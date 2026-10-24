"""告警通道：把事件推送到 webhook（企业微信机器人 / 飞书 / 通用 JSON）。

行为约定（生产环境很重要）：
  · 告警失败**绝不**中断检测主流程，只累加 failed 计数并写一条 warning 日志；
  · 发送仍是同步的（默认 5s 超时），但"何时发、失败怎么重试"交给 outbox：
    pipeline 先把告警写进 SQLite outbox，再每帧批量 drain；失败按指数退避留队，
    成功 mark_sent，超过次数 mark_dead。这样网络抖动不会堵住检测主循环；
  · 去重（dedupe_s）在**入队前**做：同一文本在窗口内只保留一条，重试不会因为
    去重被误判成"跳过"；
  · 只传文本 + 抓拍图路径。要发图片本体需先调用各平台的媒体上传接口拿 media_id，
    这属于业务定制，见 README 路线图；
  · 传 key 与传完整 URL 都行：不以 http 开头时按类型自动拼接官方接口前缀。
"""
import json
import time
import urllib.request

WECOM = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="
FEISHU = "https://open.feishu.cn/open-apis/bot/v2/hook/"


class AlertChannel:
    """type: wecom(企业微信) / feishu(飞书) / generic(原样 POST JSON)。失败只记日志，不中断检测。"""

    def __init__(self, url="", kind="generic", timeout=5, logger=None, dedupe_s=0.0):
        self.url, self.kind, self.timeout, self.logger = url, kind, timeout, logger
        self.sent, self.failed, self.skipped = 0, 0, 0
        self.dedupe_s = float(dedupe_s or 0)
        self._seen = {}                 # 告警文本 -> 上次发送时间（用于窗口内去重）
        if url and not url.startswith("http"):
            if kind == "wecom":
                self.url = WECOM + url
            elif kind == "feishu":
                self.url = FEISHU + url

    @property
    def enabled(self):
        return bool(self.url)

    def _payload(self, text):
        if self.kind == "wecom":
            return {"msgtype": "text", "text": {"content": text}}
        if self.kind == "feishu":
            return {"msg_type": "text", "content": {"text": text}}
        return {"text": text}

    def accept(self, text):
        """入队前先去重：返回 True 表示这条告警应当入队/发送。

        dedupe_s <= 0 时永远返回 True。窗口内重复时累加 skipped，语义与旧版
        send() 里的去重完全一致，只是把"登记时间"提前到了入队那一刻。
        """
        if self.dedupe_s <= 0:
            return True
        now = time.time()
        last = self._seen.get(text)
        if last is not None and now - last < self.dedupe_s:
            self.skipped += 1
            return False
        self._seen = {k: t for k, t in self._seen.items() if now - t < self.dedupe_s}   # 顺带清理过期键
        self._seen[text] = now
        return True

    def send(self, text, dedupe=True):
        """发送一条文本告警；返回是否成功。异常一律吞掉，绝不影响检测主流程。

        去重：dedupe_s > 0 且 dedupe=True 时，相同文本在该窗口内只发一次。越线事件里
        同一个目标可能在阈值附近来回抖动触发多条一模一样的消息，去重比重试更能减少骚扰。
        outbox 重试时传 dedupe=False——入队前已经去重，重试不能被去重逻辑吞掉。
        """
        if not self.enabled:
            return False
        if dedupe and not self.accept(text):
            return False
        try:
            req = urllib.request.Request(self.url,
                                        data=json.dumps(self._payload(text)).encode("utf-8"),
                                        headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                r.read()
            self.sent += 1
            return True
        except Exception as e:
            self.failed += 1
            if self.logger:
                self.logger.warning("告警发送失败: %s" % e)
            return False

    @staticmethod
    def event_text(event, source=""):
        """拼装抓拍告警文本；pipeline 入 outbox 与直接发送共用同一口径。"""
        return "[YOLO 告警] 来源 %s | 类别 %s | 置信度 %s | 时间 %s | 抓拍 %s" % (
            source, event["class"], event["conf"], event["time"], event["image"])

    def notify_event(self, event, source=""):
        return self.send(self.event_text(event, source))


def drain_outbox(store, channel, logger=None, limit=10, max_attempts=3, base_delay=2.0, rounds=1):
    """从 outbox 认领到期告警并发送，返回 {'claimed': n, 'sent': n, 'failed': n}。

    设计要点：
      · 不抛异常：单条发送失败（返回 False 或抛错）都按退避留队，超限置 dead；
      · channel.send 传 dedupe=False：去重已在入队前完成，重试必须真正重发；
      · rounds > 1 用于收尾时把剩余 pending 多刷几轮（失败项有 next_retry_at，
        同一轮不会重复认领，所以不会死循环）。
    """
    stats = {'claimed': 0, 'sent': 0, 'failed': 0}
    if store is None or channel is None:
        return stats
    for _ in range(max(int(rounds or 1), 1)):
        rows = store.claim_due(limit=limit)
        if not rows:
            break
        stats['claimed'] += len(rows)
        for row in rows:
            err = ''
            try:
                ok = bool(channel.send(row['text'], dedupe=False))
            except Exception as e:                 # send 理论上自己吞异常，这里再兜一层
                ok, err = False, str(e)
            if ok:
                store.mark_sent(row['id'])
                stats['sent'] += 1
            else:
                store.mark_failed(row['id'], err or '发送失败',
                                  max_attempts=max_attempts, base_delay=base_delay)
                stats['failed'] += 1
                if logger:
                    logger.warning(f'告警发送失败（outbox #{row["id"]}），已记录并按策略退避/置 dead')
    return stats
