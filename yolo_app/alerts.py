"""告警通道：把事件推送到 webhook（企业微信机器人 / 飞书 / 通用 JSON）。

行为约定（生产环境很重要）：
  · 告警失败**绝不**中断检测主流程，只累加 failed 计数并写一条 warning 日志；
  · 同步发送（默认 5s 超时），实现简单、语义清晰；高频告警场景再改异步队列；
  · 只传文本 + 抓拍图路径。要发图片本体需先调用各平台的媒体上传接口拿 media_id，
    这属于业务定制，见 README 路线图；
  · 传 key 与传完整 URL 都行：不以 http 开头时按类型自动拼接官方接口前缀。
"""
import json
import urllib.request

WECOM = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="
FEISHU = "https://open.feishu.cn/open-apis/bot/v2/hook/"


class AlertChannel:
    """type: wecom(企业微信) / feishu(飞书) / generic(原样 POST JSON)。失败只记日志，不中断检测。"""

    def __init__(self, url="", kind="generic", timeout=5, logger=None):
        self.url, self.kind, self.timeout, self.logger = url, kind, timeout, logger
        self.sent, self.failed = 0, 0
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

    def send(self, text):
        """发送一条文本告警；返回是否成功。异常一律吞掉，绝不影响检测主流程。"""
        if not self.enabled:
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

    def notify_event(self, event, source=""):
        text = "[YOLO 告警] 来源 %s | 类别 %s | 置信度 %s | 时间 %s | 抓拍 %s" % (
            source, event["class"], event["conf"], event["time"], event["image"])
        return self.send(text)
