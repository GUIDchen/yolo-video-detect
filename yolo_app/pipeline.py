"""处理主循环：读帧 → 推理 → ROI/越线 → 绘制 → 落盘 → 统计 → 告警。

文件、摄像头、RTSP 共用同一条链路；多进程 worker 也复用它。

资源与中断约定（这一层最容易出线上事故，务必保持）：
  · VideoSource / VideoSink / EventSink 一律在 finally 里释放——异常、Ctrl-C、面板停机
    都不能让 mp4 缺 moov 原子（播放器打不开）、摄像头句柄泄漏或异步抓拍图丢失；
  · stop_event 由 Web 面板/外部注入，主循环每帧检查一次，用于优雅停机；
  · 中断只"跳出循环并正常收尾"，不吞掉真正的异常（异常继续抛给上层记录）。

事件与告警链路（P2 新增）：
  · 越线事件、抓拍事件先写 SQLite events 表，再把告警文本放进 outbox；
  · 主循环每帧 drain 一次 outbox（最多 10 条），失败按指数退避留队，超限置 dead；
  · 收尾时再刷一轮 pending，尽力而为，失败不影响退出；
  · output.events.db 为空时完全退回旧行为：不建库，告警同步发送。
"""
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from .alerts import AlertChannel, drain_outbox
from .analytics import ConfFilter, LineCounter, RegionFilter, StationaryGate, expand_box, remap_boxes
from .detector import resolve_class_ids
from .motion import MotionGate, overlay_boxes
from .outputs import EventSink, VideoSink, write_json
from .sources import VideoSource, is_camera
from .store import EventStore


def _source_hash(source):
    """源绝对路径的 sha1 前 6 位：同一路径跨运行稳定，不同目录的同名文件不会撞车。"""
    return hashlib.sha1(str(Path(str(source)).resolve()).encode('utf-8')).hexdigest()[:6]


def output_paths(cfg, source):
    """结果文件按源命名：视频取文件名，摄像头取时间戳。

    命名模式（cfg.output.name_mode，缺省 stem）：
      · stem   源文件名主干 + .mp4；同名文件会互相覆盖，保持历史行为；
      · unique 源文件名主干 + 源绝对路径哈希前 6 位 + .mp4，目录批量处理时防覆盖。
    摄像头始终用时间戳（unique 对摄像头无意义）；显式 output.name 永远优先。
    """
    out = Path(cfg['output']['dir'])
    name = cfg['output'].get('name') or ''
    if not name:
        src = str(source)
        if src.isdigit():
            name = 'camera_' + time.strftime('%Y%m%d_%H%M%S') + '.mp4'
        elif cfg['output'].get('name_mode', 'stem') == 'unique':
            name = f'{Path(src).stem}_{_source_hash(src)}.mp4'
        else:
            name = f'{Path(src).stem}.mp4'
    return out / name


def process(source, cfg, detector, logger, show=None, record=None, on_stats=None, on_frame=None,
            stop_event=None):
    """跑完一个视频源，返回统计字典。这是全项目唯一的业务编排入口。

    参数：
        source     视频文件 / 目录项 / 摄像头编号 / RTSP 地址
        cfg        已合并好的配置字典（见 config.load_config）
        detector   已加载的模型（Detector 实例）
        show       是否弹实时窗口（多进程场景必须为 False）
        record     是否写结果视频（默认取 cfg.output.video）
        on_stats   每帧回调实时统计，Web 面板用它推送 WebSocket
        on_frame   每帧回调标注图，Web 面板用它生成 MJPEG 流
        stop_event threading.Event，置位后本轮在下一帧边界干净退出（优雅停机）
    返回：
        summary 字典：帧数、平均 FPS、逐类检测次数、唯一目标数、越线统计、
        事件列表、告警发送计数、视频产物路径（含分段）等，同时写入同名 JSON。
    """
    run_cfg = cfg['run']
    show = run_cfg['show'] if show is None else show
    record = cfg['output']['video'] if record is None else record

    # 1) 打开视频源：摄像头/RTSP 失败时给出可操作的排查提示（权限是最常见原因）
    src = VideoSource(source, cfg['camera'], run_cfg['reconnect'], run_cfg['retry_delay'],
                      reader=run_cfg.get('reader', 'auto'),
                      read_timeout=run_cfg.get('read_timeout', 10), logger=logger)
    if not src.opened:
        hint = ''
        if not is_camera(source) and not Path(str(source)).exists():
            hint = ('\n  · 文件/目录不存在：容器里跑要先把宿主机目录挂进来，'
                    '例如 -v $PWD/data:/app/data，再用 --source data/xxx.mp4')
        elif is_camera(source):
            hint += ('\n  · macOS 权限：系统设置 → 隐私与安全性 → 摄像头 → 勾选 VS Code / 终端，然后重启该应用'
                     '\n  · 查看可用摄像头：ffmpeg -f avfoundation -list_devices true -i ""'
                     '\n  · 换个编号试试：--index 1（或设 configs 里 camera.index）')
        raise RuntimeError(f'无法打开视频源: {source}{hint}')

    video_path = output_paths(cfg, source)
    sink = events = roi = line = alert = conf = store = None
    outbox_pending = 0
    max_attempts, base_delay = 3, 2.0
    stationary = StationaryGate(0)          # 未启用时当作"永远需要推理"
    motion = None
    hold_left = 0
    hits, ids, crossings = Counter(), set(), []
    idx = done = skipped = skip_stationary = skip_motion = 0
    interrupted = False
    last_res = None
    t0 = time.time()
    infer_total = infer_max = 0.0

    def db_label():
        """summary / stats 里的库路径；未配置或库不可用时为 None。"""
        return str(store.path) if (store is not None and store.available) else None

    def emit_stats():
        """把当前统计推给面板；跳帧/跳过推理的分支也要调用，否则前端 FPS 会僵住。"""
        if not on_stats:
            return
        el = max(time.time() - t0, 1e-6)
        on_stats({'frames': done, 'skipped': skipped, 'fps': round(done / el, 1),
                  'infer_ms_avg': round(infer_total / done, 2) if done else 0,
                  'infer_ms_max': round(infer_max, 2),
                  'skipped_stationary': skip_stationary, 'skipped_motion': skip_motion,
                  'dropped': src.dropped,
                  'per_class': dict(hits), 'unique_tracks': len(ids),
                  'line': dict(line.total), 'events': len(events.events) if events else 0,
                  'dropped_images': events.dropped_images if events else 0,
                  'alert_sent': alert.sent if alert else 0,
                  'alerts_sent': alert.sent if alert else 0,
                  'alert_failed': alert.failed if alert else 0,
                  'alerts_failed': alert.failed if alert else 0,
                  'alert_skipped': alert.skipped if alert else 0,
                  'outbox_pending': outbox_pending, 'events_db': db_label()})

    def refresh_outbox():
        """重新读一次 outbox 计数；只在入队/认领后调用，避免每帧查库。"""
        nonlocal outbox_pending
        if store is not None:
            outbox_pending = int(store.counts().get('pending', 0))

    def save_event(cls, conf=None, track_id=None, image=None, payload=None, ts=None):
        """把一条越线/抓拍事件写进 events 表；没有库或写库失败时返回 None。"""
        if store is None or not store.available:
            return None
        try:
            return store.add_event(ts or time.strftime('%Y-%m-%d %H:%M:%S'), idx, str(source),
                                   cls, conf, track_id, image, payload)
        except Exception as e:
            logger.warning(f'事件落库失败：{e}')
            return None

    def push_alert(text, event_id=None):
        """告警入队：先按 dedupe_s 去重，再进 outbox。

        没有可用数据库时退回旧行为（AlertChannel.send 同步发送 + 内部去重）；
        outbox 写不进去时也退回同步发送，保证告警至少不会因为库故障而消失。
        """
        if alert is None or not alert.enabled:
            return False
        try:
            if store is not None and store.available:
                if not alert.accept(text):
                    return False
                if store.enqueue_alert(text, event_id) is not None:
                    refresh_outbox()
                    return True
                alert.send(text, dedupe=False)
                return False
            alert.send(text)
            return True
        except Exception as e:
            logger.warning(f'告警入队失败：{e}')
            return False

    def drain_alerts(rounds=1):
        """处理 outbox：每帧最多 10 条；rounds>1 用于收尾多刷几轮。"""
        if store is None or alert is None or not store.available or not alert.enabled:
            return
        if rounds <= 1 and outbox_pending <= 0:
            return
        try:
            st = drain_outbox(store, alert, logger=logger, limit=10, max_attempts=max_attempts,
                              base_delay=base_delay, rounds=rounds)
            if st['claimed']:
                refresh_outbox()
        except Exception as e:
            logger.warning(f'告警 outbox 处理失败：{e}')

    try:
        # 2) 业务组件：录像、抓拍、ROI、越线、告警
        sink = VideoSink(video_path, src.fps, cfg['output']['segment_seconds']) if record else None
        ev_cfg = cfg['output']['events']
        # 事件库与抓拍开关解耦：即使不开抓拍，越线事件也能落 events 表。
        db_cfg = str(ev_cfg.get('db') or '')
        if db_cfg:
            db_path = Path(db_cfg)
            if not db_path.is_absolute():
                db_path = Path(cfg['output']['dir']) / db_path
            try:
                store = EventStore(db_path, logger=logger)
            except Exception as e:                  # 双保险：EventStore 本身已吞异常
                logger.warning(f'初始化事件库失败：{e}（本次运行不落库）')
                store = None
        events = (EventSink(Path(cfg['output']['dir']) / 'events',
                            resolve_class_ids(ev_cfg['classes'], detector.names),
                            ev_cfg['min_conf'], ev_cfg['cooldown_s'],
                            async_write=bool(ev_cfg.get('async_write', True)),
                            queue_size=int(ev_cfg.get('async_queue', 64) or 64))
                  if ev_cfg['enable'] else None)
        a_cfg = cfg.get('analytics', {})
        roi = RegionFilter(a_cfg.get('roi'), anchor=a_cfg.get('anchor', 'center'))
        conf = ConfFilter(cfg['model'].get('report_conf') or 0)
        if detector.track:                  # 静止降频依赖 track_id，没开跟踪就没有意义
            stationary = StationaryGate(a_cfg.get('stationary_frames', 0),
                                        a_cfg.get('stationary_interval', 10),
                                        a_cfg.get('stationary_min_move', 2))
        m_cfg = cfg.get('motion', {})
        if m_cfg.get('enable'):
            motion = MotionGate(m_cfg.get('threshold', 25), m_cfg.get('min_area_ratio', 0.002),
                                m_cfg.get('blur', 5),
                                a_cfg.get('roi') if m_cfg.get('mask_roi', True) else None)
        hold_frames = int(m_cfg.get('hold_frames', 0) or 0)

        crop_cfg = None
        if a_cfg.get('roi_crop') and roi.enabled:
            crop_cfg = {'margin': float(a_cfg.get('roi_margin', 0.1) or 0),
                        'imgsz': int(a_cfg.get('roi_imgsz') or 0), 'box': None, 'off': False}

        def infer_frame(frame):
            """ROI 裁剪推理：裁 → 小尺寸推理 → 平移回原图；不支持的任务自动退回全图。"""
            if crop_cfg is None or crop_cfg['off']:
                return detector.infer(frame)
            if crop_cfg['box'] is None:
                crop_cfg['box'] = expand_box(roi.roi, crop_cfg['margin'],
                                             frame.shape[1], frame.shape[0])
                x1, y1, x2, y2 = crop_cfg['box']
                if x2 - x1 < 16 or y2 - y1 < 16:      # ROI 退化成一个点，裁了没意义
                    crop_cfg['off'] = True
                    return detector.infer(frame)
            x1, y1, x2, y2 = crop_cfg['box']
            res = detector.infer(frame[y1:y2, x1:x2], imgsz=crop_cfg['imgsz'] or None)
            if getattr(res, 'masks', None) is not None or getattr(res, 'keypoints', None) is not None:
                logger.warning('ROI 裁剪只支持检测框，遇到分割/姿态结果，已退回全图推理')
                crop_cfg['off'] = True
                return detector.infer(frame)
            remap_boxes(res, x1, y1)
            res.orig_img = frame
            res.orig_shape = frame.shape[:2]
            return res
        line = LineCounter(a_cfg.get('line'), resolve_class_ids(a_cfg.get('classes'), detector.names),
                           detector.names, track_ttl=a_cfg.get('track_ttl', 300),
                           min_cross_frames=a_cfg.get('min_cross_frames', 2),
                           anchor=a_cfg.get('anchor', 'center'))
        al_cfg = cfg.get('alert', {})
        retry_cfg = al_cfg.get('retry') or {}
        max_attempts = max(int(retry_cfg.get('max_attempts', 3) or 3), 1)
        base_delay = max(float(retry_cfg.get('base_delay', 2.0) or 0.0), 0.0)
        alert = AlertChannel(al_cfg.get('webhook', ''), al_cfg.get('type', 'generic'),
                             al_cfg.get('timeout', 5), logger, dedupe_s=al_cfg.get('dedupe_s', 0))
        if store is not None and store.available:
            outbox_pending = int(store.counts().get('pending', 0))

        logger.info(f'开始处理 {source} | {src.width}x{src.height} @{src.fps:.0f}fps | 设备 {detector.device_name}'
                    f'{" | 跟踪 " + cfg["track"]["tracker"] if detector.track else ""}'
                    f'{" | ROI" if roi.enabled else ""}{" | 越线计数" if line.enabled else ""}'
                    f'{" | 双阈值 " + str(conf.min_conf) if conf.enabled else ""}'
                    f'{" | 静止降频 " + str(stationary.interval) + "帧" if stationary.enabled else ""}'
                    f'{" | 告警已启用" if alert.enabled else ""}'
                    f'{" | 运动门控" if motion is not None else ""}'
                    f'{" | ROI 裁剪推理" if crop_cfg is not None else ""}'
                    f'{" | 事件库 " + db_label() if db_label() else ""}'
                    f'{" | 读取 " + src.backend if src.backend != "opencv" else ""}')

        while True:
            if stop_event is not None and stop_event.is_set():
                interrupted = True
                logger.info('收到停止信号，处理提前收尾')
                break
            # 告警 outbox：每帧最多发 10 条；失败按退避留队，下一帧再试
            drain_alerts()
            # 3) 取帧：None 表示文件结束或直播流重连失败
            frame = src.read()
            if frame is None:
                break
            idx += 1
            if run_cfg['max_frames'] and idx > run_cfg['max_frames']:
                break
            if run_cfg['stride'] > 1 and (idx - 1) % run_cfg['stride']:
                skipped += 1
                if sink:
                    sink.write(frame)           # 跳过的帧原样写出，保持时长一致
                continue

            # 运动门控：帧差先筛一遍，没动就跳过推理（默认关闭）
            if motion is not None and not motion.update(frame):
                skip_motion += 1
                out = frame
                if hold_frames and last_res is not None and hold_left < hold_frames:
                    out = overlay_boxes(frame, last_res)      # 沿用上一帧框，避免画面突然空白
                    hold_left += 1
                if sink:
                    sink.write(out)
                if on_frame:
                    on_frame(out)
                emit_stats()
                continue
            hold_left = 0

            # 静止降频：有跟踪且所有目标都不动时，按 interval 抽帧推理（默认关闭）
            if stationary.enabled and not stationary.should_infer(last_res, idx):
                skip_stationary += 1
                if sink:
                    sink.write(frame)
                if on_frame:
                    on_frame(frame)
                emit_stats()
                continue

            # 4) 推理（可选跟踪）→ 5) ROI 过滤：区域外的框直接剔除，后续统计/绘制都看不到
            t_infer = time.perf_counter()
            res = conf.keep(roi.keep(infer_frame(frame)))       # 先 ROI 后置信度：ROI 更便宜
            last_res = res
            cost_ms = (time.perf_counter() - t_infer) * 1000
            infer_total += cost_ms
            infer_max = max(infer_max, cost_ms)

            # 6) 越线计数：按 track_id 判断目标是否从线的一侧穿到另一侧
            crossed = line.update(res, detector.names, frame_no=idx)
            if crossed:                         # 越线事件：先落库，再进告警 outbox
                for c in crossed:
                    c['frame'] = idx
                    crossings.append(c)
                    eid = save_event(c['class'], track_id=c.get('track_id'), payload=c)
                    push_alert(f'[YOLO 越线] {c["class"]} {c["direction"]} | track {c["track_id"]}', eid)
            hits.update(detector.names[int(c)] for c in res.boxes.cls.tolist())
            tids = res.boxes.id.tolist() if res.boxes.id is not None else []
            ids.update(int(t) for t in tids if t is not None)
            done += 1

            # 7) 绘制：检测框 + ROI 多边形 + 越线（都画在同一张图上，后续写盘与预览共用）
            anno = res.plot(labels=True, conf=True)
            roi.draw(anno)
            line.draw(anno)
            # 8) 告警抓拍 + 推送：事件同步记入 events 列表并落库，图片按配置异步写盘
            if events and events.check(res, detector.names, idx, anno):
                ev = events.events[-1]
                eid = save_event(ev['class'], conf=ev['conf'], track_id=ev['track_id'],
                                 image=ev['image'], payload=ev, ts=ev['time'])
                if alert is not None:
                    push_alert(alert.event_text(ev, source), eid)
            if sink:
                sink.write(anno)
            if show:
                cv2.imshow('YOLO', anno)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    interrupted = True
                    break
            if on_frame:
                on_frame(anno)
            emit_stats()
            if done % max(cfg['log']['interval'], 1) == 0:
                el = time.time() - t0 or 1e-6
                logger.info(f'帧 {idx}/{src.total or "-"} | {done / el:.1f} FPS | {dict(hits)}')
    except KeyboardInterrupt:                   # Ctrl-C：当作"提前结束"，收尾后照常出统计
        interrupted = True
        logger.warning('收到 Ctrl-C，正在收尾（写出剩余录像与统计）…')
    finally:
        src.release()
        if sink:
            sink.close()
        if events:
            events.close()                   # 排空异步抓拍队列，保证已命中的图片落盘
        if show:
            cv2.destroyAllWindows()
        # 异常向上抛时不会再走到下面的收尾刷队列，这里先把数据库连接关掉
        if sys.exc_info()[0] is not None and store is not None:
            store.close()

    # 收尾：把 outbox 里剩余的 pending 再尽力刷几轮；失败按退避留队，不影响退出
    drain_alerts(rounds=10)

    el = time.time() - t0 or 1e-6
    infer_s = infer_total / 1000
    summary = {
        'source': str(source), 'model': detector.weights,
        'frames_src': idx, 'frames_inferred': done, 'frames_skipped': skipped,
        'frames_inference_skipped': skip_stationary + skip_motion,
        'skip_reasons': {'stationary': skip_stationary, 'motion': skip_motion},
        'boxes_low_conf': conf.dropped if conf else 0,
        'elapsed_s': round(el, 1), 'avg_fps': round(done / el, 1),
        'infer_fps': round(done / infer_s, 1) if infer_s else 0,
        'infer_ms_avg': round(infer_total / done, 2) if done else 0,
        'infer_ms_max': round(infer_max, 2),
        'per_class_frames': dict(hits), 'unique_tracks': len(ids),
        'reconnects': src.reconnects, 'reader': src.backend,
        'frames_dropped': src.dropped,
        'output_video': sink.output if sink else None,
        'video_parts': sink.parts if sink else [],
        'events': events.events if events else [],
        'dropped_images': events.dropped_images if events else 0,
        'alert_sent': alert.sent if alert else 0,
        'alert_failed': alert.failed if alert else 0,
        'alert_skipped': alert.skipped if alert else 0,
        'outbox_pending': outbox_pending, 'events_db': db_label(),
    }
    if interrupted:
        summary['interrupted'] = True
    summary.update(line.summary() if line else {'line_total': {'in': 0, 'out': 0}, 'line_by_class': {}})
    if crossings:
        summary['crossings'] = crossings
    try:
        if cfg['output']['json']:
            summary['output_json'] = write_json(Path(video_path).with_suffix('.json'), summary)
        logger.info(f'完成 {source} | 推理 {done} 帧 | {summary["avg_fps"]} FPS | 唯一目标 {len(ids)} | '
                    f'越线 in {line.total["in"]}/out {line.total["out"]} | 事件 {len(summary["events"])} | '
                    f'待发告警 {outbox_pending} | 输出 {sink.output if sink else "(未写视频)"}')
    finally:
        if store is not None:
            store.close()
    return summary


def run_job(payload):
    """多进程 worker：子进程内独立加载模型与配置，互不共享状态。"""
    source, cfg, show = payload
    from .detector import Detector
    from .logging_setup import get_logger

    logger = get_logger(cfg['log']['level'], cfg['log'].get('file'))
    try:
        return process(source, cfg, Detector(cfg), logger, show=show)
    except Exception as e:
        logger.error(f'{source} 处理失败: {e}')
        return {'source': str(source), 'error': str(e)}
