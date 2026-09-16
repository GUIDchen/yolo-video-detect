"""处理主循环：读帧 → 推理 → ROI/越线 → 绘制 → 落盘 → 统计 → 告警。

文件、摄像头、RTSP 共用同一条链路；多进程 worker 也复用它。
"""
import time
from collections import Counter
from pathlib import Path

import cv2

from .alerts import AlertChannel
from .analytics import LineCounter, RegionFilter
from .detector import resolve_class_ids
from .outputs import EventSink, VideoSink, write_json
from .sources import VideoSource, is_camera


def output_paths(cfg, source):
    """结果文件按源命名：视频取文件名，摄像头取时间戳。"""
    out = Path(cfg['output']['dir'])
    name = cfg['output'].get('name') or ''
    if not name:
        stem = 'camera_' + time.strftime('%Y%m%d_%H%M%S') if str(source).isdigit() else Path(str(source)).stem
        name = f'{stem}.mp4'
    return out / name


def process(source, cfg, detector, logger, show=None, record=None, on_stats=None, on_frame=None):
    """跑完一个视频源，返回统计字典。这是全项目唯一的业务编排入口。

    参数：
        source     视频文件 / 目录项 / 摄像头编号 / RTSP 地址
        cfg        已合并好的配置字典（见 config.load_config）
        detector   已加载的模型（Detector 实例）
        show       是否弹实时窗口（多进程场景必须为 False）
        record     是否写结果视频（默认取 cfg.output.video）
        on_stats   每帧回调实时统计，Web 面板用它推送 WebSocket
        on_frame   每帧回调标注图，Web 面板用它生成 MJPEG 流
    返回：
        summary 字典：帧数、平均 FPS、逐类检测次数、唯一目标数、越线统计、
        事件列表、告警发送计数、视频产物路径（含分段）等，同时写入同名 JSON。
    """
    run_cfg = cfg['run']
    show = run_cfg['show'] if show is None else show
    record = cfg['output']['video'] if record is None else record

    # 1) 打开视频源：摄像头/RTSP 失败时给出可操作的排查提示（权限是最常见原因）
    src = VideoSource(source, cfg['camera'], run_cfg['reconnect'], run_cfg['retry_delay'])
    if not src.cap.isOpened():
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
    sink = VideoSink(video_path, src.fps, cfg['output']['segment_seconds']) if record else None

    ev_cfg = cfg['output']['events']
    events = (EventSink(Path(cfg['output']['dir']) / 'events',
                        resolve_class_ids(ev_cfg['classes'], detector.names),
                        ev_cfg['min_conf'], ev_cfg['cooldown_s']) if ev_cfg['enable'] else None)
    a_cfg = cfg.get('analytics', {})
    roi = RegionFilter(a_cfg.get('roi'))
    line = LineCounter(a_cfg.get('line'), resolve_class_ids(a_cfg.get('classes'), detector.names), detector.names)
    alert = AlertChannel(cfg.get('alert', {}).get('webhook', ''),
                         cfg.get('alert', {}).get('type', 'generic'),
                         cfg.get('alert', {}).get('timeout', 5), logger)

    logger.info(f'开始处理 {source} | {int(src.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
                f'{int(src.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @{src.fps:.0f}fps | 设备 {detector.device_name}'
                f'{" | 跟踪 " + cfg["track"]["tracker"] if detector.track else ""}'
                f'{" | ROI" if roi.enabled else ""}{" | 越线计数" if line.enabled else ""}'
                f'{" | 告警已启用" if alert.enabled else ""}')

    hits, ids = Counter(), set()
    crossings = []
    idx = done = 0
    t0 = time.time()
    while True:
        # 2) 取帧：None 表示文件结束或直播流重连失败
        frame = src.read()
        if frame is None:
            break
        idx += 1
        if run_cfg['max_frames'] and idx > run_cfg['max_frames']:
            break
        if run_cfg['stride'] > 1 and (idx - 1) % run_cfg['stride']:
            if sink:
                sink.write(frame)               # 跳过的帧原样写出，保持时长一致
            continue

        # 3) 推理（可选跟踪）→ 4) ROI 过滤：区域外的框直接剔除，后续统计/绘制都看不到
        res = roi.keep(detector.infer(frame))
        # 5) 越线计数：按 track_id 判断目标是否从线的一侧穿到另一侧
        crossed = line.update(res, detector.names)
        if crossed:                             # 越线事件：计入统计并推送告警
            for c in crossed:
                c['frame'] = idx
                crossings.append(c)
                alert.send(f'[YOLO 越线] {c["class"]} {c["direction"]} | track {c["track_id"]}')
        hits.update(detector.names[int(c)] for c in res.boxes.cls.tolist())
        tids = res.boxes.id.tolist() if res.boxes.id is not None else []
        ids.update(int(t) for t in tids if t is not None)
        done += 1

        # 6) 绘制：检测框 + ROI 多边形 + 越线（都画在同一张图上，后续写盘与预览共用）
        anno = res.plot(labels=True, conf=True)
        roi.draw(anno)
        line.draw(anno)
        # 7) 告警抓拍 + 推送（发送是同步的，失败只记日志）
        if events and events.check(res, detector.names, idx, anno):
            alert.notify_event(events.events[-1], source)
        if sink:
            sink.write(anno)
        if show:
            cv2.imshow('YOLO', anno)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if on_frame:
            on_frame(anno)
        if on_stats:
            on_stats({'frames': done, 'fps': round(done / max(time.time() - t0, 1e-6), 1),
                      'per_class': dict(hits), 'unique_tracks': len(ids),
                      'line': dict(line.total), 'events': len(events.events) if events else 0})
        if done % max(cfg['log']['interval'], 1) == 0:
            el = time.time() - t0 or 1e-6
            logger.info(f'帧 {idx}/{src.total or "-"} | {done / el:.1f} FPS | {dict(hits)}')

    src.release()
    if sink:
        sink.close()
    if show:
        cv2.destroyAllWindows()

    el = time.time() - t0 or 1e-6
    summary = {
        'source': str(source), 'model': detector.weights,
        'frames_src': idx, 'frames_inferred': done,
        'elapsed_s': round(el, 1), 'avg_fps': round(done / el, 1),
        'per_class_frames': dict(hits), 'unique_tracks': len(ids),
        'reconnects': src.reconnects,
        'output_video': sink.output if sink else None,
        'video_parts': sink.parts if sink else [],
        'events': events.events if events else [],
        'alert_sent': alert.sent, 'alert_failed': alert.failed,
    }
    summary.update(line.summary())
    if crossings:
        summary['crossings'] = crossings
    if cfg['output']['json']:
        summary['output_json'] = write_json(Path(video_path).with_suffix('.json'), summary)
    logger.info(f'完成 {source} | 推理 {done} 帧 | {summary["avg_fps"]} FPS | 唯一目标 {len(ids)} | '
                f'越线 in {line.total["in"]}/out {line.total["out"]} | 事件 {len(summary["events"])} | '
                f'输出 {sink.output if sink else "(未写视频)"}')
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
