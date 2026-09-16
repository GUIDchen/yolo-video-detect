"""结果落地：结果视频、告警抓拍、统计 JSON、留存清理。

三个类的职责：
  · VideoSink  写结果视频。两个关键取舍：
      1) H.264(avc1) 优先、mp4v 兜底 —— mp4v 在 Chromium 内核（VS Code 内置播放器/浏览器）
         里放不出来，用户看到的是黑屏；
      2) writer 延迟到"第一帧到达"才创建 —— 直播源（RTSP/摄像头）常报不出分辨率；
        `segment_seconds > 0` 时按时间切片成 result.mp4 / result_part02.mp4…（长时录制防单文件过大）。
  · EventSink  告警抓拍：命中配置类别 + 置信度达标 + 过了冷却时间才存图，避免同一目标刷屏；
  · cleanup_old 留存策略：按天数与文件数清理输出目录，防止磁盘被录像打满。

JSON 报告统一由 write_json() 落盘，字段口径见 README「输出」一节
（per_class_frames 是逐帧次数，unique_tracks 才是唯一目标数）。
"""
import json
import time
from pathlib import Path

import cv2


class VideoSink:
    """首次写入时确定尺寸；优先 H.264(avc1) 保证 VS Code/浏览器/QuickTime 能播放。

    segment_seconds > 0 时按时间切分：result.mp4 / result_part02.mp4 / ...
    """

    def __init__(self, path, fps, segment_seconds=0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps or 25.0
        self.segment_seconds = float(segment_seconds or 0)
        self.writer = None
        self.codec = None
        self.parts = []
        self._frames_in_part = 0
        self._part = 1

    def _current_path(self):
        if self._part == 1:
            return self.path
        return self.path.with_name(f'{self.path.stem}_part{self._part:02d}{self.path.suffix}')

    def _open_writer(self, path, shape):
        h, w = shape[:2]
        for codec in ('avc1', 'mp4v'):
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if vw.isOpened():
                self.codec = codec
                return vw
        raise RuntimeError(f'无法创建视频写入器: {path}')

    def write(self, frame):
        """写一帧；首帧建 writer，到点自动切分文件，跳过帧也应原样写入以保持时长。"""
        limit = self.fps * self.segment_seconds
        if self.writer is not None and self.segment_seconds and self._frames_in_part >= limit:
            self.writer.release()
            self.writer = None
            self._part += 1
            self._frames_in_part = 0
        if self.writer is None:
            path = self._current_path()
            self.writer = self._open_writer(path, frame.shape)
            self.parts.append(str(path))
        self.writer.write(frame)
        self._frames_in_part += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    @property
    def output(self):
        return self.parts[0] if self.parts else str(self.path)


class EventSink:
    """命中告警条件时抓拍一张标注图；同一告警在冷却时间内不重复抓拍。"""

    def __init__(self, out_dir, classes=None, min_conf=0.6, cooldown_s=2.0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.classes = {int(c) for c in (classes or [])}
        self.min_conf, self.cooldown = float(min_conf), float(cooldown_s)
        self._last = 0.0
        self.events = []

    def check(self, res, names, frame_no, annotated):
        """命中告警条件就抓拍一张标注图并记事件，返回是否触发了本次抓拍。

        条件：类别在配置白名单内（空=全部）且置信度 >= min_conf 且距上次抓拍超过 cooldown。
        会同步写盘一次 JPEG——冷却时间设得太小会拖慢主循环。
        """
        boxes = res.boxes
        if boxes is None or not len(boxes):
            return False
        ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        for cid, cf, tid in zip(boxes.cls.tolist(), boxes.conf.tolist(), ids):
            if cf < self.min_conf or (self.classes and int(cid) not in self.classes):
                continue
            now = time.time()
            if now - self._last < self.cooldown:
                continue
            self._last = now
            name = names[int(cid)]
            p = self.dir / f'{time.strftime("%Y%m%d_%H%M%S")}_f{frame_no}_{name}.jpg'
            cv2.imwrite(str(p), annotated)
            self.events.append({
                'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'frame': frame_no,
                'class': name, 'conf': round(float(cf), 3),
                'track_id': int(tid) if tid is not None else None, 'image': str(p),
            })
            return True
        return False


def cleanup_old(directory, keep_days=0, max_files=0, patterns=('*.mp4', '*.jpg')):
    """留存策略：按天数和文件数清理输出目录，返回删除数量。"""
    d = Path(directory)
    if not d.exists() or (not keep_days and not max_files):
        return 0
    files = sorted((f for p in patterns for f in d.rglob(p)), key=lambda f: f.stat().st_mtime)
    removed = 0
    if keep_days:
        cutoff = time.time() - keep_days * 86400
        for f in list(files):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                files.remove(f)
                removed += 1
    if max_files and len(files) > max_files:
        for f in files[:len(files) - max_files]:
            f.unlink()
            removed += 1
    return removed


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(p)
