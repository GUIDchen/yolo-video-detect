"""结果落地：结果视频、告警抓拍、统计 JSON、留存清理。

三个类的职责：
  · VideoSink  写结果视频。两个关键取舍：
      1) H.264(avc1) 优先、mp4v 兜底 —— mp4v 在 Chromium 内核（VS Code 内置播放器/浏览器）
         里放不出来，用户看到的是黑屏；
      2) writer 延迟到"第一帧到达"才创建 —— 直播源（RTSP/摄像头）常报不出分辨率；
        `segment_seconds > 0` 时按时间切片成 result.mp4 / result_part02.mp4…（长时录制防单文件过大）。
  · EventSink  告警抓拍：命中配置类别 + 置信度达标 + 过了冷却时间才存图，避免同一目标刷屏；
               图片可异步写盘（有界队列 + 丢最旧），慢磁盘不拖慢检测；
  · cleanup_old 留存策略：按天数与文件数清理输出目录，防止磁盘被录像打满。

JSON 报告统一由 write_json() 落盘，字段口径见 README「输出」一节
（per_class_frames 是逐帧次数，unique_tracks 才是唯一目标数）。
"""
import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2

_logger = logging.getLogger(__name__)


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
    """命中告警条件时抓拍一张标注图；同一告警在冷却时间内不重复抓拍。

    异步写盘（async_write=True）：check() 只做命中判断、同步维护 events 列表、把
    (路径, 图像) 塞进有界队列，真正的 imwrite 交给单个后台线程。队列满时丢**最旧**
    的一张并累加 dropped_images，生产端绝不阻塞——慢磁盘不能拖住检测主循环。
    事件列表始终在调用线程同步更新，因此 summary.events 立刻可用；只有图片文件
    可能晚一点落盘（或因为队列满被丢弃）。调用方在 check() 返回后不要再修改传入的
    annotated 图像——后台线程会直接读取该数组，不额外拷贝以省内存。

    直接构造时默认同步（async_write=False），保持老调用方行为不变；pipeline 会按
    output.events.async_write 显式传入。
    """

    def __init__(self, out_dir, classes=None, min_conf=0.6, cooldown_s=2.0,
                 async_write=False, queue_size=64, async_queue=None):
        # async_queue 是配置里的名字；queue_size 是类内参数名，两个都接受，避免调用方踩坑。
        if async_queue is not None:
            queue_size = async_queue
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.classes = {int(c) for c in (classes or [])}
        self.min_conf, self.cooldown = float(min_conf), float(cooldown_s)
        self._last = 0.0
        self.events = []
        self.async_write = bool(async_write)
        self.queue_size = max(int(queue_size or 1), 1)
        self.dropped_images = 0        # 队列满时丢掉的抓拍图数量（事件本身仍保留）
        self.write_failed = 0          # imwrite 返回 False / 抛异常的图片数量
        self._queue = queue.Queue(maxsize=self.queue_size) if self.async_write else None
        self._worker = None
        self._closed = False
        if self._queue is not None:
            self._worker = threading.Thread(target=self._run, name='event-sink', daemon=True)
            self._worker.start()

    def check(self, res, names, frame_no, annotated):
        """命中告警条件就记事件并抓拍，返回是否触发了本次抓拍。

        条件：类别在配置白名单内（空=全部）且置信度 >= min_conf 且距上次抓拍超过 cooldown。
        同步模式会当场写盘一次 JPEG；异步模式只入队，冷却时间太小时注意 dropped_images。
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
            self.events.append({
                'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'frame': frame_no,
                'class': name, 'conf': round(float(cf), 3),
                'track_id': int(tid) if tid is not None else None, 'image': str(p),
            })
            self._save(p, annotated)
            return True
        return False

    def _save(self, path, image):
        """把一次写盘请求交给后台线程；队列满时丢最旧的一条，生产端不阻塞。"""
        if self._queue is None:
            self._write_image(path, image)
            return
        item = (path, image)
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        # 丢最旧：先腾一个位置。worker 可能同时取走一条，此时取不到也不算丢图。
        dropped = False
        try:
            self._queue.get_nowait()
            self._queue.task_done()
            dropped = True
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            dropped = True          # 极端竞争下新图也塞不进去，同样算一次丢弃
        if dropped:
            self.dropped_images += 1

    def _write_image(self, path, image):
        """真正执行一次 JPEG 写盘；失败只计数 + warning，不影响检测。"""
        try:
            ok = cv2.imwrite(str(path), image)
        except Exception as e:
            self.write_failed += 1
            _logger.warning(f'抓拍写盘异常（{path}）：{e}')
            return False
        if not ok:
            self.write_failed += 1
            _logger.warning(f'抓拍写盘失败（{path}）')
        return bool(ok)

    def _run(self):
        """后台写盘线程：队列空时阻塞；收到 None 哨兵即退出。"""
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write_image(*item)
            finally:
                self._queue.task_done()

    @property
    def dropped(self):
        """dropped_images 的短别名，方便调用方/监控读取。"""
        return self.dropped_images

    def flush(self):
        """等待队列里的抓拍图全部写盘；同步模式无事可做。"""
        if self._queue is not None and not self._closed:
            self._queue.join()

    def close(self):
        """排空并停止后台线程；可重复调用。同步模式下是空操作。"""
        if self._closed:
            return
        if self._queue is None:
            self._closed = True
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=5)      # 队列满时最多等 5s，worker 会持续消费
        except queue.Full:
            _logger.warning('抓拍队列排空超时，剩余图片可能未写盘')
        if self._worker is not None:
            self._worker.join(timeout=30)
            if self._worker.is_alive():
                _logger.warning('抓拍写盘线程未在 30s 内退出')


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
