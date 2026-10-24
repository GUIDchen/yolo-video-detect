"""视频源统一抽象：本地文件 / 目录 / 摄像头 / RTSP。

设计目标：让上游（pipeline）不用关心"帧从哪来"。
  · `VideoSource.read()` 只返回"一帧 ndarray"或 None（None = 流结束）；
  · 真正的读取实现拆到 `reader.py`：文件与摄像头用 OpenCV，直播源可用 FFmpeg 子进程
    （有界队列 + 僵死看门狗 + 重连），本模块只做门面与统计透传；
  · 重连次数记录在 `.reconnects` 里、丢帧数记录在 `.dropped` 里，都写进统计报告；
  · `list_videos(dir)` 把目录批量展开成视频文件列表，是 --workers 多进程的输入来源。

加新源（GStreamer/平台 SDK）只需在 reader.py 加一个后端，其余模块无感。
"""
from pathlib import Path

from .reader import VideoReader, is_live_uri

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.ts', '.mpg', '.mpeg'}
LIVE_PREFIX = ('rtsp://', 'rtmp://', 'http://', 'https://')


def is_camera(uri):
    return isinstance(uri, int) or str(uri).isdigit()


def list_videos(path):
    """目录 → 展开成视频文件列表；文件则原样返回。"""
    p = Path(path)
    if not p.is_dir():
        return [str(path)]
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)


class VideoSource:
    """统一 open/read/release 门面；具体后端见 reader.py（opencv | ffmpeg）。"""

    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0,
                 reader='auto', read_timeout=10, logger=None):
        self.uri = int(uri) if is_camera(uri) else uri
        self.reader = VideoReader(self.uri, camera, reconnect, retry_delay,
                                  mode=reader, read_timeout=read_timeout, logger=logger)

    @property
    def is_live(self):
        return is_camera(self.uri) or is_live_uri(self.uri)

    @property
    def backend(self):
        return self.reader.backend

    @property
    def opened(self):
        return self.reader.opened

    @property
    def fps(self):
        return self.reader.fps

    @property
    def total(self):
        return self.reader.total

    @property
    def width(self):
        return self.reader.width

    @property
    def height(self):
        return self.reader.height

    @property
    def reconnects(self):
        return self.reader.reconnects

    @property
    def dropped(self):
        return self.reader.dropped

    def read(self):
        return self.reader.read()

    def release(self):
        self.reader.release()
