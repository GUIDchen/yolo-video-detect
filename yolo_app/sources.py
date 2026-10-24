"""视频源统一抽象：本地文件 / 目录 / 摄像头 / RTSP。

设计目标：让上游（pipeline）不用关心"帧从哪来"。
  · `VideoSource.read()` 只返回"一帧 ndarray"或 None（None = 流结束）；
  · 摄像头与 RTSP 这类直播源读失败时会自动重连（次数/间隔来自 run.reconnect、run.retry_delay），
    重连次数记录在 .reconnects 里并写进统计报告，方便判断链路质量；
  · 摄像头打开时会申请 width/height/fps（来自 configs 的 camera 段），
    注意这两个属性是"请求值"，实际以驱动给的能力为准（后面用 src.fps 回读真实帧率）；
  · `list_videos(dir)` 把目录批量展开成视频文件列表，是 --workers 多进程的输入来源。

加新源（GStreamer/平台 SDK）只需改 _open()/read()，其余模块无感。
"""
import time
from pathlib import Path

import cv2

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
    """统一 open/read/release；直播源读失败时按 reconnect 次数重连。"""

    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0):
        self.uri = int(uri) if is_camera(uri) else uri
        self.camera = camera or {}
        self.reconnect, self.retry_delay = reconnect, retry_delay
        self.cap = None
        self.reconnects = 0                 # 整段运行累计重连次数（写进统计）
        self._fails = 0                     # 连续失败次数，读到帧就清零
        self._open()

    @property
    def is_live(self):
        return is_camera(self.uri) or str(self.uri).startswith(LIVE_PREFIX)

    def _open(self):
        self.cap = cv2.VideoCapture(self.uri)
        c = self.camera or {}
        if is_camera(self.uri):                      # 摄像头：申请指定分辨率/帧率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.get('width', 1280))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.get('height', 720))
            self.cap.set(cv2.CAP_PROP_FPS, c.get('fps', 30))
        # 直播源只保留最新帧：缓冲里堆积的旧帧只会变成延迟，且会让重连后"追赶"很久
        if self.is_live and c.get('buffer_size'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(c['buffer_size']))
        return self.cap.isOpened()

    def read(self):
        """读一帧；直播源读失败时按连续失败次数重连，文件源直接返回 None 结束。

        语义与配置对齐：run.reconnect = 单次故障最多连续重连几次（0 = 不重连，负数 = 无限重试），
        成功读到一帧后计数清零——老实现把重连次数记成"整段运行的总次数"，直播流恢复几次后
        就再也不重连了。
        """
        ok, frame = self.cap.read()
        if ok:
            self._fails = 0
            return frame
        if not self.is_live:
            return None
        limit = self.reconnect if self.reconnect is not None else 0
        while limit < 0 or self._fails < limit:
            self._fails += 1
            self.reconnects += 1
            time.sleep(self.retry_delay)
            self.cap.release()
            if self._open():
                ok, frame = self.cap.read()
                if ok:
                    self._fails = 0
                    return frame
        return None

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def total(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        if self.cap is not None:
            self.cap.release()
