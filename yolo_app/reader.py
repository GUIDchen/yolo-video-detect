"""视频读取后端：OpenCV（默认）与 FFmpeg 子进程（直播源，带僵死看门狗）。

为什么需要第二个后端：`cv2.VideoCapture` 的缓冲不可控、没有读超时，RTSP 断流时
会一直阻塞或返回陈旧帧，延迟还会随时间累积。直播源换成 ffmpeg 子进程后：

  · 队列长度 = camera.buffer_size，满了丢最旧帧 —— 延迟不累积，丢了多少帧有计数；
  · `read_timeout` 判"多久没来帧"，到点杀掉进程重启（按 run.reconnect 语义）；
  · `-rtsp_transport tcp` + `nobuffer/low_delay` 降低抖动。

对外契约：`read() -> ndarray | None`，`release()`，`fps/total/reconnects/dropped`。
文件与摄像头仍然走 OpenCV（文件不需要看门狗，摄像头在 macOS 上走 avfoundation 更稳）。
"""
import json
import queue
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

LIVE_PREFIX = ('rtsp://', 'rtmp://', 'http://', 'https://')


def is_live_uri(uri):
    return str(uri).startswith(LIVE_PREFIX)


def choose_backend(uri, mode='auto', has_ffmpeg=None):
    """决定用哪个读取后端：auto 时只有"直播源 + ffmpeg 可用"才选 ffmpeg。"""
    if mode in ('opencv', 'ffmpeg'):
        return mode
    has = shutil.which('ffmpeg') is not None if has_ffmpeg is None else bool(has_ffmpeg)
    return 'ffmpeg' if (is_live_uri(uri) and has) else 'opencv'


def parse_rate(text):
    """'30000/1001' -> 29.97；解析不了返回 None。"""
    try:
        num, _, den = str(text).partition('/')
        return float(num) / float(den or 1)
    except Exception:
        return None


def probe_stream(uri, timeout=10):
    """ffprobe 探测分辨率/帧率/总帧数；任何失败都返回 None（由调用方回退）。"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
           '-of', 'json', str(uri)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True).stdout
        info = json.loads(out)['streams'][0]
    except Exception:
        return None
    w, h = int(info.get('width') or 0), int(info.get('height') or 0)
    if not w or not h:
        return None
    nb = str(info.get('nb_frames') or '')
    return {'width': w, 'height': h,
            'fps': parse_rate(info.get('r_frame_rate')) or 25.0,
            'total': int(nb) if nb.isdigit() else 0}


class OpenCVReader:
    """cv2.VideoCapture 后端：文件与摄像头的默认选择。"""

    backend = 'opencv'

    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0, logger=None):
        self.uri = uri
        self.camera = camera or {}
        self.reconnect, self.retry_delay = reconnect, retry_delay
        self.logger = logger
        self.cap = None
        self.reconnects = 0
        self.dropped = 0
        self._fails = 0
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.uri)
        c = self.camera
        if isinstance(self.uri, int):                  # 摄像头：申请指定分辨率/帧率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.get('width', 1280))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.get('height', 720))
            self.cap.set(cv2.CAP_PROP_FPS, c.get('fps', 30))
        if c.get('buffer_size'):                       # 直播源只留最新帧
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(c['buffer_size']))
        return self.cap.isOpened()

    @property
    def opened(self):
        return bool(self.cap is not None and self.cap.isOpened())

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self._fails = 0
            return frame
        if not (isinstance(self.uri, int) or is_live_uri(self.uri)):
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

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        if self.cap is not None:
            self.cap.release()


class FFmpegReader:
    """ffmpeg 子进程后端：读线程 + 有界队列 + 僵死看门狗。"""

    backend = 'ffmpeg'

    def __init__(self, uri, info, buffer_size=1, read_timeout=10.0, reconnect=0,
                 retry_delay=1.0, logger=None, cmd_builder=None):
        self.uri = uri
        self.info = info
        self.width, self.height = int(info['width']), int(info['height'])
        self.fps = float(info.get('fps') or 25.0)
        self.total = int(info.get('total') or 0)
        self.frame_bytes = self.width * self.height * 3
        self.live = is_live_uri(uri)
        self.read_timeout = float(read_timeout or 0)
        self.buffer_size = max(int(buffer_size or 1), 1)
        self.reconnect, self.retry_delay = reconnect, float(retry_delay or 1)
        self.logger = logger
        self._cmd_builder = cmd_builder or self._build_cmd
        self._q = queue.Queue(maxsize=self.buffer_size)
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._fails = 0
        self.reconnects = 0
        self.dropped = 0
        self._eof = False
        self._open()

    # -- 进程管理 --------------------------------------------------------
    def _build_cmd(self):
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
        if str(self.uri).startswith('rtsp://'):
            cmd += ['-rtsp_transport', 'tcp', '-fflags', 'nobuffer', '-flags', 'low_delay']
        cmd += ['-i', str(self.uri), '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
        return cmd

    def _open(self):
        self._stop.clear()
        self._eof = False
        self._drain()
        try:
            self._proc = subprocess.Popen(self._cmd_builder(), stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, bufsize=0)
        except Exception as e:
            if self.logger:
                self.logger.warning(f'ffmpeg 启动失败: {e}')
            self._proc = None
            return False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return True

    def _read_frame(self, proc):
        """从管道读满一整帧。管道 read() 会短读，必须循环 readinto 到 frame_bytes。"""
        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        got = 0
        while got < self.frame_bytes:
            n = proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        return np.frombuffer(buf, np.uint8).reshape((self.height, self.width, 3))

    def _pump(self):
        proc = self._proc
        try:
            while not self._stop.is_set():
                frame = self._read_frame(proc)
                if frame is None:
                    break
                self._put(frame)
        except Exception:
            pass
        self._send_sentinel()

    def _put(self, frame):
        """直播源队满丢最旧帧（保低延迟）；文件源阻塞等待（不丢帧）。"""
        while True:
            try:
                self._q.put_nowait(frame)
                return
            except queue.Full:
                if not self.live:
                    self._q.put(frame)
                    return
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _send_sentinel(self):
        """投递 None 唤醒阻塞中的 read()：表示"这一路流没了"。"""
        while True:
            try:
                self._q.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def _drain(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _close_proc(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            try:
                if self._proc.stdout:
                    self._proc.stdout.close()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._proc = None
        self._thread = None

    # -- 对外接口 --------------------------------------------------------
    @property
    def opened(self):
        return True

    def read(self):
        if self._eof:                       # 文件已读到结尾：直接返回，不再阻塞
            return None
        timeout = self.read_timeout if self.live else None
        try:
            item = self._q.get(timeout=timeout) if timeout else self._q.get()
        except queue.Empty:
            item = None
            if self.logger:
                self.logger.warning(f'超过 {timeout:.0f}s 没收到帧，判定直播流僵死: {self.uri}')
        if item is not None:
            self._fails = 0
            return item
        if not self.live:
            self._eof = True
            return None
        return self._restart()

    def _restart(self):
        limit = self.reconnect if self.reconnect is not None else 0
        self._close_proc()
        while limit < 0 or self._fails < limit:
            self._fails += 1
            self.reconnects += 1
            time.sleep(min(self.retry_delay * self._fails, self.retry_delay * 4))
            if not self._open():
                continue
            try:
                item = self._q.get(timeout=max(self.read_timeout, 1))
            except queue.Empty:
                item = None
            if item is not None:
                self._fails = 0
                return item
            self._close_proc()
        return None

    def release(self):
        self._close_proc()


class VideoReader:
    """门面：按配置挑后端，向上只暴露 read/release 与只读属性。"""

    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0, mode='auto',
                 read_timeout=10, logger=None):
        self.mode = mode or 'auto'
        backend = choose_backend(uri, self.mode)
        self.impl = None
        if backend == 'ffmpeg':
            info = probe_stream(uri)
            if info:
                try:
                    self.impl = FFmpegReader(uri, info, (camera or {}).get('buffer_size', 1),
                                             read_timeout, reconnect, retry_delay, logger)
                except Exception as e:
                    if logger:
                        logger.warning(f'FFmpeg 后端启动失败，回退 OpenCV: {e}')
            elif logger:
                logger.warning('ffprobe 探测失败，回退 OpenCV 读取（直播延迟可能累积）')
        if self.impl is None:
            self.impl = OpenCVReader(uri, camera, reconnect, retry_delay, logger)

    @property
    def backend(self):
        return self.impl.backend

    @property
    def opened(self):
        return self.impl.opened

    @property
    def fps(self):
        return self.impl.fps

    @property
    def total(self):
        return self.impl.total

    @property
    def width(self):
        return self.impl.width

    @property
    def height(self):
        return self.impl.height

    @property
    def reconnects(self):
        return self.impl.reconnects

    @property
    def dropped(self):
        return self.impl.dropped

    def read(self):
        return self.impl.read()

    def release(self):
        self.impl.release()


if __name__ == '__main__':      # 手动冒烟：python -m yolo_app.reader rtsp://...
    src = VideoReader(sys.argv[1])
    n = 0
    while True:
        frame = src.read()
        if frame is None:
            break
        n += 1
    src.release()
    print(f'读到最后 {n} 帧，后端 {src.backend}，重连 {src.reconnects} 次')
