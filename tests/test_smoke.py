"""冒烟测试：python -m unittest discover -s tests -v（不起框架，够用就行）。"""
import json
import shutil
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.alerts import AlertChannel
from yolo_app.analytics import LineCounter, RegionFilter, _inside
from yolo_app.config import load_config
from yolo_app.metrics import prometheus_text
from yolo_app.outputs import EventSink, VideoSink, cleanup_old, write_json
from yolo_app.sources import is_camera, list_videos

ROOT = Path(__file__).resolve().parent.parent

# 测试素材是否来自真实照片（bus.jpg）——决定要不要断言"确实检出了人"
REAL_FIXTURE = False


def ensure_test_video():
    """确保 data/test.mp4 存在：有 bus.jpg 就生成真实场景视频，否则合成一段。

    这样 `git clone` 后的干净仓库也能直接跑测试（素材不入库，见 .gitignore）。
    """
    global REAL_FIXTURE
    video = ROOT / 'data' / 'test.mp4'
    if video.exists():
        REAL_FIXTURE = (ROOT / 'data' / 'bus.jpg').exists()
        return video
    (ROOT / 'data').mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'scripts' / 'make_test_video.py'),
           '--out', str(video), '--seconds', '1', '--width', '320', '--height', '240']
    if (ROOT / 'data' / 'bus.jpg').exists():
        cmd += ['--image', str(ROOT / 'data' / 'bus.jpg')]
        REAL_FIXTURE = True
    import subprocess
    subprocess.run(cmd, check=True, cwd=str(ROOT), capture_output=True)   # 静默生成，别污染测试输出
    return video


def setUpModule():
    """整个测试模块开始前准备素材（缺就现造）。"""
    ensure_test_video()


class FakeBoxes:
    """支持索引的最小 boxes 替身，用于不加载模型就测后处理。"""

    def __init__(self, xyxy, cls, ids=None, conf=None):
        self.xyxy = np.array(xyxy, dtype=float)
        self.cls = np.array(cls, dtype=float)
        self.id = None if ids is None else np.array(ids, dtype=float)
        self.conf = None if conf is None else np.array(conf, dtype=float)

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, i):
        return FakeBoxes(self.xyxy[i].tolist(), self.cls[i].tolist(),
                         None if self.id is None else self.id[i].tolist(),
                         None if self.conf is None else self.conf[i].tolist())


class FakeRes:
    def __init__(self, boxes):
        self.boxes = boxes


class TestConfig(unittest.TestCase):
    def test_defaults_and_override(self):
        cfg = load_config(None, {'model': {'conf': 0.5}, 'run': {'stride': 2}})
        self.assertEqual(cfg['model']['conf'], 0.5)
        self.assertEqual(cfg['run']['stride'], 2)
        self.assertEqual(cfg['model']['imgsz'], 640)

    def test_new_sections_present(self):
        cfg = load_config(None, None)
        for key in ('analytics', 'alert', 'serve', 'metrics'):
            self.assertIn(key, cfg)
        self.assertIn('segment_seconds', cfg['output'])


class TestSources(unittest.TestCase):
    def test_is_camera(self):
        self.assertTrue(is_camera(0))
        self.assertTrue(is_camera('0'))
        self.assertFalse(is_camera('data/test.mp4'))

    def test_list_videos(self):
        ensure_test_video()                      # 干净仓库里素材是现造的
        vids = list_videos('data')
        self.assertTrue(any(v.endswith('.mp4') for v in vids))
        self.assertEqual(list_videos('data/test.mp4'), ['data/test.mp4'])


class TestAnalytics(unittest.TestCase):
    def test_point_in_polygon(self):
        square = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertTrue(_inside(square, (5, 5)))
        self.assertFalse(_inside(square, (15, 5)))

    def test_region_filter(self):
        res = FakeRes(FakeBoxes([[0, 0, 2, 2], [100, 100, 120, 120]], [0, 0]))
        kept = RegionFilter([[0, 0], [50, 0], [50, 50], [0, 50]]).keep(res)
        self.assertEqual(len(kept.boxes), 1)          # 只留区域内那个

    def test_line_counter_counts_crossing(self):
        counter = LineCounter([[0, 50], [100, 50]], min_cross_frames=1)   # 关防抖，逐帧判定
        top = FakeRes(FakeBoxes([[40, 5, 60, 15]], [0], [7]))       # 框中心 y=10，线之上（- 侧）
        bottom = FakeRes(FakeBoxes([[40, 85, 60, 95]], [0], [7]))   # 中心 y=90，线之下（+ 侧）
        counter.update(top, {0: 'person'})
        self.assertIsNone(counter.update(top, {0: 'person'}))        # 同侧不算穿越
        crossed = counter.update(bottom, {0: 'person'})              # 上→下 = 由 - 侧进入 + 侧 = in
        self.assertEqual(crossed[0]['direction'], 'in')
        self.assertEqual(counter.total['in'], 1)
        crossed = counter.update(top, {0: 'person'})                 # 再穿回去 = out
        self.assertEqual(crossed[0]['direction'], 'out')
        self.assertEqual(counter.total['out'], 1)


class TestLineDebounce(unittest.TestCase):
    """越线防抖：抖动不该刷计数，连续 N 帧才认定穿越。"""

    def _res(self, cy, tid=1):
        return FakeRes(FakeBoxes([[40, cy - 5, 60, cy + 5]], [0], [tid]))

    def test_single_frame_flip_is_ignored_by_default(self):
        counter = LineCounter([[0, 50], [100, 50]])          # 默认 min_cross_frames=2
        counter.update(self._res(10), {0: 'person'})         # - 侧
        self.assertIsNone(counter.update(self._res(90), {0: 'person'}))   # 抖到 + 侧 1 帧
        self.assertEqual(counter.total['in'], 0)
        self.assertIsNone(counter.update(self._res(10), {0: 'person'}))   # 抖回去，撤销
        self.assertEqual(counter.total['in'] + counter.total['out'], 0)

    def test_two_consecutive_frames_confirm_crossing(self):
        counter = LineCounter([[0, 50], [100, 50]], min_cross_frames=2)
        counter.update(self._res(10), {0: 'person'})
        self.assertIsNone(counter.update(self._res(90), {0: 'person'}))
        crossed = counter.update(self._res(90), {0: 'person'})            # 第 2 帧确认
        self.assertEqual(crossed[0]['direction'], 'in')
        self.assertEqual(counter.total['in'], 1)
        self.assertIsNone(counter.update(self._res(90), {0: 'person'}))   # 巩固不再重复计数
        self.assertEqual(counter.total['in'], 1)


class TestAlerts(unittest.TestCase):
    def test_payload_shape(self):
        self.assertEqual(AlertChannel('k', 'wecom')._payload('x')['msgtype'], 'text')
        self.assertEqual(AlertChannel('k', 'feishu')._payload('x')['msg_type'], 'text')
        self.assertEqual(AlertChannel('u', 'generic')._payload('x'), {'text': 'x'})
        self.assertTrue(AlertChannel('k', 'wecom').url.startswith('https://qyapi.weixin.qq.com/'))

    def test_send_reaches_local_webhook(self):
        got = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                got.append(json.loads(body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        ch = AlertChannel(f'http://127.0.0.1:{srv.server_port}/hook', 'wecom')
        self.assertTrue(ch.send('测试告警'))
        self.assertEqual(ch.sent, 1)
        srv.shutdown()
        srv.server_close()
        self.assertIn('测试告警', got[0]['text']['content'])

    def test_disabled_channel_is_noop(self):
        ch = AlertChannel('')
        self.assertFalse(ch.enabled)
        self.assertFalse(ch.send('x'))


class TestOutputs(unittest.TestCase):
    def test_video_sink_roundtrip(self):
        p = ROOT / 'outputs' / '_smoke.mp4'
        sink = VideoSink(p, 25)
        for i in range(10):
            sink.write(np.full((120, 160, 3), i * 20, np.uint8))
        sink.close()
        cap = cv2.VideoCapture(str(p))
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        cap.release()
        p.unlink()

    def test_video_sink_segments(self):
        p = ROOT / 'outputs' / '_smoke_seg.mp4'
        sink = VideoSink(p, 10, segment_seconds=0.5)          # 每 5 帧切一个文件
        for i in range(12):
            sink.write(np.full((64, 64, 3), i * 10, np.uint8))
        sink.close()
        self.assertEqual(len(sink.parts), 3)
        for part in sink.parts:
            Path(part).unlink()

    def test_cleanup_old_by_max_files(self):
        d = ROOT / 'outputs' / '_cleanup'
        d.mkdir(parents=True, exist_ok=True)      # 干净克隆里 outputs/ 还不存在，必须带 parents
        for i in range(5):
            write_json(d / f'f{i}.json', {'i': i})
        removed = cleanup_old(d, max_files=2, patterns=('*.json',))
        self.assertEqual(removed, 3)
        self.assertEqual(len(list(d.glob('*.json'))), 2)
        for f in d.iterdir():
            f.unlink()
        d.rmdir()

    def test_event_sink_and_write_json(self):
        d = ROOT / 'outputs' / '_events'
        sink = EventSink(d, classes=[0], min_conf=0.5, cooldown_s=0)
        res = FakeRes(FakeBoxes([[0, 0, 10, 10]], [0], [1]))
        # conf 属性缺失时用不到；这里直接给 boxes 补 conf
        res.boxes.conf = np.array([0.9])
        img = np.zeros((20, 20, 3), np.uint8)
        self.assertTrue(sink.check(res, {0: 'person'}, 1, img))
        self.assertEqual(len(sink.events), 1)
        out = write_json(ROOT / 'outputs' / '_smoke.json', {'a': 1})
        self.assertIn('a', Path(out).read_text(encoding='utf-8'))
        Path(out).unlink()
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


class TestMetrics(unittest.TestCase):
    def test_prometheus_text(self):
        text = prometheus_text({'frames': 12, 'fps': 9.5, 'unique_tracks': 3,
                                'events': 1, 'line': {'in': 2, 'out': 1},
                                'per_class': {'person': 7}})
        self.assertIn('yolo_frames_inferred_total 12', text)
        self.assertIn('yolo_line_crossings_total{direction="in"} 2', text)
        self.assertIn('yolo_detections_total{class="person"', text)


class TestPipelineEndToEnd(unittest.TestCase):
    """真跑一遍：权重存在时用测试视频跑 3 帧，校验统计与产物。"""

    def test_detect_video(self):
        if not (ROOT / 'weights' / 'yolo11n.pt').exists():
            self.skipTest('缺少 weights/yolo11n.pt（bash scripts/get_weights.sh 可拉取）')
        from yolo_app.detector import Detector
        from yolo_app.logging_setup import get_logger
        from yolo_app.pipeline import process

        cfg = load_config(None, {'run': {'max_frames': 3}, 'output': {'name': '_smoke_e2e.mp4', 'json': False}})
        det = Detector(cfg)
        summary = process(str(ensure_test_video()), cfg, det, get_logger('WARNING'))
        # 无论素材真伪都必须成立的"管道机制"：帧数、产物、统计字段
        self.assertEqual(summary['frames_inferred'], 3)
        self.assertTrue(Path(summary['output_video']).exists())
        self.assertIn('per_class_frames', summary)
        self.assertIn('unique_tracks', summary)
        if REAL_FIXTURE:      # 只有真实场景素材才断言"检出人"（合成图形没有目标）
            self.assertGreater(summary['per_class_frames'].get('person', 0), 0)
        Path(summary['output_video']).unlink()


class TestModels(unittest.TestCase):
    """模型清单/下载/对照表：不联网，只测纯逻辑。"""

    def test_repo_for(self):
        from yolo_app import models as M
        self.assertEqual(M.repo_for('yolo26n.pt'), 'YOLO26')
        self.assertEqual(M.repo_for('yolo11s.pt'), 'YOLO11')
        self.assertEqual(M.repo_for('yolov8n.pt'), 'YOLOv8')
        self.assertEqual(M.repo_for('best.pt'), 'YOLO11')        # 认不出来时兜底

    def test_presets_are_wellformed(self):
        from yolo_app import models as M
        for name, (size, speed, note) in M.PRESETS.items():
            self.assertTrue(name.endswith('.pt'))
            self.assertTrue(size.endswith('MB'))
            self.assertGreater(0, -speed)                        # speed 是正数
            self.assertTrue(note)

    def test_local_weights_and_describe(self):
        from yolo_app import models as M
        names = M.local_weights()
        self.assertIsInstance(names, list)
        self.assertTrue(M.describe('yolo11n.pt'))                # 预设里有说明
        self.assertTrue(M.describe('__not_exist__.pt'))          # 不存在也不抛异常

    def test_format_table_with_error_row(self):
        from yolo_app import models as M
        text = M.format_table([{'model': 'a.pt', 'fps': 10.0, 'elapsed': 1.0,
                                'detections': {'person': 5}, 'total': 5, 'unique_tracks': 2},
                               {'model': 'b.pt', 'error': '载入失败'}])
        self.assertIn('a.pt', text)
        self.assertIn('person:5', text)
        self.assertIn('失败', text)
        self.assertEqual(M.format_table([]), '(无结果)')



class TestConfigOverride(unittest.TestCase):
    """覆盖合并：0 / False / 空字符串这类"假值"必须能覆盖掉默认值。"""

    def test_falsy_override_wins(self):
        cfg = load_config(None, {'run': {'max_frames': 0}, 'output': {'video': False},
                                 'alert': {'webhook': ''}, 'model': {'conf': 0.1}})
        self.assertEqual(cfg['run']['max_frames'], 0)
        self.assertFalse(cfg['output']['video'])
        self.assertEqual(cfg['alert']['webhook'], '')
        self.assertEqual(cfg['model']['conf'], 0.1)

    def test_new_knobs_have_defaults(self):
        cfg = load_config(None, None)
        self.assertEqual(cfg['analytics']['track_ttl'], 300)
        self.assertEqual(cfg['alert']['dedupe_s'], 3.0)
        self.assertEqual(cfg['camera']['buffer_size'], 1)
        self.assertEqual(cfg['serve']['token'], '')


class TestLineCounterMemory(unittest.TestCase):
    """长视频里 track 状态必须有界，否则内存只增不减。"""

    def _box(self, cx, cy, tid):
        return FakeBoxes([[cx - 5, cy - 5, cx + 5, cy + 5]], [0], [tid])

    def test_track_ttl_evicts_stale_tracks(self):
        counter = LineCounter([[0, 50], [100, 50]], track_ttl=10)
        for frame in range(1, 201):
            counter.update(FakeRes(self._box(50, 90, frame)), {0: 'person'}, frame_no=frame)
            self.assertLessEqual(counter.active_tracks, 11)      # ttl=10 → 最多 11 个存活
        self.assertLessEqual(len(counter.prev), 11)

    def test_no_ttl_keeps_legacy_behaviour(self):
        counter = LineCounter([[0, 50], [100, 50]], track_ttl=0)
        for frame in range(1, 50):
            counter.update(FakeRes(self._box(50, 90, frame)), {0: 'person'}, frame_no=frame)
        self.assertEqual(counter.active_tracks, 49)


class TestAlertDedupe(unittest.TestCase):
    def test_same_text_within_window_is_skipped(self):
        ch = AlertChannel('http://127.0.0.1:1/x', 'generic', timeout=1, dedupe_s=30)
        self.assertFalse(ch.send('同一个告警'))          # 不可达，但去重键已登记
        self.assertFalse(ch.send('同一个告警'))
        self.assertEqual(ch.skipped, 1)

    def test_dedupe_off_by_default(self):
        ch = AlertChannel('http://127.0.0.1:1/x', 'generic', timeout=1)
        self.assertEqual(ch.dedupe_s, 0)


class TestMetricsExtra(unittest.TestCase):
    def test_latency_and_alert_metrics(self):
        text = prometheus_text({'frames': 5, 'infer_ms_avg': 12.345, 'infer_ms_max': 40.0,
                                'reconnects': 2, 'alert_sent': 3, 'alert_failed': 1, 'alert_skipped': 4})
        self.assertIn('yolo_infer_milliseconds_avg 12.35', text)
        self.assertIn('yolo_infer_milliseconds_max 40.00', text)
        self.assertIn('yolo_reconnects_total 2', text)
        self.assertIn('yolo_alert_skipped_total 4', text)


class TestSourceReconnect(unittest.TestCase):
    """断流重连：失败几帧后能恢复，且成功后计数清零，不会"用光"重连额度。"""

    def test_live_source_recovers(self):
        from unittest import mock
        import yolo_app.reader as R
        import yolo_app.sources as S

        state = {'reads': 0}

        class FakeCap:
            def __init__(self, uri):
                self.uri = uri

            def isOpened(self):
                return True

            def read(self):
                state['reads'] += 1
                if state['reads'] <= 1:              # 第一次读失败，模拟掉线
                    return False, None
                return True, np.zeros((4, 4, 3), np.uint8)

            def get(self, prop):
                return 25.0

            def set(self, *a):
                return True

            def release(self):
                pass

        fake_cv2 = mock.Mock()
        fake_cv2.VideoCapture = FakeCap
        fake_cv2.CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        fake_cv2.CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        fake_cv2.CAP_PROP_FPS = cv2.CAP_PROP_FPS
        fake_cv2.CAP_PROP_BUFFERSIZE = cv2.CAP_PROP_BUFFERSIZE

        with mock.patch.object(R, 'cv2', fake_cv2):
            src = S.VideoSource('rtsp://demo', {'buffer_size': 1}, reconnect=3,
                                retry_delay=0, reader='opencv')
            frame = src.read()
            self.assertIsNotNone(frame)
            self.assertEqual(src.reconnects, 1)
            self.assertIsNotNone(src.read())                  # 恢复后继续正常读
            self.assertEqual(src.reconnects, 1)               # 没有继续累加
            src.release()

    def test_offline_file_does_not_reconnect(self):
        from unittest import mock
        import yolo_app.reader as R
        import yolo_app.sources as S

        class DeadCap:
            def isOpened(self):
                return True

            def read(self):
                return False, None

            def get(self, prop):
                return 0

            def set(self, *a):
                return True

            def release(self):
                pass

        fake_cv2 = mock.Mock()
        fake_cv2.VideoCapture = lambda uri: DeadCap()
        with mock.patch.object(R, 'cv2', fake_cv2):
            src = S.VideoSource('data/not-exist.mp4', {}, reconnect=5, retry_delay=0)
            self.assertIsNone(src.read())
            self.assertEqual(src.reconnects, 0)


class TestPipelineResilience(unittest.TestCase):
    """异常与停机路径：录像必须收尾，统计必须照常产出。"""

    def _skip_without_weights(self):
        if not (ROOT / 'weights' / 'yolo11n.pt').exists():
            self.skipTest('缺少 weights/yolo11n.pt')

    def test_error_mid_run_still_finalises_video(self):
        self._skip_without_weights()
        from yolo_app.detector import Detector
        from yolo_app.logging_setup import get_logger
        from yolo_app.pipeline import process

        cfg = load_config(None, {'run': {'max_frames': 5},
                                 'output': {'name': '_resil.mp4', 'json': False,
                                            'events': {'enable': False}}})
        det = Detector(cfg)
        real = det.infer
        calls = {'n': 0}

        def boom(frame):
            calls['n'] += 1
            if calls['n'] >= 2:
                raise RuntimeError('模拟推理崩溃')
            return real(frame)

        det.infer = boom
        with self.assertRaises(RuntimeError):
            process(str(ensure_test_video()), cfg, det, get_logger('ERROR'))

        out = ROOT / 'outputs' / '_resil.mp4'
        cap = cv2.VideoCapture(str(out))          # 没 close 的话 mp4 缺 moov，isOpened 为 False
        self.assertTrue(cap.isOpened())
        cap.release()
        out.unlink()

    def test_stop_event_interrupts_cleanly(self):
        self._skip_without_weights()
        from yolo_app.detector import Detector
        from yolo_app.logging_setup import get_logger
        from yolo_app.pipeline import process

        cfg = load_config(None, {'output': {'name': '_stop.mp4', 'json': False,
                                            'events': {'enable': False}}})
        det = Detector(cfg)
        real = det.infer
        stop = threading.Event()

        def infer_then_stop(frame):
            stop.set()                            # 第一帧处理完就要停
            return real(frame)

        det.infer = infer_then_stop
        summary = process(str(ensure_test_video()), cfg, det, get_logger('ERROR'), stop_event=stop)
        self.assertTrue(summary.get('interrupted'))
        self.assertGreaterEqual(summary['frames_inferred'], 1)
        out = Path(summary['output_video'])
        self.assertTrue(out.exists())
        out.unlink()


class TestConfFilter(unittest.TestCase):
    """双阈值：低分框只送跟踪，不进统计。"""

    def _res(self, confs):
        boxes = [[0, 0, 10, 10] for _ in confs]
        return FakeRes(FakeBoxes(boxes, [0] * len(confs), conf=confs))

    def test_filters_low_confidence(self):
        from yolo_app.analytics import ConfFilter
        f = ConfFilter(0.5)
        out = f.keep(self._res([0.9, 0.2, 0.5]))
        self.assertEqual(len(out.boxes), 2)                 # 0.5 边界算保留
        self.assertEqual(f.dropped, 1)

    def test_zero_means_disabled(self):
        from yolo_app.analytics import ConfFilter
        f = ConfFilter(0)
        self.assertFalse(f.enabled)
        self.assertEqual(len(f.keep(self._res([0.1])).boxes), 1)

    def test_missing_conf_is_not_dropped(self):
        from yolo_app.analytics import ConfFilter
        f = ConfFilter(0.9)
        out = f.keep(FakeRes(FakeBoxes([[0, 0, 1, 1]], [0])))   # 无 conf 属性
        self.assertEqual(len(out.boxes), 1)
        self.assertEqual(f.dropped, 0)


class TestStationaryGate(unittest.TestCase):
    """静止降频：目标不动时抽帧推理，一动就必须恢复逐帧。"""

    def _res(self, cx, cy, tid=1):
        return FakeRes(FakeBoxes([[cx - 5, cy - 5, cx + 5, cy + 5]], [0], [tid]))

    def test_static_target_skips_frames(self):
        from yolo_app.analytics import StationaryGate
        gate = StationaryGate(frames=3, interval=5, min_move=2)
        res = self._res(50, 50)
        decisions = [gate.should_infer(res, f) for f in range(1, 21)]
        self.assertTrue(all(decisions[:2]))                 # 先观察 2 帧
        self.assertLessEqual(sum(decisions), 6)             # 20 帧里最多推理 6 次
        self.assertGreaterEqual(gate.skipped, 10)

    def test_moving_target_always_infers(self):
        from yolo_app.analytics import StationaryGate
        gate = StationaryGate(frames=2, interval=5, min_move=1)
        for frame in range(1, 12):
            self.assertTrue(gate.should_infer(self._res(50 + frame * 10, 50), frame))

    def test_disabled_by_default(self):
        from yolo_app.analytics import StationaryGate
        gate = StationaryGate()
        self.assertFalse(gate.enabled)
        self.assertTrue(gate.should_infer(None, 1))


class TestCliOverrides(unittest.TestCase):
    def test_report_conf_flag(self):
        from yolo_app.cli import build_overrides, build_parser
        a = build_parser().parse_args(['check', '--report-conf', '0.6'])
        self.assertEqual(build_overrides(a)['model']['report_conf'], 0.6)


def _writer_cmd(w, h, n, value=7):
    """生成一个"往 stdout 写 n 帧 rawvideo"的 python 命令，替代真实 ffmpeg 做单测。"""
    code = (f"import sys; buf=bytes([{value}])*({w}*{h}*3); "
            f"sys.stdout.buffer.write(buf*{n}); sys.stdout.flush()")
    return [sys.executable, '-c', code]


class TestReaderBackends(unittest.TestCase):
    """读取后端选择与 ffprobe 解析：纯逻辑，不启动真实流。"""

    def test_choose_backend_matrix(self):
        from yolo_app.reader import choose_backend
        self.assertEqual(choose_backend('rtsp://x', 'auto', has_ffmpeg=True), 'ffmpeg')
        self.assertEqual(choose_backend('rtsp://x', 'auto', has_ffmpeg=False), 'opencv')
        self.assertEqual(choose_backend('data/test.mp4', 'auto', has_ffmpeg=True), 'opencv')
        self.assertEqual(choose_backend('data/test.mp4', 'ffmpeg', has_ffmpeg=False), 'ffmpeg')
        self.assertEqual(choose_backend('rtsp://x', 'opencv', has_ffmpeg=True), 'opencv')

    def test_parse_rate(self):
        from yolo_app.reader import parse_rate
        self.assertAlmostEqual(parse_rate('30000/1001'), 29.97, places=2)
        self.assertEqual(parse_rate('25'), 25.0)
        self.assertIsNone(parse_rate('N/A'))

    def test_probe_missing_file(self):
        from yolo_app.reader import probe_stream
        self.assertIsNone(probe_stream('/tmp/definitely-not-a-video-xyz.mp4', timeout=5))


class TestFFmpegReader(unittest.TestCase):

    """FFmpeg 后端：EOF、僵死重启、队列丢弃（用 fake 子进程，不依赖真实 RTSP）。"""

    info = {'width': 4, 'height': 4, 'fps': 25, 'total': 0}

    def _make(self, cmd, uri='rtsp://demo', **kw):
        from yolo_app.reader import FFmpegReader
        return FFmpegReader(uri, self.info, cmd_builder=cmd, retry_delay=0, **kw)

    def test_file_eof(self):
        r = self._make(lambda: _writer_cmd(4, 4, 3), uri='/tmp/fake.mp4', buffer_size=4)
        frames = []
        for _ in range(5):
            f = r.read()
            if f is None:
                break
            frames.append(f)
        r.release()
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].shape, (4, 4, 3))

    def test_stalled_stream_restarts(self):
        calls = {'n': 0}

        def builder():
            calls['n'] += 1
            if calls['n'] == 1:
                return [sys.executable, '-c', 'import time; time.sleep(30)']
            return _writer_cmd(4, 4, 2)

        r = self._make(builder, read_timeout=0.5, reconnect=1, buffer_size=2)
        try:
            frame = r.read()                       # 等 0.5s → 判定僵死 → 重启 → 读到帧
            self.assertIsNotNone(frame)
            self.assertEqual(r.reconnects, 1)
        finally:
            r.release()

    def test_slow_consumer_drops_old_frames(self):
        r = self._make(lambda: _writer_cmd(4, 4, 300), buffer_size=1, read_timeout=5)
        try:
            time.sleep(0.5)                        # 让生产端跑满队列
            self.assertGreaterEqual(r.dropped, 1)  # 有界队列 + 丢最旧
        finally:
            r.release()

    def test_release_stops_thread(self):
        r = self._make(lambda: _writer_cmd(4, 4, 50), read_timeout=5)
        r.release()
        self.assertIsNone(r._proc)


class TestVideoSourceFacade(unittest.TestCase):
    def test_ffmpeg_matches_opencv_on_file(self):
        """同一文件两后端解码结果应基本一致（仅色彩转换舍入差异）。"""
        if not shutil.which('ffmpeg'):
            self.skipTest('未安装 ffmpeg')
        from yolo_app.reader import VideoReader
        video = str(ensure_test_video())
        a = VideoReader(video, mode='opencv')
        fa = a.read()
        a.release()
        b = VideoReader(video, mode='ffmpeg')
        fb = b.read()
        b.release()
        self.assertEqual(fa.shape, fb.shape)
        diff = np.abs(fa.astype(int) - fb.astype(int))
        self.assertLessEqual(int(diff.max()), 8)          # swscale 与 OpenCV 的舍入差异
        self.assertLess(float(diff.mean()), 2.0)

    def test_opencv_backend_for_file(self):
        from yolo_app.sources import VideoSource
        ensure_test_video()
        src = VideoSource(str(ROOT / 'data' / 'test.mp4'))
        try:
            self.assertEqual(src.backend, 'opencv')
            self.assertTrue(src.opened)
            self.assertGreater(src.width, 0)
            self.assertIsNotNone(src.read())
        finally:
            src.release()


class TestMotionGate(unittest.TestCase):
    """运动门控：没动就不推理；ROI 外运动忽略。"""

    def test_static_frames_have_no_motion(self):
        from yolo_app.motion import MotionGate
        gate = MotionGate(threshold=20, min_area_ratio=0.001, blur=0)
        frame = np.zeros((120, 160, 3), np.uint8)
        self.assertTrue(gate.update(frame))            # 首帧没有参照，必须推理
        for _ in range(5):
            self.assertFalse(gate.update(frame))

    def test_moving_object_detected(self):
        from yolo_app.motion import MotionGate
        gate = MotionGate(threshold=20, min_area_ratio=0.001, blur=0)
        a = np.zeros((120, 160, 3), np.uint8)
        gate.update(a)
        b = a.copy()
        b[40:80, 40:80] = 255
        self.assertTrue(gate.update(b))

    def test_roi_mask_ignores_outside_motion(self):
        from yolo_app.motion import MotionGate
        gate = MotionGate(threshold=20, min_area_ratio=0.001, blur=0,
                          roi=[[0, 0], [50, 0], [50, 120], [0, 120]])
        a = np.zeros((120, 160, 3), np.uint8)
        gate.update(a)
        outside = a.copy()
        outside[40:80, 120:150] = 255                  # ROI 外运动
        self.assertFalse(gate.update(outside))
        inside = outside.copy()
        inside[40:80, 10:40] = 255                     # ROI 内运动
        self.assertTrue(gate.update(inside))

    def test_overlay_boxes_does_not_mutate_source(self):
        from yolo_app.motion import overlay_boxes
        frame = np.zeros((60, 60, 3), np.uint8)
        out = overlay_boxes(frame, FakeRes(FakeBoxes([[10, 10, 40, 40]], [0])))
        self.assertTrue((out != frame).any())
        self.assertFalse((frame != 0).any())


class TestMotionIntegration(unittest.TestCase):
    def test_static_video_skips_most_inference(self):
        if not (ROOT / 'weights' / 'yolo11n.pt').exists():
            self.skipTest('缺少 weights/yolo11n.pt')
        import tempfile

        from yolo_app.detector import Detector
        from yolo_app.logging_setup import get_logger
        from yolo_app.pipeline import process

        path = Path(tempfile.gettempdir()) / '_static_motion.mp4'
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (160, 120))
        for _ in range(30):
            vw.write(np.zeros((120, 160, 3), np.uint8))
        vw.release()
        cfg = load_config(None, {'motion': {'enable': True},
                                 'output': {'video': False, 'json': False,
                                            'events': {'enable': False}}})
        try:
            s = process(str(path), cfg, Detector(cfg), get_logger('ERROR'))
        finally:
            path.unlink(missing_ok=True)
        self.assertGreaterEqual(s['frames_inference_skipped'], 20)
        self.assertLessEqual(s['frames_inferred'], 5)
        self.assertGreater(s['skip_reasons']['motion'], 0)


class TestRoiCropGeometry(unittest.TestCase):
    """ROI 裁剪的几何部分：外扩、裁剪、坐标平移。"""

    def test_expand_box_with_margin(self):
        from yolo_app.analytics import expand_box
        self.assertEqual(expand_box([[100, 100], [200, 100], [200, 200], [100, 200]], 0.1),
                         (90, 90, 210, 210))

    def test_expand_box_clipped_to_frame(self):
        from yolo_app.analytics import expand_box
        self.assertEqual(expand_box([[0, 0], [50, 0], [50, 50], [0, 50]], 0.5, width=40, height=40),
                         (0, 0, 40, 40))

    def test_expand_box_no_margin(self):
        from yolo_app.analytics import expand_box
        self.assertEqual(expand_box([[10, 10], [20, 20]], 0), (10, 10, 20, 20))

    def test_remap_boxes_shifts_in_place(self):
        from yolo_app.analytics import remap_boxes
        res = FakeRes(FakeBoxes([[0, 0, 10, 10]], [0]))
        remap_boxes(res, 100, 50)
        self.assertEqual(res.boxes.xyxy.tolist(), [[100.0, 50.0, 110.0, 60.0]])

    def test_remap_handles_empty(self):
        from yolo_app.analytics import remap_boxes
        res = FakeRes(FakeBoxes([], []))
        self.assertIs(remap_boxes(res, 10, 10), res)


class TestRoiCropIntegration(unittest.TestCase):
    def test_cropped_run_detects_inside_roi(self):
        if not (ROOT / 'weights' / 'yolo11n.pt').exists():
            self.skipTest('缺少 weights/yolo11n.pt')
        from yolo_app.detector import Detector
        from yolo_app.logging_setup import get_logger
        from yolo_app.pipeline import process

        cfg = load_config(None, {'run': {'max_frames': 3},
                                 'output': {'video': False, 'json': False,
                                            'events': {'enable': False}},
                                 'analytics': {'roi': [[0, 0], [320, 0], [320, 240], [0, 240]],
                                               'roi_crop': True, 'roi_imgsz': 320}})
        summary = process(str(ensure_test_video()), cfg, Detector(cfg), get_logger('ERROR'))
        self.assertEqual(summary['frames_inferred'], 3)
        if REAL_FIXTURE:                       # 只有真实场景素材才断言检出了目标
            self.assertGreater(sum(summary['per_class_frames'].values()), 0)


class TestWebApp(unittest.TestCase):
    """/healthz 与 token 鉴权：面板对外暴露时这两条是底线。"""

    def test_healthz_and_token_guard(self):
        if not (ROOT / 'weights' / 'yolo11n.pt').exists():
            self.skipTest('缺少 weights/yolo11n.pt')
        try:
            from fastapi.testclient import TestClient
        except Exception as e:                       # 干净环境没装 httpx 时跳过，不误报失败
            self.skipTest(f'缺少 TestClient 依赖: {e}')
        from yolo_app.detector import Detector
        from yolo_app.logging_setup import get_logger
        from yolo_app.webapp import create_app

        cfg = load_config(None, {'run': {'max_frames': 2},
                                 'output': {'video': False, 'json': False,
                                            'events': {'enable': False}},
                                 'serve': {'token': 'secret'}})
        app = create_app(cfg, Detector(cfg), get_logger('ERROR'), str(ensure_test_video()))
        with TestClient(app) as c:                   # 退出时触发 shutdown，验证优雅停机路径
            self.assertEqual(c.get('/healthz').status_code, 200)
            self.assertEqual(c.get('/stats').status_code, 401)       # 无 token
            r = c.get('/stats', params={'token': 'secret'})
            for _ in range(100):                     # 等检测线程推出第一帧统计
                if 'frames' in r.json():
                    break
                time.sleep(0.1)
                r = c.get('/stats', params={'token': 'secret'})
            self.assertEqual(r.status_code, 200)
            self.assertIn('frames', r.json())
            m = c.get('/metrics', headers={'X-Token': 'secret'})
            self.assertEqual(m.status_code, 200)
            self.assertIn('yolo_up 1', m.text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
