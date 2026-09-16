"""冒烟测试：python -m unittest discover -s tests -v（不起框架，够用就行）。"""
import json
import sys
import threading
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

    def __init__(self, xyxy, cls, ids=None):
        self.xyxy = np.array(xyxy, dtype=float)
        self.cls = np.array(cls, dtype=float)
        self.id = None if ids is None else np.array(ids, dtype=float)

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, i):
        return FakeBoxes(self.xyxy[i].tolist(), self.cls[i].tolist(),
                         None if self.id is None else self.id[i].tolist())


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
        counter = LineCounter([[0, 50], [100, 50]])
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
