"""锚点（center / bottom）与线段相交越线的回归测试。

覆盖四件事：
  · 线段相交的几何判定：正交穿越、不穿越、端点接触、共线重叠；
  · LineCounter 新语义：一帧大跨度不漏计、冷却期内抖动不重复计、方向 in/out；
  · analytics.anchor：bottom 取框底边中点，跨线口径与 center 明确不同；
  · pipeline 接线：analytics.anchor=bottom 经 process() 真实构造路径后生效。
"""
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.analytics import (LineCounter, RegionFilter, _anchor_xy,
                                _segments_intersect)
from yolo_app.config import load_config

ROOT = Path(__file__).resolve().parent.parent


class FakeBoxes:
    """最小 boxes 替身：只实现后处理真正用到的 xyxy / cls / id / 索引。"""

    def __init__(self, xyxy, cls, ids=None):
        self.xyxy = np.array(xyxy, dtype=float)
        self.cls = np.array(cls, dtype=float)
        self.id = None if ids is None else np.array(ids, dtype=float)

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, idx):
        return FakeBoxes(self.xyxy[idx].tolist(), self.cls[idx].tolist(),
                         None if self.id is None else self.id[idx].tolist())


class FakeRes:
    """pipeline 会调用 plot()，这里顺便把 Results 需要的字段补齐。"""

    def __init__(self, boxes, image=None):
        self.boxes = boxes
        self.orig_img = image
        self.masks = None
        self.keypoints = None

    def plot(self, **kwargs):
        return np.zeros_like(self.orig_img)


def _res(cx, cy, tid=1):
    """中心在 (cx, cy) 的 10×10 框。"""
    return FakeRes(FakeBoxes([[cx - 5, cy - 5, cx + 5, cy + 5]], [0], [tid]))


def _box_res(xyxy, tid=1):
    return FakeRes(FakeBoxes([xyxy], [0], [tid]))


LINE = [[0, 50], [100, 50]]


class TestSegmentIntersection(unittest.TestCase):
    """叉积法的四种典型情形。"""

    def test_orthogonal_crossing(self):
        self.assertTrue(_segments_intersect((50, 10), (50, 90), LINE[0], LINE[1]))
        self.assertTrue(_segments_intersect((10, 50), (90, 50), (50, 0), (50, 100)))

    def test_no_crossing(self):
        self.assertFalse(_segments_intersect((50, 10), (50, 40), LINE[0], LINE[1]))
        self.assertFalse(_segments_intersect((50, 60), (50, 90), LINE[0], LINE[1]))
        # 与线段所在直线共线，但落在计数线之外，不算相交
        self.assertFalse(_segments_intersect((110, 50), (120, 50), LINE[0], LINE[1]))

    def test_endpoint_touch(self):
        self.assertTrue(_segments_intersect((50, 50), (50, 90), LINE[0], LINE[1]))
        self.assertTrue(_segments_intersect((50, 10), (50, 50), LINE[0], LINE[1]))
        self.assertTrue(_segments_intersect((0, 50), (50, 50), LINE[0], LINE[1]))

    def test_collinear_overlap(self):
        self.assertTrue(_segments_intersect((10, 50), (90, 50), LINE[0], LINE[1]))
        self.assertTrue(_segments_intersect((0, 50), (100, 50), LINE[0], LINE[1]))


class TestAnchorPoint(unittest.TestCase):
    def test_center_and_bottom(self):
        box = [10, 20, 30, 40]
        self.assertEqual(_anchor_xy(box, 'center'), (20, 30))
        self.assertEqual(_anchor_xy(box, 'bottom'), (20, 40))

    def test_invalid_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            LineCounter(LINE, anchor='middle')


class TestLineDirection(unittest.TestCase):
    """方向沿用 - 侧 → + 侧为 in，+ 侧 → - 侧为 out。"""

    def test_negative_to_positive_is_in(self):
        counter = LineCounter(LINE, min_cross_frames=1)
        counter.update(_res(50, 10), {0: 'person'})          # 首帧在 - 侧
        crossed = counter.update(_res(50, 90), {0: 'person'})
        self.assertEqual(crossed[0]['direction'], 'in')
        self.assertEqual(counter.total, {'in': 1, 'out': 0})

    def test_positive_to_negative_is_out(self):
        counter = LineCounter(LINE, min_cross_frames=1)
        counter.update(_res(50, 90), {0: 'person'})          # 首帧在 + 侧
        crossed = counter.update(_res(50, 10), {0: 'person'})
        self.assertEqual(crossed[0]['direction'], 'out')
        self.assertEqual(counter.total, {'in': 0, 'out': 1})


class TestLineCounterCrossing(unittest.TestCase):
    def test_large_jump_within_one_frame_counts_once(self):
        counter = LineCounter(LINE)                          # 默认冷却 2 帧
        counter.update(_res(50, 10), {0: 'person'})
        crossed = counter.update(_res(50, 90), {0: 'person'})   # y 10 → 90，线在 50
        self.assertEqual(len(crossed), 1)
        self.assertEqual(crossed[0]['direction'], 'in')
        self.assertEqual(counter.total['in'], 1)

    def test_jitter_within_cooldown_counts_once(self):
        counter = LineCounter(LINE, min_cross_frames=2)
        counter.update(_res(50, 10), {0: 'person'})          # 帧 1：- 侧
        counter.update(_res(50, 90), {0: 'person'})          # 帧 2：穿到 + 侧，计 1 次
        self.assertIsNone(counter.update(_res(50, 10), {0: 'person'}))   # 帧 3：冷却期内抖回
        self.assertEqual(counter.total['in'], 1)
        self.assertEqual(counter.total['out'], 0)

    def test_endpoint_contact_counts(self):
        counter = LineCounter(LINE, min_cross_frames=1)
        counter.update(_res(50, 50), {0: 'person'})          # 首帧锚点正好压线
        crossed = counter.update(_res(50, 90), {0: 'person'})   # 从线上进入 + 侧
        self.assertEqual(crossed[0]['direction'], 'in')
        self.assertEqual(counter.total['in'], 1)

    def test_collinear_overlap_counts_once(self):
        counter = LineCounter(LINE, min_cross_frames=1)
        counter.update(_res(10, 50), {0: 'person'})          # 首帧在线上
        crossed = counter.update(_res(90, 50), {0: 'person'})   # 沿线上滑过整条线：共线重叠
        self.assertEqual(len(crossed), 1)
        self.assertEqual(sum(counter.total.values()), 1)
        # 仍沿线上滑动，不应每帧重复计数
        self.assertIsNone(counter.update(_res(80, 50), {0: 'person'}))
        self.assertEqual(sum(counter.total.values()), 1)


class TestBottomAnchor(unittest.TestCase):
    """bottom 锚点看框底边中点；这与 center 的跨线口径可能不同。"""

    def test_bottom_crosses_while_center_does_not(self):
        # 线在 y=50；框从 [10,40] 移到 [30,60]，中心 25→45 未跨线，底边 40→60 已跨线
        first = _box_res([40, 10, 60, 40])
        second = _box_res([40, 30, 60, 60])

        center = LineCounter(LINE)
        center.update(first, {0: 'person'})
        self.assertIsNone(center.update(second, {0: 'person'}))
        self.assertEqual(center.total, {'in': 0, 'out': 0})

        bottom = LineCounter(LINE, anchor='bottom')
        bottom.update(first, {0: 'person'})
        crossed = bottom.update(second, {0: 'person'})
        self.assertEqual(crossed[0]['direction'], 'in')
        self.assertEqual(bottom.total, {'in': 1, 'out': 0})

    def test_region_filter_uses_anchor(self):
        # ROI 只覆盖 y<30：中心 (50,20) 在区内，底边 (50,40) 在区外
        roi = [[0, 0], [100, 0], [100, 30], [0, 30]]
        res = FakeRes(FakeBoxes([[40, 0, 60, 40]], [0]))
        kept_center = RegionFilter(roi).keep(res)
        self.assertEqual(len(kept_center.boxes), 1)

        res = FakeRes(FakeBoxes([[40, 0, 60, 40]], [0]))
        kept_bottom = RegionFilter(roi, anchor='bottom').keep(res)
        self.assertEqual(len(kept_bottom.boxes), 0)


class TestAnchorConfig(unittest.TestCase):
    def test_default_is_center(self):
        self.assertEqual(load_config(None)['analytics']['anchor'], 'center')

    def test_default_yaml_has_anchor(self):
        cfg = load_config(str(ROOT / 'configs' / 'default.yaml'))
        self.assertEqual(cfg['analytics']['anchor'], 'center')

    def test_bottom_can_be_overridden(self):
        cfg = load_config(None, {'analytics': {'anchor': 'bottom'}})
        self.assertEqual(cfg['analytics']['anchor'], 'bottom')


class TestTrackTTL(unittest.TestCase):
    """track_ttl 淘汰逻辑在新计数语义下仍然有效。"""

    def test_stale_track_is_evicted_and_reappears_as_new(self):
        counter = LineCounter(LINE, track_ttl=2)
        counter.update(_res(50, 10, tid=1), {0: 'person'}, frame_no=1)
        for frame in range(2, 6):                            # 用别的 track 把时间推过 ttl
            counter.update(_res(50, 10, tid=99), {0: 'person'}, frame_no=frame)
        self.assertNotIn(1, counter.prev)                    # 老 track 已被回收
        self.assertLessEqual(counter.active_tracks, 2)

        # 同 id 再来时只能当新 track：首帧没有 P_prev，不产生穿越
        self.assertIsNone(counter.update(_res(50, 90, tid=1), {0: 'person'}, frame_no=6))
        self.assertEqual(counter.total, {'in': 0, 'out': 0})


class _MovingDetector:
    """不加载模型：按调用次数返回预设框，制造确定的跨线/ROI 场景。"""

    names = {0: 'person'}
    weights = 'fake.pt'
    track = True
    device_name = 'cpu'

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def infer(self, frame, imgsz=None):
        box = self.boxes[min(self.calls, len(self.boxes) - 1)]
        self.calls += 1
        return FakeRes(FakeBoxes([box], [0], [1]), frame)


def _make_video(path, frames=2, size=(100, 100)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
    if not writer.isOpened():
        raise RuntimeError('测试视频编码器不可用')
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


class TestPipelineAnchorWiring(unittest.TestCase):
    """配置里的 analytics.anchor 必须真正接到 pipeline 的组件上。

    这条用例走 process() 的真实构造路径：若 pipeline 漏传 anchor，
    bottom 会退化成 center，下面的越线计数与 ROI 过滤断言都会失败。
    """

    def _run_pipeline(self, analytics, boxes):
        from yolo_app.pipeline import process

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root / 'src.mp4'
            _make_video(video)
            cfg = load_config(None, {
                'run': {'max_frames': 2},
                'output': {'dir': str(root / 'outputs'), 'video': False, 'json': False,
                           'events': {'enable': False, 'db': ''}},
                'analytics': analytics,
                'alert': {'webhook': ''},
                'log': {'level': 'ERROR'},
            })
            logger = logging.getLogger('test_anchors')
            logger.setLevel(logging.ERROR)
            return process(str(video), cfg, _MovingDetector(boxes), logger)

    def test_bottom_anchor_reaches_pipeline(self):
        # 场景 1：中心 25→45 未跨线，底边 40→60 跨线；bottom 应计 1 次 in
        summary = self._run_pipeline(
            {'line': LINE, 'anchor': 'bottom'},
            [[40, 10, 60, 40], [40, 30, 60, 60]],
        )
        self.assertEqual(summary['line_total'], {'in': 1, 'out': 0})
        self.assertEqual(summary['crossings'][0]['direction'], 'in')

        # 场景 2：ROI 只覆盖 y<30，中心 (50,20) 在区内、底边 (50,40) 在区外；
        # bottom 生效时框应被 RegionFilter 剔除，center 则会保留。
        summary = self._run_pipeline(
            {'roi': [[0, 0], [100, 0], [100, 30], [0, 30]], 'anchor': 'bottom'},
            [[40, 0, 60, 40]],
        )
        self.assertEqual(summary['frames_inferred'], 2)
        self.assertEqual(summary['per_class_frames'], {})


if __name__ == '__main__':
    unittest.main()
