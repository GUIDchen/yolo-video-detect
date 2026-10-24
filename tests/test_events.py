"""事件落库 / 告警 outbox 重试 / 异步抓拍写盘的单测（临时目录，不联网）。

覆盖点：
  · EventStore 建表、增查、计数、重开同路径后 pending 仍在；
  · outbox 认领、mark_sent、指数退避与 dead 阈值；
  · 数据库不可写时所有 API 只返回空值，不抛异常；
  · EventSink 异步写盘：flush 后文件数与 events 一致；队列满丢最旧且不阻塞；
  · AlertChannel.send 失败（monkeypatch 返回 False）时 drain_outbox 不抛异常、attempts 增长；
  · pipeline 端到端：越线/抓拍落库、outbox 留队、summary 新字段。
"""
import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.alerts import AlertChannel, drain_outbox
from yolo_app.config import load_config
from yolo_app.logging_setup import get_logger
from yolo_app.metrics import prometheus_text
from yolo_app.outputs import EventSink
from yolo_app.store import EventStore


def _query(path, sql, params=()):
    """只读查询辅助：测试里直接看 SQLite，避免为测试给 EventStore 加临时 API。"""
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


class FakeBoxes:
    """够 EventSink / LineCounter 用的最小 boxes 替身。"""

    def __init__(self, xyxy, cls, conf=None, ids=None):
        self.xyxy = np.array(xyxy, dtype=float)
        self.cls = np.array(cls, dtype=float)
        self.conf = None if conf is None else np.array(conf, dtype=float)
        self.id = None if ids is None else np.array(ids, dtype=float)

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, i):
        return FakeBoxes(self.xyxy[i].tolist(), self.cls[i].tolist(),
                         None if self.conf is None else self.conf[i].tolist(),
                         None if self.id is None else self.id[i].tolist())


class FakeRes:
    def __init__(self, boxes, image=None):
        self.boxes = boxes
        self.orig_img = image
        self.masks = None
        self.keypoints = None

    def plot(self, **kwargs):
        return np.zeros_like(self.orig_img)


def _person_res(conf=0.9, tid=1, image=None):
    return FakeRes(FakeBoxes([[0, 0, 8, 8]], [0], [conf], [tid]), image)


class TestEventStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'nested' / 'events.db'
        self.store = EventStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_add_event_creates_dir_and_rows(self):
        eid = self.store.add_event('2024-01-02 03:04:05', 7, 'cam0', 'person', 0.875, 3,
                                   'a.jpg', {'zone': 'A'})
        self.assertIsInstance(eid, int)
        self.assertGreater(eid, 0)
        self.assertTrue(self.path.exists())
        row = _query(self.path, 'SELECT * FROM events WHERE id = ?', (eid,))[0]
        self.assertEqual(row['ts'], '2024-01-02 03:04:05')
        self.assertEqual(row['frame'], 7)
        self.assertEqual(row['source'], 'cam0')
        self.assertEqual(row['cls'], 'person')
        self.assertAlmostEqual(row['conf'], 0.875)
        self.assertEqual(row['track_id'], 3)
        self.assertEqual(row['image'], 'a.jpg')
        self.assertEqual(json.loads(row['payload']), {'zone': 'A'})
        self.assertEqual(self.store.counts()['events'], 1)

    def test_counts_and_reopen_keeps_pending(self):
        eid = self.store.add_event('t', 1, 'src', 'person', 0.9)
        oid = self.store.enqueue_alert('hello', eid)
        self.assertIsInstance(oid, int)
        self.assertEqual(self.store.counts(), {'pending': 1, 'dead': 0, 'events': 1})
        self.store.close()

        reopened = EventStore(self.path)
        try:
            self.assertEqual(reopened.counts()['pending'], 1)
            rows = reopened.claim_due(now=time.time())
            self.assertEqual([r['text'] for r in rows], ['hello'])
            self.assertEqual(rows[0]['event_id'], eid)
            self.assertEqual(reopened.counts()['pending'], 0)     # 已认领 -> sending
            self.assertEqual(reopened.claim_due(now=time.time()), [])
        finally:
            reopened.close()
        # tearDown 会再次 close；EventStore.close 是幂等的
        self.store = EventStore(self.path)

    def test_mark_sent_is_never_claimed_again(self):
        oid = self.store.enqueue_alert('done')
        self.assertEqual(len(self.store.claim_due()), 1)
        self.assertTrue(self.store.mark_sent(oid))
        self.assertEqual(self.store.counts()['pending'], 0)
        self.assertEqual(self.store.claim_due(now=1e18), [])
        row = _query(self.path, 'SELECT status, next_retry_at FROM outbox WHERE id = ?', (oid,))[0]
        self.assertEqual(row['status'], 'sent')
        self.assertIsNone(row['next_retry_at'])

    def test_backoff_grows_and_dead_after_max_attempts(self):
        oid = self.store.enqueue_alert('retry me')
        self.assertEqual(len(self.store.claim_due(now=time.time())), 1)

        before = time.time()
        self.assertTrue(self.store.mark_failed(oid, 'boom1', max_attempts=3, base_delay=2.0))
        row = _query(self.path, 'SELECT * FROM outbox WHERE id = ?', (oid,))[0]
        self.assertEqual(row['status'], 'pending')
        self.assertEqual(row['attempts'], 1)
        self.assertIn('boom1', row['last_error'])
        first_due = row['next_retry_at']
        self.assertGreaterEqual(first_due - before, 1.9)          # base * 2^0
        self.assertLessEqual(first_due - before, 2.5)
        self.assertEqual(self.store.claim_due(now=first_due - 0.01), [])   # 未到期

        self.assertEqual(len(self.store.claim_due(now=first_due + 0.01)), 1)
        before2 = time.time()
        self.store.mark_failed(oid, 'boom2', max_attempts=3, base_delay=2.0)
        row = _query(self.path, 'SELECT * FROM outbox WHERE id = ?', (oid,))[0]
        self.assertEqual(row['attempts'], 2)
        second_due = row['next_retry_at']
        self.assertGreater(second_due - first_due, 0)
        self.assertGreaterEqual(second_due - before2, 3.9)         # base * 2^1
        self.assertLessEqual(second_due - before2, 4.6)

        self.assertEqual(len(self.store.claim_due(now=second_due + 0.01)), 1)
        self.store.mark_failed(oid, 'boom3', max_attempts=3, base_delay=2.0)
        row = _query(self.path, 'SELECT * FROM outbox WHERE id = ?', (oid,))[0]
        self.assertEqual(row['status'], 'dead')
        self.assertEqual(row['attempts'], 3)
        self.assertIsNone(row['next_retry_at'])
        self.assertEqual(self.store.counts(), {'pending': 0, 'dead': 1, 'events': 0})
        self.assertEqual(self.store.claim_due(now=1e18), [])

    def test_concurrent_writes_and_claims_are_safe(self):
        """检测线程与面板线程可能同时用：多线程写事件、抢 outbox 都不能丢/重复。"""

        def add_events(base):
            for i in range(10):
                self.store.add_event('t', base + i, 'src', 'person', 0.5)

        threads = [threading.Thread(target=add_events, args=(i * 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.store.counts()['events'], 50)

        for i in range(5):
            self.store.enqueue_alert(f'msg-{i}')
        claimed = []

        def claim_all():
            while True:
                rows = self.store.claim_due(limit=2)
                if not rows:
                    return
                claimed.extend(r['id'] for r in rows)

        threads = [threading.Thread(target=claim_all) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(claimed), 5)          # 一条不多一条不少
        self.assertEqual(len(set(claimed)), 5)     # 没有重复认领

    def test_unwritable_db_never_raises(self):
        bad = EventStore(self.tmp.name)      # 把目录当数据库文件，SQLite 打不开
        try:
            self.assertFalse(bad.available)
            self.assertIsNone(bad.add_event('t', 1, 'src', 'person', 0.5))
            self.assertIsNone(bad.enqueue_alert('x'))
            self.assertEqual(bad.claim_due(), [])
            self.assertFalse(bad.mark_sent(1))
            self.assertFalse(bad.mark_failed(1, 'x'))
            self.assertEqual(bad.counts(), {'pending': 0, 'dead': 0, 'events': 0})
        finally:
            bad.close()

    def test_payload_string_and_none(self):
        eid = self.store.add_event('t', None, '', 'car', None, None, None, '{"raw": true}')
        row = _query(self.path, 'SELECT payload, frame, conf FROM events WHERE id = ?', (eid,))[0]
        self.assertEqual(row['payload'], '{"raw": true}')
        self.assertIsNone(row['frame'])
        self.assertIsNone(row['conf'])


class TestOutboxDrain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'events.db'
        self.store = EventStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _channel(self):
        return AlertChannel('http://127.0.0.1:1/hook', 'generic', timeout=1, logger=None)

    def test_send_false_does_not_raise_and_attempts_grow(self):
        eid = self.store.add_event('t', 1, 'src', 'person', 0.9)
        self.store.enqueue_alert('boom', eid)
        with mock.patch.object(AlertChannel, 'send', return_value=False):
            stats = drain_outbox(self.store, self._channel(), limit=10,
                                 max_attempts=3, base_delay=2.0)
        self.assertEqual(stats, {'claimed': 1, 'sent': 0, 'failed': 1})
        row = _query(self.path, 'SELECT * FROM outbox')[0]
        self.assertEqual(row['attempts'], 1)
        self.assertEqual(row['status'], 'pending')
        self.assertIn('发送失败', row['last_error'])

    def test_send_exception_is_contained(self):
        self.store.enqueue_alert('boom')
        with mock.patch.object(AlertChannel, 'send', side_effect=RuntimeError('网络炸了')):
            stats = drain_outbox(self.store, self._channel())
        self.assertEqual(stats['failed'], 1)
        row = _query(self.path, 'SELECT * FROM outbox')[0]
        self.assertIn('网络炸了', row['last_error'])

    def test_success_marks_sent(self):
        oid = self.store.enqueue_alert('ok')
        with mock.patch.object(AlertChannel, 'send', return_value=True):
            stats = drain_outbox(self.store, self._channel())
        self.assertEqual(stats, {'claimed': 1, 'sent': 1, 'failed': 0})
        self.assertEqual(self.store.counts()['pending'], 0)
        row = _query(self.path, 'SELECT status FROM outbox WHERE id = ?', (oid,))[0]
        self.assertEqual(row['status'], 'sent')


class TestAsyncEventSink(unittest.TestCase):
    def test_flush_writes_all_images(self):
        with tempfile.TemporaryDirectory() as d:
            sink = EventSink(d, classes=[0], min_conf=0.5, cooldown_s=0,
                             async_write=True, queue_size=256)
            img = np.zeros((24, 24, 3), np.uint8)
            for i in range(20):
                self.assertTrue(sink.check(_person_res(), {0: 'person'}, i + 1, img))
            sink.flush()
            self.assertEqual(len(sink.events), 20)
            self.assertEqual(len(list(Path(d).glob('*.jpg'))), 20)
            self.assertEqual(sink.dropped_images, 0)
            sink.close()

    def test_full_queue_drops_oldest_without_blocking(self):
        with tempfile.TemporaryDirectory() as d:
            sink = EventSink(d, classes=[0], min_conf=0.5, cooldown_s=0,
                             async_write=True, queue_size=1)

            def slow_write(path, image):        # 让 worker 慢下来，稳定制造满队列
                time.sleep(0.05)
                return True

            sink._write_image = slow_write
            img = np.zeros((8, 8, 3), np.uint8)
            t0 = time.time()
            for i in range(30):
                sink.check(_person_res(), {0: 'person'}, i + 1, img)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 2.0)                    # 生产端没有被写盘拖住
            self.assertGreater(sink.dropped_images, 0)
            self.assertEqual(len(sink.events), 30)           # 事件列表不受丢图影响
            sink.close()

    def test_sync_mode_writes_immediately(self):
        with tempfile.TemporaryDirectory() as d:
            sink = EventSink(d, classes=[0], min_conf=0.5, cooldown_s=0)
            sink.check(_person_res(), {0: 'person'}, 1, np.zeros((8, 8, 3), np.uint8))
            self.assertEqual(len(list(Path(d).glob('*.jpg'))), 1)
            self.assertFalse(sink.async_write)


class TestMetricsAndConfig(unittest.TestCase):
    def test_outbox_metrics(self):
        text = prometheus_text({'alerts_sent': 2, 'alerts_failed': 1, 'outbox_pending': 4})
        self.assertIn('yolo_alerts_sent_total 2', text)
        self.assertIn('yolo_alerts_failed_total 1', text)
        self.assertIn('yolo_outbox_pending 4', text)

    def test_new_defaults_are_safe(self):
        cfg = load_config(None, None)
        self.assertEqual(cfg['output']['events']['db'], 'events.db')
        self.assertTrue(cfg['output']['events']['async_write'])
        self.assertEqual(cfg['output']['events']['async_queue'], 64)
        self.assertEqual(cfg['alert']['retry']['max_attempts'], 3)
        self.assertEqual(cfg['alert']['retry']['base_delay'], 2.0)
        # 空字符串仍然可以覆盖默认值，退回"不落库"
        off = load_config(None, {'output': {'events': {'db': ''}}})
        self.assertEqual(off['output']['events']['db'], '')


class FakeDetector:
    """不加载模型，按调用次数把目标从上往下移，制造一次越线。"""

    names = {0: 'person'}
    weights = 'fake.pt'
    track = False
    device_name = 'cpu'

    def __init__(self):
        self.calls = 0

    def infer(self, frame, imgsz=None):
        self.calls += 1
        y = 10 if self.calls == 1 else 90
        boxes = FakeBoxes([[40, y - 5, 60, y + 5]], [0], [0.9], [1])
        return FakeRes(boxes, frame)


def _make_video(path, frames=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (160, 120))
    for _ in range(frames):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()


class TestPipelineEventsIntegration(unittest.TestCase):
    """端到端：真实 VideoSource + 假 Detector，验证落库、outbox 留队与 summary 字段。"""

    @staticmethod
    def _cfg(out, db='events.db'):
        return load_config(None, {
            'run': {'max_frames': 4},
            'output': {'dir': str(out), 'video': False, 'json': False,
                       'events': {'enable': True, 'classes': [0], 'min_conf': 0.5,
                                  'cooldown_s': 1000, 'db': db,
                                  'async_write': True, 'async_queue': 8}},
            'analytics': {'line': [[0, 60], [160, 60]], 'min_cross_frames': 1},
            'alert': {'webhook': 'http://127.0.0.1:1/hook', 'type': 'generic',
                      'timeout': 1, 'dedupe_s': 0,
                      'retry': {'max_attempts': 3, 'base_delay': 60.0}},
        })

    def test_crossing_and_capture_are_persisted(self):
        from yolo_app.pipeline import process

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root / 'src.mp4'
            _make_video(video)
            out = root / 'outputs'
            cfg = self._cfg(out)
            # 发送一律失败：验证不抛异常、attempts 增长、剩余 pending 留在队列里
            with mock.patch.object(AlertChannel, 'send', return_value=False):
                summary = process(str(video), cfg, FakeDetector(), get_logger('ERROR'))

            self.assertEqual(summary['frames_inferred'], 4)
            self.assertEqual(len(summary['crossings']), 1)
            self.assertEqual(len(summary['events']), 1)          # 抓拍事件（冷却期内只一条）
            self.assertEqual(summary['dropped_images'], 0)
            self.assertEqual(summary['outbox_pending'], 2)       # 抓拍 + 越线各一条
            self.assertEqual(summary['events_db'], str(out / 'events.db'))
            self.assertTrue((out / 'events.db').exists())
            self.assertEqual(len(list((out / 'events').glob('*.jpg'))), 1)   # 异步写盘已在 close 时排空

            events = _query(out / 'events.db', 'SELECT * FROM events ORDER BY id')
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]['cls'], 'person')
            self.assertEqual(events[0]['image'], summary['events'][0]['image'])
            self.assertEqual(json.loads(events[0]['payload'])['frame'], 1)
            self.assertEqual(events[1]['frame'], 2)
            self.assertEqual(json.loads(events[1]['payload'])['direction'], 'in')

            outbox = _query(out / 'events.db', 'SELECT * FROM outbox ORDER BY id')
            self.assertEqual(len(outbox), 2)
            self.assertTrue(all(r['status'] == 'pending' for r in outbox))
            self.assertTrue(all(r['attempts'] == 1 for r in outbox))
            self.assertIn('YOLO 告警', outbox[0]['text'])
            self.assertIn('YOLO 越线', outbox[1]['text'])

    def test_empty_db_falls_back_to_synchronous_send(self):
        """db='' 时必须完全退回旧行为：不建库、告警当帧同步发送。"""
        from yolo_app.pipeline import process

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root / 'src.mp4'
            _make_video(video)
            out = root / 'outputs'
            cfg = self._cfg(out, db='')
            calls = []

            def fake_send(channel, text, dedupe=True):
                calls.append(text)
                channel.sent += 1
                return True

            with mock.patch.object(AlertChannel, 'send', fake_send):
                summary = process(str(video), cfg, FakeDetector(), get_logger('ERROR'))

            self.assertIsNone(summary['events_db'])
            self.assertEqual(summary['outbox_pending'], 0)
            self.assertEqual(summary['alert_sent'], 2)       # 抓拍 + 越线
            self.assertEqual(len(calls), 2)
            self.assertFalse((out / 'events.db').exists())


if __name__ == '__main__':
    unittest.main()
