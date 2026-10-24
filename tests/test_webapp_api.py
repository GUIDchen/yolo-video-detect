"""Web 面板 HTTP/WebSocket 协议测试：/stats 的 seq 与 /ws 增量推送 + 断线重连。

跑法：python -m unittest discover -s tests -v
用真实模型 + data/test.mp4 跑 2 帧（约 1~3 秒），覆盖：
  · AppState 每写一次统计 seq 单调 +1，快照把 seq 与统计一起带出；
  · /stats 带 seq，原有字段（source/finished/error/frames/...）不丢；
  · /ws 首条消息是 type=stats 的完整快照；
  · 统计冻结后用 ?since=<seq> 重连，1.5 秒窗口内只收到 heartbeat，
    不重复推同 seq 的 stats（服务端只在 seq 变化时推）；
  · token 错误以 code 1008 关闭连接，坏 ?since= 参数不致命。

心跳在生产环境是 15 秒；测试通过 serve.ws_heartbeat 调到 0.25 秒，
这样 1~2 秒的窗口里能观察到"无新统计只有心跳"。
"""
import queue
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))        # 复用 test_smoke 的测试素材准备逻辑

from test_smoke import ensure_test_video      # noqa: E402  （需先插入 sys.path）

TOKEN = 'secret'
HEARTBEAT_S = 0.25                            # 生产默认 15s，测试调短以便观察心跳


class TestWebAppApi(unittest.TestCase):
    """整个类共用一个面板实例：模型只加载一次，2 帧后统计冻结，便于断言增量语义。"""

    @classmethod
    def setUpClass(cls):
        if not (ROOT / 'weights' / 'yolo11n.pt').exists():
            raise unittest.SkipTest('缺少 weights/yolo11n.pt')
        try:
            from fastapi.testclient import TestClient
            from yolo_app.config import load_config
            from yolo_app.detector import Detector
            from yolo_app.logging_setup import get_logger
            from yolo_app.webapp import create_app
        except Exception as e:                 # 干净环境缺依赖时跳过，不误报失败
            raise unittest.SkipTest(f'缺少面板测试依赖: {e}')
        cfg = load_config(None, {'run': {'max_frames': 2},
                                 'output': {'video': False, 'json': False,
                                            'events': {'enable': False}},
                                 'serve': {'token': TOKEN, 'ws_heartbeat': HEARTBEAT_S}})
        cls.app = create_app(cfg, Detector(cfg), get_logger('ERROR'), str(ensure_test_video()))
        cls.client = TestClient(cls.app)
        cls.client.__enter__()                 # 触发 lifespan startup；tearDownClass 走优雅停机

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'client', None) is not None:
            cls.client.__exit__(None, None, None)

    # ---- 工具 ----
    def _stats(self):
        r = self.client.get('/stats', params={'token': TOKEN})
        self.assertEqual(r.status_code, 200)
        return r.json()

    def _wait_finished(self, timeout=30.0):
        """等检测线程收尾：之后 seq 不再变化，才能验证"无变化不重复推送"。"""
        end = time.time() + timeout
        while time.time() < end:
            d = self._stats()
            if d.get('finished'):
                return d
            time.sleep(0.1)
        self.fail(f'检测线程 {timeout}s 内未收尾')

    @staticmethod
    def _recv_window(ws, seconds):
        """在 seconds 窗口内收集 WS 消息；后台线程兜底，避免服务端不发消息时永久阻塞。"""
        got, inbox = [], queue.Queue()

        def pump():
            try:
                while True:
                    inbox.put(ws.receive_json())
            except Exception:                  # 会话收尾/连接关闭，线程自行退出
                pass

        threading.Thread(target=pump, daemon=True).start()
        end = time.time() + seconds
        while True:
            left = end - time.time()
            if left <= 0:
                return got
            try:
                got.append(inbox.get(timeout=left))
            except queue.Empty:
                return got

    # ---- 用例 ----
    def test_appstate_seq_increments(self):
        """每写一次统计 seq +1，快照把 seq 与统计一起带出。"""
        from yolo_app.webapp import AppState
        st = AppState()
        self.assertEqual(st.snapshot()['seq'], 0)
        st.update_stats({'frames': 1})
        st.update_stats({'frames': 2})
        snap = st.snapshot()
        self.assertEqual(snap['seq'], 2)
        self.assertEqual(snap['frames'], 2)

    def test_stats_carries_seq_and_legacy_fields(self):
        d = self._wait_finished()
        self.assertIsInstance(d['seq'], int)
        self.assertGreaterEqual(d['seq'], 1)   # 2 帧至少推过一次统计
        for key in ('source', 'finished', 'error', 'frames', 'fps', 'per_class', 'line'):
            self.assertIn(key, d)

    def test_ws_first_message_is_snapshot(self):
        with self.client.websocket_connect(f'/ws?token={TOKEN}') as ws:
            msg = ws.receive_json()
        self.assertEqual(msg['type'], 'stats')
        self.assertIsInstance(msg['seq'], int)
        for key in ('frames', 'fps', 'per_class', 'line'):
            self.assertIn(key, msg)

    def test_ws_since_skips_duplicate_stats(self):
        frozen = self._wait_finished()
        seq = frozen['seq']
        with self.client.websocket_connect(f'/ws?token={TOKEN}&since={seq}') as ws:
            first = ws.receive_json()
            self.assertEqual(first['type'], 'stats')     # 协议要求先补当前快照
            self.assertEqual(first['seq'], seq)
            got = self._recv_window(ws, 1.5)
        self.assertTrue(got, '统计冻结时应当只有 heartbeat 保活，至少收到一条')
        for msg in got:
            self.assertEqual(msg['type'], 'heartbeat', f'统计未变化却收到: {msg}')
            self.assertEqual(msg['seq'], seq)

    def test_ws_invalid_since_is_ignored(self):
        with self.client.websocket_connect(f'/ws?token={TOKEN}&since=abc') as ws:
            msg = ws.receive_json()
        self.assertEqual(msg['type'], 'stats')        # 坏参数不报错，按全新连接处理

    def test_ws_bad_token_closes_1008(self):
        try:
            from starlette.websockets import WebSocketDisconnect
        except Exception as e:
            self.skipTest(f'缺少 WebSocketDisconnect: {e}')
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self.client.websocket_connect('/ws?token=wrong'):
                pass
        self.assertEqual(ctx.exception.code, 1008)


if __name__ == '__main__':
    unittest.main(verbosity=2)
