"""配置 schema 校验：合法值放行、越界值报错、可疑值只 warning。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.config import ConfigError, load_config, validate_config


class TestConfigSchema(unittest.TestCase):
    """规则覆盖：每类值域各打一条，保证报错信息可操作（字段/当前值/期望）。"""

    def test_defaults_are_valid_without_warnings(self):
        cfg = load_config(None, None)
        self.assertEqual(validate_config(cfg), [])

    def test_legal_boundary_values_pass(self):
        cfg = load_config(None, {
            'model': {'conf': 0.0, 'iou': 1.0, 'report_conf': 0.0},
            'run': {'stride': 1, 'max_frames': 0, 'workers': 1, 'read_timeout': 0.5},
            'analytics': {'roi': [[0, 0], [10, 0], [0, 10]], 'line': [[0, 0], [10, 10]],
                          'min_cross_frames': 1, 'track_ttl': 0, 'roi_margin': 0.0},
            'serve': {'port': 65535},
            'alert': {'dedupe_s': 0.0, 'retry': {'max_attempts': 1, 'base_delay': 0.1}},
        })
        self.assertEqual(validate_config(cfg), [])

    def test_imgsz_not_multiple_of_32_only_warns(self):
        cfg = load_config(None, {'model': {'imgsz': 1000}})
        warnings = validate_config(cfg)
        self.assertTrue(any('model.imgsz' in w and '32' in w for w in warnings), warnings)

    def test_imgsz_non_positive_raises(self):
        with self.assertRaises(ConfigError) as cm:
            load_config(None, {'model': {'imgsz': 0}})
        self.assertIn('model.imgsz', str(cm.exception))

    def test_conf_out_of_range_raises_actionable_message(self):
        with self.assertRaises(ConfigError) as cm:
            load_config(None, {'model': {'conf': 1.5}})
        msg = str(cm.exception)
        self.assertIn('model.conf', msg)        # 字段
        self.assertIn('1.5', msg)               # 当前值
        self.assertIn('期望', msg)               # 期望

    def test_iou_out_of_range_raises(self):
        with self.assertRaises(ConfigError):
            load_config(None, {'model': {'iou': -0.1}})

    def test_roi_too_few_points_raises(self):
        with self.assertRaises(ConfigError) as cm:
            load_config(None, {'analytics': {'roi': [[0, 0], [10, 10]]}})
        self.assertIn('analytics.roi', str(cm.exception))
        self.assertIn('3', str(cm.exception))

    def test_line_must_be_exactly_two_points(self):
        with self.assertRaises(ConfigError) as cm:
            load_config(None, {'analytics': {'line': [[0, 0], [10, 0], [20, 0]]}})
        self.assertIn('analytics.line', str(cm.exception))
        self.assertIn('2', str(cm.exception))

    def test_unknown_key_warns_but_does_not_raise(self):
        cfg = load_config(None, {'serve': {'ws_heartbeat': 5}})
        warnings = validate_config(cfg)
        self.assertIn('未知配置项 serve.ws_heartbeat，已忽略', warnings)
        self.assertEqual(cfg['serve']['ws_heartbeat'], 5)   # 只提醒不删除

    def test_strict_unknown_key_raises(self):
        with self.assertRaises(ConfigError) as cm:
            load_config(None, {'serve': {'ws_heartbeat': 5}}, strict=True)
        self.assertIn('serve.ws_heartbeat', str(cm.exception))

    def test_other_domains_reject_bad_values(self):
        cases = [
            {'run': {'stride': 0}},
            {'run': {'workers': 0}},
            {'run': {'read_timeout': 0}},
            {'run': {'reader': 'gstreamer'}},
            {'camera': {'fps': 0}},
            {'camera': {'buffer_size': 0}},
            {'analytics': {'min_cross_frames': 0}},
            {'analytics': {'track_ttl': -1}},
            {'analytics': {'roi_margin': 2}},
            {'output': {'segment_seconds': -1}},
            {'serve': {'port': 70000}},
            {'alert': {'dedupe_s': -1}},
            {'alert': {'retry': {'max_attempts': 0}}},
            {'alert': {'retry': {'base_delay': 0}}},
        ]
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ConfigError):
                load_config(None, override)

    def test_cli_returns_1_on_invalid_config(self):
        from yolo_app.cli import main

        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / 'bad.yaml'
            bad.write_text('model:\n  conf: 1.5\n', encoding='utf-8')
            with self.assertLogs('yolo_app', level='ERROR') as cm:
                rc = main(['run', '--source', 'whatever.mp4', '--config', str(bad)])
        self.assertEqual(rc, 1)
        self.assertTrue(any('配置校验失败' in line for line in cm.output), cm.output)


if __name__ == '__main__':
    unittest.main()
