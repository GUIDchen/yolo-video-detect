"""输出命名：stem 保持历史行为，unique 用源路径哈希保证跨运行稳定且不互相覆盖。"""
import hashlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo_app.config import load_config
from yolo_app.pipeline import output_paths

HASH_NAME = re.compile(r'^clip_[0-9a-f]{6}\.mp4$')


def _cfg(out_dir, mode=None, name=''):
    """构造配置；name_mode 是可选键，直接写 cfg 而不进 DEFAULTS。"""
    cfg = load_config(None, {'output': {'dir': str(out_dir), 'name': name}})
    if mode is not None:
        cfg['output']['name_mode'] = mode
    return cfg


class TestOutputNaming(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    # ---- stem：默认行为必须与改动前完全一致 ----
    def test_stem_is_default(self):
        self.assertEqual(output_paths(_cfg(self.dir), 'data/test.mp4'), self.dir / 'test.mp4')

    def test_stem_explicit_equals_default(self):
        self.assertEqual(output_paths(_cfg(self.dir, 'stem'), 'data/test.mp4'), self.dir / 'test.mp4')

    def test_camera_uses_timestamp_in_both_modes(self):
        for mode in (None, 'stem', 'unique'):
            name = output_paths(_cfg(self.dir, mode), 0).name
            self.assertRegex(name, r'^camera_\d{8}_\d{6}\.mp4$')

    # ---- unique：同名不同目录必须产生不同名字 ----
    def test_unique_same_name_different_dirs_differ(self):
        cfg = _cfg(self.dir, 'unique')
        a = output_paths(cfg, str(self.dir / 'a' / 'clip.mp4'))
        b = output_paths(cfg, str(self.dir / 'b' / 'clip.mp4'))
        self.assertNotEqual(a, b)
        self.assertRegex(a.name, HASH_NAME)
        self.assertRegex(b.name, HASH_NAME)

    def test_unique_hash_is_sha1_of_absolute_path(self):
        src = self.dir / 'a' / 'clip.mp4'
        expect = hashlib.sha1(str(src.resolve()).encode('utf-8')).hexdigest()[:6]
        self.assertEqual(output_paths(_cfg(self.dir, 'unique'), str(src)).name, f'clip_{expect}.mp4')

    def test_unique_is_stable_across_calls(self):
        cfg = _cfg(self.dir, 'unique')
        src = str(self.dir / 'a' / 'clip.mp4')
        self.assertEqual(output_paths(cfg, src), output_paths(cfg, src))
        # 相对/绝对写法指向同一文件，也应得到同一名字
        self.assertEqual(output_paths(cfg, src), output_paths(cfg, Path(src).resolve()))

    def test_unique_single_file_when_explicitly_requested(self):
        self.assertRegex(output_paths(_cfg(self.dir, 'unique'), 'data/test.mp4').name, r'^test_[0-9a-f]{6}\.mp4$')

    # ---- 显式 output.name 永远优先 ----
    def test_explicit_name_never_renamed(self):
        for mode in (None, 'stem', 'unique'):
            self.assertEqual(output_paths(_cfg(self.dir, mode, '_fixed.mp4'), 'data/test.mp4'),
                             self.dir / '_fixed.mp4')


class TestCliNamingWiring(unittest.TestCase):
    """CLI 接线：批量自动 unique、--name-mode 显式覆盖、显式 name 不改名。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.out = self.dir / 'out'

    def _mk(self, *names):
        for name in names:
            (self.dir / name).write_bytes(b'')
        return self.dir

    def _run(self, args, cfg_yaml=''):
        """打桩 Detector/process/logger：只验证配置怎么传到管道，不真的跑模型。"""
        from unittest.mock import MagicMock, patch

        seen = []

        def fake_process(source, cfg, detector, logger):
            seen.append((source, cfg))
            return {'source': str(source)}

        cfg_yaml = cfg_yaml or f'output:\n  dir: {self.out}\n'
        cfg_path = self.dir / 'cfg.yaml'
        cfg_path.write_text(cfg_yaml, encoding='utf-8')
        args = args + ['--config', str(cfg_path)]
        with patch('yolo_app.cli.Detector', return_value=MagicMock()), \
                patch('yolo_app.cli.process', side_effect=fake_process), \
                patch('yolo_app.cli.get_logger', return_value=MagicMock()):
            from yolo_app.cli import main
            rc = main(args)
        return rc, seen

    def test_batch_auto_enables_unique(self):
        self._mk('one.mp4', 'two.mp4')
        rc, seen = self._run(['run', '--source', str(self.dir)])
        self.assertEqual(rc, 0)
        self.assertEqual(len(seen), 2)
        for _, cfg in seen:
            self.assertEqual(cfg['output']['name_mode'], 'unique')
            self.assertEqual(cfg['output']['dir'], str(self.out))
        names = {output_paths(cfg, src).name for src, cfg in seen}
        self.assertEqual(len(names), 2)                      # 两个源不会共用一个输出名

    def test_single_file_keeps_stem(self):
        self._mk('one.mp4')
        rc, seen = self._run(['run', '--source', str(self.dir / 'one.mp4')])
        self.assertEqual(rc, 0)
        cfg = seen[0][1]
        self.assertNotIn('name_mode', cfg['output'])          # 不写配置也能保持历史行为
        self.assertEqual(output_paths(cfg, seen[0][0]).name, 'one.mp4')

    def test_explicit_name_mode_overrides_batch_default(self):
        self._mk('one.mp4', 'two.mp4')
        rc, seen = self._run(['run', '--source', str(self.dir), '--name-mode', 'stem'])
        self.assertEqual(rc, 0)
        for _, cfg in seen:
            self.assertEqual(cfg['output']['name_mode'], 'stem')

    def test_explicit_output_name_never_renamed(self):
        self._mk('one.mp4', 'two.mp4')
        rc, seen = self._run(['run', '--source', str(self.dir)],
                             cfg_yaml=f'output:\n  dir: {self.out}\n  name: _fixed.mp4\n')
        self.assertEqual(rc, 0)
        for src, cfg in seen:
            self.assertNotIn('name_mode', cfg['output'])
            self.assertEqual(output_paths(cfg, src), self.out / '_fixed.mp4')


if __name__ == '__main__':
    unittest.main()
