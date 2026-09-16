"""统一日志：控制台 +（可选）文件，避免业务代码里散落 print。

约定：
  · 进程内只有一个 logger（'yolo_app'），重复调用不会重复挂 handler；
  · 时间只打到 HH:MM:SS，长跑任务的进度日志更紧凑；
  · 多进程场景下每个子进程各自初始化（run_job 里就是这么做的）。
"""
import logging
import sys
from pathlib import Path


def get_logger(level='INFO', log_file=None):
    logger = logging.getLogger('yolo_app')
    if logger.handlers:                       # 避免重复添加 handler
        return logger
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter('[%(asctime)s] %(levelname)-7s %(message)s', '%H:%M:%S')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
