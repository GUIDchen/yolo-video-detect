#!/usr/bin/env python3
"""兼容入口：等价于 python -m yolo_app run（旧参数 --check 也会转到 check）。"""
import sys

from yolo_app.cli import main

if __name__ == '__main__':
    argv = sys.argv[1:]
    sys.exit(main(['check']) if '--check' in argv else main(['run'] + argv))
