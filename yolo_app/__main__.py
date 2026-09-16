"""模块入口：让 `python -m yolo_app ...` 等价于执行 CLI 主函数。

这样无需安装（不用 pip install -e .）就能在项目根目录直接运行，
也方便 Docker 里用 `ENTRYPOINT ["python", "-m", "yolo_app"]`。
"""
import sys

from .cli import main

if __name__ == '__main__':
    sys.exit(main())
