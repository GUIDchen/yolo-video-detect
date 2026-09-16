"""模型清单、下载与基准测试：让"换模型"变成一条命令的事。

三件事：
  1. PRESETS   常用官方模型清单（大小/相对速度/适用场景），`python -m yolo_app models` 直接看；
  2. download  自动选镜像下载到 weights/（国内 GitHub 不通也能拉；HF 镜像实测可用）；
  3. benchmark 对同一段视频跑多个模型，输出速度/检出对照表，用来选型。

换模型本身不需要任何代码改动——`--model <名字>` 就会走 detector.resolve_weights() 在 weights/ 里找；
本模块只是把"有哪些可选、本地有没有、怎么拉、谁更快"这几件事集中起来。
"""
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / 'weights'

# 前缀 → HuggingFace 仓库名（下载地址用）
REPO_BY_PREFIX = {
    'yolo26': 'YOLO26',
    'yolo11': 'YOLO11',
    'yolov8': 'YOLOv8',
    'yolov10': 'YOLOv10',
    'yolov5': 'YOLOv5',
}

# 下载镜像按顺序尝试：国内镜像 → 官方 HF → GitHub 代理
MIRRORS = (
    'https://hf-mirror.com/ultralytics/{repo}/resolve/main/{name}',
    'https://huggingface.co/ultralytics/{repo}/resolve/main/{name}',
    'https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}',
)

# 预设清单：name -> (体积, 相对速度, 说明)
PRESETS = {
    'yolo26n.pt': ('5.3MB', 1.0, 'YOLO26 最小档：最新一代，实时首选'),
    'yolo26s.pt': ('19MB', 0.6, 'YOLO26 小档：精度明显优于 n'),
    'yolo26m.pt': ('39MB', 0.35, 'YOLO26 中档：有 GPU 时的均衡选择'),
    'yolo26l.pt': ('49MB', 0.25, 'YOLO26 大档：精度优先'),
    'yolo26x.pt': ('109MB', 0.15, 'YOLO26 最大档：离线批处理'),
    'yolo11n.pt': ('5.4MB', 1.0, 'YOLO11 最小档：生态最成熟、兼容性最好'),
    'yolo11s.pt': ('18MB', 0.6, 'YOLO11 小档：本项目实测误检少于 n'),
    'yolo11m.pt': ('39MB', 0.35, 'YOLO11 中档'),
    'yolo11l.pt': ('49MB', 0.25, 'YOLO11 大档'),
    'yolo11x.pt': ('109MB', 0.15, 'YOLO11 最大档'),
    'yolov8n.pt': ('6.2MB', 1.0, 'YOLOv8 最小档：老项目/教程最常见'),
    'yolov8s.pt': ('22MB', 0.6, 'YOLOv8 小档'),
    'yolov8m.pt': ('50MB', 0.35, 'YOLOv8 中档'),
}


def repo_for(name):
    """按文件名前缀猜 HF 仓库；认不出来时按 YOLO11 处理。"""
    for prefix, repo in REPO_BY_PREFIX.items():
        if name.startswith(prefix):
            return repo
    return 'YOLO11'


def local_weights():
    """weights/ 下已有的权重文件（按名字排序）。"""
    if not WEIGHTS.exists():
        return []
    return sorted(p.name for p in WEIGHTS.iterdir() if p.suffix in ('.pt', '.onnx', '.engine', '.xml'))


def describe(name):
    """给出模型的一句话说明：预设里有就用预设，没有就按体积粗略判断。"""
    if name in PRESETS:
        return PRESETS[name][2]
    p = WEIGHTS / name
    if p.exists():
        mb = p.stat().st_size / 1e6
        return f'本地权重（{mb:.1f}MB）'
    return '未知模型（首次使用会尝试下载）'


def download(names, logger=print):
    """下载缺失的权重到 weights/，返回 (成功列表, 失败列表)。已存在的自动跳过。"""
    import urllib.request

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    for name in names:
        target = WEIGHTS / name
        if target.exists():
            logger(f'[跳过] {name} 已存在（{target.stat().st_size / 1e6:.1f}MB）')
            ok.append(name)
            continue
        for tpl in MIRRORS:
            url = tpl.format(repo=repo_for(name), name=name)
            try:
                logger(f'[下载] {name} ← {url}')
                with urllib.request.urlopen(url, timeout=180) as r:   # 超时给足：大模型几十 MB
                    data = r.read()
                if len(data) < 100_000:                               # 明显不是权重（可能是错误页）
                    raise ValueError(f'文件过小（{len(data)} 字节），可能不是有效权重')
                target.write_bytes(data)
                logger(f'[完成] {name} {len(data) / 1e6:.1f}MB → {target}')
                ok.append(name)
                break
            except Exception as e:
                logger(f'[失败] {url.split("/")[2]} : {e}')
        else:
            fail.append(name)
    return ok, fail


def benchmark(names, source, frames, cfg, logger):
    """对同一个视频源依次跑多个模型，返回对照行列表。

    只做推理，不写视频/JSON/抓拍，保证各模型之间的差异只来自模型本身。
    """
    import copy

    from .detector import Detector
    from .pipeline import process

    rows = []
    for name in names:
        c = copy.deepcopy(cfg)
        c['model']['weights'] = name
        c['run']['max_frames'] = frames
        c['run']['show'] = False
        c['output']['video'] = False
        c['output']['json'] = False
        c['output']['events']['enable'] = False          # 关掉抓拍，避免写盘影响计时
        try:
            det = Detector(c)
            t0 = time.time()
            s = process(source, c, det, logger)
            rows.append({
                'model': name,
                'fps': s['avg_fps'],
                'elapsed': s['elapsed_s'],
                'detections': s['per_class_frames'],
                'total': sum(s['per_class_frames'].values()),
                'unique_tracks': s['unique_tracks'],
                'classes': len(det.names),
                'device': det.device_name,
                'wall': round(time.time() - t0, 1),
            })
        except Exception as e:                            # 单个模型失败不影响其它模型
            logger(f'[跳过] {name}: {e}')
            rows.append({'model': name, 'error': str(e)})
    return rows


def format_table(rows):
    """把基准结果排成对齐的文本表（不引第三方表格库）。"""
    if not rows:
        return '(无结果)'
    head = f'{"模型":<16}{"FPS":>7}{"耗时":>8}{"检出合计":>10}{"唯一目标":>10}  逐类检出'
    lines = [head, '-' * max(len(head), 60)]
    for r in rows:
        if 'error' in r:
            lines.append(f'{r["model"]:<16}{"失败":>7}  {r["error"][:40]}')
            continue
        per = ' '.join(f'{k}:{v}' for k, v in sorted(r['detections'].items()))
        lines.append(f'{r["model"]:<16}{r["fps"]:>7}{r["elapsed"]:>8}{r["total"]:>10}'
                     f'{r["unique_tracks"]:>10}  {per}')
    return '\n'.join(lines)
