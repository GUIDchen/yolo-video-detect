"""模型封装：负责"把一帧变成检测结果"，是唯一直接依赖 ultralytics 的模块。

本模块解决三件容易踩坑的事：
  1. 权重定位 resolve_weights()：ultralytics 默认去 GitHub Releases 下权重（国内常不通），
     这里优先命中项目内 weights/ 下的同名文件，保证离线环境也能跑；
  2. 类别解析 resolve_class_ids()：统一把 ['person', 0, 'bus'] 这类混合写法解析成 [0, 5]，
     避免"类别名写对了却永远匹配不上 id"的静默失效；
  3. 推理/跟踪二选一：cfg['track']['enable'] 决定走 model.track(persist=True) 还是 model.predict。

对外只暴露 Detector.infer(frame) -> ultralytics Results 与几个只读属性，
上层的 ROI、统计、绘制都基于这个 Results，不关心底层是 PyTorch / ONNX / TensorRT。
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def resolve_weights(name):
    """裸写权重名时优先命中项目内 weights/，避免联网下载（国内 GitHub 不通）。"""
    if Path(name).exists():
        return str(Path(name))
    local = ROOT / 'weights' / name
    return str(local) if local.exists() else name


def resolve_class_ids(classes, names):
    """类别 id / 名称混合列表 → 排序去重的 id 列表；空列表返回 None（表示不过滤）。"""
    if not classes:
        return None
    out = []
    for c in classes:
        if isinstance(c, int) or str(c).isdigit():
            out.append(int(c))
        else:
            ids = [i for i, n in names.items() if str(n).lower() == str(c).lower()]
            if not ids:
                raise ValueError(f'未知类别名: {c}')
            out.extend(ids)
    return sorted(set(out))


class Detector:
    def __init__(self, cfg):
        m = cfg['model']
        self.weights = resolve_weights(m['weights'])
        self.model = YOLO(self.weights)
        self.names = self.model.names
        self.imgsz, self.conf, self.iou = m['imgsz'], m['conf'], m['iou']
        self.device, self.half = m['device'], m['half']
        self.augment = m.get('augment', False)
        self.track = cfg['track']['enable']
        self.tracker = cfg['track']['tracker']
        self.classes = self._resolve_classes(m['classes'])

    def _resolve_classes(self, cls):
        return resolve_class_ids(cls, self.names)

    def infer(self, frame, imgsz=None):
        """对单帧做推理；启用跟踪时同时返回带 track_id 的 Results。

        返回 ultralytics 的 Results，上层的统计/绘制/ROI 都基于它；
        classes=None 表示不过滤类别，half 仅在显式开启时才传给 ultralytics（8.4 起该参数已弃用）。
        imgsz 非空时覆盖配置值（ROI 裁剪推理用小尺寸换速度）。
        """
        kw = dict(conf=self.conf, iou=self.iou, imgsz=imgsz or self.imgsz, device=self.device,
                  classes=self.classes, verbose=False)
        if self.half:                       # ultralytics 8.4 起 half 已标记 deprecated，仅在需要时才传
            kw['half'] = True
        if self.augment:                    # TTA：多尺度/翻转推理后融合，精度更高、耗时约 2 倍
            kw['augment'] = True
        if self.track:
            return self.model.track(frame, persist=True, tracker=self.tracker, **kw)[0]
        return self.model.predict(frame, **kw)[0]

    @property
    def device_name(self):
        """ONNX/TensorRT 等非 torch 权重拿不到 parameters()，此时回退到配置值。"""
        try:
            return str(next(self.model.model.parameters()).device)
        except Exception:
            return str(self.device or 'auto')
