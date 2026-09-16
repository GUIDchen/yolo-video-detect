"""业务后处理：ROI 区域过滤 + 越线计数。

纯几何运算，不依赖模型与 OpenCV（draw 时才延迟 import cv2），因此非常容易单测——
tests/test_smoke.py 用 FakeBoxes 就能覆盖全部逻辑，不必真的跑推理。

两个类的分工：
  · RegionFilter：只关心"目标在不在指定区域"，把区域外的框直接从结果里剔掉，
    后续的统计、绘制、抓拍都因此只看到关注区域，等于用零成本做了业务裁剪；
  · LineCounter：只关心"同一个目标有没有跨过某条线"，靠 track_id 跨帧记忆所在侧，
    必须配合 --track；方向语义见类 docstring。

想加"区域停留时长""人群聚集"之类的统计，照这两个类的样子新增一个类，
在 pipeline.process 的推理之后插一行调用即可，主循环结构不用动。
"""
from collections import defaultdict


def _inside(poly, pt):
    """射线法：点是否在多边形内。poly=[[x,y],...]"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def _side(line, pt):
    """点在直线的哪一侧：>0 / <0，0 表示在线上。"""
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)


class RegionFilter:
    """只保留中心点落在 ROI 多边形内的检测框；roi 为空表示不过滤。"""

    def __init__(self, roi=None):
        self.roi = roi or []

    @property
    def enabled(self):
        return bool(self.roi)

    def keep(self, res):
        """就地剔除 ROI 之外的检测框（改 res.boxes），返回同一个 Results 供链式调用。"""
        if not self.enabled or res.boxes is None or not len(res.boxes):
            return res
        xyxy = res.boxes.xyxy.tolist()
        idx = [i for i, b in enumerate(xyxy) if _inside(self.roi, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))]
        res.boxes = res.boxes[idx]
        return res

    def draw(self, frame, color=(0, 255, 255)):
        import cv2
        import numpy as np
        if self.enabled:
            cv2.polylines(frame, [np.array(self.roi, dtype=np.int32)], True, color, 2)
        return frame


class LineCounter:
    """越线计数：按 track_id 跟踪目标中心点所在侧，侧别翻转即记一次穿越。

    方向约定：线由 p1→p2，其逆时针一侧为 +、顺时针一侧为 -；
    - 侧穿到 + 侧记为 in，+ 侧穿到 - 侧记为 out（水平线 p1 在左 p2 在右时，自上而下为 in）。
    """

    def __init__(self, line, classes=None, names=None):
        self.line = line or []
        self.classes = set(classes or [])
        self.names = names or {}
        self.prev = {}
        self.total = {"in": 0, "out": 0}
        self.by_class = defaultdict(lambda: {"in": 0, "out": 0})

    @property
    def enabled(self):
        return len(self.line) == 2

    def update(self, res, names):
        """更新各 track 的所在侧；返回本帧发生的穿越事件列表（无则为 None）。"""
        if not self.enabled or res.boxes is None or not len(res.boxes):
            return None
        boxes = res.boxes
        ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        xyxy = boxes.xyxy.tolist()
        crossed = []
        for i, tid in enumerate(ids):
            cid = int(boxes.cls.tolist()[i])
            if self.classes and cid not in self.classes:
                continue
            cx = (xyxy[i][0] + xyxy[i][2]) / 2
            cy = (xyxy[i][1] + xyxy[i][3]) / 2
            s = 1 if _side(self.line, (cx, cy)) >= 0 else -1
            if tid is None:
                continue
            tid = int(tid)
            prev = self.prev.get(tid)
            self.prev[tid] = s
            if prev is not None and prev != s:
                direction = "in" if prev < 0 and s > 0 else "out"
                name = names.get(cid, str(cid))
                self.total[direction] += 1
                self.by_class[name][direction] += 1
                crossed.append({"track_id": tid, "class": name, "direction": direction})
        return crossed or None

    def draw(self, frame, color=(0, 0, 255)):
        import cv2
        if self.enabled:
            (x1, y1), (x2, y2) = self.line
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, "in %d / out %d" % (self.total["in"], self.total["out"]),
                        (int(x1), max(int(y1) - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def summary(self):
        return {"line_total": dict(self.total), "line_by_class": {k: dict(v) for k, v in self.by_class.items()}}
