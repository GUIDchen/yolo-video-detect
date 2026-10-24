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

    def __init__(self, line, classes=None, names=None, track_ttl=300, min_cross_frames=2):
        self.line = line or []
        self.classes = set(classes or [])
        self.names = names or {}
        self.prev = {}                 # track_id -> 所在侧
        self.total = {"in": 0, "out": 0}
        self.by_class = defaultdict(lambda: {"in": 0, "out": 0})
        self.track_ttl = int(track_ttl or 0)   # 0 = 不淘汰（老行为）
        # 越线防抖：目标必须在"新的一侧"连续出现 N 帧才计一次穿越。
        # 框中心在阈值线附近抖动时，单帧判定会把一次穿越记成多次（去程+回程刷计数）。
        self.min_cross_frames = max(int(min_cross_frames or 1), 1)
        self._pending = {}             # track_id -> (目标侧, 已连续帧数)
        self._last_seen = {}           # track_id -> 最后出现的帧号
        self._frame = 0

    @property
    def enabled(self):
        return len(self.line) == 2

    def update(self, res, names, frame_no=None):
        """更新各 track 的所在侧；返回本帧发生的穿越事件列表（无则为 None）。

        frame_no 用于淘汰"早已消失的 track"：跟踪器每次重连/换镜都会分配新 id，
        不清理的话 prev / _last_seen 在长视频里只增不减。不传则按调用次数计龄。
        """
        self._frame = int(frame_no) if frame_no is not None else self._frame + 1
        if not self.enabled or res.boxes is None or not len(res.boxes):
            self._prune()
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
            self._last_seen[tid] = self._frame
            if prev is None:
                self.prev[tid] = s
                self._pending.pop(tid, None)
                continue
            if s == prev:                       # 回到原侧：抖动，撤销待确认的穿越
                self._pending.pop(tid, None)
                continue
            target, count = self._pending.get(tid, (s, 0))
            if target != s:                     # 这次抖到另一侧（又换向了），重新计数
                target, count = s, 0
            count += 1
            if count < self.min_cross_frames:
                self._pending[tid] = (target, count)
                continue
            self.prev[tid] = s                  # 连续 N 帧在新侧 → 认定穿越
            self._pending.pop(tid, None)
            direction = "in" if prev < 0 and s > 0 else "out"
            name = names.get(cid, str(cid))
            self.total[direction] += 1
            self.by_class[name][direction] += 1
            crossed.append({"track_id": tid, "class": name, "direction": direction})
        self._prune()
        return crossed or None

    def _prune(self):
        """淘汰超过 track_ttl 帧没再出现的 track，避免长视频里状态无限增长。"""
        if not self.track_ttl or not self._last_seen:
            return
        cutoff = self._frame - self.track_ttl
        stale = [t for t, seen in self._last_seen.items() if seen < cutoff]
        for t in stale:
            self._last_seen.pop(t, None)
            self.prev.pop(t, None)
            self._pending.pop(t, None)

    @property
    def active_tracks(self):
        return len(self.prev)

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


class ConfFilter:
    """双阈值：只把置信度达标的结果交给统计/绘制，低分框仍留给跟踪器维持轨迹。

    为什么要两个阈值：跟踪器需要"看得见"低分目标才不会断轨，而计数、抓拍、绘制
    只需要高分目标。model.conf 管前者，model.report_conf 管后者（0 = 不额外过滤）。
    """

    def __init__(self, min_conf=0.0):
        self.min_conf = float(min_conf or 0)
        self.dropped = 0               # 累计被本类剔掉的框数（写进 summary.boxes_low_conf）

    @property
    def enabled(self):
        return self.min_conf > 0

    def keep(self, res):
        """就地剔除低于 min_conf 的框，返回同一个 Results 供链式调用。"""
        if not self.enabled or res.boxes is None or not len(res.boxes):
            return res
        conf = getattr(res.boxes, 'conf', None)
        if conf is None:               # 没有置信度的结果（如撮合出来的 FakeBoxes）不误杀
            return res
        scores = conf.tolist() if hasattr(conf, 'tolist') else list(conf)
        idx = [i for i, c in enumerate(scores) if float(c) >= self.min_conf]
        if len(idx) != len(scores):
            self.dropped += len(scores) - len(idx)
            res.boxes = res.boxes[idx]
        return res


class StationaryGate:
    """静止目标降频：所有活跃轨迹连续 N 帧几乎不动时，每 interval 帧才推理一次。

    只对"开着跟踪"的场景有意义（没有 track_id 就无法判断谁静止）；默认关闭
    （analytics.stationary_frames = 0），开启前建议先用真实视频确认轨迹不被打断。
    """

    def __init__(self, frames=0, interval=10, min_move=2.0):
        self.frames = max(int(frames or 0), 0)
        self.interval = max(int(interval or 1), 1)
        self.min_move = float(min_move)
        self._last = {}                # track_id -> (cx, cy)
        self._still = {}               # track_id -> 连续静止帧数
        self.skipped = 0

    @property
    def enabled(self):
        return self.frames > 0

    def should_infer(self, res, frame_no):
        """根据上一帧结果决定本帧要不要推理；返回 True 表示需要推理。"""
        if not self.enabled or res is None or res.boxes is None or not len(res.boxes):
            self._last.clear()
            self._still.clear()
            return True
        ids = getattr(res.boxes, 'id', None)
        if ids is None:
            return True
        tids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        xyxy = res.boxes.xyxy.tolist()
        if not tids:
            return True
        for tid, box in zip(tids, xyxy):
            if tid is None:
                return True            # 有框没 id，说明跟踪没收敛，照常推理
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            prev = self._last.get(tid)
            moved = 0.0 if prev is None else ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
            self._last[tid] = (cx, cy)
            if moved <= self.min_move:
                self._still[tid] = self._still.get(tid, 0) + 1
            else:
                self._still[tid] = 0
        quiet = all(self._still.get(t, 0) >= self.frames for t in tids)
        if not quiet or int(frame_no) % self.interval == 0:
            return True
        self.skipped += 1
        return False
