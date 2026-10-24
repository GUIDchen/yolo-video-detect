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


def _sign(value, eps=1e-9):
    """浮点叉积归一成 +1 / 0 / -1；eps 用来兜住中心点 .5 坐标的舍入误差。"""
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _on_segment(a, b, pt, eps=1e-9):
    """点 pt 在 ax/ay、bx/by 张成的包围盒内（调用方已保证三点共线）。"""
    return (min(a[0], b[0]) - eps <= pt[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= pt[1] <= max(a[1], b[1]) + eps)


def _segments_intersect(p1, p2, q1, q2):
    """线段 p1→p2 与线段 q1→q2 是否相交（含端点接触与共线重叠）。

    叉积法：两点分居直线 q1q2 两侧、且 q1q2 也分居直线 p1p2 两侧时必然相交；
    其余情况逐个判断端点是否落在对方线段上，覆盖"端点刚压线"与"共线重叠"。
    """
    d1 = _side((q1, q2), p1)
    d2 = _side((q1, q2), p2)
    d3 = _side((p1, p2), q1)
    d4 = _side((p1, p2), q2)
    s1, s2, s3, s4 = _sign(d1), _sign(d2), _sign(d3), _sign(d4)
    if s1 * s2 < 0 and s3 * s4 < 0:
        return True
    return ((s1 == 0 and _on_segment(q1, q2, p1)) or
            (s2 == 0 and _on_segment(q1, q2, p2)) or
            (s3 == 0 and _on_segment(p1, p2, q1)) or
            (s4 == 0 and _on_segment(p1, p2, q2)))


def _anchor_xy(box, anchor='center'):
    """检测框锚点：center=(cx, cy)，bottom=(cx, y2)（框底边中点）。"""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    cx = (x1 + x2) / 2
    return (cx, y2) if anchor == 'bottom' else (cx, (y1 + y2) / 2)


def _normalize_anchor(anchor):
    """校验锚点取值，配置写错时尽早报错，避免静默用错口径。"""
    value = str(anchor or 'center').strip().lower()
    if value not in ('center', 'bottom'):
        raise ValueError(f"anchor 只能是 center 或 bottom，当前为 {anchor!r}")
    return value


class RegionFilter:
    """只保留锚点落在 ROI 多边形内的检测框；roi 为空表示不过滤。

    anchor 与 analytics.anchor 同一套口径：center 看框中心，bottom 看框底边中点。
    """

    def __init__(self, roi=None, anchor='center'):
        self.roi = roi or []
        self.anchor = _normalize_anchor(anchor)

    @property
    def enabled(self):
        return bool(self.roi)

    def keep(self, res):
        """就地剔除 ROI 之外的检测框（改 res.boxes），返回同一个 Results 供链式调用。"""
        if not self.enabled or res.boxes is None or not len(res.boxes):
            return res
        xyxy = res.boxes.xyxy.tolist()
        idx = [i for i, b in enumerate(xyxy) if _inside(self.roi, _anchor_xy(b, self.anchor))]
        res.boxes = res.boxes[idx]
        return res

    def draw(self, frame, color=(0, 255, 255)):
        import cv2
        import numpy as np
        if self.enabled:
            cv2.polylines(frame, [np.array(self.roi, dtype=np.int32)], True, color, 2)
        return frame


class LineCounter:
    """越线计数：按 track_id 记录上一帧锚点，用线段相交判定是否跨过计数线。

    每帧把上一帧锚点 P_prev 与当前锚点 P_cur 连成线段，只有该线段与线 L 相交
    （含端点接触与共线重叠）才算一次穿越——目标一帧内跨过整条线因此不会漏计。
    方向约定：线由 p1→p2，其逆时针一侧为 +、顺时针一侧为 -；
    - 从 - 侧穿到 + 侧记为 in，从 + 侧穿到 - 侧记为 out
      （水平线 p1 在左 p2 在右时，自上而下为 in）。

    min_cross_frames 现在是同一 track 的计数冷却帧数：一次计数后在接下来
    min_cross_frames 帧内再次相交不重复计数，用来压住线附近的来回抖动；
    track_ttl 仍负责淘汰长时间不出现的 track。
    """

    def __init__(self, line, classes=None, names=None, track_ttl=300, min_cross_frames=2,
                 anchor='center'):
        self.line = line or []
        self.classes = set(classes or [])
        self.names = names or {}
        self.anchor = _normalize_anchor(anchor)
        self.prev = {}                 # track_id -> 上一帧锚点 (x, y)
        self.total = {"in": 0, "out": 0}
        self.by_class = defaultdict(lambda: {"in": 0, "out": 0})
        self.track_ttl = int(track_ttl or 0)   # 0 = 不淘汰（老行为）
        # 计数冷却：同一 track 计数后 min_cross_frames 帧内再次相交不重复计数。
        # 注意它不再要求"连续 N 帧落在新侧"，单帧大跨度穿越也会立即计数。
        self.min_cross_frames = max(int(min_cross_frames or 1), 1)
        self._last_side = {}           # track_id -> 最近一次确定的所在侧（+1 / -1）
        self._on_line = set()          # track_id -> 上一帧锚点压线（共线重叠只计一次）
        self._last_cross = {}          # track_id -> 最后一次计数的帧号（冷却基准）
        self._last_seen = {}           # track_id -> 最后出现的帧号
        self._frame = 0

    @property
    def enabled(self):
        return len(self.line) == 2

    def update(self, res, names, frame_no=None):
        """更新各 track 的上一帧锚点；返回本帧发生的穿越事件列表（无则为 None）。

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
            pt = _anchor_xy(xyxy[i], self.anchor)
            if tid is None:
                continue
            tid = int(tid)
            prev_pt = self.prev.get(tid)
            self._last_seen[tid] = self._frame
            if prev_pt is None:                 # 第一次见到该 track：只记锚点
                self.prev[tid] = pt
                self._last_side[tid] = _sign(_side(self.line, pt))
                continue
            direction, after = None, 0
            if _segments_intersect(prev_pt, pt, self.line[0], self.line[1]):
                direction, after = self._direction(tid, prev_pt, pt)
                if direction and self._cooldown_ok(tid):
                    name = names.get(cid, str(cid))
                    self.total[direction] += 1
                    self.by_class[name][direction] += 1
                    self._last_cross[tid] = self._frame
                    crossed.append({"track_id": tid, "class": name, "direction": direction})
                # 冷却期内的相交不计数，但下面仍更新状态，冷却结束后不会补记一次
            s_cur = _sign(_side(self.line, pt))
            if direction:
                self._last_side[tid] = after     # 终点压线时记住"正穿向哪一侧"
            elif s_cur:
                self._last_side[tid] = s_cur
            if s_cur == 0:
                self._on_line.add(tid)
            else:
                self._on_line.discard(tid)
            self.prev[tid] = pt
        self._prune()
        return crossed or None

    def _direction(self, tid, p_prev, p_cur):
        """判定本次相交的方向；返回 (方向, 穿过后的侧)，None 表示不构成换侧。"""
        s_prev = _sign(_side(self.line, p_prev))
        s_cur = _sign(_side(self.line, p_cur))
        if s_prev == 0 and s_cur == 0 and tid in self._on_line:
            return None, 0                       # 沿线上滑动：同一段共线只计一次
        before = s_prev or self._last_side.get(tid, 0)
        if before == 0:                          # 起点压线且无历史：按当前侧反推来向
            before = -s_cur if s_cur else -1
        after = s_cur or (-before if before else 1)   # 终点压线：视为正在穿过去
        if before == after:                      # 碰到线又回原侧，不算穿越
            return None, s_cur
        return ("in" if before < 0 else "out"), after

    def _cooldown_ok(self, tid):
        """同一 track 距上次计数是否已过冷却帧数。"""
        last = self._last_cross.get(tid)
        return last is None or self._frame - last >= self.min_cross_frames

    def _prune(self):
        """淘汰超过 track_ttl 帧没再出现的 track，避免长视频里状态无限增长。"""
        if not self.track_ttl or not self._last_seen:
            return
        cutoff = self._frame - self.track_ttl
        stale = [t for t, seen in self._last_seen.items() if seen < cutoff]
        for t in stale:
            self._last_seen.pop(t, None)
            self.prev.pop(t, None)
            self._last_side.pop(t, None)
            self._last_cross.pop(t, None)
            self._on_line.discard(t)

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


def expand_box(points, margin=0.1, width=None, height=None):
    """多边形 → 外扩后的矩形 (x1,y1,x2,y2)，可选裁剪到画面内。

    ROI 裁剪推理用它：外扩比例补偿"目标正好压在 ROI 边界被切掉"的情况。
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    mx, my = (x2 - x1) * float(margin), (y2 - y1) * float(margin)
    x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    if width:
        x1, x2 = max(0.0, x1), min(float(width), x2)
    if height:
        y1, y2 = max(0.0, y1), min(float(height), y2)
    return int(x1), int(y1), int(x2), int(y2)


def remap_boxes(res, dx=0, dy=0):
    """把检测框平移回原图坐标（裁剪推理后调用）。

    ultralytics 的推理结果张量带 InferenceMode 标记，原地改会抛
    "Inplace update to inference tensor outside InferenceMode"，因此先 clone 再改。
    """
    boxes = getattr(res, 'boxes', None)
    if boxes is None or not len(boxes):
        return res
    data = getattr(boxes, 'data', None)
    if data is None:                    # 测试替身（FakeBoxes）只有 xyxy，直接改
        xyxy = boxes.xyxy
        xyxy[:, 0] += dx
        xyxy[:, 1] += dy
        xyxy[:, 2] += dx
        xyxy[:, 3] += dy
        return res
    data = data.clone() if hasattr(data, 'clone') else data.copy()
    boxes.data = data
    data[:, 0] += dx
    data[:, 1] += dy
    data[:, 2] += dx
    data[:, 3] += dy
    return res


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
