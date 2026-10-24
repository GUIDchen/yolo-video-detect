"""运动门控：先用帧差判断"画面有没有动"，没动就不推理。

单机 CPU 场景最大的浪费是"对着静止画面逐帧跑模型"。这里用最便宜的算法先筛一遍：

    灰度 → 高斯模糊（压噪点）→ 与上一帧 absdiff → 阈值 → 形态学开运算（去飞点）
    → 可选 ROI 掩码 → 非零像素占比 ≥ min_area_ratio 才认为有运动

代价是一次灰度转换 + 一次 absdiff（640×480 约 1ms），远低于一次推理（几十毫秒）。
已知失效场景：夜间噪点、光照突变、镜头晃动会让门控"常开"，表现为退化成逐帧推理
（不会漏检，只是省不下算力）；调大 threshold / min_area_ratio 即可。
"""
import cv2
import numpy as np


class MotionGate:
    """帧差运动检测；有运动返回 True。首帧没有参照，一律返回 True。"""

    def __init__(self, threshold=25, min_area_ratio=0.002, blur=5, roi=None):
        self.threshold = max(int(threshold), 1)
        self.min_area_ratio = max(float(min_area_ratio), 0.0)
        self.blur = int(blur) if int(blur or 0) > 1 else 0
        self.roi = roi or []
        self.mask = None
        self._mask_shape = None
        self.prev_gray = None
        self.last_ratio = 1.0          # 最近一帧的运动面积占比（写日志/调试用）
        self.enabled = True

    def _ensure_mask(self, shape):
        """ROI 掩码按帧尺寸懒构建；分辨率变了（重连换流）自动重建。"""
        if not self.roi:
            return
        if self.mask is not None and self._mask_shape == shape:
            return
        self.mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(self.mask, [np.array(self.roi, dtype=np.int32)], 255)
        self._mask_shape = shape

    def update(self, frame):
        """喂一帧，返回"是否有运动"。首帧与分辨率变化时返回 True（必须推理）。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self._ensure_mask(gray.shape)
            return True

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, th = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.dilate(th, kernel, iterations=1)

        self._ensure_mask(gray.shape)
        denom = th.size
        if self.mask is not None:
            th = cv2.bitwise_and(th, self.mask)
            denom = int(cv2.countNonZero(self.mask)) or th.size
        self.last_ratio = cv2.countNonZero(th) / float(denom)
        return self.last_ratio >= self.min_area_ratio


def overlay_boxes(frame, res, color=(0, 200, 0)):
    """把上一帧的框画到当前帧上（motion.hold_frames > 0 时用，避免画面突然"空掉"）。"""
    out = frame.copy()
    boxes = getattr(res, 'boxes', None)
    if boxes is None or not len(boxes):
        return out
    for x1, y1, x2, y2 in boxes.xyxy.tolist():
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return out
