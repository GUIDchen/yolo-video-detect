#!/usr/bin/env python3
# 生成测试视频：给图片就做平移（能得到真实检测结果），不给就用合成图形做冒烟测试
import argparse
from pathlib import Path

import cv2
import numpy as np


def from_image(image, w, h, n):
    ih, iw = image.shape[:2]
    s = max(w / iw, h / ih) * 1.25                       # 放大留出平移余量
    big = cv2.resize(image, (int(iw * s), int(ih * s)))
    for i in range(n):
        t = i / max(n - 1, 1)
        x = int((big.shape[1] - w) * t)
        y = int((big.shape[0] - h) / 2)
        yield big[y:y + h, x:x + w]


def synthetic(w, h, n):
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(f, (60 + i * 6 % (w - 120), h // 2), 40, (200, 200, 200), -1)
        cv2.rectangle(f, (i * 9 % (w - 90), 60), (i * 9 % (w - 90) + 90, 180), (120, 180, 240), -1)
        yield f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='', help='可选：本地图片路径，生成带真实目标的视频')
    p.add_argument('--out', default='data/test.mp4')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--seconds', type=int, default=4)
    a = p.parse_args()

    n = a.fps * a.seconds
    img = cv2.imread(a.image) if a.image else None
    if a.image and img is None:
        raise SystemExit(f'[错误] 读不到图片: {a.image}')
    it = from_image(img, a.width, a.height, n) if img is not None else synthetic(a.width, a.height, n)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (a.width, a.height))
    for f in it:
        vw.write(f)
    vw.release()
    print(f'[OK] {a.out} | {n} 帧 | {a.width}x{a.height}@{a.fps}fps | '
          f'模式: {"真实图片平移" if img is not None else "合成图形"}')


if __name__ == '__main__':
    main()
