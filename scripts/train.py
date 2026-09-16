#!/usr/bin/env python3
"""自有数据微调：标注好数据后一条命令训练，产物直接用 --model 加载。

    python scripts/train.py --data datasets/mydata/data.yaml --model yolo11n.pt --epochs 100
    python -m yolo_app run --source data/test.mp4 --model runs/detect/train/weights/best.pt

数据准备建议：labelImg / Roboflow 导出 YOLO txt 格式；train/val 建议 8:2，每类至少 100 张。
"""
import argparse


def main():
    p = argparse.ArgumentParser(description='YOLO 自有数据微调')
    p.add_argument('--data', required=True, help='数据集 yaml（见 configs/dataset.yaml 模板）')
    p.add_argument('--model', default='yolo11n.pt', help='预训练权重，微调通常从 n/s 开始')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default=None, help='cpu / 0 / 0,1')
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name', default='train')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, project=a.project, name=a.name, resume=a.resume)
    metrics = model.val()
    print(f'[完成] mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}')
    print(f'[下一步] python -m yolo_app run --source <视频> --model {a.project}/{a.name}/weights/best.pt')


if __name__ == '__main__':
    main()
