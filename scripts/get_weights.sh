#!/usr/bin/env bash
# 网络受限（GitHub 直连不通）时用它拉取权重：bash scripts/get_weights.sh [模型名...]
#   bash scripts/get_weights.sh                 # 默认 yolo11n.pt + 样例图
#   bash scripts/get_weights.sh yolo11s.pt      # 换更大的模型（更准更慢）
#   bash scripts/get_weights.sh yolo11n.pt yolo11m.pt yolo11x.pt
set -e
cd "$(dirname "$0")/.."
mkdir -p weights data

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(yolo11n.pt)
fi

for m in "${MODELS[@]}"; do
  if [ -f "weights/$m" ]; then
    echo "[跳过] weights/$m 已存在"
    continue
  fi
  echo "[下载] $m"
  # 国内可达的镜像；也可换成 gh-proxy.com / modelscope 的同名地址
  curl -fL --max-time 300 -o "weights/$m" \
    "https://hf-mirror.com/ultralytics/YOLO11/resolve/main/$m"
done

if [ ! -f data/bus.jpg ]; then
  curl -fL --max-time 60 -o data/bus.jpg \
    https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
fi

ls -lh weights/ data/
