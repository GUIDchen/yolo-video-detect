# 服务器/无显示器部署：opencv headless 版，不带 GUI 依赖
# 国内网络若拉不动 docker.io，用镜像源构建：
#   docker build --build-arg BASE=docker.m.daocloud.io/library/python:3.12-slim \
#                --build-arg PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/ -t yolo-app .
ARG BASE=python:3.12-slim
FROM ${BASE}

ARG PIP_INDEX=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    YOLO_OFFLINE=true \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    PIP_NO_CACHE_DIR=1

# opencv-headless 仍需 libgl/libglib 运行时
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
# CPU 版 torch 体积更小；要 GPU 请换 nvidia/cuda 基础镜像并装 cuda 版 torch
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --index-url ${PIP_INDEX} -r requirements.txt \
    && pip uninstall -y opencv-python && pip install --index-url ${PIP_INDEX} opencv-python-headless

COPY . .
EXPOSE 8000
# 健康检查打的是面板的 /healthz（毫秒级，读内存状态）；用 run 子命令跑批处理时请 --no-healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
ENTRYPOINT ["python", "-m", "yolo_app"]
CMD ["serve", "--source", "0", "--host", "0.0.0.0"]
