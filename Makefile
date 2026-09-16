# 常用命令封装：make <target>
PY := .venv/bin/python

help:            ## 显示所有可用命令
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:         ## 建虚拟环境并安装依赖
	python3.12 -m venv .venv && $(PY) -m pip install -U pip -r requirements.txt

check:           ## 环境与模型自检
	$(PY) -m yolo_app check

test:            ## 单元测试（含真模型端到端）
	$(PY) -m unittest discover -s tests -v

lint:            ## 代码检查
	$(PY) -m ruff check yolo_app tests scripts

run:             ## 检测测试视频（跟踪）
	$(PY) -m yolo_app run --source data/test.mp4 --track

camera:          ## 摄像头实时检测
	$(PY) -m yolo_app camera --index 0

serve:           ## 启动 Web 面板 http://127.0.0.1:8000
	$(PY) -m yolo_app serve --source 0

export:          ## 导出 ONNX
	$(PY) -m yolo_app export --format onnx --simplify

docker-build:    ## 构建镜像（国内可加 --build-arg BASE=... 见 README）
	docker build -t yolo-app:latest .

docker-up:       ## 后台启动容器
	docker compose up -d

clean:           ## 清理产物与缓存
	rm -rf outputs/* runs/* **/__pycache__
