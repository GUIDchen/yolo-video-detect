# 部署说明

## 1. 服务器（systemd）

```bash
sudo useradd -r -s /bin/false yolo
sudo mkdir -p /opt/yolo-video-detect && sudo chown -R yolo /opt/yolo-video-detect
# 拷贝项目、建 venv、装依赖、放权重（同 README「安装」）
sudo cp deploy/yolo-app.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now yolo-app
journalctl -u yolo-app -f          # 看日志
```

监控：`curl http://服务器:8000/metrics` 接 Prometheus，或 `--metrics-file` 落到 textfile collector 目录。

## 2. 容器（推荐给多路 RTSP）

```bash
docker compose up -d            # 源写在 docker-compose.yml 的 command 里
docker compose logs -f
```

国内网络：见根 README 的 `--build-arg BASE/PIP_INDEX` 用法。

## 3. 定时批处理（录像文件）

```bash
# crontab: 每小时处理上一小时的录像目录，结果落到 outputs/
0 * * * * cd /opt/yolo-video-detect && .venv/bin/python -m yolo_app run \
    --source /data/recordings/hourly --workers 4 --keep-days 7 --max-files 500 >> /var/log/yolo-cron.log 2>&1
```

## 4. 上线前检查清单

- [ ] `python -m yolo_app check` 通过（模型、设备、类别数）
- [ ] 用真实场景视频跑一遍，确认 `conf`/`imgsz` 的检出率与误报可接受
- [ ] 摄像头/RTSP 源在本机与服务器上都连通（`ffprobe <rtsp url>`）
- [ ] `outputs/` 的留存策略（`keep_days`/`max_files`/`segment_seconds`）与磁盘容量匹配
- [ ] 告警 webhook 用真实机器人验证过一次
- [ ] 需要 GPU 时确认 `--device 0` 生效（`check` 输出里会打印设备）
