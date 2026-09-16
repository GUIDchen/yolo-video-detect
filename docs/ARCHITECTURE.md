# 架构与代码说明

## 一句话总览

`yolo_app` 是一个**单进程可跑、多进程可扩**的视频分析管道：**源(Source) → 模型(Detector) → 后处理(Analytics) → 落地(Outputs/Metrics/Alerts)**，
命令行、Web 面板、Docker 三种入口共用同一条管道（`pipeline.process()`）。

```
        ┌──────────── CLI (cli.py) ────────────┐
        │ run / camera / serve / export / check│
        └───────────────┬──────────────────────┘
                        │ 命令行覆盖
              ┌─────────▼─────────┐
              │ config.py         │  DEFAULTS ← YAML ← CLI
              └─────────┬─────────┘
                        │ 配置字典 cfg
              ┌─────────▼──────────────────────────────────────┐
              │ pipeline.process(source, cfg, detector, ...)   │
              │                                                │
  sources.py ─┤ 读帧 → detector.infer → analytics(ROI/越线)    ├─ outputs.py（视频/事件/JSON）
  文件/摄像头 │      → 绘制 → 告警 → 写盘 → 统计               ├─ alerts.py（webhook）
  /RTSP/目录  │                                                ├─ metrics.py（Prometheus）
              └───────────────┬────────────────────────────────┘
                              │ on_frame / on_stats 回调
                        webapp.py（MJPEG + WebSocket + /metrics）
```

## 模块职责

| 文件 | 作用 | 关键符号 | 依赖 |
| --- | --- | --- | --- |
| `yolo_app/__main__.py` | 让 `python -m yolo_app` 可用 | `main()` | cli |
| `yolo_app/cli.py` | 命令行入口、参数解析、子命令分发、多进程调度 | `build_parser`、`build_overrides`、`main` | 其余全部 |
| `yolo_app/config.py` | 三层配置合并与默认值定义 | `DEFAULTS`、`load_config`、`deep_merge` | yaml |
| `yolo_app/detector.py` | 模型加载、权重定位、类别名→id 解析、单帧推理/跟踪 | `Detector`、`resolve_weights`、`resolve_class_ids` | ultralytics |
| `yolo_app/sources.py` | 统一视频源：文件/目录/摄像头/RTSP，断流重连 | `VideoSource`、`list_videos`、`is_camera` | OpenCV |
| `yolo_app/analytics.py` | 业务后处理：ROI 多边形过滤、越线计数 | `RegionFilter`、`LineCounter`、`_inside`、`_side` | 无（纯几何） |
| `yolo_app/outputs.py` | 结果落地：视频写入(含分段)、事件抓拍、JSON、留存清理 | `VideoSink`、`EventSink`、`cleanup_old`、`write_json` | OpenCV |
| `yolo_app/alerts.py` | 告警通道：企业微信/飞书/通用 webhook | `AlertChannel` | urllib（标准库） |
| `yolo_app/metrics.py` | Prometheus 文本指标生成 | `prometheus_text`、`write_metrics_file` | 无 |
| `yolo_app/pipeline.py` | **核心主循环**，串起上面所有模块；多进程 worker 入口 | `process`、`run_job`、`output_paths` | 上面全部 |
| `yolo_app/webapp.py` | FastAPI 面板：MJPEG/WebSocket/指标/统计接口 | `create_app`、`serve` | fastapi、uvicorn |
| `yolo_app/logging_setup.py` | 统一日志格式与级别 | `get_logger` | logging |

## 一帧数据的完整链路

1. **取帧** `sources.VideoSource.read()` —— 摄像头/RTSP 读失败会按 `run.reconnect` 重连。
2. **跳帧判断** `pipeline.process` —— `run.stride` 命中则原样写帧（保持视频时长）并 continue。
3. **推理/跟踪** `detector.Detector.infer()` —— `track.enable` 决定走 `model.track(persist=True)` 还是 `model.predict`。
4. **ROI 过滤** `analytics.RegionFilter.keep()` —— 框中心落在多边形外则从结果里剔除（直接改 `res.boxes`，后续统计/绘制都只看到区域内目标）。
5. **越线计数** `analytics.LineCounter.update()` —— 用 `track_id` 记住上一帧所在侧，翻转即记一次 in/out。
6. **绘制** `res.plot()` + ROI 多边形 + 越线线（都是画在同一张 ndarray 上）。
7. **事件抓拍** `outputs.EventSink.check()` —— 命中所配类别且置信度达标且过了冷却时间 → 存图 + 记事件。
8. **告警** `alerts.AlertChannel` —— 抓拍与越线都会发一条文本；失败只记日志。
9. **写盘** `outputs.VideoSink.write()` —— 首帧才建 writer（兼容未知分辨率的直播源），H.264 优先，按 `segment_seconds` 切分。
10. **回调** `on_frame` / `on_stats` —— Web 面板据此取最新画面与统计，本机跑时为空。
11. **收尾** —— 汇总统计写 `<源名>.json`，多源时再写 `summary_all.json`。

## 配置流

```
config.DEFAULTS（代码内默认，保证任何字段都有值）
      ↓ deep_merge
configs/default.yaml（可复制成 production.yaml，多环境各一份）
      ↓ deep_merge
cli 命令行参数（只覆盖显式传入的项）
      ↓
cfg 字典 —— 全程只读传递，不依赖全局变量
```

优点：新增一个配置项只需改三处（`DEFAULTS` → yaml 注释 → `build_overrides`），其它模块不用动。

## 并发模型

| 场景 | 模型 | 说明 |
| --- | --- | --- |
| 单路检测 | 单进程单线程 | 最简单，够用 |
| 目录批量 | `ProcessPoolExecutor`，一源一进程 | `--workers N`，各自加载模型（不用共享显存） |
| Web 面板 | 检测线程 + uvicorn 主线程 | `AppState` + 锁共享最新帧与统计 |

## 扩展点（改哪里）

| 要加的功能 | 只改 |
| --- | --- |
| 新输入（GStreamer、流媒体平台） | `sources.VideoSource._open/read` |
| 新后处理（越线之外：停留时长、聚集度） | `analytics.py` 加类，`pipeline.process` 里插一行 |
| 新输出（数据库、MQTT、S3） | `outputs.py` 加类，或参考 `alerts.AlertChannel` |
| 新指标 | `metrics.prometheus_text` 加一段文本 |
| 新接口 | `webapp.create_app` 加路由 |
| 新子命令 | `cli.build_parser` 加 parser，`main` 加分发 |

## 部署形态

- **本机开发**：`.venv` + `python -m yolo_app run/camera/serve`，VS Code F5 有 6 个现成配置。
- **服务器常驻**：`systemd` 跑 `serve`（Web 面板 + 指标），或用 `run` 处理定时落盘的视频目录（配合定时任务）。
- **容器**：`Dockerfile`（headless、CPU torch、HEALTHCHECK）、`docker-compose.yml` 挂载 `configs/weights/data/outputs`，RTSP 源直接写进 command。
- **GPU**：`--device 0` 本机跑，或换 `nvidia/cuda` 基础镜像 + cu 版 torch；再配 `export --format engine` 走 TensorRT。

## 关键设计取舍

- **H.264 优先**：mp4v 编码在 Chromium 内核（VS Code 内置播放器、浏览器）里放不出来，会表现为"黑屏"。因此默认 `avc1`，失败才回退。
- **权重本地回退**：ultralytics 默认从 GitHub Releases 下载权重，国内常不通；程序优先在项目内 `weights/` 找同名文件，避免卡死。
- **告警同步发送**：实现最简、语义清晰；高频告警场景再换队列（见 README 路线图）。
- **进程池而非线程池**：PyTorch 推理持有 GIL，多进程才能真正并行；代价是每进程一份模型。
- **回调而非继承**：`on_frame/on_stats` 让 Web 面板复用主循环，不复制一份检测逻辑。

## 已知边界

- 越线计数依赖跟踪稳定性，目标 ID 频繁切换会重复计数。
- 统计 `per_class_frames` 是逐帧次数，不是目标个数；唯一数看 `unique_tracks`。
- 指标为进程内累计，重启归零。
- Docker 内无 GUI；macOS 的 Docker 不支持摄像头直通。
