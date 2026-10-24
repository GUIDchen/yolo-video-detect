# 优化完善方案

> 版本：v0.5.0 · 审阅范围：整个仓库（约 1.7k 行 Python + 配置/CI/部署）
> 方法：逐模块通读代码 → 对照业界标杆（Frigate / supervision / Ultralytics solutions / vidgear / PyAV / DeepStream / k8s）
> 结论先行：**本轮不改模型、不重写架构**，先把"资源收尾、停机、防抖、可观测"这四类会直接造成线上事故的地方补齐；
> 收益最大但改造量也最大的"读取层重构"与"运动门控"列入 P1，理由与验收标准见第 5 节。

---

## 1. 现状盘点

| 维度 | 现状 |
| --- | --- |
| 代码规模 | `yolo_app/` 13 个模块约 1.6k 行；测试 1 个文件 34 个用例 |
| 输入 | 本地文件 / 目录批量 / 摄像头 / RTSP，统一走 `sources.VideoSource` |
| 推理 | Ultralytics YOLO（YOLO26 / YOLO11 / v8 均可），可选 ByteTrack 跟踪 |
| 业务 | ROI 过滤、越线计数、告警抓拍、webhook（企业微信/飞书/generic）、录像分段与留存 |
| 服务化 | FastAPI 面板：MJPEG + WebSocket + Prometheus；Docker + systemd |
| 工程化 | Makefile、ruff、CI（ruff → check → 单测 → 端到端）、CHANGELOG |

**做得好的部分（本轮不动）**：配置三层合并、`Detector` 与 ultralytics 解耦、后处理纯几何可单测、
离线权重定位、H.264 优先 + mp4v 兜底、Docker 与 systemd 双路径部署。

---

## 2. 评审发现

分级口径：**P0** = 会丢数据/损坏产物/泄漏；**P1** = 线上可感知的不稳定或性能浪费；**P2** = 工程化与可维护性。

### 2.1 P0（本轮已全部修复）

| # | 位置 | 问题 | 后果 |
| --- | --- | --- | --- |
| P0-1 | `pipeline.process` | 主循环没有 `try/finally`：推理异常、Ctrl-C、面板停机时 `VideoSink.close()` 与 `src.release()` 都不执行 | mp4 缺 moov 原子**打不开**；摄像头句柄泄漏，下次启动占用失败 |
| P0-2 | `pipeline.process` | 无法从外部停止；Web 面板关闭时 daemon 线程被硬杀 | 录像截断、统计不落盘 |
| P0-3 | `sources.VideoSource.read` | 重连计数是"整段运行总次数"，成功恢复后不清零 | 直播流断 3 次后**永久不再重连**，只能重启进程 |
| P0-4 | `analytics.LineCounter` | `prev` 字典只增不减，track 消失后状态永不回收 | 7×24 直播流内存线性增长 |
| P0-5 | `analytics.LineCounter` | 单帧判穿越，框中心在线附近抖动会反复计 in/out | 计数虚高，告警刷屏 |

### 2.2 P1（部分本轮修复，其余列入计划）

| # | 位置 | 问题 | 本轮 |
| --- | --- | --- | --- |
| P1-1 | `pipeline` | `avg_fps` 把重连等待、告警 HTTP 阻塞都算进分母，读数失真 | ✅ 拆分 `avg_fps` / `infer_fps` / `infer_ms_avg` |
| P1-2 | `alerts` | 同一告警文本在阈值抖动时重复发送 | ✅ `dedupe_s` 静默窗口 |
| P1-3 | `webapp` | 无健康检查端点；Docker HEALTHCHECK 跑 `check`（每次加载模型，5-10s） | ✅ `/healthz` + Dockerfile 改用 curl |
| P1-4 | `webapp` | 无鉴权；`--host 0.0.0.0` 时画面裸奔 | ✅ `serve.token`（401 有测试） |
| P1-5 | `webapp` | `/stream` 生成器 `while True`，检测结束后连接永久挂起 | ✅ 结束后主动收尾 |
| P1-6 | `sources` | 直播源未设采集缓冲，旧帧堆积成延迟 | ✅ `camera.buffer_size: 1` |
| P1-7 | `sources` | `cv2.VideoCapture` 无读超时，RTSP 僵死会阻塞整个循环 | ⏳ P1 计划（读取层重构） |
| P1-8 | `pipeline` | 每帧都推理，无运动门控/静止降频 | ⏳ P1 计划 |

### 2.3 P2（记录在案，未排期）

- `outputs.EventSink.check` 同步 `imwrite`，JPEG 落盘卡主循环；轻量方案是丢给单线程写盘队列。
- 告警 HTTP 同步发送（5s 超时）在越线密集时直接拖慢帧率；可改有界队列 + 后台线程。
- `camera` 申请分辨率后不回读校验，驱动给什么用什么（日志已打印实际值）。
- 目录批量里同名视频（不同子目录）输出会互相覆盖，应带相对路径哈希。
- 配置无 schema 校验：YAML 拼错键名静默忽略。
- WebSocket 每秒推全量统计，未做增量；大类别集时包体偏大。

---

## 3. 业界标杆对照

| 做法 | 出处 | 对本项目的适用性 | 本轮 |
| --- | --- | --- | --- |
| 双阈值（低阈值跟踪、高阈值判真）+ 面积/长宽比过滤 | [Frigate objects](https://docs.frigate.video/configuration/objects/) | 高，零架构改动；需重新标定 conf | 计划 P1 |
| 运动门控：先帧差筛运动区域，无运动不推理 | [Frigate motion](https://docs.frigate.video/configuration/motion_detection/) | 高（单机 CPU 收益最大），但要处理夜间噪声与 mask 误配 | 计划 P1 |
| ROI 裁剪 + 小输入尺寸（320）再放大 | [Frigate object detectors](https://docs.frigate.video/configuration/object_detectors/) | 高，与现有 `--roi` 天然契合 | 计划 P1 |
| 越线防抖：连续 N 帧满足才计数；用线段相交而非点判定 | [supervision LineZone](https://supervision.roboflow.com/latest/detection/tools/line_zone/)、[Ultralytics object_counter](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/object_counter.py) | 高，已实现防抖；线段相交判定列入 P2 | ✅ 防抖 |
| 统一 bottom-center 锚点（zones / analytics / PolygonZone 三处一致） | [Frigate zones](https://docs.frigate.video/configuration/zones/) | 中；当前用框中心，改锚点会改变既有计数口径 | 计划 P2（需迁移说明） |
| track_id 生命周期清理 | [Ultralytics solutions](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py) | 高，已用 TTL 等效实现 | ✅ |
| 读取层换 FFmpeg 子进程 / PyAV，有界队列 + 看门狗 | [PyAV cookbook](https://pyav.basswood.io/docs/stable/cookbook/basics.html)、[vidgear TQM](https://abhitronix.github.io/vidgear/latest/bonus/TQM/) | 高但改造大；是 RTSP 多路场景的必由之路 | 计划 P1 |
| 指标族照搬 `camera_fps / process_fps / skipped_fps / detector_inference_speed` | [Frigate metrics](https://docs.frigate.video/configuration/metrics/) | 高，命名直接对齐便于复用心智 | ✅ 部分（fps / infer_ms / skipped / reconnects） |
| 优雅停机顺序：停接入 → 关源 → 排空 → release writer → flush 指标 | [k8s pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) | 高，已实现"stop_event + join(10s)" | ✅ |
| 告警按业务身份去重（camera+label+zone+窗口），而非按时间戳 | [Frigate review](https://docs.frigate.video/configuration/review/) | 中；当前按文本去重，已够用 | ✅ 简化版 |
| 事件落库 + outbox 重试（幂等键/退避/死信） | [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/) | 低（单机单进程），规模上来再说 | 不排期 |
| 独立推理服务（OVMS/Triton）、DeepStream 全栈、go2rtc restream | [OVMS](https://docs.openvino.ai/2025/model-server/ovms_docs_parameters.html)、[DeepStream nvdsanalytics](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsanalytics.html) | 低，多一跳延迟 + 故障面翻倍，与"单机单进程"定位冲突 | 明确不做 |

**一条明确取舍**：Python 里"多进程 + Prometheus"是冲突的（`PROMETHEUS_MULTIPROC_DIR` 需每次清空、自定义 collector 不可用、
Gauge 不能用 `set_function`，见 [client_python 多进程文档](https://prometheus.github.io/client_python/multiprocess/)）。
因此监控口径保持"单进程内累计"，`--workers` 多进程只用于离线批处理，不进指标体系。

---

## 4. 差距分析（一句话结论）

1. **正确性**：原实现把"能跑通"当作终点，异常/中断路径完全没有收尾 —— 这是本轮最大的修复面。
2. **稳定性**：直播流的重连语义写反了（总额度 vs 连续失败），且没有读超时兜底。
3. **算法后处理**：越线判定缺防抖与轨迹回收，长时间运行会同时出现"计数漂移"和"内存增长"。
4. **可观测性**：有指标但没有延迟、跳帧、重连、告警失败这几个关键维度；健康检查靠加载模型。
5. **性能**：单机 CPU 场景真正的杠杆是"少推理"（门控/裁剪/降频），当前还是逐帧全图推理。

---

## 5. 优化清单与验收标准

### 5.1 本轮已实施（v0.5.0）

| 项 | 改法 | 验收标准 | 验证 |
| --- | --- | --- | --- |
| 资源收尾 | `pipeline.process` 主循环包 `try/finally`，release 顺序：源 → 写手 → 窗口 | 异常中断后 mp4 可被 `cv2.VideoCapture` 打开 | `test_error_mid_run_still_finalises_video` |
| 优雅停机 | `stop_event` 注入 + 每帧检查；FastAPI lifespan 置位并 `join(10s)` | 30s 视频中途 SIGTERM：日志出现"收到停止信号"，JSON 带 `interrupted: true`，mp4 帧数=已处理帧数 | 手工实测（见第 7 节） |
| Ctrl-C | `KeyboardInterrupt` 视为提前收尾，仍写统计 | 终端 Ctrl-C 后产物完整 | 代码路径 + 停机实测同源 |
| 重连语义 | 连续失败计数，成功清零；负数 = 无限重试 | 恢复后不再"用光额度" | `test_live_source_recovers` |
| 采集延迟 | 直播源设 `CAP_PROP_BUFFERSIZE` | `camera.buffer_size: 1` 生效（驱动支持时） | 配置 + 代码路径 |
| 轨迹回收 | `LineCounter` 按 `track_ttl` 淘汰 | 200 帧长跑后活跃 track 数有界 | `test_track_ttl_evicts_stale_tracks` |
| 越线防抖 | `min_cross_frames`（默认 2），抖动可撤销 | 单帧抖动不计；连续 2 帧计 1 次 | `TestLineDebounce` 两条 |
| 告警去重 | `alert.dedupe_s`（默认 3s）窗口内同文本只发一次 | 重复文本 `skipped` 递增，不重复发送 | `test_same_text_within_window_is_skipped` |
| 指标补齐 | `infer_ms_avg/max`、`frames_skipped`、`reconnects`、`alert_sent/failed/skipped` | `/metrics` 能抓到全部新指标 | `TestMetricsExtra` + 面板实测 |
| 健康检查 | `/healthz` 读内存状态（毫秒级）；Docker HEALTHCHECK 改 curl | 容器内 30s 一次探测不再加载模型 | 面板实测 |
| 面板鉴权 | `serve.token` / `--token`；`/stats /metrics /stream /ws` 校验 | 无 token 401，有 token 200 | `TestWebApp` |
| 对外暴露告警 | 监听非回环且无 token 时启动日志告警 | 日志可见 | 代码路径 |
| 流收尾 | `/stream` 在检测结束后主动结束响应 | 播放器不再永久挂起 | 代码路径 |

### 5.2 P1（v0.6.0 已全部实施）

| 项 | 实现 | 验收结果 |
| --- | --- | --- |
| 读取层重构 | `yolo_app/reader.py`：OpenCV 与 FFmpeg 两个后端；FFmpeg 走 rawvideo 管道 + 读线程 + `camera.buffer_size` 有界队列，`run.read_timeout` 判僵死并按 `run.reconnect` 重启；对外保持 `VideoSource.read()` | 模拟"子进程不产帧"后自动重启（`test_stalled_stream_restarts`）；两后端首帧 max diff 3 / mean 1.1 |
| 运动门控 | `yolo_app/motion.py` 帧差 + 模糊 + 形态学 + ROI 掩码 | 静态视频 75 帧只推理 1 次（跳过 74 帧 = 98.7%）；有运动视频跳过 0 帧 |
| 双阈值 | `analytics.ConfFilter` + `model.report_conf` | `data/test.mp4` 过滤 26 个低分框，`unique_tracks` 保持 6（轨迹不断） |
| ROI 裁剪推理 | `expand_box` / `remap_boxes` + `analytics.roi_crop` | 受控基准：全图 640 269.8ms → 裁剪 320 151.0ms（**1.79×**） |
| 静止目标降频 | `analytics.StationaryGate`（默认关闭） | 静态视频 75 帧推理 10 次；移动目标不抽帧（单测覆盖） |

遗留风险（记录在案，不阻塞）：FFmpeg 后端在**文件**上比 OpenCV 多一层管道与色彩转换开销，
因此 `auto` 只给直播源启用 FFmpeg；ROI 裁剪不支持分割/姿态（遇到时自动退回全图并告警）。

### 5.3 P2（记录，不承诺时间）

事件表落库（SQLite）+ 告警 outbox 重试、ROI 裁剪与线段相交判定、bottom-center 锚点（需迁移说明）、
配置 schema 校验、输出命名去重、WebSocket 增量推送、EventSink 异步写盘。

---

## 6. 新增配置项（v0.5.0 → v0.6.0）

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `camera.buffer_size` | 1 | 直播源采集缓冲；1 = 只留最新帧 |
| `analytics.track_ttl` | 300 | 越线计数 track 存活帧数，超时淘汰；0 = 不淘汰 |
| `analytics.min_cross_frames` | 2 | 连续 N 帧在新侧才计一次穿越；1 = 不防抖 |
| `alert.dedupe_s` | 3.0 | 相同告警文本静默窗口 |
| `serve.token` | '' | 非空时数据接口需 `?token=` 或 `X-Token` |
| `run.reader` | auto | auto = 直播源用 FFmpeg（低延迟 + 看门狗），文件/摄像头用 OpenCV |
| `run.read_timeout` | 10 | 直播源多久没收到帧判定僵死并重启（秒） |
| `model.report_conf` | 0 | 双阈值：低于该值的框只送跟踪，0 = 沿用 `model.conf` |
| `analytics.stationary_frames` | 0 | 静止降频触发帧数；0 = 关闭（需开跟踪） |
| `analytics.roi_crop` | false | 只裁 ROI 区域推理；需先设 `analytics.roi` |
| `analytics.roi_margin` | 0.1 | 裁剪外扩比例 |
| `analytics.roi_imgsz` | 0 | 裁剪推理输入尺寸；0 = 沿用 `model.imgsz` |
| `motion.enable` | false | 运动门控总开关 |
| `motion.threshold / min_area_ratio / blur` | 25 / 0.002 / 5 | 门控灵敏度 |
| `motion.mask_roi` | true | 只统计 ROI 内的运动 |
| `motion.hold_frames` | 0 | 无运动时沿用上一帧框显示多少帧 |

对应 CLI 新增：`serve --token`、`--report-conf`。其余项通过 YAML 配置。

---

## 7. 验证记录（本机 Apple Silicon，CPU）

```text
$ make test
Ran 61 tests ... OK            # v0.5.0: 34 → v0.6.0: 61（+27 条 P1 用例）

$ make lint
All checks passed!

# v0.6.0 实测（YOLO11n，CPU）
运动门控   ：有运动视频跳过 0 帧；静态视频 75 帧只推理 1 次
双阈值     ：--report-conf 0.5 过滤 26 个低分框，unique_tracks 仍为 6
静止降频   ：静态视频 75 帧推理 10 次（每 10 帧抽查）
ROI 裁剪   ：全图 640 269.8ms → 裁剪 320 151.0ms（1.79×）
读取后端   ：OpenCV vs FFmpeg 首帧 max diff 3 / mean 1.1；僵死 0.5s 判定并自动重启

# 面板实测（YOLO11n，data/test.mp4）
$ curl /healthz
{"status":"finished","frames":75,"uptime_s":7.4,"finished":true,"error":null}
$ curl /stats            → 401（未带 token）
$ curl "/stats?token=…"  → {"frames":75,"fps":20.6,"infer_ms_avg":46.81,"infer_ms_max":754.49,…}
$ curl "/metrics?token=…" | head
yolo_frames_inferred_total 75
yolo_infer_milliseconds_avg 46.81
yolo_frames_skipped_total 0

# 优雅停机实测：30s 合成视频，跑到第 158 帧时 SIGTERM
[INFO] 收到停止信号，处理提前收尾
[INFO] 完成 /tmp/long.mp4 | 推理 158 帧 | 18.8 FPS
[INFO] 面板已关闭，检测线程已收尾
产物校验：mp4 opened=True frames=158；json interrupted=True
```

---

## 8. 风险与回滚

- **越线口径变化**：`min_cross_frames` 默认 2 会让"单帧跨线"不再计数。回滚：配置 `analytics.min_cross_frames: 1`。
- **重连语义变化**：`run.reconnect` 现在指"连续失败次数"。老配置 `reconnect: 3` 含义更宽松（可反复恢复），
  不影响现有行为，只影响"是否永久放弃"。回滚：不需要，旧行为是缺陷。
- **鉴权默认关闭**：仅监听回环时无感；对外暴露需显式设 token（启动日志会提示）。
- **读取后端默认变化**：直播源默认改走 FFmpeg。回滚：`run.reader: opencv`。
- 回滚整轮：`git revert <commit>`，无数据迁移、无配置破坏性变更。

---

## 9. 参考

- Frigate：motion / objects / zones / record / review / metrics 官方文档（上文表格内逐条链接）
- supervision LineZone、Ultralytics solutions（object_counter / solutions.py）
- PyAV cookbook、vidgear TQM、FFmpeg protocols
- k8s Pod lifecycle（优雅停机）、Prometheus client_python 多进程限制
