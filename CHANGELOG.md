# 变更记录

## v0.7.0 — P2 工程化（事件落库 / 告警 outbox / 配置校验 / 锚点 / 面板增量）

### 新增
- **事件落库（`yolo_app/store.py`）**：SQLite + WAL 的 `EventStore`，events 表记录每条越线/抓拍事件，
  outbox 表记录待发告警；`claim_due` 用 `BEGIN IMMEDIATE` 原子认领，多进程共享同一库也不会重复发送。
- **告警 outbox 重试**：发送失败按 `base_delay * 2^(n-1)` 退避，超过 `alert.retry.max_attempts` 置 dead；
  `alert.dedupe_s` 在入队前去重，重试不会被去重吞掉。`output.events.db = ''` 时不建库，退回旧的同步发送。
- **EventSink 异步写盘**：抓拍图交给单线程 + 有界队列（`output.events.async_queue`，默认 64，满则丢最旧并计数），
  主循环不再被 `imwrite` 阻塞；events 列表仍在调用线程同步维护，summary 立刻可用。
- **面板 WebSocket 增量推送**：统计带单调 `seq`；`/ws?since=<seq>` 只在 seq 变化时推送，空闲 15s 心跳；
  前端按 seq 去重、断线 3s 后带 since 重连。`/stats` 同步增加 `seq`。
- **越线改线段相交判定**：用上一帧锚点 → 当前锚点的线段与计数线求交（叉积法，含端点接触与共线重叠），
  一帧内大跨度跨越不再漏计；`min_cross_frames` 语义改为同一 track 的计数冷却帧数。
- **锚点可配置**：`analytics.anchor: center | bottom`（默认 center）。bottom = 框底边中点，
  与 Frigate / supervision / DeepStream 口径一致；切换会改变既有 ROI/越线口径，需重新标定。
- **配置 schema 校验**：`validate_config()` 在启动时检查类型与取值范围，错误信息带字段、当前值、期望；
  YAML 里的未知键以 warning 形式提示（`strict=True` 时抛错），CLI 校验失败返回 1 而不是抛栈。
- **输出命名去重**：`--name-mode unique`（或 `output.name_mode`）给结果名加源路径哈希，避免同名视频互相覆盖；
  目录批量且源 > 1 时自动启用。

### 变更
- 版本号 0.6.0 → 0.7.0；测试 61 → **128**（新增 store/outbox/异步写盘/WS 协议/锚点/配置校验/命名共 67 条）。
- `docs/CODE_WALKTHROUGH.md` 补写 `reader.py` 与 `motion.py` 两章（逐行 + 常见坑）。

### 验证
- `make test` 128 个用例全绿，`ruff` 通过。
- 端到端：越线事件落库 7 条、outbox 0 滞留；`--name-mode unique` 两次调用文件名稳定（`test_e32d8f.mp4`）；
  `anchor: bottom` 下越线计 1；`/stats` 返回 `seq`。

## v0.6.0 — P1 优化（读取层 / 运动门控 / 双阈值 / ROI 裁剪）

### 新增
- **运动门控**（`yolo_app/motion.py`）：帧差 + 高斯模糊 + 形态学开运算 + 可选 ROI 掩码，判断"画面有没有动"。
  配置 `motion.enable/threshold/min_area_ratio/blur/mask_roi/hold_frames`（默认关闭）；`hold_frames > 0` 时
  无运动帧沿用上一帧框显示，避免画面突然空白。实测：有运动视频跳过 0 帧，静态视频 75 帧只推理 1 次。
- **双阈值**：`model.report_conf`（CLI `--report-conf`）。低于该值的框只送跟踪器、不进统计/绘制/抓拍/越线，
  跟踪阈值仍是 `model.conf`，轨迹不会因为提高显示阈值而断。
- **静止降频**：`analytics.stationary_frames / stationary_interval / stationary_min_move`（默认关闭）。
  开着跟踪时，所有轨迹连续 N 帧几乎不动就按 interval 抽帧推理；实测静态视频推理帧数 75 → 10。
- **ROI 裁剪推理**：`analytics.roi_crop / roi_margin / roi_imgsz`（默认关闭）。按 ROI 外扩裁剪后用
  小尺寸推理，再把框平移回原图；遇到分割/姿态结果自动退回全图并告警。
  实测 `data/test.mp4`：全图 640 推理 270ms → 裁剪 320 推理 151ms（**1.79×**）。
- summary 新增 `frames_inference_skipped`、`skip_reasons`、`boxes_low_conf`、`reader`、`frames_dropped`；
  指标新增 `yolo_inference_skipped_total{reason}`、`yolo_frames_dropped_total`。

### 变更
- **读取层拆出 `yolo_app/reader.py`**：`OpenCVReader`（文件/摄像头）与 `FFmpegReader`（直播源）。
  FFmpeg 后端用 `ffprobe` 探测分辨率/帧率，`ffmpeg -f rawvideo -pix_fmt bgr24` 管道取帧，
  后台读线程 + `camera.buffer_size` 有界队列（队满丢最旧帧并计数），`run.read_timeout` 判僵死后按
  `run.reconnect` 语义重启。`VideoSource` 变成门面，新增 `opened/width/height/dropped/backend`。
- 新增配置 `run.reader`（auto|opencv|ffmpeg，默认 auto）与 `run.read_timeout`（默认 10s）；
  auto 仅对 RTSP/RTMP/HTTP 直播源启用 FFmpeg，文件与摄像头仍走 OpenCV（可用 `run.reader` 强制）。
- Dockerfile 安装 `ffmpeg`（约 +100MB）；不想要可设 `run.reader: opencv`。
- 版本号 0.5.0 → 0.6.0；测试 34 → **61**。

### 验证
- `make test` 61 个用例全绿，`ruff` 通过。
- 两后端首帧对比：max diff 3、mean 1.1（仅色彩转换舍入差异）。
- 单进程受控基准（同机、CPU）：全图 640 270ms vs 裁剪 320 151ms。

## v0.5.0 — 健壮性与可观测性（对照 Frigate/supervision 的差距分析）

### 修复（会直接造成线上事故的路径）
- **资源收尾**：`pipeline.process` 主循环改为 `try/finally`，异常 / Ctrl-C / 面板停机都会释放视频源与写手。
  此前推理中途异常会让 mp4 缺 moov 原子（播放器打不开）、摄像头句柄泄漏。
- **重连语义**：`run.reconnect` 从"整段运行累计次数"改为"单次故障连续重试次数"，成功读到帧即清零；
  负数 = 无限重试。旧实现断流三次后就永久不再重连。
- **轨迹回收**：`LineCounter` 新增 `track_ttl`（默认 300 帧），淘汰消失的 track，避免 7×24 直播内存线性增长。
- **越线防抖**：新增 `analytics.min_cross_frames`（默认 2），连续 N 帧落在新侧才计一次，抖动自动撤销。

### 新增
- **`/healthz`**：毫秒级健康检查（读内存状态），Docker HEALTHCHECK 由"每次加载模型跑 check"改为 curl 该端点。
- **面板鉴权**：`serve.token` / `serve --token`，`/stats /metrics /stream /ws` 需 `?token=` 或 `X-Token`；
  监听非回环且未设 token 时启动日志告警。
- **优雅停机**：`pipeline.process(stop_event=...)` + FastAPI lifespan，SIGTERM 时在帧边界收尾（join 10s）。
- **告警去重**：`alert.dedupe_s`（默认 3s），相同文本窗口内只发一次，`summary.alert_skipped` 计数。
- **采集缓冲**：`camera.buffer_size`（默认 1），直播源只保留最新帧，降低延迟累积。
- **指标补全**：`yolo_infer_milliseconds_avg/max`、`yolo_frames_skipped_total`、`yolo_reconnects_total`、
  `yolo_alert_sent/failed/skipped_total`；`summary` 增加 `infer_fps`、`frames_skipped`、`interrupted`。
- **`docs/OPTIMIZATION.md`**：整仓评审 + 标杆对照（Frigate/supervision/Ultralytics/vidgear/PyAV/k8s）+
  P0/P1/P2 路线与验收标准。

### 变更
- 版本号 0.4.1 → 0.5.0；README 能力清单、参数表、路线图同步。
- `avg_fps` 语义澄清为端到端帧率，新增 `infer_fps` 表示纯推理帧率（不再把重连/告警阻塞计入分母）。
- 测试 20 → **34**（新增资源收尾、停机、重连、防抖、去重、指标、面板鉴权用例）。

### 验证
- `make test` 34 个用例全绿，`ruff` 通过。
- 面板实测：`/healthz` 返回内存状态；无 token 401、有 token 200；`/metrics` 含全部新指标。
- 优雅停机实测：30s 视频跑到第 158 帧时 SIGTERM → 日志"收到停止信号"→ JSON `interrupted: true`、mp4 可播放且 158 帧。

## v0.4.1 — 本地源码对照调试

### 新增
- **Makefile 源码对照目标**：
  - `make install-src`：以 editable 方式安装同级 `../yolo-ultralytics`（`SRC=` 可覆盖路径），并把 `ultralytics-thop` 升到 `>=2.2.3`、`ultralytics-platform` 升到 `>=0.1.45`，最后打印生效版本；
  - `make install-wheel`：回滚到基线 `ultralytics==8.4.152` / thop 2.1.6 / platform 0.1.41（`BASELINE=` 可覆盖）；
  - `make yolo-env`：打印当前生效的 ultralytics 版本与导入路径，一眼区分 wheel / 源码。
- **README 新增「对照调试：本地 ultralytics 源码」**：切换命令、三个必看陷阱（`pip freeze` 的 VCS 渲染、editable 的绝对路径绑定、thop 版本门槛）。

### 变更
- 版本号 0.4.0 → 0.4.1。
- README「测试与 CI」「实测记录」的用例数 16 → 20，与当前测试集一致。

### 验证（本机 Apple Silicon，CPU）
- 源码模式下 `make test`：20 个用例全绿；`pip check` 无冲突。
- `make install-wheel` → `make install-src` 往返切换实测通过，`make yolo-env` 分别输出 `8.4.152 -> .venv/site-packages/...` 与 `8.4.187 -> ../yolo-ultralytics/ultralytics/__init__.py`。
- `model.info()`：thop 2.1.6 时静默 `0.0 GFLOPs`，2.2.3 后 `6.7 GFLOPs`。

## v0.4.0 — 开源发布

### 新增
- **LICENSE（AGPL-3.0）**：明确本项目因依赖 Ultralytics（AGPL-3.0）而以 AGPL-3.0 发布，并说明对权重、微调产物与网络服务的要求。
- **NOTICE.md**：逐个列出运行时依赖及其许可证（依据已安装包元数据核实），并说明仓库不含权重与素材。
- **README**：新增 CI／许可证／Python 版本徽章、流水线架构图（Mermaid）、许可证与合规说明。

### 变更
- 版本号 0.3.2 → 0.4.0。

### 说明
- 本次仅许可与文档变更，未改动运行逻辑；测试与 CLI 行为保持不变。

## v0.3.2 — 模型管理：清单 / 下载 / 对照切换

### 新增
- **`models` 子命令**（`yolo_app/models.py` + `cli.py`）：
  - `python -m yolo_app models`：列出 13 个官方预设（YOLO26 / YOLO11 / YOLOv8 的 n/s/m/l/x）与本地下载状态；
  - `models --download yolo26n.pt yolo11s.pt`：自动按 镜像→官方 HF→GitHub 代理 顺序下载到 `weights/`，已存在则跳过；
  - `models --benchmark [names] --frames N --imgsz X`：对同一段视频跑多个模型，输出 FPS/检出/误检对照表（关掉录像与抓拍，保证只比模型本身）。
- **支持最新一代 YOLO26**：实测 `yolo26n.pt` / `yolo26s.pt` 可直接使用，无需任何代码改动。

### 实测数据（Apple Silicon CPU，data/test.mp4 前 20 帧，imgsz 640）
| 模型 | FPS | person | 误检 |
| --- | --- | --- | --- |
| yolo26n | 21.0 | 54 | 无 |
| yolo26s | 11.3 | 60 | tie:3 |
| yolo11n | 22.0 | 57 | train:3 |
| yolo11s | 11.3 | 59 | handbag:2 |
| yolo11m | 4.1 | 60 | 无 |
| yolov8n | 20.6 | 54 | dog:3 |

补充：`yolo26n --augment` → person 61（召回最高）；`imgsz 960` 下 yolo11n 误检暴增（tie:13）而 yolo26n 只增 1 处 —— 新模型对高分辨率更鲁棒。

### 文档与测试
- README「切换其他 YOLO 模型」重写：四步流程（清单→下载→对照→切换）+ 实测对照表 + 选型建议 + 配置里固定模型的写法。
- 测试 16 → **20**：模型仓库推断、预设完整性、本地权重枚举、对照表渲染（含失败行）。

## v0.3.1 — 精度调优、模型切换与逐行文档

### 新增
- **TTA 测试时增强**：新增 `--augment` / `model.augment`，多尺度+翻转推理融合，精度提升但耗时约 2 倍。
  实测（同一段 20 帧）：yolo11n 有 1 次 `train` 误检 → 开 TTA 后误检消失，person 57→59。
- **权重下载脚本支持多模型**：`bash scripts/get_weights.sh yolo11s.pt yolo11m.pt`，可一次拉多个；
  已有文件自动跳过。实测拉取 `yolo11s.pt`（18MB）并直接用 `--model yolo11s.pt` 跑通。

### 文档
- **新增 `docs/CODE_WALKTHROUGH.md`（约 1000 行）**：把 12 个主模块**逐行**讲透——
  每一行在做什么、为什么这么写、删掉会怎样，含 `FakeBoxes` 测试技巧、就地过滤的副作用设计、
  延迟建 VideoWriter 的原因、H.264 回退原因、多进程为何要重新加载模型等。
- **README 新增三节**：
  - 「如何提高检测精度」：9 种手段按性价比排序 + **本机实测对比表**（含"降阈值反而引入大量误检"的反面案例）；
  - 「切换其他 YOLO 模型」：模型体积/速度/适用场景对照表 + 换模型的三个坑（类别 id 变了等）；
  - 「自训练并部署」：标注→目录约定→训练→验证→导出→上线的完整命令链与经验值。

## v0.3.0 — 落地能力补齐（10 项优化逐一实现）

### 新增功能
- **模型导出**：`python -m yolo_app export --format onnx|engine|openvino|torchscript|coreml`，导出后可直接 `--model yolo11n.onnx` 推理（已实测）。
- **多路多进程并发**：`run --workers N`，一源一进程独立加载模型；>1 时自动关闭 GUI 预览。
- **ROI 区域过滤 + 越线计数**：`--roi` / `--line` / `--count-classes`，越线按 track_id 判定方向（in/out），结果写入 JSON 的 `line_total`/`line_by_class`/`crossings`，并绘制在结果视频上。
- **告警通道**：`--alert-webhook` + `--alert-type wecom|feishu|generic`，抓拍与越线都会推送，失败只记日志不中断检测。
- **Web 面板**：`serve` 子命令，FastAPI + MJPEG(`/stream`) + WebSocket(`/ws`) + Prometheus(`/metrics`) + `/stats`，自带面板页。
- **Prometheus 指标**：`metrics.py` 输出标准文本格式，可 `/metrics` 暴露或 `--metrics-file` 落盘。
- **录像分段与留存**：`--segment-seconds` 按时间切片，`--keep-days` / `--max-files` 自动清理旧输出。
- **Docker（headless）**：`Dockerfile`（`opencv-python-headless`、CPU 版 torch、HEALTHCHECK）、`docker-compose.yml`、`.dockerignore`；支持 `--build-arg BASE/PIP_INDEX` 走国内镜像源。
- **CI**：`.github/workflows/ci.yml`（ruff → check → 单测 → 合成视频端到端），附 `pyproject.toml` ruff 配置。
- **自有数据 fine-tune**：`scripts/train.py` + `configs/dataset.yaml` 模板 + 标注/训练/部署流程文档。

### 修复（本轮验证中发现）
- **类别名在业务后处理中静默失效**：`--count-classes person` / events 的类别名未解析成 id（'person' 永不等于 0），导致越线计数与抓拍条件恒不命中；抽公共 `resolve_class_ids()` 统一解析。
- **ONNX/TensorRT 权重的设备名报错**：`device_name` 假设模型是 torch Module，非 torch 后端下抛 `'str' object has no attribute 'parameters'`，改为异常回退。
- **`--line/--roi` 坐标解析**：只支持空格分隔的点，改为同时接受 `x1,y1,x2,y2` 与 `x,y x,y`，并对奇数坐标给出明确报错。
- **WebSocket 静默 404**：裸 `uvicorn` 不带 websockets/wsproto 时 `/ws` 直接 404；requirements 改为 `uvicorn[standard]`，启动时缺依赖给出明确提示。
- **`--metrics-file` 参数缺失**：配置里有 `metrics.file` 但没暴露到命令行，已补。
- **越线方向语义未定义**：补充文档与单元测试（p1→p2 逆时针侧为 +，− 侧穿到 + 侧记为 in，水平线自上而下为 in）。

### 文档与工程化（本轮补齐）
- **逐模块详细注释**：每个 Python 模块头部写明"作用 / 关键符号 / 上下游依赖 / 扩展点"，
  关键函数与易踩坑处（ROI 就地改 boxes、跳帧原样写帧、告警失败不中断、H.264 兜底、延迟建 writer）都有说明。
- **新增 \`docs/ARCHITECTURE.md\`**：模块职责表、一帧数据的 11 步完整链路、配置三层流、并发模型、
  扩展点对照表（加输入/后处理/输出/指标各改哪个文件）、四种部署形态、关键设计取舍。
- **新增 \`deploy/\`**：systemd 常驻服务模板 + 容器/定时批处理/上线检查清单。
- **新增 \`Makefile\`**（install/check/test/lint/run/camera/serve/export/docker-*）、
  **\`.gitignore\`**（忽略 .venv/outputs/runs/datasets/权重）、**\`datasets/\`** 目录占位。
- **日志落盘配置**：\`log.file\` 非空时同时写日志文件（此前 logging 支持但没接线）。
- **\`--version\`**：\`python -m yolo_app --version\` 输出 yolo_app 0.3.0。

### 测试
- 用例 7 → **16**：新增 ROI 多边形判定、越线计数方向、告警 payload/真实本地 webhook 投递、录像分段、留存清理、事件抓拍、Prometheus 文本格式、新配置段存在性。
- 端到端实测：越线计数（真实视频 person in 1）、摄像头录像（H.264 60 帧）、告警投递（本地 webhook 收到 6 条企业微信格式消息）、ONNX 导出+推理、Web 面板（浏览器内 WebSocket 收到统计、MJPEG 出图）、Docker 构建。

## v0.2.0 — 从示例脚本升级为可落地应用

### 结构
- 单文件 `detect_video.py` 拆成 `yolo_app/` 包：`cli / config / detector / sources / pipeline / outputs / logging_setup`，入口统一为 `python -m yolo_app`（无需安装）。
- `tools/` → `scripts/`；新增 `configs/`（YAML 配置）与 `tests/`（冒烟测试）。
- `detect_video.py` 保留为兼容壳，等价于 `python -m yolo_app run`。

### 新增功能
- **摄像头检测**：`python -m yolo_app camera`，支持编号、分辨率、帧率、实时窗口、后台录像。
- **子命令化**：`run`（文件/目录/RTSP）、`camera`、`check`。
- **配置系统**：内置默认值 ← `configs/default.yaml` ← 命令行，三层覆盖，支持多环境多份配置。
- **目录批量**：`--source data/` 自动遍历目录内视频，并输出 `outputs/summary_all.json`。
- **告警抓拍**：命中类别且置信度达标自动存图到 `outputs/events/`，带冷却时间，事件写入 JSON。
- **断流重连**：摄像头 / RTSP 读失败自动重连（次数、间隔可配）。
- **类别名过滤**：`--classes person,bus` 与 `--classes 0,2` 均可。
- **统一日志**：logging 输出进度（帧号/FPS/统计），不再用 print 刷屏。
- **兼容壳**：旧命令 `python detect_video.py --source ... ` 仍可用。
- **VS Code**：`.vscode/launch.json` 扩到 4 个 F5 配置（视频/摄像头/后台录制/自检），均带 `YOLO_OFFLINE` 与 `cwd`。

### 修复
- **结果视频在 VS Code 里打不开**：原 `mp4v`(mpeg4) 编码 Chromium 内核不支持，改为优先 H.264(`avc1`)、失败回退 `mp4v`。
- **权重下载卡死**：裸写 `--model yolo11n.pt` 现在自动命中项目内 `weights/`，不再触发 GitHub 下载；配 `scripts/get_weights.sh` 走镜像。
- **摄像头打不开无提示**：现在直接打印 macOS/Windows 权限排查步骤与设备枚举命令。
- **`half=False` 触发 ultralytics 弃用告警**：仅在需要时传该参数。
- 写视频的 VideoWriter 延迟到首帧确定尺寸后创建，兼容 RTSP / 摄像头（宽高未知的情况）。

### 测试
- 新增 `tests/test_smoke.py`：配置合并、源判定与目录展开、视频写入回读、JSON 落盘、**端到端跑真模型 3 帧**，共 7 个用例全绿。

## v0.1.0 — 初始版本
- 单文件 `detect_video.py`：逐帧推理、ByteTrack 跟踪、类别过滤、跳帧、结果视频与统计 JSON。
- `scripts/get_weights.sh`、`scripts/make_test_video.py`、README 与实测记录。
