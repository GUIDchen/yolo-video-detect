"""YOLO 视频/摄像头目标检测应用包。

职责：把"取帧 → 检测 → 后处理 → 落地"做成可复用的管道，并由多种入口（命令行 / Web / 容器）驱动。

模块地图（谁能调用谁）：
    cli.py       命令行入口，负责参数解析与子命令分发，是唯一的"用户入口"
    config.py    三层配置合并（默认值 ← YAML ← 命令行），输出一个纯 dict 给所有模块
    sources.py   取帧：本地文件 / 目录 / 摄像头 / RTSP，含断流重连
    detector.py  模型：权重定位、类别名解析、单帧推理与跟踪
    analytics.py 后处理：ROI 区域过滤、越线计数（纯几何，不依赖模型）
    outputs.py   落地：结果视频（H.264/分段）、事件抓拍、JSON、留存清理
    alerts.py    告警：把事件推到企业微信 / 飞书 / 任意 webhook
    metrics.py   Prometheus 文本指标
    pipeline.py  ★ 主循环：把上面这些串起来，是唯一的"业务编排层"
    webapp.py    Web 面板：复用 pipeline 的 on_frame/on_stats 回调

扩展原则：加输入改 sources，加后处理改 analytics + pipeline 里一行，加输出改 outputs，
         加接口改 webapp，别在 pipeline 里堆业务分支。
"""
__version__ = '0.4.1'
