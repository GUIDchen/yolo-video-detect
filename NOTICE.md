# 第三方组件与版权声明

本项目 Copyright (c) 2026 GUIDchen，以 **GNU Affero General Public License v3.0（AGPL-3.0）** 发布，全文见 [LICENSE](LICENSE)。

## 为什么是 AGPL-3.0

本项目核心依赖 [Ultralytics](https://github.com/ultralytics/ultralytics)，其许可证为 AGPL-3.0。AGPL 具有传染性：**分发**基于它的组合作品时，整体必须以 AGPL 提供完整源代码。因此本项目选择 AGPL-3.0，而不是 MIT。

如果你需要闭源商用，正确做法是向 Ultralytics 购买商业许可，而不是修改本项目的 LICENSE。

## 运行时依赖

许可证信息取自各已安装包的 `dist-info/METADATA`（核实日期 2026-09-28）：

| 组件 | 许可证 | 用途 |
| --- | --- | --- |
| ultralytics | **AGPL-3.0** | 检测/跟踪、导出 |
| ultralytics-thop | **AGPL-3.0** | 参数量/FLOPs 统计（ultralytics 依赖） |
| opencv-python | Apache-2.0 | 视频读写、绘制 |
| PyTorch | Apache-2.0（含若干第三方条款） | 推理后端 |
| torchvision | BSD | 图像变换 |
| numpy | BSD-3-Clause 等组合 | 数值计算 |
| fastapi | MIT | Web 面板 |
| uvicorn | BSD-3-Clause | ASGI 服务器 |
| lap | BSD-2-Clause | ByteTrack 关联 |
| PyYAML | MIT | 配置文件 |
| onnx | Apache-2.0 | 导出可选依赖 |
| onnxruntime | MIT | ONNX 推理可选依赖 |

各组件版权归其各自作者所有。

## 关于模型权重

- 本仓库**不包含**任何权重文件（`weights/*.pt`、`*.onnx`、`*.engine` 均在 `.gitignore` 中）。
- 官方预训练权重（如 `yolo11n.pt`）由 Ultralytics 发布，受 AGPL-3.0 约束。
- 基于官方权重微调得到的权重属于衍生作品，**分发时同样受 AGPL-3.0 约束**。
- 获取方式：`bash scripts/get_weights.sh` 或 `python -m yolo_app models --download yolo11n.pt`。

## 关于媒体素材

- 本仓库**不包含**视频与图片（`data/*.mp4`、`data/*.jpg` 均已忽略）。
- 测试视频由 `scripts/make_test_video.py` 现场生成，内容是合成图形，不含真人素材。
- CI 会临时下载 Ultralytics 的 `bus.jpg` 用于冒烟测试，该素材版权归 Ultralytics，仅作测试用途、不入库。

## 商标

YOLO、Ultralytics 为其各自所有者的商标。本项目是独立实现，与 Ultralytics 无隶属关系，也未获得其背书。
