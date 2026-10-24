"""配置：内置默认值 ← YAML 文件 ← 命令行覆盖，三层合并。

为什么要有 DEFAULTS：保证任何字段都有值，业务代码里可以放心 cfg['x']['y']，
不用到处写 .get() 兜底；YAML 与命令行只是"覆盖其中的一部分"。

用法：
    cfg = load_config('configs/production.yaml', {'model': {'conf': 0.5}})

分层优先级（后者覆盖前者）：DEFAULTS → YAML → overrides
所有模块都只接收这个 dict，不读全局变量，方便测试与多环境部署。
"""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    'model': {
        'weights': 'yolo11n.pt',
        'imgsz': 640,
        'conf': 0.25,
        'iou': 0.45,
        'device': None,        # None=自动, 'cpu', '0', '0,1'
        'half': False,         # FP16，仅 CUDA
        'augment': False,      # TTA 测试时增强：更准但约 2 倍耗时
        'classes': [],         # 类别 id 或名称；空 = 全部
        'report_conf': 0,      # 双阈值：低于该值的框只送跟踪、不进统计/绘制/抓拍；0 = 沿用 conf
    },
    'track': {'enable': False, 'tracker': 'bytetrack.yaml'},
    'run': {'stride': 1, 'max_frames': 0, 'reconnect': 3, 'retry_delay': 1.0,
            'show': False, 'workers': 1,
            'reader': 'auto',      # auto | opencv | ffmpeg；auto = 直播源用 ffmpeg，其余用 opencv
            'read_timeout': 10},   # 直播源多久没收到帧判定僵死（秒）
    'camera': {'index': 0, 'width': 1280, 'height': 720, 'fps': 30, 'show': True, 'record': False,
               'buffer_size': 1},   # 采集缓冲；1 = 只留最新帧，降低直播延迟
    'analytics': {
        'roi': [],             # [[x,y],...] 多边形，只统计区域内目标；空 = 不过滤
        'line': [],            # [[x1,y1],[x2,y2]] 越线，需配合 --track
        'classes': [],         # 参与越线计数的类别；空 = 全部
        'track_ttl': 300,      # 越线计数里 track 的存活帧数，超时即淘汰（防长视频内存增长）；0 = 不淘汰
        'min_cross_frames': 2, # 越线防抖：连续 N 帧落在新的一侧才计一次穿越；1 = 不防抖
        'stationary_frames': 0,   # 静止降频：连续 N 帧几乎不动就开始抽帧推理；0 = 关闭（需开跟踪）
        'stationary_interval': 10,  # 静止后每多少帧推理一次
        'stationary_min_move': 2,   # 位移小于该像素数视为"没动"
        'roi_crop': False,        # 只对 ROI 区域裁剪后推理（目标都在 ROI 内时又快又准）
        'roi_margin': 0.1,        # 裁剪外扩比例，避免目标压在 ROI 边界被切
        'roi_imgsz': 0,           # 裁剪推理的输入尺寸；0 = 沿用 model.imgsz
    },
    'motion': {
        'enable': False,       # 帧差门控：无运动不推理（静态场景省算力；光照突变会退化成逐帧）
        'threshold': 25,       # 帧差阈值
        'min_area_ratio': 0.002,  # 运动像素占比达到多少才算"有运动"
        'blur': 5,             # 高斯模糊核，压噪点；<=1 关闭
        'mask_roi': True,      # 有 ROI 时只统计 ROI 内的运动
        'hold_frames': 0,      # 无运动时沿用上一帧框显示多少帧（0 = 写原帧）
    },
    'alert': {'webhook': '', 'type': 'generic', 'timeout': 5, 'dedupe_s': 3.0},
    'serve': {'host': '127.0.0.1', 'port': 8000, 'source': '0', 'quality': 80, 'token': ''},
    'output': {
        'dir': 'outputs',
        'video': True,
        'name': '',            # 空 = 按源自动命名
        'json': True,
        'segment_seconds': 0,  # >0 按时间切分录像；0 = 不分段
        'keep_days': 0,        # >0 清理超过 N 天的输出
        'max_files': 0,        # >0 只保留最近 N 个输出文件
        'events': {'enable': True, 'classes': [0], 'min_conf': 0.6, 'cooldown_s': 2.0},
    },
    'metrics': {'file': ''},   # 非空则把 Prometheus 指标同时写到该文件
    'log': {'level': 'INFO', 'interval': 30, 'file': ''},   # file 非空则同时写日志文件
}


def deep_merge(base, patch):
    """就地深度合并 patch 到 base，返回 base。"""
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'配置文件不存在: {path}')
        deep_merge(cfg, yaml.safe_load(p.read_text(encoding='utf-8')) or {})
    deep_merge(cfg, {k: v for k, v in (overrides or {}).items() if v})
    return cfg
