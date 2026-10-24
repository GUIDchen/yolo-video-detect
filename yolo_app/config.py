"""配置：内置默认值 ← YAML 文件 ← 命令行覆盖，三层合并。

为什么要有 DEFAULTS：保证任何字段都有值，业务代码里可以放心 cfg['x']['y']，
不用到处写 .get() 兜底；YAML 与命令行只是"覆盖其中的一部分"。

用法：
    cfg = load_config('configs/production.yaml', {'model': {'conf': 0.5}})

分层优先级（后者覆盖前者）：DEFAULTS → YAML → overrides
所有模块都只接收这个 dict，不读全局变量，方便测试与多环境部署。
"""
import copy
import math
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
        'anchor': 'center',    # 锚点：center=框中心（现有口径）；bottom=框底边中点（Frigate/supervision 通行做法）
                               # 改成 bottom 会改变既有越线/ROI 口径，需重新标定
        'track_ttl': 300,      # 越线计数里 track 的存活帧数，超时即淘汰（防长视频内存增长）；0 = 不淘汰
        'min_cross_frames': 2, # 越线计数冷却帧数：同一 track 计数后 N 帧内再次相交不重复计数；1 = 不冷却
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
    'alert': {'webhook': '', 'type': 'generic', 'timeout': 5, 'dedupe_s': 3.0,
              'retry': {'max_attempts': 3, 'base_delay': 2.0}},   # 失败按 base*2^(n-1) 退避，超限置 dead
    'serve': {'host': '127.0.0.1', 'port': 8000, 'source': '0', 'quality': 80, 'token': ''},
    'output': {
        'dir': 'outputs',
        'video': True,
        'name': '',            # 空 = 按源自动命名
        'json': True,
        'segment_seconds': 0,  # >0 按时间切分录像；0 = 不分段
        'keep_days': 0,        # >0 清理超过 N 天的输出
        'max_files': 0,        # >0 只保留最近 N 个输出文件
        'events': {
            'enable': True, 'classes': [0], 'min_conf': 0.6, 'cooldown_s': 2.0,
            'db': 'events.db',      # 相对 output.dir；空字符串 = 不落库，退回旧行为
            'async_write': True,    # 抓拍图后台写盘；队列满丢最旧，不拖慢主循环
            'async_queue': 64,      # 异步写盘队列长度
        },
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


class ConfigError(Exception):
    """配置校验失败。消息必须带"字段 + 当前值 + 期望"，让用户照着就能改对。"""


# 可选键：不写进 DEFAULTS（避免默认键集膨胀），但合法出现时不算未知配置项。
# output.name_mode 由 CLI 的 --name-mode 或配置文件按需开启，见 pipeline.output_paths。
_OPTIONAL_KEYS = {'output.name_mode'}


def _bad(field, value, expect):
    """统一的报错格式：字段、当前值、期望三件套。"""
    raise ConfigError(f'配置项 {field} 非法：当前值 {value!r}，期望 {expect}')


def _num(field, value, expect):
    """转成有限浮点数；非数字、布尔、NaN/Inf 都按非法处理。"""
    if isinstance(value, bool):
        _bad(field, value, expect)
    try:
        v = float(value)
    except (TypeError, ValueError):
        _bad(field, value, expect)
    if not math.isfinite(v):
        _bad(field, value, expect)
    return v


def _int(field, value, expect):
    """转成整数；1.5 这类"看着像数字但不能当计数用"的值直接报错。"""
    v = _num(field, value, expect)
    if v != int(v):
        _bad(field, value, expect)
    return int(v)


def _section(cfg, name, field=None):
    """取一个配置段；写成标量（如 model: 640）时报可操作的错，而不是 AttributeError。"""
    value = cfg.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        _bad(field or name, value, '映射（key: value）形式的配置段')
    return value


def _points(field, value, expect, min_n=0, exact=None):
    """校验坐标点列表：空列表允许；否则必须是 [x, y] 对，数量满足 min_n / exact。"""
    if not value:
        return
    if not isinstance(value, (list, tuple)):
        _bad(field, value, expect)
    if exact is not None and len(value) != exact:
        _bad(field, value, f'空列表或恰好 {exact} 个 [x, y] 点（当前 {len(value)} 个）')
    if exact is None and len(value) < min_n:
        _bad(field, value, f'空列表或至少 {min_n} 个 [x, y] 点（当前 {len(value)} 个）')
    for pt in value:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            _bad(field, value, expect)
        for c in pt:
            _num(field, c, expect)


def _unknown_keys(cfg, defaults=None, prefix=''):
    """对比 DEFAULTS 找出未知键，返回 'a.b.c' 形式的路径列表（不抛错）。"""
    defaults = DEFAULTS if defaults is None else defaults
    found = []
    for k, v in (cfg or {}).items():
        path = f'{prefix}{k}'
        if path in _OPTIONAL_KEYS:
            continue
        if k not in defaults:
            found.append(path)
        elif isinstance(v, dict) and isinstance(defaults[k], dict):
            found.extend(_unknown_keys(v, defaults[k], path + '.'))
    return found


def _validate(cfg, strict=False):
    """校验合并后的配置：非法值抛 ConfigError，返回 warning 列表。

    分两类：
      · 非法值（越界、类型不对、点数量不够）——配置根本跑不起来，直接抛；
      · 可疑但能跑（imgsz 不是 32 的倍数、未知配置项）——返回 warning，由调用方打日志。
    strict=True 时未知配置项也升级为 ConfigError（CI / 严格部署用）。
    """
    warnings = []
    model = _section(cfg, 'model')
    run = _section(cfg, 'run')
    camera = _section(cfg, 'camera')
    analytics = _section(cfg, 'analytics')
    output = _section(cfg, 'output')
    serve = _section(cfg, 'serve')
    alert = _section(cfg, 'alert')
    retry = _section(alert, 'retry', 'alert.retry')

    # ---- model ----
    expect_imgsz = '正整数（建议 32 的整数倍，如 640）'
    imgsz = _int('model.imgsz', model.get('imgsz'), expect_imgsz)
    if imgsz <= 0:
        _bad('model.imgsz', model.get('imgsz'), expect_imgsz)
    if imgsz % 32:
        lo, hi = imgsz // 32 * 32, (imgsz // 32 + 1) * 32
        warnings.append(f'配置项 model.imgsz={imgsz} 不是 32 的整数倍，'
                        f'建议改为 {lo} 或 {hi}，避免模型内部自动对齐带来的尺寸偏差')
    for key in ('conf', 'iou'):
        expect = '0 ≤ 值 ≤ 1 的小数'
        v = _num(f'model.{key}', model.get(key), expect)
        if not 0 <= v <= 1:
            _bad(f'model.{key}', model.get(key), expect)
    expect_report = '0（沿用 model.conf）或 0 ≤ 值 ≤ 1 的小数'
    report_conf = _num('model.report_conf', model.get('report_conf'), expect_report)
    if not 0 <= report_conf <= 1:
        _bad('model.report_conf', model.get('report_conf'), expect_report)

    # ---- run ----
    expect_stride = '≥ 1 的整数（1 = 每帧都推理）'
    stride = _int('run.stride', run.get('stride'), expect_stride)
    if stride < 1:
        _bad('run.stride', run.get('stride'), expect_stride)
    expect_frames = '≥ 0 的整数（0 = 处理全部帧）'
    max_frames = _int('run.max_frames', run.get('max_frames'), expect_frames)
    if max_frames < 0:
        _bad('run.max_frames', run.get('max_frames'), expect_frames)
    expect_workers = '≥ 1 的整数'
    workers = _int('run.workers', run.get('workers'), expect_workers)
    if workers < 1:
        _bad('run.workers', run.get('workers'), expect_workers)
    expect_timeout = '> 0 的秒数'
    read_timeout = _num('run.read_timeout', run.get('read_timeout'), expect_timeout)
    if read_timeout <= 0:
        _bad('run.read_timeout', run.get('read_timeout'), expect_timeout)
    reader = run.get('reader')
    if reader not in ('auto', 'opencv', 'ffmpeg'):
        _bad('run.reader', reader, 'auto / opencv / ffmpeg 之一')

    # ---- camera ----
    expect_fps = '> 0 的帧率'
    fps = _num('camera.fps', camera.get('fps'), expect_fps)
    if fps <= 0:
        _bad('camera.fps', camera.get('fps'), expect_fps)
    expect_buffer = '≥ 1 的整数'
    buffer_size = _int('camera.buffer_size', camera.get('buffer_size'), expect_buffer)
    if buffer_size < 1:
        _bad('camera.buffer_size', camera.get('buffer_size'), expect_buffer)

    # ---- analytics ----
    _points('analytics.roi', analytics.get('roi'), '[[x, y], ...] 形式的坐标点', min_n=3)
    _points('analytics.line', analytics.get('line'), '[[x, y], [x, y]] 两个坐标点', exact=2)
    expect_cross = '≥ 1 的整数（1 = 不防抖）'
    min_cross = _int('analytics.min_cross_frames', analytics.get('min_cross_frames'), expect_cross)
    if min_cross < 1:
        _bad('analytics.min_cross_frames', analytics.get('min_cross_frames'), expect_cross)
    expect_ttl = '≥ 0 的整数（0 = 不淘汰）'
    ttl = _int('analytics.track_ttl', analytics.get('track_ttl'), expect_ttl)
    if ttl < 0:
        _bad('analytics.track_ttl', analytics.get('track_ttl'), expect_ttl)
    expect_margin = '0 ≤ 值 < 2 的小数'
    margin = _num('analytics.roi_margin', analytics.get('roi_margin'), expect_margin)
    if not 0 <= margin < 2:
        _bad('analytics.roi_margin', analytics.get('roi_margin'), expect_margin)

    # ---- output ----
    for key, expect in (('segment_seconds', '≥ 0 的整数（0 = 不分段）'),
                        ('keep_days', '≥ 0 的整数（0 = 不清理）'),
                        ('max_files', '≥ 0 的整数（0 = 不限）')):
        v = _int(f'output.{key}', output.get(key), expect)
        if v < 0:
            _bad(f'output.{key}', output.get(key), expect)

    # ---- serve ----
    expect_port = '1–65535 的整数'
    port = _int('serve.port', serve.get('port'), expect_port)
    if not 1 <= port <= 65535:
        _bad('serve.port', serve.get('port'), expect_port)

    # ---- alert ----
    expect_dedupe = '≥ 0 的秒数（0 = 不去重）'
    dedupe = _num('alert.dedupe_s', alert.get('dedupe_s'), expect_dedupe)
    if dedupe < 0:
        _bad('alert.dedupe_s', alert.get('dedupe_s'), expect_dedupe)
    expect_attempts = '≥ 1 的整数'
    attempts = _int('alert.retry.max_attempts', retry.get('max_attempts'), expect_attempts)
    if attempts < 1:
        _bad('alert.retry.max_attempts', retry.get('max_attempts'), expect_attempts)
    expect_delay = '> 0 的秒数'
    base_delay = _num('alert.retry.base_delay', retry.get('base_delay'), expect_delay)
    if base_delay <= 0:
        _bad('alert.retry.base_delay', retry.get('base_delay'), expect_delay)

    # ---- 未知键：只提醒不报错（strict 时升级为错误）----
    unknown = _unknown_keys(cfg)
    if unknown:
        if strict:
            raise ConfigError('未知配置项 ' + '、'.join(unknown) +
                              '（strict 模式不允许）；可用的键见 config.DEFAULTS')
        warnings.extend(f'未知配置项 {k}，已忽略' for k in unknown)
    return warnings


def validate_config(cfg):
    """校验已合并的配置，返回 warning 列表；非法值抛 ConfigError。

    只读：不修改 cfg，也不新增默认键。
    """
    return _validate(cfg, strict=False)


def load_config(path=None, overrides=None, strict=False):
    """三层合并 + 校验。strict=True 时未知配置项也抛 ConfigError。"""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'配置文件不存在: {path}')
        deep_merge(cfg, yaml.safe_load(p.read_text(encoding='utf-8')) or {})
    deep_merge(cfg, {k: v for k, v in (overrides or {}).items() if v})
    _validate(cfg, strict=strict)
    return cfg
