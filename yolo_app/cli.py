"""命令行入口：`python -m yolo_app {run|camera|check|export|serve}`。

结构：
    add_common()     三个子命令共用的模型/性能参数（--model/--conf/--stride/...）
    add_analytics()  业务参数（--roi/--line/--alert-webhook/分段/留存）
    build_overrides() 把命令行参数翻译成"嵌套配置覆盖字典"，交给 config 做三层合并
    build_parser()   子命令定义，是了解本工具能力的最快入口
    main()           子命令分发：check 自检 / export 导出 / serve 起面板 / run|camera 跑管道

约定：
    · 命令行只覆盖"显式传入"的项（None 表示没传），因此配置文件的默认值不会被误伤；
    · 参数既支持类别 id 也支持类别名（person,bus），解析在 parse_classes()；
    · 目录批量 + --workers>1 时走 ProcessPoolExecutor（一源一进程），此时强制关预览窗口。
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import ConfigError, load_config, validate_config
from .detector import Detector
from .logging_setup import get_logger
from .outputs import cleanup_old, write_json
from .pipeline import process, run_job
from .sources import list_videos

DEFAULT_CONFIG = 'configs/default.yaml'


def parse_classes(s):
    """'person,bus' / '0,2' → ['person','bus'] / [0,2]"""
    if not s:
        return None
    out = []
    for c in str(s).split(','):
        c = c.strip()
        if c:
            out.append(int(c) if c.isdigit() else c)
    return out or None


def parse_points(s, n):
    """坐标串 → [[x,y],...]。支持 'x1,y1 x2,y2' 与 'x1,y1,x2,y2' 两种写法；n 为期望点数（0=不限）。"""
    if not s:
        return None
    nums = [float(v) for v in re.split(r'[,;\s]+', str(s).strip()) if v]
    if len(nums) % 2:
        raise ValueError(f'坐标个数必须是偶数（成对出现）: {s}')
    pts = [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
    if n and len(pts) != n:
        raise ValueError(f'需要 {n} 个点，实际 {len(pts)} 个: {s}')
    return pts


def add_common(p):
    p.add_argument('--config', default=None, help=f'YAML 配置，默认 {DEFAULT_CONFIG}（存在则加载）')
    p.add_argument('--model', help='权重名或路径，如 yolo11n.pt / best.pt / best.onnx / best.engine')
    p.add_argument('--imgsz', type=int, help='推理分辨率，320 更快 / 1280 更准')
    p.add_argument('--conf', type=float, help='置信度阈值')
    p.add_argument('--iou', type=float, help='NMS IoU 阈值')
    p.add_argument('--report-conf', dest='report_conf', type=float,
                   help='双阈值：低于该值的框只送跟踪，不进统计/绘制/抓拍（默认沿用 --conf）')
    p.add_argument('--device', help='cpu / 0 / 0,1，默认自动')
    p.add_argument('--classes', help='类别 id 或名称，逗号分隔，如 person,bus 或 0,2')
    p.add_argument('--track', dest='track', action='store_true', default=None, help='启用 ByteTrack 跟踪')
    p.add_argument('--stride', type=int, help='跳帧间隔，2 = 每两帧推理一次')
    p.add_argument('--max-frames', dest='max_frames', type=int, help='最多处理帧数，0 = 全部')
    p.add_argument('--out-dir', dest='out_dir', help='输出目录，默认 outputs')
    p.add_argument('--no-video', dest='no_video', action='store_true', help='不写结果视频')
    p.add_argument('--half', action='store_true', default=None, help='FP16 半精度（仅 CUDA）')
    p.add_argument('--augment', action='store_true', default=None, help='TTA 测试时增强：更准但约 2 倍耗时')
    p.add_argument('--metrics-file', dest='metrics_file', help='把 Prometheus 指标同时写到该文件')
    p.add_argument('--log-level', dest='log_level', help='DEBUG / INFO / WARNING')


def add_analytics(p):
    p.add_argument('--roi', help='ROI 多边形：x1,y1 x2,y2 x3,y3 ...（只统计区域内目标）')
    p.add_argument('--line', help='越线计数线：x1,y1,x2,y2（需配合 --track）')
    p.add_argument('--count-classes', dest='count_classes', help='参与越线计数的类别，逗号分隔')
    p.add_argument('--alert-webhook', dest='alert_webhook', help='告警 webhook：完整 URL 或企业微信/飞书的 key')
    p.add_argument('--alert-type', dest='alert_type', choices=['generic', 'wecom', 'feishu'], help='告警类型')
    p.add_argument('--segment-seconds', dest='segment_seconds', type=int, help='录像分段秒数，0=不分段')
    p.add_argument('--keep-days', dest='keep_days', type=int, help='输出留存天数，0=不清理')
    p.add_argument('--max-files', dest='max_files', type=int, help='输出最多保留文件数，0=不限')


def build_overrides(a):
    """命令行参数 → 嵌套覆盖字典；只覆盖用户显式传了的项。"""
    ov = {'model': {}, 'run': {}, 'output': {}, 'log': {}, 'camera': {}, 'analytics': {}, 'alert': {}}
    m, run, out = ov['model'], ov['run'], ov['output']
    pairs = [
        (getattr(a, 'model', None), m, 'weights'),
        (getattr(a, 'imgsz', None), m, 'imgsz'),
        (getattr(a, 'conf', None), m, 'conf'),
        (getattr(a, 'iou', None), m, 'iou'),
        (getattr(a, 'report_conf', None), m, 'report_conf'),
        (getattr(a, 'device', None), m, 'device'),
        (getattr(a, 'stride', None), run, 'stride'),
        (getattr(a, 'max_frames', None), run, 'max_frames'),
        (getattr(a, 'workers', None), run, 'workers'),
        (getattr(a, 'out_dir', None), out, 'dir'),
        (getattr(a, 'segment_seconds', None), out, 'segment_seconds'),
        (getattr(a, 'keep_days', None), out, 'keep_days'),
        (getattr(a, 'max_files', None), out, 'max_files'),
    ]
    for value, target, key in pairs:
        if value is not None:
            target[key] = value
    if getattr(a, 'half', None):
        m['half'] = True
    if getattr(a, 'augment', None):
        m['augment'] = True
    cls = parse_classes(getattr(a, 'classes', None))
    if cls:
        m['classes'] = cls
    if getattr(a, 'track', None):
        ov['track'] = {'enable': True}
    if getattr(a, 'no_video', False):
        out['video'] = False
    if getattr(a, 'log_level', None):
        ov['log']['level'] = a.log_level
    if getattr(a, 'metrics_file', None):
        ov['metrics'] = {'file': a.metrics_file}
    for flag, key in (('roi', 'roi'),):
        pts = parse_points(getattr(a, flag, None), 0)
        if pts:
            ov['analytics'][key] = pts
    pts = parse_points(getattr(a, 'line', None), 2)
    if pts:
        ov['analytics']['line'] = pts
    cc = parse_classes(getattr(a, 'count_classes', None))
    if cc:
        ov['analytics']['classes'] = cc
    if getattr(a, 'alert_webhook', None):
        ov['alert']['webhook'] = a.alert_webhook
    if getattr(a, 'alert_type', None):
        ov['alert']['type'] = a.alert_type
    return ov


def build_parser():
    from . import __version__
    p = argparse.ArgumentParser(prog='python -m yolo_app', description='YOLO 视频/摄像头目标检测应用')
    p.add_argument('-V', '--version', action='version', version=f'yolo_app {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    pr = sub.add_parser('run', help='检测视频文件或目录（目录 = 批量，可多进程）')
    pr.add_argument('--source', required=True, help='视频文件 / 目录 / RTSP 地址')
    pr.add_argument('--show', dest='show', action='store_true', default=None, help='实时预览窗口，按 q 退出')
    pr.add_argument('--workers', type=int, help='并行进程数（目录批量时提速，>1 自动关闭预览窗口）')
    pr.add_argument('--name-mode', dest='name_mode', choices=['stem', 'unique'],
                    help='输出命名：stem = 源文件名（默认，同名会覆盖）；'
                         'unique = 源文件名 + 源路径哈希（目录批量默认启用，防覆盖）')
    add_analytics(pr)
    add_common(pr)

    pc = sub.add_parser('camera', help='调用摄像头实时检测')
    pc.add_argument('--index', type=int, help='摄像头编号，默认 0')
    pc.add_argument('--width', type=int, help='采集宽度')
    pc.add_argument('--height', type=int, help='采集高度')
    pc.add_argument('--fps', type=int, help='采集帧率')
    pc.add_argument('--record', dest='record', action='store_true', default=None, help='保存检测录像')
    pc.add_argument('--no-show', dest='no_show', action='store_true', help='不弹实时窗口（配合 --record 后台录制）')
    add_analytics(pc)
    add_common(pc)

    pe = sub.add_parser('export', help='导出 ONNX / TensorRT / OpenVINO 等部署格式')
    pe.add_argument('--format', default='onnx', choices=['onnx', 'engine', 'openvino', 'torchscript', 'coreml'])
    pe.add_argument('--dynamic', action='store_true', help='动态输入尺寸')
    pe.add_argument('--simplify', action='store_true', help='ONNX 图简化')
    add_common(pe)

    ps = sub.add_parser('serve', help='启动 Web 面板（MJPEG + WebSocket + Prometheus）')
    ps.add_argument('--source', default=None, help='视频文件 / 目录 / RTSP / 摄像头编号，默认取配置 serve.source')
    ps.add_argument('--host', default=None)
    ps.add_argument('--port', type=int, default=None)
    ps.add_argument('--quality', type=int, default=None, help='JPEG 质量 1-100')
    ps.add_argument('--token', default=None, help='访问 token；非空时 /stats /metrics /stream /ws 需带 ?token=')
    ps.add_argument('--show', dest='show', action='store_true', default=None, help=argparse.SUPPRESS)
    add_analytics(ps)
    add_common(ps)

    pm = sub.add_parser('models', help='列出/下载模型，或对不同模型做速度与检出对照')
    pm.add_argument('--download', nargs='*', default=None, help='下载这些模型到 weights/（如 yolo26s.pt）')
    pm.add_argument('--benchmark', nargs='*', default=None,
                    help='对这些模型跑对照测试；不给名字则测 weights/ 下全部 .pt')
    pm.add_argument('--source', default='data/test.mp4', help='benchmark 使用的视频源')
    pm.add_argument('--frames', type=int, default=20, help='benchmark 每个模型处理多少帧')
    add_common(pm)

    pk = sub.add_parser('check', help='环境与模型自检')
    add_common(pk)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg_path = a.config or (DEFAULT_CONFIG if Path(DEFAULT_CONFIG).exists() else None)

    ov = build_overrides(a)
    if a.cmd == 'camera':
        cam = {}
        for key, val in (('index', a.index), ('width', a.width), ('height', a.height), ('fps', a.fps)):
            if val is not None:
                cam[key] = val
        if a.record:
            cam['record'] = True
        ov['camera'] = cam
    if a.cmd == 'serve':
        sv = {}
        for key, val in (('source', a.source), ('host', a.host), ('port', a.port), ('quality', a.quality),
                         ('token', a.token)):
            if val is not None:
                sv[key] = val
        ov['serve'] = sv

    try:
        cfg = load_config(cfg_path, ov)
    except ConfigError as e:                    # 配置错误属于"用户可修"，只报错不甩栈
        get_logger('ERROR').error(f'配置校验失败：{e}')
        return 1
    logger = get_logger(cfg['log']['level'], cfg['log'].get('file'))
    for w in validate_config(cfg):              # 可疑但不致命：imgsz 对齐、未知配置项等
        logger.warning(w)

    if a.cmd == 'models':                       # 模型管理不需要加载模型，先拦下来
        return _models(a, cfg, logger)

    if a.cmd == 'camera':
        cfg['run']['show'] = not a.no_show
        cfg['output']['video'] = (bool(a.record) or cfg['camera']['record']) and not a.no_video
        sources = [cfg['camera']['index']]
    elif a.cmd == 'run':
        if a.show is not None:
            cfg['run']['show'] = a.show
        sources = list_videos(a.source)
        # 输出命名：显式 output.name 永远不改名；--name-mode 显式覆盖优先于批量自动开启
        name_mode = getattr(a, 'name_mode', None)
        if cfg['output'].get('name'):
            if name_mode:
                logger.warning(f'已显式设置 output.name={cfg["output"]["name"]}，忽略 --name-mode')
        elif name_mode:
            cfg['output']['name_mode'] = name_mode
            logger.info(f'输出命名方式：{name_mode}（--name-mode 显式指定）')
        elif len(sources) > 1:
            cfg['output']['name_mode'] = 'unique'
            logger.info(f'批量处理 {len(sources)} 个源，自动启用唯一输出命名（name_mode=unique），避免同名覆盖')
    elif a.cmd == 'serve':
        sources = [cfg['serve']['source']]
    else:
        sources = []

    try:
        detector = Detector(cfg)
    except Exception as e:
        logger.error(f'模型加载失败: {e}')
        return 1

    if a.cmd == 'check':
        import numpy as np
        r = detector.infer(np.zeros((detector.imgsz, detector.imgsz, 3), dtype=np.uint8))
        assert r.boxes is not None, '推理未返回 boxes，请检查 ultralytics / torch'
        logger.info(f'模型 {detector.weights} | 类别数 {len(detector.names)} | 设备 {detector.device_name} | '
                    f'跟踪 {detector.track} | 类别过滤 {detector.classes or "全部"}')
        logger.info('自检通过 ✅')
        return 0

    if a.cmd == 'export':
        return _export(a, cfg, detector, logger)

    if a.cmd == 'serve':
        from .webapp import serve
        serve(cfg, detector, logger, source=cfg['serve']['source'])
        return 0

    if not sources:
        logger.error(f'没有可处理的视频源: {getattr(a, "source", "")}')
        return 1

    removed = cleanup_old(cfg['output']['dir'], cfg['output']['keep_days'], cfg['output']['max_files'])
    if removed:
        logger.info(f'留存清理: 删除 {removed} 个旧文件')

    workers = max(int(cfg['run']['workers'] or 1), 1)
    summaries = []
    if workers > 1 and len(sources) > 1:
        if cfg['run']['show']:
            logger.warning('多进程模式不支持实时预览窗口，已关闭 --show')
        logger.info(f'多进程模式：{workers} 个进程并行处理 {len(sources)} 个源')
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s in ex.map(run_job, [(src, cfg, False) for src in sources]):
                summaries.append(s)
    else:
        for s in sources:
            try:
                summaries.append(process(s, cfg, detector, logger))
            except Exception as e:
                logger.error(f'{s} 处理失败: {e}')
                if len(sources) == 1:
                    return 1

    ok = [s for s in summaries if 'error' not in s]
    if len(summaries) > 1:
        p = write_json(Path(cfg['output']['dir']) / 'summary_all.json', summaries)
        logger.info(f'批量汇总: {p} | 成功 {len(ok)}/{len(summaries)}')
    return 0 if ok else 1


def _models(a, cfg, logger):
    """models 子命令：列表 / 下载 / 基准对照。"""
    from . import models as M

    if a.download is not None:
        names = a.download or list(M.PRESETS)
        ok, fail = M.download(names, logger.info)
        logger.info(f'下载结果：成功 {len(ok)}，失败 {len(fail)}')
        return 0 if not fail else 1

    if a.benchmark is not None:
        names = a.benchmark or [n for n in M.local_weights() if n.endswith('.pt')]
        if not names:
            logger.error('weights/ 下没有 .pt 权重；先跑：python -m yolo_app models --download yolo26n.pt')
            return 1
        logger.info(f'对照测试：{len(names)} 个模型 × {a.frames} 帧 | 源 {a.source}')
        print(M.format_table(M.benchmark(names, a.source, a.frames, cfg, logger)))
        return 0

    local = set(M.local_weights())
    print(f'{"模型":<16}{"体积":>8}{"相对速度":>10}  本地  说明')
    print('-' * 78)
    for name, (size, speed, note) in M.PRESETS.items():
        print(f'{name:<16}{size:>8}{speed:>9}×  {"✔" if name in local else " "}    {note}')
    for name in sorted(local - set(M.PRESETS)):            # 本地有但不在预设里的（自训练权重等）
        print(f'{name:<16}{"":>8}{"":>10}  ✔    {M.describe(name)}')
    print('\n下载： python -m yolo_app models --download yolo26n.pt yolo11s.pt')
    print('对照： python -m yolo_app models --benchmark            # 测本地全部 .pt')
    print('切换： python -m yolo_app run --source data/test.mp4 --model yolo26n.pt')
    return 0


def _export(a, cfg, detector, logger):
    """导出部署格式：ONNX/TensorRT 等；导出后再次推理校验能跑通。"""
    from ultralytics import YOLO
    try:
        path = YOLO(detector.weights).export(
            format=a.format, imgsz=cfg['model']['imgsz'], half=bool(cfg['model']['half']),
            dynamic=a.dynamic, simplify=a.simplify, device=cfg['model']['device'])
        logger.info(f'导出完成: {path}')
    except Exception as e:
        logger.error(f'导出失败（{a.format} 需要对应运行环境，如 TensorRT 需 NVIDIA GPU）: {e}')
        return 1
    return 0
