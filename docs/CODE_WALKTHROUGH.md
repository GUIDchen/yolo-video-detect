# 代码逐行详解

> 这份文档把每个主模块的代码**逐行拆开**讲清楚：这一行在干什么、为什么这么写、删掉会怎样。
> 源码里已经保留了模块级说明与关键注释，本文件是它们的展开版，适合第一次读这个项目时对照阅读。
>
> 阅读顺序建议：**pipeline → detector → motion → sources → reader → analytics → outputs → config → cli → 其余**。
> 带 ★ 的是核心链路，其余是辅助模块。
>
> **v0.6.0 说明**：`reader.py`（OpenCV/FFmpeg 双读取后端）与 `motion.py`（运动门控）**已覆盖（v0.6.0）**，
> 分别见第 5 章与第 3 章，内容按 v0.6.0 源码逐行核对；第 1、4 章的讲解写于 v0.5.x，
> v0.6.0 对主循环与 `sources` 门面有改动，与源码不一致处**以源码为准**。整体设计与数据流见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 0. 先建立全局印象

一帧画面的完整旅程（括号里是文件）：

```
帧 ndarray ──► sources.VideoSource.read()        取帧（reader.py 选 OpenCV/FFmpeg 后端）
           ──► motion.MotionGate.update()        帧差门控：没动就跳过推理（默认关闭）
           ──► detector.Detector.infer(frame)    变成 Results（含 boxes/id/conf）
           ──► analytics.RegionFilter.keep()     ROI 外的框剔除
           ──► analytics.LineCounter.update()    统计越线
           ──► Results.plot()                   画框 → ndarray
           ──► outputs.EventSink.check()        命中就抓拍 + alerts 推送
           ──► outputs.VideoSink.write()        写结果视频（H.264/分段）
           ──► on_frame/on_stats 回调           给 Web 面板
           ──► outputs.write_json()             落统计报告
```

所有编排都在 `pipeline.process()` 一个函数里，v0.6.0 约 250 行。**看懂它就等于看懂整个项目。**

---

## 1. yolo_app/pipeline.py ★核心

### 1.1 导入区（第 5–15 行）

```python
import time                                  # 计时：算 FPS、给摄像头文件名打时间戳
from collections import Counter              # 逐类计数，比手写 dict.get(k,0)+1 干净
from pathlib import Path                     # 路径拼接/取后缀，跨平台
import cv2                                   # 只为了 imshow（弹窗预览）
from .alerts import AlertChannel             # 告警发送
from .analytics import LineCounter, RegionFilter   # 两个后处理器
from .detector import resolve_class_ids      # 类别名→id（事件/越线都要用）
from .outputs import EventSink, VideoSink, write_json  # 三种落地方式
from .sources import VideoSource, is_camera  # 取帧 + 判断是不是摄像头
```

- `from .xxx import` 是包内相对导入，说明这五个模块属于同一个包、可以互相调用；
- **没有**导入 ultralytics、torch：pipeline 只跟"抽象的 Results"打交道，模型细节全在 detector.py —— 这是"换模型不用动主循环"的前提。

### 1.2 output_paths()（第 18–25 行，逐行）

```python
def output_paths(cfg, source):
    out = Path(cfg['output']['dir'])                        # 输出目录，默认 outputs/
    name = cfg['output'].get('name') or ''                   # 配置里显式指定的文件名（空 = 自动）
    if not name:                                             # 没指定就自动命名
        stem = 'camera_' + time.strftime('%Y%m%d_%H%M%S') if str(source).isdigit() \
               else Path(str(source)).stem                   # 摄像头→时间戳；文件→原文件名（不含扩展名）
        name = f'{stem}.mp4'                                 # 统一加 .mp4
    return out / name                                        # Path / str = 拼接路径
```

- `str(source).isdigit()`：摄像头用整数编号（0、1），其它都是字符串路径，用这个区分；
- 摄像头必须带时间戳，否则每次运行都会覆盖同一个 `camera.mp4`；
- 返回 `Path` 而不是 str，后面 `.with_suffix('.json')` 才好用。

### 1.3 process() 初始化部分（第 43–78 行，逐行）

```python
    run_cfg = cfg['run']                                     # 把常用配置取出来，少写层 cfg['run']['x']
    show = run_cfg['show'] if show is None else show          # 参数优先，其次配置；None 表示"调用方没指定"
    record = cfg['output']['video'] if record is None else record
```

```python
    src = VideoSource(source, cfg['camera'], run_cfg['reconnect'], run_cfg['retry_delay'])
    if not src.cap.isOpened():                               # 打不开：文件不存在 / 摄像头没权限 / RTSP 不通
        hint = ''
        if not is_camera(source) and not Path(str(source)).exists():
            hint = ('\n  · 文件/目录不存在：容器里跑要先把宿主机目录挂进来，'
                    '例如 -v $PWD/data:/app/data，再用 --source data/xxx.mp4')
        elif is_camera(source):
            hint += ('\n  · macOS 权限：系统设置 → 隐私与安全性 → 摄像头 → 勾选 VS Code / 终端，然后重启该应用'
                    '\n  · 查看可用摄像头：ffmpeg -f avfoundation -list_devices true -i ""'
                    '\n  · 换个编号试试：--index 1（或设 configs 里 camera.index）')
        raise RuntimeError(f'无法打开视频源: {source}{hint}')
```

- 这里**故意啰嗦**：90% 的"打不开"是权限或路径问题，一次把排查动作打全，比让人去翻文档强；
- 判断顺序有讲究：先判"文件不存在"（容器漏挂载最常见），再判摄像头权限，两类提示不混在一起。

```python
    video_path = output_paths(cfg, source)                   # 结果视频路径（同时决定 JSON 路径）
    sink = VideoSink(video_path, src.fps, cfg['output']['segment_seconds']) if record else None
```

- `src.fps` 是**回读**的真实帧率，不是我们请求的值（摄像头经常给不到 30）；
- `record=False` 时 sink 为 None，后面所有 `if sink:` 都会跳过 —— 用"空对象"代替分支开关。

```python
    ev_cfg = cfg['output']['events']
    events = (EventSink(Path(cfg['output']['dir']) / 'events',
                        resolve_class_ids(ev_cfg['classes'], detector.names),   # ← 关键：名称转 id
                        ev_cfg['min_conf'], ev_cfg['cooldown_s']) if ev_cfg['enable'] else None)
```

- `resolve_class_ids` 是之前修过的一个 bug：配置写 `[person]` 时必须转成 `[0]`，否则永远匹配不上，抓拍静默失效；
- 同理下面 LineCounter 也要转一次。

```python
    a_cfg = cfg.get('analytics', {})
    roi = RegionFilter(a_cfg.get('roi'))                     # roi=[] 时 enabled=False，等于没开
    line = LineCounter(a_cfg.get('line'), resolve_class_ids(a_cfg.get('classes'), detector.names), detector.names)
    alert = AlertChannel(cfg.get('alert', {}).get('webhook', ''),
                         cfg.get('alert', {}).get('type', 'generic'),
                         cfg.get('alert', {}).get('timeout', 5), logger)
```

- 三个后处理器都用 `.get(..., 默认值)` 读配置：**老配置文件缺字段也能跑**（向后兼容）。

```python
    logger.info(f'开始处理 {source} | {int(src.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
                f'{int(src.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @{src.fps:.0f}fps | 设备 {detector.device_name}'
                f'{" | 跟踪 " + cfg["track"]["tracker"] if detector.track else ""}'
                f'{" | ROI" if roi.enabled else ""}{" | 越线计数" if line.enabled else ""}'
                f'{" | 告警已启用" if alert.enabled else ""}')
```

- 一行把"生效的配置"打全：源、分辨率、帧率、设备、开了哪些功能。**排查现场问题第一步就是看这行日志。**

### 1.4 主循环（第 80–132 行，逐行）

```python
    hits, ids = Counter(), set()      # hits：各类别逐帧累计次数；ids：出现过的 track_id 集合
    crossings = []                    # 越线事件明细
    idx = done = 0                    # idx=读入帧序号；done=实际推理帧数（跳帧后 ≠ idx）
    t0 = time.time()                  # 计时起点
    while True:
        frame = src.read()            # 取帧（内部含直播源重连）
        if frame is None:             # None = 文件读完 / 直播重连失败
            break
        idx += 1
        if run_cfg['max_frames'] and idx > run_cfg['max_frames']:
            break                     # 限制帧数（调试/试跑用）
        if run_cfg['stride'] > 1 and (idx - 1) % run_cfg['stride']:
            if sink:
                sink.write(frame)     # 跳过的帧原样写出：保证结果视频时长/帧率与源一致
            continue                  # 注意 continue 在 write 之后，否则视频会缺帧卡顿
```

```python
        res = roi.keep(detector.infer(frame))
```

- 一行两件事：先推理，再做 ROI 过滤。`roi.keep()` **就地修改** `res.boxes`，所以后面的绘制、统计、抓拍全都只看得到 ROI 内的目标 —— 一次过滤，全链路生效。

```python
        crossed = line.update(res, detector.names)
        if crossed:
            for c in crossed:
                c['frame'] = idx                                  # 补上帧号，便于回溯
                crossings.append(c)
                alert.send(f'[YOLO 越线] {c["class"]} {c["direction"]} | track {c["track_id"]}')
```

- 越线是"事件"，既有明细列表（写进 JSON），也即时推 webhook；
- 注意这里用 `alert.send` 而不是 `notify_event`：越线没有抓拍图，消息格式不同。

```python
        hits.update(detector.names[int(c)] for c in res.boxes.cls.tolist())
        tids = res.boxes.id.tolist() if res.boxes.id is not None else []
        ids.update(int(t) for t in tids if t is not None)
        done += 1
```

- `boxes.id` 只有开 `--track` 时才有值，所以要先判断 `is not None`；
- `hits` 是"逐帧次数"，`ids` 是"唯一目标"，两者口径不同，JSON 里分别叫 `per_class_frames` 和 `unique_tracks`。

```python
        anno = res.plot(labels=True, conf=True)   # ultralytics 自带绘图：框 + "类别 置信度"
        roi.draw(anno)                            # 叠加 ROI 多边形
        line.draw(anno)                           # 叠加越线 + 实时 in/out 计数
```

- 三者在**同一张 ndarray** 上叠加：写盘、弹窗、Web 面板共用这一张图，不重复计算。

```python
        if events and events.check(res, detector.names, idx, anno):
            alert.notify_event(events.events[-1], source)   # 抓拍成功才推送
        if sink:
            sink.write(anno)
        if show:
            cv2.imshow('YOLO', anno)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                                        # q 退出预览
```

- `cv2.waitKey(1)` 必须调用，否则 GUI 不刷新；`& 0xFF` 是处理不同平台返回值差异的惯例写法；
- `events.check` 内部有冷却时间，避免同一目标每帧都抓拍。

```python
        if on_frame:
            on_frame(anno)                # Web 面板：拿最新一帧编码成 JPEG
        if on_stats:
            on_stats({'frames': done, 'fps': round(done / max(time.time() - t0, 1e-6), 1),
                      'per_class': dict(hits), 'unique_tracks': len(ids),
                      'line': dict(line.total), 'events': len(events.events) if events else 0})
        if done % max(cfg['log']['interval'], 1) == 0:
            el = time.time() - t0 or 1e-6
            logger.info(f'帧 {idx}/{src.total or "-"} | {done / el:.1f} FPS | {dict(hits)}')
```

- 回调是"可选参数"：命令行跑时是 None，Web 面板跑时才有值 —— **同一段主循环服务两种入口**，不需要复制；
- `max(..., 1e-6)` 防除零；`src.total or "-"` 是因为直播源 `CAP_PROP_FRAME_COUNT` 常为 0。

### 1.5 收尾（第 134–160 行，逐行）

```python
    src.release()                     # 释放摄像头/文件句柄（不释放会导致下次打不开）
    if sink:
        sink.close()                  # 关键：不 close 的话 mp4 的索引没写完，文件损坏
    if show:
        cv2.destroyAllWindows()
    el = time.time() - t0 or 1e-6     # 总耗时（同样防除零）
    summary = {
        'source': str(source), 'model': detector.weights,
        'frames_src': idx, 'frames_inferred': done,        # 读入帧数 vs 实际推理帧数
        'elapsed_s': round(el, 1), 'avg_fps': round(done / el, 1),
        'per_class_frames': dict(hits), 'unique_tracks': len(ids),
        'reconnects': src.reconnects,                      # 直播链路质量指标
        'output_video': sink.output if sink else None,      # 分段时返回第一段
        'video_parts': sink.parts if sink else [],         # 全部分段文件
        'events': events.events if events else [],
        'alert_sent': alert.sent, 'alert_failed': alert.failed,   # 告警成功率
    }
    summary.update(line.summary())                          # 合并 line_total / line_by_class
    if crossings:
        summary['crossings'] = crossings
    if cfg['output']['json']:
        summary['output_json'] = write_json(Path(video_path).with_suffix('.json'), summary)
    return summary
```

- 返回 dict 而不是打印：调用方（CLI 批量、Web 面板）自己决定怎么用；批量时这些 dict 汇总成 `summary_all.json`。

### 1.6 run_job()（第 163–174 行，逐行）

```python
def run_job(payload):
    source, cfg, show = payload                          # 多进程只传可 pickle 的基础数据
    from .detector import Detector                       # 在子进程里 import：进程池要求函数可序列化
    from .logging_setup import get_logger
    logger = get_logger(cfg['log']['level'], cfg['log'].get('file'))
    try:
        return process(source, cfg, Detector(cfg), logger, show=show)
    except Exception as e:                               # 一路失败不能拖垮其它路
        logger.error(f'{source} 处理失败: {e}')
        return {'source': str(source), 'error': str(e)}
```

- **模型对象不能跨进程传递**（含 CUDA 上下文），所以每个子进程自己 `Detector(cfg)` 加载一份；
- 返回 `{'error': ...}` 而不是抛异常，让主进程能继续收集其它源的结果。

---

## 2. yolo_app/detector.py ★模型封装（逐行）

```python
ROOT = Path(__file__).resolve().parent.parent       # 项目根目录（本文件在 yolo_app/ 下）

def resolve_weights(name):
    if Path(name).exists():                          # 传的是完整路径且存在 → 直接用
        return str(Path(name))
    local = ROOT / 'weights' / name                  # 否则看项目内 weights/ 有没有同名文件
    return str(local) if local.exists() else name    # 都没有 → 原样返回，交给 ultralytics 去下载
```

- `__file__` 是当前文件路径，`resolve()` 转绝对路径，`.parent.parent` 上跳两级到项目根；
- 用"文件是否存在"来决定行为，而不是靠参数开关：**对使用者更友好**（裸写 `yolo11n.pt` 也能离线跑）。

```python
def resolve_class_ids(classes, names):
    if not classes:
        return None                                  # 空列表 = 不过滤（None 是 ultralytics 的约定）
    out = []
    for c in classes:
        if isinstance(c, int) or str(c).isdigit():
            out.append(int(c))                       # 数字直接当 id
        else:
            ids = [i for i, n in names.items() if str(n).lower() == str(c).lower()]
            if not ids:
                raise ValueError(f'未知类别名: {c}')   # 拼错类别名要立刻报错，不要静默失效
            out.extend(ids)
    return sorted(set(out))                          # 去重 + 排序，保证结果稳定可测试
```

- `names` 是 ultralytics 给的 `{0: 'person', 1: 'bicycle', ...}`；
- `raise` 而不是忽略：类别名写错是常见错误，静默跳过会让人以为"检测不到"。

```python
class Detector:
    def __init__(self, cfg):
        m = cfg['model']
        self.weights = resolve_weights(m['weights'])   # 定位权重（可能被改写成 weights/xxx.pt）
        self.model = YOLO(self.weights)                # 真正加载模型（首次会构建网络）
        self.names = self.model.names                  # {id: 类别名}
        self.imgsz, self.conf, self.iou = m['imgsz'], m['conf'], m['iou']
        self.device, self.half = m['device'], m['half']
        self.augment = m.get('augment', False)         # .get：老配置文件没有这字段也能跑
        self.track = cfg['track']['enable']
        self.tracker = cfg['track']['tracker']
        self.classes = self._resolve_classes(m['classes'])
```

```python
    def infer(self, frame):
        kw = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                  classes=self.classes, verbose=False)   # verbose=False：别每帧刷 ultralytics 日志
        if self.half:
            kw['half'] = True                            # 只在开启时才传（8.4 起该参数已 deprecated）
        if self.augment:
            kw['augment'] = True                         # TTA
        if self.track:
            return self.model.track(frame, persist=True, tracker=self.tracker, **kw)[0]
        return self.model.predict(frame, **kw)[0]        # [0]：单帧推理只取第一个结果
```

- `persist=True`：跨帧保留 tracker 状态，**不加这句 track_id 每帧都会重新编号**，越线计数就废了；
- 返回统一的 `Results`，上层完全不关心是 PyTorch 还是 ONNX 还是 TensorRT。

```python
    @property
    def device_name(self):
        try:
            return str(next(self.model.model.parameters()).device)   # torch 模型能拿到真实设备
        except Exception:
            return str(self.device or 'auto')                        # ONNX/TensorRT 没有 parameters()
```

- 这是修过的 bug：加载 `.onnx` 时 `self.model.model` 不是 nn.Module，直接取参数会抛异常。

---

## 3. yolo_app/motion.py ★运动门控（逐行）

v0.6.0 新增，全文 74 行。它只回答一个问题：**这一帧需不需要推理**。位置在 `pipeline.process()`
取帧之后、`detector.infer()` 之前，默认关闭（`motion.enable: false`）；关闭时主循环完全不构造它。
命中跳过时，帧仍然写盘、仍然回调，只是不推理 —— 省的是算力，不是 I/O。

### 3.1 模块职责与算法（第 1–11 行）

模块注释把算法写成了一条五步流水线：

```
灰度 → 高斯模糊（压噪点）→ 与上一帧 absdiff → 阈值 → 形态学开运算（去飞点）
    → 可选 ROI 掩码 → 非零像素占比 ≥ min_area_ratio 才认为有运动
```

- 全部是 OpenCV 的廉价算子：一次 `cvtColor` + 一次 `absdiff`（模块注释给的量级是 640×480 约 1 ms），
  与一次几十毫秒的推理不在一个数量级，所以"先筛一遍"几乎稳赚；
- 它判断的是"画面变没变"，不是"有没有目标"：目标停住后帧差归零，门控会关闭，这一点在 3.7 展开；
- 模块注释同时列了已知失效场景（夜间噪点、光照突变、镜头晃动）与后果：门控"常开"、退化成逐帧推理。
  **不会漏检，只是省不下算力**，调大 `threshold` / `min_area_ratio` 即可。

### 3.2 MotionGate 的构造函数（第 19–28 行，逐行）

```python
    def __init__(self, threshold=25, min_area_ratio=0.002, blur=5, roi=None):
        self.threshold = max(int(threshold), 1)
        self.min_area_ratio = max(float(min_area_ratio), 0.0)
        self.blur = int(blur) if int(blur or 0) > 1 else 0
        self.roi = roi or []
        self.mask = None
        self._mask_shape = None
        self.prev_gray = None
        self.last_ratio = 1.0          # 最近一帧的运动面积占比（写日志/调试用）
        self.enabled = True
```

- 三个默认值与 `config.DEFAULTS['motion']` 一致（`threshold: 25`、`min_area_ratio: 0.002`、`blur: 5`）；
  正常走配置时由 `pipeline` 显式传值，构造函数默认值只在直接实例化（如单测）时生效；
- `max(int(threshold), 1)`：阈值 0 会让任何非零灰度差都算运动（`absdiff > 0`），门控等于常开，这里兜底成 1；
- `max(float(min_area_ratio), 0.0)`：负数没有意义。注意传 **0** 不是"关闭门控"，而是 `last_ratio >= 0` 恒成立 → 门控常开；
- `int(blur) if int(blur or 0) > 1 else 0`：`None`、0、1 都归一成 0，表示不做高斯模糊；避免把 1 当核大小传进
  `GaussianBlur`（没有模糊意义），也不用在 `update()` 里再判一次；
- `roi or []`：`None` 与空列表统一成空列表，后面只判真假；
- `mask` / `_mask_shape`：ROI 掩码及其尺寸缓存，惰性构建（见 3.3）；
- `prev_gray`：唯一的跨帧状态，保存上一帧的灰度图；
- `last_ratio` 初值 1.0：还没算过时按"有运动"处理，日志/调试不会读到 0 而误判；
- `enabled = True` 目前只是标记：pipeline 用 **`motion is not None`** 决定是否启用（配置 `motion.enable` 在构造前判断），
  不会去改 `gate.enabled`。

### 3.3 `_ensure_mask()`：ROI 掩码何时重建（第 30–38 行，逐行）

```python
    def _ensure_mask(self, shape):
        """ROI 掩码按帧尺寸懒构建；分辨率变了（重连换流）自动重建。"""
        if not self.roi:
            return
        if self.mask is not None and self._mask_shape == shape:
            return
        self.mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(self.mask, [np.array(self.roi, dtype=np.int32)], 255)
        self._mask_shape = shape
```

- 没配 ROI 时直接返回，`mask` 永远是 `None`；`update()` 会用整帧像素数作分母；
- 只有两种情况重建：`mask` 还不存在（首帧），或 `_mask_shape != shape`（分辨率变了）。
  正常帧每帧都会调一次，命中缓存时只是一次元组比较，开销可忽略；
- 参数 `shape` 是灰度图的 `(高, 宽)`，掩码取 `shape[:2]` 建同尺寸单通道图；`fillPoly` 把 `roi` 多边形内填 255；
- 分辨率变化时，`update()` 会先走"首帧"分支（`prev_gray.shape != gray.shape`）返回 `True`，
  同一次调用里 `_ensure_mask` 发现尺寸变了就重建 —— 重连换流后不会继续用旧尺寸的掩码；
- **已知边界**：重建条件只看尺寸，不看 `roi` 内容。运行中改 `gate.roi` 不会自动重建，要新建 `MotionGate`；
- **坑**：`roi` 给的是"一个多边形"的顶点列表，不是多边形列表。`fillPoly` 外层那个 `[ ... ]` 才是多边形列表，
  所以 `roi=[[[x,y],...], [[x,y],...]]` 这种多区域写法不会被支持。

### 3.4 update()：首帧、帧差与运动占比（第 40–63 行，逐行）

```python
    def update(self, frame):
        """喂一帧，返回"是否有运动"。首帧与分辨率变化时返回 True（必须推理）。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self._ensure_mask(gray.shape)
            return True

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, th = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.dilate(th, kernel, iterations=1)

        self._ensure_mask(gray.shape)
        denom = th.size
        if self.mask is not None:
            th = cv2.bitwise_and(th, self.mask)
            denom = int(cv2.countNonZero(self.mask)) or th.size
        self.last_ratio = cv2.countNonZero(th) / float(denom)
        return self.last_ratio >= self.min_area_ratio
```

**为什么首帧必定返回 `True`**（第 45–48 行）：

- 没有上一帧就没有参照，"动没动"无从判断；返回 `True` 等于告诉调用方"这一帧必须推理"；
- 如果这里返回 `False`，而画面又一直静止，`diff` 会一直是 0，门控就永远打不开：`pipeline` 的 `last_res` 始终是 `None`，
  `hold_frames` 也没有框可画，静态场景会一帧都不推理；
- 这里同时是 ROI 掩码第一次构建的位置（`_ensure_mask`），跳过就等于这一路没有掩码；
- 分辨率变化（重连换流、换摄像头）复用了同一分支：`absdiff` 要求两张图同尺寸，旧 `prev_gray` 不能再用，
  顺手重建掩码并强制推理一帧。

**逐段说明**：

- `cvtColor(..., COLOR_BGR2GRAY)`：OpenCV 默认 BGR，转灰度后计算量降到 1/3；
- `if self.blur:`：0 表示关闭；开启时先用 `GaussianBlur` 压孤立噪点，避免夜间噪点把整幅图判成"运动"；
- `absdiff`：逐像素绝对差，结果仍是 `uint8`；`prev_gray` 在算完差值后立刻更新为当前帧，
  保存的是**模糊后**的灰度图，下一帧与它比较，不会出现"模糊前后混用"的漂移；
- `threshold(diff, self.threshold, 255, THRESH_BINARY)`：灰度差 **严格大于** 阈值才置 255；
  阈值越大越不敏感；
- `MORPH_OPEN`（先腐蚀后膨胀）：去掉孤立飞点；`dilate` 再把剩下的运动区域扩一圈，
  避免小目标被开运算削没，也让碎片连成块。3×3 核每帧新建一次，分配开销可忽略；
- ROI 处理：`bitwise_and` 只保留掩码内的运动；分母改用 **ROI 面积**（`countNonZero(mask)`），
  否则 ROI 很小时，运动像素占整幅画面的比例永远达不到 `min_area_ratio`；
- `... or th.size`：掩码全 0（ROI 退化/全在画面外）时防除零。但此时 `th` 也已经被与成全 0，
  `last_ratio` 恒为 0 → 门控永远关闭（见 3.7）；
- `last_ratio` 用 `float` 除法保存，返回 `>= min_area_ratio` 的布尔值；它既是判定依据，也是调试时唯一能看到的"运动强度"。

### 3.5 overlay_boxes()：沿用上一帧的框（第 66–74 行，逐行）

```python
def overlay_boxes(frame, res, color=(0, 200, 0)):
    """把上一帧的框画到当前帧上（motion.hold_frames > 0 时用，避免画面突然"空掉"）。"""
    out = frame.copy()
    boxes = getattr(res, 'boxes', None)
    if boxes is None or not len(boxes):
        return out
    for x1, y1, x2, y2 in boxes.xyxy.tolist():
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return out
```

- `frame.copy()` 是必须的：`pipeline` 里同一帧还要 `sink.write()` 与 `on_frame()`，
  直接画在原图上会把"上一帧的框"写进结果视频；单测 `test_overlay_boxes_does_not_mutate_source` 钉的就是这个行为；
- `getattr(res, 'boxes', None)`：拿不到结果对象的结构时退化为"原图副本"，不抛异常；
- `boxes.xyxy` 是 ultralytics 的 `(N, 4)` 张量，`.tolist()` 转成 Python 浮点；
  `int()` 是因为 `cv2.rectangle` 只收整数坐标；
- 默认颜色 `(0, 200, 0)` 是 **BGR** 的绿，不是 RGB；线宽固定 2；
- 只画框，不画类别、置信度、ROI 与越线。它的定位是"占位"，避免画面突然空白，不是标注；
- 框来自上一帧结果，目标移动后框会滞后；所以只能由 `hold_frames` 限量使用，不能当跟踪。

### 3.6 与 pipeline 的接线（第 110–115、179–191 行）

构造（`pipeline.py`）：

```python
        m_cfg = cfg.get('motion', {})
        if m_cfg.get('enable'):
            motion = MotionGate(m_cfg.get('threshold', 25), m_cfg.get('min_area_ratio', 0.002),
                                m_cfg.get('blur', 5),
                                a_cfg.get('roi') if m_cfg.get('mask_roi', True) else None)
        hold_frames = int(m_cfg.get('hold_frames', 0) or 0)
```

跳过分支（在 `stride` 判断之后、`StationaryGate` 之前）：

```python
            if motion is not None and not motion.update(frame):
                skip_motion += 1
                out = frame
                if hold_frames and last_res is not None and hold_left < hold_frames:
                    out = overlay_boxes(frame, last_res)      # 沿用上一帧框，避免画面突然空白
                    hold_left += 1
                if sink:
                    sink.write(out)
                if on_frame:
                    on_frame(out)
                emit_stats()
                continue
            hold_left = 0
```

- 门控在取帧、`max_frames`、`stride` 之后执行：跳过的帧本来就不推理，无需再筛；
- 被判"无运动"的帧仍然 `sink.write`、`on_frame`、`emit_stats` —— 录像时长、面板画面、FPS 统计都不会断；
- `hold_frames` 只影响"画什么"：`> 0` 且已有上一帧结果时，用 `overlay_boxes` 叠旧框；
  默认 0 表示直接写原帧；
- `hold_left` 在运动恢复后被清零，下一次静止重新计数，不会无限沿用旧框；
- 计数进入 summary：`skip_motion` 汇入 `frames_inference_skipped` 与 `skip_reasons.motion`（第 258–259 行）。

### 3.7 常见坑与取舍

- **门控"常开"**：夜间噪点、光照突变、镜头晃动都会让帧差超过阈值，表现为退化成逐帧推理 ——
  不会漏检，只是省不下算力。优先调大 `threshold` / `min_area_ratio`，或开启 `blur` 压噪；
- **参数写成 0 不等于关掉门控**：`threshold` 被兜底为 1，`min_area_ratio=0` 时 `ratio >= 0` 恒真。
  要关闭请用 `motion.enable: false`；
- **ROI 退化会"永关"**：ROI 全在画面外或退化成线/点，掩码全 0，`ratio` 恒 0，门控永远说"没动"，
  这一路就再也不会推理。配 ROI 后先用调试日志看一次 `last_ratio`；
- **"变化检测"不是"存在检测"**：目标进入画面后停住，帧差归零，门控关闭，之后不会再有新的推理结果，
  框也停止更新。需要持续框选时应关闭门控，或接受 `hold_frames` 的短暂延续；
- **首帧必真**：每个视频开头、每次分辨率变化至少推理一帧；静态视频的实测是"75 帧只推理 1 次"
  （见 [OPTIMIZATION.md](OPTIMIZATION.md) 5.2），不是 0 次；
- **只省推理，不省解码**：帧仍要解码、写盘、编码、回调。CPU 瓶颈在解码/编码时，门控收益有限；
- **掩码只按尺寸重建**：运行中改 ROI 不生效，必须重建实例。

---

## 4. yolo_app/sources.py ★取帧（逐行）

> v0.6.0 起，本文件的 `_open()/read()` 已整体下沉到 `reader.py`（见第 5 章），`VideoSource` 只剩透传门面，
> 模块内保留 `is_camera()` 与 `list_videos()`。下面保留 v0.5.x 的实现讲解作为对照，当前代码以 `reader.py` 为准。

```python
VIDEO_EXT = {'.mp4', '.avi', ...}                      # 目录批量时认这些扩展名
LIVE_PREFIX = ('rtsp://', 'rtmp://', 'http://', 'https://')

def is_camera(uri):
    return isinstance(uri, int) or str(uri).isdigit()  # 0 / '0' 都算摄像头
```

```python
def list_videos(path):
    p = Path(path)
    if not p.is_dir():
        return [str(path)]                             # 单个文件也统一返回列表 → 上层循环不用分情况
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
```

```python
class VideoSource:
    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0):
        self.uri = int(uri) if is_camera(uri) else uri   # '0' → 0，OpenCV 需要 int 才是摄像头
        self.camera = camera or {}
        self.reconnect, self.retry_delay = reconnect, retry_delay
        self.cap = None
        self.reconnects = 0                              # 统计重连次数，写进报告
        self._open()
```

```python
    @property
    def is_live(self):
        return is_camera(self.uri) or str(self.uri).startswith(LIVE_PREFIX)   # 直播流才需要重连
```

```python
    def _open(self):
        self.cap = cv2.VideoCapture(self.uri)            # 打开：文件/摄像头/RTSP 同一个 API
        if is_camera(self.uri):                          # 摄像头才设置采集规格
            c = self.camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.get('width', 1280))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.get('height', 720))
            self.cap.set(cv2.CAP_PROP_FPS, c.get('fps', 30))
        return self.cap.isOpened()                       # 返回是否成功，供重连判断
```

- `cap.set()` 只是"请求值"，驱动可能给不了，所以后面用 `src.fps` 回读真实值。

```python
    def read(self):
        ok, frame = self.cap.read()
        if ok:
            return frame                                 # 正常返回这一帧
        if self.is_live and self.reconnects < self.reconnect:   # 直播源掉线 → 尝试重连
            self.reconnects += 1
            time.sleep(self.retry_delay)
            self.cap.release()                           # 必须先释放旧句柄，否则可能拿到失效对象
            if self._open():
                ok, frame = self.cap.read()
                return frame if ok else None
        return None                                      # None = 结束/彻底失败，由主循环 break
```

```python
    @property
    def fps(self):  return self.cap.get(cv2.CAP_PROP_FPS) or 25.0     # 取不到就按 25 算
    @property
    def total(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 直播源通常为 0
    def release(self): 
        if self.cap is not None:
            self.cap.release()
```

- `or 25.0`：OpenCV 对某些流返回 0，直接拿去建 VideoWriter 会写出坏文件。
---

## 5. yolo_app/reader.py ★读取后端（逐行）

v0.6.0 新增，全文 388 行。它把"帧从哪来"从 `sources.py` 里拆出来，提供两个可互换的后端：

- `OpenCVReader`：`cv2.VideoCapture`，文件与摄像头的默认选择；
- `FFmpegReader`：ffmpeg 子进程 + rawvideo 管道 + 读线程 + 有界队列 + 僵死看门狗，直播源的默认选择。

对外契约只有 `read() -> ndarray | None`、`release()`，以及 `fps / total / width / height / reconnects / dropped`
几个只读属性；`VideoReader` 负责按配置挑后端并把属性透传出去。

为什么要第二个后端（模块注释第 3–8 行）：`cv2.VideoCapture` 的缓冲不可控、没有读超时，RTSP 断流时会一直阻塞
或返回陈旧帧，延迟还会随时间累积。换成 ffmpeg 子进程后：队列长度 = `camera.buffer_size`、满了丢最旧帧（延迟不累积，
丢了多少有计数）；`read_timeout` 判"多久没来帧"，到点杀进程重启；`-rtsp_transport tcp` + `nobuffer/low_delay` 降抖动。

### 5.1 后端选择（第 24–36 行）

```python
LIVE_PREFIX = ('rtsp://', 'rtmp://', 'http://', 'https://')


def is_live_uri(uri):
    return str(uri).startswith(LIVE_PREFIX)


def choose_backend(uri, mode='auto', has_ffmpeg=None):
    """决定用哪个读取后端：auto 时只有"直播源 + ffmpeg 可用"才选 ffmpeg。"""
    if mode in ('opencv', 'ffmpeg'):
        return mode
    has = shutil.which('ffmpeg') is not None if has_ffmpeg is None else bool(has_ffmpeg)
    return 'ffmpeg' if (is_live_uri(uri) and has) else 'opencv'
```

- 四个前缀就是"直播"的判定口径：RTSP、RTMP、HTTP、HTTPS。**HTTP(S) 也算直播**（HLS/HTTP-FLV 属于这类），
  `auto` 会给它们选 FFmpeg；
- 显式 `mode` 直接返回，**不检查 ffmpeg 是否安装**：可用性交给后面的 `probe_stream()` 与 `_open()` 处理。
  单测 `test_choose_backend_matrix` 固定了这个矩阵，包括"显式 ffmpeg + 没有二进制仍返回 ffmpeg"；
- `auto` 要两个条件同时成立才选 ffmpeg：`is_live_uri(uri)` 为真，且 `shutil.which('ffmpeg')` 找得到可执行文件；
  `has_ffmpeg` 参数只为注入测试，生产调用不传；
- 为什么 `auto` 只给直播源用 FFmpeg（文件走 OpenCV）：
  - 文件走 OpenCV 是直接解码，FFmpeg 多一层进程、管道与 swscale 色彩转换开销（[OPTIMIZATION.md](OPTIMIZATION.md)
    5.2 的遗留风险里明确写了这一点）；
  - 文件有确定的 EOF 和总帧数，OpenCV 的 `CAP_PROP_FRAME_COUNT`、seek 都够用，不需要看门狗；
  - FFmpeg 的收益（TCP 传输、低延迟缓冲、读超时、断流重启）只对直播成立；
- 为什么摄像头也不给 FFmpeg：模块注释给的取舍是"摄像头在 macOS 上走 avfoundation 更稳"，
  而 OpenCV 已经封装了平台采集后端；直接用 ffmpeg 要写 `-f avfoundation -i "0"` 这类平台相关参数；
- 反过来，显式 `mode='ffmpeg'` 对文件也可用，测试 `test_ffmpeg_matches_opencv_on_file` 就是用它做两后端像素对比。

### 5.2 ffprobe 探测与帧率解析（第 39–64 行）

```python
def parse_rate(text):
    """'30000/1001' -> 29.97；解析不了返回 None。"""
    try:
        num, _, den = str(text).partition('/')
        return float(num) / float(den or 1)
    except Exception:
        return None


def probe_stream(uri, timeout=10):
    """ffprobe 探测分辨率/帧率/总帧数；任何失败都返回 None（由调用方回退）。"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
           '-of', 'json', str(uri)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True).stdout
        info = json.loads(out)['streams'][0]
    except Exception:
        return None
    w, h = int(info.get('width') or 0), int(info.get('height') or 0)
    if not w or not h:
        return None
    nb = str(info.get('nb_frames') or '')
    return {'width': w, 'height': h,
            'fps': parse_rate(info.get('r_frame_rate')) or 25.0,
            'total': int(nb) if nb.isdigit() else 0}
```

- `parse_rate` 用 `partition('/')` 切分：`'30000/1001'` → `30000 / 1001`；没有 `/` 时 `den` 为空，
  `den or 1` 当整数帧率。实测 `29.97002997002997`、`25.0`、`N/A` → `None`；
- ffprobe 命令只取 `v:0` 的四个字段：`-v error` 压掉无关日志，`check=True` 让非零退出进异常分支，
  `timeout=10` 防止探测本身挂死；
- **任何异常都返回 `None`**（没装 ffprobe、文件坏、RTSP 不通、JSON 结构不对），由 `VideoReader` 回退 OpenCV。
  探测失败不是致命错误，只是拿不到低延迟后端；
- 宽高任一为 0/缺失也返回 `None`：没有尺寸就算不出 `frame_bytes`，rawvideo 无法分帧；
- `nb_frames` 在直播里常见 `'N/A'`，`isdigit()` 不成立 → `total=0`；主循环日志用 `src.total or "-"` 显示为 `-`；
- fps 解析失败回退 25.0，与 `OpenCVReader.fps` 的 `or 25.0` 同一口径；
- 为什么必须在启动 ffmpeg 前探测：rawvideo 管道**没有容器头**，Python 侧只能靠宽高算每帧字节数并 `reshape`；
  fps/total 则用于日志、统计与 `VideoSink`。

### 5.3 OpenCVReader（第 67–136 行，逐行）

```python
    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0, logger=None):
        self.uri = uri
        self.camera = camera or {}
        self.reconnect, self.retry_delay = reconnect, retry_delay
        self.logger = logger
        self.cap = None
        self.reconnects = 0
        self.dropped = 0
        self._fails = 0
        self._open()
```

```python
    def _open(self):
        self.cap = cv2.VideoCapture(self.uri)
        c = self.camera
        if isinstance(self.uri, int):                  # 摄像头：申请指定分辨率/帧率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.get('width', 1280))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.get('height', 720))
            self.cap.set(cv2.CAP_PROP_FPS, c.get('fps', 30))
        if c.get('buffer_size'):                       # 直播源只留最新帧
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(c['buffer_size']))
        return self.cap.isOpened()
```

```python
    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self._fails = 0
            return frame
        if not (isinstance(self.uri, int) or is_live_uri(self.uri)):
            return None
        limit = self.reconnect if self.reconnect is not None else 0
        while limit < 0 or self._fails < limit:
            self._fails += 1
            self.reconnects += 1
            time.sleep(self.retry_delay)
            self.cap.release()
            if self._open():
                ok, frame = self.cap.read()
                if ok:
                    self._fails = 0
                    return frame
        return None
```

- 只有 `uri` 是 `int` 时才申请摄像头参数：默认 1280×720@30，与 `config.DEFAULTS['camera']` 一致；
  `set()` 是"请求值"，驱动可能给不了，所以 `fps/width/height` 都是回读 `cap.get()`；
- `buffer_size` 只要配置非 0 就设 `CAP_PROP_BUFFERSIZE`（摄像头与直播都设），`camera.buffer_size: 1` 是
  v0.6.0 的低延迟手段；驱动不支持时 `set` 返回 `False` 但代码不检查，表现为"设了没用"；
- `dropped` 在这个后端**恒为 0**：没有队列，不会丢帧。保留该字段是为了门面与统计字段统一；
- `opened`（第 94–96 行）返回 `cap.isOpened()`，`pipeline` 用它决定是否抛"无法打开视频源"；
- 重连只对摄像头/直播生效：文件读失败（含读到结尾）直接 `return None`，不重试 —— 文件不存在或损坏时重试没有意义，
  单测 `test_offline_file_does_not_reconnect` 钉住了这个行为；
- 重试语义与 FFmpeg 后端一致：`_fails` 是**连续失败计数**，成功一帧就清零；`reconnect` 是"连续失败上限"，
  负数表示无限；`reconnects` 是累计尝试次数（进报告/指标）；
- 与 FFmpeg 后端的差别：这里 `time.sleep(retry_delay)` 是固定间隔，没有退避；也没有"重启后等首帧"的额外超时。
  更大的问题是 `cap.read()` 本身可能永久阻塞 —— 这正是直播源要换 FFmpeg 的根本原因；
- `fps` 属性（第 118–120 行）用 `cap.get(...) or 25.0`：部分流返回 0，直接拿去建 `VideoWriter` 会写出坏文件。

### 5.4 FFmpegReader 初始化与 ffmpeg 命令（第 139–190 行，逐行）

```python
    def __init__(self, uri, info, buffer_size=1, read_timeout=10.0, reconnect=0,
                 retry_delay=1.0, logger=None, cmd_builder=None):
        self.uri = uri
        self.info = info
        self.width, self.height = int(info['width']), int(info['height'])
        self.fps = float(info.get('fps') or 25.0)
        self.total = int(info.get('total') or 0)
        self.frame_bytes = self.width * self.height * 3
        self.live = is_live_uri(uri)
        self.read_timeout = float(read_timeout or 0)
        self.buffer_size = max(int(buffer_size or 1), 1)
        self.reconnect, self.retry_delay = reconnect, float(retry_delay or 1)
        self.logger = logger
        self._cmd_builder = cmd_builder or self._build_cmd
        self._q = queue.Queue(maxsize=self.buffer_size)
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._fails = 0
        self.reconnects = 0
        self.dropped = 0
        self._eof = False
        self._open()
```

```python
    def _build_cmd(self):
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
        if str(self.uri).startswith('rtsp://'):
            cmd += ['-rtsp_transport', 'tcp', '-fflags', 'nobuffer', '-flags', 'low_delay']
        cmd += ['-i', str(self.uri), '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
        return cmd
```

```python
    def _open(self):
        self._stop.clear()
        self._eof = False
        self._drain()
        try:
            self._proc = subprocess.Popen(self._cmd_builder(), stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, bufsize=0)
        except Exception as e:
            if self.logger:
                self.logger.warning(f'ffmpeg 启动失败: {e}')
            self._proc = None
            return False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return True
```

- `info` 来自 `probe_stream`；宽高用 `info['width']` / `info['height']` 直接取（缺了会 `KeyError`，
  但 `VideoReader` 只在探测成功时才构造）；
- `frame_bytes = width * height * 3`：`bgr24` 每像素 3 字节，这是 5.5 循环的上界。1280×720 一帧约 2.6 MiB；
- `live = is_live_uri(uri)`：决定丢帧策略、是否启用读超时、超时后是否重启。注意它**不看 `mode`**，
  所以用 `mode='ffmpeg'` 打开文件时，这个实例仍然是"非直播"语义；
- `read_timeout = float(read_timeout or 0)`：`0` / `None` 都归一成 0；在 `read()` 里表示不设队列超时
  （重启后等首帧仍会用 `max(..., 1)`，见 5.8）；
- `buffer_size = max(int(buffer_size or 1), 1)`：队列至少 1。若允许 0，`queue.Queue(0)` 是**无限队列**，
  直播会无限堆积、延迟失控；
- `_cmd_builder` 允许注入假命令，单测用它替代真实 ffmpeg（`_writer_cmd` 生成一段写 rawvideo 的 Python）；
- ffmpeg 参数：
  - `-hide_banner -loglevel error`：压掉版本横幅与进度日志；
  - 只有 RTSP 才加 `-rtsp_transport tcp`（UDP 丢包会导致花屏/断流）与 `-fflags nobuffer -flags low_delay`
    （减少解码侧缓冲）；
  - `-an`：不需要音轨，少一路解码；
  - `-f rawvideo -pix_fmt bgr24 pipe:1`：向 stdout 输出裸帧；
- 为什么是 rawvideo 而不是让 ffmpeg 输出 mp4/ts：Python 侧不需要容器解析与解码器，读固定长度字节就是一张 BGR 帧；
  代价是数据量大、**没有帧边界**，必须靠 5.5 的循环凑满一帧；
- 为什么是 `bgr24`：OpenCV/Ultralytics 默认 BGR，省一次 `cvtColor`；
- `_open()` 的顺序很重要：`_stop.clear()`、`_eof=False`、`_drain()` 先清掉上一路流的残帧与哨兵，再起进程。
  不清空的话，重启后可能把旧流的最后一帧当成新流首帧，或者立刻被旧哨兵误判为"又断了"；
- `bufsize=0`：不经过 Python 用户态缓冲，`readinto` 直接对管道读；
- `stderr=subprocess.DEVNULL`：ffmpeg 的错误日志不进程序输出。代价是运行时错误（解码报错等）看不到，
  只剩启动失败 warning 与"僵死"日志；
- 启动失败只 warning 并返回 `False`，**不抛异常**；成功后起一个 daemon 读线程 `_pump`。

### 5.5 `_read_frame()`：管道短读与整帧（第 192–202 行，逐行）

```python
    def _read_frame(self, proc):
        """从管道读满一整帧。管道 read() 会短读，必须循环 readinto 到 frame_bytes。"""
        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        got = 0
        while got < self.frame_bytes:
            n = proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        return np.frombuffer(buf, np.uint8).reshape((self.height, self.width, 3))
```

**为什么必须循环到 `frame_bytes`（而不是读一次）**：

- 管道是**字节流，没有帧边界**。内核管道缓冲区通常只有几十 KiB，而 1280×720×3 的一帧约 2.6 MiB，
  一次 `read`/`readinto` 不可能把它全部交回来；
- `readinto` 只保证"把当前可读的字节写进缓冲区，返回写入字节数"，**短读是正常行为**，不是错误；
- 如果只读一次就 `reshape`，未填满的 `bytearray` 剩余部分是 0：画面下半部变黑；更糟的是下一帧会从**半帧处**开始，
  之后每帧都错位，而且 `cv2` / numpy 不会报错 —— 表现为持续花屏或画面撕裂；
- 循环用 `got` 记录已读字节，`view[got:]` 是 `memoryview` 切片，把下一段直接写进同一个 `bytearray`，
  全程零拷贝、不拼接临时 `bytes`；
- `readinto` 返回 0 表示对端关闭（ffmpeg 退出 / 文件读完 / 被杀），返回 `None`，`_pump` 随即退出并投递哨兵；
- 读满后 `np.frombuffer(buf, np.uint8)` 与 `reshape((height, width, 3))` 都是零拷贝视图，
  与 rawvideo 的逐行、BGR 排列一致；每帧都新建 `buf`，帧之间不共享内存；
- **已知边界**：`height/width` 只在构造时从 ffprobe 取一次。如果流中途换分辨率，`frame_bytes` 就错了，
  字节会持续错位，代码没有逐帧校验或重新探测 —— 需要重新创建 reader。

### 5.6 _pump() / _put() / 哨兵：有界队列与丢帧（第 204–242 行，逐行）

```python
    def _pump(self):
        proc = self._proc
        try:
            while not self._stop.is_set():
                frame = self._read_frame(proc)
                if frame is None:
                    break
                self._put(frame)
        except Exception:
            pass
        self._send_sentinel()
```

```python
    def _put(self, frame):
        """直播源队满丢最旧帧（保低延迟）；文件源阻塞等待（不丢帧）。"""
        while True:
            try:
                self._q.put_nowait(frame)
                return
            except queue.Full:
                if not self.live:
                    self._q.put(frame)
                    return
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
```

```python
    def _send_sentinel(self):
        """投递 None 唤醒阻塞中的 read()：表示"这一路流没了"。"""
        while True:
            try:
                self._q.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
```

- `_pump` 是唯一的帧生产者，daemon 线程；`_stop` 只在线程边界检查，正在阻塞的 `readinto` 靠
  `_close_proc()` 的 `terminate()` 打断；
- `except Exception: pass`：进程被 `terminate` 后 `readinto` 可能抛 `OSError`，daemon 线程里的异常不会回传主线程，
  统一用哨兵通知消费者；
- `_send_sentinel()` 在 `try/except` 之后无条件执行：正常 EOF 与异常退出都会让 `read()` 看到 `None`；
- **队列满时为什么"直播丢最旧、文件不丢"**：
  - 直播：观众要的是"现在"，队列尾部才是最新帧。丢头部能让队列里始终只有最近 `buffer_size` 帧；
    如果改成阻塞生产端，ffmpeg 的管道会被反压，延迟会随运行时间越积越大，`buffer_size` 就失去意义。
    每丢一帧 `dropped += 1`，最终进 summary 的 `frames_dropped`；
  - 文件：离线、可重复，帧序必须完整。丢帧会让结果视频时长、逐帧统计与源不一致；阻塞只是让 ffmpeg 等着，
    消费者取走一帧再生产，最终一帧不少；
- 丢最旧而不是丢最新：若丢最新，队列里会一直留着旧帧，延迟不降反升；
- `except queue.Empty: pass`：`put_nowait` 判满与 `get_nowait` 之间，消费者可能刚好把帧取走；
  这时 `get_nowait` 抛 `Empty`，重试 `put_nowait` 即可，不应计数；
- 哨兵同样可能遇到满队列：它选择"挤掉一帧再放 `None`"，**保证 `read()` 一定被唤醒**。
  被哨兵挤掉的帧不计入 `dropped`（它的语义是"流结束了"，不是"消费不过来"）；
- `_drain()`（第 244–249 行）在每次 `_open()` 前清空队列，理由见 5.4。

### 5.7 read() 与 _eof：文件读完为什么不能只靠哨兵（第 277–293 行，逐行）

```python
    def read(self):
        if self._eof:                       # 文件已读到结尾：直接返回，不再阻塞
            return None
        timeout = self.read_timeout if self.live else None
        try:
            item = self._q.get(timeout=timeout) if timeout else self._q.get()
        except queue.Empty:
            item = None
            if self.logger:
                self.logger.warning(f'超过 {timeout:.0f}s 没收到帧，判定直播流僵死: {self.uri}')
        if item is not None:
            self._fails = 0
            return item
        if not self.live:
            self._eof = True
            return None
        return self._restart()
```

**为什么需要 `_eof`**：

- 文件读完后，`_pump` 线程已经退出，只留下一个 `None` 哨兵。读到哨兵的那次 `read()` 取到它，返回 `None`；
- 但文件源没有超时（`timeout=None`），`self._q.get()` 是**永久阻塞**的。此时如果再调一次 `read()`，
  队列里既没有帧、也没有哨兵，就会永远卡住；
- `_eof` 把"已经结束"缓存成一个可重复查询的终态：后续 `read()` 立刻返回 `None`，
  对"幂等调用/异常重试/手动冒烟循环"都安全；
- 反过来，**直播不能缓存成终态**：直播的 `None`（哨兵或超时）语义是"这一路没了，要重启"，
  如果也置 `_eof`，断流一次就再也不会重连。所以 `_eof` 只在 `not self.live` 时置位。

**超时与僵死判定**：

- `timeout = read_timeout if live else None`：看门狗只对直播生效。文件源靠 EOF，不应被"读得慢"误杀；
- `self._q.get(timeout=timeout) if timeout else self._q.get()`：`read_timeout=0` 时走无超时分支 ——
  配置成 0 等于**关闭看门狗**，不是"立刻超时"；
- `queue.Empty` 只说明"队列在 `read_timeout` 秒内没有新帧"，不是进程存活检查。所以：
  - 消费慢（队列满）时 `get` 会立刻返回，不会误判；
  - 只有生产端**完全不再供帧**才会 `Empty`，这才是"僵死"；日志写的是"超过 Ns 没收到帧"；
- 判定僵死后并不在这一行杀进程，真正的 `terminate` 在 `_restart()` 里；
- 取到帧就 `self._fails = 0`：任何一帧成功都清零连续失败计数；
- `item is None` 有两种来源 —— 超时、或 `_pump` 的哨兵（ffmpeg 退出）。对直播，两者都走 `_restart()`；
  对文件，都归为"结束"，置 `_eof` 并返回 `None`；
- 注意哨兵 `None` 是**数据**，不是队列空的信号：`read()` 无法区分"超时"和"进程退出"，也不需要区分，
  因为两者的处置相同。

### 5.8 _restart()：重启与重试语义（第 295–312 行，逐行）

```python
    def _restart(self):
        limit = self.reconnect if self.reconnect is not None else 0
        self._close_proc()
        while limit < 0 or self._fails < limit:
            self._fails += 1
            self.reconnects += 1
            time.sleep(min(self.retry_delay * self._fails, self.retry_delay * 4))
            if not self._open():
                continue
            try:
                item = self._q.get(timeout=max(self.read_timeout, 1))
            except queue.Empty:
                item = None
            if item is not None:
                self._fails = 0
                return item
            self._close_proc()
        return None
```

- 进入前先 `_close_proc()`（第 251–270 行）：`_stop.set()` → `terminate()` → `wait(2)` → 失败则 `kill()` →
  关 `stdout` → `join(2)` → 清空 `_proc/_thread`。顺序保证读线程不会在旧进程还活着时继续往队列塞帧，
  也避免僵尸进程与线程泄漏；
- `limit = reconnect if reconnect is not None else 0`：负数表示**无限重试**，0 表示一次都不试；
- 每次尝试：`_fails += 1`（连续失败数）、`reconnects += 1`（累计尝试数，进报告）、
  `sleep(min(retry_delay * _fails, retry_delay * 4))`；
- 退避是线性的、封顶 4 倍：默认 `retry_delay=1.0` 时依次等 1、2、3、4、4… 秒。
  固定 1 秒会在长时间断流时持续打服务端，封顶则避免第 100 次等 100 秒；
- `_open()` 失败 `continue`，这次尝试照样消耗一个额度；
- 重连成功后最多等 `max(read_timeout, 1)` 秒要首帧；只有真的拿到帧才 `_fails = 0` 并返回，
  拿不到就再关掉进程进入下一轮（避免"进程起来了但永远不出帧"被当成成功）；
- 额度耗尽返回 `None`，主循环 `break`，统计照常收尾；
- **语义**：`run.reconnect=3` 是"**连续**失败 3 次就放弃"，不是"总共只能重连 3 次"。成功一帧后额度恢复
  （v0.5.0 P0-3 修的就是"用光额度后永久不再重连"）；
- `read_timeout` 在这里被复用为"重启后等首帧"的超时。最坏阻塞大致是
  `reconnect × (max(read_timeout, 1) + 退避)`，默认配置下是 3 × (10 + 1~4) 秒量级；
- 与 `OpenCVReader` 的差别：这里会先关旧进程、有退避、有"等首帧"确认；OpenCV 后端是固定延迟后直接重开。

### 5.9 VideoReader 门面（第 318–376 行，逐行）

```python
    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0, mode='auto',
                 read_timeout=10, logger=None):
        self.mode = mode or 'auto'
        backend = choose_backend(uri, self.mode)
        self.impl = None
        if backend == 'ffmpeg':
            info = probe_stream(uri)
            if info:
                try:
                    self.impl = FFmpegReader(uri, info, (camera or {}).get('buffer_size', 1),
                                             read_timeout, reconnect, retry_delay, logger)
                except Exception as e:
                    if logger:
                        logger.warning(f'FFmpeg 后端启动失败，回退 OpenCV: {e}')
            elif logger:
                logger.warning('ffprobe 探测失败，回退 OpenCV 读取（直播延迟可能累积）')
        if self.impl is None:
            self.impl = OpenCVReader(uri, camera, reconnect, retry_delay, logger)
```

- `mode or 'auto'`：`None` / 空串都归为 auto；
- 选到 ffmpeg 后先探测：探测成功才构造 `FFmpegReader`；队列长度取 `(camera or {}).get('buffer_size', 1)`；
  后面四个位置参数依次是 `read_timeout`、`reconnect`、`retry_delay`、`logger`，与 5.4 的签名一一对应，别调换；
- 探测失败只 warning，然后落到 `OpenCVReader`：宁可用有延迟的 cv2，也不要直接失败；
- `FFmpegReader` 构造抛异常（比如 `info` 缺字段）同样回退 OpenCV；
- 属性全部透传：`backend`、`opened`、`fps`、`total`、`width`、`height`、`reconnects`、`dropped`；
  `read()/release()` 直接转发；
- 为什么要有这层门面：`sources.VideoSource` 与 `pipeline` 只依赖 `read/release` 和几个只读属性，
  加后端不用动主循环 —— `sources.py` 的模块注释写的是"加新源只需在 `reader.py` 加一个后端"；
- `__main__`（第 379–388 行）是手动冒烟入口：`python -m yolo_app.reader rtsp://...`，
  读到 `None` 后打印帧数、后端与重连次数。

### 5.10 与 sources / pipeline 的接线

`sources.py` 只做透传（第 35–39 行）：

```python
    def __init__(self, uri, camera=None, reconnect=0, retry_delay=1.0,
                 reader='auto', read_timeout=10, logger=None):
        self.uri = int(uri) if is_camera(uri) else uri
        self.reader = VideoReader(self.uri, camera, reconnect, retry_delay,
                                  mode=reader, read_timeout=read_timeout, logger=logger)
```

`pipeline.process()` 打开源（第 57–59 行）：

```python
    src = VideoSource(source, cfg['camera'], run_cfg['reconnect'], run_cfg['retry_delay'],
                      reader=run_cfg.get('reader', 'auto'),
                      read_timeout=run_cfg.get('read_timeout', 10), logger=logger)
```

统计落盘（第 266–267 行）：

```python
        'reconnects': src.reconnects, 'reader': src.backend,
        'frames_dropped': src.dropped,
```

- `run.reader` 默认 `auto`、`run.read_timeout` 默认 10，均定义在 `config.DEFAULTS['run']`；
- `pipeline` 在 `if not src.opened` 时抛出带排查提示的 `RuntimeError`；
  日志里只有非 OpenCV 后端才标"读取 ffmpeg"（第 158 行），避免默认路径刷屏；
- 报告里三个读取指标口径不同：`reader` 是后端名，`reconnects` 是累计尝试次数（两个后端都记），
  `frames_dropped` 只有 FFmpeg 直播队列丢帧才会增长；OpenCV 的 `dropped` 恒 0。

### 5.11 常见坑与取舍

- **短读**：不要用一次 `read(frame_bytes)` 替换 5.5 的循环，否则会花屏且错误持续累积；
- **显式 `run.reader: ffmpeg` 不校验二进制**：`choose_backend` 直接返回 `ffmpeg`，是否存在由 `probe_stream` 决定；
  若 ffprobe 可用而 ffmpeg 缺失，`FFmpegReader._open()` 只记 warning 并返回 `False`（不抛），
  `opened` 又恒为 `True`，pipeline 的"无法打开视频源"分支不会触发。直播源还能靠 `read_timeout` + `reconnect`
  兜底返回 `None`；**文件源没有超时，会一直阻塞在 `read()` 的队列上**（启动失败时没有生产线程），这是已知边界；
- **`opened` 恒真**：FFmpeg 后端的 `opened` 直接 `return True`，启动失败不会在打开阶段暴露，要在 `read()` 阶段才看得到；
- **分辨率中途变化**：`frame_bytes` 不会重算，字节持续错位，只能重建 reader；
- **stderr 被丢弃**：运行时 ffmpeg 报错不可见；排查时临时把 `stderr` 指向终端/文件，别长期开着刷日志；
- **`buffer_size` 是延迟上限**：调大能抗抖动，但直播延迟也变大；文件源调小只影响吞吐，不影响正确性（阻塞不丢帧）；
- **`read_timeout` 只对直播生效**：FFmpeg 后端的文件源配了也没用，EOF 由哨兵 + `_eof` 处理；
  OpenCV 后端文件读到结尾直接返回 `None`；
- **必须 `release()`**：`_close_proc()` 会 `terminate → wait(2) → kill → join(2)`；不调用会留下 ffmpeg 子进程与读线程
  （pipeline 在 `finally` 里保证调用）；
- **单消费者**：队列、`_fails`、`_eof` 都没有加锁，不是为多线程同时 `read()` 设计的；
- **HTTP(S) 也走直播语义**：`auto` 会给 `http(s)://` 选 FFmpeg；服务端不支持时探测失败会回退 OpenCV，
  但日志里只有一条 warning，别以为它一定在用 FFmpeg。

---

## 6. yolo_app/analytics.py ★后处理（逐行）

这个文件**不依赖模型、也不依赖 OpenCV 顶层**（cv2/numpy 只在 draw() 里延迟 import），
好处是纯几何逻辑可以脱离推理单独测试（tests 里就是用 `FakeBoxes` 测的）。

### 6.1 点与线的几何（第 19–36 行，逐行）

```python
def _inside(poly, pt):
    """射线法：点是否在多边形内。poly=[[x,y],...]"""          # 从该点向右发射一条射线
    x, y = pt
    inside = False                                            # 初始假设在外面
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]                                      # 多边形第 i 个顶点
        x2, y2 = poly[(i + 1) % n]                            # 下一个顶点（% n 让最后一个连回第一个）
        if (y1 > y) != (y2 > y):                              # 这条边**跨越**了射线的 y（否则不相交）
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)       # 线性插值求交点 x 坐标
            if x < xi:                                        # 交点在右侧 → 穿越一次边界
                inside = not inside                           # 奇数次=在内，偶数次=在外
    return inside
```

- 为什么用中心点判定而不是"框与多边形求交"：实现简单、无歧义，业务上"目标中心在区域内"通常就是想要的语义；
- 复杂度 O(顶点数)，每帧每框几微秒，可忽略。

```python
def _side(line, pt):
    (x1, y1), (x2, y2) = line                                 # 线的两个端点
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)  # 二维叉积：正负号 = 在线的哪一侧
```

- 叉积是判断"点在直线哪侧"的标准做法；结果为 0 表示正好在线上，代码里归到 + 侧（`>= 0`）。

### 6.2 RegionFilter（第 39–63 行，逐行）

```python
    def __init__(self, roi=None):
        self.roi = roi or []                                  # 没配置 → 空列表 → enabled=False
    @property
    def enabled(self):
        return bool(self.roi)                                 # 日志/分支都用它判断，避免到处写 len()>0
```

```python
    def keep(self, res):
        if not self.enabled or res.boxes is None or not len(res.boxes):
            return res                                        # 没过ROI/没检测到 → 原样返回（链式调用不中断）
        xyxy = res.boxes.xyxy.tolist()                        # 一次性取成 python list，比逐个索引快
        idx = [i for i, b in enumerate(xyxy) if _inside(self.roi, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))]
        res.boxes = res.boxes[idx]                            # ★就地替换：后续绘制/统计只看到保留的框
        return res
```

- `(b[0]+b[2])/2, (b[1]+b[3])/2` 就是框中心；
- `res.boxes[idx]` 用整数列表索引，Ultralytics 的 Boxes 支持这种"花式索引"，一次筛完；
- **副作用是刻意设计的**：只过滤一次，全链路生效，不需要在每个下游模块再判一次。

```python
    def draw(self, frame, color=(0, 255, 255)):
        import cv2, numpy as np                               # 延迟 import：纯逻辑部分不依赖图像库
        if self.enabled:
            cv2.polylines(frame, [np.array(self.roi, dtype=np.int32)], True, color, 2)
        return frame                                          # True = 闭合多边形；画在原地，无返回值依赖
```

### 6.3 LineCounter（第 66–123 行，逐行）

```python
    def __init__(self, line, classes=None, names=None):
        self.line = line or []
        self.classes = set(classes or [])                     # 只统计这些类别（空=全部）
        self.prev = {}                                        # ★ track_id -> 上一帧所在侧（+1/-1）
        self.total = {"in": 0, "out": 0}
        self.by_class = defaultdict(lambda: {"in": 0, "out": 0})   # 分类别明细
```

```python
    def update(self, res, names):
        if not self.enabled or res.boxes is None or not len(res.boxes):
            return None
        boxes = res.boxes
        ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)   # 没开跟踪时用 None 占位
        xyxy = boxes.xyxy.tolist()
        for i, tid in enumerate(ids):
            cid = int(boxes.cls.tolist()[i])                  # 类别 id
            if self.classes and cid not in self.classes:
                continue                                      # 不在关注类别 → 跳过
            cx = (xyxy[i][0] + xyxy[i][2]) / 2                # 框中心 x
            cy = (xyxy[i][1] + xyxy[i][3]) / 2                # 框中心 y
            s = 1 if _side(self.line, (cx, cy)) >= 0 else -1  # 归一成 +1/-1，便于比较
            if tid is None:
                continue                                      # 没有 track_id 就无法判"穿越"，只能跳过
            tid = int(tid)
            prev = self.prev.get(tid)                         # 上一帧所在侧（首次为 None）
            self.prev[tid] = s                                # 更新为当前侧
            if prev is not None and prev != s:                # ★ 侧别翻转 = 发生了一次穿越
                direction = "in" if prev < 0 and s > 0 else "out"   # - 侧→+ 侧为 in，反之为 out
                name = names.get(cid, str(cid))
                self.total[direction] += 1
                self.by_class[name][direction] += 1
                crossed.append({"track_id": tid, "class": name, "direction": direction})
        return crossed or None                                # 无事件返回 None，调用方 if 判断更自然
```

- **这是全项目最依赖跟踪质量的一段**：track_id 一断，同一目标会被当成新目标，可能重复计数；
- 想更稳：降低 `--stride`（别跳帧）、换 `botsort.yaml`、或提高 `--conf` 减少抖动。

```python
    def draw(self, frame, color=(0, 0, 255)):
        (x1, y1), (x2, y2) = self.line
        cv2.line(frame, ...)                                  # 画线
        cv2.putText(frame, "in %d / out %d" % (self.total["in"], self.total["out"]), ...)  # 实时计数
    def summary(self):
        return {"line_total": dict(self.total), "line_by_class": {k: dict(v) for k, v in self.by_class.items()}}
```

- `summary()` 返回**普通 dict**（不是 defaultdict），否则 json.dumps 序列化会失败。

---

## 7. yolo_app/outputs.py ★结果落地（逐行）

### 7.1 VideoSink（视频写入，逐行）

```python
    def __init__(self, path, fps, segment_seconds=0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)   # 目录不存在就建（含多级）
        self.fps = fps or 25.0                                # 帧率兜底
        self.segment_seconds = float(segment_seconds or 0)
        self.writer = None                                    # ★ 先不建！等第一帧到了再建
        self.parts = []
        self._frames_in_part = 0
        self._part = 1
```

- **为什么延迟创建 writer**：VideoWriter 必须在构造时给宽高，而 RTSP/摄像头在打开时经常报不出分辨率，只能等第一帧 `frame.shape` 才知道。

```python
    def _current_path(self):
        if self._part == 1:
            return self.path                                  # 第一段用原始文件名
        return self.path.with_name(f'{self.path.stem}_part{self._part:02d}{self.path.suffix}')
```

- `part02` 这种补零命名能让文件在目录里自然排序。

```python
    def _open_writer(self, path, shape):
        h, w = shape[:2]                                      # shape = (高, 宽, 通道)
        for codec in ('avc1', 'mp4v'):                        # ★ 先试 H.264
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if vw.isOpened():
                self.codec = codec
                return vw
        raise RuntimeError(f'无法创建视频写入器: {path}')      # 两种都不行才报错
```

- `VideoWriter_fourcc(*'avc1')` = `VideoWriter_fourcc('a','v','c','1')`，四字符编码；
- H.264 编码在 Chromium 内核（VS Code 内置播放器/浏览器）能放，`mp4v` 会黑屏 —— 这是必须回退而不能只用 mp4v 的原因。

```python
    def write(self, frame):
        limit = self.fps * self.segment_seconds               # 每段允许的帧数
        if self.writer is not None and self.segment_seconds and self._frames_in_part >= limit:
            self.writer.release()                             # 到点：关掉当前段
            self.writer = None
            self._part += 1                                   # 段号 +1
            self._frames_in_part = 0
        if self.writer is None:
            path = self._current_path()
            self.writer = self._open_writer(path, frame.shape)
            self.parts.append(str(path))                      # 记录产物路径，写进 JSON
        self.writer.write(frame)
        self._frames_in_part += 1
```

```python
    def close(self):
        if self.writer is not None:
            self.writer.release()                             # ★ 必须调用：否则 mp4 索引没写完，文件损坏
            self.writer = None
```

### 7.2 EventSink（告警抓拍，逐行）

```python
    def __init__(self, out_dir, classes=None, min_conf=0.6, cooldown_s=2.0):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.classes = {int(c) for c in (classes or [])}      # 只关心这些类别（空=全部）
        self.min_conf, self.cooldown = float(min_conf), float(cooldown_s)
        self._last = 0.0                                      # 上次抓拍的时间戳
        self.events = []                                      # 事件明细，最终进 JSON
```

```python
    def check(self, res, names, frame_no, annotated):
        boxes = res.boxes
        if boxes is None or not len(boxes):
            return False
        ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        for cid, cf, tid in zip(boxes.cls.tolist(), boxes.conf.tolist(), ids):   # 三者一一对应
            if cf < self.min_conf or (self.classes and int(cid) not in self.classes):
                continue                                      # 置信度不够 / 类别不关心 → 跳过
            now = time.time()
            if now - self._last < self.cooldown:
                continue                                      # ★ 冷却中：同一波目标不重复抓拍
            self._last = now
            name = names[int(cid)]
            p = self.dir / f'{time.strftime("%Y%m%d_%H%M%S")}_f{frame_no}_{name}.jpg'   # 文件名自解释
            cv2.imwrite(str(p), annotated)                    # ★ 同步写盘：会拖慢主循环，冷却别设太小
            self.events.append({...})
            return True                                       # 一帧只抓一张，立刻返回
```

### 7.3 cleanup_old（留存清理，逐行）

```python
def cleanup_old(directory, keep_days=0, max_files=0, patterns=('*.mp4', '*.jpg')):
    d = Path(directory)
    if not d.exists() or (not keep_days and not max_files):
        return 0                                              # 没配策略就什么都不做
    files = sorted((f for p in patterns for f in d.rglob(p)), key=lambda f: f.stat().st_mtime)  # 按修改时间排序（旧→新）
    if keep_days:
        cutoff = time.time() - keep_days * 86400              # 截止时间戳
        for f in list(files):                                 # list() 拷贝：循环中要删元素
            if f.stat().st_mtime < cutoff:
                f.unlink(); files.remove(f); removed += 1
    if max_files and len(files) > max_files:
        for f in files[:len(files) - max_files]:              # 只保留最新的 max_files 个
            f.unlink(); removed += 1
    return removed
```

- 先按天删、再按数量删，两个策略可以叠加；
- `rglob` 递归，所以 `outputs/events/` 里的抓拍图也在清理范围内。

---

## 8. yolo_app/config.py 配置（逐行）

```python
DEFAULTS = { ... }                                            # 所有可配置项与默认值（唯一真源）
def deep_merge(base, patch):
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)                            # 两边都是 dict → 递归合并（保留未覆盖的兄弟字段）
        else:
            base[k] = v                                       # 否则直接覆盖
    return base
```

- **递归合并是关键**：`--conf 0.5` 只覆盖 `model.conf`，不会把 `model.weight`/`imgsz` 一起抹掉；
- 如果写成 `base.update(patch)`，浅层 dict 会整体替换，配置就丢字段了。

```python
def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)                             # ★ 深拷贝：否则会污染模块级 DEFAULTS
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'配置文件不存在: {path}')  # 明确报错，别静默用默认值
        deep_merge(cfg, yaml.safe_load(p.read_text(encoding='utf-8')) or {})   # 空文件 → {} 
    deep_merge(cfg, {k: v for k, v in (overrides or {}).items() if v})         # 只合并"有内容"的覆盖
    return cfg
```

- `safe_load` 而不是 `load`：不执行 YAML 里的任意 Python 对象，安全；
- `if v` 过滤掉空 dict/None，避免把配置无意义地覆盖成空。

---

## 9. yolo_app/alerts.py 告警（逐行）

```python
WECOM = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="       # 企业微信机器人固定前缀
FEISHU = "https://open.feishu.cn/open-apis/bot/v2/hook/"

    def __init__(self, url="", kind="generic", timeout=5, logger=None):
        self.sent, self.failed = 0, 0                                 # 成功/失败计数，进最终报告
        if url and not url.startswith("http"):                        # 只传了 key → 自动补全前缀
            if kind == "wecom":  self.url = WECOM + url
            elif kind == "feishu": self.url = FEISHU + url
```

```python
    def _payload(self, text):
        if self.kind == "wecom":  return {"msgtype": "text", "text": {"content": text}}    # 企业微信格式
        if self.kind == "feishu": return {"msg_type": "text", "content": {"text": text}}   # 飞书格式
        return {"text": text}                                                              # 通用 JSON
```

```python
    def send(self, text):
        if not self.enabled:
            return False                                             # 没配 webhook：什么都不做（不是错误）
        try:
            req = urllib.request.Request(self.url, data=json.dumps(...).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                r.read()                                             # 必须读掉响应体，否则连接不释放
            self.sent += 1
            return True
        except Exception as e:                                       # ★ 吞掉所有异常
            self.failed += 1
            if self.logger: self.logger.warning("告警发送失败: %s" % e)
            return False
```

- 用 **urllib 而不是 requests**：标准库就够，少一个依赖；
- 用 try/except 包住一切：**告警是旁路功能，绝不能因为它挂掉而中断检测**。

---

## 10. yolo_app/metrics.py 指标（逐行）

```python
START = time.time()                                               # 模块导入时刻 = 进程启动时刻
def prometheus_text(stats, job='yolo_app'):
    s = stats or {}
    lines = [
        '# HELP yolo_up 应用是否在跑', '# TYPE yolo_up gauge', 'yolo_up 1',
        ...
        f'yolo_frames_inferred_total {int(s.get("frames", 0))}',
        f'yolo_fps {float(s.get("fps", 0))}',
        f'yolo_line_crossings_total{{direction="in"}} {...}',      # {{ }} 是 f-string 里的字面大括号
    ]
    for k, v in (s.get('per_class') or {}).items():
        lines.append(f'yolo_detections_total{{class="{k}",job="{job}"}} {int(v)}')   # 带标签的指标
    return '\n'.join(lines) + '\n'                                   # 每行一个指标，末尾换行
```

- 文本暴露格式就是"每行一个 `指标名{标签} 值`"，加 `# HELP`/`# TYPE` 注释即可被 Prometheus 抓取；
- 所有值取自 `on_stats` 回调传进来的实时统计 —— 面板与指标共用同一份数据。

---

## 11. yolo_app/logging_setup.py 日志（逐行）

```python
def get_logger(level='INFO', log_file=None):
    logger = logging.getLogger('yolo_app')                # 具名 logger：全项目共用一个名字
    if logger.handlers:
        return logger                                     # ★ 已有 handler 就直接返回（防止重复输出两遍）
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))   # 非法级别 → 退回 INFO
    sh = logging.StreamHandler(sys.stdout)                # 输出到 stdout（不是 stderr，方便重定向收集）
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')   # utf-8：日志里有中文
        logger.addHandler(fh)
    logger.propagate = False                              # 不向 root logger 传播，避免第三方库带出重复日志
    return logger
```

- 多进程场景每个子进程各自调用一次，互不干扰（`run_job` 里就是这么做的）。
---

## 12. yolo_app/cli.py 命令行（按函数逐行）

这是最长的一个文件（约 290 行），但结构很简单：**参数声明 + 参数→配置的翻译 + 分发**，没有业务逻辑。

### 12.1 parse_classes / parse_points（参数值解析，逐行）

```python
def parse_classes(s):
    if not s:
        return None                                  # 没传 → None（表示"不覆盖配置"）
    out = []
    for c in str(s).split(','):
        c = c.strip()
        if c:
            out.append(int(c) if c.isdigit() else c)  # 数字转 int，其余保留字符串（后续按类别名解析）
    return out or None
```

```python
def parse_points(s, n):
    if not s:
        return None
    nums = [float(v) for v in re.split(r'[,;\s]+', str(s).strip()) if v]   # 逗号/分号/空格都当分隔符
    if len(nums) % 2:
        raise ValueError(f'坐标个数必须是偶数（成对出现）: {s}')              # 早报错，别等到运行时才算错
    pts = [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]          # [x,y] 两两成组
    if n and len(pts) != n:
        raise ValueError(f'需要 {n} 个点，实际 {len(pts)} 个: {s}')
    return pts
```

- 这样 `--roi "10,10 20,10 20,20"` 和 `--roi "10,10,20,10,20,20"` 都能用（用户怎么写都行，解析层兜住）；
- `n=2` 用于 `--line`（必须正好两个点）。

### 12.2 build_overrides()（命令行 → 配置覆盖，逐行）

```python
    ov = {'model': {}, 'run': {}, 'output': {}, ...}          # 与 config 的分层结构一一对应
    m, run, out = ov['model'], ov['run'], ov['output']
    pairs = [
        (getattr(a, 'model', None), m, 'weights'),            # (参数值, 目标字典, 键名)
        (getattr(a, 'conf', None),  m, 'conf'),
        (getattr(a, 'stride', None), run, 'stride'),
        ...
    ]
    for value, target, key in pairs:
        if value is not None:
            target[key] = value                               # ★ None=没传 → 不覆盖，配置里的值才保得住
```

- 用 `getattr(a, ..., None)` 是因为不同子命令参数集不同（`run` 有 `--source`，`check` 没有），缺属性不该报错；
- 表驱动写法把 20 多个"if a.x: ov[...] = a.x"压成一张表，加参数只加一行。

### 12.3 build_parser()（子命令定义）

```python
    sub = p.add_subparsers(dest='cmd', required=True)         # 必须给子命令，防止裸跑
    pr = sub.add_parser('run', ...);    add_analytics(pr); add_common(pr)
    pc = sub.add_parser('camera', ...); add_analytics(pc); add_common(pc)
```

- `add_common`/`add_analytics` 复用了参数定义：三个子命令共享同一套模型/业务参数，避免复制粘贴走样。

### 12.4 main()（分发，逐行）

```python
    ov = build_overrides(a)
    if a.cmd == 'camera':
        cam = {}
        for key, val in (('index', a.index), ('width', a.width), ...):
            if val is not None:
                cam[key] = val                                # 摄像头专属参数单独成组
        ov['camera'] = cam
    cfg = load_config(cfg_path, ov)                           # ★ 全项目只在这里读配置
    logger = get_logger(cfg['log']['level'], cfg['log'].get('file'))
```

```python
    if a.cmd == 'camera':
        cfg['run']['show'] = not a.no_show                    # 摄像头默认弹窗，--no-show 关掉
        cfg['output']['video'] = (bool(a.record) or cfg['camera']['record']) and not a.no_video
        sources = [cfg['camera']['index']]                    # 摄像头编号当"源"，与文件统一处理
    elif a.cmd == 'run':
        sources = list_videos(a.source)                       # 文件→[文件]，目录→[多个视频]
    elif a.cmd == 'serve':
        sources = [cfg['serve']['source']]
```

```python
    try:
        detector = Detector(cfg)                              # 模型只加载一次，多源复用
    except Exception as e:
        logger.error(f'模型加载失败: {e}')
        return 1                                              # 返回码：0 成功、1 失败（CI/脚本可判）
```

```python
    if a.cmd == 'check':
        r = detector.infer(np.zeros((detector.imgsz, detector.imgsz, 3), dtype=np.uint8))  # 空白图跑一次
        assert r.boxes is not None, '推理未返回 boxes，请检查 ultralytics / torch'
        logger.info(f'模型 {detector.weights} | 类别数 {len(detector.names)} | 设备 {detector.device_name} | ...')
        return 0
```

- `check` 是"最小可运行验证"：不需要视频、不依赖摄像头，就能证明 环境+模型+推理链路 是通的；
- 用 `np.zeros` 造一张全黑图，模型照样会返回空结果 —— 能返回就说明没坏。

```python
    if a.cmd == 'export':
        return _export(a, cfg, detector, logger)
    if a.cmd == 'serve':
        from .webapp import serve                             # 延迟 import：不用 Web 功能就不装 fastapi 也能跑
        serve(cfg, detector, logger, source=cfg['serve']['source'])
        return 0
```

```python
    removed = cleanup_old(cfg['output']['dir'], ...)          # 每次开跑前先清理旧产物
    workers = max(int(cfg['run']['workers'] or 1), 1)
    if workers > 1 and len(sources) > 1:                      # 只有多源才值得开进程池
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s in ex.map(run_job, [(src, cfg, False) for src in sources]):   # False = 子进程不弹窗
                summaries.append(s)
    else:
        for s in sources:
            try:
                summaries.append(process(s, cfg, detector, logger))
            except Exception as e:
                logger.error(f'{s} 处理失败: {e}')
                if len(sources) == 1:
                    return 1                                  # 单源失败要返回错误码；批量则继续跑完
```

- `ex.map` 保序，且会等全部任务完成；
- 单源失败返回 1、批量失败继续 —— 这是刻意的：批处理里一个坏文件不该让整批都失败。

### 12.5 _export()

```python
    path = YOLO(detector.weights).export(format=a.format, imgsz=..., half=..., dynamic=..., simplify=...)
```

- 导出会生成 `weights/yolo11n.onnx` 之类的文件；之后 `--model yolo11n.onnx` 即可直接用（权重定位逻辑同样生效）。

---

## 13. yolo_app/webapp.py Web 面板（按接口逐行）

### 13.1 线程模型

```
uvicorn 主线程（HTTP/WS）  ←→  AppState（一把锁 + 最新 JPEG + 最新统计）  ←→  检测线程（pipeline.process）
```

```python
class AppState:
    def __init__(self):
        self.lock = threading.Lock()      # 只保护下面三个字段
        self.jpeg = None                  # 最新一帧 JPEG 字节
        self.stats = {}                   # 最新统计
        self.error = None; self.finished = False
```

- **只保留最新帧**，不排队：实时画面天生可以丢帧，慢客户端不该拖慢检测；
- 加锁范围极小（读写变量），不会成为瓶颈。

### 13.2 create_app() 的回调（逐行）

```python
    def on_frame(anno):
        ok, buf = cv2.imencode('.jpg', anno, [int(cv2.IMWRITE_JPEG_QUALITY), quality])  # 编码成 JPEG
        if ok:
            with st.lock:
                st.jpeg = buf.tobytes()                                   # 覆盖旧帧（故意丢弃）
    def on_stats(s):
        with st.lock:
            st.stats = s
        if cfg.get('metrics', {}).get('file'):
            write_metrics_file(cfg['metrics']['file'], s)                 # 顺带落盘指标
```

```python
    def worker():
        try:
            from .pipeline import process
            process(source, cfg, detector, logger, show=False, record=record,
                    on_frame=on_frame, on_stats=on_stats)                 # ★ 复用主循环，不复制检测逻辑
        except Exception as e:
            logger.error(f'检测线程退出: {e}'); st.error = str(e)          # 源出错时面板仍能打开并显示原因
        finally:
            st.finished = True
    threading.Thread(target=worker, daemon=True).start()                  # daemon：主进程退出时自动结束
```

### 13.3 五个接口（逐行）

```python
    @app.get('/')
    def index():  return PAGE                                             # 内嵌 HTML：画面 + 统计表 + WS 客户端
    @app.get('/stats')
    def stats():  return JSONResponse({'source': ..., 'finished': ..., **st.stats})   # 一次性快照
    @app.get('/metrics')
    def metrics(): return prometheus_text(st.stats)                       # Prometheus 文本
    @app.get('/stream')
    def stream():
        def gen():
            while True:
                with st.lock: jpg = st.jpeg
                if jpg:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n...' + jpg   # MJPEG：每帧一段 multipart
                time.sleep(1 / 30)                                        # 限速 30fps，别把 CPU 喂满
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')
    @app.websocket('/ws')
    async def ws(sock):
        await sock.accept()
        while True:
            await sock.send_json({'source': ..., **st.stats})             # 每秒推一次统计
            await asyncio.sleep(1)
```

- MJPEG 是"最古老的直播方案"，但**浏览器原生支持**、零前端依赖，对本项目足够；
- WebSocket 只推**统计**（不是画面）：画面走 MJPEG，两条通道各司其职，实现最简单。

### 13.4 serve() 与依赖检查

```python
def _check_ws_support(logger):
    for mod in ('websockets', 'wsproto'):
        try: __import__(mod); return True
        except ImportError: continue
    logger.warning('未安装 websockets/wsproto，/ws 实时推送不可用；修复：pip install "uvicorn[standard]"')
```

- 这是踩过的坑：裸装 uvicorn 没有 WS 库时，`/ws` 会静默返回 404，前端只是"连不上"，极难排查；
- 现在启动时就会提示，并且 requirements 用的是 `uvicorn[standard]`。

---

## 14. 常见改法（照着改就行）

| 想做的事 | 改哪里 | 具体做法 |
| --- | --- | --- |
| 加一个命令行参数 | `config.py` → `cli.py` → `configs/default.yaml` | DEFAULTS 加键 → add_common 加参数 + build_overrides 加一行 → yaml 加注释 |
| 加一种后处理 | 新建 `analytics` 里的类 + `pipeline.process` 一行 | 参考 `RegionFilter.keep()` 的"就地改 res.boxes"或 `LineCounter.update()` 的"返回事件列表" |
| 加一种输出（数据库/MQTT） | `outputs.py` 新建类 + `pipeline` 里调用 | 实现 `write()`/`check()`，在写盘附近插两行 |
| 加一个 Web 接口 | `webapp.create_app` | 加 `@app.get` / `@app.websocket`，数据从 `st.stats` 取 |
| 加一个指标 | `metrics.prometheus_text` | 在 lines 里加一行 `f'yolo_xxx {value}'` |
| 换检测模型 | 不用改代码 | `--model yolo11s.pt` / `best.pt` / `best.onnx`，见 README「切换模型」 |

**验证习惯**：改完跑这四条就够——`ruff check yolo_app tests scripts`、`python -m unittest discover -s tests`、
`python -m yolo_app check`、`python -m yolo_app run --source data/test.mp4 --max-frames 20`。
