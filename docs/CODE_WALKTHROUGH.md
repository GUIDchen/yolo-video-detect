# 代码逐行详解

> 这份文档把每个主模块的代码**逐行拆开**讲清楚：这一行在干什么、为什么这么写、删掉会怎样。
> 源码里已经保留了模块级说明与关键注释，本文件是它们的展开版，适合第一次读这个项目时对照阅读。
>
> 阅读顺序建议：**pipeline → detector → sources → analytics → outputs → config → cli → 其余**。
> 带 ★ 的是核心链路，其余是辅助模块。

## 0. 先建立全局印象

一帧画面的完整旅程（括号里是文件）：

```
帧 ndarray ──► sources.VideoSource.read()        取帧（重连/摄像头设置都在这）
           ──► detector.Detector.infer(frame)    变成 Results（含 boxes/id/conf）
           ──► analytics.RegionFilter.keep()     ROI 外的框剔除
           ──► analytics.LineCounter.update()    统计越线
           ──► Results.plot()                   画框 → ndarray
           ──► outputs.EventSink.check()        命中就抓拍 + alerts 推送
           ──► outputs.VideoSink.write()        写结果视频（H.264/分段）
           ──► on_frame/on_stats 回调           给 Web 面板
           ──► outputs.write_json()             落统计报告
```

所有编排都在 `pipeline.process()` 一个函数里，共 ~130 行。**看懂它就等于看懂整个项目。**

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

## 3. yolo_app/sources.py ★取帧（逐行）

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

## 4. yolo_app/analytics.py ★后处理（逐行）

这个文件**不依赖模型、也不依赖 OpenCV 顶层**（cv2/numpy 只在 draw() 里延迟 import），
好处是纯几何逻辑可以脱离推理单独测试（tests 里就是用 `FakeBoxes` 测的）。

### 4.1 点与线的几何（第 19–36 行，逐行）

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

### 4.2 RegionFilter（第 39–63 行，逐行）

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

### 4.3 LineCounter（第 66–123 行，逐行）

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

## 5. yolo_app/outputs.py ★结果落地（逐行）

### 5.1 VideoSink（视频写入，逐行）

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

### 5.2 EventSink（告警抓拍，逐行）

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

### 5.3 cleanup_old（留存清理，逐行）

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

## 6. yolo_app/config.py 配置（逐行）

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

## 7. yolo_app/alerts.py 告警（逐行）

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

## 8. yolo_app/metrics.py 指标（逐行）

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

## 9. yolo_app/logging_setup.py 日志（逐行）

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

## 10. yolo_app/cli.py 命令行（按函数逐行）

这是最长的一个文件（约 290 行），但结构很简单：**参数声明 + 参数→配置的翻译 + 分发**，没有业务逻辑。

### 10.1 parse_classes / parse_points（参数值解析，逐行）

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

### 10.2 build_overrides()（命令行 → 配置覆盖，逐行）

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

### 10.3 build_parser()（子命令定义）

```python
    sub = p.add_subparsers(dest='cmd', required=True)         # 必须给子命令，防止裸跑
    pr = sub.add_parser('run', ...);    add_analytics(pr); add_common(pr)
    pc = sub.add_parser('camera', ...); add_analytics(pc); add_common(pc)
```

- `add_common`/`add_analytics` 复用了参数定义：三个子命令共享同一套模型/业务参数，避免复制粘贴走样。

### 10.4 main()（分发，逐行）

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

### 10.5 _export()

```python
    path = YOLO(detector.weights).export(format=a.format, imgsz=..., half=..., dynamic=..., simplify=...)
```

- 导出会生成 `weights/yolo11n.onnx` 之类的文件；之后 `--model yolo11n.onnx` 即可直接用（权重定位逻辑同样生效）。

---

## 11. yolo_app/webapp.py Web 面板（按接口逐行）

### 11.1 线程模型

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

### 11.2 create_app() 的回调（逐行）

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

### 11.3 五个接口（逐行）

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

### 11.4 serve() 与依赖检查

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

## 12. 常见改法（照着改就行）

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
