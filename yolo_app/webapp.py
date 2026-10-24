"""Web 面板：MJPEG 实时流 + WebSocket 统计推送 + Prometheus /metrics。

线程模型：uvicorn 主线程处理 HTTP/WS，检测跑在**后台线程**（复用 pipeline.process），
两者通过 AppState（一个 dict + 一把锁）交换"最新一张 JPEG"和"最新统计"，不做队列堆积——
实时画面只要最新帧，慢消费者自然丢帧，这是有意为之的取舍。

接口一览：
    GET  /         面板页（内嵌 HTML，实时画面 + 统计表）
    GET  /healthz  健康检查（供 Docker HEALTHCHECK / k8s liveness 用，加载的是内存状态，毫秒级）
    GET  /stream   MJPEG 流（multipart/x-mixed-replace），可直接嵌到别的前端
    GET  /stats    当前统计 JSON（带 seq 版本号）
    WS   /ws       统计增量推送：先发当前快照，之后 seq 变化才推，空闲每 15s 一条心跳
    GET  /metrics  Prometheus 文本指标

WS 消息协议（seq 是统计版本号，AppState 每收到一次 on_stats 就 +1）：
    {"type": "stats", "seq": n, "source": "...", "finished": false, "error": null, ...stats}
    {"type": "heartbeat", "seq": n}
断线重连用 ?since=<int> 告诉服务端"客户端已看到第几版"；服务端仍会先补一条当前快照，
避免客户端在没有新统计时首屏空白——真正的增量体现在"seq 不变不重推"，重复帧由客户端
按 seq 去重。心跳间隔默认 15s（serve.ws_heartbeat 可调，主要给测试用）。

鉴权：配置 serve.token 非空时，/stats /metrics /stream /ws 需要 `?token=xxx` 或 `X-Token` 头；
/ 与 /healthz 不鉴权（前者只是静态页，后者不含业务数据）。
注意：WebSocket 依赖 websockets/wsproto，裸装 uvicorn 会让 /ws 静默 404，
所以 serve() 启动前会调用 _check_ws_support() 给出明确提示。
"""
import asyncio
import threading
import time

import cv2

from .metrics import prometheus_text, write_metrics_file

PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>YOLO 实时检测面板</title>
<style>
 body{margin:0;background:#0f1115;color:#e6e6e6;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:12px 20px;border-bottom:1px solid #262b36;font-size:16px}
 .wrap{display:flex;gap:20px;padding:20px;flex-wrap:wrap}
 img{max-width:900px;width:100%;border-radius:8px;background:#000}
 .panel{min-width:260px;flex:1}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{padding:6px 8px;border-bottom:1px solid #262b36;text-align:left}
 .k{color:#8b93a7}.v{font-variant-numeric:tabular-nums}
</style></head><body>
<header>YOLO 实时检测面板 · <span id="fps" class="k">-</span></header>
<div class="wrap">
  <div style="flex:2 1 640px"><img id="stream" alt="实时流"></div>
  <div class="panel">
    <h3 style="margin-top:0">实时统计</h3>
    <table id="tbl"><tbody></tbody></table>
    <p class="k">WebSocket 推送 · <a id="metrics" href="/metrics" style="color:#4da3ff">/metrics</a> Prometheus · <a id="stats" href="/stats" style="color:#4da3ff">/stats</a> JSON</p>
  </div>
</div>
<script>
const $ = (s) => document.querySelector(s);
const token = new URLSearchParams(location.search).get('token') || '';
const q = token ? ('?token=' + encodeURIComponent(token)) : '';
$('#stream').src = '/stream' + q;
$('#metrics').href = '/metrics' + q;
$('#stats').href = '/stats' + q;
let lastSeq = -1;                 // 已渲染的统计版本号：相同 seq 不重绘
let ws = null, retry = null;

function render(d) {
  $('#fps').textContent = (d.fps || 0) + ' FPS';
  const rows = [['帧数', d.frames], ['FPS', d.fps], ['推理 ms/帧', d.infer_ms_avg], ['唯一目标', d.unique_tracks],
                ['告警抓拍', d.events], ['越线 in', (d.line || {}).in], ['越线 out', (d.line || {}).out]];
  for (const [k, v] of Object.entries(d.per_class || {})) rows.push(['检测 · ' + k, v]);
  $('#tbl').innerHTML = '<tbody>' + rows.map(r => '<tr><td class="k">' + r[0] + '</td><td class="v">' + (r[1] ?? 0) + '</td></tr>').join('') + '</tbody>';
}

function connect() {
  clearTimeout(retry);
  const p = new URLSearchParams(location.search);          // 带上面板 URL 里的 token
  if (lastSeq >= 0) p.set('since', lastSeq);               // 断线重连：告诉服务端已看到哪一版
  const qs = p.toString();
  const sock = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws' + (qs ? '?' + qs : ''));
  ws = sock;
  sock.onmessage = (e) => {
    if (sock !== ws) return;                               // 旧连接迟到的事件直接丢弃
    const d = JSON.parse(e.data);
    if (d.seq === lastSeq) return;                         // 同一 seq = 内容没变，不重绘
    lastSeq = d.seq;
    if (d.type === 'heartbeat') return;                    // 心跳只保活，不渲染
    render(d);
  };
  sock.onclose = () => {                                   // 3 秒后带 since 重连；MJPEG 是独立 <img>，不受影响
    if (sock !== ws) return;
    ws = null;
    retry = setTimeout(connect, 3000);
  };
  sock.onerror = () => sock.close();                       // 部分浏览器 onerror 后不触发 onclose，兜底
}
connect();
</script></body></html>"""


class AppState:
    """检测线程与 HTTP 处理线程之间唯一共享的状态；所有读写都过 lock。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.stats = {}
        self.seq = 0                # 统计版本号：每次 update_stats +1，WS 靠它判断"有没有新内容"
        self.error = None
        self.finished = False
        self.started = time.time()
        self.stop = threading.Event()
        self.thread = None

    def update_stats(self, stats):
        """写入新统计并推进版本号；两者在同一把锁里更新，保证 seq 与 stats 永远对应同一版。"""
        with self.lock:
            self.stats = stats
            self.seq += 1

    def snapshot(self):
        """锁内取一份可直接序列化的快照：seq + 运行状态 + 统计。

        HTTP /stats 与 WS 推送都走这里，避免读到"seq 已经变了、stats 还没换"的中间态。
        """
        with self.lock:
            return {'seq': self.seq, 'finished': self.finished, 'error': self.error, **self.stats}


def _parse_since(raw):
    """解析 WS 的 ?since=<int>；缺省/非法返回 None（按全新连接处理，坏参数不该让面板连不上）。"""
    try:
        return int(raw) if raw not in (None, '') else None
    except (TypeError, ValueError):
        return None


def create_app(cfg, detector, logger, source):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

    st = AppState()
    quality = int(cfg.get('serve', {}).get('quality', 80) or 80)
    token = str(cfg.get('serve', {}).get('token') or '')
    # 空闲心跳间隔：默认 15s；测试会调到亚秒级，才能在小窗口里验证"不重复推同 seq"
    heartbeat = max(0.05, float(cfg.get('serve', {}).get('ws_heartbeat', 15) or 15))
    tick = min(1.0, heartbeat)          # 统计轮询 1s；心跳更短时跟着缩短，保证心跳准时
    record = cfg['output']['video']

    def _authorized(request) -> bool:
        if not token:                       # 未配置 token = 不做鉴权（默认监听 127.0.0.1）
            return True
        got = request.query_params.get('token') or request.headers.get('x-token')
        return got == token

    def on_frame(anno):
        ok, buf = cv2.imencode('.jpg', anno, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            with st.lock:
                st.jpeg = buf.tobytes()

    def on_stats(s):
        st.update_stats(s)
        if cfg.get('metrics', {}).get('file'):
            write_metrics_file(cfg['metrics']['file'], s)

    def worker():
        try:
            from .pipeline import process
            process(source, cfg, detector, logger, show=False, record=record,
                    on_frame=on_frame, on_stats=on_stats, stop_event=st.stop)
        except Exception as e:                       # 面板不因源异常而挂掉
            logger.error(f'检测线程退出: {e}')
            with st.lock:
                st.error = str(e)
        finally:
            with st.lock:
                st.finished = True

    @asynccontextmanager
    async def lifespan(_app):
        """优雅停机：通知检测线程在帧边界退出，把 mp4 正常收尾再放行进程退出。"""
        yield
        st.stop.set()
        if st.thread is not None:
            await asyncio.to_thread(st.thread.join, 10)
        logger.info('面板已关闭，检测线程已收尾')

    app = FastAPI(title='YOLO 检测面板', lifespan=lifespan)
    st.thread = threading.Thread(target=worker, daemon=True)
    st.thread.start()

    @app.get('/', response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get('/healthz')
    def healthz():
        with st.lock:
            if st.error:
                status = 'error'
            elif st.finished:
                status = 'finished'
            else:
                status = 'ok'
            return JSONResponse({'status': status, 'frames': int(st.stats.get('frames', 0)),
                                 'uptime_s': round(time.time() - st.started, 1),
                                 'finished': st.finished, 'error': st.error})

    @app.get('/stats')
    def stats(request: Request):
        if not _authorized(request):
            raise HTTPException(status_code=401, detail='token 无效')
        return JSONResponse({'source': str(source), **st.snapshot()})

    @app.get('/metrics', response_class=PlainTextResponse)
    def metrics(request: Request):
        if not _authorized(request):
            raise HTTPException(status_code=401, detail='token 无效')
        with st.lock:
            return prometheus_text(st.stats)

    @app.get('/stream')
    def stream(request: Request):
        if not _authorized(request):
            raise HTTPException(status_code=401, detail='token 无效')

        def gen():
            # 检测结束（文件放完/出错/停机）后主动结束响应，否则浏览器连接会一直挂着
            while True:
                with st.lock:
                    jpg = st.jpeg
                    done = st.finished
                if jpg:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n'
                if done:
                    return
                time.sleep(1 / 30)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    @app.websocket('/ws')
    async def ws(sock: WebSocket):
        if token and (sock.query_params.get('token') or sock.headers.get('x-token')) != token:
            await sock.close(code=1008)              # 1008 = policy violation
            return
        since = _parse_since(sock.query_params.get('since'))
        await sock.accept()
        try:
            # 首条始终补当前快照：客户端无需等下一次统计更新就能画出首屏（since 只是提示，不阻塞首帧）
            snap = st.snapshot()
            last_seq, last_sent = snap['seq'], time.monotonic()
            if since is not None:
                logger.debug(f'WS 重连: 客户端 since={since}, 服务端 seq={last_seq}')
            await sock.send_json({'type': 'stats', 'source': str(source), **snap})
            while True:
                await asyncio.sleep(tick)
                snap = st.snapshot()
                if snap['seq'] != last_seq:          # 只有统计真的变过才推全量快照
                    last_seq, last_sent = snap['seq'], time.monotonic()
                    await sock.send_json({'type': 'stats', 'source': str(source), **snap})
                elif time.monotonic() - last_sent >= heartbeat:
                    last_sent = time.monotonic()     # 长时间无更新发心跳，客户端据此判断链路存活
                    await sock.send_json({'type': 'heartbeat', 'seq': snap['seq']})
        except WebSocketDisconnect:
            return

    return app


def _check_ws_support(logger):
    """uvicorn 需要 websockets/wsproto 才支持 WebSocket，缺了会静默返回 404。"""
    for mod in ('websockets', 'wsproto'):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    logger.warning('未安装 websockets/wsproto，/ws 实时推送不可用；'
                   '修复：pip install "uvicorn[standard]"')
    return False


def serve(cfg, detector, logger, source):
    import uvicorn
    _check_ws_support(logger)
    host = cfg['serve']['host']
    port = int(cfg['serve']['port'])
    app = create_app(cfg, detector, logger, source)
    if cfg['serve'].get('token'):
        logger.info('已启用 token 鉴权：访问时带 ?token=...（/healthz 除外）')
    elif host not in ('127.0.0.1', 'localhost'):
        logger.warning(f'面板监听 {host} 且未设置 serve.token，任何能访问该端口的人都能看画面；'
                       '建议设 serve.token 或 --token')
    logger.info(f'Web 面板已启动: http://{host}:{port}  (流 /stream · 统计 /ws · 指标 /metrics · 健康 /healthz)')
    uvicorn.run(app, host=host, port=port, log_level='warning')
