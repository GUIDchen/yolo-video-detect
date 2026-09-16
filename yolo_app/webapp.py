"""Web 面板：MJPEG 实时流 + WebSocket 统计推送 + Prometheus /metrics。

线程模型：uvicorn 主线程处理 HTTP/WS，检测跑在**后台线程**（复用 pipeline.process），
两者通过 AppState（一个 dict + 一把锁）交换"最新一张 JPEG"和"最新统计"，不做队列堆积——
实时画面只要最新帧，慢消费者自然丢帧，这是有意为之的取舍。

接口一览：
    GET  /         面板页（内嵌 HTML，实时画面 + 统计表）
    GET  /stream   MJPEG 流（multipart/x-mixed-replace），可直接嵌到别的前端
    GET  /stats    当前统计 JSON
    WS   /ws       每秒推送一次统计（前端表格就是靠它刷新的）
    GET  /metrics  Prometheus 文本指标

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
  <div style="flex:2 1 640px"><img src="/stream" alt="实时流"></div>
  <div class="panel">
    <h3 style="margin-top:0">实时统计</h3>
    <table id="tbl"><tbody></tbody></table>
    <p class="k">WebSocket 推送 · <a href="/metrics" style="color:#4da3ff">/metrics</a> Prometheus · <a href="/stats" style="color:#4da3ff">/stats</a> JSON</p>
  </div>
</div>
<script>
const $ = (s) => document.querySelector(s);
const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  $('#fps').textContent = (d.fps || 0) + ' FPS';
  const rows = [['帧数', d.frames], ['FPS', d.fps], ['唯一目标', d.unique_tracks],
                ['告警抓拍', d.events], ['越线 in', (d.line || {}).in], ['越线 out', (d.line || {}).out]];
  for (const [k, v] of Object.entries(d.per_class || {})) rows.push(['检测 · ' + k, v]);
  $('#tbl').innerHTML = '<tbody>' + rows.map(r => '<tr><td class="k">' + r[0] + '</td><td class="v">' + (r[1] ?? 0) + '</td></tr>').join('') + '</tbody>';
};
</script></body></html>"""


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.stats = {}
        self.error = None
        self.finished = False


def create_app(cfg, detector, logger, source):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

    st = AppState()
    quality = int(cfg.get('serve', {}).get('quality', 80) or 80)
    record = cfg['output']['video']

    def on_frame(anno):
        ok, buf = cv2.imencode('.jpg', anno, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            with st.lock:
                st.jpeg = buf.tobytes()

    def on_stats(s):
        with st.lock:
            st.stats = s
        if cfg.get('metrics', {}).get('file'):
            write_metrics_file(cfg['metrics']['file'], s)

    def worker():
        try:
            from .pipeline import process
            process(source, cfg, detector, logger, show=False, record=record,
                    on_frame=on_frame, on_stats=on_stats)
        except Exception as e:                       # 面板不因源异常而挂掉
            logger.error(f'检测线程退出: {e}')
            with st.lock:
                st.error = str(e)
        finally:
            with st.lock:
                st.finished = True

    app = FastAPI(title='YOLO 检测面板')
    threading.Thread(target=worker, daemon=True).start()

    @app.get('/', response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get('/stats')
    def stats():
        with st.lock:
            return JSONResponse({'source': str(source), 'finished': st.finished,
                                 'error': st.error, **st.stats})

    @app.get('/metrics', response_class=PlainTextResponse)
    def metrics():
        with st.lock:
            return prometheus_text(st.stats)

    @app.get('/stream')
    def stream():
        def gen():
            while True:
                with st.lock:
                    jpg = st.jpeg
                if jpg:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n'
                time.sleep(1 / 30)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    @app.websocket('/ws')
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                with st.lock:
                    payload = {'source': str(source), 'finished': st.finished, **st.stats}
                await sock.send_json(payload)
                await asyncio.sleep(1)
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
    logger.info(f'Web 面板已启动: http://{host}:{port}  (流 /stream · 统计 /ws · 指标 /metrics)')
    uvicorn.run(app, host=host, port=port, log_level='warning')
