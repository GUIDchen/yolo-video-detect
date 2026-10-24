"""Prometheus 指标（文本暴露格式），零第三方依赖。

为什么手写文本而不是用 prometheus_client：
  · 指标很少且都是进程级累计值，手写几行文本比引一个库更省事，也更透明；
  · 同一份文本既能被 `serve` 的 /metrics 直接抓取，也能 `--metrics-file` 落盘，
    交给 node_exporter 的 textfile collector（不跑 Web 面板也能被监控）。

口径：所有值都是**本进程内累计**，进程重启归零；PromQL 侧用 increase()/rate() 处理。
"""
import time
from pathlib import Path

START = time.time()


def prometheus_text(stats, job='yolo_app'):
    """stats 为 pipeline 通过 on_stats 回调给出的实时统计字典。"""
    s = stats or {}
    lines = [
        '# HELP yolo_up 应用是否在跑',
        '# TYPE yolo_up gauge',
        'yolo_up 1',
        '# HELP yolo_uptime_seconds 进程运行时长',
        '# TYPE yolo_uptime_seconds gauge',
        f'yolo_uptime_seconds {time.time() - START:.1f}',
        '# HELP yolo_frames_inferred_total 已推理帧数',
        '# TYPE yolo_frames_inferred_total counter',
        f'yolo_frames_inferred_total {int(s.get("frames", 0))}',
        '# HELP yolo_frames_skipped_total 跳帧数（--stride 生效时）',
        '# TYPE yolo_frames_skipped_total counter',
        f'yolo_frames_skipped_total {int(s.get("skipped", 0))}',
        '# HELP yolo_fps 端到端帧率',
        '# TYPE yolo_fps gauge',
        f'yolo_fps {float(s.get("fps", 0))}',
        '# HELP yolo_unique_tracks 唯一目标数（启用跟踪时有效）',
        '# TYPE yolo_unique_tracks gauge',
        f'yolo_unique_tracks {int(s.get("unique_tracks", 0))}',
        '# HELP yolo_events_total 告警抓拍次数',
        '# TYPE yolo_events_total counter',
        f'yolo_events_total {int(s.get("events", 0))}',
        '# HELP yolo_inference_skipped_total 因门控/抽帧跳过的推理次数',
        '# TYPE yolo_inference_skipped_total counter',
        f'yolo_inference_skipped_total{{reason="stationary"}} {int(s.get("skipped_stationary", 0))}',
        f'yolo_inference_skipped_total{{reason="motion"}} {int(s.get("skipped_motion", 0))}',
        '# HELP yolo_infer_milliseconds_avg 单帧推理平均耗时',
        '# TYPE yolo_infer_milliseconds_avg gauge',
        f'yolo_infer_milliseconds_avg {float(s.get("infer_ms_avg", 0)):.2f}',
        '# HELP yolo_infer_milliseconds_max 单帧推理最大耗时',
        '# TYPE yolo_infer_milliseconds_max gauge',
        f'yolo_infer_milliseconds_max {float(s.get("infer_ms_max", 0)):.2f}',
        '# HELP yolo_reconnects_total 直播源重连次数',
        '# TYPE yolo_reconnects_total counter',
        f'yolo_reconnects_total {int(s.get("reconnects", 0))}',
        '# HELP yolo_alert_sent_total 告警发送成功数',
        '# TYPE yolo_alert_sent_total counter',
        f'yolo_alert_sent_total {int(s.get("alert_sent", 0))}',
        '# HELP yolo_alert_failed_total 告警发送失败数',
        '# TYPE yolo_alert_failed_total counter',
        f'yolo_alert_failed_total {int(s.get("alert_failed", 0))}',
        '# HELP yolo_alert_skipped_total 告警去重跳过数',
        '# TYPE yolo_alert_skipped_total counter',
        f'yolo_alert_skipped_total {int(s.get("alert_skipped", 0))}',
        '# HELP yolo_line_crossings_total 越线计数',
        '# TYPE yolo_line_crossings_total counter',
        f'yolo_line_crossings_total{{direction="in"}} {int((s.get("line") or {}).get("in", 0))}',
        f'yolo_line_crossings_total{{direction="out"}} {int((s.get("line") or {}).get("out", 0))}',
        '# HELP yolo_detections_total 各类别逐帧检测次数',
        '# TYPE yolo_detections_total counter',
    ]
    for k, v in (s.get('per_class') or {}).items():
        lines.append(f'yolo_detections_total{{class="{k}",job="{job}"}} {int(v)}')
    return '\n'.join(lines) + '\n'


def write_metrics_file(path, stats, job='yolo_app'):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(prometheus_text(stats, job), encoding='utf-8')
    return str(p)
