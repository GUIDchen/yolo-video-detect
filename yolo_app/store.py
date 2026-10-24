"""事件存储：SQLite + WAL，落库检测事件与告警 outbox（待发送队列）。

为什么用 SQLite 而不是 JSON 文件：
  · 检测线程写事件、重试/面板线程读 outbox，需要跨线程、跨进程的一致队列；
  · WAL 模式下读写互不阻塞，单文件部署、零外部依赖；
  · outbox 的"认领 → 发送 → 成功/退避"需要原子更新，SQL 的 UPDATE ... WHERE 最直接。

线程安全取舍：这里采用「单连接 + threading.Lock」而不是每线程一条连接。原因：
  1) 本模块的操作都是毫秒级小事务，锁竞争可以忽略；
  2) 单连接便于保证 claim_due 的 SELECT+UPDATE 原子性，也避免连接数随线程增长；
  3) sqlite3 连接默认不允许跨线程使用，必须 check_same_thread=False，再用锁把它串行化。
WAL 允许其它进程（例如面板脚本、离线分析）并发读同一个库，不会卡住检测主流程。

失败策略：数据库不可写（磁盘满、目录只读、路径被占用）时，构造与所有写方法都只记
warning 并返回 None/0/[]，绝不把异常抛给检测主流程。此时上层可以退回同步发送。
"""
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    frame INTEGER,
    source TEXT,
    cls TEXT,
    conf REAL,
    track_id INTEGER,
    image TEXT,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    text TEXT,
    status TEXT CHECK (status IN ('pending', 'sending', 'sent', 'dead')),
    attempts INTEGER DEFAULT 0,
    next_retry_at REAL,
    last_error TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_outbox_due ON outbox(status, next_retry_at);
"""


class EventStore:
    """事件与告警 outbox 的持久化入口。所有公开方法都保证不抛异常。

    典型用法：
        store = EventStore('outputs/events.db')
        event_id = store.add_event(ts, frame, source, cls, conf, track_id, image, payload)
        store.enqueue_alert(text, event_id)
        for row in store.claim_due(limit=10):
            if channel.send(row['text'], dedupe=False):
                store.mark_sent(row['id'])
            else:
                store.mark_failed(row['id'], '发送失败')
    """

    def __init__(self, path, logger=None):
        self.path = Path(path) if path else None
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._conn = None
        self._open()

    # ------------------------------------------------------------------ 生命周期
    def _open(self):
        """建立连接并建表；失败只告警，把 self._conn 保持为 None。"""
        if self.path is None:
            return
        conn = None
        try:
            if str(self.path) != ':memory:':
                self.path.parent.mkdir(parents=True, exist_ok=True)
            # isolation_level=None 即 autocommit：单条写立即落库，多语句自己显式 BEGIN/COMMIT。
            conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                   timeout=5.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA busy_timeout=5000')
            conn.executescript(_SCHEMA)
            self._conn = conn
        except Exception as e:
            self.logger.warning(f'事件库不可用（{self.path}）：{e}；本次运行不落库，检测继续')
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._conn = None

    @property
    def available(self):
        """数据库是否真正可用；不可用时上层应退回同步发送。"""
        return self._conn is not None

    def close(self):
        """关闭连接；可重复调用。关闭后所有方法都安全地返回空值。"""
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                self.logger.warning(f'关闭事件库失败：{e}')

    # ------------------------------------------------------------------ events
    def add_event(self, ts, frame=None, source='', cls='', conf=None, track_id=None,
                  image=None, payload=None):
        """写入一条事件，返回自增 id；失败返回 None。

        payload 传 dict 会序列化成 JSON 文本，已经是字符串则原样保存。
        """
        if self._conn is None:
            return None
        if payload is not None and not isinstance(payload, str):
            try:
                payload = json.dumps(payload, ensure_ascii=False)
            except Exception:
                payload = str(payload)
        try:
            with self._lock:
                cur = self._conn.execute(
                    'INSERT INTO events (ts, frame, source, cls, conf, track_id, image, payload) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (ts, int(frame) if frame is not None else None, str(source or ''),
                     str(cls or ''), float(conf) if conf is not None else None,
                     int(track_id) if track_id is not None else None,
                     str(image) if image is not None else None, payload))
                return int(cur.lastrowid)
        except Exception as e:
            self.logger.warning(f'写事件失败：{e}')
            return None

    # ------------------------------------------------------------------ outbox
    def enqueue_alert(self, text, event_id=None):
        """把一条待发送告警放进 outbox，返回自增 id；失败返回 None。"""
        if self._conn is None:
            return None
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO outbox (event_id, text, status, attempts, next_retry_at, "
                    "last_error, created_at) VALUES (?, ?, 'pending', 0, NULL, NULL, ?)",
                    (int(event_id) if event_id is not None else None, str(text), time.time()))
                return int(cur.lastrowid)
        except Exception as e:
            self.logger.warning(f'告警入队失败：{e}')
            return None

    def claim_due(self, now=None, limit=10):
        """把到期的 pending 原子地改成 sending，并返回这批记录（dict 列表）。

        到期条件：next_retry_at 为空或 <= now。整个过程用 BEGIN IMMEDIATE 包住，
        多进程同时跑也不会把同一条告警发给两个消费者。
        """
        if self._conn is None:
            return []
        try:
            now = time.time() if now is None else float(now)
            limit = max(int(limit), 0)
        except (TypeError, ValueError):
            now, limit = time.time(), 10
        if limit == 0:
            return []
        with self._lock:
            try:
                self._conn.execute('BEGIN IMMEDIATE')
                rows = self._conn.execute(
                    'SELECT id, event_id, text, attempts, next_retry_at, last_error, created_at '
                    "FROM outbox WHERE status = 'pending' "
                    'AND (next_retry_at IS NULL OR next_retry_at <= ?) ORDER BY id LIMIT ?',
                    (now, limit)).fetchall()
                if rows:
                    self._conn.executemany(
                        "UPDATE outbox SET status = 'sending' WHERE id = ? AND status = 'pending'",
                        [(int(r['id']),) for r in rows])
                self._conn.execute('COMMIT')
                return [dict(r) for r in rows]
            except Exception as e:
                self.logger.warning(f'认领告警失败：{e}')
                try:
                    self._conn.execute('ROLLBACK')
                except Exception:
                    pass
                return []

    def mark_sent(self, outbox_id):
        """标记发送成功；返回是否真的更新到一行。"""
        if self._conn is None:
            return False
        try:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE outbox SET status = 'sent', next_retry_at = NULL, last_error = NULL "
                    'WHERE id = ?', (int(outbox_id),))
                return cur.rowcount > 0
        except Exception as e:
            self.logger.warning(f'标记告警成功失败：{e}')
            return False

    def mark_failed(self, outbox_id, error, max_attempts=3, base_delay=2.0):
        """记录一次发送失败：attempts+1，未超限则按 base*2^(n-1) 退避，超限置 dead。

        n 是本次失败后的尝试次数（第一次失败退避 base，第二次 2*base，依此类推）。
        返回是否真的更新到一行。
        """
        if self._conn is None:
            return False
        try:
            max_attempts = max(int(max_attempts or 1), 1)
            base_delay = max(float(base_delay or 0.0), 0.0)
        except (TypeError, ValueError):
            max_attempts, base_delay = 3, 2.0
        with self._lock:
            try:
                self._conn.execute('BEGIN IMMEDIATE')
                row = self._conn.execute('SELECT attempts FROM outbox WHERE id = ?',
                                         (int(outbox_id),)).fetchone()
                if row is None:
                    self._conn.execute('ROLLBACK')
                    return False
                attempts = int(row['attempts'] or 0) + 1
                if attempts >= max_attempts:
                    status, next_retry = 'dead', None
                else:
                    status = 'pending'
                    next_retry = time.time() + base_delay * (2 ** (attempts - 1))
                self._conn.execute(
                    'UPDATE outbox SET status = ?, attempts = ?, next_retry_at = ?, last_error = ? '
                    'WHERE id = ?',
                    (status, attempts, next_retry, str(error or '发送失败'), int(outbox_id)))
                self._conn.execute('COMMIT')
                return True
            except Exception as e:
                self.logger.warning(f'记录告警失败状态出错：{e}')
                try:
                    self._conn.execute('ROLLBACK')
                except Exception:
                    pass
                return False

    def counts(self):
        """返回 {'pending': n, 'dead': n, 'events': n}；不可用时全为 0。

        pending 只统计还没被认领的记录；sending/sent/dead 不计入。
        """
        empty = {'pending': 0, 'dead': 0, 'events': 0}
        if self._conn is None:
            return empty
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT (SELECT COUNT(*) FROM outbox WHERE status = 'pending') AS pending, "
                    "(SELECT COUNT(*) FROM outbox WHERE status = 'dead') AS dead, "
                    '(SELECT COUNT(*) FROM events) AS events').fetchone()
            if row is None:
                return empty
            return {'pending': int(row['pending'] or 0), 'dead': int(row['dead'] or 0),
                    'events': int(row['events'] or 0)}
        except Exception as e:
            self.logger.warning(f'读取事件库计数失败：{e}')
            return empty
