import os
import sqlite3
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from shutil import copy2

from .paths import resource_path, writable_path

class DatabaseManager:
    def __init__(self, db_path="outputs/traffic.db"):
        db_file = Path(db_path) if db_path is not None else writable_path("outputs/traffic.db")
        if not db_file.is_absolute():
            db_file = writable_path(db_file)

        self.db_path = db_file
        # 确保目录存在
        os.makedirs(self.db_path.parent, exist_ok=True)

        if not self.db_path.exists():
            seed_db = resource_path("traffic.db")
            if seed_db.exists() and seed_db != self.db_path:
                try:
                    copy2(seed_db, self.db_path)
                except Exception:
                    pass

        # 建立持久化连接并关闭同线程检查，由于YOLO处理和UI更新有时在不同线程，设置 check_same_thread=False 保证安全
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # 提升写入吞吐（实时统计场景）：WAL + 适度降低同步等级
        try:
            self.cursor.execute("PRAGMA journal_mode=WAL;")
            self.cursor.execute("PRAGMA synchronous=NORMAL;")
            self.cursor.execute("PRAGMA temp_store=MEMORY;")
            self.cursor.execute("PRAGMA cache_size=-20000;")  # ~20MB page cache
        except Exception:
            pass
        self.init_db()

    def init_db(self):
        """
        初始化数据库：创建连接，并在没有 traffic_data 表时自动创建
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                source_name TEXT,
                source_path TEXT,
                config_path TEXT,
                started_at TEXT NOT NULL,
                detect_time TEXT,
                created_at TEXT,
                ended_at TEXT,
                conf REAL,
                iou REAL,
                avg_fps REAL,
                up_count INTEGER DEFAULT 0,
                down_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0,
                detector_type TEXT,
                notes TEXT
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                run_id TEXT,
                source_name TEXT,
                detect_time TEXT,
                timestamp TEXT,
                created_at TEXT,
                frame_idx INTEGER,
                up_count INTEGER,
                down_count INTEGER,
                total_count INTEGER,
                fps REAL,
                avg_fps REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trajectory_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                run_id TEXT,
                source_name TEXT,
                detect_time TEXT,
                timestamp TEXT,
                created_at TEXT,
                frame_idx INTEGER,
                track_id INTEGER,
                x1 REAL,
                y1 REAL,
                x2 REAL,
                y2 REAL,
                cx REAL,
                cy REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                run_id TEXT,
                source_name TEXT,
                detect_time TEXT,
                timestamp TEXT,
                created_at TEXT,
                frame_idx INTEGER,
                event_type TEXT,
                direction TEXT,
                target TEXT,
                track_id INTEGER,
                value TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        self._ensure_column("sessions", "source_name", "TEXT")
        self._ensure_column("sessions", "run_id", "TEXT")
        self._ensure_column("sessions", "detect_time", "TEXT")
        self._ensure_column("sessions", "created_at", "TEXT")
        self._ensure_column("traffic_data", "source_name", "TEXT")
        self._ensure_column("traffic_data", "session_id", "INTEGER")
        self._ensure_column("traffic_data", "run_id", "TEXT")
        self._ensure_column("traffic_data", "detect_time", "TEXT")
        self._ensure_column("traffic_data", "created_at", "TEXT")
        self._ensure_column("traffic_data", "frame_idx", "INTEGER")
        self._ensure_column("traffic_data", "fps", "REAL")
        self._ensure_column("traffic_data", "avg_fps", "REAL")
        self._ensure_column("trajectory_points", "session_id", "INTEGER")
        self._ensure_column("trajectory_points", "run_id", "TEXT")
        self._ensure_column("trajectory_points", "source_name", "TEXT")
        self._ensure_column("trajectory_points", "detect_time", "TEXT")
        self._ensure_column("trajectory_points", "timestamp", "TEXT")
        self._ensure_column("trajectory_points", "created_at", "TEXT")
        self._ensure_column("trajectory_points", "frame_idx", "INTEGER")
        self._ensure_column("trajectory_points", "track_id", "INTEGER")
        self._ensure_column("trajectory_points", "x1", "REAL")
        self._ensure_column("trajectory_points", "y1", "REAL")
        self._ensure_column("trajectory_points", "x2", "REAL")
        self._ensure_column("trajectory_points", "y2", "REAL")
        self._ensure_column("trajectory_points", "cx", "REAL")
        self._ensure_column("trajectory_points", "cy", "REAL")
        self._ensure_column("event_logs", "source_name", "TEXT")
        self._ensure_column("event_logs", "session_id", "INTEGER")
        self._ensure_column("event_logs", "run_id", "TEXT")
        self._ensure_column("event_logs", "detect_time", "TEXT")
        self._ensure_column("event_logs", "created_at", "TEXT")
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL UNIQUE,
                roi_points_json TEXT,
                line_points_json TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in self.cursor.fetchall()}
        if column_name not in existing_columns:
            self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _coerce_session_id(self, run_id):
        if run_id is None:
            return None
        try:
            return int(str(run_id).strip())
        except Exception:
            return None

    def _normalize_session_row(self, row: dict) -> dict:
        data = dict(row)
        session_id = int(data.get("id") or data.get("session_id") or 0)
        source_name = str(data.get("source_name") or "")
        started_at = str(data.get("started_at") or data.get("detect_time") or data.get("created_at") or "")
        run_id = data.get("run_id") or (str(session_id) if session_id else "")
        data["id"] = session_id
        data["session_id"] = session_id
        data["run_id"] = str(run_id)
        data["source_name"] = source_name
        data["video_name"] = source_name
        data["detect_time"] = str(data.get("detect_time") or started_at or "")
        data["created_at"] = str(data.get("created_at") or started_at or "")
        return data

    def _normalize_row_common(self, row: dict) -> dict:
        data = dict(row)
        session_id = int(data.get("session_id") or data.get("id") or 0)
        source_name = str(data.get("source_name") or "")
        detect_time = str(data.get("detect_time") or data.get("timestamp") or "")
        created_at = str(data.get("created_at") or data.get("timestamp") or detect_time or "")
        run_id = data.get("run_id") or (str(session_id) if session_id else "")
        data["session_id"] = session_id
        data["run_id"] = str(run_id)
        data["source_name"] = source_name
        data["video_name"] = source_name
        data["detect_time"] = detect_time
        data["created_at"] = created_at
        return data

    def _append_session_filter(self, clauses, params, run_id=None, session_ids=None, session_id_column: str = "session_id"):
        if session_ids is not None:
            ids = []
            for item in session_ids:
                try:
                    ids.append(int(item))
                except Exception:
                    continue
            if ids:
                placeholders = ",".join(["?"] * len(ids))
                clauses.append(f"{session_id_column} IN ({placeholders})")
                params.extend(ids)
                return
            clauses.append("1=0")
            return
        if run_id is not None:
            session_id = self._coerce_session_id(run_id)
            if session_id is not None:
                clauses.append(f"({session_id_column} = ? OR run_id = ?)")
                params.extend([session_id, str(run_id)])
            else:
                clauses.append("run_id = ?")
                params.append(str(run_id))

    def start_session(self, source_name: str = "", source_path: str = "", config_path: str = "", conf: float | None = None, iou: float | None = None, detector_type: str = "", started_at: str | None = None) -> int:
        started_at = started_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            """
            INSERT INTO sessions (source_name, source_path, config_path, started_at, conf, iou, detector_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_name, source_path, config_path, started_at, conf, iou, detector_type),
        )
        session_id = int(self.cursor.lastrowid)
        self.cursor.execute(
            "UPDATE sessions SET run_id = ?, detect_time = ?, created_at = ? WHERE id = ?",
            (str(session_id), started_at, started_at, session_id),
        )
        self.conn.commit()
        return session_id

    def finalize_session(self, session_id: int, avg_fps: float | None = None, up_count: int | None = None, down_count: int | None = None, total_count: int | None = None, notes: str | None = None) -> None:
        fields = ["ended_at = ?"]
        values = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        if avg_fps is not None:
            fields.append("avg_fps = ?")
            values.append(float(avg_fps))
        if up_count is not None:
            fields.append("up_count = ?")
            values.append(int(up_count))
        if down_count is not None:
            fields.append("down_count = ?")
            values.append(int(down_count))
        if total_count is not None:
            fields.append("total_count = ?")
            values.append(int(total_count))
        if notes is not None:
            fields.append("notes = ?")
            values.append(notes)
        values.append(int(session_id))
        sql = f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?"
        self.cursor.execute(sql, values)
        self.conn.commit()

    def insert_data(self, up, down, total, session_id: int | None = None, source_name: str = "", run_id: str | None = None, detect_time: str | None = None, fps: float | None = None, avg_fps: float | None = None, created_at: str | None = None, frame_idx: int | None = None):
        """
        插入一条包含当前时间戳的实时客流记录
        """
        timestamp = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detect_time = detect_time or timestamp
        session_run_id = str(run_id or session_id or "")
        insert_sql = """
        INSERT INTO traffic_data (session_id, run_id, source_name, detect_time, timestamp, created_at, frame_idx, up_count, down_count, total_count, fps, avg_fps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(
            insert_sql,
            (
                session_id,
                session_run_id,
                source_name,
                detect_time,
                timestamp,
                timestamp,
                int(frame_idx) if frame_idx is not None else None,
                up,
                down,
                total,
                fps,
                avg_fps,
            ),
        )
        self.conn.commit()

    def insert_trajectory_points(self, points: list[dict]) -> None:
        if not points:
            return

        rows = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for item in points:
            if not isinstance(item, dict):
                continue
            rows.append(
                (
                    item.get("session_id"),
                    str(item.get("run_id") or item.get("session_id") or ""),
                    str(item.get("source_name", "")),
                    str(item.get("detect_time") or timestamp),
                    str(item.get("timestamp") or timestamp),
                    str(item.get("created_at") or timestamp),
                    int(item.get("frame_idx", 0) or 0),
                    int(item.get("track_id", 0) or 0),
                    float(item.get("x1", 0.0) or 0.0),
                    float(item.get("y1", 0.0) or 0.0),
                    float(item.get("x2", 0.0) or 0.0),
                    float(item.get("y2", 0.0) or 0.0),
                    float(item.get("cx", 0.0) or 0.0),
                    float(item.get("cy", 0.0) or 0.0),
                )
            )

        if not rows:
            return

        self.cursor.executemany(
            """
            INSERT INTO trajectory_points (
                session_id, run_id, source_name, detect_time, timestamp, created_at,
                frame_idx, track_id, x1, y1, x2, y2, cx, cy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def insert_event(self, event: dict, session_id: int | None = None, source_name: str = "", run_id: str | None = None, detect_time: str | None = None, created_at: str | None = None) -> None:
        timestamp = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detect_time = detect_time or timestamp
        session_run_id = str(run_id or session_id or "")
        raw_direction = str(event.get("direction", event.get("value", ""))).strip()
        if raw_direction.lower() == "up":
            raw_direction = "Up"
        elif raw_direction.lower() == "down":
            raw_direction = "Down"
        self.cursor.execute(
            """
            INSERT INTO event_logs (session_id, run_id, source_name, detect_time, timestamp, created_at, frame_idx, event_type, direction, target, track_id, value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                session_run_id,
                source_name,
                detect_time,
                timestamp,
                timestamp,
                int(event.get("frame_idx", 0)),
                str(event.get("event_type", "")),
                raw_direction,
                str(event.get("target", "")),
                int(event.get("track_id", 0) or 0),
                str(event.get("value", "")),
            ),
        )
        self.conn.commit()

    def query_last_n(self, n=50):
        """
        倒序查询最近的 n 条数据，返回包含字典的列表。
        每个字典格式: {'id': 1, 'timestamp': '...', 'up_count': 0, 'down_count': 0, 'total_count': 0}
        """
        query_sql = """
        SELECT * FROM traffic_data 
        ORDER BY id DESC 
        LIMIT ?
        """
        self.cursor.execute(query_sql, (n,))
        rows = self.cursor.fetchall()
        
        # 为了方便绘图，通常最老的数据放前面，这里我们可以把倒序变回正序
        results = []
        for r in reversed(rows):
            results.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "up_count": r["up_count"],
                "down_count": r["down_count"],
                "total_count": r["total_count"]
            })
        return results

    def query_sources(self):
        self.cursor.execute(
            "SELECT DISTINCT source_name FROM sessions WHERE source_name IS NOT NULL AND source_name <> '' ORDER BY source_name ASC"
        )
        return [row[0] for row in self.cursor.fetchall()]

    def query_events(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, direction: str | None = None, run_id: str | None = None, session_ids: list[int] | None = None):
        clauses = ["1=1"]
        params = []
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if source_name and source_name != "全部视频":
            clauses.append("source_name = ?")
            params.append(source_name)
        if direction and direction != "All":
            clauses.append("direction = ?")
            params.append(direction)
        self._append_session_filter(clauses, params, run_id=run_id, session_ids=session_ids)

        sql = f"""
        SELECT timestamp, event_type, direction, target, track_id, value, source_name, frame_idx, session_id, run_id, detect_time, created_at
        FROM event_logs
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp ASC, id ASC
        """
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        return [self._normalize_row_common(dict(row)) for row in rows]

    def query_traffic_history(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, bucket_minutes: int = 1, run_id: str | None = None, session_ids: list[int] | None = None):
        clauses = ["1=1"]
        params = []
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if source_name and source_name != "全部视频":
            clauses.append("source_name = ?")
            params.append(source_name)
        self._append_session_filter(clauses, params, run_id=run_id, session_ids=session_ids)

        sql = f"""
        SELECT timestamp, up_count, down_count, total_count, source_name, session_id, run_id, detect_time, created_at, frame_idx, fps, avg_fps
        FROM traffic_data
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp ASC, id ASC
        """
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()

        bucket_minutes = max(1, int(bucket_minutes))
        grouped = {}
        for row in rows:
            timestamp = str(row["timestamp"] or "")
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                bucket_minute = (dt.minute // bucket_minutes) * bucket_minutes
                bucket_start = dt.replace(minute=bucket_minute, second=0, microsecond=0)
                bucket_end = bucket_start + timedelta(minutes=bucket_minutes)
                bucket_key = bucket_start.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                bucket_key = timestamp[:16]
                bucket_start = None
                bucket_end = None

            bucket = grouped.setdefault(
                bucket_key,
                {
                    "timestamp": bucket_key,
                    "bucket_start": bucket_start.strftime("%Y-%m-%d %H:%M:%S") if bucket_start else bucket_key,
                    "bucket_end": bucket_end.strftime("%Y-%m-%d %H:%M:%S") if bucket_end else bucket_key,
                    "source_name": source_name if source_name and source_name != "全部视频" else "全部视频",
                    "video_name": source_name if source_name and source_name != "全部视频" else "全部视频",
                    "frame_idx": int(row["frame_idx"] or 0) if row["frame_idx"] is not None else 0,
                    "up": 0,
                    "down": 0,
                    "total": 0,
                    "records": 0,
                    "fps_values": [],
                    "avg_fps_values": [],
                },
            )
            bucket["up"] += int(row["up_count"] or 0)
            bucket["down"] += int(row["down_count"] or 0)
            bucket["total"] += int(row["total_count"] or 0)
            bucket["records"] += 1
            if row["fps"] is not None:
                bucket["fps_values"].append(float(row["fps"] or 0.0))
            if row["avg_fps"] is not None:
                bucket["avg_fps_values"].append(float(row["avg_fps"] or 0.0))

        results = []
        for key in sorted(grouped.keys()):
            bucket = grouped[key]
            fps_values = bucket.pop("fps_values", [])
            avg_fps_values = bucket.pop("avg_fps_values", [])
            bucket["fps"] = (sum(fps_values) / len(fps_values)) if fps_values else 0.0
            bucket["avg_fps"] = (sum(avg_fps_values) / len(avg_fps_values)) if avg_fps_values else bucket["fps"]
            results.append(bucket)
        return results

    def query_traffic_samples(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, run_id: str | None = None, session_ids: list[int] | None = None):
        clauses = ["1=1"]
        params = []
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if source_name and source_name != "全部视频":
            clauses.append("source_name = ?")
            params.append(source_name)
        self._append_session_filter(clauses, params, run_id=run_id, session_ids=session_ids)

        sql = f"""
        SELECT timestamp, up_count, down_count, total_count, source_name, session_id, run_id, detect_time, created_at, frame_idx, fps, avg_fps
        FROM traffic_data
        WHERE {' AND '.join(clauses)}
        ORDER BY frame_idx ASC, timestamp ASC, id ASC
        """
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        return [self._normalize_row_common(dict(row)) for row in rows]

    def query_fps_history(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, run_id: str | None = None, session_ids: list[int] | None = None):
        clauses = ["1=1"]
        params = []
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if source_name and source_name != "全部视频":
            clauses.append("source_name = ?")
            params.append(source_name)
        self._append_session_filter(clauses, params, run_id=run_id, session_ids=session_ids)

        sql = f"""
        SELECT timestamp, fps, avg_fps, source_name, session_id, run_id, detect_time, created_at, frame_idx
        FROM traffic_data
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp ASC, id ASC
        """
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            data = self._normalize_row_common(dict(row))
            fps_value = data.get("fps")
            avg_fps_value = data.get("avg_fps")
            if fps_value is None:
                fps_value = avg_fps_value if avg_fps_value is not None else 0.0
            data["fps"] = float(fps_value or 0.0)
            data["avg_fps"] = float(avg_fps_value or data["fps"])
            results.append(data)
        return results

    def query_sessions(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, run_id: str | None = None, session_ids: list[int] | None = None, order: str = "asc", limit: int | None = None):
        clauses = ["1=1"]
        params = []
        if start_time:
            clauses.append("started_at >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("started_at <= ?")
            params.append(end_time)
        if source_name and source_name != "全部视频":
            clauses.append("source_name = ?")
            params.append(source_name)
        self._append_session_filter(clauses, params, run_id=run_id, session_ids=session_ids, session_id_column="id")
        order_key = "DESC" if str(order).strip().lower() == "desc" else "ASC"
        sql = f"""
        SELECT * FROM sessions
        WHERE {' AND '.join(clauses)}
        ORDER BY started_at {order_key}, id {order_key}
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        self.cursor.execute(sql, params)
        return [self._normalize_session_row(dict(row)) for row in self.cursor.fetchall()]

    def query_trajectory_points(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, run_id: str | None = None, session_ids: list[int] | None = None):
        clauses = ["1=1"]
        params = []
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if source_name and source_name != "全部视频":
            clauses.append("source_name = ?")
            params.append(source_name)
        self._append_session_filter(clauses, params, run_id=run_id, session_ids=session_ids)

        sql = f"""
        SELECT session_id, run_id, source_name, detect_time, timestamp, created_at, frame_idx, track_id, x1, y1, x2, y2, cx, cy
        FROM trajectory_points
        WHERE {' AND '.join(clauses)}
        ORDER BY frame_idx ASC, timestamp ASC, track_id ASC, id ASC
        """
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        return [self._normalize_row_common(dict(row)) for row in rows]

    def query_run_batches(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, limit: int | None = None):
        sessions = self.query_sessions(start_time=start_time, end_time=end_time, source_name=source_name, order="desc", limit=limit)
        batches = []
        for row in sessions:
            batches.append({
                "session_id": row.get("id", 0),
                "run_id": row.get("run_id", ""),
                "source_name": row.get("source_name", ""),
                "video_name": row.get("video_name", ""),
                "started_at": row.get("started_at", ""),
                "ended_at": row.get("ended_at", ""),
                "display_text": f"{row.get('started_at', '')} - {row.get('source_name', '')}",
                "status": "进行中" if not row.get("ended_at") else "已结束",
                "row": row,
            })
        return batches

    def delete_old_data(self, limit=1000):
        """
        清理旧数据：只保留最近的 limit 条记录，其余老数据删掉。
        （提升检索速度，防止数据库无限膨胀）
        """
        delete_sql = """
        DELETE FROM traffic_data
        WHERE id NOT IN (
            SELECT id FROM traffic_data
            ORDER BY id DESC
            LIMIT ?
        )
        """
        self.cursor.execute(delete_sql, (limit,))
        self.conn.commit()

    def upsert_annotations(self, source_path: str, roi_points: list, line_points: list) -> None:
        import json

        roi_json = json.dumps([[float(p[0]), float(p[1])] for p in (roi_points or [])])
        line_json = json.dumps([[float(p[0]), float(p[1])] for p in (line_points or [])[:2]])
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            """
            INSERT INTO annotations (source_path, roi_points_json, line_points_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                roi_points_json = excluded.roi_points_json,
                line_points_json = excluded.line_points_json,
                updated_at = excluded.updated_at
            """,
            (str(source_path), roi_json, line_json, updated_at),
        )
        self.conn.commit()

    def query_annotations(self, source_path: str) -> dict | None:
        import json

        self.cursor.execute(
            "SELECT roi_points_json, line_points_json FROM annotations WHERE source_path = ?",
            (str(source_path),),
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        roi_points = []
        line_points = []
        try:
            roi_raw = json.loads(row["roi_points_json"] or "[]")
            roi_points = [tuple(p) for p in roi_raw]
        except Exception:
            pass
        try:
            line_raw = json.loads(row["line_points_json"] or "[]")
            line_points = [tuple(p) for p in line_raw]
        except Exception:
            pass
        return {"roi_points": roi_points, "line_points": line_points}

    def close(self):
        """
        资源释放前，安全关闭所有链接
        """
        try:
            self.cursor.close()
            self.conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    # === 使用示例 ===
    
    # 1. 实例化管理类（将自动创建 outputs/traffic.db 并初始化建表）
    db = DatabaseManager("outputs/test_traffic.db")
    
    # 2. 模拟由于 YOLO 检测带来的客流发生变化并定时触发写入
    import time
    print("开始模拟插入数据...")
    for i in range(3):
        # 参数: up_count, down_count, total_count
        db.insert_data(i*2, i, i*3)
        time.sleep(1) # 现实中可以用 PyQt的QTimer每秒触发一次
        
    # 3. 打印验证
    print("\n查询最近10条数据结果:")
    recent_data = db.query_last_n(10)
    for item in recent_data:
        print(f"[{item['timestamp']}] UP: {item['up_count']}, DOWN: {item['down_count']}, TOTAL: {item['total_count']}")