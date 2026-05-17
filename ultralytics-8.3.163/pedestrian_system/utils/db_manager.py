import sqlite3
import os
from datetime import datetime
from datetime import timedelta

class DatabaseManager:
    def __init__(self, db_path="outputs/traffic.db"):
        self.db_path = db_path
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        # 建立持久化连接并关闭同线程检查，由于YOLO处理和UI更新有时在不同线程，设置 check_same_thread=False 保证安全
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
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
                source_name TEXT,
                source_path TEXT,
                config_path TEXT,
                started_at TEXT NOT NULL,
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
                source_name TEXT,
                timestamp TEXT,
                up_count INTEGER,
                down_count INTEGER,
                total_count INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                source_name TEXT,
                timestamp TEXT,
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
        self._ensure_column("traffic_data", "source_name", "TEXT")
        self._ensure_column("event_logs", "source_name", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in self.cursor.fetchall()}
        if column_name not in existing_columns:
            self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def start_session(self, source_name: str = "", source_path: str = "", config_path: str = "", conf: float | None = None, iou: float | None = None, detector_type: str = "") -> int:
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            """
            INSERT INTO sessions (source_name, source_path, config_path, started_at, conf, iou, detector_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_name, source_path, config_path, started_at, conf, iou, detector_type),
        )
        self.conn.commit()
        return int(self.cursor.lastrowid)

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

    def insert_data(self, up, down, total, session_id: int | None = None, source_name: str = ""):
        """
        插入一条包含当前时间戳的实时客流记录
        """
        # 获取当前时间串
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        insert_sql = """
        INSERT INTO traffic_data (session_id, source_name, timestamp, up_count, down_count, total_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(insert_sql, (session_id, source_name, timestamp, up, down, total))
        self.conn.commit()

    def insert_event(self, event: dict, session_id: int | None = None, source_name: str = "") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        raw_direction = str(event.get("direction", event.get("value", ""))).strip()
        if raw_direction.lower() == "up":
            raw_direction = "Up"
        elif raw_direction.lower() == "down":
            raw_direction = "Down"
        self.cursor.execute(
            """
            INSERT INTO event_logs (session_id, source_name, timestamp, frame_idx, event_type, direction, target, track_id, value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                source_name,
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

    def query_events(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, direction: str | None = None):
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

        sql = f"""
        SELECT timestamp, event_type, direction, target, track_id, value, source_name, frame_idx
        FROM event_logs
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp ASC, id ASC
        """
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def query_traffic_history(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None, bucket_minutes: int = 1):
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

        sql = f"""
        SELECT timestamp, up_count, down_count, total_count, source_name
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
                    "up": 0,
                    "down": 0,
                    "total": 0,
                    "records": 0,
                },
            )
            bucket["up"] += int(row["up_count"] or 0)
            bucket["down"] += int(row["down_count"] or 0)
            bucket["total"] += int(row["total_count"] or 0)
            bucket["records"] += 1

        return [grouped[k] for k in sorted(grouped.keys())]

    def query_sessions(self, start_time: str | None = None, end_time: str | None = None, source_name: str | None = None):
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
        sql = f"""
        SELECT * FROM sessions
        WHERE {' AND '.join(clauses)}
        ORDER BY started_at ASC, id ASC
        """
        self.cursor.execute(sql, params)
        return [dict(row) for row in self.cursor.fetchall()]

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