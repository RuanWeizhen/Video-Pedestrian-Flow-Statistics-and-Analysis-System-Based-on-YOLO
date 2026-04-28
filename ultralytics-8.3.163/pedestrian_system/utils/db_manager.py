import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="outputs/traffic.db"):
        self.db_path = db_path
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        # 建立持久化连接并关闭同线程检查，由于YOLO处理和UI更新有时在不同线程，设置 check_same_thread=False 保证安全
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.init_db()

    def init_db(self):
        """
        初始化数据库：创建连接，并在没有 traffic_data 表时自动创建
        """
        # 使用 IF NOT EXISTS 来保证多次运行不会重复报错
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS traffic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            up_count INTEGER,
            down_count INTEGER,
            total_count INTEGER
        )
        """
        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def insert_data(self, up, down, total):
        """
        插入一条包含当前时间戳的实时客流记录
        """
        # 获取当前时间串
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        insert_sql = """
        INSERT INTO traffic_data (timestamp, up_count, down_count, total_count)
        VALUES (?, ?, ?, ?)
        """
        self.cursor.execute(insert_sql, (timestamp, up, down, total))
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