from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
import csv

class EventTablePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["时间", "事件类型", "目标ID", "计数线名称", "批次ID"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

    def reset(self):
        self.table.setRowCount(0)
        
    def add_event(self, timestamp, direction, track_id, zone, run_id=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.table.setItem(row, 1, QTableWidgetItem(direction))
        self.table.setItem(row, 2, QTableWidgetItem(str(track_id)))
        self.table.setItem(row, 3, QTableWidgetItem(zone))
        self.table.setItem(row, 4, QTableWidgetItem(str(run_id)))
        self.table.scrollToBottom()

    def add_event_record(self, event, fps=25.0):
        frame_idx = int(event.get("frame_idx", 0))
        sec = frame_idx / fps if fps > 1e-6 else 0.0
        mm = int(sec // 60)
        ss = int(sec % 60)
        timestamp = f"{mm:02d}:{ss:02d}"

        event_name = str(event.get("value") or event.get("event_type") or "")
        track_id = event.get("track_id", "")
        target = str(event.get("target", ""))
        run_id = str(event.get("run_id", ""))
        self.add_event(timestamp, event_name, track_id, target, run_id=run_id)

    def set_records(self, records):
        self.reset()
        for item in records:
            self.add_event(
                str(item.get("timestamp", "")),
                str(item.get("event", "")),
                str(item.get("track_id", "")),
                str(item.get("target", "")),
                run_id=str(item.get("run_id", "")),
            )

    def export_csv(self, file_path):
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["时间", "事件", "ID", "目标", "批次ID"])
            for row in range(self.table.rowCount()):
                writer.writerow([
                    self.table.item(row, 0).text() if self.table.item(row, 0) else "",
                    self.table.item(row, 1).text() if self.table.item(row, 1) else "",
                    self.table.item(row, 2).text() if self.table.item(row, 2) else "",
                    self.table.item(row, 3).text() if self.table.item(row, 3) else "",
                    self.table.item(row, 4).text() if self.table.item(row, 4) else "",
                ])
