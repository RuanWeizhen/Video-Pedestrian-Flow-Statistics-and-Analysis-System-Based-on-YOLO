from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt

class EventTablePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["时间", "事件", "ID", "目标"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

    def reset(self):
        self.table.setRowCount(0)
        
    def add_event(self, timestamp, direction, track_id, zone):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.table.setItem(row, 1, QTableWidgetItem(direction))
        self.table.setItem(row, 2, QTableWidgetItem(str(track_id)))
        self.table.setItem(row, 3, QTableWidgetItem(zone))
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
        self.add_event(timestamp, event_name, track_id, target)
