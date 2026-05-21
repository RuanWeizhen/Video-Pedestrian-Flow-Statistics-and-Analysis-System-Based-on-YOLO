from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
import csv


class ExperimentTablePanel(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["时间", "conf", "iou", "avg_fps", "up", "down", "批次ID"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

    def reset(self):
        self.table.setRowCount(0)

    def add_record(self, timestamp: str, conf: float, iou: float, avg_fps: float, up: int, down: int, run_id: str = ""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(timestamp)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{float(conf):.2f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{float(iou):.2f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{float(avg_fps):.2f}"))
        self.table.setItem(row, 4, QTableWidgetItem(str(int(up))))
        self.table.setItem(row, 5, QTableWidgetItem(str(int(down))))
        self.table.setItem(row, 6, QTableWidgetItem(str(run_id)))
        self.table.scrollToBottom()

    def export_csv(self, file_path):
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["时间", "conf", "iou", "avg_fps", "up", "down", "批次ID"])
            for row in range(self.table.rowCount()):
                writer.writerow([
                    self.table.item(row, 0).text() if self.table.item(row, 0) else "",
                    self.table.item(row, 1).text() if self.table.item(row, 1) else "",
                    self.table.item(row, 2).text() if self.table.item(row, 2) else "",
                    self.table.item(row, 3).text() if self.table.item(row, 3) else "",
                    self.table.item(row, 4).text() if self.table.item(row, 4) else "",
                    self.table.item(row, 5).text() if self.table.item(row, 5) else "",
                    self.table.item(row, 6).text() if self.table.item(row, 6) else "",
                ])
