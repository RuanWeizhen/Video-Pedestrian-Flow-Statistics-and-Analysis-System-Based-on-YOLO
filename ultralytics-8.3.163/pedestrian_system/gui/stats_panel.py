from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout
from PyQt5.QtCore import Qt

class StatsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        group = QGroupBox("实时统计数据")
        grid = QGridLayout()
        
        self.lbl_up = QLabel("Up 计数: 0")
        self.lbl_down = QLabel("Down 计数: 0")
        self.lbl_total = QLabel("Total 总数: 0")
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_current = QLabel("当前画面人数: 0")
        
        # 样式强调
        font = self.lbl_up.font()
        font.setPointSize(12)
        font.setBold(True)
        
        for lbl in (self.lbl_up, self.lbl_down, self.lbl_total, self.lbl_fps, self.lbl_current):
            lbl.setFont(font)
            lbl.setAlignment(Qt.AlignCenter)
        
        # 网格布局
        grid.addWidget(self.lbl_up, 0, 0)
        grid.addWidget(self.lbl_down, 0, 1)
        grid.addWidget(self.lbl_total, 1, 0, 1, 2)
        grid.addWidget(self.lbl_fps, 2, 0)
        grid.addWidget(self.lbl_current, 2, 1)
        
        group.setLayout(grid)
        main_layout.addWidget(group)
        
    def update_stats(self, stats: dict):
        self.lbl_up.setText(f"Up 计数: {stats.get('up', 0)}")
        self.lbl_down.setText(f"Down 计数: {stats.get('down', 0)}")
        self.lbl_total.setText(f"Total 总数: {stats.get('total', 0)}")
        self.lbl_fps.setText(f"FPS: {stats.get('fps', 0.0):.1f}")
        self.lbl_current.setText(f"当前人数: {stats.get('current', 0)}")
