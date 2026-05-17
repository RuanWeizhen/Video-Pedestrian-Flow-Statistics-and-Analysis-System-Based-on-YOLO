from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QGridLayout, QFrame
from PyQt5.QtCore import Qt

class StatsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        group = QGroupBox("实时统计数据")
        grid = QGridLayout()
        
        self.lbl_up = QLabel("0")
        self.lbl_down = QLabel("0")
        self.lbl_total = QLabel("0")
        self.lbl_fps = QLabel("0.0")
        self.lbl_current = QLabel("0")

        self.lbl_up_title = QLabel("Up 计数")
        self.lbl_down_title = QLabel("Down 计数")
        self.lbl_total_title = QLabel("Total 总数")
        self.lbl_fps_title = QLabel("FPS")
        self.lbl_current_title = QLabel("当前画面人数")

        card_style = (
            "QFrame { background: #ffffff; border: 1px solid #dbe6f2; border-radius: 14px; }"
            "QLabel { color: #16324f; }"
        )

        def build_card(title_label: QLabel, value_label: QLabel) -> QFrame:
            card = QFrame()
            card.setStyleSheet(card_style)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            card_layout.setSpacing(4)
            title_label.setStyleSheet("color: #6d7f95; font-size: 12px;")
            value_font = value_label.font()
            value_font.setPointSize(18)
            value_font.setBold(True)
            value_label.setFont(value_font)
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            card_layout.addStretch()
            return card
        
        # 样式强调
        font = self.lbl_up.font()
        font.setPointSize(12)
        font.setBold(True)
        
        for lbl in (self.lbl_up, self.lbl_down, self.lbl_total, self.lbl_fps, self.lbl_current):
            lbl.setAlignment(Qt.AlignCenter)
        
        # 网格布局
        grid.addWidget(build_card(self.lbl_up_title, self.lbl_up), 0, 0)
        grid.addWidget(build_card(self.lbl_down_title, self.lbl_down), 0, 1)
        grid.addWidget(build_card(self.lbl_total_title, self.lbl_total), 1, 0, 1, 2)
        grid.addWidget(build_card(self.lbl_fps_title, self.lbl_fps), 2, 0)
        grid.addWidget(build_card(self.lbl_current_title, self.lbl_current), 2, 1)
        
        group.setLayout(grid)
        main_layout.addWidget(group)
        
    def update_stats(self, stats: dict):
        self.lbl_up.setText(str(stats.get('up', 0)))
        self.lbl_down.setText(str(stats.get('down', 0)))
        self.lbl_total.setText(str(stats.get('total', 0)))
        self.lbl_fps.setText(f"{stats.get('fps', 0.0):.1f}")
        self.lbl_current.setText(str(stats.get('current', 0)))
