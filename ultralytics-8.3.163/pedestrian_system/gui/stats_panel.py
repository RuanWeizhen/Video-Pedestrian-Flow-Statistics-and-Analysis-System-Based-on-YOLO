from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame
from PyQt5.QtCore import Qt


class StatsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_up = QLabel("0")
        self.lbl_down = QLabel("0")
        self.lbl_total = QLabel("0")
        self.lbl_fps = QLabel("0.0")
        self.lbl_current = QLabel("0")
        self.lbl_timestamp = QLabel("--:--")
        self.lbl_frame_pos = QLabel("0 / 0")
        self.lbl_tracks = QLabel("0")

        title_style = "color: #6d7f95; font-size: 10px; font-weight: 500;"

        self.lbl_up_title = QLabel("累计\nUp")
        self.lbl_up_title.setAlignment(Qt.AlignCenter)
        self.lbl_up_title.setStyleSheet(title_style)

        self.lbl_down_title = QLabel("累计\nDown")
        self.lbl_down_title.setAlignment(Qt.AlignCenter)
        self.lbl_down_title.setStyleSheet(title_style)

        self.lbl_total_title = QLabel("累计\nTotal")
        self.lbl_total_title.setAlignment(Qt.AlignCenter)
        self.lbl_total_title.setStyleSheet(title_style)

        self.lbl_fps_title = QLabel("回放\nFPS")
        self.lbl_fps_title.setAlignment(Qt.AlignCenter)
        self.lbl_fps_title.setStyleSheet(title_style)

        self.lbl_current_title = QLabel("当前\n活跃")
        self.lbl_current_title.setAlignment(Qt.AlignCenter)
        self.lbl_current_title.setStyleSheet(title_style)

        self.lbl_timestamp_title = QLabel("视频\n时间")
        self.lbl_timestamp_title.setAlignment(Qt.AlignCenter)
        self.lbl_timestamp_title.setStyleSheet(title_style)

        self.lbl_frame_pos_title = QLabel("帧\n位置")
        self.lbl_frame_pos_title.setAlignment(Qt.AlignCenter)
        self.lbl_frame_pos_title.setStyleSheet(title_style)

        self.lbl_tracks_title = QLabel("活跃\nID 数")
        self.lbl_tracks_title.setAlignment(Qt.AlignCenter)
        self.lbl_tracks_title.setStyleSheet(title_style)

        card_base = (
            "QFrame {"
            "  background: #ffffff;"
            "  border: 1px solid #dfe6f0;"
            "  border-radius: 10px;"
            "}"
        )
        value_style = "color: #1a2d4a; font-size: 22px; font-weight: 700;"
        accent_value_style = "color: #2563eb; font-size: 22px; font-weight: 700;"

        def build_card(val_label: QLabel, accent: bool = False) -> QFrame:
            card = QFrame()
            card.setStyleSheet(card_base)
            card.setMinimumHeight(75)
            card.setSizePolicy(card.sizePolicy().horizontalPolicy(),
                               card.sizePolicy().Expanding if hasattr(card, 'Expanding') else card.sizePolicy().verticalPolicy())
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)
            val_label.setAlignment(Qt.AlignCenter)
            val_label.setStyleSheet(accent_value_style if accent else value_style)
            card_layout.addWidget(val_label)
            return card

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        grid.addWidget(self.lbl_up_title, 0, 0, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_up), 1, 0)
        grid.addWidget(self.lbl_down_title, 0, 1, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_down), 1, 1)
        grid.addWidget(self.lbl_current_title, 0, 2, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_current, accent=True), 1, 2)

        grid.addWidget(self.lbl_total_title, 2, 0, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_total), 3, 0)
        grid.addWidget(self.lbl_fps_title, 2, 1, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_fps, accent=True), 3, 1)
        grid.addWidget(self.lbl_timestamp_title, 2, 2, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_timestamp), 3, 2)

        grid.addWidget(self.lbl_frame_pos_title, 4, 0, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_frame_pos), 5, 0)
        grid.addWidget(self.lbl_tracks_title, 4, 1, Qt.AlignCenter)
        grid.addWidget(build_card(self.lbl_tracks), 5, 1)

        main_layout.addLayout(grid)

    def update_stats(self, stats: dict):
        self.lbl_up.setText(str(stats.get("up", 0)))
        self.lbl_down.setText(str(stats.get("down", 0)))
        self.lbl_total.setText(str(stats.get("total", 0)))
        self.lbl_fps.setText(f"{stats.get('fps', 0.0):.1f}")
        self.lbl_current.setText(str(stats.get("current", 0)))
        self.lbl_timestamp.setText(str(stats.get("timestamp", "--:--")))
        self.lbl_frame_pos.setText(str(stats.get("frame_position", "0 / 0")))
        self.lbl_tracks.setText(str(stats.get("track_count", 0)))
