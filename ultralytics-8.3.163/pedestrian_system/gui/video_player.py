from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

_BAR_BG = "#F5F7FA"

_BUTTON_STYLE = (
    "QPushButton {"
    "  background: #FFFFFF;"
    "  border: 1px solid #DDE1E8;"
    "  border-radius: 8px;"
    "  color: #5B6D85;"
    "  font-size: 18px;"
    "}"
    "QPushButton:hover {"
    "  background: #EBF0FB;"
    "  border-color: #93B4F5;"
    "  color: #3B82F6;"
    "}"
    "QPushButton:pressed {"
    "  background: #D6E2FA;"
    "  border-color: #3B82F6;"
    "  color: #1D4ED8;"
    "}"
)

_PLAY_BUTTON_STYLE = (
    "QPushButton {"
    "  background: #3B82F6;"
    "  border: 1px solid #3B82F6;"
    "  border-radius: 10px;"
    "  color: #FFFFFF;"
    "  font-size: 20px;"
    "}"
    "QPushButton:hover {"
    "  background: #2563EB;"
    "  border-color: #2563EB;"
    "}"
    "QPushButton:pressed {"
    "  background: #1D4ED8;"
    "  border-color: #1D4ED8;"
    "}"
)

_SLIDER_STYLE = (
    "QSlider::groove:horizontal {"
    "  height: 6px;"
    "  background: #E2E6ED;"
    "  border-radius: 3px;"
    "}"
    "QSlider::handle:horizontal {"
    "  background: #FFFFFF;"
    "  width: 18px;"
    "  margin: -8px 0;"
    "  border-radius: 9px;"
    "  border: 2px solid #3B82F6;"
    "}"
    "QSlider::handle:horizontal:hover {"
    "  border-color: #1D4ED8;"
    "  background: #EBF0FB;"
    "}"
    "QSlider::sub-page:horizontal {"
    "  background: #3B82F6;"
    "  border-radius: 3px;"
    "}"
    "QSlider::add-page:horizontal {"
    "  background: #E2E6ED;"
    "  border-radius: 3px;"
    "}"
)


class VideoControlBar(QWidget):
    seek_requested = pyqtSignal(int)
    play_pause_clicked = pyqtSignal()
    step_forward = pyqtSignal(int)
    step_backward = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._playing = False
        self._slider_dragging = False
        self.setFixedHeight(52)
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet(f"VideoControlBar {{ background: {_BAR_BG}; border-top: 1px solid #DDE1E8; }}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        btn_size = QSize(44, 36)
        icon_size = QSize(24, 24)

        self.btn_rewind = QPushButton()
        self.btn_rewind.setIcon(self._icon_from_style(QStyle.SP_MediaSeekBackward))
        self.btn_rewind.setIconSize(icon_size)
        self.btn_rewind.setFixedSize(btn_size)
        self.btn_rewind.setToolTip("后退 30 帧")
        self.btn_rewind.setStyleSheet(_BUTTON_STYLE)

        self.btn_play_pause = QPushButton()
        self.btn_play_pause.setIcon(self._icon_from_style(QStyle.SP_MediaPlay))
        self.btn_play_pause.setIconSize(QSize(28, 28))
        self.btn_play_pause.setFixedSize(QSize(52, 40))
        self.btn_play_pause.setToolTip("播放 / 暂停")
        self.btn_play_pause.setStyleSheet(_PLAY_BUTTON_STYLE)

        self.btn_forward = QPushButton()
        self.btn_forward.setIcon(self._icon_from_style(QStyle.SP_MediaSeekForward))
        self.btn_forward.setIconSize(icon_size)
        self.btn_forward.setFixedSize(btn_size)
        self.btn_forward.setToolTip("前进 30 帧")
        self.btn_forward.setStyleSheet(_BUTTON_STYLE)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setToolTip("拖动以跳转帧")
        self.slider.setStyleSheet(_SLIDER_STYLE)

        self.label_position = QLabel("0 / 0")
        self.label_position.setFixedWidth(130)
        self.label_position.setAlignment(Qt.AlignCenter)
        self.label_position.setStyleSheet(
            "color: #4A5568; font-size: 13px; font-weight: 600;"
            "background: #FFFFFF; border: 1px solid #DDE1E8;"
            "border-radius: 6px; padding: 4px 10px;"
        )

        layout.addWidget(self.btn_rewind)
        layout.addWidget(self.btn_play_pause)
        layout.addWidget(self.btn_forward)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label_position)

        self.btn_play_pause.clicked.connect(self._on_play_pause)
        self.btn_rewind.clicked.connect(lambda: self.step_backward.emit(30))
        self.btn_forward.clicked.connect(lambda: self.step_forward.emit(30))
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value_changed)

    @staticmethod
    def _icon_from_style(standard_pixmap):
        return QIcon(QApplication.style().standardIcon(standard_pixmap))

    def _on_play_pause(self):
        self.play_pause_clicked.emit()

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        self.seek_requested.emit(self.slider.value())

    def _on_slider_value_changed(self, value):
        if not self._slider_dragging:
            return
        self.label_position.setText(f"{value} / {self._total_frames}")

    def set_total_frames(self, total: int):
        self._total_frames = total
        if total > 0:
            self.slider.setMaximum(total - 1)
        else:
            self.slider.setMaximum(0)

    def set_position(self, current: int, total: int = -1):
        if total > 0:
            self.set_total_frames(total)
        if not self._slider_dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(min(current, self.slider.maximum()))
            self.slider.blockSignals(False)
        self.label_position.setText(f"{current} / {self._total_frames}")

    def set_playing(self, playing: bool):
        self._playing = playing
        self.btn_play_pause.setIcon(
            self._icon_from_style(
                QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
            )
        )


class VideoPlayerWidget(QWidget):
    seek_requested = pyqtSignal(int)
    play_pause_clicked = pyqtSignal()
    step_forward = pyqtSignal(int)
    step_backward = pyqtSignal(int)

    def __init__(self, video_panel, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_panel = video_panel
        self.video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_panel, 1)

        self.control_bar = VideoControlBar()
        self.control_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.control_bar)

        self.control_bar.seek_requested.connect(self.seek_requested)
        self.control_bar.play_pause_clicked.connect(self.play_pause_clicked)
        self.control_bar.step_forward.connect(self.step_forward)
        self.control_bar.step_backward.connect(self.step_backward)

    def update_frame(self, q_img, source_size=None):
        self.video_panel.update_frame(q_img, source_size)

    def set_position(self, current: int, total: int = -1):
        self.control_bar.set_position(current, total)

    def set_playing(self, playing: bool):
        self.control_bar.set_playing(playing)
