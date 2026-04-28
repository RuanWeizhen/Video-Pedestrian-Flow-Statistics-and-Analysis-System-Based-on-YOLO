from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

class AnnotationPanel(QWidget):
    params_changed = pyqtSignal(dict)
    draw_mode_changed = pyqtSignal(str)
    clear_requested = pyqtSignal()
    save_config_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        group = QGroupBox("参数与交互标注区")
        vbox = QVBoxLayout()

        # 参数调节
        hbox_conf = QHBoxLayout()
        self.lbl_conf = QLabel("Conf 置信度:")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.5)

        self.lbl_iou = QLabel("IoU 阈值:")
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.45)

        hbox_conf.addWidget(self.lbl_conf)
        hbox_conf.addWidget(self.spin_conf)
        hbox_conf.addWidget(self.lbl_iou)
        hbox_conf.addWidget(self.spin_iou)
        vbox.addLayout(hbox_conf)

        # 视效开关
        self.chk_roi = QCheckBox("显示 ROI 区域")
        self.chk_roi.setChecked(True)
        self.chk_line = QCheckBox("显示计数线")
        self.chk_line.setChecked(True)
        self.chk_trail = QCheckBox("显示轨迹")
        self.chk_trail.setChecked(True)
        self.chk_heatmap = QCheckBox("显示热力图")
        self.chk_heatmap.setChecked(True)
        self.chk_face_blur = QCheckBox("人脸模糊（隐私保护）")
        self.chk_face_blur.setChecked(False)

        vbox.addWidget(self.chk_roi)
        vbox.addWidget(self.chk_line)
        vbox.addWidget(self.chk_trail)
        vbox.addWidget(self.chk_heatmap)
        vbox.addWidget(self.chk_face_blur)

        # 标注按钮
        self.btn_draw_roi = QPushButton("绘制 ROI")
        self.btn_draw_line = QPushButton("绘制 Line")
        self.btn_clear = QPushButton("清空标注")
        self.btn_save_cfg = QPushButton("保存配置")

        vbox.addWidget(self.btn_draw_roi)
        vbox.addWidget(self.btn_draw_line)
        vbox.addWidget(self.btn_clear)
        vbox.addWidget(self.btn_save_cfg)

        group.setLayout(vbox)
        layout.addWidget(group)

        self.spin_conf.valueChanged.connect(self._emit_params)
        self.spin_iou.valueChanged.connect(self._emit_params)
        self.chk_roi.toggled.connect(self._emit_params)
        self.chk_line.toggled.connect(self._emit_params)
        self.chk_trail.toggled.connect(self._emit_params)
        self.chk_heatmap.toggled.connect(self._emit_params)
        self.chk_face_blur.toggled.connect(self._emit_params)

        self.btn_draw_roi.clicked.connect(lambda: self.draw_mode_changed.emit("roi"))
        self.btn_draw_line.clicked.connect(lambda: self.draw_mode_changed.emit("line"))
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        self.btn_save_cfg.clicked.connect(self.save_config_requested.emit)

    def _emit_params(self):
        self.params_changed.emit(self.get_params())

    def get_params(self):
        return {
            "conf": float(self.spin_conf.value()),
            "iou": float(self.spin_iou.value()),
            "show_roi": bool(self.chk_roi.isChecked()),
            "show_line": bool(self.chk_line.isChecked()),
            "show_trail": bool(self.chk_trail.isChecked()),
            "show_heatmap": bool(self.chk_heatmap.isChecked()),
            "face_blur_enabled": bool(self.chk_face_blur.isChecked()),
        }

    def set_params(self, params):
        self.spin_conf.blockSignals(True)
        self.spin_iou.blockSignals(True)
        self.chk_roi.blockSignals(True)
        self.chk_line.blockSignals(True)
        self.chk_trail.blockSignals(True)
        self.chk_heatmap.blockSignals(True)
        self.chk_face_blur.blockSignals(True)

        self.spin_conf.setValue(float(params.get("conf", self.spin_conf.value())))
        self.spin_iou.setValue(float(params.get("iou", self.spin_iou.value())))
        self.chk_roi.setChecked(bool(params.get("show_roi", self.chk_roi.isChecked())))
        self.chk_line.setChecked(bool(params.get("show_line", self.chk_line.isChecked())))
        self.chk_trail.setChecked(bool(params.get("show_trail", self.chk_trail.isChecked())))
        self.chk_heatmap.setChecked(bool(params.get("show_heatmap", self.chk_heatmap.isChecked())))
        self.chk_face_blur.setChecked(bool(params.get("face_blur_enabled", self.chk_face_blur.isChecked())))

        self.spin_conf.blockSignals(False)
        self.spin_iou.blockSignals(False)
        self.chk_roi.blockSignals(False)
        self.chk_line.blockSignals(False)
        self.chk_trail.blockSignals(False)
        self.chk_heatmap.blockSignals(False)
        self.chk_face_blur.blockSignals(False)

        self._emit_params()
