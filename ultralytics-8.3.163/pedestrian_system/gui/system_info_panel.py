from PyQt5.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget


class SystemInfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        group = QGroupBox("系统信息")
        form = QFormLayout()

        self.lbl_model_path = QLabel("-")
        self.lbl_video_resolution = QLabel("-")
        self.lbl_device = QLabel("-")
        self.lbl_current_frame = QLabel("-")
        self.lbl_total_frames = QLabel("-")
        self.lbl_total_people = QLabel("-")
        self.lbl_runtime = QLabel("-")
        self.lbl_avg_fps = QLabel("-")
        self.lbl_detector_type = QLabel("-")
        self.lbl_tracker_type = QLabel("-")
        self.lbl_onnx_enabled = QLabel("-")

        for label in (
            self.lbl_model_path,
            self.lbl_video_resolution,
            self.lbl_device,
            self.lbl_current_frame,
            self.lbl_total_frames,
            self.lbl_total_people,
            self.lbl_runtime,
            self.lbl_avg_fps,
            self.lbl_detector_type,
            self.lbl_tracker_type,
            self.lbl_onnx_enabled,
        ):
            label.setWordWrap(True)

        form.addRow("模型路径", self.lbl_model_path)
        form.addRow("视频分辨率", self.lbl_video_resolution)
        form.addRow("当前设备", self.lbl_device)
        form.addRow("当前帧号", self.lbl_current_frame)
        form.addRow("总帧数", self.lbl_total_frames)
        form.addRow("总通行人数", self.lbl_total_people)
        form.addRow("运行时间", self.lbl_runtime)
        form.addRow("平均 FPS", self.lbl_avg_fps)
        form.addRow("检测模型类型", self.lbl_detector_type)
        form.addRow("跟踪算法类型", self.lbl_tracker_type)
        form.addRow("ONNX 是否启用", self.lbl_onnx_enabled)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def update_info(self, info: dict):
        self.lbl_model_path.setText(str(info.get("model_path", "-")))
        self.lbl_video_resolution.setText(str(info.get("video_resolution", "-")))
        self.lbl_device.setText(str(info.get("device", "-")))
        self.lbl_current_frame.setText(str(info.get("current_frame", "-")))
        self.lbl_total_frames.setText(str(info.get("total_frames", "-")))
        self.lbl_total_people.setText(str(info.get("total_people", "-")))
        self.lbl_runtime.setText(str(info.get("runtime", "-")))
        self.lbl_avg_fps.setText(str(info.get("avg_fps", "-")))
        self.lbl_detector_type.setText(str(info.get("detector_type", "-")))
        self.lbl_tracker_type.setText(str(info.get("tracker_type", "-")))
        self.lbl_onnx_enabled.setText(str(info.get("onnx_enabled", "-")))
