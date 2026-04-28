from pathlib import Path

import sys
import subprocess
import shlex
import yaml
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .annotation_panel import AnnotationPanel
from .control_panel import ControlPanel
from .event_table_panel import EventTablePanel
from .log_panel import LogPanel
from .stats_panel import StatsPanel
from .trend_panel import TrendChartWidget
from .video_panel import VideoPanel
from .worker import WorkerThread


class ToolOptionsDialog(QDialog):
    def __init__(self, title: str, fields: list[tuple[str, QWidget]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._fields = {}

        for label, widget in fields:
            self._fields[label] = widget
            form.addRow(label, widget)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self, label: str):
        widget = self._fields[label]
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        return None


class ToolRunnerThread(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, cmd: list[str], cwd: str | None = None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd

    def run(self):
        try:
            self.log.emit(f"运行命令: {' '.join(self.cmd)}")
            proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.cwd)
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log.emit(line.rstrip())
            proc.wait()
            self.finished.emit(proc.returncode)
        except Exception as exc:
            self.log.emit(f"工具运行异常: {exc}")
            self.finished.emit(-1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频行人客流量统计与分析系统 - 中期答辩版")
        self.resize(1200, 800)

        self.current_config_path = None
        self.current_config = self._default_config()
        self.tool_runners = []

        self.worker = WorkerThread()
        self.init_ui()
        self._init_menus()
        self.connect_signals()
        self.apply_runtime_params(self.annotation_panel.get_params())

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        self.video_panel = VideoPanel()
        left_layout.addWidget(self.video_panel, 7)

        self.tabs = QTabWidget()
        self.log_panel = LogPanel()
        self.trend_panel = TrendChartWidget()
        self.event_table_panel = EventTablePanel()

        self.tabs.addTab(self.log_panel, "运行日志")
        self.tabs.addTab(self.trend_panel, "流量趋势图")
        self.tabs.addTab(self.event_table_panel, "事件表")

        sys_info = QWidget()
        sys_info_layout = QVBoxLayout(sys_info)
        sys_info_label = QLabel("系统信息面板（答辩版保留位）")
        sys_info_label.setAlignment(Qt.AlignCenter)
        sys_info_layout.addWidget(sys_info_label)
        self.tabs.addTab(sys_info, "系统信息")

        left_layout.addWidget(self.tabs, 3)

        right_layout = QVBoxLayout()
        self.control_panel = ControlPanel()
        self.stats_panel = StatsPanel()
        self.annotation_panel = AnnotationPanel()

        right_layout.addWidget(self.control_panel)
        right_layout.addWidget(self.stats_panel)
        right_layout.addWidget(self.annotation_panel)
        right_layout.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([860, 380])

        main_layout.addWidget(splitter)

    def _init_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        run_menu = menubar.addMenu("运行")
        config_menu = menubar.addMenu("配置")
        help_menu = menubar.addMenu("帮助")

        action_open_video = QAction("打开视频", self)
        action_load_config = QAction("加载配置", self)
        action_exit = QAction("退出", self)
        file_menu.addAction(action_open_video)
        file_menu.addAction(action_load_config)
        file_menu.addSeparator()
        file_menu.addAction(action_exit)

        action_start = QAction("开始", self)
        action_pause = QAction("暂停/继续", self)
        action_stop = QAction("停止", self)
        run_menu.addAction(action_start)
        run_menu.addAction(action_pause)
        run_menu.addAction(action_stop)

        action_save_yaml = QAction("保存YAML", self)
        action_load_yaml = QAction("加载YAML", self)
        config_menu.addAction(action_save_yaml)
        config_menu.addAction(action_load_yaml)

        action_about = QAction("关于", self)
        help_menu.addAction(action_about)

        action_open_video.triggered.connect(self.control_panel.select_video)
        action_load_config.triggered.connect(self.load_yaml_config)
        action_exit.triggered.connect(self.close)
        action_start.triggered.connect(self.start_processing)
        action_pause.triggered.connect(self.toggle_pause)
        action_stop.triggered.connect(self.stop_processing)
        action_save_yaml.triggered.connect(self.save_yaml_config)
        action_load_yaml.triggered.connect(self.load_yaml_config)
        action_about.triggered.connect(self.show_about)

    def connect_signals(self):
        self.control_panel.btn_start.clicked.connect(self.start_processing)
        self.control_panel.btn_pause.clicked.connect(self.toggle_pause)
        self.control_panel.btn_stop.clicked.connect(self.stop_processing)
        self.control_panel.btn_select_camera.clicked.connect(self.use_camera_source)

        self.worker.new_frame.connect(self.video_panel.update_frame)
        self.worker.stats_updated.connect(self.stats_panel.update_stats)
        self.worker.trend_updated.connect(self.on_trend_updated)
        self.worker.event_emitted.connect(self.on_event_emitted)
        self.worker.log_message.connect(self.log_panel.append_log)
        self.worker.finished.connect(self.on_process_finished)

        self.annotation_panel.params_changed.connect(self.apply_runtime_params)
        self.annotation_panel.draw_mode_changed.connect(self.video_panel.set_draw_mode)
        self.annotation_panel.clear_requested.connect(self.video_panel.clear_annotations)
        self.annotation_panel.save_config_requested.connect(self.save_yaml_config)

        self.video_panel.roi_changed.connect(self.on_annotations_changed)
        self.video_panel.line_changed.connect(self.on_annotations_changed)

        self.control_panel.video_selected.connect(self.on_video_selected)
        # 模型导出/量化/benchmark 信号
        self.control_panel.export_onnx_requested.connect(self.on_export_onnx_requested)
        self.control_panel.quantize_onnx_requested.connect(self.on_quantize_onnx_requested)
        self.control_panel.benchmark_onnx_requested.connect(self.on_benchmark_onnx_requested)

    def start_processing(self):
        video_path = self.control_panel.get_video_path()
        if not video_path:
            self.log_panel.append_log("请先选择视频源")
            return

        self.worker.set_config(self.current_config)
        self.worker.set_source(video_path)
        self.worker.set_runtime_params(self.annotation_panel.get_params())
        self.worker.set_annotations(self.video_panel._roi_points, self.video_panel._line_points)

        self.trend_panel.reset()
        self.event_table_panel.reset()

        self.worker.start()
        self.control_panel.btn_start.setEnabled(False)
        self.control_panel.btn_stop.setEnabled(True)
        self.control_panel.btn_pause.setEnabled(True)
        self.control_panel.btn_pause.setText("暂停")
        self.log_panel.append_log("线程启动")

    def stop_processing(self):
        self.worker.stop()
        self.log_panel.append_log("正在停止")

    def toggle_pause(self):
        if not self.worker.isRunning():
            return
        if self.worker.paused:
            self.worker.resume()
            self.control_panel.btn_pause.setText("暂停")
            self.log_panel.append_log("继续处理")
        else:
            self.worker.pause()
            self.control_panel.btn_pause.setText("继续")
            self.log_panel.append_log("已暂停")

    def on_process_finished(self):
        self.control_panel.btn_start.setEnabled(True)
        self.control_panel.btn_stop.setEnabled(False)
        self.control_panel.btn_pause.setEnabled(False)
        self.control_panel.btn_pause.setText("暂停")

    def on_trend_updated(self, payload):
        if bool(payload.get("reset", False)):
            self.trend_panel.reset()
            return
        self.trend_panel.update_trend(payload)

    def on_event_emitted(self, event):
        if bool(event.get("reset", False)):
            self.event_table_panel.reset()
            return
        self.event_table_panel.add_event_record(event)

    def on_video_selected(self, path):
        self.log_panel.append_log(f"已选择视频源: {path}")
        if path and str(path) != "0":
            import cv2
            from PyQt5.QtGui import QImage
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    target_size = self._compute_adaptive_size(frame.shape[1], frame.shape[0])
                    if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                        frame = cv2.resize(frame, target_size)
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    # Keep a reference to rgb_image.data so it doesn't get garbage collected
                    # Or use image.copy()
                    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    self.video_panel.update_frame(q_img)
                cap.release()

    def use_camera_source(self):
        self.control_panel.set_video_path("0")
        self.log_panel.append_log("视频源切换为摄像头 0")
        import cv2
        from PyQt5.QtGui import QImage
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                target_size = self._compute_adaptive_size(frame.shape[1], frame.shape[0])
                if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                    frame = cv2.resize(frame, target_size)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                self.video_panel.update_frame(q_img)
            cap.release()

    def _compute_adaptive_size(self, orig_w, orig_h, max_width=1280, max_height=800):
        if orig_w <= 0 or orig_h <= 0:
            return max_width, max_height
        scale = min(1.0, min(max_width / orig_w, max_height / orig_h))
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        return new_w, new_h

    def apply_runtime_params(self, params):
        self.worker.set_runtime_params(params)
        self.video_panel.set_show_flags(
            show_roi=params.get("show_roi", True),
            show_line=params.get("show_line", True),
        )

    def on_annotations_changed(self, _):
        self.worker.set_annotations(self.video_panel._roi_points, self.video_panel._line_points)

    def load_yaml_config(self):
        default_dir = str(Path(__file__).resolve().parents[1] / "config")
        path, _ = QFileDialog.getOpenFileName(self, "加载YAML配置", default_dir, "YAML Files (*.yaml *.yml)")
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", f"配置读取失败: {exc}")
            return

        self.current_config_path = path
        self.current_config = cfg
        self.worker.set_config(cfg)

        src = str(cfg.get("source", ""))
        if src:
            self.control_panel.set_video_path(src)

        detector_cfg = cfg.get("detector", {})
        vis_cfg = cfg.get("visualization", {})
        privacy_cfg = cfg.get("privacy", {})
        face_blur_cfg = privacy_cfg.get("face_blur", {})
        counting_cfg = cfg.get("counting", {})
        line_cfg = counting_cfg.get("line", {})
        roi_cfg = counting_cfg.get("roi", {})

        params = {
            "conf": float(detector_cfg.get("conf", 0.5)),
            "iou": float(detector_cfg.get("iou", 0.45)),
            "show_trail": bool(vis_cfg.get("show_trail", True)),
            "show_roi": bool(vis_cfg.get("draw_roi", True)),
            "show_line": bool(vis_cfg.get("draw_line", True)),
            "show_heatmap": bool(vis_cfg.get("show_heatmap", True)),
            "face_blur_enabled": bool(face_blur_cfg.get("enabled", False)),
        }
        self.annotation_panel.set_params(params)

        roi_points = roi_cfg.get("polygon", [])
        line_points = line_cfg.get("points", [])
        self.video_panel.set_annotations(roi_points=roi_points, line_points=line_points)

        self.log_panel.append_log(f"配置已加载: {path}")

    def save_yaml_config(self):
        default_dir = str(Path(__file__).resolve().parents[1] / "config")
        default_name = "pedestrian_gui_saved.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存YAML配置",
            str(Path(default_dir) / default_name),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return

        cfg = dict(self.current_config) if isinstance(self.current_config, dict) else self._default_config()
        params = self.annotation_panel.get_params()

        cfg["source"] = self.control_panel.get_video_path() or cfg.get("source", "")

        detector_cfg = cfg.setdefault("detector", {})
        detector_cfg["conf"] = float(params["conf"])
        detector_cfg["iou"] = float(params["iou"])

        vis_cfg = cfg.setdefault("visualization", {})
        vis_cfg["show_trail"] = bool(params["show_trail"])
        vis_cfg["draw_roi"] = bool(params["show_roi"])
        vis_cfg["draw_line"] = bool(params["show_line"])
        vis_cfg["show_heatmap"] = bool(params.get("show_heatmap", True))

        privacy_cfg = cfg.setdefault("privacy", {})
        face_blur_cfg = privacy_cfg.setdefault("face_blur", {})
        face_blur_cfg["enabled"] = bool(params.get("face_blur_enabled", False))

        counting_cfg = cfg.setdefault("counting", {})
        roi_cfg = counting_cfg.setdefault("roi", {})
        line_cfg = counting_cfg.setdefault("line", {})
        roi_cfg["polygon"] = [list(map(int, p)) for p in self.video_panel._roi_points]
        line_cfg["points"] = [list(map(int, p)) for p in self.video_panel._line_points]

        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"配置写入失败: {exc}")
            return

        self.current_config_path = path
        self.current_config = cfg
        self.worker.set_config(cfg)
        self.log_panel.append_log(f"配置已保存: {path}")

    def show_about(self):
        QMessageBox.information(
            self,
            "关于",
            "基于YOLO的视频行人流量统计分析系统\n中期答辩轻量级GUI演示版",
        )

    def _default_config(self):
        return {
            "source": "",
            "detector": {"conf": 0.5, "iou": 0.45},
            "visualization": {
                "show_trail": True,
                "draw_roi": True,
                "draw_line": True,
                "show_heatmap": True,
                "heatmap_alpha": 0.35,
                "heatmap_sigma": 40,
                "heatmap_interval": 5,
            },
            "privacy": {
                "face_blur": {
                    "enabled": False,
                    "model_path": "models/yolov8n-face.pt",
                    "conf": 0.4,
                    "blur_type": "gaussian",
                    "blur_kernel": 25,
                }
            },
            "counting": {
                "roi": {"enabled": True, "polygon": []},
                "line": {"enabled": True, "points": []},
            },
        }

    def _start_tool_runner(self, cmd: list[str], cwd: str | None = None):
        runner = ToolRunnerThread(cmd, cwd=cwd)
        runner.log.connect(self.log_panel.append_log)
        runner.finished.connect(lambda rc, ref=runner: self._on_tool_finished(ref, rc))
        self.tool_runners.append(runner)
        runner.start()

    def _on_tool_finished(self, runner: ToolRunnerThread, return_code: int):
        self.log_panel.append_log(f"工具退出，返回码: {return_code}")
        if runner in self.tool_runners:
            self.tool_runners.remove(runner)
        runner.deleteLater()

    def _show_tool_dialog(self, title: str, fields: list[tuple[str, QWidget]]):
        dialog = ToolOptionsDialog(title, fields, self)
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return None
        return dialog

    def on_export_onnx_requested(self):
        weights, _ = QFileDialog.getOpenFileName(self, "选择权重文件 (.pt)", "", "PyTorch Weights (*.pt *.pth)")
        if not weights:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", str(Path(weights).parent))
        if not out_dir:
            return

        imgsz = QSpinBox()
        imgsz.setRange(32, 4096)
        imgsz.setSingleStep(32)
        imgsz.setValue(800)

        opset = QSpinBox()
        opset.setRange(7, 25)
        opset.setValue(17)

        simplify = QCheckBox("启用 ONNX 简化")
        simplify.setChecked(True)

        dynamic = QCheckBox("动态 batch/shape")
        dynamic.setChecked(False)

        half = QCheckBox("导出 FP16")
        half.setChecked(False)

        dialog = self._show_tool_dialog(
            "ONNX 导出选项",
            [
                ("imgsz", imgsz),
                ("opset", opset),
                ("simplify", simplify),
                ("dynamic", dynamic),
                ("half", half),
            ],
        )
        if dialog is None:
            return

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "export_onnx.py"),
            "--weights",
            weights,
            "--output-dir",
            out_dir,
            "--imgsz",
            str(dialog.value("imgsz")),
            "--opset",
            str(dialog.value("opset")),
        ]
        if dialog.value("simplify"):
            cmd.append("--simplify")
        if dialog.value("dynamic"):
            cmd.append("--dynamic")
        if dialog.value("half"):
            cmd.append("--half")
        self._start_tool_runner(cmd, cwd=str(Path(__file__).resolve().parents[1]))

    def on_quantize_onnx_requested(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择输入 ONNX 模型", "", "ONNX Model (*.onnx)")
        if not model_path:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "保存量化后模型为", str(Path(model_path).with_suffix(".quant.onnx")), "ONNX Model (*.onnx)")
        if not out_path:
            return

        mode = QComboBox()
        mode.addItem("FP16", "fp16")
        mode.addItem("INT8", "int8")

        input_name = QLineEdit("images")
        calibration_dir = QLineEdit("")

        dialog = self._show_tool_dialog(
            "ONNX 量化选项",
            [
                ("mode", mode),
                ("input_name", input_name),
                ("calibration_dir", calibration_dir),
            ],
        )
        if dialog is None:
            return

        quant_mode = str(dialog.value("mode"))
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "quantize_onnx.py"),
            "--input",
            model_path,
            "--output",
            out_path,
            "--mode",
            quant_mode,
            "--input-name",
            dialog.value("input_name") or "images",
        ]
        if quant_mode == "int8":
            calib_dir = dialog.value("calibration_dir")
            if not calib_dir:
                self.log_panel.append_log("取消 INT8 量化：未选择校准目录")
                return
            cmd.extend(["--calibration-data", calib_dir])

        self._start_tool_runner(cmd, cwd=str(Path(__file__).resolve().parents[1]))

    def on_benchmark_onnx_requested(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 模型", "", "ONNX Model (*.onnx)")
        if not model_path:
            return

        source = QLineEdit(self.control_panel.get_video_path() or "0")
        imgsz = QSpinBox()
        imgsz.setRange(32, 4096)
        imgsz.setSingleStep(32)
        imgsz.setValue(800)

        warmup = QSpinBox()
        warmup.setRange(0, 10000)
        warmup.setValue(10)

        limit = QSpinBox()
        limit.setRange(1, 100000)
        limit.setValue(300)

        providers = QLineEdit("")

        dialog = self._show_tool_dialog(
            "ONNX Benchmark 选项",
            [
                ("source", source),
                ("imgsz", imgsz),
                ("warmup", warmup),
                ("limit", limit),
                ("providers", providers),
            ],
        )
        if dialog is None:
            return

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "benchmark_onnx.py"),
            "--model",
            model_path,
            "--source",
            dialog.value("source") or "0",
            "--imgsz",
            str(dialog.value("imgsz")),
            "--warmup",
            str(dialog.value("warmup")),
            "--limit",
            str(dialog.value("limit")),
        ]
        provider_text = dialog.value("providers")
        if provider_text:
            cmd.extend(["--providers", provider_text])

        self._start_tool_runner(cmd, cwd=str(Path(__file__).resolve().parents[1]))
