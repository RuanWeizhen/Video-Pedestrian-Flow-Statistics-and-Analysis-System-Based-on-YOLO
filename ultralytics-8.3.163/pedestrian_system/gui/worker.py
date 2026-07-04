from __future__ import annotations

import copy
from datetime import datetime
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
import threading
import queue

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

try:
    import torch
except Exception:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counting.line_counter import LineCounter
from counting.zone_counter import ZoneCounterManager
from privacy.face_blur import FaceBlur
from tracker.deepsort_wrapper import DeepSortTracker
from utils.config import load_config
from utils.db_manager import DatabaseManager
from utils.filters import StaticTrackFilter, filter_tracks_by_roi
from utils.coordinate_transform import frame_points_to_frame_points
from utils.io_utils import EventLogger, ensure_dir
from utils.paths import external_resource_path, resource_path, writable_path
from utils.perf_profiler import PerformanceProfiler
from utils.system_optimizer import apply_system_optimizations
from utils.visualization import (
    draw_counters_panel,
    draw_line_counter,
    draw_track_history,
    draw_tracks,
    draw_zone_counters,
)


# ============================================================================
# WorkerThread — 核心处理流水线（检测 → 跟踪 → 过滤 → 计数）
# ============================================================================
#
# 这是整个系统的核心引擎，运行在一个独立的 QThread 中，不阻塞 GUI 主线程。
#
# ┌─────────────────────────────────────────────────────────────────┐
# │                     主循环 (每帧执行)                            │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                 │
# │  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────┐ │
# │  │ 异步读取  │ → │ YOLO 检测    │ → │ DeepSORT 跟踪           │ │
# │  │ 视频帧    │   │ (detector)   │   │ (tracker)               │ │
# │  └──────────┘   └──────────────┘   └─────────────────────────┘ │
# │                                              ↓                  │
# │                                    ┌─────────────────┐         │
# │                                    │ ROI 区域过滤    │         │
# │                                    │ (仅保留区域内   │         │
# │                                    │  的目标)        │         │
# │                                    └─────────────────┘         │
# │                                              ↓                  │
# │                                    ┌─────────────────┐         │
# │                                    │ 静态目标过滤    │         │
# │                                    │ (排除模特/海报) │         │
# │                                    └─────────────────┘         │
# │                                              ↓                  │
# │                          ┌─────────────────────┐               │
# │                          │ 计数逻辑             │               │
# │                          │ ├─ LineCounter      │               │
# │                          │ │  (跨线计数 up/down)│               │
# │                          │ └─ ZoneCounterManager│              │
# │                          │    (区域计数 enter/  │               │
# │                          │     leave)           │               │
# │                          └─────────────────────┘               │
# │                                    ↓                            │
# │                          ┌─────────────────────┐               │
# │                          │ 结果输出             │               │
# │                          │ ├─ 数据库写入        │               │
# │                          │ ├─ CSV 事件日志      │               │
# │                          │ ├─ 轨迹点记录        │               │
# │                          │ └─ 热力图累积        │               │
# │                          └─────────────────────┘               │
# │                                    ↓                            │
# │                          ┌─────────────────────┐               │
# │                          │ 可视化渲染           │               │
# │                          │ ├─ 绘制检测框        │               │
# │                          │ ├─ 绘制轨迹线        │               │
# │                          │ ├─ 绘制计数线/区     │               │
# │                          │ ├─ 绘制热力图叠加    │               │
# │                          │ └─ 发射 QImage 到 GUI│               │
# │                          └─────────────────────┘               │
# │                                                                 │
# └─────────────────────────────────────────────────────────────────┘
#
# 性能优化措施：
#   - 异步帧读取（后台线程预读，解耦 I/O 与推理）
#   - CUDA 推理优化（torch.inference_mode、cudnn.benchmark、TF32）
#   - UI 节流（stats/trend/frame 发射均按频率限制，避免 GUI 过载）
#   - 推理尺寸自适应（限制最大分辨率，保持实时性）
# ============================================================================


class WorkerThread(QThread):
    new_frame = pyqtSignal(QImage)
    stats_updated = pyqtSignal(dict)
    trend_updated = pyqtSignal(dict)
    event_emitted = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    finished = pyqtSignal()
    frame_position = pyqtSignal(int, int)
    play_state_changed = pyqtSignal(bool)
    perf_sample = pyqtSignal(dict)
    video_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.paused = False
        self.video_path = None
        self.config_path = external_resource_path("config/pedestrian_demo.yaml")
        self.config_data = None
        self.conf = 0.5
        self.iou = 0.45
        self.show_trail = True
        self.show_roi = True
        self.show_line = True
        self.show_heatmap = True
        self.face_blur_enabled = False
        self.draw_count_points = True
        self.heatmap_alpha = 0.35
        self.heatmap_sigma = 40
        self.heatmap_interval = 5
        self.roi_points = []
        self.line_points = []
        self._seek_frame = -1
        self._detector = None
        self._detector_signature = None
        self._detector_lock = threading.RLock()
        self._detector_warmup_thread = None

    def set_source(self, path):
        self.video_path = path

    def set_config(self, cfg):
        self.config_data = copy.deepcopy(cfg) if isinstance(cfg, dict) else None

    def set_runtime_params(self, params):
        self.conf = float(params.get("conf", self.conf))
        self.iou = float(params.get("iou", self.iou))
        self.show_trail = bool(params.get("show_trail", self.show_trail))
        self.show_roi = bool(params.get("show_roi", self.show_roi))
        self.show_line = bool(params.get("show_line", self.show_line))
        self.show_heatmap = bool(params.get("show_heatmap", self.show_heatmap))
        self.face_blur_enabled = bool(params.get("face_blur_enabled", self.face_blur_enabled))
        self.draw_count_points = bool(params.get("draw_count_points", self.draw_count_points))

    def set_annotations(self, roi_points, line_points):
        self.roi_points = [tuple(map(int, p)) for p in roi_points]
        self.line_points = [tuple(map(int, p)) for p in line_points[:2]]

    def pause(self):
        self.paused = True
        self.play_state_changed.emit(False)

    def resume(self):
        self.paused = False
        self.play_state_changed.emit(True)

    def seek_to_frame(self, frame_number: int):
        self._seek_frame = max(0, int(frame_number))

    def _deep_update(self, base, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _build_runtime_config(self):
        fallback_cfg = load_config(str(self.config_path))
        cfg = copy.deepcopy(fallback_cfg)

        if isinstance(self.config_data, dict):
            cfg = self._deep_update(cfg, copy.deepcopy(self.config_data))

        cfg.setdefault("detector", {})["conf"] = float(self.conf)
        cfg.setdefault("detector", {})["iou"] = float(self.iou)

        vis_cfg = cfg.setdefault("visualization", {})
        vis_cfg["show_trail"] = bool(self.show_trail)
        vis_cfg["draw_roi"] = bool(self.show_roi)
        vis_cfg["draw_line"] = bool(self.show_line)
        vis_cfg["show_heatmap"] = bool(self.show_heatmap)
        vis_cfg.setdefault("heatmap_alpha", self.heatmap_alpha)
        vis_cfg.setdefault("heatmap_sigma", self.heatmap_sigma)
        vis_cfg.setdefault("heatmap_interval", self.heatmap_interval)

        privacy_cfg = cfg.setdefault("privacy", {})
        face_blur_cfg = privacy_cfg.setdefault("face_blur", {})
        face_blur_cfg["enabled"] = bool(self.face_blur_enabled)

        debug_cfg = cfg.setdefault("debug", {})
        debug_cfg["draw_count_points"] = bool(self.draw_count_points)

        counting_cfg = cfg.setdefault("counting", {})
        roi_cfg = counting_cfg.setdefault("roi", {})
        line_cfg = counting_cfg.setdefault("line", {})

        if self.roi_points:
            roi_cfg["enabled"] = True
            roi_cfg["polygon"] = [list(map(int, p)) for p in self.roi_points]

        if len(self.line_points) == 2:
            line_cfg["enabled"] = True
            line_cfg["points"] = [list(map(int, p)) for p in self.line_points]

        if self.video_path is not None and self.video_path != "":
            cfg["source"] = self.video_path

        cfg.setdefault("output_dir", "outputs/gui_run")
        cfg.setdefault("output", {})
        cfg["output"].setdefault("events_csv_name", "events_gui.csv")
        return cfg

    def _parse_source(self, source_value):
        if isinstance(source_value, int):
            return source_value
        if isinstance(source_value, str) and source_value.isdigit():
            return int(source_value)
        return source_value

    def _source_label(self, source_value) -> str:
        if isinstance(source_value, int):
            return f"摄像头 {source_value}"
        if isinstance(source_value, str) and source_value.isdigit():
            return f"摄像头 {source_value}"
        if isinstance(source_value, str) and source_value:
            return Path(source_value).name
        return "未知视频源"

    def _line_up_down(self, line_counter):
        if line_counter is None:
            return 0, 0
        counts = dict(line_counter.counts)
        up = int(counts.get("up", 0))
        down = int(counts.get("down", 0))
        if up == 0 and down == 0 and len(counts) >= 2:
            values = list(counts.values())
            up, down = int(values[0]), int(values[1])
        return up, down

    def _compute_adaptive_size(self, orig_w, orig_h, max_width=1280, max_height=800):
        if orig_w <= 0 or orig_h <= 0:
            return max_width, max_height, 1.0
        scale = min(1.0, min(max_width / orig_w, max_height / orig_h))
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        return new_w, new_h, scale

    def _detector_signature_from_cfg(self, detector_cfg: dict) -> tuple:
        class_ids = tuple(sorted(detector_cfg.get("class_ids", [0])))
        class_name_map = tuple(sorted((int(k), str(v)) for k, v in dict(detector_cfg.get("class_name_map", {0: "person"})).items()))
        return (
            str(detector_cfg.get("model_path", "")),
            int(detector_cfg.get("imgsz", 640)),
            float(detector_cfg.get("conf", 0.25)),
            float(detector_cfg.get("iou", 0.45)),
            str(detector_cfg.get("device", 0)),
            bool(detector_cfg.get("half", False)),
            class_ids,
            class_name_map,
        )

    def _prepare_detector(self, cfg: dict | None = None):
        cfg = cfg or self._build_runtime_config()
        detector_cfg = dict(cfg.get("detector", {}))
        signature = self._detector_signature_from_cfg(detector_cfg)

        from detector.yolo_detector import YOLODetector

        with self._detector_lock:
            if self._detector is None or self._detector_signature != signature:
                self._detector = YOLODetector(detector_cfg)
                self._detector_signature = signature
            else:
                self._detector.configure(detector_cfg, load_model=False)
            self._detector.apply_runtime_params(self.conf, self.iou)
            return self._detector

    def warmup_detector_async(self):
        cfg = self._build_runtime_config()
        detector_cfg = dict(cfg.get("detector", {}))
        signature = self._detector_signature_from_cfg(detector_cfg)

        with self._detector_lock:
            if self._detector is not None and self._detector_signature == signature:
                return
            if self._detector_warmup_thread is not None and self._detector_warmup_thread.is_alive():
                return

            def _warmup():
                try:
                    self._prepare_detector(cfg)
                except Exception:
                    pass

            self._detector_warmup_thread = threading.Thread(target=_warmup, daemon=True)
            self._detector_warmup_thread.start()

    def _async_frame_reader(self, cap, frame_queue: queue.Queue, stop_event):
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                frame_queue.put((False, None))
                return
            frame_queue.put((True, frame))

    def run(self):
        # ================================================================
        # 主处理循环：每帧执行 检测 → 跟踪 → 过滤 → 计数 → 输出 → 渲染
        # ================================================================
        self.running = True
        self.paused = False

        cap = None
        event_logger = None
        trajectory_buffer = []
        db_manager = None
        session_id = 0
        total_frames = 0
        total_time = 0.0
        up_count = 0
        down_count = 0
        total_count = 0
        _async_read_enabled = False
        _read_stop_event = None
        try:
            cfg = self._build_runtime_config()

            perf_cfg = cfg.get("performance", {})
            max_width = int(perf_cfg.get("max_width", 1280))
            max_height = int(perf_cfg.get("max_height", 800))
            ui_max_fps = float(perf_cfg.get("ui_max_fps", 15.0))
            stats_hz = float(perf_cfg.get("stats_hz", 10.0))
            trend_hz = float(perf_cfg.get("trend_hz", 2.0))
            ui_frame_interval = (1.0 / ui_max_fps) if ui_max_fps > 0 else 0.0
            stats_interval = (1.0 / stats_hz) if stats_hz > 0 else 0.0
            trend_interval = (1.0 / trend_hz) if trend_hz > 0 else 0.0

            # ----- 视频源初始化 -----
            source = self._parse_source(cfg.get("source", self.video_path))
            source_label = self._source_label(source)
            face_blur = FaceBlur(cfg.get("privacy", {}).get("face_blur", {}))

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {source}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 1e-6:
                fps = float(cfg.get("output", {}).get("fallback_fps", 25.0))
            total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            proc_width, proc_height, resize_scale = self._compute_adaptive_size(
                raw_width,
                raw_height,
                max_width=max_width,
                max_height=max_height,
            )
            self.log_message.emit(f"Original: {raw_width}x{raw_height}, fps={fps:.2f}")
            self.log_message.emit(f"Resized: {proc_width}x{proc_height}, scale={resize_scale:.3f}")

            output_dir = Path(cfg.get("output_dir", "outputs/gui_run"))
            ensure_dir(output_dir)
            events_csv_path = output_dir / cfg.get("output", {}).get("events_csv_name", "events_gui.csv")

            detector_cfg = dict(cfg["detector"])
            requested_device = detector_cfg.get("device", 0)
            if torch is not None and str(requested_device).strip().lower() != "cpu" and not torch.cuda.is_available():
                self.log_message.emit("CUDA 不可用，检测器自动切换到 CPU")
                detector_cfg["device"] = "cpu"

            try:
                # ----- 检测器初始化 & CUDA 优化 -----
                detector = self._prepare_detector(cfg)
            except Exception:
                self.log_message.emit("YOLO GPU加载失败，回退到CPU")
                detector_cfg["device"] = "cpu"
                cfg["detector"] = detector_cfg
                detector = self._prepare_detector(cfg)

            # CUDA 推理加速：固定尺寸输入下开启 cudnn benchmark / TF32
            if torch is not None:
                try:
                    if str(detector.device).strip().lower() != "cpu" and torch.cuda.is_available():
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        # 适用于 Ampere+，推理通常更快（精度影响很小）
                        if hasattr(torch, "set_float32_matmul_precision"):
                            torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

            # ----- 跟踪器 & 计数器初始化 -----
            tracker = DeepSortTracker(cfg["tracker"])

            counting_cfg = cfg.get("counting", {})
            roi_cfg = counting_cfg.get("roi", {})
            roi_polygon = frame_points_to_frame_points(
                roi_cfg.get("polygon", []) or [],
                raw_width,
                raw_height,
                proc_width,
                proc_height,
            )
            roi_anchor_point = roi_cfg.get("anchor_point", "bottom_center")
            roi_enabled = bool(roi_cfg.get("enabled", False) and len(roi_polygon) >= 3)
            roi_pts_np = np.asarray(roi_polygon, dtype=np.int32) if roi_enabled else None
            roi_label_pos = None
            if roi_pts_np is not None and roi_pts_np.size >= 2:
                roi_label_pos = (int(roi_pts_np[0][0]) + 8, int(roi_pts_np[0][1]) - 8)

            static_filter_cfg = cfg.get("static_filter", {})
            static_filter_enabled = bool(static_filter_cfg.get("enabled", False))
            static_filter = StaticTrackFilter(static_filter_cfg)
            hide_static_boxes = bool(static_filter_cfg.get("hide_static_boxes", True))

            line_counter = None
            line_cfg = counting_cfg.get("line", {})
            if line_cfg.get("enabled", False) and len(line_cfg.get("points", [])) == 2:
                line_cfg = dict(line_cfg)
                line_cfg["points"] = frame_points_to_frame_points(
                    line_cfg.get("points", []) or [],
                    raw_width,
                    raw_height,
                    proc_width,
                    proc_height,
                )
                line_counter = LineCounter(line_cfg)

            zone_manager = ZoneCounterManager(counting_cfg.get("zones", []))
            event_logger = EventLogger(events_csv_path)

            db_manager = DatabaseManager(writable_path("outputs/traffic.db"))
            db_manager.init_db()
            session_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session_id = db_manager.start_session(
                source_name=source_label,
                source_path=str(source),
                config_path=str(self.config_path),
                conf=float(detector_cfg.get("conf", self.conf)),
                iou=float(detector_cfg.get("iou", self.iou)),
                detector_type=str(detector.summary().get("model_path", "YOLO")),
                started_at=session_started_at,
            )
            session_run_id = str(session_id)
            session_meta = {
                "session_id": session_id,
                "run_id": session_run_id,
                "detect_time": session_started_at,
                "created_at": session_started_at,
            }
            last_db_insert_time = time.time()

            visualization_cfg = cfg.get("visualization", {})
            trail_length = int(visualization_cfg.get("trail_length", 30))
            show_confidence = bool(visualization_cfg.get("show_confidence", True))
            box_thickness = int(visualization_cfg.get("box_thickness", 1))
            draw_count_points = bool(cfg.get("debug", {}).get("draw_count_points", True))
            heatmap_sigma = int(visualization_cfg.get("heatmap_sigma", self.heatmap_sigma))
            heatmap_alpha = float(visualization_cfg.get("heatmap_alpha", self.heatmap_alpha))
            heatmap_interval = max(1, int(visualization_cfg.get("heatmap_interval", self.heatmap_interval)))

            heatmap_accum = np.zeros((proc_height, proc_width), dtype=np.float32)
            heatmap_overlay_cache = None
            _display_rgba = np.zeros((proc_height, proc_width, 3), dtype=np.uint8)

            qimage_bgr_format = QImage.Format_BGR888 if hasattr(QImage, "Format_BGR888") else None

            track_history = defaultdict(lambda: deque(maxlen=trail_length))
            trajectory_buffer = []
            flow_stats = defaultdict(lambda: {"up": 0, "down": 0})

            last_frame_emit = 0.0
            last_stats_emit = 0.0
            last_trend_emit = 0.0
            last_trend_minute = None
            last_trend_up = None
            last_trend_down = None

            frame_idx = 0
            total_time = 0.0
            total_frames = 0
            up_count = 0
            down_count = 0
            total_count = 0
            self.event_emitted.emit({"reset": True})
            self.trend_updated.emit({"reset": True})

            self.log_message.emit(f"开始处理视频: {source}")

            perf = PerformanceProfiler()
            sp_optimize = apply_system_optimizations()
            if "cpu_affinity" in sp_optimize:
                self.log_message.emit(f"[优化] CPU亲和性: {sp_optimize['cpu_affinity']}")
            if "gpu_memory" in sp_optimize:
                gpu_info = sp_optimize["gpu_memory"]
                if isinstance(gpu_info, dict):
                    self.log_message.emit(f"[优化] GPU: {gpu_info.get('device', 'N/A')} 显存: {gpu_info.get('total_memory_gb', 'N/A')}GB")

            _torch_inf = torch.inference_mode() if torch is not None else None
            detect_device = str(detector.device) if hasattr(detector, "device") else "cpu"

            _async_read_enabled = True
            _read_stop_event = threading.Event()
            _frame_queue = queue.Queue(maxsize=128)
            _reader_thread = threading.Thread(
                target=self._async_frame_reader,
                args=(cap, _frame_queue, _read_stop_event),
                daemon=True,
            )
            _reader_thread.start()
            self.log_message.emit(f"[优化] 异步帧读取已启用 (缓冲=128帧, device={detect_device})")

            # ========================================================
            # 主循环入口 — 每帧执行完整流水线
            # ========================================================
            while self.running:
                if self.paused:
                    self.msleep(40)
                    continue

                if self._seek_frame >= 0:
                    seek_target = self._seek_frame
                    self._seek_frame = -1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                    frame_idx = seek_target
                    track_history.clear()
                    heatmap_accum.fill(0)
                    heatmap_overlay_cache = None
                    if _async_read_enabled:
                        while not _frame_queue.empty():
                            try:
                                _frame_queue.get_nowait()
                            except queue.Empty:
                                break
                    self.log_message.emit(f"跳转到帧 {seek_target}")
                if _async_read_enabled:
                    try:
                        ok, frame = _frame_queue.get(timeout=5)
                    except queue.Empty:
                        self.log_message.emit("视频读取超时，尝试同步回退")
                        ok, frame = cap.read()
                else:
                    ok, frame = cap.read()
                if not ok:
                    self.log_message.emit("视频结束")
                    self.video_finished.emit()
                    break

                prep_t0 = time.perf_counter()
                if frame.shape[1] != proc_width or frame.shape[0] != proc_height:
                    frame = cv2.resize(frame, (proc_width, proc_height))

                # ── Step 0: 掩膜生成（推理用掩膜、显示用全图）──
                # 目标：让 YOLO 只检测 ROI 内的行人 → 降低计算量 → 提升 FPS
                # 策略：掩膜帧送推理，原始帧送显示，用户始终看到完整画面
                if roi_enabled and roi_pts_np is not None:
                    roi_mask = np.zeros(                     # 创建全黑画布 (H×W, uint8)
                        (proc_height, proc_width), dtype=np.uint8)
                    cv2.fillPoly(roi_mask, [roi_pts_np], 255)# ROI 区域填白色 (255=保留)
                    # ⬆ fillPoly 在 mask 上绘制实心多边形：
                    #   roi_pts_np = ROI 顶点数组 (N×2, int32)
                    #   255 = 填充色（白色=通过，0=黑色=遮挡）
                    inference_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                    # ⬆ bitwise_and: frame 与自身做按位与，mask=0 的位置变黑(0,0,0)
                    #   原理：mask=255 → 保留原像素值；mask=0 → 像素清零
                    #   效果：ROI 外全黑，CNN 自动忽略纯黑区域（几乎零计算）
                else:
                    inference_frame = frame                 # 无 ROI 模式，直接用原图
                prep_ms = (time.perf_counter() - prep_t0) * 1000.0

                loop_t0 = time.perf_counter()
                detector.conf = float(self.conf)
                detector.iou = float(self.iou)

                # ── Step 1: YOLO 推理（掩膜帧 → 仅检测 ROI 内目标）──
                # 使用 torch.inference_mode() 禁用梯度计算，加速推理
                inf_t0 = time.perf_counter()
                if _torch_inf is not None:
                    with _torch_inf:                  # torch.inference_mode() 上下文
                        detections = detector.detect(inference_frame)
                        # ⬆ inference_frame = 掩膜后的帧（ROI 外全黑）
                        #    YOLO 只在 ROI 区域做卷积计算，黑区跳过
                        #    → 推理时间 ≈ 正比于 ROI 面积占比
                else:
                    detections = detector.detect(inference_frame)
                    # ⬆ CPU 模式同样使用掩膜帧
                inf_ms = (time.perf_counter() - inf_t0) * 1000.0

                # ── Step 1.5: pointPolygonTest 二次精确过滤 ──
                # 掩膜推理的边缘区域仍可能漏过少量 ROI 外检测框
                # 用 OpenCV pointPolygonTest 对每个框几何中心做最终判定
                if roi_enabled and roi_pts_np is not None:
                    roi_detections = []
                    for det in detections:
                        cx, cy = det.center               # 检测框几何中心坐标
                        if cv2.pointPolygonTest(roi_pts_np, (cx, cy), False) >= 0:
                            # ⬆ pointPolygonTest 判断点与多边形的位置关系：
                            #   返回值 ≥0 → 点在多边形内部或边上
                            #   返回值 <0 → 点在多边形外部
                            #   measureDist=False → 只返回 ±1/0，不做精确距离计算（更快）
                            roi_detections.append(det)     # 保留 ROI 内检测
                    detections = roi_detections             # 替换为过滤后的列表

                # ── Step 2: DeepSORT 跟踪 ──
                # 将检测结果传入跟踪器，获取带有跨帧一致 ID 的轨迹
                # 注意：跟踪器仍使用原始 frame 提取外观特征（掩膜帧会丢失颜色纹理信息）
                post_t0 = time.perf_counter()
                all_tracks = tracker.update(frame, detections)

                # ── Step 3: ROI 区域过滤 ──
                # 仅保留计数锚点在 ROI 多边形内的轨迹
                roi_tracks = all_tracks
                if roi_enabled:
                    roi_tracks = filter_tracks_by_roi(
                        all_tracks,
                        polygon=roi_polygon,
                        anchor_point=roi_anchor_point,
                    )

                # ── Step 4: 静态目标过滤 ──
                # 检测并排除长时间几乎不动的目标（模特/海报/虚警）
                static_ids = set()
                if static_filter_enabled:
                    static_ids = static_filter.update(roi_tracks, frame_idx)

                # 排除静态目标后的轨迹，用于后续计数
                counting_tracks = [tr for tr in roi_tracks if tr.track_id not in static_ids]

                tracks_for_draw = counting_tracks if hide_static_boxes else roi_tracks
                visible_ids = {tr.track_id for tr in tracks_for_draw}

                # ── 轨迹点/热力图累积（静态过滤后）──
                if hide_static_boxes:
                    if self.show_trail or self.show_heatmap:
                        for tr in counting_tracks:
                            p = tr.count_point("bottom_center")
                            px, py = int(p[0]), int(p[1])
                            if self.show_trail:
                                track_history[tr.track_id].append((px, py))
                            if self.show_heatmap and 0 <= px < proc_width and 0 <= py < proc_height:
                                heatmap_accum[py, px] += 1.0
                else:
                    if self.show_trail:
                        for tr in roi_tracks:
                            p = tr.count_point("bottom_center")
                            track_history[tr.track_id].append((int(p[0]), int(p[1])))
                    if self.show_heatmap:
                        for tr in counting_tracks:
                            p = tr.count_point("bottom_center")
                            px, py = int(p[0]), int(p[1])
                            if 0 <= px < proc_width and 0 <= py < proc_height:
                                heatmap_accum[py, px] += 1.0

                # ── Step 5: 跨线计数 ──
                line_events = []
                if line_counter is not None:
                    line_events = line_counter.update(counting_tracks, frame_idx)
                    for event in line_events:
                        event_logger.add_event(event)
                        event_payload = dict(event)
                        event_payload.update(session_meta)
                        db_manager.insert_event(
                            event_payload,
                            session_id=session_id,
                            source_name=source_label,
                            run_id=session_run_id,
                            detect_time=session_started_at,
                            created_at=session_started_at,
                        )
                        self.event_emitted.emit(event_payload)
                        direction = event.get("direction") or event.get("label") or event.get("value")
                        minute_idx = int((frame_idx / fps) // 60)
                        if direction == "up":
                            flow_stats[minute_idx]["up"] += 1
                        elif direction == "down":
                            flow_stats[minute_idx]["down"] += 1

                # ── Step 6: 区域计数 ──
                zone_events = zone_manager.update(counting_tracks, frame_idx)
                for event in zone_events:
                    event_logger.add_event(event)
                    event_payload = dict(event)
                    event_payload.update(session_meta)
                    db_manager.insert_event(
                        event_payload,
                        session_id=session_id,
                        source_name=source_label,
                        run_id=session_run_id,
                        detect_time=session_started_at,
                        created_at=session_started_at,
                    )
                    self.event_emitted.emit(event_payload)
                current_minute = int((frame_idx / fps) // 60)
                up_now = int(flow_stats[current_minute]["up"])
                down_now = int(flow_stats[current_minute]["down"])

                # 趋势图节流：按变化 + 频率限制发信号，避免每帧触发 Matplotlib 重绘
                trend_changed = (
                    last_trend_minute is None
                    or current_minute != last_trend_minute
                    or up_now != last_trend_up
                    or down_now != last_trend_down
                )
                if trend_changed:
                    now_pc = time.perf_counter()
                    if trend_interval <= 0.0 or (now_pc - last_trend_emit) >= trend_interval:
                        self.trend_updated.emit(
                            {
                                "minute": current_minute,
                                "up": up_now,
                                "down": down_now,
                                "total": up_now + down_now,
                            }
                        )
                        last_trend_emit = now_pc
                        last_trend_minute = current_minute
                        last_trend_up = up_now
                        last_trend_down = down_now

                # ── Step 6: 全图可视化渲染（原始帧 frame，非掩膜帧）──
                # 用户始终看到完整视频画面，无黑边，仅 ROI 内检测框被绘制
                # 节流：按 UI 最大帧率限制，避免 GUI 线程过载
                now_pc = time.perf_counter()
                do_render = ui_frame_interval <= 0.0 or (now_pc - last_frame_emit) >= ui_frame_interval
                if do_render:
                    np.copyto(_display_rgba, frame)       # ⬅ 使用原始 frame（非 inference_frame）
                    # _display_rgba 是预分配的 RGBA 缓冲区，避免每帧创建新数组
                    annotated = _display_rgba             # 从此刻起所有绘制都在完整画面上

                    # ① 绘制 ROI 边界线（粉紫色多边形边框，线宽 2px）
                    if self.show_roi and roi_pts_np is not None and len(roi_pts_np) >= 3:
                        cv2.polylines(annotated, [roi_pts_np], True, (255, 0, 255), 2)
                        # ⬆ polylines: 绘制多边形轮廓（实心=isClosed=True）
                        #   (255,0,255) = 粉紫色 BGR
                        if roi_label_pos is not None:
                            cv2.putText(                  # ROI 标签文字
                                annotated, "counting_roi", roi_label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                    # ② 绘制计数锚点（脚底中心绿色圆点）+ track_id
                    if self.draw_count_points:
                        for tr in counting_tracks:
                            pt = tr.count_point("bottom_center")  # 脚底中心坐标
                            x, y = int(pt[0]), int(pt[1])
                            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                            # ⬆ 绿色实心圆，半径 5px，-1=填充
                            cv2.putText(
                                annotated, f"{tr.track_id}", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # ③ 绘制检测框（不同 ID 随机颜色，含 track_id 和可选置信度）
                    annotated = draw_tracks(
                        annotated, tracks_for_draw,       # tracks_for_draw = 供绘制用的轨迹
                        show_conf=show_confidence,         # 是否显示置信度分数
                        box_thickness=box_thickness,       # 边框粗细
                    )

                    # ④ 绘制轨迹历史线（过去 N 帧的路径折线）
                    if self.show_trail:
                        annotated = draw_track_history(annotated, track_history)

                    # ⑤ 绘制计数线（黄色或配置色线段 + 箭头 + 方向标签）
                    if self.show_line and line_counter is not None:
                        annotated = draw_line_counter(annotated, line_counter)

                    # ⑥ 绘制区域计数多边形边界 + 当前人数
                    annotated = draw_zone_counters(annotated, zone_manager)
                    annotated = draw_counters_panel(annotated, line_counter, zone_manager)

                    # ⑦ 底部计数统计文字（上下行实时数值）
                    cv2.putText(
                        annotated,
                        f"Minute {current_minute}: up={up_now} down={down_now}",
                        (20, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # ⑧ 人脸模糊（隐私保护，仅作用于显示帧）
                    annotated = face_blur.apply(annotated)

                    # ⑨ 热力图叠加（半透明覆盖，显示过往行人的空间密度分布）
                    if self.show_heatmap:
                        try:
                            if frame_idx % heatmap_interval == 0 or heatmap_overlay_cache is None:
                                if np.max(heatmap_accum) > 0:
                                    hm_blur = cv2.GaussianBlur(
                                        heatmap_accum,
                                        (0, 0),
                                        sigmaX=heatmap_sigma,
                                        sigmaY=heatmap_sigma,
                                    )
                                    hm_norm = cv2.normalize(hm_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                    heatmap_overlay_cache = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

                            if heatmap_overlay_cache is not None:
                                annotated = cv2.addWeighted(   # 半透明叠加
                                    annotated, 1.0 - heatmap_alpha,  # 原图权重
                                    heatmap_overlay_cache, heatmap_alpha,  # 热力图权重
                                    0)                         # gamma 偏移
                                # 效果：行人密集区域呈暖色(红/黄)，稀疏区域透明
                        except Exception as exc:
                            self.log_message.emit(f"热力图叠加失败: {exc}")

                    # ⑩ BGR → QImage → emit 到 GUI 主线程显示
                    h, w = annotated.shape[:2]                   # 画面尺寸
                    bpl = w * 3                                  # 每行字节数 (BGR=3通道)
                    if qimage_bgr_format is not None:
                        q_img = QImage(annotated.data, w, h, bpl, qimage_bgr_format).copy()
                        # ⬆ BGR888 格式（QImage.Format_BGR888），性能最优，零转换
                    else:
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        # ⬆ 兜底：BGR→RGB 转换后再构造 QImage
                        bpl = w * 3
                        q_img = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888).copy()
                    # .copy() 是关键：确保 QImage 持有独立内存，避免 annotated 被后续帧覆盖

                    self.new_frame.emit(q_img)                  # 发射信号 → GUI 显示
                    last_frame_emit = time.perf_counter()        # 记录上次渲染时刻

                # 只保留当前可视轨迹，防止 history 无限增长
                if self.show_trail and track_history:
                    for tid in list(track_history.keys()):
                        if tid not in visible_ids:
                            track_history.pop(tid, None)

                # ── 轨迹点记录（用于轨迹回放）──
                if tracks_for_draw:
                    frame_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for tr in tracks_for_draw:
                        px, py = tr.count_point("bottom_center")
                        src_px = int(round(px / resize_scale)) if resize_scale > 1e-9 else int(px)
                        src_py = int(round(py / resize_scale)) if resize_scale > 1e-9 else int(py)
                        src_x1 = int(round(float(tr.x1) / resize_scale)) if resize_scale > 1e-9 else int(round(float(tr.x1)))
                        src_y1 = int(round(float(tr.y1) / resize_scale)) if resize_scale > 1e-9 else int(round(float(tr.y1)))
                        src_x2 = int(round(float(tr.x2) / resize_scale)) if resize_scale > 1e-9 else int(round(float(tr.x2)))
                        src_y2 = int(round(float(tr.y2) / resize_scale)) if resize_scale > 1e-9 else int(round(float(tr.y2)))
                        trajectory_buffer.append(
                            {
                                "session_id": session_id,
                                "run_id": session_run_id,
                                "source_name": source_label,
                                "detect_time": session_started_at,
                                "timestamp": frame_timestamp,
                                "created_at": frame_timestamp,
                                "frame_idx": frame_idx,
                                "track_id": int(tr.track_id),
                                "x1": float(src_x1),
                                "y1": float(src_y1),
                                "x2": float(src_x2),
                                "y2": float(src_y2),
                                "cx": float(src_px),
                                "cy": float(src_py),
                            }
                        )

                elapsed = time.perf_counter() - loop_t0
                total_frames += 1
                total_time += elapsed
                fps_now = (1.0 / elapsed) if elapsed > 1e-6 else 0.0
                up_count, down_count = self._line_up_down(line_counter)
                
                total_count = up_count + down_count
                stats = {
                    "session_id": session_id,
                    "run_id": session_run_id,
                    "up": up_count,
                    "down": down_count,
                    "total": total_count,
                    "fps": fps_now,
                    "avg_fps": (total_frames / total_time) if total_time > 0 else 0.0,
                    "source_fps": fps,
                    "current_frame": frame_idx + 1,
                    "total_frames": total_frames_hint,
                    "current": len(counting_tracks),
                }

                now_pc = time.perf_counter()
                if stats_interval <= 0.0 or (now_pc - last_stats_emit) >= stats_interval:
                    self.stats_updated.emit(stats)
                    last_stats_emit = now_pc

                self.frame_position.emit(frame_idx + 1, total_frames_hint)

                # ── 数据库定时写入（每秒一次）──
                current_time = time.time()
                if current_time - last_db_insert_time >= 1.0:
                    try:
                        db_manager.insert_data(
                            up_count,
                            down_count,
                            total_count,
                            session_id=session_id,
                            source_name=source_label,
                            run_id=session_run_id,
                            detect_time=session_started_at,
                            fps=fps_now,
                            avg_fps=stats["avg_fps"],
                            frame_idx=frame_idx,
                        )
                        if trajectory_buffer:
                            db_manager.insert_trajectory_points(trajectory_buffer)
                            trajectory_buffer.clear()
                        # 每存几十秒可以顺手清一次旧数据防膨胀
                        if int(current_time) % 60 == 0:
                            db_manager.delete_old_data(limit=5000)
                    except Exception as e:
                        self.log_message.emit(f"数据库写入异常: {e}")
                    last_db_insert_time = current_time

                post_ms = (time.perf_counter() - post_t0) * 1000.0
                total_ms = prep_ms + inf_ms + post_ms
                perf.record_frame(prep_ms, inf_ms, post_ms, total_ms)

                if frame_idx % 30 == 0:
                    snap = perf.snapshot()
                    self.perf_sample.emit(snap)

                if frame_idx % 60 == 0:
                    avg_fps = (total_frames / total_time) if total_time > 0 else 0.0
                    snap = perf.snapshot()
                    self.log_message.emit(
                        f"[frame {frame_idx}] fps={snap['fps']:.1f} prep={snap['preprocess_ms']:.1f}ms "
                        f"inf={snap['inference_ms']:.1f}ms post={snap['postprocess_ms']:.1f}ms "
                        f"gpu={snap['gpu_util_pct']:.0f}% cpu={snap['cpu_util_pct']:.0f}%"
                        f" all={len(all_tracks)} roi={len(roi_tracks)} counting={len(counting_tracks)}"
                    )

                frame_idx += 1

            if total_frames > 0 and total_time > 0:
                final = perf.snapshot()
                self.log_message.emit(f"处理完成，平均FPS: {final['fps']:.2f}")
                self.perf_sample.emit(final)

        except Exception as exc:
            self.log_message.emit(f"处理异常: {exc}")
            self.log_message.emit(traceback.format_exc())
        finally:
            if _async_read_enabled and _read_stop_event is not None:
                _read_stop_event.set()
            if cap is not None:
                cap.release()
            if event_logger is not None:
                try:
                    event_logger.flush()
                except Exception as exc:
                    self.log_message.emit(f"事件写盘失败: {exc}")

            if trajectory_buffer:
                try:
                    db_manager.insert_trajectory_points(trajectory_buffer)
                except Exception as exc:
                    self.log_message.emit(f"轨迹写盘失败: {exc}")

            try:
                if db_manager is not None:
                    db_manager.finalize_session(
                        session_id,
                        avg_fps=(total_frames / total_time) if total_time > 0 else 0.0,
                        up_count=up_count,
                        down_count=down_count,
                        total_count=total_count,
                    )
            except Exception:
                pass

            try:
                if db_manager is not None:
                    db_manager.close()
            except Exception:
                pass

            self.running = False
            self.paused = False
            self.log_message.emit("处理结束")
            self.finished.emit()

    def stop(self):
        self.running = False
        self.paused = False
        self.wait()
