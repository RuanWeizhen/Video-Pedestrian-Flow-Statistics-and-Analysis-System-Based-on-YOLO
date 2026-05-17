from __future__ import annotations

import copy
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
import threading

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

try:
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(r"E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counting.line_counter import LineCounter
from counting.zone_counter import ZoneCounterManager
from detector.yolo_detector import YOLODetector
from privacy.face_blur import FaceBlur
from tracker.deepsort_wrapper import DeepSortTracker
from utils.config import load_config
from utils.db_manager import DatabaseManager
from utils.filters import StaticTrackFilter, filter_tracks_by_roi
from utils.io_utils import EventLogger, ensure_dir
from utils.visualization import (
    draw_counters_panel,
    draw_line_counter,
    draw_track_history,
    draw_tracks,
    draw_zone_counters,
)

class WorkerThread(QThread):
    new_frame = pyqtSignal(QImage)
    stats_updated = pyqtSignal(dict)
    trend_updated = pyqtSignal(dict)
    event_emitted = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.paused = False
        self.video_path = None
        self.config_path = Path(__file__).resolve().parents[1] / "config" / "pedestrian_demo.yaml"
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

    def resume(self):
        self.paused = False

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

    def run(self):
        self.running = True
        self.paused = False

        cap = None
        event_logger = None
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

            tracker = DeepSortTracker(cfg["tracker"])

            counting_cfg = cfg.get("counting", {})
            roi_cfg = counting_cfg.get("roi", {})
            roi_polygon = roi_cfg.get("polygon", []) or []
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
                line_counter = LineCounter(line_cfg)

            zone_manager = ZoneCounterManager(counting_cfg.get("zones", []))
            event_logger = EventLogger(events_csv_path)

            db_manager = DatabaseManager("traffic.db")
            db_manager.init_db()
            session_id = db_manager.start_session(
                source_name=source_label,
                source_path=str(source),
                config_path=str(self.config_path),
                conf=float(detector_cfg.get("conf", self.conf)),
                iou=float(detector_cfg.get("iou", self.iou)),
                detector_type=str(detector.summary().get("model_path", "YOLO")),
            )
            last_db_insert_time = time.time()

            visualization_cfg = cfg.get("visualization", {})
            trail_length = int(visualization_cfg.get("trail_length", 30))
            show_confidence = bool(visualization_cfg.get("show_confidence", True))
            box_thickness = int(visualization_cfg.get("box_thickness", 2))
            draw_count_points = bool(cfg.get("debug", {}).get("draw_count_points", True))
            heatmap_sigma = int(visualization_cfg.get("heatmap_sigma", self.heatmap_sigma))
            heatmap_alpha = float(visualization_cfg.get("heatmap_alpha", self.heatmap_alpha))
            heatmap_interval = max(1, int(visualization_cfg.get("heatmap_interval", self.heatmap_interval)))

            heatmap_accum = np.zeros((proc_height, proc_width), dtype=np.float32)
            heatmap_overlay_cache = None

            qimage_bgr_format = QImage.Format_BGR888 if hasattr(QImage, "Format_BGR888") else None

            track_history = defaultdict(lambda: deque(maxlen=trail_length))
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

            while self.running:
                if self.paused:
                    self.msleep(40)
                    continue

                ok, frame = cap.read()
                if not ok:
                    self.log_message.emit("视频结束")
                    break

                # 按原始宽高比自适应缩放，不拉伸。
                if frame.shape[1] != proc_width or frame.shape[0] != proc_height:
                    frame = cv2.resize(frame, (proc_width, proc_height))

                loop_t0 = time.perf_counter()
                detector.conf = float(self.conf)
                detector.iou = float(self.iou)

                if torch is not None:
                    with torch.inference_mode():
                        detections = detector.detect(frame)
                else:
                    detections = detector.detect(frame)
                all_tracks = tracker.update(frame, detections)

                roi_tracks = all_tracks
                if roi_enabled:
                    roi_tracks = filter_tracks_by_roi(
                        all_tracks,
                        polygon=roi_polygon,
                        anchor_point=roi_anchor_point,
                    )

                static_ids = set()
                if static_filter_enabled:
                    static_ids = static_filter.update(roi_tracks, frame_idx)

                counting_tracks = [tr for tr in roi_tracks if tr.track_id not in static_ids]

                tracks_for_draw = counting_tracks if hide_static_boxes else roi_tracks
                visible_ids = {tr.track_id for tr in tracks_for_draw}

                # 轨迹点/热力图累积：放在 static 过滤之后，减少不必要的计算
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

                line_events = []
                if line_counter is not None:
                    line_events = line_counter.update(counting_tracks, frame_idx)
                    for event in line_events:
                        event_logger.add_event(event)
                        db_manager.insert_event(event, session_id=session_id, source_name=source_label)
                        self.event_emitted.emit(dict(event))
                        direction = event.get("direction") or event.get("label") or event.get("value")
                        minute_idx = int((frame_idx / fps) // 60)
                        if direction == "up":
                            flow_stats[minute_idx]["up"] += 1
                        elif direction == "down":
                            flow_stats[minute_idx]["down"] += 1

                zone_events = zone_manager.update(counting_tracks, frame_idx)
                for event in zone_events:
                    event_logger.add_event(event)
                    db_manager.insert_event(event, session_id=session_id, source_name=source_label)
                    self.event_emitted.emit(dict(event))
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

                # 仅在需要刷新 UI 时才绘制/生成 QImage（否则跳过昂贵的 draw + copy）
                now_pc = time.perf_counter()
                do_render = ui_frame_interval <= 0.0 or (now_pc - last_frame_emit) >= ui_frame_interval
                if do_render:
                    annotated = frame  # 直接原地绘制，避免整帧 copy

                    if self.show_roi and roi_pts_np is not None and len(roi_pts_np) >= 3:
                        cv2.polylines(annotated, [roi_pts_np], True, (255, 0, 255), 2)
                        if roi_label_pos is not None:
                            cv2.putText(
                                annotated,
                                "counting_roi",
                                roi_label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 0, 255),
                                2,
                            )

                    if self.draw_count_points:
                        for tr in counting_tracks:
                            pt = tr.count_point("bottom_center")
                            x, y = int(pt[0]), int(pt[1])
                            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                            cv2.putText(
                                annotated,
                                f"{tr.track_id}",
                                (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                    annotated = draw_tracks(
                        annotated,
                        tracks_for_draw,
                        show_conf=show_confidence,
                        box_thickness=box_thickness,
                    )

                    if self.show_trail:
                        annotated = draw_track_history(annotated, track_history)

                    if self.show_line and line_counter is not None:
                        annotated = draw_line_counter(annotated, line_counter)

                    annotated = draw_zone_counters(annotated, zone_manager)
                    annotated = draw_counters_panel(annotated, line_counter, zone_manager)

                    cv2.putText(
                        annotated,
                        f"Minute {current_minute}: up={up_now} down={down_now}",
                        (20, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                    # 仅处理显示输出帧，不影响检测/跟踪/计数逻辑。
                    annotated = face_blur.apply(annotated)

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
                                annotated = cv2.addWeighted(
                                    annotated,
                                    1.0 - heatmap_alpha,
                                    heatmap_overlay_cache,
                                    heatmap_alpha,
                                    0,
                                )
                        except Exception as exc:
                            self.log_message.emit(f"热力图叠加失败: {exc}")

                    # 优先走 BGR888，避免每帧 BGR->RGB 颜色转换
                    h, w = annotated.shape[:2]
                    bytes_per_line = int(annotated.strides[0])
                    if qimage_bgr_format is not None:
                        q_img = QImage(annotated.data, w, h, bytes_per_line, qimage_bgr_format).copy()
                    else:
                        rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        bytes_per_line = int(rgb_image.strides[0])
                        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

                    self.new_frame.emit(q_img)
                    last_frame_emit = time.perf_counter()

                # 只保留当前可视轨迹，防止 history 无限增长
                if self.show_trail and track_history:
                    for tid in list(track_history.keys()):
                        if tid not in visible_ids:
                            track_history.pop(tid, None)

                elapsed = time.perf_counter() - loop_t0
                total_frames += 1
                total_time += elapsed
                fps_now = (1.0 / elapsed) if elapsed > 1e-6 else 0.0
                up_count, down_count = self._line_up_down(line_counter)
                
                total_count = up_count + down_count
                stats = {
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

                # 每1秒写入一次数据库
                current_time = time.time()
                if current_time - last_db_insert_time >= 1.0:
                    try:
                        db_manager.insert_data(up_count, down_count, total_count, session_id=session_id, source_name=source_label)
                        # 每存几十秒可以顺手清一次旧数据防膨胀
                        if int(current_time) % 60 == 0:
                            db_manager.delete_old_data(limit=5000)
                    except Exception as e:
                        self.log_message.emit(f"数据库写入异常: {e}")
                    last_db_insert_time = current_time

                if frame_idx % 60 == 0:
                    avg_fps = (total_frames / total_time) if total_time > 0 else 0.0
                    self.log_message.emit(
                        f"[frame {frame_idx}] all={len(all_tracks)} roi={len(roi_tracks)} counting={len(counting_tracks)} avg_fps={avg_fps:.2f}"
                    )

                frame_idx += 1

            if total_frames > 0 and total_time > 0:
                self.log_message.emit(f"处理完成，平均FPS: {total_frames / total_time:.2f}")

        except Exception as exc:
            self.log_message.emit(f"处理异常: {exc}")
            self.log_message.emit(traceback.format_exc())
        finally:
            if cap is not None:
                cap.release()
            if event_logger is not None:
                try:
                    event_logger.flush()
                except Exception as exc:
                    self.log_message.emit(f"事件写盘失败: {exc}")

            try:
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
                # 结束前关闭数据库连接
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
