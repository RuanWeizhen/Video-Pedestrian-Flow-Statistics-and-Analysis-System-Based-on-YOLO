from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import logging
from pathlib import Path
import math

import numpy as np

import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dashboard_widgets import DashboardSection
from .stats_panel import StatsPanel
from .trend_panel import TrendChartWidget
from .video_panel import VideoPanel
from utils.coordinate_transform import frame_points_to_display_points, get_display_transform
from utils.visualization import draw_track_history


logger = logging.getLogger(__name__)


def _frame_to_qimage(frame):
    height, width = frame.shape[:2]
    bytes_per_line = int(frame.strides[0])
    if hasattr(QImage, "Format_BGR888"):
        return QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888).copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bytes_per_line = int(rgb_frame.strides[0])
    return QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def _point_list_to_int_points(points):
    converted = []
    for point in points or []:
        if point is None:
            continue
        try:
            x, y = point
        except Exception:
            continue
        converted.append((int(round(float(x))), int(round(float(y)))))
    return converted


class TrajectoryReplayDialog(QDialog):
    def __init__(
        self,
        session_row: dict,
        video_path: str,
        trajectory_rows: list[dict],
        traffic_rows: list[dict],
        fps_rows: list[dict],
        event_rows: list[dict],
        roi_points: list | None = None,
        line_points: list | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session_row = dict(session_row or {})
        self.session_id = int(self.session_row.get("id") or self.session_row.get("session_id") or 0)
        self.run_id = str(self.session_row.get("run_id") or self.session_id or "")
        self.source_name = str(self.session_row.get("source_name") or Path(video_path).name or "未知视频")
        self.video_path = Path(str(video_path or "")).expanduser()
        self.roi_points = [tuple(map(int, p)) for p in (roi_points or [])]
        self.line_points = [tuple(map(int, p)) for p in (line_points or [])][:2]
        self._last_replay_geometry_log_key = None

        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")

        self.trajectory_by_frame = defaultdict(list)
        for row in trajectory_rows or []:
            try:
                frame_idx = int(row.get("frame_idx", 0) or 0)
                track_id = int(row.get("track_id", 0) or 0)
                point = (int(float(row.get("cx", 0.0) or 0.0)), int(float(row.get("cy", 0.0) or 0.0)))
            except Exception:
                continue
            self.trajectory_by_frame[frame_idx].append({"track_id": track_id, "point": point})
        self.trajectory_frame_indices = sorted(self.trajectory_by_frame.keys())
        self._trajectory_frame_cursor = -1
        self._trajectory_frame_index = 0
        self._trajectory_history = defaultdict(list)

        self.traffic_samples = self._normalize_sample_rows(traffic_rows or [])
        self.fps_samples = self._normalize_sample_rows(fps_rows or [])
        self.event_rows = sorted(list(event_rows or []), key=lambda item: int(item.get("frame_idx", 0) or 0))
        self.event_by_frame = defaultdict(list)
        for row in self.event_rows:
            try:
                frame_idx = int(row.get("frame_idx", 0) or 0)
            except Exception:
                frame_idx = 0
            self.event_by_frame[frame_idx].append(row)

        self.traffic_frame_indices = [int(item.get("frame_idx", idx) or idx) for idx, item in enumerate(self.traffic_samples)]
        self.fps_frame_indices = [int(item.get("frame_idx", idx) or idx) for idx, item in enumerate(self.fps_samples)]

        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        self.video_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.video_fps <= 1e-6:
            self.video_fps = 25.0
        self.video_total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.max_data_frame = max(
            [0]
            + [int(item.get("frame_idx", 0) or 0) for item in self.traffic_samples]
            + [int(item.get("frame_idx", 0) or 0) for item in self.fps_samples]
            + [int(item.get("frame_idx", 0) or 0) for item in self.event_rows]
            + list(self.trajectory_frame_indices)
        )
        if self.video_total_frames <= 0:
            self.video_total_frames = self.max_data_frame + 1
        self.max_frame_index = max(0, max(self.video_total_frames - 1, self.max_data_frame))
        self.current_frame_index = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self._exporting = False
        self.source_frame_size = (0, 0)
        self._last_render_ts = 0.0
        self._replay_fps = 0.0

        source_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if source_width <= 0 or source_height <= 0:
            ok, first_frame = self.capture.read()
            if ok and first_frame is not None:
                source_height, source_width = first_frame.shape[:2]
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if source_width > 0 and source_height > 0:
            self.source_frame_size = (int(source_width), int(source_height))

        self.setWindowFlags(
            Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint |
            Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        )
        self._build_ui()
        self._load_initial_data()

    def _build_ui(self):
        self.setWindowTitle(f"轨迹回放 - {self.source_name}")
        self.setMinimumSize(1600, 980)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)
        self.title_label = QLabel(f"批次 {self.run_id} | {self.source_name}")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #16324f;")
        self.subtitle_label = QLabel(self._build_session_summary_text())
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("color: #5f6f84;")
        header_layout.addWidget(self.title_label, 2)
        header_layout.addWidget(self.subtitle_label, 5)
        root_layout.addWidget(header_widget)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        video_section = DashboardSection("原视频同步回放", "轨迹、事件与统计状态叠加显示")
        self.video_panel = VideoPanel()
        if self.source_frame_size[0] > 0 and self.source_frame_size[1] > 0:
            self.video_panel.set_source_frame_size(*self.source_frame_size)
        self.video_panel.set_show_flags(True, True)
        self.video_panel.set_annotations([], [])
        video_section.body_layout.addWidget(self.video_panel, 1)
        video_section.setMinimumHeight(560)
        left_layout.addWidget(video_section, 1)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.current_frame_label = QLabel("帧 0 / 0")
        self.current_frame_label.setStyleSheet("font-weight: 600; color: #16324f;")
        self.current_event_label = QLabel("当前事件：无")
        self.current_event_label.setWordWrap(True)
        self.current_event_label.setStyleSheet("color: #6b7280;")

        button_row = QHBoxLayout()
        self.btn_replay_start = QPushButton("回到开头")
        self.btn_play_pause = QPushButton("播放")
        self.btn_step_prev = QPushButton("上一帧")
        self.btn_step_next = QPushButton("下一帧")
        self.btn_export_video = QPushButton("导出视频")
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("0.5x", 0.5)
        self.speed_combo.addItem("1.0x", 1.0)
        self.speed_combo.addItem("1.5x", 1.5)
        self.speed_combo.addItem("2.0x", 2.0)
        self.speed_combo.setCurrentIndex(1)
        for button in (self.btn_replay_start, self.btn_step_prev, self.btn_play_pause, self.btn_step_next, self.btn_export_video):
            button.setMinimumHeight(38)
            button_row.addWidget(button)
        button_row.addStretch()
        button_row.addWidget(QLabel("速度"))
        button_row.addWidget(self.speed_combo)
        controls_layout.addWidget(self.current_frame_label)
        controls_layout.addWidget(self.current_event_label)
        controls_layout.addLayout(button_row)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(self.max_frame_index)
        self.timeline_slider.setSingleStep(1)
        self.timeline_slider.setPageStep(max(1, int(self.video_fps)))
        controls_layout.addWidget(self.timeline_slider)
        left_layout.addWidget(controls_widget, 0)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)

        stats_section = DashboardSection("当前统计", "随播放位置同步更新")
        self.stats_panel = StatsPanel()
        stats_section.set_body_widget(self.stats_panel)
        stats_section.setMinimumHeight(280)
        right_splitter.addWidget(stats_section)

        event_section = DashboardSection("事件轨迹", "按帧查看事件记录，双击可跳转到对应位置")
        event_widget = QWidget()
        event_layout = QVBoxLayout(event_widget)
        event_layout.setContentsMargins(0, 0, 0, 0)
        self.event_table = QTableWidget(0, 6)
        self.event_table.setHorizontalHeaderLabels(["帧号", "时间", "事件", "目标", "轨迹ID", "批次ID"])
        self.event_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        event_layout.addWidget(self.event_table)
        event_section.set_body_widget(event_widget)
        event_section.setMinimumHeight(280)
        right_splitter.addWidget(event_section)

        chart_section = DashboardSection("统计曲线", "流量趋势与 FPS 曲线同步回放")
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.trend_panel = TrendChartWidget()
        chart_layout.addWidget(self.trend_panel)
        chart_section.set_body_widget(chart_widget)
        chart_section.setMinimumHeight(360)
        right_splitter.addWidget(chart_section)

        right_splitter.setSizes([180, 280, 380])
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([900, 700])
        root_layout.addWidget(main_splitter, 1)

        self.btn_replay_start.clicked.connect(self._jump_to_start)
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        self.btn_step_prev.clicked.connect(self.step_backward)
        self.btn_step_next.clicked.connect(self.step_forward)
        self.btn_export_video.clicked.connect(self.export_replay_video)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.timeline_slider.valueChanged.connect(self.seek_frame)
        self.event_table.cellDoubleClicked.connect(self._seek_from_event_row)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

    def _build_session_summary_text(self) -> str:
        started_at = str(self.session_row.get("started_at", ""))
        ended_at = str(self.session_row.get("ended_at", ""))
        source_path = str(self.session_row.get("source_path", ""))
        return f"开始：{started_at or '-'} | 结束：{ended_at or '-'} | 视频：{source_path or '-'}"

    def _normalize_sample_rows(self, rows: list[dict]) -> list[dict]:
        normalized = []
        for index, row in enumerate(rows):
            try:
                frame_idx = int(row.get("frame_idx", index) or index)
            except Exception:
                frame_idx = index
            normalized.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": str(row.get("timestamp", "")),
                    "up": int(row.get("up_count", row.get("up", 0)) or 0),
                    "down": int(row.get("down_count", row.get("down", 0)) or 0),
                    "total": int(row.get("total_count", row.get("total", 0)) or 0),
                    "fps": float(row.get("fps", row.get("avg_fps", 0.0)) or 0.0),
                    "avg_fps": float(row.get("avg_fps", row.get("fps", 0.0)) or 0.0),
                }
            )
        normalized.sort(key=lambda item: item["frame_idx"])
        return normalized

    def _load_initial_data(self):
        self.trend_panel.set_replay_data(
            self.traffic_samples,
            fps_points=self.fps_samples,
            cursor_x=0,
            title_suffix=self.source_name,
        )
        self._populate_event_table()
        self.seek_frame(0)

    def _populate_event_table(self):
        self.event_table.setRowCount(0)
        for row in self.event_rows:
            row_index = self.event_table.rowCount()
            self.event_table.insertRow(row_index)
            frame_idx = int(row.get("frame_idx", 0) or 0)
            timestamp = self._format_frame_time(frame_idx)
            values = [
                str(frame_idx),
                timestamp,
                str(row.get("event_type", "")),
                str(row.get("target", "")),
                str(row.get("track_id", "")),
                str(row.get("run_id", self.run_id)),
            ]
            for col, value in enumerate(values):
                self.event_table.setItem(row_index, col, QTableWidgetItem(str(value)))

    def _format_frame_time(self, frame_idx: int) -> str:
        seconds = max(0.0, frame_idx / max(self.video_fps, 1e-6))
        total_seconds = int(seconds)
        mm, ss = divmod(total_seconds, 60)
        return f"{mm:02d}:{ss:02d}"

    def _sync_trajectory_state(self, target_frame: int):
        if target_frame < self._trajectory_frame_cursor:
            self._trajectory_history = defaultdict(list)
            self._trajectory_frame_cursor = -1
            self._trajectory_frame_index = 0

        while self._trajectory_frame_index < len(self.trajectory_frame_indices):
            frame_idx = self.trajectory_frame_indices[self._trajectory_frame_index]
            if frame_idx > target_frame:
                break
            for item in self.trajectory_by_frame.get(frame_idx, []):
                track_id = int(item.get("track_id", 0) or 0)
                self._trajectory_history[track_id].append(tuple(item.get("point", (0, 0))))
                if len(self._trajectory_history[track_id]) > 200:
                    self._trajectory_history[track_id] = self._trajectory_history[track_id][-200:]
            self._trajectory_frame_cursor = frame_idx
            self._trajectory_frame_index += 1

    def _sample_at_or_before(self, samples: list[dict], frame_indices: list[int], target_frame: int):
        if not samples:
            return None
        index = bisect_right(frame_indices, target_frame) - 1
        if index < 0:
            return None
        return samples[index]

    def _event_text_for_frame(self, frame_idx: int) -> str:
        events = self.event_by_frame.get(frame_idx, [])
        if not events:
            return "当前事件：无"
        labels = []
        for row in events[:4]:
            event_type = str(row.get("event_type", ""))
            target = str(row.get("target", ""))
            track_id = str(row.get("track_id", ""))
            label = event_type
            if target:
                label = f"{label} · {target}"
            if track_id:
                label = f"{label} · ID {track_id}"
            labels.append(label)
        if len(events) > 4:
            labels.append(f"... 共 {len(events)} 条")
        return "当前事件：" + "；".join(labels)

    def _wrap_text_lines(self, text: str, max_chars: int = 28):
        raw = str(text or "").strip()
        if not raw:
            return ["-"]

        normalized = raw.replace("\n", " ").replace("；", " | ")
        chunks = []
        for segment in normalized.split("|"):
            segment = segment.strip()
            if not segment:
                continue
            while len(segment) > max_chars:
                chunks.append(segment[:max_chars])
                segment = segment[max_chars:]
            if segment:
                chunks.append(segment)
        return chunks or ["-"]

    def _draw_export_panel(self, frame_idx: int, stats: dict, event_text: str, trajectory_count: int, panel_width: int, panel_height: int):
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (24, 32, 46)

        cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height - 1), (70, 84, 106), 1)
        cv2.putText(panel, "Replay Export", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, self.source_name, (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (185, 200, 220), 1, cv2.LINE_AA)

        lines = [
            f"Batch: {self.run_id}",
            f"Frame: {frame_idx + 1}/{self.max_frame_index + 1}",
            f"Time: {self._format_frame_time(frame_idx)}",
            f"Up: {int(stats.get('up', 0))}   Down: {int(stats.get('down', 0))}",
            f"Total: {int(stats.get('total', 0))}   Current: {int(stats.get('current', 0))}",
            f"FPS: {float(stats.get('fps', 0.0)):.2f}",
            f"Tracked IDs: {trajectory_count}",
        ]

        y = 118
        for line in lines:
            cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 240, 246), 1, cv2.LINE_AA)
            y += 28

        cv2.putText(panel, "Current event:", (20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 140), 1, cv2.LINE_AA)
        y += 40
        for line in self._wrap_text_lines(event_text, max_chars=24)[:6]:
            cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 226, 235), 1, cv2.LINE_AA)
            y += 24

        bar_top = panel_height - 84
        bar_left = 20
        bar_width = panel_width - 40
        bar_height = 16
        progress = 0.0
        if self.max_frame_index > 0:
            progress = min(1.0, max(0.0, frame_idx / float(self.max_frame_index)))
        cv2.putText(panel, "Progress", (20, bar_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 200, 220), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (bar_left, bar_top), (bar_left + bar_width, bar_top + bar_height), (80, 92, 110), 1)
        cv2.rectangle(panel, (bar_left + 1, bar_top + 1), (bar_left + int((bar_width - 2) * progress), bar_top + bar_height - 1), (72, 150, 255), -1)

        cv2.putText(panel, f"{int(progress * 100)}%", (bar_left, panel_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 246), 1, cv2.LINE_AA)
        cv2.putText(panel, "Exported replay video", (bar_left, panel_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 155, 175), 1, cv2.LINE_AA)
        return panel

    def _compose_export_frame(self, frame, frame_idx: int, trajectory_history, current_points, current_event_text: str, stats: dict):
        annotated = self._draw_roi_and_line_on_original_frame(frame)

        annotated = draw_track_history(annotated, trajectory_history)
        for item in current_points:
            track_id = int(item.get("track_id", 0) or 0)
            point = item.get("point")
            if point is None:
                continue
            x, y = int(point[0]), int(point[1])
            color = ((track_id * 47) % 180 + 60, 180, 255)
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"ID {track_id}", (x + 8, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            annotated,
            f"Batch {self.run_id} | Frame {frame_idx + 1}/{self.max_frame_index + 1} | {self._format_frame_time(frame_idx)}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if current_event_text and current_event_text != "当前事件：无":
            cv2.putText(annotated, current_event_text[:70], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 235, 160), 1, cv2.LINE_AA)

        panel = self._draw_export_panel(frame_idx, stats, current_event_text, len(trajectory_history), 420, annotated.shape[0])
        if panel.shape[0] != annotated.shape[0]:
            panel = cv2.resize(panel, (panel.shape[1], annotated.shape[0]), interpolation=cv2.INTER_AREA)
        return np.hstack([annotated, panel])

    def _draw_roi_and_line_on_original_frame(self, frame):
        annotated = frame.copy()
        if self.roi_points and len(self.roi_points) >= 3:
            roi_pts_list = _point_list_to_int_points(self.roi_points)
            if roi_pts_list:
                roi_pts = np.asarray(roi_pts_list, dtype=np.int32)
                cv2.polylines(annotated, [roi_pts], True, (255, 0, 255), 2, cv2.LINE_AA)
        if len(self.line_points) == 2:
            line_pts = _point_list_to_int_points(self.line_points)
            if len(line_pts) == 2:
                cv2.line(annotated, line_pts[0], line_pts[1], (0, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def _log_replay_geometry(self, frame_width: int, frame_height: int):
        label_width = int(self.video_panel.label.width())
        label_height = int(self.video_panel.label.height())
        transform = get_display_transform(frame_width, frame_height, label_width, label_height)
        raw_roi_points = _point_list_to_int_points(self.roi_points)
        raw_line_points = _point_list_to_int_points(self.line_points)
        converted_roi_points = frame_points_to_display_points(raw_roi_points, frame_width, frame_height, label_width, label_height)
        converted_line_points = frame_points_to_display_points(raw_line_points, frame_width, frame_height, label_width, label_height)
        log_key = (
            self.source_name,
            frame_width,
            frame_height,
            label_width,
            label_height,
            tuple(raw_roi_points),
            tuple(raw_line_points),
        )
        if log_key == self._last_replay_geometry_log_key:
            return
        self._last_replay_geometry_log_key = log_key
        print(
            "[ReplayGeometry] "
            f"video_name={self.source_name} "
            f"frame_width={frame_width} frame_height={frame_height} "
            f"label_width={label_width} label_height={label_height} "
            f"scale={transform['scale']:.6f} offset_x={transform['offset_x']} offset_y={transform['offset_y']} "
            f"raw_roi_points={raw_roi_points} converted_roi_points={converted_roi_points} "
            f"raw_line_points={raw_line_points} converted_line_points={converted_line_points}"
        )

    def export_replay_video(self):
        default_dir = Path(self.video_path).parent if self.video_path else Path.cwd()
        default_name = f"replay_{self.run_id or self.session_id}.mp4"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出回放视频",
            str(default_dir / default_name),
            "MP4 Video (*.mp4);;AVI Video (*.avi)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if output_path.suffix.lower() not in {".mp4", ".avi"}:
            if "AVI" in str(selected_filter).upper():
                output_path = output_path.with_suffix(".avi")
            else:
                output_path = output_path.with_suffix(".mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._exporting:
            QMessageBox.information(self, "正在导出", "当前已有导出任务正在运行。")
            return

        self._exporting = True
        self.btn_export_video.setEnabled(False)
        self.btn_play_pause.setEnabled(False)
        self.btn_step_prev.setEnabled(False)
        self.btn_step_next.setEnabled(False)
        self.btn_replay_start.setEnabled(False)
        self.timeline_slider.setEnabled(False)
        self.speed_combo.setEnabled(False)

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            self._exporting = False
            self._set_export_controls_enabled(True)
            QMessageBox.warning(self, "导出失败", f"无法打开视频: {self.video_path}")
            return

        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if source_width <= 0 or source_height <= 0:
            ok, first_frame = capture.read()
            if not ok:
                capture.release()
                self._exporting = False
                self._set_export_controls_enabled(True)
                QMessageBox.warning(self, "导出失败", "无法读取视频首帧。")
                return
            source_height, source_width = first_frame.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        export_width = source_width + 420
        export_height = source_height
        fps = self.video_fps if self.video_fps > 1e-6 else 25.0
        suffix = output_path.suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if suffix == ".avi" else "mp4v"))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (export_width, export_height))
        if not writer.isOpened():
            capture.release()
            self._exporting = False
            self._set_export_controls_enabled(True)
            QMessageBox.warning(self, "导出失败", f"无法创建视频文件: {output_path}")
            return

        progress = QProgressDialog("正在导出回放视频...", "取消", 0, self.max_frame_index + 1, self)
        progress.setWindowTitle("导出回放视频")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(True)

        trajectory_history = defaultdict(list)
        trajectory_index = 0
        canceled = False

        try:
            for frame_idx in range(self.max_frame_index + 1):
                if progress.wasCanceled():
                    canceled = True
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                while trajectory_index < len(self.trajectory_frame_indices):
                    source_frame_idx = self.trajectory_frame_indices[trajectory_index]
                    if source_frame_idx > frame_idx:
                        break
                    for item in self.trajectory_by_frame.get(source_frame_idx, []):
                        trajectory_history[int(item.get("track_id", 0) or 0)].append(tuple(item.get("point", (0, 0))))
                    trajectory_index += 1

                current_points = self.trajectory_by_frame.get(frame_idx, [])
                current_event_text = self._event_text_for_frame(frame_idx)
                current_traffic = self._sample_at_or_before(self.traffic_samples, self.traffic_frame_indices, frame_idx)
                current_fps = self._sample_at_or_before(self.fps_samples, self.fps_frame_indices, frame_idx)
                stats = {
                    "up": int(current_traffic.get("up", 0) if current_traffic else 0),
                    "down": int(current_traffic.get("down", 0) if current_traffic else 0),
                    "total": int(current_traffic.get("total", 0) if current_traffic else 0),
                    "fps": float(self._replay_fps),
                    "current": len(current_points),
                }

                export_frame = self._compose_export_frame(frame, frame_idx, trajectory_history, current_points, current_event_text, stats)
                if export_frame.shape[1] != export_width or export_frame.shape[0] != export_height:
                    export_frame = cv2.resize(export_frame, (export_width, export_height), interpolation=cv2.INTER_AREA)
                writer.write(export_frame)

                progress.setValue(frame_idx + 1)
                QApplication.processEvents()

            progress.setValue(self.max_frame_index + 1)
        finally:
            writer.release()
            capture.release()
            progress.close()
            self._exporting = False
            self._set_export_controls_enabled(True)

        if canceled:
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass
            QMessageBox.information(self, "已取消", "回放视频导出已取消。")
            return

        QMessageBox.information(self, "导出完成", f"回放视频已导出: {output_path}")

    def _set_export_controls_enabled(self, enabled: bool):
        for widget in (
            self.btn_replay_start,
            self.btn_play_pause,
            self.btn_step_prev,
            self.btn_step_next,
            self.btn_export_video,
            self.timeline_slider,
            self.speed_combo,
        ):
            widget.setEnabled(enabled)

    def _update_visuals(self, frame, frame_idx: int):
        import time
        now = time.perf_counter()
        if self._last_render_ts > 0:
            elapsed = now - self._last_render_ts
            if elapsed > 0.001:
                self._replay_fps = self._replay_fps * 0.85 + (1.0 / elapsed) * 0.15
        else:
            self._replay_fps = self.video_fps
        self._last_render_ts = now

        self._sync_trajectory_state(frame_idx)
        current_points = self.trajectory_by_frame.get(frame_idx, [])
        current_event_text = self._event_text_for_frame(frame_idx)
        current_traffic = self._sample_at_or_before(self.traffic_samples, self.traffic_frame_indices, frame_idx)
        current_fps = self._sample_at_or_before(self.fps_samples, self.fps_frame_indices, frame_idx)

        stats = {
            "up": int(current_traffic.get("up", 0) if current_traffic else 0),
            "down": int(current_traffic.get("down", 0) if current_traffic else 0),
            "total": int(current_traffic.get("total", 0) if current_traffic else 0),
            "fps": float(self._replay_fps),
            "current": len(current_points),
            "timestamp": self._format_frame_time(frame_idx),
            "frame_position": f"{frame_idx + 1} / {self.max_frame_index + 1}",
            "track_count": len(self._trajectory_history),
        }
        self.stats_panel.update_stats(stats)

        self.current_frame_index = frame_idx
        self.current_frame_label.setText(f"帧 {frame_idx + 1} / {self.max_frame_index + 1} | 时间 {self._format_frame_time(frame_idx)}")
        self.current_event_label.setText(current_event_text)

        self.video_panel.set_playback_overlay(
            track_history=self._trajectory_history,
            current_points=current_points,
            event_text=current_event_text,
            frame_text=f"批次 {self.run_id} | 帧 {frame_idx + 1} / {self.max_frame_index + 1} | {self._format_frame_time(frame_idx)}",
            max_trail_length=120,
        )
        frame_width = int(frame.shape[1]) if frame is not None else 0
        frame_height = int(frame.shape[0]) if frame is not None else 0
        self._log_replay_geometry(frame_width, frame_height)
        display_frame = self._draw_roi_and_line_on_original_frame(frame)
        self.video_panel.update_frame(_frame_to_qimage(display_frame), source_size=self.source_frame_size)
        self.trend_panel.set_playback_cursor(frame_idx)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)
        self._highlight_event_row(frame_idx)

    def _highlight_event_row(self, frame_idx: int):
        target_row = -1
        for row in range(self.event_table.rowCount()):
            try:
                row_frame = int(self.event_table.item(row, 0).text()) if self.event_table.item(row, 0) else 0
            except Exception:
                row_frame = 0
            if row_frame <= frame_idx:
                target_row = row
            else:
                break
        if target_row >= 0:
            self.event_table.selectRow(target_row)
            self.event_table.scrollToItem(self.event_table.item(target_row, 0), QAbstractItemView.PositionAtCenter)

    def _read_frame(self, frame_idx: int):
        frame_idx = max(0, min(int(frame_idx), self.max_frame_index))
        if self.capture.get(cv2.CAP_PROP_POS_FRAMES) != frame_idx:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.capture.read()
        if not ok:
            return False, None, frame_idx
        return True, frame, frame_idx

    def seek_frame(self, frame_idx: int):
        frame_idx = max(0, min(int(frame_idx), self.max_frame_index))
        ok, frame, resolved_frame = self._read_frame(frame_idx)
        if not ok:
            self.timer.stop()
            self.is_playing = False
            self.btn_play_pause.setText("播放")
            QMessageBox.warning(self, "回放结束", "无法读取对应帧，回放已停止。")
            return
        self._update_visuals(frame, resolved_frame)

    def _jump_to_start(self):
        self.seek_frame(0)

    def step_forward(self):
        self.seek_frame(min(self.current_frame_index + 1, self.max_frame_index))

    def step_backward(self):
        self.seek_frame(max(self.current_frame_index - 1, 0))

    def _on_speed_changed(self, *_args):
        if self.is_playing:
            self._restart_timer()

    def _playback_interval_ms(self) -> int:
        speed = float(self.speed_combo.currentData() or 1.0)
        interval = 1000.0 / max(1e-6, self.video_fps * max(0.1, speed))
        return max(15, int(interval))

    def _restart_timer(self):
        if self.timer.isActive():
            self.timer.stop()
        self.timer.start(self._playback_interval_ms())

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.btn_play_pause.setText("暂停" if self.is_playing else "播放")
        if self.is_playing:
            self._restart_timer()
        else:
            self.timer.stop()

    def _on_timer_tick(self):
        if self.current_frame_index >= self.max_frame_index:
            self.timer.stop()
            self.is_playing = False
            self.btn_play_pause.setText("播放")
            return
        ok, frame, resolved_frame = self._read_frame(self.current_frame_index + 1)
        if not ok:
            self.timer.stop()
            self.is_playing = False
            self.btn_play_pause.setText("播放")
            return
        self._update_visuals(frame, resolved_frame)

    def _seek_from_event_row(self, row: int, _column: int):
        item = self.event_table.item(row, 0)
        if item is None:
            return
        try:
            frame_idx = int(item.text())
        except Exception:
            return
        self.seek_frame(frame_idx)

    def closeEvent(self, event):
        if self._exporting:
            event.ignore()
            QMessageBox.information(self, "正在导出", "回放视频正在导出，请先完成或取消导出。")
            return
        try:
            self.timer.stop()
            if self.capture is not None:
                self.capture.release()
        except Exception:
            pass
        super().closeEvent(event)
