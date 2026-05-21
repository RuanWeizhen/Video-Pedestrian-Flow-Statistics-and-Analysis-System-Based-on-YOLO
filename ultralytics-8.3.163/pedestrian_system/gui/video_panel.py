from PyQt5.QtCore import QEvent, QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from utils.coordinate_transform import display_to_frame_point, frame_points_to_display_points, get_display_transform

class VideoPanel(QWidget):
    roi_changed = pyqtSignal(list)
    line_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._frame_image = None
        self._image_rect = QRect()
        self._source_frame_size = None
        self._draw_mode = None
        self._roi_points = []
        self._line_points = []
        self._show_roi = True
        self._show_line = True
        self._current_pixmap = QPixmap()
        self._playback_track_history = {}
        self._playback_current_points = []
        self._playback_event_text = ""
        self._playback_frame_text = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("视频加载区")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setMouseTracking(True)
        self.label.installEventFilter(self)
        self.label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.label, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def set_source_frame_size(self, width: int | None, height: int | None):
        try:
            source_w = int(width or 0)
        except Exception:
            source_w = 0
        try:
            source_h = int(height or 0)
        except Exception:
            source_h = 0
        self._source_frame_size = (source_w, source_h) if source_w > 0 and source_h > 0 else None
        self._render()

    def update_frame(self, q_img, source_size=None):
        # worker 侧已经做过 QImage.copy() 以保证跨线程安全，这里避免二次深拷贝。
        self._frame_image = q_img
        if source_size is not None:
            try:
                source_w, source_h = source_size
                self.set_source_frame_size(source_w, source_h)
            except Exception:
                pass
        self._render()

    def set_draw_mode(self, mode):
        self._draw_mode = mode

    def set_annotations(self, roi_points=None, line_points=None):
        if roi_points is not None:
            self._roi_points = [tuple(map(int, p)) for p in roi_points]
            self.roi_changed.emit(list(self._roi_points))
        if line_points is not None:
            self._line_points = [tuple(map(int, p)) for p in line_points[:2]]
            self.line_changed.emit(list(self._line_points))
        self._render()

    def set_show_flags(self, show_roi=True, show_line=True):
        self._show_roi = bool(show_roi)
        self._show_line = bool(show_line)
        self._render()

    def set_playback_overlay(self, track_history=None, current_points=None, event_text: str = "", frame_text: str = "",
                             max_trail_length: int = 120):

        trimmed_history = {}
        for tid, pts in (track_history or {}).items():
            if len(pts) > max_trail_length:
                trimmed_history[tid] = list(pts)[-max_trail_length:]
            else:
                trimmed_history[tid] = list(pts)

        self._playback_track_history = trimmed_history
        self._playback_current_points = list(current_points or [])
        self._playback_event_text = str(event_text or "")
        self._playback_frame_text = str(frame_text or "")

    def clear_playback_overlay(self):
        self._playback_track_history = {}
        self._playback_current_points = []
        self._playback_event_text = ""
        self._playback_frame_text = ""

    def clear_annotations(self):
        self._roi_points = []
        self._line_points = []
        self.roi_changed.emit([])
        self.line_changed.emit([])
        self._render()

    def undo_last_annotation_point(self):
        if self._draw_mode == "roi":
            if self._roi_points:
                self._roi_points.pop()
                self.roi_changed.emit(list(self._roi_points))
                self._render()
        elif self._draw_mode == "line":
            if self._line_points:
                self._line_points.pop()
                self.line_changed.emit(list(self._line_points))
                self._render()

    def eventFilter(self, obj, event):
        if obj is self.label and event.type() == QEvent.MouseButtonPress and self._draw_mode:
            frame_point = self._display_to_frame(event.pos())
            if frame_point is None:
                return False

            if event.button() == Qt.LeftButton:
                if self._draw_mode == "roi":
                    self._roi_points.append(frame_point)
                    self.roi_changed.emit(list(self._roi_points))
                elif self._draw_mode == "line":
                    if len(self._line_points) >= 2:
                        self._line_points = [frame_point]
                    else:
                        self._line_points.append(frame_point)
                    self.line_changed.emit(list(self._line_points))
                self._render()
                return True

            if event.button() == Qt.RightButton and self._draw_mode == "roi":
                if self._roi_points:
                    self._roi_points.pop()
                    self.roi_changed.emit(list(self._roi_points))
                    self._render()
                    return True

        return super().eventFilter(obj, event)

    def _render(self):
        if self.label.width() <= 0 or self.label.height() <= 0:
            return

        canvas = QPixmap(self.label.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)

        if self._frame_image is not None and not self._frame_image.isNull():
            scaled_size = self._frame_image.size().scaled(self.label.size(), Qt.KeepAspectRatio)
            x = (self.label.width() - scaled_size.width()) // 2
            y = (self.label.height() - scaled_size.height()) // 2
            self._image_rect = QRect(x, y, scaled_size.width(), scaled_size.height())
            painter.drawImage(self._image_rect, self._frame_image)

            source_w, source_h = self._get_source_frame_size()
            display_w = self.label.width()
            display_h = self.label.height()

            if self._show_roi and len(self._roi_points) >= 2:
                roi_pen = QPen(Qt.magenta, 2)
                painter.setPen(roi_pen)
                points = frame_points_to_display_points(self._roi_points, source_w, source_h, display_w, display_h)
                for idx in range(len(points) - 1):
                    painter.drawLine(self._as_qpoint(points[idx]), self._as_qpoint(points[idx + 1]))
                if len(points) >= 3:
                    painter.drawLine(self._as_qpoint(points[-1]), self._as_qpoint(points[0]))

            if self._show_line and len(self._line_points) >= 1:
                line_pen = QPen(Qt.yellow, 2)
                painter.setPen(line_pen)
                pts = frame_points_to_display_points(self._line_points, source_w, source_h, display_w, display_h)
                for pt in pts:
                    painter.drawEllipse(self._as_qpoint(pt), 4, 4)
                if len(pts) == 2:
                    painter.drawLine(self._as_qpoint(pts[0]), self._as_qpoint(pts[1]))

            if self._playback_track_history:
                track_entries = sorted(
                    self._playback_track_history.items(),
                    key=lambda kv: len(kv[1]),
                    reverse=True,
                )
                visible_tracks = track_entries[:50]

                for track_id, points in visible_tracks:
                    num_points = len(points)
                    if num_points < 1:
                        continue
                    display_points = frame_points_to_display_points(points, source_w, source_h, display_w, display_h)
                    if len(display_points) < 2:
                        continue

                    base_color = QColor.fromHsv((int(track_id) * 47) % 360, 220, 255)

                    tail_alpha = 30
                    head_alpha = 255

                    tail_thick = 1
                    head_thick = 4

                    for idx in range(len(display_points) - 1):
                        ratio = float(idx) / float(max(1, num_points - 1))
                        alpha = int(tail_alpha + (head_alpha - tail_alpha) * ratio)
                        thickness = int(tail_thick + (head_thick - tail_thick) * ratio)

                        seg_color = QColor(base_color)
                        seg_color.setAlpha(alpha)
                        painter.setPen(QPen(seg_color, thickness))
                        painter.drawLine(
                            self._as_qpoint(display_points[idx]),
                            self._as_qpoint(display_points[idx + 1]),
                        )

                    if display_points:
                        painter.setPen(QPen(base_color, head_thick + 2))
                        painter.setBrush(base_color)
                        painter.drawEllipse(self._as_qpoint(display_points[-1]), 5, 5)

            if self._playback_current_points:
                for item in self._playback_current_points:
                    point = item.get("point") if isinstance(item, dict) else None
                    track_id = int(item.get("track_id", 0) or 0) if isinstance(item, dict) else 0
                    if point is None:
                        continue
                    display_point = self._frame_to_display(point)
                    color = QColor.fromHsv((track_id * 47) % 360, 220, 255)
                    painter.setPen(QPen(color, 3))
                    painter.drawEllipse(display_point, 6, 6)
                    painter.drawText(display_point.x() + 10, display_point.y() - 10, f"ID {track_id}")

            overlay_lines = []
            if self._playback_frame_text:
                overlay_lines.append(self._playback_frame_text)
            if self._playback_event_text:
                overlay_lines.append(self._playback_event_text)
            if overlay_lines:
                text = "  |  ".join(overlay_lines)
                text_rect = painter.boundingRect(0, 0, self.label.width() - 20, 80, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, text)
                rect = text_rect.adjusted(-10, -8, 12, 10)
                rect.moveTo(10, 10)
                painter.fillRect(rect, QColor(0, 0, 0, 160))
                painter.setPen(QPen(Qt.white, 1))
                painter.drawText(rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, text)

        else:
            self._image_rect = QRect()

        painter.end()
        # QPixmap 是隐式共享的；这里不做 deep copy，节省每帧内存拷贝。
        self._current_pixmap = canvas
        self.label.setPixmap(canvas)

    def save_current_frame(self, file_path):
        if self._current_pixmap.isNull():
            return False
        return bool(self._current_pixmap.save(str(file_path)))

    def _display_to_frame(self, point):
        source_w, source_h = self._get_source_frame_size()
        if source_w <= 0 or source_h <= 0:
            return None

        mapped = display_to_frame_point(point.x(), point.y(), source_w, source_h, self.label.width(), self.label.height())
        if mapped is None:
            return None
        return mapped

    def _frame_to_display(self, point):
        source_w, source_h = self._get_source_frame_size()
        if source_w <= 0 or source_h <= 0:
            return QPoint(0, 0)

        mapped = frame_points_to_display_points([point], source_w, source_h, self.label.width(), self.label.height())
        if not mapped:
            return QPoint(0, 0)
        x, y = mapped[0]
        return QPoint(x, y)

    def _get_source_frame_size(self):
        if self._source_frame_size is not None:
            return self._source_frame_size
        if self._frame_image is not None and not self._frame_image.isNull():
            return self._frame_image.width(), self._frame_image.height()
        return 0, 0

    def _as_qpoint(self, point):
        if isinstance(point, QPoint):
            return point
        try:
            x, y = point
        except Exception:
            return QPoint(0, 0)
        return QPoint(int(x), int(y))
