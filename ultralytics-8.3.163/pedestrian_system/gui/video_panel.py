from PyQt5.QtCore import QEvent, QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

class VideoPanel(QWidget):
    roi_changed = pyqtSignal(list)
    line_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._frame_image = None
        self._image_rect = QRect()
        self._draw_mode = None
        self._roi_points = []
        self._line_points = []
        self._show_roi = True
        self._show_line = True
        self._current_pixmap = QPixmap()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.label = QLabel("视频加载区")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMouseTracking(True)
        self.label.installEventFilter(self)
        self.label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def update_frame(self, q_img):
        # worker 侧已经做过 QImage.copy() 以保证跨线程安全，这里避免二次深拷贝。
        self._frame_image = q_img
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

            if self._show_roi and len(self._roi_points) >= 2:
                roi_pen = QPen(Qt.magenta, 2)
                painter.setPen(roi_pen)
                points = [self._frame_to_display(p) for p in self._roi_points]
                for idx in range(len(points) - 1):
                    painter.drawLine(points[idx], points[idx + 1])
                if len(points) >= 3:
                    painter.drawLine(points[-1], points[0])

            if self._show_line and len(self._line_points) >= 1:
                line_pen = QPen(Qt.yellow, 2)
                painter.setPen(line_pen)
                pts = [self._frame_to_display(p) for p in self._line_points]
                for pt in pts:
                    painter.drawEllipse(pt, 4, 4)
                if len(pts) == 2:
                    painter.drawLine(pts[0], pts[1])

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
        if self._frame_image is None or self._image_rect.isNull():
            return None
        if not self._image_rect.contains(point):
            return None

        rel_x = (point.x() - self._image_rect.x()) / self._image_rect.width()
        rel_y = (point.y() - self._image_rect.y()) / self._image_rect.height()
        frame_x = int(rel_x * self._frame_image.width())
        frame_y = int(rel_y * self._frame_image.height())
        frame_x = max(0, min(self._frame_image.width() - 1, frame_x))
        frame_y = max(0, min(self._frame_image.height() - 1, frame_y))
        return (frame_x, frame_y)

    def _frame_to_display(self, point):
        if self._frame_image is None or self._image_rect.isNull():
            return QPoint(0, 0)
        rel_x = point[0] / max(1, self._frame_image.width())
        rel_y = point[1] / max(1, self._frame_image.height())
        x = int(self._image_rect.x() + rel_x * self._image_rect.width())
        y = int(self._image_rect.y() + rel_y * self._image_rect.height())
        return QPoint(x, y)
