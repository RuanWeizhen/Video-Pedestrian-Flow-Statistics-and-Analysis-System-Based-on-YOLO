import collections
import time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator, MultipleLocator

class TrendChartWidget(QWidget):
    def __init__(self, max_points=100):
        super().__init__()
        self.max_points = max_points

        self._last_draw_ts = 0.0
        self._draw_min_interval = 0.25

        self.up_data = collections.deque(maxlen=self.max_points)
        self.down_data = collections.deque(maxlen=self.max_points)
        self.total_data = collections.deque(maxlen=self.max_points)
        self.x_data = collections.deque(maxlen=self.max_points)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 4.2))
        self.figure.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.12)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("流量趋势图 - 最近 100 次更新")
        self.ax.set_xlabel("采样序号 (帧/分钟)")
        self.ax.set_ylabel("流量计次")

        self.line_up, = self.ax.plot([], [], label="Up", color="green", marker=".")
        self.line_down, = self.ax.plot([], [], label="Down", color="red", marker=".")
        self.line_total, = self.ax.plot([], [], label="Total", color="blue", marker=".")

        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune="both"))
        self.ax.tick_params(axis="x", rotation=30)
        self.canvas.draw()

    def _maybe_draw(self):
        if not self.isVisible():
            return
        now = time.perf_counter()
        if now - self._last_draw_ts < self._draw_min_interval:
            return
        self._last_draw_ts = now
        self.canvas.draw_idle()
        
    def add_data(self, up, down, total, step_identifier=None):
        """
        供外部调用的专用方法：传入上行、下行与总数，自动更新趋势图。
        限制保留最多 100 帧（或100次采样）。由于调用是通过主线程信号槽，此处保证了线程安全。
        """
        self.up_data.append(up)
        self.down_data.append(down)
        self.total_data.append(total)
        
        if step_identifier is not None:
            self.x_data.append(step_identifier)
        else:
            if not self.x_data:
                self.x_data.append(0)
            else:
                self.x_data.append(self.x_data[-1] + 1)
            
        self.line_up.set_data(self.x_data, self.up_data)
        self.line_down.set_data(self.x_data, self.down_data)
        self.line_total.set_data(self.x_data, self.total_data)
        
        self.ax.set_xlim(min(self.x_data), max(self.x_data) + 1 if self.x_data else 1)
        max_y = max(max(self.total_data, default=10), 10)
        self.ax.set_ylim(0, max_y * 1.2)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune="both"))
        self.ax.tick_params(axis="x", rotation=30)

        self._maybe_draw()

    def reset(self):
        self.up_data.clear()
        self.down_data.clear()
        self.total_data.clear()
        self.x_data.clear()
        self._last_draw_ts = 0.0
        self._init_plot()

    def update_trend(self, payload):
        """
        兼容现有的 worker `trend_updated` 信号（payload字典格式）
        """
        # 可以用 payload="minute" 字段当作横坐标 step
        minute = int(payload.get("minute", 0))
        up = int(payload.get("up", 0))
        down = int(payload.get("down", 0))
        total = int(payload.get("total", up + down))
        
        self.add_data(up, down, total, step_identifier=minute)

    def load_history(self, points):
        """points: list[dict] with keys minute/up/down/total"""
        self.reset()
        running_total = 0
        for item in points:
            up = int(item.get("up", 0))
            down = int(item.get("down", 0))
            raw_total = int(item.get("total", 0))
            running_total = max(running_total, raw_total)
            self.add_data(
                up,
                down,
                running_total,
                step_identifier=int(item.get("minute", 0)),
            )

    def set_replay_data(self, trend_points, fps_points=None, cursor_x=None, title_suffix=""):
        self.ax.clear()
        trend_title = "流量趋势图 - 回放"
        if title_suffix:
            trend_title = f"{trend_title} · {title_suffix}"
        self.ax.set_title(trend_title)
        self.ax.set_xlabel("帧序号")
        self.ax.set_ylabel("流量计次")

        trend_points = sorted(trend_points or [], key=lambda item: int(item.get("frame_idx", item.get("minute", 0)) or 0))
        x_values = [int(item.get("frame_idx", item.get("minute", idx)) or idx) for idx, item in enumerate(trend_points)]
        up_values = [int(item.get("up", 0) or 0) for item in trend_points]
        down_values = [int(item.get("down", 0) or 0) for item in trend_points]
        raw_total_values = [int(item.get("total", 0) or 0) for item in trend_points]
        running_max = 0
        total_values = []
        for v in raw_total_values:
            running_max = max(running_max, v)
            total_values.append(running_max)

        self.line_up, = self.ax.plot(x_values, up_values, label="Up", color="green", marker=".")
        self.line_down, = self.ax.plot(x_values, down_values, label="Down", color="red", marker=".")
        self.line_total, = self.ax.plot(x_values, total_values, label="Total", color="blue", marker=".")

        if x_values:
            self.ax.set_xlim(min(x_values), max(x_values) + 1)
            max_y = max(max(total_values, default=10), 10)
            self.ax.set_ylim(0, max_y * 1.2)
        else:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 10)

        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune="both"))
        self.ax.tick_params(axis="x", rotation=30)
        self._replay_cursor_line = self.ax.axvline(int(cursor_x), color="#d97706", linestyle="--", linewidth=1.8, alpha=0.9) if cursor_x is not None else None
        self.canvas.draw_idle()

    def set_playback_cursor(self, cursor_x):
        if getattr(self, "_replay_cursor_line", None) is not None:
            self._replay_cursor_line.set_xdata([cursor_x, cursor_x])
        self.canvas.draw_idle()


class TotalTrendWidget(QWidget):
    def __init__(self, max_points=120):
        super().__init__()
        self.max_points = max_points
        self._last_draw_ts = 0.0
        self._draw_min_interval = 0.25
        self.total_data = collections.deque(maxlen=self.max_points)
        self.x_data = collections.deque(maxlen=self.max_points)
        self._last_cumulative = None
        self._global_max = 0
        self._stale_threshold_seconds = 10.0
        self._last_update_time = time.monotonic()
        self.setMinimumHeight(400)
        self._waiting_label = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.figure, self.ax = plt.subplots(1, 1, figsize=(6, 4))
        self.figure.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.15)
        self.canvas = FigureCanvas(self.figure)
        from PyQt5.QtWidgets import QSizePolicy
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        from PyQt5.QtWidgets import QLabel
        self._waiting_label = QLabel("等待数据...")
        self._waiting_label.setAlignment(Qt.AlignCenter)
        self._waiting_label.setStyleSheet(
            "color: #8899aa; font-size: 16px; font-weight: 600;"
            "background: rgba(240,244,250,0.6); border-radius: 12px;"
        )
        layout.addWidget(self._waiting_label)
        self._waiting_label.hide()

        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("累计通行人数趋势", fontsize=12, pad=10)
        self.ax.set_xlabel("时间/分钟", fontsize=10)
        self.ax.set_ylabel("累计通行人数", fontsize=10)
        self.ax.tick_params(axis='both', which='major', labelsize=9)
        self.ax.tick_params(axis="x", rotation=30)
        self.ax.xaxis.set_major_locator(MultipleLocator(0.5))
        self.line_total, = self.ax.plot([], [], marker="o", markersize=4, label="累计通行人数", color="#2f80ed", linewidth=2)
        self.ax.legend(loc="upper left", fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

    def _maybe_draw(self):
        if not self.isVisible():
            return
        now = time.perf_counter()
        if now - self._last_draw_ts < self._draw_min_interval:
            return
        self._last_draw_ts = now
        self.canvas.draw_idle()

    def add_cumulative_total(self, total, step_identifier=None):
        self._last_update_time = time.monotonic()
        total = int(total)
        if step_identifier is not None:
            step = int(step_identifier)
        else:
            step = 0 if not self.x_data else self.x_data[-1] + 1

        if total < self._global_max:
            total = self._global_max
        else:
            self._global_max = total

        if self.x_data and step <= self.x_data[-1]:
            idx = len(self.x_data) - 1
            while idx >= 0 and self.x_data[idx] >= step:
                if total > self.total_data[idx]:
                    self.total_data[idx] = total
                idx -= 1
            self._last_cumulative = total
            self._redraw()
            return

        self.total_data.append(total)
        self.x_data.append(step)
        self._last_cumulative = total
        self._redraw()

    def _redraw(self):
        if not self.x_data:
            if self._waiting_label:
                self._waiting_label.show()
                self.canvas.hide()
            return
        if self._waiting_label:
            self._waiting_label.hide()
            self.canvas.show()
        stale = (time.monotonic() - self._last_update_time) > self._stale_threshold_seconds
        self.line_total.set_data(self.x_data, self.total_data)
        self.ax.set_xlim(min(self.x_data), max(self.x_data) + 1)
        max_y = max(max(self.total_data, default=10), 10)
        self.ax.set_ylim(0, max_y * 1.15)
        title = "累计通行人数趋势"
        if stale:
            title = "⚠ 累计通行人数趋势 - 数据中断"
        self.ax.set_title(title, fontsize=12, pad=10)
        self._maybe_draw()

    def mark_stale(self):
        self._redraw()

    def update_trend(self, payload):
        minute = int(payload.get("minute", 0))
        total = int(payload.get("total", 0))
        self.add_total(total, step_identifier=minute)

    def add_total(self, total, step_identifier=None):
        self.add_cumulative_total(total, step_identifier=step_identifier)

    def reset(self):
        self.total_data.clear()
        self.x_data.clear()
        self._last_cumulative = None
        self._global_max = 0
        self._last_update_time = time.monotonic()
        self._last_draw_ts = 0.0
        self._init_plot()
        if self._waiting_label:
            self._waiting_label.show()
            self.canvas.hide()


class FpsChartWidget(QWidget):
    def __init__(self, max_points=200):
        super().__init__()
        self.max_points = max_points
        self._last_draw_ts = 0.0
        self._draw_min_interval = 0.25
        self.fps_data = collections.deque(maxlen=self.max_points)
        self.x_data = collections.deque(maxlen=self.max_points)
        self._step = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 3.6))
        self.figure.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.16)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("FPS 曲线 - 最近 200 帧")
        self.ax.set_xlabel("采样帧")
        self.ax.set_ylabel("FPS")
        self.line_fps, = self.ax.plot([], [], label="FPS", color="#f59e0b", linewidth=1.2, marker=".", markersize=2)
        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune="both"))
        self.ax.tick_params(axis="x", rotation=30)
        self.canvas.draw()

    def _maybe_draw(self):
        if not self.isVisible():
            return
        now = time.perf_counter()
        if now - self._last_draw_ts < self._draw_min_interval:
            return
        self._last_draw_ts = now
        self.canvas.draw_idle()

    def update_fps(self, fps_value):
        self.fps_data.append(float(fps_value))
        self.x_data.append(self._step)
        self._step += 1

        self.line_fps.set_data(self.x_data, self.fps_data)
        self.ax.set_xlim(min(self.x_data), max(self.x_data) + 1 if self.x_data else 1)
        max_fps = max(max(self.fps_data, default=10.0), 10.0)
        self.ax.set_ylim(0, max_fps * 1.2)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune="both"))
        self._maybe_draw()

    def load_history(self, fps_values):
        self.reset()
        for value in fps_values:
            self.update_fps(float(value))
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune="both"))
        self.ax.tick_params(axis="x", rotation=30)

    def set_replay_data(self, fps_points, cursor_x=None, title_suffix=""):
        self.ax.clear()
        title = "FPS 曲线 - 回放"
        if title_suffix:
            title = f"{title} · {title_suffix}"
        self.ax.set_title(title)
        self.ax.set_xlabel("帧序号")
        self.ax.set_ylabel("FPS")

        fps_points = sorted(fps_points or [], key=lambda item: int(item.get("frame_idx", item.get("minute", 0)) or 0))
        x_values = [int(item.get("frame_idx", item.get("minute", idx)) or idx) for idx, item in enumerate(fps_points)]
        fps_values = [float(item.get("fps", item.get("avg_fps", 0.0)) or 0.0) for item in fps_points]

        self.line_fps, = self.ax.plot(x_values, fps_values, label="FPS", color="orange", marker=".")
        self._replay_cursor_line = self.ax.axvline(int(cursor_x), color="#d97706", linestyle="--", linewidth=1.8, alpha=0.9) if cursor_x is not None else None
        if x_values:
            self.ax.set_xlim(min(x_values), max(x_values) + 1)
            max_fps = max(max(fps_values, default=10.0), 10.0)
            self.ax.set_ylim(0, max_fps * 1.2)
        else:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 10)
        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune="both"))
        self.ax.tick_params(axis="x", rotation=30)
        self.canvas.draw_idle()

    def set_playback_cursor(self, cursor_x):
        if getattr(self, "_replay_cursor_line", None) is not None:
            self._replay_cursor_line.set_xdata([cursor_x, cursor_x])
        self.canvas.draw_idle()

    def reset(self):
        self.fps_data.clear()
        self.x_data.clear()
        self._step = 0
        self._last_draw_ts = 0.0
        self._init_plot()
