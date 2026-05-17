import collections
import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib
matplotlib.use("Agg")  # 强制无头模式解决多线程 backend 冲突，只借用 QTAgg 画布
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class TrendChartWidget(QWidget):
    def __init__(self, max_points=100):
        super().__init__()
        self.max_points = max_points

        # Matplotlib 重绘非常耗时：限制刷新频率（只刷新最新数据）。
        self._last_draw_ts = 0.0
        self._draw_min_interval = 0.25  # seconds, ~4Hz
        
        # 使用 deque 限制最多 100 个数据点
        self.up_data = collections.deque(maxlen=self.max_points)
        self.down_data = collections.deque(maxlen=self.max_points)
        self.total_data = collections.deque(maxlen=self.max_points)
        self.x_data = collections.deque(maxlen=self.max_points)
        self.fps_data = collections.deque(maxlen=self.max_points)
        self.fps_x_data = collections.deque(maxlen=self.max_points)
        self._fps_step = 0
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False   

        self.figure, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax, self.ax_fps = axes
        self._init_plot()
        
    def _init_plot(self):
        self.ax.clear()
        self.ax_fps.clear()
        self.ax.set_title("流量趋势图 - 最近 100 次更新")
        self.ax.set_xlabel("时间/分钟")
        self.ax.set_ylabel("流量计次")
        self.ax_fps.set_title("实时 FPS 曲线")
        self.ax_fps.set_xlabel("采样帧")
        self.ax_fps.set_ylabel("FPS")
        
        # 预先画上空线条，后续仅更新数据以提升性能
        self.line_up, = self.ax.plot([], [], label="Up", color="green", marker=".")
        self.line_down, = self.ax.plot([], [], label="Down", color="red", marker=".")
        self.line_total, = self.ax.plot([], [], label="Total", color="blue", marker=".")
        self.line_fps, = self.ax_fps.plot([], [], label="FPS", color="orange", marker=".")
        
        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.ax_fps.legend(loc="upper left")
        self.ax_fps.grid(True, alpha=0.3)
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
        
        # 动态调整坐标轴刻度范围
        self.ax.set_xlim(min(self.x_data), max(self.x_data) + 1 if self.x_data else 1)
        max_y = max(max(self.total_data, default=10), 10)
        self.ax.set_ylim(0, max_y * 1.2)
        
        # 局部重绘（节流）
        self._maybe_draw()

    def update_fps(self, fps_value):
        fps_value = float(fps_value)
        self.fps_data.append(fps_value)
        self.fps_x_data.append(self._fps_step)
        self._fps_step += 1

        self.line_fps.set_data(self.fps_x_data, self.fps_data)
        self.ax_fps.set_xlim(min(self.fps_x_data), max(self.fps_x_data) + 1 if self.fps_x_data else 1)
        max_fps = max(max(self.fps_data, default=10.0), 10.0)
        self.ax_fps.set_ylim(0, max_fps * 1.2)
        self._maybe_draw()

    def reset(self):
        self.up_data.clear()
        self.down_data.clear()
        self.total_data.clear()
        self.x_data.clear()
        self.fps_data.clear()
        self.fps_x_data.clear()
        self._fps_step = 0
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
        for item in points:
            self.add_data(
                int(item.get("up", 0)),
                int(item.get("down", 0)),
                int(item.get("total", 0)),
                step_identifier=int(item.get("minute", 0)),
            )


class TotalTrendWidget(QWidget):
    def __init__(self, max_points=60):
        super().__init__()
        self.max_points = max_points
        self._last_draw_ts = 0.0
        self._draw_min_interval = 0.25
        self.total_data = collections.deque(maxlen=self.max_points)
        self.x_data = collections.deque(maxlen=self.max_points)
        self.setMinimumHeight(400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.figure, self.ax = plt.subplots(1, 1, figsize=(6, 4))
        self.figure.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15)
        self.canvas = FigureCanvas(self.figure)
        from PyQt5.QtWidgets import QSizePolicy
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("Total 趋势（简化）", fontsize=12, pad=10)
        self.ax.set_xlabel("时间/分钟", fontsize=10)
        self.ax.set_ylabel("总通行", fontsize=10)
        self.ax.tick_params(axis='both', which='major', labelsize=9)
        self.line_total, = self.ax.plot([], [], label="Total", color="#2f80ed", linewidth=2)
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

    def add_total(self, total, step_identifier=None):
        self.total_data.append(int(total))
        if step_identifier is not None:
            self.x_data.append(int(step_identifier))
        else:
            self.x_data.append(0 if not self.x_data else self.x_data[-1] + 1)

        self.line_total.set_data(self.x_data, self.total_data)
        self.ax.set_xlim(min(self.x_data), max(self.x_data) + 1 if self.x_data else 1)
        max_y = max(max(self.total_data, default=10), 10)
        self.ax.set_ylim(0, max_y * 1.2)
        self._maybe_draw()

    def update_trend(self, payload):
        minute = int(payload.get("minute", 0))
        total = int(payload.get("total", 0))
        self.add_total(total, step_identifier=minute)

    def reset(self):
        self.total_data.clear()
        self.x_data.clear()
        self._last_draw_ts = 0.0
        self._init_plot()


class FpsChartWidget(QWidget):
    def __init__(self, max_points=120):
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
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.figure, self.ax = plt.subplots(1, 1, figsize=(6, 2.8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("FPS 曲线")
        self.ax.set_xlabel("采样帧")
        self.ax.set_ylabel("FPS")
        self.line_fps, = self.ax.plot([], [], label="FPS", color="orange", marker=".")
        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
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
        self._maybe_draw()

    def load_history(self, fps_values):
        self.reset()
        for value in fps_values:
            self.update_fps(float(value))

    def reset(self):
        self.fps_data.clear()
        self.x_data.clear()
        self._step = 0
        self._last_draw_ts = 0.0
        self._init_plot()
