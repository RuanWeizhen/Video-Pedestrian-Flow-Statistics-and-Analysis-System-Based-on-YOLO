import collections
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib
matplotlib.use("Agg")  # 强制无头模式解决多线程 backend 冲突，只借用 QTAgg 画布
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class TrendChartWidget(QWidget):
    def __init__(self, max_points=100):
        super().__init__()
        self.max_points = max_points
        
        # 使用 deque 限制最多 100 个数据点
        self.up_data = collections.deque(maxlen=self.max_points)
        self.down_data = collections.deque(maxlen=self.max_points)
        self.total_data = collections.deque(maxlen=self.max_points)
        self.x_data = collections.deque(maxlen=self.max_points)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False   

        self.figure = plt.figure(figsize=(5,3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self._init_plot()
        
    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("流量趋势图 - 最近 100 次更新")
        self.ax.set_xlabel("时间/帧")
        self.ax.set_ylabel("流量计次")
        
        # 预先画上空线条，后续仅更新数据以提升性能
        self.line_up, = self.ax.plot([], [], label="Up", color="green", marker=".")
        self.line_down, = self.ax.plot([], [], label="Down", color="red", marker=".")
        self.line_total, = self.ax.plot([], [], label="Total", color="blue", marker=".")
        
        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        
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
        
        # 局部重绘
        self.canvas.draw_idle()

    def reset(self):
        self.up_data.clear()
        self.down_data.clear()
        self.total_data.clear()
        self.x_data.clear()
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
