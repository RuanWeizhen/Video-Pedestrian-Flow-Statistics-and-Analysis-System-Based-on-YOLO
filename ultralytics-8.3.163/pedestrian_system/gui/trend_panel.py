from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class TrendPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

        self.figure = plt.figure(figsize=(5,3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("流量趋势图")
        self.ax.set_xlabel("分钟")
        self.ax.set_ylabel("累计人数")
        self._x = []
        self._up = []
        self._down = []
        self._total = []
        self.canvas.draw()

    def reset(self):
        self._x = []
        self._up = []
        self._down = []
        self._total = []
        self._redraw()

    def update_trend(self, payload):
        minute = int(payload.get("minute", 0))
        up = int(payload.get("up", 0))
        down = int(payload.get("down", 0))
        total = int(payload.get("total", up + down))

        if self._x and minute == self._x[-1]:
            self._up[-1] = up
            self._down[-1] = down
            self._total[-1] = total
        else:
            self._x.append(minute)
            self._up.append(up)
            self._down.append(down)
            self._total.append(total)

        max_points = 180
        if len(self._x) > max_points:
            self._x = self._x[-max_points:]
            self._up = self._up[-max_points:]
            self._down = self._down[-max_points:]
            self._total = self._total[-max_points:]

        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.set_title("流量趋势图")
        self.ax.set_xlabel("分钟")
        self.ax.set_ylabel("累计人数")
        if self._x:
            self.ax.plot(self._x, self._up, marker="o", label="Up")
            self.ax.plot(self._x, self._down, marker="o", label="Down")
            self.ax.plot(self._x, self._total, marker="o", label="Total")
            self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()
