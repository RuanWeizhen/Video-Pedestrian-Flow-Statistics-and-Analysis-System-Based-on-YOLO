from pathlib import Path
from datetime import datetime
from collections import deque
import time
import csv
import os
import platform
import shutil

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
    QFrame,
    QSplitter,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from openpyxl import Workbook

from .annotation_panel import AnnotationPanel
from .control_panel import ControlPanel
from .event_table_panel import EventTablePanel
from .log_panel import LogPanel
from .stats_panel import StatsPanel
from .system_info_panel import SystemInfoPanel
from .trend_panel import TrendChartWidget, TotalTrendWidget, FpsChartWidget
from .video_panel import VideoPanel
from .dashboard_widgets import DashboardMetricCard, DashboardSection, SidebarButton, UserBadge, build_metric_row
from .experiment_table_panel import ExperimentTablePanel
from .worker import WorkerThread
from utils.auth_manager import AuthManager
from .employee import EmployeeManagementPage
from utils.db_manager import DatabaseManager


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
    def __init__(self, current_user: dict | None = None):
        super().__init__()
        self.current_user = current_user or {"username": "未登录", "role": "管理员"}
        self.setWindowTitle("视频行人客流量统计与分析系统")
        self.resize(1200, 800)

        self.current_config_path = None
        self.current_config = self._default_config()
        self.auth_manager = AuthManager()
        self.history_db = DatabaseManager("traffic.db")
        self.tool_runners = []
        self.recent_events = deque(maxlen=8)
        self.current_source_fps = 25.0
        self.current_video_resolution = "-"
        self.current_runtime_start = None
        self.current_frame_number = 0
        self.current_total_frames = 0
        self.current_total_people = 0
        self.current_up_count = 0
        self.current_down_count = 0
        self.current_detecting = False
        self.current_source_name = "未选择"
        self.current_avg_fps = 0.0
        self.current_device = self._device_label()
        self.current_detector_type = self._detector_type_label()
        self.current_tracker_type = "DeepSORT"
        self.current_onnx_enabled = False
        self.event_history = []
        self.trend_history = []
        self.fps_history = []

        self.worker = WorkerThread()
        self.worker.set_config(self.current_config)
        self.worker.warmup_detector_async()
        self.init_ui()
        self._init_menus()
        self.connect_signals()
        self.apply_runtime_params(self.annotation_panel.get_params())
        self.apply_user_permissions()

    def init_ui(self):
        self.setObjectName("MainWindow")

        self.video_panel = VideoPanel()
        self.log_panel = LogPanel()
        self.event_table_panel = EventTablePanel()
        self.filtered_event_table_panel = EventTablePanel()
        self.experiment_table_panel = ExperimentTablePanel()
        self.system_info_panel = SystemInfoPanel()
        self.control_panel = ControlPanel()
        self.stats_panel = StatsPanel()
        self.annotation_panel = AnnotationPanel()

        # 检测页仅保留实时检测相关按钮，模型/配置类入口迁移到系统管理页
        self.control_panel.btn_export_onnx.hide()
        self.control_panel.btn_quantize.hide()
        self.control_panel.btn_benchmark.hide()

        self.home_trend_panel = TotalTrendWidget()
        self.stats_history_panel = TrendChartWidget()
        self.stats_fps_panel = FpsChartWidget()

        self.home_metric_total = DashboardMetricCard("总通行人数", "0", "本次会话累计")
        self.home_metric_up = DashboardMetricCard("Up", "0", "上行人数")
        self.home_metric_down = DashboardMetricCard("Down", "0", "下行人数")
        self.home_metric_current = DashboardMetricCard("当前人数", "0", "当前画面")
        self.home_metric_fps = DashboardMetricCard("FPS", "0.0", "实时性能")

        self.detect_metric_total = DashboardMetricCard("总通行人数", "0", "实时统计")
        self.detect_metric_up = DashboardMetricCard("Up", "0", "上行人数")
        self.detect_metric_down = DashboardMetricCard("Down", "0", "下行人数")
        self.detect_metric_current = DashboardMetricCard("当前人数", "0", "当前画面")
        self.detect_metric_fps = DashboardMetricCard("FPS", "0.0", "实时性能")

        self.sidebar_buttons: dict[str, SidebarButton] = {}
        self.page_titles: dict[str, tuple[str, str]] = {
            "home": ("系统首页", "查看当前运行状态、关键指标与简化趋势"),
            "detection": ("行人检测", "视频画面、运行控制和参数标注集中管理"),
            "management": ("系统管理", "模型管理、配置管理和系统维护"),
            "statistics": ("数据统计", "历史分析、参数实验和结果导出"),
            "profile": ("个人信息", "查看个人资料和账号安全信息"),
            "password": ("员工管理", "管理员工账号、角色和账号状态"),
        }

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QHBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar_widget = self._build_sidebar()
        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(18, 18, 18, 18)
        content_layout.setSpacing(14)

        self.header_widget = self._build_header()
        content_layout.addWidget(self.header_widget)

        self.pages = QStackedWidget()
        content_layout.addWidget(self.pages, 1)

        root_layout.addWidget(self.sidebar_widget)
        root_layout.addWidget(self.content_widget, 1)

        self._build_pages()
        self._apply_dashboard_style()
        self._switch_page("home")

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(10)

        brand = QWidget()
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(12, 10, 12, 10)
        brand_layout.setSpacing(4)
        brand_title = QLabel("客流统计系统")
        brand_title.setObjectName("SidebarBrandTitle")
        brand_subtitle = QLabel("Dashboard UI")
        brand_subtitle.setObjectName("SidebarBrandSubtitle")
        brand_layout.addWidget(brand_title)
        brand_layout.addWidget(brand_subtitle)
        layout.addWidget(brand)

        for key, text in (
            ("home", "系统首页"),
            ("detection", "行人检测"),
            ("management", "系统管理"),
            ("statistics", "数据统计"),
            ("profile", "个人中心"),
            ("password", "员工管理"),
        ):
            button = SidebarButton(text)
            button.clicked.connect(lambda _=False, page_key=key: self._switch_page(page_key))
            self.sidebar_buttons[key] = button
            layout.addWidget(button)

        layout.addStretch()

        self.btn_logout = QPushButton("退出登录")
        self.btn_logout.setObjectName("SidebarLogoutButton")
        self.btn_logout.setCursor(Qt.PointingHandCursor)
        self.btn_logout.clicked.connect(self.close)
        layout.addWidget(self.btn_logout)
        return sidebar
        layout.addWidget(self.btn_logout)
        return sidebar

    def _build_header(self):
        header = QWidget()
        header.setObjectName("PageHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(12)

        title_box = QWidget()
        title_layout = QVBoxLayout(title_box)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        self.page_title_label = QLabel("系统首页")
        self.page_title_label.setObjectName("PageTitle")
        self.page_desc_label = QLabel("总览关键指标、趋势和最近运行摘要")
        self.page_desc_label.setObjectName("PageSubtitle")
        title_layout.addWidget(self.page_title_label)
        title_layout.addWidget(self.page_desc_label)

        self.user_badge = UserBadge("当前用户", self.current_user.get("username", "未登录"))
        self.role_badge = UserBadge("身份", self.current_user.get("role", "管理员"))

        layout.addWidget(title_box, 1)
        layout.addWidget(self.user_badge)
        layout.addWidget(self.role_badge)
        return header

    def _make_scroll_page(self, widget: QWidget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _build_pages(self):
        self.page_widgets = {}

        home_page = self._build_home_page()
        detection_page = self._build_detection_page()
        management_page = self._build_management_page()
        statistics_page = self._build_statistics_page()
        profile_page = self._build_profile_page()
        password_page = self._build_password_page()

        for key, widget in (
            ("home", home_page),
            ("detection", detection_page),
            ("management", management_page),
            ("statistics", statistics_page),
            ("profile", profile_page),
            ("password", password_page),
        ):
            self.page_widgets[key] = widget
            # 统计页内部已经实现了独立的 QScrollArea，避免嵌套滚动条
            if key == "statistics":
                self.pages.addWidget(widget)
            else:
                self.pages.addWidget(self._make_scroll_page(widget))

    def _build_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        metrics_row = build_metric_row([
            self.home_metric_total,
            self.home_metric_up,
            self.home_metric_down,
            self.home_metric_current,
            self.home_metric_fps,
        ])
        layout.addLayout(metrics_row)

        status_section = DashboardSection("系统运行状态", "当前检测状态与运行上下文")
        status_form = QFormLayout()
        self.home_status_detect = QLabel("未运行")
        self.home_status_source = QLabel("未选择")
        self.home_status_model = QLabel("未加载")
        self.home_status_config = QLabel("未加载")
        status_form.addRow("检测状态", self.home_status_detect)
        status_form.addRow("视频源", self.home_status_source)
        status_form.addRow("模型状态", self.home_status_model)
        status_form.addRow("配置状态", self.home_status_config)
        status_section.body_layout.addLayout(status_form)
        layout.addWidget(status_section)

        trend_section = DashboardSection("Total 趋势", "仅展示最近时间段总通行趋势")
        trend_section.set_body_widget(self.home_trend_panel)
        layout.addWidget(trend_section)

        summary_section = DashboardSection("最近运行摘要", "展示最近事件和状态变化")
        self.home_summary_list = QListWidget()
        self.home_summary_list.setMinimumHeight(180)
        summary_section.set_body_widget(self.home_summary_list)
        layout.addWidget(summary_section)

        self._refresh_home_status_card()

        return page

    def _build_detection_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        metrics_row = build_metric_row([
            self.detect_metric_total,
            self.detect_metric_up,
            self.detect_metric_down,
            self.detect_metric_current,
            self.detect_metric_fps,
        ])
        layout.addLayout(metrics_row)

        splitter = QSplitter(Qt.Horizontal)

        video_section = DashboardSection("视频画面", "显示检测画面与标注交互")
        video_section.set_body_widget(self.video_panel)

        right_tabs = QTabWidget()

        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(self.control_panel)
        control_layout.addStretch()
        right_tabs.addTab(control_tab, "运行控制")

        annotation_tab = QWidget()
        annotation_layout = QVBoxLayout(annotation_tab)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        annotation_layout.addWidget(self.annotation_panel)
        annotation_layout.addStretch()
        right_tabs.addTab(annotation_tab, "参数与标注")

        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_tabs = QTabWidget()
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self.log_panel)
        info_tabs.addTab(log_tab, "运行日志")

        sys_tab = QWidget()
        sys_layout = QVBoxLayout(sys_tab)
        sys_layout.setContentsMargins(0, 0, 0, 0)
        sys_layout.addWidget(self.system_info_panel)
        info_tabs.addTab(sys_tab, "系统信息")
        info_layout.addWidget(info_tabs)
        right_tabs.addTab(info_tab, "状态详情")

        splitter.addWidget(video_section)
        splitter.addWidget(right_tabs)
        splitter.setSizes([820, 420])
        layout.addWidget(splitter, 1)
        return page

    def _build_statistics_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        query_section = DashboardSection("查询条件", "按时间、视频和方向筛选统计数据")
        query_widget = QWidget()
        query_layout = QHBoxLayout(query_widget)
        query_layout.setContentsMargins(0, 0, 0, 0)
        query_layout.setSpacing(10)

        self.query_time_start = QLineEdit("")
        self.query_time_start.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.query_time_end = QLineEdit("")
        self.query_time_end.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.query_video_name = QComboBox()
        self.query_video_name.addItem("全部视频")
        self.query_direction = QComboBox()
        self.query_direction.addItems(["All", "Up", "Down"])
        self.query_bucket = QComboBox()
        self.query_bucket.addItems(["按分钟", "每5分钟"])
        self.btn_query = QPushButton("查询")
        self.btn_query_reset = QPushButton("重置")
        self.btn_query.clicked.connect(self.apply_statistics_filters)
        self.btn_query_reset.clicked.connect(self.reset_statistics_filters)

        query_layout.addWidget(QLabel("开始"))
        query_layout.addWidget(self.query_time_start)
        query_layout.addWidget(QLabel("结束"))
        query_layout.addWidget(self.query_time_end)
        query_layout.addWidget(QLabel("视频"))
        query_layout.addWidget(self.query_video_name)
        query_layout.addWidget(QLabel("方向"))
        query_layout.addWidget(self.query_direction)
        query_layout.addWidget(QLabel("统计粒度"))
        query_layout.addWidget(self.query_bucket)
        query_layout.addWidget(self.btn_query)
        query_layout.addWidget(self.btn_query_reset)
        query_section.set_body_widget(query_widget)
        layout.addWidget(query_section)

        # 把剩余内容放入可滚动区域：保证左侧侧栏和顶部 header 固定
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(14)

        summary_row = build_metric_row([
            DashboardMetricCard("查询结果总数", "0", "过滤后事件数"),
            DashboardMetricCard("Up 总数", "0", "过滤后"),
            DashboardMetricCard("Down 总数", "0", "过滤后"),
            DashboardMetricCard("平均 FPS", "0.0", "会话均值"),
        ])
        summary_widget = QWidget()
        summary_widget.setLayout(summary_row)
        summary_widget.setMinimumHeight(100)
        summary_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stats_query_total = summary_row.itemAt(0).widget()
        self.stats_query_up = summary_row.itemAt(1).widget()
        self.stats_query_down = summary_row.itemAt(2).widget()
        self.stats_query_fps = summary_row.itemAt(3).widget()
        scroll_layout.addWidget(summary_widget)

        # 趋势图与 FPS: 提高高度，确保图表不会被压缩
        chart_splitter = QSplitter(Qt.Horizontal)
        history_section = DashboardSection("历史趋势图", "展示 Up/Down/Total 历史变化")
        history_section.set_body_widget(self.stats_history_panel)
        history_section.setMinimumHeight(440)
        self.stats_history_panel.setMinimumHeight(420)
        self.stats_history_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        fps_section = DashboardSection("FPS 曲线", "展示系统实时性能变化")
        fps_section.set_body_widget(self.stats_fps_panel)
        fps_section.setMinimumHeight(360)
        self.stats_fps_panel.setMinimumHeight(320)
        self.stats_fps_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        chart_splitter.addWidget(history_section)
        chart_splitter.addWidget(fps_section)
        chart_splitter.setSizes([760, 420])
        scroll_layout.addWidget(chart_splitter)

        # 事件表
        table_section = DashboardSection("事件表", "时间、事件类型、目标 ID、计数线名称")
        table_section.set_body_widget(self.filtered_event_table_panel)
        table_section.setMinimumHeight(460)
        self.filtered_event_table_panel.setMinimumHeight(420)
        self.filtered_event_table_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_layout.addWidget(table_section)

        # 参数实验表
        exp_section = DashboardSection("参数实验结果表", "不同 conf / iou 下的性能与计数结果")
        exp_section.set_body_widget(self.experiment_table_panel)
        exp_section.setMinimumHeight(340)
        self.experiment_table_panel.setMinimumHeight(300)
        self.experiment_table_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_layout.addWidget(exp_section)

        # 导出区域放在底部，可滚动查看
        export_section = DashboardSection("导出", "导出事件表、统计结果、图表截图")
        export_widget = QWidget()
        export_layout = QHBoxLayout(export_widget)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(10)
        self.btn_export_event_table = QPushButton("导出事件表")
        self.btn_export_stats_result = QPushButton("导出统计结果")
        self.btn_export_chart_snapshot = QPushButton("导出图表截图")
        self.btn_export_history_data = QPushButton("导出数据库历史结果")
        self.btn_export_event_table.clicked.connect(self.export_filtered_events)
        self.btn_export_stats_result.clicked.connect(self.on_export_results_requested)
        self.btn_export_chart_snapshot.clicked.connect(self.export_statistics_charts)
        self.btn_export_history_data.clicked.connect(self.export_database_history_results)
        export_layout.addWidget(self.btn_export_event_table)
        export_layout.addWidget(self.btn_export_stats_result)
        export_layout.addWidget(self.btn_export_chart_snapshot)
        export_layout.addWidget(self.btn_export_history_data)
        export_layout.addStretch()
        export_section.set_body_widget(export_widget)
        export_section.setMinimumHeight(90)
        scroll_layout.addWidget(export_section)

        # 总体滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(scroll_container)
        layout.addWidget(scroll, 1)

        self.reset_statistics_filters()

        return page

    def _build_management_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(14)

        top_row = QHBoxLayout()

        model_card = DashboardSection("模型管理", "当前模型信息与模型工具")
        model_body = QWidget()
        model_layout = QVBoxLayout(model_body)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.management_model_path_label = QLabel("-")
        self.management_model_name_label = QLabel("-")
        self.management_model_type_label = QLabel("-")
        self.management_imgsz_label = QLabel("-")
        self.management_device_label = QLabel("-")
        model_form = QFormLayout()
        model_form.addRow("当前模型路径", self.management_model_path_label)
        model_form.addRow("模型名称", self.management_model_name_label)
        model_form.addRow("模型类型", self.management_model_type_label)
        model_form.addRow("输入尺寸", self.management_imgsz_label)
        model_form.addRow("当前设备", self.management_device_label)
        model_layout.addLayout(model_form)

        model_btn_row1 = QHBoxLayout()
        self.btn_export_onnx_page = QPushButton("导出 ONNX")
        self.btn_quantize_page = QPushButton("量化 ONNX")
        self.btn_benchmark_page = QPushButton("Benchmark ONNX")
        self.btn_export_onnx_page.clicked.connect(self.on_export_onnx_requested)
        self.btn_quantize_page.clicked.connect(self.on_quantize_onnx_requested)
        self.btn_benchmark_page.clicked.connect(self.on_benchmark_onnx_requested)
        for button in (self.btn_export_onnx_page, self.btn_quantize_page, self.btn_benchmark_page):
            model_btn_row1.addWidget(button)
        model_btn_row1.addStretch()
        model_layout.addLayout(model_btn_row1)

        model_card.set_body_widget(model_body)
        model_card.setMinimumHeight(250)
        top_row.addWidget(model_card)

        config_card = DashboardSection("配置管理", "当前配置文件与配置维护")
        config_body = QWidget()
        config_layout = QVBoxLayout(config_body)
        config_layout.setContentsMargins(0, 0, 0, 0)
        self.management_config_path_label = QLabel("-")
        self.management_config_summary_label = QLabel("-")
        self.management_config_summary_label.setWordWrap(True)
        config_form = QFormLayout()
        config_form.addRow("当前配置文件路径", self.management_config_path_label)
        config_form.addRow("查看配置摘要", self.management_config_summary_label)
        config_layout.addLayout(config_form)

        config_btn_row1 = QHBoxLayout()
        self.btn_import_config_page = QPushButton("导入配置")
        self.btn_save_config_page = QPushButton("保存配置")
        self.btn_restore_default_config = QPushButton("恢复默认配置")
        self.btn_view_config_summary = QPushButton("查看配置摘要")
        self.btn_import_config_page.clicked.connect(self.load_yaml_config)
        self.btn_save_config_page.clicked.connect(self.save_yaml_config)
        self.btn_restore_default_config.clicked.connect(self.restore_default_config)
        self.btn_view_config_summary.clicked.connect(self.show_config_summary)
        for button in (self.btn_import_config_page, self.btn_save_config_page, self.btn_restore_default_config, self.btn_view_config_summary):
            config_btn_row1.addWidget(button)
        config_btn_row1.addStretch()
        config_layout.addLayout(config_btn_row1)

        config_card.set_body_widget(config_body)
        config_card.setMinimumHeight(250)
        top_row.addWidget(config_card)

        scroll_layout.addLayout(top_row)

        middle_row = QHBoxLayout()

        runtime_card = DashboardSection("运行环境", "Python、PyTorch、OpenCV 和 CUDA 状态")
        runtime_body = QWidget()
        runtime_layout = QFormLayout(runtime_body)
        self.management_python_label = QLabel("-")
        self.management_torch_label = QLabel("-")
        self.management_opencv_label = QLabel("-")
        self.management_cuda_label = QLabel("-")
        self.management_run_device_label = QLabel("-")
        self.management_platform_label = QLabel("-")
        runtime_layout.addRow("Python 版本", self.management_python_label)
        runtime_layout.addRow("PyTorch 状态", self.management_torch_label)
        runtime_layout.addRow("OpenCV 状态", self.management_opencv_label)
        runtime_layout.addRow("CUDA 是否可用", self.management_cuda_label)
        runtime_layout.addRow("当前运行设备", self.management_run_device_label)
        runtime_layout.addRow("系统平台信息", self.management_platform_label)
        runtime_card.set_body_widget(runtime_body)
        runtime_card.setMinimumHeight(240)
        middle_row.addWidget(runtime_card)

        maintenance_card = DashboardSection("系统维护", "输出目录、日志目录和临时文件维护")
        maintenance_body = QWidget()
        maintenance_layout = QVBoxLayout(maintenance_body)
        maintenance_layout.setContentsMargins(0, 0, 0, 0)
        maint_btn_row1 = QHBoxLayout()
        self.btn_open_output_dir = QPushButton("打开输出目录")
        self.btn_open_log_dir = QPushButton("打开日志目录")
        self.btn_clear_cache = QPushButton("清理缓存")
        self.btn_clear_temp_files = QPushButton("清空临时文件")
        self.btn_view_system_log = QPushButton("查看系统日志")
        self.btn_open_output_dir.clicked.connect(self.open_output_dir)
        self.btn_open_log_dir.clicked.connect(self.open_log_dir)
        self.btn_clear_cache.clicked.connect(self.clear_cache)
        self.btn_clear_temp_files.clicked.connect(self.clear_temp_files)
        self.btn_view_system_log.clicked.connect(self.show_system_log)
        for button in (self.btn_open_output_dir, self.btn_open_log_dir, self.btn_clear_cache, self.btn_clear_temp_files, self.btn_view_system_log):
            maint_btn_row1.addWidget(button)
        maint_btn_row1.addStretch()
        maintenance_layout.addLayout(maint_btn_row1)
        self.management_maintenance_hint = QLabel("用于维护运行目录、临时文件与系统日志。")
        self.management_maintenance_hint.setWordWrap(True)
        maintenance_layout.addWidget(self.management_maintenance_hint)
        maintenance_card.set_body_widget(maintenance_body)
        maintenance_card.setMinimumHeight(240)
        middle_row.addWidget(maintenance_card)

        scroll_layout.addLayout(middle_row)

        benchmark_card = DashboardSection("Benchmark 结果表", "展示 ONNX Benchmark 历史结果")
        benchmark_body = QWidget()
        benchmark_layout = QVBoxLayout(benchmark_body)
        benchmark_layout.setContentsMargins(0, 0, 0, 0)
        self.management_benchmark_table = QTableWidget(0, 7)
        self.management_benchmark_table.setHorizontalHeaderLabels(["测试时间", "模型名称", "输入尺寸", "平均 FPS", "推理耗时", "设备", "备注"])
        self.management_benchmark_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.management_benchmark_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.management_benchmark_table.setMinimumHeight(260)
        self.management_benchmark_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.management_benchmark_table.horizontalHeader().setStretchLastSection(False)
        self.management_benchmark_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        benchmark_layout.addWidget(self.management_benchmark_table)
        benchmark_card.set_body_widget(benchmark_body)
        benchmark_card.setMinimumHeight(300)
        scroll_layout.addWidget(benchmark_card)

        log_card = DashboardSection("操作日志", "最近系统管理操作")
        log_body = QWidget()
        log_layout = QVBoxLayout(log_body)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.management_log_table = QTableWidget(0, 5)
        self.management_log_table.setHorizontalHeaderLabels(["时间", "操作类型", "操作内容", "操作结果", "备注"])
        self.management_log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.management_log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.management_log_table.setMinimumHeight(220)
        self.management_log_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.management_log_table.horizontalHeader().setStretchLastSection(False)
        self.management_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        log_layout.addWidget(self.management_log_table)
        log_card.set_body_widget(log_body)
        log_card.setMinimumHeight(280)
        scroll_layout.addWidget(log_card)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(scroll_container)
        layout.addWidget(scroll, 1)

        self._load_sample_management_benchmark_records()
        self._load_sample_management_logs()
        self._refresh_management_page()
        self.refresh_management_benchmark_table()
        self.refresh_management_log_table()

        layout.addStretch()
        return page

    def _build_profile_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        # 个人资料卡片
        profile_card = DashboardSection("个人资料", "当前登录用户的详细信息")
        profile_body = QWidget()
        profile_form = QFormLayout(profile_body)
        self.profile_username_label = QLabel("-")
        self.profile_role_label = QLabel("-")
        self.profile_last_login_label = QLabel("-")
        self.profile_created_label = QLabel("-")
        self.profile_status_label = QLabel("-")
        self.profile_contact_label = QLabel("未设置")
        self.profile_notes_label = QLabel("未填写")
        for label in (
            self.profile_username_label,
            self.profile_role_label,
            self.profile_last_login_label,
            self.profile_created_label,
            self.profile_status_label,
            self.profile_contact_label,
            self.profile_notes_label,
        ):
            label.setWordWrap(True)
        profile_form.addRow("用户名", self.profile_username_label)
        profile_form.addRow("角色", self.profile_role_label)
        profile_form.addRow("上次登录时间", self.profile_last_login_label)
        profile_form.addRow("注册时间", self.profile_created_label)
        profile_form.addRow("账号状态", self.profile_status_label)
        profile_form.addRow("联系方式", self.profile_contact_label)
        profile_form.addRow("备注信息", self.profile_notes_label)
        profile_card.body_layout.addWidget(profile_body)
        layout.addWidget(profile_card)

        # 账号安全卡片
        security_card = DashboardSection("账号安全", "账号安全与登录信息")
        security_body = QWidget()
        security_layout = QVBoxLayout(security_body)
        security_layout.setContentsMargins(0, 0, 0, 0)
        security_layout.setSpacing(10)

        button_row = QHBoxLayout()
        self.btn_change_password_page = QPushButton("修改密码")
        self.btn_change_password_page.clicked.connect(self._open_change_password_dialog)
        self.btn_logout_page = QPushButton("退出登录")
        self.btn_logout_page.clicked.connect(self.close)
        button_row.addWidget(self.btn_change_password_page)
        button_row.addWidget(self.btn_logout_page)
        button_row.addStretch()

        security_form = QFormLayout()
        self.sec_last_login = QLabel("-")
        self.sec_login_count = QLabel("0")
        self.sec_enabled_label = QLabel("-")
        security_form.addRow("最近登录时间", self.sec_last_login)
        security_form.addRow("登录次数", self.sec_login_count)
        security_form.addRow("是否启用账号", self.sec_enabled_label)

        security_layout.addLayout(button_row)
        security_layout.addLayout(security_form)
        security_card.body_layout.addWidget(security_body)
        layout.addWidget(security_card)

        layout.addStretch()
        return page

    def _build_password_page(self):
        page = EmployeeManagementPage(self.current_user)
        self.employee_page_widget = page
        return page

    def _switch_page(self, key: str):
        page_keys = list(self.page_widgets.keys())
        if key not in self.page_widgets:
            return
        if key == "password" and not self._is_admin():
            QMessageBox.warning(self, "权限不足", "员工管理页面仅管理员可访问。")
            return
        index = page_keys.index(key)
        self.pages.setCurrentIndex(index)
        for name, button in self.sidebar_buttons.items():
            button.setChecked(name == key)
        title, subtitle = self.page_titles.get(key, ("系统首页", ""))
        self.page_title_label.setText(title)
        self.page_desc_label.setText(subtitle)
        self._refresh_header_user_info()
        if key == "statistics":
            self._refresh_statistics_sources()
            self.apply_statistics_filters()
        if key == "password" and self._is_admin():
            if hasattr(self, "employee_page_widget"):
                self.employee_page_widget.current_user = self.current_user
                self.employee_page_widget.refresh_list()
        self.statusBar().showMessage(f"当前页面：{title}")

    def _refresh_header_user_info(self):
        username = (self.current_user or {}).get("username", "未登录")
        role = (self.current_user or {}).get("role", "管理员")
        self.user_badge.set_value(username)
        self.role_badge.set_value(role)

    def _apply_dashboard_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget#MainWindow {
                background: #eef4fb;
            }
            QWidget#Sidebar {
                background: linear-gradient(180deg, #173a63 0%, #1f4c82 100%);
            }
            QLabel#SidebarBrandTitle {
                color: #ffffff;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#SidebarBrandSubtitle {
                color: rgba(255, 255, 255, 0.75);
            }
            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                background: #ffffff;
                color: #22436a;
            }
            QPushButton:hover {
                background: #e8f1ff;
            }
            QPushButton:pressed {
                background: #d9e7fb;
            }
            QPushButton:checked {
                background: #2f80ed;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#SidebarLogoutButton {
                background: rgba(255,255,255,0.15);
                color: #ffffff;
                border: 1px solid rgba(255,255,255,0.22);
            }
            QWidget#PageHeader {
                background: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.85);
                border-radius: 16px;
            }
            QLabel#PageTitle {
                font-size: 22px;
                font-weight: 700;
                color: #16324f;
            }
            QLabel#PageSubtitle {
                color: #6a7a90;
            }
            QFrame[card="true"] {
                background: #ffffff;
                border: 1px solid #dbe6f2;
                border-radius: 16px;
            }
            QLabel#DashboardSectionTitle {
                font-size: 16px;
                font-weight: 700;
                color: #16324f;
            }
            QLabel#DashboardSectionSubtitle {
                color: #70839b;
            }
            QLabel#DashboardMetricTitle {
                color: #6d7f95;
                font-size: 12px;
            }
            QLabel#DashboardMetricValue {
                color: #16324f;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#DashboardMetricHint {
                color: #8a9ab0;
            }
            QLabel#UserBadgeTitle {
                color: #6d7f95;
                font-size: 12px;
            }
            QLabel#UserBadgeValue {
                color: #16324f;
                font-size: 14px;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #dbe6f2;
                border-radius: 14px;
                background: white;
            }
            QTabBar::tab {
                background: #edf3fb;
                color: #53667d;
                padding: 10px 16px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #2f80ed;
                font-weight: 600;
            }
            QTextEdit, QPlainTextEdit, QLineEdit, QTableWidget {
                border: 1px solid #dbe6f2;
                border-radius: 12px;
                background: #ffffff;
            }
            """
        )

    def _init_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        run_menu = menubar.addMenu("运行")
        config_menu = menubar.addMenu("配置")
        help_menu = menubar.addMenu("帮助")

        self.action_open_video = QAction("打开视频", self)
        self.action_load_config = QAction("加载配置", self)
        self.action_exit = QAction("退出", self)
        file_menu.addAction(self.action_open_video)
        file_menu.addAction(self.action_load_config)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        self.action_start = QAction("开始", self)
        self.action_pause = QAction("暂停/继续", self)
        self.action_stop = QAction("停止", self)
        run_menu.addAction(self.action_start)
        run_menu.addAction(self.action_pause)
        run_menu.addAction(self.action_stop)

        self.action_save_yaml = QAction("保存YAML", self)
        self.action_load_yaml = QAction("加载YAML", self)
        config_menu.addAction(self.action_save_yaml)
        config_menu.addAction(self.action_load_yaml)

        self.action_about = QAction("关于", self)
        help_menu.addAction(self.action_about)

        self.action_open_video.triggered.connect(self.control_panel.select_video)
        self.action_load_config.triggered.connect(self.load_yaml_config)
        self.action_exit.triggered.connect(self.close)
        self.action_start.triggered.connect(self.start_processing)
        self.action_pause.triggered.connect(self.toggle_pause)
        self.action_stop.triggered.connect(self.stop_processing)
        self.action_save_yaml.triggered.connect(self.save_yaml_config)
        self.action_load_yaml.triggered.connect(self.load_yaml_config)
        self.action_about.triggered.connect(self.show_about)

    def connect_signals(self):
        self.control_panel.btn_start.clicked.connect(self.start_processing)
        self.control_panel.btn_pause.clicked.connect(self.toggle_pause)
        self.control_panel.btn_stop.clicked.connect(self.stop_processing)
        self.control_panel.btn_select_camera.clicked.connect(self.use_camera_source)
        self.control_panel.btn_export.clicked.connect(self.on_export_results_requested)

        self.worker.new_frame.connect(self.video_panel.update_frame)
        self.worker.stats_updated.connect(self.on_stats_updated)
        self.worker.trend_updated.connect(self.on_trend_updated)
        self.worker.event_emitted.connect(self.on_event_emitted)
        self.worker.log_message.connect(self.log_panel.append_log)
        self.worker.finished.connect(self.on_process_finished)

        self.annotation_panel.params_changed.connect(self.apply_runtime_params)
        self.annotation_panel.draw_mode_changed.connect(self.video_panel.set_draw_mode)
        self.annotation_panel.clear_requested.connect(self.video_panel.clear_annotations)
        self.annotation_panel.undo_requested.connect(self.video_panel.undo_last_annotation_point)
        self.annotation_panel.load_config_requested.connect(self.load_yaml_config)
        self.annotation_panel.save_config_requested.connect(self.save_yaml_config)

        self.video_panel.roi_changed.connect(self.on_annotations_changed)
        self.video_panel.line_changed.connect(self.on_annotations_changed)

        self.control_panel.video_selected.connect(self.on_video_selected)
        self.control_panel.save_frame_requested.connect(self.on_save_frame_requested)
        # 模型导出/量化/benchmark 信号
        self.control_panel.export_onnx_requested.connect(self.on_export_onnx_requested)
        self.control_panel.quantize_onnx_requested.connect(self.on_quantize_onnx_requested)
        self.control_panel.benchmark_onnx_requested.connect(self.on_benchmark_onnx_requested)

    def _is_admin(self):
        return (self.current_user or {}).get("role") == "管理员"

    def _require_admin(self, action_name: str) -> bool:
        if self._is_admin():
            return True
        QMessageBox.warning(self, "权限不足", f"员工身份无法执行{action_name}。")
        return False

    def apply_user_permissions(self):
        is_admin = self._is_admin()
        for widget in (
            self.action_load_config,
            self.action_load_yaml,
            self.action_save_yaml,
            self.control_panel.btn_export_onnx,
            self.control_panel.btn_quantize,
            self.control_panel.btn_benchmark,
            self.btn_export_onnx_page,
            self.btn_quantize_page,
            self.btn_benchmark_page,
            self.btn_import_config_page,
            self.btn_save_config_page,
            self.btn_restore_default_config,
            self.btn_view_config_summary,
            self.btn_open_output_dir,
            self.btn_open_log_dir,
            self.btn_clear_cache,
            self.btn_clear_temp_files,
            self.btn_view_system_log,
        ):
            widget.setEnabled(is_admin)

        if hasattr(self, "sidebar_buttons") and self.sidebar_buttons.get("password") is not None:
            self.sidebar_buttons["password"].setEnabled(is_admin)

        role = (self.current_user or {}).get("role", "管理员")
        username = (self.current_user or {}).get("username", "未知用户")
        self.setWindowTitle(f"视频行人客流量统计与分析系统 - {username}（{role}）")
        self.statusBar().showMessage(f"当前用户：{username} | 身份：{role}")
        self._refresh_profile_page()
        self._refresh_management_page()

    def _refresh_management_page(self):
        config_path = self.current_config_path or "未加载配置"
        detector_cfg = self.current_config.get("detector", {}) if isinstance(self.current_config, dict) else {}
        model_path = str(detector_cfg.get("model_path", "-"))
        model_name = Path(model_path).stem if model_path and model_path != "-" else "-"
        model_type = self._detector_type_label()
        input_size = str(detector_cfg.get("imgsz", detector_cfg.get("input_size", 800)))
        current_device = self.current_device or self._device_label()

        if hasattr(self, "management_model_path_label"):
            self.management_model_path_label.setText(model_path)
        if hasattr(self, "management_model_name_label"):
            self.management_model_name_label.setText(model_name)
        if hasattr(self, "management_model_type_label"):
            self.management_model_type_label.setText(model_type)
        if hasattr(self, "management_imgsz_label"):
            self.management_imgsz_label.setText(input_size)
        if hasattr(self, "management_device_label"):
            self.management_device_label.setText(current_device)

        if hasattr(self, "management_config_path_label"):
            self.management_config_path_label.setText(config_path)
        if hasattr(self, "management_config_summary_label"):
            self.management_config_summary_label.setText(self._build_config_summary_text())

        if hasattr(self, "management_python_label"):
            self.management_python_label.setText(platform.python_version())
        if hasattr(self, "management_torch_label"):
            self.management_torch_label.setText(self._detect_torch_status())
        if hasattr(self, "management_opencv_label"):
            self.management_opencv_label.setText(self._detect_opencv_status())
        if hasattr(self, "management_cuda_label"):
            self.management_cuda_label.setText("可用" if self._detect_cuda_available() else "不可用")
        if hasattr(self, "management_run_device_label"):
            self.management_run_device_label.setText(current_device)
        if hasattr(self, "management_platform_label"):
            self.management_platform_label.setText(platform.platform())

        self._refresh_home_status_card()

    def _build_config_summary_text(self):
        cfg = self.current_config if isinstance(self.current_config, dict) else self._default_config()
        detector = cfg.get("detector", {})
        source = cfg.get("source", "") or "未设置"
        return (
            f"源: {source}\n"
            f"模型: {detector.get('model_path', '-')}\n"
            f"conf: {detector.get('conf', '-')}, iou: {detector.get('iou', '-')}\n"
            f"ROI: {len(cfg.get('counting', {}).get('roi', {}).get('polygon', []))} 点, "
            f"Line: {len(cfg.get('counting', {}).get('line', {}).get('points', []))} 点"
        )

    def _detect_torch_status(self):
        try:
            import torch
            return f"已安装，版本 {torch.__version__}"
        except Exception:
            return "未安装"

    def _detect_opencv_status(self):
        try:
            import cv2
            return f"已安装，版本 {cv2.__version__}"
        except Exception:
            return "未安装"

    def _detect_cuda_available(self):
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _load_sample_management_benchmark_records(self):
        self._management_benchmark_records = [
            {"time": "2026-05-06 09:30:00", "model": "best.pt", "imgsz": "800", "fps": "28.4", "latency": "35.2 ms", "device": "CPU", "note": "示例记录"},
            {"time": "2026-05-06 09:42:00", "model": "best.pt", "imgsz": "960", "fps": "24.6", "latency": "40.7 ms", "device": "GPU", "note": "示例记录"},
        ]

    def _load_sample_management_logs(self):
        self._management_logs = [
            {"time": "2026-05-06 09:45:00", "type": "配置", "content": "加载默认配置", "result": "成功", "note": ""},
            {"time": "2026-05-06 09:46:30", "type": "模型", "content": "刷新模型信息", "result": "成功", "note": ""},
        ]

    def refresh_management_benchmark_table(self):
        if not hasattr(self, "management_benchmark_table"):
            return
        rows = getattr(self, "_management_benchmark_records", [])
        self.management_benchmark_table.setRowCount(0)
        for row in rows:
            row_index = self.management_benchmark_table.rowCount()
            self.management_benchmark_table.insertRow(row_index)
            values = [row.get("time", ""), row.get("model", ""), row.get("imgsz", ""), row.get("fps", ""), row.get("latency", ""), row.get("device", ""), row.get("note", "")]
            for col, value in enumerate(values):
                self.management_benchmark_table.setItem(row_index, col, QTableWidgetItem(str(value)))

    def refresh_management_log_table(self):
        if not hasattr(self, "management_log_table"):
            return
        rows = getattr(self, "_management_logs", [])
        self.management_log_table.setRowCount(0)
        for row in rows:
            row_index = self.management_log_table.rowCount()
            self.management_log_table.insertRow(row_index)
            values = [row.get("time", ""), row.get("type", ""), row.get("content", ""), row.get("result", ""), row.get("note", "")]
            for col, value in enumerate(values):
                self.management_log_table.setItem(row_index, col, QTableWidgetItem(str(value)))

    def _append_management_log(self, log_type: str, content: str, result: str = "成功", note: str = ""):
        log_item = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": log_type,
            "content": content,
            "result": result,
            "note": note,
        }
        self._management_logs = [log_item] + getattr(self, "_management_logs", [])[:29]
        self.refresh_management_log_table()

    def _append_management_benchmark_record(self, model_name: str, input_size: str, device: str, note: str = ""):
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "imgsz": input_size,
            "fps": "-",
            "latency": "-",
            "device": device,
            "note": note,
        }
        self._management_benchmark_records = [record] + getattr(self, "_management_benchmark_records", [])[:19]
        self.refresh_management_benchmark_table()

    def restore_default_config(self):
        if not self._require_admin("恢复默认配置"):
            return
        self.current_config = self._default_config()
        self.current_config_path = None
        self.worker.set_config(self.current_config)
        self.worker.warmup_detector_async()
        self.annotation_panel.set_params({
            "conf": float(self.current_config.get("detector", {}).get("conf", 0.5)),
            "iou": float(self.current_config.get("detector", {}).get("iou", 0.45)),
            "show_trail": True,
            "show_roi": True,
            "show_line": True,
            "show_heatmap": True,
            "face_blur_enabled": False,
            "draw_count_points": True,
        })
        self._refresh_system_info()
        self._refresh_management_page()
        self._append_management_log("配置", "恢复默认配置", "成功")

    def show_config_summary(self):
        if not self._require_admin("查看配置摘要"):
            return
        QMessageBox.information(self, "配置摘要", self._build_config_summary_text())

    def open_output_dir(self):
        if not self._require_admin("打开输出目录"):
            return
        output_dir = Path(__file__).resolve().parents[1] / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(output_dir))
        except Exception:
            QMessageBox.information(self, "打开输出目录", f"输出目录: {output_dir}")
        self._append_management_log("维护", "打开输出目录", "成功")

    def open_log_dir(self):
        if not self._require_admin("打开日志目录"):
            return
        log_dir = Path(__file__).resolve().parents[1] / "outputs" / "gui_run"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(log_dir))
        except Exception:
            QMessageBox.information(self, "打开日志目录", f"日志目录: {log_dir}")
        self._append_management_log("维护", "打开日志目录", "成功")

    def clear_cache(self):
        if not self._require_admin("清理缓存"):
            return
        cleared = 0
        for candidate in [Path(__file__).resolve().parents[1] / "__pycache__", Path(__file__).resolve().parents[1] / "gui" / "__pycache__"]:
            if candidate.exists():
                try:
                    shutil.rmtree(candidate)
                    cleared += 1
                except Exception:
                    pass
        QMessageBox.information(self, "清理缓存", f"已清理缓存目录: {cleared} 个")
        self._append_management_log("维护", "清理缓存", "成功", f"清理 {cleared} 个目录")

    def clear_temp_files(self):
        if not self._require_admin("清空临时文件"):
            return
        temp_dir = Path(__file__).resolve().parents[1] / "outputs" / "gui_run"
        removed = 0
        if temp_dir.exists():
            for path in temp_dir.glob("*.tmp"):
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
        QMessageBox.information(self, "清空临时文件", f"已清理临时文件: {removed} 个")
        self._append_management_log("维护", "清空临时文件", "成功", f"清理 {removed} 个文件")

    def show_system_log(self):
        if not self._require_admin("查看系统日志"):
            return
        lines = [f"{row.get('time')} | {row.get('type')} | {row.get('content')} | {row.get('result')}" for row in getattr(self, "_management_logs", [])[:10]]
        QMessageBox.information(self, "系统日志", "\n".join(lines) if lines else "暂无系统日志")

    def _refresh_home_status_card(self):
        self.home_status_detect.setText("检测中" if self.current_detecting else "未运行")
        self.home_status_source.setText(self.current_source_name)
        self.home_status_model.setText(self.current_detector_type)
        self.home_status_config.setText(self.current_config_path or "默认配置")

    def _refresh_statistics_sources(self):
        current_text = self.query_video_name.currentText() if hasattr(self, "query_video_name") else "全部视频"
        self.query_video_name.blockSignals(True)
        self.query_video_name.clear()
        self.query_video_name.addItem("全部视频")
        for source_name in self.history_db.query_sources():
            self.query_video_name.addItem(source_name)
        if current_text and self.query_video_name.findText(current_text) >= 0:
            self.query_video_name.setCurrentText(current_text)
        self.query_video_name.blockSignals(False)

    def _parse_time_filter(self, text: str):
        raw = (text or "").strip()
        if not raw:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        return None

    def _get_statistics_query_context(self):
        start_time = self._parse_time_filter(self.query_time_start.text())
        end_time = self._parse_time_filter(self.query_time_end.text())
        if start_time and end_time and end_time < start_time:
            start_time, end_time = end_time, start_time

        selected_video = self.query_video_name.currentText()
        selected_dir = self.query_direction.currentText()
        bucket_minutes = 1 if self.query_bucket.currentText() == "按分钟" else 5

        events = self.history_db.query_events(
            start_time=start_time,
            end_time=end_time,
            source_name=selected_video,
            direction=selected_dir,
        )
        sessions = self.history_db.query_sessions(start_time=start_time, end_time=end_time, source_name=selected_video)
        history_rows = self.history_db.query_traffic_history(
            start_time=start_time,
            end_time=end_time,
            source_name=selected_video,
            bucket_minutes=bucket_minutes,
        )
        return {
            "start_time": start_time,
            "end_time": end_time,
            "selected_video": selected_video,
            "selected_dir": selected_dir,
            "bucket_minutes": bucket_minutes,
            "events": events,
            "sessions": sessions,
            "history_rows": history_rows,
        }

    def _parse_mmss_to_seconds(self, text: str) -> int:
        raw = (text or "").strip()
        if not raw:
            return 0
        parts = raw.split(":")
        if len(parts) != 2:
            return 0
        try:
            mm = max(0, int(parts[0]))
            ss = max(0, int(parts[1]))
            return mm * 60 + ss
        except ValueError:
            return 0

    def _event_direction(self, event_name: str) -> str:
        low = event_name.lower()
        if "up" in low or "上" in event_name:
            return "Up"
        if "down" in low or "下" in event_name:
            return "Down"
        return "All"

    def apply_statistics_filters(self):
        context = self._get_statistics_query_context()
        events = context["events"]
        self.filtered_event_table_panel.set_records([
            {
                "timestamp": item.get("timestamp", ""),
                "event": item.get("event_type", ""),
                "track_id": item.get("track_id", ""),
                "target": item.get("target", ""),
            }
            for item in events
        ])

        up_count = sum(1 for item in events if str(item.get("direction", "")).lower() == "up")
        down_count = sum(1 for item in events if str(item.get("direction", "")).lower() == "down")
        session_rows = context["sessions"]
        total_count = sum(int(row.get("total_count") or 0) for row in session_rows)
        avg_fps = 0.0
        fps_values = [float(row.get("avg_fps") or 0.0) for row in session_rows if row.get("avg_fps") is not None]
        if fps_values:
            avg_fps = sum(fps_values) / len(fps_values)

        self.stats_query_total.set_value(str(len(events)), "过滤后事件数")
        self.stats_query_up.set_value(str(up_count), "过滤后")
        self.stats_query_down.set_value(str(down_count), "过滤后")
        self.stats_query_fps.set_value(f"{avg_fps:.2f}", "历史会话均值")

        history_points = context["history_rows"]
        self.stats_history_panel.load_history([
            {
                "minute": idx,
                "up": item.get("up", 0),
                "down": item.get("down", 0),
                "total": item.get("total", 0),
            }
            for idx, item in enumerate(history_points)
        ])
        self.stats_fps_panel.load_history([float(row.get("avg_fps") or 0.0) for row in session_rows])

        self.experiment_table_panel.reset()
        for row in session_rows:
            self.experiment_table_panel.add_record(
                timestamp=str(row.get("started_at", "")),
                conf=float(row.get("conf") or 0.0),
                iou=float(row.get("iou") or 0.0),
                avg_fps=float(row.get("avg_fps") or 0.0),
                up=int(row.get("up_count") or 0),
                down=int(row.get("down_count") or 0),
            )

    def reset_statistics_filters(self):
        self.query_time_start.clear()
        self.query_time_end.clear()
        self.query_direction.setCurrentText("All")
        self.query_bucket.setCurrentText("按分钟")
        self._refresh_statistics_sources()
        self.query_video_name.setCurrentIndex(0)
        self.apply_statistics_filters()

    def _refresh_profile_page(self):
        username = (self.current_user or {}).get("username", "-")
        profile = self.auth_manager.get_user_profile(username) or {}
        def _safe_set(name, value):
            if hasattr(self, name):
                try:
                    getattr(self, name).setText(str(value))
                except Exception:
                    pass

        _safe_set("profile_username_label", profile.get("username", username))
        _safe_set("profile_role_label", profile.get("role", (self.current_user or {}).get("role", "-")))
        _safe_set("profile_last_login_label", profile.get("last_login", (self.current_user or {}).get("last_login", "-")))
        _safe_set("profile_created_label", profile.get("created_at", "-"))

        # 更新安全卡片信息（若存在）
        _safe_set("sec_last_login", profile.get("last_login", (self.current_user or {}).get("last_login", "-")))
        _safe_set("sec_login_count", profile.get("login_count", profile.get("logins", 0)))
        _safe_set("sec_enabled_label", "已启用" if profile.get("enabled", True) else "已禁用")

        # 个人资料补充字段（若存在）
        _safe_set("profile_status_label", "已启用" if profile.get("enabled", True) else "已禁用")
        _safe_set("profile_contact_label", profile.get("contact", "未设置"))
        _safe_set("profile_notes_label", profile.get("notes", ""))

    def _open_change_password_dialog(self):
        username = (self.current_user or {}).get("username", "").strip()
        if not username:
            QMessageBox.warning(self, "修改密码", "当前没有可用账号信息。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("修改密码")
        dialog_layout = QVBoxLayout(dialog)
        form = QFormLayout()
        old_edit = QLineEdit()
        old_edit.setEchoMode(QLineEdit.Password)
        new_edit = QLineEdit()
        new_edit.setEchoMode(QLineEdit.Password)
        confirm_edit = QLineEdit()
        confirm_edit.setEchoMode(QLineEdit.Password)
        form.addRow("旧密码", old_edit)
        form.addRow("新密码", new_edit)
        form.addRow("确认新密码", confirm_edit)
        dialog_layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("提交修改")
        button_box.button(QDialogButtonBox.Cancel).setText("取消")
        dialog_layout.addWidget(button_box)

        def submit_password_change():
            old_password = old_edit.text().strip()
            new_password = new_edit.text().strip()
            confirm_password = confirm_edit.text().strip()
            if not old_password or not new_password:
                QMessageBox.warning(dialog, "修改密码", "旧密码和新密码不能为空。")
                return
            if new_password != confirm_password:
                QMessageBox.warning(dialog, "修改密码", "两次输入的新密码不一致。")
                return
            success, message = self.auth_manager.change_password(username, old_password, new_password)
            if success:
                QMessageBox.information(dialog, "修改密码", message)
                dialog.accept()
                self._refresh_profile_page()
                return
            QMessageBox.warning(dialog, "修改密码", message)

        button_box.accepted.connect(submit_password_change)
        button_box.rejected.connect(dialog.reject)
        dialog.exec_()

    def _load_sample_employees(self):
        self._employee_rows = [
            {
                "id": 1001,
                "username": "admin",
                "fullname": "系统管理员",
                "role": "管理员",
                "status": "已启用",
                "created_at": "2024-01-10 09:00:00",
                "last_login": "2026-05-06 08:30:00",
            },
            {
                "id": 1002,
                "username": "zhangsan",
                "fullname": "张三",
                "role": "员工",
                "status": "已启用",
                "created_at": "2025-02-18 14:20:00",
                "last_login": "2026-05-05 17:05:00",
            },
            {
                "id": 1003,
                "username": "lisi",
                "fullname": "李四",
                "role": "员工",
                "status": "已禁用",
                "created_at": "2025-06-01 10:00:00",
                "last_login": "2026-04-28 12:10:00",
            },
        ]

    def _load_sample_employee_logs(self):
        self._employee_logs = [
            {"time": "2026-05-06 09:12:00", "operator": "admin", "action": "启用账号", "target": "zhangsan", "result": "成功"},
            {"time": "2026-05-06 09:08:00", "operator": "admin", "action": "重置密码", "target": "lisi", "result": "成功"},
            {"time": "2026-05-06 09:02:00", "operator": "admin", "action": "新增员工", "target": "wangwu", "result": "成功"},
        ]

    def refresh_employee_list(self):
        if not hasattr(self, "emp_table"):
            return
        rows = getattr(self, "_employee_rows", [])
        keyword = self.emp_search_input.text().strip().lower() if hasattr(self, "emp_search_input") else ""
        role_filter = self.emp_role_filter.currentText() if hasattr(self, "emp_role_filter") else "全部"
        status_filter = self.emp_status_filter.currentText() if hasattr(self, "emp_status_filter") else "全部"

        def match(row):
            if keyword and keyword not in str(row.get("username", "")).lower() and keyword not in str(row.get("fullname", "")).lower():
                return False
            if role_filter != "全部" and row.get("role") != role_filter:
                return False
            if status_filter != "全部" and row.get("status") != status_filter:
                return False
            return True

        filtered_rows = [row for row in rows if match(row)]
        self.emp_table.setRowCount(0)
        for row in filtered_rows:
            row_index = self.emp_table.rowCount()
            self.emp_table.insertRow(row_index)
            values = [
                row.get("id", ""),
                row.get("username", ""),
                row.get("fullname", ""),
                row.get("role", ""),
                row.get("status", ""),
                row.get("created_at", ""),
                row.get("last_login", ""),
            ]
            for col, value in enumerate(values):
                self.emp_table.setItem(row_index, col, QTableWidgetItem(str(value)))

            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(6)
            inspect_btn = QPushButton("查看")
            inspect_btn.clicked.connect(lambda _=False, rid=row.get("id"): QMessageBox.information(self, "员工信息", f"员工编号：{rid}"))
            action_layout.addWidget(inspect_btn)
            action_layout.addStretch()
            self.emp_table.setCellWidget(row_index, 7, action_widget)

    def refresh_employee_logs(self):
        if not hasattr(self, "employee_log_table"):
            return
        self.employee_log_table.setRowCount(0)
        for log in getattr(self, "_employee_logs", []):
            row_index = self.employee_log_table.rowCount()
            self.employee_log_table.insertRow(row_index)
            self.employee_log_table.setItem(row_index, 0, QTableWidgetItem(str(log.get("time", ""))))
            self.employee_log_table.setItem(row_index, 1, QTableWidgetItem(str(log.get("operator", ""))))
            self.employee_log_table.setItem(row_index, 2, QTableWidgetItem(str(log.get("action", ""))))
            self.employee_log_table.setItem(row_index, 3, QTableWidgetItem(str(log.get("target", ""))))
            self.employee_log_table.setItem(row_index, 4, QTableWidgetItem(str(log.get("result", ""))))

    def _append_employee_log(self, action_type: str, target_employee: str, result: str):
        log_item = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": (self.current_user or {}).get("username", "未知"),
            "action": action_type,
            "target": target_employee,
            "result": result,
        }
        self._employee_logs = [log_item] + getattr(self, "_employee_logs", [])[:19]
        self.refresh_employee_logs()

    def apply_employee_filters(self):
        self.refresh_employee_list()

    def reset_employee_filters(self):
        if hasattr(self, "emp_search_input"):
            self.emp_search_input.clear()
        if hasattr(self, "emp_role_filter"):
            self.emp_role_filter.setCurrentText("全部")
        if hasattr(self, "emp_status_filter"):
            self.emp_status_filter.setCurrentText("全部")
        self.refresh_employee_list()

    def _selected_employee_row(self):
        if not hasattr(self, "emp_table"):
            return None
        row_index = self.emp_table.currentRow()
        if row_index < 0:
            return None
        emp_id_item = self.emp_table.item(row_index, 0)
        if emp_id_item is None:
            return None
        emp_id = emp_id_item.text().strip()
        for row in getattr(self, "_employee_rows", []):
            if str(row.get("id")) == emp_id:
                return row
        return None

    def handle_add_employee(self):
        if not self._require_admin("新增员工"):
            return
        QMessageBox.information(self, "新增员工", "这里预留新增员工对话框接口，后续接数据库。")
        self._append_employee_log("新增员工", "新员工", "成功")

    def handle_edit_employee(self):
        if not self._require_admin("编辑员工"):
            return
        row = self._selected_employee_row()
        if row is None:
            QMessageBox.warning(self, "编辑员工", "请先选择一名员工。")
            return
        QMessageBox.information(self, "编辑员工", f"这里预留编辑接口：{row.get('username')}")
        self._append_employee_log("编辑员工", str(row.get("username", "")), "成功")

    def handle_delete_employee(self):
        if not self._require_admin("删除员工"):
            return
        row = self._selected_employee_row()
        if row is None:
            QMessageBox.warning(self, "删除员工", "请先选择一名员工。")
            return
        current_username = (self.current_user or {}).get("username", "")
        if row.get("username") == current_username and self._is_admin():
            QMessageBox.warning(self, "删除受限", "不能删除当前正在登录的管理员账号。")
            return
        confirm = QMessageBox.question(self, "确认删除", f"确认删除员工 {row.get('username')} 吗？", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        self._employee_rows = [item for item in getattr(self, "_employee_rows", []) if item.get("id") != row.get("id")]
        self.refresh_employee_list()
        self._append_employee_log("删除员工", str(row.get("username", "")), "成功")

    def handle_reset_employee_password(self):
        if not self._require_admin("重置密码"):
            return
        row = self._selected_employee_row()
        if row is None:
            QMessageBox.warning(self, "重置密码", "请先选择一名员工。")
            return
        confirm = QMessageBox.question(self, "确认重置密码", f"确认重置员工 {row.get('username')} 的密码吗？", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        QMessageBox.information(self, "重置密码", "已预留重置密码接口，后续接数据库与重置逻辑。")
        self._append_employee_log("重置密码", str(row.get("username", "")), "成功")

    def handle_enable_employee(self):
        if not self._require_admin("启用账号"):
            return
        row = self._selected_employee_row()
        if row is None:
            QMessageBox.warning(self, "启用账号", "请先选择一名员工。")
            return
        row["status"] = "已启用"
        self.refresh_employee_list()
        self._append_employee_log("启用账号", str(row.get("username", "")), "成功")

    def handle_disable_employee(self):
        if not self._require_admin("禁用账号"):
            return
        row = self._selected_employee_row()
        if row is None:
            QMessageBox.warning(self, "禁用账号", "请先选择一名员工。")
            return
        row["status"] = "已禁用"
        self.refresh_employee_list()
        self._append_employee_log("禁用账号", str(row.get("username", "")), "成功")

    def on_export_results_requested(self):
        default_dir = Path(__file__).resolve().parents[1] / "outputs" / "gui_run"
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出结果摘要",
            str(default_dir / "summary.txt"),
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return

        summary_lines = [
            f"用户: {(self.current_user or {}).get('username', '-')}",
            f"角色: {(self.current_user or {}).get('role', '-')}",
            f"总通行人数: {self.current_total_people}",
            f"Up 总数: {self.current_up_count}",
            f"Down 总数: {self.current_down_count}",
            f"过滤后事件数: {self.stats_query_total.value_label.text()}",
            f"过滤后 Up: {self.stats_query_up.value_label.text()}",
            f"过滤后 Down: {self.stats_query_down.value_label.text()}",
            f"当前设备: {self.current_device}",
            f"平均 FPS: {self.current_avg_fps:.2f}",
            f"视频分辨率: {self.current_video_resolution}",
        ]
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"结果写入失败: {exc}")
            return

        self.log_panel.append_log(f"结果已导出: {file_path}")

    def export_filtered_events(self):
        default_dir = Path(__file__).resolve().parents[1] / "outputs" / "gui_run"
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出事件表",
            str(default_dir / "events_filtered.csv"),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return
        try:
            self.filtered_event_table_panel.export_csv(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"事件表导出失败: {exc}")
            return
        self.log_panel.append_log(f"事件表已导出: {file_path}")

    def export_statistics_charts(self):
        default_dir = Path(__file__).resolve().parents[1] / "outputs" / "gui_run"
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出图表截图",
            str(default_dir / "statistics_charts.png"),
            "PNG Image (*.png)",
        )
        if not file_path:
            return
        try:
            base = Path(file_path)
            self.stats_history_panel.figure.savefig(str(base), dpi=180)
            fps_path = base.with_name(base.stem + "_fps" + base.suffix)
            self.stats_fps_panel.figure.savefig(str(fps_path), dpi=180)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"图表导出失败: {exc}")
            return
        self.log_panel.append_log(f"图表截图已导出: {file_path}")

    def export_database_history_results(self):
        default_dir = Path(__file__).resolve().parents[1] / "outputs" / "gui_run"
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出数据库历史结果",
            str(default_dir / "database_history.xlsx"),
            "Excel Workbook (*.xlsx);;CSV Files (*.csv)",
        )
        if not file_path:
            return

        context = self._get_statistics_query_context()
        history_rows = context["history_rows"]
        events = context["events"]
        sessions = context["sessions"]

        try:
            if file_path.lower().endswith(".csv") or "CSV Files" in selected_filter:
                self._export_history_csv(file_path, history_rows)
            else:
                self._export_history_xlsx(file_path, sessions, history_rows, events)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"历史结果导出失败: {exc}")
            return

        self.log_panel.append_log(f"数据库历史结果已导出: {file_path}")

    def _export_history_csv(self, file_path: str, history_rows: list[dict]):
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bucket_start", "bucket_end", "source_name", "up", "down", "total", "records"])
            for row in history_rows:
                writer.writerow([
                    row.get("bucket_start", ""),
                    row.get("bucket_end", ""),
                    row.get("source_name", ""),
                    row.get("up", 0),
                    row.get("down", 0),
                    row.get("total", 0),
                    row.get("records", 0),
                ])

    def _export_history_xlsx(self, file_path: str, sessions: list[dict], history_rows: list[dict], events: list[dict]):
        workbook = Workbook()
        summary_sheet = workbook.active
        summary_sheet.title = "Summary"
        summary_sheet.append(["item", "value"])
        summary_sheet.append(["start_time", self.query_time_start.text().strip() or "全部"])
        summary_sheet.append(["end_time", self.query_time_end.text().strip() or "全部"])
        summary_sheet.append(["video", self.query_video_name.currentText()])
        summary_sheet.append(["direction", self.query_direction.currentText()])
        summary_sheet.append(["bucket", self.query_bucket.currentText()])
        summary_sheet.append(["history_rows", len(history_rows)])
        summary_sheet.append(["events", len(events)])
        summary_sheet.append(["sessions", len(sessions)])
        summary_sheet.append([])
        summary_sheet.append(["bucket_start", "bucket_end", "source_name", "up", "down", "total", "records"])
        for row in history_rows:
            summary_sheet.append([
                row.get("bucket_start", ""),
                row.get("bucket_end", ""),
                row.get("source_name", ""),
                row.get("up", 0),
                row.get("down", 0),
                row.get("total", 0),
                row.get("records", 0),
            ])

        sessions_sheet = workbook.create_sheet("Sessions")
        sessions_sheet.append(["id", "source_name", "started_at", "ended_at", "conf", "iou", "avg_fps", "up_count", "down_count", "total_count", "detector_type", "config_path"])
        for row in sessions:
            sessions_sheet.append([
                row.get("id", ""),
                row.get("source_name", ""),
                row.get("started_at", ""),
                row.get("ended_at", ""),
                row.get("conf", ""),
                row.get("iou", ""),
                row.get("avg_fps", ""),
                row.get("up_count", ""),
                row.get("down_count", ""),
                row.get("total_count", ""),
                row.get("detector_type", ""),
                row.get("config_path", ""),
            ])

        events_sheet = workbook.create_sheet("Events")
        events_sheet.append(["timestamp", "event_type", "direction", "target", "track_id", "value", "source_name", "frame_idx"])
        for row in events:
            events_sheet.append([
                row.get("timestamp", ""),
                row.get("event_type", ""),
                row.get("direction", ""),
                row.get("target", ""),
                row.get("track_id", ""),
                row.get("value", ""),
                row.get("source_name", ""),
                row.get("frame_idx", ""),
            ])

        workbook.save(file_path)

    def start_processing(self):
        video_path = self.control_panel.get_video_path()
        if not video_path:
            self.log_panel.append_log("请先选择视频源")
            return

        self.current_runtime_start = time.time()
        self.current_frame_number = 0
        self.current_total_frames = 0
        self.current_total_people = 0
        self.current_up_count = 0
        self.current_down_count = 0
        self.current_detecting = True
        self.current_avg_fps = 0.0
        self.home_metric_total.set_value("0", "本次会话累计")
        self.home_metric_up.set_value("0", "上行人数")
        self.home_metric_down.set_value("0", "下行人数")
        self.home_metric_current.set_value("0", "当前画面")
        self.home_metric_fps.set_value("0.0", "实时性能")
        self.detect_metric_total.set_value("0", "实时统计")
        self.detect_metric_up.set_value("0", "上行人数")
        self.detect_metric_down.set_value("0", "下行人数")
        self.detect_metric_current.set_value("0", "当前画面")
        self.detect_metric_fps.set_value("0.0", "实时性能")
        self.worker.set_config(self.current_config)
        self.worker.set_source(video_path)
        self.worker.set_runtime_params(self.annotation_panel.get_params())
        self.worker.set_annotations(self.video_panel._roi_points, self.video_panel._line_points)

        self.home_trend_panel.reset()
        self.stats_history_panel.reset()
        self.stats_fps_panel.reset()
        self.event_table_panel.reset()
        self.filtered_event_table_panel.reset()
        self.home_summary_list.clear()
        self.event_history = []
        self.trend_history = []
        self.fps_history = []
        self._refresh_system_info()
        self._refresh_home_status_card()

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
        self.current_runtime_start = None
        self.current_detecting = False
        self._refresh_home_status_card()
        self._refresh_system_info()

    def on_trend_updated(self, payload):
        if bool(payload.get("reset", False)):
            self.home_trend_panel.reset()
            self.stats_history_panel.reset()
            return
        self.trend_history.append(
            {
                "minute": int(payload.get("minute", 0)),
                "up": int(payload.get("up", 0)),
                "down": int(payload.get("down", 0)),
                "total": int(payload.get("total", 0)),
            }
        )
        self.home_trend_panel.update_trend(payload)
        self.stats_history_panel.update_trend(payload)

    def on_stats_updated(self, stats):
        self.stats_panel.update_stats(stats)
        self.home_metric_total.set_value(str(int(stats.get("total", 0))), "本次会话累计")
        self.home_metric_up.set_value(str(int(stats.get("up", 0))), "上行人数")
        self.home_metric_down.set_value(str(int(stats.get("down", 0))), "下行人数")
        self.home_metric_current.set_value(str(int(stats.get("current", 0))), "当前画面")
        self.home_metric_fps.set_value(f"{float(stats.get('fps', 0.0)):.1f}", "实时性能")

        self.detect_metric_total.set_value(str(int(stats.get("total", 0))), "实时统计")
        self.detect_metric_up.set_value(str(int(stats.get("up", 0))), "上行人数")
        self.detect_metric_down.set_value(str(int(stats.get("down", 0))), "下行人数")
        self.detect_metric_current.set_value(str(int(stats.get("current", 0))), "当前画面")
        self.detect_metric_fps.set_value(f"{float(stats.get('fps', 0.0)):.1f}", "实时性能")

        self.stats_fps_panel.update_fps(stats.get("fps", 0.0))
        self.fps_history.append(float(stats.get("fps", 0.0)))
        self.current_source_fps = float(stats.get("source_fps", self.current_source_fps))
        self.current_frame_number = int(stats.get("current_frame", self.current_frame_number))
        self.current_total_frames = int(stats.get("total_frames", self.current_total_frames))
        self.current_total_people = int(stats.get("total", self.current_total_people))
        self.current_up_count = int(stats.get("up", self.current_up_count))
        self.current_down_count = int(stats.get("down", self.current_down_count))
        self.current_avg_fps = float(stats.get("avg_fps", self.current_avg_fps))
        self._refresh_system_info(stats)

    def on_event_emitted(self, event):
        if bool(event.get("reset", False)):
            self.event_table_panel.reset()
            self.filtered_event_table_panel.reset()
            self.home_summary_list.clear()
            return
        self.event_table_panel.add_event_record(event, fps=self.current_source_fps)
        event_name = str(event.get("value") or event.get("event_type") or "事件")
        track_id = event.get("track_id", "-")
        target = str(event.get("target", "-"))
        frame_idx = int(event.get("frame_idx", 0))
        sec = frame_idx / self.current_source_fps if self.current_source_fps > 1e-6 else 0.0
        mm = int(sec // 60)
        ss = int(sec % 60)
        direction = self._event_direction(event_name)
        event_record = {
            "timestamp": f"{mm:02d}:{ss:02d}",
            "seconds": int(sec),
            "event": event_name,
            "direction": direction,
            "track_id": str(track_id),
            "target": target,
            "video": self.current_source_name,
        }
        self.event_history.append(event_record)
        self.home_summary_list.insertItem(0, QListWidgetItem(f"{mm:02d}:{ss:02d} · {event_name} · ID {track_id} · {target}"))
        while self.home_summary_list.count() > 5:
            self.home_summary_list.takeItem(self.home_summary_list.count() - 1)

    def on_video_selected(self, path):
        self.log_panel.append_log(f"已选择视频源: {path}")
        self.current_source_name = "摄像头 0" if str(path) == "0" else (Path(path).name if path else "未选择")
        if self.query_video_name.findText(self.current_source_name) < 0:
            self.query_video_name.addItem(self.current_source_name)
        self._refresh_home_status_card()
        if path and str(path) != "0":
            import cv2
            from PyQt5.QtGui import QImage
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    self.current_video_resolution = f"{width} x {height}"
                    self._refresh_system_info()
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
        self.current_source_name = "摄像头 0"
        if self.query_video_name.findText(self.current_source_name) < 0:
            self.query_video_name.addItem(self.current_source_name)
        self._refresh_home_status_card()
        import cv2
        from PyQt5.QtGui import QImage
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width > 0 and height > 0:
                self.current_video_resolution = f"{width} x {height}"
                self._refresh_system_info()
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
        self.annotation_panel.update_annotation_coords(self.video_panel._roi_points, self.video_panel._line_points)

    def on_save_frame_requested(self):
        default_dir = Path(__file__).resolve().parents[1] / "outputs" / "gui_run"
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"frame_{datetime.now():%Y%m%d_%H%M%S}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前画面",
            str(default_dir / default_name),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not file_path:
            return

        if self.video_panel.save_current_frame(file_path):
            self.log_panel.append_log(f"当前画面已保存: {file_path}")
        else:
            QMessageBox.warning(self, "保存失败", "当前没有可保存的画面")

    def load_yaml_config(self):
        if not self._require_admin("加载配置"):
            return
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
        self.worker.warmup_detector_async()
        self.current_detector_type = self._detector_type_label()
        self.current_onnx_enabled = bool(cfg.get("onnx", {}).get("enabled", False))
        self._refresh_system_info()
        self._refresh_home_status_card()

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
            "draw_count_points": bool(cfg.get("debug", {}).get("draw_count_points", True)),
        }
        self.annotation_panel.set_params(params)

        roi_points = roi_cfg.get("polygon", [])
        line_points = line_cfg.get("points", [])
        self.video_panel.set_annotations(roi_points=roi_points, line_points=line_points)

        self.log_panel.append_log(f"配置已加载: {path}")
        self._append_management_log("配置管理", f"导入配置: {Path(path).name}", "成功")
        self._refresh_management_page()

    def save_yaml_config(self):
        if not self._require_admin("保存配置"):
            return
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

        debug_cfg = cfg.setdefault("debug", {})
        debug_cfg["draw_count_points"] = bool(params.get("draw_count_points", True))

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
        self.worker.warmup_detector_async()
        self.current_detector_type = self._detector_type_label()
        self.current_onnx_enabled = bool(cfg.get("onnx", {}).get("enabled", False))
        self._refresh_system_info()
        self.log_panel.append_log(f"配置已保存: {path}")
        self._append_management_log("配置管理", f"保存配置: {Path(path).name}", "成功")
        self._refresh_management_page()
        self._refresh_home_status_card()

    def show_about(self):
        QMessageBox.information(
            self,
            "关于",
            "基于YOLO的视频行人流量统计分析系统\n中期答辩轻量级GUI演示版",
        )

    def _default_config(self):
        return {
            "source": "",
            "performance": {
                # UI 刷新/曲线重绘节流（不影响检测/计数逻辑）
                "ui_max_fps": 15.0,
                "stats_hz": 10.0,
                "trend_hz": 2.0,
                # 处理分辨率上限（越小越快，精度可能略降）
                "max_width": 1280,
                "max_height": 800,
            },
            "detector": {
                "model_path": "E:/Video Pedestrian Flow Statistics and Analysis System Based on YOLO/ultralytics-8.3.163/runs/train/yolov8m_ema_p3_896_sgd_cloud/weights/best.pt",
                "conf": 0.5,
                "iou": 0.45,
            },
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
            "debug": {"draw_count_points": True},
        }

    def _device_label(self):
        try:
            import torch
        except ImportError:
            return "CPU"

        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"

    def _detector_type_label(self):
        detector_cfg = self.current_config.get("detector", {}) if isinstance(self.current_config, dict) else {}
        model_path = str(detector_cfg.get("model_path", ""))
        if not model_path:
            return "YOLO"

        path_obj = Path(model_path)
        model_dir = path_obj.parent.parent.name if len(path_obj.parts) >= 2 else path_obj.stem
        pretty = model_dir.replace("_", " ").strip()
        if pretty.lower().startswith("yolov8"):
            return pretty.replace("yolov8", "YOLOv8", 1)
        return pretty or "YOLO"

    def _format_runtime(self, seconds_value: float):
        seconds_value = max(0.0, float(seconds_value))
        total_seconds = int(seconds_value)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _refresh_system_info(self, stats: dict | None = None):
        self.current_device = self._device_label()
        self.current_detector_type = self._detector_type_label()
        runtime_seconds = 0.0
        if self.current_runtime_start is not None:
            runtime_seconds = time.time() - self.current_runtime_start

        if stats is not None:
            self.current_source_fps = float(stats.get("source_fps", self.current_source_fps))

        info = {
            "model_path": self.current_config.get("detector", {}).get("model_path", "-"),
            "video_resolution": self.current_video_resolution,
            "device": self.current_device,
            "current_frame": self.current_frame_number,
            "total_frames": self.current_total_frames if self.current_total_frames > 0 else "未知",
            "total_people": self.current_total_people,
            "runtime": self._format_runtime(runtime_seconds),
            "avg_fps": f"{self.current_avg_fps:.2f}",
            "detector_type": self.current_detector_type,
            "tracker_type": self.current_tracker_type,
            "onnx_enabled": "是" if self.current_onnx_enabled else "否",
        }
        self.system_info_panel.update_info(info)

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
        if not self._require_admin("导出 ONNX"):
            return
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
        self._append_management_log("模型管理", f"导出 ONNX: {Path(weights).name}", "已启动")

    def on_quantize_onnx_requested(self):
        if not self._require_admin("量化 ONNX"):
            return
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
        self._append_management_log("模型管理", f"量化 ONNX: {Path(model_path).name}", "已启动")

    def on_benchmark_onnx_requested(self):
        if not self._require_admin("Benchmark ONNX"):
            return
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
        self._append_management_benchmark_record(
            model_name=Path(model_path).stem,
            input_size=str(dialog.value("imgsz")),
            device=self._device_label(),
            note="已启动 Benchmark",
        )
        self._append_management_log("模型管理", f"Benchmark ONNX: {Path(model_path).name}", "已启动")

    def closeEvent(self, event):
        try:
            self.history_db.close()
        except Exception:
            pass
        try:
            self.auth_manager.close()
        except Exception:
            pass
        super().closeEvent(event)
