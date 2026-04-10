from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QGroupBox
import os

class ControlPanel(QWidget):
    video_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        group_box = QGroupBox("控制面板")
        vbox = QVBoxLayout()
        
        # 视频选择
        hbox_file = QHBoxLayout()
        self.btn_select = QPushButton("选择视频")
        self.btn_select_camera = QPushButton("打开摄像头")
        self.lbl_path = QLabel("未选择")
        self.lbl_path.setWordWrap(True)
        hbox_file.addWidget(self.btn_select)
        hbox_file.addWidget(self.btn_select_camera)
        hbox_file.addWidget(self.lbl_path)
        vbox.addLayout(hbox_file)
        
        # 运行控制
        hbox_ctrl = QHBoxLayout()
        self.btn_start = QPushButton("开始")
        self.btn_pause = QPushButton("暂停")
        self.btn_stop = QPushButton("停止")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_export = QPushButton("导出结果")
        
        hbox_ctrl.addWidget(self.btn_start)
        hbox_ctrl.addWidget(self.btn_pause)
        hbox_ctrl.addWidget(self.btn_stop)
        hbox_ctrl.addWidget(self.btn_export)
        vbox.addLayout(hbox_ctrl)
        
        group_box.setLayout(vbox)
        layout.addWidget(group_box)
        
        self.btn_select.clicked.connect(self.select_video)
        self.btn_select_camera.clicked.connect(self.select_camera)
        
    def select_video(self):
        # 默认指向项目 videos 目录
        default_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
        if not os.path.exists(default_dir):
            default_dir = ""
            
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", default_dir, "Video Files (*.mp4 *.avi)")
        if file_path:
            self.set_video_path(file_path)

    def select_camera(self):
        self.set_video_path("0")
            
    def get_video_path(self):
        return self.video_path

    def set_video_path(self, file_path):
        self.video_path = str(file_path or "")
        if self.video_path:
            if self.video_path == "0":
                self.lbl_path.setText("摄像头 0")
            else:
                self.lbl_path.setText(self.video_path)
        else:
            self.lbl_path.setText("未选择")
        self.video_selected.emit(self.video_path)
