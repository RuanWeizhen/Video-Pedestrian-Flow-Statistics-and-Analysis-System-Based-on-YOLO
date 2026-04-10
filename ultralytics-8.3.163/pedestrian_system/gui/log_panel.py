from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit
import time

class LogPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        
    def append_log(self, message):
        t_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
        self.text_edit.append(f"{t_str} {message}")
