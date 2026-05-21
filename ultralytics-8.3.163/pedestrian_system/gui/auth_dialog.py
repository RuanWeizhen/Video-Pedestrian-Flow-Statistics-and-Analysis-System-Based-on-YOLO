from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from utils.auth_manager import AuthManager


class AuthDialog(QDialog):
    def __init__(self, parent=None, auth_manager: AuthManager | None = None):
        super().__init__(parent)
        self.auth_manager = auth_manager or AuthManager()
        self.current_user: dict | None = None
        self.setWindowTitle("系统登录与注册")
        self.resize(520, 340)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        header = QLabel("行人客流统计系统入口")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 20px; font-weight: 600;")

        subheader = QLabel("请选择身份后登录。没有账号时可直接注册。")
        subheader.setAlignment(Qt.AlignCenter)
        subheader.setStyleSheet("color: #666666;")

        root_layout.addWidget(header)
        root_layout.addWidget(subheader)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_login_tab(), "登录")
        self.tabs.addTab(self._build_register_tab(), "注册")
        root_layout.addWidget(self.tabs)

    def _make_role_combo(self) -> QComboBox:
        role_combo = QComboBox()
        role_combo.addItems(["员工", "管理员"])
        return role_combo

    def _make_input(self, placeholder: str, password: bool = False) -> QLineEdit:
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        if password:
            line_edit.setEchoMode(QLineEdit.Password)
        return line_edit

    def _build_group(self, title: str):
        group_widget = QWidget()
        layout = QVBoxLayout(group_widget)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(title_label)

        form = QFormLayout()
        username_input = self._make_input("请输入用户名")
        password_input = self._make_input("请输入密码", password=True)
        role_combo = self._make_role_combo()

        form.addRow("用户名", username_input)
        form.addRow("密码", password_input)
        form.addRow("角色", role_combo)
        layout.addLayout(form)

        error_label = QLabel("")
        error_label.setWordWrap(True)
        error_label.setStyleSheet("color: #c0392b; min-height: 24px;")
        layout.addWidget(error_label)

        return group_widget, username_input, password_input, role_combo, error_label

    def _build_login_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group_widget, self.login_username, self.login_password, self.login_role, self.login_error = self._build_group("登录账号")
        layout.addWidget(group_widget)

        button_row = QHBoxLayout()
        login_button = QPushButton("登录")
        login_button.clicked.connect(self.handle_login)
        button_row.addStretch()
        button_row.addWidget(login_button)
        layout.addLayout(button_row)

        return tab

    def _build_register_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group_widget, self.register_username, self.register_password, self.register_role, self.register_error = self._build_group("注册账号")
        layout.addWidget(group_widget)

        button_row = QHBoxLayout()
        register_button = QPushButton("注册")
        register_button.clicked.connect(self.handle_register)
        button_row.addStretch()
        button_row.addWidget(register_button)
        layout.addLayout(button_row)

        return tab

    def _set_error(self, label: QLabel, message: str) -> None:
        label.setText(message)

    def handle_login(self) -> None:
        self._set_error(self.register_error, "")
        success, message, user_info = self.auth_manager.login(
            self.login_username.text(),
            self.login_password.text(),
            self.login_role.currentText(),
        )
        if not success:
            self._set_error(self.login_error, message)
            return

        self.current_user = user_info
        self.auth_manager.close()
        self.accept()

    def handle_register(self) -> None:
        self._set_error(self.login_error, "")
        success, message = self.auth_manager.register_user(
            self.register_username.text(),
            self.register_password.text(),
            self.register_role.currentText(),
        )
        if not success:
            self._set_error(self.register_error, message)
            return

        self._set_error(self.register_error, "注册成功，请返回登录页登录")
        self.login_username.setText(self.register_username.text().strip())
        self.login_password.clear()
        self.login_role.setCurrentText(self.register_role.currentText())
        self.tabs.setCurrentIndex(0)

    def closeEvent(self, event) -> None:
        self.auth_manager.close()
        super().closeEvent(event)