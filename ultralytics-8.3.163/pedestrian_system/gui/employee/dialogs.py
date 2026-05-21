from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFormLayout, QMessageBox, QCheckBox,
    QGroupBox, QScrollArea, QWidget
)
from PyQt5.QtCore import Qt

from .services import employee_service


class EmployeeDialog(QDialog):
    def __init__(self, parent=None, is_edit=False, employee_data=None):
        super().__init__(parent)
        self.is_edit = is_edit
        self.employee_data = employee_data or {}

        self.setWindowTitle("编辑员工" if is_edit else "新增员工")
        self.setMinimumWidth(400)
        self._init_ui()
        self._populate_data()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        self.inp_username = QLineEdit()
        if self.is_edit:
            self.inp_username.setReadOnly(True)
        self.inp_fullname = QLineEdit()
        self.inp_password = QLineEdit()
        self.inp_password.setEchoMode(QLineEdit.Password)
        if self.is_edit:
            self.inp_password.setPlaceholderText("留空表示不修改")

        self.cmb_role = QComboBox()
        self.cmb_role.addItems(["员工", "管理员"])

        self.cmb_status = QComboBox()
        self.cmb_status.addItems(["已启用", "已禁用"])

        self.inp_phone = QLineEdit()
        self.inp_email = QLineEdit()

        form_layout.addRow("用户名<font color='red'>*</font>：", self.inp_username)
        form_layout.addRow("姓名<font color='red'>*</font>：", self.inp_fullname)
        if not self.is_edit:
            form_layout.addRow("密码<font color='red'>*</font>：", self.inp_password)
        else:
            form_layout.addRow("密码：", self.inp_password)
        form_layout.addRow("角色：", self.cmb_role)
        form_layout.addRow("账号状态：", self.cmb_status)
        form_layout.addRow("手机号：", self.inp_phone)
        form_layout.addRow("邮箱：", self.inp_email)

        layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_confirm = QPushButton("确认" if not self.is_edit else "保存")
        self.btn_cancel = QPushButton("取消")

        self.btn_confirm.clicked.connect(self.accept_data)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _populate_data(self):
        if self.is_edit and self.employee_data:
            self.inp_username.setText(self.employee_data.get("username", ""))
            self.inp_fullname.setText(self.employee_data.get("fullname", ""))

            role = self.employee_data.get("role", "员工")
            idx = self.cmb_role.findText(role)
            if idx >= 0:
                self.cmb_role.setCurrentIndex(idx)

            status = self.employee_data.get("status", "已启用")
            idx = self.cmb_status.findText(status)
            if idx >= 0:
                self.cmb_status.setCurrentIndex(idx)

            self.inp_phone.setText(self.employee_data.get("phone", ""))
            self.inp_email.setText(self.employee_data.get("email", ""))

    def get_data(self):
        data = {
            "username": self.inp_username.text().strip(),
            "fullname": self.inp_fullname.text().strip(),
            "role": self.cmb_role.currentText(),
            "status": self.cmb_status.currentText(),
            "phone": self.inp_phone.text().strip(),
            "email": self.inp_email.text().strip(),
        }
        pwd = self.inp_password.text().strip()
        if pwd:
            data["password"] = pwd
        return data

    def accept_data(self):
        username = self.inp_username.text().strip()
        fullname = self.inp_fullname.text().strip()
        password = self.inp_password.text().strip()

        if not username:
            QMessageBox.warning(self, "校验失败", "用户名不能为空！")
            self.inp_username.setFocus()
            return
        if not fullname:
            QMessageBox.warning(self, "校验失败", "姓名不能为空！")
            self.inp_fullname.setFocus()
            return
        if not self.is_edit and not password:
            QMessageBox.warning(self, "校验失败", "密码不能为空！")
            self.inp_password.setFocus()
            return

        self.accept()


class PermissionDialog(QDialog):
    def __init__(self, parent=None, user_id=None, username=""):
        super().__init__(parent)
        self.user_id = user_id
        self.username = username
        self._permission_checkboxes = {}

        self.setWindowTitle(f"权限配置 - {username}")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self._init_ui()
        self._load_permissions()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        header = QLabel(f"为员工 <b>{self.username}</b> 配置权限")
        header.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(header)

        hint = QLabel("勾选需要授予的权限，未勾选的权限将被拒绝。")
        hint.setStyleSheet("color: #6b7280; font-size: 13px;")
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)

        perm_groups = {
            "模型与检测": ["can_switch_model"],
            "标注与编辑": ["can_edit_roi", "can_edit_line"],
            "数据导出": ["can_export_data"],
            "管理功能": ["can_manage_users", "can_reset_password", "can_clear_logs"],
        }

        label_map = employee_service.getPermissionLabels()

        for group_title, perm_names in perm_groups.items():
            group = QGroupBox(group_title)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(6)

            for perm_name in perm_names:
                cb = QCheckBox(label_map.get(perm_name, perm_name))
                cb.setProperty("perm_name", perm_name)
                self._permission_checkboxes[perm_name] = cb
                group_layout.addWidget(cb)

            scroll_layout.addWidget(group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_save = QPushButton("保存权限")
        self.btn_save.setStyleSheet(
            "QPushButton { background: #3b82f6; color: #ffffff; font-size: 14px; font-weight: 600;"
            " border: none; border-radius: 10px; padding: 10px 24px; }"
            "QPushButton:hover { background: #2563eb; }"
            "QPushButton:pressed { background: #1d4ed8; }"
        )
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setStyleSheet(
            "QPushButton { background: #e5e7eb; color: #374151; font-size: 14px; font-weight: 600;"
            " border: none; border-radius: 10px; padding: 10px 24px; }"
            "QPushButton:hover { background: #d1d5db; }"
        )

        self.btn_save.clicked.connect(self._on_save)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _load_permissions(self):
        if self.user_id is None:
            return
        try:
            perms = employee_service.getUserPermissions(self.user_id)
            for perm_name, cb in self._permission_checkboxes.items():
                cb.setChecked(perms.get(perm_name, False))
        except Exception:
            pass

    def _on_save(self):
        permissions = {}
        for perm_name, cb in self._permission_checkboxes.items():
            permissions[perm_name] = cb.isChecked()

        try:
            employee_service.updateUserPermissions(self.user_id, permissions)
            QMessageBox.information(self, "成功", f"员工 {self.username} 的权限已更新！")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存权限失败: {e}")
