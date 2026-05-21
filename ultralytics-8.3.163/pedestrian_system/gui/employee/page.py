from datetime import datetime
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QGroupBox
from PyQt5.QtCore import Qt
from .components import EmployeeSearch, EmployeeTable, OperationLog
from .dialogs import EmployeeDialog, PermissionDialog
from .services import employee_service


class DashboardSection(QGroupBox):
    def __init__(self, title, subtitle="", parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self.layout = QVBoxLayout(self)

    def set_body_widget(self, widget):
        self.layout.addWidget(widget)


class EmployeeManagementPage(QWidget):
    def __init__(self, current_user=None, parent=None, on_permission_changed=None):
        super().__init__(parent)
        self.current_user = current_user or {"username": "admin", "role": "管理员"}
        self.on_permission_changed = on_permission_changed
        self._init_ui()
        self.refresh_list()
        self.refresh_logs()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        self.search_section = DashboardSection("查询条件")
        self.search_comp = EmployeeSearch()
        self.search_comp.search_triggered.connect(self._on_search)
        self.search_comp.reset_triggered.connect(self.refresh_list)
        self.search_section.set_body_widget(self.search_comp)
        layout.addWidget(self.search_section)

        self.action_section = DashboardSection("员工操作")
        action_body = QWidget()
        action_layout = QHBoxLayout(action_body)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        self.btn_add = QPushButton("🆕 新增员工")
        self.btn_edit = QPushButton("✏️ 编辑员工")
        self.btn_delete = QPushButton("🗑️ 删除员工")
        self.btn_delete.setStyleSheet("color: red;")
        self.btn_reset_pwd = QPushButton("🔑 重置密码")
        self.btn_toggle_status = QPushButton("🛡️ 启/禁账号")

        self.btn_add.clicked.connect(self.handle_add)
        self.btn_edit.clicked.connect(self.handle_edit)
        self.btn_delete.clicked.connect(self.handle_delete)
        self.btn_reset_pwd.clicked.connect(self.handle_reset_pwd)
        self.btn_toggle_status.clicked.connect(self.handle_toggle_status)

        action_layout.addWidget(self.btn_add)
        action_layout.addWidget(self.btn_edit)
        action_layout.addWidget(self.btn_delete)
        action_layout.addWidget(self.btn_reset_pwd)
        action_layout.addWidget(self.btn_toggle_status)
        action_layout.addStretch()
        self.action_section.set_body_widget(action_body)
        layout.addWidget(self.action_section)

        self.table_section = DashboardSection("员工表格")
        self.table_comp = EmployeeTable()
        self.table_comp.permission_config_requested.connect(self.handle_permission_config)
        self.table_section.set_body_widget(self.table_comp)
        layout.addWidget(self.table_section, 1)

        self.log_section = DashboardSection("操作日志")
        self.log_comp = OperationLog()
        log_wrapper = QWidget()
        log_wrapper_layout = QVBoxLayout(log_wrapper)
        log_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        log_wrapper_layout.setSpacing(8)
        log_wrapper_layout.addWidget(self.log_comp)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_clear_logs = QPushButton("🗑️ 清空操作日志")
        self.btn_clear_logs.setCursor(Qt.PointingHandCursor)
        self.btn_clear_logs.setStyleSheet(
            "QPushButton {"
            "  background: #FEE2E2; color: #991B1B; font-size: 12px; font-weight: 600;"
            "  border: none; border-radius: 6px; padding: 6px 14px;"
            "}"
            "QPushButton:hover { background: #FECACA; }"
            "QPushButton:pressed { background: #FCA5A5; }"
        )
        self.btn_clear_logs.clicked.connect(self.handle_clear_logs)
        btn_row.addWidget(self.btn_clear_logs)
        log_wrapper_layout.addLayout(btn_row)
        self.log_section.set_body_widget(log_wrapper)
        layout.addWidget(self.log_section)

        self._apply_log_clear_permission()

    def _apply_log_clear_permission(self):
        perms = (self.current_user or {}).get("permissions", {})
        can_clear = perms.get("can_clear_logs", False)
        role = (self.current_user or {}).get("role", "")
        if role == "管理员" and not perms:
            can_clear = True
        self.btn_clear_logs.setVisible(can_clear)

    def _get_operator_info(self):
        user = self.current_user or {}
        return user.get("id"), user.get("username", "unknown")

    def _append_log_db(self, action_type, target_name="", result="成功", description="",
                        target_id=None):
        try:
            op_id, op_name = self._get_operator_info()
            employee_service.addLog(op_id, op_name, action_type,
                                     target_id=target_id, target_name=target_name,
                                     description=description, result=result)
            self.refresh_logs()
        except Exception:
            pass

    def refresh_logs(self):
        try:
            logs = employee_service.getLogs(50)
            self.log_comp.load_logs(logs)
        except Exception:
            pass

    def refresh_list(self, username="", role="全部", status="全部"):
        try:
            data = employee_service.getEmployees(username, role, status)
            self.table_comp.load_data(data)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取列表失败: {e}")

    def _on_search(self, username, role, status):
        self.refresh_list(username, role, status)

    def handle_add(self):
        dialog = EmployeeDialog(self)
        if dialog.exec_():
            data = dialog.get_data()
            try:
                new_emp = employee_service.addEmployee(data)
                QMessageBox.information(self, "成功", "新增员工成功！")
                self._append_log_db("add", target_name=data.get("username"), target_id=new_emp.get("id"))
                self.refresh_list()
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))
                self._append_log_db("add", target_name=data.get("username"), result="失败", description=str(e))

    def handle_edit(self):
        emp = self.table_comp.get_selected_employee()
        if not emp:
            QMessageBox.warning(self, "提示", "请先选择需要编辑的员工！")
            return

        dialog = EmployeeDialog(self, is_edit=True, employee_data=emp)
        if dialog.exec_():
            data = dialog.get_data()
            try:
                employee_service.updateEmployee(emp["id"], data)
                QMessageBox.information(self, "成功", "修改员工成功！")
                self._append_log_db("edit", target_name=emp.get("username"), target_id=emp.get("id"))
                self.refresh_list()
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))
                self._append_log_db("edit", target_name=emp.get("username"), result="失败", description=str(e))

    def handle_delete(self):
        emp = self.table_comp.get_selected_employee()
        if not emp:
            QMessageBox.warning(self, "提示", "请先选择需要删除的员工！")
            return

        if emp.get("username") == self.current_user.get("username"):
            QMessageBox.warning(self, "失败", "不能删除当前登录的账号！")
            return

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除员工 {emp.get('username')} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                employee_service.deleteEmployee(emp["id"])
                QMessageBox.information(self, "成功", "删除成功！")
                self._append_log_db("delete", target_name=emp.get("username"), target_id=emp.get("id"))
                self.refresh_list()
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))
                self._append_log_db("delete", target_name=emp.get("username"), result="失败", description=str(e))

    def handle_reset_pwd(self):
        current_perms = (self.current_user or {}).get("permissions", {})
        if not current_perms.get("can_reset_password", True):
            QMessageBox.warning(self, "权限不足", "您没有重置他人密码的权限。")
            return

        emp = self.table_comp.get_selected_employee()
        if not emp:
            QMessageBox.warning(self, "提示", "请先选择需要重置密码的员工！")
            return

        reply = QMessageBox.question(
            self, "确认重置",
            f"确定要重置 {emp.get('username')} 的密码为 123456 吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                employee_service.resetPassword(emp["id"])
                QMessageBox.information(self, "成功", "密码已重置！")
                self._append_log_db("reset_password", target_name=emp.get("username"), target_id=emp.get("id"))
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))
                self._append_log_db("reset_password", target_name=emp.get("username"), result="失败", description=str(e))

    def handle_toggle_status(self):
        emp = self.table_comp.get_selected_employee()
        if not emp:
            QMessageBox.warning(self, "提示", "请先选择需要操作的员工！")
            return

        try:
            employee_service.toggleStatus(emp["id"], emp["status"])
            QMessageBox.information(self, "成功", "状态已更新！")
            self._append_log_db("toggle_status", target_name=emp.get("username"), target_id=emp.get("id"))
            self.refresh_list()
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))
            self._append_log_db("toggle_status", target_name=emp.get("username"), result="失败", description=str(e))

    def handle_permission_config(self, user_id, username):
        dialog = PermissionDialog(self, user_id=user_id, username=username)
        if dialog.exec_():
            self._append_log_db("update_permissions", target_name=username, target_id=user_id)
            current_uid = (self.current_user or {}).get("id")
            if current_uid == user_id and self.on_permission_changed:
                self.on_permission_changed()
            self.refresh_list()

    def handle_clear_logs(self):
        perms = (self.current_user or {}).get("permissions", {})
        role = (self.current_user or {}).get("role", "")
        can_clear = perms.get("can_clear_logs", False) or (role == "管理员" and not perms)
        if not can_clear:
            QMessageBox.warning(self, "权限不足", "您没有清除日志的权限。")
            return

        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有操作日志吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                employee_service.clearLogs()
                self.log_comp.setRowCount(0)
                QMessageBox.information(self, "成功", "操作日志已清空！")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"清空日志失败: {e}")
