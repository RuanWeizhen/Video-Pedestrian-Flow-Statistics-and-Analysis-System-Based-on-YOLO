from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal


class EmployeeSearch(QWidget):
    search_triggered = pyqtSignal(str, str, str)
    reset_triggered = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.emp_search_input = QLineEdit()
        self.emp_search_input.setPlaceholderText("用户名搜索")

        self.emp_role_filter = QComboBox()
        self.emp_role_filter.addItems(["全部", "管理员", "员工"])

        self.emp_status_filter = QComboBox()
        self.emp_status_filter.addItems(["全部", "已启用", "已禁用"])

        self.btn_query = QPushButton("查询")
        self.btn_reset = QPushButton("重置")

        layout.addWidget(QLabel("用户名"))
        layout.addWidget(self.emp_search_input)
        layout.addWidget(QLabel("角色"))
        layout.addWidget(self.emp_role_filter)
        layout.addWidget(QLabel("状态"))
        layout.addWidget(self.emp_status_filter)
        layout.addWidget(self.btn_query)
        layout.addWidget(self.btn_reset)
        layout.addStretch()

        self.btn_query.clicked.connect(self._on_query)
        self.btn_reset.clicked.connect(self._on_reset)

    def _on_query(self):
        username = self.emp_search_input.text().strip()
        role = self.emp_role_filter.currentText()
        status = self.emp_status_filter.currentText()
        self.search_triggered.emit(username, role, status)

    def _on_reset(self):
        self.emp_search_input.clear()
        self.emp_role_filter.setCurrentIndex(0)
        self.emp_status_filter.setCurrentIndex(0)
        self.reset_triggered.emit()


class EmployeeTable(QTableWidget):
    permission_config_requested = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(0, 10, parent)
        self._init_ui()

    def _init_ui(self):
        self.setHorizontalHeaderLabels([
            "员工编号", "用户名", "姓名", "角色", "账号状态",
            "手机号", "注册时间", "上次登录时间", "权限概览", "操作"
        ])
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setStyleSheet("QTableWidget::item:hover { background-color: #e0f7fa; }")

    def load_data(self, data_list):
        self.setRowCount(0)
        for i, emp in enumerate(data_list):
            self.insertRow(i)
            self.setItem(i, 0, QTableWidgetItem(str(emp.get("id", ""))))
            self.setItem(i, 1, QTableWidgetItem(emp.get("username", "")))
            self.setItem(i, 2, QTableWidgetItem(emp.get("fullname", "")))
            self.setItem(i, 3, QTableWidgetItem(emp.get("role", "")))

            status_item = QTableWidgetItem(emp.get("status", ""))
            if emp.get("status") == "已启用":
                status_item.setForeground(Qt.green)
            else:
                status_item.setForeground(Qt.red)
            self.setItem(i, 4, status_item)

            self.setItem(i, 5, QTableWidgetItem(emp.get("phone", "-")))
            self.setItem(i, 6, QTableWidgetItem(emp.get("created_at", "")))
            self.setItem(i, 7, QTableWidgetItem(emp.get("last_login", "")))

            summary = emp.get("permission_summary", "基础权限")
            perm_item = QTableWidgetItem(summary)
            perm_item.setToolTip(summary)
            self.setItem(i, 8, perm_item)

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            btn_layout.setSpacing(4)

            btn_perm = QPushButton("权限配置")
            btn_perm.setFixedHeight(28)
            btn_perm.setStyleSheet(
                "QPushButton { background: #f59e0b; color: #ffffff; font-size: 12px; font-weight: 600;"
                " border: none; border-radius: 6px; padding: 4px 10px; }"
                "QPushButton:hover { background: #d97706; }"
            )
            emp_id = emp.get("id")
            emp_name = emp.get("username", "")
            btn_perm.clicked.connect(lambda checked=False, uid=emp_id, uname=emp_name: self.permission_config_requested.emit(uid, uname))

            btn_layout.addWidget(btn_perm)
            btn_layout.addStretch()
            self.setCellWidget(i, 9, btn_widget)

            for j in range(8):
                item = self.item(i, j)
                if item:
                    item.setData(Qt.UserRole, emp)

    def get_selected_employee(self):
        selected = self.selectedItems()
        if not selected:
            return None
        return selected[0].data(Qt.UserRole)


class OperationLog(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self._init_ui()

    def _init_ui(self):
        self.setHorizontalHeaderLabels(["操作时间", "操作人", "操作类型", "目标员工", "操作结果"])
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def add_log(self, time, operator, action, target, result):
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(time))
        self.setItem(row, 1, QTableWidgetItem(operator))
        self.setItem(row, 2, QTableWidgetItem(action))
        self.setItem(row, 3, QTableWidgetItem(target))
        self.setItem(row, 4, QTableWidgetItem(result))

    def load_logs(self, log_list):
        self.setRowCount(0)
        action_label = {
            "add": "新增员工", "edit": "编辑员工", "delete": "删除员工",
            "reset_password": "重置密码", "toggle_status": "启/禁账号",
            "update_permissions": "权限配置",
        }
        for log in log_list:
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QTableWidgetItem(log.get("created_at", "")))
            self.setItem(row, 1, QTableWidgetItem(log.get("operator_name", "")))
            self.setItem(row, 2, QTableWidgetItem(action_label.get(log.get("action_type", ""), log.get("action_type", ""))))
            self.setItem(row, 3, QTableWidgetItem(log.get("target_name", "")))
            self.setItem(row, 4, QTableWidgetItem(log.get("result", "")))
