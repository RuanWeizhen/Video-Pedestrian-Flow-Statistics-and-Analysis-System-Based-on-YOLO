from utils.auth_manager import AuthManager


class EmployeeService:
    def __init__(self, auth_manager: AuthManager | None = None):
        self.auth_manager = auth_manager or AuthManager()

    def getEmployees(self, username_query="", role_filter="全部", status_filter="全部"):
        data = self.auth_manager.get_all_users()
        result = []
        for emp in data:
            if username_query and username_query.lower() not in emp.get("username", "").lower():
                continue
            if role_filter != "全部" and emp.get("role") != role_filter:
                continue
            if status_filter != "全部" and emp.get("status") != status_filter:
                continue
            result.append(emp)
        return result

    def addEmployee(self, emp_data):
        return self.auth_manager.create_user(emp_data)

    def updateEmployee(self, emp_id, emp_data):
        return self.auth_manager.update_user(emp_id, emp_data)

    def deleteEmployee(self, emp_id):
        self.auth_manager.delete_user(emp_id)

    def resetPassword(self, emp_id):
        self.auth_manager.reset_user_password(emp_id, "123456")

    def toggleStatus(self, emp_id, current_status):
        new_status = "已禁用" if current_status == "已启用" else "已启用"
        self.auth_manager.toggle_user_status(emp_id, new_status)

    def getUserPermissions(self, user_id):
        return self.auth_manager.get_user_permissions(user_id)

    def updateUserPermissions(self, user_id, permissions):
        return self.auth_manager.update_user_permissions(user_id, permissions)

    def getAllPermissions(self):
        return self.auth_manager.get_all_permissions()

    def getPermissionLabels(self):
        return self.auth_manager.get_permission_labels()

    def addLog(self, operator_id, operator_name, action_type, target_id=None,
               target_name="", description="", result="成功"):
        self.auth_manager.add_operation_log(
            operator_id, operator_name, action_type,
            target_id=target_id, target_name=target_name,
            description=description, result=result,
        )

    def getLogs(self, limit=50):
        return self.auth_manager.get_operation_logs(limit=limit)

    def clearLogs(self):
        self.auth_manager.clear_operation_logs()


employee_service = EmployeeService()
