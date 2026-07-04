import re

file_path = 'E:/Video Pedestrian Flow Statistics and Analysis System Based on YOLO/ultralytics-8.3.163/pedestrian_system/gui/main_window.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace _build_password_page
pattern1 = re.compile(r'def _build_password_page\(self\):.*?return page\n', re.DOTALL)
replacement1 = '''def _build_password_page(self):
        page = EmployeeManagementPage(self.current_user)
        self.employee_page_widget = page
        return page
'''
content = pattern1.sub(replacement1, content, count=1)

# Replace 'self.refresh_employee_list()' out of _switch_page
pattern2 = re.compile(r'self\.refresh_employee_list\(\)\s*self\.refresh_employee_logs\(\)')
replacement2 = '''if hasattr(self, "employee_page_widget"):
                self.employee_page_widget.current_user = self.current_user
                self.employee_page_widget.refresh_list()'''
content = pattern2.sub(replacement2, content, count=1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
