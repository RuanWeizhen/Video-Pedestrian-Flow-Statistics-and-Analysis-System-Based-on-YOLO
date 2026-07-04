import re

file_path = 'E:/Video Pedestrian Flow Statistics and Analysis System Based on YOLO/ultralytics-8.3.163/pedestrian_system/gui/main_window.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Delete unused handle_ functions
methods_to_remove = [
    r'def _load_sample_employees\(self\):.*?def _load_sample_employee_logs\(self\):',
    r'def _load_sample_employee_logs\(self\):.*?def refresh_employee_list\(self\):',
    r'def refresh_employee_list\(self\):.*?def apply_employee_filters\(self\):',
    r'def apply_employee_filters\(self\):.*?def reset_employee_filters\(self\):',
    r'def reset_employee_filters\(self\):.*?def refresh_employee_logs\(self\):',
    r'def refresh_employee_logs\(self\):.*?def _append_employee_log\(self, action, target, result\):',
    r'def _append_employee_log\(self, action, target, result\):.*?def _selected_employee_row\(self\):',
    r'def _selected_employee_row\(self\):.*?def handle_add_employee\(self\):',
    r'def handle_add_employee\(self\):.*?def handle_edit_employee\(self\):',
    r'def handle_edit_employee\(self\):.*?def handle_delete_employee\(self\):',
    r'def handle_delete_employee\(self\):.*?def handle_reset_employee_password\(self\):',
    r'def handle_reset_employee_password\(self\):.*?def handle_enable_employee\(self\):',
    r'def handle_enable_employee\(self\):.*?def handle_disable_employee\(self\):',
    r'def handle_disable_employee\(self\):.*?def _refresh_video_sources\(self\):' # up to next valid func
]

pattern = re.compile(r'def _load_sample_employees\(self\):.*?def _refresh_video_sources\(self\):', re.DOTALL)
content = pattern.sub('def _refresh_video_sources(self):', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
