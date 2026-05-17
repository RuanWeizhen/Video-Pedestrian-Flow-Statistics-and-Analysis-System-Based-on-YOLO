import sys
from pathlib import Path

# GUI 入口导入链中先锁定本地 ultralytics，避免使用系统 site-packages。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .main_window import MainWindow
from .auth_dialog import AuthDialog

def run_app():
    import sys
    from PyQt5.QtWidgets import QApplication, QDialog

    app = QApplication(sys.argv)
    auth_dialog = AuthDialog()
    if auth_dialog.exec_() != QDialog.Accepted or not auth_dialog.current_user:
        sys.exit(0)

    window = MainWindow(current_user=auth_dialog.current_user)
    window.show()
    sys.exit(app.exec_())
