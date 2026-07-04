import sys
from pathlib import Path

# GUI 入口导入链中先锁定本地 ultralytics，避免使用系统 site-packages。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .main_window import MainWindow
from .auth_dialog import AuthDialog


def _write_startup_marker() -> None:
    """Write a tiny marker file to prove GUI entry started in frozen mode."""
    try:
        from utils.paths import writable_path

        marker = Path(writable_path("logs/startup.touch"))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("gui_entry_started\n", encoding="utf-8")
    except Exception:
        pass

# Setup startup logging for GUI entry as well (so double-click GUI produces logs)
try:
    from utils.paths import writable_path
    import logging
    log_dir = Path(writable_path("logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "startup.log"
    logging.basicConfig(level=logging.INFO, filename=str(log_file), filemode="a", format="%(asctime)s %(levelname)s %(message)s")
except Exception:
    import logging
    try:
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "行人检测系统" / "logs"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_log = temp_dir / "startup.log"
        logging.basicConfig(level=logging.INFO, filename=str(temp_log), filemode="a", format="%(asctime)s %(levelname)s %(message)s")
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def run_app():
    import sys
    from PyQt5.QtWidgets import QApplication, QDialog

    _write_startup_marker()
    logging.info("run_app entered")

    app = QApplication(sys.argv)
    auth_dialog = AuthDialog()
    if auth_dialog.exec_() != QDialog.Accepted or not auth_dialog.current_user:
        sys.exit(0)

    window = MainWindow(current_user=auth_dialog.current_user)
    window.show()
    sys.exit(app.exec_())
