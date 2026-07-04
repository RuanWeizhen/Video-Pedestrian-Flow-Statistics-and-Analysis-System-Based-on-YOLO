from __future__ import annotations

import sys
import os
from pathlib import Path

_frozen = bool(getattr(sys, "frozen", False))
_meipass = getattr(sys, "_MEIPASS", None)

if _frozen:
    dll_dirs = []
    if _meipass:
        meipass_path = Path(_meipass)
        dll_dirs.append(meipass_path)
        dll_dirs.append(meipass_path / "torch" / "lib")
        dll_dirs.append(meipass_path / "torchvision")
        dll_dirs.append(meipass_path / "cv2")
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass

_py_file = Path(__file__).resolve()
_repo_root = _py_file.parents[1]

if _frozen and _meipass:
    _pedestrian_root = Path(_meipass) / "pedestrian_system"
else:
    _pedestrian_root = _repo_root / "pedestrian_system"

for _p in (_repo_root, _pedestrian_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pedestrian_system.gui import run_app

if __name__ == "__main__":
    run_app()
