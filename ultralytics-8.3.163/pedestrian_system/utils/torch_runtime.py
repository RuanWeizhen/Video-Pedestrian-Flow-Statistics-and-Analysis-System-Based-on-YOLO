from __future__ import annotations

import os
import sys
from pathlib import Path


def _frozen_torch_dll_dirs() -> list[Path]:
    if not getattr(sys, "frozen", False):
        return []

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return []

    base = Path(meipass)
    return [base, base / "torch" / "lib", base / "torchvision", base / "cv2"]


def ensure_torch_preloaded():
    """Preload torch on the main thread before worker threads use YOLO."""
    if "torch" in sys.modules:
        return sys.modules["torch"]

    for dll_dir in _frozen_torch_dll_dirs():
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass

    try:
        import torch
    except Exception:
        return None

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = False
        torch.cuda.is_available()
    except Exception:
        pass

    return torch