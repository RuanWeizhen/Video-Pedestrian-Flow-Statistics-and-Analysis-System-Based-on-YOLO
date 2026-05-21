from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return source_root().parent


def bundle_root() -> Path:
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return source_root()


def runtime_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return source_root()


def app_data_root() -> Path:
    """Return a writable root for runtime data.

    In frozen mode, prefer a user-writable directory to avoid permission issues
    when the app is launched from protected locations.
    """
    if not is_frozen():
        return source_root()

    user_profile = Path(os.environ.get("USERPROFILE", "")).resolve() if os.environ.get("USERPROFILE") else None
    if user_profile and user_profile.exists():
        docs_dir = user_profile / "Documents"
        if docs_dir.exists():
            return docs_dir / "客流统计系统"
        return user_profile / "客流统计系统"

    return runtime_root() / "user_data"


def resource_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path

    for base in (bundle_root(), source_root(), project_root(), runtime_root()):
        candidate = base / path
        if candidate.exists():
            return candidate

    return bundle_root() / path


def writable_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return app_data_root() / path