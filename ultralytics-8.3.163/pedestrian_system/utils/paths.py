from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger("pedestrian_system.paths")


def _ensure_logger():
    if not _logger.handlers:
        _logger.addHandler(logging.NullHandler())
    return _logger


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
    if not is_frozen():
        return source_root()

    user_profile = Path(os.environ.get("USERPROFILE", "")).resolve() if os.environ.get("USERPROFILE") else None
    if user_profile and user_profile.exists():
        docs_dir = user_profile / "Documents"
        if docs_dir.exists():
            return docs_dir / "行人检测系统"
        return user_profile / "行人检测系统"

    return runtime_root() / "user_data"


def resource_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    log = _ensure_logger()

    if path.is_absolute():
        log.info("[resource_path] absolute path resolved: %s", path)
        return path

    search_bases = (bundle_root(), runtime_root(), source_root(), project_root())

    for base in search_bases:
        candidate = base / path
        if candidate.exists():
            log.info("[resource_path] found at: %s  (base: %s)", candidate, base)
            return candidate

    fallback = bundle_root() / path
    log.warning("[resource_path] NOT FOUND in any base, fallback to: %s  (searched: %s)", fallback, [str(b) for b in search_bases])
    return fallback


def external_resource_root() -> Path:
    if is_frozen():
        return runtime_root()
    return source_root()


def external_resource_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    log = _ensure_logger()

    if path.is_absolute():
        if not path.exists():
            log.warning("[external_resource_path] absolute path does NOT exist: %s", path)
        else:
            log.info("[external_resource_path] absolute path resolved: %s", path)
        return path

    search_bases = []
    if is_frozen():
        search_bases = [runtime_root(), source_root(), project_root()]
    else:
        search_bases = [source_root(), project_root()]

    for base in search_bases:
        candidate = base / path
        if candidate.exists():
            log.info("[external_resource_path] found at: %s  (base: %s)", candidate, base)
            return candidate

    best_guess = (runtime_root() if is_frozen() else source_root()) / path
    log.warning("[external_resource_path] NOT FOUND in any base, best-guess path: %s  (searched: %s)", best_guess, [str(b) for b in search_bases])
    return best_guess


def writable_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    log = _ensure_logger()

    if path.is_absolute():
        log.info("[writable_path] absolute path resolved: %s", path)
        return path

    resolved = app_data_root() / path
    log.info("[writable_path] resolved to: %s", resolved)
    return resolved
