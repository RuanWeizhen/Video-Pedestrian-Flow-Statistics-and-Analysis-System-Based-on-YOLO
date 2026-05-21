from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from .paths import resource_path, writable_path


def _resolve_existing_path(value, base_dir: Path):
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and not value.strip():
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return value

    path = Path(value)
    if path.is_absolute():
        return str(path)

    for candidate in (base_dir / path, resource_path(path)):
        if candidate.exists():
            return str(candidate)

    return str(resource_path(path))


def _resolve_writable_path(value):
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and not value.strip():
        return value

    path = Path(value)
    if path.is_absolute():
        return str(path)

    return str(writable_path(path))


def load_config(config_path: str) -> Dict:
    path = resource_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "detector" not in cfg:
        raise KeyError("Config missing detector section")
    if "tracker" not in cfg:
        raise KeyError("Config missing tracker section")
    if "counting" not in cfg:
        raise KeyError("Config missing counting section")

    if "model_path" in cfg and "model_path" not in cfg["detector"]:
        cfg["detector"]["model_path"] = cfg["model_path"]

    cfg["source"] = _resolve_existing_path(cfg.get("source"), path.parent)
    if "output_dir" in cfg:
        cfg["output_dir"] = _resolve_writable_path(cfg["output_dir"])

    detector_cfg = cfg.get("detector", {})
    if "model_path" in detector_cfg:
        detector_cfg["model_path"] = _resolve_existing_path(detector_cfg["model_path"], path.parent)

    privacy_cfg = cfg.get("privacy", {})
    face_blur_cfg = privacy_cfg.get("face_blur", {})
    if "model_path" in face_blur_cfg:
        face_blur_cfg["model_path"] = _resolve_existing_path(face_blur_cfg["model_path"], path.parent)

    encryption_cfg = privacy_cfg.get("encryption", {})
    if "key_path" in encryption_cfg:
        encryption_cfg["key_path"] = _resolve_writable_path(encryption_cfg["key_path"])

    return cfg
