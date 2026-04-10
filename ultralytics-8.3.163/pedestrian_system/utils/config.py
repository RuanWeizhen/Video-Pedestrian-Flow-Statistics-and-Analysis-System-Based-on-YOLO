from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


def load_config(config_path: str) -> Dict:
    path = Path(config_path)
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

    return cfg
