# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.163"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.models.yolo.model import YOLO

__all__ = ("__version__", "YOLO")
