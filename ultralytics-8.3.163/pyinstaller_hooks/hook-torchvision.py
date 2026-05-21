from __future__ import annotations

# Keep torchvision minimal for NMS / ops used by Ultralytics inference.
hiddenimports = [
    "torchvision.ops",
]
