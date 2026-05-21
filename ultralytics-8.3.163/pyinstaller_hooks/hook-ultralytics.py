from __future__ import annotations

# Minimal hook for GUI inference packaging.
# The default ultralytics hook pulls in training/export/solution submodules that
# are not needed by this pedestrian-flow desktop app.

hiddenimports = [
    "ultralytics.cfg",
    "ultralytics.engine.model",
    "ultralytics.engine.predictor",
    "ultralytics.engine.results",
    "ultralytics.models.yolo.detect",
    "ultralytics.models.yolo.detect.predict",
    "ultralytics.models.yolo.detect.train",
    "ultralytics.models.yolo.detect.val",
    "ultralytics.nn.tasks",
]