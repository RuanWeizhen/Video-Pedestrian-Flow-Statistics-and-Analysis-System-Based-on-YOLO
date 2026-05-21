from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# 强制优先加载本地 ultralytics（包含自定义模块如 EMA）。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.models.yolo.model import YOLO

from utils.common import Detection


class YOLODetector:
    def __init__(self, cfg: Dict):
        self.cfg = {}
        self.model_path = ""
        self.imgsz = 640
        self.conf = 0.25
        self.iou = 0.45
        self.device = 0
        self.half = False
        self.class_ids = {0}
        self.class_name_map = {0: "person"}

        self.model = None
        self.configure(cfg, load_model=True)

    def configure(self, cfg: Dict, load_model: bool = False) -> None:
        self.cfg = dict(cfg)
        self.model_path = self.cfg["model_path"]
        self.imgsz = int(self.cfg.get("imgsz", self.imgsz))
        self.conf = float(self.cfg.get("conf", self.conf))
        self.iou = float(self.cfg.get("iou", self.iou))
        self.device = self.cfg.get("device", self.device)
        self.half = bool(self.cfg.get("half", self.half))
        self.class_ids = set(self.cfg.get("class_ids", [0]))
        self.class_name_map = dict(self.cfg.get("class_name_map", {0: "person"}))

        if load_model or self.model is None:
            self.model = YOLO(self.model_path)

        self._normalize_device()

    def apply_runtime_params(self, conf: float | None = None, iou: float | None = None) -> None:
        if conf is not None:
            self.conf = float(conf)
        if iou is not None:
            self.iou = float(iou)

    def _normalize_device(self) -> None:
        try:
            import torch
        except Exception:
            self.device = "cpu"
            self.half = False
            return

        device_str = str(self.device).strip().lower()
        if device_str == "cpu":
            self.device = "cpu"
            self.half = False
            return

        if not torch.cuda.is_available():
            self.device = "cpu"
            self.half = False
            return

        if device_str in {"cuda", "cuda:0", "0", "gpu", "gpu:0"}:
            self.device = 0
            return

        self.device = self.device

    def detect(self, frame) -> List[Detection]:
        if self.model is None:
            self.model = YOLO(self.model_path)

        # 仅检测指定类别：减少 NMS/后处理开销
        classes = sorted(self.class_ids) if self.class_ids else None

        result = next(
            self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                half=self.half,
                classes=classes,
                verbose=False,
                stream=True,
            )
        )

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls

        # 只把必要张量搬到 CPU
        xyxy = xyxy.cpu().numpy()
        confs = confs.cpu().numpy()
        clss = clss.cpu().numpy()

        detections: List[Detection] = []
        for box, conf, cls_id_float in zip(xyxy, confs, clss):
            cls_id = int(cls_id_float)
            if cls_id not in self.class_ids:
                continue

            x1, y1, x2, y2 = map(float, box.tolist())
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(conf),
                    class_id=cls_id,
                    class_name=str(self.class_name_map.get(cls_id, f"class_{cls_id}")),
                )
            )
        return detections

    def summary(self) -> Dict:
        return {
            "model_path": self.model_path,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "half": self.half,
            "class_ids": sorted(self.class_ids),
        }
