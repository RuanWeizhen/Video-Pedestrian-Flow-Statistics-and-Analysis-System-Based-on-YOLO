from __future__ import annotations

from typing import Dict, List

from ultralytics import YOLO

from utils.common import Detection


class YOLODetector:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model_path = cfg["model_path"]
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.45))
        self.device = cfg.get("device", 0)
        self.half = bool(cfg.get("half", False))
        self.class_ids = set(cfg.get("class_ids", [0]))
        self.class_name_map = cfg.get("class_name_map", {0: "person"})

        self.model = YOLO(self.model_path)
        
        # 为了兼容在无GPU机器上直接报错回退到CPU而不等 predict 阶段，
        # 在这里执行一次快速设备检查
        import torch
        if str(self.device) != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, forced to raise early")

    def detect(self, frame) -> List[Detection]:
        result = next(
            self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                half=self.half,
                verbose=False,
                stream=True,
            )
        ).cpu()

        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.numpy()
        confs = result.boxes.conf.numpy()
        clss = result.boxes.cls.numpy()

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
