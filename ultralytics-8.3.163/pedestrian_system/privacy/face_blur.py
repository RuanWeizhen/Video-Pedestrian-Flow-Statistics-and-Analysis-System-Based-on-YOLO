from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FaceBlur:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.model_path = str(cfg.get("model_path", "models/yolov8n-face.pt"))
        self.conf = float(cfg.get("conf", 0.4))
        self.blur_type = str(cfg.get("blur_type", "gaussian")).lower()
        self.blur_kernel = int(cfg.get("blur_kernel", 25))

        if self.blur_kernel < 3:
            self.blur_kernel = 3
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

        self.model = None
        if not self.enabled:
            return

        try:
            from ultralytics import YOLO

            if not Path(self.model_path).exists():
                print(f"[Privacy] Face model not found: {self.model_path}. Face blur disabled.")
                self.enabled = False
                return

            self.model = YOLO(self.model_path)
            print("[Privacy] Face blur enabled (CPU).")
        except Exception as exc:
            print(f"[Privacy] Face blur initialization failed: {exc}")
            self.enabled = False
            self.model = None

    def _gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        max_k = min(h, w)
        k = min(self.blur_kernel, max_k if max_k % 2 == 1 else max_k - 1)
        if k < 3:
            return roi
        return cv2.GaussianBlur(roi, (k, k), 0)

    def _mosaic_blur(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi

        down_w = max(1, w // 10)
        down_h = max(1, h // 10)
        small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or self.model is None or frame is None:
            return frame

        try:
            results = self.model.predict(source=frame, conf=self.conf, verbose=False, device="cpu")
        except Exception as exc:
            print(f"[Privacy] Face blur predict failed: {exc}")
            return frame

        if not results:
            return frame

        output = frame.copy()
        boxes = results[0].boxes
        if boxes is None:
            return output

        xyxy = boxes.xyxy
        if xyxy is None:
            return output

        coords = xyxy.cpu().numpy().astype(int)
        frame_h, frame_w = output.shape[:2]

        for x1, y1, x2, y2 in coords:
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if self.blur_type == "mosaic":
                blurred = self._mosaic_blur(roi)
            else:
                blurred = self._gaussian_blur(roi)

            output[y1:y2, x1:x2] = blurred

        return output
