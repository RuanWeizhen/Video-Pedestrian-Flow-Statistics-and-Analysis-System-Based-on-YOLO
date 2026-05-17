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
        self.haar = None
        if not self.enabled:
            return
        # Try to use Ultralytics YOLO face model if available and path exists.
        try:
            from ultralytics import YOLO

            if Path(self.model_path).exists():
                try:
                    self.model = YOLO(self.model_path)
                    print("[Privacy] Face blur enabled (YOLO model).")
                except Exception as exc:
                    print(f"[Privacy] Failed to load YOLO face model: {exc}")
                    self.model = None
            else:
                print(f"[Privacy] Face model not found: {self.model_path}. Trying Haar cascade fallback.")
                self.model = None
        except Exception:
            # ultralytics not installed or failed to import
            self.model = None

        # If YOLO model not available, try OpenCV Haar cascade as a fallback so feature still works.
        if self.model is None:
            try:
                haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if Path(haar_path).exists():
                    self.haar = cv2.CascadeClassifier(haar_path)
                    if self.haar.empty():
                        self.haar = None
                    else:
                        print("[Privacy] Face blur enabled (Haar cascade fallback).")
                else:
                    print(f"[Privacy] Haar cascade not found at {haar_path}.")
            except Exception as exc:
                print(f"[Privacy] Haar cascade init failed: {exc}")

        if self.model is None and self.haar is None:
            print("[Privacy] No face detector available. Face blur disabled.")
            self.enabled = False

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
        if not self.enabled or frame is None:
            return frame

        output = frame.copy()
        frame_h, frame_w = output.shape[:2]

        # If YOLO model is available, prefer it
        coords = []
        if self.model is not None:
            try:
                results = self.model.predict(source=frame, conf=self.conf, verbose=False, device="cpu")
            except Exception as exc:
                print(f"[Privacy] Face blur predict failed: {exc}")
                results = None

            if results:
                try:
                    boxes = results[0].boxes
                    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                        coords = boxes.xyxy.cpu().numpy().astype(int)
                except Exception:
                    coords = []

        # If no YOLO detections and Haar is available, run Haar cascade
        if (not len(coords)) and self.haar is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    # convert (x, y, w, h) -> (x1, y1, x2, y2)
                    coords = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces], dtype=int)
            except Exception as exc:
                print(f"[Privacy] Haar face detection failed: {exc}")

        if coords is None or len(coords) == 0:
            return output

        for x1, y1, x2, y2 in coords:
            x1 = int(max(0, min(int(x1), frame_w - 1)))
            y1 = int(max(0, min(int(y1), frame_h - 1)))
            x2 = int(max(0, min(int(x2), frame_w)))
            y2 = int(max(0, min(int(y2), frame_h)))

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
