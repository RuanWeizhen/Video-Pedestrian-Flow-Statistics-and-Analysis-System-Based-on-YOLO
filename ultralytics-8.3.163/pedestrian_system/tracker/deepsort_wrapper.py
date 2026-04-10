from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.common import Detection, TrackResult


class DeepSortTracker:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

        self.fast_reid = bool(cfg.get("fast_reid", True))
        self.hist_bins_h = int(cfg.get("hist_bins_h", 8))
        self.hist_bins_s = int(cfg.get("hist_bins_s", 4))
        self.hist_bins_v = int(cfg.get("hist_bins_v", 4))

        embedder = cfg.get("embedder", None)
        if self.fast_reid:
            embedder = None
        elif embedder in ("none", "None", "", None):
            embedder = None

        self.tracker = DeepSort(
            max_age=int(cfg.get("max_age", 20)),
            n_init=int(cfg.get("n_init", 2)),
            nn_budget=int(cfg.get("nn_budget", 30)),
            max_iou_distance=float(cfg.get("max_iou_distance", 0.9)),
            max_cosine_distance=float(cfg.get("max_cosine_distance", 0.35)),
            embedder=embedder,
            embedder_gpu=bool(cfg.get("embedder_gpu", False)),
            half=bool(cfg.get("half", False)),
            bgr=bool(cfg.get("bgr", True)),
            polygon=bool(cfg.get("polygon", False)),
        )

    def _fast_embed_one(self, frame, det: Detection) -> np.ndarray:
        """用 HSV 直方图代替 CNN ReID，极快。"""
        h, w = frame.shape[:2]

        x1 = max(0, min(int(det.x1), w - 1))
        y1 = max(0, min(int(det.y1), h - 1))
        x2 = max(0, min(int(det.x2), w))
        y2 = max(0, min(int(det.y2), h))

        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return np.zeros(
                (self.hist_bins_h * self.hist_bins_s * self.hist_bins_v,),
                dtype=np.float32,
            )

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(
                (self.hist_bins_h * self.hist_bins_s * self.hist_bins_v,),
                dtype=np.float32,
            )

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.hist_bins_h, self.hist_bins_s, self.hist_bins_v],
            [0, 180, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def _fast_embeds(self, frame, detections: List[Detection]) -> List[np.ndarray]:
        return [self._fast_embed_one(frame, det) for det in detections]

    def update(self, frame, detections: List[Detection]) -> List[TrackResult]:
        bbs = []
        for det in detections:
            left, top, width, height = det.to_ltwh()
            bbs.append(([left, top, width, height], det.conf, det.class_name))

        if self.fast_reid:
            embeds = self._fast_embeds(frame, detections)
            tracks = self.tracker.update_tracks(bbs, embeds=embeds, frame=None)
        else:
            tracks = self.tracker.update_tracks(bbs, frame=frame)

        outputs: List[TrackResult] = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            if tr.time_since_update > 1:
                continue

            l, t, r, b = tr.to_ltrb()
            outputs.append(
                TrackResult(
                    track_id=int(tr.track_id),
                    x1=float(l),
                    y1=float(t),
                    x2=float(r),
                    y2=float(b),
                    conf=1.0,
                    class_id=0,
                    class_name="person",
                )
            )
        return outputs

    def summary(self) -> Dict:
        out = dict(self.cfg)
        out["fast_reid"] = self.fast_reid
        return out
