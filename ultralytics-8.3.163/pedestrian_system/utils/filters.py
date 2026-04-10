from __future__ import annotations

from collections import defaultdict, deque
from math import hypot
from typing import Deque, Dict, Iterable, List, Sequence, Set, Tuple


Point = Tuple[float, float]
Polygon = Sequence[Tuple[float, float]]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Ray casting point-in-polygon test.
    Returns True when the point is inside the polygon or lies on its edge.
    """
    if not polygon:
        return True

    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Treat points on polygon edges as inside
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-6:
            if min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6 and \
               min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6:
                return True

        intersects = (y1 > y) != (y2 > y)
        if intersects:
            xinters = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
            if x <= xinters:
                inside = not inside

    return inside


def filter_tracks_by_roi(
    tracks: Iterable,
    polygon: Polygon,
    anchor_point: str = "bottom_center",
) -> List:
    """
    Keep only tracks whose counting anchor point lies inside the ROI polygon.
    """
    if not polygon:
        return list(tracks)

    anchor_point = anchor_point.lower().strip()
    kept_tracks = []

    for tr in tracks:
        point = tr.count_point(anchor_point)
        if point_in_polygon(point, polygon):
            kept_tracks.append(tr)

    return kept_tracks


class StaticTrackFilter:
    """
    Mark long-time almost-stationary tracks as static targets.

    Typical usage:
        static_filter = StaticTrackFilter(cfg["static_filter"])
        static_ids = static_filter.update(roi_tracks, frame_idx)
        counting_tracks = [tr for tr in roi_tracks if tr.track_id not in static_ids]

    This is useful for suppressing:
    - mannequin / dummy detections in shop windows
    - posters or human-shaped standees
    - persistent false-positive person detections that barely move
    """

    def __init__(self, cfg: Dict | None = None):
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", True))
        self.anchor_point = str(cfg.get("anchor_point", "bottom_center")).lower().strip()

        # Number of recent anchor points stored per track
        self.history_size = int(cfg.get("history_size", 40))

        # A track must have at least this many stored points before it can be judged static
        self.min_static_frames = int(cfg.get("min_static_frames", 30))

        # Maximum movement radius (pixels) from the oldest point in history
        self.max_movement_px = float(cfg.get("max_movement_px", 18.0))

        # Forget tracks that have disappeared for too long
        self.forget_after_frames = int(cfg.get("forget_after_frames", 90))

        self.history: Dict[int, Deque[Point]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.last_seen_frame: Dict[int, int] = {}

    def update(self, tracks: Iterable, frame_idx: int) -> Set[int]:
        """
        Update internal history using current tracks and return static track IDs.
        """
        if not self.enabled:
            return set()

        static_ids: Set[int] = set()

        for tr in tracks:
            track_id = tr.track_id
            point = tr.count_point(self.anchor_point)

            self.history[track_id].append(point)
            self.last_seen_frame[track_id] = frame_idx

            hist = self.history[track_id]
            if len(hist) < self.min_static_frames:
                continue

            x0, y0 = hist[0]
            max_move = max(hypot(x - x0, y - y0) for x, y in hist)

            if max_move <= self.max_movement_px:
                static_ids.add(track_id)

        self._purge_stale(frame_idx)
        return static_ids

    def _purge_stale(self, frame_idx: int) -> None:
        stale_ids = [
            track_id
            for track_id, last_seen in self.last_seen_frame.items()
            if frame_idx - last_seen > self.forget_after_frames
        ]

        for track_id in stale_ids:
            self.last_seen_frame.pop(track_id, None)
            self.history.pop(track_id, None)

    def reset(self) -> None:
        self.history.clear()
        self.last_seen_frame.clear()

    def summary(self) -> Dict:
        return {
            "enabled": self.enabled,
            "anchor_point": self.anchor_point,
            "history_size": self.history_size,
            "min_static_frames": self.min_static_frames,
            "max_movement_px": self.max_movement_px,
            "forget_after_frames": self.forget_after_frames,
            "tracked_objects": len(self.last_seen_frame),
        }
