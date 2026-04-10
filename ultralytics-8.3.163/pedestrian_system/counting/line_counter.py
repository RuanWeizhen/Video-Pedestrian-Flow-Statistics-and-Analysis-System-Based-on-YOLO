from __future__ import annotations

from collections import deque
from math import hypot
from typing import Deque, Dict, List

from utils.geometry import point_line_side
from utils.io_utils import make_event


class LineCounter:
    def __init__(self, cfg: Dict):
        self.name = str(cfg.get("name", "line"))

        points = cfg["points"]
        if len(points) != 2:
            raise ValueError("LineCounter cfg['points'] must contain exactly 2 points.")
        self.p1 = tuple(points[0])
        self.p2 = tuple(points[1])

        labels = cfg.get("labels", {})
        self.neg_to_pos_name = str(labels.get("neg_to_pos", "neg_to_pos"))
        self.pos_to_neg_name = str(labels.get("pos_to_neg", "pos_to_neg"))

        self.min_distance_to_line = float(cfg.get("min_distance_to_line", 2.0))

        # center / bottom_center
        self.anchor_point = str(cfg.get("anchor_point", "bottom_center")).lower().strip()

        # Anti-jitter / anti-duplicate parameters
        self.cooldown_frames = int(cfg.get("cooldown_frames", 18))
        self.dedup_window_frames = int(cfg.get("dedup_window_frames", 10))
        self.dedup_distance_px = float(cfg.get("dedup_distance_px", 45.0))

        # Per-track state
        self.track_last_side: Dict[int, int] = {}
        self.track_last_count_frame: Dict[int, int] = {}

        # ✅ 同一 track 只计一次
        self.counted_track_ids = set()

        # Recent crossing events for suppressing ID-switch duplicates
        self.recent_events: Deque[Dict] = deque(maxlen=128)

        self.counts = {
            self.neg_to_pos_name: 0,
            self.pos_to_neg_name: 0,
        }

    def _should_suppress_duplicate(
        self,
        point,
        direction: str,
        frame_idx: int,
        track_id: int,
    ) -> bool:
        for evt in reversed(self.recent_events):
            if frame_idx - evt["frame_idx"] > self.dedup_window_frames:
                break

            if evt["direction"] != direction:
                continue

            if evt["track_id"] == track_id:
                continue

            dist = hypot(point[0] - evt["point"][0], point[1] - evt["point"][1])
            if dist <= self.dedup_distance_px:
                return True

        return False

    def update(self, tracks, frame_idx: int) -> List[Dict]:
        events: List[Dict] = []

        for tr in tracks:
            track_id = tr.track_id

            # ✅ 同一 track 永久只计一次
            if track_id in self.counted_track_ids:
                continue

            point = tr.count_point(self.anchor_point)
            side = point_line_side(point, self.p1, self.p2, eps=self.min_distance_to_line)

            # Near the line: do not count yet, and keep previous stable side.
            if side == 0:
                continue

            prev_side = self.track_last_side.get(track_id)

            # First stable observation of this track
            if prev_side is None:
                self.track_last_side[track_id] = side
                continue

            # Crossing detected
            if prev_side != side:
                if prev_side < 0 and side > 0:
                    direction = self.neg_to_pos_name
                else:
                    direction = self.pos_to_neg_name

                # 1) Cooldown for the same track to suppress repeated oscillation
                last_count_frame = self.track_last_count_frame.get(track_id, -10**9)
                if frame_idx - last_count_frame < self.cooldown_frames:
                    self.track_last_side[track_id] = side
                    continue

                # 2) Short-term, nearby, same-direction duplicate suppression
                if self._should_suppress_duplicate(point, direction, frame_idx, track_id):
                    self.track_last_side[track_id] = side
                    continue

                # Official count
                self.counts[direction] += 1
                self.track_last_count_frame[track_id] = frame_idx
                self.counted_track_ids.add(track_id)

                self.recent_events.append(
                    {
                        "frame_idx": frame_idx,
                        "track_id": track_id,
                        "direction": direction,
                        "point": point,
                    }
                )

                events.append(
                    make_event(
                        frame_idx=frame_idx,
                        event_type="line_crossing",
                        target=self.name,
                        track_id=track_id,
                        value=direction,
                    )
                )

                # 🔥 debug 信息（新增）
                events[-1]["point"] = point
                events[-1]["direction"] = direction
                events[-1]["side"] = side

            self.track_last_side[track_id] = side

        return events

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "line_points": [list(self.p1), list(self.p2)],
            "anchor_point": self.anchor_point,
            "min_distance_to_line": self.min_distance_to_line,
            "cooldown_frames": self.cooldown_frames,
            "dedup_window_frames": self.dedup_window_frames,
            "dedup_distance_px": self.dedup_distance_px,
            "counts": self.counts,
        }
