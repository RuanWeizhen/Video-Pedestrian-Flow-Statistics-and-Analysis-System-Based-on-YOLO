from __future__ import annotations

from collections import deque
from math import hypot
from typing import Deque, Dict, List

from utils.geometry import point_in_polygon
from utils.io_utils import make_event


class ZoneCounter:
    def __init__(self, cfg: Dict):
        self.name = str(cfg["name"])
        self.enabled = bool(cfg.get("enabled", True))

        polygon = cfg.get("polygon", [])
        if len(polygon) < 3:
            raise ValueError(f"ZoneCounter '{self.name}' polygon must contain at least 3 points.")
        self.polygon = [tuple(pt) for pt in polygon]

        self.stale_after_frames = int(cfg.get("stale_after_frames", 45))

        # center / bottom_center
        self.anchor_point = str(cfg.get("anchor_point", "bottom_center")).lower().strip()

        # Anti-jitter / anti-duplicate parameters
        self.cooldown_frames = int(cfg.get("cooldown_frames", 15))
        self.dedup_window_frames = int(cfg.get("dedup_window_frames", 12))
        self.dedup_distance_px = float(cfg.get("dedup_distance_px", 50.0))

        # Per-track state
        self.current_ids = set()
        self.last_seen_frame: Dict[int, int] = {}
        self.track_inside: Dict[int, bool] = {}
        self.last_event_frame: Dict[int, int] = {}

        # Recent events for suppressing ID-switch duplicates
        self.recent_events: Deque[Dict] = deque(maxlen=128)

        self.enter_count = 0
        self.leave_count = 0

    def _should_suppress_duplicate(
        self,
        point,
        event_name: str,
        frame_idx: int,
        track_id: int,
    ) -> bool:
        """
        Suppress duplicate zone events caused by short-term ID switch.

        If another track triggered the same enter/leave event very recently,
        near the same position, it is likely the same real person with a new ID.
        """
        for evt in reversed(self.recent_events):
            if frame_idx - evt["frame_idx"] > self.dedup_window_frames:
                break

            if evt["event_name"] != event_name:
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
            point = tr.count_point(self.anchor_point)
            inside = point_in_polygon(point, self.polygon)

            self.last_seen_frame[track_id] = frame_idx
            prev_inside = self.track_inside.get(track_id)

            # First stable observation: record state only, do not trigger event yet
            if prev_inside is None:
                self.track_inside[track_id] = inside
                if inside:
                    self.current_ids.add(track_id)
                continue

            # State changed: possible enter / leave
            if inside != prev_inside:
                event_name = "enter" if inside else "leave"

                # 1) Cooldown for the same track to suppress repeated boundary jitter
                last_event_frame = self.last_event_frame.get(track_id, -10**9)
                if frame_idx - last_event_frame < self.cooldown_frames:
                    self.track_inside[track_id] = inside
                    if inside:
                        self.current_ids.add(track_id)
                    else:
                        self.current_ids.discard(track_id)
                    continue

                # 2) Suppress short-term duplicate event caused by ID switch
                if self._should_suppress_duplicate(point, event_name, frame_idx, track_id):
                    # Only update this track's internal state, do not count as a new event
                    self.track_inside[track_id] = inside
                    continue

                # Official zone event
                if inside:
                    self.enter_count += 1
                    self.current_ids.add(track_id)
                else:
                    self.leave_count += 1
                    self.current_ids.discard(track_id)

                self.last_event_frame[track_id] = frame_idx
                self.recent_events.append(
                    {
                        "frame_idx": frame_idx,
                        "track_id": track_id,
                        "event_name": event_name,
                        "point": point,
                    }
                )

                events.append(
                    make_event(
                        frame_idx=frame_idx,
                        event_type=f"zone_{event_name}",
                        target=self.name,
                        track_id=track_id,
                        value=event_name,
                    )
                )

            else:
                # No state change, just keep current occupancy consistent
                if inside:
                    self.current_ids.add(track_id)
                else:
                    self.current_ids.discard(track_id)

            self.track_inside[track_id] = inside

        self._purge_stale(frame_idx)
        return events

    def _purge_stale(self, frame_idx: int) -> None:
        stale_ids = []
        for track_id, last_seen in list(self.last_seen_frame.items()):
            if frame_idx - last_seen > self.stale_after_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self.current_ids.discard(track_id)
            self.last_seen_frame.pop(track_id, None)
            self.track_inside.pop(track_id, None)
            self.last_event_frame.pop(track_id, None)

    def current_count(self) -> int:
        return len(self.current_ids)

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "anchor_point": self.anchor_point,
            "stale_after_frames": self.stale_after_frames,
            "cooldown_frames": self.cooldown_frames,
            "dedup_window_frames": self.dedup_window_frames,
            "dedup_distance_px": self.dedup_distance_px,
            "enter_count": self.enter_count,
            "leave_count": self.leave_count,
            "current_count": self.current_count(),
        }


class ZoneCounterManager:
    def __init__(self, zone_cfg_list: List[Dict]):
        self.zones = [ZoneCounter(cfg) for cfg in zone_cfg_list if cfg.get("enabled", True)]

    def update(self, tracks, frame_idx: int) -> List[Dict]:
        events: List[Dict] = []
        for zone in self.zones:
            events.extend(zone.update(tracks, frame_idx))
        return events

    def summary(self) -> List[Dict]:
        return [zone.summary() for zone in self.zones]
