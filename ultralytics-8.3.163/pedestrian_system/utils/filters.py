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
    
    算法：射线法（与 geometry.py 中的版本略有不同）
    增加边界检查——若点恰好落在多边形边上（共线检测），也返回 True。
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
        # 边界容差：检查点是否近似落在边上（共线且在线段范围内）
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
    
    只保留计数锚点位于 ROI（Region of Interest）多边形区域内的轨迹。
    用于限制计数范围——例如只统计门前区域的通过人数，忽略画面远处无关行人。
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
    
    静态目标过滤器 — 检测并标记长时间几乎不动的跟踪目标。
    
    核心原理：
      1. 维护每个 track 的历史锚点队列（history_size 个最近位置）
      2. 当历史足够长（≥ min_static_frames），检查所有历史位置相对于最旧位置的偏移
      3. 若所有偏移都不超过 max_movement_px → 该 track 被标记为静态
      4. 静态 track 不会参与人流计数，避免模特/海报/虚警被误统计
      
    可选：hide_static_boxes=True 时，静态目标的检测框也不绘制。
    
    关键参数：
      history_size       — 保存的历史锚点数量（默认 40）
      min_static_frames  — 最少需要多少帧数据才能判断（默认 30）
      max_movement_px    — 最大允许移动像素数（默认 18）
      forget_after_frames — 多久未出现后清除该 track 状态（默认 90）
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
        
        更新每个 track 的历史锚点，返回被判定为静态的 track_id 集合。
        
        判断逻辑：
          取历史队列中最旧点作为参考，若所有历史点到参考点的欧氏距离平方
          都 ≤ max_movement_px²，则认为该 track 静止。
          使用平方距离避免开方运算，并支持早停（一旦发现超过阈值立即跳出）。
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
            # 等价于 max(hypot(...)) <= threshold，但用平方距离避免 sqrt，并支持早停
            threshold_sq = self.max_movement_px * self.max_movement_px
            is_static = True
            for x, y in hist:
                dx = x - x0
                dy = y - y0
                if (dx * dx + dy * dy) > threshold_sq:
                    is_static = False
                    break

            if is_static:
                static_ids.add(track_id)

        self._purge_stale(frame_idx)
        return static_ids

    def _purge_stale(self, frame_idx: int) -> None:
        """清理长时间未出现的跟踪目标，避免内存泄漏"""
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
