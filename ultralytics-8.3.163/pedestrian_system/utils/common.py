from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def count_point(self, anchor_point: str = "center") -> Tuple[float, float]:
        anchor_point = anchor_point.lower().strip()
        if anchor_point == "center":
            return self.center
        if anchor_point == "bottom_center":
            return self.bottom_center
        raise ValueError(f"Unsupported anchor_point: {anchor_point}")

    def to_ltwh(self) -> Tuple[float, float, float, float]:
        left = self.x1
        top = self.y1
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return left, top, width, height


@dataclass
class TrackResult:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def count_point(self, anchor_point: str = "center") -> Tuple[float, float]:
        anchor_point = anchor_point.lower().strip()
        if anchor_point == "center":
            return self.center
        if anchor_point == "bottom_center":
            return self.bottom_center
        raise ValueError(f"Unsupported anchor_point: {anchor_point}")

    def to_ltwh(self) -> Tuple[float, float, float, float]:
        left = self.x1
        top = self.y1
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return left, top, width, height
