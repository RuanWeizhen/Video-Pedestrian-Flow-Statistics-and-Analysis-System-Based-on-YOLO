from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# Detection — 检测结果数据结构
# ============================================================================
# 表示 YOLO 检测器输出的单个检测框。
# 提供 center / bottom_center 两种计数锚点，以及 DeepSORT 所需的 ltwh 格式转换。
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
        """边界框几何中心 (cx, cy)"""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """边界框底部中心 (cx, y2)，贴近行人脚部位置，跨线计数更稳定"""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def count_point(self, anchor_point: str = "center") -> Tuple[float, float]:
        """根据锚点类型返回计数坐标"""
        anchor_point = anchor_point.lower().strip()
        if anchor_point == "center":
            return self.center
        if anchor_point == "bottom_center":
            return self.bottom_center
        raise ValueError(f"Unsupported anchor_point: {anchor_point}")

    def to_ltwh(self) -> Tuple[float, float, float, float]:
        """转为 DeepSORT 要求的 (left, top, width, height) 格式"""
        left = self.x1
        top = self.y1
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return left, top, width, height


# ============================================================================
# TrackResult — 跟踪结果数据结构
# ============================================================================
# 表示 DeepSORT 跟踪器输出的单条轨迹（已确认、跨帧关联的检测框）。
# 与 Detection 结构类似，但额外携带跨帧一致的 track_id。
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
        """边界框几何中心 (cx, cy)"""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """边界框底部中心 (cx, y2)"""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def count_point(self, anchor_point: str = "center") -> Tuple[float, float]:
        """根据锚点类型返回计数坐标"""
        anchor_point = anchor_point.lower().strip()
        if anchor_point == "center":
            return self.center
        if anchor_point == "bottom_center":
            return self.bottom_center
        raise ValueError(f"Unsupported anchor_point: {anchor_point}")

    def to_ltwh(self) -> Tuple[float, float, float, float]:
        """转为 (left, top, width, height) 格式"""
        left = self.x1
        top = self.y1
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return left, top, width, height
