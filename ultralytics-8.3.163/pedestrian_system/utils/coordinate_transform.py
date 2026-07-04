from __future__ import annotations

from typing import Iterable, Sequence


def _coerce_size(width, height) -> tuple[int, int]:
    try:
        w = int(width or 0)
    except Exception:
        w = 0
    try:
        h = int(height or 0)
    except Exception:
        h = 0
    return w, h


def get_display_transform(frame_w, frame_h, display_w, display_h) -> dict:
    """计算从帧坐标系到显示坐标系的变换参数。
    
    使用等比例缩放（letterbox）：保持宽高比，短边居中放置。
    返回字典包含：
      - scale: 缩放比例因子
      - scaled_w / scaled_h: 缩放后尺寸
      - offset_x / offset_y: 居中偏移量（黑边宽度）
    """
    frame_w, frame_h = _coerce_size(frame_w, frame_h)
    display_w, display_h = _coerce_size(display_w, display_h)

    if frame_w <= 0 or frame_h <= 0 or display_w <= 0 or display_h <= 0:
        return {
            "scale": 1.0,
            "scaled_w": max(0, display_w),
            "scaled_h": max(0, display_h),
            "offset_x": 0,
            "offset_y": 0,
        }

    scale = min(display_w / float(frame_w), display_h / float(frame_h))
    scaled_w = max(1, int(round(frame_w * scale)))
    scaled_h = max(1, int(round(frame_h * scale)))
    offset_x = int(round((display_w - scaled_w) / 2.0))
    offset_y = int(round((display_h - scaled_h) / 2.0))
    return {
        "scale": float(scale),
        "scaled_w": scaled_w,
        "scaled_h": scaled_h,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def frame_to_display_point(x, y, frame_w, frame_h, display_w, display_h):
    """将帧坐标系的点映射到显示坐标系"""
    frame_w, frame_h = _coerce_size(frame_w, frame_h)
    display_w, display_h = _coerce_size(display_w, display_h)
    if frame_w <= 0 or frame_h <= 0 or display_w <= 0 or display_h <= 0:
        return None

    transform = get_display_transform(frame_w, frame_h, display_w, display_h)
    display_x = int(round(float(x) * transform["scale"] + transform["offset_x"]))
    display_y = int(round(float(y) * transform["scale"] + transform["offset_y"]))
    return display_x, display_y


def display_to_frame_point(display_x, display_y, frame_w, frame_h, display_w, display_h):
    """将显示坐标系的点映射回帧坐标系（逆变换）"""
    frame_w, frame_h = _coerce_size(frame_w, frame_h)
    display_w, display_h = _coerce_size(display_w, display_h)
    if frame_w <= 0 or frame_h <= 0 or display_w <= 0 or display_h <= 0:
        return None

    transform = get_display_transform(frame_w, frame_h, display_w, display_h)
    left = transform["offset_x"]
    top = transform["offset_y"]
    right = left + transform["scaled_w"]
    bottom = top + transform["scaled_h"]
    if display_x < left or display_x >= right or display_y < top or display_y >= bottom:
        return None

    frame_x = (float(display_x) - left) / transform["scale"]
    frame_y = (float(display_y) - top) / transform["scale"]
    return int(round(frame_x)), int(round(frame_y))


def frame_points_to_display_points(points: Iterable[Sequence[float]], frame_w, frame_h, display_w, display_h):
    """批量帧坐标 → 显示坐标"""
    converted = []
    for point in points or []:
        if point is None:
            continue
        try:
            x, y = point
        except Exception:
            continue
        mapped = frame_to_display_point(x, y, frame_w, frame_h, display_w, display_h)
        if mapped is not None:
            converted.append(mapped)
    return converted


def display_points_to_frame_points(points: Iterable[Sequence[float]], frame_w, frame_h, display_w, display_h):
    """批量显示坐标 → 帧坐标"""
    converted = []
    for point in points or []:
        if point is None:
            continue
        try:
            x, y = point
        except Exception:
            continue
        mapped = display_to_frame_point(x, y, frame_w, frame_h, display_w, display_h)
        if mapped is not None:
            converted.append(mapped)
    return converted


def frame_points_to_frame_points(points: Iterable[Sequence[float]], source_w, source_h, target_w, target_h):
    """将点从一个帧尺寸线性映射到另一个帧尺寸。
    
    用于将人工标注的计数线/ROI 多边形从原始视频尺寸映射到处理分辨率。
    原理：X 坐标按宽度比例缩放，Y 坐标按高度比例缩放（非等比例）。
    """
    source_w, source_h = _coerce_size(source_w, source_h)
    target_w, target_h = _coerce_size(target_w, target_h)
    if source_w <= 0 or source_h <= 0 or target_w <= 0 or target_h <= 0:
        return [tuple(map(int, point)) for point in points or [] if point is not None]

    scale_x = target_w / float(source_w)
    scale_y = target_h / float(source_h)
    converted = []
    for point in points or []:
        if point is None:
            continue
        try:
            x, y = point
        except Exception:
            continue
        converted.append((int(round(float(x) * scale_x)), int(round(float(y) * scale_y))))
    return converted