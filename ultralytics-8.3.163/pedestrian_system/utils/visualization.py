from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from pathlib import Path


def draw_tracks(frame, tracks, show_conf: bool = True, box_thickness: int = 2):
    for tr in tracks:
        x1, y1, x2, y2 = map(int, [tr.x1, tr.y1, tr.x2, tr.y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        label = f"ID {tr.track_id}"
        if show_conf:
            label += f" {tr.class_name}"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_track_history(frame, track_history):
    for _, points in track_history.items():
        if len(points) < 2:
            continue
        pts = list(points)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (255, 180, 0), 2, cv2.LINE_AA)
    return frame


def draw_line_counter(frame, line_counter):
    p1 = tuple(map(int, line_counter.p1))
    p2 = tuple(map(int, line_counter.p2))
    cv2.line(frame, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)

    text = f"{line_counter.name}: {line_counter.counts}"
    x = min(p1[0], p2[0])
    y = max(30, min(p1[1], p2[1]) - 10)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_zone_counters(frame, zone_manager):
    for zone in zone_manager.zones:
        pts = [tuple(map(int, p)) for p in zone.polygon]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (255, 0, 255), 2, cv2.LINE_AA)

        anchor = pts[0]
        text = f"{zone.name} | in:{zone.enter_count} out:{zone.leave_count} now:{zone.current_count()}"
        cv2.putText(
            frame,
            text,
            (anchor[0], max(25, anchor[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_counters_panel(frame, line_counter, zone_manager):
    x, y = 15, 25
    line_height = 28

    texts = []
    if line_counter is not None:
        texts.append(f"Line {line_counter.name}: {line_counter.counts}")

    for zone in zone_manager.zones:
        texts.append(
            f"Zone {zone.name}: enter={zone.enter_count} leave={zone.leave_count} current={zone.current_count()}"
        )

    if not texts:
        return frame

    panel_width = 700
    panel_height = 15 + len(texts) * line_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    for idx, text in enumerate(texts):
        cv2.putText(
            frame,
            text,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def save_heatmap(points: List[tuple], save_path: str | Path, width: int, height: int, sigma: int = 25) -> None:
    """
    生成并保存热力图（单通道密度 -> 伪彩色）

    points: list of (x,y) 屏幕坐标，原点在左上
    save_path: 输出文件路径
    width, height: 画面尺寸
    sigma: 高斯模糊标准差，用于扩散点密度
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    heat = np.zeros((height, width), dtype=np.float32)
    for p in points:
        try:
            x, y = int(p[0]), int(p[1])
        except Exception:
            continue
        if 0 <= x < width and 0 <= y < height:
            heat[y, x] += 1.0

    if np.max(heat) <= 0:
        # 没有点，保存一张黑图
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(save_path), blank)
        return

    # 扩散并归一化
    ksize = (0, 0)
    heat_blur = cv2.GaussianBlur(heat, ksize, sigmaX=sigma, sigmaY=sigma)
    norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 伪彩色并保存
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(save_path), heatmap)
