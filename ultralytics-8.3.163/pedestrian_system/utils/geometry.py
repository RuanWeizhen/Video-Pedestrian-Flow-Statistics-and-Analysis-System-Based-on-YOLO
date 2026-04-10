from __future__ import annotations

from typing import Iterable, Sequence, Tuple


Point = Tuple[float, float]


def point_line_side(point: Point, p1: Point, p2: Point, eps: float = 1e-6) -> int:
    """Return -1 / 0 / +1 according to which side of the directed line the point lies on."""
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    if abs(cross) <= eps:
        return 0
    return 1 if cross > 0 else -1


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting algorithm."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside
