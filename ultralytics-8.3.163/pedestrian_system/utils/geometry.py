from __future__ import annotations

from typing import Iterable, Sequence, Tuple


Point = Tuple[float, float]


def point_line_side(point: Point, p1: Point, p2: Point, eps: float = 1e-6) -> int:
    """Return -1 / 0 / +1 according to which side of the directed line the point lies on.
    
    算法：二维向量叉积（cross product）
    
      给定有向线段 p1→p2 和点 P，构造向量 (P - p1) 与 (p2 - p1)：
        cross = (P.x - p1.x) * (p2.y - p1.y) - (P.y - p1.y) * (p2.x - p1.x)
      
      几何意义：
        cross > 0  → P 在 p1→p2 的左侧（逆时针方向），返回 +1
        cross < 0  → P 在 p1→p2 的右侧（顺时针方向），返回 -1
        |cross| ≤ eps → P 近似在直线上，返回 0
    
    用于跨线计数：通过比较连续帧中同一点的侧变化来判断行人是否跨越了计数线。
    """
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    if abs(cross) <= eps:
        return 0
    return 1 if cross > 0 else -1


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting algorithm.
    
    算法：射线法（Ray Casting）
    
      从待测点向右发出一条水平射线，统计该射线与多边形边界的交点数：
        - 交点数为奇数 → 点在多边形内部
        - 交点数为偶数 → 点在多边形外部
    
    优化细节：
      - 使用 (yi > y) != (yj > y) 判断射线是否与边相交（排除端点重合情况）
      - 添加 1e-12 防止除零
      - 对退化多边形（< 3 个顶点）直接返回 False
    
    用于区域计数：判断目标锚点是否在指定的 ROI / Zone 多边形区域内。
    """
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
