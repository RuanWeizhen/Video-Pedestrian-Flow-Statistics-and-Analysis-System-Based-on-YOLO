from __future__ import annotations

from collections import deque
from math import hypot
from typing import Deque, Dict, List

from utils.geometry import point_line_side
from utils.io_utils import make_event


# ============================================================================
# LineCounter — 跨线计数算法
# ============================================================================
# 功能：通过检测行人是否跨越一条有向线段来统计双向人流量。
#
# 算法核心原理（向量叉积判断跨线方向）：
#
#   给定有向线段 p1→p2，对于任意点 P：
#     cross = (P.x - p1.x) * (p2.y - p1.y) - (P.y - p1.y) * (p2.x - p1.x)
#     cross > 0  → 点在线的左侧（正方向）
#     cross < 0  → 点在线的右侧（负方向）
#     cross ≈ 0  → 点几乎在线上（靠近线，不触发计数）
#
# 计数逻辑：
#   1. 对每个跟踪目标，取其计数锚点（默认 bottom_center，即脚底中心）
#   2. 用叉积判断锚点在线的哪一侧
#   3. 记录每个目标的"上一帧所在侧"，当侧发生变化时→判定为跨线
#   4. 负→正 = "up"（上行），正→负 = "down"（下行）
#
# 防抖/去重策略（三重保障）：
#   a) 最小距离阈值（min_distance_to_line）
#      - 点离线段太近时不判断方向，避免边界振动误触发
#   b) 冷却帧（cooldown_frames）
#      - 同一 track 连续两次跨线需间隔至少 N 帧，防止震荡重复计数
#   c) 时空邻近去重（dedup_window_frames + dedup_distance_px）
#      - 若短时间内同一方向、相近位置有其他 track 触发了事件
#      - 则视为 ID 切换导致的"同一个人，不同 ID"，抑制当前事件
#   d) 单次计数（counted_track_ids）
#      - 每个 track_id 永久只计一次，避免同一个人来回走动被重复计数
#
# 关键参数：
#   points               — 定义计数线的两个端点 [(x1,y1), (x2,y2)]
#   min_distance_to_line  — 最小判定距离（像素），低于此值不判断方向
#   cooldown_frames       — 同 track 重复计数冷却帧数
#   dedup_window_frames   — 时空去重的时间窗口（帧数）
#   dedup_distance_px     — 时空去重的空间距离阈值（像素）
#   anchor_point          — 计数锚点类型："center" 或 "bottom_center"
# ============================================================================

class LineCounter:
    def __init__(self, cfg: Dict):
        # ── 上游调用者：worker.py 初始化阶段 ──
        # ── 输入：cfg = 计数线配置字典，含 points/labels/防抖参数 ──
        self.name = str(cfg.get("name", "line"))   # 计数线名称标识

        points = cfg["points"]                      # 必填：定义计数线的两个端点
        if len(points) != 2:
            raise ValueError("LineCounter cfg['points'] must contain exactly 2 points.")
        self.p1 = tuple(points[0])                  # 端点 A（线段起点，决定方向判断基准）
        self.p2 = tuple(points[1])                  # 端点 B（线段终点，p1→p2 为有向线段）

        labels = cfg.get("labels", {})
        self.neg_to_pos_name = str(labels.get("neg_to_pos", "neg_to_pos"))  # 负→正方向名称
        self.pos_to_neg_name = str(labels.get("pos_to_neg", "pos_to_neg"))  # 正→负方向名称

        self.min_distance_to_line = float(cfg.get("min_distance_to_line", 2.0))
        # ⬆ 最小判定距离：点距线段 < 此值时不判断方向，防止边界振动误触发

        # center / bottom_center
        # 计数锚点：center=目标中心点，bottom_center=目标底部中心（更稳定，行人脚步附近）
        self.anchor_point = str(cfg.get("anchor_point", "bottom_center")).lower().strip()

        # Anti-jitter / anti-duplicate parameters
        # 防抖/去重参数
        self.cooldown_frames = int(cfg.get("cooldown_frames", 18))
        # ⬆ 同一 track 连续两次跨线需间隔至少 N 帧，防止震荡重复计数
        self.dedup_window_frames = int(cfg.get("dedup_window_frames", 10))
        # ⬆ 时空去重的时间窗口（帧数）
        self.dedup_distance_px = float(cfg.get("dedup_distance_px", 45.0))
        # ⬆ 时空去重的空间距离阈值（像素），用于识别 ID 切换导致的重复事件

        # Per-track state
        # 每个跟踪目标的状态：
        #   track_last_side        — 该目标上一帧在线的哪一侧 (-1/0/+1)
        #   track_last_count_frame — 该目标上一次触发计数的帧号
        self.track_last_side: Dict[int, int] = {}
        self.track_last_count_frame: Dict[int, int] = {}

        # ✅ 同一 track 只计一次
        # 记录已计数过的 track_id，防止同一行人来回走动被重复统计
        self.counted_track_ids = set()

        # Recent crossing events for suppressing ID-switch duplicates
        # 最近跨线事件缓冲区：用于短期内同方向、相近位置的去重
        self.recent_events: Deque[Dict] = deque(maxlen=128)

        self.counts = {
            self.neg_to_pos_name: 0,                 # 负→正 方向累计计数
            self.pos_to_neg_name: 0,                 # 正→负 方向累计计数
        }

    def _should_suppress_duplicate(
        self,
        point,
        direction: str,
        frame_idx: int,
        track_id: int,
    ) -> bool:
        """
        时空邻近去重检查 ════════════════════════════════════════
        在 dedup_window_frames 帧内，若已存在同一方向、相近位置
        （欧氏距离 ≤ dedup_distance_px）的其他 track 事件，
        则判定当前事件为 ID 切换引起的重复，应抑制。
        
        典型场景：行人 A（track_id=5）跨线后被遮挡，重新出现时
        被分配新 ID（track_id=12），若短时间内同方向、同位置再
        次触发跨线，则判定为"同一个人换了 ID"，抑制重复计数。
        """
        for evt in reversed(self.recent_events):        # 从最新事件往前查
            if frame_idx - evt["frame_idx"] > self.dedup_window_frames:
                break                                   # 超出时间窗口 → 不再检查

            if evt["direction"] != direction:
                continue                                # 方向不同 → 跳过

            if evt["track_id"] == track_id:
                continue                                # 同一个 track → 跳过

            dist = hypot(point[0] - evt["point"][0], point[1] - evt["point"][1])
            # ⬆ 欧氏距离：√((Δx)² + (Δy)²)
            if dist <= self.dedup_distance_px:
                return True                             # 在去重范围内 → 抑制

        return False                                    # 未发现重复 → 允许计数

    def update(self, tracks, frame_idx: int) -> List[Dict]:
        """
        跨线计数主循环 ════════════════════════════════════════
        上游调用者：worker.py 主循环（每帧调用一次）
        下游消费者：event_logger/DatabaseManager/event_emitted 信号
        输入：tracks = List[TrackResult]（已过滤静态目标的 counting_tracks）
              frame_idx = int 当前帧号
        输出：List[Dict] = 本帧新产生的跨线事件列表
        ─────────────────────────────────────────────────────────
        算法流程（每帧对每个 track）：
          1) 已计数 → 跳过
          2) 取锚点 → 叉积判断侧（-1/0/+1）
          3) 侧=0 → 跳过（太靠近线）
          4) 首次出现 → 只记录，不计数
          5) 侧变化 → 判定方向 → 三重防抖 → 正式计数
        """
        events: List[Dict] = []

        for tr in tracks:
            track_id = tr.track_id

            # ✅ 同一 track 永久只计一次
            if track_id in self.counted_track_ids:
                continue                     # 已计数 → 跳过

            point = tr.count_point(self.anchor_point)   # 取计数锚点坐标（默认脚底中心）
            # ── 几何核心：向量叉积判断点在直线的哪一侧 ──
            side = point_line_side(point, self.p1, self.p2, eps=self.min_distance_to_line)
            # ⬆ 返回 -1(右侧) / 0(线上近旁) / +1(左侧)
            #   cross = (P.x-p1.x)*(p2.y-p1.y) - (P.y-p1.y)*(p2.x-p1.x)

            # Near the line: do not count yet, and keep previous stable side.
            if side == 0:
                continue                     # 贴近线 → 不判断方向，保持上一帧侧状态

            prev_side = self.track_last_side.get(track_id)

            # First stable observation of this track
            if prev_side is None:
                self.track_last_side[track_id] = side  # 首次出现 → 记录侧状态
                continue                                # 不触发计数

            # ⚡ Crossing detected — 侧发生变化！
            if prev_side != side:
                if prev_side < 0 and side > 0:
                    direction = self.neg_to_pos_name    # 负→正 = 上行 (up)
                else:
                    direction = self.pos_to_neg_name    # 正→负 = 下行 (down)

                # 1) Cooldown for the same track to suppress repeated oscillation
                last_count_frame = self.track_last_count_frame.get(track_id, -10**9)
                if frame_idx - last_count_frame < self.cooldown_frames:
                    self.track_last_side[track_id] = side
                    continue                  # 冷却期内 → 抑制

                # 2) Short-term, nearby, same-direction duplicate suppression
                if self._should_suppress_duplicate(point, direction, frame_idx, track_id):
                    self.track_last_side[track_id] = side
                    continue                  # ID 切换重复 → 抑制

                # ✅ Official count — 正式计数
                self.counts[direction] += 1                     # up++ 或 down++
                self.track_last_count_frame[track_id] = frame_idx # 更新最后计数帧
                self.counted_track_ids.add(track_id)              # 永久标记已计数

                self.recent_events.append(                       # 记录到去重缓冲区
                    {
                        "frame_idx": frame_idx,                  # 事件帧号
                        "track_id": track_id,                    # 触发 track ID
                        "direction": direction,                  # 方向："up" 或 "down"
                        "point": point,                          # 锚点坐标
                    }
                )

                events.append(                                   # 生成标准事件
                    make_event(
                        frame_idx=frame_idx,
                        event_type="line_crossing",
                        target=self.name,
                        track_id=track_id,
                        value=direction,
                    )
                )

                # 🔥 debug 信息
                events[-1]["point"] = point
                events[-1]["direction"] = direction
                events[-1]["side"] = side

            self.track_last_side[track_id] = side                 # 更新当前帧侧状态

        return events                       # → 写入 event_logger / DatabaseManager

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
