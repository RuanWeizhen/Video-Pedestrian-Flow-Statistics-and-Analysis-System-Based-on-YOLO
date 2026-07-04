from __future__ import annotations

from collections import deque
from math import hypot
from typing import Deque, Dict, List

from utils.geometry import point_in_polygon
from utils.io_utils import make_event


# ============================================================================
# ZoneCounter — 区域计数算法
# ============================================================================
# 功能：统计行人进入/离开指定多边形区域（如店铺、走廊区域）的次数。
#
# 算法核心原理（射线法判断点在多边形内外）：
#
#   从点向任意方向（通常水平向右）发出一条射线，统计该射线与多边形
#   边界的交点数量。若交点数为奇数→点在多边形内部，偶数→外部。
#
# 计数逻辑：
#   1. 对每个跟踪目标，取其计数锚点（默认 bottom_center）
#   2. 用射线法判断锚点是否在多边形区域内
#   3. 记录每个目标的"上一帧是否在区域内"，当状态改变时判定 enter/leave：
#      - 外→内 = "enter"（进入区域）
#      - 内→外 = "leave"（离开区域）
#   4. 首次观察到目标时（prev_inside 为 None），只记录状态，不触发事件
#
# 防抖/去重策略（同 LineCounter）：
#   a) 冷却帧（cooldown_frames）
#      — 同 track 连续两次事件需间隔至少 N 帧
#   b) 时空邻近去重（dedup_window_frames + dedup_distance_px）
#      — 短时间、同类型、相近位置的异 track 事件视为 ID 切换重复
#
# 过期清理：
#   长时间未出现的 track（超过 stale_after_frames 帧）会从状态字典中清除
#   避免内存无限增长。
#
# 关键参数：
#   polygon              — 定义区域的多边形顶点列表 [(x1,y1), (x2,y2), ...]
#   stale_after_frames    — track 失联多少帧后清理（默认 45）
#   cooldown_frames       — 同 track 事件冷却帧数
#   dedup_window_frames   — 时空去重时间窗口
#   dedup_distance_px     — 时空去重空间距离
#   anchor_point          — 计数锚点类型
# ============================================================================

class ZoneCounter:
    def __init__(self, cfg: Dict):
        # ── 上游调用者：ZoneCounterManager.__init__() ──
        # ── 输入：cfg = 区域计数配置字典，含 name/polygon/防抖参数 ──
        self.name = str(cfg["name"])                 # 区域名称（如 "商店A"）
        self.enabled = bool(cfg.get("enabled", True)) # 是否启用此区域

        polygon = cfg.get("polygon", [])              # 必填：定义区域的多边形顶点
        if len(polygon) < 3:
            raise ValueError(f"ZoneCounter '{self.name}' polygon must contain at least 3 points.")
        self.polygon = [tuple(pt) for pt in polygon]  # 转为 (x,y) 元组列表

        self.stale_after_frames = int(cfg.get("stale_after_frames", 45))
        # ⬆ 过期时间：track 失联超过此帧数后从内存中清除

        # center / bottom_center
        self.anchor_point = str(cfg.get("anchor_point", "bottom_center")).lower().strip()

        # Anti-jitter / anti-duplicate parameters
        self.cooldown_frames = int(cfg.get("cooldown_frames", 15))
        # ⬆ 同 track 事件冷却帧数：二次触发需间隔至少 N 帧
        self.dedup_window_frames = int(cfg.get("dedup_window_frames", 12))
        # ⬆ 时空去重时间窗口
        self.dedup_distance_px = float(cfg.get("dedup_distance_px", 50.0))
        # ⬆ 时空去重空间距离（像素）

        # Per-track state — 每个跟踪目标的状态映射
        self.current_ids = set()                       # 当前在区域内的 track_id 集合
        self.last_seen_frame: Dict[int, int] = {}      # track_id → 最后出现帧号
        self.track_inside: Dict[int, bool] = {}        # track_id → 是否在区域内
        self.last_event_frame: Dict[int, int] = {}     # track_id → 上次触发事件帧号

        # Recent events for suppressing ID-switch duplicates
        self.recent_events: Deque[Dict] = deque(maxlen=128)

        self.enter_count = 0                            # 累计进入次数
        self.leave_count = 0                            # 累计离开次数

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
        """
        区域计数主循环 ════════════════════════════════════════
        上游调用者：ZoneCounterManager.update()
        下游消费者：event_logger/DatabaseManager/event_emitted 信号
        输入：tracks = List[TrackResult]（已过滤静态目标的 counting_tracks）
              frame_idx = int 当前帧号
        输出：List[Dict] = 本帧新产生的区域进出事件列表
        ─────────────────────────────────────────────────────────
        算法流程（每帧对每个 track）：
          1) 取锚点 → 射线法判断是否在区域内 (point_in_polygon)
          2) 首次出现 → 只记录状态，不触发事件
          3) 状态改变 → 判定 enter/leave → 双重防抖 → 正式计数
          4) 清理过期 track
        """
        events: List[Dict] = []

        for tr in tracks:
            track_id = tr.track_id
            point = tr.count_point(self.anchor_point)          # 取计数锚点
            # ── 几何核心：射线法判断点是否在多边形内 ──
            inside = point_in_polygon(point, self.polygon)
            # ⬆ 从点向右发水平射线，统计与多边形交点：
            #    奇数交点 → True（内部），偶数交点 → False（外部）

            self.last_seen_frame[track_id] = frame_idx         # 更新最后出现帧号
            prev_inside = self.track_inside.get(track_id)      # 上一帧是否在区域内

            # First stable observation: record state only, do not trigger event yet
            if prev_inside is None:
                self.track_inside[track_id] = inside           # 首次 → 只记录状态
                if inside:
                    self.current_ids.add(track_id)             # 若在内部，加入当前集合
                continue                                       # 不触发事件

            # ⚡ State changed: possible enter / leave — 状态改变！
            if inside != prev_inside:
                event_name = "enter" if inside else "leave"    # 外→内=enter，内→外=leave

                # 1) Cooldown for the same track to suppress repeated boundary jitter
                last_event_frame = self.last_event_frame.get(track_id, -10**9)
                if frame_idx - last_event_frame < self.cooldown_frames:
                    self.track_inside[track_id] = inside       # 冷却期 → 只更新状态
                    if inside:
                        self.current_ids.add(track_id)
                    else:
                        self.current_ids.discard(track_id)
                    continue                                   # 抑制事件

                # 2) Suppress short-term duplicate event caused by ID switch
                if self._should_suppress_duplicate(point, event_name, frame_idx, track_id):
                    self.track_inside[track_id] = inside       # 时空去重 → 只更新状态
                    continue                                   # 抑制事件

                # ✅ Official zone event — 正式区域事件
                if inside:
                    self.enter_count += 1                      # 进入计数 +1
                    self.current_ids.add(track_id)             # 加入当前集合
                else:
                    self.leave_count += 1                      # 离开计数 +1
                    self.current_ids.discard(track_id)         # 从当前集合移除

                self.last_event_frame[track_id] = frame_idx    # 更新事件帧号
                self.recent_events.append(                     # 记录到去重缓冲区
                    {
                        "frame_idx": frame_idx,
                        "track_id": track_id,
                        "event_name": event_name,              # "enter" 或 "leave"
                        "point": point,
                    }
                )

                events.append(                                 # 生成标准事件
                    make_event(
                        frame_idx=frame_idx,
                        event_type=f"zone_{event_name}",        # "zone_enter" 或 "zone_leave"
                        target=self.name,
                        track_id=track_id,
                        value=event_name,
                    )
                )

            else:
                # No state change, just keep current occupancy consistent
                # 无状态变化，同步确保 current_ids 准确
                if inside:
                    self.current_ids.add(track_id)
                else:
                    self.current_ids.discard(track_id)

            self.track_inside[track_id] = inside               # 更新当前区域状态

        self._purge_stale(frame_idx)                           # 清理过期 track
        return events

    def _purge_stale(self, frame_idx: int) -> None:
        # 清理长时间未出现的跟踪目标：
        # 若某 track 的最后出现帧距当前帧超过 stale_after_frames 帧，
        # 则从所有状态字典中移除，释放内存
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
    """管理多个 ZoneCounter 实例，统一调度计数更新和汇总。"""
    def __init__(self, zone_cfg_list: List[Dict]):
        self.zones = [ZoneCounter(cfg) for cfg in zone_cfg_list if cfg.get("enabled", True)]

    def update(self, tracks, frame_idx: int) -> List[Dict]:
        events: List[Dict] = []
        for zone in self.zones:
            events.extend(zone.update(tracks, frame_idx))
        return events

    def summary(self) -> List[Dict]:
        return [zone.summary() for zone in self.zones]
