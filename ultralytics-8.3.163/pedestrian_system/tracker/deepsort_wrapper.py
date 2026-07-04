from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.common import Detection, TrackResult


# ============================================================================
# DeepSortTracker — DeepSORT 多目标跟踪器
# ============================================================================
# 功能：基于 DeepSORT 算法实现跨帧的行人 ID 关联（Re-Identification）。
#
# DeepSORT 原理概述：
#   1. 卡尔曼滤波预测 — 根据上一帧的轨迹状态，预测当前帧每个目标的位置
#   2. 匈牙利匹配 — 将检测框与预测框进行二分图最优匹配，融合运动和外观信息：
#      - 运动匹配：使用马氏距离（Mahalanobis Distance）衡量检测与预测的位置偏差
#      - 外观匹配：使用余弦距离衡量 ReID 特征向量的相似度
#   3. 级联匹配 — 优先匹配"活跃"轨迹（最近更新过的），再匹配丢失较久的
#   4. 轨迹管理 — 新检测初始化为"暂定"轨迹，连续命中 n_init 帧后"确认"
#      - 确认轨迹连续丢失超过 max_age 帧则删除
#
# 快速 ReID 模式（fast_reid=True，默认启用）：
#   使用 HSV 颜色直方图作为外观特征，代替 CNN 特征提取网络
#   优点：CPU 友好、速度极快（无需 GPU）
#   代价：外观区分能力弱于深度学习特征，但行人场景下通常够用
#
# 关键参数：
#   max_age              — 轨迹最大丢失容忍帧数（默认 20）
#   n_init               — 新轨迹确认所需连续命中帧数（默认 2）
#   nn_budget            — 外观特征库最大容量（默认 30）
#   max_iou_distance     — IoU 匹配距离阈值（默认 0.9）
#   max_cosine_distance  — 余弦距离匹配阈值（默认 0.35）
# ============================================================================

class DeepSortTracker:
    def __init__(self, cfg: Dict):
        # ── 上游调用者：worker.py 初始化阶段 ──
        # ── 输入：cfg = 跟踪器配置字典，含 max_age/n_init/ReID 等参数 ──
        self.cfg = cfg

        # 快速 ReID 模式开关：True=HSV直方图(CPU快)，False=CNN特征提取(GPU慢但准)
        self.fast_reid = bool(cfg.get("fast_reid", True))
        # HSV 直方图分箱参数，决定外观特征向量维度 = H_bins × S_bins × V_bins = 128
        self.hist_bins_h = int(cfg.get("hist_bins_h", 8))  # Hue 色调通道 8 个桶
        self.hist_bins_s = int(cfg.get("hist_bins_s", 4))  # Saturation 饱和度 4 个桶
        self.hist_bins_v = int(cfg.get("hist_bins_v", 4))  # Value 亮度 4 个桶

        embedder = cfg.get("embedder", None)      # CNN ReID 外观模型路径（如 "osnet_x0_25"）
        if self.fast_reid:
            embedder = None                       # 快速模式不使用 CNN embedder
        elif embedder in ("none", "None", "", None):
            embedder = None

        # ── 核心：初始化 DeepSORT 底层跟踪器 ──
        self.tracker = DeepSort(
            max_age=int(cfg.get("max_age", 30)),
            # ⬆ 轨迹最大丢失容忍帧数。轨迹超过此帧未匹配则永久删除
            n_init=int(cfg.get("n_init", 2)),
            # ⬆ 新轨迹确认所需连续命中帧数。暂定→确认的最小观测次数
            nn_budget=int(cfg.get("nn_budget", 30)),
            # ⬆ 外观特征库每类最大保存数（旧特征优先淘汰）
            max_iou_distance=float(cfg.get("max_iou_distance", 0.9)),
            # ⬆ 匈牙利匹配中 IoU 距离阈值。>阈值的检测-轨迹对不会匹配
            max_cosine_distance=float(cfg.get("max_cosine_distance", 0.35)),
            # ⬆ 匈牙利匹配中余弦距离阈值。>阈值的外观特征对不会匹配
            embedder=embedder,
            embedder_gpu=bool(cfg.get("embedder_gpu", False)),
            half=bool(cfg.get("half", False)),
            bgr=bool(cfg.get("bgr", True)),
            polygon=bool(cfg.get("polygon", False)),
        )

    def _fast_embed_one(self, frame, det: Detection) -> np.ndarray:
        """用 HSV 直方图代替 CNN ReID，极快。
        
        算法步骤：
          1. 从原图中裁剪检测框区域 (x1:y1, x2:y2)
          2. 将 BGR 图像转为 HSV 色彩空间
          3. 分别计算 H/S/V 三个通道的直方图（默认 8×4×4=128 维）
          4. L2 归一化后展平为一维特征向量
        
        此方法利用颜色分布作为行人外观特征，避免了深度网络推理开销。
        
        输入：frame = 原始全图 BGR（含完整色彩信息），det = 单个检测框
        输出：np.ndarray shape=(128,)，L2归一化的 HSV 颜色直方图
        """
        h, w = frame.shape[:2]                    # 帧尺寸

        # 边界裁剪：确保坐标不超出图像范围
        x1 = max(0, min(int(det.x1), w - 1))      # 左上 x，钳制到 [0, w-1]
        y1 = max(0, min(int(det.y1), h - 1))      # 左上 y，钳制到 [0, h-1]
        x2 = max(0, min(int(det.x2), w))          # 右下 x，钳制到 [0, w]
        y2 = max(0, min(int(det.y2), h))          # 右下 y，钳制到 [0, h]

        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return np.zeros(                      # 框太小（≤1像素）→ 返回零向量
                (self.hist_bins_h * self.hist_bins_s * self.hist_bins_v,),
                dtype=np.float32,
            )

        crop = frame[y1:y2, x1:x2]                # 从原图剪切 ROI 区域
        if crop.size == 0:
            return np.zeros(                      # 空裁剪 → 返回零向量
                (self.hist_bins_h * self.hist_bins_s * self.hist_bins_v,),
                dtype=np.float32,
            )

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)# BGR → HSV 色彩空间（对光照变化更鲁棒）

        # 计算 3D 直方图：H/S/V 三个通道联合统计颜色分布
        hist = cv2.calcHist(
            [hsv],                                 # 输入 HSV 图像
            [0, 1, 2],                             # 统计 H(0), S(1), V(2) 三个通道
            None,                                  # 无 mask
            [self.hist_bins_h, self.hist_bins_s, self.hist_bins_v],
            # ⬆ 每个通道的分箱数：8(H)×4(S)×4(V)=128
            [0, 180, 0, 256, 0, 256],
            # ⬆ 各通道取值范围：H[0,180], S[0,256], V[0,256]
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        # ⬆ L2 归一化 → 展平为一维 128 维向量 → 转为 float32
        return hist

    def _fast_embeds(self, frame, detections: List[Detection]) -> List[np.ndarray]:
        # 批量提取所有检测框的 HSV 特征
        return [self._fast_embed_one(frame, det) for det in detections]

    def update(self, frame, detections: List[Detection]) -> List[TrackResult]:
        """
        跟踪更新主入口 ═══════════════════════════════════════════
        上游调用者：worker.py 主循环（每帧调用一次）
        下游消费者：LineCounter.update() / ZoneCounterManager.update()
        输入：frame = 原始全图 BGR（用于提取外观特征，不能用掩膜帧）
              detections = YOLODetector.detect() 的输出列表
        输出：List[TrackResult] = 已确认且活跃的轨迹列表
        ─────────────────────────────────────────────────────────────
        内部流程：
          1) Detection → DeepSORT 格式 (left, top, w, h) 转换
          2) ReID 特征提取（HSV 或 CNN）
          3) tracker.update_tracks() → 卡尔曼预测 + 级联匹配 + IoU匹配
          4) 过滤暂定/丢失轨迹 → 封装为 TrackResult
        """
        bbs = []
        for det in detections:
            left, top, width, height = det.to_ltwh()  # xyxy → ltwh 格式转换
            bbs.append(([left, top, width, height], det.conf, det.class_name))
            # ⬆ DeepSORT 期望格式：([l,t,w,h], confidence, class_name)

        if self.fast_reid:
            embeds = self._fast_embeds(frame, detections)  # 批量提取 HSV 特征 (128维)
            tracks = self.tracker.update_tracks(bbs, embeds=embeds, frame=None)
            # ⬆ 核心匹配调用（传入预提取的特征，不传 frame）
        else:
            tracks = self.tracker.update_tracks(bbs, frame=frame)
            # ⬆ CNN ReID 模式：传入 frame，DeepSORT 内部自行提取 CNN 特征

        # update_tracks() 内部执行（简化算法流程）：
        #  ┌─ 1) 卡尔曼滤波预测 ────────────────────────────────┐
        #  │   对每条已确认轨迹，用卡尔曼滤波根据上一帧状态      │
        #  │   预测当前帧的边界框位置 (x,y,w,h) 和速度 (vx,vy)  │
        #  ├─ 2) 级联匹配（Cascade Matching）────────────────────┤
        #  │   优先匹配"time_since_update 最小"的轨迹（活跃轨迹）│
        #  │   使用马氏距离(运动) + 余弦距离(外观) 加权组合      │
        #  │   匹配成功的检测→更新轨迹状态和特征库               │
        #  ├─ 3) IoU 匹配 ──────────────────────────────────────┤
        #  │   对仍未匹配的检测和轨迹，用 IoU 距离做二次匹配    │
        #  ├─ 4) 轨迹管理 ──────────────────────────────────────┤
        #  │   未匹配检测 → 初始化为"暂定"轨迹 (tentative)      │
        #  │   暂定轨迹命中≥n_init次 → 升级为"确认" (confirmed) │
        #  │   确认轨迹丢失≥max_age次 → 永久删除                │
        #  └──────────────────────────────────────────────────────┘

        outputs: List[TrackResult] = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue                 # 跳过未确认的暂定轨迹（不参与计数）
            if tr.time_since_update > 1:
                continue                 # 跳过丢失超过 1 帧的轨迹（避免闪烁 ID）

            l, t, r, b = tr.to_ltrb()    # DeepSORT 内部格式 → xyxy
            outputs.append(
                TrackResult(
                    track_id=int(tr.track_id),  # DeepSORT 分配的唯一整数 ID
                    x1=float(l),                 # 左上 x
                    y1=float(t),                 # 左上 y
                    x2=float(r),                 # 右下 x
                    y2=float(b),                 # 右下 y
                    conf=1.0,                    # 跟踪置信度固定为1.0（已确认）
                    class_id=0,                  # 固定为 person 类别
                    class_name="person",
                )
            )
        return outputs                     # → 传入 LineCounter / ZoneCounter

    def summary(self) -> Dict:
        out = dict(self.cfg)
        out["fast_reid"] = self.fast_reid
        return out
