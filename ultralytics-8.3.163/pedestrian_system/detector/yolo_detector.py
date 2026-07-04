from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# 强制优先加载本地 ultralytics（包含自定义模块如 EMA）。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.models.yolo.model import YOLO

from utils.common import Detection


# ============================================================================
# YOLODetector — YOLO 目标检测器
# ============================================================================
# 功能：封装 Ultralytics YOLO 模型，用于从视频帧中检测行人目标。
#
# 核心流程：
#   1. 加载 YOLO 模型权重文件（.pt / .onnx / .engine 等）
#   2. 对每帧图像执行推理，输出边界框、置信度、类别
#   3. 过滤出指定类别的检测结果（默认 class_id=0 即 "person"）
#   4. 将原始输出封装为统一的 Detection 数据结构返回
#
# 关键参数：
#   imgsz   — 推理输入尺寸，越大越精确但越慢（默认 640）
#   conf    — 置信度阈值，低于此值的检测被丢弃（默认 0.25）
#   iou     — NMS 的 IoU 阈值，用于抑制重叠检测框（默认 0.45）
#   device  — 推理设备，"cpu"、"0"（GPU 0）、"cuda" 等
#   half    — 是否使用 FP16 半精度推理（仅 CUDA 有效）
# ============================================================================

class YOLODetector:
    def __init__(self, cfg: Dict):
        # ── 上游调用者：WorkerThread._prepare_detector() ──
        # ── 输入：cfg = 配置字典，包含 model_path/imgsz/conf/iou/device/half/class_ids ──
        self.cfg = {}
        self.model_path = ""                   # 模型权重文件路径，如 "yolov8m.pt" 或 "yolov8m.engine"
        self.imgsz = 640                       # 推理输入尺寸（像素），内部自动 resize+letterbox
        self.conf = 0.25                       # 置信度阈值：低于此值的检测框直接丢弃
        self.iou = 0.45                        # NMS 的 IoU 阈值：重叠超过此值的框被抑制
        self.device = 0                        # 推理设备：0=GPU:0，cpu=CPU，"cuda:1"=指定GPU
        self.half = False                      # FP16 半精度推理（仅 CUDA 可用，速度↑精度≈不变）
        self.class_ids = {0}                   # 需要检测的类别 ID 集合：0=COCO 中的 person
        self.class_name_map = {0: "person"}     # 类别 ID → 名称映射

        self.model = None                      # YOLO 模型实例，首次构造时通过 configure() 加载
        self.configure(cfg, load_model=True)   # 应用配置并加载模型权重文件

    def configure(self, cfg: Dict, load_model: bool = False) -> None:
        # ── 功能：运行时更新检测器参数，可选重新加载模型 ──
        self.cfg = dict(cfg)
        self.model_path = self.cfg["model_path"]                     # 必填项：模型文件路径
        self.imgsz = int(self.cfg.get("imgsz", self.imgsz))          # 推理尺寸，越大越精确但越慢
        self.conf = float(self.cfg.get("conf", self.conf))           # 置信度阈值（0.0~1.0）
        self.iou = float(self.cfg.get("iou", self.iou))              # IoU 阈值（0.0~1.0）
        self.device = self.cfg.get("device", self.device)            # 设备："0"/"cpu"/"cuda:1"
        self.half = bool(self.cfg.get("half", self.half))            # FP16 半精度加速
        self.class_ids = set(self.cfg.get("class_ids", [0]))         # 检测类别集合，默认 {0}=person
        self.class_name_map = dict(self.cfg.get("class_name_map", {0: "person"}))

        if load_model or self.model is None:
            self.model = YOLO(self.model_path)  # ── 核心：加载 Ultralytics YOLO 模型 ──
            # 支持格式：.pt（PyTorch权重）、.engine（TensorRT）、.onnx（ONNX）等

        self._normalize_device()                # 规范化设备参数，自动回退 CPU

    def apply_runtime_params(self, conf: float | None = None, iou: float | None = None) -> None:
        # ── 运行时热更新阈值，无需重新加载模型 ──
        if conf is not None:
            self.conf = float(conf)
        if iou is not None:
            self.iou = float(iou)

    def _normalize_device(self) -> None:
        """
        规范化设备标识，统一各种 GPU 表示 → 设备索引 0。
        若 PyTorch 不存在或 CUDA 不可用，自动回退到 CPU 并禁用 FP16。
        """
        try:
            import torch                          # 检查 PyTorch 是否安装
        except Exception:
            self.device = "cpu"                   # 无 PyTorch → 强制 CPU
            self.half = False                     # CPU 不支持 FP16
            return

        device_str = str(self.device).strip().lower()
        if device_str == "cpu":
            self.device = "cpu"                   # 用户显式指定 CPU
            self.half = False                     # CPU 禁用半精度
            return

        if not torch.cuda.is_available():
            self.device = "cpu"                   # GPU 不可用 → 自动回退 CPU
            self.half = False
            return

        if device_str in {"cuda", "cuda:0", "0", "gpu", "gpu:0"}:
            self.device = 0                       # 统一映射为 GPU 设备索引 0
            return

        self.device = self.device                 # 保持用户指定的设备（如 "cuda:2"）

    def detect(self, frame) -> List[Detection]:
        """
        YOLO 推理主入口 ═══════════════════════════════════════════
        上游调用者：worker.py 主循环（每帧调用，传入 frame 或 inference_frame）
        下游消费者：DeepSortTracker.update() 用检测结果做跟踪匹配
        输入：frame = numpy array (H×W×3, BGR)，可能是掩膜后的 inference_frame
        输出：List[Detection] = 检测框列表，每个元素包含 x1,y1,x2,y2,conf,class_id,class_name
        ─────────────────────────────────────────────────────────────
        内部流程：
          1) model.predict() → 预处理+CNN前向+解码+NMS → result
          2) result.boxes → 提取坐标/置信度/类别
          3) GPU→CPU 转移 → numpy
          4) 遍历过滤 → 构造 Detection 对象列表
        """
        if self.model is None:
            self.model = YOLO(self.model_path)   # 惰性加载：若模型未初始化则补载

        # 仅检测指定类别：class_ids=[0] 则只输出 person，减少 NMS/后处理开销
        classes = sorted(self.class_ids) if self.class_ids else None

        result = next(                           # ── 核心推理调用 ──
            self.model.predict(                  # stream=True → 返回生成器，取第一个元素
                source=frame,                    # 输入帧（numpy BGR 数组或用掩膜处理后的帧）
                imgsz=self.imgsz,                # 推理尺寸（内部做 resize+letterbox 保持比例）
                conf=self.conf,                  # 置信度阈值，低于此值的检测直接丢弃
                iou=self.iou,                    # NMS 的 IoU 阈值，抑制重叠框
                device=self.device,              # 推理设备：0=GPU / cpu
                half=self.half,                  # FP16 半精度（CUDA 下加速 ~2×）
                classes=classes,                 # 只检测 person，跳过无关类别
                verbose=False,                   # 不打印进度条
                stream=True,                     # 流式模式，低内存占用
            )
        )
        # model.predict() 内部完整流程：
        #  ① 预处理：frame → resize(imgsz) → /255 归一化 → HWC→CHW → 组装 batch=1
        #  ② CNN 前向：backbone(提取多尺度特征) → neck(特征融合) → head(分类+回归)
        #  ③ 后处理：anchor解码 → sigmoid/softmax → 置信度过滤 → NMS去重 → result

        boxes = result.boxes                     # 检测结果容器
        if boxes is None or len(boxes) == 0:
            return []                            # 无检测 → 返回空列表

        xyxy = boxes.xyxy                        # shape=(N,4) 边界框坐标 (x1,y1,x2,y2)
        confs = boxes.conf                       # shape=(N,) 置信度分数
        clss = boxes.cls                         # shape=(N,) 类别 ID（浮点数，需转 int）

        # 将 GPU 张量搬迁到 CPU 并转为 NumPy 数组（detach 由内部自动完成）
        xyxy = xyxy.cpu().numpy()                # (N,4) float32
        confs = confs.cpu().numpy()              # (N,)  float32
        clss = clss.cpu().numpy()                # (N,)  float32

        detections: List[Detection] = []
        for box, conf, cls_id_float in zip(xyxy, confs, clss):
            cls_id = int(cls_id_float)           # 类别 ID 浮点数 → 整数
            if cls_id not in self.class_ids:
                continue                         # 跳过非目标类别（二次保险过滤）

            x1, y1, x2, y2 = map(float, box.tolist())
            detections.append(                   # 封装为统一的 Detection 数据结构
                Detection(
                    x1=x1,                       # 检测框左上角 x
                    y1=y1,                       # 检测框左上角 y
                    x2=x2,                       # 检测框右下角 x
                    y2=y2,                       # 检测框右下角 y
                    conf=float(conf),            # 置信度分数（0.0~1.0）
                    class_id=cls_id,             # 类别 ID
                    class_name=str(self.class_name_map.get(cls_id, f"class_{cls_id}")),
                )
            )
        return detections                       # → 传入 DeepSORT tracker.update()

    def summary(self) -> Dict:
        return {
            "model_path": self.model_path,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "half": self.half,
            "class_ids": sorted(self.class_ids),
        }
