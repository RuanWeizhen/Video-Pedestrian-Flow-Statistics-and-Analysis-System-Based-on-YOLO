# ======================================================================
# YOLOv8m + EMA-P3 + imgsz 896 云端强模型训练脚本
# ======================================================================
# 目标：在高端 GPU（如 RTX 4090D）上训练高容量行人检测模型，
#       冲击更高 mAP@0.5，以便更好地服务客流统计系统的检测端。
#
# 三个关键设计决策：
# │
# ├─ ① YOLOv8m（而非 n/s）

# │
# ├─ ② EMA 注意力模块（插入 P3 检测分支前）

# │
# └─ ③ imgsz 896（而非默认的 640）

#
# 训练策略：
#   - 基于 COCO 预训练的 yolov8m.pt 做迁移学习
#   - 优化器使用 SGD + Cosine LR Schedule（带 warmup）
#   - 单类检测模式（single_cls=True，只检测 person）
#   - Box Loss 权重加高（box=8.0），定位优先于分类
#   - 数据增强温和（mosaic=0.3，不启用 mixup/copy_paste）
#   - 最后 20 轮关闭 Mosaic 增强，贴近真实数据分布做微调收尾
# ======================================================================

from ultralytics import YOLO           # Ultralytics YOLO 训练/推理 API
from pathlib import Path               # 跨平台路径处理
import os                              # 环境变量设置


# ── CUDA 显存优化：启用可扩展内存段 ──
# PyTorch 默认的显存分配器是缓存式（caching allocator），
# 当显存碎片化严重时即使剩余显存足够也会 OOM。
# expandable_segments:True → 允许分配器将相邻空闲段合并为大段，
# 显著减少碎片化，对长时间训练（80 epochs）尤为重要。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Ultralytics 配置/缓存目录 ──
# 云端容器化环境通常 /tmp 是高速临时存储（内存盘或本地 NVMe），
# 将配置文件放到 /tmp 避免持续写入网络挂载的慢速存储。
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"


def main():
    # ================================================================
    # 路径配置
    # ================================================================
    # 项目根目录（云端 autodl-tmp 挂载点，通常是高性能本地 SSD）
    root = Path(
        "/root/autodl-tmp/Video Pedestrian Flow Statistics and Analysis System Based on YOLO/ultralytics-8.3.163"
    )

    # 数据集路径：指向 datasets/pedestrian_all/data.yaml
    # 该 YAML 文件定义了 train/val 图像路径和类别信息（单类 pedestrian）
    data_yaml = root / "datasets" / "pedestrian_all" / "data.yaml"

    # 训练输出目录：所有权重、日志、指标将保存在 runs/train/ 下
    project_dir = root / "runs" / "train"

    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {data_yaml}")

    # ================================================================
    # 自动生成 YOLOv8m-EMA-P3 模型结构 YAML
    # ================================================================
    # 说明：Ultralytics 官方没有内置 EMA 注意力模块的模型配置，
    # 所以需要在此脚本中动态生成自定义的模型结构文件。
    #
    # 网络结构概述（完整数据流）：
    #
    #   Input (BGR, 3×896×896)
    #     │
    #   ┌─ Backbone ─────────────────────────────────────────────┐
    #   │  Conv(k=3,s=2) → Conv(k=3,s=2) → C2f×3 → Conv →       │
    #   │  C2f×6 → Conv → C2f×6 → Conv → C2f×3 → SPPF           │
    #   │  输出：P3(8↓)、P4(16↓)、P5(32↓) 三尺度特征             │
    #   └────────────────────────────────────────────────────────┘
    #     │
    #   ┌─ Neck (FPN + PAN) ────────────────────────────────────┐
    #   │  P5 → Upsample → Concat(P4) → C2f →                    │
    #   │       Upsample → Concat(P3) → C2f → ★ EMA(8) ★ →      │
    #   │       Conv↓ → Concat → C2f →                            │
    #   │       Conv↓ → Concat → C2f                              │
    #   │  输出：P3/8、P4/16、P5/32 增强后的特征图               │
    #   └────────────────────────────────────────────────────────┘
    #     │
    #   ┌─ Head ─────────────────────────────────────────────────┐
    #   │  Detect(nc=80)：在 P3/P4/P5 三尺度上做分类+回归       │
    #   │  每个 Anchor-Free 位置输出：4 个 bbox 偏移 + 1 个 cls │
    #   └────────────────────────────────────────────────────────┘
    #
    # EMA 模块位置：P3（最高分辨率层，8倍下采样）→ 增强小目标
    #   参数 (8) 表示 group size/channel subdivision parameter
    model_yaml = root / "ultralytics" / "cfg" / "models" / "v8" / "yolov8m_ema_p3.yaml"

    yaml_text = """
# ===================================================================
# Ultralytics YOLOv8m-EMA-P3  自定义模型结构定义
# ===================================================================
# 基于 YOLOv8m 标准 backbone + neck，在 P3 分支前插入 EMA 注意力。
# 检测头保持 P3/P4/P5 三尺度，单类行人检测（最终训练时覆写 nc=1）。

nc: 80                          # 输出类别数（COCO 默认 80，训练时自动覆写为 1）

scales:
  n: [0.33, 0.25, 1024]         # Nano:  33% depth, 25% width, max 1024 channels
  s: [0.33, 0.50, 1024]         # Small: 33% depth, 50% width
  m: [0.67, 0.75, 768]          # Medium: 67% depth, 75% width  ← 当前配置
  l: [1.00, 1.00, 512]          # Large:  100% depth, 100% width
  x: [1.00, 1.25, 512]          # X-Large: 100% depth, 125% width

# ── Backbone（主干网络）──
# 每行格式：[from, repeats, module, args]
#   from: 第 N 层的输出作为输入（-1 = 上一层）
#   repeats: 模块重复次数
#   module: 模块类型名
#   args:   [输出通道, 卷积核大小, 步长]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 0: 第1层卷积  (k=3,s=2) → 降采样×2, 64通道
  - [-1, 1, Conv, [128, 3, 2]]          # 1: 第2层卷积  (k=3,s=2) → 降采样×4, 128通道
  - [-1, 3, C2f, [128, True]]           # 2: C2f模块×3 → P2 特征（4倍降采样）
  - [-1, 1, Conv, [256, 3, 2]]          # 3: 卷积降采样 → 降采样×8
  - [-1, 6, C2f, [256, True]]           # 4: C2f模块×6 → P3 特征（8倍降采样）★ 最高分辨率
  - [-1, 1, Conv, [512, 3, 2]]          # 5: 卷积降采样 → 降采样×16
  - [-1, 6, C2f, [512, True]]           # 6: C2f模块×6 → P4 特征（16倍降采样）
  - [-1, 1, Conv, [1024, 3, 2]]         # 7: 卷积降采样 → 降采样×32
  - [-1, 3, C2f, [1024, True]]          # 8: C2f模块×3 → P5 特征（32倍降采样）
  - [-1, 1, SPPF, [1024, 5]]            # 9: 空间金字塔池化(SPPF) → 多尺度感受野融合

# ── Head（Neck + Detection Head）──
# Neck 采用 FPN(特征金字塔) + PAN(路径聚合网络) 双路径结构：
#   FPN:  Top-down    path: P5 → P4 → P3（上采样+融合）
#   PAN:  Bottom-up   path: P3 → P4 → P5（下采样+融合）
# 两路径结合 → 各尺度同时拥有高层语义和低层细节
head:
  # FPN Top-down 路径 ─────────────────────────────────────
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 10: P5 上采样×2（最近邻插值）
  - [[-1, 6], 1, Concat, [1]]                      # 11: 与 P4(层6) 通道拼接
  - [-1, 3, C2f, [512]]                             # 12: C2f×3 → 融合后 P4

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 13: 上采样×2
  - [[-1, 4], 1, Concat, [1]]                      # 14: 与 P3(层4) 通道拼接
  - [-1, 3, C2f, [256]]                             # 15: C2f×3 → 融合后 P3

  # ★ EMA 注意力插入点：P3 分支（最高分辨率，8倍降采样）★
  # EMA(8) = Efficient Multi-scale Attention with sub-feature grouping factor 8
  # 作用机制：将 P3 特征图按通道分组，组间做跨空间交互，
  #          产生加权的注意力特征图 → 增强小行人局部细节
  - [-1, 1, EMA, [8]]                              # 16: EMA 注意力模块

  # PAN Bottom-up 路径 ─────────────────────────────────────
  - [-1, 1, Conv, [256, 3, 2]]                      # 17: 卷积降采样×2
  - [[-1, 12], 1, Concat, [1]]                      # 18: 与 FPN 的 P4(层12) 拼接
  - [-1, 3, C2f, [512]]                              # 19: C2f×3 → PAN 增强 P4

  - [-1, 1, Conv, [512, 3, 2]]                      # 20: 卷积降采样×2
  - [[-1, 9], 1, Concat, [1]]                       # 21: 与 backbone P5(层9) 拼接
  - [-1, 3, C2f, [1024]]                             # 22: C2f×3 → PAN 增强 P5

  # Detection Head → 三尺度卷积预测 ──────────────────────────
  - [[16, 19, 22], 1, Detect, [nc]]                # 23: P3(层16)/P4(层19)/P5(层22) 检测
"""
    # 写入 YAML 文件（自动覆盖已存在的同名文件）
    model_yaml.write_text(yaml_text.strip() + "\n", encoding="utf-8")

    # ── 训练信息输出 ──
    print("=" * 80)
    print(f"数据集配置: {data_yaml}")
    print(f"模型配置: {model_yaml}")
    print("预训练权重: yolov8m.pt")                          # COCO 预训练的官方权重
    print("实验名称: yolov8m_ema_p3_896_sgd_cloud")
    print("=" * 80)

    # ================================================================
    # 加载模型：自定义结构 + COCO 预训练权重迁移
    # ================================================================
    # YOLO(str(yaml)).load("yolov8m.pt") 的工作机制：
    #   1. 读取 yaml 中定义的 backbone + neck + head 结构
    #   2. 下载/加载 yolov8m.pt（COCO 80类预训练权重）
    #   3. 按层名称匹配，将 backbone/neck 的权重复制到新结构
    #      - EMA 模块（新组件）随机初始化
    #      - 其余层（Conv/C2f/SPPF/Detect）继承预训练权重
    #   4. 训练时根据 data.yaml 中的 nc 自动调整 Detect 头输出维度
    model = YOLO(str(model_yaml)).load("yolov8m.pt")

    # ================================================================
    # train() 训练参数详解
    # ================================================================
    model.train(
        # ── 数据 & 模型 ──
        data=str(data_yaml),                                     # 数据集 YAML 路径

        # ── 训练周期与尺寸 ──
        epochs=80,                                               # 总训练轮数
        imgsz=896,                                               # 输入图像尺寸（像素）
        # ⬆ 896×896，比默认 640 大 40%，每个行人像素更多
        #    代价：显存 ≈ 与面积成正比（~2×），单 batch 更小

        # ── Batch Size ──
        batch=-1,                                                # AutoBatch：自动选最优 batch
        # ⬆ -1：Ultralytics 内置自动 batch size 选择器
        #    它会从默认值开始逐步增大 batch，直到显存使用率 ~90%
        #    4090D 24GB 显存下，896×896 通常得到 batch=8~16
        device=0,                                                # GPU 设备索引（单卡训练）
        workers=8,                                                # 数据加载线程数
        # ⬆ workers=8：8 个并行线程预加载+预处理图像
        #    经验值 = CPU 核心数，云端 16 核取 8 已足够

        # ── 缓存策略 ──
        cache=False,                                             # 不缓存数据集到内存/RAM磁盘
        # ⬆ 训练集较大时缓存会额外占用磁盘空间（~图像总大小的 1.5×）
        #    磁盘空间紧张时设为 False，牺牲少量 I/O 速度

        # ── Early Stopping ──
        patience=35,                                             # 早停耐心值
        # ⬆ 若验证集 mAP 连续 35 个 epoch 不提升 → 提前终止训练
        #    总 80 epochs，耐心 35 ≈ 一半周期，合理

        # ════════════════════════════════════════════════════════════
        # 优化器配置：SGD + Cosine LR Schedule
        # ════════════════════════════════════════════════════════════
        # SGD 在大型数据集上的泛化能力通常优于 Adam/AdamW，
        # 虽然收敛速度稍慢，但最终 mAP 往往更高。
        # Cosine LR：学习率从 lr0 余弦衰减到 lr0×lrf，
        # 比阶梯衰减更平滑，减少 loss 振荡。
        optimizer="SGD",                                         # 随机梯度下降

        lr0=0.006,                                               # 初始学习率
        # ⬆ SGD 初始 lr=0.006（较 Adam 的 0.001 偏大是正常的）
        #    因为 SGD 无自适应学习率，需要更大初始值

        lrf=0.01,                                                # 最终学习率因子
        # ⬆ 最终 lr = lr0 × lrf = 0.006 × 0.01 = 0.00006
        #    衰减 100 倍，确保训练结尾足够小，精细收敛

        momentum=0.937,                                          # SGD 动量
        # ⬆ 高动量 → 加速收敛方向的梯度累积、平滑振荡
        #    0.937 是 YOLO 训练的经验最优值

        weight_decay=0.0005,                                     # L2 正则化系数
        # ⬆ 权重衰减 → 防止过拟合，SGD 下约 5e-4 为标准值

        cos_lr=True,                                             # 启用余弦学习率调度
        # ⬆ lr(t) = lrf + 0.5*(lr0 - lrf)*(1 + cos(π*t/T))
        #    训练初始 + 最终阶段 lr 变化缓慢，中间快速衰减

        warmup_epochs=3.0,                                       # 学习率预热周期数
        # ⬆ 前 3 个 epoch 线性增加 lr，防止初始梯度爆炸
        warmup_momentum=0.8,                                     # 预热阶段动量
        warmup_bias_lr=0.05,                                     # 预热阶段偏置层学习率

        # ════════════════════════════════════════════════════════════
        # Loss 权重配置（单类行人检测 → 定位优先）
        # ════════════════════════════════════════════════════════════
        # YOLOv8 的 Loss 由三部分组成：
        #   Box Loss (CIoU)   — 边界框回归，衡量预测框与 GT 的重合度
        #   Cls Loss (BCE)    — 分类损失，判断是否有目标+类别
        #   DFL Loss           — Distribution Focal Loss，边框分布精细化
        # 权重含义：值越大，训练时该 Loss 项对梯度更新的贡献越大
        box=8.0,                                                 # Box Loss 权重
        # ⬆ 默认 7.5，提高到 8.0 → 更强调定位精度
        #    行人检测的难点在于密集场景下的精确边界框回归
        cls=0.3,                                                 # Cls Loss 权重
        # ⬆ 默认 0.5，降低到 0.3 → 弱化分类损失
        #    单类检测（只有 person），分类任务简单，不需要高权重
        dfl=1.5,                                                 # DFL Loss 权重
        # ⬆ 默认 1.5，维持不变

        # ════════════════════════════════════════════════════════════
        # 数据增强参数（温和策略 → 兼顾泛化与收敛）
        # ════════════════════════════════════════════════════════════
        # 数据增强的目的是提升模型的泛化能力，防止对训练集过拟合。
        # 但如果增强太强，训练分布与真实分布差异过大，反而降低性能。
        # 此处选择了比 YOLO 默认值更温和的增强策略。

        # 颜色抖动 — HSV 色彩空间扰动
        hsv_h=0.015,                                             # 色调扰动幅度（0.0~0.015）
        hsv_s=0.7,                                               # 饱和度扰动幅度（0.0~1.0）
        hsv_v=0.4,                                               # 亮度扰动幅度（0.0~1.0）

        # 几何变换 — 随机仿射
        degrees=3.0,                                             # 随机旋转角度范围（±3°）
        translate=0.03,                                          # 随机平移比例（±3%）
        scale=0.25,                                              # 随机缩放比例（0.75~1.25）
        shear=0.0,                                               # 剪切变换（关闭）
        perspective=0.0,                                         # 透视变换（关闭）

        # 翻转
        flipud=0.0,                                              # 上下翻转概率（关闭，行人极少倒立出现）
        fliplr=0.5,                                              # 左右翻转概率（50%，行人对称属性）

        # Mosaic 增强 — 4 张图拼接为 1 张
        mosaic=0.3,                                              # Mosaic 启用概率（30%）
        # ⬆ 默认 1.0（每张都用），降低到 0.3 → 更多原图参与训练
        #    在数据量充足+单类场景下，Mosaic 太强可能导致虚警

        # 更高级的增强（全部关闭 → 保持训练分布接近真实分布）
        mixup=0.0,                                               # 图像混合增强（关闭）
        copy_paste=0.0,                                          # 实例级复制粘贴增强（关闭）
        auto_augment=None,                                       # 自动增强策略（关闭）
        erasing=0.0,                                             # 随机擦除增强（关闭）

        # Mosaic 关闭策略
        close_mosaic=20,                                         # 最后 20 个 epoch 关闭 Mosaic
        # ⬆ 训练后期关闭 Mosaic → 模型在"纯原图"上微调收尾
        #    消除 Mosaic 拼接产生的边界伪影对最终性能的影响

        # ════════════════════════════════════════════════════════════
        # 训练配置
        # ════════════════════════════════════════════════════════════
        amp=True,                                                # 自动混合精度（FP16+FP32 混合训练）
        # ⬆ AMP：大部分计算用 FP16（快 2×），关键操作用 FP32（保精度）
        #    显著降低显存占用 → 允许更大的 batch size

        single_cls=True,                                         # 单类模式
        # ⬆ 所有检测框统一视为 "person"（类别 0）
        #    忽略 data.yaml 中的类别名称，训练速度略有提升

        rect=False,                                              # 矩形训练（关闭）
        # ⬆ False：每张图独立做 letterbox resize（保持原图宽高比）
        #    True：整个 batch 统一 resize 到相同尺寸（减少黑边但损失多样性）

        seed=0,                                                  # 随机种子（固定可复现）
        deterministic=False,                                     # 确定性模式（关闭）
        # ⬆ False：允许 cudnn 使用非确定性算法 → 速度更快但每次结果略有差异
        #    True：完全可复现的训练结果，但速度显著降低

        # ════════════════════════════════════════════════════════════
        # 保存 & 验证
        # ════════════════════════════════════════════════════════════
        project=str(project_dir),                                # 训练输出根目录
        name="yolov8m_ema_p3_896_sgd_cloud",                     # 实验名称（=子目录名）
        exist_ok=True,                                           # 允许覆盖同名实验目录
        save=True,                                               # 保存训练权重
        save_period=10,                                          # 每 10 个 epoch 保存一次 checkpoint
        val=True,                                                # 每个 epoch 后执行验证集评估
        # ⬆ 验证频率：每 epoch 一次（计算 mAP@0.5/mAP@0.5:0.95 等指标）

        # ════════════════════════════════════════════════════════════
        # 日志 & 可视化
        # ════════════════════════════════════════════════════════════
        plots=False,                                             # 不生成训练曲线图
        # ⬆ 云端无 GUI 环境时 matplotlib 绘图可能报错，
        #    训练结束后可手动用 results.csv 在本地画图
        verbose=True,                                            # 详细日志输出
        resume=False,                                            # 不从中断点恢复（从头训练）
    )

    # ── 训练完成 ──
    print("=" * 80)
    print("YOLOv8m-EMA-P3 强模型训练完成。")
    print(f"结果目录: {project_dir / 'yolov8m_ema_p3_896_sgd_cloud'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    # ── 典型运行方式 ──
    # 命令行：python train_yolov8m_ema_p3_cloud.py
    # 后台运行（不受终端断开影响）：
    #   nohup python train_yolov8m_ema_p3_cloud.py > train.log 2>&1 &
    #   然后用 tail -f train.log 实时查看训练进度
