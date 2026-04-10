# ==================================================
# 主训练阶段：YOLOv8s + 分辨率 800 + SGD（适配 6GB 显存）
# 日期：2026-03-04
# ==================================================
from ultralytics import YOLO
import os

# 可选：减少显存碎片（Windows 下不一定有效，但保留无妨）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # 加载标准 YOLOv8s 预训练模型
    model = YOLO("yolov8s.pt")

    model.train(
        data="datasets/pedestrian_all/data.yaml",
        epochs=120,
        imgsz=800,

        # ===== 6GB 显存建议：使用 AutoBatch 自动选择安全 batch =====
        batch=-1,                 # 自动 batch（大约占用 60% 显存），更稳
        device=0,
        workers=4,                # 笔记本/Windows 通常 4 比 8 更稳定
        cache="disk",             # 大数据集建议用磁盘缓存，避免吃爆内存
        patience=30,

        # ===== 优化器设置 =====
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ===== 损失权重（单类行人：定位优先） =====
        box=8.0,
        cls=0.3,
        dfl=1.5,

        # ===== 数据增强（温和且符合行人场景） =====
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,                # 建议剪切设为 0，避免影响框回归稳定性
        perspective=0.0,
        flipud=0.0,               # 行人几乎不会倒立出现，建议关闭上下翻转
        fliplr=0.5,               # 左右翻转可保留
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,

        # auto_augment / erasing 更偏分类任务；检测任务建议关掉，保证实验可解释
        auto_augment=None,
        erasing=0.0,

        # 最后若干轮关闭 Mosaic，让收敛更稳（建议 10 轮）
        close_mosaic=10,

        # ===== 其他设置 =====
        amp=True,                 # 混合精度，省显存/加速
        single_cls=True,          # 明确单类检测（行人）
        rect=False,                # 关闭矩形训练
        deterministi1c=False,  # 可选：冲上限建议关
        project="runs/train",
        name="yolov8s_800_sgd_rectoff",
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()
