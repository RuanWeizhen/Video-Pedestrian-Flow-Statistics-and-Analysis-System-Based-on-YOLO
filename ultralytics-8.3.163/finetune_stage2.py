# ==================================================
# 精修阶段：加载主训练 best.pt，使用更低学习率和更弱增强进行微调
# 目标：进一步稳定收敛，提升最终 mAP
# ==================================================
from ultralytics import YOLO
import os

# 可选：减少显存碎片（Windows 下不一定生效，但保留无妨）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # 加载主训练得到的最佳权重
    model = YOLO("runs/train/yolov8s_800_sgd_rectoff/weights/best.pt")

    model.train(
        data="datasets/pedestrian_all/data.yaml",
        epochs=50,
        imgsz=800,

        # 6GB 显存建议继续使用自动 batch
        batch=-1,
        device=0,
        workers=4,
        cache="disk",
        patience=20,

        # 优化器与学习率：低学习率精修
        optimizer="SGD",
        lr0=3e-4,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        # 损失权重保持与主训练一致
        box=8.0,
        cls=0.3,
        dfl=1.5,

        # 精修阶段尽量贴近真实分布，只保留极弱增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.0,
        scale=0.05,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.2,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
        close_mosaic=0,

        # 其他设置
        amp=True,
        single_cls=True,
        rect=False,
        deterministic=False,
        project="runs/train",
        name="yolov8s_800_sgd_finetune_clean",
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()
