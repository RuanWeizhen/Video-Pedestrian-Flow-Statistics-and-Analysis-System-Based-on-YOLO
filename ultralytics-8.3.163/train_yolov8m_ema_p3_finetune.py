# ==================================================
# YOLOv8m-EMA-P3 最终精修阶段
# 从强模型 best.pt 开始，低学习率 + 高分辨率 + 关闭强增强
# 目标：进一步提升 mAP@0.5，冲击 0.85
# ==================================================

from ultralytics import YOLO
from pathlib import Path
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"


def main():
    root = Path(
        "/root/autodl-tmp/Video Pedestrian Flow Statistics and Analysis System Based on YOLO/ultralytics-8.3.163"
    )

    data_yaml = root / "datasets" / "pedestrian_all" / "data.yaml"

    best_pt = (
        root
        / "runs"
        / "train"
        / "yolov8m_ema_p3_896_sgd_cloud"
        / "weights"
        / "best.pt"
    )

    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {data_yaml}")

    if not best_pt.exists():
        raise FileNotFoundError(f"未找到 YOLOv8m best.pt: {best_pt}")

    print("=" * 80)
    print(f"数据集配置: {data_yaml}")
    print(f"精修权重: {best_pt}")
    print("精修策略: YOLOv8m-EMA-P3 + imgsz=1024 + 低学习率 + 关闭 Mosaic")
    print("=" * 80)

    model = YOLO(str(best_pt))

    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=1024,

        batch=-1,
        device=0,
        workers=8,
        cache=False,
        patience=20,

        optimizer="SGD",
        lr0=0.0003,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        warmup_epochs=0.5,
        warmup_momentum=0.85,
        warmup_bias_lr=0.001,

        box=8.0,
        cls=0.3,
        dfl=1.5,

        # 精修阶段：关闭强几何增强
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        degrees=0.0,
        translate=0.0,
        scale=0.03,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.1,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
        close_mosaic=0,

        amp=True,
        single_cls=True,
        rect=False,
        seed=0,
        deterministic=False,

        project=str(root / "runs" / "train"),
        name="yolov8m_ema_p3_1024_finetune",
        exist_ok=True,

        save=True,
        save_period=10,
        val=True,
        plots=False,
        verbose=True,
        resume=False,
    )

    print("=" * 80)
    print("YOLOv8m-EMA-P3 精修完成。")
    print("请重点查看 yolov8m_ema_p3_1024_finetune/weights/best.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()
