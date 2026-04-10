# ==================================================
# EMA 快速筛选实验：YOLOv8s + EMA(P3) + 分辨率 800 + SGD
# 目标：
# 1. 验证在小目标分支加入 EMA 是否优于当前基线
# 2. 先跑 30 轮做快速筛选，再决定是否继续长训
# ==================================================

from ultralytics import YOLO


def main():
    # ========= 路径配置 =========
    model_yaml = r"ultralytics/cfg/models/v8/yolov8s_ema_p3.yaml"
    pretrained_pt = r"yolov8s.pt"
    data_yaml = r"datasets/pedestrian_all/data.yaml"

    # ========= 加载模型 =========
    model = YOLO(model_yaml)
    model.load(pretrained_pt)

    # ========= 训练参数 =========
    EPOCHS = 30
    IMGSZ = 800
    DEVICE = 0
    WORKERS = 4

    # ========= 开始训练 =========
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,

        # 显存与数据加载
        batch=-1,
        device=DEVICE,
        workers=WORKERS,
        cache="disk",
        patience=20,

        # 优化器
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 损失权重
        box=8.0,
        cls=0.3,
        dfl=1.5,

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
        close_mosaic=10,

        # 其他设置
        amp=True,
        single_cls=True,
        rect=False,
        seed=0,
        deterministic=False,
        project=r"E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\runs\train",
        name="yolov8s_ema_p3_800_sgd",
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True
    )

    print("EMA 快速筛选实验完成。")
    print("请重点对比第 10 轮和第 30 轮的 mAP@0.5：")
    print("基线参考：第10轮 = 0.742，第30轮 = 0.759")
    print("若 EMA 版未超过基线，可考虑停止该路线。")


if __name__ == "__main__":
    main()
