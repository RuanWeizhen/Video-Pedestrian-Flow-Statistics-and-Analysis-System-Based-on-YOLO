from ultralytics import YOLO
from pathlib import Path


def main():
    # ========= 路径配置 =========
    project_dir = Path(
        r"E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\runs\train\yolov8s_ema_p3_800_sgd"
    )
    last_pt = project_dir / "weights" / "last.pt"

    if not last_pt.exists():
        raise FileNotFoundError(f"未找到续训权重文件: {last_pt}")

    print(f"检测到 last.pt：{last_pt}")
    print("开始从第30轮结果继续训练到第60轮...")

    # ========= 加载上一次训练的 last.pt =========
    model = YOLO(str(last_pt))

    # ========= 续训到总共 60 轮 =========
    # 说明：
    # 1. 这里不能只写 resume=True 而不改总轮数，否则通常会按原来的30轮配置结束
    # 2. 重新指定 epochs=60，表示总训练轮数目标为60，而不是再加60轮
    # 3. 其余关键参数保持与原 EMA 实验一致，确保对比公平
    model.train(
        data=r"datasets/pedestrian_all/data.yaml",
        epochs=60,                 # 总轮数目标为60
        imgsz=800,

        # 显存与数据加载
        batch=-1,
        device=0,
        workers=4,
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
        verbose=True,
        resume=False,              # 这里用 last.pt 接着训，不走 resume=True
    )

    print("EMA 已完成从30轮续训到60轮。")
    print("请重点查看第60轮的指标，并与基线第60轮 mAP50=0.769 对比。")


if __name__ == "__main__":
    main()
