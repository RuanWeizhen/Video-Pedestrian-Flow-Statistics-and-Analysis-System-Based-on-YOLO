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
    print("开始从第60轮结果继续训练到第90轮...")

    # ========= 加载上一次训练的 last.pt =========
    model = YOLO(str(last_pt))

    # ========= 续训到总共 90 轮 =========
    # 说明：
    # 1. 这里设置 epochs=90，表示总训练目标轮数为 90，而不是再加 90 轮
    # 2. 其余关键参数与前面的 EMA 实验保持一致，保证对比公平
    # 3. 使用同一个实验目录，权重和结果会继续写入原文件夹
    model.train(
        data=r"datasets/pedestrian_all/data.yaml",
        epochs=90,                 # 总轮数目标为90
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
        resume=False,              # 这里仍然采用 last.pt 接着训
    )

    print("EMA 已完成从60轮续训到90轮。")
    print("请重点查看第90轮的指标，并与当前主模型 best 指标进行对比。")


if __name__ == "__main__":
    main()
