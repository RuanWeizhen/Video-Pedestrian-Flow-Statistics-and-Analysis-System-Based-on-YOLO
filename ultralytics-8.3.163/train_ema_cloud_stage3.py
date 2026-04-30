# ==================================================
# 云端续训阶段：从本机90轮 EMA 权重继续冲刺
# 目标：提高 mAP@0.5，冲击 0.85
# 方案：低学习率 + 高分辨率 + 减弱增强 + 后期关闭 Mosaic
# ==================================================

from ultralytics import YOLO
from pathlib import Path
import os


# 可选：减少显存碎片，Linux/AutoDL 一般可以保留
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    # ========= 项目根目录 =========
    root = Path(
        "/root/autodl-tmp/Video Pedestrian Flow Statistics and Analysis System Based on YOLO/ultralytics-8.3.163"
    )

    # ========= 数据集配置 =========
    data_yaml = root / "datasets" / "pedestrian_all" / "data.yaml"

    # ========= 本机90轮上传后的权重目录 =========
    source_run_name = "yolov8s_ema_p3_800_sgd"
    last_pt = root / "runs" / "train" / source_run_name / "weights" / "last.pt"
    best_pt = root / "runs" / "train" / source_run_name / "weights" / "best.pt"

    # ========= 云端新实验输出目录 =========
    project_dir = root / "runs" / "train"
    new_run_name = "yolov8s_ema_p3_896_sgd_cloud_stage3"

    # ========= 文件检查 =========
    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {data_yaml}")

    if not last_pt.exists():
        raise FileNotFoundError(f"未找到续训权重文件: {last_pt}")

    print("=" * 80)
    print(f"数据集配置: {data_yaml}")
    print(f"续训权重: {last_pt}")
    print(f"新实验名称: {new_run_name}")
    print("=" * 80)

    # ==================================================
    # 说明：
    # 这里采用“权重续训”，不是严格 resume=True 的断点续训。
    # 原因：
    # 1. 本机 Windows 路径迁移到 Linux 后，严格 resume 容易因为旧路径报错；
    # 2. 从90轮权重继续冲刺时，更适合重新设置较低学习率；
    # 3. 这样更适合冲击最终 mAP，而不是机械恢复原训练调度。
    # ==================================================

    model = YOLO(str(last_pt))

    model.train(
        data=str(data_yaml),

        # ========= 追加训练轮数 =========
        # 注意：resume=False 时，这里的 epochs=60 表示“再训练60轮”
        # 相当于从原90轮权重继续冲刺到大约150轮水平
        epochs=60,

        # ========= 云端显存更大，适当提高分辨率 =========
        # 如果你云端是 RTX 5090 / 4090D，896 通常可以跑
        # 如果 AutoBatch 选出的 batch 太小，比如 batch=1，可改回 832 或 800
        imgsz=896,

        # ========= 显存与数据加载 =========
        batch=-1,
        device=0,
        workers=8,
        cache="disk",
        patience=50,

        # ========= 优化器：低学习率冲刺 =========
        optimizer="SGD",
        lr0=0.003,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        # 已经训练过90轮，warmup 不宜太长
        warmup_epochs=1.0,
        warmup_momentum=0.85,
        warmup_bias_lr=0.01,

        # ========= 损失权重：保持定位优先 =========
        box=8.0,
        cls=0.3,
        dfl=1.5,

        # ========= 数据增强：比主训练更弱，更利于后期收敛 =========
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=3.0,
        translate=0.03,
        scale=0.20,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.5,

        # 后期冲刺阶段不宜再用太强 Mosaic
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,

        auto_augment=None,
        erasing=0.0,

        # 追加60轮中，最后20轮关闭 Mosaic，让模型贴近真实分布
        close_mosaic=20,

        # ========= 其他设置 =========
        amp=True,
        single_cls=True,
        rect=False,
        seed=0,
        deterministic=False,

        project=str(project_dir),
        name=new_run_name,
        exist_ok=True,

        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,

        # 关键：跨机器续训不要用 resume=True
        resume=False,
    )

    print("=" * 80)
    print("云端 Stage3 续训完成。")
    print(f"请查看结果目录: {project_dir / new_run_name}")
    print(f"重点查看: {project_dir / new_run_name / 'weights' / 'best.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
