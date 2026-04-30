# ==================================================
# 云端强模型容量训练：YOLOv8m + EMA-P3 + imgsz 896
# 目标：提高模型容量，冲击更高 mAP@0.5
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
    project_dir = root / "runs" / "train"

    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {data_yaml}")

    # ==================================================
    # 自动写入 YOLOv8m-EMA-P3 配置文件
    # 说明：
    # 1. 使用 YOLOv8 标准 m 规模；
    # 2. 在 P3 检测分支后加入 EMA 注意力；
    # 3. 检测头仍为 P3/P4/P5 三尺度。
    # ==================================================
    model_yaml = root / "ultralytics" / "cfg" / "models" / "v8" / "yolov8m_ema_p3.yaml"

    yaml_text = """
# Ultralytics YOLOv8m-EMA-P3 model

nc: 80

scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]

  # P3 分支加入 EMA 注意力
  - [-1, 1, EMA, [8]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]

  - [[16, 19, 22], 1, Detect, [nc]]
"""
    model_yaml.write_text(yaml_text.strip() + "\n", encoding="utf-8")

    print("=" * 80)
    print(f"数据集配置: {data_yaml}")
    print(f"模型配置: {model_yaml}")
    print("预训练权重: yolov8m.pt")
    print("实验名称: yolov8m_ema_p3_896_sgd_cloud")
    print("=" * 80)

    # 加载 YOLOv8m-EMA-P3 结构，并迁移 yolov8m.pt 可匹配权重
    model = YOLO(str(model_yaml)).load("yolov8m.pt")

    model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=896,

        # 4090D 显存足够，AutoBatch 自动选安全 batch
        batch=-1,
        device=0,
        workers=8,

        # 目前磁盘剩余不多，训练集不强制缓存，避免空间不足警告
        cache=False,

        patience=35,

        # 优化器：比 v8s 后期续训略高，但低于初始主训，适合大模型稳定学习
        optimizer="SGD",
        lr0=0.006,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,

        # 单类行人检测：定位优先
        box=8.0,
        cls=0.3,
        dfl=1.5,

        # 数据增强：比主训练温和，兼顾泛化与收敛
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=3.0,
        translate=0.03,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,

        # 最后 20 轮关闭 Mosaic，贴近真实分布
        close_mosaic=20,

        amp=True,
        single_cls=True,
        rect=False,
        seed=0,
        deterministic=False,

        project=str(project_dir),
        name="yolov8m_ema_p3_896_sgd_cloud",
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,

        # 先关掉 plots，避免因为画图依赖中断训练
        plots=False,
        verbose=True,
        resume=False,
    )

    print("=" * 80)
    print("YOLOv8m-EMA-P3 强模型训练完成。")
    print(f"结果目录: {project_dir / 'yolov8m_ema_p3_896_sgd_cloud'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
