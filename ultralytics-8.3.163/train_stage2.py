from ultralytics import YOLO
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def main():
    # 1. 加载改进模型的结构
    model = YOLO('yolov8s_p2_cbam.yaml')  # 你的改进模型配置文件

    # 2. 加载第一阶段训练好的标准模型权重
    #   注意：这里不是用 model.load()，而是直接加载权重文件，因为结构不同
    #   Ultralytics 会自动匹配相同层，新增层随机初始化
    model = YOLO('runs/train/pedestrian_baseline/weights/best.pt')  # 直接加载权重，自动适配结构

    # 3. 开始微调（保守超参数）
    model.train(
        data='datasets/pedestrian_all/data.yaml',
        epochs=100,  # 微调轮数
        imgsz=640,
        batch=-1,  # 自动选择 batch（标准模型下 batch=7，改进模型可能略低，让系统自动测）
        device=0,
        workers=8,
        cache=True,
        patience=20,

        # 优化器与学习率（大幅降低，保护已学特征）
        optimizer='AdamW',
        lr0=0.0005,  # 仅为标准模型初始学习率的一半
        lrf=0.01,
        cos_lr=True,  # 余弦退火帮助微调
        warmup_epochs=3,

        # 数据增强（温和起步，逐步可恢复至目标强度）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,  # 保留尺度变化
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.3,  # 从 0.3 开始，防止初期不稳定
        mixup=0.1,
        copy_paste=0.05,
        auto_augment='randaugment',
        erasing=0.4,

        close_mosaic=10,
        amp=True,
        project='runs/train',
        name='pedestrian_stage2',  # 新实验目录
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )


if __name__ == '__main__':
    main()