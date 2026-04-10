from ultralytics import YOLO
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def main():
    # 1. 创建改进模型结构（P2 + CBAM）
    model = YOLO('yolov8s_p2_cbam.yaml')

    # 2. 加载第一阶段标准模型权重（自动匹配相同层，新增层随机初始化）
    model.load(r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\runs\train\pedestrian_baseline\weights\best.pt')

    # 3. 开始微调（适当提高学习率，恢复数据增强）
    model.train(
        data='datasets/pedestrian_all/data.yaml',
        epochs=100,
        imgsz=640,
        batch=-1,  # 自动选择（改进模型可能略低）
        device=0,
        workers=8,
        cache=True,
        patience=20,

        # 优化器与学习率（提高至 0.001，仍低于标准模型初始值）
        optimizer='AdamW',
        lr0=0.001,  # 原 0.0005 → 0.001
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,

        # 数据增强（恢复至接近标准模型水平）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,  # 从 0.3 提高至 0.5
        mixup=0.1,
        copy_paste=0.05,
        auto_augment='randaugment',
        erasing=0.4,

        close_mosaic=10,
        amp=True,
        project='runs/train',
        name='pedestrian_stage2_fixed',  # 新实验目录
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )


if __name__ == '__main__':
    main()