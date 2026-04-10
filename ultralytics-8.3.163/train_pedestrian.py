from ultralytics import YOLO
import os

# 可选：减少显存碎片（Windows 下可能无效，但保留无妨）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    # 加载标准 YOLOv8s 预训练模型
    model = YOLO('yolov8s.pt')

    # 开始训练
    model.train(
        data='datasets/pedestrian_all/data.yaml',   # 你的多源数据集
        epochs=150,                                  # 训练轮数，可根据需要增加
        imgsz=640,
        batch=-1,                                    # 自动选择 batch size（推荐 3）
        device=0,
        workers=8,                                    # 若 CPU 压力大，可降至 4
        cache=True,                                    # 缓存图像（内存不足自动降级）
        patience=30,                                   # 早停耐心值

        # 优化器与学习率
        optimizer='AdamW',
        lr0=0.001,                                     # 初始学习率（适中）
        lrf=0.01,                                       # 最终学习率因子
        cos_lr=True,                                    # 余弦退火
        warmup_epochs=3,                                # 预热轮数
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 数据增强（针对性优化，温和版）
        hsv_h=0.015,                                    # 色调增强
        hsv_s=0.7,                                      # 饱和度增强
        hsv_v=0.4,                                      # 明度增强（模拟光照变化）
        degrees=10.0,                                   # 旋转
        translate=0.1,                                  # 平移
        scale=0.5,                                      # 缩放（模拟小目标远近）
        shear=2.0,                                      # 剪切
        flipud=0.5,                                     # 上下翻转（适应俯视视角）
        fliplr=0.5,                                     # 左右翻转
        mosaic=0.5,                                     # Mosaic 增强（温和）
        mixup=0.1,                                      # Mixup 增强（温和）
        copy_paste=0.05,                                # Copy-Paste 增强（温和）
        auto_augment='randaugment',                     # 自动增强
        erasing=0.4,                                    # 随机擦除（模拟遮挡）

        # 其他设置
        close_mosaic=10,                                 # 最后 10 轮关闭 mosaic
        amp=True,                                        # 混合精度加速
        project='runs/train',
        name='pedestrian_baseline',                      # 实验名称
        exist_ok=True,
        save=True,
        save_period=10,                                  # 每 10 轮保存一次权重
        val=True,
        verbose=True,
    )

if __name__ == '__main__':
    main()