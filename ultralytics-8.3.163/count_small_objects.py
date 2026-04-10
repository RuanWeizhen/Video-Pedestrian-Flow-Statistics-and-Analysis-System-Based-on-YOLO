import torch
import pickle
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

# 设置路径
dataset_root = Path('datasets/pedestrian_all')
yaml_path = dataset_root / 'data.yaml'
cache_files = {
    'train': dataset_root / 'labels' / 'train.cache',
    'val': dataset_root / 'labels' / 'val.cache'
}
SMALL_THRESH = 32 * 32  # COCO 小目标定义

total_objects = 0
small_objects = 0

def load_cache(cache_path):
    """尝试用 torch.load 加载缓存，失败则用 pickle"""
    try:
        return torch.load(cache_path, map_location='cpu')
    except Exception:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

# 首先尝试从缓存文件加载
cache_loaded = False
for split, cache_path in cache_files.items():
    if cache_path.exists():
        print(f"加载缓存文件 {cache_path}...")
        try:
            cache_data = load_cache(cache_path)
            # 假设 cache_data 是字典，键为图片路径，值为包含 labels 和 shape 的字典
            for img_path, info in tqdm(cache_data.items(), desc=f"处理 {split}"):
                if 'shape' not in info:
                    continue
                h, w = info['shape'][:2]  # shape 通常是 (height, width, channels)
                labels = info['labels']    # numpy 数组，每行 [class, x, y, w, h]
                for label in labels:
                    if len(label) >= 5:
                        _, _, _, w_norm, h_norm = label[:5]
                        area = w_norm * w * h_norm * h
                        total_objects += 1
                        if area < SMALL_THRESH:
                            small_objects += 1
            cache_loaded = True
        except Exception as e:
            print(f"加载缓存文件 {cache_path} 失败: {e}")
            # 失败后继续尝试手动遍历

if not cache_loaded:
    print("缓存加载失败，将手动遍历图像和标签文件...")
    # 读取 data.yaml
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    path = Path(data_cfg['path'])
    for split in ['train', 'val']:
        # 获取图像目录和标签目录
        # 注意：data.yaml 中的 train/val 可能是 'images/train' 或 'train/images'，这里适配常见格式
        img_dir = path / data_cfg[split].replace('images', 'images')  # 保持原样，但可能需要调整
        # 如果上述路径不存在，尝试用另一种组合
        if not img_dir.exists():
            img_dir = path / data_cfg[split]
        label_dir = path / data_cfg[split].replace('images', 'labels')

        if not img_dir.exists():
            print(f"图像目录 {img_dir} 不存在，跳过")
            continue

        # 获取所有图像文件
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
        for img_file in tqdm(img_files, desc=f"处理 {split}"):
            # 获取图像尺寸
            try:
                with Image.open(img_file) as img:
                    w, h = img.size
            except Exception as e:
                print(f"无法读取图像 {img_file}: {e}")
                continue

            label_file = label_dir / (img_file.stem + '.txt')
            if not label_file.exists():
                continue

            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w_norm, h_norm = map(float, parts)
                area = w_norm * w * h_norm * h
                total_objects += 1
                if area < SMALL_THRESH:
                    small_objects += 1

if total_objects > 0:
    print(f"总目标数: {total_objects}")
    print(f"小目标 (面积 < {SMALL_THRESH} 像素) 数量: {small_objects}")
    print(f"小目标比例: {small_objects / total_objects:.2%}")
else:
    print("未找到任何目标，请检查数据集路径。")