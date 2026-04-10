import os
import shutil
from tqdm import tqdm

# 源路径：person9000
src_root = r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\datasets\person9000'
# 目标路径：coco_temp
dst_root = r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\datasets\coco_temp'

prefix = 'person'

# 确保目标目录存在
os.makedirs(os.path.join(dst_root, 'images'), exist_ok=True)
os.makedirs(os.path.join(dst_root, 'labels'), exist_ok=True)

# 处理 train 和 valid 两个部分
for split in ['train', 'valid']:
    src_img_dir = os.path.join(src_root, split, 'images')
    src_lab_dir = os.path.join(src_root, split, 'labels')

    if not os.path.exists(src_img_dir):
        print(f"警告：{src_img_dir} 不存在，跳过")
        continue

    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(img_files, desc=f'Copying {split}'):
        base = os.path.splitext(img_file)[0]
        new_img = f"{prefix}_{base}.jpg"  # 假设原图为 jpg，如果实际是 png 请对应修改
        new_lab = f"{prefix}_{base}.txt"

        # 复制图片
        src_img = os.path.join(src_img_dir, img_file)
        dst_img = os.path.join(dst_root, 'images', new_img)
        shutil.copy(src_img, dst_img)

        # 复制标签
        lab_file = base + '.txt'
        src_lab = os.path.join(src_lab_dir, lab_file)
        if os.path.exists(src_lab):
            dst_lab = os.path.join(dst_root, 'labels', new_lab)
            shutil.copy(src_lab, dst_lab)
        else:
            print(f"警告：标签 {lab_file} 不存在")