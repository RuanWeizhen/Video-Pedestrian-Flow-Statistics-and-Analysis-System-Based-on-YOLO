import os
import random
import shutil
from tqdm import tqdm


def split_dataset(source_images_dir, source_labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    source_images_dir: coco_temp/images
    source_labels_dir: coco_temp/labels
    output_dir: pedestrian_all 根目录
    """
    # 创建输出子目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 获取所有图片文件
    images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        for img in tqdm(img_list, desc=f'Copying {split}'):
            # 复制图片
            src_img = os.path.join(source_images_dir, img)
            dst_img = os.path.join(output_dir, 'images', split, img)
            shutil.copy(src_img, dst_img)

            # 复制标签
            label_file = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(source_labels_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, 'labels', split, label_file)
                shutil.copy(src_label, dst_label)
            else:
                print(f"警告：标签 {label_file} 不存在")


if __name__ == '__main__':
    source_img_dir = r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\datasets\coco_temp\images'
    source_lab_dir = r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\datasets\coco_temp\labels'
    output_dir = r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\datasets\pedestrian_all'

    split_dataset(source_img_dir, source_lab_dir, output_dir)