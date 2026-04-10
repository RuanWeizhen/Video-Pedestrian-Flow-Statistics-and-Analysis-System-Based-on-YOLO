import json
import os
import shutil
from tqdm import tqdm  # 如未安装：pip install tqdm


def convert_coco_to_yolo(coco_ann_file, image_dir, output_dir, class_id=0, prefix='coco'):
    """
    将 COCO 标注转换为 YOLO 格式，并复制图片到输出目录
    """
    with open(coco_ann_file, 'r') as f:
        coco = json.load(f)

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # 图片信息映射
    images_info = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco['images']}

    # 收集 person 标注 (category_id=1)
    annotations = {}
    for ann in coco['annotations']:
        if ann['category_id'] != 1:
            continue
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann['bbox'])

    for img_id, bboxes in tqdm(annotations.items(), desc=f'Processing {os.path.basename(coco_ann_file)}'):
        file_name, width, height = images_info[img_id]

        # 添加前缀避免文件名冲突
        base_name = os.path.splitext(file_name)[0]
        new_img_name = f"{prefix}_{base_name}.jpg"
        new_label_name = f"{prefix}_{base_name}.txt"

        # 复制图片
        src_img = os.path.join(image_dir, file_name)
        dst_img = os.path.join(output_dir, 'images', new_img_name)
        if not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)

        # 生成标签
        label_path = os.path.join(output_dir, 'labels', new_label_name)
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"完成！共处理 {len(annotations)} 张图片，输出至 {output_dir}")


if __name__ == '__main__':
    # 根据你的实际路径修改
    coco_root = r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\datasets\coco2017'

    train_ann = os.path.join(coco_root, 'annotations_trainval2017', 'annotations', 'instances_train2017.json')
    val_ann   = os.path.join(coco_root, 'annotations_trainval2017', 'annotations', 'instances_val2017.json')
    train_img = os.path.join(coco_root, 'train2017')
    val_img = os.path.join(coco_root, 'val2017')

    # 临时输出目录
    temp_output = r'E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\datasets\coco_temp'

    # 转换训练集
    convert_coco_to_yolo(train_ann, train_img, temp_output, prefix='coco_train')
    # 转换验证集
    convert_coco_to_yolo(val_ann, val_img, temp_output, prefix='coco_val')