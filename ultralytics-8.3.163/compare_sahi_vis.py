from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction


def main():
    # ===== 你的模型权重 =====
    model_path = r"runs/train/yolov8s_800_sgd_rectoff/weights/best.pt"

    # ===== 验证集图片目录（如果你的 val 图片路径不同，就改这里） =====
    image_dir = Path(r"datasets/pedestrian_all/images/val")

    # ===== 输出目录 =====
    out_root = Path("outputs/sahi_compare")
    out_normal = out_root / "normal"
    out_sliced = out_root / "sliced"
    out_normal.mkdir(parents=True, exist_ok=True)
    out_sliced.mkdir(parents=True, exist_ok=True)

    # ===== 加载 Ultralytics 模型到 SAHI =====
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0"
    )

    # ===== 先抽一小批图做对比，避免一开始跑太久 =====
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])[:200]

    print(f"共选取 {len(image_paths)} 张图进行对比")

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")

        # 1) 普通推理
        result_normal = get_prediction(
            str(img_path),
            detection_model
        )
        result_normal.export_visuals(
            export_dir=str(out_normal),
            file_name=img_path.stem,
            hide_conf=False,
            hide_labels=False
        )

        # 2) SAHI 切片推理
        result_sliced = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        result_sliced.export_visuals(
            export_dir=str(out_sliced),
            file_name=img_path.stem,
            hide_conf=False,
            hide_labels=False
        )

    print("对比完成。")
    print(f"普通推理结果保存在: {out_normal}")
    print(f"SAHI 推理结果保存在: {out_sliced}")


if __name__ == "__main__":
    main()
