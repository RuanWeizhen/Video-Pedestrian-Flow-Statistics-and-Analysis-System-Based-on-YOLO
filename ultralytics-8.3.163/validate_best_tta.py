from ultralytics import YOLO


def main():
    model = YOLO("runs/train/yolov8s_800_sgd_rectoff/weights/best.pt")

    # 普通验证
    metrics_base = model.val(
        data="datasets/pedestrian_all/data.yaml",
        imgsz=800,
        batch=4,
        device=0,
        split="val"
    )
    print("base mAP50:", metrics_base.box.map50)
    print("base mAP50-95:", metrics_base.box.map)

    # TTA 验证
    metrics_tta = model.val(
        data="datasets/pedestrian_all/data.yaml",
        imgsz=800,
        batch=4,
        device=0,
        split="val",
        augment=True
    )
    print("tta mAP50:", metrics_tta.box.map50)
    print("tta mAP50-95:", metrics_tta.box.map)


if __name__ == "__main__":
    main()
