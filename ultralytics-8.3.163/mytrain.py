from ultralytics import YOLO

def main():
    # 加载预训练模型
    model = YOLO('yolov8s.pt')

    # 开始训练
    model.train(
        data='datasets/person9000/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project='runs/train',
        name='person_test',
        exist_ok=True,
        amp=True
    )

if __name__ == '__main__':
    # Windows 下多进程的入口必须放在这里
    main()