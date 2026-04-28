from ultralytics import YOLO
import pandas as pd

def main():
    model_path = r"runs/train/yolov8s_ema_p3_800_sgd/weights/best.pt"
    data_path = r"datasets/pedestrian_all/data.yaml"

    model = YOLO(model_path)
    metrics = model.val(
        data=data_path,
        split="val",
        imgsz=800,
        save_json=False,
        plots=False,
        verbose=True
    )

    # 提取核心指标
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    # 生成表
    df = pd.DataFrame([
        {
            "实验名称": "EMA-90",
            "模型": "YOLOv8s+EMA",
            "权重文件": "best.pt",
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "mAP@0.5": round(map50, 4),
            "mAP@0.5:0.95": round(map5095, 4),
        }
    ])

    print("\nEMA 90轮实验 best 指标表：")
    print(df.to_string(index=False))

    df.to_csv("ema90_best_metrics_table.csv", index=False, encoding="utf-8-sig")
    print("\n已保存为 ema90_best_metrics_table.csv")

if __name__ == "__main__":
    main()
