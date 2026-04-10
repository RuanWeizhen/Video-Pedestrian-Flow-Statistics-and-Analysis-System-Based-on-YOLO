from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v8/yolov8s_ema_p3.yaml")
model.info()
