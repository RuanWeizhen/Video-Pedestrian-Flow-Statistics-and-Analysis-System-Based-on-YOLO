# YOLOv8 + DeepSORT + Line/Zone Counting Framework

A lightweight project template for **pedestrian detection, tracking, line crossing counting, and polygon zone counting**.

## 1. Features

- YOLOv8 detection
- DeepSORT multi-object tracking
- Line crossing counting
- Polygon zone counting
- Trajectory drawing
- Video visualization and saving
- CSV event export + JSON summary export
- YAML configuration driven

## 2. Recommended install

```bash
pip install -r requirements.txt
```

## 3. Project structure

```text
yolov8_deepsort_counting_framework/
├── main.py
├── requirements.txt
├── config/
│   └── pedestrian_demo.yaml
├── detector/
│   ├── __init__.py
│   └── yolo_detector.py
├── tracker/
│   ├── __init__.py
│   └── deepsort_wrapper.py
├── counting/
│   ├── __init__.py
│   ├── line_counter.py
│   └── zone_counter.py
└── utils/
    ├── __init__.py
    ├── common.py
    ├── config.py
    ├── geometry.py
    ├── io_utils.py
    └── visualization.py
```

## 4. How to run

Edit `config/pedestrian_demo.yaml` first, especially:

- `model_path`
- `source`
- `output_dir`
- `line.points`
- `zones[*].polygon`

Then run:

```bash
python main.py --config config/pedestrian_demo.yaml
```

## 5. What you should modify first

### Model path
```yaml
model_path: "runs/train/yolov8s_800_sgd_rectoff/weights/best.pt"
```

### Video path
```yaml
source: "videos/test.mp4"
```

### Only detect person
This template defaults to class id `0` as person.

### Counting line
```yaml
line:
  enabled: true
  points:
    - [300, 500]
    - [1500, 500]
  labels:
    neg_to_pos: "up"
    pos_to_neg: "down"
```

### Zone polygon
```yaml
zones:
  - name: "waiting_area"
    enabled: true
    polygon:
      - [100, 100]
      - [700, 100]
      - [700, 650]
      - [100, 650]
```

## 6. Output files

The program writes to `output_dir`:

- `result.mp4` : annotated video
- `events.csv` : line/zone event log
- `summary.json` : final counting summary

## 7. Notes

- This template is designed for **single-camera pedestrian flow statistics**.
- DeepSORT is initialized as a separate tracker instance for one video stream.
- By default, DeepSORT uses a lightweight embedder on CPU to avoid competing with YOLO for limited GPU memory.

## 8. Thesis-friendly module mapping

- `detector/` -> pedestrian detection module
- `tracker/` -> multi-object tracking module
- `counting/` -> line/zone counting module
- `utils/visualization.py` -> visualization module
- `utils/io_utils.py` -> result saving module
