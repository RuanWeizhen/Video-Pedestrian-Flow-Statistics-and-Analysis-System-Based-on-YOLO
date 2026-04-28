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

## 9. ONNX export, quantization, and benchmark

This project now ships standalone tooling under `tools/`.

### 9.1 Export ONNX

```bash
python tools/export_onnx.py --weights "runs/train/your_best.pt" --imgsz 800 --simplify
```

Optional flags:

- `--dynamic`: export with dynamic shape/batch support
- `--half`: export fp16 ONNX when supported by the exporter
- `--opset`: choose ONNX opset, default `17`

### 9.2 FP16 conversion

```bash
python tools/quantize_onnx.py --input exports/your_best/weights/best.onnx --output exports/your_best_fp16.onnx --mode fp16
```

### 9.3 INT8 quantization

Static INT8 quantization needs a calibration image directory.

```bash
python tools/quantize_onnx.py --input exports/your_best/weights/best.onnx --output exports/your_best_int8.onnx --mode int8 --calibration-data datasets/calibration_images
```

Recommended calibration data:

- 100-500 representative images
- same scene distribution as deployment if possible
- include day/night, occlusion, and far-distance samples

### 9.4 Benchmark ONNX Runtime

```bash
python tools/benchmark_onnx.py --model exports/your_best_fp16.onnx --source videos/test.mp4 --imgsz 800 --warmup 20 --limit 200
```

If CUDA provider is installed in ONNX Runtime, you can force providers:

```bash
python tools/benchmark_onnx.py --model exports/your_best_fp16.onnx --source videos/test.mp4 --providers CUDAExecutionProvider,CPUExecutionProvider
```

### 9.5 Suggested pip packages

For export and quantization, install:

```bash
pip install onnx onnxruntime onnxconverter-common
```

For GPU benchmark with ONNX Runtime, install the CUDA-enabled wheel that matches your platform.

### 9.6 Workflow summary

1. Export `.pt` to ONNX.
2. Convert ONNX to FP16 or INT8.
3. Benchmark the exported model with `benchmark_onnx.py`.
4. Compare latency/FPS with the PyTorch pipeline.

## 8. Thesis-friendly module mapping

- `detector/` -> pedestrian detection module
- `tracker/` -> multi-object tracking module
- `counting/` -> line/zone counting module
- `utils/visualization.py` -> visualization module
- `utils/io_utils.py` -> result saving module
