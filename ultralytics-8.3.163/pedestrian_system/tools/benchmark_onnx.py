from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX model on a video source.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, required=True, help="Video file path or camera index")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames")
    parser.add_argument("--limit", type=int, default=300, help="Maximum frames to benchmark")
    parser.add_argument("--providers", type=str, default="", help="Comma-separated ONNX Runtime providers")
    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def letterbox(image: np.ndarray, new_shape: int = 800, color=(114, 114, 114)):
    shape = image.shape[:2]
    ratio = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def build_session(model_path: Path, providers: list[str]):
    import onnxruntime as ort

    if providers:
        return ort.InferenceSession(str(model_path), providers=providers)

    available = ort.get_available_providers()
    preferred = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
    return ort.InferenceSession(str(model_path), providers=preferred or available)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    session = build_session(model_path, providers)

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    warmup = max(0, int(args.warmup))
    limit = max(1, int(args.limit))
    total_time = 0.0
    processed = 0

    while processed < warmup + limit:
        ok, frame = cap.read()
        if not ok:
            break

        img, _, _ = letterbox(frame, args.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        start = time.perf_counter()
        _ = session.run(output_names, {input_name: img})
        elapsed = time.perf_counter() - start

        if processed >= warmup:
            total_time += elapsed

        processed += 1

    cap.release()

    benchmark_frames = max(0, processed - warmup)
    avg_latency_ms = (total_time / benchmark_frames * 1000.0) if benchmark_frames > 0 else 0.0
    fps = (benchmark_frames / total_time) if total_time > 0 else 0.0

    print("Benchmark finished")
    print(f"Model: {model_path}")
    print(f"Frames benchmarked: {benchmark_frames}")
    print(f"Average latency: {avg_latency_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Providers: {session.get_providers()}")


if __name__ == "__main__":
    main()