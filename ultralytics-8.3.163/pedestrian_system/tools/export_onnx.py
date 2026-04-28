from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX.")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights file")
    parser.add_argument("--imgsz", type=int, default=800, help="Export image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Run ONNX simplification")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic batch/shape")
    parser.add_argument("--half", action="store_true", help="Export fp16 ONNX if supported")
    parser.add_argument("--output-dir", type=str, default="exports", help="Directory for exported artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        project=str(output_dir),
        name=weights_path.stem,
    )

    print(f"Exported ONNX: {exported_path}")


if __name__ == "__main__":
    main()