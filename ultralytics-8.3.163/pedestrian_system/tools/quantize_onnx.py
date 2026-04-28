from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to FP16 or INT8.")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX model path")
    parser.add_argument("--mode", type=str, choices=["fp16", "int8"], required=True, help="Quantization mode")
    parser.add_argument("--calibration-data", type=str, default="", help="Calibration data directory for INT8 static quantization")
    parser.add_argument("--input-name", type=str, default="images", help="Input tensor name")
    return parser.parse_args()


def convert_to_fp16(input_path: Path, output_path: Path) -> None:
    try:
        from onnxconverter_common.float16 import convert_float_to_float16
        import onnx
    except ImportError as exc:
        raise ImportError(
            "FP16 conversion requires onnx and onnxconverter-common. Install: pip install onnx onnxconverter-common"
        ) from exc

    model = onnx.load(str(input_path))
    fp16_model = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16_model, str(output_path))


def convert_to_int8(input_path: Path, output_path: Path, calibration_data: Path, input_name: str) -> None:
    try:
        import onnx
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        raise ImportError(
            "INT8 quantization requires onnx and onnxruntime. Install: pip install onnx onnxruntime"
        ) from exc

    if not calibration_data.exists():
        raise FileNotFoundError(f"Calibration data directory not found: {calibration_data}")

    class ImageCalibrationDataReader(CalibrationDataReader):
        def __init__(self, data_dir: Path):
            self.data = []
            for path in sorted(data_dir.glob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                self.data.append(path)
            self.iterator = iter(self.data)

        def get_next(self):
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise ImportError("INT8 calibration requires opencv-python and numpy") from exc

            try:
                image_path = next(self.iterator)
            except StopIteration:
                return None

            image = cv2.imread(str(image_path))
            if image is None:
                return self.get_next()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float32") / 255.0
            image = image.transpose(2, 0, 1)[None, ...]
            return {input_name: image}

    reader = ImageCalibrationDataReader(calibration_data)
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "fp16":
        convert_to_fp16(input_path, output_path)
        print(f"FP16 ONNX saved to: {output_path}")
    else:
        calibration_data = Path(args.calibration_data)
        if not args.calibration_data:
            raise ValueError("INT8 mode requires --calibration-data")
        convert_to_int8(input_path, output_path, calibration_data, args.input_name)
        print(f"INT8 ONNX saved to: {output_path}")


if __name__ == "__main__":
    main()