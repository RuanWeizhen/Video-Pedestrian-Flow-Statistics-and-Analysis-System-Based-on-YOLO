from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for key in list(sys.modules.keys()):
    if key == "ultralytics" or key.startswith("ultralytics."):
        del sys.modules[key]

import ultralytics.nn.modules.block  # noqa: E402,F401 - 注册 EMA 等自定义模块供 torch.load 反序列化


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


def _check_dependencies(simplify: bool) -> None:
    missing = []
    try:
        import onnx  # noqa: F401
        print(f"  onnx: OK (version {onnx.__version__})")
    except ImportError:
        missing.append("onnx")

    try:
        import onnxruntime  # noqa: F401
        print(f"  onnxruntime: OK (version {onnxruntime.__version__})")
    except ImportError:
        missing.append("onnxruntime")

    if simplify:
        try:
            import onnxsim  # noqa: F401
            print(f"  onnxsim: OK (version {onnxsim.__version__})")
        except ImportError:
            missing.append("onnxsim (required for --simplify)")

    if missing:
        raise ImportError(f"缺少依赖: {', '.join(missing)}。请运行: pip install {' '.join(missing)}")


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"ERROR: 权重文件不存在: {weights_path}")
        sys.exit(1)

    print(f"=== ONNX 导出开始 ===")
    print(f"权重文件: {weights_path}")
    print(f"imgsz: {args.imgsz}")
    print(f"opset: {args.opset}")
    print(f"simplify: {args.simplify}")
    print(f"dynamic: {args.dynamic}")
    print(f"half: {args.half}")

    _check_dependencies(args.simplify)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    try:
        from ultralytics import YOLO

        print(f"加载模型: {weights_path}")
        model = YOLO(str(weights_path))
        print("模型加载完成，开始导出...")

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

        exported_path = Path(str(exported_path)).resolve()
        print(f"Ultralytics 报告导出路径: {exported_path}")

        if not exported_path.exists():
            print(f"WARNING: Ultralytics 报告路径不存在，尝试在输出目录中搜索 .onnx 文件...")
            onnx_files = list(output_dir.rglob("*.onnx"))
            if onnx_files:
                exported_path = onnx_files[0].resolve()
                print(f"  找到: {exported_path}")
            else:
                print("ERROR: 未找到任何 .onnx 文件，导出可能失败")
                sys.exit(1)

        target_path = output_dir / f"{weights_path.stem}.onnx"
        if exported_path != target_path:
            print(f"扁平化输出: {exported_path} -> {target_path}")
            shutil.copy2(exported_path, target_path)
            # 如果导出的 ONNX 文件名与 weights 名不同（如 best.onnx vs 实际导出名）
            if exported_path.name != target_path.name:
                print(f"  注意: 实际导出文件名为 {exported_path.name}，已复制为 {target_path.name}")
        else:
            print(f"ONNX 文件已位于正确位置: {target_path}")

        file_size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"SUCCESS: ONNX 文件已生成")
        print(f"  路径: {target_path}")
        print(f"  大小: {file_size_mb:.2f} MB")

        if getattr(model, "ckpt", None) is not None:
            print("该模型包含 EMA 权重，已自动使用 EMA 权重导出" if args.half else "")

    except Exception as exc:
        print(f"ERROR: ONNX 导出失败")
        print(f"异常类型: {type(exc).__name__}")
        print(f"异常信息: {exc}")
        print(f"完整 Traceback:")
        traceback.print_exc()

        print(f"\n=== 诊断建议 ===")
        print(f"1. 检查 onnx/onnxruntime 版本: pip list | findstr onnx")
        print(f"2. 如果 --simplify 失败，先尝试不加该参数导出")
        print(f"3. 如果 --half 失败，先尝试不加该参数导出")
        print(f"4. 如果显存不足，降低 --imgsz 值（如 640）")
        print(f"5. 确认模型路径正确且 .pt 文件完整")
        sys.exit(1)


if __name__ == "__main__":
    main()
