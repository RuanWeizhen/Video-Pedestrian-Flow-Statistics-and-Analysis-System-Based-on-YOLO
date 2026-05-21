from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to FP16.")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX model path")
    parser.add_argument("--mode", type=str, choices=["fp16"], default="fp16", help="Quantization mode (only fp16 supported)")
    parser.add_argument("--input-name", type=str, default="images", help=argparse.SUPPRESS)
    return parser.parse_args()


def _count_fp_weights(model) -> tuple[int, int]:
    fp16 = 0
    fp32 = 0
    for init in model.graph.initializer:
        num_elems = max(1, int(np.prod(init.dims) if init.dims else 0))
        raw_len = len(init.raw_data) if init.raw_data else 0
        if raw_len == num_elems * 2:
            fp16 += 1
        elif raw_len == num_elems * 4:
            fp32 += 1
    return fp16, fp32


def convert_to_fp16(input_path: Path, output_path: Path) -> None:
    model = onnx.load(str(input_path))

    fp16_count, fp32_count = _count_fp_weights(model)
    print(f"模型权重: FP16={fp16_count}, FP32={fp32_count}")

    if fp32_count == 0 and fp16_count > 0:
        print("模型已为纯 FP16，直接复制到输出路径")
        onnx.save(model, str(output_path))
        return

    print("开始转换 FP32 → FP16...")

    graph = model.graph
    converted_names = set()
    new_initializers = []

    for init in graph.initializer:
        num_elems = max(1, int(np.prod(init.dims) if init.dims else 0))
        raw_len = len(init.raw_data) if init.raw_data else 0
        if raw_len == num_elems * 4:
            arr = numpy_helper.to_array(init)
            new_initializers.append(numpy_helper.from_array(arr.astype(np.float16), name=init.name))
            converted_names.add(init.name)
        else:
            new_initializers.append(init)

    print(f"已转换 {len(converted_names)} 个权重为 FP16，正在插入 Cast 节点...")

    all_initializer_names = {init.name for init in graph.initializer}

    new_nodes = []
    cast_counter = 0

    for node in graph.node:
        new_inputs = list(node.input)

        for idx, inp in enumerate(node.input):
            if inp in converted_names:
                cast_name = f"/quant/Cast_FP16toFP32_{cast_counter}"
                cast_output = f"{inp}_fp32_cast_{cast_counter}"
                cast_counter += 1

                cast_node = helper.make_node(
                    "Cast",
                    inputs=[inp],
                    outputs=[cast_output],
                    name=cast_name,
                    to=onnx.TensorProto.FLOAT,
                )
                new_nodes.append(cast_node)
                new_inputs[idx] = cast_output

        new_nodes.append(helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name,
            **{attr.name: helper.get_attribute_value(attr) for attr in node.attribute},
        ))

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=new_initializers,
        value_info=list(graph.value_info),
    )

    new_model = helper.make_model(new_graph, producer_name="quantize_onnx_fp16")
    new_model.ir_version = model.ir_version
    while len(new_model.opset_import) > 0:
        new_model.opset_import.pop()
    for opset in model.opset_import:
        new_model.opset_import.add().CopyFrom(opset)

    onnx.save(new_model, str(output_path))

    final_fp16, final_fp32 = _count_fp_weights(new_model)
    print(f"转换完成: FP16={final_fp16}, FP32={final_fp32}, Cast 节点数={cast_counter}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_to_fp16(input_path, output_path)
    print(f"FP16 ONNX saved to: {output_path}")


if __name__ == "__main__":
    main()
