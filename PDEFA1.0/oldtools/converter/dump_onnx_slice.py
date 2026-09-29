import onnx
from onnx import numpy_helper

m = onnx.load("yolov8s.onnx")
print(f"ir_version={m.ir_version} opset={[o.version for o in m.opset_import]}")

init_map = {i.name: i for i in m.graph.initializer}

for n in m.graph.node:
    if n.op_type == "Slice":
        print(f"\n=== Slice: '{n.name}' ===")
        for i, inp_name in enumerate(n.input):
            if inp_name in init_map:
                arr = numpy_helper.to_array(init_map[inp_name])
                print(f"  in[{i}] '{inp_name}': dtype={arr.dtype} shape={arr.shape} values={arr.tolist()}")
            else:
                print(f"  in[{i}] '{inp_name}': <runtime>")