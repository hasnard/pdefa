import onnx
m = onnx.load('yolov8s.onnx')
print(f"Toplam node: {len(m.graph.node)}")
for i, n in enumerate(m.graph.node):
    if i >= 180:
        print(f'  {i:3d}  {n.op_type:12s}  in={list(n.input)}  out={list(n.output)}  name={n.name}')