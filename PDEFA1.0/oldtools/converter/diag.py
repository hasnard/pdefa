import onnx
from onnx import numpy_helper
import numpy as np

print("=== Engine vs PyTorch çıktı ===")
ref = np.fromfile('yolov8s_pt.f32', dtype=np.float32).reshape(1, 84, 8400)
eng = np.fromfile('yolov8s_eng.f32', dtype=np.float32).reshape(1, 84, 8400)
print(f"  ref: min={ref.min():.3f} max={ref.max():.3f} mean={ref.mean():.3f}")
print(f"  eng: min={eng.min():.3f} max={eng.max():.3f} mean={eng.mean():.3f}")

print()
print("=== Kanal başına oran (ref vs eng) ===")
for c in [0, 1, 2, 3, 4, 40, 80, 83]:
    r = ref[0, c, :]
    e = eng[0, c, :]
    rm, em = r.mean(), e.mean()
    ratio = em / rm if abs(rm) > 1e-6 else 0
    print(f"  ch{c:3d}: ref.mean={rm:9.3f}  eng.mean={em:9.3f}  oran={ratio:6.3f}")

print()
print("=== İlk 10 değer, kanal 0 (x_center) ===")
print(f"  ref: {[f'{v:.2f}' for v in ref[0,0,:10]]}")
print(f"  eng: {[f'{v:.2f}' for v in eng[0,0,:10]]}")

print()
print("=== ONNX /model.22 kritik Constant değerleri ===")
m = onnx.load('yolov8s.onnx')
for n in m.graph.node:
    if n.op_type != "Constant": continue
    if not n.name.startswith('/model.22/'): continue
    for a in n.attribute:
        if a.name == "value":
            arr = numpy_helper.to_array(a.t)
            nf = arr.size
            if nf <= 24:
                print(f"  {n.name:42s} shape={str(arr.shape):15s} {arr.flatten()[:20].tolist()}")
            else:
                print(f"  {n.name:42s} shape={str(arr.shape):15s} nfloats={nf}  first8={arr.flatten()[:8].tolist()}")
        elif a.name in ("value_int", "value_float"):
            v = a.i if a.name == "value_int" else a.f
            print(f"  {n.name:42s} {a.name} = {v}")
        elif a.name == "value_ints":
            print(f"  {n.name:42s} value_ints = {list(a.ints)}")