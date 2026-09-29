# pdefa - AGPL-3.0-or-later - see LICENSE
"""Manual smoke test during development.

Usage:
    python tools/manual_test.py <image.png>
"""
import sys
import pdefa

IMG = sys.argv[1] if len(sys.argv) > 1 else "test.png"

print("=" * 60)
print(" pdefa — manual test")
print("=" * 60)

print("\n[1] Meta")
print("  version :", pdefa.version())
print("  license :", pdefa.__license__)
print("  simd    :", pdefa.simd_level())
print("  cpu     :", pdefa.cpu_brand())

print("\n[2] Context")
ctx = pdefa.Context()
print("  ctx OK")

print("\n[3] Image")
img = pdefa.load_image(IMG)
print(f"  {img}")

print("\n[4] Preprocess")
tensor, meta = pdefa.prepare_for_engine(IMG, 640, 640)
print(f"  tensor : {tensor}")
print(f"  meta   : {meta}")

print("\n[5] Tensor access (no numpy)")
arr = tensor.tolist()
print(f"  shape={tensor.shape}  sample={arr[0][0][0][:4]}")

print("\n[6] Post-process")
b = pdefa.Box(100, 200, 300, 400)
print(f"  in  : {b}")
print(f"  out : {pdefa.map_box_to_original(b, meta)}")
x, y = pdefa.map_point_to_original(320.0, 320.0, meta)
print(f"  point: ({x:.1f}, {y:.1f})")

print("\n[7] Ops")
A = pdefa.zeros(4, 8)
B = pdefa.zeros(8, 3)
C = pdefa.ops_matmul(A, B)
print(f"  matmul C shape: {C.shape}")

T = pdefa.randn(2, 4, seed=0)
pdefa.ops_relu(T)
pdefa.ops_silu(T)
print(f"  relu+silu OK: {T.tolist()}")

print("\n" + "=" * 60)
print(" ✅ DONE")
print("=" * 60)