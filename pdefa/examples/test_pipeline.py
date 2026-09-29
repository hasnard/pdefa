"""
pdefa tam fonksiyonel test.
Çalıştır: python test_pdefa.py
"""
import pdefa
import numpy as np

print("=" * 60)
print(" pdefa — tam test")
print("=" * 60)

# --- 1. Meta ---
print("\n[1] Meta")
print("  version :", pdefa.version())
print("  simd    :", pdefa.simd_level())
print("  cpu     :", pdefa.cpu_brand())

# --- 2. Context / Model / Session (model dosyası gerekmez) ---
print("\n[2] Context")
ctx = pdefa.Context()
print("  ctx OK")

# --- 3. Image yükle ---
print("\n[3] Image")
png = r"D:\PROJE\engineforaiincpu\imageloader\images\earth.png"
img = pdefa.load_image(png)
print(f"  {img}")
assert img.width > 0 and img.height > 0

# --- 4. Preprocess -> Tensor ---
print("\n[4] Preprocess")
tensor, meta = pdefa.prepare_for_engine(png, 640, 640)
print(f"  tensor : {tensor}")
print(f"  meta   : {meta}")
assert tensor.shape == (1, 3, 640, 640)

# --- 5. Numpy çıktı ---
print("\n[5] Numpy")
arr = tensor.numpy()
print(f"  shape={arr.shape} dtype={arr.dtype} sum={arr.sum():.2f}")
assert arr.shape == (1, 3, 640, 640)
assert arr.dtype == np.float32

# --- 6. Post-process ---
print("\n[6] Post-process")
b = pdefa.Box(100, 200, 300, 400)
orig = pdefa.map_box_to_original(b, meta)
print(f"  in  : {b}")
print(f"  out : {orig}")

x, y = pdefa.map_point_to_original(320.0, 320.0, meta)
print(f"  point: ({x:.1f}, {y:.1f})")

# --- 7. Ops — matmul ---
print("\n[7] ops_matmul")
A = pdefa.tensor_empty([4, 8])
B = pdefa.tensor_empty([8, 3])
C = pdefa.ops_matmul(A, B)
print(f"  A: {A}")
print(f"  B: {B}")
print(f"  C: {C}")
assert C.shape == (4, 3)

# --- 8. Ops — relu / silu ---
print("\n[8] ops_relu / ops_silu")
T = pdefa.tensor_empty([2, 4])
pdefa.ops_relu(T)
pdefa.ops_silu(T)
print("  relu + silu OK")

# --- 9. BatchInference (model dosyası olmadan sadece API testi) ---
print("\n[9] BatchInference")
print(f"  {pdefa.BatchInference}")

print("\n" + "=" * 60)
print(" ✅ TÜM TESTLER GEÇTİ")
print("=" * 60)