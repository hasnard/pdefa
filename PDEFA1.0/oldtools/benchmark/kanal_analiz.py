import numpy as np
pt  = np.fromfile("../converter/yolov8s_pt.f32",  dtype=np.float32).reshape(84, 8400)
eng = np.fromfile("../converter/yolov8s_eng.f32", dtype=np.float32).reshape(84, 8400)

print(f"pt  [0:5, 0:5]:\n{pt[:5, :5]}")
print(f"eng [0:5, 0:5]:\n{eng[:5, :5]}")
print()
print(f"{'ch':>3} {'cos':>8} {'pt_sum':>14} {'eng_sum':>14} {'pt_max':>10} {'eng_max':>10}")
for ch in range(84):
    c = np.dot(pt[ch], eng[ch]) / (np.linalg.norm(pt[ch]) * np.linalg.norm(eng[ch]) + 1e-9)
    print(f"{ch:>3} {c:>+8.4f} {pt[ch].sum():>14.2f} {eng[ch].sum():>14.2f} {pt[ch].max():>10.3f} {eng[ch].max():>10.3f}")