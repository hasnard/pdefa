# D:\png\diagnose_ch5.py
import numpy as np, torch, pdefa
from ultralytics import YOLO

PNG = r"D:\png\sayfa_1.png"
PT  = r"D:\png\yolov8n.pt"
ENG = r"D:\png\yolov8n.engine"

model = YOLO(PT).model.eval().fuse()
tensor, _ = pdefa.prepare_for_engine(PNG, 640, 640)
with torch.no_grad():
    pt_out = model(torch.from_numpy(tensor.numpy().copy()))
    if isinstance(pt_out, (list, tuple)): pt_out = pt_out[0]
    pt_out = pt_out.numpy()

ctx = pdefa.Context(); m = pdefa.Model(ctx, ENG); s = pdefa.Session(ctx, m)
s.set_input(0, tensor); s.run()
eng_out = s.output(0).numpy()

print("ch5 degerleri:")
print(f"  PT  : min={pt_out[0,5].min():.6f}  max={pt_out[0,5].max():.6f}  mean={pt_out[0,5].mean():.6f}")
print(f"  Eng : min={eng_out[0,5].min():.6f}  max={eng_out[0,5].max():.6f}  mean={eng_out[0,5].mean():.6f}")

# En buyuk 10 fark nerede?
diff = np.abs(pt_out[0] - eng_out[0])
flat = np.argsort(diff.flatten())[::-1][:10]
print("\nEn buyuk 10 fark:")
for idx in flat:
    c, p = divmod(idx, 8400)
    print(f"  ch{c:3d} pos{p:5d}: PT={pt_out[0,c,p]:10.4f}  Eng={eng_out[0,c,p]:10.4f}  diff={diff[c,p]:.4f}")