# pdefa - AGPL-3.0-or-later - see LICENSE
"""Per-channel error diagnosis.

Usage:
    python diagnose_channel.py <engine> <pt> <image>
"""
import sys
import numpy as np
import torch
import pdefa
from ultralytics import YOLO

if len(sys.argv) < 4:
    print("usage: diagnose_channel.py <engine> <pt> <image>")
    sys.exit(1)

ENG, PT, PNG = sys.argv[1], sys.argv[2], sys.argv[3]

model = YOLO(PT).model.eval().fuse()
tensor, _ = pdefa.prepare_for_engine(PNG, 640, 640)
np_in = np.asarray(tensor.tolist(), dtype=np.float32)

with torch.no_grad():
    pt_out = model(torch.from_numpy(np_in))
    if isinstance(pt_out, (list, tuple)):
        pt_out = pt_out[0]
    pt_out = pt_out.numpy()

ctx = pdefa.Context()
m = pdefa.Model(ctx, ENG)
s = pdefa.Session(ctx, m)
s.set_input(0, tensor)
s.run()
eng_out = np.asarray(s.output(0).tolist(), dtype=np.float32)

CH = 5
print(f"ch{CH}:")
print(f"  PT  min={pt_out[0, CH].min():.6f}  max={pt_out[0, CH].max():.6f}  mean={pt_out[0, CH].mean():.6f}")
print(f"  Eng min={eng_out[0, CH].min():.6f}  max={eng_out[0, CH].max():.6f}  mean={eng_out[0, CH].mean():.6f}")

diff = np.abs(pt_out[0] - eng_out[0])
top = np.argsort(diff.flatten())[::-1][:10]
print("\nTop-10 largest diffs:")
for idx in top:
    c, p = divmod(idx, diff.shape[1])
    print(f"  ch{c:3d} pos{p:5d}: PT={pt_out[0, c, p]:10.4f}  "
          f"Eng={eng_out[0, c, p]:10.4f}  diff={diff[c, p]:.4f}")