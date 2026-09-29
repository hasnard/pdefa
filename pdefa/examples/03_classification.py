# pdefa - AGPL-3.0-or-later - see LICENSE
"""ImageNet classification example.

Usage:
    python 03_classification.py <resnet18.engine> <image.jpg>
"""
import sys
import pdefa

if len(sys.argv) < 3:
    print("usage: python 03_classification.py <model.engine> <image>")
    sys.exit(1)

ENGINE, IMAGE = sys.argv[1], sys.argv[2]

ctx = pdefa.Context()
model = pdefa.Model(ctx, ENGINE)
session = pdefa.Session(ctx, model)

tensor, _ = pdefa.prepare_for_engine(IMAGE, 224, 224, scale=1.0 / 255.0)
session.set_input(0, tensor)
session.run()

out = session.output(0).tolist()
flat = out[0]

top5 = sorted(range(len(flat)), key=lambda i: flat[i], reverse=True)[:5]
print("Top-5:")
for rank, idx in enumerate(top5, 1):
    print(f"  {rank}. class={idx:4d}  prob={flat[idx]:.4f}")