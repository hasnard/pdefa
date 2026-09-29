# pdefa - AGPL-3.0-or-later - see LICENSE
"""Segmentation example.

Usage:
    python 04_segmentation.py <unet.engine> <image.jpg>
"""
import sys
import pdefa

if len(sys.argv) < 3:
    print("usage: python 04_segmentation.py <model.engine> <image>")
    sys.exit(1)

ENGINE, IMAGE = sys.argv[1], sys.argv[2]

ctx = pdefa.Context()
model = pdefa.Model(ctx, ENGINE)
session = pdefa.Session(ctx, model)

tensor, meta = pdefa.prepare_for_model(model, IMAGE)
session.set_input(0, tensor)
session.run()

mask = session.output(0)
print(f"Mask shape: {mask.shape}")
arr = mask.tolist()
print(f"First values: {arr[0][0][:4] if arr[0] else 'empty'}")