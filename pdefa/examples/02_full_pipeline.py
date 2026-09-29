# pdefa - AGPL-3.0-or-later - see LICENSE
"""Full pipeline with explicit control.

Usage:
    python 02_full_pipeline.py <model.engine> <image>
"""
import sys
import pdefa

if len(sys.argv) < 3:
    print("usage: python 02_full_pipeline.py <model.engine> <image>")
    sys.exit(1)

ENGINE, IMAGE = sys.argv[1], sys.argv[2]

ctx = pdefa.Context()
model = pdefa.Model(ctx, ENGINE)
session = pdefa.Session(ctx, model)

info = model.input_info(0)
print(f"Model input : {info.name} {info.shape}")

tensor, meta = pdefa.prepare_for_model(model, IMAGE)
print(f"Tensor      : {tensor.shape}")
print(f"Meta        : orig={meta.orig_w}x{meta.orig_h}  ratio={meta.ratio:.3f}")

session.set_input(0, tensor)
session.run()
out = session.output(0)
print(f"Output      : shape={out.shape}")

boxes = pdefa.decode_yolo(out, meta, conf_thres=0.25, iou_thres=0.45)
print(f"\n{len(boxes)} detections:")
for b in boxes[:10]:
    print(f"  cls={b['cls']:2d}  conf={b['conf']:.3f}  "
          f"box=({b['x1']:.0f},{b['y1']:.0f})-({b['x2']:.0f},{b['y2']:.0f})")

boxes2 = session.run_and_decode(meta)
print(f"\nShortcut    : {len(boxes2)} detections")