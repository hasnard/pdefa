# pdefa - AGPL-3.0-or-later - see LICENSE
"""Simplest YOLO detection example.

Usage:
    python 01_simple_detection.py <model.engine> <image.jpg>
"""
import sys
import pdefa

if len(sys.argv) < 3:
    print("usage: python 01_simple_detection.py <model.engine> <image>")
    sys.exit(1)

ENGINE, IMAGE = sys.argv[1], sys.argv[2]

model = pdefa.load(ENGINE)
boxes = model(IMAGE, conf_thres=0.25)

print(f"{len(boxes)} detections:")
for b in boxes:
    print(f"  cls={b['cls']}  conf={b['conf']:.3f}  "
          f"box=({b['x1']:.0f},{b['y1']:.0f})-({b['x2']:.0f},{b['y2']:.0f})")