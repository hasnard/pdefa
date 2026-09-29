"""pdefa YOLO inference example.

Usage:
    python examples/yolo_inference.py yolov8n.engine bus.jpg
    python examples/yolo_inference.py yolov8n.engine earth.png
"""

import sys
import pdefa

COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python examples/yolo_inference.py <model.engine> <image.jpg>")
        return 1

    engine_path, image_path = sys.argv[1], sys.argv[2]

    print(f"pdefa {pdefa.version()}  |  {pdefa.simd_level()}  |  {pdefa.cpu_brand()}")

    ctx     = pdefa.Context()
    model   = pdefa.Model(ctx, engine_path)
    session = pdefa.Session(ctx, model)

    tensor, meta = pdefa.prepare_for_model(model, image_path)
    session.set_input(0, tensor)
    session.run()
    out = session.output(0)

    dets = pdefa.decode_yolo(out, meta, conf_thres=0.25, iou_thres=0.45)
    ms   = session.stats()["last_run_ms"]

    print(f"Image      : {image_path}  ({meta.orig_w}x{meta.orig_h})")
    print(f"Inference  : {ms:.2f} ms")
    print(f"Detections : {len(dets)}")
    print("-" * 60)

    for i, d in enumerate(dets):
        cls_id = int(d["cls"])
        name = COCO[cls_id] if 0 <= cls_id < len(COCO) else f"cls{cls_id}"
        print(f"{i:<3} {name:<14} conf={d['conf']:.3f}  "
              f"box=({d['x1']:.0f}, {d['y1']:.0f}, {d['x2']:.0f}, {d['y2']:.0f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())