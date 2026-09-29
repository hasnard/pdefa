# pdefa - AGPL-3.0-or-later - see LICENSE
"""Batch inference example.

Usage:
    python 05_batch_inference.py <model.engine> <img1> <img2> [...]
"""
import sys
import pdefa

if len(sys.argv) < 3:
    print("usage: python 05_batch_inference.py <model.engine> <img> [<img> ...]")
    sys.exit(1)

ENGINE = sys.argv[1]
IMAGES = sys.argv[2:]

ctx = pdefa.Context()
model = pdefa.Model(ctx, ENGINE)

with pdefa.BatchInference(ctx, model, batch_size=4, num_threads=4) as batch:
    results = batch.run(IMAGES)
    for i, (out_tensor, meta) in enumerate(results):
        print(f"[{i}] {IMAGES[i]}  out shape={out_tensor.shape}")