#!/usr/bin/env python3
"""
PyTorch -> ONNX (one-liner wrapper around torch.onnx.export).

Usage:
    python pytorch_to_onnx.py model.pt model.onnx --input 1,3,640,640
    python pytorch_to_onnx.py model.pt model.onnx                # default 1,3,640,640
"""
import argparse
import os
import sys

import torch


def export(pt_path, onnx_path, input_shape=(1, 3, 640, 640), opset=17):
    # Load
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Unwrap Ultralytics-style checkpoint
    if isinstance(obj, dict):
        for key in ("model", "net", "module", "ema"):
            if key in obj and isinstance(obj[key], torch.nn.Module):
                obj = obj[key]
                break
        else:
            raise TypeError(f"No nn.Module in dict. Keys: {list(obj.keys())}")

    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(obj).__name__}")

    obj = obj.float()          # FP16 -> FP32 (Ultralytics fix)
    obj.eval()
    dummy = torch.randn(*input_shape)

    torch.onnx.export(
        obj, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )

    size = os.path.getsize(onnx_path)
    print(f"Wrote: {onnx_path} ({size:,} bytes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pt", help="Input .pt / .pth")
    p.add_argument("onnx", help="Output .onnx")
    p.add_argument("--input", default="1,3,640,640",
                   help="Input shape (default 1,3,640,640)")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    shape = tuple(int(x) for x in args.input.split(","))
    export(args.pt, args.onnx, input_shape=shape, opset=args.opset)


if __name__ == "__main__":
    main()