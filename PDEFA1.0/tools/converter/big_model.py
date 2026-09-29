#!/usr/bin/env python3
"""
BigDetector — 20M parametrelik detection modeli.
Backbone + detection head. Depthwise YOK, saf 3x3 conv.

Input:  [1, 3, 640, 640]
Output: [1, 21, 10, 10]
"""

import os
import numpy as np
import torch
import torch.nn as nn

from onnx_to_engine import convert


class BigDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Backbone (640 → 320 → 160 → 80 → 40 → 20 → 10)
        self.conv1 = nn.Conv2d(3,    32,  3, 2, 1)
        self.conv2 = nn.Conv2d(32,   64,  3, 2, 1)
        self.conv3 = nn.Conv2d(64,   128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128,  256, 3, 2, 1)
        self.conv5 = nn.Conv2d(256,  512, 3, 2, 1)
        self.conv6 = nn.Conv2d(512,  1024, 3, 2, 1)

        # Detection head
        head_ch = 3 * (5 + num_classes)  # 21
        self.head1 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.head2 = nn.Conv2d(512, 256, 3, 1, 1)
        self.head3 = nn.Conv2d(256, head_ch, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))   # 640 → 320
        x = self.relu(self.conv2(x))   # 320 → 160
        x = self.relu(self.conv3(x))   # 160 → 80
        x = self.relu(self.conv4(x))   # 80  → 40
        x = self.relu(self.conv5(x))   # 40  → 20
        x = self.relu(self.conv6(x))   # 20  → 10
        x = self.relu(self.head1(x))   # 10  → 10
        x = self.relu(self.head2(x))   # 10  → 10
        x = self.head3(x)              # 10  → 10 (out)
        return x


if __name__ == "__main__":
    model = BigDetector(num_classes=2).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total params: {total:,}")

    dummy = torch.randn(1, 3, 640, 640)

    # 1) ONNX
    print("\n[1/3] ONNX...")
    torch.onnx.export(
        model, dummy, "big_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print(f"      OK: big_model.onnx ({os.path.getsize('big_model.onnx')} bytes)")

    # 2) .engine
    print("\n[2/3] .engine...")
    convert("big_model.onnx", "big_model.engine")
    print(f"      OK: big_model.engine")

    # 3) Referanslar
    print("\n[3/3] Referanslar...")
    dummy.numpy().astype(np.float32).tofile("big_model_input.f32")
    with torch.no_grad():
        ref = model(dummy).numpy().astype(np.float32)
    ref.tofile("big_model_output.f32")
    print(f"      Input:  {tuple(dummy.shape)}, sum={dummy.numpy().sum():.4f}")
    print(f"      Output: {ref.shape}, sum={ref.sum():.6f}")

    print("\n" + "=" * 60)
    print("  HAZIR")
    print("=" * 60)