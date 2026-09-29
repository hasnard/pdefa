#!/usr/bin/env python3
"""
MiniDetector — Basit bir detection modeli.
Backbone + detection head. Post-processing yok.

Input:  [1, 3, 640, 640]
Output: [1, 3*(5+num_classes), 20, 20]
"""

import torch
import torch.nn as nn

from onnx_to_engine import convert


class MiniDetector(nn.Module):
    """
    5 katmanlı backbone + tek detection head.
    Her conv 3x3, stride=2 → 640 → 320 → 160 → 80 → 40 → 20
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Backbone
        self.conv1 = nn.Conv2d(3,    16,  3, 2, 1)
        self.conv2 = nn.Conv2d(16,   32,  3, 2, 1)
        self.conv3 = nn.Conv2d(32,   64,  3, 2, 1)
        self.conv4 = nn.Conv2d(64,   128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128,  256, 3, 2, 1)
        self.relu = nn.ReLU()

        # Detection head
        # 3 anchor * (4 bbox + 1 objectness + num_classes)
        head_ch = 3 * (5 + num_classes)
        self.head = nn.Conv2d(256, head_ch, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = MiniDetector(num_classes=2).eval()
    dummy = torch.randn(1, 3, 640, 640)

    # 1) ONNX export
    print("📤 PyTorch → ONNX...")
    torch.onnx.export(
        model, dummy, "mini_detector.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print("✅ mini_detector.onnx")

    # 2) ONNX → .engine
    print("\n📤 ONNX → .engine...")
    convert("mini_detector.onnx", "mini_detector.engine")
    print("\n✅ mini_detector.engine hazır")