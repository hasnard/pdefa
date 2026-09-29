#!/usr/bin/env python3
"""Conv+BN+ReLU içeren test modeli. BN fusion doğrulaması için."""

import numpy as np
import torch
import torch.nn as nn

from onnx_to_engine import convert


class BNCNN(nn.Module):
    """Conv → BatchNorm → ReLU zinciri — 3 katman."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)   # bias YOK — BN ekliyor
        self.bn1   = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(2)
        self.fc   = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = BNCNN().eval()

    # Rastgele ağırlıklar (BN running stats dahil)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.1, 0.1)
                m.running_var.uniform_(0.9, 1.1)
                m.weight.uniform_(0.8, 1.2)
                m.bias.uniform_(-0.1, 0.1)

    dummy = torch.randn(1, 3, 64, 64)

    print("📤 PyTorch → ONNX...")
    torch.onnx.export(
        model, dummy, "bn_cnn.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print("✅ bn_cnn.onnx")

    print("\n📤 ONNX → .engine...")
    convert("bn_cnn.onnx", "bn_cnn.engine")

    # Referans çıktı
    with torch.no_grad():
        ref = model(dummy).numpy()
    dummy.numpy().astype(np.float32).tofile("bn_cnn_input.f32")
    ref.astype(np.float32).tofile("bn_cnn_output.f32")

    print(f"   Input:  {tuple(dummy.shape)}, sum={dummy.numpy().sum():.4f}")
    print(f"   Output: {ref.shape}, sum={ref.sum():.4f}")
    print("✅ bn_cnn.engine hazır")