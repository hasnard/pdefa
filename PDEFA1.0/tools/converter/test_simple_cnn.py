#!/usr/bin/env python3
"""Basit CNN — PyTorch → ONNX → .engine"""
import numpy as np
import torch
import torch.nn as nn

from onnx_to_engine import convert


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN().eval()

    # Dummy input
    dummy = torch.randn(1, 3, 64, 64)

    # 1) PyTorch → ONNX
    print("📤 PyTorch → ONNX...")
    torch.onnx.export(
        model, dummy, "simple_cnn.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print("✅ simple_cnn.onnx")

    # 2) ONNX → .engine
    print("\n📤 ONNX → .engine...")
    convert("simple_cnn.onnx", "simple_cnn.engine")
    print("\n✅ simple_cnn.engine hazır")

        # 3) Referans input + output kaydet
    with torch.no_grad():
        # Ara katman çıktıları
        x = dummy
        x1 = model.conv1(x)      # Conv2d 1
        x2 = model.relu1(x1)     # ReLU 1
        x3 = model.pool1(x2)     # MaxPool 1
        x4 = model.conv2(x3)     # Conv2d 2
        x5 = model.relu2(x4)     # ReLU 2
        x6 = model.pool2(x5)     # MaxPool 2
        ref_output = model(dummy)
    
    # Kaydet
    dummy.numpy().astype(np.float32).tofile("simple_cnn_input.f32")
    ref_output.numpy().astype(np.float32).tofile("simple_cnn_output.f32")
    
    x1.numpy().astype(np.float32).tofile("layer_conv1.f32")
    x3.numpy().astype(np.float32).tofile("layer_pool1.f32")
    x6.numpy().astype(np.float32).tofile("layer_pool2.f32")
    
    print(f"   Input:   shape={tuple(dummy.shape)}, sum={dummy.numpy().sum():.4f}")
    print(f"   Conv1:   shape={tuple(x1.shape)}, sum={x1.numpy().sum():.4f}")
    print(f"   Pool1:   shape={tuple(x3.shape)}, sum={x3.numpy().sum():.4f}")
    print(f"   Pool2:   shape={tuple(x6.shape)}, sum={x6.numpy().sum():.4f}")
    print(f"   Output:  shape={tuple(ref_output.shape)}, sum={ref_output.numpy().sum():.4f}")
    print(f"   Kaydedildi: layer_conv1.f32, layer_pool1.f32, layer_pool2.f32")