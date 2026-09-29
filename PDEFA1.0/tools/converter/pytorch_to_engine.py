#!/usr/bin/env python3
"""
Engine-AI — PyTorch → .engine format converter

Kullanım:
    python pytorch_to_engine.py model.pt model.engine
"""

import struct
import sys
import os
from pathlib import Path
import torch

# ---- Sabitler (C++ tarafıyla eşleşmeli) ----
MAGIC   = 0x4E474E45  # "ENGN"
VERSION = 1

OP_CONV2D     = 1
OP_MATMUL     = 2
OP_ADD        = 3
OP_MUL        = 4
OP_RELU       = 10
OP_LEAKY_RELU = 11
OP_SIGMOID    = 12
OP_TANH       = 13
OP_SILU       = 14
OP_GELU       = 15
OP_MAXPOOL2D  = 20
OP_AVGPOOL2D  = 21
OP_BATCHNORM2D= 22
OP_LAYERNORM  = 23
OP_SOFTMAX    = 30
OP_CONCAT     = 31
OP_UPSAMPLE   = 32
OP_RESIZE     = 33
OP_INPUT      = 40
OP_OUTPUT     = 41

DT_F32  = 12
DT_I8   = 2
DT_I32  = 6
DT_I64  = 8

# ============================================================================
#  FLAT NAMESPACE HELPERS (C++ Motoru ile Uyumlu Indexleme)
# ============================================================================

def W(idx: int) -> int:
    """Ağırlık indexini C++ motorunun beklediği <= -1000 formatına çevirir."""
    return -1000 - idx

def EXT(idx: int) -> int:
    """Dış girdi (External Input) indexini < 0 formatına çevirir."""
    return -1 - idx

# ============================================================================

def pack_op_params(
    stride_h=1, stride_w=1, pad_h=0, pad_w=0, dil_h=1, dil_w=1, groups=1,
    kernel_h=0, kernel_w=0,
    scale_h=0, scale_w=0, target_h=0, target_w=0,
    axis=-1, eps=1e-5, leaky_slope=0.01,
    upsample_mode=0, fused_activation=0,
):
    PARAMS_SIZE = 128
    buf = bytearray()
    # 7 × int64: stride, pad, dil, groups (offset 0-55)
    buf += struct.pack('<7q', stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups)
    # 2 × int64: kernel (offset 56-71)
    buf += struct.pack('<2q', kernel_h, kernel_w)
    # 1 × int32: axis (offset 72-75)
    buf += struct.pack('<i', axis)
    # 4 bytes padding (int64 alignment) (offset 76-79)
    buf += b'\x00' * 4
    # 4 × int64: scale, target (offset 80-111)
    buf += struct.pack('<4q', scale_h, scale_w, target_h, target_w)
    # 1 × uint8: upsample_mode (offset 112)
    buf += struct.pack('<B', upsample_mode)
    # 3 bytes padding (float alignment)
    buf += b'\x00' * 3
    # 1 × float: eps (offset 116-119)
    buf += struct.pack('<f', eps)
    # 1 × uint8: fused_activation (offset 120)
    buf += struct.pack('<B', fused_activation)
    # 3 bytes padding
    buf += b'\x00' * 3
    # 1 × float: leaky_slope (offset 124-127)
    buf += struct.pack('<f', leaky_slope)
    # Zaten 128 olmalı
    while len(buf) < PARAMS_SIZE:
        buf += b'\x00'
    return bytes(buf[:PARAMS_SIZE])

class WeightEntry:
    def __init__(self, name, tensor):
        self.name = name
        self.tensor = tensor
        self.data_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()

class Node:
    def __init__(self, op, name, inputs, outputs, params=None):
        self.op = op
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = params or {}

def write_engine(path, weights, nodes, graph_inputs, graph_outputs, brand=b"engine-ai"):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(weights)))
        f.write(struct.pack('<I', len(nodes)))
        f.write(struct.pack('<I', len(graph_inputs)))
        f.write(struct.pack('<I', len(graph_outputs)))
        brand_bytes = brand[:63] + b'\x00' * (64 - len(brand[:63]))
        f.write(brand_bytes)

        weight_offsets = []
        cursor_offset = 0
        for w in weights:
            name_bytes = w.name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            t = w.tensor
            dt = DT_F32 if t.dtype == torch.float32 else DT_I8
            f.write(struct.pack('<B', dt))
            rank = len(t.shape)
            f.write(struct.pack('<B', rank))
            for d in t.shape:
                f.write(struct.pack('<q', int(d)))

            size_bytes = len(w.data_bytes)
            f.write(struct.pack('<Q', cursor_offset))
            f.write(struct.pack('<Q', size_bytes))
            weight_offsets.append(cursor_offset)
            cursor_offset += size_bytes

        for n in nodes:
            f.write(struct.pack('<B', n.op))
            name_bytes = n.name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<H', len(n.inputs)))
            f.write(struct.pack('<H', len(n.outputs)))
            for i in n.inputs:
                f.write(struct.pack('<i', i))
            for o in n.outputs:
                f.write(struct.pack('<i', o))
            params_bytes = pack_op_params(**n.params)
            f.write(params_bytes)

        for gi in graph_inputs:
            f.write(struct.pack('<i', gi))
        for go in graph_outputs:
            f.write(struct.pack('<i', go))

        for w in weights:
            f.write(w.data_bytes)

    print(f"✅ Yazıldı: {path}")
    print(f"   Ağırlık: {len(weights)}")
    print(f"   Node:    {len(nodes)}")

# ============================================================================
#  ÖRNEK: Basit bir CNN modelini .engine'e çevir (YENİ MİMARİ İLE)
# ============================================================================

def example_convert():
    weights = [
        WeightEntry("conv1.weight", torch.randn(32, 3, 3, 3)), # Index: 0
        WeightEntry("conv1.bias",   torch.zeros(32)),          # Index: 1
        WeightEntry("conv2.weight", torch.randn(64, 32, 3, 3)),# Index: 2
        WeightEntry("conv2.bias",   torch.zeros(64)),          # Index: 3
        WeightEntry("fc.weight",    torch.randn(10, 64 * 8 * 8)), # Index: 4
        WeightEntry("fc.bias",      torch.zeros(10)),          # Index: 5
    ]

    nodes = [
        Node(OP_INPUT, "input", [], [0], {}),
        
        # YENİ: Ağırlıkları W(0) ve W(1) ile sarmalıyoruz ki -1000'e dönüşsünler!
        Node(OP_CONV2D, "conv1", [0, W(0), W(1)], [1],
             {"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1, "fused_activation": 1}),
             
        Node(OP_MAXPOOL2D, "pool1", [1], [2],
             {"kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2}),
             
        # YENİ: Ağırlıkları W(2) ve W(3) ile sarmalıyoruz
        Node(OP_CONV2D, "conv2", [2, W(2), W(3)], [3],
             {"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1, "fused_activation": 1}),
             
        Node(OP_MAXPOOL2D, "pool2", [3], [4],
             {"kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2}),
             
        Node(OP_OUTPUT, "output", [4], [-1], {}),
    ]

    graph_inputs  = [0]
    graph_outputs = [4]

    write_engine("example.engine", weights, nodes, graph_inputs, graph_outputs)

if __name__ == "__main__":
    example_convert()