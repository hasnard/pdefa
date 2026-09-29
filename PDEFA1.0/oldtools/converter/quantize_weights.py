#!/usr/bin/env python3
"""Weight INT8 quantize — symmetric per-channel."""
import numpy as np
import onnx
from onnx import numpy_helper

def quantize_weight_perchannel(w):
    """w: [Cout, ...] np.float32 → (w_q: int8, scale: float32[Cout])"""
    Cout = w.shape[0]
    w_flat = w.reshape(Cout, -1)
    max_abs = np.abs(w_flat).max(axis=1)
    scale = max_abs / 127.0
    scale[scale == 0] = 1.0  # sıfır bölmesini önle
    w_q = np.round(w_flat / scale[:, None]).clip(-128, 127).astype(np.int8)
    return w_q.reshape(w.shape), scale.astype(np.float32)

def quantize_onnx_weights(onnx_in, onnx_out):
    model = onnx.load(onnx_in)
    graph = model.graph
    
    new_initializers = []
    quantized_scales = {}  # name -> scale
    
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        # Sadece 4D Conv weight'leri ve 2D MatMul weight'leri quantize et
        if arr.ndim == 4 or (arr.ndim == 2 and arr.shape[0] >= 32):
            w_q, scale = quantize_weight_perchannel(arr)
            new_init = numpy_helper.from_array(w_q, name=init.name + "_q")
            scale_init = numpy_helper.from_array(
                scale, name=init.name + "_scale")
            new_initializers.append(new_init)
            new_initializers.append(scale_init)
            quantized_scales[init.name] = init.name + "_scale"
            print(f"  {init.name}: {arr.shape} → INT8 + scale[{len(scale)}]")
        else:
            new_initializers.append(init)
    
    # Graph'ı güncelle
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    
    onnx.save(model, onnx_out)
    print(f"✅ {onnx_out}")

if __name__ == "__main__":
    import sys
    quantize_onnx_weights(sys.argv[1], sys.argv[2])