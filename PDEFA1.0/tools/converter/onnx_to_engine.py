#!/usr/bin/env python3
"""
ONNX -> .engine converter (v2)

Degisiklikler (v2):
  - .engine v2: input/output TensorInfo (name + shape + dtype) eklendi
  - Eski v1 dosyalari motor tarafinda hala okunabilir (fallback)
  - File existence check
  - Topolojik siralama (Kahn)
  - Split -> Slice donusumu
  - Identity/Dropout/Cast passthrough
  - Node-identity tabanli Conv+SiLU fusion
  - Graph output: gercek graph.output'tan
  - Dead-node eliminasyonu
  - Net hata mesajlari

Kullanim: python onnx_to_engine.py model.onnx model.engine
"""
import os
import sys
import struct
from collections import defaultdict, deque

import numpy as np
import onnx
from onnx import numpy_helper

# ============================================================
#  SABITLER
# ============================================================
MAGIC = 0x4E474E45
VERSION = 2                     # v2: input/output infos

OP_CONV2D           = 1
OP_MATMUL           = 2
OP_ADD              = 3
OP_MUL              = 4
OP_SUB              = 5
OP_DIV              = 6
OP_RELU             = 10
OP_LEAKY_RELU       = 11
OP_SIGMOID          = 12
OP_TANH             = 13
OP_SILU             = 14
OP_GELU             = 15
OP_HARDSWISH        = 16  
OP_HARDSIGMOID      = 17  
OP_MAXPOOL2D        = 20
OP_AVGPOOL2D        = 21
OP_BATCHNORM2D      = 22
OP_LAYERNORM        = 23
OP_SOFTMAX          = 30
OP_CONCAT           = 31
OP_UPSAMPLE         = 32
OP_RESIZE           = 33
OP_INPUT            = 40
OP_OUTPUT           = 41
OP_RESHAPE          = 42
OP_TRANSPOSE        = 43
OP_SLICE            = 44
OP_GATHER           = 45
OP_SHAPE            = 46
OP_EXPAND           = 47
OP_UNSQUEEZE        = 48
OP_SQUEEZE          = 49
OP_TOPK             = 50
OP_GATHER_ELEMENTS  = 51
OP_MOD              = 52
OP_REDUCE_MAX       = 53
OP_REDUCE_MIN       = 54
OP_REDUCE_PROD      = 55
OP_NONZERO          = 56
OP_CONSTANT_OF_SHAPE= 57
OP_TILE             = 58
OP_RANGE            = 59
OP_EQUAL            = 60
OP_GREATER          = 61
OP_EXP              = 62
OP_MIN              = 63
OP_MAX              = 64
OP_REDUCE_SUM       = 65
OP_CLIP             = 66
OP_ARGMAX           = 67
OP_STACK            = 68


DT_F32 = 12
DT_I8  = 2
PARAMS_SIZE = 128

ONNX_OP_MAP = {
    "Conv":               OP_CONV2D,
    "MatMul":             OP_MATMUL,
    "Gemm":               OP_MATMUL,
    "Add":                OP_ADD,
    "Mul":                OP_MUL,
    "Sub":                OP_SUB,
    "Div":                OP_DIV,
    "Relu":               OP_RELU,
    "LeakyRelu":          OP_LEAKY_RELU,
    "Sigmoid":            OP_SIGMOID,
    "Tanh":               OP_TANH,
    "SiLU":               OP_SILU,
    "GELU":               OP_GELU,
    "HardSwish":          OP_HARDSWISH,      
    "HardSigmoid":        OP_HARDSIGMOID,    
    "MaxPool":            OP_MAXPOOL2D,
    "AveragePool":        OP_AVGPOOL2D,
    "GlobalAveragePool":  OP_AVGPOOL2D,
    "ReduceMean":         OP_AVGPOOL2D,      # GlobalAveragePool'un opset-17 karsiligi
    "BatchNormalization": OP_BATCHNORM2D,
    "Clip":               OP_CLIP,
    "ArgMax":             OP_ARGMAX,
    "Stack":              OP_STACK,
    "LayerNormalization": OP_LAYERNORM,
    "Softmax":            OP_SOFTMAX,
    "Concat":             OP_CONCAT,
    "Resize":             OP_RESIZE,
    "Upsample":           OP_UPSAMPLE,
    "Reshape":            OP_RESHAPE,
    "Flatten":            OP_RESHAPE,
    "Transpose":          OP_TRANSPOSE,
    "Slice":              OP_SLICE,
    "Gather":             OP_GATHER,
    "Shape":              OP_SHAPE,
    "Expand":             OP_EXPAND,
    "Unsqueeze":          OP_UNSQUEEZE,
    "Squeeze":            OP_SQUEEZE,
    "TopK":             OP_TOPK,
    "GatherElements":   OP_GATHER_ELEMENTS,
    "Mod":              OP_MOD,
    "ReduceMax":        OP_REDUCE_MAX,
    "ReduceMin":        OP_REDUCE_MIN,
    "ReduceProd":       OP_REDUCE_PROD,
    "Exp":              OP_EXP,
    "Min":              OP_MIN,
    "Max":              OP_MAX,
    "Tile":             OP_TILE,
    "ReduceSum":        OP_REDUCE_SUM,
    
}

PASSTHROUGH_OPS = {"Identity", "Dropout", "Cast"}


# ============================================================
#  YARDIMCILAR
# ============================================================
def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.STRING: return a.s.decode('utf-8')
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
    return default


def get_ints(node, name, default=None):
    v = get_attr(node, name, default)
    if v is None: return default
    if isinstance(v, list): return v
    return [v]

def get_tensor_constant_value(graph, name):
    """Bir ismin değerini initializer VEYA Constant node'dan döndür.
    Her zaman bir liste döner (skaler olsa bile [x])."""
    init = next((i for i in graph.initializer if i.name == name), None)
    if init is not None:
        v = onnx.numpy_helper.to_array(init).tolist()
        return v if isinstance(v, list) else [v]

    for n in graph.node:
        if n.op_type != "Constant":
            continue
        if name not in n.output:
            continue
        for a in n.attribute:
            if a.name == "value":
                return onnx.numpy_helper.to_array(a.t).tolist()
            if a.name == "value_int":
                return [a.i]
            if a.name == "value_ints":
                return list(a.ints)
            if a.name == "value_float":
                return [a.f]
            if a.name == "value_floats":
                return list(a.floats)
    return None

def topo_sort_nodes(graph):
    nodes = list(graph.node)
    producer = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                producer[o] = i

    deps = defaultdict(set)
    for i, n in enumerate(nodes):
        for inp in n.input:
            if inp and inp in producer and producer[inp] != i:
                deps[i].add(producer[inp])

    in_deg = {i: len(deps[i]) for i in range(len(nodes))}
    rev = defaultdict(list)
    for i, parents in deps.items():
        for p in parents:
            rev[p].append(i)

    q = deque([i for i in range(len(nodes)) if in_deg[i] == 0])
    order = []
    while q:
        i = q.popleft()
        order.append(i)
        for j in rev[i]:
            in_deg[j] -= 1
            if in_deg[j] == 0:
                q.append(j)

    if len(order) != len(nodes):
        raise RuntimeError("Topolojik siralama basarisiz - dongu var mi?")
    return [nodes[i] for i in order]


def expand_split_to_slices(graph):
    new_nodes = []
    for node in graph.node:
        if node.op_type != "Split":
            new_nodes.append(node)
            continue

        axis = get_attr(node, "axis", 0)
        split_sizes = get_attr(node, "split", None)
        if split_sizes is None and len(node.input) > 1 and node.input[1]:
            init = next((i for i in graph.initializer if i.name == node.input[1]), None)
            if init is not None:
                split_sizes = onnx.numpy_helper.to_array(init).tolist()
        if split_sizes is None:
            n_out = len(node.output)
            split_sizes = [-1] * n_out

        data_input = node.input[0]
        cum = 0
        for i, out_name in enumerate(node.output):
            if not out_name:
                continue
            size = split_sizes[i]
            starts_name = f"{node.name}_start_{i}"
            ends_name   = f"{node.name}_end_{i}"
            axes_name   = f"{node.name}_axis_{i}"

            starts_arr = np.array([cum], dtype=np.int64)
            ends_arr   = np.array([cum + size] if size > 0 else [2**31 - 1], dtype=np.int64)
            axes_arr   = np.array([axis], dtype=np.int64)

            new_nodes.append(onnx.helper.make_node(
                "Constant", [], [starts_name],
                value=onnx.helper.make_tensor(
                    starts_name, onnx.TensorProto.INT64, [1], starts_arr)))
            new_nodes.append(onnx.helper.make_node(
                "Constant", [], [ends_name],
                value=onnx.helper.make_tensor(
                    ends_name, onnx.TensorProto.INT64, [1], ends_arr)))
            new_nodes.append(onnx.helper.make_node(
                "Constant", [], [axes_name],
                value=onnx.helper.make_tensor(
                    axes_name, onnx.TensorProto.INT64, [1], axes_arr)))

            new_nodes.append(onnx.helper.make_node(
                "Slice",
                [data_input, starts_name, ends_name, axes_name],
                [out_name],
                name=f"{node.name}_slice_{i}"))
            cum += size if size > 0 else 0

    graph.ClearField("node")
    graph.node.extend(new_nodes)


def expand_reduce_sum_to_chain(graph):
    """ONNX ReduceSum(axes=[2,3])'i iki ayrı ReduceSum(axis=2) + ReduceSum(axis=3)'e böler.
    Motorun engine'i tek-eksenli reduce yapıyor."""
    new_nodes = []
    for node in graph.node:
        if node.op_type != "ReduceSum":
            new_nodes.append(node)
            continue

        axes = get_ints(node, "axes", None)
        if axes is None and len(node.input) > 1 and node.input[1]:
            axes = get_tensor_constant_value(graph, node.input[1])

        if not axes:
            # Tek eksen varsayımı (default = -1)
            axes = [-1]

        if len(axes) <= 1:
            new_nodes.append(node)
            continue

        # Çok eksen → tek tek
        cur_input = node.input[0]
        for i, ax in enumerate(axes):
            is_last = (i == len(axes) - 1)
            out_name = node.output[0] if is_last else f"{node.name}_partial_{i}"

            axes_name = f"{node.name}_axis_{i}"
            axes_arr = np.array([ax], dtype=np.int64)
            new_nodes.append(onnx.helper.make_node(
                "Constant", [], [axes_name],
                value=onnx.helper.make_tensor(
                    axes_name, onnx.TensorProto.INT64, [1], axes_arr)))

            new_nodes.append(onnx.helper.make_node(
                "ReduceSum",
                [cur_input, axes_name],
                [out_name],
                name=f"{node.name}_axis_{i}"))
            cur_input = out_name

    graph.ClearField("node")
    graph.node.extend(new_nodes)


def fuse_conv_silu(graph):
    producer = {}
    for n in graph.node:
        for o in n.output:
            if o:
                producer[o] = n

    silu_skip_names = set()
    conv_silu_out = {}
    matched = 0

    for n in graph.node:
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        a, b = n.input
        for x, y in ((a, b), (b, a)):
            sig = producer.get(x)
            if not (sig and sig.op_type == "Sigmoid" and len(sig.input) == 1):
                continue
            if sig.input[0] != y:
                continue
            conv = producer.get(y)
            if not (conv and conv.op_type == "Conv"):
                continue

            silu_skip_names.add(sig.name)
            silu_skip_names.add(n.name)
            conv_silu_out[conv.name] = n.output[0]
            matched += 1
            break

    print(f"   [FUSE] matched={matched}, skip_names={len(silu_skip_names)}")
    return silu_skip_names, conv_silu_out


def pack_op_params(
    stride_h=1, stride_w=1, pad_h=0, pad_w=0, dil_h=1, dil_w=1, groups=1,
    kernel_h=0, kernel_w=0,
    scale_h=0, scale_w=0, target_h=0, target_w=0,
    axis=-1, eps=1e-5, leaky_slope=0.01,
    upsample_mode=0, fused_activation=0,
):
    buf = bytearray()
    buf += struct.pack('<7q', stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups)
    buf += struct.pack('<2q', kernel_h, kernel_w)
    buf += struct.pack('<i', axis)
    buf += b'\x00' * 4
    buf += struct.pack('<4q', scale_h, scale_w, target_h, target_w)
    buf += struct.pack('<B', upsample_mode)
    buf += b'\x00' * 3
    buf += struct.pack('<f', eps)
    buf += struct.pack('<B', fused_activation)
    buf += b'\x00' * 3
    buf += struct.pack('<f', leaky_slope)
    while len(buf) < PARAMS_SIZE:
        buf += b'\x00'
    return bytes(buf[:PARAMS_SIZE])


# ============================================================
#  ONNX dtype -> EngineDType
# ============================================================
def onnx_dtype_to_engine(elem_type):
    m = {
        onnx.TensorProto.FLOAT:    12,   # F32
        onnx.TensorProto.FLOAT16:  10,   # F16
        onnx.TensorProto.BFLOAT16: 11,   # BF16
        onnx.TensorProto.INT8:      2,   # I8
        onnx.TensorProto.UINT8:     3,   # U8
        onnx.TensorProto.INT16:     4,
        onnx.TensorProto.UINT16:    5,
        onnx.TensorProto.INT32:     6,   # I32
        onnx.TensorProto.UINT32:    7,
        onnx.TensorProto.INT64:     8,   # I64
        onnx.TensorProto.UINT64:    9,
        onnx.TensorProto.BOOL:      1,   # Bool
        onnx.TensorProto.DOUBLE:   13,   # F64
    }
    return m.get(elem_type, 12)


def extract_tensor_info(tp):
    """ONNX ValueInfoProto -> {name, shape, dtype}"""
    shape = []
    for d in tp.type.tensor_type.shape.dim:
        if d.HasField('dim_value'):
            shape.append(int(d.dim_value))
        else:
            shape.append(-1)   # dinamik boyut
    return {
        "name":  tp.name,
        "shape": shape,
        "dtype": onnx_dtype_to_engine(tp.type.tensor_type.elem_type),
    }


# ============================================================
#  CONVERT
# ============================================================
def convert(onnx_path, engine_path):
    if not os.path.exists(onnx_path):
        print(f"HATA: ONNX dosyasi yok: {onnx_path}")
        print(f"   Once export et:")
        print(f"   python -c \"from ultralytics import YOLO; "
              f"YOLO('yolov8n.pt').export(format='onnx', opset=17, imgsz=640, dynamic=False)\"")
        sys.exit(1)

    print(f"ONNX okunuyor: {onnx_path}")
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"HATA: ONNX yuklenemedi / bozuk: {e}")
        sys.exit(1)

    graph = model.graph
    print(f"   ONNX node sayisi: {len(graph.node)}")

    # ---- 1) Topolojik siralama ----
    try:
        sorted_nodes = topo_sort_nodes(graph)
        graph.ClearField("node")
        graph.node.extend(sorted_nodes)
        print(f"   [OK] Topolojik siralama")
    except Exception as e:
        print(f"   [WARN] Topolojik siralama atlandi: {e}")

    # ---- 2) Split -> Slice ----
    n_before = len(graph.node)
    expand_split_to_slices(graph)
    n_after = len(graph.node)
    if n_after != n_before:
        print(f"   [OK] Split -> Slice: {n_before} -> {n_after} node")

    # ---- 2.5) ReduceSum (multi-axis) → zincir ----
    n_before = len(graph.node)
    expand_reduce_sum_to_chain(graph)
    n_after = len(graph.node)
    if n_after != n_before:
        print(f"   [OK] ReduceSum expand: {n_before} -> {n_after} node")

    # ---- 3) Conv+SiLU fusion ----
    silu_skip, conv_silu_out = fuse_conv_silu(graph)
    print(f"   [OK] {len(conv_silu_out)} Conv+SiLU fusion (fused into Conv nodes)")

    # ---- 4) Weights ----
    weights = []
    weight_name_to_idx = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        weight_name_to_idx[init.name] = len(weights)
        weights.append((init.name, arr, False))
    print(f"   {len(weights)} agirlik")

    # ---- 5) Tensor -> intermediate index ----
    tensor_to_inter = {}
    next_inter = [0]

    def alloc_inter():
        i = next_inter[0]; next_inter[0] += 1; return i

    graph_inputs_inter = []
    for inp in graph.input:
        if inp.name in weight_name_to_idx:
            continue
        ext_idx = -1 - len(graph_inputs_inter)
        tensor_to_inter[inp.name] = ext_idx
        graph_inputs_inter.append(ext_idx)

    # ---- 5.5) v2: Input/Output infos ----
    input_infos  = []
    output_infos = []

    for inp in graph.input:
        if inp.name in weight_name_to_idx:
            continue
        input_infos.append(extract_tensor_info(inp))

    for out in graph.output:
        output_infos.append(extract_tensor_info(out))

    # ---- 6) Node conversion ----
    engine_nodes = []
    skipped = []
    passthrough_remap = {}

    def resolve_name(name):
        while name in passthrough_remap:
            name = passthrough_remap[name]
        return name

    for node in graph.node:
        op_type = node.op_type

        if node.name in silu_skip:
            continue

        if op_type == "Constant":
            v_tensor = None
            for a in node.attribute:
                if a.name == "value":
                    v_tensor = a.t; break
            if v_tensor is not None:
                arr = numpy_helper.to_array(v_tensor)
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                weight_name_to_idx[node.output[0]] = len(weights)
                weights.append((f"{node.name}::const", arr, False))
            else:
                for aname in ("value_int", "value_float"):
                    val = get_attr(node, aname, None)
                    if val is not None:
                        arr = np.ascontiguousarray(np.array([val], dtype=np.float32))
                        weight_name_to_idx[node.output[0]] = len(weights)
                        weights.append((f"{node.name}::const", arr, False))
                        break
            continue

        if op_type in PASSTHROUGH_OPS:
            if node.input and node.input[0]:
                in_name = resolve_name(node.input[0])
                for o in node.output:
                    if o:
                        passthrough_remap[o] = in_name
            continue

        if op_type not in ONNX_OP_MAP:
            skipped.append((op_type, node.name))
            for o in node.output:
                if o:
                    passthrough_remap[o] = f"__SKIPPED_{o}"
            continue

        engine_op = ONNX_OP_MAP[op_type]

        node_inputs = []
        for inp_name in node.input:
            if not inp_name:
                continue
            inp_name = resolve_name(inp_name)
            if inp_name.startswith("__SKIPPED_"):
                skipped.append(("CONSUMER", node.name))
                node_inputs = None; break
            if inp_name in weight_name_to_idx:
                node_inputs.append(-1000 - weight_name_to_idx[inp_name])
            elif inp_name in tensor_to_inter:
                    node_inputs.append(tensor_to_inter[inp_name])
            else:
                    print(f"   [MISSING-INPUT] node='{node.name}' "
                          f"op={op_type} input='{inp_name}'")
                    tensor_to_inter[inp_name] = alloc_inter()
                    node_inputs.append(tensor_to_inter[inp_name])
        if node_inputs is None:
            continue




               # ---- Clip: opset<11 attr, opset>=11 ek input ----
        clip_min_val = None
        clip_max_val = None
        if op_type == "Clip":
            clip_min_val = get_attr(node, "min", None)
            clip_max_val = get_attr(node, "max", None)

            def _to_float(x):
                if x is None: return None
                if isinstance(x, (int, float)): return float(x)
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    return float(x[0])
                return None

            if clip_min_val is None and len(node.input) > 1 and node.input[1]:
                clip_min_val = _to_float(get_tensor_constant_value(graph, node.input[1]))
            if clip_max_val is None and len(node.input) > 2 and node.input[2]:
                clip_max_val = _to_float(get_tensor_constant_value(graph, node.input[2]))
            node_inputs = node_inputs[:1]


        # TopK'nın 2. input'u K tensor'ı — weight olarak yaz
        if op_type == "TopK" and len(node.input) >= 2:
            # node.inputs[1] zaten weight veya constant olarak geliyor olabilir
            # Eğer intermediate ise, K'yı constant olarak yaz
            pass  # K zaten normal input olarak isleniyor


        if op_type == "Transpose":
            perm = get_attr(node, "perm", None)
            if perm is not None:
                perm_arr = np.ascontiguousarray(np.array(perm, dtype=np.float32))
                pname = f"{node.name}::perm"
                weight_name_by_idx = len(weights)
                weight_name_to_idx[pname] = weight_name_by_idx
                weights.append((pname, perm_arr, False))
                node_inputs.append(-1000 - weight_name_by_idx)

        node_outputs = []
        if node.name in conv_silu_out:
            target = conv_silu_out[node.name]
            if target not in tensor_to_inter:
                tensor_to_inter[target] = alloc_inter()
            node_outputs.append(tensor_to_inter[target])
        else:
            for out_name in node.output:
                if not out_name:
                    continue
                out_name = resolve_name(out_name)
                if out_name not in tensor_to_inter:
                    tensor_to_inter[out_name] = alloc_inter()
                node_outputs.append(tensor_to_inter[out_name])

        params = {}
        if op_type == "Conv":
            s = get_ints(node, "strides", [1, 1])
            p = get_ints(node, "pads", [0, 0, 0, 0])
            d = get_ints(node, "dilations", [1, 1])
            grp = get_attr(node, "group", 1)
            # DEBUG: grup>1 olanları yazdır
            if grp != 1:
                print(f"   [GROUP] {node.name[:50]} group={grp}")
            params = {
                "stride_h": s[0] if len(s) > 0 else 1,
                "stride_w": s[1] if len(s) > 1 else 1,
                "pad_h":    p[0] if len(p) >= 2 else 0,
                "pad_w":    p[1] if len(p) >= 2 else 0,
                "dil_h":    d[0] if len(d) > 0 else 1,
                "dil_w":    d[1] if len(d) > 1 else 1,
                "groups":   grp,
            }
            if node.name in conv_silu_out:
                params["fused_activation"] = 5
        elif op_type in ("MaxPool", "AveragePool"):
            k = get_ints(node, "kernel_shape", [2, 2])
            s = get_ints(node, "strides", [1, 1])
            p = get_ints(node, "pads", [0, 0, 0, 0])
            params = {
                "kernel_h": k[0] if len(k) > 0 else 2,
                "kernel_w": k[1] if len(k) > 1 else 2,
                "stride_h": s[0] if len(s) > 0 else 1,
                "stride_w": s[1] if len(s) > 1 else 1,
                "pad_h":    p[0] if len(p) >= 2 else 0,
                "pad_w":    p[1] if len(p) >= 2 else 0,
            }
        elif op_type == "ReduceMean":
            # GlobalAveragePool esdegeri
            axes = get_ints(node, "axes", None)
            keepdims = get_attr(node, "keepdims", 1)
            params = {
                "kernel_h": 0,
                "kernel_w": 0,
                "stride_h": 1,
                "stride_w": 1,
                "pad_h":    0,
                "pad_w":    0,
                "axis":     -1,
                "_reduce_mean": 1,
            }
        elif op_type == "BatchNormalization":
            params = {"eps": get_attr(node, "epsilon", 1e-5)}
        elif op_type == "LeakyRelu":
            params = {"leaky_slope": get_attr(node, "alpha", 0.01)}
        elif op_type == "Concat":
            params = {"axis": get_attr(node, "axis", 0)}
        elif op_type == "Softmax":
            params = {"axis": get_attr(node, "axis", -1)}
        elif op_type == "Gather":

            
            params = {"axis": get_attr(node, "axis", 0)}


        elif op_type == "LayerNormalization":
            params = {
                "axis": get_attr(node, "axis", -1),
                "eps":  get_attr(node, "epsilon", 1e-5),
            }    
        elif op_type == "Clip":
            lo = clip_min_val if clip_min_val is not None else -3.4028235e+38
            hi = clip_max_val if clip_max_val is not None else  3.4028235e+38
            params = {
                "eps":          float(lo),   # clip_min
                "leaky_slope":  float(hi),   # clip_max
            }
        elif op_type == "ArgMax":
            params = {"axis": get_attr(node, "axis", 0)}
        elif op_type == "Stack":
            params = {"axis": get_attr(node, "axis", 0)} 


        elif op_type == "TopK":
            # K input'u 2. input (input[1])
            axis = get_attr(node, "axis", -1)
            largest = get_attr(node, "largest", 1)
            sorted_ = get_attr(node, "sorted", 1)
            params = {
                "axis": axis,
                "largest": largest,
                "sorted": sorted_,
            }
        elif op_type == "GatherElements":
            params = {"axis": get_attr(node, "axis", 0)}
        elif op_type == "Mod":
            params = {"fmod": get_attr(node, "fmod", 0)}
        elif op_type == "ReduceSum":
            axes = get_ints(node, "axes", None)
            if axes is None and len(node.input) > 1 and node.input[1]:
                axes = get_tensor_constant_value(graph, node.input[1])
            if not axes:
                axes = [-1]
            params = {"axis": axes[0]}
        elif op_type == "ReduceMax":
            axes = get_ints(node, "axes", [-1])
            params = {"axis": axes[0] if axes else -1}
            axes = get_ints(node, "axes", [-1])
            params = {"axis": axes[0] if axes else -1}
        elif op_type == "ReduceMin":
            axes = get_ints(node, "axes", [-1])
            params = {"axis": axes[0] if axes else -1}
        
           
        elif op_type in ("Resize", "Upsample"):
            scales = get_attr(node, "scales", None)
            if scales is not None and len(scales) >= 4:
                params = {
                    "scale_h": int(scales[2]),
                    "scale_w": int(scales[3]),
                    "upsample_mode":
                        1 if get_attr(node, "mode", "nearest") == "linear" else 0,
                }

        if engine_op == OP_MATMUL and len(node_inputs) >= 2:
            w_neg = node_inputs[1]
            if w_neg <= -1000:
                w_idx = -w_neg - 1000
                if w_idx < len(weights):
                    wname, warr = weights[w_idx][:2]
                    if warr.ndim == 2:
                        warr_t = np.ascontiguousarray(warr.T)
                        new_idx = len(weights)
                        weights.append((f"{wname}_T", warr_t, False))
                        node_inputs[1] = -1000 - new_idx

        engine_nodes.append({
            "op": engine_op,
            "name": node.name or f"{op_type}_{len(engine_nodes)}",
            "inputs": node_inputs,
            "outputs": node_outputs,
            "params": params,
        })

    # ---- 7) Graph outputs ----
    graph_outputs_inter = []
    for out in graph.output:
        name = resolve_name(out.name)
        if name in tensor_to_inter:
            graph_outputs_inter.append(tensor_to_inter[name])
        else:
            print(f"   [WARN] Graph output '{out.name}' intermediate'da yok")
    if not graph_outputs_inter and engine_nodes:
        graph_outputs_inter = list(engine_nodes[-1]["outputs"])
        print(f"   [WARN] Fallback: son node output")

    # ---- 8) Rapor ----
    if skipped:
        unique_ops = sorted(set(op for op, _ in skipped if op != "CONSUMER"))
        print(f"\n[WARN] Desteklenmeyen op'lar ({len(skipped)} node etkilendi):")
        for op in unique_ops:
            count = sum(1 for o, _ in skipped if o == op)
            print(f"     - {op}: {count} node")

    # ---- 9) Yaz ----
    write_engine(engine_path, weights, engine_nodes,
                 graph_inputs_inter, graph_outputs_inter,
                 input_infos, output_infos)

    print(f"\n[OK] Yazildi: {engine_path}")
    print(f"   Weights: {len(weights)}")
    print(f"   Nodes:   {len(engine_nodes)}")
    print(f"   Inputs:  {graph_inputs_inter}")
    print(f"   Outputs: {graph_outputs_inter}")
    if input_infos:
        for i, info in enumerate(input_infos):
            print(f"   in[{i}]  = {info['name']} {tuple(info['shape'])} dtype={info['dtype']}")
    if output_infos:
        for i, info in enumerate(output_infos):
            print(f"   out[{i}] = {info['name']} {tuple(info['shape'])} dtype={info['dtype']}")


# ============================================================
#  WRITE (v2)
# ============================================================
def _write_tensor_info(f, info):
    name = info.get("name", "") or ""
    nb = name.encode('utf-8')
    f.write(struct.pack('<I', len(nb)))
    if nb:
        f.write(nb)

    shape = list(info.get("shape", []))
    ndim = len(shape)
    f.write(struct.pack('<B', ndim))
    for d in shape:
        f.write(struct.pack('<q', int(d)))
    f.write(struct.pack('<B', int(info.get("dtype", DT_F32))))


def write_engine(path, weights, nodes, graph_inputs, graph_outputs,
                 input_infos=None, output_infos=None,
                 brand=b"engine-ai"):
    if input_infos  is None: input_infos  = []
    if output_infos is None: output_infos = []

    with open(path, 'wb') as f:
        # ---- Header ----
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(weights)))
        f.write(struct.pack('<I', len(nodes)))
        f.write(struct.pack('<I', len(graph_inputs)))
        f.write(struct.pack('<I', len(graph_outputs)))
        brand_bytes = brand[:63] + b'\x00' * (64 - len(brand[:63]))
        f.write(brand_bytes)

        # ---- Weight descriptors ----
        cursor = 0
        for item in weights:
            if len(item) == 3:
                wname, arr, is_int8 = item
            else:
                wname, arr = item
                is_int8 = False
            nb = wname.encode('utf-8')
            f.write(struct.pack('<I', len(nb)))
            f.write(nb)
            f.write(struct.pack('<B', DT_I8 if is_int8 else DT_F32))
            f.write(struct.pack('<B', arr.ndim))
            for d in arr.shape:
                f.write(struct.pack('<q', int(d)))
            size = arr.nbytes
            f.write(struct.pack('<Q', cursor))
            f.write(struct.pack('<Q', size))
            cursor += size

        # ---- Nodes ----
        _PACK_KEYS = {
            "stride_h","stride_w","pad_h","pad_w","dil_h","dil_w","groups",
            "kernel_h","kernel_w",
            "scale_h","scale_w","target_h","target_w",
            "axis","eps","leaky_slope",
            "upsample_mode","fused_activation",
        }
        for n in nodes:
            if not isinstance(n["op"], int):
                raise ValueError(
                    f"Node '{n['name']}' op={n['op']!r} (type={type(n['op']).__name__})")
            f.write(struct.pack('<B', n["op"]))
            nb = n["name"].encode('utf-8')
            f.write(struct.pack('<I', len(nb)))
            f.write(nb)
            f.write(struct.pack('<H', len(n["inputs"])))
            f.write(struct.pack('<H', len(n["outputs"])))
            for i in n["inputs"]:
                f.write(struct.pack('<i', i))
            for o in n["outputs"]:
                f.write(struct.pack('<i', o))
            # pack_op_params sadece bilinen alanlari kabul eder
            params = {k: v for k, v in n["params"].items() if not k.startswith("_")}
            params = {k: v for k, v in params.items() if k in _PACK_KEYS}
            f.write(pack_op_params(**params))

        # ---- Graph I/O indices ----
        for gi in graph_inputs:
            f.write(struct.pack('<i', gi))
        for go in graph_outputs:
            f.write(struct.pack('<i', go))

        # ---- v2: input/output infos ----
        for info in input_infos:
            _write_tensor_info(f, info)
        for info in output_infos:
            _write_tensor_info(f, info)

        # ---- Weight data ----
        for item in weights:
            f.write(item[1].tobytes())


# ============================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Kullanim: python onnx_to_engine.py model.onnx model.engine")
        sys.exit(1)
    try:
        convert(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"\nHATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)