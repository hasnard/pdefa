"""ctypes bindings for pdefa_native (engine + imageloader + YOLO post-process)."""
import ctypes
import platform
from dataclasses import dataclass
from importlib.resources import files
from typing import Tuple


# ==================================================================
#  Native library yükleme
# ==================================================================

def _load_native():
    system = platform.system()
    name = {
        "Windows": "pdefa_native.dll",
        "Darwin":  "libpdefa_native.dylib",
        "Linux":   "libpdefa_native.so",
    }.get(system, "libpdefa_native.so")
    p = files("pdefa") / "lib" / name
    if not p.is_file():
        raise RuntimeError(f"Native library not found: {p}")
    return ctypes.CDLL(str(p))


_lib = _load_native()


# ==================================================================
#  Fonksiyon imzaları (argtypes / restype)
# ==================================================================

_lib.pdefa_version.restype        = ctypes.c_char_p
_lib.pdefa_last_error.restype     = ctypes.c_char_p
_lib.engine_version.restype       = ctypes.c_char_p
_lib.engine_simd_level.restype    = ctypes.c_char_p
_lib.engine_cpu_brand.restype     = ctypes.c_char_p
_lib.engine_status_string.argtypes = [ctypes.c_int]
_lib.engine_status_string.restype  = ctypes.c_char_p

# Context
_lib.engine_context_create.argtypes  = [ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_context_create.restype   = ctypes.c_int
_lib.engine_context_destroy.argtypes = [ctypes.c_void_p]
_lib.engine_context_set_num_threads.argtypes = [ctypes.c_void_p, ctypes.c_int32]
_lib.engine_context_set_num_threads.restype  = ctypes.c_int
_lib.engine_context_get_num_threads.argtypes = [ctypes.c_void_p]
_lib.engine_context_get_num_threads.restype  = ctypes.c_int32

# Tensor
_lib.engine_tensor_create.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int32, ctypes.c_int,
]
_lib.engine_tensor_create.restype = ctypes.c_int

_lib.engine_tensor_destroy.argtypes = [ctypes.c_void_p]
_lib.engine_tensor_numel.argtypes   = [ctypes.c_void_p]
_lib.engine_tensor_numel.restype    = ctypes.c_int64
_lib.engine_tensor_rank.argtypes    = [ctypes.c_void_p]
_lib.engine_tensor_rank.restype     = ctypes.c_int32

_lib.engine_tensor_shape.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int32),
]
_lib.engine_tensor_shape.restype = ctypes.c_int

_lib.engine_tensor_data.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.engine_tensor_data.restype = ctypes.c_int

_lib.engine_tensor_dtype.argtypes = [ctypes.c_void_p]
_lib.engine_tensor_dtype.restype  = ctypes.c_int

# Image
_lib.pdefa_image_load.argtypes = [ctypes.c_char_p]
_lib.pdefa_image_load.restype  = ctypes.c_void_p
_lib.pdefa_image_free.argtypes = [ctypes.c_void_p]

_lib.pdefa_image_width.argtypes    = [ctypes.c_void_p]
_lib.pdefa_image_width.restype     = ctypes.c_int32
_lib.pdefa_image_height.argtypes   = [ctypes.c_void_p]
_lib.pdefa_image_height.restype    = ctypes.c_int32
_lib.pdefa_image_channels.argtypes = [ctypes.c_void_p]
_lib.pdefa_image_channels.restype  = ctypes.c_int32
_lib.pdefa_image_format.argtypes   = [ctypes.c_void_p]
_lib.pdefa_image_format.restype    = ctypes.c_int32
_lib.pdefa_image_pixels.argtypes   = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
_lib.pdefa_image_pixels.restype    = ctypes.POINTER(ctypes.c_uint8)

# Preprocess
class _PreConfig(ctypes.Structure):
    _fields_ = [
        ("target_w",  ctypes.c_int32),
        ("target_h",  ctypes.c_int32),
        ("scale",     ctypes.c_float),
        ("pad_value", ctypes.c_float),
    ]

class _PreMeta(ctypes.Structure):
    _fields_ = [
        ("target_w", ctypes.c_int32), ("target_h", ctypes.c_int32),
        ("orig_w",   ctypes.c_int32), ("orig_h",   ctypes.c_int32),
        ("pad_x",    ctypes.c_int32), ("pad_y",    ctypes.c_int32),
        ("scaled_w", ctypes.c_int32), ("scaled_h", ctypes.c_int32),
        ("ratio",    ctypes.c_float),
    ]

_lib.pdefa_preprocess_letterbox.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_PreConfig),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(_PreMeta),
]
_lib.pdefa_preprocess_letterbox.restype = ctypes.c_int

_lib.pdefa_prepare_for_engine.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_float, ctypes.c_float,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(_PreMeta),
]
_lib.pdefa_prepare_for_engine.restype = ctypes.c_int

_lib.pdefa_load_image_as_tensor.argtypes = [
    ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.pdefa_load_image_as_tensor.restype = ctypes.c_int

# Post-process
_lib.pdefa_map_box_to_original.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(_PreMeta),
    ctypes.POINTER(ctypes.c_float),
]
_lib.pdefa_map_box_to_original.restype = ctypes.c_int

_lib.pdefa_map_point_to_original.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(_PreMeta),
]
_lib.pdefa_map_point_to_original.restype = ctypes.c_int

# Model / Session
_lib.engine_model_load.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_model_load.restype     = ctypes.c_int
_lib.engine_model_destroy.argtypes = [ctypes.c_void_p]
_lib.engine_model_name.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
_lib.engine_model_name.restype     = ctypes.c_int

_lib.engine_session_create.argtypes     = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_session_create.restype      = ctypes.c_int
_lib.engine_session_destroy.argtypes    = [ctypes.c_void_p]
_lib.engine_session_set_input.argtypes  = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
_lib.engine_session_set_input.restype   = ctypes.c_int
_lib.engine_session_run.argtypes        = [ctypes.c_void_p]
_lib.engine_session_run.restype         = ctypes.c_int
_lib.engine_session_get_output.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_session_get_output.restype  = ctypes.c_int

# Yeni: model info / session stats / debug / ops
class _SessionStats(ctypes.Structure):
    _fields_ = [("last_run_ms", ctypes.c_double),
                ("runs", ctypes.c_uint64)]

_lib.engine_model_input_info.argtypes = [
    ctypes.c_void_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int),
]
_lib.engine_model_input_info.restype = ctypes.c_int

_lib.engine_model_output_info.argtypes = [
    ctypes.c_void_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int),
]
_lib.engine_model_output_info.restype = ctypes.c_int

_lib.engine_session_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_SessionStats)]
_lib.engine_session_get_stats.restype  = ctypes.c_int

_lib.engine_session_debug_intermediate.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_session_debug_intermediate.restype = ctypes.c_int

_lib.engine_ops_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.engine_ops_matmul.restype  = ctypes.c_int
_lib.engine_ops_relu.argtypes   = [ctypes.c_void_p]
_lib.engine_ops_relu.restype    = ctypes.c_int
_lib.engine_ops_silu.argtypes   = [ctypes.c_void_p]
_lib.engine_ops_silu.restype    = ctypes.c_int
# ==================================================================
#  OPS — PART 1: binary elementwise
# ==================================================================

_lib.engine_ops_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_add.restype  = ctypes.c_int

_lib.engine_ops_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_sub.restype  = ctypes.c_int

_lib.engine_ops_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_mul.restype  = ctypes.c_int

_lib.engine_ops_div.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_div.restype  = ctypes.c_int

# Unary in-place
_lib.engine_ops_sigmoid.argtypes = [ctypes.c_void_p]
_lib.engine_ops_sigmoid.restype  = ctypes.c_int
_lib.engine_ops_tanh.argtypes    = [ctypes.c_void_p]
_lib.engine_ops_tanh.restype     = ctypes.c_int
_lib.engine_ops_gelu.argtypes    = [ctypes.c_void_p]
_lib.engine_ops_gelu.restype     = ctypes.c_int
_lib.engine_ops_clip.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
_lib.engine_ops_clip.restype     = ctypes.c_int
_lib.engine_ops_softmax.argtypes = [ctypes.c_void_p, ctypes.c_int32]
_lib.engine_ops_softmax.restype  = ctypes.c_int

# ==================================================================
#  OPS — PART 2: reduce / argmax / gather / shape
# ==================================================================

_lib.engine_ops_reduce_max.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                        ctypes.c_int32,
                                        ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_reduce_max.restype  = ctypes.c_int

_lib.engine_ops_reduce_sum.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                        ctypes.c_int32,
                                        ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_reduce_sum.restype  = ctypes.c_int

_lib.engine_ops_reduce_mean.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                         ctypes.c_int32,
                                         ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_reduce_mean.restype  = ctypes.c_int

_lib.engine_ops_argmax.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                    ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_argmax.restype  = ctypes.c_int

_lib.engine_ops_gather.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_gather.restype  = ctypes.c_int

_lib.engine_ops_reshape.argtypes = [ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_int64),
                                     ctypes.c_int32,
                                     ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_reshape.restype  = ctypes.c_int

_lib.engine_ops_transpose.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_int32),
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_transpose.restype  = ctypes.c_int

_lib.engine_ops_concat.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_int32, ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_concat.restype  = ctypes.c_int

_lib.engine_ops_stack.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.c_int32, ctypes.c_int32,
                                   ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_ops_stack.restype  = ctypes.c_int

# ==================================================================
#  RANDOM
# ==================================================================

_lib.engine_random_randn.argtypes = [ctypes.POINTER(ctypes.c_int64),
                                      ctypes.c_int32, ctypes.c_uint64,
                                      ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_random_randn.restype  = ctypes.c_int

_lib.engine_random_rand.argtypes  = [ctypes.POINTER(ctypes.c_int64),
                                      ctypes.c_int32, ctypes.c_uint64,
                                      ctypes.POINTER(ctypes.c_void_p)]
_lib.engine_random_rand.restype   = ctypes.c_int


# ==================================================================
#  OPS — Python wrapper'lar
# ==================================================================

def _ops_binary(a, b, fn) -> Tensor:
    if not isinstance(a, Tensor):
        a = _scalar_tensor(float(a))
    if not isinstance(b, Tensor):
        b = _scalar_tensor(float(b))
    h = ctypes.c_void_p()
    _check(fn(a.raw, b.raw, ctypes.byref(h)), "ops_binary")
    return Tensor(h, owns=True)


def ops_add(a: Tensor, b: Tensor) -> Tensor: return _ops_binary(a, b, _lib.engine_ops_add)
def ops_sub(a: Tensor, b: Tensor) -> Tensor: return _ops_binary(a, b, _lib.engine_ops_sub)
def ops_mul(a: Tensor, b: Tensor) -> Tensor: return _ops_binary(a, b, _lib.engine_ops_mul)
def ops_div(a: Tensor, b: Tensor) -> Tensor: return _ops_binary(a, b, _lib.engine_ops_div)


def ops_sigmoid(t: Tensor) -> Tensor:
    _check(_lib.engine_ops_sigmoid(t.raw), "ops_sigmoid"); return t

def ops_tanh(t: Tensor) -> Tensor:
    _check(_lib.engine_ops_tanh(t.raw), "ops_tanh"); return t

def ops_gelu(t: Tensor) -> Tensor:
    _check(_lib.engine_ops_gelu(t.raw), "ops_gelu"); return t

def ops_clip(t: Tensor, lo: float, hi: float) -> Tensor:
    _check(_lib.engine_ops_clip(t.raw, float(lo), float(hi)), "ops_clip"); return t

def ops_softmax(t: Tensor, axis: int = -1) -> Tensor:
    _check(_lib.engine_ops_softmax(t.raw, int(axis)), "ops_softmax"); return t


def _ops_reduce(a: Tensor, axis: int, keepdims: bool, fn) -> Tensor:
    h = ctypes.c_void_p()
    _check(fn(a.raw, int(axis), 1 if keepdims else 0, ctypes.byref(h)), "ops_reduce")
    return Tensor(h, owns=True)


def ops_reduce_max(a, axis=-1, keepdims=False):  return _ops_reduce(a, axis, keepdims, _lib.engine_ops_reduce_max)
def ops_reduce_sum(a, axis=-1, keepdims=False):  return _ops_reduce(a, axis, keepdims, _lib.engine_ops_reduce_sum)
def ops_reduce_mean(a, axis=-1, keepdims=False): return _ops_reduce(a, axis, keepdims, _lib.engine_ops_reduce_mean)


def ops_argmax(a: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    h = ctypes.c_void_p()
    _check(_lib.engine_ops_argmax(a.raw, int(axis), 1 if keepdims else 0,
                                   ctypes.byref(h)), "ops_argmax")
    return Tensor(h, owns=True)


def ops_gather(a: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    h = ctypes.c_void_p()
    _check(_lib.engine_ops_gather(a.raw, indices.raw, int(axis),
                                   ctypes.byref(h)), "ops_gather")
    return Tensor(h, owns=True)


def ops_reshape(a: Tensor, shape) -> Tensor:
    n = len(shape)
    arr = (ctypes.c_int64 * n)(*shape)
    h = ctypes.c_void_p()
    _check(_lib.engine_ops_reshape(a.raw, arr, n, ctypes.byref(h)), "ops_reshape")
    return Tensor(h, owns=True)


def ops_transpose(a: Tensor, perm) -> Tensor:
    n = len(perm)
    arr = (ctypes.c_int32 * n)(*perm)
    h = ctypes.c_void_p()
    _check(_lib.engine_ops_transpose(a.raw, arr, n, ctypes.byref(h)), "ops_transpose")
    return Tensor(h, owns=True)


def ops_concat(inputs, axis: int = 0) -> Tensor:
    n = len(inputs)
    arr = (ctypes.c_void_p * n)(*[t.raw for t in inputs])
    h = ctypes.c_void_p()
    _check(_lib.engine_ops_concat(arr, n, int(axis), ctypes.byref(h)), "ops_concat")
    return Tensor(h, owns=True)


def ops_stack(inputs, axis: int = 0) -> Tensor:
    n = len(inputs)
    arr = (ctypes.c_void_p * n)(*[t.raw for t in inputs])
    h = ctypes.c_void_p()
    _check(_lib.engine_ops_stack(arr, n, int(axis), ctypes.byref(h)), "ops_stack")
    return Tensor(h, owns=True)


def random_randn(shape, seed: int = 0) -> Tensor:
    n = len(shape)
    arr = (ctypes.c_int64 * n)(*shape)
    h = ctypes.c_void_p()
    _check(_lib.engine_random_randn(arr, n, int(seed), ctypes.byref(h)), "random_randn")
    return Tensor(h, owns=True)


def random_rand(shape, seed: int = 0) -> Tensor:
    n = len(shape)
    arr = (ctypes.c_int64 * n)(*shape)
    h = ctypes.c_void_p()
    _check(_lib.engine_random_rand(arr, n, int(seed), ctypes.byref(h)), "random_rand")
    return Tensor(h, owns=True)





def _reshape_list(flat, shape):
    """Flat list → nested list (numpy .tolist() formatı)."""
    if len(shape) == 0:
        return flat[0] if flat else None
    if len(shape) == 1:
        return list(flat)
    # Multi-dim
    from math import prod
    inner_size = prod(shape[1:])
    if inner_size == 0:
        return []
    return [_reshape_list(flat[i*inner_size:(i+1)*inner_size], shape[1:])
            for i in range(shape[0])]


def _flatten_list(data):
    """Nested list → (flat, shape)."""
    if not isinstance(data, (list, tuple)):
        return [data], []
    shape = []
    cur = data
    while isinstance(cur, (list, tuple)):
        shape.append(len(cur))
        if len(cur) == 0:
            break
        cur = cur[0]
    # Flatten
    def rec(x, depth):
        if depth == len(shape):
            yield x
        else:
            for v in x:
                yield from rec(v, depth + 1)
    return list(rec(data, 0)), shape


def _scalar_tensor(v: float):
    """0-boyutlu skaler tensor (broadcast için)."""
    h = ctypes.c_void_p()
    shape = (ctypes.c_int64 * 1)(1)
    _check(_lib.engine_tensor_create(ctypes.byref(h), shape, 1, 0),
           "tensor_create")
    t = Tensor(h, owns=True)
    arr = (ctypes.c_float * 1)(float(v))
    _check(_lib.engine_tensor_copy_from(t.raw, arr, 4), "scalar_copy")
    return t












# ==================================================================
#  Hata yönetimi
# ==================================================================

class PdefaError(RuntimeError):
    pass


def _check(rc: int, ctx: str = ""):
    if rc != 0:
        msg = _lib.pdefa_last_error().decode("utf-8", "replace") if _lib.pdefa_last_error() else "unknown"
        raise PdefaError(f"{ctx}: {msg}" if ctx else msg)


# ==================================================================
#  Meta
# ==================================================================

def version() -> str:    return _lib.pdefa_version().decode()
def simd_level() -> str: return _lib.engine_simd_level().decode()
def cpu_brand() -> str:  return _lib.engine_cpu_brand().decode()


@dataclass
class PreprocessMeta:
    target_w: int = 0
    target_h: int = 0
    orig_w:   int = 0
    orig_h:   int = 0
    pad_x:    int = 0
    pad_y:    int = 0
    scaled_w: int = 0
    scaled_h: int = 0
    ratio:    float = 1.0

    @classmethod
    def from_c(cls, c: _PreMeta) -> "PreprocessMeta":
        return cls(
            target_w=c.target_w, target_h=c.target_h,
            orig_w=c.orig_w,     orig_h=c.orig_h,
            pad_x=c.pad_x,       pad_y=c.pad_y,
            scaled_w=c.scaled_w, scaled_h=c.scaled_h,
            ratio=float(c.ratio),
        )


@dataclass
class Box:
    x1: float; y1: float; x2: float; y2: float


@dataclass
class TensorInfo:
    name: str
    shape: tuple
    dtype: int


# ==================================================================
#  Tensor
# ==================================================================

class Tensor:
    def __init__(self, handle, owns: bool = True):
        self._owns = owns
        if isinstance(handle, ctypes.c_void_p):
            self._handle = handle
        elif handle is None:
            self._handle = ctypes.c_void_p()
        else:
            self._handle = ctypes.c_void_p(handle)

    def __del__(self):
        try:
            if (getattr(self, "_owns", False)
                    and getattr(self, "_handle", None)
                    and self._handle
                    and self._handle.value):
                _lib.engine_tensor_destroy(self._handle)
        except Exception:
            pass

    @property
    def raw(self):
        return self._handle

    @property
    def numel(self) -> int:
        return int(_lib.engine_tensor_numel(self._handle))

    @property
    def ndim(self) -> int:
        return int(_lib.engine_tensor_rank(self._handle))

    @property
    def shape(self):
        n = self.ndim
        if n <= 0:
            return ()
        buf = (ctypes.c_int64 * n)()
        nd  = ctypes.c_int32(0)
        _lib.engine_tensor_shape(self._handle, buf, ctypes.byref(nd))
        return tuple(int(buf[i]) for i in range(nd.value))



        # ------------------------------------------------------------------
    #  YENİ — numpy'siz veri erişimi
    # ------------------------------------------------------------------

    def tolist(self):
        """Nested Python list — numpy olmadan."""
        ptr  = ctypes.c_void_p()
        size = ctypes.c_size_t()
        _check(_lib.engine_tensor_data(self._handle, ctypes.byref(ptr),
                                        ctypes.byref(size)), "tensor_data")
        dt = _lib.engine_tensor_dtype(self._handle)
        if dt == 5:      # I32
            n = size.value // 4
            raw = (ctypes.c_int32 * n).from_address(ptr.value)
        elif dt == 6:    # I64
            n = size.value // 8
            raw = (ctypes.c_int64 * n).from_address(ptr.value)
        elif dt == 3:    # I8
            n = size.value
            raw = (ctypes.c_int8 * n).from_address(ptr.value)
        elif dt == 4:    # U8
            n = size.value
            raw = (ctypes.c_uint8 * n).from_address(ptr.value)
        else:            # F32 (default)
            n = size.value // 4
            raw = (ctypes.c_float * n).from_address(ptr.value)
        flat = list(raw)
        return _reshape_list(flat, self.shape)

    def to_numpy(self):
        """Opsiyonel — numpy kurulu değilse ImportError."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy kurulu değil; .tolist() kullanın")
        ptr  = ctypes.c_void_p()
        size = ctypes.c_size_t()
        _check(_lib.engine_tensor_data(self._handle, ctypes.byref(ptr),
                                        ctypes.byref(size)), "tensor_data")
        buf = (ctypes.c_uint8 * size.value).from_address(ptr.value)
        return np.frombuffer(bytes(buf), dtype=np.float32).reshape(self.shape)

    def numpy(self):
        """Geri uyumluluk — to_numpy() ile aynı."""
        return self.to_numpy()

    def fill(self, value: float):
        """In-place skaler doldur."""
        n = self.numel
        arr = (ctypes.c_float * n)(*([float(value)] * n))
        _check(_lib.engine_tensor_copy_from(self._handle, arr, n * 4), "fill")
        return self

    def copy_from_list(self, data):
        """Python list → bu tensor (in-place)."""
        flat, shape = _flatten_list(data)
        if tuple(shape) != self.shape:
            raise PdefaError(f"shape mismatch: {tuple(shape)} != {self.shape}")
        arr = (ctypes.c_float * len(flat))(*flat)
        _check(_lib.engine_tensor_copy_from(self._handle, arr,
                                             len(flat) * 4), "copy_from_list")
        return self

    # ------------------------------------------------------------------
    #  Aritmetik operatörler
    # ------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, Tensor):
            return ops_add(self, other)
        return ops_add(self, _scalar_tensor(float(other)))

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return ops_sub(self, other)
        return ops_sub(self, _scalar_tensor(float(other)))

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return ops_mul(self, other)
        return ops_mul(self, _scalar_tensor(float(other)))

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return ops_div(self, other)
        return ops_div(self, _scalar_tensor(float(other)))

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires Tensor")
        sa, sb = self.shape, other.shape
        if len(sa) != 2 or len(sb) != 2:
            raise PdefaError("matmul: 2D tensors required")
        if sa[1] != sb[0]:
            raise PdefaError(
                f"matmul: inner dims mismatch {sa} @ {sb}"
            )
        M, N = sa[0], sb[1]
        out = tensor_empty([M, N])
        _check(_lib.engine_ops_matmul(self._handle, other._handle, out._handle),
               "ops_matmul")
        return out

    def __neg__(self):
        return ops_mul(self, _scalar_tensor(-1.0))


    def __repr__(self):
        return f"<Tensor shape={self.shape} numel={self.numel}>"


# ==================================================================
#  Image
# ==================================================================

class Image:
    def __init__(self, handle):
        if isinstance(handle, ctypes.c_void_p):
            self._handle = handle
        elif handle is None:
            self._handle = ctypes.c_void_p()
        else:
            self._handle = ctypes.c_void_p(handle)

    def __del__(self):
        try:
            if (getattr(self, "_handle", None)
                    and self._handle
                    and self._handle.value):
                _lib.pdefa_image_free(self._handle)
        except Exception:
            pass

    @property
    def width(self) -> int:
        return int(_lib.pdefa_image_width(self._handle))

    @property
    def height(self) -> int:
        return int(_lib.pdefa_image_height(self._handle))

    @property
    def channels(self) -> int:
        return int(_lib.pdefa_image_channels(self._handle))

    @property
    def format(self) -> int:
        return int(_lib.pdefa_image_format(self._handle))

    def pixels_bytes(self) -> bytes:
        size = ctypes.c_size_t(0)
        ptr  = _lib.pdefa_image_pixels(self._handle, ctypes.byref(size))
        if not ptr or size.value == 0:
            return b""
        return bytes(bytearray(ptr[i] for i in range(size.value)))

    def tobytes(self) -> bytes:
        """Ham pixel verisi (H*W*C bytes)."""
        size = ctypes.c_size_t(0)
        ptr  = _lib.pdefa_image_pixels(self._handle, ctypes.byref(size))
        if not ptr or size.value == 0:
            return b""
        return bytes(bytearray(ptr[i] for i in range(size.value)))

    def tolist(self):
        """Nested Python list [H][W][C] — numpy'siz."""
        b = self.tobytes()
        if not b:
            return []
        H, W, C = self.height, self.width, self.channels
        out = []
        i = 0
        for h in range(H):
            row = []
            for w in range(W):
                row.append(list(b[i:i+C]))
                i += C
            out.append(row)
        return out

    def numpy(self):
        """[OPSİYONEL] numpy kurulu değilse ImportError."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy kurulu değil; '.tolist()' kullanın")
        b = self.tobytes()
        if not b:
            return None
        arr = np.frombuffer(b, dtype=np.uint8)
        return arr.reshape(self.height, self.width, self.channels)

    def __repr__(self):
        return f"<Image {self.width}x{self.height} ch={self.channels} fmt={self.format}>"


def load_image(path: str) -> Image:
    h = _lib.pdefa_image_load(path.encode("utf-8"))
    if not h:
        raise PdefaError(f"load_image({path}): {_lib.pdefa_last_error().decode()}")
    return Image(h)


# ==================================================================
#  Preprocess
# ==================================================================

def preprocess_letterbox(
    img: Image,
    target_w: int = 640, target_h: int = 640,
    scale: float = 1.0 / 255.0, pad_value: float = 0.0,
) -> Tuple[Tensor, PreprocessMeta]:
    cfg = _PreConfig(target_w=target_w, target_h=target_h,
                     scale=scale, pad_value=pad_value)
    t = ctypes.c_void_p()
    meta = _PreMeta()
    rc = _lib.pdefa_preprocess_letterbox(img._handle, ctypes.byref(cfg),
                                          ctypes.byref(t), ctypes.byref(meta))
    _check(rc, "preprocess_letterbox")
    return Tensor(t, owns=True), PreprocessMeta.from_c(meta)


def prepare_for_engine(
    path: str,
    target_w: int = 640, target_h: int = 640,
    scale: float = 1.0 / 255.0, pad_value: float = 0.0,
) -> Tuple[Tensor, PreprocessMeta]:
    t = ctypes.c_void_p()
    meta = _PreMeta()
    rc = _lib.pdefa_prepare_for_engine(
        path.encode("utf-8"), target_w, target_h, scale, pad_value,
        ctypes.byref(t), ctypes.byref(meta))
    _check(rc, "prepare_for_engine")
    return Tensor(t, owns=True), PreprocessMeta.from_c(meta)


def load_image_as_tensor(path: str, target_w: int = 640, target_h: int = 640) -> Tensor:
    t = ctypes.c_void_p()
    rc = _lib.pdefa_load_image_as_tensor(
        path.encode("utf-8"), target_w, target_h, ctypes.byref(t))
    _check(rc, "load_image_as_tensor")
    return Tensor(t, owns=True)


# ==================================================================
#  Post-process
# ==================================================================

def map_box_to_original(box: Box, meta: PreprocessMeta) -> Box:
    c = _PreMeta(
        target_w=meta.target_w, target_h=meta.target_h,
        orig_w=meta.orig_w, orig_h=meta.orig_h,
        pad_x=meta.pad_x, pad_y=meta.pad_y,
        scaled_w=meta.scaled_w, scaled_h=meta.scaled_h,
        ratio=meta.ratio,
    )
    in_box  = (ctypes.c_float * 4)(box.x1, box.y1, box.x2, box.y2)
    out_box = (ctypes.c_float * 4)()
    rc = _lib.pdefa_map_box_to_original(in_box, ctypes.byref(c), out_box)
    _check(rc, "map_box_to_original")
    return Box(out_box[0], out_box[1], out_box[2], out_box[3])


def map_point_to_original(x: float, y: float, meta: PreprocessMeta) -> Tuple[float, float]:
    c = _PreMeta(
        target_w=meta.target_w, target_h=meta.target_h,
        orig_w=meta.orig_w, orig_h=meta.orig_h,
        pad_x=meta.pad_x, pad_y=meta.pad_y,
        scaled_w=meta.scaled_w, scaled_h=meta.scaled_h,
        ratio=meta.ratio,
    )
    cx = ctypes.c_float(x); cy = ctypes.c_float(y)
    rc = _lib.pdefa_map_point_to_original(ctypes.byref(cx), ctypes.byref(cy), ctypes.byref(c))
    _check(rc, "map_point_to_original")
    return float(cx.value), float(cy.value)


# ==================================================================
#  Context / Model / Session
# ==================================================================

class Context:
    def __init__(self):
        self._handle = ctypes.c_void_p()
        _check(_lib.engine_context_create(ctypes.byref(self._handle)), "context_create")

    def __del__(self):
        try:
            if getattr(self, "_handle", None) and self._handle and self._handle.value:
                _lib.engine_context_destroy(self._handle)
        except Exception:
            pass

    @property
    def raw(self):
        return self._handle

    def set_num_threads(self, n: int):
        """Motor thread pool boyutunu ayarla. n > 0 olmalı."""
        _check(_lib.engine_context_set_num_threads(self._handle, int(n)),
               "set_num_threads")
        return self

    def get_num_threads(self) -> int:
        """Aktif thread sayısı."""
        return int(_lib.engine_context_get_num_threads(self._handle))


class Model:
    def __init__(self, ctx: Context, path: str):
        self._handle = ctypes.c_void_p()
        _check(_lib.engine_model_load(ctx.raw, path.encode("utf-8"),
                                       ctypes.byref(self._handle)), "model_load")

    def __del__(self):
        try:
            if getattr(self, "_handle", None) and self._handle and self._handle.value:
                _lib.engine_model_destroy(self._handle)
        except Exception:
            pass

    @property
    def raw(self):
        return self._handle

    @property
    def name(self) -> str:
        p = ctypes.c_char_p()
        _check(_lib.engine_model_name(self._handle, ctypes.byref(p)), "model_name")
        return p.value.decode() if p.value else ""

    def input_info(self, idx: int) -> TensorInfo:
        return _model_info(self._handle, idx, _lib.engine_model_input_info)

    def output_info(self, idx: int) -> TensorInfo:
        return _model_info(self._handle, idx, _lib.engine_model_output_info)


def _model_info(handle, idx: int, fn) -> TensorInfo:
    name  = ctypes.c_char_p()
    shape = (ctypes.c_int64 * 8)()
    ndim  = ctypes.c_int32(0)
    dt    = ctypes.c_int(0)
    rc = fn(handle, idx,
            ctypes.byref(name),
            shape, ctypes.byref(ndim), ctypes.byref(dt))
    if rc != 0:
        raise PdefaError(f"model_info(idx={idx}): rc={rc}")
    n = ndim.value
    return TensorInfo(
        name=name.value.decode() if name.value else f"tensor_{idx}",
        shape=tuple(int(shape[i]) for i in range(n)),
        dtype=int(dt.value),
    )


class Session:
    def run_and_decode(self, meta: "PreprocessMeta" = None,
                       conf_thres: float = 0.25,
                       iou_thres: float = 0.45):
        self.run()
        out = self.output(0)              # Tensor
        return decode_yolo(out, meta, conf_thres, iou_thres)
    

    def __init__(self, ctx: Context, model: Model):
        self._handle = ctypes.c_void_p()
        _check(_lib.engine_session_create(ctx.raw, model.raw,
                                           ctypes.byref(self._handle)), "session_create")
        # Input tensor'ları canlı tut — aksi halde Python geçici
        # Tensor'ü GC'ye bırakır, motor dangling pointer okur.
        self._input_refs = {}

    def __del__(self):
        try:
            if getattr(self, "_handle", None) and self._handle and self._handle.value:
                _lib.engine_session_destroy(self._handle)
        except Exception:
            pass




  
    @property
    def raw(self):
        return self._handle

    def set_input(self, index: int, tensor: Tensor):
        _check(_lib.engine_session_set_input(self._handle, index, tensor.raw),
               "session_set_input")
        self._input_refs[index] = tensor   # ← canlı tut

    def run(self):
        _check(_lib.engine_session_run(self._handle), "session_run")

    def output(self, index: int = 0) -> Tensor:
        h = ctypes.c_void_p()
        _check(_lib.engine_session_get_output(self._handle, index, ctypes.byref(h)),
               "session_get_output")
        return Tensor(h, owns=False)

    def stats(self):
        s = _SessionStats()
        _check(_lib.engine_session_get_stats(self._handle, ctypes.byref(s)),
               "session_get_stats")
        return {"last_run_ms": s.last_run_ms, "runs": int(s.runs)}

    def debug_intermediate(self, idx: int) -> Tensor:
        h = ctypes.c_void_p()
        _check(_lib.engine_session_debug_intermediate(
            self._handle, idx, ctypes.byref(h)), "debug_intermediate")
        return Tensor(h, owns=False)


# ==================================================================
#  Ops
# ==================================================================

def tensor_empty(shape, dtype: int = 0) -> Tensor:
    n = len(shape)
    arr = (ctypes.c_int64 * n)(*shape)
    h = ctypes.c_void_p()
    _check(_lib.engine_tensor_create(ctypes.byref(h), arr, n, dtype), "tensor_create")
    return Tensor(h, owns=True)


def ops_matmul(a: Tensor, b: Tensor, out: Tensor = None) -> Tensor:
    if out is None:
        sa, sb = a.shape, b.shape
        if len(sa) != 2 or len(sb) != 2:
            raise PdefaError("ops_matmul: 2D tensors required")
        if sa[1] != sb[0]:
            raise PdefaError(
                f"ops_matmul: inner dims mismatch {sa} @ {sb}"
            )
        M, N = sa[0], sb[1]
        out = tensor_empty([M, N], dtype=0)
    _check(_lib.engine_ops_matmul(a.raw, b.raw, out.raw), "ops_matmul")
    return out


def ops_relu(t: Tensor) -> Tensor:
    _check(_lib.engine_ops_relu(t.raw), "ops_relu")
    return t


def ops_silu(t: Tensor) -> Tensor:
    _check(_lib.engine_ops_silu(t.raw), "ops_silu")
    return t




# ==================================================================
#  Module-level tensor creation (numpy'siz)
# ==================================================================

def randn(*shape, seed=0):
    """Normal dağılım N(0,1). randn(3, 4) veya randn([3, 4])."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    return random_randn(list(shape), seed)


def rand(*shape, seed=0):
    """Uniform [0, 1)."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    return random_rand(list(shape), seed)


def zeros(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    t = tensor_empty(list(shape))
    t.fill(0.0)
    return t


def ones(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    t = tensor_empty(list(shape))
    t.fill(1.0)
    return t


def empty(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    return tensor_empty(list(shape))


def from_list(data) -> Tensor:
    """Python nested list → Tensor."""
    flat, shape = _flatten_list(data)
    t = tensor_empty(shape)
    arr = (ctypes.c_float * len(flat))(*flat)
    _check(_lib.engine_tensor_copy_from(t.raw, arr, len(flat) * 4), "from_list")
    return t












# ==================================================================
#  YOLO Decode — numpy'siz, pdefa.Tensor üzerinden
# ==================================================================

def decode_yolo(output, meta=None, conf_thres=0.25,
                iou_thres=0.45, max_det=300):
    """
    YOLOv8/v9 ham çıktısını bbox listesine çevirir.

    Args:
        output: pdefa.Tensor (1, 84, 8400) veya (84, 8400) veya
                Python nested list.
        meta  : PreprocessMeta veya None
    Returns:
        [{'x1','y1','x2','y2','conf','cls'}, ...]
    """
    # 1) Tensor → nested list
    if isinstance(output, Tensor):
        arr = output.tolist()
    else:
        arr = output

    # 2) Batch unwrap
    while (isinstance(arr, list) and len(arr) == 1
           and isinstance(arr[0], list) and len(arr[0]) > 4):
        arr = arr[0]

    if not isinstance(arr, list) or len(arr) < 5:
        raise PdefaError(f"decode_yolo: beklenmeyen şekil {type(arr)}")

    n_ch = len(arr)
    n_anchors = len(arr[0])
    n_classes = n_ch - 4

    # 3) Her anchor için max sınıf
    boxes = []
    for i in range(n_anchors):
        best_c = 0
        best_s = arr[4][i]
        for c in range(1, n_classes):
            s = arr[4 + c][i]
            if s > best_s:
                best_s = s
                best_c = c
        if best_s < conf_thres:
            continue

        cx = arr[0][i]; cy = arr[1][i]
        w  = arr[2][i]; h  = arr[3][i]
        boxes.append((cx - w * 0.5, cy - h * 0.5,
                      cx + w * 0.5, cy + h * 0.5,
                      best_s, best_c))

    if not boxes:
        return []

    # 4) Letterbox → orijinal koordinat
    if meta is not None:
        scale = 1.0 / meta.ratio
        px, py = meta.pad_x, meta.pad_y
        ow, oh = meta.orig_w, meta.orig_h
        moved = []
        for b in boxes:
            x1 = (b[0] - px) * scale
            y1 = (b[1] - py) * scale
            x2 = (b[2] - px) * scale
            y2 = (b[3] - py) * scale
            x1 = 0.0 if x1 < 0 else (ow if x1 > ow else x1)
            y1 = 0.0 if y1 < 0 else (oh if y1 > oh else y1)
            x2 = 0.0 if x2 < 0 else (ow if x2 > ow else x2)
            y2 = 0.0 if y2 < 0 else (oh if y2 > oh else y2)
            moved.append((x1, y1, x2, y2, b[4], b[5]))
        boxes = moved

    # 5) NMS
    keep = _nms(boxes, iou_thres)
    if len(keep) > max_det:
        keep = keep[:max_det]

    return [
        {"x1": boxes[i][0], "y1": boxes[i][1],
         "x2": boxes[i][2], "y2": boxes[i][3],
         "conf": boxes[i][4], "cls": boxes[i][5]}
        for i in keep
    ]


def _iou(a, b):
    x1 = a[0] if a[0] > b[0] else b[0]
    y1 = a[1] if a[1] > b[1] else b[1]
    x2 = a[2] if a[2] < b[2] else b[2]
    y2 = a[3] if a[3] < b[3] else b[3]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _nms(boxes, iou_thres):
    """Sınıf bazlı NMS — numpy'siz."""
    from collections import defaultdict
    by_class = defaultdict(list)
    for i, b in enumerate(boxes):
        by_class[b[5]].append(i)

    keep_all = []
    for cls, idxs in by_class.items():
        idxs.sort(key=lambda i: boxes[i][4], reverse=True)
        while idxs:
            cur = idxs.pop(0)
            keep_all.append(cur)
            if not idxs:
                break
            cur_box = boxes[cur]
            idxs = [j for j in idxs if _iou(cur_box, boxes[j]) <= iou_thres]

    keep_all.sort(key=lambda i: boxes[i][4], reverse=True)
    return keep_all

# ==================================================================
#  Model input shape'i otomatik al
# ==================================================================

def _model_input_shape(model: "Model", idx: int = 0):
    """Model.input_shape(0) -> (1, 3, 640, 640)"""
    info = model.input_info(idx)
    return info.shape


def prepare_for_model(model: "Model", path: str,
                      input_idx: int = 0,
                      scale: float = 1.0 / 255.0,
                      pad_value: float = 0.0):
    """
    Model input shape'ine göre otomatik preprocess.

    Kullanım:
        tensor, meta = pdefa.prepare_for_model(model, "image.png")
        # otomatik olarak model'in beklediği H, W ile hazırlar
    """
    shape = _model_input_shape(model, input_idx)
    if len(shape) != 4:
        raise PdefaError(f"prepare_for_model: beklenen 4D input, gelen {shape}")
    _, _, H, W = shape
    return prepare_for_engine(path, W, H, scale, pad_value)






# ==================================================================
#  Kolay API — pdefa.load("model.engine")("image.jpg")
# ==================================================================

class EngineModel:
    """
    Tek satırda model yükle + inference + decode.

    Kullanım:
        model = pdefa.load("yolov8n.engine")
        boxes = model("image.jpg", conf_thres=0.25)
    """
    def __init__(self, ctx: "Context", model: "Model", session: "Session"):
        self.ctx     = ctx
        self.model   = model
        self.session = session

    def __call__(self, image_path: str,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45):
        tensor, meta = prepare_for_model(self.model, image_path)
        self.session.set_input(0, tensor)
        self.session.run()
        return decode_yolo(self.session.output(0), meta,
                           conf_thres, iou_thres)

    def set_num_threads(self, n: int):
        self.ctx.set_num_threads(n)
        return self


def load(engine_path: str) -> EngineModel:
    """Model dosyasını yükle, tek satırda inference için."""
    ctx     = Context()
    model   = Model(ctx, engine_path)
    session = Session(ctx, model)
    return EngineModel(ctx, model, session)



   