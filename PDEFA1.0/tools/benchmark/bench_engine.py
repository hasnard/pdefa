#!/usr/bin/env python3
"""bench_engine.py — Engine-AI icin cross-platform (Windows DLL + Linux SO) ctypes benchmark."""
import ctypes as C
import os, sys, time, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CONV = os.path.join(ROOT, 'tools', 'converter')

IS_WIN = platform.system() == 'Windows'

# Windows: engine.dll build/bin/Release altinda
# Linux:   libengine.so build/lib altinda
if IS_WIN:
    CANDIDATES = [
        os.path.join(ROOT, 'build', 'bin', 'Release', 'engine.dll'),
        os.path.join(ROOT, 'build', 'bin', 'engine.dll'),
        os.path.join(ROOT, 'build', 'engine.dll'),
    ]
else:
    CANDIDATES = [
        os.path.join(ROOT, 'build', 'lib', 'libengine.so'),
        os.path.join(ROOT, 'build', 'libengine.so'),
    ]

SO = next((p for p in CANDIDATES if os.path.exists(p)), None)
if SO is None:
    print("[HATA] Engine kutuphanesi bulunamadi. Denenen:")
    for p in CANDIDATES:
        print("  " + p)
    sys.exit(1)
print("[lib] " + SO)

lib = C.CDLL(SO)

lib.engine_version.restype = C.c_char_p
lib.engine_cpu_brand.restype = C.c_char_p
lib.engine_simd_level.restype = C.c_char_p
lib.engine_last_error.restype = C.c_char_p

lib.engine_context_create.argtypes = [C.POINTER(C.c_void_p)]
lib.engine_context_create.restype  = C.c_int
lib.engine_context_set_num_threads.argtypes = [C.c_void_p, C.c_int32]
lib.engine_context_set_num_threads.restype  = C.c_int
lib.engine_context_destroy.argtypes = [C.c_void_p]

lib.engine_model_load.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
lib.engine_model_load.restype  = C.c_int
lib.engine_model_destroy.argtypes = [C.c_void_p]

lib.engine_tensor_create_from_data.argtypes = [
    C.POINTER(C.c_void_p), C.POINTER(C.c_int64), C.c_int32,
    C.c_int, C.c_void_p, C.c_size_t]
lib.engine_tensor_create_from_data.restype = C.c_int
lib.engine_tensor_destroy.argtypes = [C.c_void_p]

lib.engine_session_create.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)]
lib.engine_session_create.restype  = C.c_int
lib.engine_session_set_input.argtypes = [C.c_void_p, C.c_int32, C.c_void_p]
lib.engine_session_set_input.restype  = C.c_int
lib.engine_session_run.argtypes = [C.c_void_p]
lib.engine_session_run.restype  = C.c_int
lib.engine_session_destroy.argtypes = [C.c_void_p]

def check(status, label):
    if status != 0:
        err = lib.engine_last_error()
        msg = err.decode() if err else "?"
        raise RuntimeError(label + " failed: status=" + str(status) + " (" + msg + ")")

print("[cpu]  " + lib.engine_cpu_brand().decode())
print("[simd] " + lib.engine_simd_level().decode())
print("[ver]  " + lib.engine_version().decode())

MODEL = os.path.join(CONV, "yolov8s.engine")
INPUT = os.path.join(CONV, "yolov8s_input.f32")

for p in (MODEL, INPUT):
    if not os.path.exists(p):
        print("[HATA] eksik: " + p)
        sys.exit(1)

# Thread sayisini komut satirindan al, varsayilan 4 (Windows'ta 12 yapabilirsin)
THREADS = int(os.environ.get('BENCH_THREADS', '4'))
WARMUP, RUNS = 10, 50

ctx = C.c_void_p()
check(lib.engine_context_create(C.byref(ctx)), "context_create")
check(lib.engine_context_set_num_threads(ctx, THREADS), "set_num_threads")

model = C.c_void_p()
check(lib.engine_model_load(ctx, MODEL.encode('utf-8'), C.byref(model)), "model_load")

in_np = np.fromfile(INPUT, dtype=np.float32)
SHAPE = (1, 3, 640, 640)
assert in_np.size == 1*3*640*640, "input size mismatch: " + str(in_np.size)
shape_arr = (C.c_int64 * 4)(*SHAPE)

in_t = C.c_void_p()
check(lib.engine_tensor_create_from_data(
    C.byref(in_t), shape_arr, 4, 0,
    in_np.ctypes.data_as(C.c_void_p), in_np.nbytes), "tensor_create_from_data")

sess = C.c_void_p()
check(lib.engine_session_create(ctx, model, C.byref(sess)), "session_create")
check(lib.engine_session_set_input(sess, 0, in_t), "set_input")

print("[warmup] {} run".format(WARMUP))
for _ in range(WARMUP):
    check(lib.engine_session_run(sess), "session_run(warmup)")

times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    check(lib.engine_session_run(sess), "session_run")
    times.append((time.perf_counter() - t0) * 1000.0)

ts = sorted(times)
n = len(ts)
median = ts[n // 2]
p95 = ts[int(0.95 * n)]
mn, mx = ts[0], ts[-1]
fps = 1000.0 / median

print()
print("=" * 60)
print("  Engine-AI  YOLOv8s 640x640  ({} thread)".format(THREADS))
print("=" * 60)
print("  min    = {:8.3f} ms".format(mn))
print("  median = {:8.3f} ms   ({:6.1f} FPS)".format(median, fps))
print("  p95    = {:8.3f} ms".format(p95))
print("  max    = {:8.3f} ms".format(mx))
print("=" * 60)
print("Engine-AI {:.3f} ms".format(median))

lib.engine_tensor_destroy(in_t)
lib.engine_session_destroy(sess)
lib.engine_model_destroy(model)
lib.engine_context_destroy(ctx)