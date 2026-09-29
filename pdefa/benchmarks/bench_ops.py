# pdefa - AGPL-3.0-or-later - see LICENSE
"""Ops micro-benchmark. Usage: python bench_ops.py"""
import time
import pdefa


def bench(name, fn, iters=100, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:30s} {dt:8.3f} ms/iter")


def main():
    print("pdefa ops benchmark")
    print(f"  simd: {pdefa.simd_level()}   cpu: {pdefa.cpu_brand()}")
    print()

    A = pdefa.randn(512, 512, seed=0)
    B = pdefa.randn(512, 512, seed=1)
    bench("matmul 512x512", lambda: A @ B)

    T = pdefa.randn(1024, 1024, seed=0)
    bench("relu 1024x1024",    lambda: pdefa.ops_relu(T))
    bench("sigmoid 1024x1024", lambda: pdefa.ops_sigmoid(T))
    bench("softmax 1024x1024", lambda: pdefa.ops_softmax(T, axis=-1))

    R = pdefa.randn(64, 1000, seed=0)
    bench("reduce_sum axis=1", lambda: pdefa.ops_reduce_sum(R, axis=1))


if __name__ == "__main__":
    main()