01_math.inc       → NR, KC sabitleri + exp/silu/transpose
02_microkernels   → 01'deki transpose8x8_ps kullanır
03_weight_cache   → bağımsız
04_im2col         → 01'deki transpose8x8_ps kullanır
05_gemm           → 02'deki microkernel'leri çağırır
06_output         → 01'deki silu, transpose kullanır
07_dispatch       → hepsini kullanır