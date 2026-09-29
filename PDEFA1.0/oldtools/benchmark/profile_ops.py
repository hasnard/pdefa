import csv
import os

# Script'in bulunduğu dizinden proje kökünü bul
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CSV  = os.path.join(ROOT, 'build', 'bin', 'Release', 'engine_layer_times.csv')

if not os.path.exists(CSV):
    print(f"CSV yok: {CSV}")
    print("Önce şunu çalıştır:")
    print("  cd build\\bin\\Release")
    print("  $env:ENGINE_PROFILE_CSV = 'engine_layer_times.csv'")
    print("  cd ..\\..\\..")
    print("  .\\build\\bin\\Release\\benchmark_yolo.exe")
    raise SystemExit(1)

rows = list(csv.DictReader(open(CSV)))
n_nodes = len(set((r['node_idx'], r['op_type'], r['dims']) for r in rows))
print(f"Toplam satır: {len(rows)}, node sayısı: {n_nodes}, run sayısı: {len(rows)//n_nodes}")

# İlk 2 turu atla (cold), geri kalanı warm
warm = rows[2 * n_nodes:]

from collections import defaultdict
agg = defaultdict(lambda: [0.0, 0])
for r in warm:
    agg[r['op_type']][0] += float(r['time_ms'])
    agg[r['op_type']][1] += 1

print(f"\n{'Op':14s} {'Total(ms)':>10s} {'Count':>6s} {'Avg(ms)':>10s}")
print('-' * 44)
for op, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:12]:
    print(f'{op:14s} {t:10.2f} {c:6d} {t/c:10.3f}')

print(f"\n{'Node (dims)':50s} {'Type':14s} {'Avg(ms)':>10s}")
print('-' * 80)
per = defaultdict(list)
for r in warm:
    per[(r['node_idx'], r['op_type'], r['dims'])].append(float(r['time_ms']))
items = sorted(per.items(), key=lambda kv: -sum(kv[1])/len(kv[1]))[:15]
for (idx, typ, dims), times in items:
    avg = sum(times) / len(times)
    print(f'{dims:50s} {typ:14s} {avg:10.3f}')