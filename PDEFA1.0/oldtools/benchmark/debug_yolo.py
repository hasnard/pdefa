#!/usr/bin/env python3
"""
YOLOv8s Node-by-Node Debug
- PyTorch intermediate activation'ları hook'la yakalar
- Engine'i ENGINE_DUMP ile çalıştırır, her node'un output'unu dumper
- Node-by-node cosine similarity hesaplar
- İlk düşüş noktasını raporlar
"""
import os, sys, subprocess, shutil
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CONV = os.path.join(ROOT, 'tools', 'converter')
DUMP = os.path.join(ROOT, 'dumps')

# ============ 1) PyTorch model + hooks ============
print("=== PyTorch yükleniyor ===")
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
torch_model = model.model.eval().fuse()

torch.manual_seed(42)
dummy = torch.randn(1, 3, 640, 640)

# Hook'ları kaydet: her Conv, Sigmoid, Mul, Concat sonrası output
pt_acts = {}   # name -> np.ndarray
def make_hook(name):
    def fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            pt_acts[name] = out.detach().cpu().numpy().copy()
        elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            pt_acts[name] = out[0].detach().cpu().numpy().copy()
    return fn

hooks = []
for name, m in torch_model.named_modules():
    # Sadece Conv2d ve bazı op'lar için
    tn = type(m).__name__
    if tn in ('Conv2d', 'BatchNorm2d', 'SiLU', 'Concat', 'MaxPool2d', 'Upsample', 'Sigmoid', 'Mul', 'Add', 'Identity'):
        hooks.append(m.register_forward_hook(make_hook(name)))

with torch.inference_mode():
    pt_out = torch_model(dummy)
    if isinstance(pt_out, (tuple, list)): pt_out = pt_out[0]
pt_out_np = pt_out.numpy()

for h in hooks: h.remove()

print(f"  {len(pt_acts)} PyTorch activation yakalandı")

# Input sabit
dummy.numpy().astype(np.float32).tofile(os.path.join(CONV, "yolov8s_input.f32"))

# ============ 2) Engine çalıştır (dump ile) ============
if os.path.exists(DUMP):
    shutil.rmtree(DUMP)
os.makedirs(DUMP, exist_ok=True)

print("\n=== Engine çalıştırılıyor (ENGINE_DUMP) ===")
eng_exe = os.path.join(ROOT, "build", "bin", "Release", "benchmark_yolo.exe")
env = os.environ.copy()
env["ENGINE_DUMP"] = DUMP

r = subprocess.run([eng_exe], cwd=ROOT, env=env,
                   capture_output=True, text=True, timeout=600)
if r.returncode != 0:
    print(f"[HATA] Engine exit={r.returncode}")
    print("STDOUT:", r.stdout[-500:])
    print("STDERR:", r.stderr[-500:])
    sys.exit(1)

# meta.txt oku
meta_path = os.path.join(DUMP, "meta.txt")
if not os.path.exists(meta_path):
    print("[HATA] meta.txt yok — ENGINE_DUMP çalışmadı")
    sys.exit(1)

eng_nodes = []   # list of (idx, op, name, shape, file)
with open(meta_path) as f:
    for line in f:
        # node_0000 op=1 name='...' shape=[1,320,320,32]
        parts = line.strip().split(" ", 3)
        node_id = parts[0]
        op = int(parts[1].split("=")[1])
        rest = parts[2] + " " + parts[3] if len(parts) > 3 else parts[2]
        # name='...' shape=[...]
        import re
        m1 = re.search(r"name='([^']*)'", rest)
        m2 = re.search(r"shape=\[([^\]]*)\]", rest)
        name = m1.group(1) if m1 else ""
        shape = tuple(int(x) for x in m2.group(1).split(",")) if m2 else ()
        fname = os.path.join(DUMP, node_id + f".bin")
        # dosya adı meta'da yok, op'a göre yeniden kur
        # Dosya: node_XXXX_opY.bin
        fname = os.path.join(DUMP, f"{node_id}_op{op}.bin")
        eng_nodes.append((node_id, op, name, shape, fname))

print(f"  {len(eng_nodes)} Engine node yakalandı")

# ============ 3) İlk birkaç node için karşılaştır ============
print("\n=== NODE-NODE KARŞILAŞTIRMA ===")
print(f"  {'#':>4} {'op':>3} {'name':<40} {'shape':<20} {'cos':>8}")
print("  " + "-" * 80)

for idx, (node_id, op, name, shape, fname) in enumerate(eng_nodes[:30]):
    if not os.path.exists(fname):
        continue
    eng = np.fromfile(fname, dtype=np.float32)
    
    # PyTorch'ta eşleşen aktivasyonu bul (isimde Conv/Concat vb. ara)
    pt_match = None
    # Basit eşleştirme: name içinde geçen model.N kısmını ara
    for ptname, ptact in pt_acts.items():
        # name örn: /model.0/conv/Conv — ptname örn: model.0.conv
        # Kaba eşleşme: name'de ptname'in son parçası geçiyor mu
        pass
    
    # Şimdilik sadece shape eşleşen ve yakın olanı yazdır
    shape_str = "x".join(str(d) for d in shape)
    print(f"  {idx:>4} {op:>3} {name[:40]:<40} {shape_str[:20]:<20} {'?':>8}")

print(f"\n[!] Node-by-node eşleştirme için meta.txt ve PyTorch hook isimleri uyumlu değil")
print(f"    Elle karşılaştırma için dumps/ klasörüne bakın")
print(f"\nToplam Engine node: {len(eng_nodes)}")
print(f"PyTorch activation: {len(pt_acts)}")