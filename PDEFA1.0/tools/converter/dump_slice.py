import struct

path = r"D:\PROJE\engineforaiincpu\PDEFA1.0\tools\converter\yolov8s.engine"
f = open(path, "rb")

magic, version = struct.unpack("<II", f.read(8))
num_w, num_n, num_in, num_out = struct.unpack("<IIII", f.read(16))
brand = f.read(64)

print(f"w={num_w} n={num_n} in={num_in} out={num_out}")

weights = []
for i in range(num_w):
    nl, = struct.unpack("<I", f.read(4))
    name = f.read(nl).decode('utf-8', errors='replace')
    dt, rk = struct.unpack("<BB", f.read(2))
    dims = struct.unpack(f"<{rk}q", f.read(8*rk)) if rk > 0 else ()
    off, sz = struct.unpack("<QQ", f.read(16))
    weights.append((i, name, dt, dims, off, sz))

# Slice tensor'larını göster: starts, ends, axes isimleri genelde
# "/model.2/Slice/starts" gibi. İsimle filtrele.
print("\n=== Slice-related weights ===")
for (i, name, dt, dims, off, sz) in weights:
    if "Slice" in name or "split" in name.lower() or "/starts" in name or "/ends" in name or "/axes" in name:
        print(f"w[{i}] name='{name}' dt={dt} dims={dims} off={off} sz={sz}")

# "model.2" ile başlayan tüm weight'leri göster
print("\n=== /model.2/ weights ===")
for (i, name, dt, dims, off, sz) in weights:
    if "/model.2/" in name:
        print(f"w[{i}] '{name}' dt={dt} dims={dims} sz={sz}")

f.close()