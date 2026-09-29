import struct

path = 'yolov8s.engine'
with open(path, 'rb') as f:
    magic, ver, nw, nn, ni, no = struct.unpack('<6I', f.read(24))
    f.read(64)  # brand
    print(f"magic=0x{magic:08x}  weights={nw}  nodes={nn}")
    for i in range(nw):
        nlen = struct.unpack('<I', f.read(4))[0]
        name = f.read(nlen).decode()
        dt, rank = struct.unpack('<BB', f.read(2))
        dims = []
        for _ in range(rank):
            d = struct.unpack('<q', f.read(8))[0]
            dims.append(d)
        off, sz = struct.unpack('<QQ', f.read(16))
        nfloats = sz // 4
        # İlginç olanlar: 8400, 1600, 400, 1, 2, 8, 16, 24, 32 içeren küçük tensorlar
        if nfloats <= 32 and 'model.22' in name:
            print(f'  W[{i:3d}] {name:50s} dims={dims} nfloats={nfloats}')