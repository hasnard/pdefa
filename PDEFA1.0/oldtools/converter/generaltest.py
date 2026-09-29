import struct, glob, os

for fpath in glob.glob(r'D:\PROJE\**\*.engine', recursive=True):
    try:
        with open(fpath, 'rb') as f:
            f.read(8)
            nw = struct.unpack('<I', f.read(4))[0]
            nn = struct.unpack('<I', f.read(4))[0]
            f.read(8 + 64)
            for _ in range(nw):
                nl = struct.unpack('<I', f.read(4))[0]
                f.read(nl)
                f.read(1)
                rank = f.read(1)[0]
                f.read(8*rank + 16)
            op = f.read(1)[0]
            nl = struct.unpack('<I', f.read(4))[0]
            name = f.read(nl).decode(errors='replace')
            print(f"{fpath}\n   nodes={nn}  first='{name}'")
    except Exception as e:
        print(f"{fpath}: ERR {e}")