"""
100 tane random PNG uretir (hem non-interlaced hem interlaced).
Cikti: D:\\pngtest\\NNN_orig.png  ve  NNN_int.png
"""
import os
import random
import struct
import zlib
from PIL import Image

OUT_DIR = r"D:\pngtest"
ADAM7 = [
    (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
    (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)
]

def png_chunk(ctype, data):
    chunk = ctype + data
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

def save_interlaced(img, path):
    """RGBA modunda, Adam7 pattern ile interlaced PNG yaz."""
    img = img.convert("RGBA")
    w, h = img.size
    pixels = img.load()

    raw = b""
    for (xs, ys, xstep, ystep) in ADAM7:
        pw = (w - xs + xstep - 1) // xstep
        ph = (h - ys + ystep - 1) // ystep
        if pw == 0 or ph == 0:
            continue
        for py in range(ph):
            raw += b"\x00"  # filter = None
            for px in range(pw):
                sx = xs + px * xstep
                sy = ys + py * ystep
                r, g, b, a = pixels[sx, sy]
                raw += bytes((r, g, b, a))

    z = zlib.compress(raw, 6)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 1)  # interlace=1

    with open(path, "wb") as f:
        f.write(sig)
        f.write(png_chunk(b"IHDR", ihdr))
        f.write(png_chunk(b"IDAT", z))
        f.write(png_chunk(b"IEND", b""))

def random_image(w, h):
    """Random pattern uret (gradient + noise karisik)."""
    img = Image.new("RGBA", (w, h))
    px = img.load()
    mode = random.choice(["gradient", "noise", "blocks"])
    for y in range(h):
        for x in range(w):
            if mode == "gradient":
                r = (x * 255) // max(w - 1, 1)
                g = (y * 255) // max(h - 1, 1)
                b = ((x + y) * 255) // max(w + h - 2, 1)
                a = 255
            elif mode == "noise":
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                a = random.randint(0, 255)
            else:  # blocks
                bx, by = x // 8, y // 8
                r = (bx * 37) % 256
                g = (by * 53) % 256
                b = ((bx + by) * 71) % 256
                a = 255 if (bx + by) % 2 == 0 else 128
            px[x, y] = (r, g, b, a)
    return img

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)  # tekrarlanabilir

    N = 100
    for i in range(N):
        # Rastgele boyut: 1x1 ile 200x200 arasi
        w = random.randint(1, 200)
        h = random.randint(1, 200)

        img = random_image(w, h)

        orig = os.path.join(OUT_DIR, f"{i:03d}_orig.png")
        inter = os.path.join(OUT_DIR, f"{i:03d}_int.png")

        img.save(orig, "PNG", interlace=False)
        save_interlaced(img, inter)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N} uretildi")

    print(f"Bitti: {N} PNG cifti -> {OUT_DIR}")