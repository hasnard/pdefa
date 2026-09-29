import sys, struct, zlib
from PIL import Image

def adler32(data):
    a, b = 1, 0
    for byte in data:
        a = (a + byte) % 65521
        b = (b + a) % 65521
    return (b << 16) | a

def png_chunk(ctype, data):
    chunk = ctype + data
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

# Adam7 pattern: (x_start, y_start, x_step, y_step)
ADAM7 = [
    (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
    (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)
]

def make_interlaced(src_path, dst_path):
    img = Image.open(src_path).convert("RGBA")
    w, h = img.size
    pixels = img.load()

    # Her pass için piksel topla ve filtre 0 (None) ekle
    raw = b""
    for (xs, ys, xstep, ystep) in ADAM7:
        # Pass boyutları
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

    # zlib ile sıkıştır
    z = zlib.compress(raw, 9)

    # PNG yaz
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 1)  # interlace=1

    with open(dst_path, "wb") as f:
        f.write(sig)
        f.write(png_chunk(b"IHDR", ihdr))
        f.write(png_chunk(b"IDAT", z))
        f.write(png_chunk(b"IEND", b""))

    print(f"Yazildi: {dst_path}  ({w}x{h}, Adam7 interlaced)")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "D:/wikipedia.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "D:/interlaced_forced.png"
    make_interlaced(src, dst)