from PIL import Image
import sys

if len(sys.argv) < 3:
    print("kullanim: python compare.py <img1> <img2>")
    sys.exit(1)

a = Image.open(sys.argv[1]).convert("RGBA")
b = Image.open(sys.argv[2]).convert("RGBA")

if a.size != b.size:
    print("BOYUT FARKLI:", a.size, "vs", b.size)
    sys.exit(1)

pa = a.load()
pb = b.load()
diff = 0
max_diff = 0

for y in range(a.size[1]):
    for x in range(a.size[0]):
        for c in range(4):
            d = abs(pa[x, y][c] - pb[x, y][c])
            if d:
                diff += 1
                if d > max_diff:
                    max_diff = d

total = a.size[0] * a.size[1] * 4
print("Toplam kanal :", total)
print("Farkli kanal :", diff)
print("Maks fark    :", max_diff)
print("SONUC: AYNI" if diff == 0 else "SONUC: FARKLI")