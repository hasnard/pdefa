"""Test JPEG dosyalari uretir."""
import os
import numpy as np
from PIL import Image

OUT = "tests/test_images"
os.makedirs(OUT, exist_ok=True)

# 1) Kirmizi 8x8
arr = np.zeros((8, 8, 3), dtype=np.uint8)
arr[:, :, 0] = 255
Image.fromarray(arr).save(f"{OUT}/test_red_8x8.jpg", "JPEG", quality=95)
print(f"  {OUT}/test_red_8x8.jpg")

# 2) Gray 16x16 gradient
arr = np.zeros((16, 16), dtype=np.uint8)
for y in range(16):
    for x in range(16):
        arr[y, x] = x * 16
Image.fromarray(arr, mode="L").save(f"{OUT}/test_gray_16x16.jpg", "JPEG", quality=95)
print(f"  {OUT}/test_gray_16x16.jpg")

# 3) 64x64 gradient RGB
arr = np.zeros((64, 64, 3), dtype=np.uint8)
for y in range(64):
    for x in range(64):
        arr[y, x] = (x * 4, y * 4, (x + y) * 2)
Image.fromarray(arr).save(f"{OUT}/test_photo_64x64.jpg", "JPEG", quality=95)
print(f"  {OUT}/test_photo_64x64.jpg")

print("Bitti.")
