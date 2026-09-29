# pdefa - AGPL-3.0-or-later - see LICENSE
from pathlib import Path
import pytest
import pdefa

DATA = Path(__file__).parent / "data"


def _img():
    p = DATA / "cat.jpg"
    if not p.exists():
        pytest.skip(f"missing: {p}")
    return str(p)


def test_letterbox_shape():
    img = pdefa.load_image(_img())
    tensor, _ = pdefa.preprocess_letterbox(img, 640, 640)
    assert tensor.shape == (1, 3, 640, 640)


def test_letterbox_meta():
    img = pdefa.load_image(_img())
    _, meta = pdefa.preprocess_letterbox(img, 640, 640)
    assert meta.orig_w == img.width
    assert meta.orig_h == img.height
    assert meta.ratio > 0.0           
    assert meta.pad_x >= 0 and meta.pad_y >= 0


def test_letterbox_normalized():
    img = pdefa.load_image(_img())
    t, _ = pdefa.preprocess_letterbox(img, 64, 64, scale=1.0 / 255.0)
    arr = t.tolist()
    flat = [x for ch in arr[0] for row in ch for x in row]
    assert min(flat) >= 0.0
    assert max(flat) <= 1.0 + 1e-6


def test_prepare_for_engine():
    t, m = pdefa.prepare_for_engine(_img(), 640, 640)
    assert t.shape == (1, 3, 640, 640)
    assert m.orig_w > 0 and m.orig_h > 0