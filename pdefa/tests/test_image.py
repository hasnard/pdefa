# pdefa - AGPL-3.0-or-later - see LICENSE
from pathlib import Path
import pytest
import pdefa

DATA = Path(__file__).parent / "data"


def _img(name):
    p = DATA / name
    if not p.exists():
        pytest.skip(f"missing test image: {p}")
    return str(p)


def test_load_jpeg():
    img = pdefa.load_image(_img("cat.jpg"))
    assert img.width > 0 and img.height > 0
    assert img.channels in (1, 3, 4)
    assert img.format in (0, 1, 2)


def test_load_png():
    img = pdefa.load_image(_img("bus.png"))
    assert img.channels == 3


def test_tobytes_size():
    img = pdefa.load_image(_img("cat.jpg"))
    raw = img.tobytes()
    assert len(raw) == img.width * img.height * img.channels


def test_tolist_shape():
    img = pdefa.load_image(_img("cat.jpg"))
    arr = img.tolist()
    assert len(arr) == img.height
    assert len(arr[0]) == img.width
    assert len(arr[0][0]) == img.channels


def test_missing_file_raises():
    with pytest.raises(pdefa.PdefaError):
        pdefa.load_image("does_not_exist_12345.jpg")