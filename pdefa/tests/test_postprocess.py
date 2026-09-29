# pdefa - AGPL-3.0-or-later - see LICENSE
from pathlib import Path
import pytest
import pdefa

DATA = Path(__file__).parent / "data"


@pytest.fixture
def cat_and_meta():
    p = DATA / "cat.jpg"
    if not p.exists():
        pytest.skip(f"missing: {p}")
    img = pdefa.load_image(str(p))
    _, meta = pdefa.preprocess_letterbox(img, 640, 640)
    return img, meta


def test_map_box_roundtrip(cat_and_meta):
    img, meta = cat_and_meta
    box_in = pdefa.Box(meta.pad_x, meta.pad_y,
                       meta.pad_x + 100, meta.pad_y + 100)
    box_out = pdefa.map_box_to_original(box_in, meta)
    assert 0 <= box_out.x1 < img.width
    assert 0 <= box_out.y1 < img.height
    assert box_out.x2 > box_out.x1
    assert box_out.y2 > box_out.y1


def test_map_point(cat_and_meta):
    img, meta = cat_and_meta
    x, y = pdefa.map_point_to_original(320.0, 320.0, meta)
    assert 0 <= x <= img.width
    assert 0 <= y <= img.height


def test_decode_yolo_empty(cat_and_meta):
    _, meta = cat_and_meta
    t = pdefa.zeros(1, 84, 8400)
    boxes = pdefa.decode_yolo(t, meta, conf_thres=0.9)
    assert isinstance(boxes, list)
    assert len(boxes) == 0