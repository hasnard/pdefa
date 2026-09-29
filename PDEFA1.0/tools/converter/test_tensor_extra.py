# pdefa - AGPL-3.0-or-later - see LICENSE
"""Extra tensor tests — no numpy required."""
import pdefa


def test_tolist_roundtrip():
    src = [[1.0, 2.0], [3.0, 4.0]]
    t = pdefa.from_list(src)
    assert t.tolist() == src


def test_tolist_3d():
    src = [[[1.0, 2.0], [3.0, 4.0]],
           [[5.0, 6.0], [7.0, 8.0]]]
    t = pdefa.from_list(src)
    assert t.shape == (2, 2, 2)
    assert t.tolist() == src


def test_fill_changes_values():
    t = pdefa.zeros(3, 3)
    t.fill(2.5)
    for row in t.tolist():
        for v in row:
            assert abs(v - 2.5) < 1e-6


def test_from_list_deep_copy():
    """Listeyi değiştirmek tensor'ı etkilememeli."""
    src = [[1.0, 2.0]]
    t = pdefa.from_list(src)
    src[0][0] = 999.0
    assert t.tolist() == [[1.0, 2.0]]


def test_tensor_shape_3d():
    t = pdefa.zeros(2, 3, 4)
    assert t.shape == (2, 3, 4)
    assert t.numel == 24
    assert t.ndim == 3


def test_randn_3d():
    t = pdefa.randn(2, 3, 4, seed=42)
    assert t.shape == (2, 3, 4)
    assert t.numel == 24