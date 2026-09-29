# pdefa - AGPL-3.0-or-later - see LICENSE
import pdefa


def test_zeros_shape():
    t = pdefa.zeros(3, 4)
    assert t.shape == (3, 4)
    assert t.numel == 12
    assert t.ndim == 2


def test_ones_values():
    assert pdefa.ones(2, 3).tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_from_list_roundtrip():
    src = [[1.0, 2.0], [3.0, 4.0]]
    assert pdefa.from_list(src).tolist() == src


def test_operator_add():
    c = pdefa.ones(2, 2) + pdefa.ones(2, 2)
    assert c.tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_operator_sub():
    c = pdefa.ones(2, 2) - pdefa.ones(2, 2)
    assert c.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_operator_scalar_mul():
    c = pdefa.ones(2, 2) * 3.0
    assert c.tolist() == [[3.0, 3.0], [3.0, 3.0]]


def test_matmul_identity():
    a = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])
    i = pdefa.from_list([[1.0, 0.0], [0.0, 1.0]])
    assert (a @ i).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_fill_inplace():
    t = pdefa.zeros(2, 2)
    t.fill(0.5)
    assert t.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_copy_from_list():
    t = pdefa.zeros(2, 3)
    t.copy_from_list([[1, 2, 3], [4, 5, 6]])
    assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_deterministic_randn():
    a = pdefa.randn(3, 3, seed=42)
    b = pdefa.randn(3, 3, seed=42)
    assert a.tolist() == b.tolist()


def test_different_seeds_differ():
    a = pdefa.randn(3, 3, seed=1)
    b = pdefa.randn(3, 3, seed=2)
    assert a.tolist() != b.tolist()