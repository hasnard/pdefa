# pdefa - AGPL-3.0-or-later - see LICENSE
"""Error path tests."""
import pytest
import pdefa


def test_softmax_invalid_axis():
    t = pdefa.from_list([[1.0, 2.0]])
    with pytest.raises(pdefa.PdefaError):
        pdefa.ops_softmax(t, axis=99)


def test_reshape_wrong_numel():
    t = pdefa.from_list([[1.0, 2.0, 3.0]])
    with pytest.raises(pdefa.PdefaError):
        pdefa.ops_reshape(t, [2, 2])  # 3 → 4 olmaz


def test_reduce_invalid_axis():
    t = pdefa.from_list([[1.0, 2.0]])
    with pytest.raises(pdefa.PdefaError):
        pdefa.ops_reduce_sum(t, axis=99)


def test_matmul_wrong_rank():
    a = pdefa.zeros(2, 3, 4)
    b = pdefa.zeros(4, 5)
    with pytest.raises(pdefa.PdefaError):
        _ = a @ b


def test_copy_from_list_shape_mismatch():
    t = pdefa.zeros(2, 2)
    with pytest.raises(pdefa.PdefaError):
        t.copy_from_list([[1.0, 2.0, 3.0]])


def test_context_set_zero_threads():
    ctx = pdefa.Context()
    with pytest.raises(pdefa.PdefaError):
        ctx.set_num_threads(0)


def test_context_set_negative_threads():
    ctx = pdefa.Context()
    with pytest.raises(pdefa.PdefaError):
        ctx.set_num_threads(-1)