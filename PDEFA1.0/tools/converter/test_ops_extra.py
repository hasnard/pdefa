# pdefa - AGPL-3.0-or-later - see LICENSE
"""Extra ops tests: gelu, silu, reduce_max, gather, softmax variants."""
import math
import pdefa


def test_gelu_zero():
    t = pdefa.from_list([[0.0]])
    pdefa.ops_gelu(t)
    assert abs(t.tolist()[0][0]) < 1e-6


def test_gelu_positive():
    t = pdefa.from_list([[1.0]])
    pdefa.ops_gelu(t)
    # gelu(1) ≈ 0.8413
    assert abs(t.tolist()[0][0] - 0.8413) < 1e-3


def test_silu_zero():
    t = pdefa.from_list([[0.0]])
    pdefa.ops_silu(t)
    assert abs(t.tolist()[0][0]) < 1e-6


def test_silu_one():
    t = pdefa.from_list([[1.0]])
    pdefa.ops_silu(t)
    # silu(1) = 1/(1+e^-1) ≈ 0.7311
    assert abs(t.tolist()[0][0] - 0.7311) < 1e-3


def test_reduce_max_axis1():
    t = pdefa.from_list([[1.0, 5.0, 2.0], [3.0, 4.0, 6.0]])
    r = pdefa.ops_reduce_max(t, axis=1, keepdims=False)
    assert r.tolist() == [5.0, 6.0]


def test_reduce_max_axis0():
    t = pdefa.from_list([[1.0, 5.0], [3.0, 4.0]])
    r = pdefa.ops_reduce_max(t, axis=0, keepdims=False)
    assert r.tolist() == [3.0, 5.0]


def test_reduce_sum_keepdims():
    t = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])
    r = pdefa.ops_reduce_sum(t, axis=1, keepdims=True)
    assert r.shape == (2, 1)


def test_reduce_mean_axis0():
    t = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])
    r = pdefa.ops_reduce_mean(t, axis=0, keepdims=False)
    assert r.tolist() == [2.0, 3.0]


def test_gather_axis0():
    a = pdefa.from_list([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = pdefa.from_list([2, 0])
    r = pdefa.ops_gather(a, idx, axis=0)
    # axis=0'da 2 satır seçtik, her biri 2 kolon
    arr = r.tolist()
    assert len(arr) == 2
    assert arr[0] == [5.0, 6.0]
    assert arr[1] == [1.0, 2.0]


def test_softmax_axis0():
    t = pdefa.from_list([[1.0, 1.0], [1.0, 1.0]])
    pdefa.ops_softmax(t, axis=0)
    arr = t.tolist()
    # Her sütun toplamı 1 olmalı
    assert abs(arr[0][0] + arr[1][0] - 1.0) < 1e-5
    assert abs(arr[0][1] + arr[1][1] - 1.0) < 1e-5


def test_softmax_axis0_asymmetric():
    t = pdefa.from_list([[2.0], [1.0]])
    pdefa.ops_softmax(t, axis=0)
    arr = t.tolist()
    assert arr[0][0] > arr[1][0]  # 2 > 1


def test_ops_relu_on_2d():
    t = pdefa.from_list([[-1.0, 2.0], [-3.0, 4.0]])
    pdefa.ops_relu(t)
    assert t.tolist() == [[0.0, 2.0], [0.0, 4.0]]


def test_clip_negative_positive():
    t = pdefa.from_list([[-10.0, 0.0, 10.0]])
    pdefa.ops_clip(t, -2.0, 2.0)
    assert t.tolist() == [[-2.0, 0.0, 2.0]]