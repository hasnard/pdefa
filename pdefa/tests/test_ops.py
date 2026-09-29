# pdefa - AGPL-3.0-or-later - see LICENSE
import pdefa


def test_relu():
    t = pdefa.from_list([[-1.0, 0.0, 2.0]])
    pdefa.ops_relu(t)
    assert t.tolist() == [[0.0, 0.0, 2.0]]


def test_sigmoid():
    t = pdefa.from_list([[0.0]])
    pdefa.ops_sigmoid(t)
    assert abs(t.tolist()[0][0] - 0.5) < 1e-6


def test_tanh():
    t = pdefa.from_list([[0.0]])
    pdefa.ops_tanh(t)
    assert abs(t.tolist()[0][0]) < 1e-6


def test_clip():
    t = pdefa.from_list([[-5.0, 0.0, 5.0]])
    pdefa.ops_clip(t, -1.0, 1.0)
    assert t.tolist() == [[-1.0, 0.0, 1.0]]


def test_softmax_sums_to_one():
    t = pdefa.from_list([[1.0, 2.0, 3.0]])
    pdefa.ops_softmax(t, axis=-1)
    assert abs(sum(t.tolist()[0]) - 1.0) < 1e-5


def test_reduce_sum_axis0():
    t = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])
    assert pdefa.ops_reduce_sum(t, axis=0, keepdims=False).tolist() == [4.0, 6.0]


def test_reduce_mean_axis1():
    t = pdefa.from_list([[1.0, 3.0], [2.0, 4.0]])
    assert pdefa.ops_reduce_mean(t, axis=1, keepdims=False).tolist() == [2.0, 3.0]


def test_argmax():
    t = pdefa.from_list([[1.0, 5.0, 2.0]])
    assert pdefa.ops_argmax(t, axis=1, keepdims=False).tolist() == [1]


def test_add_broadcast():
    a = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])
    b = pdefa.from_list([[10.0, 20.0]])
    assert pdefa.ops_add(a, b).tolist() == [[11.0, 22.0], [13.0, 24.0]]


def test_mul_scalar():
    a = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])
    assert pdefa.ops_mul(a, 2.0).tolist() == [[2.0, 4.0], [6.0, 8.0]]


def test_reshape():
    t = pdefa.from_list([[1.0, 2.0, 3.0, 4.0]])
    assert pdefa.ops_reshape(t, [2, 2]).shape == (2, 2)


def test_transpose():
    t = pdefa.from_list([[1.0, 2.0], [3.0, 4.0]])
    assert pdefa.ops_transpose(t, [1, 0]).tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_concat():
    a = pdefa.from_list([[1.0, 2.0]])
    b = pdefa.from_list([[3.0, 4.0]])
    assert pdefa.ops_concat([a, b], axis=0).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_stack():
    a = pdefa.from_list([[1.0, 2.0]])
    b = pdefa.from_list([[3.0, 4.0]])
    assert pdefa.ops_stack([a, b], axis=0).shape == (2, 1, 2)