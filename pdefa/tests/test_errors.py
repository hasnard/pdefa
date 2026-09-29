# pdefa - AGPL-3.0-or-later - see LICENSE
import pytest
import pdefa


def test_bad_model_path():
    ctx = pdefa.Context()
    with pytest.raises(pdefa.PdefaError):
        pdefa.Model(ctx, "nonexistent_model_xyz.engine")


def test_shape_mismatch_matmul():
    a = pdefa.zeros(2, 3)
    b = pdefa.zeros(4, 5)
    with pytest.raises(pdefa.PdefaError):
        _ = a @ b


def test_invalid_image_path():
    with pytest.raises(pdefa.PdefaError):
        pdefa.load_image("does_not_exist.jpg")