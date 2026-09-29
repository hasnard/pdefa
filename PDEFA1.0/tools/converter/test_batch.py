# pdefa - AGPL-3.0-or-later - see LICENSE
"""BatchInference tests."""
import pytest
from pathlib import Path
import pdefa

DATA = Path(__file__).parent / "data"


@pytest.fixture
def tiny_engine():
    p = DATA / "tiny.engine"
    if not p.exists():
        pytest.skip(f"missing: {p}")
    return str(p)


@pytest.fixture
def image_paths():
    paths = []
    for name in ("cat.jpg", "bus.png"):
        p = DATA / name
        if not p.exists():
            pytest.skip(f"missing: {p}")
        paths.append(str(p))
    return paths


def test_batch_inference(tiny_engine, image_paths):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    with pdefa.BatchInference(ctx, model, batch_size=2, num_threads=2) as batch:
        results = batch.run(image_paths)
        assert len(results) == 2
        for out_tensor, meta in results:
            assert out_tensor.shape[0] == 1


def test_batch_inference_single_image(tiny_engine, image_paths):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    with pdefa.BatchInference(ctx, model, batch_size=1) as batch:
        results = batch.run([image_paths[0]])
        assert len(results) == 1


def test_batch_context_manager(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    batch = pdefa.BatchInference(ctx, model, batch_size=2)
    with batch:
        pass  # __enter__/__exit__ çalışmalı