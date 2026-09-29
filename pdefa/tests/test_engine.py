# pdefa - AGPL-3.0-or-later - see LICENSE
"""Engine-level tests: Session, Context, Model IO, EngineModel."""
import pytest
from pathlib import Path
import pdefa

DATA = Path(__file__).parent / "data"


@pytest.fixture
def tiny_engine():
    p = DATA / "tiny.engine"
    if not p.exists():
        pytest.skip(f"missing: {p} (build with tools/converter)")
    return str(p)


# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------

def test_session_output_shape(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    s = pdefa.Session(ctx, model)
    s.set_input(0, pdefa.zeros(1, 3, 64, 64))
    s.run()
    out = s.output(0)
    assert out.shape == (1, 4)


def test_session_output_deterministic(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    s = pdefa.Session(ctx, model)
    s.set_input(0, pdefa.ones(1, 3, 64, 64))
    s.run()
    out1 = s.output(0).tolist()
    s.run()
    out2 = s.output(0).tolist()
    assert out1 == out2


def test_session_output_changes_with_input(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    s = pdefa.Session(ctx, model)

    s.set_input(0, pdefa.zeros(1, 3, 64, 64))
    s.run()
    out_zero = s.output(0).tolist()

    s.set_input(0, pdefa.ones(1, 3, 64, 64))
    s.run()
    out_ones = s.output(0).tolist()

    assert out_zero != out_ones


def test_session_multiple_runs(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    s = pdefa.Session(ctx, model)
    for _ in range(5):
        s.set_input(0, pdefa.zeros(1, 3, 64, 64))
        s.run()
    assert s.stats()["runs"] >= 5


# ------------------------------------------------------------------
# Model IO info
# ------------------------------------------------------------------

def test_model_input_info(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    info = model.input_info(0)
    assert info.shape == (1, 3, 64, 64)
    assert isinstance(info.name, str)


def test_model_output_info(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    info = model.output_info(0)
    assert info.shape == (1, 4)
    assert isinstance(info.name, str)


# ------------------------------------------------------------------
# Context threads
# ------------------------------------------------------------------

def test_context_set_get_threads():
    ctx = pdefa.Context()
    ctx.set_num_threads(4)
    assert ctx.get_num_threads() == 4
    ctx.set_num_threads(8)
    assert ctx.get_num_threads() == 8


def test_context_threads_affect_engine(tiny_engine):
    """Motor 1 thread ile de çalışmalı."""
    ctx = pdefa.Context()
    ctx.set_num_threads(1)
    model = pdefa.Model(ctx, tiny_engine)
    s = pdefa.Session(ctx, model)
    s.set_input(0, pdefa.zeros(1, 3, 64, 64))
    s.run()
    assert s.output(0).shape == (1, 4)


# ------------------------------------------------------------------
# EngineModel (convenience API)
# ------------------------------------------------------------------

def test_engine_model_load(tiny_engine):
    model = pdefa.load(tiny_engine)
    assert model is not None
    assert hasattr(model, "set_num_threads")
    assert hasattr(model, "session")


def test_engine_model_set_num_threads(tiny_engine):
    model = pdefa.load(tiny_engine)
    model.set_num_threads(2)
    assert model.ctx.get_num_threads() == 2