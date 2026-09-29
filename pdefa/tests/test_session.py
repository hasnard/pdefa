# pdefa - AGPL-3.0-or-later - see LICENSE
from pathlib import Path
import pytest
import pdefa

DATA = Path(__file__).parent / "data"


@pytest.fixture
def tiny_engine():
    p = DATA / "tiny.engine"
    if not p.exists():
        pytest.skip(f"missing: {p} (build with tools/converter)")
    return str(p)


def test_session_stats(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    s = pdefa.Session(ctx, model)
    s.set_input(0, pdefa.zeros(1, 3, 64, 64))
    s.run()
    stats = s.stats()
    assert stats["runs"] >= 1
    assert stats["last_run_ms"] > 0


def test_input_info(tiny_engine):
    ctx = pdefa.Context()
    model = pdefa.Model(ctx, tiny_engine)
    info = model.input_info(0)
    assert len(info.shape) >= 1
    assert info.shape[0] == 1