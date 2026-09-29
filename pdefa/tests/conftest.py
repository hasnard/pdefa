# pdefa - AGPL-3.0-or-later - see LICENSE
import pytest
from pathlib import Path
import pdefa

DATA = Path(__file__).parent / "data"


def _require(name: str) -> Path:
    p = DATA / name
    if not p.exists():
        pytest.skip(f"missing test data: {p} (see tests/README.md)")
    return p


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def ctx():
    return pdefa.Context()


@pytest.fixture(scope="session")
def model(ctx):
    return pdefa.Model(ctx, str(_require("tiny.engine")))


@pytest.fixture
def session(ctx, model):
    return pdefa.Session(ctx, model)


@pytest.fixture(scope="session")
def cat_image():
    return pdefa.load_image(str(_require("cat.jpg")))