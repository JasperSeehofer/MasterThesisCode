import types
from pathlib import Path

import numpy as np
import pytest

from master_thesis_code.plotting import apply_style

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--save-test-plots",
        action="store_true",
        default=False,
        help="Save integration test plots to test-artifacts/ instead of a tmp dir",
    )


@pytest.fixture(scope="session")
def plot_output_dir(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Directory where integration tests write plot PNGs.

    With ``--save-test-plots`` the plots land in ``<repo>/test-artifacts/evaluation/``
    so CI can publish them. Otherwise a throwaway tmpdir is used.
    """
    if request.config.getoption("--save-test-plots"):
        out = _REPO_ROOT / "test-artifacts" / "evaluation"
        out.mkdir(parents=True, exist_ok=True)
        return out
    return tmp_path_factory.mktemp("plots")


@pytest.fixture(autouse=True, scope="session")
def _plotting_style() -> None:
    """Apply the project matplotlib style once per test session.

    This ensures every test that produces a plot uses the
    ``emri_thesis.mplstyle`` settings (font sizes, DPI, constrained
    layout, etc.) rather than matplotlib defaults.
    """
    apply_style()


@pytest.fixture(params=["numpy"] + (["cupy"] if _CUPY_AVAILABLE else []))
def xp(request: pytest.FixtureRequest) -> types.ModuleType:
    """Array module fixture.

    Parameterized so tests run on numpy (always) and cupy (when a GPU is
    available).  Pass this fixture to any test that operates on arrays so the
    same test exercises both backends without duplication.

    Usage::

        def test_something(xp: types.ModuleType) -> None:
            arr = xp.zeros(10)
            ...
    """
    if request.param == "cupy":
        return cp  # type: ignore[no-any-return]
    return np
