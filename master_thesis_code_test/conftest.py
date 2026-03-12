import types

import numpy as np
import pytest

from master_thesis_code.plotting import apply_style

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


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
