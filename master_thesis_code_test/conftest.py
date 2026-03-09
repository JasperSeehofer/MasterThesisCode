import types

import numpy as np
import pytest

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


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
        return cp  # type: ignore[return-value]
    return np  # type: ignore[return-value]
