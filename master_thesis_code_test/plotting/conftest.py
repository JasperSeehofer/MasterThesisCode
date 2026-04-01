"""Shared pytest fixtures for plotting smoke tests.

Provides fake DataFrames, arrays, and catalog data used by all
``test_*_plots.py`` modules. The session-scoped ``_plotting_style``
fixture in the root ``conftest.py`` handles ``apply_style()`` — do not
duplicate it here.
"""

from collections.abc import Generator

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture()
def sample_h_values() -> npt.NDArray[np.float64]:
    """50-element array of Hubble constant values."""
    return np.linspace(0.5, 1.0, 50)


@pytest.fixture()
def sample_posterior() -> npt.NDArray[np.float64]:
    """50-element fake posterior array."""
    return np.random.default_rng(42).random(50)


@pytest.fixture()
def sample_redshifts() -> npt.NDArray[np.float64]:
    """30-element redshift array."""
    return np.linspace(0.1, 2.0, 30)


@pytest.fixture()
def sample_masses() -> npt.NDArray[np.float64]:
    """30-element mass array, strictly positive, log-spaced."""
    return np.geomspace(1e4, 1e7, 30)


@pytest.fixture()
def sample_distances() -> npt.NDArray[np.float64]:
    """30 distances in Gpc."""
    return np.linspace(0.1, 10.0, 30)


@pytest.fixture()
def sample_times() -> npt.NDArray[np.float64]:
    """100 generation times drawn from an exponential distribution."""
    return np.random.default_rng(42).exponential(5.0, size=100)


@pytest.fixture()
def sample_parameter_names() -> list[str]:
    """14 EMRI parameter names."""
    return [
        "M",
        "mu",
        "a",
        "p0",
        "e0",
        "Y0",
        "d_L",
        "qS",
        "phiS",
        "qK",
        "phiK",
        "Phi_phi0",
        "Phi_theta0",
        "Phi_r0",
    ]


@pytest.fixture()
def sample_covariance_matrix() -> npt.NDArray[np.float64]:
    """14x14 positive semidefinite covariance matrix."""
    rng = np.random.default_rng(42)
    A = rng.random((14, 14))
    return A @ A.T


@pytest.fixture()
def sample_uncertainties() -> dict[str, npt.NDArray[np.float64]]:
    """Dictionary of parameter uncertainties for violin plots."""
    rng = np.random.default_rng(42)
    return {"M": rng.random(20), "mu": rng.random(20), "d_L": rng.random(20)}
