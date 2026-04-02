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
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None]:
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


@pytest.fixture()
def sample_crb_row() -> pd.Series:
    """Single CRB CSV row with 14 param values + 105 delta columns + metadata."""
    from master_thesis_code.plotting._data import PARAMETER_NAMES

    rng = np.random.default_rng(42)
    n = len(PARAMETER_NAMES)
    # Build a positive-definite matrix
    A = rng.random((n, n))
    cov = A @ A.T
    data: dict[str, float] = {}
    # Parameter values
    param_values = [
        1e6, 10.0, 0.6, 10.0, 0.2, 1.0, 1000.0,
        1.0, 2.0, 0.5, 3.0, 0.1, 0.2, 0.3,
    ]
    for name, val in zip(PARAMETER_NAMES, param_values):
        data[name] = val
    # Lower-triangle delta columns (row >= col)
    for i in range(n):
        for j in range(i + 1):
            data[f"delta_{PARAMETER_NAMES[i]}_delta_{PARAMETER_NAMES[j]}"] = cov[i, j]
    # Metadata
    data["T"] = 1.0
    data["dt"] = 10.0
    data["SNR"] = 25.0
    data["generation_time"] = 5.0
    data["host_galaxy_index"] = 0.0
    return pd.Series(data)


@pytest.fixture()
def sample_crb_dataframe(sample_crb_row: pd.Series) -> pd.DataFrame:
    """DataFrame with 12 CRB rows (varied data for violin plots)."""
    rng = np.random.default_rng(99)
    rows: list[pd.Series] = []
    for _ in range(12):
        row = sample_crb_row.copy()
        # Add small multiplicative noise to delta columns so violins have width
        for key in row.index:
            if str(key).startswith("delta_"):
                row[key] *= rng.uniform(0.8, 1.2)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_event_posteriors() -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """H-values grid and 20 per-event posterior arrays for convergence tests."""
    h_values = np.linspace(0.5, 1.0, 50)
    rng = np.random.default_rng(42)
    event_posteriors: list[npt.NDArray[np.float64]] = []
    for _ in range(20):
        center = 0.73 + rng.normal(0, 0.02)
        posterior = np.exp(-0.5 * ((h_values - center) / 0.05) ** 2)
        event_posteriors.append(posterior)
    return h_values, event_posteriors


@pytest.fixture()
def sample_injection_campaign() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Variable array and detected boolean array for detection efficiency tests."""
    rng = np.random.default_rng(42)
    variable = rng.uniform(0.1, 3.0, 200)
    detected = rng.random(200) < 0.7 * np.exp(-variable)
    return variable, detected
