"""Smoke tests for fisher_plots factory functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._data import PARAMETER_NAMES
from master_thesis_code.plotting.fisher_plots import (
    plot_characteristic_strain,
    plot_fisher_corner,
    plot_fisher_ellipses,
    plot_parameter_uncertainties,
)

# 14-element array aligned to PARAMETER_NAMES order:
# M, mu, a, p0, e0, x0, luminosity_distance, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
_SAMPLE_PARAM_VALUES = np.array(
    [1e6, 10.0, 0.6, 10.0, 0.2, 1.0, 1000.0, 1.0, 2.0, 0.5, 3.0, 0.1, 0.2, 0.3]
)


# ---------------------------------------------------------------------------
# Error Ellipses (FISH-02)
# ---------------------------------------------------------------------------


def test_plot_fisher_ellipses_single(
    sample_covariance_matrix: np.ndarray,
) -> None:
    """Smoke test: single-event error ellipses return (Figure, ndarray[Axes])."""
    fig, axes = plot_fisher_ellipses(sample_covariance_matrix, _SAMPLE_PARAM_VALUES)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_plot_fisher_ellipses_multi(
    sample_covariance_matrix: np.ndarray,
) -> None:
    """Smoke test: multi-event overlay returns (Figure, ndarray[Axes])."""
    fig, axes = plot_fisher_ellipses(
        sample_covariance_matrix,
        _SAMPLE_PARAM_VALUES,
        events=[
            (sample_covariance_matrix, _SAMPLE_PARAM_VALUES),
            (sample_covariance_matrix, _SAMPLE_PARAM_VALUES * 1.1),
        ],
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_fisher_ellipses_custom_pairs(
    sample_covariance_matrix: np.ndarray,
) -> None:
    """Smoke test: custom parameter pairs return (Figure, ndarray[Axes])."""
    fig, axes = plot_fisher_ellipses(
        sample_covariance_matrix, _SAMPLE_PARAM_VALUES, pairs=[("M", "a")]
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Characteristic Strain (FISH-04)
# ---------------------------------------------------------------------------


def test_plot_characteristic_strain() -> None:
    """Smoke test: characteristic strain returns (Figure, Axes) with >= 3 lines."""
    fig, ax = plot_characteristic_strain()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.get_lines()) >= 3
    plt.close(fig)


def test_plot_characteristic_strain_custom_range() -> None:
    """Smoke test: custom frequency range returns (Figure, Axes)."""
    fig, ax = plot_characteristic_strain(f_min=1e-4, f_max=0.1)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parameter Uncertainties (FISH-05)
# ---------------------------------------------------------------------------


def test_plot_uncertainties_violin(sample_crb_dataframe: pd.DataFrame) -> None:
    """Smoke test: DataFrame input produces violin/bar plot."""
    param_values_df = sample_crb_dataframe[PARAMETER_NAMES]
    fig, ax = plot_parameter_uncertainties(sample_crb_dataframe, param_values_df)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_uncertainties_bar(sample_crb_row: pd.Series) -> None:
    """Smoke test: Series input produces bar chart."""
    param_values_series = sample_crb_row[PARAMETER_NAMES]
    fig, ax = plot_parameter_uncertainties(sample_crb_row, param_values_series)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Corner Plot (FISH-03)
# ---------------------------------------------------------------------------


def test_plot_fisher_corner_basic(
    sample_covariance_matrix: np.ndarray,
) -> None:
    """Smoke test: corner plot with default 6 params returns (Figure, ndarray)."""
    fig, axes = plot_fisher_corner(sample_covariance_matrix, _SAMPLE_PARAM_VALUES)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (6, 6)
    plt.close(fig)


def test_plot_fisher_corner_custom_params(
    sample_covariance_matrix: np.ndarray,
) -> None:
    """Smoke test: corner plot with custom 3 params returns correct shape."""
    fig, axes = plot_fisher_corner(
        sample_covariance_matrix, _SAMPLE_PARAM_VALUES, params=["M", "mu", "a"]
    )
    assert isinstance(fig, Figure)
    assert axes.shape == (3, 3)
    plt.close(fig)


def test_plot_fisher_corner_multi_event(
    sample_covariance_matrix: np.ndarray,
) -> None:
    """Smoke test: multi-event corner overlay has more artists than single."""
    fig_single, _ = plot_fisher_corner(
        sample_covariance_matrix, _SAMPLE_PARAM_VALUES, params=["M", "mu"]
    )
    n_artists_single = len(fig_single.get_children())
    plt.close(fig_single)

    fig_multi, _ = plot_fisher_corner(
        sample_covariance_matrix,
        _SAMPLE_PARAM_VALUES,
        params=["M", "mu"],
        overlay_events=[
            (sample_covariance_matrix, _SAMPLE_PARAM_VALUES * 1.1),
        ],
    )
    n_artists_multi = len(fig_multi.get_children())
    assert n_artists_multi >= n_artists_single
    plt.close(fig_multi)
