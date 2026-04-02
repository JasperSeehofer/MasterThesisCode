"""Smoke tests for evaluation_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from master_thesis_code.plotting.evaluation_plots import (
    plot_detection_contour,
    plot_generation_time_histogram,
    plot_injected_vs_recovered,
    plot_mean_cramer_rao_bounds,
    plot_sky_localization_3d,
    plot_uncertainty_violins,
)


def test_plot_mean_cramer_rao_bounds(
    sample_covariance_matrix: npt.NDArray[np.float64],
    sample_parameter_names: list[str],
) -> None:
    """Smoke test: plot_mean_cramer_rao_bounds returns (Figure, Axes)."""
    from matplotlib.axes import Axes

    fig, ax = plot_mean_cramer_rao_bounds(sample_covariance_matrix, sample_parameter_names)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_uncertainty_violins(
    sample_uncertainties: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Smoke test: plot_uncertainty_violins returns (Figure, Axes)."""
    from matplotlib.axes import Axes

    fig, ax = plot_uncertainty_violins(sample_uncertainties)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_sky_localization_3d() -> None:
    """Smoke test: plot_sky_localization_3d returns (Figure, Any)."""
    rng = np.random.default_rng(42)
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 10)
    sky_error = rng.random(10)
    fig, ax = plot_sky_localization_3d(theta, phi, sky_error)
    assert isinstance(fig, Figure)
    # ax is Axes3D, not regular Axes — just check it exists
    assert ax is not None


def test_plot_detection_contour(
    sample_redshifts: npt.NDArray[np.float64],
    sample_masses: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_detection_contour returns (Figure, Axes)."""
    from matplotlib.axes import Axes

    fig, ax = plot_detection_contour(sample_redshifts, sample_masses)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_generation_time_histogram(
    sample_times: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_generation_time_histogram returns (Figure, Axes)."""
    from matplotlib.axes import Axes

    fig, ax = plot_generation_time_histogram(sample_times)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_injected_vs_recovered(
    sample_injected_recovered: tuple[
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
    ],
) -> None:
    """Smoke test: multi-panel scatter with error bars."""
    injected, recovered, uncertainties = sample_injected_recovered
    fig, axes = plot_injected_vs_recovered(injected, recovered, uncertainties=uncertainties)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_plot_injected_vs_recovered_no_errors(
    sample_injected_recovered: tuple[
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
    ],
) -> None:
    """Smoke test: multi-panel scatter without error bars."""
    injected, recovered, _uncertainties = sample_injected_recovered
    fig, axes = plot_injected_vs_recovered(injected, recovered)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_plot_injected_vs_recovered_custom_params(
    sample_injected_recovered: tuple[
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
    ],
) -> None:
    """Smoke test: multi-panel scatter with custom parameter subset."""
    injected, recovered, _uncertainties = sample_injected_recovered
    fig, axes = plot_injected_vs_recovered(injected, recovered, parameters=["M", "a"])
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
