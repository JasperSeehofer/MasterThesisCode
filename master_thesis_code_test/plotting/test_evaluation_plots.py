"""Smoke tests for evaluation_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import NO_DATA
from master_thesis_code.plotting.evaluation_plots import (
    plot_detection_contour,
    plot_generation_time_histogram,
    plot_injected_vs_recovered,
    plot_mean_cramer_rao_bounds,
    plot_sky_localization_3d,
    plot_uncertainty_violins,
)


def _first_mappable(ax: Axes) -> ScalarMappable:
    """Return the first ScalarMappable (image / quadmesh / collection) on *ax*.

    Collections (QuadMesh from hist2d) are ScalarMappable subclasses, so a
    single isinstance check covers them.
    """
    if ax.images:
        return ax.images[0]
    for coll in ax.collections:
        if isinstance(coll, ScalarMappable) and coll.get_array() is not None:
            return coll
    raise AssertionError("no mappable found on axes")


def _asserts_set_bad_is_no_data(mappable: ScalarMappable) -> None:
    bad_rgba = mappable.get_cmap().get_bad()
    assert tuple(bad_rgba) == to_rgba(NO_DATA), (
        f"set_bad color {tuple(bad_rgba)} != NO_DATA {to_rgba(NO_DATA)}"
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


def test_plot_mean_cramer_rao_bounds_has_explicit_norm_and_set_bad(
    sample_covariance_matrix: npt.NDArray[np.float64],
    sample_parameter_names: list[str],
) -> None:
    """CRB covariance imshow: robust norm (not autoscale) + set_bad(NO_DATA)."""
    from matplotlib.colors import Normalize

    _, ax = plot_mean_cramer_rao_bounds(sample_covariance_matrix, sample_parameter_names)
    mappable = _first_mappable(ax)
    assert isinstance(mappable.norm, Normalize)
    assert mappable.norm.vmin is not None and mappable.norm.vmax is not None
    _asserts_set_bad_is_no_data(mappable)


def test_plot_detection_contour_uses_lognorm_and_set_bad(
    sample_redshifts: npt.NDArray[np.float64],
    sample_masses: npt.NDArray[np.float64],
) -> None:
    """detection_contour hist2d: LogNorm/robust + set_bad; empty bins do not crash."""
    from matplotlib.colors import Normalize

    _, ax = plot_detection_contour(sample_redshifts, sample_masses)
    mappable = _first_mappable(ax)
    assert isinstance(mappable.norm, Normalize)
    _asserts_set_bad_is_no_data(mappable)


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
