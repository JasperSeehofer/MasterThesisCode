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
    plot_pdet_surface,
    plot_sky_localization_3d,
    plot_uncertainty_violins,
)


def _write_injection_csv(path: str, *, n: int = 400, seed: int = 0) -> None:
    """Write a synthetic injection-campaign CSV with the production columns."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    z = rng.uniform(0.1, 1.5, n)
    luminosity_distance = rng.uniform(0.4, 11.0, n)  # Gpc
    M = 10 ** rng.uniform(4.5, 7.0, n)
    snr = rng.uniform(5.0, 60.0, n)
    df = pd.DataFrame(
        {
            "z": z,
            "M": M,
            "phiS": rng.uniform(0, 2 * np.pi, n),
            "qS": rng.uniform(0, np.pi, n),
            "SNR": snr,
            "h_inj": np.full(n, 0.73),
            "luminosity_distance": luminosity_distance,
        }
    )
    df.to_csv(path, index=False)


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


def test_plot_pdet_surface_true_log_axis_norm_setbad_contour(tmp_path: object) -> None:
    """fig20: pcolormesh true log-y, explicit Normalize(0,1)+set_bad, P_det=0.5 contour."""
    from pathlib import Path

    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Normalize

    d = Path(str(tmp_path))
    _write_injection_csv(str(d / "injection_h_0p73_task_0.csv"), n=600, seed=1)
    _write_injection_csv(str(d / "injection_h_0p73_task_1.csv"), n=600, seed=2)

    fig, ax = plot_pdet_surface(str(d / "injection_h_0p73_task_*.csv"), snr_threshold=20.0)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # True log mass axis (not 0..n_m_bins fake indices)
    assert ax.get_yscale() == "log"
    # The mesh y-range spans real masses (>= 1e4), not small fake indices.
    ylo, yhi = ax.get_ylim()
    assert yhi > 1e4, f"y-axis upper {yhi} suggests fake-index, not true mass edges"
    # Mappable: a QuadMesh from pcolormesh with explicit Normalize(0,1) + set_bad.
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert meshes, "expected a pcolormesh QuadMesh"
    mesh = meshes[0]
    assert isinstance(mesh.norm, Normalize)
    assert mesh.norm.vmin == 0.0 and mesh.norm.vmax == 1.0
    assert tuple(mesh.get_cmap().get_bad()) == to_rgba(NO_DATA)
    # At least one contour collection (the 0.5 / 0.9 horizon).
    contour_colls = [c for c in ax.collections if not isinstance(c, QuadMesh)]
    assert len(contour_colls) >= 1, "expected a P_det horizon contour collection"


def test_plot_pdet_surface_nan_bins_do_not_raise(tmp_path: object) -> None:
    """Empty (d_L, M) cells -> NaN -> set_bad, must not crash the masked draw."""
    from pathlib import Path

    d = Path(str(tmp_path))
    # Few rows -> many empty bins -> NaN cells that must render as no-data.
    _write_injection_csv(str(d / "injection_h_0p73_task_0.csv"), n=25, seed=3)
    fig, ax = plot_pdet_surface(str(d / "injection_h_0p73_task_*.csv"), snr_threshold=20.0)
    assert isinstance(fig, Figure)


def test_plot_pdet_surface_xlabel_is_gpc(tmp_path: object) -> None:
    """fig20 d_L is in Gpc (c/H0 ~ 4.1 Gpc, z~1.5 -> ~11 Gpc), label must say Gpc."""
    from pathlib import Path

    d = Path(str(tmp_path))
    _write_injection_csv(str(d / "injection_h_0p73_task_0.csv"), n=300, seed=4)
    _, ax = plot_pdet_surface(str(d / "injection_h_0p73_task_*.csv"))
    assert "Gpc" in ax.get_xlabel(), f"xlabel {ax.get_xlabel()!r} should carry Gpc units"


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
