"""Smoke tests for model_plots factory functions."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import NO_DATA
from master_thesis_code.plotting.model_plots import (
    plot_detection_probability_grid,
    plot_detection_probability_zM,
    plot_emri_distribution,
    plot_emri_rate,
    plot_emri_sampling,
)


def _first_mappable(ax: Axes) -> ScalarMappable:
    """Return the first ScalarMappable (image / quadmesh / collection) on *ax*.

    Collections (QuadMesh from hist2d, QuadContourSet from contourf) are all
    ScalarMappable subclasses, so a single isinstance check covers them.
    """
    if ax.images:
        return ax.images[0]
    for coll in ax.collections:
        if isinstance(coll, ScalarMappable) and coll.get_array() is not None:
            return coll
    raise AssertionError("no mappable found on axes")


def _asserts_set_bad_is_no_data(mappable: ScalarMappable) -> None:
    cmap = mappable.get_cmap()
    bad_rgba = cmap.get_bad()
    assert tuple(bad_rgba) == to_rgba(NO_DATA), (
        f"set_bad color {tuple(bad_rgba)} != NO_DATA {to_rgba(NO_DATA)}"
    )


def test_plot_emri_distribution() -> None:
    z = np.linspace(0.1, 2.0, 15)
    m = np.geomspace(1e4, 1e7, 10)
    Z, M_grid = np.meshgrid(z, m)
    dist = np.random.default_rng(42).random((10, 15))
    fig, ax = plot_emri_distribution(Z, M_grid, dist)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_emri_rate(sample_masses: np.ndarray) -> None:
    rates = np.random.default_rng(42).random(len(sample_masses))
    fig, ax = plot_emri_rate(sample_masses, rates)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_emri_distribution_has_explicit_norm_and_set_bad() -> None:
    """contourf density map: robust norm + set_bad(NO_DATA)."""
    z = np.linspace(0.1, 2.0, 15)
    m = np.geomspace(1e4, 1e7, 10)
    Z, M_grid = np.meshgrid(z, m)
    dist = np.random.default_rng(7).random((10, 15))
    _, ax = plot_emri_distribution(Z, M_grid, dist)
    mappable = _first_mappable(ax)
    assert mappable.norm is not None
    _asserts_set_bad_is_no_data(mappable)


def test_plot_emri_sampling() -> None:
    rng = np.random.default_rng(42)
    z_events = rng.uniform(0.1, 2.0, 50)
    m_events = 10 ** rng.uniform(4, 7, 50)
    z_bins = np.linspace(0.1, 2.0, 10)
    m_bins = np.geomspace(1e4, 1e7, 10)
    fig, ax = plot_emri_sampling(z_events, m_events, z_bins, m_bins)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_emri_sampling_uses_lognorm_and_set_bad() -> None:
    """hist2d counts: LogNorm (not linear-from-zero autoscale) + set_bad; empty bins -> NO_DATA."""
    from matplotlib.colors import LogNorm

    rng = np.random.default_rng(42)
    z_events = rng.uniform(0.1, 2.0, 50)
    m_events = 10 ** rng.uniform(4, 7, 50)
    z_bins = np.linspace(0.1, 2.0, 10)
    m_bins = np.geomspace(1e4, 1e7, 10)
    _, ax = plot_emri_sampling(z_events, m_events, z_bins, m_bins)
    mappable = _first_mappable(ax)
    assert isinstance(mappable.norm, LogNorm), "emri_sampling counts should use LogNorm"
    _asserts_set_bad_is_no_data(mappable)


def test_plot_detection_probability_grid_has_explicit_norm_and_set_bad() -> None:
    """pdet contourf is bounded 0..1 -> explicit Normalize(0,1) + set_bad(NO_DATA)."""
    from matplotlib.colors import Normalize

    d_L = np.linspace(0.1, 10.0, 12)
    M = np.geomspace(1e4, 1e7, 8)
    D, MG = np.meshgrid(d_L, M)
    prob = np.random.default_rng(42).random((8, 12))
    prob[0, 0] = np.nan  # empty region must render as NO_DATA, not crash
    _, ax = plot_detection_probability_grid(D, MG, prob)
    mappable = _first_mappable(ax)
    assert isinstance(mappable.norm, Normalize)
    assert mappable.norm.vmin == 0.0 and mappable.norm.vmax == 1.0
    _asserts_set_bad_is_no_data(mappable)


def test_plot_detection_probability_grid() -> None:
    d_L = np.linspace(0.1, 10.0, 12)
    M = np.geomspace(1e4, 1e7, 8)
    D, MG = np.meshgrid(d_L, M)
    prob = np.random.default_rng(42).random((8, 12))
    fig, ax = plot_detection_probability_grid(D, MG, prob)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_detection_probability_grid_with_contours() -> None:
    d_L = np.linspace(0.1, 10.0, 12)
    M = np.geomspace(1e4, 1e7, 8)
    D, MG = np.meshgrid(d_L, M)
    prob = np.random.default_rng(42).random((8, 12))
    fig, ax = plot_detection_probability_grid(D, MG, prob, contour_levels=[0.5, 0.9])
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_detection_probability_grid_with_scatter() -> None:
    rng = np.random.default_rng(42)
    d_L = np.linspace(0.1, 10.0, 12)
    M = np.geomspace(1e4, 1e7, 8)
    D, MG = np.meshgrid(d_L, M)
    prob = rng.random((8, 12))
    inj_dl = rng.uniform(0.1, 10.0, 30)
    inj_M = 10 ** rng.uniform(4, 7, 30)
    det_mask = rng.random(30) > 0.3
    fig, ax = plot_detection_probability_grid(
        D, MG, prob, injected_coords=(inj_dl, inj_M), detected_mask=det_mask
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_detection_probability_zM() -> None:
    z = np.linspace(0.1, 2.0, 12)
    M = np.geomspace(1e4, 1e7, 8)
    Z, MG = np.meshgrid(z, M)
    prob = np.random.default_rng(42).random((8, 12))
    fig, ax = plot_detection_probability_zM(Z, MG, prob)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
