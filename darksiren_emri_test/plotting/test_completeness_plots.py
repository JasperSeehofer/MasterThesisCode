"""Smoke tests for the per-pixel HEALPix completeness plots (Change 5)."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from darksiren_emri.galaxy_catalogue.pixel_completeness import PixelCompleteness
from darksiren_emri.plotting.completeness_plots import (
    empty_pixel_fraction,
    plot_completeness_sky_map,
    plot_sky_averaged_completeness,
)


def _small_map(nside: int = 4) -> PixelCompleteness:
    """A small PixelCompleteness with a spread of m_th plus one empty (ZoA) pixel."""
    npix = 12 * nside * nside
    rng = np.random.default_rng(0)
    m_th = rng.uniform(16.0, 21.0, size=npix)
    m_th[0] = -np.inf  # empty / ZoA pixel
    return PixelCompleteness(m_th, nside=nside)


def test_plot_m_th_sky_map_returns_fig_ax() -> None:
    fig, ax = plot_completeness_sky_map(_small_map(), quantity="m_th")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_fk_sky_map_returns_fig_ax() -> None:
    fig, ax = plot_completeness_sky_map(_small_map(), quantity="f_k", z=0.05)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_sky_averaged_completeness_returns_fig_ax() -> None:
    fig, ax = plot_sky_averaged_completeness(_small_map())
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_empty_pixel_fraction_counts_zoa() -> None:
    frac = empty_pixel_fraction(_small_map())
    assert 0.0 < frac < 1.0  # exactly one empty pixel out of 12*nside^2
