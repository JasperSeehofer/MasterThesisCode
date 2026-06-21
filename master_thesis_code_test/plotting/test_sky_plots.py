"""Smoke tests for sky_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from master_thesis_code.plotting.sky_plots import plot_sky_localization_mollweide

# ---------------------------------------------------------------------------
# Mollweide Sky Map (SKY-01)
# ---------------------------------------------------------------------------


def test_plot_sky_localization_mollweide_basic(
    sample_sky_data: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
    ],
) -> None:
    """Smoke test: basic Mollweide sky map returns (Figure, Axes)."""
    theta_s, phi_s, snr, _ = sample_sky_data
    fig, ax = plot_sky_localization_mollweide(theta_s, phi_s, snr)
    assert isinstance(fig, Figure)


def test_plot_sky_localization_mollweide_with_ellipses(
    sample_sky_data: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
    ],
) -> None:
    """Smoke test: Mollweide sky map with localization ellipses."""
    theta_s, phi_s, snr, covariances = sample_sky_data
    fig, ax = plot_sky_localization_mollweide(theta_s, phi_s, snr, covariances=covariances)
    assert isinstance(fig, Figure)
    ellipse_patches = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipse_patches) == 10


def test_plot_sky_localization_mollweide_colorbar(
    sample_sky_data: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
    ],
) -> None:
    """Smoke test: Mollweide sky map has a colorbar."""
    theta_s, phi_s, snr, _ = sample_sky_data
    fig, ax = plot_sky_localization_mollweide(theta_s, phi_s, snr)
    # Colorbar adds an extra axes to the figure
    assert len(fig.axes) > 1


def test_plot_sky_localization_mollweide_explicit_snr_norm(
    sample_sky_data: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
    ],
) -> None:
    """fig05: SNR scatter gets an explicit norm (Log/robust), not silent autoscale."""
    from matplotlib.colors import LogNorm, Normalize

    theta_s, phi_s, snr, _ = sample_sky_data
    _, ax = plot_sky_localization_mollweide(theta_s, phi_s, snr)
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert scatters, "expected a scatter PathCollection"
    sc = scatters[0]
    # The norm must be an explicit LogNorm (all SNR > 0 in fixtures) — at minimum
    # a non-default Normalize whose bounds were set from the data, not autoscale.
    assert isinstance(sc.norm, Normalize)
    assert sc.norm.vmin is not None and sc.norm.vmax is not None
    if np.all(snr > 0):
        assert isinstance(sc.norm, LogNorm), "all-positive SNR should map via LogNorm"
