"""Smoke tests for sky_plots factory functions."""

import numpy as np
import numpy.typing as npt
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
    fig, ax = plot_sky_localization_mollweide(
        theta_s, phi_s, snr, covariances=covariances
    )
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
