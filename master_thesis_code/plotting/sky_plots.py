"""Mollweide sky localization map factory function.

Plots EMRI source positions on a Mollweide projection with SNR-colored
scatter markers and optional localization error ellipses derived from
the Fisher matrix sky sub-covariance.

All functions follow the project convention: data in, ``(fig, ax)`` out.
None call ``plt.show()`` or ``plt.savefig()``.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from master_thesis_code.plotting._colors import CMAP, EDGE
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure, make_colorbar
from master_thesis_code.plotting._labels import LABELS
from master_thesis_code.plotting.fisher_plots import _ellipse_params


def plot_sky_localization_mollweide(
    theta_s: npt.NDArray[np.float64],
    phi_s: npt.NDArray[np.float64],
    snr: npt.NDArray[np.float64],
    *,
    covariances: list[npt.NDArray[np.float64]] | None = None,
    n_sigma: float = 1.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot EMRI sky positions on a Mollweide projection.

    Parameters
    ----------
    theta_s : npt.NDArray[np.float64]
        Source colatitude in radians, range ``[0, pi]``.
    phi_s : npt.NDArray[np.float64]
        Source longitude in radians, range ``[0, 2*pi]``.
    snr : npt.NDArray[np.float64]
        Signal-to-noise ratio for each source (used for color mapping).
    covariances : list[npt.NDArray[np.float64]] | None
        Optional list of 2x2 sky sub-covariance matrices for error
        ellipses.  Length must match ``theta_s``.
    n_sigma : float
        Number of standard deviations for ellipse boundary.
    ax : Axes | None
        Existing Mollweide axes to draw on.  Created if ``None``.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes with the sky map.
    """
    if ax is None:
        fig, ax = get_figure(preset="double", subplot_kw={"projection": "mollweide"})
    else:
        fig = _fig_from_ax(ax)

    # Coordinate transform: colatitude -> latitude, longitude -> [-pi, pi]
    lat = np.pi / 2.0 - theta_s
    lon = ((phi_s - np.pi + np.pi) % (2.0 * np.pi)) - np.pi

    sc = ax.scatter(lon, lat, c=snr, cmap=CMAP, s=12, alpha=0.8, zorder=5)
    make_colorbar(sc, fig, ax, label=LABELS["SNR"])

    if covariances is not None:
        for i, cov_2x2 in enumerate(covariances):
            w, h, angle = _ellipse_params(cov_2x2, n_sigma)
            ellipse = Ellipse(
                xy=(float(lon[i]), float(lat[i])),
                width=w,
                height=h,
                angle=angle,
                facecolor="none",
                edgecolor=EDGE,
                linewidth=0.8,
                transform=ax.transData,
            )
            ax.add_patch(ellipse)

    ax.grid(True, alpha=0.3)
    return fig, ax
