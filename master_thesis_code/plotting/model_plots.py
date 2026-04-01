"""Factory functions for cosmological model plots.

Extracted from ``Model1CrossCheck`` and ``DetectionProbability`` in
``cosmological_model.py``.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _fig_from_ax


def plot_emri_distribution(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    distribution: npt.NDArray[np.float64],
    *,
    title: str = "EMRI distribution",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of the EMRI event distribution in (z, M) space."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    cs = ax.contourf(redshifts, masses, distribution, cmap="viridis", levels=30)
    fig.colorbar(cs, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Mass [M_sun]")
    ax.set_title(title)
    return fig, ax


def plot_emri_rate(
    masses: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Log-log plot of EMRI rate vs MBH mass."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.plot(masses, rates)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("MBH mass [M_sun]")
    ax.set_ylabel("EMRI rate R [1/Gyr]")
    return fig, ax


def plot_emri_sampling(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    redshift_bins: npt.NDArray[np.float64],
    mass_bins: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """2D histogram of sampled EMRI events."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    h = ax.hist2d(redshifts, masses, bins=[redshift_bins, mass_bins], cmap="viridis")
    fig.colorbar(h[3], ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Mass [M_sun]")
    ax.set_title("EMRI distribution sampling")
    return fig, ax


def plot_detection_probability_grid(
    d_L_range: npt.NDArray[np.float64],
    M_range: npt.NDArray[np.float64],
    detection_prob: npt.NDArray[np.float64],
    *,
    title: str = "Detection probability",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of detection probability in (d_L, M) space."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    cs = ax.contourf(d_L_range, M_range, detection_prob, cmap="viridis", levels=50)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel("Luminosity distance [Gpc]")
    ax.set_ylabel("Mass [M_sun]")
    ax.set_yscale("log")
    ax.set_title(title)
    return fig, ax
