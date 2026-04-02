"""Factory functions for galaxy catalog plots.

Extracted from ``GalaxyCatalogueHandler.visualize_galaxy_catalog()`` in
``handler.py`` and ``GalaxyCatalog`` plotting methods.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import EDGE
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
from master_thesis_code.plotting._labels import LABELS


def plot_bh_mass_distribution(
    masses: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of estimated black hole masses."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    log_bins = np.geomspace(masses.min(), masses.max(), bins).tolist()
    ax.hist(masses, bins=log_bins, edgecolor=EDGE, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["M"])
    ax.set_ylabel("Count")
    return fig, ax


def plot_redshift_distribution(
    redshifts: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of galaxy redshifts."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(redshifts, bins=bins, edgecolor=EDGE, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Count")
    return fig, ax


def plot_glade_completeness(
    distance_range: npt.NDArray[np.float64],
    completeness: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """GLADE catalog completeness as a function of distance."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(distance_range, completeness)
    ax.set_xlabel("Distance [Mpc]")
    ax.set_ylabel("Completeness [%]")
    return fig, ax


def plot_comoving_volume_sampling(
    samples: npt.NDArray[np.float64],
    *,
    bins: int = 20,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of comoving volume MCMC samples."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(samples, bins=bins, density=True, edgecolor=EDGE, alpha=0.7)
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Density")
    return fig, ax
