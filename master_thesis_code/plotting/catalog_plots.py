"""Factory functions for galaxy catalog plots.

Extracted from ``GalaxyCatalogueHandler.visualize_galaxy_catalog()`` in
``handler.py`` and ``GalaxyCatalog`` plotting methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _fig_from_ax


def plot_bh_mass_distribution(
    masses: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of estimated black hole masses."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    log_bins = np.geomspace(masses.min(), masses.max(), bins).tolist()
    ax.hist(masses, bins=log_bins, edgecolor="black", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BH mass [M_sun]")
    ax.set_ylabel("Count")
    ax.set_title("Black hole mass distribution")
    return fig, ax


def plot_redshift_distribution(
    redshifts: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of galaxy redshifts."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.hist(redshifts, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Count")
    ax.set_title("Galaxy redshift distribution")
    return fig, ax


def plot_glade_completeness(
    distance_range: npt.NDArray[np.float64],
    completeness: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """GLADE catalog completeness as a function of distance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.plot(distance_range, completeness)
    ax.set_xlabel("Distance [Mpc]")
    ax.set_ylabel("Completeness [%]")
    ax.set_title("GLADE catalog completeness")
    return fig, ax


def plot_comoving_volume_sampling(
    samples: npt.NDArray[np.float64],
    *,
    bins: int = 20,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of comoving volume MCMC samples."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.hist(samples, bins=bins, density=True, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Density")
    ax.set_title("Comoving volume sampling")
    return fig, ax
