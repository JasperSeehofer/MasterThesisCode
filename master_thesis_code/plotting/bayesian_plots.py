"""Factory functions for Bayesian H₀ inference plots.

Extracted from ``BayesianStatistics.visualize()`` and
``BayesianStatistics.visualize_galaxy_weights()`` in ``cosmological_model.py``.

All functions take data in and return ``(fig, ax)`` out.  None call
``plt.show()`` or ``plt.savefig()`` — the caller decides where to save.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _fig_from_ax


def plot_combined_posterior(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    true_h: float,
    *,
    label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single combined Hubble constant posterior."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    normalized = posterior / np.max(posterior) if np.max(posterior) > 0 else posterior
    ax.plot(h_values, normalized, label=label)
    ax.axvline(true_h, color="green", linestyle="dashed", label=f"True H = {true_h}")
    ax.set_xlabel("Hubble constant h")
    ax.set_ylabel("Posterior (normalized)")
    ax.legend()
    return fig, ax


def plot_event_posteriors(
    h_values: npt.NDArray[np.float64],
    posterior_data: dict[int, list[float]],
    true_h: float,
    *,
    title: str = "Individual event posteriors",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot posteriors from individual EMRI detections."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    for idx, posterior in posterior_data.items():
        arr = np.array(posterior)
        if np.max(arr) > 0:
            ax.plot(h_values, arr / np.max(arr), alpha=0.3, linewidth=0.5)
    ax.axvline(true_h, color="green", linestyle="dashed")
    ax.set_xlabel("Hubble constant h")
    ax.set_ylabel("Posterior (normalized)")
    ax.set_title(title)
    return fig, ax


def plot_subset_posteriors(
    h_values: npt.NDArray[np.float64],
    subset_posteriors: list[npt.NDArray[np.float64]],
    true_h: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot combined posteriors for random subsets of detections."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    for posterior in subset_posteriors:
        normalized = posterior / np.max(posterior) if np.max(posterior) > 0 else posterior
        ax.plot(h_values, normalized, alpha=0.5, linewidth=0.8)
    ax.axvline(true_h, color="green", linestyle="dashed")
    ax.set_xlabel("Hubble constant h")
    ax.set_ylabel("Posterior (normalized)")
    ax.set_title("Subset posteriors")
    return fig, ax


def plot_detection_redshift_distribution(
    redshifts: npt.NDArray[np.float64],
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of detection redshifts."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.hist(redshifts, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Count")
    ax.set_title("Detection redshift distribution")
    return fig, ax


def plot_number_of_possible_hosts(
    host_counts: npt.NDArray[np.float64],
    *,
    bins: int = 30,
    label: str = "Possible hosts",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of number of possible host galaxies per detection."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.hist(host_counts, bins=bins, edgecolor="black", alpha=0.7, label=label)
    ax.set_xlabel("Number of possible hosts")
    ax.set_ylabel("Count")
    ax.legend()
    return fig, ax
