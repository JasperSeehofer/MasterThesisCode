"""Factory functions for data evaluation plots.

Extracted from ``DataEvaluation.visualize()`` in ``evaluation.py``.

All functions take data in and return ``(fig, ax)`` out.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _fig_from_ax


def plot_mean_cramer_rao_bounds(
    covariance_matrix: npt.NDArray[np.float64],
    parameter_names: list[str],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of mean Cramér-Rao bound covariance matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = _fig_from_ax(ax)

    im = ax.imshow(covariance_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(parameter_names)))
    ax.set_yticks(range(len(parameter_names)))
    ax.set_xticklabels(parameter_names, rotation=45, ha="right")
    ax.set_yticklabels(parameter_names)
    fig.colorbar(im, ax=ax)
    ax.set_title("Mean Cramér-Rao bounds")
    return fig, ax


def plot_uncertainty_violins(
    uncertainties: dict[str, npt.NDArray[np.float64]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Violin plot of relative parameter uncertainties."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = _fig_from_ax(ax)

    names = list(uncertainties.keys())
    data = [uncertainties[name] for name in names]

    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Relative uncertainty")
    ax.set_title("Parameter uncertainty distributions")
    return fig, ax


def plot_sky_localization_3d(
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    sky_error: npt.NDArray[np.float64],
) -> tuple[Figure, Any]:
    """3D scatter plot of sky-localization uncertainty."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(theta, phi, sky_error, c=sky_error, cmap="viridis", alpha=0.6)
    ax.set_xlabel("theta")
    ax.set_ylabel("phi")
    ax.set_zlabel("Sky localization error")
    fig.colorbar(sc, ax=ax, label="Error")
    return fig, ax


def plot_detection_contour(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    title: str = "Detection distribution",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """2D histogram of detections in redshift-mass space."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    mass_bins = np.geomspace(masses.min(), masses.max(), bins)
    h = ax.hist2d(redshifts, masses, bins=[bins, mass_bins], cmap="viridis")  # type: ignore[arg-type]
    fig.colorbar(h[3], ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Mass [M_sun]")
    ax.set_title(title)
    return fig, ax


def plot_generation_time_histogram(
    generation_times: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of waveform generation times."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.hist(generation_times, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(float(np.mean(generation_times)), color="red", linestyle="dashed", label="Mean")
    ax.set_xlabel("Generation time [s]")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Waveform generation time distribution")
    return fig, ax
