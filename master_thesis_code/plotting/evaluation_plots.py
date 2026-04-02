"""Factory functions for data evaluation plots.

Extracted from ``DataEvaluation.visualize()`` in ``evaluation.py``.

All functions take data in and return ``(fig, ax)`` out.
"""

from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from master_thesis_code.plotting._colors import CMAP, CYCLE, EDGE, MEAN, REFERENCE
from master_thesis_code.plotting._data import label_key
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
from master_thesis_code.plotting._labels import LABELS

_DEFAULT_RECOVERY_PARAMS: list[str] = ["M", "mu", "luminosity_distance", "a", "e0", "qS"]


def plot_mean_cramer_rao_bounds(
    covariance_matrix: npt.NDArray[np.float64],
    parameter_names: list[str],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of mean Cramer-Rao bound covariance matrix."""
    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    im = ax.imshow(covariance_matrix, cmap=CMAP, aspect="auto")
    tick_labels = [LABELS.get(label_key(p), p) for p in parameter_names]
    ax.set_xticks(range(len(parameter_names)))
    ax.set_yticks(range(len(parameter_names)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_uncertainty_violins(
    uncertainties: dict[str, npt.NDArray[np.float64]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Violin plot of relative parameter uncertainties."""
    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    names = list(uncertainties.keys())
    data = [uncertainties[name] for name in names]
    tick_labels = [LABELS.get(label_key(n), n) for n in names]

    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Relative uncertainty")
    return fig, ax


def plot_sky_localization_3d(
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    sky_error: npt.NDArray[np.float64],
) -> tuple[Figure, Any]:
    """3D scatter plot of sky-localization uncertainty."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(theta, phi, sky_error, c=sky_error, cmap=CMAP, alpha=0.6)
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
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """2D histogram of detections in redshift-mass space."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    mass_bins = np.geomspace(masses.min(), masses.max(), bins)
    h = ax.hist2d(redshifts, masses, bins=[bins, mass_bins], cmap=CMAP)  # type: ignore[arg-type]
    fig.colorbar(h[3], ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(LABELS["M"])
    return fig, ax


def plot_generation_time_histogram(
    generation_times: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of waveform generation times."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(generation_times, bins=bins, edgecolor=EDGE, alpha=0.7)
    ax.axvline(float(np.mean(generation_times)), color=MEAN, linestyle="dashed", label="Mean")
    ax.set_xlabel(LABELS["t"])
    ax.set_ylabel("Count")
    ax.legend()
    return fig, ax


def plot_injected_vs_recovered(
    injected: dict[str, npt.NDArray[np.float64]],
    recovered: dict[str, npt.NDArray[np.float64]],
    *,
    uncertainties: dict[str, npt.NDArray[np.float64]] | None = None,
    parameters: list[str] | None = None,
    ncols: int = 3,
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Multi-panel scatter grid comparing injected vs recovered parameters.

    Each parameter gets a main scatter panel (with identity line and
    optional 1-sigma CRB error bars) and a residual sub-panel showing
    ``recovered - injected``.

    Parameters
    ----------
    injected, recovered:
        Dicts mapping parameter name to array of values.
    uncertainties:
        Optional dict mapping parameter name to 1-sigma CRB errors.
    parameters:
        Subset of parameter names to plot.  Defaults to
        ``["M", "mu", "luminosity_distance", "a", "e0", "qS"]``.
    ncols:
        Number of columns in the grid layout.

    Returns
    -------
    tuple[Figure, npt.NDArray[np.object_]]
        Figure and 2D array of all axes (shape ``(nrows * 2, ncols)``).
    """
    if parameters is None:
        parameters = list(_DEFAULT_RECOVERY_PARAMS)

    n_params = len(parameters)
    nrows = ceil(n_params / ncols)
    fig = plt.figure(figsize=(7.0, 2.8 * nrows))
    gs = GridSpec(
        nrows * 2,
        ncols,
        height_ratios=[3, 1] * nrows,
        hspace=0.05,
        wspace=0.35,
    )

    all_axes: list[list[Axes]] = []
    for _ in range(nrows * 2):
        all_axes.append([])

    for idx, p in enumerate(parameters):
        row = idx // ncols
        col = idx % ncols
        gs_main = gs[row * 2, col]
        gs_resid = gs[row * 2 + 1, col]

        ax_main: Axes = fig.add_subplot(gs_main)
        ax_resid: Axes = fig.add_subplot(gs_resid, sharex=ax_main)

        inj = injected[p]
        rec = recovered[p]

        # Identity line
        lo = min(float(inj.min()), float(rec.min()))
        hi = max(float(inj.max()), float(rec.max()))
        ax_main.plot([lo, hi], [lo, hi], color=REFERENCE, linestyle="--", linewidth=1)

        # Main scatter / errorbar
        if uncertainties is not None and p in uncertainties:
            ax_main.errorbar(
                inj,
                rec,
                yerr=uncertainties[p],
                fmt=".",
                color=CYCLE[0],
                capsize=2,
                markersize=3,
                alpha=0.7,
            )
        else:
            ax_main.scatter(inj, rec, s=9, color=CYCLE[0], alpha=0.7, rasterized=True)

        # y-axis label on leftmost column only
        if col == 0:
            lbl = LABELS.get(label_key(p), p)
            ax_main.set_ylabel(f"{lbl} (recovered)")

        # Hide x-tick labels on main panel (shared with residual)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Residual sub-panel
        residual = rec - inj
        if uncertainties is not None and p in uncertainties:
            ax_resid.errorbar(
                inj,
                residual,
                yerr=uncertainties[p],
                fmt=".",
                color=CYCLE[0],
                capsize=2,
                markersize=3,
                alpha=0.7,
            )
        else:
            ax_resid.scatter(inj, residual, s=9, color=CYCLE[0], alpha=0.7, rasterized=True)

        ax_resid.axhline(0, color=REFERENCE, linestyle="--", linewidth=1)

        # x-axis label on bottom row only
        is_bottom_row = row == nrows - 1
        if is_bottom_row:
            lbl = LABELS.get(label_key(p), p)
            ax_resid.set_xlabel(f"{lbl} (injected)")

        # Residual y-axis label on leftmost column
        if col == 0:
            ax_resid.set_ylabel(r"$\Delta$")

        all_axes[row * 2].append(ax_main)
        all_axes[row * 2 + 1].append(ax_resid)

    # Pad rows that have fewer columns than ncols
    for row_axes in all_axes:
        while len(row_axes) < ncols:
            row_axes.append(None)  # type: ignore[arg-type]

    axes_array: npt.NDArray[np.object_] = np.array(all_axes, dtype=object)
    return fig, axes_array
