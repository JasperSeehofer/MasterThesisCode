"""Factory functions for cosmological model plots.

Extracted from ``Model1CrossCheck`` and ``DetectionProbability`` in
``cosmological_model.py``.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import CMAP, CYCLE, EDGE
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure, make_colorbar
from master_thesis_code.plotting._labels import LABELS


def _plot_detection_heatmap(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prob: npt.NDArray[np.float64],
    xlabel: str,
    ylabel: str,
    *,
    contour_levels: list[float] | None = None,
    injected_coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    detected_mask: npt.NDArray[np.bool_] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Shared implementation for detection probability heatmaps.

    Parameters
    ----------
    x, y:
        Meshgrid arrays for the two axes.
    prob:
        Detection probability values on the grid, expected in [0, 1].
    xlabel, ylabel:
        LaTeX axis labels.
    contour_levels:
        Probability thresholds for contour lines (default ``[0.5, 0.9]``).
    injected_coords:
        Tuple ``(x_array, y_array)`` of injected event coordinates for
        scatter overlay.
    detected_mask:
        Boolean mask selecting detected events.  When *injected_coords*
        is given, detected events are shown as filled circles and missed
        events as open circles.  Ignored when *injected_coords* is None.
    ax:
        Optional pre-existing axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if contour_levels is None:
        contour_levels = [0.5, 0.9]

    cs = ax.contourf(
        x, y, prob,
        levels=np.linspace(0, 1, 51),
        cmap=CMAP,
        vmin=0,
        vmax=1,
    )

    # Contour lines at specified probability thresholds
    contours = ax.contour(
        x, y, prob,
        levels=contour_levels,
        colors=EDGE,
        linewidths=1.0,
    )
    ax.clabel(contours, inline=True, fontsize=8)

    # Scatter overlay for injected population
    if injected_coords is not None:
        inj_x, inj_y = injected_coords
        if detected_mask is not None:
            ax.scatter(
                inj_x[detected_mask],
                inj_y[detected_mask],
                marker="o",
                facecolors=CYCLE[0],
                edgecolors=EDGE,
                s=10,
                alpha=0.6,
                zorder=3,
            )
            ax.scatter(
                inj_x[~detected_mask],
                inj_y[~detected_mask],
                marker="o",
                facecolors="none",
                edgecolors=CYCLE[3],
                s=10,
                alpha=0.6,
                zorder=3,
            )
        else:
            ax.scatter(
                inj_x,
                inj_y,
                marker="o",
                facecolors=CYCLE[0],
                edgecolors=EDGE,
                s=10,
                alpha=0.6,
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    make_colorbar(cs, fig, ax, label=r"$P_\mathrm{det}$")
    return fig, ax


def plot_emri_distribution(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    distribution: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of the EMRI event distribution in (z, M) space."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    cs = ax.contourf(redshifts, masses, distribution, cmap=CMAP, levels=30)
    fig.colorbar(cs, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(LABELS["M"])
    return fig, ax


def plot_emri_rate(
    masses: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Log-log plot of EMRI rate vs MBH mass."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(masses, rates)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(LABELS["M"])
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
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    h = ax.hist2d(redshifts, masses, bins=[redshift_bins, mass_bins], cmap=CMAP)
    fig.colorbar(h[3], ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(LABELS["M"])
    return fig, ax


def plot_detection_probability_grid(
    d_L_range: npt.NDArray[np.float64],
    M_range: npt.NDArray[np.float64],
    detection_prob: npt.NDArray[np.float64],
    *,
    contour_levels: list[float] | None = None,
    injected_coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    detected_mask: npt.NDArray[np.bool_] | None = None,
    title: str = "",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of detection probability in (d_L, M) space.

    Parameters
    ----------
    d_L_range, M_range:
        Meshgrid arrays for luminosity distance and mass.
    detection_prob:
        Detection probability on the grid.
    contour_levels:
        Probability thresholds for contour lines (default ``[0.5, 0.9]``).
    injected_coords:
        ``(d_L_array, M_array)`` for scatter overlay of injected events.
    detected_mask:
        Boolean mask selecting detected events (filled circles).
        Missed events are shown as open circles.
    title:
        Optional figure title.  Omitted when empty (thesis convention).
    ax:
        Optional pre-existing axes.
    """
    fig, ax = _plot_detection_heatmap(
        d_L_range,
        M_range,
        detection_prob,
        LABELS["d_L"],
        LABELS["M"],
        contour_levels=contour_levels,
        injected_coords=injected_coords,
        detected_mask=detected_mask,
        ax=ax,
    )
    if title:
        ax.set_title(title)
    return fig, ax


def plot_detection_probability_zM(
    z_range: npt.NDArray[np.float64],
    M_range: npt.NDArray[np.float64],
    detection_prob: npt.NDArray[np.float64],
    *,
    contour_levels: list[float] | None = None,
    injected_coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    detected_mask: npt.NDArray[np.bool_] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of detection probability in (z, M) space.

    Parameters
    ----------
    z_range, M_range:
        Meshgrid arrays for redshift and mass.
    detection_prob:
        Detection probability on the grid.
    contour_levels:
        Probability thresholds for contour lines (default ``[0.5, 0.9]``).
    injected_coords:
        ``(z_array, M_array)`` for scatter overlay of injected events.
    detected_mask:
        Boolean mask selecting detected events (filled circles).
        Missed events are shown as open circles.
    ax:
        Optional pre-existing axes.
    """
    return _plot_detection_heatmap(
        z_range,
        M_range,
        detection_prob,
        LABELS["z"],
        LABELS["M"],
        contour_levels=contour_levels,
        injected_coords=injected_coords,
        detected_mask=detected_mask,
        ax=ax,
    )
