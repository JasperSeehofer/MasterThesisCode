"""Factory functions for physical relations plots.

Extracted from ``physical_relations.visualize()``.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import CYCLE
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
from master_thesis_code.plotting._labels import LABELS


def plot_distance_redshift(
    redshifts: npt.NDArray[np.float64],
    distances: npt.NDArray[np.float64],
    *,
    h0_values: list[float] | None = None,
    distance_fn: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]] | None = None,
    label: str = "d_L(z)",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot luminosity distance vs redshift.

    Parameters
    ----------
    redshifts:
        Redshift array for the primary curve.
    distances:
        Luminosity distances for the primary curve.
    h0_values:
        Optional list of H0 values for comparison curves.  Requires
        *distance_fn* to be provided.
    distance_fn:
        Callable ``(redshifts, h0) -> distances`` used to compute
        comparison curves for each value in *h0_values*.
    label:
        Label for the primary curve.
    ax:
        Optional pre-existing Axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(redshifts, distances, label=label)

    if h0_values is not None:
        if distance_fn is None:
            msg = "distance_fn must be provided when h0_values is set"
            raise ValueError(msg)
        for i, h0 in enumerate(h0_values):
            d = distance_fn(redshifts, h0)
            color = CYCLE[i % len(CYCLE)]
            ax.plot(redshifts, d, color=color, label=f"$h = {h0}$")

    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(LABELS["d_L"])
    ax.legend()
    return fig, ax
