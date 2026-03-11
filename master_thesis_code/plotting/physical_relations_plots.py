"""Factory functions for physical relations plots.

Extracted from ``physical_relations.visualize()``.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting.simulation_plots import _fig_from_ax


def plot_distance_redshift(
    redshifts: npt.NDArray[np.float64],
    distances: npt.NDArray[np.float64],
    *,
    label: str = "d_L(z)",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot luminosity distance vs redshift."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = _fig_from_ax(ax)

    ax.plot(redshifts, distances, label=label)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Luminosity distance [Gpc]")
    ax.legend()
    return fig, ax
