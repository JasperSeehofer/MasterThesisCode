"""Smoke tests for physical_relations_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting.physical_relations_plots import plot_distance_redshift


def test_plot_distance_redshift(
    sample_redshifts: npt.NDArray[np.float64],
    sample_distances: npt.NDArray[np.float64],
) -> None:
    fig, ax = plot_distance_redshift(sample_redshifts, sample_distances)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
