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


def test_plot_distance_redshift_multi_h0(
    sample_redshifts: npt.NDArray[np.float64],
    sample_distances: npt.NDArray[np.float64],
) -> None:
    """Multi-H0 mode plots additional comparison curves."""

    def distance_fn(
        z: npt.NDArray[np.float64], h0: float
    ) -> npt.NDArray[np.float64]:
        return z * 4000.0 / h0

    fig, ax = plot_distance_redshift(
        sample_redshifts,
        sample_distances,
        h0_values=[0.674, 0.73],
        distance_fn=distance_fn,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # Original curve + 2 H0 curves = at least 3 lines
    assert len(ax.get_lines()) >= 3
