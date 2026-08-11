"""Smoke tests for physical_relations_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from darksiren_emri.plotting.physical_relations_plots import plot_distance_redshift


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
    """Multi-H0 mode draws one direct-labelled curve per H0 value.

    The redesigned grammar (Observatory + Atlas) plots exactly the per-H0
    family -- the fiducial curve is one of those members -- and direct-labels
    each curve at its right endpoint instead of carrying a legend.
    """

    def distance_fn(z: npt.NDArray[np.float64], h0: float) -> npt.NDArray[np.float64]:
        return z * 4000.0 / h0

    h0_values = [0.674, 0.73]
    fig, ax = plot_distance_redshift(
        sample_redshifts,
        sample_distances,
        h0_values=h0_values,
        distance_fn=distance_fn,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # One curve per H0 value.
    assert len(ax.get_lines()) == len(h0_values)
    # Direct-labelled (no legend); one annotation per curve.
    assert ax.get_legend() is None
    assert len(ax.texts) == len(h0_values)
    # Luminosity-distance axis is labelled in Gpc (unit-bug fix).
    assert "Gpc" in ax.get_ylabel()
