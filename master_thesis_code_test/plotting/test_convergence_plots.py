"""Smoke tests for convergence_plots.py factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from master_thesis_code.plotting.convergence_plots import (
    plot_detection_efficiency,
    plot_h0_convergence,
)


class TestPlotH0Convergence:
    """Tests for the H0 convergence two-panel plot."""

    def test_basic_return_type(
        self,
        sample_event_posteriors: tuple[
            npt.NDArray[np.float64], list[npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Returns (Figure, ndarray of 2 Axes)."""
        h_values, posteriors = sample_event_posteriors
        fig, axes = plot_h0_convergence(h_values, posteriors[:5])
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2,)

    def test_seed_reproducible(
        self,
        sample_event_posteriors: tuple[
            npt.NDArray[np.float64], list[npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Same seed produces identical CI width arrays."""
        h_values, posteriors = sample_event_posteriors
        _, axes1 = plot_h0_convergence(h_values, posteriors, seed=123)
        _, axes2 = plot_h0_convergence(h_values, posteriors, seed=123)
        # Right panel (CI width) should have identical data
        lines1 = axes1[1].get_lines()
        lines2 = axes2[1].get_lines()
        assert len(lines1) > 0
        np.testing.assert_array_equal(
            lines1[0].get_ydata(), lines2[0].get_ydata()
        )

    def test_custom_subsets(
        self,
        sample_event_posteriors: tuple[
            npt.NDArray[np.float64], list[npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Custom subset_sizes accepted without error."""
        h_values, posteriors = sample_event_posteriors
        fig, axes = plot_h0_convergence(
            h_values, posteriors, subset_sizes=[1, 3, 5]
        )
        assert isinstance(fig, Figure)

    def test_truth_line(
        self,
        sample_event_posteriors: tuple[
            npt.NDArray[np.float64], list[npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Truth line appears as vertical lines on left panel."""
        h_values, posteriors = sample_event_posteriors
        _, axes = plot_h0_convergence(h_values, posteriors[:5], true_h=0.73)
        # Left panel should have at least one vertical line
        ax_post = axes[0]
        vlines = [
            line
            for line in ax_post.get_lines()
            if len(set(line.get_xdata())) == 1
        ]
        assert len(vlines) >= 1


class TestPlotDetectionEfficiency:
    """Tests for the detection efficiency curve plot."""

    def test_basic_return_type(
        self,
        sample_injection_campaign: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.bool_]
        ],
    ) -> None:
        """Returns (Figure, Axes)."""
        variable, detected = sample_injection_campaign
        fig, ax = plot_detection_efficiency(variable, detected)
        assert isinstance(fig, Figure)
        assert hasattr(ax, "set_xlabel")

    def test_ci_band(
        self,
        sample_injection_campaign: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.bool_]
        ],
    ) -> None:
        """CI band appears as a PolyCollection (fill_between)."""
        variable, detected = sample_injection_campaign
        _, ax = plot_detection_efficiency(variable, detected)
        polys = [
            c for c in ax.collections if isinstance(c, PolyCollection)
        ]
        assert len(polys) >= 1

    def test_custom_bins(
        self,
        sample_injection_campaign: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.bool_]
        ],
    ) -> None:
        """Custom bins=10 accepted without error."""
        variable, detected = sample_injection_campaign
        fig, ax = plot_detection_efficiency(variable, detected, bins=10)
        assert isinstance(fig, Figure)

    def test_xlabel(
        self,
        sample_injection_campaign: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.bool_]
        ],
    ) -> None:
        """Custom xlabel is applied."""
        variable, detected = sample_injection_campaign
        _, ax = plot_detection_efficiency(variable, detected, xlabel="z")
        assert ax.get_xlabel() == "z"
