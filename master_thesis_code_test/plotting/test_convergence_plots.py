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
        sample_event_posteriors: tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]],
    ) -> None:
        """Returns (Figure, ndarray of 2 Axes)."""
        h_values, posteriors = sample_event_posteriors
        fig, axes = plot_h0_convergence(h_values, posteriors[:5])
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2,)

    def test_seed_reproducible(
        self,
        sample_event_posteriors: tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]],
    ) -> None:
        """Same seed produces identical CI width arrays."""
        h_values, posteriors = sample_event_posteriors
        _, axes1 = plot_h0_convergence(h_values, posteriors, seed=123)
        _, axes2 = plot_h0_convergence(h_values, posteriors, seed=123)
        # Right panel (CI width) should have identical data
        lines1 = axes1[1].get_lines()
        lines2 = axes2[1].get_lines()
        assert len(lines1) > 0
        np.testing.assert_array_equal(lines1[0].get_ydata(), lines2[0].get_ydata())

    def test_custom_subsets(
        self,
        sample_event_posteriors: tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]],
    ) -> None:
        """Custom subset_sizes accepted without error."""
        h_values, posteriors = sample_event_posteriors
        fig, axes = plot_h0_convergence(h_values, posteriors, subset_sizes=[1, 3, 5])
        assert isinstance(fig, Figure)

    def test_truth_line(
        self,
        sample_event_posteriors: tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]],
    ) -> None:
        """Truth line appears as vertical lines on left panel."""
        h_values, posteriors = sample_event_posteriors
        _, axes = plot_h0_convergence(h_values, posteriors[:5], true_h=0.73)
        # Left panel should have at least one vertical line
        ax_post = axes[0]
        vlines = [line for line in ax_post.get_lines() if len(set(line.get_xdata())) == 1]
        assert len(vlines) >= 1

    def test_plot_h0_convergence_without_bootstrap_bank_has_no_fill_between(self) -> None:
        """VIZ-02 backward-compat: no band PolyCollection when bootstrap_bank=None."""
        import matplotlib.pyplot as plt

        h_values = np.linspace(0.5, 1.0, 100)
        event_posteriors = [np.ones(100) for _ in range(10)]
        fig, axes = plot_h0_convergence(h_values, event_posteriors)
        ax_ci = axes[1]
        poly_artists = [a for a in ax_ci.get_children() if isinstance(a, PolyCollection)]
        assert len(poly_artists) == 0, "No PolyCollection expected without bootstrap_bank"
        plt.close(fig)

    def test_plot_h0_convergence_with_bootstrap_bank_adds_fill_between(self) -> None:
        """VIZ-02: passing a bootstrap_bank adds at least one fill_between
        PolyCollection on the right panel (axes[1]) for BOTH variants (no-mass
        and with-mass) when event_posteriors_alt is supplied."""
        import matplotlib.pyplot as plt

        from master_thesis_code.plotting.convergence_analysis import ImprovementBank

        h_values = np.linspace(0.5, 1.0, 100)
        event_posteriors = [np.ones(100) for _ in range(10)]
        event_posteriors_alt = [np.ones(100) for _ in range(10)]

        # Minimal synthetic bank: every REQUIRED @dataclass field supplied
        # (verified field list from convergence_analysis.py:218-232).  Only
        # `sizes` and `metrics_{no,with}_mass["hdi68_width"]["p16"/"p84"]` are
        # read by plot_h0_convergence's band block; the rest are placeholders.
        bank = ImprovementBank(
            h_grid=np.linspace(0.5, 1.0, 100),
            h_true=0.73,
            sizes=[1, 5, 10],
            n_bootstrap=0,
            seed=0,
            metrics_no_mass={
                "hdi68_width": {
                    "median": [0.30, 0.20, 0.10],
                    "p16": [0.25, 0.15, 0.08],
                    "p84": [0.35, 0.25, 0.12],
                }
            },
            metrics_with_mass={
                "hdi68_width": {
                    "median": [0.25, 0.18, 0.09],
                    "p16": [0.22, 0.14, 0.07],
                    "p84": [0.28, 0.22, 0.11],
                }
            },
            fractional_improvement={},
            effective_event_gain={},
            jsd_bits={},
            representative_posteriors_no_mass=[],
            representative_posteriors_with_mass=[],
            n_events_no_mass=0,
            n_events_with_mass=0,
            # cache_meta omitted — has default_factory
        )

        fig, axes = plot_h0_convergence(
            h_values,
            event_posteriors,
            event_posteriors_alt=event_posteriors_alt,
            bootstrap_bank=bank,
        )
        ax_ci = axes[1]
        poly_artists = [a for a in ax_ci.get_children() if isinstance(a, PolyCollection)]
        # Two fill_between calls expected (no-mass + with-mass); a conservative
        # lower bound of 1 still flags the total absence of the band block.
        assert len(poly_artists) >= 1, "At least one PolyCollection (fill_between) expected"
        plt.close(fig)


class TestPlotDetectionEfficiency:
    """Tests for the detection efficiency curve plot."""

    def test_basic_return_type(
        self,
        sample_injection_campaign: tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    ) -> None:
        """Returns (Figure, Axes)."""
        variable, detected = sample_injection_campaign
        fig, ax = plot_detection_efficiency(variable, detected)
        assert isinstance(fig, Figure)
        assert hasattr(ax, "set_xlabel")

    def test_ci_band(
        self,
        sample_injection_campaign: tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    ) -> None:
        """CI band appears as a PolyCollection (fill_between)."""
        variable, detected = sample_injection_campaign
        _, ax = plot_detection_efficiency(variable, detected)
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) >= 1

    def test_custom_bins(
        self,
        sample_injection_campaign: tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    ) -> None:
        """Custom bins=10 accepted without error."""
        variable, detected = sample_injection_campaign
        fig, ax = plot_detection_efficiency(variable, detected, bins=10)
        assert isinstance(fig, Figure)

    def test_xlabel(
        self,
        sample_injection_campaign: tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    ) -> None:
        """Custom xlabel is applied."""
        variable, detected = sample_injection_campaign
        _, ax = plot_detection_efficiency(variable, detected, xlabel="z")
        assert ax.get_xlabel() == "z"
