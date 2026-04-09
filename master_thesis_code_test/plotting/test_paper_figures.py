"""Smoke tests for paper_figures.py factory functions."""

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from master_thesis_code.plotting._style import apply_style


@pytest.fixture(autouse=True)
def _apply_style() -> None:
    apply_style()


class TestPaperFigureImports:
    """Verify all 4 paper figure functions are importable and have correct signatures."""

    def test_plot_h0_posterior_comparison_importable(self) -> None:
        from master_thesis_code.plotting.paper_figures import plot_h0_posterior_comparison

        sig = inspect.signature(plot_h0_posterior_comparison)
        assert "data_dir" in sig.parameters

    def test_plot_single_event_likelihoods_importable(self) -> None:
        from master_thesis_code.plotting.paper_figures import plot_single_event_likelihoods

        sig = inspect.signature(plot_single_event_likelihoods)
        assert "data_dir" in sig.parameters

    def test_plot_posterior_convergence_importable(self) -> None:
        from master_thesis_code.plotting.paper_figures import plot_posterior_convergence

        sig = inspect.signature(plot_posterior_convergence)
        assert "data_dir" in sig.parameters

    def test_plot_snr_distribution_importable(self) -> None:
        from master_thesis_code.plotting.paper_figures import plot_snr_distribution

        sig = inspect.signature(plot_snr_distribution)
        assert "data_dir" in sig.parameters


class TestPaperFigureGracefulDegradation:
    """Paper figure functions return or raise gracefully when data is missing."""

    def test_h0_posterior_raises_on_missing_data(self, tmp_path: Path) -> None:
        from master_thesis_code.plotting.paper_figures import plot_h0_posterior_comparison

        with pytest.raises(FileNotFoundError):
            plot_h0_posterior_comparison(data_dir=tmp_path)

    def test_single_event_raises_on_missing_data(self, tmp_path: Path) -> None:
        from master_thesis_code.plotting.paper_figures import plot_single_event_likelihoods

        with pytest.raises((FileNotFoundError, OSError)):
            plot_single_event_likelihoods(data_dir=tmp_path)

    def test_snr_distribution_placeholder_on_missing_data(self, tmp_path: Path) -> None:
        from master_thesis_code.plotting.paper_figures import plot_snr_distribution

        fig, ax = plot_snr_distribution(data_dir=tmp_path)
        assert fig is not None
        plt.close(fig)


class TestNoStandaloneMain:
    """Verify standalone main() was removed per D-01."""

    def test_no_main_function(self) -> None:
        import master_thesis_code.plotting.paper_figures as pf

        assert not hasattr(pf, "main"), "main() should have been removed per D-01"

    def test_no_data_root_constant(self) -> None:
        import master_thesis_code.plotting.paper_figures as pf

        assert not hasattr(pf, "_DATA_ROOT"), "_DATA_ROOT should have been removed per D-02"


class TestKDESmoothing:
    """Tests for _kde_smooth_posterior and plot_h0_posterior_kde."""

    def test_kde_smooth_returns_correct_shape(self) -> None:
        from master_thesis_code.plotting.paper_figures import _kde_smooth_posterior

        h = np.linspace(0.60, 0.90, 31)
        posterior = np.exp(-0.5 * ((h - 0.73) / 0.04) ** 2)
        h_fine, kde_fine = _kde_smooth_posterior(h, posterior)
        assert len(h_fine) == 500
        assert len(kde_fine) == 500

    def test_kde_map_within_grid_spacing(self) -> None:
        from master_thesis_code.plotting.paper_figures import _kde_smooth_posterior

        h = np.linspace(0.60, 0.90, 15)
        posterior = np.exp(-0.5 * ((h - 0.73) / 0.04) ** 2)
        grid_spacing = float(np.diff(h).mean())
        h_fine, kde_fine = _kde_smooth_posterior(h, posterior)
        discrete_map = h[np.argmax(posterior)]
        kde_map = h_fine[np.argmax(kde_fine)]
        assert abs(kde_map - discrete_map) < grid_spacing

    def test_kde_zero_posterior_returns_copy(self) -> None:
        from master_thesis_code.plotting.paper_figures import _kde_smooth_posterior

        h = np.linspace(0.60, 0.90, 31)
        posterior = np.zeros_like(h)
        h_out, kde_out = _kde_smooth_posterior(h, posterior)
        assert np.array_equal(h_out, h)
        assert np.array_equal(kde_out, posterior)

    def test_auto_detect_grid_spacing_15pt(self) -> None:
        h = np.linspace(0.60, 0.90, 15)
        spacing = float(np.diff(h).mean())
        assert abs(spacing - 0.0214) < 0.001

    def test_auto_detect_grid_spacing_31pt(self) -> None:
        h = np.linspace(0.60, 0.90, 31)
        spacing = float(np.diff(h).mean())
        assert abs(spacing - 0.0100) < 0.001

    def test_plot_h0_posterior_kde_returns_fig_axes(self, tmp_path: Path) -> None:
        """plot_h0_posterior_kde returns (Figure, Axes) for a valid data dir."""
        import json

        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        from master_thesis_code.plotting.paper_figures import plot_h0_posterior_kde

        # Synthetic posterior JSON files
        h_vals = np.linspace(0.60, 0.90, 31).tolist()
        posterior = np.exp(-0.5 * ((np.array(h_vals) - 0.73) / 0.04) ** 2).tolist()
        data_no = {"h_values": h_vals, "posterior": posterior, "map_h": 0.73}
        data_with = {"h_values": h_vals, "posterior": posterior, "map_h": 0.73}
        (tmp_path / "combined_posterior.json").write_text(json.dumps(data_no))
        (tmp_path / "combined_posterior_with_bh_mass.json").write_text(json.dumps(data_with))

        fig, ax = plot_h0_posterior_kde(data_dir=tmp_path)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPosteriorConvergenceDualVariant:
    """Tests for plot_posterior_convergence showing both analysis variants."""

    def _write_per_event_json(
        self,
        directory: Path,
        h_vals: list[float],
        n_events: int,
        rng: np.random.Generator,
    ) -> None:
        """Write synthetic per-event posterior JSON files.

        Each file is named ``h_{int}_{frac}.json`` (matching the loader's
        naming convention) and contains event IDs as keys with
        single-element list values.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for h in h_vals:
            int_part = int(h)
            frac = round((h - int_part) * 100)
            fname = f"h_{int_part}_{frac:02d}.json"
            data = {str(i): [float(rng.uniform(0.1, 2.0))] for i in range(n_events)}
            (directory / fname).write_text(json.dumps(data))

    def test_posterior_convergence_with_synthetic_data(self, tmp_path: Path) -> None:
        """plot_posterior_convergence returns (Figure, Axes) and plots both variants."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        from master_thesis_code.plotting.paper_figures import plot_posterior_convergence

        rng = np.random.default_rng(42)
        # 15-point h-grid from 0.60 to 0.88 (step 0.02)
        h_vals = [round(0.60 + i * 0.02, 2) for i in range(15)]
        n_events = 20

        self._write_per_event_json(tmp_path / "posteriors", h_vals, n_events, rng)
        self._write_per_event_json(tmp_path / "posteriors_with_bh_mass", h_vals, n_events, rng)

        fig, ax = plot_posterior_convergence(
            data_dir=tmp_path,
            subset_sizes=[5, 10, 15],
            n_subsets=5,
        )

        assert isinstance(fig, Figure)
        # Both variants should be plotted as errorbar containers.
        assert len(ax.containers) >= 2, (
            f"Expected at least 2 errorbar containers (one per variant), got {len(ax.containers)}"
        )
        plt.close(fig)
