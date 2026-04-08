"""Smoke tests for paper_figures.py factory functions."""

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
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
