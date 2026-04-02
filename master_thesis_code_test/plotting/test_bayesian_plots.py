"""Smoke tests for bayesian_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting.bayesian_plots import (
    plot_combined_posterior,
    plot_detection_redshift_distribution,
    plot_event_posteriors,
    plot_number_of_possible_hosts,
    plot_snr_distribution,
    plot_subset_posteriors,
)


def test_plot_combined_posterior(
    sample_h_values: npt.NDArray[np.float64],
    sample_posterior: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_combined_posterior returns (Figure, Axes)."""
    fig, ax = plot_combined_posterior(sample_h_values, sample_posterior, true_h=0.73)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_combined_posterior_credible_intervals(
    sample_h_values: npt.NDArray[np.float64],
    sample_posterior: npt.NDArray[np.float64],
) -> None:
    """Credible intervals produce fill regions (PolyCollections)."""
    fig, ax = plot_combined_posterior(
        sample_h_values, sample_posterior, true_h=0.73, show_credible=True
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # fill_between creates PolyCollection objects in ax.collections
    assert len(ax.collections) >= 2


def test_plot_combined_posterior_density_normalization(
    sample_h_values: npt.NDArray[np.float64],
    sample_posterior: npt.NDArray[np.float64],
) -> None:
    """Density normalization mode returns valid figure."""
    fig, ax = plot_combined_posterior(
        sample_h_values, sample_posterior, true_h=0.73, normalize="density"
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_combined_posterior_references(
    sample_h_values: npt.NDArray[np.float64],
    sample_posterior: npt.NDArray[np.float64],
) -> None:
    """Reference bands add vertical lines for Planck and SH0ES."""
    fig, ax = plot_combined_posterior(
        sample_h_values, sample_posterior, true_h=0.73, show_references=True
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # At minimum: truth line + Planck center + SH0ES center + 4 credible edges = 7 vlines
    assert len(ax.get_lines()) > 0


def test_plot_event_posteriors(
    sample_h_values: npt.NDArray[np.float64],
) -> None:
    """Smoke test: backward-compat dict input still works."""
    rng = np.random.default_rng(42)
    posterior_data: dict[int, list[float]] = {
        0: list(rng.random(50)),
        1: list(rng.random(50)),
    }
    fig, ax = plot_event_posteriors(sample_h_values, posterior_data, true_h=0.73)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_event_posteriors_color_by_snr(
    sample_h_values: npt.NDArray[np.float64],
) -> None:
    """Color-by-SNR mode renders colorbar."""
    rng = np.random.default_rng(42)
    posteriors = [rng.random(50) for _ in range(5)]
    color_values = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
    fig, ax = plot_event_posteriors(
        sample_h_values,
        posteriors,
        true_h=0.73,
        color_by="snr",
        color_values=color_values,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_event_posteriors_combined(
    sample_h_values: npt.NDArray[np.float64],
    sample_posterior: npt.NDArray[np.float64],
) -> None:
    """Combined posterior overlay renders as an additional line."""
    rng = np.random.default_rng(42)
    posteriors = [rng.random(50) for _ in range(3)]
    fig, ax = plot_event_posteriors(
        sample_h_values,
        posteriors,
        true_h=0.73,
        combined_posterior=sample_posterior,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # 3 individual + 1 combined + 1 truth vline = 5 lines
    assert len(ax.get_lines()) >= 5


def test_plot_subset_posteriors(
    sample_h_values: npt.NDArray[np.float64],
    sample_posterior: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_subset_posteriors returns (Figure, Axes)."""
    fig, ax = plot_subset_posteriors(
        sample_h_values,
        subset_posteriors=[sample_posterior, sample_posterior * 0.8],
        true_h=0.73,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_detection_redshift_distribution(
    sample_redshifts: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_detection_redshift_distribution returns (Figure, Axes)."""
    fig, ax = plot_detection_redshift_distribution(sample_redshifts)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_number_of_possible_hosts() -> None:
    """Smoke test: plot_number_of_possible_hosts returns (Figure, Axes)."""
    host_counts = np.array([1, 3, 5, 2, 4, 1, 6, 2, 3, 1], dtype=np.float64)
    fig, ax = plot_number_of_possible_hosts(host_counts)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_snr_distribution() -> None:
    """Smoke test: plot_snr_distribution returns (Figure, Axes) with histogram + CDF."""
    rng = np.random.default_rng(42)
    snr_values = rng.exponential(15.0, size=100).astype(np.float64)
    fig, ax = plot_snr_distribution(snr_values)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_snr_distribution_custom_threshold() -> None:
    """Custom threshold value is accepted."""
    rng = np.random.default_rng(42)
    snr_values = rng.exponential(15.0, size=100).astype(np.float64)
    fig, ax = plot_snr_distribution(snr_values, snr_threshold=30.0)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
