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


def test_plot_event_posteriors(
    sample_h_values: npt.NDArray[np.float64],
) -> None:
    """Smoke test: plot_event_posteriors returns (Figure, Axes)."""
    rng = np.random.default_rng(42)
    posterior_data = {
        0: list(rng.random(50)),
        1: list(rng.random(50)),
    }
    fig, ax = plot_event_posteriors(sample_h_values, posterior_data, true_h=0.73)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


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
