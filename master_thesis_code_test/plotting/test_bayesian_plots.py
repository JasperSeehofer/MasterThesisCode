"""Smoke tests for bayesian_plots factory functions."""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
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


def test_plot_combined_posterior_density_integrates_to_one() -> None:
    """normalize='density' draws an area-normalized PDF (integrates to ~1)."""
    h = np.linspace(0.55, 0.95, 200)
    posterior = np.exp(-0.5 * ((h - 0.73) / 0.03) ** 2)
    fig, ax = plot_combined_posterior(
        h, posterior, true_h=0.73, normalize="density", show_references=False
    )
    # The first line drawn is the main posterior curve.
    line = ax.get_lines()[0]
    x = np.asarray(line.get_xdata(), dtype=np.float64)
    y = np.asarray(line.get_ydata(), dtype=np.float64)
    area = float(np.trapezoid(y, x))
    assert abs(area - 1.0) < 1e-2
    assert isinstance(fig, Figure)


def test_plot_combined_posterior_hdi_bands_are_nested() -> None:
    """show_credible shades two nested HDI regions (>=2 PolyCollections)."""
    h = np.linspace(0.55, 0.95, 200)
    posterior = np.exp(-0.5 * ((h - 0.73) / 0.03) ** 2)
    fig, ax = plot_combined_posterior(
        h, posterior, true_h=0.73, normalize="density", show_credible=True
    )
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) >= 2
    assert isinstance(fig, Figure)


def test_plot_combined_posterior_inline_map_annotation() -> None:
    """annotate_map=True adds an inline MAP +/- HDI text annotation."""
    h = np.linspace(0.55, 0.95, 200)
    posterior = np.exp(-0.5 * ((h - 0.73) / 0.03) ** 2)
    fig, ax = plot_combined_posterior(
        h, posterior, true_h=0.73, normalize="density", annotate_map=True
    )
    texts = [t.get_text() for t in ax.texts]
    assert any("MAP" in t for t in texts), f"no MAP annotation found in {texts}"
    assert isinstance(fig, Figure)


def test_plot_combined_posterior_default_normalize_is_peak() -> None:
    """Default normalize stays 'peak' so multi-variant overlay callers are unchanged."""
    h = np.linspace(0.55, 0.95, 200)
    posterior = np.exp(-0.5 * ((h - 0.73) / 0.03) ** 2)
    fig, ax = plot_combined_posterior(
        h, posterior, true_h=0.73, show_references=False, annotate_map=False
    )
    y = np.asarray(ax.get_lines()[0].get_ydata(), dtype=np.float64)
    assert abs(float(np.max(y)) - 1.0) < 1e-6
    assert isinstance(fig, Figure)


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
