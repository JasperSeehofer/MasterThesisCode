"""Tests for _helpers.py: figure presets, figsize override, _fig_from_ax."""

import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

from master_thesis_code.plotting import apply_style, get_figure
from master_thesis_code.plotting._helpers import (
    _fig_from_ax,
    compute_credible_interval,
    diverging_norm,
    make_heatmap_norm,
)


def test_get_figure_preset_single_width() -> None:
    apply_style()
    fig, ax = get_figure(preset="single")
    w, _ = fig.get_size_inches()
    assert abs(w - 3.375) < 0.01, f"single preset width {w}, expected ~3.375"
    plt.close(fig)


def test_get_figure_preset_double_width() -> None:
    apply_style()
    fig, ax = get_figure(preset="double")
    w, _ = fig.get_size_inches()
    assert abs(w - 7.0) < 0.01, f"double preset width {w}, expected ~7.0"
    plt.close(fig)


def test_get_figure_no_args_uses_mplstyle_default() -> None:
    apply_style()
    fig, ax = get_figure()
    w, h = fig.get_size_inches()
    assert abs(w - 6.4) < 0.01, f"default width {w}, expected 6.4"
    assert abs(h - 4.0) < 0.01, f"default height {h}, expected 4.0"
    plt.close(fig)


def test_get_figure_figsize_overrides_preset() -> None:
    apply_style()
    fig, ax = get_figure(figsize=(10, 5), preset="single")
    w, h = fig.get_size_inches()
    assert abs(w - 10) < 0.01, f"figsize width {w}, expected 10"
    assert abs(h - 5) < 0.01, f"figsize height {h}, expected 5"
    plt.close(fig)


def test_get_figure_explicit_figsize() -> None:
    apply_style()
    fig, ax = get_figure(figsize=(10, 5))
    w, h = fig.get_size_inches()
    assert abs(w - 10) < 0.01
    assert abs(h - 5) < 0.01
    plt.close(fig)


def test_fig_from_ax_round_trip() -> None:
    apply_style()
    fig, ax = get_figure()
    assert isinstance(ax, Axes)
    recovered = _fig_from_ax(ax)
    assert isinstance(recovered, Figure)
    assert recovered is fig
    plt.close(fig)


def test_fig_from_ax_importable_from_helpers() -> None:
    """Verify _fig_from_ax is importable from _helpers (not just simulation_plots)."""
    from master_thesis_code.plotting._helpers import _fig_from_ax as fn

    assert callable(fn)


class TestComputeCredibleInterval:
    """Unit tests for compute_credible_interval in _helpers.py."""

    def _gaussian_posterior(
        self, h_values: npt.NDArray[np.float64], mu: float, sigma: float
    ) -> npt.NDArray[np.float64]:
        """Gaussian posterior evaluated on h_values."""
        return np.exp(-0.5 * ((h_values - mu) / sigma) ** 2).astype(np.float64)

    def test_gaussian_68ci_equals_two_sigma(self) -> None:
        """Gaussian(mu=0.73, sigma=0.04): 68% CI width should equal 2*sigma within 0.003."""
        h_values = np.linspace(0.60, 0.90, 101)
        posterior = self._gaussian_posterior(h_values, mu=0.73, sigma=0.04)
        lo, hi = compute_credible_interval(h_values, posterior, level=0.68)
        ci_width = hi - lo
        assert abs(ci_width - 2 * 0.04) < 0.003, (
            f"Expected CI width ~{2 * 0.04:.4f}, got {ci_width:.4f}"
        )

    def test_uniform_68ci_equals_68_percent_range(self) -> None:
        """Uniform posterior: 68% CI width should equal 0.68 * range within 0.01."""
        h_values = np.linspace(0.60, 0.90, 31)
        posterior = np.ones(31, dtype=np.float64)
        lo, hi = compute_credible_interval(h_values, posterior, level=0.68)
        ci_width = hi - lo
        expected = 0.68 * (0.90 - 0.60)
        assert abs(ci_width - expected) < 0.01, (
            f"Expected CI width ~{expected:.4f}, got {ci_width:.4f}"
        )

    def test_zero_posterior_returns_nan(self) -> None:
        """Zero-valued posterior should return (nan, nan)."""
        h_values = np.linspace(0.60, 0.90, 31)
        posterior = np.zeros(31, dtype=np.float64)
        lo, hi = compute_credible_interval(h_values, posterior)
        assert math.isnan(lo), f"Expected lo=nan, got {lo}"
        assert math.isnan(hi), f"Expected hi=nan, got {hi}"

    def test_returns_tuple_of_floats(self) -> None:
        """Return type should be tuple[float, float]."""
        h_values = np.linspace(0.60, 0.90, 51)
        posterior = self._gaussian_posterior(h_values, mu=0.73, sigma=0.04)
        result = compute_credible_interval(h_values, posterior)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"
        lo, hi = result
        assert isinstance(lo, float), f"Expected float lo, got {type(lo)}"
        assert isinstance(hi, float), f"Expected float hi, got {type(hi)}"

    def test_level_parameter(self) -> None:
        """level=0.95 on Gaussian(sigma=0.04): CI width should be ~2*1.96*sigma within 0.005."""
        h_values = np.linspace(0.60, 0.90, 101)
        posterior = self._gaussian_posterior(h_values, mu=0.73, sigma=0.04)
        lo, hi = compute_credible_interval(h_values, posterior, level=0.95)
        ci_width = hi - lo
        expected = 2 * 1.96 * 0.04
        assert abs(ci_width - expected) < 0.005, (
            f"Expected 95% CI width ~{expected:.4f}, got {ci_width:.4f}"
        )


class TestDivergingNorm:
    """Unit tests for diverging_norm (TwoSlopeNorm wrapper)."""

    def test_returns_twoslopenorm_with_vcenter(self) -> None:
        norm = diverging_norm(vcenter=0.73)
        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vcenter == 0.73

    def test_vcenter_stays_inside_explicit_bounds(self) -> None:
        norm = diverging_norm(vcenter=0.73, vmin=0.6, vmax=0.9)
        assert norm.vmin is not None and norm.vmax is not None
        assert norm.vmin < norm.vcenter < norm.vmax

    def test_out_of_range_vcenter_is_nudged_to_midpoint(self) -> None:
        """vcenter below vmin would crash TwoSlopeNorm; diverging_norm nudges it."""
        norm = diverging_norm(vcenter=0.73, vmin=0.8, vmax=0.9)
        assert norm.vmin is not None and norm.vmax is not None
        assert norm.vmin < norm.vcenter < norm.vmax

    def test_composes_into_midpoint_mapping(self) -> None:
        """A residual array spanning 0.6..0.86 maps vcenter 0.73 to the colormap midpoint."""
        norm = diverging_norm(vcenter=0.73, vmin=0.6, vmax=0.86)
        assert abs(float(norm(0.73)) - 0.5) < 1e-9


class TestMakeHeatmapNorm:
    """Unit tests for make_heatmap_norm (robust / log norm builder)."""

    def test_robust_uses_finite_percentiles(self) -> None:
        data = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
        norm = make_heatmap_norm(data, mode="robust", low=0.0, high=100.0)
        assert isinstance(norm, Normalize)
        assert norm.vmin is not None and norm.vmax is not None
        assert math.isclose(norm.vmin, 1.0, rel_tol=1e-9)
        assert math.isclose(norm.vmax, 5.0, rel_tol=1e-9)

    def test_robust_ignores_nan(self) -> None:
        """NaN must not poison the percentile clip."""
        data = np.array([10.0, 20.0, 30.0, np.nan])
        norm = make_heatmap_norm(data, mode="robust", low=0.0, high=100.0)
        assert norm.vmin is not None and norm.vmax is not None
        assert np.isfinite(norm.vmin) and np.isfinite(norm.vmax)

    def test_log_over_positive_values_only(self) -> None:
        norm = make_heatmap_norm(np.array([1.0, 10.0, 100.0]), mode="log")
        assert isinstance(norm, LogNorm)
        assert norm.vmin is not None and norm.vmin > 0

    def test_log_does_not_crash_on_zeros_negatives_nan(self) -> None:
        """Zeros / negatives / NaN must be masked before LogNorm (no ValueError)."""
        data = np.array([0.0, -5.0, np.nan, 2.0, 8.0])
        norm = make_heatmap_norm(data, mode="log")
        assert isinstance(norm, LogNorm)
        assert norm.vmin is not None and norm.vmin > 0
        assert norm.vmax is not None and norm.vmax >= norm.vmin

    def test_log_all_nonpositive_falls_back_to_safe_range(self) -> None:
        data = np.array([0.0, -1.0, -2.0])
        norm = make_heatmap_norm(data, mode="log")
        assert isinstance(norm, LogNorm)
        assert norm.vmin is not None and norm.vmin > 0
