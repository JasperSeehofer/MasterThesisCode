"""Unit tests for plotting/_metrics.py and the HDI helper.

Pure-function checks against analytical limits.  No simulation data
required; runs in milliseconds on the CPU-only test profile.
"""

import numpy as np
import numpy.typing as npt

from master_thesis_code.plotting._helpers import (
    compute_credible_interval,
    compute_hdi_interval,
)
from master_thesis_code.plotting._metrics import (
    bias_pct,
    effective_event_gain,
    hdi_width,
    jsd_between,
    kl_from_uniform,
    map_h,
    rel_precision,
)


def _gaussian_grid(
    mean: float = 0.73,
    sigma: float = 0.02,
    n: int = 4001,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    h = np.linspace(0.55, 0.92, n)
    p = np.exp(-0.5 * ((h - mean) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    return h, p


# ---------------------------------------------------------------------------
# HDI helper
# ---------------------------------------------------------------------------


def test_hdi_width_gaussian_one_sigma() -> None:
    """For a Gaussian, 68.3% HDI width = 2 * 0.99445 sigma."""
    sigma = 0.02
    h, p = _gaussian_grid(sigma=sigma)
    width = hdi_width(h, p, level=0.683)
    expected = 2.0 * 0.99445 * sigma
    assert abs(width - expected) < 0.01 * expected, (
        f"HDI width {width:.5f} differs from analytical {expected:.5f} by more than 1%"
    )


def test_hdi_matches_central_for_symmetric_gaussian() -> None:
    """HDI and central CI should agree on a symmetric posterior."""
    h, p = _gaussian_grid()
    lo_h, hi_h = compute_hdi_interval(h, p, level=0.683)
    lo_c, hi_c = compute_credible_interval(h, p, level=0.683)
    assert abs(lo_h - lo_c) < 1e-3
    assert abs(hi_h - hi_c) < 1e-3


def test_hdi_handles_zero_posterior() -> None:
    h = np.linspace(0.6, 0.9, 100)
    p = np.zeros_like(h)
    lo, hi = compute_hdi_interval(h, p)
    assert np.isnan(lo) and np.isnan(hi)
    assert np.isnan(hdi_width(h, p))


# ---------------------------------------------------------------------------
# Single-posterior metrics
# ---------------------------------------------------------------------------


def test_map_h_picks_mode() -> None:
    h, p = _gaussian_grid(mean=0.71)
    assert abs(map_h(h, p) - 0.71) < 1e-3


def test_rel_precision_positive_finite() -> None:
    h, p = _gaussian_grid(mean=0.73, sigma=0.02)
    rp = rel_precision(h, p)
    assert 0.0 < rp < 1.0


def test_bias_pct_zero_for_truth_centered_gaussian() -> None:
    h, p = _gaussian_grid(mean=0.73)
    b = bias_pct(h, p, h_true=0.73)
    assert abs(b) < 0.5  # within 0.5% — limited by grid spacing


def test_kl_from_uniform_zero_for_flat() -> None:
    h = np.linspace(0.6, 0.9, 200)
    p = np.ones_like(h)
    kl = kl_from_uniform(h, p)
    assert abs(kl) < 1e-10


def test_kl_from_uniform_positive_for_peaked() -> None:
    h, p = _gaussian_grid(sigma=0.01)
    kl = kl_from_uniform(h, p)
    # Narrower than the prior range -> positive information gain
    assert kl > 0.5


# ---------------------------------------------------------------------------
# Two-posterior comparison metrics
# ---------------------------------------------------------------------------


def test_jsd_self_zero() -> None:
    h, p = _gaussian_grid()
    assert jsd_between(h, p, p) < 1e-10


def test_jsd_disjoint_bounded_by_one_bit() -> None:
    """JSD in bits is bounded by 1 for any pair of distributions."""
    h = np.linspace(0.6, 0.9, 1000)
    p_left = np.exp(-0.5 * ((h - 0.65) / 0.005) ** 2)
    p_right = np.exp(-0.5 * ((h - 0.85) / 0.005) ** 2)
    j = jsd_between(h, p_left, p_right)
    assert 0.5 < j <= 1.0 + 1e-9


def test_jsd_symmetric() -> None:
    h, p_a = _gaussian_grid(mean=0.71)
    _, p_b = _gaussian_grid(mean=0.74)
    j_ab = jsd_between(h, p_a, p_b)
    j_ba = jsd_between(h, p_b, p_a)
    assert abs(j_ab - j_ba) < 1e-12


# ---------------------------------------------------------------------------
# effective_event_gain
# ---------------------------------------------------------------------------


def test_effective_event_gain_identity() -> None:
    """Two identical curves -> K = 1 everywhere."""
    sizes = np.array([10, 50, 100, 500], dtype=np.float64)
    widths = 1.0 / np.sqrt(sizes)
    K = effective_event_gain(sizes, widths, sizes, widths)
    assert np.allclose(K, 1.0, atol=1e-6)


def test_effective_event_gain_factor_two() -> None:
    """If query has 1/sqrt(2*N) scaling, K should be ~2.

    Width_with(N) == width_without(2*N) means the with-M_z analysis at N
    matches the without-M_z analysis at 2N -> K = 2.
    """
    sizes = np.array([10, 20, 50, 100, 200, 500], dtype=np.float64)
    widths_ref = 1.0 / np.sqrt(sizes)
    widths_query = 1.0 / np.sqrt(2.0 * sizes)
    K = effective_event_gain(sizes, widths_ref, sizes, widths_query)
    # K should be ~2 except at the endpoints where extrapolation is
    # impossible (returns nan).
    valid = ~np.isnan(K)
    assert valid.sum() >= 1
    assert np.allclose(K[valid], 2.0, rtol=1e-3)


def test_effective_event_gain_out_of_range_returns_nan() -> None:
    sizes = np.array([10, 50, 100], dtype=np.float64)
    widths_ref = np.array([0.3, 0.15, 0.1])
    sizes_q = np.array([10.0])
    widths_q = np.array([0.5])  # wider than anything in the reference
    K = effective_event_gain(sizes, widths_ref, sizes_q, widths_q)
    assert np.isnan(K[0])
