"""Tests for the Route 1 adaptive Gauss-Hermite order in ``completion_mass_factor_g``.

[PHYSICS] Route 1 (2026-08-12), RATIFIED by author 2026-08-12: per-row fast
n=8 Gauss-Hermite contraction with a relative-half-width / breakpoint-straddle
fallback to the pinned n=64 convention.
"""

import math

import numpy as np
import numpy.typing as npt

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    _G_I_ADAPT_MAX_RELWIDTH,
    _G_I_ADAPT_T,
    completion_mass_factor_g,
)
from master_thesis_code.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN


def test_adaptive_matches_convention_on_narrow_sigma() -> None:
    z_nodes: npt.NDArray[np.float64] = np.linspace(0.05, 1.2, 60, dtype=np.float64)
    d_L_fraction: npt.NDArray[np.float64] = np.linspace(0.97, 1.03, 60, dtype=np.float64)
    det_M_z = 8.0e5
    proj = 0.4
    sigma_cond_M = 5e-7  # harvest-realistic

    g_adaptive = completion_mass_factor_g(z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M)
    g_convention = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M, adaptive=False
    )

    np.testing.assert_allclose(g_adaptive, g_convention, rtol=1e-12)
    assert np.all(g_adaptive > 0.0)


def test_forced_straddle_is_bit_identical_to_convention() -> None:
    sigma_cond_M = 0.12
    det_M_z = 2.0e5
    z_nodes: npt.NDArray[np.float64] = np.linspace(0.95, 1.05, 40, dtype=np.float64)
    d_L_fraction: npt.NDArray[np.float64] = np.linspace(0.99, 1.01, 40, dtype=np.float64)
    proj = 0.3

    g_adaptive = completion_mass_factor_g(z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M)
    g_convention = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M, adaptive=False
    )

    # All rows straddle the kink -> adaptive takes the identical single-group
    # n=64 path -> exact bit-for-bit equality, not just close.
    assert np.array_equal(g_adaptive, g_convention)


def test_mixed_mask_routes_rows_correctly() -> None:
    # Low-z rows: scale = det_M_z/(1+z) sits within ~1e3 of the 1e5 kink, so
    # the +-6 sigma window straddles it (fallback to n=64). High-z rows put
    # scale near 3-4e4, far from every breakpoint -> fast n=8 path. sigma is
    # kept small enough that the relative-half-width criterion never fires on
    # its own, isolating the straddle criterion.
    z_nodes: npt.NDArray[np.float64] = np.concatenate(
        [np.linspace(0.99, 1.01, 10), np.linspace(3.0, 5.0, 10)]
    ).astype(np.float64)
    d_L_fraction: npt.NDArray[np.float64] = np.linspace(0.98, 1.02, 20, dtype=np.float64)
    det_M_z = 2.0e5
    proj = 0.2
    sigma_cond_M = 0.001
    proj_arg = proj

    w = math.sqrt(2.0) * sigma_cond_M * _G_I_ADAPT_T
    scale = det_M_z / (1.0 + z_nodes)
    mu_cond = 1.0 + proj_arg * (d_L_fraction - 1.0)
    lo_bound = (mu_cond - w) * scale
    hi_bound = (mu_cond + w) * scale
    breakpoints = (M_SOURCE_FRAME_MIN, 1.0e5, M_SOURCE_FRAME_MAX)
    straddles = np.zeros_like(mu_cond, dtype=np.bool_)
    for b in breakpoints:
        straddles |= (lo_bound < b) & (b < hi_bound)
    expected_fallback = (w > _G_I_ADAPT_MAX_RELWIDTH * mu_cond) | (mu_cond <= 0.0) | straddles

    assert np.any(expected_fallback)
    assert np.any(~expected_fallback)

    g_adaptive = completion_mass_factor_g(z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M)
    g_convention = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M, adaptive=False
    )

    np.testing.assert_allclose(
        g_adaptive[expected_fallback], g_convention[expected_fallback], rtol=1e-13
    )
    np.testing.assert_allclose(
        g_adaptive[~expected_fallback], g_convention[~expected_fallback], rtol=1e-10
    )


def test_relwidth_criterion_triggers_fallback() -> None:
    sigma_cond_M = 0.01  # w = 6*sqrt(2)*0.01 ~ 0.0849 > 0.02 * mu_cond (mu_cond ~ 1)
    z_nodes: npt.NDArray[np.float64] = np.linspace(0.3, 0.7, 25, dtype=np.float64)
    d_L_fraction: npt.NDArray[np.float64] = np.linspace(0.995, 1.005, 25, dtype=np.float64)
    det_M_z = 8.0e5  # scale far from the 1e5 kink -> breakpoint-free window
    proj = 0.1

    w = math.sqrt(2.0) * sigma_cond_M * _G_I_ADAPT_T
    mu_cond = 1.0 + proj * (d_L_fraction - 1.0)
    assert np.all(w > _G_I_ADAPT_MAX_RELWIDTH * mu_cond)

    g_adaptive = completion_mass_factor_g(z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M)
    g_convention = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M, adaptive=False
    )

    assert np.array_equal(g_adaptive, g_convention)


def test_sigma_to_zero_point_evaluation() -> None:
    sigma_cond_M = 1e-14
    z_nodes: npt.NDArray[np.float64] = np.linspace(0.1, 2.0, 30, dtype=np.float64)
    d_L_fraction: npt.NDArray[np.float64] = np.linspace(0.95, 1.05, 30, dtype=np.float64)
    det_M_z = 5.0e5
    proj = 0.5

    g_adaptive = completion_mass_factor_g(z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M)
    g_convention = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M, adaptive=False
    )

    assert np.all(np.isfinite(g_adaptive))
    assert np.all(g_adaptive > 0.0)
    np.testing.assert_allclose(g_adaptive, g_convention, rtol=1e-10)


def test_explicit_n_hermite_override_bypasses_adaptive() -> None:
    z_nodes: npt.NDArray[np.float64] = np.linspace(0.1, 1.5, 20, dtype=np.float64)
    d_L_fraction: npt.NDArray[np.float64] = np.linspace(0.96, 1.04, 20, dtype=np.float64)
    det_M_z = 6.0e5
    proj = 0.25
    sigma_cond_M = 1e-6

    g_true_adaptive = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M, n_hermite=32, adaptive=True
    )
    g_false_adaptive = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M, n_hermite=32, adaptive=False
    )

    assert np.array_equal(g_true_adaptive, g_false_adaptive)


def test_fast_order_is_pinned_at_eight() -> None:
    """Adversarial-verify caveat 2: a silent fast-order change must fail loudly."""
    z_nodes: npt.NDArray[np.float64] = np.linspace(0.05, 1.2, 40, dtype=np.float64)
    d_L_fraction: npt.NDArray[np.float64] = np.linspace(0.99, 1.01, 40, dtype=np.float64)
    det_M_z = 8.0e5
    proj = 0.4
    sigma_cond_M = 5e-7

    g_adaptive = completion_mass_factor_g(z_nodes, d_L_fraction, det_M_z, proj, sigma_cond_M)
    g_explicit_8 = completion_mass_factor_g(
        z_nodes,
        d_L_fraction,
        det_M_z,
        proj,
        sigma_cond_M,
        n_hermite=8,
        adaptive=False,
    )

    assert np.array_equal(g_adaptive, g_explicit_8)
