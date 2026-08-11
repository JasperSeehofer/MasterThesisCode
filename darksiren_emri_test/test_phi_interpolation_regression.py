"""Regression + equivalence tests for the phi(M) interpolation swap.

[PHYSICS] Track B perf branch (2026-08-12), RATIFIED by author 2026-08-12.

The dark-host mass density ``phi(M)`` (``dark_mass_density_per_mass``,
``bayesian_statistics.py``) is an exact piecewise power law on the Babak band:
``dn/dlog10 M ~ M^-0.3`` (Eq. 5) x ``R0 ~ M^-0.19`` (Eq. 23) x
``Gamma ~ M^+0.06`` (Eqs. 26-27, min-cap never binds: max ratio 0.1253 < 1)
x ``kappa`` (Eq. 30 surrogate: ``M^0.5`` below ``M_turn = 1e5``, 1 above).
Hence ``log10 phi`` is exactly affine in ``log10 M`` on each side of the
single kink at ``log10 M = 5``, and linear interpolation of ``ln phi`` in
``log10 M`` off a kink-aligned grid is analytically EXACT — the only residual
is the log/exp/lerp floating-point roundtrip, O(few ULP).

Pins below were generated from the pre-swap exact-chain code on branch
``perf/realistic-venue`` (parent commit of the [PHYSICS] swap commit); the
swap must reproduce them, so the diff is regression-visible.

References:
    Babak et al. (2017), arXiv:1703.09722, Eqs. (5), (23), (26)-(27), (30),
    (31)x(34).
"""

import functools
import math

import numpy as np
import numpy.typing as npt
import pytest

from darksiren_emri.bayesian_inference import bayesian_statistics as bs
from darksiren_emri.bayesian_inference.bayesian_statistics import (
    _phi_dark_mass_log10_grid,
    completion_mass_factor_g,
    dark_mass_density_per_mass,
)
from darksiren_emri.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from darksiren_emri.dark_siren_injection import dark_mass_log10_density_unnormalised
from darksiren_emri.emri_rate import duty_cycle_Gamma, kappa_cap

# Old-code exact values of dark_mass_density_per_mass (M_sun^-1), pinned
# 2026-08-12 pre-swap (see module docstring). Column 1: M [M_sun].
_PINS: list[tuple[float, float]] = [
    (1.00000000000000e4, 2.06028919839687905e-05),
    (3.16227766016838e4, 7.06200768180058898e-06),
    (9.99900000000000e4, 2.42085388186384599e-06),
    (1.00000000000000e5, 2.42062874166481597e-06),  # the kappa_cap kink
    (1.00010000000000e5, 2.42028263380716473e-06),
    (1.00000000000000e6, 8.99348854093638078e-08),
    (3.00000000000000e6, 1.86915273861576530e-08),
    (9.99000000000000e6, 3.34618183648661988e-09),
    (1.00000000000000e7, 3.34139782543959458e-09),
]


def _exact_chain_density(M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The pre-swap definition, re-composed verbatim from its parts."""
    _, _, _, Z_phi = _phi_dark_mass_log10_grid()
    inside = (M >= M_SOURCE_FRAME_MIN) & (M <= M_SOURCE_FRAME_MAX)
    safe = np.where(inside, M, M_SOURCE_FRAME_MIN)
    density = dark_mass_log10_density_unnormalised(safe) / (safe * math.log(10.0)) / Z_phi
    return np.asarray(np.where(inside, density, 0.0), dtype=np.float64)


def test_regression_pins_old_exact_values() -> None:
    M = np.array([m for m, _ in _PINS], dtype=np.float64)
    expected = np.array([v for _, v in _PINS], dtype=np.float64)
    np.testing.assert_allclose(dark_mass_density_per_mass(M), expected, rtol=1e-12)
    np.testing.assert_allclose(dark_mass_density_per_mass(M, exact=True), expected, rtol=1e-12)


def test_off_band_is_exactly_zero() -> None:
    off = np.array([0.0, 9.9e3, 1.01e7, 1.0e12], dtype=np.float64)
    assert np.all(dark_mass_density_per_mass(off) == 0.0)


def test_interp_matches_exact_chain_on_dense_log_uniform_sample() -> None:
    rng = np.random.default_rng(20260812)
    log10_M = rng.uniform(
        math.log10(M_SOURCE_FRAME_MIN), math.log10(M_SOURCE_FRAME_MAX), size=200_000
    )
    M = 10.0**log10_M
    np.testing.assert_allclose(
        dark_mass_density_per_mass(M), _exact_chain_density(M), rtol=5e-13, atol=0.0
    )


def test_kink_neighbourhood_is_resolved() -> None:
    """The interpolation grid must place a node exactly on the kappa_cap kink."""
    M_turn = 1.0e5
    eps = np.spacing(M_turn)
    M = np.array([M_turn - eps, M_turn, M_turn + eps], dtype=np.float64)
    np.testing.assert_allclose(
        dark_mass_density_per_mass(M), _exact_chain_density(M), rtol=5e-13, atol=0.0
    )


def test_affinity_premise_still_holds() -> None:
    """Guard: if emri_rate ever stops being piecewise power-law (or the Gamma
    min-cap starts binding, or the kink moves off M = 1e5), the analytic
    exactness argument of the swap is void — this test is the tripwire."""
    M_band = np.logspace(math.log10(M_SOURCE_FRAME_MIN), math.log10(M_SOURCE_FRAME_MAX), 2001)
    assert float(np.max(duty_cycle_Gamma(M_band))) < 1.0
    assert kappa_cap(np.array([1.0e5 - 1e-6]))[0] < 1.0 == kappa_cap(np.array([1.0e5]))[0]
    for lo, hi in [(4.0, 5.0), (5.0, 7.0)]:
        x = np.linspace(lo, hi, 501)
        y = np.log10(dark_mass_log10_density_unnormalised(10.0**x))
        assert float(np.max(np.abs(np.diff(y, 2)))) < 1e-12


def test_completion_mass_factor_g_is_unchanged() -> None:
    """End-to-end through the only production consumer (g_i contraction)."""
    z_nodes = np.linspace(0.05, 1.2, 40)
    d_L_fraction = np.linspace(0.7, 1.3, 40)
    g = completion_mass_factor_g(
        z_nodes, d_L_fraction, det_M_z=8.0e5, proj_d_L_to_M=0.4, sigma_cond_M=0.12
    )
    x_nodes_w = np.polynomial.hermite.hermgauss(64)
    scale = 8.0e5 / (1.0 + z_nodes)
    mu_cond = 1.0 + 0.4 * (d_L_fraction - 1.0)
    x_M = mu_cond[:, None] + math.sqrt(2.0) * 0.12 * x_nodes_w[0][None, :]
    M_source = x_M * scale[:, None]
    phi_x = _exact_chain_density(np.asarray(M_source, dtype=np.float64)) * scale[:, None]
    g_exact = (phi_x @ x_nodes_w[1]) / math.sqrt(math.pi)
    np.testing.assert_allclose(g, g_exact, rtol=1e-12)
    assert np.all(g >= 0.0)


def test_patch_seam_tracks_the_underlying_density(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two caches must be cleared together, or the patch seam is invisible.

    Precedent: ``test_closed_loop_gfrac.py::
    test_g_i_flat_phi_gives_the_minus_one_jacobian_slope`` — see the
    :func:`~darksiren_emri.bayesian_inference.bayesian_statistics.
    _phi_ln_dark_mass_affine_coeffs` docstring note. Monkeypatching
    ``dark_mass_log10_density_unnormalised`` to a flat-in-M density and
    clearing BOTH ``_phi_dark_mass_log10_grid`` and
    ``_phi_ln_dark_mass_affine_coeffs`` must make the default (interpolated)
    path track the ``exact=True`` path again, and must move the value away
    from the unpatched pin at ``M = 1e6``.
    """
    grid_cache = bs._phi_dark_mass_log10_grid
    affine_cache = bs._phi_ln_dark_mass_affine_coeffs
    assert isinstance(grid_cache, functools._lru_cache_wrapper)
    assert isinstance(affine_cache, functools._lru_cache_wrapper)
    monkeypatch.setattr(
        bs,
        "dark_mass_log10_density_unnormalised",
        lambda M: np.asarray(M, dtype=np.float64),
    )
    grid_cache.cache_clear()
    affine_cache.cache_clear()
    try:
        M = np.geomspace(M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX, 37)
        interp = bs.dark_mass_density_per_mass(M)
        exact = bs.dark_mass_density_per_mass(M, exact=True)
        np.testing.assert_allclose(interp, exact, rtol=1e-12)

        patched_at_1e6 = float(bs.dark_mass_density_per_mass(np.array([1.0e6]))[0])
        pinned_at_1e6 = next(v for m, v in _PINS if m == 1.00000000000000e6)
        assert abs(patched_at_1e6 - pinned_at_1e6) > 1e-3 * abs(pinned_at_1e6)
    finally:
        grid_cache.cache_clear()
        affine_cache.cache_clear()


def test_normalisation_is_preserved() -> None:
    """INTEGRAL phi(M) dM = 1 on the band, same as the exact chain."""
    log10_M, M_grid, _, _ = _phi_dark_mass_log10_grid()
    integral = float(
        np.trapezoid(dark_mass_density_per_mass(M_grid) * M_grid * math.log(10.0), log10_M)
    )
    assert integral == pytest.approx(1.0, rel=1e-9)
