"""Tests for the closed-loop two-channel calibration harness (G4b / §9).

All tests are CPU-only and cheap. Tests that need the production injection pool
(the object that defines ``S_4D``) are skipped when it is absent, so the suite
runs on a clone without the results tree.
"""

import functools
import math
import os

import numpy as np
import pytest

from master_thesis_code.bayesian_inference import bayesian_statistics as bs
from master_thesis_code.validation import closed_loop_gfrac as cl

_POOL_AVAILABLE = os.path.isdir(cl.DEFAULT_INJECTION_DIR) and os.path.isfile(cl.DEFAULT_CRB_CSV)
needs_pool = pytest.mark.skipif(
    not _POOL_AVAILABLE,
    reason="production injection pool / CRB CSV not present in this checkout",
)


# ── grid + readout ───────────────────────────────────────────────────────────


def test_canonical_h_grid_is_the_production_41_point_grid() -> None:
    """The h grid must be the 41-point production grid, sorted and bracketing 0.73."""
    g = cl.CANONICAL_H_GRID
    assert len(g) == 41
    assert g[0] == pytest.approx(0.60)
    assert g[-1] == pytest.approx(0.86)
    assert all(b > a for a, b in zip(g, g[1:], strict=False))
    assert any(abs(h - 0.73) < 1e-12 for h in g)


def test_posterior_readout_recovers_a_known_peak() -> None:
    """A Gaussian log posterior must give MAP, refined MAP and mean at its centre."""
    h = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
    ln_p = -0.5 * ((h - 0.7304) / 0.02) ** 2
    out = cl.posterior_readout(h, ln_p)
    assert out["map"] == pytest.approx(0.730, abs=1e-9)
    assert out["map_refined"] == pytest.approx(0.7304, abs=2e-3)
    assert out["mean"] == pytest.approx(0.7304, abs=2e-3)
    assert out["railed_low"] == 0.0 and out["railed_high"] == 0.0


def test_posterior_readout_flags_rails() -> None:
    """A monotone log posterior must be flagged as railed at the corresponding edge."""
    h = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
    assert cl.posterior_readout(h, -100.0 * h)["railed_low"] == 1.0
    assert cl.posterior_readout(h, +100.0 * h)["railed_high"] == 1.0


def test_score_against_bands_implements_the_frozen_prereg() -> None:
    """CONFIRM / REFUTE / MIXED must follow §9's frozen bands exactly."""
    assert cl.score_against_bands(np.full(10, 0.735), 0.73)["verdict"] == "CONFIRM"
    assert cl.score_against_bands(np.full(10, 0.765), 0.73)["verdict"] == "REFUTE"
    assert cl.score_against_bands(np.full(10, 0.750), 0.73)["verdict"] == "MIXED"
    # A negative displacement of any size is MIXED, never CONFIRM-by-absolute-value.
    assert cl.score_against_bands(np.full(10, 0.700), 0.73)["verdict"] == "MIXED"


# ── g_i limiting cases (the derivation package's §5) ─────────────────────────


def test_g_i_sigma_Mz_to_zero_is_the_point_evaluation() -> None:
    r"""``sigma_cond -> 0``: ``g_i -> phi(mu_cond M_z,det/(1+z)) M_z,det/(1+z)``.

    Derivation package §5(b): a dark host's mass is never measured, so the
    Gauss-Hermite average collapses to a point read-out of the population
    density — finite and non-zero, with no ``1/sigma`` blow-up.
    """
    z = np.linspace(0.05, 0.9, 12)
    d_L_frac = np.linspace(0.9, 1.1, 12)
    det_M_z = 6.0e5
    proj = 3.0e-7
    g = cl.completion_mass_factor_g(z, d_L_frac, det_M_z, proj, 1e-14)
    scale = det_M_z / (1.0 + z)
    mu_cond = 1.0 + proj * (d_L_frac - 1.0)
    expected = bs.dark_mass_density_per_mass(mu_cond * scale) * scale
    assert np.allclose(g, expected, rtol=1e-10)
    assert np.all(g > 0.0)


def test_g_i_flat_phi_gives_the_minus_one_jacobian_slope(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Flat ``phi`` ⇒ ``g_i (1+z)/M_z,det`` constant and ``dln g/dln(1+z) = -1``.

    Derivation package §5(a) / GATEB refutation report row 5 ("toy monkeypatch:
    ``g (1+z)/M_z,det`` constant to 4e-16; ``dln g/dln(1+z) = -1`` exactly").
    This is the sign check that localises the *real* slope to phi's curvature
    rather than to the mass Jacobian.
    """
    # phi(M) = unnormalised(M) / (M ln10) / Z_phi, so a phi that is FLAT IN M
    # (the §5(a) toy) needs the log10-density to be proportional to M.
    monkeypatch.setattr(
        bs,
        "dark_mass_log10_density_unnormalised",
        lambda M: np.asarray(M, dtype=np.float64),
    )
    cache = bs._phi_dark_mass_log10_grid
    assert isinstance(cache, functools._lru_cache_wrapper)
    cache.cache_clear()
    try:
        z = np.linspace(0.05, 1.2, 25)
        det_M_z = 6.0e5
        g = cl.completion_mass_factor_g(z, np.ones_like(z), det_M_z, 0.0, 1e-14)
        reduced = g * (1.0 + z) / det_M_z
        assert np.allclose(reduced, reduced[0], rtol=1e-12)
        slope = np.gradient(np.log(g), np.log(1.0 + z))
        assert np.allclose(slope, -1.0, atol=1e-6)
    finally:
        cache.cache_clear()


def test_g_i_real_phi_slope_is_positive_above_the_kink() -> None:
    r"""Above the ``kappa_cap`` kink the real phi gives ``dln g/dln(1+z) = +0.43``.

    Derivation package §6.3 (``s_dex = -0.43``) and GATEB row 2 ("real-phi above
    break +0.43000 exactly"). Below ``M = 1e5`` the ``kappa_cap`` surrogate
    flips the local slope — this test pins that the kink is present, i.e. that
    the synthetic universe's phi is the kinked production one.
    """
    z = np.linspace(0.05, 1.0, 20)
    det_M_z = 1.0e6  # source masses stay >> 1e5 over this z range
    g = cl.completion_mass_factor_g(z, np.ones_like(z), det_M_z, 0.0, 1e-14)
    slope = np.gradient(np.log(g), np.log(1.0 + z))
    assert np.allclose(slope, 0.43, atol=2e-3)
    # And the kink is real: far below 1e5 the local slope differs.
    g_low = cl.completion_mass_factor_g(z, np.ones_like(z), 5.0e4, 0.0, 1e-14)
    slope_low = np.gradient(np.log(g_low), np.log(1.0 + z))
    assert not np.allclose(slope_low, 0.43, atol=0.1)


# ── error model ──────────────────────────────────────────────────────────────


@needs_pool
def test_sigma_triples_match_the_production_error_scale() -> None:
    """The bootstrap pool must reproduce the production fractional error scale."""
    t = cl.load_sigma_triples(cl.DEFAULT_CRB_CSV)
    assert t.shape[1] == 3
    assert t.shape[0] > 1000
    assert np.median(t[:, 0]) == pytest.approx(0.037, abs=0.01)  # sigma_dL/d_L
    assert np.median(t[:, 1]) < 1e-6  # sigma_Mz/M_z — the mass is essentially exact
    assert np.all(np.abs(t[:, 2]) < 1.0)


# ── end-to-end ───────────────────────────────────────────────────────────────


@needs_pool
def test_two_seed_smoke_completes() -> None:
    """A 2-seed, tiny-N sweep must complete and produce a scored aggregate."""
    cfg = cl.ClosedLoopConfig(n_events=60)
    res = cl.run_sweep(cfg, [cl.DEFAULT_BASE_SEED, cl.DEFAULT_BASE_SEED + 1], workers=1)
    assert res["aggregate"]["n_seeds"] == 2
    assert res["aggregate"]["scoring"]["verdict"] in {"CONFIRM", "REFUTE", "MIXED"}
    for rec in res["per_seed"]:
        assert len(rec["ln_post_1d"]) == 41
        assert len(rec["ln_post_2d"]) == 41
        assert math.isfinite(rec["map_2d"])
        assert cl.CANONICAL_H_GRID[0] <= rec["map_2d"] <= cl.CANONICAL_H_GRID[-1]


@needs_pool
def test_seed_is_deterministic() -> None:
    """The same seed must reproduce the same universe and the same MAPs."""
    cfg = cl.ClosedLoopConfig(n_events=60)
    ctx = cl.build_context(cfg)
    a = cl.run_seed(4242, ctx)
    b = cl.run_seed(4242, ctx)
    assert a["map_1d"] == b["map_1d"] and a["map_2d"] == b["map_2d"]
    assert a["ln_post_2d"] == b["ln_post_2d"]


@needs_pool
def test_known_host_redshift_control_recovers_h_true() -> None:
    """With every host redshift known exactly, both channels must land on 0.73.

    This is the harness's own plumbing check: the distance ladder, the noise
    draw, the quadrature and the ``alpha(h)`` normalisation are all exercised,
    and the only thing removed is the redshift ambiguity. A failure here means
    the harness is broken, not that the estimator is.
    """
    cfg = cl.ClosedLoopConfig(n_events=120, f_cat=1.0)
    ctx = cl.build_context(cfg)
    rec = cl.run_seed(cl.DEFAULT_BASE_SEED, ctx)
    assert rec["map_1d"] == pytest.approx(0.73, abs=1e-9)
    assert rec["map_2d"] == pytest.approx(0.73, abs=1e-9)


@needs_pool
def test_universe_follows_the_estimators_own_population() -> None:
    """The drawn universe must sit inside phi's band and exercise the kink."""
    cfg = cl.ClosedLoopConfig(n_events=400)
    ctx = cl.build_context(cfg)
    u = cl.draw_universe(ctx, np.random.default_rng(7))
    assert np.all(u.M_true >= 1.0e4) and np.all(u.M_true <= 1.0e7)
    assert np.all(u.z_true > 0.0) and np.all(u.z_true < ctx.z_max_true)
    # GATEB amendment 2: the kappa_cap kink at 1e5 must be inside the support
    # the synthetic events actually populate.
    assert np.any(u.M_true < 1.0e5)
    assert np.any(u.M_true > 1.0e5)
