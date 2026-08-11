"""Tests for the venue-transfer instrument (prereg VT-D0, results/venue_transfer_20260811).

All tests are CPU-only and cheap; toy contexts are pool-free. Tests needing
the pinned production inputs (CRB CSV / frozeng emit / pruned catalogue) are
skipped when absent, so the suite runs on a bare clone. The full-file V-T3
pin-integrity check is ``slow``-marked (streams a 1.7 GB CSV).

The central toy certification here is the vector-σ estimator core: with a
constant σ_z vector it must reproduce the gate's scalar ball path
BIT-IDENTICALLY (the same property V-T5 certifies against the committed v2
records, exact gate-shape mode). Chunked mode is deterministic and agrees to
O(1 ULP) — BLAS ``@`` accumulation is shape-dependent (module divergence 2).
"""

import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import roots_legendre

from master_thesis_code.physical_relations import dist_vectorized
from master_thesis_code.validation import calibration_gate as cg
from master_thesis_code.validation import closed_loop_gfrac as cl
from master_thesis_code.validation import venue_transfer as vt

_PINNED_FILES_AVAILABLE = all(
    os.path.isfile(p) for p in (vt.CRB_CSV_PATH, vt.FROZENG_EMIT_JSON, vt.PRUNED_CATALOGUE_CSV)
)

needs_pins = pytest.mark.skipif(
    not _PINNED_FILES_AVAILABLE,
    reason="pinned production inputs not present in this checkout",
)


# ── Seed plan (VT-D7) ────────────────────────────────────────────────────────


def test_seed_blocks_match_prereg_and_are_disjoint() -> None:
    """Base 20260808, +40000-decade offsets, a seed in exactly one cell."""
    assert vt.VT_BASE_SEED == 20260808
    assert vt.CELL_SPECS["T0"].seed_offsets == (40000,)
    assert vt.CELL_SPECS["Ta"].seed_offsets == (41000,)
    assert vt.CELL_SPECS["Tb"].seed_offsets == (42000,)
    assert vt.CELL_SPECS["Tc"].seed_offsets == (43000, 44000, 45000)
    assert vt.CELL_SPECS["Tc"].n_seeds == (200, 400, 200)
    all_seeds: set[int] = set()
    for spec in vt.CELL_SPECS.values():
        for i, t in enumerate(spec.truths):
            block = vt.venue_cell_seeds(spec, t, 0, None)
            assert len(block) == spec.n_seeds[i]
            assert not (all_seeds & set(block))
            all_seeds.update(block)
    # Everything inside the registered v3 envelope.
    lo = vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[0]
    hi = vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[1]
    assert all(lo <= s <= hi for s in all_seeds)


def test_v3_seeds_disjoint_from_v1_and_v2_envelopes() -> None:
    """No v3 seed falls in v1 (+[0,9049]) or v2 (+[20000,29049]) envelopes."""
    envelopes = (vt.V1_SEED_OFFSET_ENVELOPE, vt.V2_SEED_OFFSET_ENVELOPE)
    for spec in vt.CELL_SPECS.values():
        for t in spec.truths:
            for s in vt.venue_cell_seeds(spec, t, 0, None):
                for lo, hi in envelopes:
                    assert not (vt.VT_BASE_SEED + lo <= s <= vt.VT_BASE_SEED + hi)
    # The v2 offsets themselves (incl. reserved O1 at +28000) sit inside the
    # declared v2 envelope this test guards against.
    for spec2 in cg.CELL_SPECS.values():
        for off in spec2.seed_offsets:
            assert vt.V2_SEED_OFFSET_ENVELOPE[0] <= off <= vt.V2_SEED_OFFSET_ENVELOPE[1]


def test_reserved_blocks_disjoint_and_not_in_cells() -> None:
    """W1/O2 reserved blocks are outside every built cell's seed range."""
    reserved: set[int] = set()
    for lo, hi in vt.RESERVED_SEED_OFFSET_BLOCKS.values():
        block = set(range(vt.VT_BASE_SEED + lo, vt.VT_BASE_SEED + hi + 1))
        assert not (reserved & block)
        reserved.update(block)
    for spec in vt.CELL_SPECS.values():
        for t in spec.truths:
            assert not (reserved & set(vt.venue_cell_seeds(spec, t, 0, None)))


def test_venue_cell_seeds_chunking() -> None:
    """--seed-range chunks tile the block without overlap and reject overruns."""
    spec = vt.CELL_SPECS["Tc"]
    a = vt.venue_cell_seeds(spec, 0.730, 0, 100)
    b = vt.venue_cell_seeds(spec, 0.730, 100, 300)
    assert a + b == vt.venue_cell_seeds(spec, 0.730, 0, None)
    with pytest.raises(ValueError):
        vt.venue_cell_seeds(spec, 0.730, 300, 200)
    # Per-truth block sizes differ (0.690: 200) — chunk bound respects them.
    with pytest.raises(ValueError):
        vt.venue_cell_seeds(spec, 0.690, 100, 150)


def test_cell_matrix_matches_prereg_table() -> None:
    """Cell matrix (§5): ball modes, σ_z modes, truths, prereg spellings."""
    assert vt.CELL_SPECS["T0"].balls == "real_k" and vt.CELL_SPECS["T0"].sigma_mode == "zero"
    assert vt.CELL_SPECS["Ta"].balls == "poisson4" and vt.CELL_SPECS["Ta"].sigma_mode == "flat035"
    assert vt.CELL_SPECS["Tb"].balls == "real_k" and vt.CELL_SPECS["Tb"].sigma_mode == "flat035"
    assert vt.CELL_SPECS["Tc"].balls == "real_k" and vt.CELL_SPECS["Tc"].sigma_mode == "glade"
    assert vt.CELL_SPECS["Tc"].truths == (0.690, 0.730, 0.770)
    for name, prereg in (("T0", "T-0"), ("Ta", "T-a"), ("Tb", "T-b"), ("Tc", "T-c")):
        assert vt.CELL_SPECS[name].prereg_cell == prereg
    cfg = vt.VenueConfig(cell="Tc", h_true=0.730, balls="real_k", sigma_mode="glade")
    assert cfg.h_grid == cl.CANONICAL_H_GRID
    assert cfg.flat_sigma_z == 0.035 and cfg.lambda_poisson == 4.0


def test_registered_pin_constants_match_prereg_literals() -> None:
    """Module pins carry the prereg §V-T3 literals (md5s, census, σ_z stats)."""
    assert vt.CRB_CSV_MD5 == "9a1f2a14384a9281c97ca3be312ddaab"
    assert vt.FROZENG_EMIT_MD5 == "34c50e91028b6a6458a2b145db545705"
    assert vt.K_CENSUS_PINS["n_events_evaluated"] == 1588
    assert vt.K_CENSUS_PINS["zeros"] == 606
    assert vt.K_CENSUS_PINS["nonempty_n"] == 982
    assert vt.K_CENSUS_PINS["sum_K"] == 1_193_703
    assert vt.K_CENSUS_PINS["max"] == 245_364
    # Full-precision pins are consistent with the prereg's printed truncations
    # (module docstring divergence 3).
    assert abs(float(vt.K_CENSUS_PINS["mean"]) - 751.702) < 1e-3
    assert abs(float(vt.K_CENSUS_PINS["p99"]) - 11325.26) < 5e-3
    assert abs(float(vt.K_CENSUS_PINS["nonempty_mean"]) - 1215.58) < 5e-3
    assert vt.SIGMA_PINS["n"] == 20_834_171
    assert vt.SIGMA_PINS["median"] == float("0.0393412950539589")
    assert vt.SIGMA_PINS["min"] == float("0.0005317263419419")
    assert vt.SIGMA_PINS["n_lt_5e-3"] == 231_098
    assert vt.R_DOSE_BAND == (0.75, 1.25)


# ── Census / stats helpers ───────────────────────────────────────────────────


def test_k_census_on_synthetic_counts() -> None:
    """k_census computes the VT-D2 statistic set on a known array."""
    K = np.asarray([0, 0, 1, 2, 4, 10], dtype=np.int64)
    c = vt.k_census(K)
    assert c["n_events_evaluated"] == 6
    assert c["zeros"] == 2 and c["ones"] == 1
    assert c["sum_K"] == 17 and c["max"] == 10
    assert c["nonempty_n"] == 4
    assert c["nonempty_median"] == pytest.approx(3.0)


def test_sigma_stats_on_synthetic_sample() -> None:
    """sigma_stats counts the spec-z-like tail correctly."""
    sz = np.asarray([1e-4, 4e-3, 6e-3, 0.02, 0.04], dtype=np.float64)
    s = vt.sigma_stats(sz)
    assert s["n"] == 5
    assert s["n_lt_5e-3"] == 2
    assert s["n_lt_1e-2"] == 3
    assert s["min"] == pytest.approx(1e-4)
    assert s["median"] == pytest.approx(6e-3)


# ── σ_z sampler (VT-D3) ──────────────────────────────────────────────────────


def _toy_sampler_context(n_cat: int = 1000) -> vt.VenueContext:
    """A VenueContext with only sampler tables (decile machinery tests)."""
    rng = np.random.default_rng(0)
    z = rng.uniform(0.0, 1.0, size=n_cat)
    # sigma encodes its own decile: sigma = decile + fractional noise in (0, 0.5).
    order = np.argsort(np.argsort(z, kind="stable"), kind="stable")
    dec_true = (10 * order) // n_cat
    sz = dec_true + rng.uniform(0.0, 0.5, size=n_cat)
    edges, pools = vt.build_sigma_sampler(z, sz)
    return vt.VenueContext(
        vcfg=vt.VenueConfig(cell="custom", h_true=0.730, balls="real_k", sigma_mode="glade"),
        gctx=None,  # type: ignore[arg-type]  # sampler tests never touch gctx
        event_rows=np.zeros(0, dtype=np.int64),
        d_L=np.zeros(0),
        M_row=np.zeros(0),
        sigma_dL=np.zeros(0),
        sigma_Mz=np.zeros(0),
        rho=np.zeros(0),
        z_true=np.zeros(0),
        K=np.zeros(0, dtype=np.int64),
        n_horizon_dropped=0,
        z_decile_edges=edges,
        sigma_pool_deciles=pools,
    )


def test_sigma_sampler_pools_partition_the_frame() -> None:
    """10 pools of n/10 each; pool b holds exactly decile b's sigmas."""
    vctx = _toy_sampler_context(1000)
    assert len(vctx.sigma_pool_deciles) == 10
    assert all(p.size == 100 for p in vctx.sigma_pool_deciles)
    for b, pool in enumerate(vctx.sigma_pool_deciles):
        assert np.all((pool >= b) & (pool < b + 0.5))


def test_sigma_sampler_draws_are_decile_matched() -> None:
    """A member's z decile determines its σ pool (σ encodes the decile)."""
    vctx = _toy_sampler_context(1000)
    rng = np.random.default_rng(7)
    z_members = np.asarray([0.05, 0.15, 0.55, 0.95, 0.31])
    draws = vt.draw_member_sigma_z(vctx, z_members, rng)
    # Expected deciles from the edges themselves.
    expected = np.searchsorted(vctx.z_decile_edges, z_members, side="right")
    assert np.all(np.floor(draws).astype(int) == expected)


def test_sigma_sampler_z_beyond_last_edge_maps_to_top_decile() -> None:
    """Impostors above the frame's z range draw from decile 9 (divergence 8)."""
    vctx = _toy_sampler_context(1000)
    rng = np.random.default_rng(8)
    draws = vt.draw_member_sigma_z(vctx, np.asarray([2.5, 5.0]), rng)
    assert np.all(np.floor(draws).astype(int) == 9)


def test_sigma_sampler_is_deterministic() -> None:
    """Same seed and member order => identical draws (V-T2 ingredient)."""
    vctx = _toy_sampler_context(1000)
    z_members = np.random.default_rng(3).uniform(0.0, 1.2, size=200)
    a = vt.draw_member_sigma_z(vctx, z_members, np.random.default_rng(11))
    b = vt.draw_member_sigma_z(vctx, z_members, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)


def test_sigma_sampler_requires_tables() -> None:
    """A context without pools refuses to draw (guards non-glade misuse)."""
    vctx = _toy_sampler_context(1000)
    vctx.sigma_pool_deciles = []
    with pytest.raises(RuntimeError, match="sampler tables"):
        vt.draw_member_sigma_z(vctx, np.asarray([0.5]), np.random.default_rng(0))


# ── Pinned-K ball generator (VT-D2) ──────────────────────────────────────────


def _fake_ball_context(K: npt.NDArray[np.int64], n_events: int) -> vt.VenueContext:
    """A VenueContext over the toy ladder of the gate's ball unit tests."""
    gcfg = cg.GateConfig(
        cell="custom", h_true=0.730, ball=True, lambda_ball=0.0, sigma_z=0.0, n_events=n_events
    )
    z = np.linspace(1e-6, 2.0, 2000)
    dl = 4000.0 * z * (1.0 + 0.5 * z)  # monotone toy ladder [Mpc]
    w = z**2 / (1.0 + z)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(z))])
    cdf /= cdf[-1]
    gctx = cg.GateContext(
        gate_config=gcfg,
        cl_ctx=None,  # type: ignore[arg-type]  # draw_ball_pinned never touches cl_ctx
        csv_dl_sorted=np.asarray([1.0]),
        triples=np.asarray([[0.04, 1e-8, 0.0]]),
        decile_rows=[np.asarray([0], dtype=np.int64)] * 10,
        imp_z_nodes=z,
        imp_z_cdf=cdf,
        imp_dl_nodes=dl,
    )
    return vt.VenueContext(
        vcfg=vt.VenueConfig(cell="custom", h_true=0.730, balls="real_k", sigma_mode="zero"),
        gctx=gctx,
        event_rows=np.arange(n_events, dtype=np.int64),
        d_L=np.zeros(n_events),
        M_row=np.full(n_events, 5.0e5),
        sigma_dL=np.full(n_events, 0.04),
        sigma_Mz=np.full(n_events, 1e-8),
        rho=np.zeros(n_events),
        z_true=np.zeros(n_events),
        K=K,
        n_horizon_dropped=0,
    )


def _toy_universe(n: int, rng: np.random.Generator) -> cl.SyntheticUniverse:
    """A small noiseless universe on the toy ladder of the fake context."""
    z = rng.uniform(0.1, 0.8, size=n)
    d_L = 4000.0 * z * (1.0 + 0.5 * z)
    M = np.full(n, 5.0e5)
    return cl.SyntheticUniverse(
        z_true=z,
        M_true=M,
        d_L_true=d_L,
        d_L_obs=d_L.copy(),
        M_z_obs=M * (1.0 + z),
        sigma_dL=np.full(n, 0.04),
        sigma_Mz=np.full(n, 1e-8),
        rho=np.zeros(n),
        in_catalogue=np.zeros(n, dtype=bool),
        n_drawn=n,
    )


def test_draw_ball_pinned_respects_pinned_multiplicity() -> None:
    """Each event's ball has exactly K_i members; the host z is one of them."""
    rng = np.random.default_rng(5)
    K = np.asarray([1, 2, 5, 100, 3], dtype=np.int64)
    vctx = _fake_ball_context(K, 5)
    uni = _toy_universe(5, rng)
    ball = vt.draw_ball_pinned(vctx, uni, rng)
    np.testing.assert_array_equal(ball.K, K)
    assert ball.n_impostors_total == int(np.sum(K - 1))
    for i in range(5):
        members = ball.z_obs[ball.event_idx == i]
        assert members.size == K[i]
        assert np.any(np.isclose(members, uni.z_true[i], atol=1e-12))


def test_draw_ball_pinned_applies_no_sigma() -> None:
    """z_obs are TRUE member redshifts — the σ_z texture is the caller's step."""
    rng = np.random.default_rng(6)
    K = np.full(20, 4, dtype=np.int64)
    vctx = _fake_ball_context(K, 20)
    uni = _toy_universe(20, rng)
    ball = vt.draw_ball_pinned(vctx, uni, rng)
    # Every host appears exactly (impostors are new draws, hosts untouched).
    for i in range(20):
        members = ball.z_obs[ball.event_idx == i]
        assert np.any(members == uni.z_true[i])


def test_draw_ball_pinned_impostors_lie_in_window() -> None:
    """Impostors respect W_i = [z(d_obs(1-4s)), z(d_obs(1+4s))] on the ladder."""
    rng = np.random.default_rng(9)
    K = np.full(30, 50, dtype=np.int64)
    vctx = _fake_ball_context(K, 30)
    uni = _toy_universe(30, rng)
    ball = vt.draw_ball_pinned(vctx, uni, rng)
    gctx = vctx.gctx
    d_lo = uni.d_L_obs * (1.0 - 4.0 * uni.sigma_dL)
    d_hi = uni.d_L_obs * (1.0 + 4.0 * uni.sigma_dL)
    z_lo = np.interp(np.maximum(d_lo, 0.0), gctx.imp_dl_nodes, gctx.imp_z_nodes)
    z_hi = np.interp(d_hi, gctx.imp_dl_nodes, gctx.imp_z_nodes)
    for i in range(30):
        members = ball.z_obs[ball.event_idx == i]
        imp = members[~np.isclose(members, uni.z_true[i], atol=1e-12)]
        assert np.all(imp >= z_lo[i] - 1e-9) and np.all(imp <= z_hi[i] + 1e-9)


def test_draw_ball_pinned_is_deterministic() -> None:
    """Same seed => identical balls (V-T2 ingredient)."""
    K = np.asarray([3, 7, 1, 12], dtype=np.int64)
    vctx = _fake_ball_context(K, 4)
    uni = _toy_universe(4, np.random.default_rng(2))
    b1 = vt.draw_ball_pinned(vctx, uni, np.random.default_rng(13))
    b2 = vt.draw_ball_pinned(vctx, uni, np.random.default_rng(13))
    np.testing.assert_array_equal(b1.z_obs, b2.z_obs)
    np.testing.assert_array_equal(b1.event_idx, b2.event_idx)
    np.testing.assert_array_equal(b1.K, b2.K)


def test_draw_ball_pinned_degenerate_window_keeps_host_only() -> None:
    """A window with zero population mass yields a host-only ball (counted)."""
    K = np.asarray([50], dtype=np.int64)
    vctx = _fake_ball_context(K, 1)
    rng = np.random.default_rng(4)
    uni = _toy_universe(1, rng)
    uni.d_L_obs[:] = 1e9  # far beyond the toy ladder: F_hi == F_lo == 1
    ball = vt.draw_ball_pinned(vctx, uni, rng)
    assert ball.n_degenerate_windows == 1
    assert ball.K[0] == 1
    assert ball.n_impostors_total == 0


# ── Pair chunking + capped g evaluation ──────────────────────────────────────


def test_pair_chunks_tile_the_range() -> None:
    """Chunks tile [0, n_pairs) contiguously; degenerate cases behave."""
    chunks = vt._pair_chunks(12, 5)
    assert chunks == [(0, 5), (5, 10), (10, 12)]
    assert vt._pair_chunks(12, 0) == [(0, 12)]
    assert vt._pair_chunks(0, 4) == []


def test_g_ball_capped_matches_gate_g_ball() -> None:
    """Exact mode == the gate's _g_ball bit-for-bit; tiny caps agree to ULPs."""
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=10)
    rng = np.random.default_rng(43)
    uni = _real_ladder_universe(10, rng)
    uni.sigma_Mz[:] = 0.05  # give g a nontrivial conditional width
    ball = cg.draw_ball(gctx, uni, rng)
    n_pairs = ball.z_obs.size
    z_nodes = np.clip(ball.z_obs[:, None] + np.linspace(-0.02, 0.02, 7)[None, :], 1e-6, 1.4)
    d_L_frac = (
        np.asarray(dist_vectorized(z_nodes.reshape(-1), h=0.73)).reshape(z_nodes.shape)
        / uni.d_L_obs[ball.event_idx][:, None]
    )
    valid = np.ones(n_pairs, dtype=bool)
    valid[::5] = False  # exercise the invalid-row zeroing too
    ref = cg._g_ball(gctx, uni, ball.event_idx, z_nodes, d_L_frac, valid)
    # Exact gate-shape mode (node_chunk <= 0): bit-identical.
    out_exact = vt._g_ball_capped(gctx, uni, ball.event_idx, z_nodes, d_L_frac, valid, node_chunk=0)
    np.testing.assert_array_equal(out_exact, ref)
    # A generous finite cap that never splits these small events: bit-identical.
    out_big = vt._g_ball_capped(
        gctx, uni, ball.event_idx, z_nodes, d_L_frac, valid, node_chunk=200_000
    )
    np.testing.assert_array_equal(out_big, ref)
    # A tiny cap splits events: deterministic, and equal to O(1 ULP) (BLAS
    # shape-dependent accumulation — module divergence 2).
    out_tiny = vt._g_ball_capped(gctx, uni, ball.event_idx, z_nodes, d_L_frac, valid, node_chunk=13)
    np.testing.assert_allclose(out_tiny, ref, rtol=1e-13, atol=0.0)
    out_tiny2 = vt._g_ball_capped(
        gctx, uni, ball.event_idx, z_nodes, d_L_frac, valid, node_chunk=13
    )
    np.testing.assert_array_equal(out_tiny, out_tiny2)


# ── Vector-σ estimator core (the V-T5 toy certification) ─────────────────────


def _real_ladder_context(sigma_z: float, lambda_ball: float, n_events: int) -> cg.GateContext:
    """GateContext on the REAL flat-ΛCDM ladder, no injection pool needed."""
    gcfg = cg.GateConfig(
        cell="custom",
        h_true=0.730,
        ball=True,
        lambda_ball=lambda_ball,
        sigma_z=sigma_z,
        n_events=n_events,
    )
    cl_cfg = cg.to_closed_loop_config(gcfg)
    z_max = 1.5
    tables = [cl._z_of_dl_table(h, z_max) for h in cl_cfg.h_grid]
    gl_nodes, gl_weights = roots_legendre(cl_cfg.n_quad)
    cl_ctx = cl.ClosedLoopContext(
        config=cl_cfg,
        detection=None,  # type: ignore[arg-type]  # not used by the ball path
        sigma_triples=np.asarray([[0.02, 1e-8, 0.0]]),
        z_max_true=z_max,
        gen_z_nodes=np.linspace(1e-6, z_max, 100),
        gen_z_cdf=np.linspace(0.0, 1.0, 100),
        gen_log10_M_nodes=np.linspace(4.0, 7.0, 100),
        gen_M_cdf=np.linspace(0.0, 1.0, 100),
        z_of_dl_tables=tables,
        log_alpha=np.zeros(len(cl_cfg.h_grid)),
        s_phi_tables=[],
        gl_nodes=np.asarray(gl_nodes, dtype=np.float64),
        gl_weights=np.asarray(gl_weights, dtype=np.float64),
    )
    z_nodes = np.linspace(1e-6, 3.0, 3000)
    dl_nodes = np.asarray(dist_vectorized(z_nodes, h=0.730), dtype=np.float64)
    w = cl._w_pop(z_nodes, 0.730)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(z_nodes))])
    cdf /= cdf[-1]
    return cg.GateContext(
        gate_config=gcfg,
        cl_ctx=cl_ctx,
        csv_dl_sorted=np.asarray([1.0]),
        triples=np.asarray([[0.02, 1e-8, 0.0]]),
        decile_rows=[np.asarray([0], dtype=np.int64)] * 10,
        imp_z_nodes=z_nodes,
        imp_z_cdf=cdf,
        imp_dl_nodes=dl_nodes,
    )


def _real_ladder_universe(n: int, rng: np.random.Generator) -> cl.SyntheticUniverse:
    """Noiseless events on the real ladder at h_true = 0.730."""
    z = rng.uniform(0.15, 0.9, size=n)
    d_L = np.asarray(dist_vectorized(z, h=0.730), dtype=np.float64)
    M = np.full(n, 5.0e5)
    return cl.SyntheticUniverse(
        z_true=z,
        M_true=M,
        d_L_true=d_L,
        d_L_obs=d_L.copy(),
        M_z_obs=M * (1.0 + z),
        sigma_dL=np.full(n, 0.02),
        sigma_Mz=np.full(n, 1e-8),
        rho=np.zeros(n),
        in_catalogue=np.zeros(n, dtype=bool),
        n_drawn=n,
    )


def test_vector_core_constant_sigma_bit_reproduces_gate_scalar_path() -> None:
    """Constant σ vector == the gate's scalar σ_z ball path, bit-identical.

    This is the toy form of the V-T5 no-drift certification: the vector-σ
    generalization must BE the committed gate math when every candidate
    shares one σ.
    """
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=25)
    rng = np.random.default_rng(17)
    uni = _real_ladder_universe(25, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    ln1_g, ln2_g, slope_g = cg.log_channel_posteriors_ball(gctx, uni, ball)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)
    ln1_v, ln2_v, slope_v = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs
    )
    np.testing.assert_array_equal(ln1_v, ln1_g)
    np.testing.assert_array_equal(ln2_v, ln2_g)
    np.testing.assert_array_equal(slope_v, slope_g)


def test_vector_core_zero_sigma_bit_reproduces_gate_point_path() -> None:
    """All-zero σ vector == the gate's σ_z = 0 point-evaluation path."""
    gctx = _real_ladder_context(sigma_z=0.0, lambda_ball=4.0, n_events=25)
    rng = np.random.default_rng(19)
    uni = _real_ladder_universe(25, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    ln1_g, ln2_g, slope_g = cg.log_channel_posteriors_ball(gctx, uni, ball)
    sigma_pairs = np.zeros(ball.z_obs.size, dtype=np.float64)
    ln1_v, ln2_v, slope_v = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs
    )
    np.testing.assert_array_equal(ln1_v, ln1_g)
    np.testing.assert_array_equal(ln2_v, ln2_g)
    np.testing.assert_array_equal(slope_v, slope_g)


def test_vector_core_chunking_is_deterministic_and_ulp_close() -> None:
    """Chunked mode: bit-deterministic per geometry, ULP-close across geometries."""
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=20)
    rng = np.random.default_rng(23)
    uni = _real_ladder_universe(20, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)
    ln1_a, ln2_a, sl_a = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, chunk_pairs=0
    )
    ln1_b, ln2_b, sl_b = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, chunk_pairs=3
    )
    ln1_c, ln2_c, _sl_c = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, sigma_pairs, chunk_pairs=3
    )
    # Same geometry => bit-identical (V-T2 determinism).
    np.testing.assert_array_equal(ln1_b, ln1_c)
    np.testing.assert_array_equal(ln2_b, ln2_c)
    # Different geometry => O(1 ULP) agreement (module divergence 2) and the
    # same grid argmax (no readout-level effect).
    np.testing.assert_allclose(ln1_a, ln1_b, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(ln2_a, ln2_b, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(sl_a, sl_b, rtol=1e-9, atol=1e-9)
    assert int(np.argmax(ln1_a)) == int(np.argmax(ln1_b))
    assert int(np.argmax(ln2_a)) == int(np.argmax(ln2_b))


def test_vector_core_heterogeneous_sigma_changes_the_answer() -> None:
    """A genuinely heterogeneous σ vector differs from any flat approximation."""
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=15)
    rng = np.random.default_rng(29)
    uni = _real_ladder_universe(15, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    het = np.where(np.arange(ball.z_obs.size) % 2 == 0, 0.005, 0.06).astype(np.float64)
    ln1_h, ln2_h, _ = vt.log_channel_posteriors_ball_sigma_vector(gctx, uni, ball, het)
    ln1_f, ln2_f, _ = vt.log_channel_posteriors_ball_sigma_vector(
        gctx, uni, ball, np.full(ball.z_obs.size, float(np.mean(het)))
    )
    assert np.all(np.isfinite(ln1_h)) and np.all(np.isfinite(ln2_h))
    assert not np.array_equal(ln1_h, ln1_f)


def test_vector_core_mixed_zero_and_positive_sigma_composes_exactly() -> None:
    """Mixed σ (spec-z-like zero rows + photo-z rows) = sum of the pure paths.

    With log_alpha = 0, ``ln P = Σ_i ln L_i``, so a two-event universe with
    one σ = 0 event and one σ > 0 event must equal the sum of the two
    single-event runs bit-for-bit.
    """
    gctx = _real_ladder_context(sigma_z=0.0, lambda_ball=0.0, n_events=2)
    rng = np.random.default_rng(31)
    uni = _real_ladder_universe(2, rng)
    ball = vt.draw_ball_pinned(
        _fake_venue_context_from_gate(gctx, np.asarray([3, 3], dtype=np.int64)), uni, rng
    )
    sigma = np.where(ball.event_idx == 0, 0.0, 0.03)
    ln1_mix, ln2_mix, _ = vt.log_channel_posteriors_ball_sigma_vector(gctx, uni, ball, sigma)

    def _single(i: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mask = ball.event_idx == i
        sub_uni = cl.SyntheticUniverse(
            z_true=uni.z_true[i : i + 1],
            M_true=uni.M_true[i : i + 1],
            d_L_true=uni.d_L_true[i : i + 1],
            d_L_obs=uni.d_L_obs[i : i + 1],
            M_z_obs=uni.M_z_obs[i : i + 1],
            sigma_dL=uni.sigma_dL[i : i + 1],
            sigma_Mz=uni.sigma_Mz[i : i + 1],
            rho=uni.rho[i : i + 1],
            in_catalogue=uni.in_catalogue[i : i + 1],
            n_drawn=1,
        )
        sub_ball = cg.HostBall(
            z_obs=ball.z_obs[mask],
            event_idx=np.zeros(int(mask.sum()), dtype=np.int64),
            K=ball.K[i : i + 1],
            n_impostors_total=int(mask.sum()) - 1,
            n_degenerate_windows=0,
        )
        a, b, _ = vt.log_channel_posteriors_ball_sigma_vector(gctx, sub_uni, sub_ball, sigma[mask])
        return a, b

    ln1_0, ln2_0 = _single(0)
    ln1_1, ln2_1 = _single(1)
    np.testing.assert_array_equal(ln1_mix, ln1_0 + ln1_1)
    np.testing.assert_array_equal(ln2_mix, ln2_0 + ln2_1)


def _fake_venue_context_from_gate(
    gctx: cg.GateContext, K: npt.NDArray[np.int64]
) -> vt.VenueContext:
    """Wrap a gate context into a minimal VenueContext for ball draws."""
    n = K.size
    return vt.VenueContext(
        vcfg=vt.VenueConfig(cell="custom", h_true=0.730, balls="real_k", sigma_mode="zero"),
        gctx=gctx,
        event_rows=np.arange(n, dtype=np.int64),
        d_L=np.zeros(n),
        M_row=np.full(n, 5.0e5),
        sigma_dL=np.full(n, 0.02),
        sigma_Mz=np.full(n, 1e-8),
        rho=np.zeros(n),
        z_true=np.zeros(n),
        K=K,
        n_horizon_dropped=0,
    )


def test_vector_core_rejects_shape_mismatch() -> None:
    """A σ vector not aligned with the pairs raises immediately."""
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=5)
    rng = np.random.default_rng(37)
    uni = _real_ladder_universe(5, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    with pytest.raises(ValueError, match="shape"):
        vt.log_channel_posteriors_ball_sigma_vector(gctx, uni, ball, np.zeros(ball.z_obs.size + 1))


# ── Per-seed driver on a toy pinned context ──────────────────────────────────


def _toy_pinned_context(sigma_mode: str, n: int = 12) -> vt.VenueContext:
    """A full toy VenueContext: real ladder, pinned rows, small K, sampler."""
    gctx = _real_ladder_context(sigma_z=0.0, lambda_ball=0.0, n_events=n)
    rng = np.random.default_rng(41)
    z_true = rng.uniform(0.15, 0.9, size=n)
    d_L = np.asarray(dist_vectorized(z_true, h=0.730), dtype=np.float64)
    K = rng.integers(1, 8, size=n).astype(np.int64)
    z_cat = np.random.default_rng(0).uniform(0.0, 1.5, size=2000)
    sz_cat = np.random.default_rng(1).uniform(0.001, 0.08, size=2000)
    edges, pools = vt.build_sigma_sampler(z_cat, sz_cat)
    return vt.VenueContext(
        vcfg=vt.VenueConfig(cell="custom", h_true=0.730, balls="real_k", sigma_mode=sigma_mode),
        gctx=gctx,
        event_rows=np.arange(n, dtype=np.int64),
        d_L=d_L,
        M_row=np.full(n, 5.0e5),
        sigma_dL=np.full(n, 0.02),
        sigma_Mz=np.full(n, 1e-8),
        rho=np.zeros(n),
        z_true=z_true,
        K=K,
        n_horizon_dropped=0,
        z_decile_edges=edges,
        sigma_pool_deciles=pools,
    )


def test_run_seed_venue_record_schema_and_json_safety() -> None:
    """The §6 record carries the gate fields + the venue fields, JSON-safe."""
    required = {
        "seed", "cell", "h_true", "balls", "sigma_mode", "f_incl", "n_events",
        "n_events_run", "n_horizon_dropped", "z_median", "M_source_median",
        "frac_below_kink", "K_mean", "K_sum", "n_impostors_total",
        "n_degenerate_windows", "texture_corr", "sigma_z_mean_pairs",
        "sigma_z_median_pairs", "frac_pairs_sigma_lt_5e-3", "map_1d", "map_2d",
        "map_1d_refined", "map_2d_refined", "mean_1d", "mean_2d",
        "railed_low_1d", "railed_high_1d", "railed_low_2d", "railed_high_2d",
        "sum_dlog_gfrac_dh", "pit_1d", "pit_2d", "hpd50_1d", "hpd68_1d",
        "hpd90_1d", "hpd50_2d", "hpd68_2d", "hpd90_2d", "post_sd_1d",
        "post_sd_2d", "edge_mass_1d", "edge_mass_2d", "ln_post_1d", "ln_post_2d",
    }  # fmt: skip
    vctx = _toy_pinned_context("glade")
    rec = vt.run_seed_venue(12345, vctx)
    assert required <= set(rec)
    json.dumps(rec)
    assert rec["K_sum"] == int(np.sum(vctx.K))
    assert len(rec["ln_post_1d"]) == 41


def test_run_seed_venue_sigma_modes_dose_statistics() -> None:
    """Realized dose stats: 0 for T-0, exactly 0.035 for flat, empirical for glade."""
    rec0 = vt.run_seed_venue(100, _toy_pinned_context("zero"))
    assert rec0["sigma_z_mean_pairs"] == 0.0
    assert rec0["frac_pairs_sigma_lt_5e-3"] == 1.0
    recf = vt.run_seed_venue(100, _toy_pinned_context("flat035"))
    assert recf["sigma_z_mean_pairs"] == pytest.approx(0.035, abs=1e-15)
    assert recf["frac_pairs_sigma_lt_5e-3"] == 0.0
    recg = vt.run_seed_venue(100, _toy_pinned_context("glade"))
    assert 0.001 <= recg["sigma_z_mean_pairs"] <= 0.08
    assert recg["sigma_z_median_pairs"] > 0.0


def test_run_seed_venue_is_deterministic() -> None:
    """V-T2: same seed, same context => bit-identical record (glade path)."""
    vctx = _toy_pinned_context("glade")
    a = vt.run_seed_venue(777, vctx)
    b = vt.run_seed_venue(777, vctx)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_run_seed_venue_t0_anchor_shape_maps_near_truth() -> None:
    """σ_z = 0 with exact hosts: both channels' MAP at the injected truth."""
    vctx = _toy_pinned_context("zero")
    rec = vt.run_seed_venue(555, vctx)
    # 12 events with 2% dL noise and exact host members => MAP within a couple
    # of grid steps of the injected truth (a shape check, not a DS band).
    assert abs(rec["map_1d"] - 0.730) <= 0.02
    assert abs(rec["map_2d"] - 0.730) <= 0.02


# ── h-grain parallel mode (divergence 11) ────────────────────────────────────


def _toy_mode_context(
    balls: str,
    sigma_mode: str,
    *,
    chunk_pairs: int = vt.DEFAULT_CHUNK_PAIRS,
    k_hi: int = 8,
    n: int = 12,
) -> vt.VenueContext:
    """A toy VenueContext for any (balls, sigma_mode) venue cell type."""
    if balls == "poisson4":
        gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=n)
    else:
        gctx = _real_ladder_context(sigma_z=0.0, lambda_ball=0.0, n_events=n)
    rng = np.random.default_rng(41)
    z_true = rng.uniform(0.15, 0.9, size=n)
    d_L = np.asarray(dist_vectorized(z_true, h=0.730), dtype=np.float64)
    K = rng.integers(1, k_hi, size=n).astype(np.int64)
    z_cat = np.random.default_rng(0).uniform(0.0, 1.5, size=2000)
    sz_cat = np.random.default_rng(1).uniform(0.001, 0.08, size=2000)
    edges, pools = vt.build_sigma_sampler(z_cat, sz_cat)
    return vt.VenueContext(
        vcfg=vt.VenueConfig(
            cell="custom",
            h_true=0.730,
            balls=balls,
            sigma_mode=sigma_mode,
            chunk_pairs=chunk_pairs,
        ),
        gctx=gctx,
        event_rows=np.arange(n, dtype=np.int64),
        d_L=d_L,
        M_row=np.full(n, 5.0e5),
        sigma_dL=np.full(n, 0.02),
        sigma_Mz=np.full(n, 1e-8),
        rho=np.zeros(n),
        z_true=z_true,
        K=K,
        n_horizon_dropped=0,
        z_decile_edges=edges,
        sigma_pool_deciles=pools,
    )


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize(
    ("balls", "sigma_mode"),
    [
        ("real_k", "zero"),  # T-0
        ("real_k", "flat035"),  # T-b
        ("real_k", "glade"),  # T-c (maximal path)
        ("poisson4", "flat035"),  # T-a (the gate's draw_ball verbatim)
    ],
)
def test_hgrain_record_byte_identical_to_seed_grain(
    balls: str, sigma_mode: str, workers: int
) -> None:
    """Divergence 11 acceptance bar: same (seed, cell) => byte-identical
    per-seed record for any worker count, across every venue cell type."""
    vctx = _toy_mode_context(balls, sigma_mode)
    ref = vt.run_seed_venue(4242, vctx)
    par = vt.run_seed_venue_hgrain(4242, vctx, workers=workers)
    assert json.dumps(par, sort_keys=True) == json.dumps(ref, sort_keys=True)


def test_hgrain_real_k_capped_multichunk_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-K capped case: multi-chunk pair partition + split g calls engaged
    (the campaign's chunked geometry, scaled down) — still byte-identical,
    and identical across worker counts."""
    monkeypatch.setattr(vt, "_G_NODE_CHUNK", 40)  # < n_quad=50 => per-row g splits
    vctx = _toy_mode_context("real_k", "glade", chunk_pairs=7, k_hi=30)
    assert len(vt._pair_chunks(int(np.sum(vctx.K)), 7)) > 1  # chunking engaged
    ref = json.dumps(vt.run_seed_venue(31415, vctx), sort_keys=True)
    par2 = json.dumps(vt.run_seed_venue_hgrain(31415, vctx, workers=2), sort_keys=True)
    par5 = json.dumps(vt.run_seed_venue_hgrain(31415, vctx, workers=5), sort_keys=True)
    assert par2 == ref
    assert par5 == ref


def test_hgrain_is_deterministic() -> None:
    """V-T2 carried to the new mode: same seed twice => bit-identical."""
    vctx = _toy_mode_context("real_k", "glade")
    a = vt.run_seed_venue_hgrain(777, vctx, workers=3)
    b = vt.run_seed_venue_hgrain(777, vctx, workers=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_hgrain_estimator_bitwise_matches_serial() -> None:
    """The h-grain estimator twin equals the serial loop bit-for-bit."""
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=20)
    rng = np.random.default_rng(53)
    uni = _real_ladder_universe(20, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    sigma_pairs = np.full(ball.z_obs.size, 0.035, dtype=np.float64)
    a1, a2, asl = vt.log_channel_posteriors_ball_sigma_vector(gctx, uni, ball, sigma_pairs)
    b1, b2, bsl = vt.log_channel_posteriors_ball_sigma_vector_hgrain(
        gctx, uni, ball, sigma_pairs, workers=3
    )
    np.testing.assert_array_equal(a1, b1)
    np.testing.assert_array_equal(a2, b2)
    np.testing.assert_array_equal(asl, bsl)


def test_hgrain_estimator_rejects_shape_mismatch() -> None:
    """The h-grain twin carries the serial function's alignment guard."""
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=5)
    rng = np.random.default_rng(37)
    uni = _real_ladder_universe(5, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    with pytest.raises(ValueError, match="shape"):
        vt.log_channel_posteriors_ball_sigma_vector_hgrain(
            gctx, uni, ball, np.zeros(ball.z_obs.size + 1)
        )


def test_run_cell_venue_grain_dispatch_and_equality(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_cell_venue: grain='h' per-seed records + aggregate equal grain='seed';
    unknown grains are refused; the mode is recorded in the document."""
    toy = _toy_mode_context("real_k", "glade")
    monkeypatch.setattr(vt, "build_venue_context", lambda vcfg, check_pins=True: toy)
    monkeypatch.setattr(cg, "_git_state", lambda: ("abc", {"import_path": [], "other": []}))
    try:
        vt._VCTX = None
        doc_seed = vt.run_cell_venue(toy.vcfg, [11, 12], 1, grain="seed")
        vt._VCTX = None
        doc_h = vt.run_cell_venue(toy.vcfg, [11, 12], 3, grain="h")
        with pytest.raises(ValueError, match="grain"):
            vt.run_cell_venue(toy.vcfg, [11], 1, grain="event")
    finally:
        vt._VCTX = None
    assert doc_seed["grain"] == "seed"
    assert doc_h["grain"] == "h"
    assert json.dumps(doc_seed["per_seed"], sort_keys=True) == json.dumps(
        doc_h["per_seed"], sort_keys=True
    )
    assert json.dumps(doc_seed["aggregate"], sort_keys=True) == json.dumps(
        doc_h["aggregate"], sort_keys=True
    )


def test_cli_grain_flag_default_and_choices() -> None:
    """--grain defaults to the registered 'seed' mode; bad values are refused."""
    assert vt.build_parser().parse_args(["--cell", "Tc"]).grain == "seed"
    assert vt.build_parser().parse_args(["--cell", "Tc", "--grain", "h"]).grain == "h"
    with pytest.raises(SystemExit):
        vt.build_parser().parse_args(["--cell", "Tc", "--grain", "event"])


# ── Classification (DS-VT bands) ─────────────────────────────────────────────


def _fake_channel_aggregate(
    c90: float,
    r_low: float,
    r_high: float,
    bias: float,
    *,
    ds1_inside: bool = False,
    ds2_pass: bool = False,
) -> dict[str, Any]:
    """A minimal channel-aggregate block for classification tests."""
    cov = {}
    for lv in ("hpd50", "hpd68", "hpd90"):
        cov[lv] = {"value": c90, "inside_2sigma": ds1_inside, "inside_3sigma": ds1_inside}
    return {
        "ds1_coverage": cov,
        "ds2_ks": {"status": "PASS" if ds2_pass else "FAIL"},
        "ds3_map_bias": {"bias": bias},
        "ds4_rails": {"railed_low_frac": r_low, "railed_high_frac": r_high},
    }


def test_classify_collapse_reproduced() -> None:
    """v2-like pattern (C90=0, no rails, bias ~ +σ̄) at N=400 => COLLAPSE-REPRODUCED."""
    ch = _fake_channel_aggregate(0.0, 0.0, 0.0, 0.040)
    out = vt.classify_channel(ch, 400, 0.040, degenerate_exempt=False)
    assert out["label"] == "COLLAPSE-REPRODUCED"
    assert out["r_dose"] == pytest.approx(1.0)
    assert out["collapse_band"] == 0.02
    assert not out["rail_emergent"]


def test_classify_collapse_needs_r_dose_in_band() -> None:
    """Same pattern but R_dose outside [0.75, 1.25] => OTHER."""
    ch = _fake_channel_aggregate(0.0, 0.0, 0.0, 0.080)
    out = vt.classify_channel(ch, 400, 0.040, degenerate_exempt=False)
    assert out["r_dose"] == pytest.approx(2.0)
    assert out["label"] == "OTHER"


def test_classify_calibrated() -> None:
    """DS-1 inside 3σ + DS-2 PASS + |bias|<=0.010 + no rails => CALIBRATED."""
    ch = _fake_channel_aggregate(0.90, 0.0, 0.0, 0.004, ds1_inside=True, ds2_pass=True)
    out = vt.classify_channel(ch, 400, 0.040, degenerate_exempt=False)
    assert out["label"] == "CALIBRATED"


def test_classify_rail_emergent_flags_and_other() -> None:
    """A rail fraction >= 0.90 flags RAIL-EMERGENT (forces MIXED at readout)."""
    ch = _fake_channel_aggregate(0.0, 0.95, 0.0, 0.040)
    out = vt.classify_channel(ch, 400, 0.040, degenerate_exempt=False)
    assert out["rail_emergent"]
    assert out["label"] == "OTHER"  # rails outside the collapse band


def test_classify_collapse_band_tracks_n() -> None:
    """Bands: 0.02 at N=400, 0.04 at N=200, 0.08 at N=100 (prereg locked)."""
    ch = _fake_channel_aggregate(0.03, 0.03, 0.0, 0.040)
    assert vt.classify_channel(ch, 400, 0.04, degenerate_exempt=False)["label"] == "OTHER"
    assert (
        vt.classify_channel(ch, 200, 0.04, degenerate_exempt=False)["label"]
        == "COLLAPSE-REPRODUCED"
    )
    assert vt.classify_channel(ch, 100, 0.04, degenerate_exempt=False)["collapse_band"] == 0.08


def test_classify_degenerate_exempt_label() -> None:
    """T-0 is never classified on DS-1/DS-2 (VT-D8 exemption carried)."""
    ch = _fake_channel_aggregate(0.0, 0.0, 0.0, 0.0)
    out = vt.classify_channel(ch, 200, 0.0, degenerate_exempt=True)
    assert out["label"] == "DS1-DS2-EXEMPT"
    assert np.isnan(out["r_dose"])


def test_vt1_anchor_statuses() -> None:
    """V-T1 edges: PASS / ANCHOR-MARGINAL / HARD-TRIGGER at the locked bands."""
    ok = vt._vt1_anchor(_fake_channel_aggregate(0.0, 0.01, 0.0, 0.005))
    assert ok["status"] == "PASS"
    marg = vt._vt1_anchor(_fake_channel_aggregate(0.0, 0.0, 0.0, 0.02))
    assert marg["status"] == "ANCHOR-MARGINAL"
    hard_bias = vt._vt1_anchor(_fake_channel_aggregate(0.0, 0.0, 0.0, 0.031))
    assert hard_bias["status"] == "HARD-TRIGGER"
    hard_rail = vt._vt1_anchor(_fake_channel_aggregate(0.0, 0.06, 0.0, 0.0))
    assert hard_rail["status"] == "HARD-TRIGGER"


def test_aggregate_venue_structure_and_t0_anchor_block() -> None:
    """aggregate_venue emits both channels, classifications, dose, VT-1 block."""
    vctx = _toy_pinned_context("zero")
    records = [vt.run_seed_venue(s, vctx) for s in range(3)]
    vcfg = vt.VenueConfig(cell="T0", h_true=0.730, balls="real_k", sigma_mode="zero")
    agg = vt.aggregate_venue(records, vcfg)
    assert agg["headline_channel"] == "1d"
    assert agg["prereg_cell"] == "T-0"
    assert agg["ds1_ds2_degenerate_pit_exempt"]
    assert "vt1_anchor" in agg
    assert agg["vt1_anchor"]["channel_1d"]["status"] in (
        "PASS",
        "ANCHOR-MARGINAL",
        "HARD-TRIGGER",
    )
    assert agg["classification_1d"]["label"] == "DS1-DS2-EXEMPT"
    assert agg["dose"]["sigma_bar_pairs"] == 0.0
    json.dumps(agg)


# ── CLI guards (V-T4 + divergences 9 / reserved arms) ────────────────────────


def test_main_rejects_allow_dirty_on_registered_cells() -> None:
    """--allow-dirty without --smoke/--validate is refused outright (V-T4)."""
    with pytest.raises(SystemExit, match="--smoke or --validate"):
        vt.main(["--cell", "Tc", "--allow-dirty"])


def test_main_rejects_n_events_cap_on_registered_cells() -> None:
    """--n-events-cap without --smoke/--validate is refused (divergence 9)."""
    with pytest.raises(SystemExit, match="n-events-cap"):
        vt.main(["--cell", "Tc", "--n-events-cap", "50"])


def test_main_refuses_reserved_arms() -> None:
    """W1/O2 are reserved seed blocks, NOT built — the CLI says so."""
    with pytest.raises(SystemExit, match="NOT built"):
        vt.main(["--cell", "W1"])
    with pytest.raises(SystemExit, match="NOT built"):
        vt.main(["--cell", "O2"])


def test_main_rejects_off_registry_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truth outside the cell's registered set is refused."""
    monkeypatch.setattr(
        cg, "_enforce_clean_import_path", lambda allow: ("x", {"import_path": [], "other": []})
    )
    with pytest.raises(SystemExit, match="registered set"):
        vt.main(["--cell", "Tb", "--truth", "0.700"])


def test_run_cell_venue_refuses_dirty_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The imported gate clean rule guards run_cell_venue (V-T4 clause 1)."""
    monkeypatch.setattr(
        cg,
        "_git_state",
        lambda: ("abc", {"import_path": ["?? master_thesis_code/x.py"], "other": []}),
    )
    vcfg = vt.VenueConfig(cell="Tc", h_true=0.730, balls="real_k", sigma_mode="glade")
    with pytest.raises(SystemExit, match="import path"):
        vt.run_cell_venue(vcfg, [1], 1, allow_dirty=False)


def test_out_path_guard_is_wired() -> None:
    """The gate's production-directory guard protects venue outputs too."""
    with pytest.raises(SystemExit):
        cg._guard_out_path("results/run_20260804_frozeng/iiib/venue.json")
    cg._guard_out_path("results/venue_transfer_20260811/T0_results.json")  # must not raise


# ── Horizon guard (VT-D1 abort (d)) ──────────────────────────────────────────


def test_horizon_guard_fires_on_excess_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """> 5 % of pinned events beyond 0.999 x dl_max => SystemExit (abort (d))."""
    n_rows = 20

    def _fake_rows(csv_path: str) -> tuple[Any, ...]:
        d_L = np.linspace(1.0, 20.0, n_rows)
        ones = np.full(n_rows, 0.02)
        return d_L, np.full(n_rows, 1e6), ones, np.full(n_rows, 1e-8), np.zeros(n_rows)

    def _fake_k(path: str) -> tuple[Any, Any]:
        return np.arange(n_rows, dtype=np.int64), np.full(n_rows, 3, dtype=np.int64)

    class _FakeDet:
        def get_dl_max(self, h: float) -> float:
            return 10.0  # half the events beyond the horizon

    class _FakeClCtx:
        detection = _FakeDet()

    class _FakeGateCtx:
        cl_ctx = _FakeClCtx()

    monkeypatch.setattr(vt, "_load_pinned_rows", _fake_rows)
    monkeypatch.setattr(vt, "load_pinned_k", _fake_k)
    monkeypatch.setattr(cg, "build_gate_context", lambda gcfg: _FakeGateCtx())
    vcfg = vt.VenueConfig(cell="Tc", h_true=0.730, balls="real_k", sigma_mode="zero")
    with pytest.raises(SystemExit, match="horizon-drop guard"):
        vt.build_venue_context(vcfg, check_pins=False)


# ── V-T3 pin integrity on the real pinned files (slow) ───────────────────────


@needs_pins
@pytest.mark.slow
def test_v_t3_pin_integrity_on_real_files() -> None:
    """The registered md5 / census / σ_z pins all reproduce (V-T3)."""
    vcfg = vt.VenueConfig(cell="Tc", h_true=0.730, balls="real_k", sigma_mode="glade")
    block = vt.check_pin_integrity(vcfg)
    assert block["crb_csv_md5"]["match"], block["crb_csv_md5"]
    assert block["frozeng_emit_md5"]["match"], block["frozeng_emit_md5"]
    assert all(block["k_census"]["match"].values()), block["k_census"]["match"]
    assert all(block["sigma_stats"]["match"].values()), block["sigma_stats"]["match"]
    assert block["pass"]
