"""Tests for the calibration-gate instrument (prereg b50ccc65, stage-4 leg 1).

All tests are CPU-only and cheap. Tests needing the production injection pool
or the CRB CSV are skipped when absent, so the suite runs on a bare clone.

V2 (prereg §10) lives here: the ``hpd_contains`` port must agree
boolean-exactly with ``pp_coverage._hpd_contains`` on 1000 random synthetic
posteriors — ``pp_coverage`` is imported in this TEST only, never by the
runtime module.
"""

import json
import os

import numpy as np
import pytest

from master_thesis_code.validation import calibration_gate as cg
from master_thesis_code.validation import closed_loop_gfrac as cl
from master_thesis_code.validation import pp_coverage as pp

_POOL_AVAILABLE = os.path.isdir(cl.DEFAULT_INJECTION_DIR) and os.path.isfile(cl.DEFAULT_CRB_CSV)
_CSV_AVAILABLE = os.path.isfile(cl.DEFAULT_CRB_CSV)
_R0_AVAILABLE = os.path.isfile(cg.R0_RESULTS_JSON)

needs_pool = pytest.mark.skipif(
    not _POOL_AVAILABLE,
    reason="production injection pool / CRB CSV not present in this checkout",
)
needs_csv = pytest.mark.skipif(not _CSV_AVAILABLE, reason="production CRB CSV not present")
needs_r0 = pytest.mark.skipif(not _R0_AVAILABLE, reason="committed R0 results JSON not present")


# ── V2 — HPD port certification (prereg §10) ─────────────────────────────────


def test_v2_hpd_port_agrees_boolean_exactly_with_pp_coverage() -> None:
    """1000 random synthetic posteriors: port must match the original exactly."""
    rng = np.random.default_rng(20260808)
    h = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
    for _ in range(1000):
        mu = rng.uniform(0.58, 0.88)
        sd = rng.uniform(0.005, 0.15)
        post = np.exp(-0.5 * ((h - mu) / sd) ** 2)
        if rng.random() < 0.3:  # sprinkle bimodal / railed shapes
            mu2 = rng.uniform(0.58, 0.88)
            post = post + rng.uniform(0.1, 1.0) * np.exp(-0.5 * ((h - mu2) / sd) ** 2)
        post /= np.trapezoid(post, h)
        h_true = float(rng.uniform(0.60, 0.86))
        level = float(rng.choice([0.50, 0.68, 0.90]))
        assert cg.hpd_contains(h, post, h_true, level) == pp._hpd_contains(h, post, h_true, level)


# ── P–P readout layer ────────────────────────────────────────────────────────


def test_pp_readout_symmetric_gaussian_center() -> None:
    """Truth at the posterior centre: PIT ~ 0.5, all HPDs contain, tiny edge mass."""
    h = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
    ln_post = -0.5 * ((h - 0.730) / 0.02) ** 2
    out = cg.pp_readout(h, ln_post, 0.730)
    assert out["pit"] == pytest.approx(0.5, abs=0.02)
    assert out["hpd50"] == 1.0 and out["hpd68"] == 1.0 and out["hpd90"] == 1.0
    assert out["post_sd"] == pytest.approx(0.02, rel=0.1)
    assert out["edge_mass"] < 1e-6


def test_pp_readout_truth_in_far_tail() -> None:
    """Truth far in the tail: PIT near an extreme, 50% HPD excludes it."""
    h = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
    ln_post = -0.5 * ((h - 0.650) / 0.01) ** 2
    out = cg.pp_readout(h, ln_post, 0.850)
    assert out["pit"] > 0.999
    assert out["hpd50"] == 0.0 and out["hpd90"] == 0.0


def test_pp_readout_railed_posterior_has_edge_mass() -> None:
    """A low-railed posterior concentrates mass in the first grid interval."""
    h = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
    ln_post = -300.0 * (h - h[0])
    out = cg.pp_readout(h, ln_post, 0.730)
    assert out["edge_mass"] > 0.5
    assert out["pit"] > 0.99


def test_pp_readout_nonfinite_input_returns_nans() -> None:
    """Non-finite ln_post must give NaN outputs, not raise (abort-b counting)."""
    h = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
    ln_post = np.full(h.size, -np.inf)
    ln_post[0] = np.nan
    out = cg.pp_readout(h, ln_post, 0.730)
    assert all(np.isnan(v) for v in out.values())


def test_ks_distance_known_values() -> None:
    """KS distance: exact for a degenerate sample, ~0 for a perfect grid."""
    # All mass at 0.5: D = 0.5 exactly.
    assert cg.ks_distance(np.full(1000, 0.5)) == pytest.approx(0.5, abs=1e-9)
    # Perfect uniform grid (midpoints): D = 1/(2n).
    n = 400
    q = (np.arange(n) + 0.5) / n
    assert cg.ks_distance(q) == pytest.approx(0.5 / n, abs=1e-12)


# ── Cell registry / seed plan (prereg §5) ────────────────────────────────────


def test_cell_seed_blocks_match_prereg_and_are_disjoint() -> None:
    """Seed blocks: base 20260808, registered offsets, a seed in exactly one cell."""
    assert cg.GATE_BASE_SEED == 20260808
    assert cg.CELL_SPECS["A"].seed_offsets == (0, 1000, 2000)
    assert cg.CELL_SPECS["B0"].seed_offsets == (3000,)
    assert cg.CELL_SPECS["B1"].seed_offsets == (4000,)
    assert cg.CELL_SPECS["B2"].seed_offsets == (5000, 6000, 7000)
    assert cg.CELL_SPECS["V1"].seed_offsets == (9000,)
    assert cg.CELL_SPECS["V1"].n_seeds == 50
    all_seeds: set[int] = set()
    for spec in cg.CELL_SPECS.values():
        for t in spec.truths:
            block = cg.cell_seeds(spec, t, 0, None)
            assert len(block) == spec.n_seeds
            assert not (all_seeds & set(block))
            all_seeds.update(block)


def test_cell_seeds_chunking() -> None:
    """--seed-range chunks tile the block without overlap and reject overruns."""
    spec = cg.CELL_SPECS["B2"]
    a = cg.cell_seeds(spec, 0.730, 0, 100)
    b = cg.cell_seeds(spec, 0.730, 100, 300)
    assert a + b == cg.cell_seeds(spec, 0.730, 0, None)
    with pytest.raises(ValueError):
        cg.cell_seeds(spec, 0.730, 300, 200)


def test_cell_configs_match_prereg_table() -> None:
    """Cell matrix (§5): ball flags, sigma_z doses, lambda, texture."""
    assert not cg.CELL_SPECS["A"].ball
    assert cg.CELL_SPECS["B0"].sigma_z == 0.0
    assert cg.CELL_SPECS["B1"].sigma_z == 0.010
    assert cg.CELL_SPECS["B2"].sigma_z == 0.035
    for name in ("B0", "B1", "B2"):
        assert cg.CELL_SPECS[name].ball
        assert cg.CELL_SPECS[name].lambda_ball == 4.0
    assert cg.CELL_SPECS["V1"].ball and cg.CELL_SPECS["V1"].lambda_ball == 0.0
    for spec in cg.CELL_SPECS.values():
        assert spec.sigma_texture == "dl_binned"


def test_to_closed_loop_config_pins_shipped_estimator_convention() -> None:
    """f_cat=0 and numerator_pdet=off are pinned regardless of the cell."""
    gcfg = cg.GateConfig(cell="B2", h_true=0.69, ball=True, lambda_ball=4.0, sigma_z=0.035)
    cl_cfg = cg.to_closed_loop_config(gcfg)
    assert cl_cfg.f_cat == 0.0
    assert cl_cfg.numerator_pdet == "off"
    assert cl_cfg.h_true == 0.69
    assert cl_cfg.h_grid == cl.CANONICAL_H_GRID


# ── σ–d_L texture loader ─────────────────────────────────────────────────────


@needs_csv
def test_texture_loader_triples_match_parent_exactly() -> None:
    """load_sigma_triples_with_dl must apply the parent's row filter verbatim."""
    parent = cl.load_sigma_triples(cl.DEFAULT_CRB_CSV)
    triples, d_L = cg.load_sigma_triples_with_dl(cl.DEFAULT_CRB_CSV)
    np.testing.assert_array_equal(parent, triples)
    assert d_L.shape == (triples.shape[0],)
    assert np.all(d_L > 0.0)


# ── Ball generator (pool-free, fake context) ─────────────────────────────────


def _fake_gate_context(sigma_z: float, lambda_ball: float, n_events: int = 50) -> cg.GateContext:
    """A GateContext with analytic tables; enough for draw_ball unit tests."""
    gcfg = cg.GateConfig(
        cell="custom",
        h_true=0.730,
        ball=True,
        lambda_ball=lambda_ball,
        sigma_z=sigma_z,
        n_events=n_events,
    )
    z = np.linspace(1e-6, 2.0, 2000)
    dl = 4000.0 * z * (1.0 + 0.5 * z)  # monotone toy ladder [Mpc]
    w = z**2 / (1.0 + z)  # smooth toy population density
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(z))])
    cdf /= cdf[-1]
    return cg.GateContext(
        gate_config=gcfg,
        cl_ctx=None,  # type: ignore[arg-type]  # draw_ball never touches cl_ctx
        csv_dl_sorted=np.asarray([1.0]),
        triples=np.asarray([[0.04, 1e-8, 0.0]]),
        decile_rows=[np.asarray([0], dtype=np.int64)] * 10,
        imp_z_nodes=z,
        imp_z_cdf=cdf,
        imp_dl_nodes=dl,
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


def test_draw_ball_lambda_zero_sigma_zero_is_exact_hosts() -> None:
    """V1 plumbing shape: lambda=0, sigma_z=0 => every ball is the exact host."""
    gctx = _fake_gate_context(sigma_z=0.0, lambda_ball=0.0)
    rng = np.random.default_rng(1)
    uni = _toy_universe(50, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    assert np.all(ball.K == 1)
    assert ball.n_impostors_total == 0
    np.testing.assert_array_equal(ball.event_idx, np.arange(50))
    np.testing.assert_array_equal(ball.z_obs, uni.z_true)


def test_draw_ball_poisson_counts_and_membership() -> None:
    """lambda=4: K = 1 + Poisson impostors, host z present, impostors in-window."""
    gctx = _fake_gate_context(sigma_z=0.0, lambda_ball=4.0)
    rng = np.random.default_rng(2)
    uni = _toy_universe(200, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    assert np.all(ball.K >= 1)
    assert ball.n_impostors_total == int(np.sum(ball.K - 1))
    # Poisson(4) mean within 4 sigma of 4 for 200 events.
    assert abs(float(np.mean(ball.K - 1)) - 4.0) < 4.0 * np.sqrt(4.0 / 200)
    assert np.all(np.diff(ball.event_idx) >= 0)  # nondecreasing (grouped)
    # sigma_z = 0: the true host z appears verbatim in every event's ball.
    for i in range(200):
        members = ball.z_obs[ball.event_idx == i]
        assert np.any(np.isclose(members, uni.z_true[i], atol=1e-12))


def test_draw_ball_is_deterministic() -> None:
    """Same seed => identical balls (V3 ingredient)."""
    gctx = _fake_gate_context(sigma_z=0.035, lambda_ball=4.0)
    uni = _toy_universe(60, np.random.default_rng(3))
    b1 = cg.draw_ball(gctx, uni, np.random.default_rng(7))
    b2 = cg.draw_ball(gctx, uni, np.random.default_rng(7))
    np.testing.assert_array_equal(b1.z_obs, b2.z_obs)
    np.testing.assert_array_equal(b1.event_idx, b2.event_idx)
    np.testing.assert_array_equal(b1.K, b2.K)


# ── Ball estimator on a pool-free real-ladder context ────────────────────────


def _real_ladder_context(sigma_z: float, lambda_ball: float, n_events: int) -> cg.GateContext:
    """GateContext on the REAL flat-ΛCDM ladder, no injection pool needed.

    ``z_of_dl_tables`` and the Gauss-Legendre rule are built exactly as the
    parent builds them; ``log_alpha`` is zero (a constant offset per h does not
    move the argmax of a sum over events, so MAP tests remain valid).
    """
    from scipy.special import roots_legendre

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
    from master_thesis_code.physical_relations import dist_vectorized

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
    from master_thesis_code.physical_relations import dist_vectorized

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


def test_ball_posteriors_v1_limit_peak_at_truth() -> None:
    """V1 analog (noiseless): lambda=0, sigma_z=0 => both channels' MAP = 0.730."""
    gctx = _real_ladder_context(sigma_z=0.0, lambda_ball=0.0, n_events=40)
    rng = np.random.default_rng(11)
    uni = _real_ladder_universe(40, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    ln1, ln2, _slope = cg.log_channel_posteriors_ball(gctx, uni, ball)
    h = np.asarray(cl.CANONICAL_H_GRID)
    assert np.all(np.isfinite(ln1)) and np.all(np.isfinite(ln2))
    assert h[int(np.argmax(ln1))] == pytest.approx(0.730, abs=1e-12)
    assert h[int(np.argmax(ln2))] == pytest.approx(0.730, abs=1e-12)


def test_ball_posteriors_finite_with_impostors_and_photoz() -> None:
    """Maximal path (lambda=4, sigma_z=0.035): finite posteriors, both channels."""
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=15)
    rng = np.random.default_rng(13)
    uni = _real_ladder_universe(15, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    ln1, ln2, slope = cg.log_channel_posteriors_ball(gctx, uni, ball)
    assert np.all(np.isfinite(ln1)) and np.all(np.isfinite(ln2))
    assert np.isfinite(slope[0])


# ── Aggregation ──────────────────────────────────────────────────────────────


def _synthetic_records(
    n: int, rng: np.random.Generator, *, railed_low: bool = False
) -> list[dict[str, float | list[float] | str | int]]:
    """Minimal per-seed records for aggregate tests (calibrated by construction)."""
    records = []
    for i in range(n):
        pit = float(rng.random())
        rec: dict[str, float | list[float] | str | int] = {
            "seed": i,
            "n_proposed": 100000,
            "K_mean": 1.0,
            "n_impostors_total": 0,
            "n_degenerate_windows": 0,
            "texture_corr": 0.82,
            "sum_dlog_gfrac_dh": 0.0,
            "ln_post_1d": [0.0] * 41,
            "ln_post_2d": [0.0] * 41,
        }
        for ch in ("1d", "2d"):
            rec[f"pit_{ch}"] = pit
            rec[f"map_{ch}"] = 0.600 if railed_low else float(0.73 + 0.01 * rng.standard_normal())
            rec[f"map_{ch}_refined"] = rec[f"map_{ch}"]
            rec[f"mean_{ch}"] = rec[f"map_{ch}"]
            rec[f"railed_low_{ch}"] = 1.0 if railed_low else 0.0
            rec[f"railed_high_{ch}"] = 0.0
            for lv, p in (("50", 0.50), ("68", 0.68), ("90", 0.90)):
                rec[f"hpd{lv}_{ch}"] = 1.0 if rng.random() < p else 0.0
            rec[f"post_sd_{ch}"] = 0.02
            rec[f"edge_mass_{ch}"] = 1.0 if railed_low else 1e-8
        records.append(rec)
    return records


def test_aggregate_calibrated_records_pass_ds1_ds2() -> None:
    """Calibrated-by-construction records => DS-1 PASS, DS-2 PASS, no rails."""
    rng = np.random.default_rng(20260808)
    records = _synthetic_records(400, rng)
    gcfg = cg.GateConfig(cell="A", h_true=0.730, ball=False, lambda_ball=0.0, sigma_z=0.0)
    agg = cg.aggregate_gate(records, gcfg)  # type: ignore[arg-type]
    for ch in ("channel_1d", "channel_2d"):
        assert agg[ch]["ds1_status"] == "PASS"
        assert agg[ch]["ds2_ks"]["status"] == "PASS"
        assert agg[ch]["ds4_rails"]["railed_low_frac"] == 0.0
        assert not agg[ch]["edge_guard"]["edge_contaminated"]
    assert not agg["abort_b_triggered"]


def test_aggregate_railed_records_flag_ds4_and_edge_guard() -> None:
    """Fully low-railed records => R_low = 1 and the §8 edge guard fires."""
    rng = np.random.default_rng(1)
    records = _synthetic_records(200, rng, railed_low=True)
    gcfg = cg.GateConfig(cell="B2", h_true=0.730, ball=True, lambda_ball=4.0, sigma_z=0.035)
    agg = cg.aggregate_gate(records, gcfg)  # type: ignore[arg-type]
    for ch in ("channel_1d", "channel_2d"):
        assert agg[ch]["ds4_rails"]["railed_low_frac"] == 1.0
        assert agg[ch]["edge_guard"]["edge_contaminated"]
        assert agg[ch]["ds3_map_bias"]["status"] == "DEFECT-SCALE"


# ── R0 retro-read / V5 (committed data) ──────────────────────────────────────


@needs_r0
def test_v5_r0_reproduction_and_retro_read() -> None:
    """V5: the readout layer reproduces the committed R0 aggregate <= 1e-12 rel."""
    doc = cg.retro_read_r0()
    assert doc["v5"]["pass"], doc["v5"]["mismatches"]
    agg = doc["aggregate"]
    # Committed anchors (prereg §7 DS-4): starved 1D R_low = 1.000 (200/200),
    # registered-run 2D R_low/R_high = 0.005/0.035.
    assert agg["channel_1d"]["ds4_rails"]["railed_low_frac"] == pytest.approx(1.0)
    assert agg["channel_2d"]["ds4_rails"]["railed_low_frac"] == pytest.approx(0.005, abs=1e-9)
    assert agg["channel_2d"]["ds4_rails"]["railed_high_frac"] == pytest.approx(0.035, abs=1e-9)
    assert agg["n_seeds"] == 200


# ── Output-path guard ────────────────────────────────────────────────────────


def test_out_path_guard_refuses_production_dirs() -> None:
    """Never write into a production run/campaign directory (prereg §0)."""
    with pytest.raises(SystemExit):
        cg._guard_out_path("results/run_20260804_postfix/iiib/gate.json")
    with pytest.raises(SystemExit):
        cg._guard_out_path("results/campaign51_20260728/gate.json")
    cg._guard_out_path("results/calibration_gate_20260808/B2_results.json")  # must not raise


# ── JSON round-trip of a record (schema stability) ───────────────────────────


def test_record_fields_match_prereg_section_6(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_seed_gate's record carries the full §6 field list and is JSON-safe.

    The universe draw is monkeypatched (it needs the injection pool); the rest
    of run_seed_gate — ball, posteriors, readouts, record assembly — is the
    real code path on the pool-free real-ladder context.
    """
    required = {
        "seed", "cell", "h_true", "sigma_texture", "sigma_z", "f_incl", "lambda_ball",
        "n_events", "n_proposed", "z_median", "M_source_median", "frac_below_kink",
        "K_mean", "n_impostors_total", "map_1d", "map_2d", "map_1d_refined",
        "map_2d_refined", "mean_1d", "mean_2d", "railed_low_1d", "railed_high_1d",
        "railed_low_2d", "railed_high_2d", "sum_dlog_gfrac_dh", "pit_1d", "pit_2d",
        "hpd50_1d", "hpd68_1d", "hpd90_1d", "hpd50_2d", "hpd68_2d", "hpd90_2d",
        "post_sd_1d", "post_sd_2d", "edge_mass_1d", "edge_mass_2d",
        "ln_post_1d", "ln_post_2d",
    }  # fmt: skip
    gctx = _real_ladder_context(sigma_z=0.035, lambda_ball=4.0, n_events=10)
    monkeypatch.setattr(cg, "draw_universe_gate", lambda ctx, rng: _real_ladder_universe(10, rng))
    rec = cg.run_seed_gate(12345, gctx)
    assert required <= set(rec)
    json.dumps(rec)  # must be JSON-serialisable
