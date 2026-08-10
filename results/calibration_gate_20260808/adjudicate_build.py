"""ADVERSARIAL adjudication of the calibration-gate build (independent verification).

Written by the verification pass, NOT the build. Attacks, per the verification
mandate:

  T1  KS null calibration — the DS-2 critical values must produce the analytic
      false-alarm rates at the prereg N (400/300/200), measured by MC against
      ``cg.ks_distance`` and cross-checked against ``scipy.stats.kstwo``.
  T2  PIT correctness — ``pp_readout``'s PIT must equal the analytic
      (grid-truncated) Gaussian CDF.
  T3  HPD correctness — ``hpd_contains`` must agree with the analytic Gaussian
      HPD region on a fine grid away from region boundaries, and empirical
      HPD coverage of a calibrated ensemble must be nominal.
  T4  Known-calibrated ensemble PASSES the full DS-1/DS-2 machinery
      (fixed-truth frequentist design exactly as the cells run).
  T5  Known-miscalibrated ensembles FAIL: coherent shift, overconfident width,
      railed ensemble (DS-4 + edge guard).
  T6  DS-1 bands quote-compare against the prereg table at N = 400/300/200.
  T7  Ball-path estimator equivalence — independent Gauss-Legendre-200
      re-integration of the prereg §4.2 formula (windows, kernel, 1/K prior,
      -N ln alpha, -745 exclusion) vs ``log_channel_posteriors_ball``, both
      sigma_z > 0 and the sigma_z = 0 point-evaluation branch, with
      NONZERO log_alpha.
  T8  V4 numbers — CSV corr(ln sigma_frac, ln d_L) anchor and the decile
      rank-matching attenuation, reproduced independently.
  P1  Cross-session determinism — re-run both committed smokes via the CLI and
      byte-compare every per-seed record.
  P2  Inheritance — cell-A/independent-texture ``run_seed_gate`` must be
      bit-identical to the parent ``cl.run_seed`` on the same seed; the
      dl_binned draw must differ; V4 corrs of validate_results.json reproduced.
  P3  DS-7 — independent p_bar MC (different seed) and ratio arithmetic.

Run:  cd <repo-root> && uv run python results/calibration_gate_20260808/adjudicate_build.py
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy.stats import kstwo, norm

from master_thesis_code.validation import calibration_gate as cg
from master_thesis_code.validation import closed_loop_gfrac as cl

RESULTS = {}
DIR = "results/calibration_gate_20260808"
SCRATCH = os.environ.get("ADJ_SCRATCH", "/tmp/claude-1000/adj_gate")
os.makedirs(SCRATCH, exist_ok=True)

FINE = np.linspace(0.50, 0.96, 2301)


def record(name, ok, **info):
    RESULTS[name] = {"pass": bool(ok), **info}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")


# ── T1: KS null calibration ──────────────────────────────────────────────────
def t1():
    rng = np.random.default_rng(424242)
    for n in (400, 300, 200):
        d95 = 1.358 / math.sqrt(n)
        d99 = 1.628 / math.sqrt(n)
        reps = 20000
        ds = np.empty(reps)
        for i in range(reps):
            ds[i] = cg.ks_distance(rng.random(n))
        rate95 = float(np.mean(ds > d95))
        rate99 = float(np.mean(ds > d99))
        exact95 = float(kstwo.sf(d95, n))
        exact99 = float(kstwo.sf(d99, n))
        se95 = math.sqrt(exact95 * (1 - exact95) / reps)
        se99 = math.sqrt(exact99 * (1 - exact99) / reps)
        ok = (
            abs(rate95 - exact95) < 4 * se95
            and abs(rate99 - exact99) < 4 * se99
            and 0.03 < rate95 < 0.07
            and 0.005 < rate99 < 0.02
        )
        record(
            f"T1_ks_null_N{n}",
            ok,
            mc_rate95=round(rate95, 4),
            exact95=round(exact95, 4),
            mc_rate99=round(rate99, 4),
            exact99=round(exact99, 4),
        )


# ── T2: PIT analytic ─────────────────────────────────────────────────────────
def t2():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        mu = rng.uniform(0.60, 0.86)
        sd = rng.uniform(0.01, 0.08)
        h_true = rng.uniform(0.55, 0.92)
        ln_post = -0.5 * ((FINE - mu) / sd) ** 2
        pit = cg.pp_readout(FINE, ln_post, h_true)["pit"]
        lo, hi = FINE[0], FINE[-1]
        z = norm.cdf
        analytic = (z((h_true - mu) / sd) - z((lo - mu) / sd)) / (
            z((hi - mu) / sd) - z((lo - mu) / sd)
        )
        analytic = min(max(analytic, 0.0), 1.0)
        worst = max(worst, abs(pit - analytic))
    record("T2_pit_analytic", worst < 5e-4, worst_abs_err=round(worst, 8))


# ── T3: HPD analytic + empirical coverage ────────────────────────────────────
def t3():
    rng = np.random.default_rng(11)
    step = FINE[1] - FINE[0]
    n_checked = 0
    n_bad = 0
    for _ in range(3000):
        mu = rng.uniform(0.62, 0.84)
        sd = rng.uniform(0.01, 0.05)
        level = float(rng.choice([0.50, 0.68, 0.90]))
        h_true = rng.uniform(0.55, 0.92)
        zb = norm.ppf(0.5 * (1 + level))
        edge_dist = abs(abs(h_true - mu) - zb * sd)
        if edge_dist < 3 * step:  # skip within grid resolution of the boundary
            continue
        post = np.exp(-0.5 * ((FINE - mu) / sd) ** 2)
        post /= np.trapezoid(post, FINE)
        got = cg.hpd_contains(FINE, post, h_true, level)
        want = abs(h_true - mu) <= zb * sd
        n_checked += 1
        n_bad += int(got != want)
    record("T3_hpd_analytic", n_bad == 0 and n_checked > 2000, checked=n_checked, bad=n_bad)

    # empirical coverage of a calibrated fixed-truth ensemble through pp_readout
    rng = np.random.default_rng(20260808)
    n = 2000
    truth, sd = 0.730, 0.03
    hits = {0.50: 0, 0.68: 0, 0.90: 0}
    pits = []
    for _ in range(n):
        h_hat = rng.normal(truth, sd)
        ln_post = -0.5 * ((FINE - h_hat) / sd) ** 2
        out = cg.pp_readout(FINE, ln_post, truth)
        pits.append(out["pit"])
        for lv in hits:
            hits[lv] += int(out[f"hpd{int(round(lv * 100))}"])
    ok = True
    cov = {}
    for lv, k in hits.items():
        sig = math.sqrt(lv * (1 - lv) / n)
        cov[str(lv)] = round(k / n, 4)
        ok &= abs(k / n - lv) < 3.5 * sig
    d = cg.ks_distance(np.asarray(pits))
    ok &= d < 1.358 / math.sqrt(n)
    record("T3_fixed_truth_calibrated_coverage", ok, coverage=cov, ks=round(d, 4))


# ── T4/T5: full aggregation machinery on constructed ensembles ───────────────
def _ensemble_records(rng, n, truth, sd_true, sd_post, shift=0.0, railed=False):
    recs = []
    grid = FINE
    for i in range(n):
        h_hat = rng.normal(truth, sd_true) + shift
        if railed:
            ln_post = -300.0 * (grid - grid[0])
        else:
            ln_post = -0.5 * ((grid - h_hat) / sd_post) ** 2
        pp = cg.pp_readout(grid, ln_post, truth)
        r = cl.posterior_readout(grid, ln_post)
        rec = {"seed": i}
        for ch in ("1d", "2d"):
            rec[f"pit_{ch}"] = pp["pit"]
            rec[f"map_{ch}"] = r["map"]
            rec[f"map_{ch}_refined"] = r["map_refined"]
            rec[f"mean_{ch}"] = r["mean"]
            rec[f"railed_low_{ch}"] = r["railed_low"]
            rec[f"railed_high_{ch}"] = r["railed_high"]
            for lv in (50, 68, 90):
                rec[f"hpd{lv}_{ch}"] = pp[f"hpd{lv}"]
            rec[f"post_sd_{ch}"] = pp["post_sd"]
            rec[f"edge_mass_{ch}"] = pp["edge_mass"]
        recs.append(rec)
    return recs


def t4_t5():
    truth, sd = 0.730, 0.03
    rng = np.random.default_rng(99)
    good = cg._channel_aggregate(_ensemble_records(rng, 400, truth, sd, sd), "1d", truth)
    ok = (
        good["ds1_status"] == "PASS"
        and good["ds2_ks"]["status"] == "PASS"
        and good["ds3_map_bias"]["status"] == "IN-BAND"
        and not good["edge_guard"]["edge_contaminated"]
    )
    record(
        "T4_calibrated_passes",
        ok,
        ds1=good["ds1_status"],
        ds2=good["ds2_ks"]["status"],
        ds3=good["ds3_map_bias"]["status"],
    )

    shifted = cg._channel_aggregate(
        _ensemble_records(rng, 400, truth, sd, sd, shift=2.0 * sd), "1d", truth
    )
    ok = shifted["ds1_status"] == "FAIL" and shifted["ds2_ks"]["status"] == "FAIL"
    record(
        "T5_shifted_fails",
        ok,
        ds1=shifted["ds1_status"],
        ds2=shifted["ds2_ks"]["status"],
        c90=round(shifted["ds1_coverage"]["hpd90"]["value"], 3),
    )

    overconf = cg._channel_aggregate(
        _ensemble_records(rng, 400, truth, sd, 0.5 * sd), "1d", truth
    )
    ok = overconf["ds1_status"] == "FAIL" and overconf["ds2_ks"]["status"] == "FAIL"
    record(
        "T5_overconfident_fails",
        ok,
        ds1=overconf["ds1_status"],
        ds2=overconf["ds2_ks"]["status"],
        c90=round(overconf["ds1_coverage"]["hpd90"]["value"], 3),
    )

    railed = cg._channel_aggregate(
        _ensemble_records(rng, 200, truth, sd, sd, railed=True), "1d", truth
    )
    ok = (
        railed["ds4_rails"]["railed_low_frac"] == 1.0
        and railed["edge_guard"]["edge_contaminated"]
    )
    record("T5_railed_ds4_edge_guard", ok, r_low=railed["ds4_rails"]["railed_low_frac"])


# ── T6: DS-1 band quote-compare ──────────────────────────────────────────────
def t6():
    prereg = {
        400: {"sig": (0.0250, 0.0233, 0.0150), "b2": ((0.450, 0.550), (0.633, 0.727), (0.870, 0.930)), "b3": ((0.425, 0.575), (0.610, 0.750), (0.855, 0.945))},
        300: {"sig": (0.0289, 0.0269, 0.0173), "b2": ((0.442, 0.558), (0.626, 0.734), (0.865, 0.935)), "b3": None},
        200: {"sig": (0.0354, 0.0330, 0.0212), "b2": ((0.429, 0.571), (0.614, 0.746), (0.858, 0.942)), "b3": None},
    }
    ok_all = True
    detail = {}
    rng = np.random.default_rng(5)
    for n, want in prereg.items():
        recs = _ensemble_records(rng, n, 0.730, 0.03, 0.03)
        agg = cg._channel_aggregate(recs, "1d", 0.730)
        for j, lv in enumerate((0.50, 0.68, 0.90)):
            c = agg["ds1_coverage"][f"hpd{int(lv * 100)}"]
            ok_all &= abs(c["binomial_sigma"] - want["sig"][j]) < 6e-4
            ok_all &= abs(c["band_2sigma"][0] - want["b2"][j][0]) < 1.5e-3
            ok_all &= abs(c["band_2sigma"][1] - want["b2"][j][1]) < 1.5e-3
            if want["b3"]:
                ok_all &= abs(c["band_3sigma"][0] - want["b3"][j][0]) < 1.5e-3
                ok_all &= abs(c["band_3sigma"][1] - want["b3"][j][1]) < 1.5e-3
        detail[n] = "checked"
    # KS critical values quote-compare
    for n, (w95, w99) in {400: (0.0679, 0.0814), 300: (0.0784, 0.0940), 200: (0.0960, 0.1151)}.items():
        ok_all &= abs(1.358 / math.sqrt(n) - w95) < 6e-5
        ok_all &= abs(1.628 / math.sqrt(n) - w99) < 6e-5
    record("T6_prereg_band_quote_compare", ok_all, detail=detail)


# ── T7: ball-path independent re-integration ─────────────────────────────────
def _ladder_ctx(sigma_z, lambda_ball, log_alpha_mode="nonzero"):
    from scipy.special import roots_legendre
    from master_thesis_code.physical_relations import dist_vectorized

    gcfg = cg.GateConfig(
        cell="custom", h_true=0.730, ball=True, lambda_ball=lambda_ball,
        sigma_z=sigma_z, n_events=6,
    )
    cl_cfg = cg.to_closed_loop_config(gcfg)
    z_max = 1.5
    tables = [cl._z_of_dl_table(h, z_max) for h in cl_cfg.h_grid]
    gl_nodes, gl_weights = roots_legendre(cl_cfg.n_quad)
    n_h = len(cl_cfg.h_grid)
    if log_alpha_mode == "nonzero":
        log_alpha = 0.3 * np.sin(np.arange(n_h) * 0.7) - 0.1  # arbitrary, h-dependent
    else:
        log_alpha = np.zeros(n_h)
    cl_ctx = cl.ClosedLoopContext(
        config=cl_cfg, detection=None, sigma_triples=np.asarray([[0.05, 0.05, 0.3]]),
        z_max_true=z_max,
        gen_z_nodes=np.linspace(1e-6, z_max, 100), gen_z_cdf=np.linspace(0, 1, 100),
        gen_log10_M_nodes=np.linspace(4, 7, 100), gen_M_cdf=np.linspace(0, 1, 100),
        z_of_dl_tables=tables, log_alpha=np.asarray(log_alpha, dtype=np.float64),
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
        gate_config=gcfg, cl_ctx=cl_ctx, csv_dl_sorted=np.asarray([1.0]),
        triples=np.asarray([[0.05, 0.05, 0.3]]),
        decile_rows=[np.asarray([0], dtype=np.int64)] * 10,
        imp_z_nodes=z_nodes, imp_z_cdf=cdf, imp_dl_nodes=dl_nodes,
    )


def _ladder_universe(n, rng):
    from master_thesis_code.physical_relations import dist_vectorized

    z = rng.uniform(0.15, 0.9, size=n)
    d_L = np.asarray(dist_vectorized(z, h=0.730), dtype=np.float64)
    sig_d = rng.uniform(0.03, 0.08, size=n)
    sig_m = np.full(n, 0.05)
    rho = np.full(n, 0.3)
    M = np.full(n, 5.0e5)
    d_obs = d_L * (1.0 + sig_d * rng.standard_normal(n) * 0.5)
    M_z = M * (1.0 + z)
    return cl.SyntheticUniverse(
        z_true=z, M_true=M, d_L_true=d_L, d_L_obs=d_obs,
        M_z_obs=M_z * (1.0 + 0.02 * rng.standard_normal(n)),
        sigma_dL=sig_d, sigma_Mz=sig_m, rho=rho,
        in_catalogue=np.zeros(n, dtype=bool), n_drawn=n,
    )


def _brute_force_lnpost(gctx, uni, ball, n_gl=200):
    """Independent §4.2 evaluation with my own GL-200 assembly."""
    from numpy.polynomial.legendre import leggauss
    from master_thesis_code.bayesian_inference.bayesian_statistics import (
        completion_mass_factor_g,
    )
    from master_thesis_code.physical_relations import dist_vectorized

    cfg = gctx.cl_ctx.config
    gcfg = gctx.gate_config
    n = uni.z_true.size
    xg, wg = leggauss(n_gl)
    s_dd = uni.sigma_dL**2
    s_dm = uni.rho * uni.sigma_dL * uni.sigma_Mz
    s_mm = uni.sigma_Mz**2
    proj = np.where(s_dd > 0, s_dm / np.maximum(s_dd, 1e-300), 0.0)
    sig_c = np.sqrt(np.maximum(s_mm - proj * s_dm, 1e-30))
    ln1 = np.empty(len(cfg.h_grid))
    ln2 = np.empty(len(cfg.h_grid))
    for k, h in enumerate(cfg.h_grid):
        dtab, ztab = gctx.cl_ctx.z_of_dl_tables[k]
        L1 = np.zeros(n)
        L2 = np.zeros(n)
        for p in range(ball.z_obs.size):
            i = int(ball.event_idx[p])
            zo = float(ball.z_obs[p])
            zlo = max(float(np.interp(uni.d_L_obs[i] * (1 - 4 * uni.sigma_dL[i]), dtab, ztab)), 1e-6)
            zhi = min(float(np.interp(uni.d_L_obs[i] * (1 + 4 * uni.sigma_dL[i]), dtab, ztab)), float(ztab[-1]))
            if gcfg.sigma_z > 0:
                a = max(zlo, zo - 5 * gcfg.sigma_z)
                b = min(zhi, zo + 5 * gcfg.sigma_z)
                if b <= a:
                    continue
                zz = 0.5 * (b + a) + 0.5 * (b - a) * xg
                dl = np.asarray(dist_vectorized(np.maximum(zz, 1e-8), h=h), dtype=np.float64)
                fr = dl / uni.d_L_obs[i]
                integ = norm.pdf(zz, zo, gcfg.sigma_z) * norm.pdf(fr, 1.0, uni.sigma_dL[i])
                g = completion_mass_factor_g(
                    zz, fr, float(uni.M_z_obs[i]), float(proj[i]), float(sig_c[i]),
                    n_hermite=cfg.n_hermite,
                )
                L1[i] += 0.5 * (b - a) * float(np.sum(integ * wg))
                L2[i] += 0.5 * (b - a) * float(np.sum(integ * g * wg))
            else:
                if not (zlo <= zo <= zhi):
                    continue
                dl = float(np.asarray(dist_vectorized(np.asarray([max(zo, 1e-8)]), h=h))[0])
                fr = dl / uni.d_L_obs[i]
                pg = norm.pdf(fr, 1.0, uni.sigma_dL[i])
                g = completion_mass_factor_g(
                    np.asarray([zo]), np.asarray([fr]), float(uni.M_z_obs[i]),
                    float(proj[i]), float(sig_c[i]), n_hermite=cfg.n_hermite,
                )[0]
                L1[i] += pg
                L2[i] += pg * g
        K = np.maximum(ball.K, 1)
        L1 = L1 / K
        L2 = L2 / K
        la = gctx.cl_ctx.log_alpha[k]
        ln1[k] = sum(math.log(v) if v > 0 else -745.0 for v in L1) - n * la
        ln2[k] = sum(math.log(v) if v > 0 else -745.0 for v in L2) - n * la
    return ln1, ln2


def t7():
    # sigma_z > 0 branch, nonzero log_alpha
    gctx = _ladder_ctx(0.035, 3.0)
    rng = np.random.default_rng(21)
    uni = _ladder_universe(6, rng)
    ball = cg.draw_ball(gctx, uni, rng)
    ln1, ln2, _ = cg.log_channel_posteriors_ball(gctx, uni, ball)
    b1, b2 = _brute_force_lnpost(gctx, uni, ball)
    e1 = float(np.max(np.abs(ln1 - b1)))
    e2 = float(np.max(np.abs(ln2 - b2)))
    record("T7_ball_integral_sigmaz", e1 < 1e-7 and e2 < 1e-7, max_err_1d=e1, max_err_2d=e2)

    # sigma_z = 0 point branch
    gctx0 = _ladder_ctx(0.0, 3.0)
    rng = np.random.default_rng(22)
    uni0 = _ladder_universe(6, rng)
    ball0 = cg.draw_ball(gctx0, uni0, rng)
    ln1, ln2, _ = cg.log_channel_posteriors_ball(gctx0, uni0, ball0)
    b1, b2 = _brute_force_lnpost(gctx0, uni0, ball0)
    e1 = float(np.max(np.abs(ln1 - b1)))
    e2 = float(np.max(np.abs(ln2 - b2)))
    record("T7_ball_point_branch", e1 < 1e-9 and e2 < 1e-9, max_err_1d=e1, max_err_2d=e2)

    # -745 exclusion semantics: force one event's single candidate out of window
    gctx0b = _ladder_ctx(0.0, 0.0)
    rng = np.random.default_rng(23)
    unix = _ladder_universe(4, rng)
    ballx = cg.draw_ball(gctx0b, unix, rng)
    z_mod = ballx.z_obs.copy()
    z_mod[0] = 1.45  # far above event 0's window at every h
    ballx = cg.HostBall(
        z_obs=z_mod, event_idx=ballx.event_idx, K=ballx.K,
        n_impostors_total=0, n_degenerate_windows=0,
    )
    ln1, ln2, _ = cg.log_channel_posteriors_ball(gctx0b, unix, ballx)
    b1, b2 = _brute_force_lnpost(gctx0b, unix, ballx)
    e1 = float(np.max(np.abs(ln1 - b1)))
    penalty_present = bool(np.all(ln1 < -700))  # every h carries the -745 event
    record("T7_zero_event_exclusion", e1 < 1e-9 and penalty_present, max_err=e1)


# ── T8: V4 texture numbers, independently ────────────────────────────────────
def t8():
    triples, dl = cg.load_sigma_triples_with_dl(cl.DEFAULT_CRB_CSV)
    csv_corr = float(np.corrcoef(np.log(triples[:, 0]), np.log(dl))[0, 1])
    # decile rank-matching self-simulation on the CSV's own d_L values
    rng = np.random.default_rng(20260808)
    n_rows = dl.size
    rank = np.argsort(np.argsort(dl))
    dec_row = np.clip((10 * rank) // n_rows, 0, 9)
    pools = [np.where(dec_row == b)[0] for b in range(10)]
    reps = 20
    corrs = []
    dl_sorted = np.sort(dl)
    for _ in range(reps):
        ev_dl = rng.choice(dl, size=1500, replace=True)
        q = np.searchsorted(dl_sorted, ev_dl, side="right") / n_rows
        dec = np.clip((q * 10).astype(int), 0, 9)
        s = np.empty(1500)
        for b in range(10):
            m = dec == b
            if m.any():
                s[m] = triples[rng.choice(pools[b], size=int(m.sum())), 0]
        corrs.append(float(np.corrcoef(np.log(s), np.log(ev_dl))[0, 1]))
    med = float(np.median(corrs))
    ok = abs(csv_corr - 0.816) < 0.01 and 0.63 < med < 0.75 and (med < 0.72)
    record(
        "T8_v4_texture_numbers", ok,
        csv_corr=round(csv_corr, 4), rank_matched_median=round(med, 4),
        note="confirms 0.82 anchor is the CSV corr and that decile matching attenuates below the V4 band",
    )


# ── P1: cross-session smoke determinism ──────────────────────────────────────
def p1():
    # NOTE: --truth is required for B2 — the stored smoke ran truth 0.730; without
    # it the CLI defaults to the cell's FIRST registered truth (0.690, seed block
    # +5000), which is a different registered configuration, not nondeterminism.
    # (First adjudication pass omitted it and mis-flagged a determinism failure.)
    cells = (("V1", f"{DIR}/smoke_V1.json", []), ("B2", f"{DIR}/smoke_B2_h0p730.json", ["--truth", "0.730"]))
    for cell, ref_path, extra in cells:
        out = os.path.join(SCRATCH, f"rerun_smoke_{cell}.json")
        r = subprocess.run(
            [sys.executable, "-m", "master_thesis_code.validation.calibration_gate",
             "--cell", cell, "--smoke", "--allow-dirty", "--out", out, *extra],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            record(f"P1_smoke_rerun_{cell}", False, stderr=r.stderr[-500:])
            continue
        with open(ref_path) as fh:
            ref = json.load(fh)
        with open(out) as fh:
            new = json.load(fh)
        same_seeds = ref["seeds"] == new["seeds"]
        per_seed_identical = json.dumps(ref["per_seed"], sort_keys=True) == json.dumps(
            new["per_seed"], sort_keys=True
        )
        agg_ref = {k: v for k, v in ref["aggregate"].items()}
        agg_new = {k: v for k, v in new["aggregate"].items()}
        agg_identical = json.dumps(agg_ref, sort_keys=True) == json.dumps(agg_new, sort_keys=True)
        v3 = new.get("v3_smoke", {}).get("pass")
        record(
            f"P1_smoke_rerun_{cell}",
            same_seeds and per_seed_identical and agg_identical and v3,
            per_seed_identical=per_seed_identical, agg_identical=agg_identical, v3_smoke=v3,
        )


# ── P2: inheritance + texture-changes-draw + V4 corr reproduction ────────────
def p2():
    gcfg_ind = cg.GateConfig(
        cell="A", h_true=0.730, ball=False, lambda_ball=0.0, sigma_z=0.0,
        sigma_texture="independent",
    )
    gctx_ind = cg.build_gate_context(gcfg_ind)
    seed = 20261808  # cell A truth-0.730 block start
    rec_gate = cg.run_seed_gate(seed, gctx_ind)
    rec_parent = cl.run_seed(seed, ctx=gctx_ind.cl_ctx)
    same1 = rec_gate["ln_post_1d"] == rec_parent["ln_post_1d"]
    same2 = rec_gate["ln_post_2d"] == rec_parent["ln_post_2d"]
    record("P2_independent_texture_inherits_parent_bitexact", same1 and same2,
           ln1_equal=same1, ln2_equal=same2)

    # determinism in-process
    rec_again = cg.run_seed_gate(seed, gctx_ind)
    record(
        "P2_same_seed_bit_identical",
        json.dumps(rec_gate, sort_keys=True) == json.dumps(rec_again, sort_keys=True),
    )

    # dl_binned must differ from the parent draw; compare against timing_A record
    gcfg_dlb = cg.GateConfig(
        cell="A", h_true=0.730, ball=False, lambda_ball=0.0, sigma_z=0.0,
        sigma_texture="dl_binned",
    )
    gctx_dlb = cg.GateContext(
        gate_config=gcfg_dlb, cl_ctx=gctx_ind.cl_ctx,
        csv_dl_sorted=gctx_ind.csv_dl_sorted, triples=gctx_ind.triples,
        decile_rows=gctx_ind.decile_rows, imp_z_nodes=gctx_ind.imp_z_nodes,
        imp_z_cdf=gctx_ind.imp_z_cdf, imp_dl_nodes=gctx_ind.imp_dl_nodes,
    )
    rec_dlb = cg.run_seed_gate(seed, gctx_dlb)
    differs = rec_dlb["ln_post_1d"] != rec_parent["ln_post_1d"]
    with open(f"{DIR}/timing_A_fullN.json") as fh:
        timing = json.load(fh)
    t_rec = timing["per_seed"][0]
    matches_timing = (
        t_rec["seed"] == seed
        and rec_dlb["ln_post_1d"] == t_rec["ln_post_1d"]
        and rec_dlb["ln_post_2d"] == t_rec["ln_post_2d"]
    )
    record("P2_dlbinned_differs_and_reproduces_timingA", differs and matches_timing,
           differs_from_parent=differs, reproduces_timing_record=matches_timing)

    # dl_binned decile property: every event's sigma comes from its matched decile pool
    rng = np.random.default_rng(seed)
    uni = cg.draw_universe_gate(gctx_dlb, rng)
    q = np.searchsorted(gctx_dlb.csv_dl_sorted, uni.d_L_true, side="right") / gctx_dlb.csv_dl_sorted.size
    dec = np.clip((q * 10).astype(np.int64), 0, 9)
    ok_pool = True
    for i in range(uni.z_true.size):
        pool_sigmas = gctx_dlb.triples[gctx_dlb.decile_rows[dec[i]], 0]
        if not np.any(np.isclose(pool_sigmas, uni.sigma_dL[i], rtol=0, atol=0)):
            ok_pool = False
            break
    record("P2_dlbinned_decile_membership_property", ok_pool)

    # V4 corr reproduction against validate_results.json
    with open(f"{DIR}/validate_results.json") as fh:
        val = json.load(fh)
    corrs = []
    for i in range(3):
        rng = np.random.default_rng(cg.GATE_BASE_SEED + i)
        u = cg.draw_universe_gate(gctx_dlb, rng)
        corrs.append(float(np.corrcoef(np.log(u.sigma_dL), np.log(u.d_L_true))[0, 1]))
    match = np.allclose(corrs, val["v4"]["corrs"], rtol=0, atol=1e-12)
    record("P2_v4_corrs_reproduced", match, mine=[round(c, 6) for c in corrs],
           committed=[round(c, 6) for c in val["v4"]["corrs"]])

    # P3: DS-7 independent p_bar (different seed MC) + arithmetic
    rng = np.random.default_rng(777)
    n_mc = 400_000
    u_z = rng.random(n_mc)
    z = np.interp(u_z, gctx_ind.cl_ctx.gen_z_cdf, gctx_ind.cl_ctx.gen_z_nodes)
    u_m = rng.random(n_mc)
    M = 10.0 ** np.interp(u_m, gctx_ind.cl_ctx.gen_M_cdf, gctx_ind.cl_ctx.gen_log10_M_nodes)
    from master_thesis_code.physical_relations import dist_vectorized

    d_L = np.asarray(dist_vectorized(z, h=0.730), dtype=np.float64)
    p = np.asarray(
        gctx_ind.cl_ctx.detection.detection_probability_with_bh_mass_interpolated(
            d_L, M * (1.0 + z), 0.0, 0.0, h=0.730
        ),
        dtype=np.float64,
    )
    p_bar_mine = float(np.mean(p))
    with open(f"{DIR}/timing_B2_fullN.json") as fh:
        tb2 = json.load(fh)
    ds7 = tb2["aggregate"]["ds7"]
    se = math.sqrt(p_bar_mine * (1 - p_bar_mine) / n_mc)
    ok = abs(p_bar_mine - ds7["p_bar"]) < 5 * se
    ratio_check = abs(ds7["ratio"] - 1500 / (ds7["mean_n_proposed"] * ds7["p_bar"])) < 1e-12
    corr_check = abs(ds7["ratio_corrected"] - ds7["ratio"] * ds7["expected_batch_overcount"]) < 1e-12
    record("P3_ds7_pbar_and_arithmetic", ok and ratio_check and corr_check,
           p_bar_mine=round(p_bar_mine, 5), p_bar_reported=round(ds7["p_bar"], 5),
           ratio_arith=ratio_check, corrected_arith=corr_check)


def main():
    t1()
    t2()
    t3()
    t4_t5()
    t6()
    t7()
    t8()
    p1()
    p2()
    n_fail = sum(1 for v in RESULTS.values() if not v["pass"])
    out_path = os.path.join(DIR, "adjudicate_results.json")
    with open(out_path, "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'} — written {out_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
