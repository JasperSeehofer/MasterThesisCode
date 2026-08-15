"""L4-DER Part 2 — switch decomposition of the coded tilt (runbook 11 §1).

The L4-T2 audit refuted the Part-1 D1-D6 ledger AS COMPOSED: every predicted total is
strongly negative where every measured tilt is strongly positive (pulls -69 to -211
sigma). Part 2's derivation (``L4_DER_PART2_20260815.md``) locates the failure in the
composition frame itself and identifies the missing positive source:

    In the venue's photo-z-starved regime (sigma_z-kernel width >> the GW factor's
    z-space width), the coded ratio-form GW factor ``norm.pdf(d_L(z,h)/d_obs; 1,
    sigma_d)`` is INTEGRATED over z, and its z-space Gaussian mass is proportional to
    D(z*)/D'(z*) with D(z*) = h * d_obs — it GROWS with h, contributing a per-event
    tilt G_e = (1/h) * (1 - D D''/D'^2) > 0 that no D-term ever priced.  Summed over
    the pinned population this is ~ +N/h * (1 - <D D''/D'^2>) ~ +1.06e3 nats/h — the
    dominant positive source.  Closed-form identity:  sum_e G_e = N/h - sum_e x_e/h
    (x = D D''/D'^2), i.e. EXACTLY the M6R verifier's "-N/h + D'-tracking" J-repair
    prediction with the opposite overall sign — which is WHY the A-M2' Jacobian repair
    (which makes the GW z-mass h-independent) measured on-prediction while the base
    estimator's own tilt stayed unexplained.

This script replaces the failed additive isolated-term ledger with EXACT single-switch
A/B differences on a bit-validated mirror of the production estimator, at the two
h-grid neighbours of truth (h = 0.725, 0.735), on the SAME per-seed realizations the
MN0X instrument ran (seed replay is deterministic; the mirror's base output is
validated bit-for-bit against the committed per-seed ``ln_post_1d`` vectors):

- ``base``    : the c1 (1D) path of ``venue_transfer._channel_terms_at_h``, verbatim.
- ``jac``     : the installed A-M2' per-node Jacobian variant, verbatim (reference:
                its instrument effect measured -1132.9 +- 35.9 nats/h cross-arm).
- ``mass``    : ANALYTIC per-event kill of the mass-growth term — multiply each
                event's integrand by the z-CONSTANT peak factor
                ``(d d_L/dz)(z*_e(h), h) / d_obs_e`` (z*_e(h) = the integrand-peak
                redshift, found from the estimator's own per-h z(d_L) table).  Being
                per-event constant it adds ``ln f_e(h)`` to ``ln L_e`` exactly, so the
                switch needs no quadrature: Delta T_mass = sum_e Delta ln f_e / Delta
                h.  This isolates the event-level mass growth WITHOUT reshaping the
                integrand (the jac-vs-mass difference then measures the per-node
                reshaping share).
- ``exp``     : the D3 exponent-scale swap — replace ``norm.pdf(d_L/d_obs; 1, s)`` by
                ``norm.pdf(d_obs/d_L; 1, s)`` (correct mu-scale exponent, height
                unchanged) — D3 with the full responsibility weighting the audit's
                isolated reading lacked.
- ``frozen``  : freeze the event window edges ``z_lo(h), z_hi(h)`` (and the kernel
                clip limits / point-branch validity they induce) at their h_true
                values — the window-MOTION term (M7-class edge flux), priced on the
                instrument for the first time (M7's toy closure is a toy-transfer
                risk; three toy-transfer failures are on record).

Dose structure: the same 15 seeds re-drawn at dose_scales = (1.0, f_i), f_i in
{0.25, 0.5} (paired dosing; f_i = 1.0 is the full-dose context), compared against the
MEASURED per-cell tilts of S31/S32/S33 (f_h = 1.0 row, M6R_L0_output.json, restated).

Output: ``L4_DER_PART2_output.json``.  Report: ``L4_DER_PART2_20260815.md``.
Status: PRESENTED, NOT ADJUDICATED — numbers only; the author adjudicates.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

RESULTS_DIR = Path(__file__).parent
sys.path.insert(0, str(RESULTS_DIR))

from l4_t2_audit import build_population_context  # noqa: E402

from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import calibration_gate as cg  # noqa: E402
from darksiren_emri.validation import closed_loop_gfrac as cl  # noqa: E402
from darksiren_emri.validation import venue_transfer as vt  # noqa: E402

N_SEEDS = 15
DOSE_LEVELS = ("0.25", "0.5", "1.0")
SWITCHES_FULL = ("base", "jac", "exp", "frozen")
SWITCHES_DOSE = ("base", "exp", "frozen")
H_TRUE = 0.730

# Measured references (restated verbatim; never recomputed here).
T_MEASURED = {
    "MN0X": (2624.945881166521, 18.82124210436552, 100),
    "S31": (3399.5950204674814, 70.47238869041148, 15),
    "S32": (2962.8060496205076, 74.92793185514462, 15),
    "S33": (2667.0932815894994, 53.818052089761046, 15),
}
DT_J_MEASURED = (-1132.9, 35.9)  # T(AM2P) - T(MN0X), cross-arm, M6R/L4-T1 restated
ALPHA_TILT_ANALYTIC = 1.036 * 982 / 0.730  # +1393.63 nats/h (registered constant)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def build_dose_context(f_i: float) -> vt.VenueContext:
    """Rebuild the MN0X venue context with dose_scales = (1.0, f_i).

    Identical to ``build_population_context`` except for the dose override
    (the dose mask is applied AFTER every RNG draw, so the seed stream — and
    hence the pinned realization — is identical across dose levels; paired
    dosing).
    """
    cfg = _load_json(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json")["config"]
    vcfg = vt.VenueConfig(
        cell=cfg["cell"],
        h_true=cfg["h_true"],
        balls=cfg["balls"],
        sigma_mode=cfg["sigma_mode"],
        flat_sigma_z=cfg["flat_sigma_z"],
        lambda_poisson=cfg["lambda_poisson"],
        dose_target=cfg["dose_target"],
        dose_scales=(1.0, f_i),
        crb_reference_csv=cfg["crb_reference_csv"],
        frozeng_emit_json=cfg["frozeng_emit_json"],
        pruned_catalogue_csv=cfg["pruned_catalogue_csv"],
        injection_data_dir=cfg["injection_data_dir"],
        n_events_cap=cfg["n_events_cap"],
        chunk_pairs=cfg["chunk_pairs"],
        h_grid=cfg["h_grid"],
    )
    return vt.build_venue_context(vcfg)


def mirror_ln1(
    vctx: vt.VenueContext,
    universe: cl.SyntheticUniverse,
    ball: cg.HostBall,
    sig_z: npt.NDArray[np.float64],
    k: int,
    *,
    switch: str,
    k_freeze: int,
) -> float:
    """The c1 (1D) path of ``_channel_terms_at_h``, verbatim, with switches.

    ``switch = "base"`` reproduces the production estimator's ``ln1[k]``
    bit-for-bit (validated against the committed MN0X per-seed vectors);
    ``"jac"`` mirrors the installed A-M2' variant; ``"exp"`` swaps the
    exponent scale (D3); ``"frozen"`` computes the event windows from the
    h-grid index ``k_freeze`` (h_true) instead of ``k``.
    """
    gctx = vctx.gctx
    cfg = gctx.cl_ctx.config
    h = cfg.h_grid[k]
    n = universe.z_true.size
    x = gctx.cl_ctx.gl_nodes
    w_gl = gctx.cl_ctx.gl_weights
    ev = ball.event_idx
    z_obs = ball.z_obs
    d_obs_e = universe.d_L_obs
    sig_e = universe.sigma_dL
    d_obs_p = d_obs_e[ev]
    sig_p = sig_e[ev]
    K = np.maximum(ball.K, 1)
    n_pairs = int(z_obs.size)
    chunks = vt._pair_chunks(n_pairs, vctx.vcfg.chunk_pairs)

    k_win = k_freeze if switch == "frozen" else k
    d_L_nodes_w, z_tab_w = gctx.cl_ctx.z_of_dl_tables[k_win]
    z_hi_e = np.interp(d_obs_e * (1.0 + cl._SIGMA_WINDOW * sig_e), d_L_nodes_w, z_tab_w)
    z_lo_e = np.interp(d_obs_e * (1.0 - cl._SIGMA_WINDOW * sig_e), d_L_nodes_w, z_tab_w)
    z_lo_e = np.maximum(z_lo_e, 1e-6)
    z_hi_e = np.minimum(z_hi_e, z_tab_w[-1])
    z_lo_p = z_lo_e[ev]
    z_hi_p = z_hi_e[ev]

    c1 = np.zeros(n_pairs, dtype=np.float64)
    for a0, a1 in chunks:
        sl = np.arange(a0, a1, dtype=np.int64)
        sig_c = sig_z[sl]
        q = sig_c > 0.0
        if np.any(q):
            rows_q = sl[q]
            zo = z_obs[rows_q]
            so = sig_c[q]
            a = np.maximum(z_lo_p[rows_q], zo - cg._IMPOSTOR_KERNEL_WINDOW * so)
            b = np.minimum(z_hi_p[rows_q], zo + cg._IMPOSTOR_KERNEL_WINDOW * so)
            valid = b > a
            half = 0.5 * (b - a)
            mid = 0.5 * (b + a)
            z_nodes = mid[:, None] + half[:, None] * x[None, :]
            d_L_n = np.asarray(
                dist_vectorized(np.maximum(z_nodes.reshape(-1), 1e-8), h=h),
                dtype=np.float64,
            ).reshape(z_nodes.shape)
            d_L_frac = d_L_n / d_obs_p[rows_q][:, None]
            if switch == "exp":
                p_gw = norm.pdf(1.0 / d_L_frac, loc=1.0, scale=sig_p[rows_q][:, None])
            else:
                p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[rows_q][:, None])
            kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])
            if switch == "jac":
                eps = vt.M2P_JACOBIAN_EPS_Z
                z_flat = np.maximum(z_nodes.reshape(-1), 1e-8)
                d_hi = np.asarray(dist_vectorized(z_flat + eps, h=h), dtype=np.float64)
                d_lo = np.asarray(
                    dist_vectorized(np.maximum(z_flat - eps, 1e-8), h=h), dtype=np.float64
                )
                dd_dz = ((d_hi - d_lo) / (2.0 * eps)).reshape(z_nodes.shape)
                jac = dd_dz / d_obs_p[rows_q][:, None]
                integ = kern * p_gw * jac
            else:
                integ = kern * p_gw
            c1q = half * (integ @ w_gl)
            c1[rows_q] = np.where(valid, c1q, 0.0)
        if not np.all(q):
            rows_p = sl[~q]
            zo = z_obs[rows_p]
            valid_p = (zo >= z_lo_p[rows_p]) & (zo <= z_hi_p[rows_p])
            d_pt = np.asarray(dist_vectorized(np.maximum(zo, 1e-8), h=h), dtype=np.float64)
            frac = d_pt / d_obs_p[rows_p]
            if switch == "exp":
                p_gw_p = norm.pdf(1.0 / frac, loc=1.0, scale=sig_p[rows_p])
            else:
                p_gw_p = norm.pdf(frac, loc=1.0, scale=sig_p[rows_p])
            c1[rows_p] = np.where(valid_p, p_gw_p, 0.0)

    L1 = np.bincount(ev, weights=c1, minlength=n) / K
    ok1 = (L1 > 0.0) & np.isfinite(L1)
    lnL1 = np.where(ok1, np.log(np.where(ok1, L1, 1.0)), cg._LN_ZERO_EVENT)
    return float(np.sum(lnL1)) - float(n) * gctx.cl_ctx.log_alpha[k]


def mass_kill_delta_lnf(
    vctx: vt.VenueContext,
    universe: cl.SyntheticUniverse,
    k: int,
) -> npt.NDArray[np.float64]:
    """Per-event ``ln f_e(h_k)`` of the analytic mass-kill switch.

    ``f_e(h) = (d d_L/dz)(z*_e(h), h) / d_obs_e`` with ``z*_e(h)`` from the
    estimator's own per-h z(d_L) table (the integrand-peak redshift where
    ``d_L(z, h) = d_obs``).  Multiplying event e's integrand by this
    z-constant factor adds ``ln f_e`` to ``ln L_e`` exactly — no quadrature.
    """
    gctx = vctx.gctx
    h = gctx.cl_ctx.config.h_grid[k]
    d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[k]
    d_obs = np.asarray(universe.d_L_obs, dtype=np.float64)
    z_star = np.interp(d_obs, d_L_nodes, z_tab)
    eps = 1e-6
    d_hi = np.asarray(dist_vectorized(z_star + eps, h=h), dtype=np.float64)
    d_lo = np.asarray(dist_vectorized(np.maximum(z_star - eps, 1e-8), h=h), dtype=np.float64)
    dd_dz = (d_hi - d_lo) / (2.0 * eps)
    result: npt.NDArray[np.float64] = np.log(dd_dz / d_obs)
    return result


def analytic_identity_block(vctx: vt.VenueContext) -> dict[str, float]:
    """The closed-form mass-growth identity at the pinned z_true (no noise)."""
    z = np.asarray(vctx.z_true, dtype=np.float64)
    eps = 1e-5
    d0 = np.asarray(dist_vectorized(z, h=1.0), dtype=np.float64)
    dp = (
        np.asarray(dist_vectorized(z + eps, h=1.0), dtype=np.float64)
        - np.asarray(dist_vectorized(z - eps, h=1.0), dtype=np.float64)
    ) / (2.0 * eps)
    dpp = (
        np.asarray(dist_vectorized(z + eps, h=1.0), dtype=np.float64)
        - 2.0 * d0
        + np.asarray(dist_vectorized(z - eps, h=1.0), dtype=np.float64)
    ) / eps**2
    x = d0 * dpp / dp**2
    g = (1.0 - x) / H_TRUE
    return {
        "N": float(z.size),
        "N_over_h": float(z.size / H_TRUE),
        "sum_x_over_h": float(np.sum(x) / H_TRUE),
        "P_sum_G": float(np.sum(g)),
        "mean_x": float(np.mean(x)),
        "m6r_tracking_piece_reference": 291.0,
    }


# ── worker plumbing (fork start method; contexts built in the parent) ────────

_CTXS: dict[str, tuple[vt.VenueContext, int, int, int]] = {}


def _seed_task(args: tuple[str, int]) -> dict[str, Any]:
    """One (dose_level, seed): draw once, run every switch at both h-points."""
    dose, seed = args
    vctx, i_lo, i_hi, i_true = _CTXS[dose]
    universe, ball, sigma_pairs = vt._draw_seed_realization(seed, vctx)
    switches = SWITCHES_FULL if dose == "1.0" else SWITCHES_DOSE
    out: dict[str, Any] = {"dose": dose, "seed": seed}
    dh = float(vctx.gctx.cl_ctx.config.h_grid[i_hi] - vctx.gctx.cl_ctx.config.h_grid[i_lo])
    for sw in switches:
        lo = mirror_ln1(vctx, universe, ball, sigma_pairs, i_lo, switch=sw, k_freeze=i_true)
        hi = mirror_ln1(vctx, universe, ball, sigma_pairs, i_hi, switch=sw, k_freeze=i_true)
        out[f"ln1_lo_{sw}"] = lo
        out[f"ln1_hi_{sw}"] = hi
        out[f"T_{sw}"] = (hi - lo) / dh
    lnf_lo = mass_kill_delta_lnf(vctx, universe, i_lo)
    lnf_hi = mass_kill_delta_lnf(vctx, universe, i_hi)
    out["dT_mass"] = float(np.sum(lnf_hi - lnf_lo) / dh)
    return out


def _mean_se(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def main() -> None:
    mn0x = _load_json(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json")
    per_seed = mn0x["per_seed"][:N_SEEDS]
    seeds = [int(r["seed"]) for r in per_seed]
    h_grid = np.asarray(mn0x["config"]["h_grid"], dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_grid - H_TRUE)))
    i_lo, i_hi = i_true - 1, i_true + 1
    dh = float(h_grid[i_hi] - h_grid[i_lo])
    stored = {
        int(r["seed"]): (
            float(r["ln_post_1d"][i_lo]),
            float(r["ln_post_1d"][i_hi]),
        )
        for r in per_seed
    }

    print("building contexts (full dose + 0.25 + 0.5) ...", flush=True)
    vctx_full, a_lo, a_hi = build_population_context()
    assert (a_lo, a_hi) == (i_lo, i_hi)
    _CTXS["1.0"] = (vctx_full, i_lo, i_hi, i_true)
    for f_lab, f_val in (("0.25", 0.25), ("0.5", 0.5)):
        _CTXS[f_lab] = (build_dose_context(f_val), i_lo, i_hi, i_true)

    n_alpha = float(per_seed[0]["n_events"])
    log_alpha = np.asarray(vctx_full.gctx.cl_ctx.log_alpha, dtype=np.float64)
    alpha_tilt_numeric = float(-n_alpha * (log_alpha[i_hi] - log_alpha[i_lo]) / dh)

    tasks = [(dose, s) for dose in DOSE_LEVELS for s in seeds]
    print(f"running {len(tasks)} seed-dose tasks ...", flush=True)
    with mp.get_context("fork").Pool(processes=min(8, mp.cpu_count())) as pool:
        rows = pool.map(_seed_task, tasks)

    # Validation: full-dose mirror base vs the committed instrument vectors.
    val_diffs = []
    for r in rows:
        if r["dose"] == "1.0":
            slo, shi = stored[r["seed"]]
            val_diffs.append(max(abs(r["ln1_lo_base"] - slo), abs(r["ln1_hi_base"] - shi)))
    validation_max_abs = float(max(val_diffs))
    print(f"validation: max |mirror - stored ln_post_1d| = {validation_max_abs:.3e}")

    results: dict[str, Any] = {
        "note": (
            "L4-DER Part 2 switch decomposition. All tilts nats/h, grid-neighbour "
            "central difference at h_true=0.730 (h=0.725, 0.735). dT_<switch> = "
            "T(switched) - T(base), i.e. MINUS the term the switch removes/replaces. "
            "PRESENTED, NOT ADJUDICATED."
        ),
        "seeds": seeds,
        "n_seeds": N_SEEDS,
        "validation_max_abs_ln1_diff": validation_max_abs,
        "alpha_tilt_numeric": alpha_tilt_numeric,
        "alpha_tilt_analytic": ALPHA_TILT_ANALYTIC,
        "identity_block": analytic_identity_block(vctx_full),
        "measured_references": {
            k: {"T1_mean": v[0], "T1_se": v[1], "n": v[2]} for k, v in T_MEASURED.items()
        },
        "dT_J_measured_cross_arm": {"mean": DT_J_MEASURED[0], "se": DT_J_MEASURED[1]},
        "per_seed_rows": rows,
        "by_dose": {},
    }

    for dose in DOSE_LEVELS:
        drows = [r for r in rows if r["dose"] == dose]
        switches = SWITCHES_FULL if dose == "1.0" else SWITCHES_DOSE
        blk: dict[str, Any] = {}
        t_base_m, t_base_se = _mean_se([r["T_base"] for r in drows])
        blk["T_base"] = {"mean": t_base_m, "se": t_base_se}
        blk["T_candidate_sum"] = {"mean": t_base_m - alpha_tilt_numeric, "se": t_base_se}
        dtm_m, dtm_se = _mean_se([r["dT_mass"] for r in drows])
        blk["dT_mass_analytic"] = {"mean": dtm_m, "se": dtm_se}
        for sw in switches:
            if sw == "base":
                continue
            dt_m, dt_se = _mean_se([r[f"T_{sw}"] - r["T_base"] for r in drows])
            blk[f"dT_{sw}"] = {"mean": dt_m, "se": dt_se}
        if dose == "1.0":
            blk["dT_jac_minus_dT_mass"] = {
                "mean": blk["dT_jac"]["mean"] - dtm_m,
                "note": "per-node reshaping share of the Jacobian repair",
            }
        # Leftover after removing the three located channels from the candidate sum:
        # T_cand + dT_mass + dT_exp + dT_frozen  (each dT = minus the owned term).
        leftover = [
            (r["T_base"] - alpha_tilt_numeric)
            + r["dT_mass"]
            + (r["T_exp"] - r["T_base"])
            + (r["T_frozen"] - r["T_base"])
            for r in drows
        ]
        lo_m, lo_se = _mean_se(leftover)
        blk["leftover_drift_plus_interactions"] = {"mean": lo_m, "se": lo_se}
        results["by_dose"][dose] = blk

    out_path = RESULTS_DIR / "L4_DER_PART2_output.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out_path}")

    print("\n=== summary (1D, nats/h) ===")
    print(f"alpha tilt (numeric/analytic): {alpha_tilt_numeric:+.1f} / {ALPHA_TILT_ANALYTIC:+.1f}")
    ib = results["identity_block"]
    print(
        f"identity: P = sum G = {ib['P_sum_G']:+.1f} = N/h ({ib['N_over_h']:+.1f}) "
        f"- sum x/h ({ib['sum_x_over_h']:.1f}) [M6R tracking ref +291]"
    )
    for dose in DOSE_LEVELS:
        blk = results["by_dose"][dose]
        meas = {"0.25": "S31", "0.5": "S32", "1.0": "S33"}[dose]
        tm, ts, _ = T_MEASURED[meas]
        line = (
            f"f_i={dose}: T_base {blk['T_base']['mean']:+.1f}±{blk['T_base']['se']:.1f} "
            f"(meas {meas} {tm:+.1f}±{ts:.1f}) | dT_mass {blk['dT_mass_analytic']['mean']:+.1f} "
            f"| dT_exp {blk['dT_exp']['mean']:+.1f}±{blk['dT_exp']['se']:.1f} "
            f"| dT_frozen {blk['dT_frozen']['mean']:+.1f}±{blk['dT_frozen']['se']:.1f} "
            f"| leftover {blk['leftover_drift_plus_interactions']['mean']:+.1f}"
        )
        if "dT_jac" in blk:
            line += f" | dT_jac {blk['dT_jac']['mean']:+.1f}±{blk['dT_jac']['se']:.1f}"
        print(line)


if __name__ == "__main__":
    main()
