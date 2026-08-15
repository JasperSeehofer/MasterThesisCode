"""A-FULL candidate pre-measurement (ledger row #109 item 3 — draft support, committed data only).

Measures, on the same bit-validated mirror geometry as the switch decomposition (15 MN0X
seed replays, grid-neighbour tilt at h_true), the venue tilt of three CORRECT-FORM
estimator candidates, so the A-FULL draft's central prediction ("tilt ~= 0") is a
measured number instead of an assertion:

- FULL-A  : the d_obs-DENSITY GW factor alone — ``p_gw_full(z) = norm.pdf(d_obs/d_L; 1,
            sigma_d) / d_L`` (Part 1 F3: prefactor AND mu-scale exponent together),
            kernel and 1/K unchanged.  Reported without the alpha term (Part 1 F2) and
            with it (reference).
- FULL-B  : FULL-A x the estimator's own population measure ``w_pop(z; h) = (dV_c/dz)/
            (1+z)`` as a numerator node weight (the Gray-consistency pairing with the
            alpha the code already carries; ``closed_loop_gfrac._w_pop``, the same
            function ``alpha(h)`` integrates).  Reported with alpha (the paired form)
            and without.
- FULL-C  : FULL-B + per-candidate kernel renormalisation (divide by the retained kernel
            mass W_k(h), the A-REN switch verbatim) — the complete candidate: density
            form + population weight + renormalised kernel.

Dose levels: full (1.0) for all variants; 0.25 additionally for FULL-C (dose structure of
the residual tilt).  Output: ``L4_AFULL_PREMEASURE_output.json``.
Status: PRESENTED, NOT ADJUDICATED — numbers for the draft; the author adjudicates.
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

from l4_der_part2_switch_decomposition import (  # noqa: E402
    H_TRUE,
    N_SEEDS,
    _load_json,
    build_dose_context,
)
from l4_t2_audit import build_population_context  # noqa: E402

from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import calibration_gate as cg  # noqa: E402
from darksiren_emri.validation import closed_loop_gfrac as cl  # noqa: E402
from darksiren_emri.validation import venue_transfer as vt  # noqa: E402

VARIANTS = ("full_a", "full_b", "full_c")
_W_K_FLOOR = vt._W_K_FLOOR if hasattr(vt, "_W_K_FLOOR") else 1e-12


def full_ln1(
    vctx: vt.VenueContext,
    universe: cl.SyntheticUniverse,
    ball: cg.HostBall,
    sig_z: npt.NDArray[np.float64],
    k: int,
    *,
    variant: str,
) -> float:
    """c1/1D mirror with the correct-form (d_obs-density) GW factor and options.

    Returns the CANDIDATE-SUM log-likelihood only (no alpha term); callers add
    ``-n * log_alpha[k]`` analytically where a paired form is reported.
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

    d_L_nodes_w, z_tab_w = gctx.cl_ctx.z_of_dl_tables[k]
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
            z_flat = np.maximum(z_nodes.reshape(-1), 1e-8)
            d_L_n = np.asarray(dist_vectorized(z_flat, h=h), dtype=np.float64).reshape(
                z_nodes.shape
            )
            # Correct-form GW factor: N(d_obs; d_L(z,h), sigma_d * d_L(z,h)) —
            # density in d_obs (prefactor + mu-scale exponent together).
            ratio = d_obs_p[rows_q][:, None] / d_L_n
            p_gw = norm.pdf(ratio, loc=1.0, scale=sig_p[rows_q][:, None]) / d_L_n
            kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])
            integ = kern * p_gw
            if variant in ("full_b", "full_c", "full_d", "full_e"):
                integ = integ * np.asarray(cl._w_pop(z_flat, h), dtype=np.float64).reshape(
                    z_nodes.shape
                )
            if variant in ("full_d", "full_e"):
                z_s, s_phi = gctx.cl_ctx.s_phi_tables[k]
                integ = integ * np.interp(z_nodes, z_s, s_phi)
            if variant in ("full_c", "full_e"):
                w_k = norm.cdf((b - zo) / so) - norm.cdf((a - zo) / so)
                integ = integ / np.maximum(w_k, _W_K_FLOOR)[:, None]
            c1q = half * (integ @ w_gl)
            c1[rows_q] = np.where(valid, c1q, 0.0)
        if not np.all(q):
            rows_p = sl[~q]
            zo = z_obs[rows_p]
            valid_p = (zo >= z_lo_p[rows_p]) & (zo <= z_hi_p[rows_p])
            d_pt = np.asarray(dist_vectorized(np.maximum(zo, 1e-8), h=h), dtype=np.float64)
            ratio_p = d_obs_p[rows_p] / d_pt
            p_gw_p = norm.pdf(ratio_p, loc=1.0, scale=sig_p[rows_p]) / d_pt
            if variant in ("full_b", "full_c", "full_d", "full_e"):
                p_gw_p = p_gw_p * np.asarray(cl._w_pop(np.maximum(zo, 1e-8), h), dtype=np.float64)
            if variant in ("full_d", "full_e"):
                z_s, s_phi = gctx.cl_ctx.s_phi_tables[k]
                p_gw_p = p_gw_p * np.interp(np.maximum(zo, 1e-8), z_s, s_phi)
            c1[rows_p] = np.where(valid_p, p_gw_p, 0.0)

    L1 = np.bincount(ev, weights=c1, minlength=n) / K
    ok1 = (L1 > 0.0) & np.isfinite(L1)
    lnL1 = np.where(ok1, np.log(np.where(ok1, L1, 1.0)), cg._LN_ZERO_EVENT)
    return float(np.sum(lnL1))


_CTXS: dict[str, tuple[vt.VenueContext, int, int]] = {}


def _seed_task(args: tuple[str, int]) -> dict[str, Any]:
    dose, seed = args
    vctx, i_lo, i_hi = _CTXS[dose]
    universe, ball, sigma_pairs = vt._draw_seed_realization(seed, vctx)
    variants = ("full_d", "full_e")
    dh = float(vctx.gctx.cl_ctx.config.h_grid[i_hi] - vctx.gctx.cl_ctx.config.h_grid[i_lo])
    out: dict[str, Any] = {"dose": dose, "seed": seed}
    for v in variants:
        lo = full_ln1(vctx, universe, ball, sigma_pairs, i_lo, variant=v)
        hi = full_ln1(vctx, universe, ball, sigma_pairs, i_hi, variant=v)
        out[f"T_cand_{v}"] = (hi - lo) / dh
    return out


def main() -> None:
    mn0x = _load_json(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json")
    seeds = [int(r["seed"]) for r in mn0x["per_seed"][:N_SEEDS]]
    h_grid = np.asarray(mn0x["config"]["h_grid"], dtype=np.float64)
    i_true = int(np.argmin(np.abs(h_grid - H_TRUE)))
    i_lo, i_hi = i_true - 1, i_true + 1
    dh = float(h_grid[i_hi] - h_grid[i_lo])
    n_alpha = float(mn0x["per_seed"][0]["n_events"])

    print("building contexts ...", flush=True)
    vctx_full, a_lo, a_hi = build_population_context()
    assert (a_lo, a_hi) == (i_lo, i_hi)
    _CTXS["1.0"] = (vctx_full, i_lo, i_hi)
    _CTXS["0.25"] = (build_dose_context(0.25), i_lo, i_hi)

    log_alpha = np.asarray(vctx_full.gctx.cl_ctx.log_alpha, dtype=np.float64)
    alpha_tilt = float(-n_alpha * (log_alpha[i_hi] - log_alpha[i_lo]) / dh)

    tasks = [("1.0", s) for s in seeds] + [("0.25", s) for s in seeds]
    print(f"running {len(tasks)} tasks ...", flush=True)
    with mp.get_context("fork").Pool(processes=min(8, mp.cpu_count())) as pool:
        rows = pool.map(_seed_task, tasks)

    def _stats(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "se": float(arr.std(ddof=1) / np.sqrt(arr.size))}

    results: dict[str, Any] = {
        "note": (
            "A-FULL candidate pre-measurement: venue tilt of the correct-form estimator "
            "variants at h_true, mirror geometry, 15 MN0X seed replays. T_cand = "
            "candidate-sum tilt (no alpha); T_paired = T_cand + alpha_tilt (the coded "
            "-N ln alpha added analytically). PRESENTED, NOT ADJUDICATED."
        ),
        "seeds": seeds,
        "alpha_tilt_numeric": alpha_tilt,
        "per_seed_rows": rows,
        "summary": {},
    }
    for dose in ("1.0", "0.25"):
        drows = [r for r in rows if r["dose"] == dose]
        variants = ("full_d", "full_e")
        blk: dict[str, Any] = {}
        for v in variants:
            cand = _stats([r[f"T_cand_{v}"] for r in drows])
            blk[v] = {
                "T_cand": cand,
                "T_paired_with_alpha": {"mean": cand["mean"] + alpha_tilt, "se": cand["se"]},
            }
        results["summary"][dose] = blk
        for v in variants:
            b = blk[v]
            print(
                f"dose {dose} {v}: T_cand {b['T_cand']['mean']:+.1f}±{b['T_cand']['se']:.1f}"
                f" | paired-with-alpha {b['T_paired_with_alpha']['mean']:+.1f}",
                flush=True,
            )

    out = RESULTS_DIR / "L4_AFULL_PREMEASURE_D_output.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
