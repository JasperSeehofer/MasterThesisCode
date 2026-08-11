"""Route-1 offline convergence study on the harvested query distribution.

PERF/MEASUREMENT ONLY — no re-run of the instrument, no edits under
``master_thesis_code/``. Reads ``route1_harvest.npz`` (produced by
``harvest_route1.py``) and for every harvested z-node recomputes the g_i
integrand contraction

    g_i = INTEGRAL dx_M N(x_M; mu_cond, sigma_cond) * phi_x(x_M)
        = (1/sqrt(pi)) SUM_j w_j * phi_x(mu_cond + sqrt(2) sigma_cond t_j) * scale

at Gauss-Hermite order n in {8, 12, 16, 24, 64, 128, 256} using
``numpy.polynomial.hermite.hermgauss`` and the ACTUAL
``dark_mass_density_per_mass`` (default/affine path — the as-committed
production convention) — never a re-typed phi.

Produces four tables (Method §2-5 of ROUTE1_STUDY.md) as CSV/JSON under this
directory, plus the numbers folded into ROUTE1_STUDY.md by hand/second pass.

Usage (from repo root):
    uv run python results/venue_transfer_20260811/perf/route1_study/analyze_route1.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import numpy.polynomial.hermite as np_herm
from scipy.special import roots_legendre

from master_thesis_code.bayesian_inference.bayesian_statistics import (
    dark_mass_density_per_mass,
)
from master_thesis_code.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN

STUDY_DIR = Path(__file__).resolve().parent
BREAKPOINTS = (M_SOURCE_FRAME_MIN, 1.0e5, M_SOURCE_FRAME_MAX)  # {1e4, 1e5, 1e7}

N_CANDIDATES = [8, 12, 16, 24, 64, 128, 256]
N_REFERENCE = 256
T_TOL_CANDIDATES = [4, 5, 6]
N_LOW_CANDIDATES = [8, 12, 16, 24]


_ROW_CHUNK = 200_000  # keeps (chunk, n=256) intermediate ~= 400 MB


def g_i_hermgauss(
    mu_cond: np.ndarray, sigma_cond: np.ndarray, scale: np.ndarray, n: int
) -> np.ndarray:
    """g_i at Gauss-Hermite order ``n`` via ``numpy.polynomial.hermite.hermgauss``.

    Chunked over rows to bound peak memory (10M rows x n=256 nodes would be
    ~20GB unchunked).
    """
    t_nodes, t_weights = np_herm.hermgauss(n)
    n_rows = mu_cond.size
    out = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, n_rows)
        mu_c = mu_cond[start:end]
        sig_c = sigma_cond[start:end]
        sc_c = scale[start:end]
        x_M = mu_c[:, None] + math.sqrt(2.0) * sig_c[:, None] * t_nodes[None, :]
        M_source = x_M * sc_c[:, None]
        phi_x = dark_mass_density_per_mass(M_source) * sc_c[:, None]
        out[start:end] = (phi_x @ t_weights) / math.sqrt(math.pi)
    return out


def g_i_split_gauss_legendre(
    mu_cond: np.ndarray, sigma_cond: np.ndarray, scale: np.ndarray, n_per_segment: int = 64
) -> np.ndarray:
    """Reference for straddling nodes: split-interval Gauss-Legendre in x_M.

    Splits the x_M integral at the interior breakpoints (mapped through
    ``x = M/scale``) into sub-intervals and integrates each with
    ``n_per_segment`` Gauss-Legendre nodes, writing the Gaussian weight
    N(x_M; mu_cond, sigma_cond) explicitly (no e^{-t^2} substitution — this
    is the whole point: it does not inherit the Hermite kernel's own
    quadrature convention, so it is an independent check on the kink).

    Integration domain is truncated to +-12 sigma (safely beyond both
    Hermite reference orders' effective support) intersected with the
    physical mass band [M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX]/scale.
    """
    xi, wi = roots_legendre(n_per_segment)
    n_rows = mu_cond.size
    out = np.zeros(n_rows, dtype=np.float64)
    lo_x = M_SOURCE_FRAME_MIN / scale
    hi_x = M_SOURCE_FRAME_MAX / scale
    dom_lo = np.maximum(mu_cond - 12.0 * sigma_cond, lo_x)
    dom_hi = np.minimum(mu_cond + 12.0 * sigma_cond, hi_x)
    kink_x = BREAKPOINTS[1] / scale  # interior breakpoint only
    for i in range(n_rows):
        a, b = dom_lo[i], dom_hi[i]
        if not (b > a):
            continue
        edges = sorted({a, b} | ({kink_x[i]} if a < kink_x[i] < b else set()))
        total = 0.0
        for seg_lo, seg_hi in zip(edges[:-1], edges[1:], strict=True):
            half = 0.5 * (seg_hi - seg_lo)
            mid = 0.5 * (seg_hi + seg_lo)
            x_seg = mid + half * xi
            gauss_w = np.exp(-0.5 * ((x_seg - mu_cond[i]) / sigma_cond[i]) ** 2) / (
                sigma_cond[i] * math.sqrt(2.0 * math.pi)
            )
            phi_x = dark_mass_density_per_mass(x_seg * scale[i]) * scale[i]
            total += half * float(np.sum(wi * gauss_w * phi_x))
        out[i] = total
    return out


def straddles(
    mu_cond: np.ndarray, sigma_cond: np.ndarray, scale: np.ndarray, t_tol: float
) -> np.ndarray:
    """Straddle test per the approved spec: does
    ``[mu_cond - sqrt(2)*sigma_cond*t_tol, mu_cond + sqrt(2)*sigma_cond*t_tol] * scale``
    cross any of ``{1e4, 1e5, 1e7}``? ``scale`` is per-row and strictly
    positive, so the window is built in x_M space and then multiplied by
    ``scale`` (row-wise) before comparing against the fixed mass breakpoints
    — NOT compared directly in dimensionless x_M units (that was a bug in an
    earlier revision of this script, caught before results were reported).
    """
    half_width = math.sqrt(2.0) * sigma_cond * t_tol
    win_lo_M = (mu_cond - half_width) * scale
    win_hi_M = (mu_cond + half_width) * scale
    out = np.zeros(mu_cond.size, dtype=bool)
    for bp_M in BREAKPOINTS:
        out |= (win_lo_M <= bp_M) & (bp_M <= win_hi_M)
    return out


def main() -> None:
    data = np.load(STUDY_DIR / "route1_harvest.npz")
    det_M_z = data["det_M_z"]
    proj = data["proj_d_L_to_M"]
    sigma_cond_M = data["sigma_cond_M"]
    z_node = data["z_node"]
    d_L_fraction = data["d_L_fraction"]

    n_rows = z_node.size
    print(f"[analyze] {n_rows} harvested z-nodes", flush=True)

    scale = det_M_z / (1.0 + z_node)
    mu_cond = 1.0 + proj * (d_L_fraction - 1.0)
    sigma_cond = sigma_cond_M

    # --- harvest statistics ---
    def _quantiles(a: np.ndarray) -> dict[str, float]:
        qs = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
        vals = np.quantile(a, qs)
        return dict(zip(["min", "p1", "p10", "p50", "p90", "p99", "max"], vals.tolist(), strict=True))

    harvest_stats = {
        "n_calls": int(data["n_calls"][0]),
        "n_total_rows_before_subsample": int(data["n_total_rows_before_subsample"][0]),
        "n_rows_analyzed": n_rows,
        "subsampled": bool(data["subsampled"][0]),
        "wall_seed_s": float(data["wall_seed_s"][0]),
        "n_events_cap": int(data["n_events_cap"][0]),
        "seed": int(data["seed"][0]),
        "sigma_cond_M_quantiles": _quantiles(sigma_cond_M),
        "scale_quantiles": _quantiles(scale),
        "mu_cond_quantiles": _quantiles(mu_cond),
    }
    with open(STUDY_DIR / "harvest_stats.json", "w") as f:
        json.dump(harvest_stats, f, indent=2, sort_keys=True)
    print(f"[analyze] harvest stats -> harvest_stats.json: {harvest_stats}", flush=True)

    # --- reference at n=256 ---
    print("[analyze] computing reference g_i at n=256 ...", flush=True)
    g_ref = g_i_hermgauss(mu_cond, sigma_cond, scale, N_REFERENCE)

    # --- g_i at all candidate n ---
    g_by_n: dict[int, np.ndarray] = {}
    for n in N_CANDIDATES:
        print(f"[analyze] computing g_i at n={n} ...", flush=True)
        g_by_n[n] = g_i_hermgauss(mu_cond, sigma_cond, scale, n)

    def rel_err(g: np.ndarray, ref: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.abs(ref), 1e-300)
        return np.abs(g - ref) / denom

    # --- straddle masks per t_tol ---
    straddle_masks: dict[int, np.ndarray] = {}
    for t_tol in T_TOL_CANDIDATES:
        straddle_masks[t_tol] = straddles(mu_cond, sigma_cond, scale, float(t_tol))
        frac = float(np.mean(straddle_masks[t_tol]))
        print(f"[analyze] t_tol={t_tol}: fallback_fraction={frac:.6f}", flush=True)

    # --- Table 2: convergence sweep, split by straddling ---
    table2_rows = []
    for n in N_CANDIDATES:
        errs = rel_err(g_by_n[n], g_ref)
        for t_tol in T_TOL_CANDIDATES:
            mask = straddle_masks[t_tol]
            for label, sel in (("straddling", mask), ("non_straddling", ~mask)):
                sub = errs[sel]
                if sub.size == 0:
                    max_e = p999 = float("nan")
                else:
                    max_e = float(np.max(sub))
                    p999 = float(np.quantile(sub, 0.999))
                table2_rows.append(
                    {
                        "n": n,
                        "t_tol": t_tol,
                        "group": label,
                        "n_rows": int(sub.size),
                        "max_rel_err_vs_n256": max_e,
                        "p999_rel_err_vs_n256": p999,
                    }
                )
    with open(STUDY_DIR / "table2_convergence_sweep.json", "w") as f:
        json.dump(table2_rows, f, indent=2, sort_keys=True)

    # --- avg node count and fallback fraction per (n_low, t_tol) ---
    table_nodecount_rows = []
    for t_tol in T_TOL_CANDIDATES:
        frac = float(np.mean(straddle_masks[t_tol]))
        for n_low in N_LOW_CANDIDATES:
            avg_nodes = frac * 64.0 + (1.0 - frac) * n_low
            table_nodecount_rows.append(
                {
                    "n_low": n_low,
                    "t_tol": t_tol,
                    "fallback_fraction": frac,
                    "avg_nodes_per_z_node": avg_nodes,
                }
            )
    with open(STUDY_DIR / "table_nodecount.json", "w") as f:
        json.dump(table_nodecount_rows, f, indent=2, sort_keys=True)

    # --- Table 3: n=64 self-convergence at straddling nodes (any t_tol=4,
    # the most inclusive candidate, defines "straddling" for this table;
    # report per t_tol too since the straddling SET depends on it) ---
    table3_rows = []
    g64 = g_by_n[64]
    for t_tol in T_TOL_CANDIDATES:
        mask = straddle_masks[t_tol]
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            table3_rows.append(
                {"t_tol": t_tol, "n_rows": 0, "vs_n256": None, "vs_split_gl": None}
            )
            continue
        err_vs_256 = rel_err(g64[idx], g_ref[idx])
        print(
            f"[analyze] t_tol={t_tol}: computing split-GL reference for "
            f"{idx.size} straddling rows ...",
            flush=True,
        )
        g_split = g_i_split_gauss_legendre(
            mu_cond[idx], sigma_cond[idx], scale[idx], n_per_segment=64
        )
        err_vs_split = rel_err(g64[idx], g_split)
        table3_rows.append(
            {
                "t_tol": t_tol,
                "n_rows": int(idx.size),
                "vs_n256": {
                    "max": float(np.max(err_vs_256)),
                    "median": float(np.median(err_vs_256)),
                    "p99": float(np.quantile(err_vs_256, 0.99)),
                },
                "vs_split_gl": {
                    "max": float(np.max(err_vs_split)),
                    "median": float(np.median(err_vs_split)),
                    "p99": float(np.quantile(err_vs_split, 0.99)),
                },
            }
        )
    with open(STUDY_DIR / "table3_n64_self_convergence.json", "w") as f:
        json.dump(table3_rows, f, indent=2, sort_keys=True)

    # --- Table 4: projected speedup ---
    table4_rows = []
    for row in table_nodecount_rows:
        avg_nodes = row["avg_nodes_per_z_node"]
        speedup = 1.0 / (0.089 + 0.911 * (avg_nodes / 64.0))
        table4_rows.append({**row, "projected_seed_wall_speedup": speedup})
    with open(STUDY_DIR / "table4_projected_speedup.json", "w") as f:
        json.dump(table4_rows, f, indent=2, sort_keys=True)

    # --- Table 5: acceptance scan ---
    # non-straddling max rel error < 1e-12 (vs n=256)
    # overall (with fallback) max rel error vs the n=64 CONVENTION < 1e-10
    table5_rows = []
    for t_tol in T_TOL_CANDIDATES:
        mask = straddle_masks[t_tol]
        for n_low in N_LOW_CANDIDATES:
            g_low = g_by_n[n_low]
            g_mixed = np.where(mask, g64, g_low)  # fallback applies n=64 on straddle
            err_non_straddle_vs_256 = rel_err(g_low[~mask], g_ref[~mask])
            err_overall_vs_conv64 = rel_err(g_mixed, g64)
            max_non_straddle = (
                float(np.max(err_non_straddle_vs_256)) if np.any(~mask) else 0.0
            )
            max_overall = float(np.max(err_overall_vs_conv64))
            pass_non_straddle = max_non_straddle < 1e-12
            pass_overall = max_overall < 1e-10
            table5_rows.append(
                {
                    "n_low": n_low,
                    "t_tol": t_tol,
                    "max_rel_err_non_straddling_vs_n256": max_non_straddle,
                    "max_rel_err_overall_vs_n64_convention": max_overall,
                    "pass_non_straddling_lt_1e12": pass_non_straddle,
                    "pass_overall_lt_1e10": pass_overall,
                    "accept": pass_non_straddle and pass_overall,
                }
            )
    with open(STUDY_DIR / "table5_acceptance_scan.json", "w") as f:
        json.dump(table5_rows, f, indent=2, sort_keys=True)

    winners = [r for r in table5_rows if r["accept"]]
    print(f"[analyze] acceptance-scan winners: {len(winners)}", flush=True)
    for w in winners:
        speedup_row = next(
            r
            for r in table4_rows
            if r["n_low"] == w["n_low"] and r["t_tol"] == w["t_tol"]
        )
        print(
            f"    n_low={w['n_low']} t_tol={w['t_tol']} "
            f"fallback_frac={speedup_row['fallback_fraction']:.6f} "
            f"avg_nodes={speedup_row['avg_nodes_per_z_node']:.3f} "
            f"speedup={speedup_row['projected_seed_wall_speedup']:.4f}x",
            flush=True,
        )

    print("[analyze] done. All tables written under route1_study/.", flush=True)


if __name__ == "__main__":
    main()
