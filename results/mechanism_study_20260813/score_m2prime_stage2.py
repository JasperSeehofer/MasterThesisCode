#!/usr/bin/env python
"""Independent stage-2 readout scorer for A-M2' / A-NULL (PREREGISTRATION_M2PRIME_ABLATION.md).

Written and committed BEFORE the arm data exists (analysis-code-freeze discipline, thread 17).
Recomputes every statistic from the raw per-seed ``ln_post_1d`` / ``ln_post_2d`` vectors of the
two registered stage-2 arms; the stored per-seed scalars and the ``aggregate`` block are never
used as inputs, only as optional cross-checks.

Inputs (paths, all optional except MN0X which must be present):
    --am2p   AM2P_h0p730_results_seeds0_25.json   (A-M2', DS-M1 only)
    --anull  ANULL_h0p730_results_seeds0_15.json  (A-NULL, DS-M1 + DS-N1)
    --mn0x   MN0X_h0p730_results_seeds0_100.json  (committed paired reference for DS-N1)

Neither AM2P nor ANULL is required to exist at commit time: this script fails cleanly (reports
each arm's file as MISSING) rather than raising, and the branch is withheld until both are
present (§5 execution-completeness clause).

Emits ``score_m2prime_stage2_output.json`` next to this file and a compact stdout table.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
H_TRUE = 0.730
HPD_LEVELS = (0.50, 0.68, 0.90)

# ---- registered pins (PREREGISTRATION_M2PRIME_ABLATION.md §4) -------------
B_REF = 0.037250  # MN0X, N=100, committed (§4 DS-M1 TERM-INNOCENT anchor)
IN_BAND = 0.010  # DS-M1 TERM-OWNS/TERM-PARTIAL edge
DEFECT = 0.030  # DS-M1 TERM-PARTIAL/TERM-INNOCENT edge
NULL_TOL = 0.004  # DS-M1 TERM-INNOCENT proximity-to-b_ref edge
HPD90_OWNS = 0.60  # DS-M1 TERM-OWNS coverage conjunct
LN_1P7 = math.log(1.7)  # DS-N1 floor-aware integer shift law base
N_EVENTS_PIN = 982  # DS-N1 shift-law upper bound on m(h)
DS_N1_LAW_TOL = 1e-6  # nats, DS-N1 §4 tolerance
ANULL_SEED_BASE = 20260808 + 50000  # ARMS.md: base + 50000..50014, PAIRED with MN0X's first 15
ANULL_N_PAIRED = 15

DEFAULT_AM2P = HERE / "AM2P_h0p730_results_seeds0_25.json"
DEFAULT_ANULL = HERE / "ANULL_h0p730_results_seeds0_15.json"
DEFAULT_MN0X = HERE / "MN0X_h0p730_results_seeds0_100.json"


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# ── recomputation kernel (ports of score_mechanism_isolation.py's) ─────────
def posterior_readout(
    h_grid: NDArray[np.float64], ln_post: NDArray[np.float64]
) -> dict[str, float]:
    i = int(np.argmax(ln_post))
    return {
        "map_index": float(i),
        "map": float(h_grid[i]),
        "railed_low": float(i == 0),
        "railed_high": float(i == len(h_grid) - 1),
    }


def hpd_contains(
    h_grid: NDArray[np.float64], post: NDArray[np.float64], h_true: float, level: float
) -> bool:
    dh = np.gradient(h_grid)
    mass = post * dh
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = min(int(np.searchsorted(csum, level)), order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h_grid, post))
    return bool(p_true >= thresh)


def pp_readout(
    h_grid: NDArray[np.float64], ln_post: NDArray[np.float64], h_true: float
) -> dict[str, float]:
    p = np.exp(ln_post - float(np.max(ln_post)))
    norm_c = float(np.trapezoid(p, h_grid))
    post = p / norm_c
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(h_grid))])
    pit = float(np.interp(h_true, h_grid, cum))
    mean = float(np.trapezoid(post * h_grid, h_grid))
    var = float(np.trapezoid(post * h_grid**2, h_grid)) - mean**2
    r = {"pit": pit, "post_sd": math.sqrt(max(var, 0.0))}
    for lv in HPD_LEVELS:
        r[f"hpd{int(round(lv * 100))}"] = float(hpd_contains(h_grid, post, h_true, lv))
    return r


def ks_distance(pits: list[float]) -> float:
    q = np.sort(np.asarray(pits, dtype=np.float64))
    n = q.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.max(np.maximum(i / n - q, q - (i - 1.0) / n)))


def ks_critical(n: int) -> dict[str, float]:
    """Asymptotic Kolmogorov critical values D_alpha = c(alpha)/sqrt(n)."""
    if n <= 0:
        return {"D_95": float("nan"), "D_99": float("nan")}
    return {"D_95": 1.36 / math.sqrt(n), "D_99": 1.63 / math.sqrt(n)}


def classify(bias: float, hpd90: float, b_ref: float) -> str:
    if abs(bias) <= IN_BAND and hpd90 >= HPD90_OWNS:
        return "TERM-OWNS"
    if IN_BAND < abs(bias) < DEFECT:
        return "TERM-PARTIAL"
    if abs(bias) >= DEFECT and abs(bias - b_ref) <= NULL_TOL:
        return "TERM-INNOCENT"
    return "OTHER"


# ── DS-M1: per-arm, both channels ───────────────────────────────────────────
def score_ds_m1(d: dict[str, Any]) -> dict[str, Any]:
    grid = np.array(d["config"]["h_grid"], dtype=np.float64)
    recs = d["per_seed"]
    out: dict[str, Any] = {"n_records": len(recs), "seeds": sorted(r["seed"] for r in recs)}
    for ch in ("1d", "2d"):
        maps: list[float] = []
        pits: list[float] = []
        sds: list[float] = []
        rl: list[float] = []
        rh: list[float] = []
        hpd: dict[int, list[float]] = {50: [], 68: [], 90: []}
        nonfinite = 0
        for r in recs:
            ln = np.array(r[f"ln_post_{ch}"], dtype=np.float64)
            if not np.all(np.isfinite(ln)):
                nonfinite += 1
                continue
            pr = posterior_readout(grid, ln)
            pp = pp_readout(grid, ln, H_TRUE)
            maps.append(pr["map"])
            rl.append(pr["railed_low"])
            rh.append(pr["railed_high"])
            pits.append(pp["pit"])
            sds.append(pp["post_sd"])
            for lv in (50, 68, 90):
                hpd[lv].append(pp[f"hpd{lv}"])
        n = len(maps)
        if n == 0:
            out[ch] = {"n": 0, "status": "NO FINITE SEEDS"}
            continue
        bias_list = [m - H_TRUE for m in maps]
        mean_bias = sum(bias_list) / n
        sd = st.stdev(bias_list) if n > 1 and len(set(bias_list)) > 1 else 0.0
        se = sd / math.sqrt(n)
        hpd90_cov = sum(hpd[90]) / n
        ks = ks_distance(pits)
        crit = ks_critical(n)
        out[ch] = {
            "n": n,
            "nonfinite": nonfinite,
            "mean_bias": mean_bias,
            "se": se,
            "sd": sd,
            "post_sd_median": st.median(sds),
            "hpd50_cov": sum(hpd[50]) / n,
            "hpd68_cov": sum(hpd[68]) / n,
            "hpd90_cov": hpd90_cov,
            "pit_ks_D": ks,
            "pit_ks_D95": crit["D_95"],
            "pit_ks_D99": crit["D_99"],
            "pit_ks_status": "FAIL" if ks > crit["D_95"] else "PASS",
            "railed_low_frac": sum(rl) / n,
            "railed_high_frac": sum(rh) / n,
            "class": classify(mean_bias, hpd90_cov, B_REF),
        }
    return out


# ── DS-N1: A-NULL vs MN0X first-15 paired seeds ─────────────────────────────
def score_ds_n1(d_anull: dict[str, Any], d_mn0x: dict[str, Any]) -> dict[str, Any]:
    grid_a = list(d_anull["config"]["h_grid"])
    grid_m = list(d_mn0x["config"]["h_grid"])
    grid_match = grid_a == grid_m

    anull_recs = {r["seed"]: r for r in d_anull["per_seed"]}
    mn0x_recs = {r["seed"]: r for r in d_mn0x["per_seed"]}
    expected_seeds = [ANULL_SEED_BASE + j for j in range(ANULL_N_PAIRED)]

    rows: list[dict[str, Any]] = []
    all_index_eq = True
    all_law_ok = True
    n_paired = 0
    for seed in expected_seeds:
        ra = anull_recs.get(seed)
        rm = mn0x_recs.get(seed)
        row: dict[str, Any] = {
            "seed": seed,
            "anull_present": ra is not None,
            "mn0x_present": rm is not None,
        }
        if ra is None or rm is None:
            row["status"] = "MISSING"
            all_index_eq = False
            all_law_ok = False
            rows.append(row)
            continue
        n_paired += 1
        for ch in ("1d", "2d"):
            ln_a = np.array(ra[f"ln_post_{ch}"], dtype=np.float64)
            ln_m = np.array(rm[f"ln_post_{ch}"], dtype=np.float64)
            i_a = int(np.argmax(ln_a))
            i_m = int(np.argmax(ln_m))
            idx_eq = i_a == i_m
            delta = ln_a - ln_m
            m_arr = np.round(delta / LN_1P7)
            resid = np.abs(delta - m_arr * LN_1P7)
            law_ok = bool(
                np.all(resid <= DS_N1_LAW_TOL) and np.all((m_arr >= 0) & (m_arr <= N_EVENTS_PIN))
            )
            row[f"{ch}_map_index_anull"] = i_a
            row[f"{ch}_map_index_mn0x"] = i_m
            row[f"{ch}_index_eq"] = idx_eq
            row[f"{ch}_m_at_anull_map_index"] = int(m_arr[i_a])
            row[f"{ch}_m_at_mn0x_map_index"] = int(m_arr[i_m])
            row[f"{ch}_law_ok"] = law_ok
            row[f"{ch}_max_law_resid"] = float(resid.max())
            row[f"{ch}_m_min"] = int(m_arr.min())
            row[f"{ch}_m_max"] = int(m_arr.max())
            all_index_eq &= idx_eq
            all_law_ok &= law_ok
        row["status"] = "OK"
        rows.append(row)

    all_present = n_paired == ANULL_N_PAIRED
    ds_n1_pass = bool(all_present and grid_match and all_index_eq and all_law_ok)
    return {
        "grid_match": grid_match,
        "n_expected": ANULL_N_PAIRED,
        "n_paired": n_paired,
        "all_present": all_present,
        "all_index_eq": bool(all_index_eq),
        "all_law_ok": bool(all_law_ok),
        "status": "PASS" if ds_n1_pass else "FAIL",
        "rows": rows,
    }


# ── branch determination (§5, split-precedence + execution-completeness) ───
def determine_branch(
    am2p_present: bool,
    anull_present: bool,
    ds_m1_am2p: dict[str, Any] | None,
    ds_n1: dict[str, Any] | None,
    validity_ok: bool,
) -> dict[str, Any]:
    if not (am2p_present and anull_present):
        missing = []
        if not am2p_present:
            missing.append("A-M2' (AM2P)")
        if not anull_present:
            missing.append("A-NULL (ANULL)")
        return {
            "status": "NOT PRESENTED",
            "reason": f"execution-completeness clause (§5): missing arm(s): {', '.join(missing)}",
        }

    assert ds_m1_am2p is not None and ds_n1 is not None

    if ds_n1["status"] == "FAIL" or not validity_ok:
        return {
            "status": "PRESENTED, NOT ADJUDICATED",
            "branch": "1. STUDY-CONFOUNDED",
            "fired_by": "DS-N1 FAIL" if ds_n1["status"] == "FAIL" else "a §6 validity check failed",
        }

    cls_1d = ds_m1_am2p.get("1d", {}).get("class")
    cls_2d = ds_m1_am2p.get("2d", {}).get("class")

    if cls_1d != cls_2d:
        return {
            "status": "PRESENTED, NOT ADJUDICATED",
            "branch": "5. OTHER / SPLIT",
            "fired_by": f"split-precedence: 1D class {cls_1d} != 2D class {cls_2d}",
        }

    branch_map = {
        "TERM-OWNS": "2. M2'-OWNS",
        "TERM-PARTIAL": "3. M2'-PARTIAL",
        "TERM-INNOCENT": "4. M2'-INNOCENT",
    }
    branch = branch_map.get(cls_1d, "5. OTHER / SPLIT")
    return {
        "status": "PRESENTED, NOT ADJUDICATED",
        "branch": branch,
        "fired_by": f"A-M2' class {cls_1d} (both channels)",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--am2p", type=Path, default=DEFAULT_AM2P)
    ap.add_argument("--anull", type=Path, default=DEFAULT_ANULL)
    ap.add_argument("--mn0x", type=Path, default=DEFAULT_MN0X)
    ap.add_argument("--out", type=Path, default=HERE / "score_m2prime_stage2_output.json")
    args = ap.parse_args(argv)

    P = print
    out: dict[str, Any] = {"inputs": {}}

    if not args.mn0x.exists():
        P(f"FATAL: committed reference MN0X file not found: {args.mn0x}")
        return 2
    d_mn0x = load(args.mn0x)

    am2p_present = args.am2p.exists()
    anull_present = args.anull.exists()
    out["inputs"] = {
        "am2p_path": str(args.am2p),
        "am2p_present": am2p_present,
        "anull_path": str(args.anull),
        "anull_present": anull_present,
        "mn0x_path": str(args.mn0x),
        "mn0x_present": True,
    }

    P("=== INPUTS ===")
    P(
        f"  AM2P  {args.am2p.name}: {'FOUND' if am2p_present else 'MISSING (analysis-code-freeze: expected before data exists)'}"
    )
    P(
        f"  ANULL {args.anull.name}: {'FOUND' if anull_present else 'MISSING (analysis-code-freeze: expected before data exists)'}"
    )
    P(f"  MN0X  {args.mn0x.name}: FOUND (committed paired reference)")
    P()

    # -- MN0X cross-check only (never trusted as an input to any class edge) --
    ds_m1_mn0x = score_ds_m1(d_mn0x)
    b_mn0x = ds_m1_mn0x["1d"]["mean_bias"]
    P(
        "=== MN0X CROSS-CHECK (recomputed vs registered b_ref=+0.037250; not an input to any class) ==="
    )
    P(f"  recomputed 1D bias = {b_mn0x:+.6f}  |delta from b_ref| = {abs(b_mn0x - B_REF):.2e}")
    out["mn0x_cross_check"] = ds_m1_mn0x

    ds_m1_am2p = None
    ds_m1_anull = None

    if am2p_present:
        d_am2p = load(args.am2p)
        ds_m1_am2p = score_ds_m1(d_am2p)
        out["ds_m1_am2p"] = ds_m1_am2p
        P("=== DS-M1: A-M2' ===")
        for ch in ("1d", "2d"):
            c = ds_m1_am2p[ch]
            if c.get("n", 0) == 0:
                P(f"  {ch}: NO FINITE SEEDS")
                continue
            P(
                f"  {ch}: N={c['n']} bias={c['mean_bias']:+.6f} SE={c['se']:.6f} "
                f"hpd50/68/90={c['hpd50_cov']:.3f}/{c['hpd68_cov']:.3f}/{c['hpd90_cov']:.3f} "
                f"KS_D={c['pit_ks_D']:.4f} ({c['pit_ks_status']}) post_sd_med={c['post_sd_median']:.6f} "
                f"rails={c['railed_low_frac'] + c['railed_high_frac']:.3f} nonfin={c['nonfinite']} "
                f"-> {c['class']}"
            )
        P()
    else:
        out["ds_m1_am2p"] = None

    if anull_present:
        d_anull = load(args.anull)
        ds_m1_anull = score_ds_m1(d_anull)
        out["ds_m1_anull"] = ds_m1_anull
        P("=== DS-M1: A-NULL (also enters DS-N1 below) ===")
        for ch in ("1d", "2d"):
            c = ds_m1_anull[ch]
            if c.get("n", 0) == 0:
                P(f"  {ch}: NO FINITE SEEDS")
                continue
            P(
                f"  {ch}: N={c['n']} bias={c['mean_bias']:+.6f} SE={c['se']:.6f} "
                f"hpd50/68/90={c['hpd50_cov']:.3f}/{c['hpd68_cov']:.3f}/{c['hpd90_cov']:.3f} "
                f"KS_D={c['pit_ks_D']:.4f} ({c['pit_ks_status']}) post_sd_med={c['post_sd_median']:.6f} "
                f"rails={c['railed_low_frac'] + c['railed_high_frac']:.3f} nonfin={c['nonfinite']} "
                f"-> {c['class']}"
            )
        P()

        ds_n1 = score_ds_n1(d_anull, d_mn0x)
        out["ds_n1"] = ds_n1
        P("=== DS-N1: A-NULL vs MN0X first-15 paired seeds ===")
        P(
            f"  grid_match={ds_n1['grid_match']} n_paired={ds_n1['n_paired']}/{ds_n1['n_expected']} "
            f"all_index_eq={ds_n1['all_index_eq']} all_law_ok={ds_n1['all_law_ok']} -> {ds_n1['status']}"
        )
        for row in ds_n1["rows"]:
            if row["status"] != "OK":
                P(f"    seed {row['seed']}: {row['status']}")
                continue
            P(
                f"    seed {row['seed']}: 1d idx_eq={row['1d_index_eq']} "
                f"m@map={row['1d_m_at_anull_map_index']} law_ok={row['1d_law_ok']} | "
                f"2d idx_eq={row['2d_index_eq']} m@map={row['2d_m_at_anull_map_index']} "
                f"law_ok={row['2d_law_ok']}"
            )
        P()
    else:
        out["ds_m1_anull"] = None
        out["ds_n1"] = None

    validity_ok = True
    if ds_m1_mn0x["1d"].get("n", 0) > 0:
        validity_ok = abs(b_mn0x - B_REF) <= 1e-9  # exact-record cross-check on committed data

    branch = determine_branch(
        am2p_present, anull_present, ds_m1_am2p, out.get("ds_n1"), validity_ok
    )
    out["branch"] = branch
    P("=== BRANCH (§5; PRESENTED, NOT ADJUDICATED — author call only) ===")
    P(f"  status: {branch['status']}")
    if "branch" in branch:
        P(f"  branch: {branch['branch']}")
    P(f"  reason: {branch.get('reason', branch.get('fired_by'))}")
    P()

    args.out.write_text(json.dumps(out, indent=1))
    P(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
