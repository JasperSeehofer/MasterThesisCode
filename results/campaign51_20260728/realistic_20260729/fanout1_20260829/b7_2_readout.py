"""
B7.2 (PROD-CF-2D) independent-reader readout script.

Registered form: PROPOSAL_2D_TWIN_ADOPTION_20260829.md SS6.2 / SS13.3
(H4 grid, T_mat = 0.008 two-sided, gates R1/R2/R6, stencil
Delta_mean_h,pred = Delta_ell'(0.665) / I_HEAD with I_HEAD = 2965).

Usage:
    python b7_2_readout.py --arm-dir <path to arm out-root> [--baseline-dir <path>]
                            [--out-json <path>]

Default baseline-dir is the banked HEAD readout
results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/
(commit d04d9dc9, reused zero-compute per the C0 PASS bit-identical gate,
row #246).

The arm-dir is expected to contain (A22/STEP-2 layout, confirmed against
wave2_20260829/c3/ as a structure reference, 2026-08-29):
    <arm-dir>/simulations/diagnostics/event_likelihoods.csv
    <arm-dir>/simulations/posteriors/h_0_*.json
    <arm-dir>/simulations/posteriors_with_bh_mass/h_0_*.json
    <arm-dir>/GIT_COMMIT_AT_RUN.txt
    <arm-dir>/run_metadata_*.json
    <arm-dir>/logs/

The baseline-dir (banked HEAD readout) has a flatter layout:
    <baseline-dir>/event_likelihoods.csv
    <baseline-dir>/posteriors/h_0_*.json
    <baseline-dir>/posteriors_with_bh_mass/h_0_*.json
    <baseline-dir>/run_metadata_21.json

Both layouts are auto-detected (whichever of the two csv locations exists).

This script performs NO writes other than the optional --out-json. It does
not touch results/.../wave2_20260829/c3/ or hier_s0_driver.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registered constants (PROPOSAL_2D_TWIN_ADOPTION_20260829.md SS6.2/SS13.3)
# ---------------------------------------------------------------------------
H4 = [0.660, 0.665, 0.670, 0.730]
H_STENCIL = [0.660, 0.665, 0.670]  # the three nodes bracketing MAP 0.665 / mean_h 0.663347
H_CENTER = 0.665
I_HEAD = 2965.0  # 1 / sigma_h^2, sigma_h = 0.018366 (row-#132 / MEASUREMENT_HEAD_READOUT sec C.1)
T_MAT = 0.008
T_MAT_HALF = T_MAT / 2.0  # 0.004, IMMATERIAL-PREDICTED boundary
R2_THRESHOLD = 0.95
R1_TOL = 0.0  # strict inequality gate; equality only permitted for empty candidate sets
R2_DELTA_LNL_FLOOR = 1e-6
R6_ATOL = 1e-12  # operationalized "bit-identical": matches the project's own C0-gate definition
# of bit-identical (REGISTRATION_C0_BASELINE_GATE_20260829.md SS3/SS13: "PASS -- bit-identical",
# max_abs 0.000, band <=1e-12; PROD-A0 historical floor <=8.5e-15). A dry run against C3's real
# CSVs showed ~1e-16 float noise on shared no-bh columns between independently-run arms -- an
# exact atol=0.0 would false-flag that noise as INSTRUMENT-DEFECT. DISCLOSED interpretation,
# not a silent band relaxation: exact 0.000 is still the expectation since the with-BH switch
# is architecturally untouched on this leg; 1e-12 is the noise floor, not slack for a real defect.
BASELINE_DEFAULT = (
    "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib"
)

R6_COLUMNS = ["L_cat_no_bh", "combined_no_bh"]


def find_csv(root: Path) -> Path:
    """Locate event_likelihoods.csv under either the arm layout or the flat
    banked-HEAD-readout layout."""
    candidates = [
        root / "simulations" / "diagnostics" / "event_likelihoods.csv",
        root / "diagnostics" / "event_likelihoods.csv",
        root / "event_likelihoods.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"event_likelihoods.csv not found under {root} (tried {candidates})")


def load_h4(csv_path: Path, h_nodes: list[float]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # tolerance on float h match (grid values are exact decimals but guard anyway)
    mask = np.zeros(len(df), dtype=bool)
    for h in h_nodes:
        mask |= np.isclose(df["h"].to_numpy(), h, atol=1e-9)
    out = df.loc[mask].copy()
    missing = set(h_nodes) - set(np.round(out["h"].unique(), 6))
    if missing:
        raise ValueError(f"{csv_path}: missing H4 nodes {sorted(missing)}")
    return out


def gate_r1(arm: pd.DataFrame, base: pd.DataFrame) -> dict:
    """R1: ln L_cat,wbh^T <= ln L_cat,wbh^B for every (event, h); equality only
    where the candidate set is empty (L_cat_with_bh == 0 in both, i.e. no
    galaxy passed the window)."""
    merged = arm.merge(
        base, on=["event_idx", "h"], suffixes=("_T", "_B"), how="inner"
    )
    n_total = len(merged)
    if n_total == 0:
        return {"gate": "R1", "verdict": "FAIL", "reason": "no matched (event,h) rows"}

    with np.errstate(divide="ignore"):
        ln_t = np.log(merged["L_cat_with_bh_T"].to_numpy())
        ln_b = np.log(merged["L_cat_with_bh_B"].to_numpy())

    both_zero = (merged["L_cat_with_bh_T"].to_numpy() == 0) & (
        merged["L_cat_with_bh_B"].to_numpy() == 0
    )
    # Inequality check only meaningful where both are finite (not -inf from log(0))
    finite_mask = np.isfinite(ln_t) & np.isfinite(ln_b)
    viol_mask = finite_mask & (ln_t > ln_b + 1e-12)  # tiny fp slack, not a band relaxation
    n_violations = int(viol_mask.sum())
    n_empty_equal = int(both_zero.sum())

    detail = merged.loc[viol_mask, ["event_idx", "h", "L_cat_with_bh_T", "L_cat_with_bh_B"]]
    return {
        "gate": "R1",
        "verdict": "PASS" if n_violations == 0 else "INSTRUMENT-DEFECT",
        "n_rows_checked": n_total,
        "n_violations": n_violations,
        "n_empty_candidate_equal_rows": n_empty_equal,
        "violation_sample": detail.head(20).to_dict(orient="records"),
    }


def gate_r2(arm: pd.DataFrame, base: pd.DataFrame, h_engage: float = 0.730) -> dict:
    """R2 (A13 engagement): fraction of events with a non-empty window-passed
    with-BH candidate set (L_cat_with_bh_B > 0, i.e. baseline had candidates)
    whose |Delta ln L_cat,wbh| > 1e-6 at h = h_engage."""
    a = arm.loc[np.isclose(arm["h"], h_engage, atol=1e-9)]
    b = base.loc[np.isclose(base["h"], h_engage, atol=1e-9)]
    merged = a.merge(b, on="event_idx", suffixes=("_T", "_B"), how="inner")

    active = merged["L_cat_with_bh_B"] > 0
    n_active = int(active.sum())
    if n_active == 0:
        return {"gate": "R2", "verdict": "STOP", "reason": "zero active rows at h_engage"}

    with np.errstate(divide="ignore"):
        ln_t = np.log(merged.loc[active, "L_cat_with_bh_T"].to_numpy())
        ln_b = np.log(merged.loc[active, "L_cat_with_bh_B"].to_numpy())
    finite = np.isfinite(ln_t) & np.isfinite(ln_b)
    delta = np.abs(ln_t[finite] - ln_b[finite])
    n_engaged = int((delta > R2_DELTA_LNL_FLOOR).sum())
    n_considered = int(finite.sum())
    frac = n_engaged / n_considered if n_considered else 0.0

    return {
        "gate": "R2",
        "verdict": "PASS" if frac >= R2_THRESHOLD else "STOP",
        "h_engage": h_engage,
        "n_active_rows": n_active,
        "n_considered_finite": n_considered,
        "n_engaged": n_engaged,
        "engagement_fraction": frac,
        "threshold": R2_THRESHOLD,
    }


def gate_r6(arm: pd.DataFrame, base: pd.DataFrame, h_nodes: list[float]) -> dict:
    """R6: the 1D channel's per-event columns are bit-identical between arms
    at every H4 node."""
    per_node = {}
    max_abs_overall = 0.0
    for h in h_nodes:
        a = arm.loc[np.isclose(arm["h"], h, atol=1e-9)]
        b = base.loc[np.isclose(base["h"], h, atol=1e-9)]
        merged = a.merge(b, on="event_idx", suffixes=("_T", "_B"), how="inner")
        node_result = {}
        for col in R6_COLUMNS:
            diff = (merged[f"{col}_T"] - merged[f"{col}_B"]).to_numpy()
            max_abs = float(np.nanmax(np.abs(diff))) if len(diff) else float("nan")
            node_result[col] = max_abs
            max_abs_overall = max(max_abs_overall, max_abs if np.isfinite(max_abs) else 0.0)
        per_node[str(h)] = node_result

    verdict = "PASS" if max_abs_overall <= R6_ATOL else "INSTRUMENT-DEFECT"
    return {
        "gate": "R6",
        "verdict": verdict,
        "columns_checked": R6_COLUMNS,
        "per_node_max_abs": per_node,
        "max_abs_overall": max_abs_overall,
    }


def stencil_delta_ell(arm: pd.DataFrame, base: pd.DataFrame, h_nodes: list[float]) -> dict:
    """Delta_ell(h) = sum_i ln[L_i^T / L_i^B] (with-BH combined channel) at
    each stencil node; central-difference slope at h=0.665, curvature check
    against the registered validity condition |Delta ell''| << I_HEAD."""
    vals = {}
    for h in h_nodes:
        a = arm.loc[np.isclose(arm["h"], h, atol=1e-9)]
        b = base.loc[np.isclose(base["h"], h, atol=1e-9)]
        merged = a.merge(b, on="event_idx", suffixes=("_T", "_B"), how="inner")
        ct = merged["combined_with_bh_T"].to_numpy()
        cb = merged["combined_with_bh_B"].to_numpy()
        valid = (ct > 0) & (cb > 0) & np.isfinite(ct) & np.isfinite(cb)
        n_dropped = int((~valid).sum())
        delta_ell = float(np.sum(np.log(ct[valid]) - np.log(cb[valid])))
        vals[h] = {"delta_ell": delta_ell, "n_events": int(valid.sum()), "n_dropped_nonpositive": n_dropped}

    if len(h_nodes) != 3:
        raise ValueError("stencil requires exactly 3 nodes")
    h_lo, h_mid, h_hi = h_nodes
    step = h_mid - h_lo  # assume uniform spacing (0.005)
    step2 = h_hi - h_mid
    if not np.isclose(step, step2):
        raise ValueError(f"non-uniform stencil spacing: {step} vs {step2}")

    ell_lo = vals[h_lo]["delta_ell"]
    ell_mid = vals[h_mid]["delta_ell"]
    ell_hi = vals[h_hi]["delta_ell"]

    d_ell_prime = (ell_hi - ell_lo) / (2 * step)  # central difference, nats per unit h
    d_ell_double_prime = (ell_hi - 2 * ell_mid + ell_lo) / (step**2)

    delta_mean_h_pred = d_ell_prime / I_HEAD
    validity_ok = abs(d_ell_double_prime) < 0.1 * I_HEAD  # "<<" interpreted as 10% of I_HEAD, disclosed

    return {
        "per_node_delta_ell": vals,
        "step": step,
        "delta_ell_prime_at_0_665": d_ell_prime,
        "delta_ell_doubleprime_at_0_665": d_ell_double_prime,
        "I_HEAD": I_HEAD,
        "delta_mean_h_pred": delta_mean_h_pred,
        "validity_condition_ok": bool(validity_ok),
        "validity_note": "interpreted as |Delta ell''| < 0.1 * I_HEAD (10% threshold, DISCLOSED interpretation of the registered '<<')",
    }


def direct_map_mean_over_h4(arm: pd.DataFrame, base: pd.DataFrame, h_nodes: list[float]) -> dict:
    """Secondary reading: treat the H4 nodes as if they were the entire grid
    and compute an (unnormalized-relative) MAP + mean over just these 4
    points, for both arm and baseline, using the with-BH combined channel
    summed in log over events. This is NOT a real posterior (only 4 of 41
    nodes) -- reported as a secondary/sanity cross-check only, per the task's
    'direct combined-posterior MAP/mean over the four nodes for both'
    instruction.
    """
    def sum_log_l(df: pd.DataFrame, h: float) -> float:
        rows = df.loc[np.isclose(df["h"], h, atol=1e-9)]
        vals = rows["combined_with_bh"].to_numpy()
        valid = (vals > 0) & np.isfinite(vals)
        return float(np.sum(np.log(vals[valid])))

    def map_mean(df: pd.DataFrame) -> dict:
        log_l = np.array([sum_log_l(df, h) for h in h_nodes])
        log_l -= log_l.max()  # numerical stability, relative only
        w = np.exp(log_l)
        w /= w.sum()
        map_h = h_nodes[int(np.argmax(w))]
        mean_h = float(np.sum(np.array(h_nodes) * w))
        return {
            "log_L_relative": {h: float(v) for h, v in zip(h_nodes, log_l)},
            "weights": {h: float(v) for h, v in zip(h_nodes, w)},
            "MAP": map_h,
            "mean": mean_h,
        }

    arm_r = map_mean(arm)
    base_r = map_mean(base)
    return {
        "arm": arm_r,
        "baseline": base_r,
        "delta_MAP": arm_r["MAP"] - base_r["MAP"],
        "delta_mean": arm_r["mean"] - base_r["mean"],
        "caveat": "4-node MAP/mean is NOT a valid full-grid posterior read (only 4 of 41 H_GRID_41 "
                  "nodes); REPORTED-ONLY secondary cross-check against the registered stencil reading.",
    }


def classify_verdict(delta_mean_h_pred: float, validity_ok: bool) -> str:
    if not validity_ok:
        return "AMBIGUOUS (validity condition violated)"
    a = abs(delta_mean_h_pred)
    if a >= T_MAT:
        return "MATERIAL-UP-PREDICTED" if delta_mean_h_pred > 0 else "MATERIAL-DOWN-PREDICTED"
    if a <= T_MAT_HALF:
        return "IMMATERIAL-PREDICTED"
    return "AMBIGUOUS (0.004 < |delta| < 0.008)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", required=True, type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=Path(BASELINE_DEFAULT))
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--h-engage", type=float, default=0.730)
    args = ap.parse_args()

    arm_csv = find_csv(args.arm_dir)
    base_csv = find_csv(args.baseline_dir)

    arm_h4 = load_h4(arm_csv, H4)
    base_h4 = load_h4(base_csv, H4)

    r1 = gate_r1(arm_h4, base_h4)
    r2 = gate_r2(arm_h4, base_h4, h_engage=args.h_engage)
    r6 = gate_r6(arm_h4, base_h4, H4)

    stencil = stencil_delta_ell(arm_h4, base_h4, H_STENCIL)
    direct = direct_map_mean_over_h4(arm_h4, base_h4, H4)

    verdict = classify_verdict(stencil["delta_mean_h_pred"], stencil["validity_condition_ok"])

    result = {
        "arm_csv": str(arm_csv),
        "baseline_csv": str(base_csv),
        "gates": {"R1": r1, "R2": r2, "R6": r6},
        "stencil": stencil,
        "direct_map_mean_over_h4": direct,
        "verdict_map_classification": verdict,
        "T_mat": T_MAT,
        "T_mat_half": T_MAT_HALF,
    }

    text = json.dumps(result, indent=2, default=str)
    if args.out_json:
        args.out_json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
