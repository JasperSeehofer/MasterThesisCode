"""Mechanical readout script for the pre-registered N-2 sel-1d counterfactual.

Reads the two new (1d-toggled) event_likelihoods.csv files and scores them
against the pre-registration in
results/run_20260804_postfix/gate_vii/PREREGISTRATION_N2_SEL1D.md.

No interpretation beyond the mechanical branch scoring is performed here;
this script only computes numbers and writes them to readout.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
N2_DIR = REPO / "results" / "run_20260805_n2sel1d"
POSTFIX_DIR = REPO / "results" / "run_20260804_postfix"

VENUES = {
    "iiib": {
        "n2_csv": N2_DIR / "iiib" / "diagnostics" / "event_likelihoods.csv",
        "postfix_csv": POSTFIX_DIR / "iiib" / "diagnostics" / "event_likelihoods.csv",
        "metadata": N2_DIR / "iiib" / "run_metadata_0.json",
    },
    "joint_r1": {
        "n2_csv": N2_DIR / "joint_r1" / "diagnostics" / "event_likelihoods.csv",
        "postfix_csv": POSTFIX_DIR / "joint_r1" / "diagnostics" / "event_likelihoods.csv",
        "metadata": N2_DIR / "joint_r1" / "run_metadata_0.json",
    },
}

EXPECTED_ROWS = 65108  # 41 h * 1588 events
EXPECTED_H_COUNT = 41
EXPECTED_EVENT_COUNT = 1588

P2_BAND = (10.0, 30.0)  # nats/h
P3_THRESHOLD = 0.90

NULL_COLUMNS_2D = ["combined_with_bh", "L_cat_with_bh", "B_num_wbh"]
NULL_COLUMNS_CATALOGUE = ["L_cat_no_bh", "L_cat_with_bh"]
NULL_COLUMNS_SELECTION = ["w_G", "w_tilde_G", "alpha_G_phi", "r_Malm", "D_tilde_phi"]

NONNULL_COLUMNS = ["B_num", "combined_no_bh"]


def load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n2_df = pd.read_csv(cfg["n2_csv"])
    postfix_df = pd.read_csv(cfg["postfix_csv"])
    with open(cfg["metadata"]) as f:
        metadata = json.load(f)
    return n2_df, postfix_df, metadata


def fingerprints(n2_df: pd.DataFrame, postfix_df: pd.DataFrame, metadata: dict) -> dict:
    args = metadata.get("cli_args") or metadata.get("args") or metadata
    return {
        "n2_csv_rows": int(len(n2_df)),
        "n2_csv_rows_match_expected": len(n2_df) == EXPECTED_ROWS,
        "n2_n_h": int(n2_df["h"].nunique()),
        "n2_n_events": int(n2_df["event_idx"].nunique()),
        "postfix_csv_rows": int(len(postfix_df)),
        "postfix_n_h": int(postfix_df["h"].nunique()),
        "postfix_n_events": int(postfix_df["event_idx"].nunique()),
        "metadata_top_level_keys": list(metadata.keys()),
        "metadata_selection_flag_present": _find_key(metadata, "selection_in_completion_numerator"),
        "metadata_freeze_g_frac_ref_h_present": _find_key(metadata, "freeze_g_frac_ref_h"),
    }


def _find_key(d: dict, key: str, path: str = "") -> str | None:
    """Recursively search a (possibly nested) dict for `key`, return path=value."""
    if key in d:
        return f"{path}{key}={d[key]!r}"
    for k, v in d.items():
        if isinstance(v, dict):
            found = _find_key(v, key, path=f"{path}{k}.")
            if found is not None:
                return found
    return None


def merged_no_bh(n2_df: pd.DataFrame, postfix_df: pd.DataFrame) -> pd.DataFrame:
    m = n2_df.merge(
        postfix_df,
        on=["event_idx", "h"],
        suffixes=("_1d", "_off"),
        how="inner",
    )
    m["ln_combined_no_bh_1d"] = np.log(m["combined_no_bh_1d"].astype(float).clip(lower=1e-300))
    m["ln_combined_no_bh_off"] = np.log(m["combined_no_bh_off"].astype(float).clip(lower=1e-300))
    m["delta_ln"] = m["ln_combined_no_bh_1d"] - m["ln_combined_no_bh_off"]
    return m


def p2_tilt(m: pd.DataFrame) -> dict:
    """Sigma_events ln(combined_no_bh) tilt, chord over full grid and central diff at 0.73."""
    per_h_sum_1d = m.groupby("h")["ln_combined_no_bh_1d"].sum().sort_index()
    per_h_sum_off = m.groupby("h")["ln_combined_no_bh_off"].sum().sort_index()
    per_h_sum_delta = per_h_sum_1d - per_h_sum_off

    h_grid = per_h_sum_delta.index.to_numpy(dtype=float)
    h_lo, h_hi = float(h_grid.min()), float(h_grid.max())

    chord = float(
        (per_h_sum_delta.loc[h_hi] - per_h_sum_delta.loc[h_lo]) / (h_hi - h_lo)
    )

    h_sorted = np.sort(h_grid)
    i73 = int(np.argmin(np.abs(h_sorted - 0.73)))
    if 0 < i73 < h_sorted.size - 1:
        h_m, h_p = h_sorted[i73 - 1], h_sorted[i73 + 1]
        central_diff = float(
            (per_h_sum_delta.loc[h_p] - per_h_sum_delta.loc[h_m]) / (h_p - h_m)
        )
    else:
        central_diff = float("nan")
        h_m = h_p = float("nan")

    return {
        "h_grid_bounds": [h_lo, h_hi],
        "chord_nats_per_h": chord,
        "chord_in_band": P2_BAND[0] <= chord <= P2_BAND[1],
        "central_diff_nats_per_h_at_073": central_diff,
        "central_diff_in_band": P2_BAND[0] <= central_diff <= P2_BAND[1],
        "central_diff_neighbors": [float(h_m), float(h_p)],
        "sum_delta_ln_level_at_073": float(
            per_h_sum_delta.loc[h_sorted[i73]]
        ),
        "sum_delta_ln_by_h": {str(h): float(v) for h, v in per_h_sum_delta.items()},
        "band": list(P2_BAND),
    }


def p3_sign_coherence(m: pd.DataFrame) -> dict:
    """Fraction of events with positive per-event delta-slope (central diff at h=0.73)."""
    h_grid = np.sort(m["h"].unique())
    i73 = int(np.argmin(np.abs(h_grid - 0.73)))
    if not (0 < i73 < h_grid.size - 1):
        return {"error": "h=0.73 not an interior grid point"}
    h_m, h_p = float(h_grid[i73 - 1]), float(h_grid[i73 + 1])

    piv = m.pivot_table(index="event_idx", columns="h", values="delta_ln")
    slope = (piv[h_p] - piv[h_m]) / (h_p - h_m)
    n_events = int(slope.notna().sum())
    n_positive = int((slope > 0).sum())
    frac_positive = float(n_positive / n_events) if n_events else float("nan")

    return {
        "neighbors": [h_m, h_p],
        "n_events": n_events,
        "n_positive": n_positive,
        "frac_positive": frac_positive,
        "meets_threshold": frac_positive >= P3_THRESHOLD if n_events else False,
        "threshold": P3_THRESHOLD,
        "slope_summary": {
            "min": float(slope.min()),
            "q05": float(slope.quantile(0.05)),
            "median": float(slope.median()),
            "mean": float(slope.mean()),
            "q95": float(slope.quantile(0.95)),
            "max": float(slope.max()),
        },
    }


def p6_map_1d(n2_df: pd.DataFrame) -> dict:
    """1D MAP (argmax Sigma ln combined_no_bh) in the counterfactual run."""
    grouped = n2_df.groupby("h")["combined_no_bh"].apply(
        lambda s: np.log(s.astype(float).clip(lower=1e-300)).sum()
    )
    grouped = grouped.sort_index()
    h_grid = grouped.index.to_numpy(dtype=float)
    argmax_h = float(grouped.idxmax())
    argmax_val = float(grouped.max())
    railed = bool(np.isclose(argmax_h, 0.600, atol=1e-6))

    top5 = grouped.sort_values(ascending=False).head(5)

    # delta-ln from 0.600 to the next few h points -- how hard is the rail
    h_sorted = np.sort(h_grid)
    idx600 = int(np.argmin(np.abs(h_sorted - 0.600)))
    rail_hardness = []
    for k in range(idx600, min(idx600 + 6, len(h_sorted))):
        h_k = h_sorted[k]
        rail_hardness.append(
            {
                "h": float(h_k),
                "sum_ln": float(grouped.loc[h_k]),
                "delta_ln_from_0600": float(grouped.loc[h_k] - grouped.loc[h_sorted[idx600]]),
            }
        )

    # posterior moments (exp-normalized, trapezoid mass)
    log_l = grouped.to_numpy()
    log_l_shifted = log_l - log_l.max()
    weights_unnorm = np.exp(log_l_shifted)
    dh = np.gradient(h_grid)
    mass = weights_unnorm * dh
    mass_norm = mass / mass.sum()
    mean_h = float(np.sum(mass_norm * h_grid))
    var_h = float(np.sum(mass_norm * (h_grid - mean_h) ** 2))
    std_h = float(np.sqrt(var_h))

    return {
        "argmax_h": argmax_h,
        "argmax_sum_ln": argmax_val,
        "railed_at_0600": railed,
        "top5_h_and_sum_ln": [(float(h), float(v)) for h, v in top5.items()],
        "rail_hardness_delta_ln_from_0600": rail_hardness,
        "posterior_mean_h": mean_h,
        "posterior_std_h": std_h,
        "n_h_grid_points": int(len(h_grid)),
    }


def null_check(n2_df: pd.DataFrame, postfix_df: pd.DataFrame) -> dict:
    m = n2_df.merge(
        postfix_df,
        on=["event_idx", "h"],
        suffixes=("_1d", "_off"),
        how="outer",
        indicator=True,
    )
    results: dict = {"row_join_mismatches": int((m["_merge"] != "both").sum())}

    all_null_cols = NULL_COLUMNS_2D + NULL_COLUMNS_CATALOGUE + NULL_COLUMNS_SELECTION
    # dedupe preserving order
    seen = set()
    dedup_cols = []
    for c in all_null_cols:
        if c not in seen:
            dedup_cols.append(c)
            seen.add(c)

    null_results = {}
    any_violation = False
    for col in dedup_cols:
        a = m[f"{col}_1d"].astype(float)
        b = m[f"{col}_off"].astype(float)
        diff = (a - b).abs()
        n_diff = int((diff > 0).sum())
        max_diff = float(diff.max())
        violated = n_diff > 0
        any_violation = any_violation or violated
        null_results[col] = {
            "max_abs_diff": max_diff,
            "n_differing_cells": n_diff,
            "n_total_cells": int(len(diff)),
            "VIOLATED": violated,
        }
    results["null_columns"] = null_results
    results["any_null_violated"] = any_violation

    nonnull_results = {}
    for col in NONNULL_COLUMNS:
        a = m[f"{col}_1d"].astype(float)
        b = m[f"{col}_off"].astype(float)
        diff = (a - b).abs()
        n_diff = int((diff > 0).sum())
        nonnull_results[col] = {
            "max_abs_diff": float(diff.max()),
            "n_differing_cells": n_diff,
            "n_total_cells": int(len(diff)),
            "confirmed_nonnull": n_diff > 0,
        }
    results["nonnull_columns"] = nonnull_results

    return results


def p6_branch(p2: dict, map_result: dict) -> str:
    """Mechanical branch per prereg: (a) REAL-BUT-SMALL / (b) ESCALATE / (c) MIXED."""
    both_in_band = p2["chord_in_band"] and p2["central_diff_in_band"]
    if map_result["railed_at_0600"] and both_in_band:
        return "a_REAL_BUT_SMALL"
    if not map_result["railed_at_0600"]:
        return "b_ESCALATE"
    return "c_MIXED"


def channel_difference_context(n2_df: pd.DataFrame, postfix_df: pd.DataFrame) -> dict:
    """2D-1D channel-difference tilt: does the g_frac +243.5 tilt partially annihilate?

    g_frac_diff(h) = ln(combined_with_bh) - ln(combined_no_bh), summed over events;
    compare the delta of this quantity between the 1d and off runs to the P-2 tilt.
    """
    m = n2_df.merge(
        postfix_df,
        on=["event_idx", "h"],
        suffixes=("_1d", "_off"),
        how="inner",
    )
    for suf in ("_1d", "_off"):
        m[f"ln_gfrac_diff{suf}"] = np.log(
            m[f"combined_with_bh{suf}"].astype(float).clip(lower=1e-300)
        ) - np.log(m[f"combined_no_bh{suf}"].astype(float).clip(lower=1e-300))

    per_h_1d = m.groupby("h")["ln_gfrac_diff_1d"].sum().sort_index()
    per_h_off = m.groupby("h")["ln_gfrac_diff_off"].sum().sort_index()
    per_h_delta = per_h_1d - per_h_off

    h_grid = per_h_delta.index.to_numpy(dtype=float)
    h_lo, h_hi = float(h_grid.min()), float(h_grid.max())
    chord = float((per_h_delta.loc[h_hi] - per_h_delta.loc[h_lo]) / (h_hi - h_lo))

    h_sorted = np.sort(h_grid)
    i73 = int(np.argmin(np.abs(h_sorted - 0.73)))
    if 0 < i73 < h_sorted.size - 1:
        h_m, h_p = h_sorted[i73 - 1], h_sorted[i73 + 1]
        central_diff = float((per_h_delta.loc[h_p] - per_h_delta.loc[h_m]) / (h_p - h_m))
    else:
        central_diff = float("nan")

    return {
        "note": (
            "channel-difference (2D-1D) tilt change; expected to roughly mirror "
            "-1 * P-2 (the g_frac tilt annihilates by approx the P-2 amount)"
        ),
        "chord_nats_per_h": chord,
        "central_diff_nats_per_h_at_073": central_diff,
    }


def main() -> None:
    report: dict = {}
    per_venue_branch = {}
    per_venue_p2 = {}
    per_venue_p3 = {}
    per_venue_map = {}

    for venue, cfg in VENUES.items():
        vreport: dict = {}
        n2_df, postfix_df, metadata = load(cfg)

        vreport["fingerprints"] = fingerprints(n2_df, postfix_df, metadata)

        m = merged_no_bh(n2_df, postfix_df)

        p2 = p2_tilt(m)
        vreport["P2_tilt"] = p2
        per_venue_p2[venue] = p2

        p3 = p3_sign_coherence(m)
        vreport["P3_sign_coherence"] = p3
        per_venue_p3[venue] = p3

        map_result = p6_map_1d(n2_df)
        vreport["P6_map_1d"] = map_result
        per_venue_map[venue] = map_result

        branch = p6_branch(p2, map_result)
        vreport["mechanical_branch"] = branch
        per_venue_branch[venue] = branch

        vreport["null_check"] = null_check(n2_df, postfix_df)

        vreport["channel_difference_context"] = channel_difference_context(n2_df, postfix_df)

        report[venue] = vreport

    # Joint verdict
    branches = set(per_venue_branch.values())
    if branches == {"a_REAL_BUT_SMALL"}:
        joint_branch = "a_REAL_BUT_SMALL"
    elif "b_ESCALATE" in branches:
        joint_branch = "b_ESCALATE"
    else:
        joint_branch = "c_MIXED"

    any_null_violation = any(
        report[v]["null_check"]["any_null_violated"] for v in VENUES
    )

    report["joint_verdict"] = {
        "per_venue_branch": per_venue_branch,
        "joint_branch": joint_branch,
        "any_null_violation_ANY_VENUE": any_null_violation,
    }

    out_path = N2_DIR / "readout.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Wrote {out_path}")
    print(json.dumps(report["joint_verdict"], indent=2))


if __name__ == "__main__":
    main()
