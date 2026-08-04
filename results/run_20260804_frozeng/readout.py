"""Mechanical readout script for the pre-registered frozen-g_frac evaluate.

Reads the two new (frozen) event_likelihoods.csv files and scores them
against the pre-registration in
results/run_20260804_postfix/gate_vii/PREREGISTRATION_FROZEN_GFRAC.md.

No interpretation beyond the mechanical branch scoring is performed here;
this script only computes numbers and writes them to readout.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FROZEN_DIR = REPO / "results" / "run_20260804_frozeng"
POSTFIX_DIR = REPO / "results" / "run_20260804_postfix"

VENUES = {
    "iiib": {
        "frozen_csv": FROZEN_DIR / "iiib" / "diagnostics" / "event_likelihoods.csv",
        "postfix_csv": POSTFIX_DIR / "iiib" / "diagnostics" / "event_likelihoods.csv",
        "proxy_map": 0.660,
    },
    "joint_r1": {
        "frozen_csv": FROZEN_DIR / "joint_r1" / "diagnostics" / "event_likelihoods.csv",
        "postfix_csv": POSTFIX_DIR / "joint_r1" / "diagnostics" / "event_likelihoods.csv",
        "proxy_map": 0.640,
    },
}

CONFIRM_BAND = (0.63, 0.665)
REFUTE_BAND = (0.78, 0.80)
CONTEXT_H_VALUES = [0.700, 0.730, 0.780, 0.800]

EXPECTED_ROWS = 65108  # 41 h * 1588 events
EXPECTED_H_COUNT = 41
EXPECTED_EVENT_COUNT = 1588


def grid_steps(h_grid: np.ndarray) -> np.ndarray:
    """Local grid step at each h value (min distance to neighbor)."""
    h_sorted = np.sort(h_grid)
    steps = np.empty_like(h_sorted)
    steps[0] = h_sorted[1] - h_sorted[0]
    steps[-1] = h_sorted[-1] - h_sorted[-2]
    steps[1:-1] = np.minimum(h_sorted[1:-1] - h_sorted[:-2], h_sorted[2:] - h_sorted[1:-1])
    return dict(zip(h_sorted.tolist(), steps.tolist()))


def nearest_grid_step(h_grid: np.ndarray, h0: float) -> float:
    steps = grid_steps(h_grid)
    nearest = min(steps, key=lambda h: abs(h - h0))
    return steps[nearest]


def compute_2d_map(df: pd.DataFrame) -> dict:
    grouped = df.groupby("h")["combined_with_bh"].apply(lambda s: np.log(s.astype(float)).sum())
    grouped = grouped.sort_index()
    h_grid = grouped.index.to_numpy()
    argmax_h = float(grouped.idxmax())
    argmax_val = float(grouped.max())
    top5 = grouped.sort_values(ascending=False).head(5)
    context = {}
    for h_ctx in CONTEXT_H_VALUES:
        # nearest grid point to context h (grid may not contain it exactly)
        nearest_idx = (np.abs(h_grid - h_ctx)).argmin()
        nearest_h = float(h_grid[nearest_idx])
        val = float(grouped.loc[nearest_h])
        context[str(h_ctx)] = {
            "nearest_grid_h": nearest_h,
            "delta_ln_from_argmax": argmax_val - val,
        }
    return {
        "argmax_h": argmax_h,
        "argmax_sum_ln": argmax_val,
        "top5_h_and_sum_ln": [(float(h), float(v)) for h, v in top5.items()],
        "context_deltas": context,
        "n_h_grid_points": int(len(h_grid)),
        "h_grid": h_grid.tolist(),
    }


def branch_score(map_h: float) -> str:
    if CONFIRM_BAND[0] <= map_h <= CONFIRM_BAND[1]:
        return "CONFIRM"
    if REFUTE_BAND[0] <= map_h <= REFUTE_BAND[1]:
        return "REFUTE"
    return "MIXED_OR_OTHER"


def secondary_read_1d_bit_identity(frozen_df: pd.DataFrame, postfix_df: pd.DataFrame) -> dict:
    merged = frozen_df.merge(
        postfix_df,
        on=["event_idx", "h"],
        suffixes=("_frozen", "_postfix"),
        how="outer",
        indicator=True,
    )
    results = {}
    mismatched_rows = int((merged["_merge"] != "both").sum())
    results["row_join_mismatches"] = mismatched_rows
    for col in ["combined_no_bh", "L_cat_no_bh"]:
        a = merged[f"{col}_frozen"].astype(float)
        b = merged[f"{col}_postfix"].astype(float)
        diff = (a - b).abs()
        results[col] = {
            "max_abs_diff": float(diff.max()),
            "n_differing_cells": int((diff > 0).sum()),
            "n_total_cells": int(len(diff)),
        }
    return results


def secondary_read_g_frac_constancy(frozen_df: pd.DataFrame, postfix_df: pd.DataFrame) -> dict:
    per_event = frozen_df.groupby("event_idx")["g_frac"].nunique()
    n_violating = int((per_event != 1).sum())
    max_spread = float(
        frozen_df.groupby("event_idx")["g_frac"].apply(lambda s: s.max() - s.min()).max()
    )

    # frozen value per event (should be single value; take first/mean since nunique should be 1)
    frozen_g = frozen_df.groupby("event_idx")["g_frac"].first()

    postfix_h073 = postfix_df[np.isclose(postfix_df["h"].astype(float), 0.73)]
    postfix_g_at_073 = postfix_h073.set_index("event_idx")["g_frac"]

    aligned = frozen_g.align(postfix_g_at_073, join="inner")
    diff = (aligned[0].astype(float) - aligned[1].astype(float)).abs()

    return {
        "n_events": int(len(per_event)),
        "n_events_violating_h_constancy": n_violating,
        "max_spread_across_h_within_event": max_spread,
        "n_events_matched_to_postfix_h073": int(len(diff)),
        "max_abs_diff_vs_postfix_h073": float(diff.max()) if len(diff) else None,
    }


def full_grid_posterior_moments(df: pd.DataFrame) -> dict:
    grouped = df.groupby("h")["combined_with_bh"].apply(lambda s: np.log(s.astype(float)).sum())
    grouped = grouped.sort_index()
    h_grid = grouped.index.to_numpy()
    log_l = grouped.to_numpy()

    # exp-normalize (subtract max for numerical stability)
    log_l_shifted = log_l - log_l.max()
    weights_unnorm = np.exp(log_l_shifted)

    # non-uniform grid spacing (trapezoidal weights)
    dh = np.gradient(h_grid)
    mass = weights_unnorm * dh
    mass_norm = mass / mass.sum()

    mean_h = float(np.sum(mass_norm * h_grid))
    var_h = float(np.sum(mass_norm * (h_grid - mean_h) ** 2))
    std_h = float(np.sqrt(var_h))

    return {
        "posterior_mean_h": mean_h,
        "posterior_std_h": std_h,
    }


def secondary_read_bit_identical_columns(frozen_df: pd.DataFrame, postfix_df: pd.DataFrame) -> dict:
    """Pre-registration secondary read (4): w_tilde_G, r_Malm, alpha_G_phi, D_tilde_phi
    expected bit-identical between frozen and unfrozen runs."""
    merged = frozen_df.merge(
        postfix_df,
        on=["event_idx", "h"],
        suffixes=("_frozen", "_postfix"),
        how="inner",
    )
    results = {}
    for col in ["w_tilde_G", "r_Malm", "alpha_G_phi", "D_tilde_phi"]:
        a = merged[f"{col}_frozen"].astype(float)
        b = merged[f"{col}_postfix"].astype(float)
        diff = (a - b).abs()
        results[col] = {
            "max_abs_diff": float(diff.max()),
            "n_differing_cells": int((diff > 0).sum()),
        }
    return results


def main() -> None:
    report: dict = {}
    per_venue_map = {}

    for venue, cfg in VENUES.items():
        vreport: dict = {}

        frozen_df = pd.read_csv(cfg["frozen_csv"])
        postfix_df = pd.read_csv(cfg["postfix_csv"])

        # Fingerprints
        vreport["fingerprints"] = {
            "frozen_csv_rows": int(len(frozen_df)),
            "frozen_csv_rows_expected": EXPECTED_ROWS,
            "frozen_csv_rows_match": len(frozen_df) == EXPECTED_ROWS,
            "frozen_n_h": int(frozen_df["h"].nunique()),
            "frozen_n_events": int(frozen_df["event_idx"].nunique()),
            "postfix_csv_rows": int(len(postfix_df)),
        }

        # Step 1: 2D MAP
        map_result = compute_2d_map(frozen_df)
        vreport["map_2d"] = map_result
        per_venue_map[venue] = map_result["argmax_h"]

        # Step 2: branch score (per venue, applied jointly below too)
        vreport["branch_score_this_venue"] = branch_score(map_result["argmax_h"])

        # Step 3: secondary read (i) 1D bit-identity
        vreport["secondary_1d_bit_identity"] = secondary_read_1d_bit_identity(frozen_df, postfix_df)

        # Step 4: secondary read (ii) g_frac h-constancy
        vreport["secondary_g_frac_constancy"] = secondary_read_g_frac_constancy(frozen_df, postfix_df)

        # Extra: secondary read (4) from prereg -- bit-identical selection columns
        vreport["secondary_selection_columns_bit_identity"] = secondary_read_bit_identical_columns(
            frozen_df, postfix_df
        )

        # Step 5: proxy vs live comparison
        live_map = map_result["argmax_h"]
        proxy_map = cfg["proxy_map"]
        step_at_live = nearest_grid_step(np.array(map_result["h_grid"]), live_map)
        vreport["proxy_vs_live"] = {
            "proxy_predicted_map": proxy_map,
            "live_map": live_map,
            "diff": live_map - proxy_map,
            "grid_step_at_live_map": step_at_live,
            "diff_in_grid_steps": (live_map - proxy_map) / step_at_live if step_at_live else None,
        }

        # Step 6: full-grid posterior mean/std
        vreport["full_grid_posterior_moments"] = full_grid_posterior_moments(frozen_df)

        report[venue] = vreport

    # Joint mechanical branch verdict
    both_confirm = all(
        CONFIRM_BAND[0] <= per_venue_map[v] <= CONFIRM_BAND[1] for v in per_venue_map
    )
    both_refute = all(
        REFUTE_BAND[0] <= per_venue_map[v] <= REFUTE_BAND[1] for v in per_venue_map
    )
    if both_confirm:
        joint_verdict = "CONFIRM"
    elif both_refute:
        joint_verdict = "REFUTE"
    else:
        joint_verdict = "MIXED"

    report["joint_verdict"] = {
        "per_venue_2d_map": per_venue_map,
        "confirm_band": CONFIRM_BAND,
        "refute_band": REFUTE_BAND,
        "verdict": joint_verdict,
    }

    out_path = FROZEN_DIR / "readout.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Wrote {out_path}")
    print(json.dumps(report["joint_verdict"], indent=2))


if __name__ == "__main__":
    main()
