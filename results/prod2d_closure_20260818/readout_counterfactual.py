"""Registered readout for PREREGISTRATION_PROD_COUNTERFACTUAL.md v2 (Sec 1 gates, Sec 3 reads).

No branch adjudication here -- compute and report only. Conventions inherited
verbatim from tier0_bootstrap_jackknife.py: pivot per column to
(n_events, n_h), trapezoid weights np.gradient(h_grid), physics-floor
zero-handling, Sigma log L posterior, mean_h/sigma_h via gradient-weighted
moments.

Usage:
    python readout_counterfactual.py [--output readout_counterfactual_output.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]

VENUES = ("iiib", "joint_r1")
VARIANTS = ("v0", "v1", "v2k05", "v2k2")
CHANNELS = ("combined_no_bh", "combined_with_bh")
TRUTH = 0.73

BASELINE_MEAN_H_2D = {"iiib": 0.7842, "joint_r1": 0.7966}
BASELINE_MEAN_H_1D = {"iiib": 0.6040, "joint_r1": 0.6074}

PROBE_H = (0.72, 0.78)

N0_TOL = 1e-10
N1_REPORT_TOL = 1e-15  # reporting threshold, not a hard gate cutoff
N2_REL_THRESH = 1e-6
N2_FRACTION_GATE = 0.10

METADATA_WHITELIST = {
    "git_commit",
    "timestamp",
    "working_directory",
    "seed",
    "random_seed",
    "catalogue_mass_overlap",
    "catalogue_mass_error_scale",
    "selection_in_completion_numerator",
    "h_value",
    "simulation_index",
}


def _is_whitelisted(key: str) -> bool:
    if key in METADATA_WHITELIST:
        return True
    if key.startswith("SLURM_"):
        return True
    return False


def _load_cell_raw(variant: str, venue: str) -> pd.DataFrame:
    path = REPO_ROOT / "results" / "prod2d_closure_20260818" / "counterfactual" / f"{variant}_{venue}" / "event_likelihoods.csv"
    return pd.read_csv(path)


def _dedupe(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    dup_mask = df.duplicated(subset=["event_idx", "h"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        df = df[~dup_mask].copy()
    return df, n_dup


def _load_baseline(venue: str) -> pd.DataFrame:
    path = REPO_ROOT / "results" / "run_20260804_postfix" / venue / "diagnostics" / "event_likelihoods.csv"
    return pd.read_csv(path)


def _physics_floor_apply(likelihoods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate posterior_combination._physics_floor per-row (tier0 convention)."""
    result = likelihoods.copy()
    n_events = result.shape[0]
    exclude_mask = np.zeros(n_events, dtype=bool)
    for i in range(n_events):
        row = result[i]
        zero_mask = row == 0.0
        if not zero_mask.any():
            continue
        nonzero = row[~zero_mask]
        if nonzero.size == 0:
            exclude_mask[i] = True
        else:
            result[i, zero_mask] = float(nonzero.min())
    return result, exclude_mask


def _pivot(df: pd.DataFrame, channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (h_grid sorted, event_idx array, matrix [n_events, n_h])."""
    h_grid = np.sort(df["h"].unique())
    piv = df.pivot(index="event_idx", columns="h", values=channel).reindex(columns=h_grid)
    if piv.isna().any().any():
        raise ValueError(f"{channel}: pivot has missing (event, h) cells -- ragged CSV")
    event_idx = piv.index.to_numpy()
    mat = piv.to_numpy(dtype=np.float64)
    return h_grid, event_idx, mat


def _mean_h_sigma_h(logL: np.ndarray, h_grid: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    logpost = logL.sum(axis=0)
    lp = logpost - logpost.max()
    post = np.exp(lp)
    norm = float((post * weights).sum())
    post_n = post / norm
    mean_h = float((post_n * h_grid * weights).sum())
    var = float((post_n * (h_grid - mean_h) ** 2 * weights).sum())
    sigma_h = float(np.sqrt(max(var, 0.0)))
    return mean_h, sigma_h


def _mean_h_convention(df: pd.DataFrame, channel: str) -> dict[str, Any]:
    h_grid, event_idx, mat = _pivot(df, channel)
    floored, exclude_mask = _physics_floor_apply(mat)
    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        floored = floored[~exclude_mask]
        event_idx = event_idx[~exclude_mask]
    logL = np.log(floored)
    weights = np.gradient(h_grid)
    mean_h, sigma_h = _mean_h_sigma_h(logL, h_grid, weights)
    return {
        "mean_h": mean_h,
        "sigma_h": sigma_h,
        "n_events": int(logL.shape[0]),
        "n_excluded_physics_floor": n_excluded,
        "n_h": int(h_grid.size),
        "h_grid": h_grid,
    }


def _max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0.0, 1.0, denom)
    diff = np.abs(a - b) / denom
    # where both are exactly zero, diff is 0 by construction (a-b=0)
    return float(np.max(diff))


def gate_n0(venue: str, baseline_df: pd.DataFrame, cells: dict[str, pd.DataFrame]) -> dict[str, Any]:
    v0 = cells["v0"]
    result: dict[str, Any] = {"venue": venue, "probes": []}
    all_pass = True
    for probe_h in PROBE_H:
        base_rows = baseline_df[np.isclose(baseline_df["h"], probe_h)].set_index("event_idx").sort_index()
        v0_rows = v0[np.isclose(v0["h"], probe_h)].set_index("event_idx").sort_index()
        common_idx = base_rows.index.intersection(v0_rows.index)
        base_rows = base_rows.loc[common_idx]
        v0_rows = v0_rows.loc[common_idx]
        probe_result = {"h": probe_h, "n_events_matched": int(len(common_idx))}
        for channel in CHANNELS:
            a = base_rows[channel].to_numpy(dtype=np.float64)
            b = v0_rows[channel].to_numpy(dtype=np.float64)
            rel = _max_rel_diff(a, b)
            passed = bool(rel <= N0_TOL)
            all_pass = all_pass and passed
            probe_result[f"max_rel_diff_{channel}"] = rel
            probe_result[f"pass_{channel}"] = passed
        result["probes"].append(probe_result)
    result["gate_pass"] = all_pass
    return result


def metadata_diff_check(venue: str, cells_dirs: dict[str, Path]) -> dict[str, Any]:
    ref_path = REPO_ROOT / "results" / "run_20260817_fusion_counterfactual" / f"off_{venue}" / "run_metadata_0.json"
    ref = json.loads(ref_path.read_text())
    ref_cli = ref.get("cli_args", {})
    out: dict[str, Any] = {"venue": venue, "reference": str(ref_path), "cells": {}}
    for variant, cell_dir in cells_dirs.items():
        # pick first available run_metadata file in the cell dir
        meta_files = sorted(cell_dir.glob("run_metadata_*.json"))
        if not meta_files:
            out["cells"][variant] = {"error": "no run_metadata file found"}
            continue
        meta_path = meta_files[0]
        meta = json.loads(meta_path.read_text())
        cli = meta.get("cli_args", {})
        keys = set(cli.keys()) | set(ref_cli.keys())
        differing: dict[str, Any] = {}
        for k in sorted(keys):
            a = cli.get(k, "<absent>")
            b = ref_cli.get(k, "<absent>")
            if a != b:
                differing[k] = {"cell_value": a, "reference_value": b}
        # also top-level (non cli_args) keys of interest
        top_keys = {"git_commit", "timestamp", "random_seed"}
        for k in top_keys:
            a = meta.get(k, "<absent>")
            b = ref.get(k, "<absent>")
            if a != b:
                differing[k] = {"cell_value": a, "reference_value": b}
        not_whitelisted = {k: v for k, v in differing.items() if not _is_whitelisted(k)}
        out["cells"][variant] = {
            "metadata_file_used": str(meta_path),
            "differing_keys": differing,
            "not_whitelisted_keys": not_whitelisted,
            "flag_any_not_whitelisted": bool(not_whitelisted),
        }
    return out


def gate_n1(venue: str, baseline_df: pd.DataFrame, cells: dict[str, pd.DataFrame]) -> dict[str, Any]:
    base_h_grid, base_event_idx, base_mat = _pivot(baseline_df, "combined_no_bh")
    result: dict[str, Any] = {"venue": venue, "variants": {}}
    for variant in ("v1", "v2k05", "v2k2"):
        df = cells[variant]
        h_grid, event_idx, mat = _pivot(df, "combined_no_bh")
        # match h grids
        common_h = np.intersect1d(base_h_grid, h_grid)
        common_idx = np.intersect1d(base_event_idx, event_idx)
        base_mask_h = np.isin(base_h_grid, common_h)
        var_mask_h = np.isin(h_grid, common_h)
        base_mask_e = np.isin(base_event_idx, common_idx)
        var_mask_e = np.isin(event_idx, common_idx)
        # order both by sorted common_idx / common_h (both are already sorted arrays)
        a = base_mat[np.ix_(base_mask_e, base_mask_h)]
        b = mat[np.ix_(var_mask_e, var_mask_h)]
        # ensure orderings align: base_event_idx[base_mask_e] and event_idx[var_mask_e] both equal sorted(common_idx)
        rel = _max_rel_diff(a, b)
        result["variants"][variant] = {
            "n_events_matched": int(common_idx.size),
            "n_h_matched": int(common_h.size),
            "max_rel_diff_combined_no_bh": rel,
            "bit_identical_1e15": bool(rel <= N1_REPORT_TOL),
        }
    return result


def gate_n2(venue: str, baseline_df: pd.DataFrame, v1_df: pd.DataFrame) -> dict[str, Any]:
    probe_h = 0.72
    base_rows = baseline_df[np.isclose(baseline_df["h"], probe_h)].set_index("event_idx").sort_index()
    v1_rows = v1_df[np.isclose(v1_df["h"], probe_h)].set_index("event_idx").sort_index()
    common_idx = base_rows.index.intersection(v1_rows.index)
    base_rows = base_rows.loc[common_idx]
    v1_rows = v1_rows.loc[common_idx]

    cat_supported_mask = base_rows["L_cat_with_bh"].to_numpy(dtype=np.float64) > 0.0
    n_cat_supported = int(cat_supported_mask.sum())

    a = base_rows["combined_with_bh"].to_numpy(dtype=np.float64)
    b = v1_rows["combined_with_bh"].to_numpy(dtype=np.float64)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0.0, 1.0, denom)
    rel_diff = np.abs(a - b) / denom

    changed_mask = (rel_diff >= N2_REL_THRESH) & cat_supported_mask
    n_changed = int(changed_mask.sum())
    fraction = n_changed / n_cat_supported if n_cat_supported > 0 else float("nan")

    return {
        "venue": venue,
        "h": probe_h,
        "n_events_matched": int(len(common_idx)),
        "n_catalogue_supported": n_cat_supported,
        "n_changed_ge_1e-6_rel": n_changed,
        "fraction_changed": fraction,
        "gate_pass_ge_10pct": bool(fraction >= N2_FRACTION_GATE) if n_cat_supported > 0 else False,
    }


def read_r1_r2(venue: str, baseline_df: pd.DataFrame, cells: dict[str, pd.DataFrame]) -> dict[str, Any]:
    base_2d = _mean_h_convention(baseline_df, "combined_with_bh")
    base_1d = _mean_h_convention(baseline_df, "combined_no_bh")

    out: dict[str, Any] = {
        "venue": venue,
        "baseline": {
            "mean_h_2D": base_2d["mean_h"],
            "sigma_h_2D": base_2d["sigma_h"],
            "mean_h_1D": base_1d["mean_h"],
            "sigma_h_1D": base_1d["sigma_h"],
            "n_events": base_2d["n_events"],
            "n_h": base_2d["n_h"],
            "header_ref_mean_h_2D": BASELINE_MEAN_H_2D[venue],
            "header_ref_mean_h_1D": BASELINE_MEAN_H_1D[venue],
        },
        "variants": {},
    }

    for variant in ("v1", "v2k05", "v2k2"):
        df = cells[variant]
        r2d = _mean_h_convention(df, "combined_with_bh")
        r1d = _mean_h_convention(df, "combined_no_bh")
        delta_2d = r2d["mean_h"] - base_2d["mean_h"]
        out["variants"][variant] = {
            "mean_h_2D": r2d["mean_h"],
            "sigma_h_2D": r2d["sigma_h"],
            "delta_vs_banked_v0_2D": delta_2d,
            "mean_h_1D": r1d["mean_h"],
            "sigma_h_1D": r1d["sigma_h"],
            "mean_h_1D_matches_baseline_1D": bool(abs(r1d["mean_h"] - base_1d["mean_h"]) < 1e-6),
            "n_events": r2d["n_events"],
            "n_h": r2d["n_h"],
        }
    return out


def main(argv: list[str] | None = None) -> None:
    t_start = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "readout_counterfactual_output.json")
    args = parser.parse_args(argv)

    output: dict[str, Any] = {
        "preregistration": "PREREGISTRATION_PROD_COUNTERFACTUAL.md v2",
        "truth": TRUTH,
        "venues": {},
    }

    for venue in VENUES:
        print(f"=== {venue} ===", flush=True)
        baseline_df = _load_baseline(venue)

        cells: dict[str, pd.DataFrame] = {}
        cells_dirs: dict[str, Path] = {}
        dedupe_report: dict[str, Any] = {}
        for variant in VARIANTS:
            raw = _load_cell_raw(variant, venue)
            df, n_dup = _dedupe(raw, f"{variant}_{venue}")
            cells[variant] = df
            cells_dirs[variant] = REPO_ROOT / "results" / "prod2d_closure_20260818" / "counterfactual" / f"{variant}_{venue}"
            dedupe_report[variant] = {
                "raw_rows": int(len(raw)),
                "deduped_rows": int(len(df)),
                "n_duplicates_removed": n_dup,
                "n_events": int(df["event_idx"].nunique()),
                "n_h": int(df["h"].unique().size),
                "expected_rows": 1588 * 2 if variant == "v0" else 1588 * 41,
                "matches_expected": bool(len(df) == (1588 * 2 if variant == "v0" else 1588 * 41)),
            }

        venue_out: dict[str, Any] = {"sanity_dedupe": dedupe_report}

        # --- N-0 ---
        venue_out["N0_gate"] = gate_n0(venue, baseline_df, cells)

        # --- metadata check ---
        venue_out["N0_metadata_diff"] = metadata_diff_check(venue, cells_dirs)

        # --- N-1 ---
        venue_out["N1_gate"] = gate_n1(venue, baseline_df, cells)

        # --- N-2 ---
        venue_out["N2_gate"] = gate_n2(venue, baseline_df, cells["v1"])

        # --- R1/R2 ---
        venue_out["R1_R2_reads"] = read_r1_r2(venue, baseline_df, cells)

        # --- h-grid sanity ---
        h_grid_41 = np.sort(cells["v1"]["h"].unique())
        venue_out["h_grid_sanity"] = {
            "n_h": int(h_grid_41.size),
            "h_grid": [float(x) for x in h_grid_41],
        }

        output["venues"][venue] = venue_out

    # --- venue differential (item 6) ---
    dv1_iiib = output["venues"]["iiib"]["R1_R2_reads"]["variants"]["v1"]["delta_vs_banked_v0_2D"]
    dv1_joint = output["venues"]["joint_r1"]["R1_R2_reads"]["variants"]["v1"]["delta_vs_banked_v0_2D"]
    output["venue_differential_deltaV1_joint_minus_iiib"] = dv1_joint - dv1_iiib

    runtime_s = time.time() - t_start
    output["runtime_seconds"] = runtime_s

    args.output.write_text(json.dumps(output, indent=2, default=str))
    print(f"Wrote {args.output} in {runtime_s:.2f}s", flush=True)


if __name__ == "__main__":
    main()
