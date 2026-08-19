"""Registered readout for PREREGISTRATION_TILT_BATTERY.md v2 (Sec 1 reads R-E/R-J, Sec 2 gates).

No branch adjudication here -- compute and report only. Conventions inherited
verbatim from readout_counterfactual.py / tier0_bootstrap_jackknife.py: pivot
per column to (n_events, n_h), trapezoid weights np.gradient(h_grid),
physics-floor zero-handling, Sigma log L posterior, mean_h/sigma_h via
gradient-weighted moments.

Usage:
    python readout_battery.py [--output readout_battery_output.json]
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
BATTERY_ROOT = HERE / "battery"
BASELINE_ROOT = HERE / "postfix_baseline"

VENUES = ("iiib", "joint_r1")
CHANNELS = ("combined_no_bh", "combined_with_bh")

# Instrument variants (v0 = continuity probe, eoff = E-off, jker = J-kernel)
VARIANTS = ("v0", "eoff", "jker")

PROBE_H = (0.675, 0.700)

BASELINE_MEAN_H_2D = {"iiib": 0.6771, "joint_r1": 0.6788}
BASELINE_MEAN_H_1D = {"iiib": 0.6010, "joint_r1": 0.6020}

N0_TOL = 1e-10
N1_TOL = 0.0  # bit-identical required (reported as max rel diff; gate is "== 0" in spirit)
N2E_REL_THRESH = 1e-6
N2E_FRACTION_GATE = 0.10
N2J_RATIO_THRESH = 1e-4

METADATA_WHITELIST = {
    "git_commit",
    "timestamp",
    "working_directory",
    "seed",
    "random_seed",
    "eddington_m",
    "sigma4d_mass_kernel",
    "h_value",
    "simulation_index",
}


def _is_whitelisted(key: str) -> bool:
    if key in METADATA_WHITELIST:
        return True
    if key.startswith("SLURM_"):
        return True
    return False


def _load_cell(variant: str, venue: str) -> pd.DataFrame:
    path = BATTERY_ROOT / f"{variant}_{venue}" / "event_likelihoods.csv"
    return pd.read_csv(path)


def _load_baseline(venue: str) -> pd.DataFrame:
    path = BASELINE_ROOT / venue / "event_likelihoods.csv"
    return pd.read_csv(path)


def _dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    dup_mask = df.duplicated(subset=["event_idx", "h"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        df = df[~dup_mask].copy()
    return df, n_dup


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
    return float(np.max(diff))


# ---------------------------------------------------------------------------
# Score item 1: N-0 continuity + metadata
# ---------------------------------------------------------------------------


def gate_n0(venue: str, baseline_df: pd.DataFrame, v0_df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"venue": venue, "probes": []}
    all_pass = True
    for probe_h in PROBE_H:
        base_rows = baseline_df[np.isclose(baseline_df["h"], probe_h)].set_index("event_idx").sort_index()
        v0_rows = v0_df[np.isclose(v0_df["h"], probe_h)].set_index("event_idx").sort_index()
        common_idx = base_rows.index.intersection(v0_rows.index)
        base_rows = base_rows.loc[common_idx]
        v0_rows = v0_rows.loc[common_idx]
        probe_result: dict[str, Any] = {"h": probe_h, "n_events_matched": int(len(common_idx))}
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
    ref_path = BASELINE_ROOT / venue / "run_metadata_0.json"
    ref = json.loads(ref_path.read_text())
    ref_cli = ref.get("cli_args", {})
    out: dict[str, Any] = {"venue": venue, "reference": str(ref_path), "cells": {}}
    for variant, cell_dir in cells_dirs.items():
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


# ---------------------------------------------------------------------------
# Score item 2: N-1 (E) full-grid bit-identical check on combined_no_bh
# ---------------------------------------------------------------------------


def gate_n1_eoff(venue: str, baseline_df: pd.DataFrame, eoff_df: pd.DataFrame) -> dict[str, Any]:
    base_h_grid, base_event_idx, base_mat = _pivot(baseline_df, "combined_no_bh")
    h_grid, event_idx, mat = _pivot(eoff_df, "combined_no_bh")
    common_h = np.intersect1d(base_h_grid, h_grid)
    common_idx = np.intersect1d(base_event_idx, event_idx)
    base_mask_h = np.isin(base_h_grid, common_h)
    var_mask_h = np.isin(h_grid, common_h)
    base_mask_e = np.isin(base_event_idx, common_idx)
    var_mask_e = np.isin(event_idx, common_idx)
    a = base_mat[np.ix_(base_mask_e, base_mask_h)]
    b = mat[np.ix_(var_mask_e, var_mask_h)]
    rel = _max_rel_diff(a, b)
    n_exact = int(np.sum(a == b))
    n_total = int(a.size)
    return {
        "venue": venue,
        "n_events_matched": int(common_idx.size),
        "n_h_matched": int(common_h.size),
        "n_cells": n_total,
        "n_bit_identical_cells": n_exact,
        "max_rel_diff_combined_no_bh": rel,
        "gate_pass_bit_identical": bool(rel == 0.0),
    }


# ---------------------------------------------------------------------------
# Score item 3: N-2 engagement gates (E and J)
# ---------------------------------------------------------------------------


def gate_n2_e(venue: str, baseline_df: pd.DataFrame, eoff_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"venue": venue, "probes": []}
    for probe_h in PROBE_H:
        base_rows = baseline_df[np.isclose(baseline_df["h"], probe_h)].set_index("event_idx").sort_index()
        e_rows = eoff_df[np.isclose(eoff_df["h"], probe_h)].set_index("event_idx").sort_index()
        common_idx = base_rows.index.intersection(e_rows.index)
        base_rows = base_rows.loc[common_idx]
        e_rows = e_rows.loc[common_idx]

        cat_supported_mask = base_rows["L_cat_with_bh"].to_numpy(dtype=np.float64) > 0.0
        n_cat_supported = int(cat_supported_mask.sum())

        a = base_rows["combined_with_bh"].to_numpy(dtype=np.float64)
        b = e_rows["combined_with_bh"].to_numpy(dtype=np.float64)
        denom = np.maximum(np.abs(a), np.abs(b))
        denom = np.where(denom == 0.0, 1.0, denom)
        rel_diff = np.abs(a - b) / denom

        changed_mask = (rel_diff >= N2E_REL_THRESH) & cat_supported_mask
        n_changed = int(changed_mask.sum())
        fraction = n_changed / n_cat_supported if n_cat_supported > 0 else float("nan")

        out["probes"].append(
            {
                "h": probe_h,
                "n_events_matched": int(len(common_idx)),
                "n_catalogue_supported": n_cat_supported,
                "n_changed_ge_1e-6_rel": n_changed,
                "fraction_changed": fraction,
                "gate_pass_ge_10pct": bool(fraction >= N2E_FRACTION_GATE) if n_cat_supported > 0 else False,
            }
        )
    out["gate_pass"] = any(p["gate_pass_ge_10pct"] for p in out["probes"])
    return out


def _load_selection_tables(cell_dir: Path) -> dict[float, dict[str, Any]]:
    tables: dict[float, dict[str, Any]] = {}
    for f in sorted(cell_dir.glob("selection_tables_h_*.json")):
        d = json.loads(f.read_text())
        tables[round(float(d["h"]), 6)] = d
    return tables


def gate_n2_j(venue: str, eoff_dir: Path, jker_dir: Path, v0_dir: Path) -> dict[str, Any]:
    """N-2(J): max_h |r_Malm(jker)/r_Malm(point-mode reference) - 1| > 1e-4.

    Point-mode reference at all 41 h is the eoff cell's tables (eddington_m
    only touches the catalogue leg, not the selection tables -- verified
    below against v0 at the 2 probe h before use).
    """
    v0_tables = _load_selection_tables(v0_dir)
    eoff_tables = _load_selection_tables(eoff_dir)
    jker_tables = _load_selection_tables(jker_dir)

    # verification: eoff vs v0 at the 2 probe h -- eddington_m should not
    # touch selection tables at all (E only affects the catalogue leg).
    verify: dict[str, Any] = {}
    verify_all_match = True
    for probe_h in PROBE_H:
        key = round(probe_h, 6)
        matched_key = None
        for k in eoff_tables:
            if abs(k - probe_h) < 1e-6:
                matched_key = k
                break
        v0_key = None
        for k in v0_tables:
            if abs(k - probe_h) < 1e-6:
                v0_key = k
                break
        if matched_key is None or v0_key is None:
            verify[str(probe_h)] = {"error": "missing table at probe h", "eoff_key": matched_key, "v0_key": v0_key}
            verify_all_match = False
            continue
        e_tab = eoff_tables[matched_key]
        v_tab = v0_tables[v0_key]
        row = {}
        row_match = True
        for field in ("beta_G_phi", "beta_Gbar_phi", "sigma_phi", "sigma_4d", "r_Malm"):
            a = float(e_tab[field])
            b = float(v_tab[field])
            rel = abs(a - b) / max(abs(a), abs(b), 1.0)
            row[field] = {"eoff": a, "v0": b, "rel_diff": rel}
            if rel > 1e-12:
                row_match = False
        row["all_fields_match"] = row_match
        verify_all_match = verify_all_match and row_match
        verify[str(probe_h)] = row

    # r_Malm ratio table over the 41-h grid: jker / eoff(point-mode reference)
    common_h = sorted(set(jker_tables.keys()) & set(eoff_tables.keys()))
    ratio_rows = []
    for h in common_h:
        r_point = float(eoff_tables[h]["r_Malm"])
        r_kernel = float(jker_tables[h]["r_Malm"])
        ratio = r_kernel / r_point if r_point != 0 else float("nan")
        ratio_rows.append({"h": h, "r_Malm_point": r_point, "r_Malm_kernel": r_kernel, "ratio": ratio, "abs_ratio_minus_1": abs(ratio - 1.0)})

    max_dev = max((row["abs_ratio_minus_1"] for row in ratio_rows), default=float("nan"))
    ratios_only = [row["ratio"] for row in ratio_rows]

    return {
        "venue": venue,
        "eddington_m_selection_table_invariance_check": {
            "probes": verify,
            "all_match": verify_all_match,
            "note": "confirms eddington_m does not affect selection tables (catalogue-leg-only claim); if all_match is False this is flagged loudly",
        },
        "n_h_ratio_table": len(ratio_rows),
        "ratio_table": ratio_rows,
        "max_h_abs_ratio_minus_1": max_dev,
        "gate_pass_gt_1e-4": bool(max_dev > N2J_RATIO_THRESH),
        "ratio_min": float(np.min(ratios_only)) if ratios_only else float("nan"),
        "ratio_median": float(np.median(ratios_only)) if ratios_only else float("nan"),
        "ratio_max": float(np.max(ratios_only)) if ratios_only else float("nan"),
    }


# ---------------------------------------------------------------------------
# Score item 4/5: R-E and R-J reads
# ---------------------------------------------------------------------------


def read_re_rj(venue: str, baseline_df: pd.DataFrame, eoff_df: pd.DataFrame, jker_df: pd.DataFrame) -> dict[str, Any]:
    base_2d = _mean_h_convention(baseline_df, "combined_with_bh")
    base_1d = _mean_h_convention(baseline_df, "combined_no_bh")
    eoff_2d = _mean_h_convention(eoff_df, "combined_with_bh")
    eoff_1d = _mean_h_convention(eoff_df, "combined_no_bh")
    jker_2d = _mean_h_convention(jker_df, "combined_with_bh")
    jker_1d = _mean_h_convention(jker_df, "combined_no_bh")

    s_edd_new = base_2d["mean_h"] - eoff_2d["mean_h"]
    s_edd_1d_delta = base_1d["mean_h"] - eoff_1d["mean_h"]
    delta_j = jker_2d["mean_h"] - base_2d["mean_h"]
    delta_j_1d = jker_1d["mean_h"] - base_1d["mean_h"]

    return {
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
        "R_E": {
            "eoff_mean_h_2D": round(eoff_2d["mean_h"], 6),
            "eoff_sigma_h_2D": round(eoff_2d["sigma_h"], 6),
            "eoff_mean_h_1D": round(eoff_1d["mean_h"], 6),
            "eoff_sigma_h_1D": round(eoff_1d["sigma_h"], 6),
            "s_edd_new_2D": round(s_edd_new, 6),
            "s_edd_new_1D_delta": round(s_edd_1d_delta, 6),
            "s_edd_new_1D_delta_note": "expected 0 given N-1 bit-identical combined_no_bh",
            "n_events": eoff_2d["n_events"],
            "n_h": eoff_2d["n_h"],
        },
        "R_J": {
            "jker_mean_h_2D": round(jker_2d["mean_h"], 6),
            "jker_sigma_h_2D": round(jker_2d["sigma_h"], 6),
            "jker_mean_h_1D": round(jker_1d["mean_h"], 6),
            "jker_sigma_h_1D": round(jker_1d["sigma_h"], 6),
            "delta_j_2D": round(delta_j, 6),
            "delta_j_1D": round(delta_j_1d, 6),
            "n_events": jker_2d["n_events"],
            "n_h": jker_2d["n_h"],
        },
    }


# ---------------------------------------------------------------------------
# Score item 6: sanity
# ---------------------------------------------------------------------------


def sanity_report(venue: str, cells: dict[str, pd.DataFrame], dedupe: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"venue": venue, "row_counts": {}, "h_grids": {}}
    for variant, df in cells.items():
        n_events = df["event_idx"].nunique()
        n_h = df["h"].unique().size
        out["row_counts"][variant] = {
            "n_rows": int(len(df)),
            "n_events": int(n_events),
            "n_h": int(n_h),
            "expected_rows": 1588 * 2 if variant == "v0" else 1588 * 41,
            "matches_expected": bool(len(df) == (1588 * 2 if variant == "v0" else 1588 * 41)),
            "n_duplicates_removed": dedupe[variant]["n_duplicates_removed"],
        }
        out["h_grids"][variant] = [float(x) for x in np.sort(df["h"].unique())]
    return out


def main(argv: list[str] | None = None) -> None:
    t_start = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "readout_battery_output.json")
    args = parser.parse_args(argv)

    output: dict[str, Any] = {
        "preregistration": "PREREGISTRATION_TILT_BATTERY.md v2",
        "venues": {},
    }

    for venue in VENUES:
        print(f"=== {venue} ===", flush=True)
        baseline_df = _load_baseline(venue)

        cells: dict[str, pd.DataFrame] = {}
        cells_dirs: dict[str, Path] = {}
        dedupe_report: dict[str, Any] = {}
        for variant in VARIANTS:
            raw = _load_cell(variant, venue)
            df, n_dup = _dedupe(raw)
            cells[variant] = df
            cells_dirs[variant] = BATTERY_ROOT / f"{variant}_{venue}"
            dedupe_report[variant] = {"n_duplicates_removed": n_dup}

        venue_out: dict[str, Any] = {}

        # --- item 1: N-0 + metadata ---
        venue_out["N0_gate"] = gate_n0(venue, baseline_df, cells["v0"])
        venue_out["N0_metadata_diff"] = metadata_diff_check(venue, cells_dirs)

        # --- item 2: N-1 (E) full grid ---
        venue_out["N1_E_gate"] = gate_n1_eoff(venue, baseline_df, cells["eoff"])

        # --- item 3: N-2 (E) and N-2 (J) ---
        venue_out["N2_E_gate"] = gate_n2_e(venue, baseline_df, cells["eoff"])
        venue_out["N2_J_gate"] = gate_n2_j(
            venue,
            cells_dirs["eoff"],
            cells_dirs["jker"],
            cells_dirs["v0"],
        )

        # --- items 4/5: R-E, R-J ---
        venue_out["reads"] = read_re_rj(venue, baseline_df, cells["eoff"], cells["jker"])

        # --- item 6: sanity ---
        venue_out["sanity"] = sanity_report(venue, cells, dedupe_report)

        output["venues"][venue] = venue_out

    runtime_s = time.time() - t_start
    output["runtime_seconds"] = runtime_s

    args.output.write_text(json.dumps(output, indent=2, default=str))
    print(f"Wrote {args.output} in {runtime_s:.2f}s", flush=True)


if __name__ == "__main__":
    main()
