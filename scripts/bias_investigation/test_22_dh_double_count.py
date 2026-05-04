"""Test 22 (Tier 3f): verify D(h) is double-applied in the joint posterior.

Hypothesis: D(h) appears TWICE in the joint posterior:

  (a) INSIDE per-event L_comp (Phase 32 commit fc7c84c):
      L_comp_i(h) = numerator_i(h) / D(h)
      then combined_i = f_i · L_cat_i + (1-f_i) · L_comp_i

  (b) OUTSIDE in combine_log_space (Phase 43-H1 commit 2853c32):
      joint_log = Σ log combined_i  −  N · log D(h)

For events with f_i ≈ 0 (completion-dominated), the contribution becomes:
  log(num_i / D) − log D = log num_i  −  2·log D

For events with f_i ≈ 1 (catalog-dominated):
  log L_cat_i − log D

Net effect: completion-dominated events have D applied 2×, catalog-dominated 1×.
The mix produces a structural bias in MAP whose magnitude depends on the f_i
distribution.

Phase 43's verification used a TOY D(h) ∝ h³ (increasing with h), so the −N log D
correction did the right thing back then. Phase 44 changed the actual D(h) to
DECREASING with h. Now −N log D over-corrects for completion-dominated events
and shifts MAP up.

This test: recompute joint posterior WITHOUT the outer −N log D term at both
h_true=0.65 (closure fine-grid) and h_true=0.73 (production 412 events). If
both MAPs land within 2·σ_boot of truth, double-counting is the bug.

Output: scripts/bias_investigation/outputs/phase45/dh_double_count.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

# Closure (fine grid) inputs
CLOSURE_DIR = PROJECT_ROOT / "simulations" / "cluster_run_closure_h065_20260503_finegrid"
CLOSURE_POSTERIORS_1D = CLOSURE_DIR / "posteriors"
CLOSURE_POSTERIORS_2D = CLOSURE_DIR / "posteriors_with_bh_mass"

# Production (h_true=0.73) inputs
PROD_DIAG_CSV = (
    PROJECT_ROOT / "simulations" / "cluster_run_phase45_20260501" / "event_likelihoods.csv"
)
PROD_COMBINED_JSON = PROJECT_ROOT / "results" / "phase45_v2_posteriors" / "combined_posterior.json"
PROD_COMBINED_2D_JSON = (
    PROJECT_ROOT / "results" / "phase45_v2_posteriors_with_bh_mass" / "combined_posterior.json"
)

H_TRUTH_CLOSURE = 0.65
H_TRUTH_PRODUCTION = 0.73
N_BOOTSTRAP = 1000
RNG_SEED = 20260504


def _h_from_filename(path: Path) -> float:
    m = re.match(r"h_(\d+)_(\d+)\.json", path.name)
    if m is None:
        return float("nan")
    return float(f"{m.group(1)}.{m.group(2)}")


def load_per_h_likelihoods(directory: Path) -> tuple[list[float], np.ndarray]:
    files = sorted(directory.glob("h_*.json"), key=_h_from_filename)
    h_values: list[float] = [_h_from_filename(f) for f in files]
    event_indices: set[int] = set()
    per_h_data: list[dict[int, float]] = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        per_h: dict[int, float] = {}
        for k, v in d.items():
            try:
                ev = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                if len(v) == 0:
                    continue
                val = v[0]
            else:
                val = v
            event_indices.add(ev)
            per_h[ev] = float(val)
        per_h_data.append(per_h)
    events_sorted = sorted(event_indices)
    log_L = np.full((len(events_sorted), len(h_values)), np.nan)
    for j, per_h in enumerate(per_h_data):
        for i, ev in enumerate(events_sorted):
            if ev in per_h:
                log_L[i, j] = float(np.log(max(per_h[ev], 1e-300)))
    full_mask = ~np.isnan(log_L).any(axis=1)
    log_L = log_L[full_mask]
    return h_values, log_L


def load_per_event_log_L_from_csv(
    csv_path: Path, h_grid: np.ndarray, channel: str
) -> tuple[np.ndarray, list[int]]:
    """Build per-event log-L matrix from event_likelihoods.csv.

    Same dedup-latest logic as test_19.
    """
    column = "combined_no_bh" if channel == "no_bh" else "combined_with_bh"
    diag_raw = pd.read_csv(csv_path)
    diag_raw["_row_order"] = np.arange(len(diag_raw))
    diag = (
        diag_raw.sort_values("_row_order")
        .groupby(["event_idx", "h"], as_index=False)
        .last()
        .drop(columns=["_row_order"])
    )
    event_idx_list = sorted(diag["event_idx"].unique().tolist())
    log_L = np.full((len(event_idx_list), len(h_grid)), np.nan)
    for i, ev in enumerate(event_idx_list):
        sub = diag[diag["event_idx"] == ev].sort_values("h")
        for j, hv in enumerate(h_grid):
            idx = int(np.argmin(np.abs(sub["h"].values - hv)))
            if abs(sub["h"].values[idx] - hv) < 1e-4:
                log_L[i, j] = float(np.log(max(sub[column].values[idx], 1e-300)))
    full_mask = ~np.isnan(log_L).any(axis=1)
    log_L = log_L[full_mask]
    event_idx_list = [event_idx_list[i] for i, ok in enumerate(full_mask) if ok]
    return log_L, event_idx_list


def parabolic_refine(h_grid: np.ndarray, log_post: np.ndarray) -> float:
    i = int(np.argmax(log_post))
    if i <= 0 or i >= len(h_grid) - 1:
        return float(h_grid[i])
    h0, h1, h2 = h_grid[i - 1], h_grid[i], h_grid[i + 1]
    y0, y1, y2 = log_post[i - 1], log_post[i], log_post[i + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(h1)
    return float(h1 - 0.5 * (h2 - h0) * (y2 - y0) / (2 * denom))


def map_with_corrections(
    log_L: np.ndarray,
    log_D: np.ndarray,
    h_grid: np.ndarray,
    *,
    n_log_D_coeff: float,
) -> tuple[float, float, np.ndarray]:
    """Joint MAP with custom −c·N·log D coefficient.

    Returns (discrete MAP, continuous MAP via parabolic refine, joint_log array).
    """
    n_events = log_L.shape[0]
    sum_log_L = log_L.sum(axis=0)
    joint = sum_log_L - n_log_D_coeff * n_events * log_D
    discrete = float(h_grid[int(np.argmax(joint))])
    continuous = parabolic_refine(h_grid, joint)
    return discrete, continuous, joint


def bootstrap_map_with_corrections(
    log_L: np.ndarray,
    log_D: np.ndarray,
    h_grid: np.ndarray,
    n_boot: int,
    coeff: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_events = log_L.shape[0]
    maps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n_events, size=n_events, replace=True)
        sum_log_L = log_L[idx].sum(axis=0)
        joint = sum_log_L - coeff * n_events * log_D
        maps[b] = parabolic_refine(h_grid, joint)
    return maps


def analyze_case(
    label: str,
    log_L: np.ndarray,
    log_D: np.ndarray,
    h_grid: np.ndarray,
    h_truth: float,
    rng: np.random.Generator,
) -> dict:
    print(f"\n=== {label} ===")
    print(f"  N events: {log_L.shape[0]}, h-grid: [{h_grid[0]:.3f}..{h_grid[-1]:.3f}]")

    results = {}
    for coeff, tag in [
        (0.0, "no_outer_correction"),
        (1.0, "current_outer_minus_N_log_D"),
    ]:
        d_map, c_map, _ = map_with_corrections(log_L, log_D, h_grid, n_log_D_coeff=coeff)
        boot_maps = bootstrap_map_with_corrections(log_L, log_D, h_grid, N_BOOTSTRAP, coeff, rng)
        sigma_boot = float(np.std(boot_maps, ddof=1))
        bias = c_map - h_truth
        z = bias / sigma_boot if sigma_boot > 0 else float("inf")
        verdict = (
            "PASS (|z|≤2)"
            if abs(z) <= 2
            else ("MARGINAL (2<|z|≤3)" if abs(z) <= 3 else "FAIL (|z|>3)")
        )
        print(
            f"  c={coeff:.1f}  discrete={d_map:.4f}  continuous={c_map:.4f}  "
            f"bias={bias:+.4f}  σ_boot={sigma_boot:.4f}  z={z:+.2f}  {verdict}"
        )
        results[tag] = {
            "coeff_outer_minus_N_log_D": coeff,
            "discrete_map": d_map,
            "continuous_map": c_map,
            "bias": bias,
            "sigma_boot": sigma_boot,
            "z_score": z,
            "verdict": verdict,
        }
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print("TIER 3f — D(h) double-application test")
    print("=" * 70)

    # --- Closure h_true=0.65 fine grid ---
    # Use the closure log_D from the closure_finegrid analyzer's saved JSON
    closure_finegrid_json = OUTPUT_DIR / "closure_h065_finegrid.json"
    with open(closure_finegrid_json) as f:
        closure_data = json.load(f)

    closure_results = {}
    for label_short, posteriors_dir, channel_data in [
        ("closure_h065_1D", CLOSURE_POSTERIORS_1D, closure_data["without_bh_mass (1D)"]),
        ("closure_h065_2D", CLOSURE_POSTERIORS_2D, closure_data["with_bh_mass (2D)"]),
    ]:
        h_values, log_L = load_per_h_likelihoods(posteriors_dir)
        h_grid = np.asarray(h_values)
        # Reconstruct log_D from D_term: D_term = -N log D so log D = -D_term / N
        n_events = channel_data["n_events"]
        D_term = np.asarray(channel_data["D_term_per_h"])
        log_D = -D_term / n_events

        closure_results[label_short] = analyze_case(
            label_short, log_L, log_D, h_grid, H_TRUTH_CLOSURE, rng
        )

    # --- Production h_true=0.73 (412 events from event_likelihoods.csv) ---
    # log_D for production is from combined_posterior.json
    print("\n--- Loading production data ---")
    with open(PROD_COMBINED_JSON) as f:
        prod_combined = json.load(f)
    h_grid_prod = np.asarray(prod_combined["h_values"])
    log_D_prod = np.log(np.asarray(prod_combined["D_h_per_h"]))

    print("Building per-event log-L (1D, no_bh) from production CSV...")
    log_L_1d, _ = load_per_event_log_L_from_csv(PROD_DIAG_CSV, h_grid_prod, "no_bh")
    print(f"  {log_L_1d.shape[0]} events × {log_L_1d.shape[1]} h-values")

    print("Building per-event log-L (2D, with_bh) from production CSV...")
    log_L_2d, _ = load_per_event_log_L_from_csv(PROD_DIAG_CSV, h_grid_prod, "with_bh")
    print(f"  {log_L_2d.shape[0]} events × {log_L_2d.shape[1]} h-values")

    prod_results = {
        "production_h073_1D": analyze_case(
            "production_h073_1D", log_L_1d, log_D_prod, h_grid_prod, H_TRUTH_PRODUCTION, rng
        ),
        "production_h073_2D": analyze_case(
            "production_h073_2D", log_L_2d, log_D_prod, h_grid_prod, H_TRUTH_PRODUCTION, rng
        ),
    }

    summary = {
        "rng_seed": RNG_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "h_truth_closure": H_TRUTH_CLOSURE,
        "h_truth_production": H_TRUTH_PRODUCTION,
        "closure": closure_results,
        "production": prod_results,
        "interpretation": (
            "Test the hypothesis that D(h) is applied 2× in the joint: once "
            "inside L_comp = num/D (Phase 32 fc7c84c), and once outside via "
            "joint = Σ log combined_i − N log D (Phase 43-H1 2853c32). If c=0 "
            "(no outer correction) gives MAPs within 2·σ_boot of truth at BOTH "
            "h_true=0.65 and h_true=0.73, the outer correction is double-counting."
        ),
    }
    out_path = OUTPUT_DIR / "dh_double_count.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
