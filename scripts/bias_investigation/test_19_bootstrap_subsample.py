"""Test 19 (Tier 2): bootstrap-subsample MAP distribution at h_true=0.73.

Tests whether the +0.025 residual at h=0.73 is consistent with sample-size
statistical fluctuation. Reuses the cached cluster per-event likelihoods
(no cluster work needed).

For each subsample size N in {100, 200, 300, 412} and each of B=1000 bootstrap
iterations:
  - draw N event indices with replacement from the 412-event pool;
  - compute joint log p(h) = Σ log L_i(h) − N · log D(h) on the cluster h-grid;
  - record MAP via discrete argmax + parabolic refinement near peak.

Outputs the distribution of MAPs per N. The N=412 distribution gives σ_boot
under resampling-with-replacement (matches T08 procedure). Smaller N exposes
how MAP wanders as a function of sample size — if N=412 sits in the upper
tail of the smaller-N distributions, the +0.025 is plausibly an unlucky
realization on the existing sample.

Also runs an event-stratified ablation: drops the top-10 |pull| events
identified by Audit A4, recomputes MAP. If MAP collapses toward truth, the
bias is concentrated in those events.

Outputs:
  scripts/bias_investigation/outputs/phase45/bootstrap_subsample.json

Run from project root:
    uv run python scripts/bias_investigation/test_19_bootstrap_subsample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CLUSTER_DIR = PROJECT_ROOT / "simulations" / "cluster_run_phase45_20260501"
DIAGNOSTIC_CSV = CLUSTER_DIR / "event_likelihoods.csv"
PREPARED_CSV = CLUSTER_DIR / "prepared_cramer_rao_bounds.csv"
COMBINED_JSON = PROJECT_ROOT / "results" / "phase45_v2_posteriors" / "combined_posterior.json"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"

H_TRUTH = 0.73
H_BIAS_DIRECTION = 0.755

SUBSAMPLE_SIZES = [100, 200, 300, 412]
N_BOOTSTRAP = 1000
RNG_SEED = 20260504  # date of run

# Top-10 ablation per A4: drop the events with the largest |pull| toward h_bias
TOP_K_ABLATION = 10


def load_per_event_log_L(
    diagnostic_csv: Path, h_grid: np.ndarray, channel: str
) -> tuple[np.ndarray, list[int]]:
    """Build per-event log-L matrix on cluster h_grid.

    Returns (log_L[n_events, n_h], event_idx_list). The append-mode CSV is
    deduplicated to the latest entry per (event_idx, h) — same protocol as
    test_16.
    """
    diag_raw = pd.read_csv(diagnostic_csv)
    diag_raw["_row_order"] = np.arange(len(diag_raw))
    diag = (
        diag_raw.sort_values("_row_order")
        .groupby(["event_idx", "h"], as_index=False)
        .last()
        .drop(columns=["_row_order"])
    )

    column = "combined_no_bh" if channel == "no_bh" else "combined_with_bh"
    event_idx_list = sorted(diag["event_idx"].unique().tolist())
    log_L = np.full((len(event_idx_list), len(h_grid)), np.nan)
    for i, ev in enumerate(event_idx_list):
        sub = diag[diag["event_idx"] == ev].sort_values("h")
        for j, hv in enumerate(h_grid):
            idx = int(np.argmin(np.abs(sub["h"].values - hv)))
            if abs(sub["h"].values[idx] - hv) < 1e-4:
                log_L[i, j] = float(np.log(max(sub[column].values[idx], 1e-300)))

    # Drop events with any NaN entries (incomplete h coverage)
    full_mask = ~np.isnan(log_L).any(axis=1)
    log_L = log_L[full_mask]
    event_idx_list = [event_idx_list[i] for i, ok in enumerate(full_mask) if ok]
    return log_L, event_idx_list


def parabolic_refine(h_grid: np.ndarray, log_post: np.ndarray) -> float:
    """Three-point parabolic interpolation around discrete argmax for sub-grid MAP.

    Falls back to grid argmax at the edges.
    """
    i = int(np.argmax(log_post))
    if i <= 0 or i >= len(h_grid) - 1:
        return float(h_grid[i])
    h0, h1, h2 = h_grid[i - 1], h_grid[i], h_grid[i + 1]
    y0, y1, y2 = log_post[i - 1], log_post[i], log_post[i + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(h1)
    h_max = h1 - 0.5 * (h2 - h0) * (y2 - y0) / (2 * (y0 - 2 * y1 + y2))
    return float(h_max)


def map_for_indices(
    log_L: np.ndarray, log_D: np.ndarray, h_grid: np.ndarray, idx: np.ndarray
) -> float:
    """Joint MAP for a subsample of events (indices into log_L)."""
    n_events = len(idx)
    sum_log_L = log_L[idx].sum(axis=0)
    log_post = sum_log_L - n_events * log_D
    return parabolic_refine(h_grid, log_post)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print("TIER 2 — Bootstrap-subsample MAP distribution at h_true=0.73")
    print("=" * 70)

    # 1. Load combined posterior for h-grid and D(h).
    with open(COMBINED_JSON) as f:
        cp = json.load(f)
    h_grid = np.asarray(cp["h_values"])
    D_h = np.asarray(cp["D_h_per_h"])
    log_D = np.log(D_h)
    print(f"Loaded h-grid: {len(h_grid)} values, [{h_grid.min():.3f}, {h_grid.max():.3f}]")

    # 2. Load per-event log-L matrices for both channels.
    print("Building per-event log-L matrix (1D channel, no_bh)...")
    log_L_1d, event_idx_1d = load_per_event_log_L(DIAGNOSTIC_CSV, h_grid, "no_bh")
    print(f"  {log_L_1d.shape[0]} events × {log_L_1d.shape[1]} h-values")

    print("Building per-event log-L matrix (2D channel, with_bh)...")
    log_L_2d, event_idx_2d = load_per_event_log_L(DIAGNOSTIC_CSV, h_grid, "with_bh")
    print(f"  {log_L_2d.shape[0]} events × {log_L_2d.shape[1]} h-values")

    # 3. Sanity check: full-sample MAP recovers cluster value (0.755 / 0.745).
    full_idx_1d = np.arange(log_L_1d.shape[0])
    full_idx_2d = np.arange(log_L_2d.shape[0])
    full_map_1d = map_for_indices(log_L_1d, log_D, h_grid, full_idx_1d)
    full_map_2d = map_for_indices(log_L_2d, log_D, h_grid, full_idx_2d)
    print(f"\nFull-sample sanity: 1D MAP={full_map_1d:.4f}, 2D MAP={full_map_2d:.4f}")
    print("  (cluster reported 0.7550 / 0.7450)")

    results: dict = {
        "rng_seed": RNG_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "subsample_sizes": SUBSAMPLE_SIZES,
        "h_truth": H_TRUTH,
        "full_sample": {
            "n_events_1d": int(log_L_1d.shape[0]),
            "n_events_2d": int(log_L_2d.shape[0]),
            "map_1d": full_map_1d,
            "map_2d": full_map_2d,
        },
        "channels": {},
    }

    # 4. Bootstrap subsample for each channel and N.
    for channel, log_L, n_total in [
        ("no_bh", log_L_1d, log_L_1d.shape[0]),
        ("with_bh", log_L_2d, log_L_2d.shape[0]),
    ]:
        print(f"\n--- Channel: {channel} ({n_total} events) ---")
        ch_results: dict = {}
        for N in SUBSAMPLE_SIZES:
            if N > n_total:
                continue
            maps = np.empty(N_BOOTSTRAP)
            for b in range(N_BOOTSTRAP):
                idx = rng.choice(n_total, size=N, replace=True)
                maps[b] = map_for_indices(log_L, log_D, h_grid, idx)
            mean_map = float(np.mean(maps))
            std_map = float(np.std(maps, ddof=1))
            q05, q16, q50, q84, q95 = np.percentile(maps, [5, 16, 50, 84, 95]).tolist()
            # Position of the full-sample MAP within this distribution.
            full_map = full_map_1d if channel == "no_bh" else full_map_2d
            quantile_full = float(np.mean(maps <= full_map))
            print(
                f"  N={N:>4}: μ={mean_map:.4f}  σ={std_map:.4f}  "
                f"q[5,16,50,84,95]=[{q05:.3f},{q16:.3f},{q50:.3f},{q84:.3f},{q95:.3f}]  "
                f"P(MAP_b ≤ MAP_full={full_map:.3f})={quantile_full:.3f}"
            )
            ch_results[f"N={N}"] = {
                "N": N,
                "map_mean": mean_map,
                "map_std": std_map,
                "map_quantiles": {
                    "q05": q05,
                    "q16": q16,
                    "q50": q50,
                    "q84": q84,
                    "q95": q95,
                },
                "full_map_quantile_position": quantile_full,
                "full_map": float(full_map),
            }
        results["channels"][channel] = ch_results

    # 5. Top-K ablation per A4. Drop top-10 |pull| events from 1D channel and
    #    recompute MAP. The "pull" is log_L_combined(h_bias=0.755) - log_L(h_truth=0.73).
    print(f"\n--- Top-{TOP_K_ABLATION} ablation (1D channel) ---")
    i_truth = int(np.argmin(np.abs(h_grid - H_TRUTH)))
    i_bias = int(np.argmin(np.abs(h_grid - H_BIAS_DIRECTION)))
    pull = log_L_1d[:, i_bias] - log_L_1d[:, i_truth]
    abs_pull = np.abs(pull)
    top_k_idx = np.argsort(abs_pull)[::-1][:TOP_K_ABLATION]
    keep_mask = np.ones(log_L_1d.shape[0], dtype=bool)
    keep_mask[top_k_idx] = False
    ablated_idx = np.where(keep_mask)[0]
    map_ablated = map_for_indices(log_L_1d, log_D, h_grid, ablated_idx)
    map_top_k_only = map_for_indices(log_L_1d, log_D, h_grid, top_k_idx)
    print(f"  Full sample MAP (1D):              {full_map_1d:.4f}")
    print(
        f"  Drop top-{TOP_K_ABLATION} |pull| events:        {map_ablated:.4f}  "
        f"(Δ vs full = {map_ablated - full_map_1d:+.4f})"
    )
    print(f"  Top-{TOP_K_ABLATION} only (sanity check):       {map_top_k_only:.4f}")
    results["ablation"] = {
        "channel": "no_bh",
        "top_k": TOP_K_ABLATION,
        "top_k_event_indices": [int(event_idx_1d[i]) for i in top_k_idx],
        "full_sample_map": full_map_1d,
        "drop_top_k_map": map_ablated,
        "top_k_only_map": map_top_k_only,
        "delta_drop_top_k": map_ablated - full_map_1d,
    }

    # 6. Verdict.
    n_412_results = results["channels"]["no_bh"].get("N=412")
    if n_412_results is not None:
        sigma_boot = n_412_results["map_std"]
        bias_observed = full_map_1d - H_TRUTH
        z_score = bias_observed / sigma_boot if sigma_boot > 0 else float("inf")
        print("\n=== VERDICT (1D channel) ===")
        print(f"  σ_boot (N=412) = {sigma_boot:.4f}")
        print(f"  bias = MAP - h_truth = {bias_observed:+.4f}")
        print(f"  z = bias / σ_boot = {z_score:+.2f}")
        if abs(z_score) < 2:
            verdict = "WITHIN_2_SIGMA — bias plausibly statistical fluctuation"
        elif abs(z_score) < 3:
            verdict = "MARGINAL_2_TO_3_SIGMA — borderline; would benefit from D(h) audit"
        else:
            verdict = "BEYOND_3_SIGMA — systematic, requires D(h) or fresh-injection audit"
        print(f"  verdict: {verdict}")
        results["verdict"] = {
            "sigma_boot_N412": sigma_boot,
            "bias_observed": bias_observed,
            "z_score": z_score,
            "verdict": verdict,
        }

    # 7. Save.
    out_path = OUTPUT_DIR / "bootstrap_subsample.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
