"""Phase 32 Plan 02: Validation of completion term fix.

Runs evaluation across h-grid, computes MAP comparison, bias-vs-N convergence,
and extracts per-event L_comp decomposition.

% ASSERT_CONVENTION: natural_units=SI, metric_signature=mostly_plus, distance=luminosity_distance_Gpc
"""

import glob
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_LOGGER = logging.getLogger(__name__)


def load_posteriors_from_dir(posteriors_dir: str) -> dict[float, dict[int, float]]:
    """Load per-event likelihoods from posterior JSON files.

    Returns: {h_value: {event_idx: likelihood_value}}
    """
    result: dict[float, dict[int, float]] = {}
    files = sorted(glob.glob(os.path.join(posteriors_dir, "h_*.json")))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        h = d["h"]
        events = {}
        for k, v in d.items():
            if k == "h":
                continue
            if isinstance(v, list) and len(v) > 0:
                events[int(k)] = v[0]
            elif isinstance(v, int | float):
                events[int(k)] = v
        result[h] = events
    return result


def compute_combined_posterior(
    posteriors: dict[float, dict[int, float]],
    max_events: int | None = None,
) -> tuple[list[float], list[float]]:
    """Compute combined posterior p(h|data) = product of per-event likelihoods.

    Args:
        posteriors: {h: {event_idx: likelihood}}
        max_events: If set, use only the first N events (sorted by index).

    Returns:
        (h_values, normalized_posterior)
    """
    h_sorted = sorted(posteriors.keys())

    # Determine which events to use
    all_events = set()
    for h in h_sorted:
        all_events |= set(posteriors[h].keys())
    event_indices = sorted(all_events)
    if max_events is not None:
        event_indices = event_indices[:max_events]

    # Compute log-posterior for each h
    log_posts = []
    for h in h_sorted:
        log_sum = 0.0
        for ev in event_indices:
            val = posteriors[h].get(ev, 1e-300)
            log_sum += np.log(max(val, 1e-300))
        log_posts.append(log_sum)

    log_posts_arr = np.array(log_posts)
    log_posts_arr -= log_posts_arr.max()
    posts = np.exp(log_posts_arr)
    total = posts.sum()
    if total > 0:
        posts /= total

    return h_sorted, list(posts)


def find_map(h_values: list[float], posterior: list[float]) -> float:
    """Find MAP estimate (mode of posterior)."""
    return h_values[np.argmax(posterior)]


def run_evaluation_sweep(h_grid: list[float], working_dir: str) -> None:
    """Run the evaluation pipeline for each h in the grid."""
    for i, h in enumerate(h_grid):
        _LOGGER.info(f"Running evaluation for h={h:.2f} ({i + 1}/{len(h_grid)})...")
        t0 = time.time()

        # Use subprocess to run the evaluation
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "master_thesis_code",
                working_dir,
                "--evaluate",
                "--h_value",
                str(h),
                "--num_workers",
                "6",
                "--log_level",
                "WARNING",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        elapsed = time.time() - t0
        if result.returncode != 0:
            _LOGGER.error(f"  FAILED (h={h}): {result.stderr[-500:]}")
            raise RuntimeError(f"Evaluation failed for h={h}")
        _LOGGER.info(f"  Done h={h:.2f} in {elapsed:.1f}s")


def main() -> None:
    working_dir = "simulations"

    # --- Step 1: Run evaluation sweep ---
    h_grid = [round(0.60 + i * 0.01, 2) for i in range(27)]  # 0.60 to 0.86
    _LOGGER.info(f"H-grid: {h_grid}")

    _LOGGER.info("Running evaluation sweep with fixed code...")
    t0_total = time.time()
    run_evaluation_sweep(h_grid, working_dir)
    _LOGGER.info(f"Sweep complete in {time.time() - t0_total:.1f}s")

    # --- Step 2: Load post-fix posteriors ---
    posteriors_no_bh = load_posteriors_from_dir(os.path.join(working_dir, "posteriors"))
    posteriors_with_bh = load_posteriors_from_dir(
        os.path.join(working_dir, "posteriors_with_bh_mass")
    )

    # --- Step 3: MAP comparison ---
    h_vals, post_no_bh = compute_combined_posterior(posteriors_no_bh)
    h_vals_bh, post_with_bh = compute_combined_posterior(posteriors_with_bh)

    map_after_no_bh = find_map(h_vals, post_no_bh)
    map_after_with_bh = find_map(h_vals_bh, post_with_bh)

    # Pre-fix baseline (from archived posteriors, excluding contaminated h=0.73)
    pre_fix_no_bh = load_posteriors_from_dir(
        os.path.join(working_dir, "archive/pre_fix_posteriors/posteriors")
    )
    pre_fix_with_bh = load_posteriors_from_dir(
        os.path.join(working_dir, "archive/pre_fix_posteriors/posteriors_with_bh_mass")
    )
    # Remove contaminated h=0.73
    pre_fix_no_bh = {h: v for h, v in pre_fix_no_bh.items() if abs(h - 0.73) > 0.001}
    pre_fix_with_bh = {h: v for h, v in pre_fix_with_bh.items() if abs(h - 0.73) > 0.001}

    h_vals_pre, post_pre_no_bh = compute_combined_posterior(pre_fix_no_bh)
    h_vals_pre_bh, post_pre_with_bh = compute_combined_posterior(pre_fix_with_bh)

    map_before_no_bh = find_map(h_vals_pre, post_pre_no_bh)
    map_before_with_bh = find_map(h_vals_pre_bh, post_pre_with_bh)

    n_events = len(set().union(*[set(posteriors_no_bh[h].keys()) for h in posteriors_no_bh]))

    map_comparison = {
        "MAP_before_no_bh": map_before_no_bh,
        "MAP_after_no_bh": map_after_no_bh,
        "MAP_before_with_bh": map_before_with_bh,
        "MAP_after_with_bh": map_after_with_bh,
        "bias_before_no_bh": (map_before_no_bh - 0.73) / 0.73,
        "bias_after_no_bh": (map_after_no_bh - 0.73) / 0.73,
        "bias_before_with_bh": (map_before_with_bh - 0.73) / 0.73,
        "bias_after_with_bh": (map_after_with_bh - 0.73) / 0.73,
        "n_events": n_events,
        "snr_threshold": 20,
        "h_grid": h_vals,
        "posterior_no_bh": post_no_bh,
        "posterior_with_bh": post_with_bh,
        "posterior_pre_no_bh": post_pre_no_bh,
        "posterior_pre_with_bh": post_pre_with_bh,
        "h_grid_pre": h_vals_pre,
        "production_baseline_no_bh": {"MAP": 0.66, "n_events": 531, "snr_threshold": 15},
        "production_baseline_with_bh": {"MAP": 0.68, "n_events": 527, "snr_threshold": 15},
    }

    out_dir = ".gpd/phases/32-completion-term-fix/validation"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "map_comparison.json"), "w") as f:
        json.dump(map_comparison, f, indent=2)
    _LOGGER.info(f"MAP comparison saved to {out_dir}/map_comparison.json")

    # Print results
    print("\n" + "=" * 60)
    print("MAP COMPARISON (before vs after fix)")
    print("=" * 60)
    print(f"Events: {n_events} (SNR >= 20)")
    print("\nWithout BH mass:")
    print(
        f"  Before: MAP h = {map_before_no_bh:.2f}, bias = {(map_before_no_bh - 0.73) / 0.73:.4f}"
    )
    print(f"  After:  MAP h = {map_after_no_bh:.2f}, bias = {(map_after_no_bh - 0.73) / 0.73:.4f}")
    print(f"  Shift:  {map_after_no_bh - map_before_no_bh:+.2f}")
    print("\nWith BH mass:")
    print(
        f"  Before: MAP h = {map_before_with_bh:.2f}, bias = {(map_before_with_bh - 0.73) / 0.73:.4f}"
    )
    print(
        f"  After:  MAP h = {map_after_with_bh:.2f}, bias = {(map_after_with_bh - 0.73) / 0.73:.4f}"
    )
    print(f"  Shift:  {map_after_with_bh - map_before_with_bh:+.2f}")

    # --- Step 4: Bias-vs-N convergence ---
    N_values = [10, 20, 30, 40, 50, 60]
    # Adjust max N to actual event count
    N_values = [n for n in N_values if n <= n_events]
    if n_events not in N_values:
        N_values.append(n_events)

    bias_no_bh = []
    bias_with_bh = []
    map_no_bh_vs_n = []
    map_with_bh_vs_n = []

    for N in N_values:
        h_v, p_no_bh = compute_combined_posterior(posteriors_no_bh, max_events=N)
        h_v_bh, p_with_bh = compute_combined_posterior(posteriors_with_bh, max_events=N)
        m_no_bh = find_map(h_v, p_no_bh)
        m_with_bh = find_map(h_v_bh, p_with_bh)
        b_no_bh = (m_no_bh - 0.73) / 0.73
        b_with_bh = (m_with_bh - 0.73) / 0.73
        bias_no_bh.append(b_no_bh)
        bias_with_bh.append(b_with_bh)
        map_no_bh_vs_n.append(m_no_bh)
        map_with_bh_vs_n.append(m_with_bh)

    bias_vs_n = {
        "N_values": N_values,
        "bias_without_bh": bias_no_bh,
        "bias_with_bh": bias_with_bh,
        "MAP_without_bh": map_no_bh_vs_n,
        "MAP_with_bh": map_with_bh_vs_n,
    }

    with open(os.path.join(out_dir, "bias_vs_n.json"), "w") as f:
        json.dump(bias_vs_n, f, indent=2)
    _LOGGER.info(f"Bias-vs-N saved to {out_dir}/bias_vs_n.json")

    print("\n" + "=" * 60)
    print("BIAS-vs-N CONVERGENCE")
    print("=" * 60)
    print(f"{'N':>5} {'MAP_no_bh':>10} {'bias_no_bh':>12} {'MAP_bh':>10} {'bias_bh':>12}")
    for i, N in enumerate(N_values):
        print(
            f"{N:5d} {map_no_bh_vs_n[i]:10.2f} {bias_no_bh[i]:12.4f} {map_with_bh_vs_n[i]:10.2f} {bias_with_bh[i]:12.4f}"
        )

    # --- Step 5: Per-event L_comp decomposition ---
    diag_csv = os.path.join(working_dir, "diagnostics/event_likelihoods.csv")
    if os.path.exists(diag_csv):
        diag = pd.read_csv(diag_csv)
        _LOGGER.info(f"Diagnostic CSV: {len(diag)} rows")

        # Select representative events (pick 5 with varied L_comp)
        h_vals_diag = sorted(diag["h"].unique())
        event_indices = sorted(diag["event_idx"].unique())

        # Sample events at different L_comp levels
        if len(event_indices) >= 5:
            sample_events = [
                event_indices[0],
                event_indices[len(event_indices) // 4],
                event_indices[len(event_indices) // 2],
                event_indices[3 * len(event_indices) // 4],
                event_indices[-1],
            ]
        else:
            sample_events = event_indices

        lcomp_decomp = {
            "events": [int(e) for e in sample_events],
            "h_values": [float(h) for h in h_vals_diag],
            "L_comp": {},
            "L_cat": {},
            "f_i": {},
        }

        for ev in sample_events:
            ev_data = diag[diag["event_idx"] == ev].sort_values("h")
            lcomp_decomp["L_comp"][str(ev)] = ev_data["L_comp"].tolist()
            lcomp_decomp["L_cat"][str(ev)] = ev_data["L_cat_no_bh"].tolist()
            lcomp_decomp["f_i"][str(ev)] = ev_data["f_i"].tolist()

        with open(os.path.join(out_dir, "lcomp_decomposition.json"), "w") as f:
            json.dump(lcomp_decomp, f, indent=2)
        _LOGGER.info(f"L_comp decomposition saved to {out_dir}/lcomp_decomposition.json")

        # Print summary
        print("\n" + "=" * 60)
        print("PER-EVENT L_comp DECOMPOSITION (sample events)")
        print("=" * 60)
        for ev in sample_events:
            ev_data = diag[diag["event_idx"] == ev].sort_values("h")
            lcomp_vals = ev_data["L_comp"].values
            lcomp_range = lcomp_vals.max() / max(lcomp_vals.min(), 1e-300)
            print(f"\n  Event {ev}:")
            print(f"    f_i range: [{ev_data['f_i'].min():.4f}, {ev_data['f_i'].max():.4f}]")
            print(
                f"    L_comp range: [{lcomp_vals.min():.4e}, {lcomp_vals.max():.4e}], ratio={lcomp_range:.2f}"
            )
    else:
        _LOGGER.warning("No diagnostic CSV found — L_comp decomposition unavailable")
        _LOGGER.warning("(Diagnostics are only written when evaluating for a single h at a time)")

    # --- Step 6: Check stop conditions ---
    print("\n" + "=" * 60)
    print("STOP CONDITION CHECKS")
    print("=" * 60)

    # Check if MAP moved away from 0.73
    if map_after_no_bh < map_before_no_bh and abs(map_before_no_bh - 0.73) < abs(
        map_after_no_bh - 0.73
    ):
        print("  [FAIL] MAP moved AWAY from 0.73 for 'without BH mass' channel")
    else:
        print("  [PASS] MAP shift direction OK for 'without BH mass'")

    if map_after_with_bh < map_before_with_bh and abs(map_before_with_bh - 0.73) < abs(
        map_after_with_bh - 0.73
    ):
        print("  [FAIL] MAP moved AWAY from 0.73 for 'with BH mass' channel")
    else:
        print("  [PASS] MAP shift direction OK for 'with BH mass'")

    # Check NaN/zero L_comp
    if os.path.exists(diag_csv):
        diag = pd.read_csv(diag_csv)
        n_zero_lcomp = (diag["L_comp"] == 0).sum()
        n_nan_lcomp = diag["L_comp"].isna().sum()
        n_total = len(diag)
        pct_problematic = (n_zero_lcomp + n_nan_lcomp) / n_total * 100
        print(
            f"  L_comp zeros: {n_zero_lcomp}/{n_total}, NaN: {n_nan_lcomp}/{n_total} ({pct_problematic:.1f}%)"
        )
        if pct_problematic > 10:
            print("  [FAIL] >10% of events have zero or NaN L_comp")
        else:
            print("  [PASS] < 10% problematic L_comp values")


if __name__ == "__main__":
    main()
