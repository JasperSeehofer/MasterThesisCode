#!/usr/bin/env python3
"""Test 2: Sharp drop analysis at h=0.70 → 0.72.

The sum of log-likelihoods drops by 92 units between h=0.70 and h=0.72.
This test identifies WHICH events cause the drop and checks whether their
d_L values cluster near a specific threshold (suggesting P_det grid boundary).

Expected if P_det boundary: events cluster near a d_L value.
Expected if smooth effect: events spread across d_L range.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.physical_relations import dist, dist_to_redshift

RESULTS_DIR = PROJECT_ROOT / "results" / "h_sweep_20260401"
POSTERIORS_DIR = RESULTS_DIR / "posteriors"
CRB_FILE = RESULTS_DIR / "cramer_rao_bounds.csv"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRUE = 0.73


def load_posterior(h_val: float) -> dict[int, float]:
    h_str = f"h_{h_val}".replace(".", "_")
    fpath = POSTERIORS_DIR / f"{h_str}.json"
    with open(fpath) as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        if k == "h":
            continue
        val = v[0] if isinstance(v, list) and len(v) > 0 else (0.0 if isinstance(v, list) else float(v))
        result[int(k)] = val
    return result


def main() -> None:
    crb = pd.read_csv(CRB_FILE)
    print(f"Loaded {len(crb)} detections")

    # Load posteriors at all h values
    h_values = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.73, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86]
    posteriors = {h: load_posterior(h) for h in h_values}

    # Compute z_true and d_L for each detection
    det_info: dict[int, dict] = {}
    for idx, row in crb.iterrows():
        d_L = row["luminosity_distance"]
        d_L_err = np.sqrt(row["delta_luminosity_distance_delta_luminosity_distance"])
        z = dist_to_redshift(d_L, h=H_TRUE)
        det_info[int(idx)] = {"d_L": d_L, "z_true": z, "sigma_dL": d_L_err}

    # ================================================================
    # Analysis 1: Sharp drop between consecutive h-bins
    # ================================================================
    print("\n=== LOG-LIKELIHOOD DROPS BETWEEN CONSECUTIVE H-BINS ===")
    for i in range(len(h_values) - 1):
        h_lo, h_hi = h_values[i], h_values[i + 1]
        common = sorted(set(posteriors[h_lo].keys()) & set(posteriors[h_hi].keys()))
        deltas = []
        for d in common:
            l_lo = posteriors[h_lo][d]
            l_hi = posteriors[h_hi][d]
            if l_lo > 0 and l_hi > 0:
                deltas.append(np.log(l_hi) - np.log(l_lo))
        if deltas:
            total = sum(deltas)
            print(f"  h={h_lo:.2f}→{h_hi:.2f}: Δ sum log L = {total:+8.1f}  (N={len(deltas)}, mean={np.mean(deltas):+.4f})")

    # ================================================================
    # Analysis 2: Events causing the h=0.70→0.72 drop
    # ================================================================
    print("\n=== TOP 50 EVENTS CAUSING h=0.70→0.72 DROP ===")
    common_70_72 = sorted(set(posteriors[0.70].keys()) & set(posteriors[0.72].keys()) & set(det_info.keys()))

    event_deltas = []
    for d in common_70_72:
        l70 = posteriors[0.70][d]
        l72 = posteriors[0.72][d]
        if l70 > 0 and l72 > 0:
            delta = np.log(l72) - np.log(l70)
            event_deltas.append((d, delta, det_info[d]["d_L"], det_info[d]["z_true"], det_info[d]["sigma_dL"]))

    event_deltas.sort(key=lambda x: x[1])  # most negative first

    print(f"{'rank':>5s} {'det':>5s} {'Δlog L':>10s} {'d_L':>8s} {'z_true':>8s} {'σ_dL':>8s} {'d_L(z,0.70)':>12s} {'d_L(z,0.72)':>12s}")
    for rank, (det_id, delta, d_L, z, sigma) in enumerate(event_deltas[:50]):
        # d_L at the trial h values for this event's true redshift
        dl_at_70 = dist(z, h=0.70)
        dl_at_72 = dist(z, h=0.72)
        print(f"{rank + 1:5d} {det_id:5d} {delta:+10.4f} {d_L:8.4f} {z:8.4f} {sigma:8.4f} {dl_at_70:12.4f} {dl_at_72:12.4f}")

    # ================================================================
    # Analysis 3: d_L clustering of drop events
    # ================================================================
    all_dl = np.array([det_info[d]["d_L"] for d in common_70_72 if posteriors[0.70][d] > 0 and posteriors[0.72][d] > 0])
    all_z = np.array([det_info[d]["z_true"] for d in common_70_72 if posteriors[0.70][d] > 0 and posteriors[0.72][d] > 0])
    all_deltas = np.array([d[1] for d in event_deltas])

    # Check for d_L clustering in top-50 worst events
    top50_dl = np.array([d[2] for d in event_deltas[:50]])
    top50_z = np.array([d[3] for d in event_deltas[:50]])

    print(f"\n=== d_L STATISTICS ===")
    print(f"All events: d_L range [{all_dl.min():.4f}, {all_dl.max():.4f}], median={np.median(all_dl):.4f}")
    print(f"Top-50 drop: d_L range [{top50_dl.min():.4f}, {top50_dl.max():.4f}], median={np.median(top50_dl):.4f}")
    print(f"All events: z range [{all_z.min():.4f}, {all_z.max():.4f}], median={np.median(all_z):.4f}")
    print(f"Top-50 drop: z range [{top50_z.min():.4f}, {top50_z.max():.4f}], median={np.median(top50_z):.4f}")

    # ================================================================
    # Analysis 4: Check ALL h-bin transitions for each event
    # ================================================================
    print("\n=== TRANSITION EVENTS: WHERE DOES LIKELIHOOD FIRST BECOME ZERO? ===")
    transition_h = {}  # det_id -> first h where L=0
    for d in sorted(det_info.keys()):
        for h in h_values:
            if posteriors[h].get(d, 0) == 0:
                transition_h[d] = h
                break

    if transition_h:
        print(f"Events with zero likelihood at some h: {len(transition_h)}")
        h_counts = {}
        for h in transition_h.values():
            h_counts[h] = h_counts.get(h, 0) + 1
        for h in sorted(h_counts.keys()):
            print(f"  First zero at h={h:.2f}: {h_counts[h]} events")
    else:
        print("No events have zero likelihood at any h (all nonzero)")

    # ================================================================
    # Plot
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Δlog L vs d_L
    ax = axes[0, 0]
    ax.scatter(all_dl, all_deltas, s=8, alpha=0.4, c="steelblue")
    ax.scatter(top50_dl, all_deltas[:50], s=20, alpha=0.8, c="red", label="Top 50 drop")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("d_L [Gpc]")
    ax.set_ylabel("Δlog L (h=0.72 - h=0.70)")
    ax.set_title("Sharp drop events: Δlog L vs d_L")
    ax.legend()

    # Plot 2: Δlog L vs z_true
    ax = axes[0, 1]
    ax.scatter(all_z, all_deltas, s=8, alpha=0.4, c="steelblue")
    ax.scatter(top50_z, all_deltas[:50], s=20, alpha=0.8, c="red", label="Top 50 drop")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("z_true")
    ax.set_ylabel("Δlog L (h=0.72 - h=0.70)")
    ax.set_title("Sharp drop events: Δlog L vs z_true")
    ax.legend()

    # Plot 3: d_L histogram comparing top-50 vs all
    ax = axes[1, 0]
    ax.hist(all_dl, bins=30, alpha=0.5, density=True, label="All events")
    ax.hist(top50_dl, bins=15, alpha=0.5, density=True, label="Top 50 drop")
    ax.set_xlabel("d_L [Gpc]")
    ax.set_ylabel("density")
    ax.set_title("d_L distribution: all vs top-50 drop events")
    ax.legend()

    # Plot 4: Full log-likelihood profile summed
    ax = axes[1, 1]
    sums = {}
    common_all = sorted(set.intersection(*[set(posteriors[h].keys()) for h in h_values]))
    for h in h_values:
        s = sum(np.log(posteriors[h][d]) for d in common_all if posteriors[h][d] > 0)
        sums[h] = s
    ax.plot(list(sums.keys()), list(sums.values()), "o-", color="steelblue")
    ax.axvline(0.73, color="r", ls="--", lw=1, label="h_true")
    ax.set_xlabel("h")
    ax.set_ylabel("sum log L")
    ax.set_title("Combined log-likelihood profile")
    ax.legend()

    plt.tight_layout()
    outpath = OUTPUT_DIR / "test_02_sharp_drop.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
    plt.close()

    # Summary
    print("\n=== SUMMARY ===")
    median_delta = np.median(all_deltas)
    print(f"Median per-event Δlog L (0.72 vs 0.70): {median_delta:+.4f}")
    print(f"Total Δlog L: {np.sum(all_deltas):+.1f}")
    if np.median(top50_dl) > np.median(all_dl) * 1.1:
        print("Top-50 drop events are at HIGHER d_L — consistent with P_det boundary hypothesis")
    elif np.median(top50_dl) < np.median(all_dl) * 0.9:
        print("Top-50 drop events are at LOWER d_L — inconsistent with P_det boundary hypothesis")
    else:
        print("Top-50 drop events have similar d_L distribution — suggests uniform effect, not boundary")


if __name__ == "__main__":
    main()
