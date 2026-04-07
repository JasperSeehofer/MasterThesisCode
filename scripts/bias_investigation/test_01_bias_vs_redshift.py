#!/usr/bin/env python3
"""Test 1: Per-event bias vs redshift correlation.

For each detection, compute z_true from d_L and plot log(L(0.66)/L(0.73))
vs z_true to check if the bias is redshift-dependent.

Expected if unbiased: random scatter, no trend.
Expected if biased: systematic correlation (high-z → low h, low-z → high h).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.physical_relations import dist_to_redshift

RESULTS_DIR = PROJECT_ROOT / "results" / "h_sweep_20260401"
POSTERIORS_DIR = RESULTS_DIR / "posteriors"
CRB_FILE = RESULTS_DIR / "cramer_rao_bounds.csv"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRUE = 0.73


def load_posterior(h_val: float) -> dict[int, float]:
    """Load per-detection likelihoods at a given h value."""
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
    # Load CRB data for d_L values
    crb = pd.read_csv(CRB_FILE)
    print(f"Loaded {len(crb)} detections from CRB CSV")

    # Compute z_true for each detection
    z_true_map: dict[int, float] = {}
    dl_map: dict[int, float] = {}
    for idx, row in crb.iterrows():
        d_L = row["luminosity_distance"]
        z = dist_to_redshift(d_L, h=H_TRUE)
        z_true_map[int(idx)] = z
        dl_map[int(idx)] = d_L

    # Load posteriors at key h values
    h_values = [0.60, 0.64, 0.66, 0.68, 0.70, 0.72, 0.73, 0.74, 0.76, 0.80, 0.86]
    posteriors = {h: load_posterior(h) for h in h_values}

    # Compute per-event log-ratios
    det_ids = sorted(set(posteriors[0.66].keys()) & set(posteriors[0.73].keys()) & set(z_true_map.keys()))
    print(f"Common detections across all datasets: {len(det_ids)}")

    z_arr = []
    dl_arr = []
    log_ratio_66_73 = []
    peak_h_arr = []

    for d in det_ids:
        l66 = posteriors[0.66][d]
        l73 = posteriors[0.73][d]
        if l66 <= 0 or l73 <= 0:
            continue
        z_arr.append(z_true_map[d])
        dl_arr.append(dl_map[d])
        log_ratio_66_73.append(np.log(l66) - np.log(l73))

        # Find peak h for this event
        best_h = max(h_values, key=lambda h: posteriors[h].get(d, 0))
        peak_h_arr.append(best_h)

    z_arr = np.array(z_arr)
    dl_arr = np.array(dl_arr)
    log_ratio_66_73 = np.array(log_ratio_66_73)
    peak_h_arr = np.array(peak_h_arr)

    print(f"\nEvents with both L(0.66)>0 and L(0.73)>0: {len(z_arr)}")
    print(f"Redshift range: [{z_arr.min():.4f}, {z_arr.max():.4f}]")
    print(f"d_L range: [{dl_arr.min():.4f}, {dl_arr.max():.4f}] Gpc")

    # Correlation analysis
    corr = np.corrcoef(z_arr, log_ratio_66_73)[0, 1]
    print(f"\nPearson correlation(z_true, log(L(0.66)/L(0.73))): {corr:.4f}")

    # Bin by redshift
    z_bins = np.linspace(z_arr.min(), z_arr.max(), 11)
    print("\nBinned analysis:")
    print(f"{'z_bin':>12s} {'N':>5s} {'mean_logR':>10s} {'std_logR':>10s} {'frac>0':>8s} {'mean_peak_h':>12s}")
    for i in range(len(z_bins) - 1):
        mask = (z_arr >= z_bins[i]) & (z_arr < z_bins[i + 1])
        if mask.sum() == 0:
            continue
        z_mid = 0.5 * (z_bins[i] + z_bins[i + 1])
        mean_lr = log_ratio_66_73[mask].mean()
        std_lr = log_ratio_66_73[mask].std()
        frac_pos = (log_ratio_66_73[mask] > 0).mean()
        mean_peak = peak_h_arr[mask].mean()
        print(f"{z_mid:12.4f} {mask.sum():5d} {mean_lr:10.4f} {std_lr:10.4f} {frac_pos:8.3f} {mean_peak:12.4f}")

    # Plot 1: log-ratio vs redshift
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(z_arr, log_ratio_66_73, c=peak_h_arr, cmap="RdYlBu_r",
                         s=8, alpha=0.5, vmin=0.60, vmax=0.86)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("z_true")
    ax.set_ylabel("log(L(h=0.66) / L(h=0.73))")
    ax.set_title(f"Per-event log-ratio vs redshift (r={corr:.3f})")
    plt.colorbar(scatter, ax=ax, label="peak h")

    # Plot 2: peak h vs redshift
    ax = axes[0, 1]
    ax.scatter(z_arr, peak_h_arr, s=8, alpha=0.3)
    ax.axhline(0.73, color="r", ls="--", lw=1, label="h_true=0.73")
    ax.set_xlabel("z_true")
    ax.set_ylabel("peak h")
    ax.set_title("Per-event peak h vs redshift")
    ax.legend()

    # Plot 3: d_L vs log-ratio
    ax = axes[1, 0]
    ax.scatter(dl_arr, log_ratio_66_73, c=z_arr, cmap="viridis", s=8, alpha=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("d_L [Gpc]")
    ax.set_ylabel("log(L(h=0.66) / L(h=0.73))")
    ax.set_title("Per-event log-ratio vs luminosity distance")

    # Plot 4: histogram of peak h, colored by z bin
    ax = axes[1, 1]
    z_median = np.median(z_arr)
    low_z_mask = z_arr < z_median
    ax.hist(peak_h_arr[low_z_mask], bins=h_values, alpha=0.6, label=f"z < {z_median:.3f}")
    ax.hist(peak_h_arr[~low_z_mask], bins=h_values, alpha=0.6, label=f"z >= {z_median:.3f}")
    ax.axvline(0.73, color="r", ls="--", lw=1)
    ax.set_xlabel("peak h")
    ax.set_ylabel("count")
    ax.set_title("Peak h distribution by redshift half")
    ax.legend()

    plt.tight_layout()
    outpath = OUTPUT_DIR / "test_01_bias_vs_redshift.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
    plt.close()

    # Summary
    print("\n=== SUMMARY ===")
    if abs(corr) > 0.3:
        print(f"STRONG redshift correlation detected (r={corr:.3f})")
        print("Bias is REDSHIFT-DEPENDENT — consistent with galaxy density or P_det effect")
    elif abs(corr) > 0.1:
        print(f"MODERATE redshift correlation (r={corr:.3f})")
        print("Bias has redshift component but may also have uniform component")
    else:
        print(f"WEAK/NO redshift correlation (r={corr:.3f})")
        print("Bias is largely UNIFORM across redshifts — points to formula or normalization error")


if __name__ == "__main__":
    main()
