#!/usr/bin/env python3
"""Test 5: P_det grid boundary mapping.

For each h value, compute d_L(z, h) for the actual detection redshifts and
check how many fall outside the P_det grid's dl_max. Also compute the critical
redshift z_crit(h) beyond which P_det returns 0.

Requires: simulations/injections/ directory with injection campaign CSVs.

Expected: at h < 0.73, more events exceed dl_max → asymmetric P_det suppression.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.physical_relations import dist_to_redshift

RESULTS_DIR = PROJECT_ROOT / "results" / "h_sweep_20260401"
CRB_FILE = RESULTS_DIR / "cramer_rao_bounds.csv"
INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRUE = 0.73


def get_dl_max_from_injections() -> dict[float, float]:
    """Read injection CSVs and compute dl_max per h-value grid."""
    if not INJECTION_DIR.exists():
        print(f"ERROR: {INJECTION_DIR} does not exist. Fetch from cluster first.")
        print("Falling back to estimated dl_max from detection data.")
        return {}

    dl_max_per_h = {}
    for csv_file in sorted(INJECTION_DIR.glob("*.csv")):
        # Extract h value from filename (format varies)
        try:
            df = pd.read_csv(csv_file)
            if "luminosity_distance" in df.columns:
                dl_max = float(df["luminosity_distance"].max()) * 1.1
                # Try to extract h from filename
                name = csv_file.stem
                # Common patterns: h_0_73, injections_h0.73, etc.
                for part in name.split("_"):
                    try:
                        h_val = float(part.replace("h", ""))
                        dl_max_per_h[h_val] = dl_max
                        break
                    except ValueError:
                        continue
                if name not in str(dl_max_per_h):
                    print(f"  Read {csv_file.name}: {len(df)} injections, dl_max={dl_max:.4f}")
        except Exception as e:
            print(f"  Warning: could not read {csv_file.name}: {e}")

    return dl_max_per_h


def estimate_dl_max_from_detections(crb: pd.DataFrame) -> float:
    """Estimate dl_max from the detection data if injections aren't available."""
    # The P_det grid's dl_max = max(injection_dl) * 1.1
    # Detections are a subset of injections (those with SNR > threshold)
    # So max(detection_dl) < max(injection_dl)
    # Conservative estimate: dl_max ~ max(detection_dl) * 1.3
    return float(crb["luminosity_distance"].max()) * 1.3


def main() -> None:
    crb = pd.read_csv(CRB_FILE)
    print(f"Loaded {len(crb)} detections")

    # Get detection z_true and d_L
    det_dL = crb["luminosity_distance"].values
    det_z = np.array([dist_to_redshift(dl, h=H_TRUE) for dl in det_dL])
    print(f"Detection d_L range: [{det_dL.min():.4f}, {det_dL.max():.4f}] Gpc")
    print(f"Detection z range: [{det_z.min():.4f}, {det_z.max():.4f}]")

    # Get dl_max
    dl_max_per_h = get_dl_max_from_injections()

    if not dl_max_per_h:
        # Estimate from detection data
        dl_max_est = estimate_dl_max_from_detections(crb)
        print(f"\nEstimated dl_max (from detections): {dl_max_est:.4f} Gpc")
        print("Using single dl_max for all h (since real grid may vary by h)")
        # Also try the d_L scaling approach
        # At h=0.73, the grid covers up to dl_max_est
        # At trial h, d_L(z, h) = (0.73/h) * d_L(z, 0.73)
        # So effective dl_max at h = dl_max_est * (h/0.73) from the detection's perspective
    else:
        print(f"\nP_det grid dl_max per h: {dl_max_per_h}")
        dl_max_est = dl_max_per_h.get(0.73, max(dl_max_per_h.values()))

    h_values = np.arange(0.60, 0.87, 0.01)

    # For each h, compute how many detections would have d_L(z_true, h) > dl_max
    # d_L(z, h) = (0.73/h) * d_L(z, 0.73) = (0.73/h) * det_dL
    print(f"\n=== d_L EXCEEDANCE ANALYSIS (dl_max_est = {dl_max_est:.4f} Gpc) ===")
    print(f"{'h':>6s} {'dl_max_eff':>12s} {'N_exceed':>10s} {'frac_exceed':>12s} {'z_crit':>8s}")

    n_exceed_arr = []
    z_crit_arr = []
    dl_max_eff_arr = []

    for h in h_values:
        # d_L at the trial h for each detection's true redshift
        dl_at_h = (H_TRUE / h) * det_dL

        # If P_det grid is built at h=0.73, and we query at h:
        # The P_det grid has dl_max fixed. We're querying with d_L(z, h).
        # But the grid is ALSO built per-h, so there might be separate grids.
        # For now, use fixed dl_max (worst-case single-grid assumption)
        n_exceed = np.sum(dl_at_h > dl_max_est)
        frac_exceed = n_exceed / len(det_dL)

        # Critical redshift: where does d_L(z, h) = dl_max?
        # d_L(z, h) = (0.73/h) * d_L(z, 0.73) = dl_max
        # d_L(z, 0.73) = dl_max * (h/0.73)
        try:
            z_crit = dist_to_redshift(dl_max_est, h=h)
        except Exception:
            z_crit = np.nan

        n_exceed_arr.append(n_exceed)
        z_crit_arr.append(z_crit)
        dl_max_eff_arr.append(dl_max_est)

        print(f"{h:6.2f} {dl_max_est:12.4f} {n_exceed:10d} {frac_exceed:12.3f} {z_crit:8.4f}")

    # Also check: for each detection, at what h does d_L first exceed dl_max?
    print("\n=== PER-DETECTION: h WHERE d_L(z_true, h) FIRST EXCEEDS dl_max ===")
    exceed_h = []
    for i in range(len(det_dL)):
        # d_L(z_true, h) > dl_max when (0.73/h) * det_dL[i] > dl_max
        # h < 0.73 * det_dL[i] / dl_max
        h_thresh = H_TRUE * det_dL[i] / dl_max_est
        if h_thresh < 0.86:  # within our h range
            exceed_h.append(h_thresh)

    if exceed_h:
        exceed_h = np.array(exceed_h)
        print(f"  {len(exceed_h)}/{len(det_dL)} detections exceed dl_max at some h < 0.86")
        print(
            f"  Threshold h distribution: min={exceed_h.min():.3f}, median={np.median(exceed_h):.3f}, max={exceed_h.max():.3f}"
        )

        # Histogram of threshold h values
        h_bins = np.arange(0.60, 0.87, 0.02)
        counts, _ = np.histogram(exceed_h, bins=h_bins)
        print("  Events hitting dl_max boundary per h-bin:")
        for i in range(len(h_bins) - 1):
            if counts[i] > 0:
                print(f"    h=[{h_bins[i]:.2f}, {h_bins[i + 1]:.2f}): {counts[i]} events")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(h_values, n_exceed_arr, "o-", ms=4)
    ax.axvline(H_TRUE, color="r", ls="--", lw=1, label="h_true")
    ax.set_xlabel("h")
    ax.set_ylabel("N events exceeding dl_max")
    ax.set_title(f"Events beyond P_det grid (dl_max={dl_max_est:.3f})")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(h_values, z_crit_arr, "o-", ms=4)
    ax.axhline(
        np.median(det_z), color="orange", ls=":", label=f"median z_det={np.median(det_z):.3f}"
    )
    ax.axvline(H_TRUE, color="r", ls="--", lw=1, label="h_true")
    ax.set_xlabel("h")
    ax.set_ylabel("z_crit (max detectable z)")
    ax.set_title("Critical redshift vs h")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(det_z, bins=30, alpha=0.5, label="Detection z_true")
    if len(exceed_h) > 0:
        # For each z, find the h where it hits boundary
        z_for_exceed = det_z[det_dL > dl_max_est * (0.73 / 0.73)]  # always beyond at some h
        ax.axvline(np.median(det_z), color="orange", ls=":", label="median z")
    ax.set_xlabel("z_true")
    ax.set_ylabel("count")
    ax.set_title("Detection redshift distribution")
    ax.legend()

    ax = axes[1, 1]
    if len(exceed_h) > 0:
        ax.hist(exceed_h, bins=20, alpha=0.7, color="coral")
        ax.axvline(H_TRUE, color="r", ls="--", lw=1, label="h_true")
        ax.set_xlabel("h threshold (where d_L exceeds dl_max)")
        ax.set_ylabel("count")
        ax.set_title("Per-event boundary crossing h")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No events exceed dl_max\nin [0.60, 0.86] range",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    plt.tight_layout()
    outpath = OUTPUT_DIR / "test_05_pdet_boundary.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
    plt.close()

    # Summary
    print("\n=== SUMMARY ===")
    n_at_66 = n_exceed_arr[h_values.tolist().index(0.66)] if 0.66 in h_values else 0
    n_at_73 = n_exceed_arr[h_values.tolist().index(0.73)] if 0.73 in h_values else 0
    print(f"Events beyond dl_max at h=0.66: {n_at_66}")
    print(f"Events beyond dl_max at h=0.73: {n_at_73}")
    if n_at_66 > n_at_73 + 10:
        print("SIGNIFICANT asymmetry: more events clipped at h=0.66 → P_det boundary IS a factor")
    else:
        print("Similar exceedance at h=0.66 and h=0.73 → P_det boundary is NOT the primary issue")


if __name__ == "__main__":
    main()
