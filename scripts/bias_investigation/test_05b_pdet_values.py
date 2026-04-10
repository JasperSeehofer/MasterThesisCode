#!/usr/bin/env python3
"""Test 5b: P_det values at detection distances.

Check what P_det returns at the actual detection d_L values.
If P_det ≈ 1.0 for all d_L < 1 Gpc, then P_det provides NO selection
correction for nearby sources, and the galaxy density bias is uncorrectable.

This is the smoking gun test.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from master_thesis_code.constants import SNR_THRESHOLD
from master_thesis_code.physical_relations import dist_to_redshift

RESULTS_DIR = PROJECT_ROOT / "results" / "h_sweep_20260401"
CRB_FILE = RESULTS_DIR / "cramer_rao_bounds.csv"
INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRUE = 0.73


def main() -> None:
    crb = pd.read_csv(CRB_FILE)
    print(f"Loaded {len(crb)} detections, SNR_THRESHOLD={SNR_THRESHOLD}")

    # Build the actual P_det interpolator
    print("\nBuilding SimulationDetectionProbability from injection data...")
    pdet = SimulationDetectionProbability(
        injection_data_dir=str(INJECTION_DIR),
        snr_threshold=SNR_THRESHOLD,
    )
    print(f"P_det h-grid: {pdet._h_grid}")

    # Evaluate P_det at detection distances
    det_dL = crb["luminosity_distance"].values
    det_M = crb["M"].values
    det_phi = crb["phiS"].values
    det_theta = crb["qS"].values
    det_z = np.array([dist_to_redshift(dl, h=H_TRUE) for dl in det_dL])

    # P_det at actual detection parameters for various h values
    h_test_values = [0.60, 0.66, 0.70, 0.73, 0.80, 0.86]

    print("\n=== P_det AT DETECTION d_L VALUES (1D, marginalized over M) ===")
    print(
        f"{'h':>6s} {'mean':>8s} {'min':>8s} {'max':>8s} {'std':>8s} {'frac<0.9':>10s} {'frac<0.5':>10s}"
    )
    for h in h_test_values:
        # d_L at trial h for these sources
        dl_trial = (H_TRUE / h) * det_dL
        p_vals = pdet.detection_probability_without_bh_mass_interpolated(
            dl_trial, det_phi, det_theta, h=h
        )
        p_vals = np.atleast_1d(p_vals)
        print(
            f"{h:6.2f} {np.mean(p_vals):8.4f} {np.min(p_vals):8.4f} {np.max(p_vals):8.4f} "
            f"{np.std(p_vals):8.4f} {(p_vals < 0.9).mean():10.3f} {(p_vals < 0.5).mean():10.3f}"
        )

    # P_det across a wide d_L range at h=0.73
    print("\n=== P_det PROFILE vs d_L AT h=0.73 ===")
    dl_range = np.linspace(0.01, 3.0, 300)
    phi_const = np.full_like(dl_range, 1.0)
    theta_const = np.full_like(dl_range, 0.8)
    p_profile_73 = pdet.detection_probability_without_bh_mass_interpolated(
        dl_range, phi_const, theta_const, h=0.73
    )
    p_profile_73 = np.atleast_1d(p_profile_73)

    # Find where P_det drops below 0.5
    idx_half = np.where(p_profile_73 < 0.5)[0]
    dl_half = dl_range[idx_half[0]] if len(idx_half) > 0 else np.nan
    print(f"P_det drops below 0.5 at d_L = {dl_half:.3f} Gpc")
    print(f"Detection d_L max = {det_dL.max():.3f} Gpc")
    print(f"Ratio: d_L_max / d_L_half = {det_dL.max() / dl_half:.3f}")

    # P_det profile at detection d_L range
    dl_det_range = np.linspace(0.01, det_dL.max() * 1.2, 100)
    p_det_range = pdet.detection_probability_without_bh_mass_interpolated(
        dl_det_range, np.full_like(dl_det_range, 1.0), np.full_like(dl_det_range, 0.8), h=0.73
    )
    p_det_range = np.atleast_1d(p_det_range)

    print(f"\n=== P_det IN DETECTION d_L RANGE [0, {det_dL.max():.3f}] ===")
    print(f"P_det at d_L=0.01: {p_det_range[0]:.4f}")
    print(f"P_det at d_L={det_dL.max():.3f}: {p_det_range[-1]:.4f}")
    print(
        f"P_det range in detection region: [{np.min(p_det_range):.4f}, {np.max(p_det_range):.4f}]"
    )
    print(f"P_det variation (max-min): {np.max(p_det_range) - np.min(p_det_range):.4f}")

    # Critical check: how does P_det RATIO between h=0.66 and h=0.73 vary?
    print("\n=== P_det RATIO: P_det(d_L, h=0.66) / P_det(d_L, h=0.73) ===")
    for i_det in range(min(10, len(det_dL))):
        dl = det_dL[i_det]
        dl_at_66 = (H_TRUE / 0.66) * dl
        dl_at_73 = dl
        p66 = float(
            pdet.detection_probability_without_bh_mass_interpolated(
                dl_at_66, det_phi[i_det], det_theta[i_det], h=0.66
            )
        )
        p73 = float(
            pdet.detection_probability_without_bh_mass_interpolated(
                dl_at_73, det_phi[i_det], det_theta[i_det], h=0.73
            )
        )
        ratio = p66 / p73 if p73 > 0 else np.nan
        print(
            f"  det {i_det}: d_L={dl:.4f}, P_det(h=0.66)={p66:.4f}, P_det(h=0.73)={p73:.4f}, ratio={ratio:.4f}"
        )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: P_det profile vs d_L
    ax = axes[0, 0]
    for h in [0.66, 0.70, 0.73, 0.80]:
        p = pdet.detection_probability_without_bh_mass_interpolated(
            dl_range, phi_const, theta_const, h=h
        )
        ax.plot(dl_range, np.atleast_1d(p), label=f"h={h}")
    ax.axvspan(
        0, det_dL.max(), alpha=0.1, color="green", label=f"detection range (< {det_dL.max():.2f})"
    )
    ax.set_xlabel("d_L [Gpc]")
    ax.set_ylabel("P_det")
    ax.set_title("P_det profile vs luminosity distance")
    ax.legend()

    # Plot 2: P_det in detection range (zoomed)
    ax = axes[0, 1]
    dl_zoom = np.linspace(0.01, det_dL.max() * 1.5, 200)
    for h in [0.66, 0.70, 0.73, 0.80]:
        p = pdet.detection_probability_without_bh_mass_interpolated(
            dl_zoom, np.full_like(dl_zoom, 1.0), np.full_like(dl_zoom, 0.8), h=h
        )
        ax.plot(dl_zoom, np.atleast_1d(p), label=f"h={h}")
    ax.set_xlabel("d_L [Gpc]")
    ax.set_ylabel("P_det")
    ax.set_title("P_det in detection d_L range (zoomed)")
    ax.legend()

    # Plot 3: P_det at each detection location
    ax = axes[1, 0]
    for h in [0.66, 0.73, 0.80]:
        dl_trial = (H_TRUE / h) * det_dL
        p_vals = np.atleast_1d(
            pdet.detection_probability_without_bh_mass_interpolated(
                dl_trial, det_phi, det_theta, h=h
            )
        )
        ax.scatter(det_z, p_vals, s=5, alpha=0.3, label=f"h={h}")
    ax.set_xlabel("z_true")
    ax.set_ylabel("P_det at detection")
    ax.set_title("P_det at each detection vs z_true")
    ax.legend()

    # Plot 4: P_det ratio (h=0.66 / h=0.73) per detection
    ax = axes[1, 1]
    dl_at_66 = (H_TRUE / 0.66) * det_dL
    p66 = np.atleast_1d(
        pdet.detection_probability_without_bh_mass_interpolated(
            dl_at_66, det_phi, det_theta, h=0.66
        )
    )
    p73 = np.atleast_1d(
        pdet.detection_probability_without_bh_mass_interpolated(det_dL, det_phi, det_theta, h=0.73)
    )
    ratio = np.where(p73 > 0, p66 / p73, np.nan)
    ax.scatter(det_z, ratio, s=5, alpha=0.5)
    ax.axhline(1.0, color="r", ls="--", lw=1)
    ax.set_xlabel("z_true")
    ax.set_ylabel("P_det(h=0.66) / P_det(h=0.73)")
    ax.set_title("P_det asymmetry between h=0.66 and h=0.73")

    plt.tight_layout()
    outpath = OUTPUT_DIR / "test_05b_pdet_values.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
    plt.close()

    # Summary
    print("\n=== SUMMARY ===")
    p73_at_dets = np.atleast_1d(
        pdet.detection_probability_without_bh_mass_interpolated(det_dL, det_phi, det_theta, h=0.73)
    )
    if np.mean(p73_at_dets) > 0.95:
        print(f"P_det ≈ {np.mean(p73_at_dets):.3f} at detection distances (nearly constant)")
        print("→ P_det provides NO SELECTION CORRECTION for these nearby sources")
        print("→ Galaxy density bias is UNCORRECTABLE by P_det at these distances")
        print("→ This is a fundamental limitation, not a bug")
    else:
        print(
            f"P_det varies significantly at detection distances (mean={np.mean(p73_at_dets):.3f})"
        )
        print("→ P_det should provide some correction")


if __name__ == "__main__":
    main()
