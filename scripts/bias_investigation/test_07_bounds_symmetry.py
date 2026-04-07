#!/usr/bin/env python3
"""Test 7: Integration bounds symmetrization.

The numerator uses h-dependent z-bounds (from d_L ± 4σ via dist_to_redshift(h)),
while the denominator uses h-independent bounds (galaxy z ± 4σ_z).

At h=0.66, the numerator z-window shifts DOWNWARD relative to h=0.73.
This means:
  - Different z-ranges where P_det and GW likelihood are evaluated
  - Potential mismatch between where the numerator "sees" and where
    the denominator normalizes

This test checks whether using SYMMETRIC bounds changes the bias.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import fixed_quad
from scipy.stats import multivariate_normal, norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.physical_relations import dist, dist_to_redshift, dist_vectorized

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRUE = 0.73
N_QUAD = 50


def make_detection(z_true: float, frac_err: float = 0.05) -> dict:
    d_L = dist(z_true, h=H_TRUE)
    sigma_dL = frac_err * d_L
    return {
        "d_L": d_L, "sigma_dL": sigma_dL,
        "phi": 1.0, "theta": 0.8,
        "sigma_phi": 0.05, "sigma_theta": 0.05,
    }


def build_gaussian_3d(det: dict) -> multivariate_normal:
    cov = np.diag([det["sigma_phi"]**2, det["sigma_theta"]**2, (det["sigma_dL"] / det["d_L"])**2])
    return multivariate_normal(mean=[det["phi"], det["theta"], 1.0], cov=cov, allow_singular=True)


def compute_likelihood(
    det: dict,
    gal_z: float,
    gal_z_err: float,
    gaussian_3d: multivariate_normal,
    h: float,
    bounds_mode: str = "asymmetric",  # "asymmetric" (production), "numerator", "denominator", "union"
) -> tuple[float, float]:
    """Compute (numerator, denominator) with configurable integration bounds."""
    sigma_mult = 4.0

    # Numerator z-bounds (h-dependent)
    num_z_upper = dist_to_redshift(det["d_L"] + sigma_mult * det["sigma_dL"], h=h)
    num_z_lower = dist_to_redshift(max(0.001, det["d_L"] - sigma_mult * det["sigma_dL"]), h=h)

    # Denominator z-bounds (h-independent)
    den_z_upper = gal_z + sigma_mult * gal_z_err
    den_z_lower = max(0.001, gal_z - sigma_mult * gal_z_err)

    # Choose bounds based on mode
    if bounds_mode == "asymmetric":
        # Production behavior: different bounds for num and den
        pass
    elif bounds_mode == "numerator":
        # Both use numerator's (h-dependent) bounds
        den_z_lower = num_z_lower
        den_z_upper = num_z_upper
    elif bounds_mode == "denominator":
        # Both use denominator's (galaxy) bounds
        num_z_lower = den_z_lower
        num_z_upper = den_z_upper
    elif bounds_mode == "union":
        # Both use the wider of the two
        wide_lower = min(num_z_lower, den_z_lower)
        wide_upper = max(num_z_upper, den_z_upper)
        num_z_lower = den_z_lower = wide_lower
        num_z_upper = den_z_upper = wide_upper

    gal_z_dist = norm(loc=gal_z, scale=gal_z_err)

    def num_integrand(z):
        d_L_trial = dist_vectorized(z, h=h)
        d_L_frac = d_L_trial / det["d_L"]
        phi = np.full_like(z, det["phi"])
        theta = np.full_like(z, det["theta"])
        gw = gaussian_3d.pdf(np.vstack([phi, theta, d_L_frac]).T)
        return gw * gal_z_dist.pdf(z)  # P_det = 1 for isolation

    def den_integrand(z):
        return gal_z_dist.pdf(z)

    num, _ = fixed_quad(num_integrand, num_z_lower, num_z_upper, n=N_QUAD)
    den, _ = fixed_quad(den_integrand, den_z_lower, den_z_upper, n=N_QUAD)
    return float(num), float(den)


def main() -> None:
    z_true = 0.12
    det = make_detection(z_true)
    gaussian_3d = build_gaussian_3d(det)
    h_values = np.arange(0.60, 0.87, 0.01)

    # First, show how bounds differ
    print("=== INTEGRATION BOUNDS AT KEY h VALUES ===")
    print(f"Detection: z_true={z_true}, d_L={det['d_L']:.4f}, σ_dL={det['sigma_dL']:.4f}")
    gal_z = z_true
    gal_z_err = 0.002
    print(f"Galaxy: z={gal_z}, σ_z={gal_z_err}")
    print()
    print(f"{'h':>6s} {'num_z_lo':>10s} {'num_z_hi':>10s} {'den_z_lo':>10s} {'den_z_hi':>10s} {'overlap':>10s}")
    for h in [0.60, 0.66, 0.70, 0.73, 0.80, 0.86]:
        num_upper = dist_to_redshift(det["d_L"] + 4 * det["sigma_dL"], h=h)
        num_lower = dist_to_redshift(max(0.001, det["d_L"] - 4 * det["sigma_dL"]), h=h)
        den_upper = gal_z + 4 * gal_z_err
        den_lower = max(0.001, gal_z - 4 * gal_z_err)
        overlap = max(0, min(num_upper, den_upper) - max(num_lower, den_lower))
        total = (num_upper - num_lower) + (den_upper - den_lower)
        print(f"{h:6.2f} {num_lower:10.4f} {num_upper:10.4f} {den_lower:10.4f} {den_upper:10.4f} {overlap:10.4f}")

    # Run with different bounds modes (true host only first)
    modes = ["asymmetric", "numerator", "denominator", "union"]
    results_single = {}

    print("\n=== SINGLE GALAXY (TRUE HOST) — BOUNDS COMPARISON ===")
    for mode in modes:
        likes = []
        for h in h_values:
            num, den = compute_likelihood(det, gal_z, gal_z_err, gaussian_3d, h, bounds_mode=mode)
            likes.append(num / den if den > 0 else 0)
        likes = np.array(likes)
        if likes.max() > 0:
            likes /= likes.max()
        peak = h_values[np.argmax(likes)]
        results_single[mode] = (likes, peak)
        print(f"  {mode:>15s}: peak h = {peak:.3f} (offset = {peak - H_TRUE:+.3f})")

    # Run with 100 uniform galaxies + true host
    print("\n=== 100 UNIFORM GALAXIES + TRUE HOST — BOUNDS COMPARISON ===")
    galaxy_positions = np.concatenate([[z_true], np.linspace(0.02, 0.25, 100)])
    results_multi = {}

    for mode in modes:
        likes = []
        for h in h_values:
            total_num = 0.0
            total_den = 0.0
            for gz in galaxy_positions:
                num, den = compute_likelihood(det, gz, 0.002, gaussian_3d, h, bounds_mode=mode)
                total_num += num
                total_den += den
            likes.append(total_num / total_den if total_den > 0 else 0)
        likes = np.array(likes)
        if likes.max() > 0:
            likes /= likes.max()
        peak = h_values[np.argmax(likes)]
        results_multi[mode] = (likes, peak)
        print(f"  {mode:>15s}: peak h = {peak:.3f} (offset = {peak - H_TRUE:+.3f})")

    # Run with low-z biased galaxies (mimicking GLADE)
    print("\n=== 100 LOW-Z BIASED GALAXIES — BOUNDS COMPARISON ===")
    galaxy_positions_lowz = np.concatenate([
        [z_true],
        np.linspace(0.02, z_true, 70),
        np.linspace(z_true, 0.25, 30),
    ])
    results_lowz = {}

    for mode in modes:
        likes = []
        for h in h_values:
            total_num = 0.0
            total_den = 0.0
            for gz in galaxy_positions_lowz:
                num, den = compute_likelihood(det, gz, 0.002, gaussian_3d, h, bounds_mode=mode)
                total_num += num
                total_den += den
            likes.append(total_num / total_den if total_den > 0 else 0)
        likes = np.array(likes)
        if likes.max() > 0:
            likes /= likes.max()
        peak = h_values[np.argmax(likes)]
        results_lowz[mode] = (likes, peak)
        print(f"  {mode:>15s}: peak h = {peak:.3f} (offset = {peak - H_TRUE:+.3f})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, results, title in [
        (axes[0], results_single, "Single galaxy (true host)"),
        (axes[1], results_multi, "101 galaxies (uniform)"),
        (axes[2], results_lowz, "101 galaxies (low-z biased)"),
    ]:
        for mode in modes:
            likes, peak = results[mode]
            ax.plot(h_values, likes, label=f"{mode} (peak={peak:.3f})", lw=1.5)
        ax.axvline(H_TRUE, color="r", ls="--", lw=1)
        ax.set_xlabel("h")
        ax.set_ylabel("L(h) / L_max")
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.suptitle("Test 7: Integration bounds symmetrization (P_det=1)", fontsize=12)
    plt.tight_layout()
    outpath = OUTPUT_DIR / "test_07_bounds_symmetry.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
    plt.close()

    # Summary
    print("\n=== SUMMARY ===")
    asym_peak = results_lowz["asymmetric"][1]
    union_peak = results_lowz["union"][1]
    den_peak = results_lowz["denominator"][1]
    print(f"Low-z biased galaxies:")
    print(f"  Asymmetric (production): peak = {asym_peak:.3f}")
    print(f"  Union bounds:            peak = {union_peak:.3f}")
    print(f"  Denominator bounds:      peak = {den_peak:.3f}")
    if abs(asym_peak - union_peak) > 0.02:
        print(f"  Bounds difference = {abs(asym_peak - union_peak):.3f} → SIGNIFICANT")
    else:
        print(f"  Bounds difference = {abs(asym_peak - union_peak):.3f} → NEGLIGIBLE")


if __name__ == "__main__":
    main()
