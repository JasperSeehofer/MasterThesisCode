#!/usr/bin/env python3
"""Test 4: Single detection + 100 galaxies + P_det=1.

Tests whether adding many non-host galaxies at uniform redshifts biases
the combined likelihood away from h=0.73 when P_det=1.

Expected if correct: peak near h=0.73 (small shift acceptable).
Expected if galaxy density dominates: peak shifts to ~0.68.
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

from master_thesis_code.datamodels.detection import Detection
from master_thesis_code.galaxy_catalogue.handler import HostGalaxy
from master_thesis_code.physical_relations import dist, dist_to_redshift, dist_vectorized

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRUE = 0.73
SIGMA_MULT = 4.0
N_QUAD = 50


def make_synthetic_detection(z_true: float, frac_dl_err: float = 0.05) -> Detection:
    d_L = dist(z_true, h=H_TRUE)
    sigma_dL = frac_dl_err * d_L
    params = pd.Series({
        "luminosity_distance": d_L,
        "delta_luminosity_distance_delta_luminosity_distance": sigma_dL**2,
        "phiS": 1.0,
        "delta_phiS_delta_phiS": 0.05**2,
        "qS": 0.8,
        "delta_qS_delta_qS": 0.05**2,
        "M": 5e5,
        "delta_M_delta_M": 50.0**2,
        "delta_phiS_delta_qS": 0.0,
        "delta_phiS_delta_M": 0.0,
        "delta_qS_delta_M": 0.0,
        "delta_luminosity_distance_delta_M": 0.0,
        "delta_qS_delta_luminosity_distance": 0.0,
        "delta_phiS_delta_luminosity_distance": 0.0,
        "SNR": 25.0,
        "host_galaxy_index": 0,
    })
    return Detection(params)


def build_gaussian_3d(det: Detection) -> multivariate_normal:
    cov = [
        [det.phi_error**2, det.theta_phi_covariance, det.d_L_phi_covariance / det.d_L],
        [det.theta_phi_covariance, det.theta_error**2, det.d_L_theta_covariance / det.d_L],
        [det.d_L_phi_covariance / det.d_L, det.d_L_theta_covariance / det.d_L,
         det.d_L_uncertainty**2 / det.d_L**2],
    ]
    return multivariate_normal(mean=[det.phi, det.theta, 1.0], cov=cov, allow_singular=True)


def compute_likelihood_for_galaxy(
    det: Detection,
    gal: HostGalaxy,
    gaussian_3d: multivariate_normal,
    h: float,
) -> tuple[float, float]:
    """Compute (numerator, denominator) for one galaxy at given h."""
    num_z_upper = dist_to_redshift(det.d_L + SIGMA_MULT * det.d_L_uncertainty, h=h)
    num_z_lower = dist_to_redshift(max(0.001, det.d_L - SIGMA_MULT * det.d_L_uncertainty), h=h)
    den_z_upper = gal.z + SIGMA_MULT * gal.z_error
    den_z_lower = max(0.001, gal.z - SIGMA_MULT * gal.z_error)

    gal_z_dist = norm(loc=gal.z, scale=gal.z_error)

    def num_integrand(z):
        d_L_trial = dist_vectorized(z, h=h)
        d_L_frac = d_L_trial / det.d_L
        phi = np.full_like(z, gal.phiS)
        theta = np.full_like(z, gal.qS)
        gw = gaussian_3d.pdf(np.vstack([phi, theta, d_L_frac]).T)
        return gw * gal_z_dist.pdf(z)

    def den_integrand(z):
        return gal_z_dist.pdf(z)

    num, _ = fixed_quad(num_integrand, num_z_lower, num_z_upper, n=N_QUAD)
    den, _ = fixed_quad(den_integrand, den_z_lower, den_z_upper, n=N_QUAD)
    return float(num), float(den)


def run_scenario(
    det: Detection,
    galaxies: list[HostGalaxy],
    gaussian_3d: multivariate_normal,
    h_values: np.ndarray,
    label: str,
) -> tuple[np.ndarray, float]:
    """Compute combined likelihood L(h) = sum(num) / sum(den) across all galaxies."""
    likelihoods = []

    for h in h_values:
        total_num = 0.0
        total_den = 0.0
        for gal in galaxies:
            num, den = compute_likelihood_for_galaxy(det, gal, gaussian_3d, h)
            total_num += num
            total_den += den
        L_h = total_num / total_den if total_den > 0 else 0
        likelihoods.append(L_h)

    likelihoods = np.array(likelihoods)
    if likelihoods.max() > 0:
        likelihoods /= likelihoods.max()

    peak_h = h_values[np.argmax(likelihoods)]
    print(f"  {label}: peak h = {peak_h:.3f} (offset = {peak_h - H_TRUE:+.3f})")
    return likelihoods, peak_h


def main() -> None:
    z_true = 0.12  # representative of the dominant z=0.08-0.13 population
    h_values = np.arange(0.60, 0.87, 0.01)
    det = make_synthetic_detection(z_true)
    gaussian_3d = build_gaussian_3d(det)

    print(f"Detection: z_true={z_true}, d_L={det.d_L:.4f} Gpc, σ_dL={det.d_L_uncertainty:.4f} Gpc")
    print(f"  σ_dL/d_L = {det.d_L_uncertainty/det.d_L:.4f}")
    print()

    # Scenario A: True host only
    true_host = HostGalaxy.from_attributes(
        phiS=det.phi, qS=det.theta, z=z_true, z_error=0.002, M=5e5, M_error=1e4
    )
    L_a, peak_a = run_scenario(det, [true_host], gaussian_3d, h_values, "A: True host only")

    # Scenario B: True host + 50 uniform galaxies
    np.random.seed(42)
    n_bg = 50
    bg_galaxies = []
    for z_bg in np.linspace(0.02, 0.25, n_bg):
        bg_galaxies.append(HostGalaxy.from_attributes(
            phiS=det.phi, qS=det.theta, z=z_bg, z_error=0.002, M=5e5, M_error=1e4
        ))
    L_b, peak_b = run_scenario(det, [true_host] + bg_galaxies, gaussian_3d, h_values,
                                "B: True host + 50 uniform")

    # Scenario C: True host + 100 uniform galaxies
    bg_galaxies_100 = []
    for z_bg in np.linspace(0.02, 0.25, 100):
        bg_galaxies_100.append(HostGalaxy.from_attributes(
            phiS=det.phi, qS=det.theta, z=z_bg, z_error=0.002, M=5e5, M_error=1e4
        ))
    L_c, peak_c = run_scenario(det, [true_host] + bg_galaxies_100, gaussian_3d, h_values,
                                "C: True host + 100 uniform")

    # Scenario D: True host + 100 galaxies biased toward LOW z (more at z<z_true)
    bg_low_z = []
    z_vals_low = np.concatenate([np.linspace(0.02, z_true, 70), np.linspace(z_true, 0.25, 30)])
    for z_bg in z_vals_low:
        bg_low_z.append(HostGalaxy.from_attributes(
            phiS=det.phi, qS=det.theta, z=z_bg, z_error=0.002, M=5e5, M_error=1e4
        ))
    L_d, peak_d = run_scenario(det, [true_host] + bg_low_z, gaussian_3d, h_values,
                                "D: True host + 100 low-z biased")

    # Scenario E: True host + 100 galaxies biased toward HIGH z
    bg_high_z = []
    z_vals_high = np.concatenate([np.linspace(0.02, z_true, 30), np.linspace(z_true, 0.25, 70)])
    for z_bg in z_vals_high:
        bg_high_z.append(HostGalaxy.from_attributes(
            phiS=det.phi, qS=det.theta, z=z_bg, z_error=0.002, M=5e5, M_error=1e4
        ))
    L_e, peak_e = run_scenario(det, [true_host] + bg_high_z, gaussian_3d, h_values,
                                "E: True host + 100 high-z biased")

    # Scenario F: No true host — only 100 uniform galaxies (worst case)
    L_f, peak_f = run_scenario(det, bg_galaxies_100, gaussian_3d, h_values,
                                "F: 100 uniform (NO true host)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(h_values, L_a, "o-", ms=3, label=f"A: True host only (peak={peak_a:.3f})")
    ax.plot(h_values, L_b, "s-", ms=3, label=f"B: +50 uniform (peak={peak_b:.3f})")
    ax.plot(h_values, L_c, "^-", ms=3, label=f"C: +100 uniform (peak={peak_c:.3f})")
    ax.plot(h_values, L_f, "x-", ms=3, label=f"F: 100 uniform, no host (peak={peak_f:.3f})")
    ax.axvline(H_TRUE, color="r", ls="--", lw=1, alpha=0.7, label="h_true=0.73")
    ax.set_xlabel("h")
    ax.set_ylabel("L(h) / L_max")
    ax.set_title(f"Galaxy count effect (z_true={z_true})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(h_values, L_a, "o-", ms=3, label=f"A: True host (peak={peak_a:.3f})")
    ax.plot(h_values, L_d, "v-", ms=3, label=f"D: Low-z biased (peak={peak_d:.3f})")
    ax.plot(h_values, L_e, "^-", ms=3, label=f"E: High-z biased (peak={peak_e:.3f})")
    ax.axvline(H_TRUE, color="r", ls="--", lw=1, alpha=0.7, label="h_true=0.73")
    ax.set_xlabel("h")
    ax.set_ylabel("L(h) / L_max")
    ax.set_title("Galaxy density gradient effect")
    ax.legend(fontsize=8)

    plt.suptitle("Test 4: Multi-galaxy scenarios with P_det=1", fontsize=12)
    plt.tight_layout()
    outpath = OUTPUT_DIR / "test_04_multi_galaxy_pdet1.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
    plt.close()

    # Summary
    print("\n=== SUMMARY ===")
    print(f"A (true host only):      peak = {peak_a:.3f}, offset = {peak_a - H_TRUE:+.3f}")
    print(f"B (+50 uniform):         peak = {peak_b:.3f}, offset = {peak_b - H_TRUE:+.3f}")
    print(f"C (+100 uniform):        peak = {peak_c:.3f}, offset = {peak_c - H_TRUE:+.3f}")
    print(f"D (+100 low-z biased):   peak = {peak_d:.3f}, offset = {peak_d - H_TRUE:+.3f}")
    print(f"E (+100 high-z biased):  peak = {peak_e:.3f}, offset = {peak_e - H_TRUE:+.3f}")
    print(f"F (100 uniform, no host): peak = {peak_f:.3f}, offset = {peak_f - H_TRUE:+.3f}")
    print()
    if abs(peak_b - H_TRUE) < 0.02:
        print("Uniform galaxies cause < 0.02 offset → P_det=1 galaxy density bias is SMALL")
    else:
        print(f"Uniform galaxies cause {abs(peak_b - H_TRUE):.3f} offset → galaxy density bias is SIGNIFICANT")
    if abs(peak_d - peak_e) > 0.03:
        print(f"Galaxy density gradient causes {abs(peak_d - peak_e):.3f} shift → density distribution MATTERS")


if __name__ == "__main__":
    main()
