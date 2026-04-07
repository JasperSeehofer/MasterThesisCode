#!/usr/bin/env python3
"""Test 3: Single detection + true host galaxy (baseline sanity).

Constructs a synthetic detection at z_true=0.15 with a realistic Fisher matrix,
creates one HostGalaxy at z_true, uses P_det=1.0, and verifies the likelihood
peaks at h=0.73.

Expected: peak at h=0.73 ± 0.01.
If this fails, the core formula is wrong.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from master_thesis_code.datamodels.detection import Detection
from master_thesis_code.galaxy_catalogue.handler import HostGalaxy
from master_thesis_code.physical_relations import dist, dist_to_redshift

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

H_TRUE = 0.73


class MockDetectionProbability:
    """Returns P_det = 1.0 for all inputs."""

    def detection_probability_without_bh_mass_interpolated(
        self, d_L, phi, theta, *, h
    ):
        return np.ones_like(np.atleast_1d(np.asarray(d_L, dtype=np.float64)))

    def detection_probability_with_bh_mass_interpolated(
        self, d_L, M, phi, theta, *, h
    ):
        return np.ones_like(np.atleast_1d(np.asarray(d_L, dtype=np.float64)))


def make_synthetic_detection(z_true: float, h_true: float, frac_dl_err: float = 0.05) -> Detection:
    """Create a synthetic detection at z_true with given fractional d_L error."""
    d_L = dist(z_true, h=h_true)
    sigma_dL = frac_dl_err * d_L
    phi = 1.0
    theta = 0.8
    sigma_phi = 0.05
    sigma_theta = 0.05
    M = 5e5  # solar masses
    sigma_M = 50.0

    params = pd.Series({
        "luminosity_distance": d_L,
        "delta_luminosity_distance_delta_luminosity_distance": sigma_dL**2,
        "phiS": phi,
        "delta_phiS_delta_phiS": sigma_phi**2,
        "qS": theta,
        "delta_qS_delta_qS": sigma_theta**2,
        "M": M,
        "delta_M_delta_M": sigma_M**2,
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


def build_gaussian_3d(detection: Detection) -> multivariate_normal:
    """Build the 3D GW likelihood Gaussian (phi, theta, d_L_frac)."""
    cov = [
        [detection.phi_error**2, detection.theta_phi_covariance, detection.d_L_phi_covariance / detection.d_L],
        [detection.theta_phi_covariance, detection.theta_error**2, detection.d_L_theta_covariance / detection.d_L],
        [detection.d_L_phi_covariance / detection.d_L, detection.d_L_theta_covariance / detection.d_L,
         detection.d_L_uncertainty**2 / detection.d_L**2],
    ]
    return multivariate_normal(mean=[detection.phi, detection.theta, 1.0], cov=cov, allow_singular=True)


def compute_single_host_likelihood(
    detection: Detection,
    galaxy: HostGalaxy,
    gaussian_3d: multivariate_normal,
    h: float,
    n_quad: int = 50,
) -> tuple[float, float]:
    """Compute numerator and denominator for a single (detection, galaxy) pair.

    Reimplements the core logic from bayesian_statistics.py single_host_likelihood
    WITHOUT the global state dependency, for testing.
    """
    from master_thesis_code.physical_relations import dist_vectorized
    from scipy.integrate import fixed_quad
    from scipy.stats import norm

    sigma_mult = 4.0

    # Integration bounds for numerator (h-dependent)
    num_z_upper = dist_to_redshift(detection.d_L + sigma_mult * detection.d_L_uncertainty, h=h)
    num_z_lower = dist_to_redshift(max(0.001, detection.d_L - sigma_mult * detection.d_L_uncertainty), h=h)

    # Integration bounds for denominator (galaxy-based, h-independent)
    den_z_upper = galaxy.z + sigma_mult * galaxy.z_error
    den_z_lower = max(0.001, galaxy.z - sigma_mult * galaxy.z_error)

    galaxy_z_dist = norm(loc=galaxy.z, scale=galaxy.z_error)

    def numerator_integrand(z):
        d_L_trial = dist_vectorized(z, h=h)
        d_L_frac = d_L_trial / detection.d_L
        phi = np.full_like(z, galaxy.phiS)
        theta = np.full_like(z, galaxy.qS)
        # P_det = 1.0 (mock)
        gw_likelihood = gaussian_3d.pdf(np.vstack([phi, theta, d_L_frac]).T)
        return gw_likelihood * galaxy_z_dist.pdf(z)

    def denominator_integrand(z):
        # P_det = 1.0 (mock)
        return galaxy_z_dist.pdf(z)

    num, _ = fixed_quad(numerator_integrand, num_z_lower, num_z_upper, n=n_quad)
    den, _ = fixed_quad(denominator_integrand, den_z_lower, den_z_upper, n=n_quad)

    return float(num), float(den)


def main() -> None:
    h_values = np.arange(0.60, 0.87, 0.01)

    # Test at multiple redshifts
    z_tests = [0.03, 0.05, 0.10, 0.15, 0.20]

    fig, axes = plt.subplots(1, len(z_tests), figsize=(4 * len(z_tests), 4), sharey=True)

    for ax_idx, z_true in enumerate(z_tests):
        detection = make_synthetic_detection(z_true, H_TRUE, frac_dl_err=0.05)
        galaxy = HostGalaxy.from_attributes(
            phiS=detection.phi,
            qS=detection.theta,
            z=z_true,
            z_error=0.002,
            M=5e5,
            M_error=1e4,
        )
        gaussian_3d = build_gaussian_3d(detection)

        likelihoods = []
        numerators = []
        denominators = []

        for h in h_values:
            num, den = compute_single_host_likelihood(detection, galaxy, gaussian_3d, h)
            likelihoods.append(num / den if den > 0 else 0)
            numerators.append(num)
            denominators.append(den)

        likelihoods = np.array(likelihoods)
        numerators = np.array(numerators)
        denominators = np.array(denominators)

        # Normalize for plotting
        if likelihoods.max() > 0:
            likelihoods /= likelihoods.max()

        peak_h = h_values[np.argmax(likelihoods)]
        offset = peak_h - H_TRUE

        print(f"\nz_true = {z_true:.2f}:")
        print(f"  d_L = {detection.d_L:.4f} Gpc, σ_dL = {detection.d_L_uncertainty:.4f} Gpc")
        print(f"  Peak h = {peak_h:.3f} (offset = {offset:+.3f})")
        print(f"  L(0.66) / L(0.73) = {likelihoods[np.argmin(abs(h_values-0.66))] / max(likelihoods[np.argmin(abs(h_values-0.73))], 1e-30):.6f}")

        # Diagnostic: check num and den separately
        print(f"  Numerator at h=0.73: {numerators[np.argmin(abs(h_values-0.73))]:.6e}")
        print(f"  Denominator at h=0.73: {denominators[np.argmin(abs(h_values-0.73))]:.6e}")
        print(f"  Numerator at h=0.66: {numerators[np.argmin(abs(h_values-0.66))]:.6e}")
        print(f"  Denominator at h=0.66: {denominators[np.argmin(abs(h_values-0.66))]:.6e}")

        ax = axes[ax_idx]
        ax.plot(h_values, likelihoods, "o-", ms=3)
        ax.axvline(H_TRUE, color="r", ls="--", lw=1, alpha=0.7)
        ax.axvline(peak_h, color="g", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("h")
        ax.set_title(f"z={z_true:.2f}\npeak={peak_h:.3f}")
        if ax_idx == 0:
            ax.set_ylabel("L(h) / L_max")

    plt.suptitle("Test 3: Single galaxy (true host), P_det=1\nExpected: peak at h=0.73", fontsize=12)
    plt.tight_layout()
    outpath = OUTPUT_DIR / "test_03_single_galaxy_sanity.png"
    fig.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")
    plt.close()

    # Summary
    print("\n=== SUMMARY ===")
    print("If all peaks are at h=0.73 ± 0.01, the core formula is CORRECT.")
    print("Any offset indicates a formula error in the single-galaxy case.")


if __name__ == "__main__":
    main()
