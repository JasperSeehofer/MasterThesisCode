"""Decisive isolation test for the host-mass kernel truncation hypothesis (H2).

Production models the per-galaxy host-mass prior as N(M; M_g, sigma_M) * R_eff(M),
and (G2d) approximates it by the shifted Gaussian N(M; M_g*(1+alpha*sigma_rel^2),
sigma_M) via the exponential-tilt identity — EXACT only when (i) R_eff is locally
log-linear over the Gaussian width and (ii) the Gaussian is untruncated. At the
catalogue's sigma_rel = sigma_M/M ~ 0.6 both break: the +/-4sigma window spans
M_g*(1 -> -1.4) (mass below 0 and past the EMRI [M_min,M_max]=[1e4,1e7] bounds),
and R_eff curves over that range.

This compares the FIRST MOMENT of the effective host-mass prior:
  (A) EXACT  : <M> under N*R_eff truncated+renormalized on [M_min,M_max]  (fine quad)
  (B) G2d    : production's shifted effective mass (eddington_shifted_host_mass)
  (C) bare   : M_g
A large (B)-(A) = the production mass kernel believes in the wrong effective mass
=> the 2D mass-marginalised likelihood peaks at a biased M => biased (1+z) => H0.
"""

import numpy as np

from master_thesis_code.bayesian_inference.bayesian_statistics import eddington_shifted_host_mass
from master_thesis_code.emri_rate import R_eff_per_mbh

M_MIN, M_MAX = 1e4, 1e7


def exact_truncated_mean(M_g: float, sigma_rel: float) -> float:
    """<M> under N(M;M_g,sigma_M)*R_eff(M) truncated+renormalised on [M_MIN,M_MAX]."""
    sigma_M = sigma_rel * M_g
    # fine linear grid over the physical support intersected with the +/-6sigma window
    lo = max(M_MIN, M_g - 6 * sigma_M)
    hi = min(M_MAX, M_g + 6 * sigma_M)
    if hi <= lo:
        return M_g
    M = np.linspace(lo, hi, 20001)
    gauss = np.exp(-0.5 * ((M - M_g) / sigma_M) ** 2)
    w = np.asarray(R_eff_per_mbh(M), dtype=np.float64)
    p = gauss * w
    Z = np.trapezoid(p, M)
    return float(np.trapezoid(M * p, M) / Z) if Z > 0 else M_g


def main() -> None:
    print("=== (B) G2d vs (A) exact truncated effective host mass, sigma_rel=0.6 ===")
    print(
        f"{'M_g':>10} {'A_exact':>12} {'B_G2d':>12} {'C_bare':>10} "
        f"{'(B-A)/M_g':>10} {'(A-M_g)/M_g':>11}"
    )
    sig = 0.6
    for M_g in [1.5e4, 3e4, 1e5, 3e5, 1e6, 3e6, 7e6]:
        A = exact_truncated_mean(M_g, sig)
        B = eddington_shifted_host_mass(M_g, sig * M_g)
        C = M_g
        print(
            f"{M_g:10.2e} {A:12.4e} {B:12.4e} {C:10.2e} "
            f"{(B - A) / M_g:10.4f} {(A - M_g) / M_g:11.4f}"
        )

    print("\n=== sigma_rel dependence at M_g=3e5 (mid-population) ===")
    print(f"{'sig_rel':>8} {'A_exact':>12} {'B_G2d':>12} {'(B-A)/M_g':>10}")
    M_g = 3e5
    for sig in [0.05, 0.15, 0.30, 0.45, 0.60, 0.75]:
        A = exact_truncated_mean(M_g, sig)
        B = eddington_shifted_host_mass(M_g, sig * M_g)
        print(f"{sig:8.2f} {A:12.4e} {B:12.4e} {(B - A) / M_g:10.4f}")

    print("\n=== P(M<0) and P(M>M_max) under the untruncated linear Gaussian ===")
    from scipy.stats import norm

    for M_g in [1.5e4, 3e5, 7e6]:
        for sig in [0.6]:
            sM = sig * M_g
            below0 = norm.cdf(0.0, M_g, sM)
            above = 1 - norm.cdf(M_MAX, M_g, sM)
            belowmin = norm.cdf(M_MIN, M_g, sM)
            print(
                f"M_g={M_g:.2e} sig_rel={sig}: P(M<0)={below0 * 100:5.2f}%  "
                f"P(M<M_min)={belowmin * 100:5.2f}%  P(M>M_max)={above * 100:5.2f}%"
            )


if __name__ == "__main__":
    main()
