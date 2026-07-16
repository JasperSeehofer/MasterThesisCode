"""Diagnostic: is the volume_trunc numerator failure a quadrature artifact?

volume_trunc integrates the in-catalogue numerator over the WIDE per-host galaxy
window [z_g - 4σ, z_g + 4σ] instead of the NARROW event-level GW window. For a
shallow low-z photo-z host (σ_z/z ~ O(1)) the host window is wide (~0.18 in z)
while the GW likelihood peak in z is narrow (~0.003). This compares the shared
fixed_quad(n=50) used in production against a high-accuracy adaptive quad, across
h, to separate a quadrature-resolution artifact from a genuine estimator tilt.

Result (2026-07-12): n=50 over the wide host window returns ~0.0 for every h
(the sharp GW peak falls between the sparse Gauss-Legendre nodes), while the
exact integral is 0.24-0.65 and monotonically INCREASING in h. So (1) the n=50
machinery is numerically invalid over the wide window (peak aliasing), and (2)
even the exact numerator tilts high in this regime. Both push H0 high -> the
production A/B posterior collapsed onto h=0.80.

Run: uv run python results/volume_trunc_ab_20260712/quadrature_diagnostic.py
"""

import numpy as np
from scipy.integrate import fixed_quad, quad
from scipy.stats import norm

from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist,
    dist_to_redshift,
)


def main() -> None:
    h_grid = [0.60, 0.70, 0.73, 0.80, 0.86]
    z_g, sz = 0.05, 0.033  # sigma_z/z ~ 0.66 (representative seed600 photo-z host)
    d_L_det = float(dist(z_g, h=0.73))
    dL_unc = 0.05 * d_L_det
    sig_dl_frac = 0.05

    def gw_dlfrac(z: np.ndarray, h: float) -> np.ndarray:
        f = np.asarray(dist(z, h=h)) / d_L_det
        return np.exp(-0.5 * ((f - 1.0) / sig_dl_frac) ** 2) / (np.sqrt(2 * np.pi) * sig_dl_frac)

    print(f"host z_g={z_g} sz={sz} (sz/z={sz / z_g:.2f}); d_L_det={d_L_det:.4f} Gpc")
    print(
        f"{'h':>6} | {'GW-window(n50)':>16} | {'host-window(n50)':>18} | {'host(exact)':>12} | n50/exact"
    )
    for h in h_grid:
        den_lo, den_hi = max(z_g - 4 * sz, 0.0), z_g + 4 * sz

        def wpop(z: np.ndarray, h: float = h) -> np.ndarray:
            return np.asarray(comoving_volume_element(z, h=h)) / (1.0 + z)

        def prior_un(z: np.ndarray, h: float = h) -> np.ndarray:
            return norm(z_g, sz).pdf(z) * wpop(z, h)

        z_norm = fixed_quad(prior_un, den_lo, den_hi, n=50)[0]

        def pg(z: np.ndarray, h: float = h, z_norm: float = z_norm) -> np.ndarray:
            return prior_un(z, h) / z_norm

        num_lo = dist_to_redshift(d_L_det - 4 * dL_unc, h=h)
        num_hi = dist_to_redshift(d_L_det + 4 * dL_unc, h=h)
        n_gw = fixed_quad(lambda z, h=h: gw_dlfrac(z, h) * pg(z, h), num_lo, num_hi, n=50)[0]
        n_host_50 = fixed_quad(lambda z, h=h: gw_dlfrac(z, h) * pg(z, h), den_lo, den_hi, n=50)[0]
        n_host_ex = quad(
            lambda z, h=h: float(gw_dlfrac(z, h) * pg(z, h)), den_lo, den_hi, limit=400
        )[0]
        print(
            f"{h:6.2f} | {n_gw:16.4f} | {n_host_50:18.4f} | {n_host_ex:12.4f} | "
            f"{n_host_50 / n_host_ex if n_host_ex else float('nan'):.4f}"
        )


if __name__ == "__main__":
    main()
