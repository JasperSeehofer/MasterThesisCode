"""C7 — correctness / robustness checks on the kernel driver.

1. fixed_quad cross-check: the batched GL-50 reduction used by the driver must
   reproduce scipy.integrate.fixed_quad on the same integrand (the code claims
   bit-parity; verify it holds for OUR integrand too).
2. h-invariance of the kernel: w_pop(z;h)/Z_g(h) must be exactly h-independent
   (comoving_volume_element carries a pure h^-3 prefactor that cancels).  If it
   is, every h-dependence of N_g comes from the GW term and the window - which
   is what makes "the kernel shifts the peak" a statement about the PRIOR SHAPE,
   not about an h-dependent prior.
3. Quadrature-order robustness at realistic sigma_z: n=50 vs n=400.  (The
   volume_trunc post-mortem found fixed_quad n=50 ALIASING the GW peak once the
   numerator window was widened to the host window; the default numerator window
   is the narrow GW window, so this must be re-checked, not assumed.)
4. The analytic point-kernel tilt Delta ln N(0.73 -> 0.86), which the numerical
   eps->0 limit CANNOT deliver (a delta-like prior inside the wide GW window is
   unresolvable by GL-50 - the +460 nat entry in c7_vs_production.py's eps=1e-4
   row is that aliasing artifact, not physics).

Read-only.  Run from the repo root with .venv/bin/python.
"""

from pathlib import Path

import numpy as np
from c7_kernel_measure import deconv_profile, load_incat_events, point_peak_h  # noqa: E402
from scipy.integrate import fixed_quad

from master_thesis_code.bayesian_inference.bayesian_statistics import _gaussian_pdf
from master_thesis_code.constants import H as H_TRUE
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
)

HERE = Path(__file__).parent


def main() -> None:
    ev = load_incat_events()
    n = len(ev)
    d_L = ev.d_L.to_numpy()
    s_dL = ev.sigma_dL.to_numpy()
    s_fr = ev.sigma_frac_cond.to_numpy()
    z_t = ev.z_true.to_numpy()

    # ---- 1. fixed_quad cross-check on host 0 -------------------------------
    i, h, eps = 0, 0.79, 0.4
    sz = eps * z_t[i]
    lo = max(z_t[i] - 4 * sz, 1e-6)
    hi = z_t[i] + 4 * sz
    Zg = fixed_quad(
        lambda z: (
            _gaussian_pdf(z, z_t[i], sz)
            * np.asarray(comoving_volume_element(z, h=h), float)
            / (1 + z)
        ),
        lo,
        hi,
        n=50,
    )[0]
    zhi = dist_to_redshift(d_L[i] + 4 * s_dL[i], h=h)
    zlo = dist_to_redshift(d_L[i] - 4 * s_dL[i], h=h)
    Ng = fixed_quad(
        lambda z: (
            np.exp(-0.5 * ((dist_vectorized(z, h=h) / d_L[i] - 1) / s_fr[i]) ** 2)
            * _gaussian_pdf(z, z_t[i], sz)
            * np.asarray(comoving_volume_element(z, h=h), float)
            / (1 + z)
            / Zg
        ),
        zlo,
        zhi,
        n=50,
    )[0]
    Ng_drv = deconv_profile(
        np.array([h]),
        d_L[i : i + 1],
        s_dL[i : i + 1],
        s_fr[i : i + 1],
        z_t[i : i + 1],
        np.array([sz]),
    )[0, 0]
    print(
        f"1. fixed_quad vs driver: {Ng:.12e} vs {Ng_drv:.12e}   "
        f"rel diff = {abs(Ng - Ng_drv) / Ng:.3e}"
    )

    # ---- 2. h-invariance of w_pop/Z_g --------------------------------------
    z = np.linspace(lo, hi, 7)
    r = []
    for hh in (0.60, 0.73, 0.86, 1.20):
        Z = fixed_quad(
            lambda t, hh=hh: (  # bind loop var
                _gaussian_pdf(t, z_t[i], sz)
                * np.asarray(comoving_volume_element(t, h=hh), float)
                / (1 + t)
            ),
            lo,
            hi,
            n=50,
        )[0]
        r.append(np.asarray(comoving_volume_element(z, h=hh), float) / (1 + z) / Z)
    r = np.array(r)
    print(
        f"2. w_pop/Z_g h-invariance: max rel spread over h in [0.6,1.2] = "
        f"{np.max(np.ptp(r, axis=0) / np.mean(r, axis=0)):.3e}"
    )

    # ---- 3. quadrature order ------------------------------------------------
    hgrid = np.array([0.60, 0.73, 0.86])
    for eps in (0.25, 0.49, 0.80):
        sz = eps * z_t
        a = deconv_profile(hgrid, d_L, s_dL, s_fr, z_t, sz, n_nodes=50)
        b = deconv_profile(hgrid, d_L, s_dL, s_fr, z_t, sz, n_nodes=400)
        rel = np.abs(a - b) / np.abs(b)
        tilt_a = np.log(a[:, 2] / a[:, 1])
        tilt_b = np.log(b[:, 2] / b[:, 1])
        print(
            f"3. eps={eps}: n=50 vs n=400 max rel diff = {rel.max():.3e}; "
            f"median tilt(0.73->0.86) {np.median(tilt_a):+.4f} vs {np.median(tilt_b):+.4f}"
        )

    # ---- 4. analytic point-kernel tilt --------------------------------------
    fr73 = np.asarray(dist_vectorized(z_t, h=0.73), float) / d_L
    fr86 = np.asarray(dist_vectorized(z_t, h=0.86), float) / d_L
    tilt_pt = -0.5 * ((fr86 - 1) / s_fr) ** 2 + 0.5 * ((fr73 - 1) / s_fr) ** 2
    print(
        "4. POINT kernel Delta ln N_g(0.73->0.86), analytic: "
        + "  ".join(f"p{p}={np.percentile(tilt_pt, p):+.1f}" for p in (5, 50, 95))
    )
    print(
        f"   (point-kernel peak is exactly h_true: median {np.median(point_peak_h(z_t, d_L)):.6f}, "
        f"h_true={H_TRUE})"
    )


if __name__ == "__main__":
    main()
