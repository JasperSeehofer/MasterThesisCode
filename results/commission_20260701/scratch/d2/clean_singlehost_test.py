"""Clean controlled coverage sub-test isolating the in-catalogue numerator z-prior.

Setup (removes all confounds):
  - fully complete catalogue over the detectable range (f ~ 1), so B_num=0,
    beta_G=D and p_i reduces to the in-catalogue term ONLY.
  - EXACTLY one candidate host per event = the true host, with a photo-z.
  - detection selection p_det on the events (Malmquist) is retained and handled
    by the D(h) denominator, exactly as in production.

In this limit the per-event likelihood is  p_i(h) = num(h) / D(h)  with
  num_A(h)  = int p_GW(A(z)/h) N(z; z_g, sig_z) dz              (FLAT prior)
  num_V(h)  = int p_GW(A(z)/h) N(z; z_g, sig_z) w_pop(z)/Z dz    (VOLUME prior)
  D(h)      = int p_det(A(z)/h) w_pop(z) dz.
Only the numerator's z-prior differs -> a pure test of which prior is calibrated.
"""

from __future__ import annotations

import numpy as np

import pp_coverage_test as m


def run(h_true, h_grid, n_real, n_events=250, sigma_z=0.035, sigma_dl_frac=0.05, seed=0):
    # Denominator D(h) = int p_det w_pop dz (intrinsic prior, selection in denom)
    zint = np.linspace(m.Z_MIN, m.Z_MAX_POP, 3000)
    wpop_i = m.wpop_of_z(zint)
    Dh = np.trapezoid(m.p_det(m.A_of_z(zint)[:, None] / h_grid[None, :]) * wpop_i[:, None],
                      zint, axis=0)
    logDh = np.log(Dh)

    master = np.random.default_rng(seed)
    res = {"A": {50: 0, 68: 0, 90: 0, "rail": 0, "maps": []},
           "V": {50: 0, 68: 0, 90: 0, "rail": 0, "maps": []}}

    for _ in range(n_real):
        rng = np.random.default_rng(master.integers(1 << 62))
        # detected true hosts (Malmquist selection)
        z_host = m.sample_from_density(m.detected_pop_density(h_true),
                                       m.Z_MIN, m.Z_MAX_POP, n_events, rng)
        dL_host = m.A_of_z(z_host) / h_true
        dL_obs = np.clip(dL_host + rng.normal(0.0, sigma_dl_frac * dL_host), 1e-3, None)
        sig_dl = sigma_dl_frac * dL_obs
        z_g = np.clip(z_host + rng.normal(0.0, sigma_z, n_events), m.Z_MIN, None)

        logL = {"A": np.zeros(h_grid.size), "V": np.zeros(h_grid.size)}
        for i in range(n_events):
            z_lo = max(m.Z_MIN, float(m.z_from_dL(dL_obs[i] - 5 * sig_dl[i], h_grid.min())) - 4 * sigma_z)
            z_hi = min(m._ZG[-1], float(m.z_from_dL(dL_obs[i] + 5 * sig_dl[i], h_grid.max())) + 4 * sigma_z)
            zq = np.linspace(z_lo, z_hi, 160)
            wq = np.gradient(zq)
            dLg = m.A_of_z(zq)[:, None] / h_grid[None, :]
            pGW = m.norm_pdf(dLg, dL_obs[i], sig_dl[i])
            K = m.norm_pdf(zq, z_g[i], sigma_z)                 # (nz,)
            Kw = K * m.wpop_of_z(zq)
            Kw = Kw / max(np.trapezoid(Kw, zq), 1e-300)          # normalized volume prior
            num_A = np.einsum("z,zh,z->h", wq, pGW, K)
            num_V = np.einsum("z,zh,z->h", wq, pGW, Kw)
            logL["A"] += np.log(np.clip(num_A, 1e-300, None)) - logDh
            logL["V"] += np.log(np.clip(num_V, 1e-300, None)) - logDh

        for key in ("A", "V"):
            post = m.posterior_from_logL(logL[key], h_grid)
            mi = int(np.argmax(post))
            res[key]["maps"].append(float(h_grid[mi]))
            if mi == 0 or mi == h_grid.size - 1:
                res[key]["rail"] += 1
            for lv in (50, 68, 90):
                if m.hpd_contains(h_grid, post, h_true, lv / 100):
                    res[key][lv] += 1
    return res


if __name__ == "__main__":
    import time
    h_grid = np.arange(0.600, 0.8601, 0.004)
    for h_true in (0.66, 0.72, 0.78):
        t = time.time()
        r = run(h_true, h_grid, n_real=120, seed=int(h_true * 1e4))
        for key, lab in (("A", "FLAT (production)"), ("V", "VOLUME (correct)")):
            n = len(r[key]["maps"])
            mm = np.mean(r[key]["maps"])
            print(f"h_true={h_true:.2f} [{lab:18s}] "
                  f"cov50={r[key][50]/n:.2f} cov68={r[key][68]/n:.2f} cov90={r[key][90]/n:.2f} "
                  f"rail={r[key]['rail']/n:.2f} MAP_mean={mm:.4f} bias={mm-h_true:+.4f}")
        print(f"  ({time.time()-t:.0f}s)")
