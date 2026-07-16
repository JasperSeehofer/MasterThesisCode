"""H2: controlled 2D mass-kernel -> H0 bias toy (sign + magnitude), swept.

Isolates the host-MASS kernel's effect on the inferred H0. Two arms differ ONLY
in p_M(M):
  production : N(M; M_g_eff, sigma_M)  [LINEAR, sigma_M=0.6 M_g, G2d-shifted mean]
  correct    : LogNormal(M; M_g, sigma_lnM) * R_eff(M) truncated on [M_MIN,M_MAX]
The production-minus-correct H0 shift is the mass-kernel-induced bias. Common-mode
z-marginalisation (Jensen) bias is identical in both arms and cancels in the diff.
Swept over the photo-z width sigma_z (looser photo-z -> the mass channel carries
more of the z constraint -> larger mass-kernel leverage, the real GLADE regime).
"""

import numpy as np

from master_thesis_code.bayesian_inference.bayesian_statistics import eddington_shifted_host_mass
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.physical_relations import dist

H_TRUE = 0.73
M_MIN, M_MAX = 1e4, 1e7
SIGMA_LNM = 0.24 * np.log(10)
SIGMA_REL_LIN = 0.6

# dist(z,h) = A(z)/h exactly (H0=100h; Omega_m h-independent). Precompute A(z).
_ZT = np.linspace(1e-4, 1.2, 6000)
_AT = np.array([dist(z, h=1.0) for z in _ZT])  # A(z) = dist(z, h=1)


def A_of_z(z: np.ndarray) -> np.ndarray:
    return np.interp(z, _ZT, _AT)


def _sample_Mtrue(n: int, rng: np.random.Generator) -> np.ndarray:
    grid = np.logspace(np.log10(M_MIN), np.log10(M_MAX), 4000)
    pdf = np.asarray(R_eff_per_mbh(grid), dtype=np.float64) * grid
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)


def _mass_kernel(M_grid: np.ndarray, M_g: float, arm: str) -> np.ndarray:
    if arm == "production":
        sig = SIGMA_REL_LIN * M_g
        M_eff = eddington_shifted_host_mass(M_g, sig)
        k = np.exp(-0.5 * ((M_grid - M_eff) / sig) ** 2)
    else:  # correct
        lnk = -0.5 * ((np.log(M_grid) - np.log(M_g)) / SIGMA_LNM) ** 2
        k = np.exp(lnk) / M_grid * np.asarray(R_eff_per_mbh(M_grid), dtype=np.float64)
    return k / max(np.trapezoid(k, M_grid), 1e-300)


def run(sigma_z: float, sigma_mz: float, n_events: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Wide grid so neither arm rails at loose photo-z (the common-mode z-bias can
    # push both means high); the production-minus-correct diff stays clean as long
    # as neither posterior is clipped at an edge (check correct_mean << 1.20).
    h_grid = np.linspace(0.50, 1.20, 220)
    M_grid = np.logspace(np.log10(M_MIN), np.log10(M_MAX), 400)
    sigma_dl = 0.05
    nzq = 60

    z_true = rng.uniform(0.30, 0.50, n_events)
    M_true = _sample_Mtrue(n_events, rng)
    dL_obs = (A_of_z(z_true) / H_TRUE) * (1.0 + rng.normal(0.0, sigma_dl, n_events))
    Mz_obs = M_true * (1.0 + z_true) * (1.0 + rng.normal(0.0, sigma_mz, n_events))
    z_gal = z_true + rng.normal(0.0, sigma_z, n_events)
    M_g = np.clip(M_true * np.exp(rng.normal(0.0, SIGMA_LNM, n_events)), M_MIN, M_MAX)

    res = {}
    for arm in ("correct", "production"):
        logL = np.zeros(h_grid.size)
        for i in range(n_events):
            z_lo = max(1e-4, z_gal[i] - 5 * sigma_z)
            z_hi = z_gal[i] + 5 * sigma_z
            zq = np.linspace(z_lo, z_hi, nzq)
            kz = np.exp(-0.5 * ((zq - z_gal[i]) / sigma_z) ** 2)  # flat-prior photo-z anchor
            kz /= max(np.trapezoid(kz, zq), 1e-300)
            pM = _mass_kernel(M_grid, float(M_g[i]), arm)
            Mz_model = M_grid[None, :] * (1.0 + zq[:, None])
            sig_mz = sigma_mz * Mz_obs[i]
            gw_mass = np.exp(-0.5 * ((Mz_obs[i] - Mz_model) / sig_mz) ** 2)
            mass_marg = np.trapezoid(gw_mass * pM[None, :], M_grid, axis=1)  # (nz,)
            dLg = (A_of_z(zq)[:, None]) / h_grid[None, :]
            sig_dl = sigma_dl * dL_obs[i]
            pGW = np.exp(-0.5 * ((dL_obs[i] - dLg) / sig_dl) ** 2)
            num = np.trapezoid((kz * mass_marg)[:, None] * pGW, zq, axis=0)
            logL += np.log(np.clip(num, 1e-300, None))
        post = np.exp(logL - logL.max())
        post /= np.trapezoid(post, h_grid)
        res[arm] = {
            "MAP": float(h_grid[np.argmax(post)]),
            "mean": float(np.trapezoid(h_grid * post, h_grid)),
        }
    res["diff_mean"] = res["production"]["mean"] - res["correct"]["mean"]
    res["control_bias"] = res["correct"]["mean"] - H_TRUE
    return res


if __name__ == "__main__":
    print("H2 mass-kernel -> H0 (truth 0.73). diff = production - correct (mass-kernel bias).")
    print(
        f"{'sigma_z':>8} {'sig_z/z':>8} {'sigma_mz':>9} {'ctrl_bias':>10} "
        f"{'corr_mean':>10} {'prod_mean':>10} {'DIFF':>9}"
    )
    for sigma_z in (0.02, 0.06, 0.12, 0.20):
        for sigma_mz in (0.01,):
            r = run(sigma_z, sigma_mz, n_events=1500, seed=0)
            print(
                f"{sigma_z:8.2f} {sigma_z / 0.40:8.2f} {sigma_mz:9.3f} "
                f"{r['control_bias']:+10.4f} {r['correct']['mean']:10.4f} "
                f"{r['production']['mean']:10.4f} {r['diff_mean']:+9.4f}"
            )
