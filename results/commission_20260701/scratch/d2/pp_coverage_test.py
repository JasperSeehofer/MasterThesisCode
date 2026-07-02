"""Independent P-P / coverage test of a dark-siren H0 estimator.

From-scratch synthetic reimplementation (does NOT import the repo's inference
code).  Builds a small synthetic universe + photo-z catalogue with completion
term + interlopers, and runs a frequentist coverage test on four per-event
likelihood constructions of the form  p_i(h) = (beta_G L_cat + B_num)/D(h):

  A_prod   PRODUCTION z-prior : in-cat numerator = BARE Gaussian N(z;z_gal,sig_z)
                               (NO dVc/dz weight); local self-normalized L_cat.
  B_corr   CORRECT            : per-galaxy volume-prior deconvolution
                               p_g(z)=N(z_g;z,sig)*w_pop(z)/Z_g; local
                               self-normalized L_cat -> the calibration fix.
  B_naive  DIAGNOSTIC         : numerator multiplied by w_pop but NOT renormalized.
  A_global DIAGNOSTIC         : production's literal GLOBAL selection denominator
                               (normalization-sensitive; kept for comparison).

The completion term B_num, selection denominators D(h)/beta_G/beta_Gbar, the
detection/population model and the sky-localization candidate weighting are
IDENTICAL across estimators; only the in-catalogue numerator's z-prior differs,
isolating the cause of any miscalibration.  NOTE: the completion term is only
weakly H0-informative, so with an incomplete catalogue the MAP response to H0 is
compressed (slope<1) for ALL estimators -- the DECISIVE, confound-free test is
the fully-complete single-host experiment in clean_singlehost_test.py.

Units: h in [100 km/s/Mpc].  Cosmology: flat LambdaCDM, Omega_m=0.3.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import erfc

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Cosmology (flat LambdaCDM).  d_L(z,h) = A(z)/h ; volume-prior shape w_pop(z).
# ----------------------------------------------------------------------------
C_KM_S = 299_792.458
OMEGA_M = 0.30
OMEGA_L = 0.70

_ZG = np.linspace(0.0, 1.5, 15_001)
_E = np.sqrt(OMEGA_M * (1.0 + _ZG) ** 3 + OMEGA_L)
_invE = 1.0 / _E
_I = np.concatenate([[0.0], np.cumsum(0.5 * (_invE[1:] + _invE[:-1]) * np.diff(_ZG))])
# A(z) = (1+z)*(c/100)*I(z) in Mpc, /1000 -> Gpc.  d_L(z,h)=A(z)/h [Gpc].
_A_GPC = (1.0 + _ZG) * (C_KM_S / 100.0) * _I / 1000.0
# w_pop(z) = dVc/dz/dOmega * 1/(1+z)  propto  I(z)^2/E(z)/(1+z)   (1/h^3 cancels)
_WPOP = np.where(_ZG > 0.0, _I**2 / _E / (1.0 + _ZG), 0.0)


def A_of_z(z: np.ndarray) -> np.ndarray:
    return np.interp(z, _ZG, _A_GPC)


def z_of_A(a: np.ndarray) -> np.ndarray:
    return np.interp(a, _A_GPC, _ZG)


def wpop_of_z(z: np.ndarray) -> np.ndarray:
    return np.interp(z, _ZG, _WPOP)


def dL_of_zh(z: np.ndarray, h: float) -> np.ndarray:
    return A_of_z(z) / h


def z_from_dL(dL: np.ndarray, h: float) -> np.ndarray:
    return z_of_A(np.asarray(dL) * h)


# ----------------------------------------------------------------------------
# Detection probability p_det(d_L) and catalogue completeness f(z).
# ----------------------------------------------------------------------------
D50 = 1.85       # Gpc, 50% detection distance
WPDET = 0.30     # Gpc, detection roll-off width
ZF = 0.30        # completeness roll-off redshift
WF = 0.10        # completeness roll-off width
FMAX = 0.90      # peak completeness (nearby)

Z_MIN = 1e-4
Z_MAX_POP = 0.95  # population / catalogue redshift ceiling


def p_det(dL: np.ndarray) -> np.ndarray:
    return 0.5 * erfc((np.asarray(dL) - D50) / (np.sqrt(2.0) * WPDET))


def f_complete(z: np.ndarray) -> np.ndarray:
    return FMAX * 0.5 * erfc((np.asarray(z) - ZF) / (np.sqrt(2.0) * WF))


def norm_pdf(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (np.sqrt(2.0 * np.pi) * sig)


# ----------------------------------------------------------------------------
# Global (event-independent) selection tables on the h-grid.
# D(h)=int p_det w_pop dz ; beta_Gbar=int (1-f) p_det w_pop ; beta_G=D-beta_Gbar.
# n0 = catalogue number-density normalization (galaxies per unit int f w_pop).
# ----------------------------------------------------------------------------
def build_global_tables(h_grid: np.ndarray) -> dict:
    zint = np.linspace(Z_MIN, Z_MAX_POP, 3000)
    wpop = wpop_of_z(zint)
    fz = f_complete(zint)
    A = A_of_z(zint)
    dL = A[:, None] / h_grid[None, :]      # (nz, nh)
    pdet = p_det(dL)                        # (nz, nh)
    Dh = np.trapezoid(pdet * wpop[:, None], zint, axis=0)
    beta_Gbar = np.trapezoid(pdet * ((1.0 - fz) * wpop)[:, None], zint, axis=0)
    beta_G = Dh - beta_Gbar
    int_f_wpop = float(np.trapezoid(fz * wpop, zint))  # for n0
    return {"Dh": Dh, "beta_G": beta_G, "beta_Gbar": beta_Gbar, "int_f_wpop": int_f_wpop}


# ----------------------------------------------------------------------------
# Sampling helpers.
# ----------------------------------------------------------------------------
def sample_from_density(density_fn, z_lo, z_hi, n, rng, ngrid=2000):
    zg = np.linspace(z_lo, z_hi, ngrid)
    pdf = np.clip(density_fn(zg), 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    u = rng.random(n)
    return np.interp(u, cdf, zg)


def detected_pop_density(h_true):
    def dens(z):
        return wpop_of_z(z) * p_det(A_of_z(z) / h_true)
    return dens


def catalogue_density(z):
    # in-catalogue galaxies trace f(z) * w_pop(z)
    return f_complete(z) * wpop_of_z(z)


# ----------------------------------------------------------------------------
# One realization: build catalogue + events, return per-event log-likelihoods
# on the h-grid for every estimator.
# ----------------------------------------------------------------------------
def run_realization(h_true, h_grid, glob, rng, n_events=200, n_cat=8000,
                    sigma_z=0.035, sigma_dl_frac=0.05, box_halfwidth=0.03,
                    sky_big=30000.0):
    nh = h_grid.size

    # --- catalogue (shared) : true z ~ f*w_pop ; photo-z scatter ---
    z_cat_true = sample_from_density(catalogue_density, Z_MIN, Z_MAX_POP, n_cat, rng)
    z_cat_obs = np.clip(z_cat_true + rng.normal(0.0, sigma_z, n_cat), Z_MIN, None)
    order = np.argsort(z_cat_obs)
    z_cat_obs = z_cat_obs[order]
    # n0 : galaxies per unit (int f w_pop) -> catalogue density normalization
    n0 = n_cat / glob["int_f_wpop"]

    # Sigma_global(h) = sum_cat p_det(A(z_obs)/h)  (discrete, literal production)
    A_cat = A_of_z(z_cat_obs)
    Sigma_global = p_det(A_cat[:, None] / h_grid[None, :]).sum(axis=0)  # (nh,)
    g_denom = glob["beta_G"] / np.clip(Sigma_global, 1e-300, None)  # ~ 1/n0

    Dh = glob["Dh"]
    beta_G = glob["beta_G"]

    # accumulators of log p_i(h) over events, per estimator
    logL = {k: np.zeros(nh) for k in ("A_prod", "B_corr", "B_naive", "A_global")}

    # --- events ---
    z_host = sample_from_density(detected_pop_density(h_true), Z_MIN, Z_MAX_POP, n_events, rng)
    dL_host = A_of_z(z_host) / h_true
    dL_obs = dL_host + rng.normal(0.0, sigma_dl_frac * dL_host)
    dL_obs = np.clip(dL_obs, 1e-3, None)
    sig_dl = sigma_dl_frac * dL_obs
    in_cat = rng.random(n_events) < np.clip(f_complete(z_host), 0.0, 1.0)

    for i in range(n_events):
        dlo = dL_obs[i]
        sdl = sig_dl[i]
        # per-event z-integration window covering the GW peak for all h + photo-z tails
        z_lo = max(Z_MIN, float(z_from_dL(dlo - 5 * sdl, h_grid.min())) - 4 * sigma_z)
        z_hi = min(_ZG[-1], float(z_from_dL(dlo + 5 * sdl, h_grid.max())) + 4 * sigma_z)
        zq = np.linspace(z_lo, z_hi, 100)
        Aq = A_of_z(zq)
        wq_trap = np.gradient(zq)                      # trapezoid-ish weights
        dLg = Aq[:, None] / h_grid[None, :]            # (nz, nh)
        pGW = norm_pdf(dLg, dlo, sdl)                  # (nz, nh)
        pdetg = p_det(dLg)                             # (nz, nh)
        wpopq = wpop_of_z(zq)                          # (nz,)
        fq = np.clip(f_complete(zq), 0.0, 1.0)         # (nz,)

        # --- candidate hosts: interlopers from catalogue box + (host if in-cat) ---
        # A sky-localization weight w_sky models the (dropped) angular selection:
        # the true host sits at the event's sky centre (w_sky~1); box interlopers
        # scatter across the localization area (w_sky = exp(-0.5*u), u~U(0,sky_big)),
        # so only a realistic handful survive with appreciable weight.
        zc_ref = float(z_from_dL(dlo, 0.73))
        lo = np.searchsorted(z_cat_obs, zc_ref - box_halfwidth)
        hi = np.searchsorted(z_cat_obs, zc_ref + box_halfwidth)
        cand_zobs = list(z_cat_obs[lo:hi])
        w_sky = list(np.exp(-0.5 * rng.uniform(0.0, sky_big, hi - lo)))
        if in_cat[i]:
            cand_zobs.append(float(np.clip(z_host[i] + rng.normal(0.0, sigma_z), Z_MIN, None)))
            w_sky.append(float(np.exp(-0.5 * rng.chisquare(2))))  # host near centre
        cand_zobs = np.array(cand_zobs)
        w_sky = np.array(w_sky)
        nC = cand_zobs.size

        # completion numerator B_num(h) = int (1-f) pGW w_pop dz   (same for all)
        B_num = (wq_trap * (1.0 - fq) * wpopq) @ pGW    # matvec -> (nh,)

        if nC > 0:
            # per-galaxy photo-z kernels  N(z; z_obs_c, sigma_z)  (nz, nC), each
            # candidate weighted by w_sky.  Sum over candidates FIRST (weighted
            # per-z), then one matvec each: sum_x[h] = (wq * (K@w_sky)) @ pGW.
            K = norm_pdf(zq[:, None], cand_zobs[None, :], sigma_z)            # (nz,nC)
            Kw = K * wpopq[:, None]                                           # * w_pop
            Zg = wq_trap @ Kw                                                 # (nC,) per-gal norm
            Kex = Kw / np.clip(Zg[None, :], 1e-300, None)                     # deconvolved p_g
            wKA = wq_trap * (K @ w_sky)                                       # (nz,)
            wKw = wq_trap * (Kw @ w_sky)                                      # (nz,)
            wKex = wq_trap * (Kex @ w_sky)                                    # (nz,)
            sum_numA = wKA @ pGW                # (nh,)  bare numerator
            sum_den_bare = wKA @ pdetg          # bare per-host selection
            sum_num_naive = wKw @ pGW           # naive volume-multiply
            sum_den_naive = wKw @ pdetg
            sum_num_ex = wKex @ pGW             # volume-prior deconvolved
            sum_den_ex = wKex @ pdetg
            sum_numA_g = g_denom * sum_numA     # production global-denom in-cat term
        else:
            z0 = np.zeros(nh)
            sum_numA = sum_num_naive = sum_num_ex = z0
            sum_den_ex = sum_den_naive = sum_den_bare = z0
            sum_numA_g = z0

        # --- combine per estimator:  p_i = (beta_G L_cat + B_num)/D(h) ---
        # All four are the convex form  w_G L_cat + (1-w_G) L_comp  (self-normalized,
        # normalization-robust) EXCEPT A_global which uses production's literal
        # global selection denominator (kept as a normalization-sensitive diagnostic).
        def _lcat(num, den):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(den > 0, num / den, 0.0)

        # A_prod  : BARE numerator, local self-normalized ratio (production z-prior)
        pA = (beta_G * _lcat(sum_numA, sum_den_bare) + B_num) / Dh
        # B_corr  : volume-prior deconvolved numerator, local self-normalized (correct)
        pBc = (beta_G * _lcat(sum_num_ex, sum_den_ex) + B_num) / Dh
        # B_naive : naive volume-multiply, local self-normalized (diagnostic)
        pBna = (beta_G * _lcat(sum_num_naive, sum_den_naive) + B_num) / Dh
        # A_global: production's literal GLOBAL selection denominator (diagnostic)
        pAg = (sum_numA_g + B_num) / Dh

        logL["A_prod"] += np.log(np.clip(pA, 1e-300, None))
        logL["B_corr"] += np.log(np.clip(pBc, 1e-300, None))
        logL["B_naive"] += np.log(np.clip(pBna, 1e-300, None))
        logL["A_global"] += np.log(np.clip(pAg, 1e-300, None))

    return logL


# ----------------------------------------------------------------------------
# Posterior utilities: normalize, MAP, HPD coverage.
# ----------------------------------------------------------------------------
def posterior_from_logL(logL, h_grid):
    p = np.exp(logL - logL.max())
    p /= np.trapezoid(p, h_grid)
    return p


def hpd_contains(h_grid, post, h_true, level):
    """True if h_true is inside the highest-posterior-density credible region."""
    dh = np.gradient(h_grid)
    mass = post * dh
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    # threshold density = the post level at which cumulative mass reaches `level`
    k = np.searchsorted(csum, level)
    k = min(k, order.size - 1)
    thresh = post[order[k]]
    p_true = float(np.interp(h_true, h_grid, post))
    return p_true >= thresh


def analyze(logL_list, h_grid, h_true):
    levels = {50: 0.50, 68: 0.68, 90: 0.90}
    cov = {L: 0 for L in levels}
    rail = 0
    maps = []
    edge = 0
    for logL in logL_list:
        post = posterior_from_logL(logL, h_grid)
        mi = int(np.argmax(post))
        maps.append(float(h_grid[mi]))
        if mi == 0 or mi == h_grid.size - 1:
            rail += 1
        for L, lv in levels.items():
            if hpd_contains(h_grid, post, h_true, lv):
                cov[L] += 1
    n = len(logL_list)
    return {
        "n": n,
        "coverage": {L: cov[L] / n for L in levels},
        "rail_fraction": rail / n,
        "map_mean": float(np.mean(maps)),
        "map_std": float(np.std(maps)),
        "map_median": float(np.median(maps)),
    }


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    h_grid = np.arange(0.600, 0.8601, 0.004)   # 66 points
    glob = build_global_tables(h_grid)
    estimators = ("A_prod", "B_corr", "B_naive", "A_global")

    H_TRUE = 0.72
    N_REAL = 120
    N_EVENTS = 160
    N_CAT = 6000

    print(f"h-grid: {h_grid[0]:.3f}..{h_grid[-1]:.3f} ({h_grid.size} pts)")
    print(f"D(h) range: {glob['Dh'].min():.4e}..{glob['Dh'].max():.4e}  "
          f"w_G=beta_G/D range: {(glob['beta_G']/glob['Dh']).min():.3f}.."
          f"{(glob['beta_G']/glob['Dh']).max():.3f}")

    # --- primary coverage run at H_TRUE ---
    master = np.random.default_rng(20260701)
    logs = {k: [] for k in estimators}
    for r in range(N_REAL):
        rng = np.random.default_rng(master.integers(1 << 62))
        out = run_realization(H_TRUE, h_grid, glob, rng, n_events=N_EVENTS, n_cat=N_CAT)
        for k in estimators:
            logs[k].append(out[k])
        if (r + 1) % 20 == 0:
            print(f"  realization {r+1}/{N_REAL}  ({time.time()-t0:.0f}s)")

    results = {"config": {
        "H_TRUE": H_TRUE, "N_REAL": N_REAL, "sigma_z": 0.035, "sigma_dl_frac": 0.05,
        "n_events": N_EVENTS, "n_cat": N_CAT, "D50": D50, "WPDET": WPDET,
        "ZF": ZF, "WF": WF, "FMAX": FMAX, "OMEGA_M": OMEGA_M,
        "h_grid": [float(h_grid[0]), float(h_grid[-1]), int(h_grid.size)],
    }, "primary": {}}
    for k in estimators:
        results["primary"][k] = analyze(logs[k], h_grid, H_TRUE)
        a = results["primary"][k]
        print(f"[{k:8s}] cov50={a['coverage'][50]:.2f} cov68={a['coverage'][68]:.2f} "
              f"cov90={a['coverage'][90]:.2f} rail={a['rail_fraction']:.2f} "
              f"MAP={a['map_mean']:.4f}+-{a['map_std']:.4f} (truth {H_TRUE})")

    # --- MAP-tracks-truth sweep ---
    print("\nMAP-tracks-truth sweep:")
    sweep_truths = [0.66, 0.69, 0.72, 0.75, 0.78]
    N_SWEEP = 25
    sweep = {k: {"truth": [], "map_mean": [], "map_std": []} for k in estimators}
    for ht in sweep_truths:
        master_s = np.random.default_rng(int(round(ht * 1e4)) + 7)
        slogs = {k: [] for k in estimators}
        for r in range(N_SWEEP):
            rng = np.random.default_rng(master_s.integers(1 << 62))
            out = run_realization(ht, h_grid, glob, rng, n_events=N_EVENTS, n_cat=N_CAT)
            for k in estimators:
                slogs[k].append(out[k])
        for k in estimators:
            a = analyze(slogs[k], h_grid, ht)
            sweep[k]["truth"].append(ht)
            sweep[k]["map_mean"].append(a["map_mean"])
            sweep[k]["map_std"].append(a["map_std"])
        line = "  ".join(f"{k}:{analyze(slogs[k], h_grid, ht)['map_mean']:.3f}"
                         for k in estimators)
        print(f"  h_true={ht:.2f}  {line}")

    # slope of MAP vs truth (1 = perfect tracking, 0 = independent)
    for k in estimators:
        t = np.array(sweep[k]["truth"])
        m = np.array(sweep[k]["map_mean"])
        slope = float(np.polyfit(t, m, 1)[0])
        sweep[k]["slope"] = slope
        print(f"  slope(MAP vs truth) [{k:8s}] = {slope:.3f}")

    results["sweep"] = sweep
    (HERE / "coverage_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote coverage_results.json  ({time.time()-t0:.0f}s total)")
    return results, logs, h_grid, sweep, H_TRUE


if __name__ == "__main__":
    main()
