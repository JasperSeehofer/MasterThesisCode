"""M5 toy: minimal faithful mirror of the venue-transfer ball estimator.

Isolates the sigma_z-dosed MAP displacement in a standalone model:
 - real fiducial LCDM d_L(z,h) = D(z)/h  (spline table)
 - w_pop(z,h) = (dV_c/dz)/(1+z) population, hard d_L horizon -> alpha(h)
 - +-4 sigma_d ball window in d_L, impostors ~ w_pop | W
 - estimator: (1/K) sum_k int N(z;z_obs_k,sigma_k) N(D(z)/(h d_obs);1,sigma_d) dz
              clipped to [max(z_lo(h), z_obs-5s), min(z_hi(h), z_obs+5s)], GL-50
              minus N ln alpha(h)
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

from darksiren_emri.physical_relations import comoving_volume_element, dist_vectorized

H_TRUE = 0.73
DMAX = 6.0  # Gpc horizon (fixed detector-frame cut)
SIGMA_WINDOW = 4.0
KERN_WINDOW = 5.0
NQ = 50
_x, _w = leggauss(NQ)

# --- D(z) at h=1 and its inverse -------------------------------------------
_ZT = np.linspace(0.0, 3.0, 40001)
_DT = np.asarray(dist_vectorized(np.maximum(_ZT, 1e-10), h=1.0), dtype=float)


def D(z):
    return np.interp(z, _ZT, _DT)


def Zof(d):
    return np.interp(d, _DT, _ZT)


# v(z) = w_pop(z, h=1) shape; w_pop(z,h) = v(z)/h^3
_VT = np.asarray(comoving_volume_element(np.maximum(_ZT, 1e-10), h=1.0), dtype=float) / (1.0 + _ZT)
_VC = np.concatenate([[0.0], np.cumsum(0.5 * (_VT[1:] + _VT[:-1]) * np.diff(_ZT))])


def Vcdf(z):
    return np.interp(z, _ZT, _VC)


def Vinv(c):
    return np.interp(c, _VC, _ZT)


def v_of(z):
    return np.interp(z, _ZT, _VT)


def log_alpha(h):
    zmax = Zof(h * DMAX)
    return np.log(Vcdf(zmax)) - 3.0 * np.log(h)  # up to h-independent const


SIG_D_POOL = None


def load_sigd():
    global SIG_D_POOL
    import pandas as pd

    df = pd.read_csv("results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv")
    d = df["luminosity_distance"].values
    SIG_D_POOL = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].values) / d


def draw(seed, n_ev, K, sigma_z, scatter=True):
    rng = np.random.default_rng(seed)
    zmax = Zof(H_TRUE * DMAX)
    z_true = Vinv(rng.random(n_ev) * Vcdf(zmax))
    sig_d = SIG_D_POOL[rng.integers(0, SIG_D_POOL.size, n_ev)]
    d_true = D(z_true) / H_TRUE
    d_obs = d_true * (1.0 + sig_d * rng.standard_normal(n_ev))
    z_lo = Zof(np.maximum(d_obs * (1 - SIGMA_WINDOW * sig_d), 0.0) * H_TRUE)
    z_hi = Zof(d_obs * (1 + SIGMA_WINDOW * sig_d) * H_TRUE)
    F_lo, F_hi = Vcdf(z_lo), Vcdf(z_hi)
    n_imp = max(K - 1, 0)
    ev = np.repeat(np.arange(n_ev), n_imp)
    u = F_lo[ev] + (F_hi[ev] - F_lo[ev]) * rng.random(ev.size)
    z_imp = Vinv(u)
    z_cand = np.concatenate([z_true, z_imp])
    ev_all = np.concatenate([np.arange(n_ev), ev])
    z_obs = z_cand.copy()
    if sigma_z > 0 and scatter:
        z_obs = z_cand + sigma_z * rng.standard_normal(z_cand.size)
    return dict(
        n_ev=n_ev,
        z_true=z_true,
        sig_d=sig_d,
        d_obs=d_obs,
        ev=ev_all,
        z_obs=z_obs,
        z_cand=z_cand,
        K=np.bincount(ev_all, minlength=n_ev),
    )


def lnpost(R, h_grid, sigma_z, *, truncate=True, weights=None, point_kernel=False):
    ev, z_obs, n_ev = R["ev"], R["z_obs"], R["n_ev"]
    d_obs_p = R["d_obs"][ev]
    sig_p = R["sig_d"][ev]
    K = np.maximum(R["K"], 1).astype(float)
    if weights is None:
        wk = np.ones(ev.size)
    else:
        wk = weights
    Wsum = np.bincount(ev, weights=wk, minlength=n_ev)
    out = np.empty(len(h_grid))
    for j, h in enumerate(h_grid):
        z_lo = Zof(d_obs_p * (1 - SIGMA_WINDOW * sig_p) * h)
        z_hi = Zof(d_obs_p * (1 + SIGMA_WINDOW * sig_p) * h)
        z_lo = np.maximum(z_lo, 1e-6)
        if sigma_z > 0 and not point_kernel:
            a = z_obs - KERN_WINDOW * sigma_z
            b = z_obs + KERN_WINDOW * sigma_z
            if truncate:
                a = np.maximum(z_lo, a)
                b = np.minimum(z_hi, b)
            valid = b > a
            half = 0.5 * (b - a)
            mid = 0.5 * (b + a)
            zn = mid[:, None] + half[:, None] * _x[None, :]
            frac = (D(np.maximum(zn, 1e-8)) / h) / d_obs_p[:, None]
            integ = norm.pdf(zn, loc=z_obs[:, None], scale=sigma_z) * norm.pdf(
                frac, loc=1.0, scale=sig_p[:, None]
            )
            c = np.where(valid, half * (integ @ _w), 0.0)
        else:
            valid = (z_obs >= z_lo) & (z_obs <= z_hi) if truncate else np.ones(ev.size, bool)
            frac = (D(np.maximum(z_obs, 1e-8)) / h) / d_obs_p
            c = np.where(valid, norm.pdf(frac, loc=1.0, scale=sig_p), 0.0)
        L = np.bincount(ev, weights=wk * c, minlength=n_ev) / Wsum
        ok = (L > 0) & np.isfinite(L)
        out[j] = np.sum(np.where(ok, np.log(np.where(ok, L, 1.0)), -745.0)) - n_ev * log_alpha(h)
    return out


def argmax_refined(h_grid, ln):
    i = int(np.argmax(ln))
    if 0 < i < len(h_grid) - 1:
        y0, y1, y2 = ln[i - 1], ln[i], ln[i + 1]
        dh = h_grid[1] - h_grid[0]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return h_grid[i] - 0.5 * dh * (y2 - y0) / den
    return h_grid[i]
