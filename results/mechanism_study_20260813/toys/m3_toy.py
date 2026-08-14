"""M3 toy: isolate the h-dependent truncation of the UNRENORMALISED z-kernel.

Arm A: domain = [max(z_lo(h), zo-5s), min(z_hi(h), zo+5s)]  (code as written)
Arm B: identical, but z_lo/z_hi built with W_d = 12 sigma_d instead of 4
       (truncation by the GW window made negligible; kernel window unchanged)
Everything else (candidates, z_obs, sigma_z, d_obs, GL order) is identical.
Reported: d/dh [ sum_i (lnL_A - lnL_B) ] near h=0.73, scaled to 982 events,
compared with the slope needed to displace the joint MAP by +0.0372:
   S_need = bias / sigma_post^2 = 0.0372 / 0.004376^2 ~ 1943 per unit h.
"""

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import norm

from darksiren_emri.physical_relations import dist_vectorized
from darksiren_emri.validation import closed_loop_gfrac as cl

rng = np.random.default_rng(12345)
H_TRUE = 0.73
N_EV = 200
K = 400
NQ = 50
SIG_Z = 0.042  # realized sigma_bar_pairs of T-c
KW = 5.0  # _IMPOSTOR_KERNEL_WINDOW
x, w = roots_legendre(NQ)

trip = cl.load_sigma_triples(cl.DEFAULT_CRB_CSV)
sig_d = rng.choice(trip[:, 0], size=N_EV)

# events: z ~ comoving-volume-ish over a plausible detected range
zg = np.linspace(0.05, 1.2, 2000)
wz = zg**2 / (1.0 + zg)
cdf = np.concatenate([[0.0], np.cumsum(0.5 * (wz[1:] + wz[:-1]) * np.diff(zg))])
cdf /= cdf[-1]
z_true = np.interp(rng.random(N_EV), cdf, zg)
d_true = np.asarray(dist_vectorized(z_true, h=H_TRUE))
d_obs = d_true * (1.0 + sig_d * rng.standard_normal(N_EV))

# z<->d table at h_true for drawing candidates inside the (4 sigma) ball window
zt = np.linspace(1e-6, 3.0, 4000)
dt = np.asarray(dist_vectorized(zt, h=H_TRUE))
zlo0 = np.interp(d_obs * (1 - 4 * sig_d), dt, zt)
zhi0 = np.interp(d_obs * (1 + 4 * sig_d), dt, zt)
u = rng.random((N_EV, K))
z_cand = zlo0[:, None] + (zhi0 - zlo0)[:, None] * u  # true member z (uniform in window)
z_obs = z_cand + SIG_Z * rng.standard_normal((N_EV, K))  # zero-mean sigma_z scatter
so = np.full((N_EV, K), SIG_Z)


def lnL(h, Wd):
    d_n = np.asarray(dist_vectorized(zt, h=h))
    zlo = np.maximum(np.interp(d_obs * (1 - Wd * sig_d), d_n, zt), 1e-6)
    zhi = np.minimum(np.interp(d_obs * (1 + Wd * sig_d), d_n, zt), zt[-1])
    a = np.maximum(zlo[:, None], z_obs - KW * so)
    b = np.minimum(zhi[:, None], z_obs + KW * so)
    valid = b > a
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    zn = mid[..., None] + half[..., None] * x
    dn = np.asarray(dist_vectorized(np.maximum(zn.reshape(-1), 1e-8), h=h)).reshape(zn.shape)
    frac = dn / d_obs[:, None, None]
    pgw = norm.pdf(frac, loc=1.0, scale=sig_d[:, None, None])
    kern = norm.pdf(zn, loc=z_obs[..., None], scale=so[..., None])
    c = half * ((kern * pgw) @ w)
    c = np.where(valid, c, 0.0)
    L = c.sum(axis=1) / K
    return np.where(L > 0, np.log(np.where(L > 0, L, 1.0)), -745.0)


dh = 0.005
hs = [H_TRUE - dh, H_TRUE + dh]
A = [lnL(h, 4.0) for h in hs]
B = [lnL(h, 12.0) for h in hs]
diff = [A[i] - B[i] for i in (0, 1)]
slope_per_ev = (diff[1] - diff[0]).sum() / (2 * dh) / N_EV
S = slope_per_ev * 982.0
S_need = 0.037237 / 0.004376**2
print(f"mean |lnL_A - lnL_B| per event at h-: {np.abs(diff[0]).mean():.3e}")
print(f"max  |lnL_A - lnL_B| per event      : {np.abs(diff[0]).max():.3e}")
print(f"M3 slope per event  d(dlnL)/dh      : {slope_per_ev:.4e}")
print(f"M3 stacked slope (982 ev)           : {S:.4e} per unit h")
print(f"needed slope for +0.0372 MAP shift  : {S_need:.4e}")
print(f"ratio M3/needed                     : {S / S_need:.3e}")
print(f"implied MAP shift from M3 alone     : {S * 0.004376**2:+.3e} in h")
