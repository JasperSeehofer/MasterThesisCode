"""S5 — membership-clean REAL-data closure test + z-resolved survival (FIX-B).

Subset: real fallback events with observed d_L above x_c = dist(z_c=0.45, 0.73).
There membership is ~complete (fallback fraction ~1 for dark events, catalogue
empty, f~0), so the subset selection is a pure cut in the observable x.
The correctly conditioned subset statistic is

    Sigma_i log B_num_i(h)  -  N * log beta_sub(h),
    beta_sub(h) = INT_{z(x_c,h)}^{z_max(h)} (1-f_bar) p_det dVc/(1+z) dz

(the model probability mass of the dark-detected channel in the cut region;
kernel smearing across the boundary is symmetric and second order).

Two p_det models inside beta_sub:
  (a) pooled survival S(d_L)  (production model)         -> residual = estimator error
  (b) z-resolved survival S_z(d_L) = P(d_hor >= d_L | z near) (FIX-B candidate)

MC calibration: synthetic dark events (true rule = actual SNR>=20 of pool rows
resampled locally in z) under the same cut and the same two statistics.
"""

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist,
    dist_to_redshift,
    dist_vectorized,
)

RUN = Path("/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260719_seed1000_exp40")
OUT = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/completion_bias")
INJ = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/data/injections")

s1 = json.load(open(OUT / "s1_results.json"))
h_grid = np.array(s1["h_grid"])
i73 = int(np.argmin(np.abs(h_grid - 0.73)))
Z_C = 0.45
X_C = float(dist(Z_C, h=0.73))

# ---------------- tables ----------------
comp = from_cache_or_build()
zt = np.linspace(1e-6, 1.5, 901)
dl_tab = np.array([np.asarray(dist_vectorized(zt, h=float(h))) for h in h_grid])
dvc_tab = np.array([np.asarray(comoving_volume_element(zt, h=float(h))) for h in h_grid])
fbar_tab = np.array([np.clip(np.asarray(comp.f_bar(zt, float(h))), 0, 1) for h in h_grid])

detprob = SimulationDetectionProbability(
    injection_data_dir=str(INJ), snr_threshold=20.0, dl_bins=60, mass_bins=40,
    estimator="local_linear", expected_z_max=1.5,
)
dl_max = float(detprob.get_dl_max(0.73))
dlq = np.linspace(1e-4, dl_max * 1.001, 4000)
pdet_tab = np.asarray(
    detprob.detection_probability_without_bh_mass_interpolated_zero_fill(
        dlq, np.zeros_like(dlq), np.zeros_like(dlq), h=0.73
    )
)
pdet_pool = lambda d: np.interp(d, dlq, pdet_tab, left=pdet_tab[0], right=0.0)  # noqa: E731

# z-resolved survival S_z(d): P(d_hor >= d | injections with z near)
pool = pd.concat([pd.read_csv(f) for f in sorted(INJ.glob("injection_h_0p73_task_*.csv"))])
pool_z = pool["z"].to_numpy()
pool_dhor = (pool["SNR"] * pool["luminosity_distance"] / 20.0).to_numpy()
ZB = np.linspace(0, 1.5, 31)  # 0.05-wide z bins
zbc = 0.5 * (ZB[:-1] + ZB[1:])
surv_z = np.zeros((len(zbc), len(dlq)))
for j, (a, b) in enumerate(zip(ZB[:-1], ZB[1:])):
    dh = np.sort(pool_dhor[(pool_z >= a) & (pool_z < b)])
    if len(dh):
        surv_z[j] = 1.0 - np.searchsorted(dh, dlq, side="left") / len(dh)


def pdet_zres(z: np.ndarray, d: np.ndarray) -> np.ndarray:
    jz = np.clip(np.digitize(z, ZB) - 1, 0, len(zbc) - 1)
    jd = np.clip(np.searchsorted(dlq, d), 0, len(dlq) - 1)
    return surv_z[jz, jd]


# ---------------- conditioned denominators ----------------
u200, w200 = leggauss(200)
def beta_sub(pdet_kind: str) -> np.ndarray:
    out = np.empty(len(h_grid))
    for i in range(len(h_grid)):
        z_lo = float(np.interp(X_C, dl_tab[i], zt))
        z_hi = min(float(np.interp(dl_max, dl_tab[i], zt)), 1.5)
        a, b = z_lo, z_hi
        zn = (a + b) / 2 + (b - a) / 2 * u200
        dln = np.interp(zn, zt, dl_tab[i])
        fn = np.interp(zn, zt, fbar_tab[i])
        dvcn = np.interp(zn, zt, dvc_tab[i])
        pd_ = pdet_pool(dln) if pdet_kind == "pool" else pdet_zres(zn, dln)
        out[i] = (b - a) / 2 * np.sum(w200 * (1 - fn) * pd_ * dvcn / (1 + zn))
    return out


beta_sub_pool = beta_sub("pool")
beta_sub_zres = beta_sub("zres")


def peak(hv, y):
    j = int(np.argmax(y))
    o = {"argmax_h": float(hv[j]), "railed": bool(j in (0, len(hv) - 1))}
    if 0 < j < len(hv) - 1:
        hm, h0, hp = hv[j - 1 : j + 2]
        ym, y0, yp = y[j - 1 : j + 2]
        d = ym - 2 * y0 + yp
        o["parabolic_h"] = float(h0 - 0.5 * (hp - hm) * (yp - ym) / (2 * d))
        d2 = 2 * (ym / ((h0 - hm) * (hp - hm)) - y0 / ((hp - h0) * (h0 - hm)) + yp / ((hp - h0) * (hp - hm)))
        o["sigma"] = float(np.sqrt(-1.0 / d2)) if d2 < 0 else None
    return o


# ---------------- real subset ----------------
det_re = re.compile(r"Detection (\d+): no catalogue hosts")
fb: set[int] = set()
for e in (RUN / "logs").glob("evaluate_*.err.gz"):
    fb |= {int(m.group(1)) for m in det_re.finditer(gzip.open(e, "rt").read())}
crb = pd.read_csv(RUN / "fetched_seed1000" / "prepared_cramer_rao_bounds.csv")
cfb = crb.loc[sorted(fb)]
x_real = cfb["luminosity_distance"].to_numpy()
sub_mask = x_real > X_C
sub_ids = cfb.index[sub_mask]
diag = pd.read_csv(RUN / "simulations" / "diagnostics" / "event_likelihoods.csv")
dsub = diag[diag["event_idx"].isin(set(sub_ids))].pivot(index="event_idx", columns="h", values="B_num")
dsub = dsub[np.round(h_grid, 4)]
S_real = np.log(dsub.to_numpy()).sum(axis=0)
n_real = dsub.shape[0]

res = {
    "z_c": Z_C,
    "x_c_gpc": X_C,
    "n_real_subset": int(n_real),
    "real_over_beta_sub_pool": peak(h_grid, S_real - n_real * np.log(beta_sub_pool)),
    "real_over_beta_sub_zres": peak(h_grid, S_real - n_real * np.log(beta_sub_zres)),
}

# ---------------- MC calibration ----------------
rng = np.random.default_rng(7)
N_MC = 200000
dens = (1 - fbar_tab[i73]) * dvc_tab[i73] / (1 + zt)
cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(zt))))
cdf /= cdf[-1]
z_d = np.interp(rng.uniform(0, 1, N_MC), cdf, zt)
dl_d = np.interp(z_d, zt, dl_tab[i73])
# true rule: accept with the pool's LOCAL detected fraction (empirical SNR>=20 rate)
det_frac = np.array(
    [
        (pool.loc[(pool_z >= a) & (pool_z < b), "SNR"] >= 20).mean()
        if ((pool_z >= a) & (pool_z < b)).sum()
        else 0.0
        for a, b in zip(ZB[:-1], ZB[1:])
    ]
)
p_true = lambda z: np.interp(z, zbc, det_frac)  # noqa: E731
acc = rng.uniform(0, 1, N_MC) < p_true(z_d)
z_e, dl_e = z_d[acc], dl_d[acc]
order = np.argsort(x_real)
sf_pool_real = (np.sqrt(cfb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy()) / x_real)[order]
dl_sorted = x_real[order]
sigf = sf_pool_real[np.clip(np.searchsorted(dl_sorted, dl_e), 0, len(dl_sorted) - 1)]
x_e = dl_e * (1 + sigf * rng.standard_normal(len(dl_e)))
m = x_e > X_C
z_e, x_e, sigf = z_e[m], x_e[m], sigf[m]

u50, w50 = leggauss(50)
lb = np.empty((len(h_grid), len(x_e)))
for i in range(len(h_grid)):
    z_lo = np.maximum(np.interp(np.maximum(x_e * (1 - 4 * sigf), 0), dl_tab[i], zt), 1e-6)
    z_hi = np.minimum(np.interp(x_e * (1 + 4 * sigf), dl_tab[i], zt), 1.5)
    a, b = z_lo, z_hi
    zn = (a + b)[:, None] / 2 + ((b - a) / 2)[:, None] * u50[None, :]
    dln = np.interp(zn, zt, dl_tab[i])
    dvcn = np.interp(zn, zt, dvc_tab[i])
    fn = np.interp(zn, zt, fbar_tab[i])
    pgw = norm.pdf(dln / x_e[:, None], loc=1.0, scale=sigf[:, None])
    B = ((b - a) / 2) * np.sum(w50 * (1 - fn) * pgw * dvcn / (1 + zn), axis=1)
    with np.errstate(divide="ignore"):
        lb[i] = np.log(B)
fin = np.isfinite(lb).all(axis=0)
S_mc = lb[:, fin].sum(axis=1)
n_mc = int(fin.sum())
res["n_mc_subset"] = n_mc
res["mc_truerule_over_beta_sub_pool"] = peak(h_grid, S_mc - n_mc * np.log(beta_sub_pool))
res["mc_truerule_over_beta_sub_zres"] = peak(h_grid, S_mc - n_mc * np.log(beta_sub_zres))
res["beta_sub_pool"] = beta_sub_pool.tolist()
res["beta_sub_zres"] = beta_sub_zres.tolist()

with open(OUT / "s5_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
