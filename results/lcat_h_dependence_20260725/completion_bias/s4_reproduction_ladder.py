"""S4 — full in-silico reproduction ladder for the fallback-ensemble peak.

Ladder (all ensemble peaks on the 41-h grid, production sky-aware D(h) from logs):
  L1: self-consistent dark channel, conditioned denominator beta_Gbar  -> closure
  L2: same events, production statistic B/D                            -> subset term
  L3: composition pinned to OBSERVED fallback z-profile, B(f_bar)/D    -> membership
  L4: L3 + ZoA-flat completeness (f=0) with the empirically measured
      per-z-bin ZoA fraction of real fallback events                    -> pixel term
  Real: 0.612

Also: decompose the composition mismatch into selection-model error
(V0 survival-p_det composition vs V4 pool-true composition) and ball-membership
(V4 composition vs observed), and check total detected z-hist vs model
p_pop * p_true (generator/analysis population consistency, mechanism 1).
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
    dist_to_redshift,
    dist_vectorized,
)

RUN = Path("/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260719_seed1000_exp40")
OUT = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/completion_bias")
INJ = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/data/injections")

s1 = json.load(open(OUT / "s1_results.json"))
h_grid = np.array(s1["h_grid"])
D_prod = np.array(s1["D_h"])
bgbar_prod = np.array(s1["beta_Gbar_h"])
i73 = int(np.argmin(np.abs(h_grid - 0.73)))

# ---------- real fallback events ----------
det_re = re.compile(r"Detection (\d+): no catalogue hosts")
fb: set[int] = set()
for e in (RUN / "logs").glob("evaluate_*.err.gz"):
    fb |= {int(m.group(1)) for m in det_re.finditer(gzip.open(e, "rt").read())}
crb = pd.read_csv(RUN / "fetched_seed1000" / "prepared_cramer_rao_bounds.csv")
cfb = crb.loc[sorted(fb)]
dl_obs = cfb["luminosity_distance"].to_numpy()
sig_obs = np.sqrt(cfb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
z_obs = np.array([dist_to_redshift(d, h=0.73) for d in dl_obs])

# real ZoA fraction per z bin (f_k < 0.01 at event pixel)
comp = from_cache_or_build()
fk_real = np.array(
    [
        float(comp.f_k(zz, comp.ang2pix(p, t), 0.73))
        for zz, p, t in zip(z_obs, cfb["phiS"], cfb["qS"])
    ]
)

# all detected events (for mechanism-1 check)
snr_ok = crb["SNR"] >= 20.0
rel = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"]) / crb["luminosity_distance"]
kept = crb[snr_ok & (rel < 0.10)]
z_all = np.array([dist_to_redshift(d, h=0.73) for d in kept["luminosity_distance"]])

# ---------- tables ----------
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
pdet = lambda d: np.interp(d, dlq, pdet_tab, left=pdet_tab[0], right=0.0)  # noqa: E731

pool = pd.concat([pd.read_csv(f) for f in sorted(INJ.glob("injection_h_0p73_task_*.csv"))])
zb = np.linspace(0, 1.5, 61)
zbc = 0.5 * (zb[:-1] + zb[1:])
det_frac = np.array(
    [
        (pool.loc[(pool["z"] >= a) & (pool["z"] < b), "SNR"] >= 20).mean()
        if ((pool["z"] >= a) & (pool["z"] < b)).sum()
        else 0.0
        for a, b in zip(zb[:-1], zb[1:])
    ]
)
p_true = lambda z: np.interp(z, zbc, det_frac)  # noqa: E731

# ---------- MC draws ----------
rng = np.random.default_rng(20260726)
N_MC = 120000
dens = (1 - fbar_tab[i73]) * dvc_tab[i73] / (1 + zt)
cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(zt))))
cdf /= cdf[-1]
z_dark = np.interp(rng.uniform(0, 1, N_MC), cdf, zt)
dl_dark = np.interp(z_dark, zt, dl_tab[i73])
u = rng.uniform(0, 1, N_MC)
acc_V0 = u < pdet(dl_dark)
acc_V4 = u < p_true(z_dark)

order = np.argsort(dl_obs)
dl_s, sf_s = dl_obs[order], (sig_obs / dl_obs)[order]
u50, w50 = leggauss(50)


def logB(x, sigf, zoA_mask=None):
    """(n_h, N) log B_num; zoA_mask=True rows use f=0 (ZoA pixel)."""
    out = np.empty((len(h_grid), len(x)))
    for i in range(len(h_grid)):
        z_lo = np.maximum(np.interp(np.maximum(x * (1 - 4 * sigf), 0), dl_tab[i], zt), 1e-6)
        z_hi = np.minimum(np.interp(x * (1 + 4 * sigf), dl_tab[i], zt), 1.5)
        a, b = z_lo, z_hi
        zn = (a + b)[:, None] / 2 + ((b - a) / 2)[:, None] * u50[None, :]
        dln = np.interp(zn, zt, dl_tab[i])
        dvcn = np.interp(zn, zt, dvc_tab[i])
        fn = np.interp(zn, zt, fbar_tab[i])
        if zoA_mask is not None:
            fn = np.where(zoA_mask[:, None], 0.0, fn)
        pgw = norm.pdf(dln / x[:, None], loc=1.0, scale=sigf[:, None])
        B = ((b - a) / 2) * np.sum(w50 * (1 - fn) * pgw * dvcn / (1 + zn), axis=1)
        with np.errstate(divide="ignore"):
            out[i] = np.log(B)
    return out


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


res: dict = {}
bins = np.concatenate((np.linspace(0, 0.9, 19), [1.1, 1.5]))
h_obs_hist, _ = np.histogram(z_obs, bins=bins, density=True)

# empirical ZoA fraction per bin from real fallback events
zoA_frac = np.zeros(len(bins) - 1)
for j, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
    m = (z_obs >= a) & (z_obs < b)
    zoA_frac[j] = float(np.mean(fk_real[m] < 0.01)) if m.sum() else 1.0
res["zoA_frac_by_bin"] = zoA_frac.tolist()

ladder = {}
for name, acc in [("V0", acc_V0), ("V4", acc_V4)]:
    z_ev = z_dark[acc]
    dl_ev = dl_dark[acc]
    nn = np.clip(np.searchsorted(dl_s, dl_ev), 0, len(dl_s) - 1)
    sigf = sf_s[nn]
    x = dl_ev * (1 + sigf * rng.standard_normal(len(dl_ev)))
    lb = logB(x, sigf)
    fin = np.isfinite(lb).all(axis=0)
    n = int(fin.sum())
    S = lb[:, fin].sum(axis=1)
    ladder[f"{name}_over_betaGbar_prod"] = peak(h_grid, S - n * np.log(bgbar_prod))
    ladder[f"{name}_over_D_prod"] = peak(h_grid, S - n * np.log(D_prod))
    hist, _ = np.histogram(z_ev[fin], bins=bins, density=True)
    res[f"{name}_z_hist"] = hist.tolist()

    # composition reweight to observed profile
    ratio = np.where(hist > 0, h_obs_hist / np.maximum(hist, 1e-300), 0.0)
    w = ratio[np.clip(np.digitize(z_ev[fin], bins) - 1, 0, len(ratio) - 1)]
    w *= n / w.sum()
    S3 = (lb[:, fin] * w[None, :]).sum(axis=1)
    ladder[f"{name}_obsreweight_over_D_prod"] = peak(h_grid, S3 - n * np.log(D_prod))

    # + ZoA-flat completeness with empirical per-bin ZoA fraction
    binidx = np.clip(np.digitize(z_ev[fin], bins) - 1, 0, len(bins) - 2)
    is_zoa = rng.uniform(0, 1, n) < zoA_frac[binidx]
    lb_zoa = logB(x[fin], sigf[fin], zoA_mask=is_zoa)
    fin2 = np.isfinite(lb_zoa).all(axis=0)
    S4 = (lb_zoa[:, fin2] * w[fin2][None, :]).sum(axis=1) * (w[fin2].sum() / w[fin2].sum())
    ladder[f"{name}_obsreweight_zoa_over_D_prod"] = peak(h_grid, S4 - int(fin2.sum()) * 0 - w[fin2].sum() * np.log(D_prod))
    res[f"{name}_n"] = n
res["ladder"] = ladder
res["obs_z_hist"] = h_obs_hist.tolist()
res["bins"] = bins.tolist()

# selection-model error curves
res["p_true_by_z"] = {"z": zbc.tolist(), "frac": det_frac.tolist()}
res["p_surv_on_dist"] = pdet(np.interp(zbc, zt, dl_tab[i73])).tolist()

# mechanism-1 check: all detected events vs model p_pop * p_true
model_all = dvc_tab[i73] / (1 + zt) * p_true(zt)
model_all /= np.trapezoid(model_all, zt)
hist_all, _ = np.histogram(z_all, bins=bins, density=True)
model_binned = [
    float(np.trapezoid(model_all[(zt >= a) & (zt < b)], zt[(zt >= a) & (zt < b)]) / (b - a))
    for a, b in zip(bins[:-1], bins[1:])
]
res["all_detected_z_hist"] = hist_all.tolist()
res["model_pop_ptrue_binned"] = model_binned

with open(OUT / "s4_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(ladder, indent=2))
print("real peak (from s1):", s1["peak_sum_log_BnumOverD"])
