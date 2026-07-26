"""S7 — fix-candidate predictions on the PRODUCTION fallback statistic + compact JSON.

F2' (z-resolved survival): rebuild D(h) and beta_Gbar(h) with the pool-LOCAL
horizon survival S_z(d_L) = P(d_hor >= d_L | z-bin) instead of the pooled
S(d_L) — the 1/d amplitude law is exact within fixed z, so this is the
generator-consistent selection model. Evaluate the REAL production fallback
sum Sigma log B_num (diagnostics) over D_zres -> predicted peak shift.

F3 (generator-consistent mixture denominator, ESTIMATE): the injected mixture is
F*p_cat + (1-F)*(1-fbar)*p_pop, not p_pop (Option-A). Catalogue part of the
selection integral is Sigma_glob(h) (from run logs, exact rate-weighted
catalogue sum). D_gen(h) = A*Sigma_glob_hat(h) + (1-A)*beta_Gbar_hat(h) with
hats normalized at h=0.73 and A = model P(in-catalogue | detected, h=0.73)
estimated generator-side. Report slope change of log D.

Also assembles E1_summary.json from s1..s6 outputs.
"""

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

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

# Sigma_glob(h) (no-BH channel) from run logs
sg_re = re.compile(r"\(with_bh=False\) sum_w_Dg\(h=([0-9.]+)\) = ([0-9.eE+-]+)")
sig_glob: dict[float, float] = {}
for log in RUN.glob("master_thesis_code_*.log"):
    for m in sg_re.finditer(log.read_text()):
        sig_glob[round(float(m.group(1)), 4)] = float(m.group(2))
Sglob = np.array([sig_glob[round(h, 4)] for h in h_grid])

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

pool = pd.concat([pd.read_csv(f) for f in sorted(INJ.glob("injection_h_0p73_task_*.csv"))])
pool_z = pool["z"].to_numpy()
pool_dhor = (pool["SNR"] * pool["luminosity_distance"] / 20.0).to_numpy()
ZB = np.linspace(0, 1.5, 31)
zbc = 0.5 * (ZB[:-1] + ZB[1:])
surv_z = np.zeros((len(zbc), len(dlq)))
for j, (a, b) in enumerate(zip(ZB[:-1], ZB[1:])):
    dh = np.sort(pool_dhor[(pool_z >= a) & (pool_z < b)])
    if len(dh):
        surv_z[j] = 1.0 - np.searchsorted(dh, dlq, side="left") / len(dh)


def pdet_zres(z, d):
    jz = np.clip(np.digitize(z, ZB) - 1, 0, len(zbc) - 1)
    jd = np.clip(np.searchsorted(dlq, d), 0, len(dlq) - 1)
    return surv_z[jz, jd]


u100, w100 = leggauss(100)
D_zres = np.empty(len(h_grid))
bgbar_zres = np.empty(len(h_grid))
for i in range(len(h_grid)):
    z_hi = min(float(np.interp(dl_max, dl_tab[i], zt)), 1.5)
    a, b = 1e-6, z_hi
    zn = (a + b) / 2 + (b - a) / 2 * u100
    dln = np.interp(zn, zt, dl_tab[i])
    fn = np.interp(zn, zt, fbar_tab[i])
    dvcn = np.interp(zn, zt, dvc_tab[i])
    pd_ = pdet_zres(zn, dln)
    core = pd_ * dvcn / (1 + zn)
    D_zres[i] = (b - a) / 2 * np.sum(w100 * core)
    bgbar_zres[i] = (b - a) / 2 * np.sum(w100 * (1 - fn) * core)

# real production fallback sums
det_re = re.compile(r"Detection (\d+): no catalogue hosts")
fb: set[int] = set()
for e in (RUN / "logs").glob("evaluate_*.err.gz"):
    fb |= {int(m.group(1)) for m in det_re.finditer(gzip.open(e, "rt").read())}
diag = pd.read_csv(RUN / "simulations" / "diagnostics" / "event_likelihoods.csv")
dfb = diag[diag["event_idx"].isin(fb)].pivot(index="event_idx", columns="h", values="B_num")
dfb = dfb[np.round(h_grid, 4)]
S_real = np.log(dfb.to_numpy()).sum(axis=0)
N = dfb.shape[0]


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


res = {
    "real_prod_over_D_prod": peak(h_grid, S_real - N * np.log(D_prod)),
    "real_prod_over_D_zres_FIX2": peak(h_grid, S_real - N * np.log(D_zres)),
    "real_prod_over_bgbar_zres": peak(h_grid, S_real - N * np.log(bgbar_zres)),
    "dlogD_prod_dh_073": float(np.gradient(np.log(D_prod), h_grid)[i73]),
    "dlogD_zres_dh_073": float(np.gradient(np.log(D_zres), h_grid)[i73]),
    "dlogbgbar_zres_dh_073": float(np.gradient(np.log(bgbar_zres), h_grid)[i73]),
    "D_zres": D_zres.tolist(),
    "beta_Gbar_zres": bgbar_zres.tolist(),
}

# ---- F3 estimate: generator-consistent mixture denominator --------------------
# F (model in-catalog fraction of the injected population)
p_pop = dvc_tab[i73] / (1 + zt)
F = float(np.trapezoid(fbar_tab[i73] * p_pop, zt) / np.trapezoid(p_pop, zt))
# mean detectabilities at truth (pooled survival for both, adequacy for slope est.)
pdet_tab_iso = np.asarray(
    detprob.detection_probability_without_bh_mass_interpolated_zero_fill(
        dlq, np.zeros_like(dlq), np.zeros_like(dlq), h=0.73
    )
)
pdet_iso = lambda d: np.interp(d, dlq, pdet_tab_iso, left=pdet_tab_iso[0], right=0.0)  # noqa: E731
pd73 = pdet_iso(dl_tab[i73])
mean_pdet_dark = float(
    np.trapezoid((1 - fbar_tab[i73]) * pd73 * p_pop, zt)
    / np.trapezoid((1 - fbar_tab[i73]) * p_pop, zt)
)
# catalogue-channel z density estimate: host-found events / p_true(z)
crb = pd.read_csv(RUN / "fetched_seed1000" / "prepared_cramer_rao_bounds.csv")
snr_ok = crb["SNR"] >= 20.0
rel = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"]) / crb["luminosity_distance"]
kept = crb[snr_ok & (rel < 0.10)]
hf_ids = set(kept.index.astype(int)) - fb
z_hf = np.array([dist_to_redshift(d, h=0.73) for d in kept.loc[sorted(hf_ids), "luminosity_distance"]])
det_frac = np.array(
    [
        (pool.loc[(pool_z >= a) & (pool_z < b), "SNR"] >= 20).mean()
        if ((pool_z >= a) & (pool_z < b)).sum()
        else 0.0
        for a, b in zip(ZB[:-1], ZB[1:])
    ]
)
hist_hf, _ = np.histogram(z_hf, bins=ZB, density=True)
with np.errstate(divide="ignore", invalid="ignore"):
    p_cat_shape = np.where(det_frac > 0.01, hist_hf / det_frac, 0.0)
p_cat_shape /= np.trapezoid(p_cat_shape, zbc)
mean_pdet_cat = float(np.trapezoid(p_cat_shape * np.interp(zbc, zt, pd73), zbc))
A_cat = F * mean_pdet_cat / (F * mean_pdet_cat + (1 - F) * mean_pdet_dark)
Sglob_hat = Sglob / Sglob[i73]
bgbar_hat = bgbar_prod / bgbar_prod[i73]
D_gen_hat = A_cat * Sglob_hat + (1 - A_cat) * bgbar_hat
res["F3_estimate"] = {
    "F_incat_population": F,
    "mean_pdet_cat_est": mean_pdet_cat,
    "mean_pdet_dark": mean_pdet_dark,
    "A_P_cat_given_det": float(A_cat),
    "dlogSglob_dh_073": float(np.gradient(np.log(Sglob), h_grid)[i73]),
    "dlogD_gen_dh_073": float(np.gradient(np.log(D_gen_hat), h_grid)[i73]),
    "dlogD_prod_dh_073": float(np.gradient(np.log(D_prod), h_grid)[i73]),
    "real_prod_over_D_gen": peak(h_grid, S_real - N * np.log(D_gen_hat)),
    "note": "p_cat shape estimated from host-found events / p_true(z); crude, slope-level only",
}

# ---- assemble compact summary --------------------------------------------------
summary = {
    "s1": {k: v for k, v in s1.items() if not isinstance(v, list)},
    "s2": {k: v for k, v in json.load(open(OUT / "s2_mc_results.json")).items() if not isinstance(v, list)},
    "s3_peaks": {
        k: v
        for k, v in json.load(open(OUT / "s3_results.json")).items()
        if "peak" in k or "mean_slope" in k or k.startswith("real_")
    },
    "s4_ladder": json.load(open(OUT / "s4_results.json"))["ladder"],
    "s5": {k: v for k, v in json.load(open(OUT / "s5_results.json")).items() if not isinstance(v, list)},
    "s6": json.load(open(OUT / "s6_results.json")),
    "s7": {k: v for k, v in res.items() if not isinstance(v, list)},
}
with open(OUT / "s7_results.json", "w") as f:
    json.dump(res, f, indent=2)
with open(OUT / "E1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
