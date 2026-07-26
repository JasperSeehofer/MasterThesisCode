"""S6 — test the oracle-selection numerator: B~(x;h) = INT (1-f) p_det(z,h) L(x|z) dVc/(1+z) dz.

In this pipeline detection is decided by the TRUE-parameter SNR (oracle
selection, independent of the observed x given z), so
    p(x | dark, det, h) = INT L(x|z) p_det(z,h) (1-f) p_pop dz / beta_Gbar(h),
i.e. p_det belongs INSIDE the completion numerator (contrast MFG data-threshold
selection where it cancels). Test in fully self-consistent MC (acceptance ==
survival model), full range and deep subset:

  A  : B  (production, p_det-free) / beta_Gbar   — expect: full-range ~closes,
       deep subset biased low (~0.63, reproducing s5)
  B~ : B~ (p_det inside)          / beta_Gbar    — expect: closes everywhere

Then apply B~ to the REAL fallback events (harness validated vs diagnostics at
z>0.45) and report the predicted shift of the production fallback ensemble.
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
D_prod = np.array(s1["D_h"])
bgbar_prod = np.array(s1["beta_Gbar_h"])
i73 = int(np.argmin(np.abs(h_grid - 0.73)))
Z_C = 0.45
X_C = float(dist(Z_C, h=0.73))

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

u100, w100 = leggauss(100)
u50, w50 = leggauss(50)

# isotropic beta_Gbar and cut beta_sub, both with pooled survival p_det
def beta_tables():
    bg = np.empty(len(h_grid))
    bsub = np.empty(len(h_grid))
    for i in range(len(h_grid)):
        z_hi = min(float(np.interp(dl_max, dl_tab[i], zt)), 1.5)
        for out, z_lo in ((bg, 1e-6), (bsub, float(np.interp(X_C, dl_tab[i], zt)))):
            a, b = z_lo, z_hi
            zn = (a + b) / 2 + (b - a) / 2 * u100
            core = (
                (1 - np.interp(zn, zt, fbar_tab[i]))
                * pdet_pool(np.interp(zn, zt, dl_tab[i]))
                * np.interp(zn, zt, dvc_tab[i])
                / (1 + zn)
            )
            out[i] = (b - a) / 2 * np.sum(w100 * core)
    return bg, bsub


bg_iso, bsub_iso = beta_tables()


def logB_matrix(x, sigf, with_pdet: bool):
    out = np.empty((len(h_grid), len(x)))
    for i in range(len(h_grid)):
        z_lo = np.maximum(np.interp(np.maximum(x * (1 - 4 * sigf), 0), dl_tab[i], zt), 1e-6)
        z_hi = np.minimum(np.interp(x * (1 + 4 * sigf), dl_tab[i], zt), 1.5)
        a, b = z_lo, z_hi
        zn = (a + b)[:, None] / 2 + ((b - a) / 2)[:, None] * u50[None, :]
        dln = np.interp(zn, zt, dl_tab[i])
        dvcn = np.interp(zn, zt, dvc_tab[i])
        fn = np.interp(zn, zt, fbar_tab[i])
        pgw = norm.pdf(dln / x[:, None], loc=1.0, scale=sigf[:, None])
        integ = (1 - fn) * pgw * dvcn / (1 + zn)
        if with_pdet:
            integ = integ * pdet_pool(dln)
        B = ((b - a) / 2) * np.sum(w50 * integ, axis=1)
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


# ---------------- MC, fully self-consistent (acceptance == survival model) -----
det_re = re.compile(r"Detection (\d+): no catalogue hosts")
fb: set[int] = set()
for e in (RUN / "logs").glob("evaluate_*.err.gz"):
    fb |= {int(m.group(1)) for m in det_re.finditer(gzip.open(e, "rt").read())}
crb = pd.read_csv(RUN / "fetched_seed1000" / "prepared_cramer_rao_bounds.csv")
cfb = crb.loc[sorted(fb)]
x_real_all = cfb["luminosity_distance"].to_numpy()
sig_real_all = np.sqrt(cfb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
order = np.argsort(x_real_all)
dl_s, sf_s = x_real_all[order], (sig_real_all / x_real_all)[order]

rng = np.random.default_rng(11)
N_MC = 200000
dens = (1 - fbar_tab[i73]) * dvc_tab[i73] / (1 + zt)
cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(zt))))
cdf /= cdf[-1]
z_d = np.interp(rng.uniform(0, 1, N_MC), cdf, zt)
dl_d = np.interp(z_d, zt, dl_tab[i73])
acc = rng.uniform(0, 1, N_MC) < pdet_pool(dl_d)
z_e, dl_e = z_d[acc], dl_d[acc]
sigf = sf_s[np.clip(np.searchsorted(dl_s, dl_e), 0, len(dl_s) - 1)]
x_e = dl_e * (1 + sigf * rng.standard_normal(len(dl_e)))

res: dict = {}
for tag, wp in [("A_prod_kernel", False), ("Btilde_pdet_inside", True)]:
    lb = logB_matrix(x_e, sigf, with_pdet=wp)
    fin = np.isfinite(lb).all(axis=0)
    S = lb[:, fin].sum(axis=1)
    n = int(fin.sum())
    res[f"{tag}_fullrange_over_betaGbar"] = peak(h_grid, S - n * np.log(bg_iso))
    m = fin & (x_e > X_C)
    Ssub = lb[:, m].sum(axis=1)
    res[f"{tag}_deepsubset_over_betasub"] = peak(h_grid, Ssub - int(m.sum()) * np.log(bsub_iso))
    res[f"{tag}_n_full"] = n
    res[f"{tag}_n_sub"] = int(m.sum())

# ---------------- apply to REAL fallback events --------------------------------
x_r = x_real_all
sf_r = sig_real_all / x_real_all
lbA = logB_matrix(x_r, sf_r, with_pdet=False)
lbB = logB_matrix(x_r, sf_r, with_pdet=True)
finr = np.isfinite(lbA).all(axis=0) & np.isfinite(lbB).all(axis=0)
nr = int(finr.sum())
res["real_n"] = nr
res["real_A_over_Dprod"] = peak(h_grid, lbA[:, finr].sum(axis=1) - nr * np.log(D_prod))
res["real_Btilde_over_Dprod"] = peak(h_grid, lbB[:, finr].sum(axis=1) - nr * np.log(D_prod))
res["real_A_over_bgbarprod"] = peak(h_grid, lbA[:, finr].sum(axis=1) - nr * np.log(bgbar_prod))
res["real_Btilde_over_bgbarprod"] = peak(h_grid, lbB[:, finr].sum(axis=1) - nr * np.log(bgbar_prod))
# deep real subset with conditioned denominators
mr = finr & (x_r > X_C)
res["real_deep_A_over_betasub"] = peak(h_grid, lbA[:, mr].sum(axis=1) - int(mr.sum()) * np.log(bsub_iso))
res["real_deep_Btilde_over_betasub"] = peak(h_grid, lbB[:, mr].sum(axis=1) - int(mr.sum()) * np.log(bsub_iso))
res["real_deep_n"] = int(mr.sum())

with open(OUT / "s6_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
