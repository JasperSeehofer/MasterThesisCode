"""S3 — attribute the real-vs-MC gap: per-event dlogB_num/dh at fixed z.

Real per-event B_num(h) curves come from the shipped diagnostics
(event_likelihoods.csv, B_num column, 41 h). MC curves come from s2 (rerun of
the generator+estimator). If mean slopes agree at fixed z, the whole bias is
composition (membership); if not, B_num internals differ at fixed z (f_k vs
f_bar / pixel effects / sigma / p_det marginal).

Also V4: MC acceptance with the POOL-LOCAL true detection rate p_true(z)
(fraction of pool injections with SNR>=20 near z) instead of the survival
p_det(d_L) — tests the horizon-trick fidelity (mechanism 4).
"""

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

RUN = Path("/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260719_seed1000_exp40")
OUT = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/completion_bias")
INJ = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/data/injections")

s2 = json.load(open(OUT / "s2_mc_results.json"))
h_grid = np.array(s2["h_grid"])
i73 = int(np.argmin(np.abs(h_grid - 0.73)))

# --- real fallback events -----------------------------------------------------
det_re = re.compile(r"Detection (\d+): no catalogue hosts")
fb: set[int] = set()
for e in (RUN / "logs").glob("evaluate_*.err.gz"):
    fb |= {int(m.group(1)) for m in det_re.finditer(gzip.open(e, "rt").read())}
diag = pd.read_csv(RUN / "simulations" / "diagnostics" / "event_likelihoods.csv")
dfb = diag[diag["event_idx"].isin(fb)].pivot(index="event_idx", columns="h", values="B_num")
dfb = dfb[np.round(h_grid, 4)]
crb = pd.read_csv(RUN / "fetched_seed1000" / "prepared_cramer_rao_bounds.csv")

from master_thesis_code.physical_relations import dist_to_redshift, dist_vectorized  # noqa: E402

dl_obs = crb.loc[dfb.index, "luminosity_distance"].to_numpy()
sig_obs = np.sqrt(
    crb.loc[dfb.index, "delta_luminosity_distance_delta_luminosity_distance"].to_numpy()
)
z_obs = np.array([dist_to_redshift(d, h=0.73) for d in dl_obs])

logB_real = np.log(dfb.to_numpy())  # (N, 41)
slope_real = np.gradient(logB_real, h_grid, axis=1)[:, i73]

# --- MC events (regenerate identically to s2: same seed/pipeline) -------------
# cheaper: recompute via saved per-event arrays? s2 didn't save them; redo the draw
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.stats import norm  # noqa: E402

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build  # noqa: E402
from master_thesis_code.physical_relations import comoving_volume_element  # noqa: E402

comp = from_cache_or_build()
zt = np.linspace(1e-6, 1.5, 901)
fbar73 = np.clip(np.asarray(comp.f_bar(zt, 0.73)), 0, 1)
dl_tab = {i: np.asarray(dist_vectorized(zt, h=float(h))) for i, h in enumerate(h_grid)}
dvc_tab = {i: np.asarray(comoving_volume_element(zt, h=float(h))) for i, h in enumerate(h_grid)}
fbar_tab = {i: np.clip(np.asarray(comp.f_bar(zt, float(h))), 0, 1) for i, h in enumerate(h_grid)}

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

# pool-local true detection rate p_true(z)
pool = pd.concat([pd.read_csv(f) for f in sorted(INJ.glob("injection_h_0p73_task_*.csv"))])
zb = np.linspace(0, 1.5, 61)
det_frac = np.zeros(len(zb) - 1)
for j in range(len(zb) - 1):
    m = (pool["z"] >= zb[j]) & (pool["z"] < zb[j + 1])
    det_frac[j] = (pool.loc[m, "SNR"] >= 20).mean() if m.sum() > 0 else 0.0
zbc = 0.5 * (zb[:-1] + zb[1:])
p_true = lambda z: np.interp(z, zbc, det_frac)  # noqa: E731

rng = np.random.default_rng(20260726)
N_MC = 60000
dens = (1 - fbar73) * dvc_tab[i73] / (1 + zt)
cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(zt))))
cdf /= cdf[-1]
z_true = np.interp(rng.uniform(0, 1, N_MC), cdf, zt)
dl_true = np.interp(z_true, zt, dl_tab[i73])
u_acc = rng.uniform(0, 1, N_MC)
acc_surv = u_acc < pdet(dl_true)      # V0 acceptance (survival model)
acc_true = u_acc < p_true(z_true)     # V4 acceptance (pool-local true rate)

# sigma from real fallback NN in d_L (same as s2)
order = np.argsort(dl_obs)
dl_s, sf_s = dl_obs[order], (sig_obs / dl_obs)[order]
u50, w50 = leggauss(50)


def logB_curves(z_ev, dl_ev, sigf, x):
    logB = np.empty((len(h_grid), len(x)))
    for i in range(len(h_grid)):
        z_lo = np.maximum(np.interp(np.maximum(x * (1 - 4 * sigf), 0), dl_tab[i], zt), 1e-6)
        z_hi = np.minimum(np.interp(x * (1 + 4 * sigf), dl_tab[i], zt), 1.5)
        a, b = z_lo, z_hi
        zn = (a + b)[:, None] / 2 + ((b - a) / 2)[:, None] * u50[None, :]
        dln = np.interp(zn, zt, dl_tab[i])
        dvcn = np.interp(zn, zt, dvc_tab[i])
        fn = np.interp(zn, zt, fbar_tab[i])
        pgw = norm.pdf(dln / x[:, None], loc=1.0, scale=sigf[:, None])
        B = ((b - a) / 2) * np.sum(w50 * (1 - fn) * pgw * dvcn / (1 + zn), axis=1)
        with np.errstate(divide="ignore"):
            logB[i] = np.log(B)
    return logB


def peak(hv, y):
    j = int(np.argmax(y))
    o = {"argmax_h": float(hv[j]), "railed": j in (0, len(hv) - 1)}
    if 0 < j < len(hv) - 1:
        hm, h0, hp = hv[j - 1 : j + 2]
        ym, y0, yp = y[j - 1 : j + 2]
        o["parabolic_h"] = float(h0 - 0.5 * (hp - hm) * (yp - ym) / (2 * (ym - 2 * y0 + yp)))
    return o


res = {}
for name, acc in [("V0", acc_surv), ("V4_pool_true_pdet", acc_true)]:
    z_ev = z_true[acc]
    dl_ev = dl_true[acc]
    nn = np.clip(np.searchsorted(dl_s, dl_ev), 0, len(dl_s) - 1)
    sigf = sf_s[nn]
    x = dl_ev * (1 + sigf * rng.standard_normal(len(dl_ev)))
    lb = logB_curves(z_ev, dl_ev, sigf, x)
    fin = np.isfinite(lb).all(axis=0)
    S = lb[:, fin].sum(axis=1)
    logD = np.log(np.array(s2["D_iso"]))
    res[f"{name}_n"] = int(fin.sum())
    res[f"{name}_peak_over_D"] = peak(h_grid, S - fin.sum() * logD)
    res[f"{name}_peak_over_betaGbar"] = peak(
        h_grid, S - fin.sum() * np.log(np.array(s2["beta_Gbar_iso"]))
    )
    # per-event slope at truth binned by z
    sl = np.gradient(lb[:, fin], h_grid, axis=0)[i73]
    res[f"{name}_mean_slope"] = float(sl.mean())
    if name == "V0":
        mc_z = z_ev[fin]
        mc_slope = sl

# --- binned slope comparison ---------------------------------------------------
bins = np.concatenate((np.linspace(0, 0.9, 19), [1.1, 1.5]))
rows = []
for a, b in zip(bins[:-1], bins[1:]):
    mr = (z_obs >= a) & (z_obs < b)
    mm = (mc_z >= a) & (mc_z < b)
    rows.append(
        {
            "z_lo": float(a),
            "z_hi": float(b),
            "n_real": int(mr.sum()),
            "n_mc": int(mm.sum()),
            "mean_slope_real": float(slope_real[mr].mean()) if mr.sum() else None,
            "mean_slope_mc": float(mc_slope[mm].mean()) if mm.sum() else None,
        }
    )
res["slope_by_z"] = rows
res["mean_slope_real_all"] = float(slope_real.mean())
res["det_frac_pool_by_z"] = {"z": zbc.tolist(), "frac": det_frac.tolist()}
# survival-model p_det evaluated on the d_L=dist(z) curve for comparison
res["pdet_surv_on_dist_curve"] = {"z": zbc.tolist(), "pdet": pdet(np.interp(zbc, zt, dl_tab[i73])).tolist()}

# composition counterfactual: real events, but slopes replaced by MC mean at same z
mc_mean_by_z = {(r["z_lo"], r["z_hi"]): r["mean_slope_mc"] for r in rows}
repl = []
for zi in z_obs:
    for (a, b), v in mc_mean_by_z.items():
        if a <= zi < b and v is not None:
            repl.append(v)
            break
res["real_composition_with_mc_slopes_mean"] = float(np.mean(repl))
res["dlogD_iso_dh_073"] = float(np.gradient(np.log(np.array(s2["D_iso"])), h_grid)[i73])

with open(OUT / "s3_results.json", "w") as f:
    json.dump(res, f, indent=2)
for r in rows:
    print(r)
for k, v in res.items():
    if not isinstance(v, (list, dict)):
        print(k, "=", v)
print("V0 peaks:", res["V0_peak_over_D"], res["V0_peak_over_betaGbar"])
print("V4 peaks:", res["V4_pool_true_pdet_peak_over_D"], res["V4_pool_true_pdet_peak_over_betaGbar"])
