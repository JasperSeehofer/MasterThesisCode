"""S2 — Direct Monte-Carlo self-consistency test of the fallback estimator (mechanism #6).

Draw synthetic dark events from the generator's own density
    z ~ (1 - f_bar(z, 0.73)) * dVc/dz / (1+z),  z <= 1.5
(dark_siren_injection._draw_dark_redshifts), detect with the estimator's own
selection model p_det(d_L) (survival grid from the SAME injection pool), scatter
the observed d_L with the per-event Fisher sigma (matched to the real fallback
events' sigma_frac|d_L), then apply the SAME estimator p_i(h) = B_num(h)/D(h)
(isotropic-sky variant of p_Di lines ~2180-2278, copied verbatim in structure:
4-sigma window, z_upper cap 1.5, fixed_quad n=50, Gaussian kernel in
d_L_fraction, (1-f) dVc/(1+z) weight) over the same 41-h grid.

Variants:
  V0  : scattered observation (matches production prepare step)  -> core test
  V0n : noiseless observation                                    -> sigma-kernel term
  V3  : V0 reweighted to the OBSERVED fallback z-profile          -> membership term

Also: D_iso(h)/beta_Gbar_iso(h) tables (fixed_quad n=100, same as pipeline) and
comparison of their log-slopes with the production sky-aware D(h) from run logs.

Everything is read-only w.r.t. master_thesis_code (imports only).
"""

import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: E402
    SimulationDetectionProbability,
)
from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build  # noqa: E402
from master_thesis_code.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist_vectorized,
)

RUN = Path("/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs/run_20260719_seed1000_exp40")
OUT = Path("/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/completion_bias")
INJ = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725/data/injections"

H_TRUE = 0.73
Z_MAX = 1.5
N_MC = 60000  # pre-selection draws
SEED = 20260726
rng = np.random.default_rng(SEED)

s1 = json.load(open(OUT / "s1_results.json"))
h_grid = np.array(s1["h_grid"])
D_prod = np.array(s1["D_h"])
bgbar_prod = np.array(s1["beta_Gbar_h"])

# ---------------------------------------------------------------- tables
print("building completeness...", flush=True)
comp = from_cache_or_build()
zt = np.linspace(1e-6, Z_MAX, 901)
t0 = time.time()
fbar_tab = np.empty((len(h_grid), len(zt)))
for i, h in enumerate(h_grid):
    fbar_tab[i] = np.clip(np.asarray(comp.f_bar(zt, float(h))), 0.0, 1.0)
print(f"f_bar tables {time.time()-t0:.1f}s", flush=True)

dl_tab = np.empty((len(h_grid), len(zt)))
dvc_tab = np.empty((len(h_grid), len(zt)))
for i, h in enumerate(h_grid):
    dl_tab[i] = np.asarray(dist_vectorized(zt, h=float(h)), dtype=np.float64)
    dvc_tab[i] = np.asarray(comoving_volume_element(zt, h=float(h)), dtype=np.float64)

print("building detection probability...", flush=True)
t0 = time.time()
detprob = SimulationDetectionProbability(
    injection_data_dir=INJ,
    snr_threshold=20.0,
    dl_bins=60,
    mass_bins=40,
    estimator="local_linear",
    expected_z_max=1.5,
)
dl_max = float(detprob.get_dl_max(H_TRUE))
dlq = np.linspace(1e-4, dl_max * 1.001, 4000)
pdet_tab = np.asarray(
    detprob.detection_probability_without_bh_mass_interpolated_zero_fill(
        dlq, np.zeros_like(dlq), np.zeros_like(dlq), h=H_TRUE
    ),
    dtype=np.float64,
)
print(f"detprob {time.time()-t0:.1f}s, dl_max={dl_max:.4f}", flush=True)


def pdet(d: np.ndarray) -> np.ndarray:
    return np.interp(d, dlq, pdet_tab, left=pdet_tab[0], right=0.0)


# ---------------------------------------------------------------- D(h), beta_Gbar(h) isotropic
from numpy.polynomial.legendre import leggauss  # noqa: E402

u100, w100 = leggauss(100)
D_iso = np.empty(len(h_grid))
bgbar_iso = np.empty(len(h_grid))
zmax_h = np.empty(len(h_grid))
for i, h in enumerate(h_grid):
    z_max_h = float(np.interp(dl_max, dl_tab[i], zt))  # dist_to_redshift(dl_max, h)
    zmax_h[i] = z_max_h
    a, b = 1e-6, min(z_max_h, Z_MAX)
    zn = (a + b) / 2 + (b - a) / 2 * u100
    dln = np.interp(zn, zt, dl_tab[i])
    dvcn = np.interp(zn, zt, dvc_tab[i])
    fn = np.interp(zn, zt, fbar_tab[i])
    pd_n = pdet(dln)
    core = pd_n * dvcn / (1.0 + zn)
    D_iso[i] = (b - a) / 2 * np.sum(w100 * core)
    bgbar_iso[i] = (b - a) / 2 * np.sum(w100 * (1.0 - fn) * core)

# ---------------------------------------------------------------- real fallback sigma pool
det_re = re.compile(r"Detection (\d+): no catalogue hosts")
fb: set[int] = set()
for e in (RUN / "logs").glob("evaluate_*.err.gz"):
    fb |= {int(m.group(1)) for m in det_re.finditer(gzip.open(e, "rt").read())}
crb = pd.read_csv(RUN / "fetched_seed1000" / "prepared_cramer_rao_bounds.csv")
crb_fb = crb.loc[sorted(fb)]
dl_fb = crb_fb["luminosity_distance"].to_numpy()
sig_fb = np.sqrt(crb_fb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
sigfrac_fb = sig_fb / dl_fb
order = np.argsort(dl_fb)
dl_fb_s, sigfrac_fb_s = dl_fb[order], sigfrac_fb[order]

# ---------------------------------------------------------------- MC generation at h=0.73
i73 = int(np.argmin(np.abs(h_grid - H_TRUE)))
dens = (1.0 - fbar_tab[i73]) * dvc_tab[i73] / (1.0 + zt)
cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(zt))))
cdf /= cdf[-1]
z_true = np.interp(rng.uniform(0, 1, N_MC), cdf, zt)
dl_true = np.interp(z_true, zt, dl_tab[i73])
accept = rng.uniform(0, 1, N_MC) < pdet(dl_true)
z_ev = z_true[accept]
dl_ev = dl_true[accept]
n_ev = len(z_ev)
print(f"MC: {n_ev} detected dark events of {N_MC}", flush=True)

# sigma_frac: nearest neighbour in d_L from the real fallback events
idx_nn = np.clip(np.searchsorted(dl_fb_s, dl_ev), 0, len(dl_fb_s) - 1)
sigf_ev = sigfrac_fb_s[idx_nn]
# variants of the observation
eps = rng.standard_normal(n_ev)
x_scat = dl_ev * (1.0 + sigf_ev * eps)
x_noiseless = dl_ev.copy()

# keep only positive observations (guard; sigma_frac<=0.1 so fine)
u50, w50 = leggauss(50)


def sum_log_Bnum(x: np.ndarray, sigf: np.ndarray) -> np.ndarray:
    """Sigma_i log B_num_i(h) for all h (isotropic; drops the per-event
    h-independent sin(theta)/4pi constant). Returns (n_h, n_events) log B."""
    logB = np.empty((len(h_grid), len(x)))
    for i in range(len(h_grid)):
        dl2z = lambda d: np.interp(d, dl_tab[i], zt)  # noqa: E731
        z_lo = np.maximum(dl2z(np.maximum(x * (1 - 4 * sigf), 0.0)), 1e-6)
        z_hi = np.minimum(dl2z(x * (1 + 4 * sigf)), Z_MAX)
        bad = z_lo >= z_hi
        a, b = z_lo, z_hi
        zn = (a + b)[:, None] / 2 + ((b - a) / 2)[:, None] * u50[None, :]  # (N,50)
        dln = np.interp(zn, zt, dl_tab[i])
        dvcn = np.interp(zn, zt, dvc_tab[i])
        fn = np.interp(zn, zt, fbar_tab[i])
        pgw = norm.pdf(dln / x[:, None], loc=1.0, scale=sigf[:, None])
        integ = (1.0 - fn) * pgw * dvcn / (1.0 + zn)
        B = ((b - a) / 2) * np.sum(w50[None, :] * integ, axis=1)
        B[bad] = 0.0
        with np.errstate(divide="ignore"):
            logB[i] = np.log(B)
    return logB


def peak(hv: np.ndarray, y: np.ndarray) -> dict:
    j = int(np.argmax(y))
    out = {"argmax_h": float(hv[j]), "railed": j in (0, len(hv) - 1)}
    if 0 < j < len(hv) - 1:
        hm, h0, hp = hv[j - 1 : j + 2]
        ym, y0, yp = y[j - 1 : j + 2]
        denom = ym - 2 * y0 + yp
        out["parabolic_h"] = float(h0 - 0.5 * (hp - hm) * (yp - ym) / (2 * denom))
        d2 = 2 * (ym / ((h0 - hm) * (hp - hm)) - y0 / ((hp - h0) * (h0 - hm)) + yp / ((hp - h0) * (hp - hm)))
        out["sigma"] = float(np.sqrt(-1.0 / d2)) if d2 < 0 else None
    return out


print("evaluating B_num on 41-h grid (V0 scattered)...", flush=True)
t0 = time.time()
logB_scat = sum_log_Bnum(x_scat, sigf_ev)
print(f"  {time.time()-t0:.1f}s", flush=True)
logB_nl = sum_log_Bnum(x_noiseless, sigf_ev)

res: dict = {
    "n_mc_drawn": N_MC,
    "n_mc_detected": n_ev,
    "seed": SEED,
    "h_grid": h_grid.tolist(),
    "D_iso": D_iso.tolist(),
    "beta_Gbar_iso": bgbar_iso.tolist(),
    "z_max_h(dl_max)": zmax_h.tolist(),
    "dl_max": dl_max,
}

logD = np.log(D_iso)
logbg = np.log(bgbar_iso)
finite = np.isfinite(logB_scat).all(axis=0)
print(f"events with finite logB at all h: {finite.sum()}/{n_ev}")
res["n_events_finite_all_h"] = int(finite.sum())

for name, logB in [("V0_scattered", logB_scat), ("V0n_noiseless", logB_nl)]:
    S = logB[:, finite].sum(axis=1)
    res[f"{name}_peak_over_D"] = peak(h_grid, S - finite.sum() * logD)
    res[f"{name}_peak_over_betaGbar"] = peak(h_grid, S - finite.sum() * logbg)
    res[f"{name}_sum_logB"] = S.tolist()

# per-event slope consistency at truth (V0)
g = np.gradient(logB_scat[:, finite], h_grid, axis=0)
res["V0_mean_dlogBnum_dh_at_073"] = float(np.mean(g[i73]))
res["dlogD_iso_dh_at_073"] = float(np.gradient(logD, h_grid)[i73])
res["dlogbgbar_iso_dh_at_073"] = float(np.gradient(logbg, h_grid)[i73])
res["dlogD_prod_dh_at_073"] = float(np.gradient(np.log(D_prod), h_grid)[i73])
res["dlogbgbar_prod_dh_at_073"] = float(np.gradient(np.log(bgbar_prod), h_grid)[i73])

# V3: reweight V0 events to the OBSERVED fallback z-profile
z_obs_fb = np.interp(dl_fb, dl_tab[i73], zt)  # z at true cosmology from observed d_L
bins = np.concatenate((np.linspace(0, 0.9, 19), [1.1, 1.5]))
h_obs, _ = np.histogram(z_obs_fb, bins=bins, density=True)
h_mc, _ = np.histogram(z_ev[finite], bins=bins, density=True)
ratio = np.where(h_mc > 0, h_obs / np.maximum(h_mc, 1e-300), 0.0)
w_ev = ratio[np.clip(np.digitize(z_ev[finite], bins) - 1, 0, len(ratio) - 1)]
w_ev *= finite.sum() / w_ev.sum()
S3 = (logB_scat[:, finite] * w_ev[None, :]).sum(axis=1)
res["V3_membership_reweighted_peak_over_D"] = peak(h_grid, S3 - finite.sum() * logD)
res["V3_weights_ess"] = float(w_ev.sum() ** 2 / np.sum(w_ev**2))
res["z_hist_bins"] = bins.tolist()
res["z_hist_obs_fallback"] = h_obs.tolist()
res["z_hist_mc_dark_detected"] = h_mc.tolist()

with open(OUT / "s2_mc_results.json", "w") as f:
    json.dump(res, f, indent=2)
for k, v in res.items():
    if not isinstance(v, list):
        print(k, "=", v)
