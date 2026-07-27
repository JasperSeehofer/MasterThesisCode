"""P1 parity audit, part B — probe-side rebuild, clamp decomposition, axis translation.

Rebuilds the z2/z3 probe estimators DIRECTLY from the canonical pool (same
conventions, functions copied from z2_zres_slopes.py / z3_production_axis.py
verbatim; z2 is NOT exec'd so its z2_results.json is not rewritten) and
validates every rebuilt table against the stored z2/z3 values (<1e-3 rel).

Computes:
  (i)  binned-profile vs row-by-row parity for the SAME estimator (q_lm and
       shrunk-joint queried at the 9.06M catalogue rows vs at the 300x60
       profile cells) — isolates the binning error from the estimator/grid
       convention error (the latter = production-interpolator vs probe-grid,
       from p1a's Sigma_off vs the row-by-row q_lm here).
  (iv) clamp decomposition of the probe increment: hybrid sums where only
       the unclamped (lm < 6.0) or only the clamped weight switches
       conditioning, on the SAME D_gen gap axis as the pre-registered
       -6.5 +/- 4 gate (z3 machinery: gap = 92 + N*(dlnDgen_3D - dlnDgen)).
  (ii) d_L convention delta: step (searchsorted, probe) vs linear-in-d_L
       (production) reads of the SAME stored survival tables.
  (translation) the A-cell axis multiplier: exact per-event re-prediction
       Sum_i ln(1 + s_i(r-1)) from the cellApp diagnostics for any Sigma
       table pair — production on/off (tautology check vs measured
       -0.51/-1.18), probe shrunk/mz (the parity-corrected A-cell
       prediction), and the clamp hybrids.

Inputs: pool CSVs, catalog_zw_profile.json, z2/z3_results.json,
generator_norm_Dgen_table.json, cellApp event_likelihoods.csv, and the
catalogue-row cache written by p1a. Writes p1b_results.json.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = "/home/jasper/Repositories/MasterThesisCode"
BASE = f"{REPO}/results/lcat_h_dependence_20260725"
ZS = f"{BASE}/zres_survival"
HERE = f"{BASE}/mass_ab_20260727/p1_parity"
SCRATCH = os.environ.get(
    "P1_CACHE_DIR",
    "/tmp/claude-1000/-home-jasper-Repositories-MasterThesisCode/"
    "7cb124bf-2889-4fc6-987f-d69ae709b032/scratchpad",
)
sys.path.insert(0, REPO)

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.physical_relations import dist_vectorized  # noqa: E402

SNR_THR = 20.0
N_EVENTS = 3454
N0 = 10.0

# ---------------- pool (z2 conventions, lines 51-89) ----------------
pool = pd.concat(
    [pd.read_csv(f) for f in sorted(glob.glob(f"{BASE}/data/injections/injection_h_0p73_task_*.csv"))],
    ignore_index=True,
)
N = len(pool)
z_p = pool["z"].to_numpy(dtype=np.float64)
u_p = np.log1p(z_p)
lm_p = np.log10(pool["M"].to_numpy(dtype=np.float64))
snr_p = pool["SNR"].to_numpy(dtype=np.float64)
dl_p = pool["luminosity_distance"].to_numpy(dtype=np.float64)
d_hor = snr_p * dl_p / SNR_THR
srt = np.argsort(d_hor, kind="mergesort")
dh_s, u_s, lm_s = d_hor[srt], u_p[srt], lm_p[srt]
DLQ = np.linspace(1e-4, float(dh_s[-1]) * 1.02, 3000)
LM_NODES = np.linspace(float(lm_p.min()), float(lm_p.max()), 41)
SIG_U_SCOTT1D = float(N ** (-1.0 / 5.0) * u_p.std())
SIG_U_REPO = float(N ** (-1.0 / 6.0) * u_p.std())
SIG_LM = float(N ** (-1.0 / 6.0) * lm_p.std())
M_MAX_POOL = float(lm_p.max())

# Abramson factors (z2 lines 78-89)
UB = np.linspace(0.0, float(np.log1p(1.5)), 401)
ubc = 0.5 * (UB[:-1] + UB[1:])
du = UB[1] - UB[0]
hist_u, _ = np.histogram(u_p, bins=UB, density=True)
kh = int(np.ceil(4 * SIG_U_SCOTT1D / du))
kk = np.exp(-0.5 * (np.arange(-kh, kh + 1) * du / SIG_U_SCOTT1D) ** 2)
pilot = np.clip(np.convolve(hist_u, kk / kk.sum(), mode="same"), 1e-12, None)
f_at = np.interp(u_p, ubc, pilot)
G = float(np.exp(np.mean(np.log(f_at))))
lam_s = np.sqrt(G / f_at)[srt]

idx_dlq = np.searchsorted(dh_s, DLQ, side="left")
idx_dlq_c = np.minimum(idx_dlq, N - 1)
inside = idx_dlq < N


def build_surv_lm() -> np.ndarray:
    surv = np.empty((len(LM_NODES), len(DLQ)))
    for k, ln_ in enumerate(LM_NODES):
        w = np.exp(-0.5 * ((lm_s - ln_) / SIG_LM) ** 2)
        suf = np.cumsum(w[::-1])[::-1]
        surv[k] = np.where(inside, suf[idx_dlq_c], 0.0) / w.sum()
    return surv


def build_surv_ulm(sigma_u: float):
    UN = np.linspace(0.0, float(np.log1p(1.5)), 61)
    LN = np.linspace(float(lm_p.min()), float(lm_p.max()), 31)
    sig_i = sigma_u * lam_s
    surv = np.empty((len(UN), len(LN), len(DLQ)))
    ess = np.empty((len(UN), len(LN)))
    tot_ab = np.empty((len(UN), len(LN)))
    for a, un in enumerate(UN):
        wu = np.exp(-0.5 * ((u_s - un) / sig_i) ** 2)
        for b, ln_ in enumerate(LN):
            w = wu * np.exp(-0.5 * ((lm_s - ln_) / SIG_LM) ** 2)
            tot = w.sum()
            tot_ab[a, b] = tot
            ess[a, b] = tot * tot / np.dot(w, w) if tot > 0 else 0.0
            suf = np.cumsum(w[::-1])[::-1]
            surv[a, b] = np.where(inside, suf[idx_dlq_c], 0.0) / tot
    return UN, LN, surv, ess, tot_ab


surv_lm = build_surv_lm()
UN2, LN2, surv_ulm, ess_ulm, tot_ulm = build_surv_ulm(SIG_U_REPO)

# m-only marginal on the joint machinery (z3 lines 148-162) + (K5) shrink
surv_m_matched = np.empty((len(LN2), len(DLQ)))
for b, ln_ in enumerate(LN2):
    w = np.exp(-0.5 * ((lm_s - ln_) / SIG_LM) ** 2)
    suf = np.cumsum(w[::-1])[::-1]
    surv_m_matched[b] = np.where(inside, suf[idx_dlq_c], 0.0) / w.sum()
finite_ok = np.isfinite(tot_ulm) & (tot_ulm > 0)
w_shrink = np.where(finite_ok, ess_ulm / (ess_ulm + N0), 0.0)
w_shrink = np.where(np.isfinite(w_shrink), w_shrink, 0.0)
surv_shrunk = (
    w_shrink[:, :, None] * surv_ulm + (1.0 - w_shrink)[:, :, None] * surv_m_matched[None, :, :]
)


def _j_step(dq: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(DLQ, dq), 0, len(DLQ) - 1)


def q_lm(lmq, dq, linear_dl=False):
    k = np.interp(lmq, LM_NODES, np.arange(len(LM_NODES)))
    k0 = np.clip(np.floor(k).astype(int), 0, len(LM_NODES) - 2)
    fr = k - k0
    if linear_dl:
        pos = np.interp(dq, DLQ, np.arange(len(DLQ)))
        j0 = np.clip(np.floor(pos).astype(int), 0, len(DLQ) - 2)
        fd = np.clip(pos - j0, 0, 1)
        lo = (1 - fr) * surv_lm[k0, j0] + fr * surv_lm[k0 + 1, j0]
        hi = (1 - fr) * surv_lm[k0, j0 + 1] + fr * surv_lm[k0 + 1, j0 + 1]
        return (1 - fd) * lo + fd * hi
    j = _j_step(dq)
    return (1 - fr) * surv_lm[k0, j] + fr * surv_lm[k0 + 1, j]


def q_tab3(s, zq, lmq, dq, linear_dl=False):
    """Bilinear (u, m) query of a (61, 31, 3000) table; step or linear d_L."""
    uq = np.log1p(zq)
    a = np.interp(uq, UN2, np.arange(len(UN2)))
    a0 = np.clip(np.floor(a).astype(int), 0, len(UN2) - 2)
    fa = a - a0
    bq = np.interp(lmq, LN2, np.arange(len(LN2)))
    b0 = np.clip(np.floor(bq).astype(int), 0, len(LN2) - 2)
    fb = bq - b0
    if linear_dl:
        pos = np.interp(dq, DLQ, np.arange(len(DLQ)))
        j0 = np.clip(np.floor(pos).astype(int), 0, len(DLQ) - 2)
        fd = np.clip(pos - j0, 0, 1)
        out = 0.0
        for jj, wd in ((j0, 1 - fd), (j0 + 1, fd)):
            out = out + wd * (
                (1 - fa) * (1 - fb) * s[a0, b0, jj]
                + fa * (1 - fb) * s[a0 + 1, b0, jj]
                + (1 - fa) * fb * s[a0, b0 + 1, jj]
                + fa * fb * s[a0 + 1, b0 + 1, jj]
            )
        return out
    j = _j_step(dq)
    return (
        (1 - fa) * (1 - fb) * s[a0, b0, j]
        + fa * (1 - fb) * s[a0 + 1, b0, j]
        + (1 - fa) * fb * s[a0, b0 + 1, j]
        + fa * fb * s[a0 + 1, b0 + 1, j]
    )


# ---------------- catalogue: binned profile + row cache ----------------
prof = json.load(open(f"{ZS}/catalog_zw_profile.json"))
z_edges = np.array(prof["z_edges"])
zb_cat = 0.5 * (z_edges[:-1] + z_edges[1:])
W_zlm = np.array(prof["W_z_lm"])
lm_edges = np.array(prof["lm_edges"])
lmb_cat = 0.5 * (lm_edges[:-1] + lm_edges[1:])
cells = np.argwhere(W_zlm > 0)
Wc = W_zlm[cells[:, 0], cells[:, 1]]
zc_c = zb_cat[cells[:, 0]]
lm_c = lmb_cat[cells[:, 1]]
cl_c = lm_c >= M_MAX_POOL  # clamped catalogue cells

rows = np.load(f"{SCRATCH}/p1_catalogue_zM.npz")
z_r_all, M_r_all = rows["z"], rows["M"]
okr = (
    np.isfinite(M_r_all) & (M_r_all > 0.0) & np.isfinite(z_r_all) & (z_r_all >= 0.0) & (z_r_all < 1.5)
)
z_r, M_r = z_r_all[okr], M_r_all[okr]
w_r = np.asarray(R_eff_per_mbh(M_r), dtype=np.float64) / (1.0 + z_r)
lm_r = np.log10(M_r * (1.0 + z_r))
cl_r = lm_r >= M_MAX_POOL

# ---------------- h tables ----------------
tabj = json.load(open(f"{BASE}/generator_norm_Dgen_table.json"))
h_grid = np.array(tabj["h"])
i73 = int(np.argmin(np.abs(h_grid - 0.73)))
i86 = int(np.argmin(np.abs(h_grid - 0.86)))
i80 = int(np.argmin(np.abs(h_grid - 0.80)))
n_hat_w = np.array(tabj["n_hat_w"])
Dgen_run = np.array(tabj["D_gen"])
z2res = json.load(open(f"{ZS}/z2_results.json"))
bg_zres = np.array(z2res["beta_Gbar_zres_table"])
z3res = json.load(open(f"{ZS}/z3_results.json"))

variants = [
    "mz_binned",
    "shrunk_binned",
    "hyb_unclamped_binned",  # unclamped cells switch to shrunk-joint, clamped stay mz
    "hyb_clamped_binned",
    "mz_row",
    "shrunk_row",
    "mz_binned_lineardl",
    "shrunk_binned_lineardl",
]
Sg = {k: np.zeros(len(h_grid)) for k in variants}
for i, h in enumerate(h_grid):
    dlb = np.asarray(dist_vectorized(zb_cat, h=float(h)), dtype=np.float64)
    dlc = dlb[cells[:, 0]]
    v_mz = q_lm(lm_c, dlc)
    v_sh = q_tab3(surv_shrunk, zc_c, lm_c, dlc)
    Sg["mz_binned"][i] = np.sum(Wc * v_mz)
    Sg["shrunk_binned"][i] = np.sum(Wc * v_sh)
    Sg["hyb_unclamped_binned"][i] = np.sum(Wc * np.where(cl_c, v_mz, v_sh))
    Sg["hyb_clamped_binned"][i] = np.sum(Wc * np.where(cl_c, v_sh, v_mz))
    Sg["mz_binned_lineardl"][i] = np.sum(Wc * q_lm(lm_c, dlc, linear_dl=True))
    Sg["shrunk_binned_lineardl"][i] = np.sum(Wc * q_tab3(surv_shrunk, zc_c, lm_c, dlc, linear_dl=True))
    # row-by-row (full catalogue; z < 1.5 support like the profile)
    dlr = np.asarray(dist_vectorized(z_r, h=float(h)), dtype=np.float64)
    Sg["mz_row"][i] = np.sum(w_r * q_lm(lm_r, dlr))
    Sg["shrunk_row"][i] = np.sum(w_r * q_tab3(surv_shrunk, z_r, lm_r, dlr))

# validation vs stored z3 numbers
val = {
    "mz_binned_v073_vs_z3": float(
        Sg["mz_binned"][i73] / z3res["Sigma_glob"]["Mz_only_production"]["v073"] - 1.0
    ),
    "shrunk_binned_v073_vs_z3": float(
        Sg["shrunk_binned"][i73] / z3res["Sigma_glob"]["shrunk_joint_z2_bw_K5"]["v073"] - 1.0
    ),
}
assert abs(val["mz_binned_v073_vs_z3"]) < 1e-3, val
assert abs(val["shrunk_binned_v073_vs_z3"]) < 1e-3, val


def dln_73_86(y):
    return float(np.log(y[i86] / y[i73]))


def gap(sg):
    d_gen_3d = dln_73_86(Dgen_run)
    dgen = sg / n_hat_w + bg_zres
    return 92.0 + N_EVENTS * (d_gen_3d - dln_73_86(dgen))


gaps = {k: gap(v) for k, v in Sg.items()}
base = gaps["mz_binned"]
increments = {k: gaps[k] - base for k in variants}
increments["mz_row_base_shrunk_row"] = gaps["shrunk_row"] - gaps["mz_row"]

# clamp fractions (weights)
clamp = {
    "profile_W_frac_clamped": float(Wc[cl_c].sum() / Wc.sum()),
    "rows_W_frac_clamped": float(w_r[cl_r].sum() / w_r.sum()),
    "pool_m_max": M_MAX_POOL,
}

# ---------------- A-cell exact per-event translation ----------------
diag = pd.read_csv(f"{BASE}/mass_ab_20260727/cellApp/simulations/diagnostics/event_likelihoods.csv")
H_VENUE = [0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86]
hv_idx = [int(np.argmin(np.abs(h_grid - h))) for h in H_VENUE]

# production cluster tables (see p1a header for provenance)
SIG_OFF = {0.60: 2.700808e8, 0.65: 2.759635e8, 0.70: 2.815376e8, 0.73: 2.847445e8,
           0.76: 2.878574e8, 0.80: 2.918739e8, 0.86: 2.976373e8}
SIG_ON = {0.60: 2.767679e8, 0.65: 2.826148e8, 0.70: 2.885021e8, 0.73: 2.920316e8,
          0.76: 2.955726e8, 0.80: 3.003153e8, 0.86: 3.074504e8}
SIG_GO = {0.60: 2.838463e8, 0.65: 2.899200e8, 0.70: 2.956500e8, 0.73: 2.989355e8,
          0.76: 3.021176e8, 0.80: 3.062085e8, 0.86: 3.120567e8}


def acell_delta(ratio_by_h: dict[float, float]) -> dict[str, float]:
    """Exact A-cell profile shift for L_cat_wbh -> L_cat_wbh / rho(h).

    rho(h) = Sigma_variant(h)/Sigma_base(h). Returns the flag delta of the
    2D ln profile relative to h=0.73 at 0.80 and 0.86 and the raw shifts.
    """
    shifts = {}
    for h in H_VENUE:
        dh = diag[np.isclose(diag.h, h)]
        lc = dh.w_G * dh.L_cat_with_bh
        comb = dh.combined_with_bh
        rho = ratio_by_h[h]
        new = comb + lc * (1.0 / rho - 1.0)
        shifts[h] = float(np.sum(np.log(new) - np.log(comb)))
    return {
        "shift_073": shifts[0.73],
        "delta_at_080": shifts[0.80] - shifts[0.73],
        "delta_at_086": shifts[0.86] - shifts[0.73],
        "raw_shifts": {str(h): shifts[h] for h in H_VENUE},
    }


trans = {}
trans["production_on_vs_off"] = acell_delta({h: SIG_ON[h] / SIG_OFF[h] for h in H_VENUE})
trans["production_gridonly_vs_off"] = acell_delta({h: SIG_GO[h] / SIG_OFF[h] for h in H_VENUE})
trans["production_conditioning_only"] = acell_delta({h: SIG_ON[h] / SIG_GO[h] for h in H_VENUE})
trans["probe_shrunk_vs_mz"] = acell_delta(
    {h: float(Sg["shrunk_binned"][j] / Sg["mz_binned"][j]) for h, j in zip(H_VENUE, hv_idx)}
)
trans["probe_hyb_unclamped_vs_mz"] = acell_delta(
    {h: float(Sg["hyb_unclamped_binned"][j] / Sg["mz_binned"][j]) for h, j in zip(H_VENUE, hv_idx)}
)
trans["probe_hyb_clamped_vs_mz"] = acell_delta(
    {h: float(Sg["hyb_clamped_binned"][j] / Sg["mz_binned"][j]) for h, j in zip(H_VENUE, hv_idx)}
)

# effective multipliers
s73 = diag[np.isclose(diag.h, 0.73)]
s86 = diag[np.isclose(diag.h, 0.86)]
sum_s_73 = float(np.sum(s73.w_G * s73.L_cat_with_bh / s73.combined_with_bh))
sum_s_86 = float(np.sum(s86.w_G * s86.L_cat_with_bh / s86.combined_with_bh))
ddln_probe = dln_73_86(Sg["shrunk_binned"]) - dln_73_86(Sg["mz_binned"])
mult = {
    "sum_s_i_2D_at_073": sum_s_73,
    "sum_s_i_2D_at_086": sum_s_86,
    "probe_ddln_shrunk_minus_mz_073_086": ddln_probe,
    "z3_gate_increment": z3res["production_axis_gaps"]["increment_Mzonly_to_shrunk_joint"],
    "z3_implied_Dgen_axis_multiplier": float(
        z3res["production_axis_gaps"]["increment_Mzonly_to_shrunk_joint"] / (-ddln_probe)
    ),
}

res = {
    "validation_vs_z3": val,
    "clamp": clamp,
    "Sigma_073": {k: float(v[i73]) for k, v in Sg.items()},
    "dln_073_086": {k: dln_73_86(v) for k, v in Sg.items()},
    "dln_073_080": {k: float(np.log(v[i80] / v[i73])) for k, v in Sg.items()},
    "Dgen_axis_gaps": gaps,
    "Dgen_axis_increments_vs_mz_binned": increments,
    "binned_vs_row_parity": {
        "mz_value_ratio_073": float(Sg["mz_binned"][i73] / Sg["mz_row"][i73]),
        "shrunk_value_ratio_073": float(Sg["shrunk_binned"][i73] / Sg["shrunk_row"][i73]),
        "mz_dln_073_086_binned": dln_73_86(Sg["mz_binned"]),
        "mz_dln_073_086_row": dln_73_86(Sg["mz_row"]),
        "shrunk_dln_073_086_binned": dln_73_86(Sg["shrunk_binned"]),
        "shrunk_dln_073_086_row": dln_73_86(Sg["shrunk_row"]),
    },
    "dl_convention_delta": {
        "mz_value_ratio_lineardl_over_step_073": float(
            Sg["mz_binned_lineardl"][i73] / Sg["mz_binned"][i73]
        ),
        "shrunk_value_ratio_lineardl_over_step_073": float(
            Sg["shrunk_binned_lineardl"][i73] / Sg["shrunk_binned"][i73]
        ),
        "mz_dln_delta": dln_73_86(Sg["mz_binned_lineardl"]) - dln_73_86(Sg["mz_binned"]),
        "shrunk_dln_delta": dln_73_86(Sg["shrunk_binned_lineardl"]) - dln_73_86(Sg["shrunk_binned"]),
    },
    "acell_translation": trans,
    "multipliers": mult,
    "shrinkage_uniform_vs_percell_note": {
        "z3_percell_increment": z3res["production_axis_gaps"]["increment_Mzonly_to_shrunk_joint"],
        "z3_uniform_wbar_estimate": -6.96,
        "source": "z3 pre-registration (doc §3.9 P2 decomposition)",
    },
}
with open(f"{HERE}/p1b_results.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("Dgen_axis_gaps",)}, indent=1))
print(f"wrote {HERE}/p1b_results.json")
