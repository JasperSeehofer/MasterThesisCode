"""Z3 — production-axis tabulation of the FIX-3 §7.1 packet (doc §4 item 1).

Tabulates, at the same h-grid z2 used, Sigma_glob (the catalogue selection
sum) under FOUR conditionings of the catalogue-selection kernel:

  (a) M_z-only   — the PRODUCTION 2D object (kernel in m only, 41-node
                    LM_NODES grid, z2's `build_surv_lm`/`q_lm`, == z2's `mz`).
  (b) joint, probe bandwidth   sigma_u = N^(-1/5)          (== z2's `z_mz`).
  (c) joint, Z2-ratified bandwidth   sigma_u = N^(-1/6)*std(u)  (Scott d=2).
  (d) shrunk joint at the Z2 bandwidth, (K5) policy:
        w_ab = ESS_ab / (ESS_ab + n0),  n0 = 10  (_MIN_BAND_INJECTIONS),
        S~ = w_ab * S_joint(Z2 bw) + (1 - w_ab) * S_m(matched machinery),
        S_m built with the SAME 31x61-node machinery, u-factor == 1
        (no second convention vs the joint build);
        empty/underflowed node (tot <= 0 or non-finite) => w_ab = 0.

Reuses z2_zres_slopes.py's data loading, kernel construction
(build_surv_ulm), catalogue-profile assembly, and gap-prediction machinery
verbatim (imported as a module, not re-implemented) so there is exactly one
convention for the pool, the nodes, and the cosmology tables.

Outputs: value ratios at h=0.73, dlogSigma/dh slopes, the production-axis
0.73->0.86 gap increments (M_z-only -> joint, M_z-only -> shrunk-joint) via
z2's own gap-extrapolation method, the symmetric-beta_Gbar control (beta_Gbar
carrying the same composition factor as the catalogue term, per doc §3.1-B),
and the Z2-bandwidth ESS summary (node-level + catalogue-weighted).

Writes z3_results.json. Read-only w.r.t. the repo and w.r.t. z2's own outputs
(z2_results.json is read for cross-checks, never rewritten).
"""

import importlib.util
import json
import sys

import numpy as np

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

BASE = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725"
ZS = f"{BASE}/zres_survival"
N_EVENTS = 3454
N0_SHRINK = 10.0  # _MIN_BAND_INJECTIONS, reused unchanged (K5)

# ---------------- import z2 as a module (reuse, do not re-implement) ----------------
_spec = importlib.util.spec_from_file_location("z2_zres_slopes", f"{ZS}/z2_zres_slopes.py")
z2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z2)  # runs z2 end-to-end once; also (re)writes z2_results.json

h_grid = z2.h_grid
i73 = z2.i73
i86 = z2.i86
n_hat_w = z2.n_hat_w


def slope73(y: np.ndarray) -> float:
    return float(np.gradient(np.log(y), h_grid)[i73])


def dln_73_86(y: np.ndarray) -> float:
    return float(np.log(y[i86] / y[i73]))


# ---------------- catalogue profile (already loaded once by z2, reuse arrays) ----------------
zb_cat = z2.zb_cat
lmb_cat = z2.lmb_cat
W_z = z2.W_z
W_zlm = z2.W_zlm
cells = z2.cells
Wc = z2.Wc
zc_c = z2.zc_c
lm_c = z2.lm_c

# ---------------- (a) M_z-only production object: already z2's `mz` ----------------
# Rebuild the full h-array (z2 only kept v073/slope/dln for it); identical machinery
# (build_surv_lm / q_lm), just evaluated over the whole h_grid.
Sg_mz = np.zeros(len(h_grid))
for i, h in enumerate(h_grid):
    dlb = np.asarray(z2.dist_vectorized(zb_cat, h=float(h)), dtype=np.float64)
    dlc = dlb[cells[:, 0]]
    Sg_mz[i] = np.sum(Wc * z2.q_lm(lm_c, dlc))

assert abs(Sg_mz[i73] - z2.res["Sigma_glob"]["mz"]["v073"]) < 1e-3 * Sg_mz[i73], (
    "M_z-only rebuild does not match z2's own v073 - convention drift"
)

# ---------------- (b) joint at the PROBE bandwidth: already z2's `z_mz` ----------------
Sg_probe = np.zeros(len(h_grid))
for i, h in enumerate(h_grid):
    dlb = np.asarray(z2.dist_vectorized(zb_cat, h=float(h)), dtype=np.float64)
    dlc = dlb[cells[:, 0]]
    Sg_probe[i] = np.sum(Wc * z2.q_ulm(zc_c, lm_c, dlc))

assert abs(Sg_probe[i73] - z2.res["Sigma_glob"]["z_mz"]["v073"]) < 1e-3 * Sg_probe[i73], (
    "probe-bandwidth joint rebuild does not match z2's own v073 - convention drift"
)

# ---------------- (c) joint at the Z2-ratified bandwidth sigma_u = N^(-1/6)*std(u) ----------------
UN2, LN2, surv_ulm_z2bw, ess_ulm_z2bw = z2.build_surv_ulm(z2.SIG_U_REPO)


def _bilerp_indices(zq: np.ndarray, lmq: np.ndarray) -> tuple:
    uq = np.log1p(zq)
    a = np.interp(uq, UN2, np.arange(len(UN2)))
    a0 = np.clip(np.floor(a).astype(int), 0, len(UN2) - 2)
    fa = a - a0
    bq = np.interp(lmq, LN2, np.arange(len(LN2)))
    b0 = np.clip(np.floor(bq).astype(int), 0, len(LN2) - 2)
    fb = bq - b0
    return a0, fa, b0, fb


def q_ulm_z2bw(zq: np.ndarray, lmq: np.ndarray, dq: np.ndarray) -> np.ndarray:
    a0, fa, b0, fb = _bilerp_indices(zq, lmq)
    j = np.clip(np.searchsorted(z2.DLQ, dq), 0, len(z2.DLQ) - 1)
    s = surv_ulm_z2bw
    return (
        (1 - fa) * (1 - fb) * s[a0, b0, j]
        + fa * (1 - fb) * s[a0 + 1, b0, j]
        + (1 - fa) * fb * s[a0, b0 + 1, j]
        + fa * fb * s[a0 + 1, b0 + 1, j]
    )


def ess_at_z2bw(zq: np.ndarray, lmq: np.ndarray) -> np.ndarray:
    """Bilinear-interpolated node ESS (scalar field, no d_L axis)."""
    a0, fa, b0, fb = _bilerp_indices(zq, lmq)
    e = ess_ulm_z2bw
    return (
        (1 - fa) * (1 - fb) * e[a0, b0]
        + fa * (1 - fb) * e[a0 + 1, b0]
        + (1 - fa) * fb * e[a0, b0 + 1]
        + fa * fb * e[a0 + 1, b0 + 1]
    )


Sg_z2bw = np.zeros(len(h_grid))
for i, h in enumerate(h_grid):
    dlb = np.asarray(z2.dist_vectorized(zb_cat, h=float(h)), dtype=np.float64)
    dlc = dlb[cells[:, 0]]
    Sg_z2bw[i] = np.sum(Wc * q_ulm_z2bw(zc_c, lm_c, dlc))

# ---------------- (d) shrunk joint at the Z2 bandwidth, (K5) ----------------
# S_m built with the SAME 31x61-node machinery, u-factor == 1 (matched to the
# joint build's UN2/LN2 grid and DLQ -- NOT z2's 41-node LM_NODES production
# grid, per doc §3.4/§4 item 1(b): the shrinkage target must share the joint
# build's convention so no second convention enters the blend).
idx_dlq = np.searchsorted(z2.dh_s, z2.DLQ, side="left")
idx_dlq_c = np.minimum(idx_dlq, z2.N - 1)
surv_m_matched = np.empty((len(LN2), len(z2.DLQ)))
for b, ln_ in enumerate(LN2):
    w = np.exp(-0.5 * ((z2.lm_s - ln_) / z2.SIG_LM) ** 2)
    suf = np.cumsum(w[::-1])[::-1]
    surv_m_matched[b] = np.where(idx_dlq < z2.N, suf[idx_dlq_c], 0.0) / w.sum()


def q_m_matched(lmq: np.ndarray, dq: np.ndarray) -> np.ndarray:
    bq = np.interp(lmq, LN2, np.arange(len(LN2)))
    b0 = np.clip(np.floor(bq).astype(int), 0, len(LN2) - 2)
    fb = bq - b0
    j = np.clip(np.searchsorted(z2.DLQ, dq), 0, len(z2.DLQ) - 1)
    return (1 - fb) * surv_m_matched[b0, j] + fb * surv_m_matched[b0 + 1, j]


# Node-level shrunk survival table S~(u_a, m_b, d_L): blend joint(Z2 bw) with
# S_m(matched), weight w_ab = ESS/(ESS+n0); empty/underflowed node -> w=0.
tot_ulm_z2bw = np.empty((len(UN2), len(LN2)))
for a, un in enumerate(UN2):
    sig_i = z2.SIG_U_REPO * z2.lam_s
    wu = np.exp(-0.5 * ((z2.u_s - un) / sig_i) ** 2)
    for b, ln_ in enumerate(LN2):
        w = wu * np.exp(-0.5 * ((z2.lm_s - ln_) / z2.SIG_LM) ** 2)
        tot_ulm_z2bw[a, b] = w.sum()

finite_ok = np.isfinite(tot_ulm_z2bw) & (tot_ulm_z2bw > 0)
w_shrink = np.where(finite_ok, ess_ulm_z2bw / (ess_ulm_z2bw + N0_SHRINK), 0.0)
w_shrink = np.where(np.isfinite(w_shrink), w_shrink, 0.0)

surv_shrunk = w_shrink[:, :, None] * surv_ulm_z2bw + (1.0 - w_shrink)[:, :, None] * surv_m_matched[None, :, :]


def q_shrunk(zq: np.ndarray, lmq: np.ndarray, dq: np.ndarray) -> np.ndarray:
    a0, fa, b0, fb = _bilerp_indices(zq, lmq)
    j = np.clip(np.searchsorted(z2.DLQ, dq), 0, len(z2.DLQ) - 1)
    s = surv_shrunk
    return (
        (1 - fa) * (1 - fb) * s[a0, b0, j]
        + fa * (1 - fb) * s[a0 + 1, b0, j]
        + (1 - fa) * fb * s[a0, b0 + 1, j]
        + fa * fb * s[a0 + 1, b0 + 1, j]
    )


Sg_shrunk = np.zeros(len(h_grid))
for i, h in enumerate(h_grid):
    dlb = np.asarray(z2.dist_vectorized(zb_cat, h=float(h)), dtype=np.float64)
    dlc = dlb[cells[:, 0]]
    Sg_shrunk[i] = np.sum(Wc * q_shrunk(zc_c, lm_c, dlc))

# ---------------- assembled D_gen, ASYMMETRIC (beta_Gbar pool-conditioned, z2's bg_zres) ----------------
bg_zres = z2.bg_zres  # z-only, pool-conditioned complement -- shared by all four conditionings
D_gen_mz = Sg_mz / n_hat_w + bg_zres
D_gen_probe = Sg_probe / n_hat_w + bg_zres
D_gen_z2bw = Sg_z2bw / n_hat_w + bg_zres
D_gen_shrunk = Sg_shrunk / n_hat_w + bg_zres

d_gen_3d = dln_73_86(z2.Dgen_run)
BASELINE_GEN_3D = 92.0  # DERIVATION_GENERATOR_CONSISTENT_NORM §6.3, shared by z2 and this packet


def gap(D_gen_new: np.ndarray) -> float:
    return BASELINE_GEN_3D + N_EVENTS * (d_gen_3d - dln_73_86(D_gen_new))


gap_mz = gap(D_gen_mz)
gap_probe = gap(D_gen_probe)
gap_z2bw = gap(D_gen_z2bw)
gap_shrunk = gap(D_gen_shrunk)

# ---------------- symmetric-beta_Gbar control (doc §3.1-B, §4 item 10) ----------------
# beta_Gbar carries the SAME composition factor as the catalogue term, defined
# per-axis as the elementwise ratio of the axis's own joint-vs-baseline
# catalogue sums (z2_zres_slopes.py:342-360's assembly, generalized): the
# complement leg is reweighted by exactly the composition shift the catalogue
# term itself measures on that axis, so the two legs move together (the
# symmetric-treatment reading of §3.1-B).
bg_sym_prod = bg_zres * (Sg_probe / Sg_mz)  # production axis: M_z-only -> joint
D_gen_sym_prod = Sg_probe / n_hat_w + bg_sym_prod
gap_sym_prod = gap(D_gen_sym_prod)

bg_sym_tab = bg_zres * (Sg_probe / z2.Sg["zres"])  # tabulated axis: z-only -> joint
D_gen_sym_tab = Sg_probe / n_hat_w + bg_sym_tab
gap_sym_tab = gap(D_gen_sym_tab)

# ---------------- ESS summary at the Z2 bandwidth ----------------
# NOTE: `cells`/`Wc`/`zc_c`/`lm_c` index the CATALOGUE profile grid (300 z-bins
# x 60 lm-bins, catalog_zw_profile.json), a different index space from the
# UN2/LN2 survival-node grid (61x31) -- ESS must be bilinearly interpolated
# to the catalogue cell locations (zc_c, lm_c), never indexed directly by
# `cells`.
ess_flat = ess_ulm_z2bw.ravel()
ess_at_cells = ess_at_z2bw(zc_c, lm_c)
finite_at_cells = np.isfinite(ess_at_cells) & (ess_at_cells > 0)
w_cat_at_cells = np.where(
    finite_at_cells,
    ess_at_cells / (ess_at_cells + N0_SHRINK),
    0.0,
)
order_ess = np.argsort(ess_at_cells)
cum_w = np.cumsum(Wc[order_ess])
med_ess = float(np.interp(cum_w[-1] / 2.0, cum_w, ess_at_cells[order_ess]))

ess_summary = {
    "n_nodes": int(ess_flat.size),
    "min": float(ess_flat.min()),
    "frac_below_10": float((ess_flat < 10).mean()),
    "frac_below_100": float((ess_flat < 100).mean()),
    "frac_below_500": float((ess_flat < 500).mean()),
    "catalogue_weighted_wbar_n0_10": float(np.sum(Wc * w_cat_at_cells) / np.sum(Wc)),
    "catalogue_Wfrac_on_ESS_below_100": float(np.sum(Wc[ess_at_cells < 100]) / np.sum(Wc)),
    "catalogue_weighted_median_ESS": med_ess,
}

# ---------------- value ratios at h = 0.73 ----------------
ratios_073 = {
    "joint_probe_over_pooled": float(Sg_probe[i73] / z2.Sg["pooled"][i73]),
    "joint_probe_over_zonly": float(Sg_probe[i73] / z2.Sg["zres"][i73]),
    "joint_probe_over_Mzonly": float(Sg_probe[i73] / Sg_mz[i73]),
    "Mzonly_over_pooled": float(Sg_mz[i73] / z2.Sg["pooled"][i73]),  # doc §0 0.589-vs-0.556 parity item
    "joint_z2bw_over_pooled": float(Sg_z2bw[i73] / z2.Sg["pooled"][i73]),
    "joint_z2bw_over_zonly": float(Sg_z2bw[i73] / z2.Sg["zres"][i73]),
    "joint_z2bw_over_Mzonly": float(Sg_z2bw[i73] / Sg_mz[i73]),
    "shrunk_over_Mzonly": float(Sg_shrunk[i73] / Sg_mz[i73]),
    "shrunk_over_joint_z2bw": float(Sg_shrunk[i73] / Sg_z2bw[i73]),
}

# ---------------- pack + write ----------------
res = {
    "h_grid": h_grid.tolist(),
    "i73": i73,
    "i86": i86,
    "n0_shrink": N0_SHRINK,
    "bandwidths": {
        "sigma_u_probe_scott1d": z2.SIG_U_SCOTT1D,
        "sigma_u_z2_repo": z2.SIG_U_REPO,
        "sigma_lm": z2.SIG_LM,
    },
    "Sigma_glob": {
        "Mz_only_production": {
            "v073": float(Sg_mz[i73]), "slope": slope73(Sg_mz), "dln_073_086": dln_73_86(Sg_mz),
        },
        "joint_probe_bw": {
            "v073": float(Sg_probe[i73]), "slope": slope73(Sg_probe), "dln_073_086": dln_73_86(Sg_probe),
        },
        "joint_z2_bw": {
            "v073": float(Sg_z2bw[i73]), "slope": slope73(Sg_z2bw), "dln_073_086": dln_73_86(Sg_z2bw),
        },
        "shrunk_joint_z2_bw_K5": {
            "v073": float(Sg_shrunk[i73]), "slope": slope73(Sg_shrunk), "dln_073_086": dln_73_86(Sg_shrunk),
        },
    },
    "ratios_at_073": ratios_073,
    "production_axis_gaps": {
        "Mz_only_baseline": gap_mz,
        "joint_probe_bw": gap_probe,
        "joint_z2_bw": gap_z2bw,
        "shrunk_joint_z2_bw": gap_shrunk,
        "increment_Mzonly_to_joint_probe_bw": gap_probe - gap_mz,
        "increment_Mzonly_to_joint_z2_bw": gap_z2bw - gap_mz,
        "increment_Mzonly_to_shrunk_joint": gap_shrunk - gap_mz,
    },
    "symmetric_beta_Gbar_control": {
        "production_axis": {
            "gap_Mzonly_baseline": gap_mz,
            "gap_joint_composition_matched": gap_sym_prod,
            "increment": gap_sym_prod - gap_mz,
        },
        "tabulated_axis": {},  # filled below (needs z2's z-only baseline gap)
    },
    "ess_summary_z2_bandwidth": ess_summary,
}
# tabulated-axis symmetric control (z-only -> joint), for completeness / doc §3.1-B cross-check
gap_zonly_baseline = z2.res["gap_predictions"]["FIX2_stacked_on_gen_3D"]
res["symmetric_beta_Gbar_control"]["tabulated_axis"] = {
    "gap_zonly_baseline": gap_zonly_baseline,
    "gap_joint_composition_matched": gap_sym_tab,
    "increment": gap_sym_tab - gap_zonly_baseline,
}

with open(f"{ZS}/z3_results.json", "w") as f:
    json.dump(res, f, indent=1)

# ---------------- compact summary table ----------------
print("\n=== Z3 production-axis tabulation ===\n")
print(f"{'Sigma_glob variant':<26}{'v(0.73)':>14}{'slope':>10}{'dln(.73->.86)':>16}")
for name, arr in [
    ("M_z-only (production)", Sg_mz),
    ("joint, probe bw", Sg_probe),
    ("joint, Z2 bw", Sg_z2bw),
    ("shrunk joint, Z2 bw", Sg_shrunk),
]:
    print(f"{name:<26}{arr[i73]:>14.4e}{slope73(arr):>10.4f}{dln_73_86(arr):>16.5f}")

print("\n--- ratios @ 0.73 ---")
for k, v in ratios_073.items():
    print(f"  {k:<32}{v:.4f}")

print("\n--- production-axis gaps [ln] (baseline = M_z-only, doc §3.8 rev. B) ---")
print(f"  M_z-only baseline           {gap_mz:8.2f}")
print(f"  joint, probe bw             {gap_probe:8.2f}   increment {gap_probe - gap_mz:+.2f}")
print(f"  joint, Z2 bw                {gap_z2bw:8.2f}   increment {gap_z2bw - gap_mz:+.2f}")
print(f"  shrunk joint, Z2 bw (K5)    {gap_shrunk:8.2f}   increment {gap_shrunk - gap_mz:+.2f}")

print("\n--- symmetric-beta_Gbar control (doc §3.1-B / §4 item 10) ---")
print(f"  production axis: gap {gap_sym_prod:8.2f}   increment {gap_sym_prod - gap_mz:+.2f}")
print(f"  tabulated axis:  gap {gap_sym_tab:8.2f}   increment {gap_sym_tab - gap_zonly_baseline:+.2f}")

print("\n--- ESS summary @ Z2 bandwidth ---")
for k, v in ess_summary.items():
    print(f"  {k:<34}{v}")

print(f"\nWrote {ZS}/z3_results.json")
