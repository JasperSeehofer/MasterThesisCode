"""Reviewer scratch: quantitative adjudication of the flagged venue-drift mechanism.

Mechanism (my own algebra, re-derived — see C2_star_review.md):
  venue accepted-event law = model class-G law x S_bar_phi(z_ev), renormalized by
  Sigma~^{phi4D}/Sigma~^{4D} = <S_bar_phi>_{model,1}. Hence for any per-event weight
  omega(w2) (a function of the DATA only):

    E_venue[omega] / E_model[omega] = <S_phi>_{m,omega} / <S_phi>_{m,1}
                                    = E_b[omega] * E_b[1/S_phi] / E_b[omega/S_phi]      (R_pred)

  with E_b over the banked venue-accepted measure (all 200 accepted/seed; omega=0 off F-0).

Predicted observed ratio LHS/RHS = R_pred(omega), omega_identity = (1-w2)*1_F0,
omega_BR = (1-w2)/(1+(r2-1)*w2)*1_F0.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(
    "/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/"
    "abb9d681-b424-483f-92ff-341423c5a742/scratchpad"
)
ROOT = Path(
    "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/"
    "realistic_20260729/p3_2d_fleet_20260825"
)
R2 = 2.6124925
C2_STAR = 0.06124403326364123
SEEDS = list(range(900101, 900125))

tab = np.load(SCRATCH / "phi_table.npz")
z_grid, s_grid = tab["z"], tab["s"]


def s_phi(z):
    # endpoint-clamped np.interp — the committed convention (:6368 twin factor)
    return np.interp(np.asarray(z, dtype=np.float64), z_grid, s_grid)


def load_seed(arm, seed):
    work = ROOT / f"{arm}_{seed}_work" / f"seed{seed}" / "simulations"
    prep = pd.read_csv(work / "prepared_cramer_rao_bounds.csv")
    diag = pd.read_csv(work / "diagnostics" / "event_likelihoods.csv")
    diag = diag[np.isclose(diag["h"], 0.73)].copy()
    return prep, diag


# ---- Mapping + F-0 verification on one seed --------------------------------
prep, diag = load_seed("bt", 900101)
sigma_dl = np.sqrt(prep["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
rel = sigma_dl / prep["luminosity_distance"].to_numpy()
f0_mask = (rel < 0.10) & (prep["SNR"].to_numpy() >= 20.0)
idx_f0_0based = set(np.flatnonzero(f0_mask).tolist())
idx_f0_1based = set((np.flatnonzero(f0_mask) + 1).tolist())
idx_diag = set(diag["event_idx"].astype(int).tolist())
print("F-0 mapping check (seed 900101 bt):")
print("  n_prep=200, n_F0(prep)=", int(f0_mask.sum()), " n_diag=", len(idx_diag))
print("  diag==F0(0-based):", idx_diag == idx_f0_0based,
      "  diag==F0(1-based):", idx_diag == idx_f0_1based)
if idx_diag == idx_f0_0based:
    BASE = 0
elif idx_diag == idx_f0_1based:
    BASE = 1
else:
    inter0 = len(idx_diag & idx_f0_0based)
    inter1 = len(idx_diag & idx_f0_1based)
    print("  partial overlap: 0-based", inter0, " 1-based", inter1)
    BASE = 0 if inter0 >= inter1 else 1
print("  using BASE =", BASE)


def w2_of(diag):
    a = diag["alpha_G_phi"].to_numpy(np.float64)
    l = diag["L_cat_with_bh"].to_numpy(np.float64)
    b = diag["B_num_wbh"].to_numpy(np.float64)
    live = l > 0.0
    a2 = a[live] * l[live]
    den = a2 + b[live]
    w = np.where(den != 0.0, a2 / den, 0.0)
    return w, live


# ---- Per-seed accumulation --------------------------------------------------
rows = []
for seed in SEEDS:
    prep, diag = load_seed("bt", seed)
    n_drawn = len(prep)
    assert n_drawn == 200, (seed, n_drawn)
    w, live = w2_of(diag)
    ev = diag["event_idx"].astype(int).to_numpy()[live] - BASE
    z_ev = prep["z_true"].to_numpy(np.float64)[ev]
    s_ev = s_phi(z_ev)
    s_all = s_phi(prep["z_true"].to_numpy(np.float64))

    om_id = 1.0 - w
    om_br = (1.0 - w) / (1.0 + (R2 - 1.0) * w)

    lhs_id = C2_STAR / n_drawn * om_id.sum()          # replicate stage_lhs2d
    lhs_br = C2_STAR / n_drawn * om_br.sum()          # replicate scorer :675

    rec = dict(
        seed=seed, n_live=int(live.sum()),
        lhs_id=lhs_id, lhs_br=lhs_br,
        A_id=om_id.sum(), C_id=(om_id / s_ev).sum(),
        A_br=om_br.sum(), C_br=(om_br / s_ev).sum(),
        B=(1.0 / s_all).sum(), n=n_drawn,
        s_min_all=float(s_all.min()), s_med_acc=float(np.median(s_ev)),
        # diagnostics: arithmetic means
        s_wmean_id=float((om_id * s_ev).sum() / om_id.sum()),
        s_mean_all=float(s_all.mean()),
        z_wmean_id=float((om_id * z_ev).sum() / om_id.sum()),
        z_mean_all=float(prep["z_true"].mean()),
    )
    rec["R_id"] = rec["A_id"] * rec["B"] / (rec["n"] * rec["C_id"])
    rec["R_br"] = rec["A_br"] * rec["B"] / (rec["n"] * rec["C_br"])
    rows.append(rec)

df = pd.DataFrame(rows)


def msem(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


# frozen numbers (PA-2D-9)
LHS_ID, SE_LHS_ID = 0.00500770, 0.00011615
RHS_ID, SE_RHS_ID = 0.01451300, 0.00045293
LHS_BR, SE_LHS_BR = 0.00332207, 0.00009164
RHS_BR, SE_RHS_BR = 0.00908280, 0.00023752

obs_id = LHS_ID / RHS_ID
obs_id_se = obs_id * np.hypot(SE_LHS_ID / LHS_ID, SE_RHS_ID / RHS_ID)
obs_br = LHS_BR / RHS_BR
obs_br_se = obs_br * np.hypot(SE_LHS_BR / LHS_BR, SE_RHS_BR / RHS_BR)

print("\n== replication check (my per-seed LHS vs frozen) ==")
print("  LHS2(id)  mine:", msem(df.lhs_id), " frozen:", (LHS_ID, SE_LHS_ID))
print("  LHS2(BR)  mine:", msem(df.lhs_br), " frozen:", (LHS_BR, SE_LHS_BR))

print("\n== per-seed predicted ratios ==")
Rid_m, Rid_se = msem(df.R_id)
Rbr_m, Rbr_se = msem(df.R_br)

# pooled (sums across seeds) — more stable for the harmonic tail
A_id, C_id = df.A_id.sum(), df.C_id.sum()
A_br, C_br = df.A_br.sum(), df.C_br.sum()
B, N = df.B.sum(), df.n.sum()
Rid_pool = A_id * B / (N * C_id)
Rbr_pool = A_br * B / (N * C_br)

print(f"  R_pred(identity): per-seed {Rid_m:.5f} +/- {Rid_se:.5f}   pooled {Rid_pool:.5f}")
print(f"  R_pred(BR):       per-seed {Rbr_m:.5f} +/- {Rbr_se:.5f}   pooled {Rbr_pool:.5f}")
print(f"  observed LHS/RHS: identity {obs_id:.5f} +/- {obs_id_se:.5f}"
      f"   BR {obs_br:.5f} +/- {obs_br_se:.5f}")
print(f"  crude 1/r2 = {1 / R2:.5f}")

print("\n== corrected-LHS residual test (venue-corrected vs crude x r2) ==")
for name, lhs, se_l, rhs, se_r, Rm, Rse in [
    ("identity", LHS_ID, SE_LHS_ID, RHS_ID, SE_RHS_ID, Rid_m, Rid_se),
    ("BR",       LHS_BR, SE_LHS_BR, RHS_BR, SE_RHS_BR, Rbr_m, Rbr_se),
]:
    corr = lhs / Rm
    corr_se = corr * np.hypot(se_l / lhs, Rse / Rm)
    resid = corr - rhs
    resid_se = np.hypot(corr_se, se_r)
    crude = lhs * R2
    crude_resid = crude - rhs
    print(f"  {name}: venue-corrected LHS = {corr:.6f} +/- {corr_se:.6f}; "
          f"residual vs RHS = {resid:+.6f} +/- {resid_se:.6f}  "
          f"(crude x r2 residual {crude_resid:+.6f}); band eps2 = 0.001914")

print("\n== diagnostics ==")
print("  <S_phi> arithmetic, (1-w2)1_F0-weighted:", msem(df.s_wmean_id))
print("  <S_phi> arithmetic, all-accepted:       ", msem(df.s_mean_all))
print("  z means: weighted", msem(df.z_wmean_id), " all", msem(df.z_mean_all))
print("  min S_phi over all accepted (worst seed):", df.s_min_all.min())
print("  harmonic <S_phi>_m1 = N/B =", N / B)

df.to_csv(SCRATCH / "venue_drift_per_seed.csv", index=False)
json.dump(
    {
        "R_pred_identity": {"mean": Rid_m, "sem": Rid_se, "pooled": Rid_pool},
        "R_pred_BR": {"mean": Rbr_m, "sem": Rbr_se, "pooled": Rbr_pool},
        "obs_identity": {"ratio": obs_id, "se": obs_id_se},
        "obs_BR": {"ratio": obs_br, "se": obs_br_se},
        "crude_1_over_r2": 1 / R2,
    },
    open(SCRATCH / "venue_drift_adjudication.json", "w"),
    indent=2,
)
print("\nwrote per-seed CSV + summary JSON to scratchpad")
