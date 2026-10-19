"""
Independent re-derivation for END-OF-FAN-OUT VERIFIER PASS item 11 (B7.2).

Written from scratch by the verifier -- does NOT import or call
fanout1_20260829/b7_2_readout.py. Reads only the raw CSVs. Re-implements the
gates and stencil independently to check the decisive numbers reported in
B7_2_TWIN_CF_READOUT_RECORD.md sec6 / b7_2_readout.json against a fresh
computation.
"""

import numpy as np
import pandas as pd

ARM_CSV = (
    "results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/"
    "simulations/diagnostics/event_likelihoods.csv"
)
BASE_CSV = (
    "results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/"
    "event_likelihoods.csv"
)

H4 = [0.660, 0.665, 0.670, 0.730]
I_HEAD_CLAIMED = 2965.0
T_MAT = 0.008
T_MAT_HALF = 0.004

arm = pd.read_csv(ARM_CSV)
base = pd.read_csv(BASE_CSV)

print("arm shape", arm.shape, "unique h", sorted(arm["h"].unique()))
print("base shape", base.shape, "unique h in H4 subset:",
      sorted(base.loc[np.isclose(base["h"].values[:, None], H4, atol=1e-9).any(axis=1), "h"].unique()))

# restrict base to H4 nodes only
base_h4 = base.loc[np.isclose(base["h"].to_numpy()[:, None], H4, atol=1e-9).any(axis=1)].copy()
arm_h4 = arm.copy()  # already only H4

for h in H4:
    na = (np.isclose(arm_h4["h"], h, atol=1e-9)).sum()
    nb = (np.isclose(base_h4["h"], h, atol=1e-9)).sum()
    print(f"h={h}: n_arm={na} n_base={nb}")

# ---------------------------------------------------------------------
# R1: ln L_cat,wbh^T <= ln L_cat,wbh^B for every (event,h)
# ---------------------------------------------------------------------
merged_all = arm_h4.merge(base_h4, on=["event_idx", "h"], suffixes=("_T", "_B"), how="inner")
print("\nR1: merged rows =", len(merged_all))

Lt = merged_all["L_cat_with_bh_T"].to_numpy()
Lb = merged_all["L_cat_with_bh_B"].to_numpy()
both_zero = (Lt == 0) & (Lb == 0)
with np.errstate(divide="ignore"):
    ln_t = np.log(Lt)
    ln_b = np.log(Lb)
finite_mask = np.isfinite(ln_t) & np.isfinite(ln_b)
viol = finite_mask & (ln_t > ln_b + 1e-12)
print("R1: n_rows_checked =", len(merged_all))
print("R1: n_violations =", int(viol.sum()))
print("R1: n_empty_candidate_equal_rows (both zero) =", int(both_zero.sum()))
print("R1 VERDICT:", "PASS" if viol.sum() == 0 else "FAIL/INSTRUMENT-DEFECT")

# ---------------------------------------------------------------------
# R2: A13 engagement at h=0.730
# ---------------------------------------------------------------------
h_engage = 0.730
a2 = arm_h4.loc[np.isclose(arm_h4["h"], h_engage, atol=1e-9)]
b2 = base_h4.loc[np.isclose(base_h4["h"], h_engage, atol=1e-9)]
m2 = a2.merge(b2, on="event_idx", suffixes=("_T", "_B"), how="inner")
active = m2["L_cat_with_bh_B"] > 0
n_active = int(active.sum())
with np.errstate(divide="ignore"):
    lt2 = np.log(m2.loc[active, "L_cat_with_bh_T"].to_numpy())
    lb2 = np.log(m2.loc[active, "L_cat_with_bh_B"].to_numpy())
finite2 = np.isfinite(lt2) & np.isfinite(lb2)
delta2 = np.abs(lt2[finite2] - lb2[finite2])
n_engaged = int((delta2 > 1e-6).sum())
n_considered = int(finite2.sum())
frac = n_engaged / n_considered if n_considered else 0.0
print(f"\nR2: n_active_rows={n_active} n_considered={n_considered} n_engaged={n_engaged} "
      f"engagement_fraction={frac}")
print("R2 VERDICT:", "PASS" if frac >= 0.95 else "STOP")

# ---------------------------------------------------------------------
# R6: 1D channel bit-identical
# ---------------------------------------------------------------------
r6_cols = ["L_cat_no_bh", "combined_no_bh"]
max_abs_overall = 0.0
for h in H4:
    a = arm_h4.loc[np.isclose(arm_h4["h"], h, atol=1e-9)]
    b = base_h4.loc[np.isclose(base_h4["h"], h, atol=1e-9)]
    m = a.merge(b, on="event_idx", suffixes=("_T", "_B"), how="inner")
    for col in r6_cols:
        diff = (m[f"{col}_T"] - m[f"{col}_B"]).to_numpy()
        ma = float(np.max(np.abs(diff))) if len(diff) else float("nan")
        max_abs_overall = max(max_abs_overall, ma if np.isfinite(ma) else 0.0)
        print(f"R6: h={h} col={col} max_abs={ma!r}")
print("R6 max_abs_overall =", max_abs_overall)
print("R6 VERDICT:", "PASS" if max_abs_overall <= 1e-12 else "INSTRUMENT-DEFECT")

# ---------------------------------------------------------------------
# Stencil: Delta ell(h) = sum_i ln[combined_with_bh^T / combined_with_bh^B]
# at h in {0.660, 0.665, 0.670}; central-difference slope + curvature.
# ---------------------------------------------------------------------
H3 = [0.660, 0.665, 0.670]
delta_ell = {}
for h in H3:
    a = arm_h4.loc[np.isclose(arm_h4["h"], h, atol=1e-9)]
    b = base_h4.loc[np.isclose(base_h4["h"], h, atol=1e-9)]
    m = a.merge(b, on="event_idx", suffixes=("_T", "_B"), how="inner")
    ct = m["combined_with_bh_T"].to_numpy()
    cb = m["combined_with_bh_B"].to_numpy()
    valid = (ct > 0) & (cb > 0) & np.isfinite(ct) & np.isfinite(cb)
    de = float(np.sum(np.log(ct[valid]) - np.log(cb[valid])))
    delta_ell[h] = de
    print(f"\nDelta_ell({h}) = {de!r}  n_events={int(valid.sum())}  n_dropped={int((~valid).sum())}")

ell_lo, ell_mid, ell_hi = delta_ell[0.660], delta_ell[0.665], delta_ell[0.670]
step = 0.665 - 0.660
d_ell_prime = (ell_hi - ell_lo) / (2 * step)
d_ell_dprime = (ell_hi - 2 * ell_mid + ell_lo) / (step**2)

print("\nstep =", step)
print("Delta_ell'(0.665)  =", d_ell_prime)
print("Delta_ell''(0.665) =", d_ell_dprime)

delta_mean_h_pred = d_ell_prime / I_HEAD_CLAIMED
print("\nI_HEAD (claimed, not independently re-derived from sigma_h source) =", I_HEAD_CLAIMED)
print("Delta_mean_h,pred = Delta_ell'/I_HEAD =", delta_mean_h_pred)

validity_ok = abs(d_ell_dprime) < 0.1 * I_HEAD_CLAIMED
print("validity |Delta_ell''| < 0.1*I_HEAD ?", validity_ok, f"({abs(d_ell_dprime)} vs {0.1*I_HEAD_CLAIMED})")

if not validity_ok:
    verdict = "AMBIGUOUS (validity condition violated)"
elif abs(delta_mean_h_pred) >= T_MAT:
    verdict = "MATERIAL-UP-PREDICTED" if delta_mean_h_pred > 0 else "MATERIAL-DOWN-PREDICTED"
elif abs(delta_mean_h_pred) <= T_MAT_HALF:
    verdict = "IMMATERIAL-PREDICTED"
else:
    verdict = "AMBIGUOUS (0.004 < |delta| < 0.008)"

print("\nVERDICT =", verdict)

print("\n--- comparison to claimed record values ---")
claims = {
    "delta_ell(0.660)": (-3.03067408140695, ell_lo),
    "delta_ell(0.665)": (-2.9931484145135636, ell_mid),
    "delta_ell(0.670)": (-2.956380531717331, ell_hi),
    "delta_ell_prime": (7.429354968961904, d_ell_prime),
    "delta_ell_dprime": (-30.31136388614181, d_ell_dprime),
    "delta_mean_h_pred": (0.002505684643832008, delta_mean_h_pred),
    "R1_violations": (0, int(viol.sum())),
    "R2_engagement_fraction": (1.0, frac),
    "R6_max_abs": (0.0, max_abs_overall),
}
for k, (claimed, mine) in claims.items():
    if isinstance(claimed, float):
        diff = abs(claimed - mine)
        ok = diff < 1e-6 * max(1.0, abs(claimed))
    else:
        ok = claimed == mine
        diff = None
    print(f"{k}: claimed={claimed!r} rederived={mine!r} diff={diff} MATCH={ok}")
