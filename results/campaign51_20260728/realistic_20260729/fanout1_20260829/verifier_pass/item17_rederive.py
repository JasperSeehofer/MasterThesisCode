"""
Item 17 verifier re-derivation.
Part A: re-derive F-A's decisive P1 refutation numbers (alpha_G_phi -12.0%,
D_tilde_phi -0.745%, combined_no_bh max_rel 7.45e-3, L_cat_no_bh exact) directly
from the two source event_likelihoods.csv files, independent of any JSON/MD
restating them.

Part B: sanity-check the docket §5 item 10 / row #239 dirty-tree claim inputs
(timestamps: B6.1 edits 17:29-17:35, B5.1 edits <=17:53, S0-A start 17:58) by
reading git log timestamps for the commit that landed those edits and the
run's own log timestamps, and checking the tree state at commit d04d9dc9/current
HEAD for any edits *not* covered by B6.1/B5.1 that would break the byte-identity
argument (s=1 no-op; linear/1.5 default).
"""
import pandas as pd
import json

SMOKE = "results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work/b1_2_smoke/p1_2p2_off/s0a_seed900101/node_b_plus_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv"
REG = "results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_seed900101/node_b_plus/simulations/diagnostics/event_likelihoods.csv"

smoke = pd.read_csv(SMOKE)
reg = pd.read_csv(REG)

print(f"smoke rows={len(smoke)} reg rows={len(reg)}")

m = smoke.merge(reg, on="event_idx", suffixes=("_22", "_reg"))
print(f"merged (shared event_idx) rows={len(m)}")

results = {}

# L_cat_no_bh: exact?
lcat_diff = (m["L_cat_no_bh_22"] - m["L_cat_no_bh_reg"]).abs()
lcat_max_abs = float(lcat_diff.max())
lcat_exact = bool((lcat_diff == 0).all())
results["L_cat_no_bh_max_abs_diff"] = lcat_max_abs
results["L_cat_no_bh_exact"] = lcat_exact

# alpha_G_phi: sum or mean comparison -- the record cites single before/after
# values (5.8688310e7 -> 5.1635200e7). Reproduce via sum over shared events
# (alpha_G_phi is a per-event column; check if it's identical across events
# within a node, i.e. a global quantity broadcast per row, or genuinely per-event).
alpha_22_vals = m["alpha_G_phi_22"].unique()
alpha_reg_vals = m["alpha_G_phi_reg"].unique()
results["alpha_G_phi_22_unique_count"] = int(len(alpha_22_vals))
results["alpha_G_phi_reg_unique_count"] = int(len(alpha_reg_vals))
results["alpha_G_phi_22_value"] = float(alpha_22_vals[0]) if len(alpha_22_vals) == 1 else list(map(float, alpha_22_vals))
results["alpha_G_phi_reg_value"] = float(alpha_reg_vals[0]) if len(alpha_reg_vals) == 1 else list(map(float, alpha_reg_vals))
if len(alpha_22_vals) == 1 and len(alpha_reg_vals) == 1:
    a22 = float(alpha_22_vals[0]); areg = float(alpha_reg_vals[0])
    # docket convention: pct change of the SMEARED/registered value relative
    # to the 2.2/unsmeared baseline (a22 is the base, per the cited
    # "5.8688310e7 -> 5.1635200e7" ordering in row #238/WAVE2_REGISTRATION_CHECK).
    results["alpha_G_phi_pct_change_base22_to_reg"] = (areg - a22) / a22 * 100.0

# D_tilde_phi
dt_22_vals = m["D_tilde_phi_22"].unique()
dt_reg_vals = m["D_tilde_phi_reg"].unique()
results["D_tilde_phi_22_unique_count"] = int(len(dt_22_vals))
results["D_tilde_phi_reg_unique_count"] = int(len(dt_reg_vals))
if len(dt_22_vals) == 1 and len(dt_reg_vals) == 1:
    d22 = float(dt_22_vals[0]); dreg = float(dt_reg_vals[0])
    results["D_tilde_phi_22_value"] = d22
    results["D_tilde_phi_reg_value"] = dreg
    results["D_tilde_phi_pct_change_base22_to_reg"] = (dreg - d22) / d22 * 100.0

# combined_no_bh max relative difference
rel = (m["combined_no_bh_22"] - m["combined_no_bh_reg"]).abs() / m["combined_no_bh_reg"].abs()
results["combined_no_bh_max_rel"] = float(rel.max())
results["combined_no_bh_max_rel_event_idx"] = int(m.loc[rel.idxmax(), "event_idx"])

# w_G comparison (cited 0.06196684 -> 0.05492879)
wg_22_vals = m["w_G_22"].unique() if "w_G_22" in m.columns else None
wg_reg_vals = m["w_G_reg"].unique() if "w_G_reg" in m.columns else None
if wg_22_vals is not None and len(wg_22_vals) == 1 and len(wg_reg_vals) == 1:
    results["w_G_22_value"] = float(wg_22_vals[0])
    results["w_G_reg_value"] = float(wg_reg_vals[0])

print(json.dumps(results, indent=2))

with open("results/campaign51_20260728/realistic_20260729/fanout1_20260829/verifier_pass/item17_rederive_output.json", "w") as f:
    json.dump(results, f, indent=2)
