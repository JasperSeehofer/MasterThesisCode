"""Independent re-derivation for verifier item 1 (B1.1 wave-1 record).

Re-derives, from raw source only (no import of hier_s0_driver.py's own
gate_parity/gate_eng helpers, no trust of the record's own restated numbers):

1. GATE PARITY residual: registered-run truth node vs banked bc_900101 CSV,
   at h=0.73, on shared event_idx values (combined_no_bh, combined_with_bh).
2. The timing claim: parse the raw log for the truth and b_plus evaluate_s
   values, and recompute the 18.6x ratio against the registered anchor
   63.97 s (from PREREGISTRATION_HIER_HTHETA_20260826.md sec 7.1: mean of
   64.996/62.944).
3. The ternary at bayesian_statistics.py (current HEAD): read the actual
   source text of the global_denom_no_bh assignment, independent of any
   line-number citation, and confirm which branch it takes under
   catalogue_global_selection="phi".
"""

import re
from pathlib import Path

import pandas as pd

REPO = Path("/home/jasper/Repositories/darksiren-emri")
FANOUT = REPO / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"

# ---------------------------------------------------------------------------
# 1. GATE PARITY residual, full N=106, seed 900101
# ---------------------------------------------------------------------------
truth_csv = (
    FANOUT
    / "hier_s0_registered_run/s0a_seed900101/node_truth/simulations/diagnostics/event_likelihoods.csv"
)
banked_csv = (
    REPO
    / "results/campaign51_20260728/realistic_20260729/p3_b0_work/bc_900101_work/seed900101/simulations/diagnostics/event_likelihoods.csv"
)

df_t = pd.read_csv(truth_csv)
df_b = pd.read_csv(banked_csv)

df_t_h = df_t[abs(df_t["h"] - 0.73) < 1e-9]
df_b_h = df_b[abs(df_b["h"] - 0.73) < 1e-9]

merged = df_t_h.merge(df_b_h, on="event_idx", suffixes=("_reg", "_bank"))
n_shared = len(merged)

results = {}
for col in ["combined_no_bh", "combined_with_bh"]:
    a = merged[f"{col}_reg"].to_numpy()
    b = merged[f"{col}_bank"].to_numpy()
    abs_diff = abs(a - b)
    rel_diff = abs_diff / abs(b)
    results[col] = (n_shared, abs_diff.max(), rel_diff.max())

print("=== GATE PARITY re-derivation (independent pandas, zero trust of driver code) ===")
print(f"truth CSV: {truth_csv}")
print(f"banked CSV: {banked_csv}")
print(f"n rows @ h=0.73: registered={len(df_t_h)}, banked={len(df_b_h)}, shared={n_shared}")
for col, (n, mx_abs, mx_rel) in results.items():
    print(f"  {col}: n={n} max_abs_diff={mx_abs:.6e} max_rel_diff={mx_rel:.6e}")

# ---------------------------------------------------------------------------
# 2. Timing claim
# ---------------------------------------------------------------------------
log_path = FANOUT / "hier_s0_registered_run/logs/s0a_seed900101_full.log"
text = log_path.read_text(errors="replace")

truth_m = re.search(r"node=truth theta=\(0\.0,1\.0\)\].*?evaluate_s=([\d.]+)", text)
bplus_m = re.search(r"node=b_plus theta=\(0\.02,1\.0\)\].*?evaluate_s=([\d.]+)", text)

truth_evaluate_s = float(truth_m.group(1))
bplus_evaluate_s = float(bplus_m.group(1))

anchor_1 = 64.996
anchor_2 = 62.944
anchor_mean = (anchor_1 + anchor_2) / 2.0

print()
print("=== Timing re-derivation (raw log parse, independent regex) ===")
print(f"log: {log_path}")
print(f"truth evaluate_s = {truth_evaluate_s}")
print(f"b_plus evaluate_s = {bplus_evaluate_s}")
print(f"registered anchor mean (64.996+62.944)/2 = {anchor_mean}")
print(f"b_plus / anchor_mean = {bplus_evaluate_s / anchor_mean:.4f}x")
print(f"b_plus / truth_evaluate_s = {bplus_evaluate_s / truth_evaluate_s:.4f}x")
print(f"truth_evaluate_s / anchor_mean = {truth_evaluate_s / anchor_mean:.4f}x (should be ~1)")

# ---------------------------------------------------------------------------
# 3. The ternary itself, read as text, no line-number reliance
# ---------------------------------------------------------------------------
bs_path = REPO / "darksiren_emri/bayesian_inference/bayesian_statistics.py"
bs_text = bs_path.read_text()

m = re.search(
    r"global_denom_no_bh:\s*float\s*=\s*\(\s*"
    r"(self\._global_cat_selection_phi\.get\(self\.h,\s*0\.0\))\s*"
    r"if\s+(getattr\(self,\s*\"_catalogue_global_selection\",\s*\"s3d\"\)\s*==\s*\"phi\")\s*"
    r"else\s+(self\._global_cat_denom_no_bh\.get\(self\.h,\s*0\.0\))\s*\)",
    bs_text,
)
print()
print("=== Ternary re-derivation (regex over current bayesian_statistics.py, no line numbers) ===")
if m:
    print("MATCHED. global_denom_no_bh ternary reads:")
    print(f"  if-true branch (phi):  {m.group(1)}")
    print(f"  condition:             {m.group(2)}")
    print(f"  else branch (default): {m.group(3)}")
    # find actual current line number for disclosure
    line_no = bs_text[: m.start()].count("\n") + 1
    print(f"  current HEAD line number of this assignment: {line_no}")
else:
    print("NO MATCH -- ternary text differs from what the record describes (possible refutation).")

# Now check whether combined_without_bh_mass (phi/absolute_marginal branch) is
# built from D_tilde_phi (theta-engaged via global_denom_with_bh) rather than
# from global_denom_no_bh directly -- this is the F-A mechanism claim.
m2 = re.search(
    r"combined_without_bh_mass\s*=\s*float\(\s*"
    r"\(beta_G_phi \* L_cat_without_bh_mass \+ B_num_phi\) / D_tilde_phi\s*\)",
    bs_text,
)
print()
print("=== F-A mechanism re-derivation: is combined_no_bh built from D_tilde_phi? ===")
print("MATCHED" if m2 else "NO MATCH")
if m2:
    line_no2 = bs_text[: m2.start()].count("\n") + 1
    print(f"  current HEAD line number: {line_no2}")

m3 = re.search(
    r"path_a\s*=\s*path_a_mixture_objects\(\s*"
    r"beta_G_phi,\s*beta_Gbar_phi,\s*sigma_phi,\s*global_denom_with_bh\s*\)",
    bs_text,
)
print("path_a_mixture_objects(..., sigma_4d=global_denom_with_bh) call found:", bool(m3))
if m3:
    line_no3 = bs_text[: m3.start()].count("\n") + 1
    print(f"  current HEAD line number: {line_no3}")

print()
print("=== hier_s0_driver.py ln-transform citation check (must-fix item 3) ===")
driver_path = FANOUT / "hier_s0_driver.py"
driver_text = driver_path.read_text()
m4 = re.search(r"np\.where\(vals > 0\.0, np\.log\(vals\), np\.nan\)", driver_text)
if m4:
    ln_no = driver_text[: m4.start()].count("\n") + 1
    print(f"np.where(...np.log...) actually at driver.py line {ln_no} (current HEAD)")
