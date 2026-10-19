#!/usr/bin/env python3
"""
Item 16 (B1.2 PA-HIER-31) verifier re-derivation.

Re-derives the F-A finding's decisive numbers FROM SOURCE CSVs (raw event-likelihood
diagnostics), independently of any prose restating them in
PREREGISTRATION_HIER_HTHETA_20260826.md or WAVE2_REGISTRATION_CHECK_20260829.md.

Comparison: seed 900101, node b_plus (b=+0.02, h=0.73), 9 events shared between:
  - "2.2"/unsmeared form: hier_s0_work/b1_2_smoke/p1_2p2_off/s0a_seed900101/
        node_b_plus_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv
  - "all"/smeared form:   hier_s0_registered_run/s0a_seed900101/node_b_plus/
        simulations/diagnostics/event_likelihoods.csv
"""

import pandas as pd
import numpy as np
import os

ROOT = "results/campaign51_20260728/realistic_20260729/fanout1_20260829"

path_22_unsmeared = os.path.join(
    ROOT,
    "hier_s0_work/b1_2_smoke/p1_2p2_off/s0a_seed900101/"
    "node_b_plus_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv",
)
path_all_smeared = os.path.join(
    ROOT,
    "hier_s0_registered_run/s0a_seed900101/node_b_plus/"
    "simulations/diagnostics/event_likelihoods.csv",
)

for p in (path_22_unsmeared, path_all_smeared):
    assert os.path.exists(p), f"missing source file: {p}"

df_22 = pd.read_csv(path_22_unsmeared)
df_all = pd.read_csv(path_all_smeared)

print(f"'2.2'/unsmeared rows: {len(df_22)}  (file: {path_22_unsmeared})")
print(f"'all'/smeared  rows: {len(df_all)}  (file: {path_all_smeared})")

merged = df_22.merge(df_all, on="event_idx", how="inner", suffixes=("_22", "_all"))
print(f"shared event_idx count: {len(merged)}")
assert len(merged) == 9, f"expected 9 shared events per the registered claim, got {len(merged)}"

numeric_cols = [c for c in df_22.columns if c != "event_idx" and c != "h"]

print("\n--- per-column max relative difference (2.2/unsmeared vs all/smeared), 9 shared events ---")
results = {}
for col in numeric_cols:
    a = merged[f"{col}_22"].to_numpy(dtype=float)
    b = merged[f"{col}_all"].to_numpy(dtype=float)
    # relative diff with zero-safe denominator (matches "bit-identical" style checks used in the record)
    denom = np.where(np.abs(b) > 0, np.abs(b), np.nan)
    rel = np.abs(a - b) / denom
    rel_valid = rel[~np.isnan(rel)]
    max_rel = np.nanmax(rel) if len(rel_valid) else 0.0
    max_abs = np.max(np.abs(a - b))
    results[col] = (max_rel, max_abs)
    flag = ""
    if col in ("L_cat_no_bh", "combined_no_bh", "alpha_G_phi", "D_tilde_phi", "w_G"):
        flag = "  <== decisive column"
    print(f"{col:25s} max_rel={max_rel:.6e}  max_abs={max_abs:.6e}{flag}")

print("\n--- decisive numbers vs registered claim ---")


def pct_change(col):
    a = merged[f"{col}_22"].to_numpy(dtype=float)
    b = merged[f"{col}_all"].to_numpy(dtype=float)
    # registered numbers are single representative values (event with the largest/labelled shift);
    # report both the aggregate max_rel and the per-event values for full disclosure.
    return a, b


print(f"L_cat_no_bh max_rel (registered: 0.0, bit-identical): {results['L_cat_no_bh'][0]:.6e}")
print(f"combined_no_bh max_rel (registered: 7.45e-3 / 7.447e-3): {results['combined_no_bh'][0]:.6e}")

a_alpha, b_alpha = pct_change("alpha_G_phi")
print("\nalpha_G_phi per event (2.2/unsmeared -> all/smeared):")
for i, (x, y) in enumerate(zip(a_alpha, b_alpha)):
    pct = 100.0 * (y - x) / x if x != 0 else float("nan")
    print(f"  event {merged['event_idx'].iloc[i]}: {x:.7e} -> {y:.7e}  ({pct:+.3f}%)")

a_dt, b_dt = pct_change("D_tilde_phi")
print("\nD_tilde_phi per event (2.2/unsmeared -> all/smeared):")
for i, (x, y) in enumerate(zip(a_dt, b_dt)):
    pct = 100.0 * (y - x) / x if x != 0 else float("nan")
    print(f"  event {merged['event_idx'].iloc[i]}: {x:.7e} -> {y:.7e}  ({pct:+.3f}%)")

a_wg, b_wg = pct_change("w_G")
print("\nw_G per event (2.2/unsmeared -> all/smeared):")
for i, (x, y) in enumerate(zip(a_wg, b_wg)):
    pct = 100.0 * (y - x) / x if x != 0 else float("nan")
    print(f"  event {merged['event_idx'].iloc[i]}: {x:.7e} -> {y:.7e}  ({pct:+.3f}%)")

# Registered representative numbers (from PA-HIER-31(b) / WAVE2 check §0):
#   alpha_G_phi: 5.8688310e7 -> 5.1635200e7 (-12.0%)
#   D_tilde_phi: 9.470921e8 -> 9.40039e8 (-0.745%)
#   w_G: 0.06196684 -> 0.05492879
# These look like single-event (or mean) representative values; check whether they match
# any specific event or the mean across the 9 shared events.
print("\n--- checking whether the registered single-number citations match a specific event or the mean ---")
print(f"mean alpha_G_phi (2.2): {a_alpha.mean():.7e}  mean (all): {b_alpha.mean():.7e}")
print(f"mean D_tilde_phi (2.2): {a_dt.mean():.7e}  mean (all): {b_dt.mean():.7e}")
print(f"mean w_G (2.2): {a_wg.mean():.7e}  mean (all): {b_wg.mean():.7e}")

print("\nDONE")
