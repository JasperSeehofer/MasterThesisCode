#!/usr/bin/env python3
"""
Item 14 (B8.2 [CAL] harness design note) — independent re-derivation of the
"honest cost correction" decisive number.

Re-executes the arithmetic FROM the source numbers quoted in:
  - results/.../SYNTHESIS_DOCKET_1_20260829.md:202  (docket anchor)
  - results/.../hier_s0_registered_run/logs/s0a_seed900101_full.log:875 (truth-node timing)
  - B8_2_HARNESS_DESIGN_20260829.md §0, §6 (production N, universes, bracket)

Never trusts the record's restated arithmetic -- recomputes from the primitive
numbers only.
"""

# ---------------------------------------------------------------------------
# 1. Docket anchor: "24 arms x 65 s x 14 cores ~ 6 CPU-h per harness sweep"
#    (SYNTHESIS_DOCKET_1_20260829.md:202)
# ---------------------------------------------------------------------------
docket_arms = 24
docket_wall_s_per_arm = 65.0
docket_cores = 14
docket_cpu_h = docket_arms * docket_wall_s_per_arm * docket_cores / 3600.0
print(f"[1] docket anchor CPU-h = {docket_cpu_h:.4f}  (claimed '~6')")

# ---------------------------------------------------------------------------
# 2. Truth-node timing anchor from the actual log line (source of record):
#    s0a_seed900101_full.log:875
#    "[S0-A seed=900101 node=truth theta=(0.0,1.0)] n_events=106 evaluate_s=64.73 wall_s=67.72"
# ---------------------------------------------------------------------------
anchor_n_events = 106
anchor_evaluate_s = 64.73
anchor_cores = 14
anchor_cpu_h = anchor_evaluate_s * anchor_cores / 3600.0
print(f"[2] single-call anchor CPU-h at N={anchor_n_events}: {anchor_cpu_h:.4f} (note claims '~0.25')")

# Sanity check against the OTHER n_events=106 line in the SAME log
# (node=b_plus, theta=(0.02,1.0)) -- same N, different theta:
bplus_evaluate_s = 1190.93
ratio_theta = bplus_evaluate_s / anchor_evaluate_s
print(f"[2b] SAME N=106, different theta (b_plus): evaluate_s={bplus_evaluate_s}s "
      f"-> {ratio_theta:.2f}x the truth-node cost at IDENTICAL N.")
print("     (Harness pins theta=(0,1) == the 'truth' node's theta per B8.2 SS3/SS7 A10 "
      "invariants, so this 18x outlier does not apply to the harness's own bracket -- "
      "but the note never states this check.)")

# ---------------------------------------------------------------------------
# 3. Linear-scaling extrapolation to production N=1588
#    (64.73 * 1588 / 106 = 970 s wall; note's own arithmetic)
# ---------------------------------------------------------------------------
production_N = 1588
linear_wall_s = anchor_evaluate_s * production_N / anchor_n_events
linear_cpu_h = linear_wall_s * anchor_cores / 3600.0
print(f"[3] linear-scaling extrapolation: wall_s={linear_wall_s:.2f} "
      f"(note claims '970 s'), CPU-h={linear_cpu_h:.4f} (note claims '3.8')")

# Fixed-cost-dominated floor claimed by the note: 1.0 CPU-h/universe.
# This is NOT independently derivable from the two anchors alone (106 and
# 174-188 events both ~65s) -- it is a judgment-call floor, not a formula.
fixed_floor_cpu_h = 1.0
linear_ceiling_cpu_h = linear_cpu_h  # ~3.77

print(f"[3b] per-universe bracket used downstream: [{fixed_floor_cpu_h}, {linear_ceiling_cpu_h:.2f}] CPU-h "
      "(low end is an UNDERIVED judgment floor, not a formula output)")

# ---------------------------------------------------------------------------
# 4. Mandatory-cell totals -- two ways to read "mandatory total" from SS6's own table
# ---------------------------------------------------------------------------
n_U_cellS = 100
n_U_cellT = 25

cellS_lo, cellS_hi = n_U_cellS * fixed_floor_cpu_h, n_U_cellS * linear_ceiling_cpu_h
cellT_lo, cellT_hi = n_U_cellT * fixed_floor_cpu_h, n_U_cellT * linear_ceiling_cpu_h
print(f"[4a] Cell S (n_U={n_U_cellS}) CPU-h = [{cellS_lo:.1f}, {cellS_hi:.1f}]  (table says '100-380')")
print(f"[4b] Cell T (n_U={n_U_cellT}) CPU-h = [{cellT_lo:.1f}, {cellT_hi:.1f}]  (table says '25-95')")

cellS_plus_T_lo = cellS_lo + cellT_lo
cellS_plus_T_hi = cellS_hi + cellT_hi
print(f"[4c] Cell S + Cell T ONLY: [{cellS_plus_T_lo:.1f}, {cellS_plus_T_hi:.1f}] CPU-h "
      f"(note's stated mandatory total: '130-475')")

# Other table rows (SS6): smoke, PROD-A0 gate, N-ladder, pilot S, pilot T
smoke_lo, smoke_hi = 0.5, 1.25          # 5 x N=20, sub-anchor cost, soft estimate
prodA0_lo, prodA0_hi = fixed_floor_cpu_h, linear_ceiling_cpu_h   # 1 universe at N=1588
# N-ladder: N=106 (=anchor, 0.2517), N=400 (bracket fixed..linear), N=1588 (bracket)
n400_lo = anchor_cpu_h  # fixed-cost-dominated floor same as anchor
n400_hi = anchor_evaluate_s * 400 / anchor_n_events * anchor_cores / 3600.0
ladder_lo = anchor_cpu_h + n400_lo + fixed_floor_cpu_h
ladder_hi = anchor_cpu_h + n400_hi + linear_ceiling_cpu_h
pilotS = n_U_cellS * anchor_cpu_h   # pilot is at N=200, close to anchor N -> single value
pilotT = n_U_cellT * anchor_cpu_h
print(f"[4d] smoke [{smoke_lo},{smoke_hi}], PROD-A0 [{prodA0_lo:.2f},{prodA0_hi:.2f}], "
      f"N-ladder [{ladder_lo:.2f},{ladder_hi:.2f}], pilotS={pilotS:.2f}, pilotT={pilotT:.2f}")

full_table_sum_lo = smoke_lo + prodA0_lo + ladder_lo + pilotS + pilotT + cellS_lo + cellT_lo
full_table_sum_hi = smoke_hi + prodA0_hi + ladder_hi + pilotS + pilotT + cellS_hi + cellT_hi
print(f"[4e] FULL literal sum of every row in SS6's table: [{full_table_sum_lo:.1f}, {full_table_sum_hi:.1f}] CPU-h")
print("     -> does NOT equal the document's own '130-475' mandatory-total line "
      f"(discrepancy: low {full_table_sum_lo-130:.1f} CPU-h / {(full_table_sum_lo/130-1)*100:.0f}%, "
      f"high {full_table_sum_hi-475:.1f} CPU-h / {(full_table_sum_hi/475-1)*100:.0f}%)")

# ---------------------------------------------------------------------------
# 5. The headline correction factor: 20-80x
# ---------------------------------------------------------------------------
factor_lo_using_stated = cellS_plus_T_lo / docket_cpu_h
factor_hi_using_stated = cellS_plus_T_hi / docket_cpu_h
print(f"[5a] correction factor via Cell S+T only: {factor_lo_using_stated:.1f}x - {factor_hi_using_stated:.1f}x "
      "(note claims '20-80x')")

factor_lo_full = full_table_sum_lo / docket_cpu_h
factor_hi_full = full_table_sum_hi / docket_cpu_h
print(f"[5b] correction factor via FULL table sum: {factor_lo_full:.1f}x - {factor_hi_full:.1f}x "
      "(bigger correction than claimed if setup/pilot rows count as 'mandatory')")

# Ratio-decomposition cross-check (universes ratio x per-universe-cost ratio)
universe_ratio = (n_U_cellS + n_U_cellT) / docket_arms
cost_ratio_lo = fixed_floor_cpu_h / (docket_wall_s_per_arm * docket_cores / 3600.0)
cost_ratio_hi = linear_ceiling_cpu_h / (docket_wall_s_per_arm * docket_cores / 3600.0)
print(f"[5c] decomposition check: universe_ratio={universe_ratio:.3f} "
      f"({n_U_cellS+n_U_cellT} vs {docket_arms} arms) x per-universe-cost-ratio "
      f"[{cost_ratio_lo:.2f}, {cost_ratio_hi:.2f}] "
      f"= [{universe_ratio*cost_ratio_lo:.1f}x, {universe_ratio*cost_ratio_hi:.1f}x]")

# ---------------------------------------------------------------------------
# 6. Wall-time cross-check at 14 cores
# ---------------------------------------------------------------------------
wall_lo_stated_scope = cellS_plus_T_lo / 14
wall_hi_stated_scope = cellS_plus_T_hi / 14
print(f"[6] Cell S+T CPU-h / 14 cores = [{wall_lo_stated_scope:.1f}, {wall_hi_stated_scope:.1f}] h wall "
      "(note's mandatory-total wall claim: '13-46 h')")
wall_lo_full = full_table_sum_lo / 14
wall_hi_full = full_table_sum_hi / 14
print(f"    FULL-sum CPU-h / 14 cores = [{wall_lo_full:.1f}, {wall_hi_full:.1f}] h wall")
print("    Neither scope reproduces '13-46 h' cleanly via CPU-h/14 -- the wall-time line "
      "is not shown to follow from the CPU-h bracket by the same divisor used elsewhere "
      "in the same table (Cell S row: 100-380 CPU-h -> 7-27 h wall IS exactly /14).")
