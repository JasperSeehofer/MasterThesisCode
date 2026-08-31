# WAVE-3 BLIND HEAD READOUT — A14 DELTA READ — 2026-08-31

Launched under rows #278/#279; registration `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §0.2
(F2) / §8 (A14, T_mat = 0.008); baseline certified by the C0′ off-gate PASS (row #281,
`REGISTRATION_C0_BASELINE_GATE_20260829.md` §14). Jobs 6746275 (iiib) + 6746276 (joint_r1), 41
tasks each, all COMPLETED (sacct dump: `sacct_dump_20260829_31.txt`, 84/84 wave-3 rows); commit
at run `1e092e82`; pins STOP-gated in every task. Retrieved to `wave3_20260830/{iiib,joint_r1}/`
(one transient Lustre ENODATA during the first rsync, clean on resume; cluster-side readability
0 failures).

**Pre-read STOP-checks (§1.3/§5 of the measurement doc):** both venues: 41 distinct h
(H_GRID_41), 1588 distinct events, ZERO non-positive `combined_no_bh`/`combined_with_bh`
(no sentinel signal).

**Scorer:** the frozen T0 CSV convention (Σ ln combined over events per h; discrete posterior on
the 41-grid) — same scorer as the banked row #213 readout.

| venue | channel | banked mean_h | wave-3 mean_h | Δmean_h | verdict |
|---|---|---|---|---|---|
| iiib | 2D (with-BH) | 0.66643 | 0.66855 | **+0.002127** | **A14 PASS** (≤ 0.008) |
| iiib | 1D (no-BH) | 0.60532 | 0.60532 | +0.000000 | exact-zero (leg untouched, as gated) |
| joint_r1 | 2D (with-BH) | 0.66622 | 0.66974 | **+0.003519** | **A14 PASS** (≤ 0.008) |
| joint_r1 | 1D (no-BH) | 0.61189 | 0.61189 | +0.000000 | exact-zero |

**Reading:** the row #223 production adoption (`catalogue_numerator_survival_2d` default
`off`→`mz_sel`, center `eff`) is **NOT MATERIAL** at the registered threshold on both venues —
the blind readout moves the 2D mean by +0.0021/+0.0035, a quarter to a half of T_mat, and the 1D
channel is bit-stable. MAP moves: iiib 0.665→0.665 (none), joint_r1 0.660→0.665 (one grid step).

**A4 status (per row #280's restated form):** conditions met on this read — C0′ gate PASSED
(comparand = banked baseline, valid), both venues inside T_mat — but ratification remains
**pending falsifier (ii)** (class-G fleet Option A′ rung 1, not yet run). A4 therefore RETURNS
to the author with these numbers rather than auto-ratifying. Tree-1 verifier item 20's input now
exists; the part-2 verifier append can run.

---

## CORRECTION NOTE (append-only, 2026-08-31 ~13:00) — scorer weights

The item-20 end verifier (opus; PART 2 appended to `END_VERIFIER_REPORT_PART1_20260830.md`)
found the table above used UNIT grid weights, not the frozen T0 scorer's `np.gradient`
trapezoid weights (H_GRID_41 is non-uniform: 0.010 wings / 0.005 peak), and the "banked mean_h"
column therefore does not equal the row #213 published numbers. Orchestrator re-derived with the
reference implementation (`bscale_counterfactual_exploratory.py:23-30`) and reproduces the
verifier exactly. **Corrected numbers of record:**

| venue | banked 2D mean_h | wave-3 2D mean_h | Δmean_h | verdict |
|---|---|---|---|---|
| iiib | 0.663347 | 0.665854 | **+0.002507** | **A14 PASS** (31.3 % of T_mat) |
| joint_r1 | 0.663013 | 0.667127 | **+0.004114** | **A14 PASS** (51.4 % of T_mat) |

1D: bit-identical at all 41 nodes, both venues (stronger than the rounding-identity reported
above). MAP moves are weighting-independent (iiib none; joint_r1 0.660→0.665). The corrected
iiib delta lands on the §8 registered point prediction (≈ +0.0025). **The A14 PASS verdict
stands; only the numbers change.** Verifier [DO] adopted: the T0 scorer is to be frozen as an
importable helper with the §C.1 numbers as a regression test before the next readout.
