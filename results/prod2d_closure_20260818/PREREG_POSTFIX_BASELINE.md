# PRE-REGISTRATION (lite) — post-B_scale-fix production baseline

**Date:** 2026-08-19 · **Scope:** the new production baseline posterior after the [PHYSICS]
B_scale removal (ledger rows #130–#132 era; runbook 22 supersession item 3). Registered
BEFORE launch. Continuity-check class: the expected values are fully pinned by the
verified banked-data counterfactual (`bscale_counterfactual_exploratory.py`), so this run
doubles as the implementation cross-check of the physics change on the full pipeline.

## Design

Cluster, `cluster/evaluate.sbatch` pattern: {iiib, joint_r1} × 41 h-values, derived form
(default flags), explicit `--selection_in_completion_numerator off` (basis of record),
same banked seed61000 inputs/symlinks/seeds as the runs of record. Run dirs
`run_20260819_postfix_baseline_{venue}`. ~82 tasks × ~3 min ≈ 4 CPU-h.

## Registered bands

- **N-B0 (continuity, scored):** T0-convention 2D mean_h must reproduce the counterfactual
  prediction **0.6771 (iiib) / 0.6788 (joint_r1)** to within ±0.0030; 1D mean_h within
  ±0.0030 of 0.6010/0.6020. Failure ⇒ STOP: implementation discrepancy between the shipped
  derived path and the registered counterfactual — investigate before ANY use.
- **N-B1:** run_metadata cli_args diff vs the fusion-counterfactual off referent shows only
  whitelisted keys (as in the counterfactual prereg §2) plus `completion_b_scale` present
  (= "derived").
- **Products:** the new baselines of record (means, σ_h, MAP, per-event
  `event_likelihoods.csv` both venues) — the reference the re-ranked battery reads against.
  Presentation carries: 2D offset vs truth now expected ≈ **−0.053/−0.051** (below truth;
  the re-exposed base tilt = the open tilt ledger; correctness-over-bias-removal applied).
- P7-8 single-realization disclosure carried.

## VERDICT

*(append-only after execution)*

**VERDICT (2026-08-19, appended after execution):** jobs 6372475/6372476, 82/82 COMPLETED.
**N-B0 PASS, exact:** 2D mean_h = 0.6771 (iiib) / 0.6788 (joint_r1) — equal to the
registered counterfactual prediction to 4 decimals; 1D 0.6010/0.6020 likewise. σ_h(2D) =
0.0239/0.0225, MAP 0.675/0.675. **N-B1 PASS** (completion_b_scale=derived, sel=off, commit
e65d263c). **New baselines of record:** production 2D offset vs truth = **−0.0529 (iiib) /
−0.0512 (joint_r1)** — below truth as predicted; the re-exposed base tilt is the open tilt
ledger. The [PHYSICS] derived-form implementation is cross-validated end-to-end on the full
pipeline (prediction → shipped code → cluster fleet, exact agreement).
