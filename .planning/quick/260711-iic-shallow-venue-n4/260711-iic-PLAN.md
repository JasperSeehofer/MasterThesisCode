---
quick_id: 260711-iic
slug: shallow-venue-n4
status: complete
date: 2026-07-11
branch: physics/zero-host-completion-fallback
---

# Quick Task 260711-iic — N-4: the separate shallow-venue +0.0132/+0.0138 regime

## Goal

Characterize the SEPARATE shallow-venue 1D residual (seed600: comp_frac ≈ 0.4%,
z_median 0.046, era-corrected +0.0138 / raw +0.0132) via the two cheap N-4 probes
(`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`). Harness-only, no `/physics-change`.
This is DISTINCT from the deep-incompleteness floor closed in 260711-hx1.

## Tasks

1. **feat** — make the harness detection horizon tunable: add `d50_gpc`/`w_pdet_gpc`
   (config + `--d50-gpc`/`--w-pdet-gpc`) to `pp_coverage.py`, threaded through
   `detection_probability` and every call site (population sampler, D(h), beta_G,
   p_det factors). Default (1.85/0.30) bit-identical (regression guard: exact zs=0.3
   sz=0.035 results byte-identical to committed exactmode).

2. **results** — pre-registered RUNBOOK (written BEFORE runs):
   - **(a) depth ladder** d50 ∈ {1.85…0.23} (z_med 0.28→0.044), w=0.162·d50, calibrated
     volume kernel + no truncation → does the estimator develop a +0.013 offset as the
     venue shallows? + **Set B** σ_z ∈ {0.005,0.015,0.035} at the shallow rung to localize.
   - **(b) jackknife** on the on-disk seed600 run_live per-event JSONs (no re-eval) —
     DONE inline: reproduces +0.0132; residual is broad/systematic, NOT outlier-driven.

3. **docs** — SUMMARY (a+b) with the P-A/P-B verdict, STATE row, [L7]/N-4 ledger update.

## Pre-registered predictions (full form in the RUNBOOK)
- **P-A calibrated-stays** ⇒ shallow +0.0132 is seed600-DATA-specific (cross-seed needs campaign).
- **P-B shallow-bias** ⇒ estimator-intrinsic low-z break (truncated volume kernel when σ_z/z~1);
  Set B: bias scaling with σ_z ⇒ σ_z/z Eddington effect.

## must_haves
- Default d50/w byte-identical to pre-probe harness.
- Depth ladder holds the estimator CALIBRATED (volume, no truncation) and varies ONLY depth.
- SUMMARY reports both (a) depth-sweep and (b) jackknife; honest systematic-vs-scatter caveat (needs campaign).
