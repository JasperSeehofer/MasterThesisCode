# RUNBOOK — pp_coverage shallow-venue N-4 probe, 2026-07-11

**Provenance:** quick task `260711-iic-shallow-venue-n4`; handoff item **N-4** (the
separate shallow-venue 1D +0.0132/+0.0138 regime), `.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`
+ `.planning/BIAS-INVESTIGATION-20260710.md` §1/[L3]. Code adds `d50_gpc`/`w_pdet_gpc`
(config + `--d50-gpc`/`--w-pdet-gpc`) to `master_thesis_code/validation/pp_coverage.py`,
making the detection horizon tunable. Harness-only, no `/physics-change`.

**The shallow +0.0132 is a DIFFERENT regime from the deep floor** (260711-hx1): seed600
is comp_frac ≈ 0.4% (the deep membership-leak + noise-model mechanisms are ~absent),
z_median 0.046, z_max 0.12 — far shallower than the commission venue (z_median 0.28).
After the L_cat fix + Ω_m-era correction the seed600 1D residual is +0.0138 (raw +0.0132),
single-seed +2.6σ, and its cause ended at "p(G|D,H0)-weight or scatter" (weight exonerated).

## N-4 has two sub-probes

**(a) Venue-matched harness depth sweep (THIS RUNBOOK's runs):** does a CALIBRATED
estimator (volume kernel, NO z_support truncation, comp_frac ≈ 0) develop a +0.013-like
offset when the venue is made shallow (z_median → 0.046)? Depth is set by `--d50-gpc`
with `--w-pdet-gpc` scaled to keep a constant fractional roll-off (w = 0.162·d50, the
default ratio), so each rung is a self-similar Malmquist at a different depth. z_median
per rung (verified): d50 {1.85,1.0,0.6,0.4,0.30,0.23} → z_med {0.28,0.17,0.11,0.074,0.056,0.044}.
The shallowest rung (d50=0.23) matches seed600 (z_med 0.044). σ_z=0.035 gives σ_z/z_med
0.12 → 0.80 across the ladder (the fractional photo-z scatter grows as the venue shallows).

**(b) Jackknife/influence on the EXISTING seed600 per-event JSONs (DONE inline, no
re-eval):** `results/pv_correction_test_20260703/run_live/simulations/posteriors/`
reproduces the combined grid-MAP 0.745 and posterior mean 0.74320 (residual **+0.01320**,
= ledger raw). Verdict recorded in SUMMARY.md: the residual is **broad/systematic, NOT a
heavy-tailed outlier subset** — 62% of events tilt high, median per-event tilt positive,
Gini(|influence|)=0.65, 90% of |influence| spread across 52% of the sample, and removing
the highest-|tilt| events INCREASES the residual (the informative events are a
net-negative counterweight; the systematic high-drift lives in the shallow bulk).

## Pre-registered predictions for (a) — written BEFORE any depth-sweep run

Criteria: cov68 within ±0.085 of 0.68; SEM = map_std/√120; calibrated reference =
d50=1.85 (the commission-validated volume-kernel setting, bias ≈ 0).

- **P-A CALIBRATED-STAYS ⇒ the shallow +0.0132 is seed600-DATA-specific.** The
  volume-kernel/no-truncation estimator stays calibrated at every depth (|bias| < 2·SEM,
  cov68 nominal) down to z_med 0.044 ⇒ deep-venue calibration does NOT break with depth;
  the seed600 offset is a data/realization property, not an intrinsic shallow-estimator
  bias (cross-seed systematic-vs-scatter then genuinely needs the multi-seed campaign, as
  the handoff flags — do NOT force it locally).
- **P-B SHALLOW-BIAS ⇒ the shallow offset is ESTIMATOR-INTRINSIC at low z.** The
  estimator develops a positive MAP bias as d50 shrinks, reaching ~+0.01 near d50≈0.23
  (z_med 0.044) with cov68 degrading ⇒ the volume-kernel calibration that holds at deep z
  breaks at shallow z (candidate mechanism: the host-z kernel N(z;z_gal,σ_z)·w_pop is
  truncated at Z_MIN when σ_z/z ~ 1, so the volume/Eddington-in-z correction — derived
  assuming an un-truncated kernel — no longer cancels). Set B then localizes it in σ_z.

- **Set B (σ_z at the shallow rung):** at d50=0.23, sweep σ_z ∈ {0.005, 0.015, 0.035}.
  If the shallow bias (P-B) SCALES with σ_z (vanishes at σ_z=0.005) ⇒ it is the
  σ_z/z-driven truncated-kernel Eddington effect (matches the commission bare-vs-volume
  finding, now depth-amplified). If it persists at σ_z=0.005 ⇒ a depth effect independent
  of photo-z scatter.

## Common settings

`--kernel volume --n-realizations 120 --n-events 250 --truths 0.62 0.73 0.84 --seed 20260701`
(NO `--z-support` → comp_frac 0, the clean calibrated estimator). Output dir:
`results/pp_coverage_shallowvenue_20260711/`.

## Set A — depth ladder (σ_z = 0.035)

d50 ∈ {1.85, 1.0, 0.6, 0.4, 0.30, 0.23}, w = round(0.162·d50, 4):

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 \
  --d50-gpc {D50} --w-pdet-gpc {W} --truths 0.62 0.73 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_shallowvenue_20260711/pp_depth_d50{D50}.json
```

## Set B — σ_z at the shallow rung (d50 = 0.23, w = 0.0373)

σ_z ∈ {0.005, 0.015} (0.035 is the shallowest Set-A rung):

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --n-z-quad 320 \
  --d50-gpc 0.23 --w-pdet-gpc 0.0373 --truths 0.62 0.73 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_shallowvenue_20260711/pp_shallow_sz{SZ}.json
```
(σ_z=0.005 uses `--n-z-quad 320` so the narrow kernel is sampled by ≥4 pts/σ.)

## Regression guard

Default `--d50-gpc`/`--w-pdet-gpc` (1.85/0.30) reproduce the committed
`results/pp_coverage_exactmode_20260711/pp_exact_zs0.3_sz0.035.json` results
byte-for-byte (verified in the feat commit).
