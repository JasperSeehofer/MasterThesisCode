# PRE-REGISTRATION (free read) — production completion-class decomposition of the base tilt

**Date:** 2026-08-20 · **Status:** registered BEFORE execution. Autonomous execution under
the author's grant ("follow the scientific clues"); branches return as [RULE]s. Zero
compute: a read on banked outputs. **No production change.**

**Motivation (row #136):** production's ensemble is ~95% out-of-catalogue and 81.5% of
events (iiib) have NO catalogue support at all, so most events' 1D likelihood is
p_i ≈ B_num(h)/D̃^φ(h) — a ratio of two integrals over the SAME population model. If the
estimator is correct that ratio is unbiased in h. The base tilt (1D mean 0.601, MAP railed;
2D −0.053) must therefore be carried by an identifiable class. This read asks WHICH.

## Data and classes

Baselines of record: `results/prod2d_closure_20260818/postfix_baseline/{iiib,joint_r1}/
event_likelihoods.csv` (derived-form B_scale, off-basis). Join `crb.iloc[event_idx]` for
`in_catalog`. Classes (evaluated per venue, membership fixed across h):

- **C-A** — `in_catalog == True` (true host in the catalogue): iiib 4.79%.
- **C-B** — `in_catalog == False` AND `L_cat_no_bh > 0` at ≥ 1 h node (impostor-only
  catalogue support).
- **C-C** — `L_cat_no_bh == 0` at every h node (pure completion leg).

## Statistics (T0 conventions: trapezoid weights, Σ log L, physics floor)

Per venue × class × channel (1D `combined_no_bh`, 2D `combined_with_bh`): class posterior
mean_h, MAP, σ_h, class size; plus each class's share of the full-sample Σ log-likelihood
SLOPE at the full-sample mean (the additive attribution of the tilt).

## Registered bands (fixed before running)

- **COMPLETION-CARRIES:** C-C's 1D mean is within ±0.010 of the full-sample 1D mean AND
  C-C's slope share ≥ 50% ⇒ the completion leg carries the base tilt (⇒ opens a fresh
  completion-leg derivation hunt as the next front).
- **CATALOGUE-CARRIES:** C-C's 1D mean is within ±0.020 of truth (0.73) while C-A ∪ C-B's
  mean is ≤ 0.65 ⇒ the catalogue legs carry it.
- **BOTH-TILT:** every class mean ≤ 0.65 ⇒ the tilt is common-mode (shared denominator
  D̃^φ or the population model itself), not leg-specific — the strongest pointer to a
  normalization/population defect.
- **MIXED:** anything else, reported with all class numbers.

## Registered caveats

1. Classes have different n and different z/sky selection, so class posteriors differ in
   width and in regime; this is an ATTRIBUTION read (who carries the slope), never a
   calibration read.
2. Class membership correlates with distance/sky-coverage, so a class difference conflates
   "leg" with "regime"; a class-carried tilt is a pointer for the next derivation, not a
   proof of leg-ownership.
3. Single realization (P7-8): iiib and joint_r1 share one universe.

---

## VERDICT

*(append-only after execution)*
