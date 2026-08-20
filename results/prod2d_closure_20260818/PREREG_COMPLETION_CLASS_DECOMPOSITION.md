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

**VERDICT (2026-08-20, appended after execution; [RULE]s to the author):**

**Band fired: COMPLETION-CARRIES** (both venues, both channels).

| venue | class | n (%) | 1D mean | 1D σ_h | 1D MAP | 1D slope share | 2D mean |
|---|---|---|---|---|---|---|---|
| iiib | C-A in-catalogue | 76 (4.8%) | 0.8279 | 0.0298 | 0.86 | −0.91 | 0.7821 |
| iiib | C-B impostor-only | 907 (57.1%) | 0.6160 | 0.0141 | 0.61 | −0.04 | 0.8441 |
| iiib | **C-C pure completion** | **605 (38.1%)** | **0.6001** | **0.0011** | **0.60** | **+1.95** | **0.6004** |
| joint_r1 | C-A | 76 (4.8%) | 0.8187 | 0.0372 | 0.86 | −1.17 | 0.8125 |
| joint_r1 | C-B | 1021 (64.3%) | 0.6080 | 0.0100 | 0.60 | +0.30 | 0.7945 |
| joint_r1 | **C-C** | **491 (30.9%)** | **0.6004** | **0.0021** | **0.60** | **+1.87** | **0.6011** |

Full-sample means: 1D 0.6010/0.6020, 2D 0.6771/0.6788. C-C reproduces the full-sample 1D
mean to 0.001 and carries ~195% of its slope (the catalogue-supported classes pull the
other way — the production posterior is a balance point, as the gate review predicted).
**Identical in the 2D channel** (C-C 0.6004/0.6011): the base tilt is NOT mass-channel
structure.

**The mechanism, stated as a testable defect (free-read follow-up, same data):** for the
pure-completion class the per-event score at TRUTH is

    d ln p_i/dh |_(h=0.73) = **−0.635 ± 0.017** (iiib; sd 0.417 over 605 events)
                             **−0.565 ± 0.020** (joint_r1; sd 0.440 over 491)

i.e. **37σ / 28σ from zero**. A correctly normalized likelihood has zero expected score at
truth when the data come from its own model; this one does not, and 605 events convert a
−0.6/event score into a −384 nats/h ensemble slope ⇒ the delta-like rail at the grid edge
(σ_h = 0.0011). Decomposition (identity verified to machine precision, B_num − D̃^φ = p_i):
d ln B_num,i/dh = −1.871 per event vs the GLOBAL d ln D̃^φ/dh = −1.236 — the completion
NUMERATOR falls with h faster than the selection normalization that is supposed to cancel
it, by 0.635 per event.

**Interpretation (two candidates, and the experiment that separates them — already
registered as A-2's B-OUT):**
(i) **internal misnormalization** of the completion leg (numerator/denominator not a
matched pair in h — the same defect class as the removed B_scale, which lived on this exact
leg), or (ii) **population/selection misspecification** — the injected universe's detected
n(z) differs from the estimator's assumed population × completeness, so the score is
non-zero at truth even for a self-consistent estimator.
**B-OUT decides it:** it draws dark hosts FROM the estimator's own population model, so a
non-zero score there ⇒ (i) internal defect; a zero score ⇒ (ii) misspecification and the
production tilt is a data-vs-model mismatch. This is now the top-ranked derivation target.

Caveats §"Registered caveats" apply: class membership correlates with regime, so this is
attribution (who carries the slope), not leg-ownership proof.
