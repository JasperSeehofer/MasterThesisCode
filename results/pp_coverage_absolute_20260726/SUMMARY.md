# `mixture_mode="absolute"` calibration campaign — SUMMARY

**Date:** 2026-07-26. **Harness:** `master_thesis_code/validation/pp_coverage.py`
(new `mixture_mode="absolute"`). **Spec:** `results/lcat_h_dependence_20260725/
DERIVATION_ESTIMATOR_REDESIGN.md` Variant 1 (Sec 6, validation gate 1).
**Commands:** `RUNBOOK.md`. n_realizations=500, n_events=250 per cell/mode
unless noted.

## TL;DR verdict

**`absolute` mode makes NO material difference to coverage or MAP bias in any
of the three cells tested, including the deep completion-governed cell.**
This is a genuine, unflattering finding, not a bug: the calibration-harness's
`z_support` truncation mechanism does not — and structurally cannot — create
the impostor-only candidate balls that Variant 1's absolute-mass numerator is
designed to suppress (see "Harness-fidelity caveat" below). The harness
regression check (existing `two_branch`/`gray`/`conditioned`/`exact` modes)
is byte-identical, confirming the new mode was added without perturbing
production-analog code paths already validated by prior campaigns.

## Coverage tables (50/68/90% HPD), old (`two_branch`) vs new (`absolute`)

### Cell A — shallow / complete catalogue (`z_support=1.0`, i.e. > Z_MAX_POP=0.95; completion_fraction ~ 0)

| h_true | mode | cov50 | cov68 | cov90 | MAP bias | map_std |
|---|---|---|---|---|---|---|
| 0.62 | two_branch | 0.600 | 0.748 | 0.888 | -0.0023 | 0.0064 |
| 0.62 | absolute   | 0.600 | 0.748 | 0.888 | -0.0023 | 0.0064 |
| 0.72 | two_branch | 0.534 | 0.718 | 0.920 | -0.0022 | 0.0067 |
| 0.72 | absolute   | 0.534 | 0.718 | 0.920 | -0.0022 | 0.0067 |
| 0.84 | two_branch | 0.522 | 0.678 | 0.890 | -0.0023 | 0.0072 |
| 0.84 | absolute   | 0.522 | 0.678 | 0.890 | -0.0023 | 0.0072 |

Byte-identical, as predicted by the derivation's limiting case (a): with
`z_support >= Z_MAX_POP` there are no zero-host events and no completion
term, and `absolute`'s host-branch formula `(N_i + 0)/D = N_i/D` collapses to
exactly the `two_branch` host-branch formula `N_i/D`. This is a hard
algebraic identity, not an approximate agreement -- the two JSON result
blocks are equal to the last printed digit. Reasonably well-calibrated
(cov68/cov90 slightly under nominal, cov50 close), consistent with the
prior shallow-venue harness record.

### Cell B — intermediate (`z_support=0.3`; completion_fraction ~ 0.22 / 0.39 / 0.55)

| h_true | mode | cov50 | cov68 | cov90 | MAP bias |
|---|---|---|---|---|---|
| 0.62 | two_branch | 0.052 | 0.102 | 0.242 | +0.0157 |
| 0.62 | absolute   | 0.050 | 0.100 | 0.232 | +0.0159 |
| 0.72 | two_branch | 0.002 | 0.014 | 0.058 | +0.0232 |
| 0.72 | absolute   | 0.002 | 0.016 | 0.054 | +0.0236 |
| 0.84 | two_branch | 0.004 | 0.006 | 0.012 | +0.0192 |
| 0.84 | absolute   | 0.002 | 0.006 | 0.012 | +0.0193 |

**Both modes are badly miscalibrated** (cov50 as low as 0.2-0.4% of nominal
50%; cov90 8-24% of nominal 90%) and **both carry essentially the same
positive MAP bias (+0.016 to +0.024)**. The differences between `two_branch`
and `absolute` here (<=0.01 in coverage fraction, <=0.0004 in bias) are
inside Monte-Carlo noise at n_realizations=500 (binomial CI half-width at
cov50~0.05 is ~0.02) -- i.e. statistically indistinguishable, not a partial
improvement.

### Cell C — deep (`z_support=0.2`; completion_fraction ~ 0.71 / 0.79 / 0.85)

| h_true | mode | cov50 | cov68 | cov90 | MAP bias | map_std |
|---|---|---|---|---|---|---|
| 0.62 | two_branch | 0.002 | 0.022 | 0.104 | +0.0324 | 0.0106 |
| 0.62 | absolute   | 0.002 | 0.018 | 0.104 | +0.0327 | 0.0107 |
| 0.72 | two_branch | 0.024 | 0.056 | 0.170 | +0.0384 | 0.0150 |
| 0.72 | absolute   | 0.024 | 0.054 | 0.168 | +0.0387 | 0.0150 |
| 0.84 | two_branch | 0.024 | 0.040 | 0.200 | +0.0190 | 0.0045 |
| 0.84 | absolute   | 0.024 | 0.038 | 0.198 | +0.0190 | 0.0045 |

**Still badly miscalibrated in both modes** (cov50 far below nominal 50% in
all three truths; cov90 roughly half of nominal 90%) and **still carrying
essentially the same positive MAP bias** (+0.019 to +0.039). The largest
single coverage delta between modes is 0.004 (cov68 at h=0.62/0.72) --
inside Monte-Carlo noise. **The deep, completion-governed cell (71-85% of
events routed to the completion branch, squarely inside the mission's
22-85% target range) shows the SAME near-null result as cells A and B: no
detectable improvement from `absolute` mode.**

## Ensemble MAP bias per cell, old vs new (h_true=0.72 row, representative)

| Cell | completion_fraction | two_branch bias | absolute bias | delta (absolute - two_branch) |
|---|---|---|---|---|
| shallow (A) | 0.00 | -0.0022 | -0.0022 | 0.0000 (exact algebraic identity) |
| intermediate (B) | 0.39 | +0.0232 | +0.0236 | +0.0004 (noise) |
| deep (C) | 0.79 | +0.0384 | +0.0387 | +0.0003 (noise) |

At n_realizations=500, the bootstrap SE on map_bias is of order map_std/sqrt(500)
~ 0.0007 (deep cell, h=0.72: map_std=0.015). Every measured delta above is
smaller than this SE -- consistent with zero effect, not a small positive
effect.

## Is the deep-cell miscalibration cured?

**NO.** Based on all three cells (A/B/C, n_realizations=500 each), `absolute`
mode's coverage and bias track `two_branch` mode to within Monte-Carlo noise
everywhere, including the completion-dominated cell C (71-85% completion
fraction) that the mission specifically targeted as the previously-railed
regime. This is the *opposite* of the production prediction (derivation Sec
3.5: "≥90% of the joint tilt removed", "max |S_V1| = 1.18 vs max |S_cur| =
28.4") -- but, per the harness-fidelity caveat below, this harness's
`z_support` mechanism is not exercising the tilt Variant 1 targets in the
first place, so this null result does not falsify Variant 1's production
claim; it shows that THIS harness cannot detect whatever effect Variant 1
would have.

## Harness-fidelity caveat (why this is not a refutation of Variant 1)

The derivation's mechanism for Variant 1's improvement is specifically
**impostor-ball suppression**: production candidate balls sometimes contain
ONLY galaxies that are not the true host (foreground impostors), and the
current `L_cat` self-normalization forces those impostor balls to carry O(1)
weight against the completion term regardless of how implausible they are;
Variant 1's absolute-mass numerator lets such balls defer continuously to
the completion term (`A_i/B_num -> 0` as the ball's absolute plausibility
mass -> 0).

**This harness has no such mechanism.** Every event in `pp_coverage.py` has
exactly ONE candidate: a Gaussian kernel centered on the event's own noisy
observed redshift `z_gal` (drawn as `z_host + N(0, sigma_z)` from the TRUE
host). There is no shared discrete galaxy catalogue that events query and
that could return a candidate set containing zero true hosts -- the
"catalogue membership" test (`z_host >= z_support` or, with
`membership_on_observed`, `z_gal >= z_support`) either routes an event to
its own well-matched single candidate (host branch) or to the
pure-completion branch (zero-host fallback); there is no third case of "one
or more badly-mismatched candidates and no good one." Consequently the
harness's `z_support`-induced coverage failure in cells B/C is being driven
by something Variant 1's mechanism does not touch -- most likely the
completion term `B_num`'s own model fidelity (the sigma_z/sigma(z) asymmetry
and Malmquist-in-z shape documented elsewhere in this harness's module
docstring, or the flat z_support hard-cut itself being a coarser
approximation than production's smooth `f(z)` completeness weighting) --
and swapping the catalogue-term normalization cannot fix a bias that lives
in the completion term.

**This is an explicit, load-bearing negative finding, not an approximation
glossed over:** the mission asked whether the harness confirms the
deep-cell rail is cured, and the honest answer, given what this harness can
and cannot represent, is that this experiment is not capable of testing that
claim's mechanism, and the numbers it does produce show no improvement. A
faithful Variant-1 harness analog would need a genuine multi-galaxy
candidate-set generator (with a configurable impostor fraction/incompleteness
model) -- that does not exist in `pp_coverage.py` and was out of scope to
build under this mission's time box. Confirming/refuting Variant 1's actual
claim requires the production-code gates the derivation already specifies
(Sec 6, gates 2-3: seed600 must-not-change, seed1000 EXP-40 deep-venue
closure), not this harness.

## Regression check (existing modes unaffected)

`two_branch` rerun of `results/pp_coverage_deepvenue_20260710/
pp_zs0.3_sz0.035_volume.json` (z_support=0.3, sigma_z=0.035,
n_realizations=120) reproduces every shared field (`coverage`, `map_mean`,
`map_std`, `map_median`, `map_bias`, `rail_fraction`, `completion_fraction`)
bit-for-bit; the only diff is two diagnostic keys
(`dlogL_dh_host_mean`/`dlogL_dh_completion_mean`) the 2026-07-10 reference
predates (added by an unrelated later commit). `gray`, `conditioned`, and
`exact` modes are untouched by this change (new `elif` branch only).

## What this campaign does and does not establish

- **Established:** the harness-level implementation of `mixture_mode=
  "absolute"` is internally consistent (exact algebraic collapse to
  `two_branch` in the complete-catalogue limit -- a hard-zero check, not an
  approximate one) and does not disturb any existing mode.
- **Established:** in the three synthetic-universe regimes this harness CAN
  express, `absolute` mode provides no measurable calibration or bias
  improvement over `two_branch`.
- **NOT established (harness cannot test it):** whether Variant 1 fixes the
  production host-misassociation rail, because the rail's actual mechanism
  (impostor-only candidate balls in a real multi-galaxy catalogue query) has
  no analog in this harness's one-candidate-per-event generative model.

## Recommendation

Treat this campaign as evidence that the `pp_coverage.py` harness, as
currently structured, is the wrong instrument to validate Variant 1's central
claim, and proceed to the derivation's own gate 2/3 (production-code
seed600/seed1000 re-evaluations) rather than iterating further on this
harness. If a harness-level test of the impostor mechanism is wanted, it
would require extending the generative model with a genuine multi-candidate
catalogue (e.g., a Poisson-sampled field of background galaxies per event,
some of which fall inside the localization volume without being the true
host) -- a materially larger change than adding a mixture-mode branch, and
outside this mission's scope.
