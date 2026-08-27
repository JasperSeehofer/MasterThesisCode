# P3-2D residual accounting — [OPUS-ORCH 2026-08-27]

**Status: does NOT re-open the PARKED verdict.** [P3-2D] was PARKED at "UNATTRIBUTED-bounded" per
author ruling, row #211 (`607ac886`), with a STUCK symptom card banked
(`STUCK_P3_2D_SYMPTOM_CARD_20260826.md`). This document partially attributes the residual factor
that made it STUCK — moving the unattributed bound from ×2.5 to ×1.96 — but the residual remains
real, unattributed, and the PARK stands. Nothing here authorizes resuming compute on [P3-2D]; that
requires a fresh author [RULE] exactly as before. All three corrections below are, by construction,
**post-hoc reweightings of the frozen banked statistic** — precisely the route the A20 reviewer
rejected as a disqualified re-score option in `C2_star_review.md` option (a). They stand as
**attributions plus one free pre-registered prediction for a re-run**, nothing more.

## 0. What was frozen going in

- RHS2 (twin) = 0.01451300 ± 0.00045293 — held frozen, banked (`PREREGISTRATION_P3_2D_20260825.md:310-311`,
  reproduced in `p32d_c1_reweight_20260827.md` §1).
- LHS2 (bt/twin, identity, original) = 0.00500770 ± 0.00011615 (`PREREGISTRATION_P3_2D_20260825.md:309`,
  reproduced bit-for-bit in `p32d_c1_reweight_20260827.md` §3).
- X = RHS2/LHS2 = **2.898 ± 0.113** (SEs in quadrature).

This is the C2*-based bounded identity from `CLAIM_P3_2D_20260825.md` §2, C2* =
0.06124403326364123 from `ca_rhs_work2d/p3_2d_companion_v2.json` (verified exact match,
`p32d_c1_reweight_20260827.md` §1).

## 1. The multiplicative ladder (bt/twin arm, identity weight)

| step | mechanism | factor | LHS2 after | X = RHS2/LHS2 after |
|---|---|---|---|---|
| 0 | (start) | — | 0.00500770 ± 0.00011615 | 2.898 ± 0.113 |
| 1 | S̄_φ(z) double-application (banked, rows #209/#210, NOT NEW) | ×1.1585 | 0.00580132 ± 0.00014764 | **2.502 ± 0.101** |
| 2 | venue latent-mass floor vs the F2 MINOR-6 `S_4D(M<=0):=0` guard (NEW) | ×1.1944 | 0.00692891 ± 0.00018403 | **2.095 ± 0.086** |
| 3 | LHS-side dead-row convention violation vs PA-2D-1 F16 (NEW) | ×1.0680 | 0.00739968 ± 0.00024951 | **1.961 ± 0.090** |

**Total attributed factor: 1.4776** of the original 2.898. **Still unattributed: ×1.961 ± 0.090**
(bt/twin), **×2.348 ± 0.113** (bc/coded, same three corrections applied; per-arm figures in
`p32d_c1_reweight_20260827.md` for step 1 and the companion working notes for steps 2-3).

The deficit ratio LHS2/RHS2 moves **0.345 → 0.510** of unity: **25.2% of the 0.655 deficit closed
(37% of the log-gap)**. The arm split (bt vs bc) is essentially untouched by all three corrections,
because they are common-mode — exactly as the F8 coherence clause in `CLAIM_P3_2D_20260825.md`
expects of a genuine shared-mechanism fix.

## 2. Step 1 — S̄_φ(z) double-application (banked, reproduced not new)

Already banked at rows #209/#210 (`aaabc829`, `936236db`) and reproduced independently this session
in `p32d_c1_reweight_20260827.md` §4-5 via self-normalized importance sampling (as opposed to the
earlier harmonic-ratio-division route in `venue_drift_adjudication.py`): the class-G venue applies
the mass-marginal survival `S̄_φ(z)` to a set of latents whose acceptance already carries it once,
inflating the "dead" (`1-w2 = 0`) share and depressing the LHS2 sum. Correction factor
×1.15735 ± 0.00678 (bt, per-seed mean ± SEM, N=24 seeds), reproducing the C2_star_review.md
elimination-derived figure X_id = 2.506 almost exactly (this ladder's step-1 X = 2.502 ± 0.101).
**This step is CONFIRMED, NOT NEW** — it is restated here only so the ladder is complete and
auditable end-to-end from the original 2.898.

## 3. Step 2 (NEW) — venue latent-mass floor vs the model's zero-below-zero guard

**Mechanism.** The class-G venue clips latent mass draws to a floor of 1 M_sun
(`correspondence_1d.py:1708`), then obtains `S_4D` for those floored latents by grid-edge clamping
in `detection_probability_with_bh_mass_interpolated` (`simulation_detection_probability.py`
`_build_grid_2d` ~:1787-1902, accessor ~:2018-2117, "true nearest" clamp at the low-M grid edge,
documented at `p3_2d_companion.py:270-283`). This clamp returns a comparatively large `S_4D`
(mean `s4d_at_truth` = 0.826 on floor rows vs 0.718 on real-mass rows).

The companion object that defines Σ̃^4D and hence C2* takes the opposite convention: it applies the
**F2 MINOR-6 guard `S_4D(M<=0) := 0`** (`p3_2d_companion.py:281`, `:399`, `:909`) — i.e. the target
law the identity is measured against gives these floored-mass events measure **zero**, while the
venue that actually draws and accepts them gives them **16.52% of its accepted class-G
population**:

- 793/4800 latents across the fleet have `M_true == 1.0` exactly (the floor value).
- 380 of those 793 pass F-0.
- 372 of those 380 have `w2 == 1.0` exactly — i.e. they contribute **exactly zero** to the
  `Σ(1-w2)` LHS numerator, while still counting fully in the `/200` per-seed denominator
  (`p3_2d_fleet.py:632`, `n_drawn = meta["n_events"]`).

Because the venue and the target law disagree only on the *measure* of this floored-mass subset
(venue: nonzero, clamped-high `S_4D`; target: hard zero), removing/reweighting the floor-population
contribution to match the target's zero-measure convention raises LHS2 by **×1.1944**.

**This directly refutes, in sign, the Stage-L searcher's own C7 candidate**
(`stageL_p32d_fresh_search_20260827.md`), which predicted the mass-floor clip makes LHS2 **too
big**, bounded at ~10%. Measured: it makes LHS2 **too small**, and the effect is ~65% larger in
magnitude than the searcher's own ceiling (16.52% vs a predicted ≤10%, opposite sign).

## 4. Step 3 (NEW) — LHS-side dead-row convention violation

**Mechanism.** `p3_2d_fleet.py:632` computes `sum_acc = np.sum(1.0 - w2[live])` with
`live = L_cat_with_bh > 0` — i.e. rows with `A2 = alpha_G_phi * L_cat_with_bh = 0` are **dropped
from the sum entirely** rather than contributing a summand of 1. The registered convention,
**PA-2D-1 F16**, is explicit: "A2 = 0 => w2 = 0 => summand 1" (a dead catalogue-side likelihood
should count as a full completion-class contribution, not be excluded). The RHS instrument already
uses the registered convention (`ca_rhs_scorer.py:1324`, `w2 = np.divide(a2, denom, out=np.zeros_like(a2),
where=denom > 0.0)` — dead rows fall to `w2=0` and are *included*, not filtered out before the sum).

So the two sides of the identity run **different dead-row conventions**, and only the LHS side is
biased — downward, since it silently discards rows that should each contribute a full unit of
`(1-w2)=1` to the sum. Restoring PA-2D-1 F16 on the LHS raises LHS2 by **×1.0680**.

A strict variant that instead zeroes out the 3 pathological rows with both `A2 = 0` and `B2 = 0`
(rather than including them at summand 1, since `w2` is undefined 0/0 there) gives LHS2 =
0.00735830 (vs 0.00739968), X = 1.972 (vs 1.961) — **the choice between these two conventions is
worth <1% and does not change the qualitative picture.**

## 5. What remains unattributed

**×1.961 ± 0.090 (bt/twin), ×2.348 ± 0.113 (bc/coded).** Unchanged from before this session's work:

- **C2* is correct** (row #209, `936236db`) — not implicated by any of the three corrections above,
  which act entirely on the LHS2 construction, not on C2* itself.
- **The completion-mass axis is exonerated twice** (rows #209/#210, `936236db`/`aaabc829`) — the
  RHS-side "unlinked donor mass" hypothesis (PA-2D-9) and its alternative counterfactual
  construction (PA-2D-10, X_alt = 0.9997 ± 0.0003) both came back confound-free / refuted as
  operationalized.
- **The ×2.5 factor was never going to come from C1 alone** — the maximum achievable factor from
  the C1 (S̄_φ double-application) mechanism by itself is **1.831**, proved arithmetically (the
  ceiling on a self-normalized importance-sampling correction bounded by the S̄_φ(z) range observed
  in-fleet, `z_true ∈ [0.005, 0.34]` ⇒ `S̄_φ(z) ∈ [0.481, 0.989]` ⇒ max per-event reweight ≤ 1/0.481
  ≈ 2.08, damped by the ESS-weighted mixture to 1.831 in practice). C1 alone (step 1) delivered
  1.1585 of that ceiling; steps 2 and 3 are genuinely separate mechanisms, not a re-derivation of
  C1 by another route.

## 6. C2 (sky-frame) — lead dead on the class-G leg; one genuinely open question moved to the completion leg

`f_bar` is algebraically the pixel-average of `f_k` on the SAME shared cached completeness object
both venue sides load (`pixel_completeness.py:514`; HARD CONSISTENCY REQUIREMENT C1 at `:33-43`).
Measured: `E[1-f_k]/(1-f_bar) = 1.0000` exactly at every z tested, over all 12,288 HEALPix pixels.
The catalogue-selected 2D venue uses per-pixel `f_k` throughout, evaluated at the host's real
catalogue-inherited sky position. **No mismatch on the class-G leg — this lead is dead there.**

**A genuinely new, still-open question was opened on the COMPLETION leg, not closed.**
`E[1-f_k] = 1-f_bar` settles the MARGINAL only; the identity needs the JOINT. The completion venue
draws sky isotropically and z independently from `(1-f_bar(z))·w_pop·S̄_φ`, while the per-event
completion numerator carries `(1-f_k)` at the event's OWN pixel — so the completion draw law may not
couple sky and z the way the estimator's numerator does. **This must WAIT behind the class-G repair**
(steps 2-3 above): a venue with 16.5% invalid (mass-floored) latents cannot arbitrate anything on
the other side of the identity until its own construction is fixed. Cheap discriminator, registered
for later, not run here: bank per-draw `(sky, z, w2)` on 2-3 local RHS chunks and reweight by
`u = (1-f_k)/(1-f_bar)`.

## 7. Corrections to the C1 measurer's own report (record, not new physics)

These matter because the discipline this project runs on (`docs/METHODS_FALSE_ATTRIBUTION_DISCIPLINE.md`)
treats an overstated independence claim as a finding in its own right, not a footnote:

1. **C1's "independent-method cross-check" claim is FALSE.** Algebraically, the SNIS
   (self-normalized importance sampling) statistic in `p32d_c1_reweight_20260827.md` is the
   **identical statistic** to the banked `R_pred` estimator — a valuable independent
   **RE-IMPLEMENTATION**, but not methodological independence. Both compute the same
   self-normalized ratio; they differ only in code path, not in what is being estimated.
2. **C1's claim that RHS2's SEM is underestimated ~6.5× is REFUTED, with a hard bound.** The RHS2
   summand is bounded in [0,1] (it is a mean of indicator-like weights), so
   `SE <= sqrt(m(1-m)/N) = 7.47e-4` at `m = 0.01451`, `N = 25600` — a hard ceiling. The claimed
   ~0.003 SEM is **4× above that bound**, which is not possible for a plain bounded sample mean.
   C1 applied importance-sampling effective-sample-size (ESS) logic — appropriate for an
   importance-weighted estimator — to what is, on the RHS side, a **plain bounded sample mean**, a
   category error. The banked SE (a 128-chunk scatter SE) is reproduced to 10 digits and stands.
   The "~2.5σ_comb" significance language for the residual stands.
3. **The Stage-L searcher's own C7 candidate was REFUTED IN SIGN** (see §3 above): predicted the
   mass-floor clip makes LHS2 too big, bounded at ~10%; measured, it makes LHS2 too small and is
   worth 16.52%.

## 8. The free pre-registered prediction

Registered here, BEFORE any re-run, so a future re-run is a genuine test rather than a fit:

> **LHS2(bt) = 0.00739968 ± 0.00024951, X = 1.961 ± 0.090.**

If a fresh re-run of the class-G venue — with the mass-floor clip and dead-row convention both
corrected to match the target law (F2 MINOR-6 guard and PA-2D-1 F16 respectively) — lands at this
value within its own statistical error, the two NEW mechanisms (steps 2 and 3 above) are confirmed
end-to-end, not just as post-hoc reweightings of the frozen statistic. If it does not land there,
the reweighting model above is wrong somewhere — equally informative, and it means the residual
×1.96 bound from this document should NOT be treated as load-bearing until the discrepancy is
resolved. Either outcome is useful; neither outcome reopens the PARKED verdict on its own, since
running that re-run is itself compute that requires a fresh author [RULE] under the PARK.

## 9. Sources

- `p32d_c1_reweight_20260827.md` — step 1 remeasurement, ESS/tail diagnostics, and the three C1-report
  corrections in §7 above.
- `p32d_c2_skyframe_20260827.md` — the sky-frame (C2) analysis in §6 above.
- `CLAIM_P3_2D_20260825.md` §2 — the C2*-based bounded identity this ladder is built on; see also
  the `## RESIDUAL ACCOUNTING [OPUS-ORCH 2026-08-27]` section appended there.
- `STUCK_P3_2D_SYMPTOM_CARD_20260826.md`, row #211 (`607ac886`) — the PARK this document does not
  reopen.
- `stageL_p32d_fresh_search_20260827.md` — the C7 candidate refuted in sign (§3 above) and the
  73%/56% RHS-tail warning independently reproduced in `p32d_c1_reweight_20260827.md` §7.
- `docs/METHODS_FALSE_ATTRIBUTION_DISCIPLINE.md` — the discipline under which §7's corrections to
  the C1 report are recorded rather than quietly absorbed.
- Code citations: `correspondence_1d.py:1708` (mass-floor clip), `p3_2d_companion.py:270-283,281,399,909`
  (F2 MINOR-6 guard), `p3_2d_fleet.py:632` (LHS dead-row exclusion), `ca_rhs_scorer.py:1315-1324`
  (RHS dead-row inclusion, registered convention), `pixel_completeness.py:33-43,514` (C1 sky-frame
  consistency requirement).
