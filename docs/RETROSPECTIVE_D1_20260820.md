# Retrospective — the B-SEL correspondence arm and the D-1 over-void (rows #140–#144)

**Date:** 2026-08-20 · Produced by a six-agent adversarial retrospective (four lenses, a
steelman instructed to argue the orchestrator was wrong, a synthesis chair), with every
decisive number re-verified by the orchestrator directly. It concludes that **two successive
rulings of mine were wrong in opposite directions**, and it found a harness defect that
invalidates a gate we have been trusting since G-0.

## What happened, in three sentences

Arm B-SEL drew hosts from the estimator's own assumed detected-dark density, was registered
as matched "in BOTH population and selection", returned **−0.1120 ± 0.0017**, and that was
banked as a genuine estimator defect (row #140). Two bisection arms failed to move it, which
triggered a premise check (D-1); D-1 reported a surviving-vs-model CDF gap of 0.0792 against
a registered band of ≤0.05, and I **voided** row #140 (row #143). The retrospective shows the
premise ambiguity was real but the void was an over-escalation, because **the D-1 band was
statistically meaningless at the N it ran at**.

## Failure 1 — "model-matched" was a predicate over a pipeline, not a harness

B-SEL is a composition: draw → donor-row assignment → quality filter → score. Only the first
stage samples from the density the estimator evaluates; the rest borrow real-world machinery
and are load-bearing modelling assumptions. The registration named *population* and
*selection* — both describe the draw. The donor resampling and quality filter were not
omitted through carelessness; they were not perceived as claims at all, because they were
pre-existing harness machinery. **Building implied trusting.**

The adversarial verifier missed it for a different, systemic reason: **it never saw the arm.**
The verifier stamp on the correspondence pre-registration belongs to its v1 (B-0, B-σ, B-D2,
E-DEN). A-3 through A-6 were filed as inline `AMENDMENT (registered now, pre-run)` blocks by
the executing session and inherited that stamp by proximity — even though A-3 did not extend
the v1 question but replaced it with a stronger claim carrying its own decisive band.
`CLAUDE.md` already makes *author* approval non-transitive; nothing makes *verifier* coverage
non-transitive.

A related detail will recur if unnamed: a premise check *was* attempted at row #140 (the
"residual-mismatch bound"), and it **argued the filter's direction** — "it drops the FARTHEST
events, which pushes the posterior HIGH" — instead of measuring it. D-1 found the sign
reversed. An argument from direction has the form of a check and none of its content.

## Failure 2 — the D-1 band could not have passed (verified arithmetic)

For a one-sample max-CDF-gap statistic on n draws against a fully specified density:

| n | E[D] under the null | D_crit(5%) | false-fail rate of a 0.05 band |
|---|---|---|---|
| 174 (D-1's surviving set) | **0.0659** | 0.1029 | **0.58** |
| 200 (the drawn control) | 0.0614 | 0.0960 | 0.63 |
| 1588 (row #137's provenance check, where 0.05 came from) | 0.0218 | 0.0341 | ≈1.00 |

**The registered band sat below the expected fluctuation of a perfectly matched sample.**
D-1's observed 0.0792 at n = 174 has **p = 0.225** — entirely consistent with a perfect
match. The 0.05 threshold was imported from a check with nine times the sample size, where
it was in fact too *loose*. So the MIRROR-MISMATCHED verdict is not evidence of mismatch, and
**the void of row #140 is withdrawn**.

## Failure 3 — the mirror's positive control carries no information (new, and the worst of the three)

Directly verified in the banked outputs: `bf1_seed900101.json` and `bf1_seed900102.json` have
`max(log_posterior) − min(log_posterior) = 0.0` **exactly**, across all 46 grid nodes, in both
seeds — a perfectly flat likelihood. For contrast, a real arm (`bsel_seed900101`) spans 13.85
nats. B-F1's celebrated "0.7300 — truth to four decimals" is therefore not a measurement of
an unbiased estimator; it is what the pipeline emits when the likelihood carries no
h-dependence at all. (It is not even the flat-posterior moment of this grid, which is 0.6776 —
so the reported 0.73 comes from some other path and warrants its own bug hunt.)

The `_UnityCompleteness` f ≡ 1 shim that produces this is the same shim used by **G-1, the
mirror's STOP gate** (`correspondence_1d.py:1843, 2049`). Consequences: G-1's "PASS" was
vacuous; row #136's claim that B-F1 proves the catalogue-arm bias is a completeness artefact
is unsupported; and the harness has been running since G-0 without a working positive control.

## What the steelman established that survives — and what it overturns

Two quantitative results stand and I reproduced both:

1. **The survival mismatch cannot own −0.112.** Using the campaign's own z-resolved dark-class
   score as the sensitivity kernel, the assumption-free bound |∫s dΔF| ≤ D·TV(s) with
   D = 0.0792 caps the mismatch's contribution at 34–62% of the ensemble slope, leaving a
   residual ≥ 0.073 — fifteen times the self-consistency band. To own the whole effect the
   mismatch would need a mean z-shift of ≈0.17; the measured shift is +0.018.
2. **The three bisection arms were never inert.** On the unsaturated scale, σ_h runs
   0.0216 → 0.0182 → 0.0150, i.e. the ensemble slope rose **46 → 55 → 67 nats/h, a 44%
   change**. `mean_h` is floor-saturated at the 0.60 grid edge (coverage 0/0/0 in all three),
   which compressed that into the ~6% "nothing moves" pattern I read as inertness. A driver
   entirely outside the estimator cannot produce a 44% spread across three internal repairs.
   **My row #143 item 2 — "the driver was outside the estimator all along" — is retracted.**

## Corrected status of record

**Row #140 is reinstated as PROVISIONAL-WITH-A-BOUND:** B-SEL's −0.112 contains a
survival-time data-vs-model contribution bounded at ≤34–62% of the slope; the residual
internal component is ≥0.073 and cannot reach the self-consistent band. What blocks banking
it as a defect is **not** D-1 — it is Failure 3: the arm has no working positive control, so
"biased" and "carries no information" are not yet distinguishable by anything in the harness.

**D-2 as registered is inadequate** and must be redesigned before it runs: it changes two
things at once (removes the filter *and* replaces donor rows with analytic errors drawn to
match the estimator's assumed model), which re-creates self-consistency blindness by
construction, and it inherits the broken control.

## Proposed amendment A15 — gates and controls must have demonstrated operating characteristics

*(Style-matched to A10–A14 in `docs/RESEARCH_CYCLE.md`; author ruling required.)*

**A15 — Power-calibrated gates, sensitive controls.** No scored threshold may be registered
without its operating characteristics at the ACTUAL N it will run at: state the statistic's
null distribution, the false-fail rate of the chosen band, and the effect size the band can
detect at reasonable power. A band lying inside the null's expected fluctuation is void on
its face, and a threshold imported from another measurement is invalid unless its N and
statistic match. Symmetrically, **no null/positive control counts as a control until it is
shown to be informative** — a control must be demonstrated capable of failing (e.g. its
likelihood must carry the dependence it claims to certify; a flat likelihood is not a pass).
*Evidence:* D-1's 0.05 band false-fails 58% of the time at n = 174 and was imported from an
n = 1588 check where it was too loose; separately, the mirror's B-F1/G-1 unity-completeness
control returned a perfectly flat log-posterior (span 0.0 nats) whose "truth to four decimals"
certified nothing, and it had gated every arm since G-0. *Relation to existing rules:* A13
requires an INSTRUMENT to be shown to move the output; A15 is the same demand one level up,
applied to the GATES and CONTROLS that adjudicate instruments. *Cost:* minutes per gate — one
null-distribution calculation and one control-sensitivity check.

## What this does not change

Production's base tilt is untouched: the dark class at 0.6001, its score of −0.635 ± 0.017 at
truth, and its high-z localization are production-native reads on banked data with no mirror
in the chain. So are the B_scale removal, the s_Edd re-measurement, J_α, the f-treatment
closure, and the post-fix baselines. The mirror's *qualitative* achievement also stands: it
reproduces production's dark-class rail (B-OUT 0.6007 vs 0.6001), which no self-consistent
harness had ever done.

## The single measurement that settles it

**Build a positive control that can fail, then re-run the isolation test.** Concretely: an
arm whose universe is generated by the estimator's own forward model end-to-end (draw AND
observation AND selection from the same objects the likelihood integrates), with an
injected-bias variant to prove the arm can detect a known displacement. If that arm returns
truth, the harness is trustworthy and B-SEL's residual ≥0.073 becomes a defect claim; if it
returns a bias, the harness is the defect. Everything else — D-2, the D̃^φ class composition,
the remaining bisection — is downstream of that one control.
