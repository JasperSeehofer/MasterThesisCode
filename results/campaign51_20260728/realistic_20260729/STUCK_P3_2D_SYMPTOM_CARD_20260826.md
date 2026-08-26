# STUCK SYMPTOM CARD — [P3-2D] (2026-08-26; for the Stage-L searcher / fresh-eyes session)

**Independence rule: a searcher receives THIS CARD ONLY — never the suspect list or the
forensic history. The exoneration record lives in rows #207–#210 and PA-2D-9/10; consult it
only at intake, after the search returns.**

## Observed signatures (numbers, venues, what survived which controls)

- A bounded-identity comparison between (i) a banked per-seed statistic over ~24 accepted-event
  realizations of a 2-D (distance, mass) mirror venue and (ii) the matching expectation over
  ~25,600 synthetic draws from the model's completion-class law is off by a stable global
  factor: banked/model = 0.345 ± 0.013 (primary) and 0.366 ± 0.014 (a registered nonlinear
  rescale transform of the same objects) — arm-independent (two estimator arrangements give
  0.345/0.366 alike), i.e. a common-mode ×2.5–2.9 deficit, ~5× the pre-registered band.
- The normalization constant relating the two sides was independently re-derived blind and is
  exact (two algebraic forms agree to float round-off; its one nontrivial input is a
  20.8M-row zero-evaluate contraction, itself arbitered to 1e-9).
- Per-event mass-observable linkage on the model side is verified: two counterfactual
  re-scorings that alter the mass assignment (independent-host swap; own-mass re-redshifting)
  move the statistic by ×0.05 and ×0.9997 respectively — neither reproduces ×2.5.
- A real, measured selection double-weighting exists in the empirical venue's draw law
  (13.5–16% tilt) — sign-correct but ~7× too small; the fix is authorized but unrun.
- The 1-D member of the same identity family closes in band in the same codebase; the 2-D
  extension added: a latent mass dimension, a two-stage accepted-latent draw (weighted z-draw
  + Bernoulli survival thinning + whole-event rejection), and the new analytic contraction.
- The per-seed scatter of both sides is small and well-estimated (SEMs 2–3% of the means);
  this is not a variance problem.

## Abstraction ladder (search vocabulary per rung)

1. **Field-specific:** dark-siren / galaxy-catalogue mirror-venue calibration; selection
   function (Malmquist) thinning in mock universes; completion-class vs catalogue-class
   mixtures. Vocabulary: "mock data challenge normalization", "selection-weighted mock
   catalogue calibration", "detection-probability thinning bias".
2. **Method class:** identity/consistency tests between an empirical accepted sample
   (acceptance–rejection with whole-event restarts) and an analytic/self-normalized
   expectation; importance-sampling identities with acceptance indicators. Vocabulary:
   "self-normalized importance sampling bias", "acceptance-rejection realized density vs
   target density", "rejection sampling with restart biases the joint law".
3. **Math class:** O(1) multiplicative mismatch between an implemented sampler's stationary
   law and the density its normalizing contraction integrates; mixture-weight vs
   component-law confusion in two-class populations (the accepted-event population is a
   MIXTURE of classes; each side may condition on a different class measure). Vocabulary:
   "mixture class-conditional vs joint expectation", "conditioning vs marginalization
   mismatch in two-population estimators".
4. **Generic inference pathology:** two estimators of "the same" quantity disagreeing by a
   stable O(1) factor equal to neither side's variance scale — classically a ratio-of-
   normalizers slip (per-event N vs per-accepted N; per-class vs per-mixture measure).

## Sought

Methods/warnings literature on: consistency identities between accepted-sample sums and
model-side expectations in two-class (catalogue/completion) mixtures; which normalization
(drawn-count, accepted-count, class-count) each side of such an identity must carry; known
failure modes where a whole-event-rejection sampler realizes a different joint law than the
per-draw target (restart-induced re-weighting).
