# PROPOSAL — [HIER] hierarchical photo-z self-calibration: the (h, θ)-grid experiment

**Date:** 2026-08-25 · **Status:** PROPOSED — presented, then STOP (author-gated per the
reviewable-artifact convention; the author's row-#192 approval opened the EXPLORATION — this
document is the decision surface for the first EXPERIMENT) · **Thread:** `[HIER]` ·
**Grounding:** `STAGE_L_HIER_20260825.md` (+ verification appendix); ledger rows #192–#193;
the banked z-structured score-at-truth tilt (≈0 below z≈0.4, ≈−1.08 by z≈0.9; dark class
−0.635 ± 0.017); `docs/SIGMA_Z_SIGMA_M_FORECAST.md` (the information bound this cannot beat).

## 1. The claim being explored (stage-0 form, with Refute-by)

**[HYPOTHESIS]** Part of the N-coherent ensemble bias is *error-model mis-specification* —
the per-event Gaussian photo-z kernel with catalogue-quoted σ_z mis-states the true error law
(bias curve and/or scatter scale, z-dependently) — and a hierarchical layer that infers a
low-dimensional error-model θ JOINTLY with h across the ensemble would (a) absorb the
z-structured tilt into θ and (b) yield calibrated (coverage-passing) posteriors, at the price
of honest width. **Refute by:** on the mirror venue, where truth-θ is known, the joint (h, θ)
posterior evaluated at truth-θ still rails / coverage still fails — then the coherence lever
is dead in our regime and the thread closes with a documented bound.

**Field position (quote-verified):** Hanselman+ 2024 §IV.5 — *"it should be possible to
simultaneously infer the weighting scheme as well as H0 by generalizing the idea laid out in
[86]"* ([86] = Vijaykumar+ 2024, ApJ 972, 157) — named, never built; no siren-context
ensemble error-model+H₀ joint inference exists; no small-N validity statements exist; the
survey-scale self-calibration class lives in the inverse (minority-outlier) regime. The
experiment is novel either way it lands; the ABSENCE is already banked (row #193).

## 2. The instrument (feasible core)

θ enters only the per-event z-kernels, so the joint posterior is computable WITHOUT new
architecture: evaluate the existing per-event likelihood on an (h × θ) grid —
**θ = (b, s): a bias slope Δz = b·(1+z) and a scatter scale σ_z → s·σ_z** (2D; ~5×5 nodes,
refined once) — giving per-event L(h, θ) cubes; the ensemble joint posterior and its
h-marginal follow by summation. Truth-θ = (0, 1) on the mirror venue by construction.
Registered reads: (i) the score-at-truth tilt recomputed AT the θ-marginalized posterior and
AT truth-θ (does the z-structure reabsorb?); (ii) coverage/P–P over seeds at the h-marginal
(the stage-4 currency — the author's posterior-statistics ladder, row #188); (iii) the width
vs the F5 forecast (the honest-trade check); (iv) the (h, θ) degeneracy structure (the
identifiability answer the literature lacks).

## 3. Expectations (registered two-sided)

| outcome | signature | disposition |
|---|---|---|
| Error-model share is real | tilt reabsorbs; coverage improves at truth-θ and at marginal; h-bias shrinks | a physics-change candidate for the production kernel (its own gate) |
| Honest-width only | coverage improves ONLY via widening; bias → width trade | a calibration result: report + the thesis's posterior-honesty chapter |
| Unidentifiable | (h, θ) ridge; truth-θ uninformative at N | the lever is dead at our N — banked bound + the small-N validity statement the field lacks |

## 4. Costing — [ORCH-COST, cluster-first per row #185]

25 θ-nodes × the 12-seed mirror fleet ≈ **~50–100 CPU-h as one cluster array** (embarrassingly
parallel in (seed, θ); array-friendly by construction per row #185's registration rule). Zero
new estimator physics (θ applied as kernel transforms at evaluate-time via existing
instrumentation patterns — an instrumentation flag pair, byte-identical at truth-θ). Design +
prereg + review chain before any launch, as always.

## 5. Decision table

| # | item | tag | recommendation |
|---|---|---|---|
| 1 | Authorize the (h, θ)-grid experiment (prereg → review chain → cluster array) | [DO] | approve — the refutation is as valuable as the confirmation |
| 2 | Read [86] (Vijaykumar+ 2024) + the §IV.5 generalization properly before the prereg | [DO] | approve (stage-L obligation, ≤1 day) |
| 3 | Scope guard: no production kernel change from this thread without its own physics-change gate | [STANDING structure] | note only |
| 4 | Sequencing vs [P3-2D]/[P3-WBHZERO] | [RULE] | those finish first (the cluster budget and the review bandwidth are shared); [HIER] prereg drafting can proceed in parallel |

**STOP.** Presented for the author's ruling.
