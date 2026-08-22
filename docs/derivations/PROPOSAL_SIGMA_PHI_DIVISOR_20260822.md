# PHYSICS-CHANGE PROPOSAL — the no-BH catalogue divisor: Σ³ᴰ → Σ^φ (the fourth Path-A slot)

**Date:** 2026-08-22 · **Status:** PROPOSED — presented, then STOP (author-gated;
row #172 item 2) · **Subject:** `bayesian_inference/bayesian_statistics.py` (trigger file;
nothing changes until ruled) · **Evidence:** rows #168, #171–#172;
`CLAIM_P3_RPHI_20260822.md` (+ its A20 amendments); `A20_REVIEW_P3_RPHI_20260822.md`.

## 1. Old formula (item 1)

No-BH catalogue leg: `L_cat_no_bh = A_ball / Σ³ᴰ` with `Σ³ᴰ = _global_cat_denom_no_bh`
(`:4823`, built at `:3826` from the **separately fitted, mass-blind S_3D** survival), combined
as `β_G_φ·L_cat_no_bh/D̃_φ` (`:5400`).

## 2. New formula (item 2)

`L_cat_no_bh = A_ball / Σ^φ` with `Σ^φ = _global_cat_selection_phi` (`:3878` — ALREADY BUILT
by Path A on the same rows/weights/eligibility, currently feeding only the weight chain).
One-slot change; the with-BH leg (`A/Σ⁴ᴰ` against `α_G_φ = Σ⁴ᴰ/n̂_w^φ`) is untouched — it is
the internal control that already pairs correctly.

## 3. Reference / derivation (item 3)

The A20/RPHI review's ratified argument (banked verbatim): the estimator's catalogue target is
`A_ball/n̂_w^φ` with `n̂_w^φ ≡ Σ^φ/β_G^φ` a DEFINED code object (`path_a_mixture_objects`,
`:2441`); `β_G_φ·A/Σ` equals it **iff Σ = Σ^φ, uniquely** — algebraically exact, no
catalogue-faithfulness assumption. The β-ratio alternative is refuted (injects a second rate
density `n̂_w^3D` into a mixture committed to `n̂_w^φ`). Path A's own docstrings assert
`r_φ ≡ 1` (`:1751-1753`, `:2422-2423`) — false for this slot (production object: r_φ(0.73) =
0.9119 per the code's own gate (ii-b) note; realistic venue: 0.8860 measured) — the change
makes the documented invariant TRUE. Framework refs: MFG (2019) Eqs. (5)–(7) selection
consistency (same population + detection model in every numerator's normalization);
`FIXB_PATHA_PACKAGE.md` §3.2 (the three ruled slots this completes to four);
`bscale_completion_normalization.md` §2 (the with-BH cancellation this mirrors).

## 4. Dimensional analysis (item 4)

Σ^φ and Σ³ᴰ are the same catalogue-weighted survival sums (dimensionless weights × survival);
`L_cat`'s units unchanged; `r_φ = Σ^φ/Σ³ᴰ` dimensionless ∈ (0.85, 0.92) measured.

## 5. Limiting cases (item 5)

- Tower identity holds (S_3D = ∫φ S_4D dM exactly) ⇒ Σ^φ = Σ³ᴰ ⇒ byte-identical — the change
  is inert exactly when the documented assumption is true.
- S̄ → c·S̄ homogeneity: with Σ^φ the catalogue leg becomes c-degree-matched to its weight
  chain (the review's invariance test); with Σ³ᴰ it is not.
- Single-candidate, σ_z→0: reduces to the selected-prior single-host form with the φ-consistent
  normalization (the A-FULL structure).

## 6. Validity conditions + verification plan (item 6, the assumption register)

- **Measured venue effect:** −0.004309 ± 0.000736 (12 B-SEL seeds; anti-conservative
  direction; venue-conditional — amendment 5 binding).
- **Verification plan before adoption:** (i) production-object r_φ(h) measurement (the
  committed-leaf instrument at the production catalogue/venue — zero-evaluate, ~5 min);
  (ii) an instrumentation counterfactual flag (`catalogue_global_selection ∈ {"s3d","phi"}`,
  default byte-identical) + a production-run counterfactual read (the row #119 M-pattern);
  (iii) regression: the S̄→cS̄ degree test as a unit test; the with-BH channel bit-unchanged;
  (iv) the b0 catalogued-host venue inherits the corrected slot for the identity test.
- **Scope:** no-BH leg only; interacts with (but is separable from) the twin/basis decisions —
  the corrected slot changes the catalogue leg's LEVEL by 1/r_φ(h) independent of the S̄_φ
  numerator question.

## 7. Decision table

| # | item | tag | recommendation |
|---|---|---|---|
| 1 | Adopt Σ^φ as the no-BH divisor (production) | [RULE] | after (i)+(ii) of the plan — measure first |
| 2 | The verification plan (i)–(iii) | [DO] | approve (≲1 CPU-h + one instrumentation commit) |
| 3 | Fold the corrected slot into the FC/FT + b0 chain | [RULE] | yes — the b0 test should run on the corrected slot |

**STOP.** Presented for the author's ruling.
