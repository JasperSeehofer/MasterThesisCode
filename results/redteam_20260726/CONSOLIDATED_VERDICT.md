# Redteam consolidated verdict — 2026-07-26

Two independent adversarial reviews of the production stack
(`generator_marginal + --pdet_z_resolved`), ordered by the author as a merge
gate for production adoption. Full reports: `MATH_REVIEW.md`,
`PHYSICS_METHODOLOGY_REVIEW.md` (this file is the synthesis; the reports
control where they differ).

## Verdicts

| Review | Verdict | Anti-tuning |
|---|---|---|
| Mathematical soundness | SOUND-WITH-CAVEATS | **NO EVIDENCE OF TUNING** — derivations re-derived independently and correct; `C_NORM` cancels algebraically; `constants.H` enters inference only in a post-hoc diagnostic; commit chronology clean (no estimator constant altered after a readout; pre-registration precedes submission) |
| Physics + methodology | SOUND-WITH-CAVEATS | **NO NUMERICAL ANCHOR to 0.73** found (`m_th`, `f(z,h)`, `F`, `W_cat`, `d_hor` all h-free or exactly h-scaled) — estimator should close at a different truth; golden-event pulls (mean +0.06, std 0.94, n=133) are independent evidence closure is calibrated, not tuned |

**Merge gate: CLEARED.** The caveats rescope claims and order follow-ups;
they do not invalidate the estimator or the adoption.

## Binding claim rescopes (apply to all downstream docs and the paper)

1. **R1 (from math F1): the 41-pt grid does not resolve the peak.** Exactly
   one grid node has p/p_max > 1e-3 on all three deep venues; parabola
   neighbours sit ~200 ln below the peak; the ±2-step lnP ratio (2.5–3.3×)
   is inconsistent with the Gaussian 4×. The quoted −0.00030 ± 0.00035 bias
   and σ ≈ 2.6e-4 widths are **extrapolations**. Grid-supported claim:
   **MAP node = 0.730 in 4/4 valid venues ⇒ |bias| ≲ 0.0025** (half-spacing).
   Remedy ordered: dense-core grid (spacing ≲ 1e-4) on out-of-sample venues.
2. **R2 (from physics): what the closure validates.** 133 golden events
   (3.9%, exact host IDs, z ∈ [0.011, 0.166]) carry ~100% of the curvature;
   the closure validates host association + generator consistency, **not**
   the completeness/selection machinery (out-voted, tilt ~1.45/h vs the
   ~1.7/h needed to move MAP by 4e-4). Completion-machinery claims rest on
   the E1 exoneration + P–P fallback gate, not on this campaign.
3. **R3 (both): mock-only precision.** The δ-kernel host-z numerator inside
   `generator_marginal` (point/point pairing) carries ~95% of the ln-gap
   cure and deletes peculiar-velocity (~1.25%) and GLADE z-floor (~2.67%)
   errors that dominate the retained 0.54% distance error on real data
   (σ_h degrades ×3.3–×6.8 in the optimistic host-known limit). All
   precision claims are **mock-internal**; real-data mode requires the
   photo-z kernel re-derivation.

## Ordered follow-ups (tracked as GitHub issues)

- **T-5 / F1**: dense-core grid runs, seeds 2000 + 3000 (out-of-sample;
  seed1000 is the development venue), spacing 1e-4 — measure MAP/σ instead
  of extrapolating. [submitted this session]
- **T-1**: blind alternative-truth mock at sealed h_inj — the decisive
  anti-tuning test (plus cheap d_L-rescaling variant as mechanism check).
- **F2/F3**: make the δ-kernel numerator separately selectable from the
  normalization leg (decomposition flag); required for the real-data
  photo-z derivation and for honest per-leg attribution.
- **F4**: `dgen_catalog_selection="4d_exact"` is isotropic/z-pooled while
  β_Ḡ in the same sum is sky-aware/z-conditional, against the code's own
  guardrail comment; packet initially recommended 3d_shared; no rationale
  recorded. Record a derivation-backed rationale or A/B it. (Mitigating:
  frozen before the first probe — not tuned.)
- **F5–F7 (medium, bundled)**: 1.5σ candidate-ball convergence study (ball
  radius became load-bearing under absolute-mass modes); 0.10 fractional-d_L
  cut absent from α(h); physics-floor floored-bin counts unreported in
  combine output.
- **F9 (low)**: adaptive kernel omits Abramson's 1/σ_k factor — measured
  effect <1% on the real 50k pool; citation/doc fix only.

## Out-of-sample note (math §9)

seed1000 is the development venue. The out-of-sample deep evidence is seeds
2000/3000 (+90000 small); all close at the 0.730 node. The seed900 fixpool
re-run and the P–P impostor harness remain in flight and are not part of
this verdict.
