# The completion numerator's data measure — derivation memo

**Date:** 2026-08-20 · **Status:** derivation complete; **falsifier registered (§5) and NOT
yet run** — per amendment A14 this attribution is PROVISIONAL until B-DEN returns. §6 is the
`/physics-change` gate presentation for the instrument and awaits the author.

**Object:** the internal misnormalization confirmed by B-SEL — the completion leg is biased
low by **−0.112 ± 0.0017** in a universe matched to the estimator's own population *and*
selection (ledger row #140), and the fused-convention bisection has excluded the
numerator/denominator detection-weight asymmetry as its cause (**−0.1163 ± 0.0010**, A-4
verdict CONVENTION-NOT-IT).

## 1. What the code computes

Completion numerator (`bayesian_statistics.py:4852-4877`), per dark event with measured
distance d̂ ≡ `_comp_det_d_L`:

    B_num(h) = ∫ dz (1 − f_k(z)) · p_gw(z;h) · [dV_c/dz]/(1+z),
    p_gw(z;h) = N( d_L(z;h)/d̂ ; μ_frac, σ_frac ) · sinθ/(4π)

Denominator (`precompute_phi_selection_integrals`), same population measure
p_pop = dV_c/dz/(1+z) — verified identical — with the detection weight attached:

    β_Ḡ^φ(h) = ∫ dz (1 − f̄(z)) · S̄_φ(z;h) · p_pop(z;h)

## 2. The mismatch: the two legs are normalized over different measures

A mixture likelihood over DETECTED data requires that the numerator, integrated over the
data space, reproduce the denominator. Integrate B_num over d̂ at fixed (z, h). Substituting
u = d_L/d̂, so d̂ = d_L/u and dd̂ = −(d_L/u²) du:

    ∫ dd̂  N(d_L/d̂ ; μ_frac, σ_frac) = d_L(z;h) ∫ du N(u; μ_frac, σ_frac)/u²
                                      = d_L(z;h) · [1/μ_frac² + 3σ_frac²/μ_frac⁴ + …]
                                      ≈ d_L(z;h)     (μ_frac ≈ 1, σ_frac ≪ 1)

**It is not 1 — it is proportional to d_L(z;h).** The term `p_gw` is a density in the
dimensionless ratio, not in the observable d̂; converting costs a factor d_L(z;h), which
depends on BOTH the integration variable and h. Hence

    ∫ dd̂ B_num(d̂;h) ≈ (sinθ/4π) ∫ dz (1−f) p_pop(z;h) · d_L(z;h)
                     = (sinθ/4π)(1/h) ∫ dz (1−f) p_pop(z;h) · a(z),   d_L = a(z)/h

whereas the denominator it is divided by is ∫ dz (1−f̄) S̄_φ p_pop. **The numerator's implicit
normalization weights the population by an extra d_L(z;h); the denominator does not.** The
mismatch is not a constant — it carries an explicit 1/h.

## 3. Predicted sign and size (the part that makes this testable)

Write the code's ratio in terms of the correctly paired one:

    ln[B_num/D̃^φ]_code = ln[B_num/D̃^φ]_correct + ln( ⟨a⟩_w / h ) + const,
    ⟨a⟩_w ≡ ∫(1−f̄)S̄_φ p_pop a dz / ∫(1−f̄)S̄_φ p_pop dz

Differentiating, and using that an unbiased estimator has zero expected score at truth:

    **E[score]_code ≈ −1/h + d ln⟨a⟩_w/dh ≈ −1.37 + (subleading)**  per event at h = 0.73.

Measured on production's dark class: **−0.635 ± 0.017**. Same sign, same order, roughly a
factor 2 apart — the residual is exactly what the subleading term (⟨a⟩_w's own h-dependence,
plus the f_k-vs-f̄ and S̄-weighting differences between the two integrals) is expected to
supply. The leading term is parameter-free.

**Why the saddle-point intuition fails here (and why an earlier guess of mine was wrong).**
Inside the numerator's z-integral the GW factor is sharp, pinning d_L(z*;h) = d̂ — a
CONSTANT — so the missing factor contributes no h-slope *within* the numerator. The defect is
not in the numerator's shape at all: it is that the numerator and denominator normalize over
different population weightings, and the denominator is a broad integral where the missing
d_L ∝ 1/h does not cancel. This also explains why the fused convention changed nothing
(A-4): adding S̄_φ to the numerator fixes the *detection* weighting, not the *measure*.

## 4. What this predicts elsewhere (consistency checks, all already banked)

- **Catalogue-supported events should NOT show the same low pull**, since their z-integral is
  pinned by a narrow host kernel rather than normalized against the broad dark integral.
  Observed: the in-catalogue class sits at 0.828 (score +1.507), the opposite direction.
- **The effect must be present in the 2D channel identically**, since the measure issue is in
  the distance factor, not the mass factor. Observed: dark class 0.6001 (1D) vs 0.6004 (2D).
- **It must survive in a model-matched universe.** Observed: B-SEL −0.112, B-SELF −0.116.

## 5. Falsifier (registered per A14; B-DEN, A-5)

Instrument the completion numerator's event term with the proper data-space density —
divide by d_L(z;h), equivalently evaluate `N(d̂; d_L, σ_frac·d_L)` — everything else
untouched, and run it in B-SEL's model-matched universe.

- **MEASURE-OWNS-IT:** |bias| ≤ max(0.005, 2·SE) with C68 inside the N=15 binomial band ⇒
  this memo's attribution is confirmed and the correct form is a `/physics-change`.
- **MEASURE-PARTIAL:** |bias| ≤ ½·0.112 but material ⇒ contributing term, not the whole.
- **MEASURE-NOT-IT:** otherwise ⇒ **this memo is wrong**; the next target is D̃^φ's class
  composition (a dark event's numerator carries only the dark term while its denominator
  carries α_G^φ + β_Ḡ^φ).

## 6. `/physics-change` gate presentation (author approval required before implementation)

- **Old:** `p_gw = N(d_L(z;h)/d̂ ; μ_frac, σ_frac)·sinθ/(4π)` — a density in the distance
  RATIO (`bayesian_statistics.py:4852-4877`), integrated against dz.
- **New (behind a default-off flag, e.g. `--completion_event_measure {ratio,data}`):**
  `p_gw = N(d̂ ; d_L(z;h), σ_frac·d_L(z;h))·sinθ/(4π)` — the same Gaussian measurement model
  expressed as a density in the observable, so that ∫dd̂ p_gw = 1 and the numerator normalizes
  to the denominator's measure.
- **Reference:** Mandel, Farr & Gair (2019) arXiv:1809.02063 Eqs. (5)–(7) (numerator and
  selection normalization must be the same measure over the same data space); Gray et al.
  (2020) arXiv:1908.06050 Eq. (A.19); this memo §2.
- **Dimensional analysis:** the new form is 1/Gpc (a density in d̂); the old is dimensionless.
  The ratio to the denominator's units is what the derivation fixes — this is precisely the
  defect, and it is invisible to a units check of the integrand alone because d̂ is a
  per-event constant that hides the scale.
- **Limiting cases:** σ_frac → 0 ⇒ both forms collapse to the same delta at d_L = d̂ (pinned
  test); at fixed z and h the two differ only by the constant d_L/d̂², so a single-event
  likelihood SHAPE in z is unchanged — only the cross-event/denominator normalization moves
  (pinned test); default flag ⇒ bit-identical production (N-0 gate against banked output).
- **Regression bed:** B-SEL's 12 banked seeds (model-matched, −0.112) are the reference the
  instrument must move; the production off-basis baselines are the bit-identity pins.
- **Expected consequence if confirmed:** the dark class stops railing; production's 1D and 2D
  posteriors both move up substantially. The size is NOT predicted here beyond the leading
  −1/h term — it is measured by B-DEN before any production default changes.
