# PHYSICS-CHANGE PROPOSAL — the catalogue-class mixture weight under the latent model (the [P2]+[P3] completion)

**Date:** 2026-08-22 · **Status:** PROPOSED — presented, then STOP (author-gated per
`/physics-change`; row #163 item "Author the derivation + proposal")
**Subject file:** `bayesian_inference/bayesian_statistics.py` (trigger file; nothing changes
until the author rules) · **Evidence base:** rows #159–#164; `PREREGISTRATION_P3_TWIN_20260822.md`
(verdict + amendments 4–11); L6-DER3 §4; `PROPOSAL_GRAY_CONVENTION_PAPER_INTEGRATION_20260817.md`.

## 0. The claim being repaired (and what the measurements did and did not establish)

The completion leg's S̄_φ pairing was repaired by `fused` (rows #117–#118, confirmed O6/O7).
The catalogue leg still carries the coded arrangement: per-event term
`(α_G_φ/r_Malm)·L_cat/D̃_φ`, where the class weight α_G_φ ∝ β_G_φ = ∫ f̄·S̄_φ·w_pop dz is
GLOBAL (all-sky, z-integrated) and `L_cat` is an S̄-free per-candidate average. The measured
twin cell (per-candidate S̄_φ(z_g) inside the average, global weight retained) moves the
12-seed headline by **+0.0155 ± 0.0037**, and the effect is **~94–98% the per-event
host-z-dependent suppression** of the leg's mixture weight (amendment 9), with a **null
residual h-tilt** (SHAPE-NULL, row #164).

**Scope warning the author must weigh first [MEASURED-context, decisive]:** the B-SEL venue is
ALL-IMPOSTOR by construction (AMENDMENT A-2: the true host is never a candidate). In this
venue, ANY suppression of the catalogue leg reduces the drag mechanically. **The +0.0155 is
evidence of LEVERAGE, not of correctness** — in a venue with genuinely catalogued hosts the
same per-event factor also suppresses true-host contributions, and its net calibration effect
is unmeasured. Correctness must come from (a) the derivation below and (b) the §5 verification
plan's catalogued-host venue leg. No production adoption is proposed on the venue evidence
alone.

## 1. The derivation (latent-thresholded model, no-BH channel)

Per event, conditional on detection, marginalizing host identity, the likelihood of the data x
splits over the host's catalogue membership. Writing the population intensity λ(z,Ω) =
w_pop(z)·(sky measure), completeness f_k(z,Ω), and the mass-marginal survival S̄_φ(z;h)
(the generator-matched acceptance factor, proven at the A20/O4 review):

    p(x | det, h) ∝ [ ∫ dz dΩ f_k·λ·S̄_φ·p_gw(x|z,Ω,h) ]_cat + [ ∫ dz dΩ (1−f_k)·λ·S̄_φ·p_gw ]_dark
                    ————————————————————————————————————————— all over  ∫ f_k λ S̄_φ + ∫ (1−f̄) λ S̄_φ

The dark branch is exactly the fused completion leg (numerator carries S̄_φ; normalizer
β̄_Ḡ_φ = ∫(1−f̄)·S̄_φ·w_pop). For the catalogue branch, the intensity ∫ f_k·λ·S̄_φ·p_gw is
represented by the catalogue itself: the cone's candidates ARE the realized draw of f_k·λ, so

    [cat] ∝ Σ_g  S̄_φ(z_g; h) · ⟨p_gw⟩_{kernel_g(h)}        (per-candidate survival, INSIDE the sum)

with kernel_g the host-z posterior kernel. **The latent model puts S̄_φ(z_g;h) on each
candidate; there is no arrangement of it as a global, candidate-independent class factor** —
the global β_G_φ = ∫f̄·S̄_φ·w_pop is the EXPECTATION of Σ_g S̄_φ(z_g) over realizations, not
its per-event value. The coded arrangement replaces a per-event, h-dependent random weight by
its ensemble mean; the twin cell restores the per-event form. That is the entire content of
the fork — and it is why the measured effect is per-event level-like (amendment 9): the
correction IS the fluctuation of Σ_g S̄_φ(z_g) about its ensemble mean, event by event.

**Consistency requirement (the derivation's testable core):** under the generator, the
posterior's expected catalogue-class responsibility must equal the generator's realized
catalogue-hosted fraction. The coded arrangement satisfies this only in ensemble mean; the
per-event form satisfies it event-conditionally. In the all-impostor B-SEL venue the realized
fraction is 0 and NEITHER arrangement is calibrated there — the identity must be tested in a
venue with catalogued hosts (§5).

## 2. Old formula / New formula (items 1–2 of the gate package)

- **Old:** per-event catalogue term `(α_G_φ/r_Malm) · L_cat · / D̃_φ` with
  `L_cat = weighted-average of S̄-free normalized host kernels` (`:5902-5907` scalar,
  `:6511-6522`-region batch; class weight global via `β_G_φ` `:2065`, `α_G_φ` `:2423-2427`).
- **New:** the same term with each candidate's kernel carrying `S̄_φ(z;h)` inside the z-integral
  — mechanically, the ALREADY-IMPLEMENTED `catalogue_numerator_survival="phi"` cell
  (merged instrumentation, default off) promoted to default, **together with the matching
  normalizer bookkeeping**: the class-weight chain (n̂_w, r_Malm, α_G_φ) re-derived so the
  global factor is not double-counted — the ensemble-mean content of β_G_φ that the per-event
  factors now carry must be divided OUT of the class weight (replace β_G_φ by
  β_G ≡ ∫ f̄·w_pop — the S̄-free catalogue mass — in α_G_φ's construction, so
  weight × per-candidate-S̄ has the same ensemble mean as before). **This normalizer half was
  NOT part of the measured twin cell** — the twin held α_G_φ fixed (disclosed at
  registration), which is why its ~+0.0155 includes an ensemble-level suppression the
  completed arrangement would largely return. The completed pair's expected net effect is the
  FLUCTUATION term only — plausibly of order the shape+level heterogeneity, NOT +0.0155.

## 3. Reference · Dimensional analysis · Limiting case (items 3–5)

- **Reference:** L6-DER3 §4 ("the catalogue leg is the same fork, per-galaxy"); the A-FULL
  addendum's marginalization (Σ_k host_k/imp_k structure); Gray et al. (2020) Eq. (A.10) and
  MFG (2019) as the convention being departed from — with `docs/LITERATURE_WARNINGS.md` MFG-a
  still UNCHECKED-verbatim (a §5 obligation before the paper quotes it) and G20-d's
  completeness-floor scope note.
- **Dimensional:** S̄_φ dimensionless ∈ [0,1]; β_G → β_G_φ swap preserves the weight's units;
  the per-candidate factor multiplies a normalized kernel — the term's measure is unchanged.
- **Limiting cases:** S̄_φ ≡ 1 recovers the coded arrangement exactly (both halves);
  S̄_φ constant < 1: per-candidate factors × the re-derived weight cancel exactly by
  construction (the K-flat/normalizer identity — measurable as a regression test);
  σ_z → 0, single candidate at the true host: the arrangement reduces to the selected-prior
  single-host form (the FULL-D structure the venue chain validated).

## 4. What is proposed (decision table)

| # | item | tag | recommendation |
|---|---|---|---|
| 1 | Adopt the completed per-event pairing (numerator factor + re-derived class weight) as a REGISTERED CANDIDATE, not production | [RULE] | measure first (§5); no production change now |
| 2 | The §5 verification plan | [DO] | approve |
| 3 | If §5's identity test passes in the catalogued-host venue: production adoption returns as its own 6-item gate | [STANDING structure] | note only |

## 5. Verification plan (before any adoption)

1. **Normalizer completion instrument (zero/low compute):** implement the β_G (S̄-free) class
   weight as a paired flag; measure the COMPLETED arrangement on the 12 B-SEL seeds — the
   registered expectation is |Δ̄_completed| ≪ +0.0155 (the fluctuation term), two-sided.
2. **Catalogued-host venue leg (the decisive one):** a b0/catalogue-mode arm (hosts genuinely
   in the candidate set) scoring the class-responsibility identity: E[posterior catalogue
   responsibility] vs the generator's realized catalogue fraction, coded vs completed
   arrangements — the calibration criterion neither B-SEL nor C-SG can test. Costing to be
   lined before launch (A6/A17); A21/A22 throughout.
3. **MFG-a verbatim verification** (Stage-L obligation) before any paper-facing claim.

**STOP.** Presented for the author's ruling; no code path changes, no registration executes,
until items 1–2 of §4 are ruled.

---

## APPENDIX A (2026-08-22, post-amendment-12; the §1/§2 inconsistency RESOLVED by derivation — presented for ratification)

**The inconsistency (amendment 12):** §2's prose said β_G_φ→β_G "in α_G_φ's construction";
α_G_φ also enters D̃_φ. Completing D̃_φ flips the measured verdict (+0.0344 vs −0.0028) — a
15× lever. Which reading does the latent model force?

**Resolution — the numerator-only completion is the derivation; D̃_φ stays S̄-weighted.**
In §1's display the denominator is the TOTAL SELECTED MASS, `∫f_k λ S̄_φ + ∫(1−f̄) λ S̄_φ` —
both class terms S̄-weighted. That is D̃_φ = α_G_φ + β̄_Ḡ_φ **as coded**: the denominator is
the selected-population normalization ("α untouched", exactly as the venue chain's A-FULL
result held), and un-selecting it (the +0.0344 arm) departs from the latent model. The
symbol α_G_φ plays TWO roles: (i) in D̃_φ it is the selected catalogue MASS — stays
S̄-weighted; (ii) as the numerator's class weight it is the ensemble stand-in for the
per-event Σ_g S̄_φ(z_g) — and THERE the completion replaces class-mean-S̄ content with the
per-candidate factors (β_G × per-candidate S̄_φ, whose ensemble mean is β_G_φ, preserving
D̃_φ-consistency in expectation). §2's wording was the error; §1's display was right; the
instrument's construction is the derivation-coherent candidate. The reviewer's +0.0344 arm is
banked as the measured cost of the WRONG reading, not as a live candidate.

**Consequences if ratified:**
1. The registered candidate's venue effect stands at **−0.002810 ± 0.000467** (6σ, small,
   negative; a magnitude bound under this now-derivation-fixed convention).
2. The catalogued-host responsibility-identity test UNBLOCKS (the candidate is no longer
   ambiguous) and proceeds per its granted [DO] with the costing line
   (~8 paired b0 runs ≈ 4–5 CPU-h + one instrumentation flag).
3. The remaining bias story is unchanged: the impostor drag's convention component is bounded
   small under the coherent completion; the drag itself (−0.079 within −0.108) stays venue
   physics *pending* the identity test's correctness verdict — stated as pending, not assumed
   (amendment-16 discipline).

**[RULE] asked of the author:** ratify Appendix A's resolution (numerator-only completion is
the candidate; D̃_φ stays as coded), unblocking §5 item 2.

---

## APPENDIX A — REFUTED (2026-08-22; adversarial derivation check, banked verbatim in `A20_REVIEW_APPENDIX_A_20260822.md`; the author's conditional ratification does NOT take effect)

The verifier's verdict: **APPENDIX-FALLS.** Half survives — D̃_φ stays selected (the +0.0344
reading refuted by two independent routes). The ratified half is refuted: **β_G_φ also stays.**
Under `absolute_marginal`, `L_cat` divides by the GLOBAL selection sum (verified at source,
`:4896-4901`), so β_G_φ/Σ_glob = 1/n̂ is a **measure conversion whose S̄ content cancels** —
the repo's own ratified derivation (`bscale_completion_normalization.md` §2: "no remaining slot
for any factor on either leg"). The R = β_G/β_G_φ rescale is a **B_scale-class un-derived
multiplier** (a ratio of one leg's integral under two detection models — the exact defect class
rows #130–#131 removed from the completion leg). Two independent confirmations: the
self-normalization table (only the twin leaves the class mass unchanged) and the S̄→cS̄
homogeneity test (only the twin is invariant — on the fused basis).

## APPENDIX B — the verifier's replacement resolution (PROPOSED; returns to the author)

(i) **D̃_φ stays selected** (Appendix A's surviving half; the +0.0344/+0.0424 arms banked as
the cost of the wrong reading). (ii) **β_G_φ stays** — no double count exists; the
derivation-coherent candidate is **the twin cell as measured: Δ̄ = +0.015524 ± 0.003657**
(12/12 positive, 4.24σ), with amendment 15's "double-count +0.018334" RE-LABELLED as the
spurious R-inflation and −0.002810 as the twin contaminated by it. (iii) **Basis conditional:**
the twin's coherence (S̄→cS̄ invariance) holds only with the completion leg `fused`; ALL banked
P3 numbers are off-basis, where no arrangement is invariant — **P3 adoption is inseparable from
the off-vs-fused basis fork and must be re-measured on `fused`.** (iv) The catalogued-host
identity test proceeds with THREE arms (coded / twin / R-rescaled), on the fused basis, as the
empirical adjudicator. **Secondary finding (separate thread, [AGENT] pending re-measurement):**
the no-BH `L_cat`'s global divisor is the mass-blind Σ³ᴰ while its weight β_G_φ is
φ-marginal-4D — a possible residual r_φ = Σ^φ/Σ³ᴰ ≈ 0.91 slot mismatch contested against Path
A's "all three slots" claim; opened as `[P3-RPHI]` stage 0.

**[RULE] asked of the author:** ratify Appendix B (i)–(iv), or rule otherwise. Nothing adopts
until then.
