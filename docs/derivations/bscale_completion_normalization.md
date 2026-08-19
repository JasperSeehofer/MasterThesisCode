# The completion-leg normalization factor B_scale — derivation memo

**Date:** 2026-08-19 · **Author of record:** orchestrator derivation for the author's review
(row #130 gate-exit item 2, approved verbatim "all approved" 2026-08-19). **Status:**
DERIVATION COMPLETE — verdict at §6; the corresponding `/physics-change` presentation is §7
and awaits the author's explicit approval before any code changes.

**Object under examination** (`bayesian_statistics.py:4904-4906`):

    B_scale       = beta_Gbar_phi / beta_Gbar        # β_Ḡ^φ(h) / β_Ḡ(h)
    B_num_phi     = B_num     · B_scale              # 1D completion numerator
    B_num_wbh_phi = B_num_wbh · B_scale              # 2D completion numerator

introduced by `FIXB_PATHA_PACKAGE` (2026-08-04) via the line
`B_num^φ = β_Ḡ^φ · L_comp (= B_num · β_Ḡ^φ/β_Ḡ)`
(`docs/derivations/fixb_pathA_phi_marginal_selection.md:82`), justified as a "convention
transfer". Measured on the production runs of record: B_scale(h) rises 0.6503 → 0.6765 over
h ∈ [0.60, 0.86] (d ln B_scale/dh ≈ +0.16); the banked-data counterfactual
(`results/prod2d_closure_20260818/bscale_counterfactual_exploratory.py`, reproduces the
banked posteriors bit-for-bit) shows its h-slope moves the production 2D mean by
+0.119/+0.137 (iiib/joint_r1).

## 1. The mixture from first principles

Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)–(7): for a detected event with data
d_i under population prior p_pop and detection probability P_det,

    p_i(h) = ∫ p(d_i|θ) p_pop(θ;h) dθ  /  ∫ P_det(θ) p_pop(θ;h) dθ .

Two equivalent selection conventions exist and the codebase uses both:

- **OFF convention** (runs of record: `--selection_in_completion_numerator off`): no P_det in
  the numerator; the FULL detection-weighted population integral is the denominator.
- **FUSED convention** ([P2]/[P1], commit 2b10b8b8): P_det (S̄_φ / S_4D via g_sel) inside
  the numerator; the denominator unchanged (β-form). Equivalence holds per-convention as
  long as numerator and denominator use the SAME population and detection model (MFG A2).

Split the population into the catalogue class G (delivered galaxies g, rate weights w_g)
and the dark class Ḡ (density (1−f)·dV_c/(1+z), mass prior φ). The two-class numerator is

    num_i(h) = num_G,i(h) + num_Ḡ,i(h),      p_i = num_i / D(h),

with ONE common denominator D(h) = ∫ P_det p_pop over BOTH classes. In the path-A φ
convention, D = D̃^φ = α_G^φ + β_Ḡ^φ (the catalogue leg deliberately carries the
Malmquist-aware α_G^φ = β_G^φ·r_Malm; that design choice is not under examination here).

## 2. Units: both numerator legs are already commensurate

**Catalogue leg.** The code's `alpha_G_phi · L_cat` reduces algebraically to
Σ_g w_g N_g(h) / n̂_w^φ(h) (the Σ^4D inside α_G^φ = Σ^4D/n̂_w^φ cancels L_cat's
denominator; verified in the gate review and reproduced by the counterfactual assembly).
The division by n̂_w^φ = Σᶲ/β_G^φ is exactly the conversion of a PER-GALAXY sum into the
p_pop measure that the β integrals are expressed in: Σᶲ = Σ_g w_g S̄_φ(z_g) is the
catalogue's detection-weighted count, β_G^φ = ∫ f̄ S̄_φ p_pop dz its population-measure
counterpart, so Σ_g w_g N_g / n̂_w^φ is the class-G event-likelihood integral **in p_pop
units**.

**Dark leg.** B_num (OFF form) = ∫ (1−f_k) p_gw dV_c/(1+z) dz is the DIRECT class-Ḡ
event-likelihood integral in those same p_pop units — dV_c/(1+z) with the completeness
deficit (1−f) IS the dark-class slice of p_pop. Under FUSED, B_num additionally carries
S̄_φ (1D) / g_sel (2D) inside the same integral, matching the numerator convention.
Dimensionally the 2D legs also share the x_M measure (mz_integral and g_i are both
densities in x_M — the addability invariant of the mass_marginal derivation).

**Consequence:** num_G and num_Ḡ are commensurate AS CONSTRUCTED. The MFG assembly is
complete with

    p_i = ( Σ_g w_g N_g / n̂_w^φ  +  B_num ) / D̃^φ        (per channel, per convention)

and there is **no remaining slot for any factor on either leg**. Every consistent
assembly gives B_scale ≡ 1.

## 3. The "convention transfer" claim, examined

The fixb_pathA line rests on reading L_comp = B_num/β_Ḡ as a convention-free per-unit
completion likelihood whose φ-convention numerator is β_Ḡ^φ·L_comp. This would be valid
IFF B_num factored as β_Ḡ(h) × (convention-free shape), i.e. if the legacy detection model
β_Ḡ were an amplitude convention wrapped around physical content. It is not:

1. B_num contains **no detection model at all** in the OFF runs of record (the survival
   enters only under FUSED, and then it is S̄_φ — already the φ model). There is no legacy
   normalization inside B_num to "transfer out".
2. The code itself documents L_comp as **"Diagnostic-only completion likelihood"**
   (`bayesian_statistics.py:4861-4863`) — a reporting ratio, not a derivational object.
3. The multiplication imports d ln(β_Ḡ^φ/β_Ḡ)/dh — the DIFFERENCE of two detection
   models' volume-response slopes (φ-contracted S̄_φ vs the separately-fitted mass-blind
   S_3D with sky-banded completeness) — as a coherent h-dependent multiplier on the leg
   carrying ~93–95% of the mixture weight (w̃_G ≈ 0.06). This is precisely the MFG-A2
   violation (two detection models inside one likelihood) that the path-A package was
   written to eliminate from the catalogue leg; the transfer line re-installed it on the
   completion leg. Being dimensionless, it is invisible to dimensional analysis — the same
   failure mode as the Phase-14 /(1+z) bug.

## 4. The nearby genuine inconsistency (bounded, separate)

B_num's completeness deficit uses the sky-banded f_k(z) while β_Ḡ^φ uses the isotropic
f̄(z). That IS a real (numerator vs denominator) population inconsistency — but its remedy
is to evaluate the pair with one f-treatment (a separate, bounded item; the f_k vs f̄
difference is a completeness-anisotropy correction, not a detection-model swap), NOT to
multiply by a global ratio of detection models: B_scale ≈ 0.65 is dominated by
S̄_φ-vs-S_3D normalization and carries no f-banding content. This item joins the D3/F10
J_α point-vs-kernel entry on the tilt ledger as a documented, bounded, unmeasured term.

## 5. Limiting cases and consistency checks

- **S̄_φ ≡ S_3D limit:** if the two detection models coincided, β_Ḡ^φ = β_Ḡ and
  B_scale = 1 — the shipped formula degenerates to the derived one, confirming that the
  factor measures exactly the model mismatch, not physics.
- **Harness cross-check:** the validation harness (`validation/pp_coverage.py`)
  implements the same two-class mixture with NO B_scale analog and calibrates in-venue
  (rows #109–#116) — the derived form is the one that passes coverage.
- **Counterfactual (verified, banked data):** removing B_scale (≡1) lands the 2D mean at
  0.6771/0.6788, i.e. ≈ 0.05 BELOW truth — the removal re-exposes the shared base low tilt
  (the un-transferred row-#111 1D correct-form terms + remaining ledger entries). This is
  the expected behavior of a balance-point posterior and is NOT evidence for keeping the
  factor; it fixes the sign of the next budget iteration.
- **1D consistency:** the same factor multiplies the railed 1D channel's completion leg
  (B_num_phi), where the rail hides it (removal moves 1D 0.604 → 0.601) — consistent with
  a common-mode completion-leg multiplier, and with why no 1D instrument ever saw it.

## 6. Verdict

**B_scale = β_Ḡ^φ/β_Ḡ is a DEFECT (un-derived normalization), not a convention choice.**
The derived form is

    B_num_phi     = B_num          (1D)
    B_num_wbh_phi = B_num_wbh      (2D)

under both selection conventions, with each convention's own survival placement inside
B_num as already implemented. No derivation supports any other factor; the burden-of-proof
question posed by the gate review is answered: the formula cannot be derived.

## 7. Physics-change gate presentation (author approval required before implementation)

- **Old formula** (`bayesian_statistics.py:4904-4906`): `B_num^φ = B_num · β_Ḡ^φ/β_Ḡ`
  (both channels), per fixb_pathA §2 line 82.
- **New formula:** `B_num^φ = B_num` (both channels; delete the B_scale multiplication;
  keep L_comp as the documented diagnostic). fixb_pathA §2's transfer line is corrected by
  an appended erratum note referencing this memo.
- **Reference:** Mandel, Farr & Gair (2019) arXiv:1809.02063 Eqs. (5)–(7) + this memo §§1–3
  (units/measure argument); Gray et al. (2020) arXiv:1908.06050 two-class structure.
- **Dimensional analysis:** B_scale is dimensionless — the change is invisible to units;
  the binding check is the MEASURE argument (§2): both legs are p_pop-measure integrals as
  constructed, x_M-density-paired in 2D.
- **Limiting cases:** (i) S̄_φ ≡ S_3D ⇒ old = new (no-op); (ii) harness (no B_scale)
  calibrates in-venue; (iii) counterfactual removal on banked data = the new formula's
  posterior EXACTLY (0.6771/0.6788 / 1D 0.601-0.602) — the validation bed is already
  banked and verified.
- **Regression test:** pin the old combined values at two probe h per venue (pre-change),
  assert the new path reproduces the bscale-removed counterfactual values; keep an
  instrument flag `--completion_b_scale legacy` for one release to preserve
  counterfactual reproducibility of the historical runs (default = derived form).
- **Expected consequence (stated for honesty):** the production 2D posterior moves from
  +0.054/+0.067 ABOVE truth to ≈ −0.053/−0.051 BELOW truth. The bias does not vanish —
  the tilt ledger's remaining entries (lapsed row-#111 1D correct-form terms, f_k vs f̄,
  J_α, s_Edd re-measurement) become the open budget, now on a derivation-complete
  completion normalization. This is the correctness-over-bias-removal value applied
  literally.
