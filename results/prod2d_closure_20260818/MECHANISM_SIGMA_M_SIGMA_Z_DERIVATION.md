# Mechanism derivation — why σ_M × σ_z produces a positive h-pull in the 2D channel

**Date:** 2026-08-19 · **Status:** DRAFT v1 (runbook 21 §3 item 2) · **Author-gated:** this is
analysis, not a code or estimator change; any fix branch returns via `/physics-change`.
**Blindness declaration:** written while cluster job 6364821 is RUNNING. Evidence already seen:
T0 (`tier0_output.json`), the five interim off cells (anchor (0.035, 0.55), σ_z ladder
{0.010, 0.002} at σ_m = 0.55, V-prod off, V-deep 1D off-basis reads), and the prodcal
σ_m = 0.30 class (+0.009…+0.012). **NOT seen:** all fused T2 grid cells, the σ_m ∈
{0.30, 0.10, 0.02} rungs of this campaign, T1 toggles, V-prod fused. §5's predictions for
those cells are blind and registered here before readout.

## 1. The object under derivation

Interim harness evidence (runbook 21 §1): at production-mapped errors (σ_z = 0.035,
σ_m_gal = 0.55) the V-deep off cell shows a 2D bias **+0.038…+0.042**; the σ_z ladder at
fixed σ_m = 0.55 collapses it (+0.040 → +0.007 at σ_z = 0.010 → +0.002 at σ_z = 0.002); the
σ_m ladder at fixed σ_z = 0.035 (prodcal 0.30 class) gives ~×3.8 growth from 0.30 to 0.55.
So the 2D mass-error bias needs BOTH errors — a genuine σ_M × σ_z coupling, and it is
positive in h. The question: what in the 2D likelihood geometry produces that sign and that
coupling?

## 2. The 2D geometry (harness form; production analogs cited)

Per event with observed (d_L,obs, M_z,det), per candidate galaxy g at photo-z z_g with
observed mass M_g, the catalogue-leg contribution at hypothesis h is (pp_coverage.py:2313-2336;
production analog `mz_integral`, bayesian_statistics.py:5522):

    w_g(h) = ∫ dz N(z; z_g, σ_z) · w_pop(z) · p_GW(d_L(z;h) | d_L,obs) · D_g(z;h)

    D_g(z;h) = N( μ_cond(z;h) ; μ_gal,g(z) , s ),   s² = σ_cond² + σ_gal²
    μ_cond(z;h) = 1 + ρ(σ_M/σ_D)·(d_L(z;h)/d_L,obs − 1)   [≈ 1; σ_cond ≈ 0 at the
                                                            production operating point, row #118]
    μ_gal,g(z) = M_g (1+z) / M_z,det
    σ_gal = σ_m_gal · μ_gal,g(z)          [heteroscedastic: width ∝ the ratio itself]

The completion leg multiplies its integrand by g_i / g_sel (pp_coverage.py:1760/1827;
production bayesian_statistics.py:2022/2155), which at σ_cond → 0 and φ flat-in-lnM reduces to
g_i ≈ 1/(μ_cond · ln(M_max/M_min)) — approximately z- and h-FLAT wherever the implied source
mass M_z,det/(1+z) stays inside the φ support, with hard edges where it exits.

Three structural facts drive everything below:

- **(F1) The mass overlap is an INVERSE read.** D_g prefers the z where μ_gal = μ_cond ≈ 1,
  i.e. (1+z_pk,g) = M_z,det / M_g. The candidate's noisy mass enters INVERTED. For
  fractional noise M_g = M_true(1+δ), δ ~ N(0, σ_m): E[1/(1+δ)] = 1 + σ_m² + O(σ_m⁴), so the
  mass-preferred redshift is systematically HIGH: E[Δ(1+z_pk)]/(1+z) ≈ +σ_m². (At σ_m = 0.55
  the Gaussian fractional model's inverse moment is formally divergent/heavy-tailed; the
  quadrature-realized preference is finite but large and still high-sided — the +σ_m² law is
  the small-σ_m limit of a monotone effect.)
- **(F2) The heteroscedastic width tilts the same way.** With s ∝ μ_gal ∝ (1+z), D_g falls
  slower on the high-z side than the low-z side. Second-order expansion of ln D around
  μ = 1 (u = μ−1): ln D ≈ −u²/(2σ_m²) + u³/σ_m² − u + const; the perturbation ⟨u⟩ =
  ⟨u⁴⟩/σ_m² − ⟨u²⟩ = +2σ_m² — again a high-(1+z) preference, again ∝ σ_m².
- **(F3) The completion leg carries no σ_m penalty.** D_g's peak density scales as
  1/(√(2π)·σ_m·μ_gal): every doubling of σ_m halves the catalogue candidates' weight, while
  g_i (the completion mass factor) is σ_m-independent. σ_z acts the same way on the z kernel
  (peak candidate weight ∝ 1/σ_z at fixed offset, and the alignment probability of a
  candidate with the GW shell degrades as σ_z grows).

## 3. Why σ_z is the lever arm (the coupling)

p_GW is narrow in z (σ_dL/dL is 1e-3…1e-2 class), so at each h the z-integral is pinned near
z*(h) solving d_L(z;h) = d_L,obs; z*(h) is INCREASING in h (d_L = a(z)/h ⇒ a(z*) = h·d_L,obs).
The per-candidate likelihood in h is therefore approximately the z-profile of the slowly
varying factors evaluated along z*(h):

    L_g(h) ≈ N(z*(h); z_g, σ_z) · D_g(z*(h)) · w_pop(z*(h))

A high-z preference in D_g (F1+F2) becomes a high-h preference — but ONLY to the extent the
photo-z kernel allows z* to sit above z_g. The stationary point shifts by

    δz ≈ σ_z² · ∂ ln D_g/∂z |_(z_g)      ⇒      δh/h ≈ δz · d ln a/dz |_(z*)

i.e. the realized tilt is the mass-side slope (∝ σ_m², F1+F2) times the photo-z variance
σ_z² — the product form of the observed coupling: **collapse with either error, positive sign
fixed by the inverse-mass read.** This is Malmquist/Eddington-type: a noisy inverted
observable plus an asymmetric kernel, exactly the class named in runbook 21 §3.2.

**Magnitude check (honest gap):** with σ_z = 0.035, the within-kernel shift caps at
δz ≈ σ_z² × O(1…10) ≈ (1…12)e-3, i.e. δh/h up to ~1-2% — the right sign and coupling but
likely a factor ~3-20 SHORT of the measured +0.040/0.73 ≈ 5.5%. So M-A below is unlikely to
be the whole story; the leg-rebalance channel (M-B) has no such cap. The T2 surface and the
production regression discriminate.

## 4. Three candidate sub-mechanisms and their discriminating predictions

- **M-A — within-kernel inverse-mass shift (F1+F2 through §3).** Predicts bias ∝ σ_z^2
  exactly (the σ_z² lever), ∝ σ_m² at small-moderate σ_m; per-event h-slope concentrated in
  events whose TRUE host is in the ball with a well-aligned candidate; magnitude small
  (≤ ~0.02 at the anchor).
- **M-B — catalogue→completion re-balance (F3).** Both σ_m and σ_z DILUTE the catalogue
  (host-bearing) term's peak density relative to the σ-independent completion floor; the
  posterior inherits more of the completion leg's own high-h tilt (the raised-d50/Malmquist
  completion geometry — the V-prod off interim shows +0.008…+0.015 of exactly that flavor,
  and b_num's support starts at z_support with a rising w_pop). Predicts: per-event positive
  h-slope correlates with per-event completion share (production columns g_frac,
  B_num_wbh/(L_cat_with_bh + B_num_wbh)); magnitude NOT capped by σ_z² (the mixture weight
  can move order-unity); scaling in σ_m closer to the density dilution 1/σ_m saturating once
  the completion term dominates; still needs σ_z > 0 (at σ_z → 0 the true host's kernel
  density ∝ 1/σ_z grows and re-anchors the catalogue leg — the coupling survives).
- **M-C — completion-leg mass-support truncation (edge of φ).** g_i is z-flat except where
  M_z,det/(1+z) exits [M_min, M_max]; events with extreme M_z,det get one-sided z-support
  truncation ⇒ h-tilt concentrated in the mass-extreme tail. Predicts per-event slope
  correlates with |ln(M_z,det/M_ref)| extremity, not with σ_M or completion share.

**Composite expectation:** M-B dominant (magnitude), M-A additive (same sign, gives the clean
σ_z² small-σ_z limb), M-C a tail contributor at most (production φ has the Babak kink but
broad support).

## 5. Blind registered predictions (before the fused T2 cells return)

Response-surface model: bias(σ_z, σ_m) ≈ A · (σ_z/0.035)^p · r(σ_m), A = +0.040 ± 0.004
(anchor read), p ∈ [1.4, 2.0] (pure M-A ⇒ p = 2; measured interim ladder 0.040→0.007
suggests p ≈ 1.5 at the top limb — an M-B share), r(0.55) = 1, r(0.30) ≈ (0.30/0.55)^q with
q ∈ [1.5, 2.2] (prodcal class ⇒ q ≈ 1.9).

Per-cell blind predictions (2D bias, fused cells, V-deep, production noise; fused ≈ off per
the certified +0.001-class selection lever):

| σ_z \ σ_m | 0.55 | 0.30 | 0.10 | 0.02 |
|---|---|---|---|---|
| 0.035 | +0.040 (seen, off) | +0.010…+0.015 | +0.001…+0.004 | |bias| ≤ 0.002 |
| 0.010 | +0.007 (seen, off) | +0.002…+0.004 | |bias| ≤ 0.002 | |bias| ≤ 0.002 |
| 0.002 | ≤ +0.003 (seen, off) | |bias| ≤ 0.002 | |bias| ≤ 0.002 | |bias| ≤ 0.002 |

Falsifiers: (i) a fused cell FAR off this surface (>3× or sign flip at a σ_m ≥ 0.30 rung)
kills the product form; (ii) NO correlation of production per-event h-slope with completion
share AND none with per-event σ_M kills M-B and M-A respectively (then M-C or a new class);
(iii) bias at (0.002, 0.55) > +0.005 kills the σ_z lever-arm claim outright.

## 6. Production-native discriminant (feeds the §3-item-1 regression prereg)

The T0 CSV already carries per event × h: L_cat_with_bh, B_num_wbh, g_frac, combined legs —
so the per-event 2D h-slope AND the per-event completion share are both production-native
free reads. The regression separates M-A/M-B/M-C by which covariate carries the positive
slope mass: candidate-σ_M (M-A), completion share (M-B), M_z,det extremity (M-C). Bands to
be registered in the regression prereg BEFORE running (separate document).

## 7. What this is not

No estimator defect is asserted here. F1-F3 are properties of a CORRECT likelihood evaluated
with a point-estimate treatment of the galaxy mass error inside `mz` (the σ_gal Gaussian IS
the modeled error) — whether the residual is a defect (fix fork a/b) or an information limit
(fork c, document-as-systematic) is exactly the author's fork, to be presented with the
budget. Consistent with [[author-values-correctness-over-bias-removal]].
