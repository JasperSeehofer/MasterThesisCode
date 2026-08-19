# J_α — the catalogue-selection mass kernel: derivation memo

**Date:** 2026-08-20 · **Status:** derivation complete; §5 is the /physics-change gate
presentation, awaiting the author. **Provenance:** D3/F10 documented open term (fixb_pathA
§"Mass evaluation"); promoted to a formal correction candidate by the author (ledger row
#134, verbatim); measured production response: ΔJ = −0.0025 (iiib) / −0.0061 (joint_r1),
r_Malm +4.0–7.8% (`PREREGISTRATION_TILT_BATTERY.md` VERDICT — the validation bed).

## 1. The inconsistency, precisely

The with-BH-mass catalogue-selection sum point-evaluates the survival at the measured mass:

    Σ⁴ᴰ(h) = Σ_g w_g · S_4D(d_L(z_g;h), M_g(1+z_g))        (:2707-2731, point form)

while the per-event catalogue numerator and its D_g marginalize the SAME galaxy's mass
measurement error: the mz overlap carries σ_gal = M_error·(1+z)/M_z,det (:5515-class) and
its mean is the Eddington-shifted M_eff (`eddington_shifted_host_mass`, the G2d
moment-matched posterior mean under the mass prior). One quantity — the galaxy's unknown
true mass — is treated as a random variable in the numerator and as a point in the
selection normalization. That is the MFG (2019) A2 violation: numerator and selection
normalization must use the same population AND measurement model.

## 2. The two self-consistent resolutions, adjudicated

**(a) Kernel-everywhere.** The catalogue datum is (M_g, σ_g); the galaxy's true mass has
posterior p(M | M_g, σ_g) with G2d moment-matched form N(M_eff,g, σ_g²). A2 then fixes the
selection term to the SAME posterior expectation:

    Σ⁴ᴰ_kernel(h) = Σ_g w_g · E_{M ~ N(M_eff,g, σ_g²)} [ S_4D(d_L(z_g;h), M(1+z_g)) ]

— exactly the battery's registered instrument (P2: erf-sum at the Eddington mean, no
R_eff inside, σ_g = BH_MASS_ERROR linear). Σᶲ is untouched (no per-galaxy mass evaluation
in it), so the change is r_Malm → r_Malm·J_α(h) with J_α = Σ⁴ᴰ_kernel/Σ⁴ᴰ_point — the
form fixb_pathA D3 predicted.

**(b) Point-everywhere.** Treat catalogue masses as exact: then A2 requires the NUMERATOR
to drop σ_gal, D_g's integral, and the Eddington shift. This is refuted by measurement
reality (fractional σ_M ≈ 0.9 median is not ignorable) and would reinstate the raw
inverse-mass Eddington biases the G2d treatment was derived to remove (G2d doc; ledger
history). Point-everywhere is a coherent estimator for a NOISELESS catalogue — not this
one.

**Verdict: kernel-everywhere is the derived correct form.** The current mixed form is the
defect; the battery's `--sigma4d_mass_kernel kernel` instrument IS the corrected Σ⁴ᴰ.

## 3. Why the effect is small (mechanism of the measured −0.002/−0.006)

S_4D is piecewise-linear-in-M between p_det grid nodes; a Gaussian expectation of a linear
function equals its point value — J_α ≠ 1 only through S_4D's curvature and the horizon
edge across the kernel width. With σ_g/M ≈ 0.9 the kernel spans the erf roll-off for
near-horizon galaxies, raising their effective detectability weight (heavy-mass tail is
louder), hence r_Malm up 4–8% and a modest low-h pull. The sign and smallness are thus
structural, not tuned — consistent with the fixb_pathA "adverse-direction, second-tier"
expectation.

## 4. Next-order term, disclosed and bounded out of scope

The rate weight w_g = R_eff(M_g)/(1+z_g) is also point-evaluated. It appears in BOTH Σ⁴ᴰ
and Σᶲ (and the numerator's w_g), so its measurement-error correction largely cancels in
r_Malm and in the catalogue-leg ratio; the residual is a curvature term of R_eff over σ_g,
common-mode across legs. It is NOT part of D3's documented inconsistency (which is the
S_4D evaluation only) and is left as a documented next-order entry — measuring it would be
a separate registered instrument if ever warranted.

## 5. /physics-change gate presentation (author approval before the default flips)

- **Old:** `--sigma4d_mass_kernel` default `point` — Σ⁴ᴰ point form (:2707-2731).
- **New:** default `kernel` — the §2(a) expectation (code already shipped and tested as the
  battery instrument; the change is the DEFAULT, plus keeping `point` available for
  historical reproduction, mirroring the B_scale pattern).
- **Reference:** Mandel, Farr & Gair (2019) arXiv:1809.02063 A2; fixb_pathA §D3 (σ_lnM/σ_g
  as a declared physics input); this memo §2.
- **Dimensional:** probability in [0,1] either way; J_α dimensionless.
- **Limiting cases (pinned by the existing instrument tests):** σ_g → 0 ⇒ kernel ≡ point;
  S linear in M over the kernel ⇒ kernel ≡ point at M_eff; default-flip regression = the
  battery jker cells (banked): adopting the default reproduces mean_h 0.6746/0.6727
  (= baseline + ΔJ) — the validation bed is already measured.
- **Expected consequence (honest):** production 2D offsets move −0.0529/−0.0512 →
  ≈ −0.0554/−0.0573 (slightly further below truth). This is a correctness adoption, not a
  bias repair; the base tilt remains Option B's target.
- **Alternative on the fork:** (c-style) keep `point` as default and document J_α as a
  quantified systematic (−0.002/−0.006) — coherent only as a temporary state, since §2
  establishes the mixed form as a genuine A2 defect; recommended only if the author prefers
  to batch the default flip with the Option B outcome.
