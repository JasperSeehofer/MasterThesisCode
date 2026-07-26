# Host-z PV/photo-z kernel for real-data mode (issue #40b) — DERIVATION SKELETON

**Status: DRAFT SKELETON — physics content to be derived and ratified by the
author (physics-change protocol). Nothing in this document is implemented;
`--host_z_kernel` currently exposes only the existing `point` and
`volume_deconv` kernels (issue #40a decomposition flag).**

**Scope.** Derive the in-catalogue numerator host-redshift kernel for
*real-data* mode — where the observed catalogue redshift carries
peculiar-velocity (PV) and photometric-measurement errors that the mock's
point/point pairing deletes — and specify how it enters (i) the per-galaxy
numerator $N_g$, (ii) the per-galaxy machinery $Z_g$/$D_g$, and (iii) the
selection normalization, without double-counting. Redteam findings F2/F3
(math) and P-2 (physics) in `results/redteam_20260726/` are the driving
record; rescope R3 makes all current precision claims mock-internal until
this derivation lands.

---

## 0. Why this is needed (measured, not hypothetical)

From the redteam consolidated verdict and P-2 (2026-07-26):

- The δ-kernel host-z numerator inside `generator_marginal` carries **~95%**
  of the −898.8 ln cure and is the dominant driver of the measured
  σ_h ≈ 2×10⁻⁴ widths.
- Neglected host-z error terms vs the retained σ_dL/d_L = 0.54%:
  PV at 200 km/s → **1.25%** median (2.3× larger); GLADE+ parse-time z-floor
  0.0015 → **2.67%** median (4.9× larger).
- Restoring them (optimistic host-known limit) degrades the golden-set
  combined width from 1.9×10⁻⁴ to 6.3×10⁻⁴ (PV only, ×3.3) or 1.3×10⁻³
  (PV + z-floor, ×6.8).

The mock's point/point pairing is *generator-exact* (the generator draws
hosts at catalogue z verbatim — DERIVATION_GENERATOR_CONSISTENT_NORM.md
§4.3), so the δ-kernel is correct *for the mock*. Real data breaks the
premise; the kernel must broaden.

## 1. Current state of the code (what already exists)

Important: the codebase **already contains** a broadened host-z kernel — the
`volume_deconv` quadrature kernel (G2b derivation) with the residual-PV term
folded in (issue #16, `bayesian_statistics.py`: `sigma_z_pv = (1+z_g)·σ_v/c`,
added in quadrature to the catalogue σ_z into `host_z_error_eff`;
Davis et al. 2011, arXiv:1012.2912 for the (1+z) factor). The δ-kernel path
*bypasses* it in the numerator. The derivation must therefore answer:

> **Q1 — Is the existing `volume_deconv` kernel (Gaussian in z, PV in
> quadrature, volume-deconvolved, per-galaxy renormalized by $Z_g$) already
> the correct real-data kernel, or does real-data mode need a new form?**

Candidate answers to develop:
- (a) YES as-is — then real-data mode is `--host_z_kernel volume_deconv` on
  top of a normalization mode whose selection leg is re-derived for scattered
  generation (see §4), and issue #40b reduces to validation.
- (b) NO — fat-tailed photo-z outliers matter (Turski et al. 2023 find
  Gaussian vs modified-Lorentzian gives measurably different H₀ posteriors
  for well-localized events); a new kernel form is needed.
- (c) NO — the GLADE+ per-galaxy σ_z model (flag-dependent: spec σ_z≈0.0017
  vs photo σ_z≈0.035, handler.py parse) needs a per-flag kernel rather than
  one Gaussian.

## 2. Verified literature anchors (2026-07-26 scan; full note in PR)

| Source | Kernel | Where it enters | Verified |
|---|---|---|---|
| Laghi et al. 2021 (arXiv:2102.01708, EMRI/LISA) | Gaussian in z, fixed σ_z = 0.0015 (⇔ 500 km/s rms PV), their Eq. (2.8)/(2.10) | Numerator (galaxy-weighting) only; selection term separate, ~1% correction | full text |
| Turski et al. 2023 (arXiv:2302.12037) | Gaussian AND modified Lorentzian $f(\Delta z)=A(1+\Delta z^2/2as^2)^{-a}$, their Eq. (1); σ = 0.052·z+0.008 (2MPZ), 0.085·z+0.019 (WISC) — linear in z, NOT (1+z) | Numerator (their Eq. 4); out-of-catalogue term keeps smooth dV_c/dz prior | full text |
| Gray et al. 2020 (arXiv:1908.06050) | Gaussian in z (baseline, per Turski's characterization) | in-catalogue term | functional form only — **eq. number unverified; do not cite an eq. without opening the PDF** |
| GLADE+ (Dálya et al. 2022, arXiv:2110.06184) | catalogue reports per-galaxy σ_z; PV correction via BORG for z<0.05 | input | partial — **check whether σ_z already includes PV before adding ours (double-count risk with issue #16's σ_v=200 km/s residual)** |

Anti-anchor: a secondary review attributed σ_z = 0.013(1+z)³ to Gray et al. —
**rejected**: it exactly matches this repo's own known-unreferenced dead code
(`datamodels/galaxy.py:66`, GitHub #7). Do not cite.
Chen, Fishbach & Holz 2018 and Howlett & Davis 2020 turned out weak matches
for a kernel-form citation — verify independently before citing.

Consistent across all verified sources: (i) the kernel is **not
h-dependent**; (ii) it belongs in the **per-galaxy numerator sum**; (iii) the
**selection/normalization keeps the smooth dV_c/dz prior**, not the
discrete-galaxy kernel.

## 3. The derivation to be done (AUTHOR)

**3.1 Kernel form.** Write $p(z_g \mid z)$ for real data as [AUTHOR: choose
and justify — Gaussian $\mathcal N(z_g; z, \sigma_{\rm eff}(z))$ with
$\sigma_{\rm eff}^2 = \sigma_{z,\rm cat}^2 + \sigma_{z,\rm pv}^2$, vs
per-flag mixture, vs fat-tailed form]. State the PV term
$\sigma_{z,\rm pv} = (1+z)\,\sigma_v/c$ and the chosen $\sigma_v$ with the
double-counting audit against GLADE+'s BORG PV correction (§2) and against
issue #16's already-implemented residual term.

**3.2 Numerator.** $N_g = \int dz\, p(x_{\rm GW} \mid z, \Omega_g[, M], h)\,
p_g(z)$ with $p_g(z) \propto p(z_g \mid z)\, w_{\rm pop}(z)$, $Z_g$
renormalized — i.e. the G2b structure with the ratified kernel. [AUTHOR:
confirm the volume-deconvolution weight $w_{\rm pop} = dV_c/dz\,(1+z)^{-1}$
survives unchanged for the real-data kernel.]

**3.3 Selection/normalization leg.** The mock's generator-consistent
normalization (n̂_w, D_gen, point-evaluated Σ_glob) is derived from the
*generator's* recipe. Real data has no generator. [AUTHOR: derive the
real-data selection normalization — presumably the absolute_marginal /
Mandel-Farr-Gair α(h) form with the smooth dV_c/dz prior; decide whether
Σ_glob must be kernel-smeared (the existing `--smear_global_selection`
machinery) for num/denom symmetry.]

**3.4 Dimensional analysis.** [AUTHOR: units table — kernel is a density in
z (dimensionless argument), N_g carries the GW-likelihood units unchanged,
$Z_g$ dimensionless normalization.]

**3.5 Limiting cases (minimum set).**
- σ_eff → 0 recovers the δ-kernel/point numerator exactly (already pinned by
  `test_volume_deconv_numerator_collapses_to_point_as_sigma_to_zero`).
- σ_v → 0, σ_z,cat → 0: same limit through the PV term alone.
- z ≫ σ_eff (deep venue): kernel → Gaussian, volume weight locally flat →
  recovers the bare-Gaussian kernel (G2b §; existing behavior).
- h-independence: kernel must be exactly h-invariant (D1 §2 fact 2 analog).

**3.6 Predicted impact (pre-register before any A/B).** From P-2: expect
σ_h inflation ×3.3–×6.8 on the golden set; MAP shift prediction [AUTHOR:
derive sign/size or pre-register "no detectable bias" with a threshold].

## 4. Validation plan (after ratification)

1. Implement as a new `host_z_kernel` choice (extend
   `HOST_Z_KERNEL_CHOICES`; the #40a flag already decouples the numerator
   kernel from the normalization leg — see `resolve_host_z_kernel`).
2. Regression: default path byte-identical (pipeline-parity golden).
3. Limiting-case tests per §3.5 (imitate the existing collapse test).
4. 3-way A/B per math-review F2 recommendation: {point, volume_deconv,
   ratified real-data kernel} on a fixed venue, per-leg attribution of the
   ln-gap.
5. P–P/coverage calibration with scattered-z synthetic universes
   (pp_coverage harness `kernel` switch is precedent) — the decisive test
   that widths are calibrated, not just inflated.
6. Blind alternative-truth mock (#39) remains the anti-tuning gate.

## References

- Redteam: `results/redteam_20260726/CONSOLIDATED_VERDICT.md` (R3, F2/F3),
  `MATH_REVIEW.md` F2/F3, `PHYSICS_METHODOLOGY_REVIEW.md` P-2.
- G2b: `docs/derivations/G2b_host_z_volume_prior.md` (volume_deconv kernel).
- Generator norm: `results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md`.
- Laghi et al. 2021, arXiv:2102.01708; Turski et al. 2023, arXiv:2302.12037;
  Davis et al. 2011, arXiv:1012.2912; Dálya et al. 2022, arXiv:2110.06184;
  Gray et al. 2020, arXiv:1908.06050 (eq. numbers unverified);
  Mandel, Farr & Gair 2019, arXiv:1809.02063.
