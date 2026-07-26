# Host-z PV/photo-z kernel for real-data mode (issue #40b) — DERIVATION SKELETON

**Status: RATIFIED (2026-07-26) — all five [RATIFY] gates approved by the
author with the stated recommendations ("all approved with your
recommendations"): (1) corrected-host residual σ_v = 150 km/s; (2)
uncorrected-host σ_v = 500 km/s replacing the 0.0015 fill; (3) parse the
PV-correction flag, WITH the empirical precondition that the null↔flag
coincidence be verified on the raw GLADE+.txt BEFORE the implementation
merges (raw catalogue re-download in progress 2026-07-26 — the original was
deleted from every machine); (4) w_pop unchanged; (5) real-data mode =
absolute_marginal normalization + broadened volume_deconv numerator,
explicitly 1D-complete / 2D-OPEN (three-way A/B: the 2D channel needs its
own mass-marginal derivation).**

**Implementation decision (post-ratification):** per-class PV terms are
applied at PARSE time — the (1+z)·σ_v/c term is computed per row (z known at
parse) and folded into the stored z_error with the class resolved by the
PV-correction flag; the runtime `SIGMA_V_PEC_KM_S` quadrature in
`bayesian_statistics.py` is removed (counted-once invariant, single
application site). The reduced-catalogue schema stays 8 columns (no flag
column stored); the reduced CSV must be REGENERATED from GLADE+.txt and
re-staged to the cluster. The width changes touch `host_z_error_eff`
(denominator windows, Z_g) on the production mock path — golden
regeneration is a reviewed value-update step of this change.

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

- The δ-kernel host-z numerator inside `generator_marginal` carries
  **~85–87% of the ln movement** (MEASURED 2026-07-26 by the three-way
  per-leg A/B enabled by the #40a flag — refining the redteam's ~95%
  estimate; `results/lcat_h_dependence_20260725/threeway_ab/THREEWAY_AB_READOUT.md`)
  and is the dominant driver of the measured σ_h ≈ 2×10⁻⁴ widths. Refinement:
  the normalization legs ALONE de-rail the 1D channel (truth MAP); the
  δ-kernel buys peak depth — EXCEPT the 2D channel, where only the δ-kernel
  brings the MAP to truth (kernel numerator leaves +29 ln at 0.80). The 2D
  mass-marginal treatment is therefore a first-class dependency of this
  derivation, not an afterthought.
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

## 3. The derivation (RATIFIED 2026-07-26 — all five gates approved with
## the stated recommendations; audit evidence in
## `results/lcat_h_dependence_20260725/threeway_ab/GLADE_PV_AUDIT.md`)

Author delegation 2026-07-26 ("go with what you recommend and what is
scientifically correct"), followed by explicit per-gate ratification of the
drafted recommendations ("all approved with your recommendations").

**3.1 Kernel form — Gaussian with per-host, flag-resolved σ_eff (answer to
Q1: (a) structurally, with the (c) per-flag width audit REQUIRED, and (b) as
a robustness ablation only).**

$$p(z_g \mid z) = \mathcal N\!\left(z_g;\, z,\, \sigma_{\rm eff}\right),
\qquad \sigma_{\rm eff}^2 = \sigma_{z,\rm meas}^2 + \sigma_{z,\rm pv}^2 ,$$

with the PV term counted **exactly once** per host, resolved by the GLADE+
PV-correction flag (paper col 30, 1-based; currently NOT parsed — must be
added):

- **BORG-corrected hosts** (z ≤ 0.05 ∩ 2M++ ∩ B-band — the EMRI venue):
  GLADE+'s reported PV error is already the total
  $\sigma_{\rm tot}^2 = \sigma_{\rm borg}^2 + \sigma_{\rm vir}^2$ (Dálya et
  al. 2022 §2.2, Eq. 1 — the non-linear virial residual is INCLUDED). The
  parse-time quadrature (handler.py) correctly folds it into the stored
  z_error; the correct additional likelihood residual is therefore
  **σ_v = 0** (trusting the catalogue) or a conservative
  **σ_v = 150 km/s** (Carrick et al. 2015 §4.2.1, verified) — NOT the
  current 200 km/s, which double-counts σ_vir.
- **Uncorrected hosts**: apply ONE full-dispersion term
  $\sigma_{z,\rm pv} = (1+z)\,\sigma_v/c$ with **σ_v ≈ 500 km/s** (Laghi et
  al. 2021, verified; Del Pozzo et al. 2018 lineage), REPLACING the uncited
  parse-time 0.0015 fill (which is a full-PV stand-in in disguise,
  ≈430–450 km/s).
- **Photo-z hosts** (flag 1, σ_z ≈ 0.035): unchanged — the PV term is a
  2% width effect at all z (both scale as (1+z)); immaterial.
- **Fat tails** (Turski et al. 2023 modified Lorentzian): robustness
  ablation in validation §4, not the baseline — GLADE+'s BORG-corrected
  spec-z errors are Gaussian-characterized in the source paper, and the
  fat-tail evidence in Turski concerns photo-z catalogues.

**Defect this fixes (measured, audit §Layer 3):** the current stack applies
the parse-time floor ⊕ 200 km/s on top of catalogue σ_tot, inflating the
spec-z (golden-event) kernel width by ~+40% over the defensible value. The
current SIGMA_V_PEC_KM_S = 200 km/s rationale ("residual nonlinear
dispersion on top") is invalid for BORG-corrected hosts — σ_vir already
covers it.

**[RATIFY-1 — RATIFIED 2026-07-26: 150 km/s]** corrected-host residual: 0 vs 150 km/s (recommendation:
**150 km/s**, conservative, cited, and robust to GLADE+ underestimating
σ_vir for unresolved halo masses).
**[RATIFY-2 — RATIFIED 2026-07-26]** uncorrected-host σ_v = 500 km/s replacing the 0.0015 fill.
**[RATIFY-3 — RATIFIED 2026-07-26; null↔flag check RUN 2026-07-27 on the
re-downloaded raw catalogue (23,181,758 rows): uncorrected-with-value
violations = 0; the flag is 1/0/null with null dominating (~19M rows,
uncorrected by construction); one violation subclass found — 119,299 rows
flagged corrected but with NULL PV error (99.7% photometric, median
z = 0.055, only 374 spec-z). Resolution: "corrected" requires flag == 1 AND
a reported σ_tot; flagged-but-null rows take the conservative full
500 km/s term (numerically a ~2% width effect for their photo-z majority).]** parse the PV-correction flag; verify on the cluster copy of
GLADE+.txt that col-31 nulls coincide with flag = 0 (audit's one
locally-unverifiable assumption) BEFORE implementation.

**3.2 Numerator.** Unchanged G2b structure with the §3.1 width:
$N_g = \int dz\; p(x_{\rm GW} \mid z, \Omega_g[, M], h)\; p_g(z)$,
$p_g(z) = \mathcal N(z_g; z, \sigma_{\rm eff})\, w_{\rm pop}(z)/Z_g$,
$w_{\rm pop} = \frac{dV_c}{dz}(1+z)^{-1}$. The volume-deconvolution weight
survives unchanged: it derives from the population prior (G2b §1), which is
independent of the *measurement* kernel width; only σ_eff changes.
**[RATIFY-4 — RATIFIED 2026-07-26]**.

**3.3 Selection/normalization leg.** Real data has no generator, so the
generator-consistent leg (n̂_w = W_cat/V_f, D_gen) does not apply. The
real-data leg is the absolute-marginal/α(h) form (Mandel, Farr & Gair 2019:
one selection factor; Chen, Fishbach & Holz 2018 Eq. 15 structure) with the
smooth dV_c/dz prior in the out-of-catalogue term — consistent with every
verified literature source (§2: the discrete-galaxy kernel enters the
NUMERATOR only; selection keeps the smooth prior; do not double-apply).
Σ_glob smearing: NOT required for consistency — the selection integrand is
smooth on the σ_z scale (the Jacobian/volume factors vary slowly over
Δz ≈ 2×10⁻³), so kernel-smearing Σ_glob is an O((σ_z/z)²) correction; keep
`--smear_global_selection` available as the diagnostic it already is.
**[RATIFY-5 — RATIFIED 2026-07-26, 1D-complete/2D-open]** (real-data mode = `absolute_marginal` normalization +
`volume_deconv` numerator kernel with §3.1 widths; note the #40a flag
already makes this combination expressible).

**3.4 Dimensional analysis.** z, σ_z, σ_v/c: dimensionless. $p(z_g|z)$:
density in z, units [z]⁻¹ = dimensionless. $w_{\rm pop}$: Mpc³ per unit z;
$Z_g = \int \mathcal N\, w_{\rm pop}\, dz$: Mpc³; $p_g = \mathcal N\,
w_{\rm pop}/Z_g$: [z]⁻¹ — dimensionless density, as required. $N_g$ carries
the GW-likelihood units unchanged. σ_pv term: (1+z)·[km/s]/[km/s] —
dimensionless. Consistent.

**3.5 Limiting cases (minimum set).**
- σ_eff → 0 recovers the δ-kernel/point numerator exactly (already pinned by
  `test_volume_deconv_numerator_collapses_to_point_as_sigma_to_zero`).
- σ_v → 0, σ_z,meas → 0: same limit through the PV term alone.
- Corrected host with σ_v(residual) = 0 and catalogue σ_tot → the pure
  catalogue-width Gaussian (no repo-added broadening) — new test required.
- z ≫ σ_eff (deep venue): volume weight locally flat → bare-Gaussian kernel
  (G2b; existing behavior).
- h-independence: σ_eff contains no h anywhere — structurally exact.
- Photo-z host: flag-resolved widths change the kernel by < 2% — new
  tolerance test.

**3.6 Predicted impact (pre-registered BEFORE any A/B with the ratified
kernel).**
- Mock golden set, kernel restored (P-2, optimistic host-known limit):
  σ_h ×3.3 (PV only) to ×6.8 (PV + floor) vs the point/point width. With the
  §3.1 de-double-counted widths the inflation must land BELOW the P-2
  figures (P-2 used the current, ~40% too wide, spec-z kernel):
  **prediction: ×2.5–×5.5**, i.e. σ_h ≈ 5×10⁻⁴–1.1×10⁻³ per deep venue.
- Bias: no detectable MAP shift at the |Δh| ≤ 5×10⁻⁴ level (the kernel is
  symmetric in z_g and h-invariant; the volume deconvolution already removes
  the Jensen/Eddington asymmetry per G2b). A measured shift beyond this
  falsifies the h-invariance/symmetry argument and re-opens §3.2.
- Spec-z kernel width itself: −40% vs current volume_deconv for
  BORG-corrected hosts (direct, testable at the kernel level without a
  posterior run).

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
