# H0 Posterior Bias — Resolution Changelog

Chronological record of the investigation and resolution of systematic bias
in the H0 posterior for EMRI dark siren inference. True injection value: h = 0.73.

For detailed per-test diagnostic results, see
[`scripts/bias_investigation/FINDINGS.md`](../scripts/bias_investigation/FINDINGS.md).

---

## Phase 9 — Galactic Confusion Noise (2026-03-29)

**Commit:** `3bed9fc` — `[PHYSICS] feat(09-01): add galactic confusion noise to LISA A/E-channel PSD`

- **Issue:** LISA PSD was missing galactic foreground noise, making SNR estimates
  unrealistically high and distorting Fisher-matrix parameter uncertainties.
- **Fix:** Implemented `_confusion_noise()` in `LisaTdiConfiguration`, following
  Babak et al. (2023) arXiv:2303.15929 Eq. (17) with observation-time-dependent
  knee frequency.
- **Impact on bias:** Indirect — more realistic Fisher bounds feed into detection
  probability and per-event uncertainties.

---

## Phase 10 — Five-Point Stencil Derivatives (2026-03-29)

**Commit:** `a87eeab` — `[PHYSICS] feat(10-01): wire five-point stencil into Fisher matrix`

- **Issue:** Fisher matrix used O(epsilon) forward difference, producing less accurate
  Cramer-Rao bounds on all 14 EMRI parameters including d_L.
- **Fix:** Wired `five_point_stencil_derivative()` as default in
  `compute_fisher_information_matrix()` (`use_five_point_stencil=True`).
  O(epsilon^4) central difference with coefficients (-1, 8, -8, 1) / 12epsilon.
- **Impact on bias:** More accurate Fisher bounds improve d_L and sky-localization
  uncertainties, tightening the per-event GW likelihood.
- **Reference:** Vallisneri (2008) arXiv:gr-qc/0703086.

---

## Phase 11.1 — KDE to IS Detection Probability (2026-03-31)

**Commits:**
- `e86e826` — replace KDE `DetectionProbability` with `SimulationDetectionProbability`
- `caf8ce6` — delete old KDE class and clean up references

- **Issue:** KDE-based P_det had bandwidth sensitivity, poor tail coverage, and
  systematic over/underestimation at grid boundaries.
- **Fix:** Replaced with simulation-based importance sampling (IS) estimator using
  injection campaign data. `SimulationDetectionProbability` builds a histogram
  grid from injected events with proper IS weights, then interpolates via
  `RegularGridInterpolator`.
- **Impact on bias:** Removed KDE artifacts from P_det surface. Variance reduction
  factor 11.8-24.9x in boundary bins. Backward-compatible API.

---

## Gaussian Index Bug Investigation (2026-03-31)

- **Issue:** Investigated whether `bayesian_statistics.py` lines 631/831 used
  Gaussian index [0] (3D, without BH mass) instead of [1] (4D, with BH mass)
  in the "with BH mass" numerator.
- **Finding:** Under delta-function approximation, M_frac ~ 1 with sigma ~ 1e-7,
  so 3D vs 4D Gaussian makes no numerical difference. **Not a root cause.**
- **Status:** Eliminated as hypothesis.

---

## Phases 14-15 — Likelihood Derivation and /(1+z) Fix (2026-03-31)

**Commits:**
- `c466e86` — derive d_L-only dark siren likelihood (Phase 14)
- `1d4e9a1` — remove spurious /(1+z) Jacobian from with-BH-mass numerator (Phase 15)
- `46e2662` — validation: /(1+z) fix insufficient for bias

- **Phase 14:** First-principles derivation of dark siren likelihood from Bayes'
  theorem. Verified sky-localization weight placement, dimensional consistency,
  and mapped all 12 terms to code. Deliverable: `derivations/dark_siren_likelihood.md`.
- **Phase 15:** Code audit found spurious `/(1+z)` factor at `bayesian_statistics.py`
  line 646 (and 871). Removed per Jacobian absorption identity.
- **Impact on bias:** The /(1+z) factor was a code anomaly but **not the dominant
  bias mechanism**. Posterior remained monotonically decreasing after fix.
- **Conclusion:** Necessary correction but insufficient for bias resolution.

---

## Phases 17-20 — Injection Campaign Validation (2026-03-31 to 2026-04-01)

**Key commits:**
- `60fe875` — characterize waveform failure modes in injection campaign
- `6a1ac4d` — add quality flags to `SimulationDetectionProbability`
- `f15df43` — IS-weighted histogram estimator for P_det grid
- `74affb4` — P_det validation framework

- **Phase 17:** Injection physics audit — confirmed 14-parameter consistency,
  d_L round-trip accuracy to 2e-13 precision.
- **Phase 18:** Grid quality assessment — Wilson confidence intervals, detection
  yield analysis, quality flags.
- **Phase 19:** IS estimator construction with proper weights from 463k injections.
- **Phase 20:** Validation framework — VALD-01 PASS (916 bins), VALD-02 PASS
  (alpha_grid = alpha_MC exactly).
- **Conclusion:** P_det surface is validated as **not the bias source**.

---

## Phases 21-23 — Posterior Numerical Stability (2026-04-02)

**Commits:**
- `7df0baa` — implement posterior combination module with 4 strategies
- `db5eb2b` — implement physics-floor strategy with per-event min-nonzero floor

- **Issue:** Float64 underflow in product of 500+ per-event likelihoods. Some events
  have p(d_i|h) = 0.0 at certain h-bins (no catalog host found), causing
  zero-product collapse of the joint posterior.
- **Fix:** Log-space accumulation (`posterior_combination.py`) with four strategies:
  log-sum, per-event floor, per-event nonzero-min, global floor. Physics-motivated
  floor from faintest catalog galaxy at error volume boundary.
- **Impact on bias:** Eliminated numerical artifacts but did not resolve the
  systematic catalog-driven bias.

---

## Phases 24-25 — Completeness Correction (2026-04-04)

**Commits:**
- `2341b80` — refactor `GladeCatalogCompleteness` with f(z, h) interface
- `f60a75a` — completeness-corrected dark siren likelihood (Gray et al. 2020 Eq. 9)

- **Issue:** GLADE catalog reaches only ~21% completeness at typical EMRI distances
  (>796 Mpc, z > 0.08). The asymmetric galaxy density distribution systematically
  biases the posterior toward lower H0: at trial h < h_true, galaxies at z < z_true
  can match the observed d_L, and there are more galaxies at lower z due to
  catalog incompleteness.
- **Fix:** Implemented completeness-corrected likelihood per Gray et al. (2020)
  arXiv:1908.06050 Eq. 9:

  ```
  p_i(H0) = f(z, H0) * L_cat + (1 - f(z, H0)) * L_comp
  ```

  where L_cat is the catalog term (sum over cataloged galaxies) and L_comp is the
  completion term integrating over uncataloged hosts weighted by a comoving volume
  prior. f(z, H0) from GLADE+ B-band luminosity comparison.
- **Impact on bias:** Primary mitigation for catalog-driven bias. However, the
  completion term itself carries systematic bias (see Current Status below).

---

## P_det Grid Extrapolation Fix (2026-04-08)

**Commit:** `44d5358` — `[PHYSICS] fix P_det grid extrapolation causing 44% completeness fallback`

- **Issue:** `RegularGridInterpolator` used `fill_value=0.0` for out-of-grid queries.
  702 events received P_det = 0, causing the completeness correction to fall back
  to catalog-only likelihood (L_comp contribution = 0).
- **Fix:** Changed `fill_value` from `0.0` to `None` (nearest-neighbor extrapolation).
- **Impact on bias:** Eliminated 702 zero-likelihood completeness fallbacks. Improved
  MAP h from 0.663 to 0.680 (+0.017). Reduced bias from -9.2% to -6.9%.

---

## Current Status (2026-04-08)

### Production Results (531 detections, SNR threshold = 20)

| Pipeline | MAP h | Bias | Notes |
|----------|-------|------|-------|
| Without BH mass | 0.66 | -9.6% | Grid-limited CI |
| With BH mass | 0.68 | -6.8% | Grid-limited CI |
| Thesis baseline | 0.712 / 0.742 | -2.5% / +1.6% | Pre-completeness correction |

### Root Cause (confirmed by diagnostic tests)

The residual bias is driven by **systematic GLADE catalog incompleteness** at z > 0.08.
This is NOT a code bug but a fundamental limitation of the dark siren method with an
incomplete galaxy catalog. See `scripts/bias_investigation/FINDINGS.md` for the full
7-test diagnostic analysis confirming:

- The likelihood formula is mathematically correct (Test 3)
- The P_det grid is properly constructed (Tests 5, 5b)
- The integration bounds are adequate (Test 7)
- The bias is entirely driven by galaxy density gradient (Test 4)

### Eliminated Hypotheses

| Hypothesis | Phase | Verdict |
|------------|-------|---------|
| Fisher matrix accuracy | Phase 10 | Fixed, insufficient alone |
| PSD missing confusion noise | Phase 9 | Fixed, insufficient alone |
| KDE P_det artifacts | Phase 11.1 | Fixed, insufficient alone |
| Gaussian evaluation index bug | Investigation | Not a factor |
| Spurious /(1+z) Jacobian | Phase 15 | Fixed, insufficient alone |
| P_det grid boundary/construction | Phases 17-20 | Validated correct |
| Numerical underflow | Phases 21-23 | Fixed, not the cause |
| P_det extrapolation zeros | Phase 27+ | Fixed, partial improvement |

### Active Work (v2.1 Milestone)

The v2.1 milestone (phases 29-34) targets further bias reduction through:
- Completion term prior improvements (source population prior vs. dVc/dz)
- P_det grid resolution increase
- Fisher matrix quality improvements
- Diagnostic separation of L_cat vs L_comp contributions

### Key Artifacts

| Artifact | Location |
|----------|----------|
| Diagnostic test scripts (7 tests) | `scripts/bias_investigation/` |
| Diagnostic findings summary | `scripts/bias_investigation/FINDINGS.md` |
| Bias evolution analysis | `.gpd/quick/2-analyze-evolution-of-residual-bias-ac/` |
| Dark siren likelihood derivation | `derivations/dark_siren_likelihood.md` |
| Active debug session | `.gpd/debug/h0-posterior-bias-worsening.md` |
| Resolved debug session | `.gpd/debug/resolved/h0-posterior-residual-bias.md` |

### References

- Gray, R. et al. (2020). Cosmological inference using gravitational wave standard
  sirens: A mock data challenge. *Phys. Rev. D* **101**, 122001. arXiv:1908.06050.
- Babak, S. et al. (2023). LISA sensitivity and SNR calculations. arXiv:2303.15929.
- Vallisneri, M. (2008). Use and abuse of the Fisher information matrix. *Phys. Rev. D*
  **77**, 042001. arXiv:gr-qc/0703086.
- Chen, H.-Y. et al. (2018). A two percent Hubble constant measurement from standard
  sirens within five years. *Nature* **562**, 545-547. arXiv:1709.08079.
