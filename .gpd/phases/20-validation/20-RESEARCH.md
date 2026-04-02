# Phase 20: Validation - Research

**Researched:** 2026-04-01
**Domain:** Statistical validation, hypothesis testing for binomial proportions, Monte Carlo diagnostics
**Confidence:** HIGH

## Summary

Phase 20 validates that the enhanced sampling design (Phase 19) produces unbiased P_det estimates consistent with the uniform baseline (Phase 18). This is a pure statistical validation phase -- no new physics, no new Monte Carlo sampling, no GPU compute. The inputs are the existing Phase 18 injection CSVs (uniform sampling) and the Phase 19 IS estimator + allocation design. The outputs are statistical test results, diagnostic tables, and pass/fail verdicts.

The validation has four distinct components: (1) per-bin agreement between uniform and IS-weighted P_det estimates via Wilson CI overlap, (2) monotonicity of P_det in redshift at fixed mass, (3) physical boundary condition verification, and (4) Farr (2019) N_eff criterion satisfaction. All four are straightforward to implement using existing tools (numpy, scipy, astropy, the Phase 18/19 analysis modules). The main subtlety is multiple testing correction when comparing ~150 bins across 7 h-values, and handling sparse bins where statistical tests have low power.

Additionally, VALD-02 requires comparing the grid-based P_det against a direct MC selection integral alpha(h) = (1/N) sum(I_det,i) without gridding (the LVK standard approach from Mandel, Farr & Gair 2019). This is a one-line computation per h-value and provides a global consistency check.

**Primary recommendation:** Use Wilson CI overlap as the per-bin agreement test (no additional statistical test needed -- if the 2-sigma Wilson CIs from both methods overlap, the estimates are consistent). Apply Benjamini-Hochberg FDR correction at q=0.05 across all bins. Test monotonicity via isotonic regression residuals. Verify boundary conditions by direct inspection of extreme bins.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| Phase 18 uniform baseline P_det grid | prior artifact | The reference grid against which enhanced P_det is compared | Load and compute Wilson CIs for all bins | plan / execution / verification |
| Phase 19 IS estimator (`analysis/importance_sampling.py`) | prior artifact | The enhanced estimator being validated | Apply with w=1 (uniform recovery) and with non-uniform weights | plan / execution / verification |
| Phase 19 sampling design (`analysis/sampling_design.py`) | prior artifact | Neyman allocation and VRF predictions to verify | Compare predicted VRF with observed CI improvement | plan / execution |
| Farr (2019) arXiv:1904.10879 | benchmark | N_eff > 4*N_det criterion for unbiased selection function | Verify globally and per-bin; report worst-case bin | plan / execution / verification |
| Mandel, Farr & Gair (2019) arXiv:1809.02063 | method | Direct MC selection integral alpha(h) as gridless comparison | Compute alpha(h) for each h-value from raw injection data | plan / execution |
| Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133 | method | Wilson score CI; the interval used for per-bin comparison | Use Wilson CIs from both uniform and weighted estimators | plan / execution |
| Phase 18 grid quality report | prior artifact | Baseline CI half-widths, boundary bin counts, empty/unreliable bin census | Reference for expected bin counts and quality flags | plan / verification |

**Missing or weak anchors:** No published EMRI-specific P_det validation protocol exists. The validation approach follows standard GW injection campaign methodology (Farr 2019, LVK O3 injection sets) adapted to the 2D grid structure.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Grid space | (d_L, M) not (z, M) | (z, M) | SimulationDetectionProbability |
| Distance units | Gpc | Mpc | physical_relations.py |
| Mass units | Solar masses, source-frame | kg, redshifted mass | ParameterSpace |
| Hubble parameter | Dimensionless h | H0 in km/s/Mpc | constants.py |
| SNR threshold | 15.0 | 20 | analysis/importance_sampling.py |
| Grid resolution | 15x10 (d_L x M) for validation target | 30x20 | Phase 18/19 recommendation |
| CI method | Wilson score, 2-sigma (95.45%) | Clopper-Pearson | Brown et al. (2001), Phase 18 convention |
| Boundary definition | 0.05 < P_det < 0.95 | Other thresholds | Phase 18/19 convention |
| Sufficient statistics threshold | N_total >= 10 | N_total >= 30 | Phase 18 convention |
| Multiple testing correction | Benjamini-Hochberg FDR at q=0.05 | Bonferroni | Standard recommendation for >100 tests |

**CRITICAL: All comparisons use the same conventions as Phases 18-19. The grid is in (d_L, M) space. Monotonicity tests require converting d_L back to z via the cosmology model.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| `P_hat_U(B) = N_det(B) / N_total(B)` | Uniform P_det estimator | Standard binomial | Baseline estimate per bin |
| `P_hat_W(B) = sum(w_i * I_i) / sum(w_i)` | IS-weighted P_det estimator | Tiwari (2018) Eq. 5-8 | Enhanced estimate per bin |
| Wilson CI: `(p + z^2/2n +/- z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)` | Wilson score interval | Brown et al. (2001) | Per-bin CI for both estimators |
| `N_eff(B) = (sum w)^2 / sum(w^2)` | Kish effective sample size | Kish (1965) | Replace n with N_eff in Wilson CI for weighted estimator |
| `alpha(h) = (1/N) sum_{i=1}^{N} I(SNR_i >= threshold)` | Direct MC selection integral | Mandel, Farr & Gair (2019) | Global P_det without gridding |
| BH critical value: `p_(i) <= (i/m) * q` | Benjamini-Hochberg procedure | Benjamini & Hochberg (1995) | Multiple testing correction |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Wilson CI overlap test | Tests whether two binomial proportions are consistent | Per-bin uniform vs weighted comparison | Agresti & Coull (1998) |
| Benjamini-Hochberg FDR | Controls false discovery rate across ~1000 comparisons | All per-bin comparisons pooled | Benjamini & Hochberg (1995) |
| Isotonic regression (PAVA) | Fits monotone non-increasing function to noisy P_det vs z | Monotonicity verification per M-column | Barlow et al. (1972); sklearn.isotonic |
| Direct MC integral | Computes global alpha(h) without gridding | VALD-02 grid vs gridless comparison | Mandel, Farr & Gair (2019) |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Wilson CI with N_eff | 1/N_eff | N_eff >= 5 per bin | Conservative for small N_eff | Clopper-Pearson (exact but wider) |
| Normal approx in BH procedure | 1/sqrt(n) | n >= 10 per bin | Wilson interval p-values are well-calibrated for n >= 10 | Fisher exact test for n < 10 |
| IS bias O(1/N) | 1/N_eff per bin | N_eff > 50 | Negligible for this validation | None needed (bias is <0.02 for N_eff > 50) |

## Standard Approaches

### Approach 1: Wilson CI Overlap + BH FDR (RECOMMENDED)

**What:** For each bin in each h-value, compute the Wilson CI for both the uniform estimator (using N_total) and the IS-weighted estimator (using N_eff). Two estimates are "consistent" if their 2-sigma CIs overlap. Pool all per-bin comparisons across all h-values and apply Benjamini-Hochberg FDR correction at q=0.05.

**Why standard:** Wilson CI overlap is the simplest valid method for comparing two binomial proportions that correctly handles small samples and extreme proportions. It avoids the need for a formal two-sample test (which would require choosing between z-test, Fisher exact, Barnard exact, etc. depending on sample size). BH FDR is the standard correction for large numbers of comparisons (Bonferroni is far too conservative for ~1000 tests).

**Track record:** Wilson intervals recommended by Brown, Cai, DasGupta (2001) as the single best method for binomial CIs. BH FDR correction is the standard in genomics, neuroimaging, and any field with high-dimensional multiple testing.

**Key steps:**

1. Load Phase 18 injection data and build 15x10 grids per h-value using `build_grid_with_ci()`
2. For each bin, compute uniform P_det and Wilson CI (from `GridResult`)
3. Apply IS weights (w=1 for uniform data) and compute weighted P_det and Wilson CI with N_eff
4. For each bin with N_total >= 10 in both methods: check if CIs overlap
5. Pool all non-overlapping bins; apply BH correction; report discoveries at q=0.05
6. Success criterion: zero discoveries (all bins consistent) or discoveries only in bins with N_total < 20

**Known difficulties at each step:**

- Step 3: When all weights are 1, the IS-weighted Wilson CI must exactly equal the uniform Wilson CI. This was verified in Phase 19 (machine precision). The interesting case is when we simulate non-uniform weights.
- Step 4: Many bins will have N_total < 10 in one or both methods. These are excluded from the formal test but flagged.
- Step 5: With ~100 eligible bins (N_total >= 10 in 15x10 grid) times 7 h-values = ~700 tests, BH at q=0.05 allows ~35 false discoveries. This is very permissive.

### Approach 2: Two-Proportion Z-Test per Bin (FALLBACK)

**What:** For bins with sufficient statistics (N_total >= 30 in both methods), use `statsmodels.stats.proportion.proportions_ztest` to formally test H0: P_uniform = P_weighted.

**When to switch:** If CI overlap is deemed insufficiently rigorous (e.g., reviewer requests a formal p-value for each bin).

**Tradeoffs:** More formal but requires larger N per bin; not valid for N < 30 (use Fisher exact test for N < 30); adds statsmodels dependency (likely already available but not currently used in the analysis scripts).

### Anti-Patterns to Avoid

- **Chi-squared test on sparse bins:** Chi-squared requires expected cell counts >= 5 in all cells. Many P_det bins have N_det = 0 or 1. Do not use chi-squared for per-bin testing.
- **Bonferroni correction:** With ~700-1000 comparisons, Bonferroni (alpha/m = 0.05/1000 = 5e-5) is absurdly conservative. Any real but small disagreement will be undetectable.
- **Comparing point estimates without CIs:** "P_det differs by 0.03" is meaningless without knowing the CI width. Always compare via CIs.
- **Testing bins with N_total < 10:** Statistical tests have no power here. Flag and exclude, do not test.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Wilson score CI formula | `(p + z^2/2n +/- z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)` | Brown et al. (2001) | Already implemented in `is_weighted_wilson_ci()` and astropy |
| Kish N_eff | `(sum w)^2 / sum(w^2)` | Kish (1965) | Already implemented in `kish_n_eff()` |
| Farr criterion | `N_eff > 4 * N_det` | Farr (2019) | Already implemented in `farr_criterion_check()` |
| Phase 18 grid statistics | 15x10 grid: 3-5 boundary bins per h, median CI hw 0.02-0.06 | Phase 18 grid quality report | Baseline for comparison |
| Phase 19 VRF predictions | VRF 7-42x across boundary bins | Phase 19 sampling design report | Verify VRF is achieved |
| IS uniform recovery | max|diff| < 1e-14 for all h-values | Phase 19 verification | Already verified; re-run as sanity check |

**Key insight:** Most of the statistical machinery (Wilson CI, N_eff, Farr criterion) is already implemented in `analysis/importance_sampling.py` and `analysis/grid_quality.py`. Phase 20 assembles these into a validation report, adds monotonicity + boundary checks, and adds the VALD-02 gridless comparison. Very little new code is needed.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| `GridResult` from `build_grid_with_ci()` | Per-bin P_det, CIs, N_total, N_detected, reliability flags | `analysis/grid_quality.py` | Requires injection CSVs |
| `quality_flags()` from `SimulationDetectionProbability` | Per-bin N_total, N_detected, N_eff, CI bounds | `simulation_detection_probability.py` | Requires injection CSVs |
| P_det monotonicity in d_L | P_det should be non-increasing in d_L at fixed M | Physical: farther = fainter | Always valid for marginal P_det |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| arXiv:1904.10879 | Farr | 2019 | N_eff criterion for injection campaign adequacy | Eq. (6): N_eff > 4*N_det; per-bin version |
| arXiv:1809.02063 | Mandel, Farr, Gair | 2019 | Selection integral framework; direct MC sum | Eq. (8): alpha = (1/N_draw) sum I_det |
| Stat. Sci. 16:101-133 | Brown, Cai, DasGupta | 2001 | Wilson score CI coverage properties | Table 1: Wilson has best coverage among simple intervals |
| JRSS-B 57:289-300 | Benjamini, Hochberg | 1995 | FDR procedure for multiple testing | Algorithm: rank p-values, compare to (i/m)*q |
| arXiv:1712.00482 | Tiwari | 2018 | IS estimator for GW sensitive volume | Eq. 5-8: self-normalized estimator formula |
| arXiv:2404.16930 | Essick, Farr | 2024 | Quick recipes for GW selection effects | Modern summary of injection-based methods |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| numpy | `np.histogram2d`, `np.argsort` | Grid construction, BH procedure | Standard |
| scipy.stats | `norm.ppf` | Wilson CI z-value | Standard |
| astropy.stats | `binom_conf_interval` | Wilson CI (already used in Phase 18) | Handles edge cases (k=0, k=n) |
| sklearn.isotonic | `IsotonicRegression` | Monotonicity test via PAVA | Standard implementation of isotonic regression |
| analysis/importance_sampling.py | `weighted_histogram_estimator`, `kish_n_eff`, `is_weighted_wilson_ci`, `farr_criterion_check` | All IS-weighted statistics | Phase 19 implementation |
| analysis/grid_quality.py | `build_grid_with_ci`, `load_injection_data`, `GridResult` | Grid construction with CIs | Phase 18 implementation |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| matplotlib | Side-by-side P_det grid visualizations, CI comparison plots | Validation report figures |
| pandas | Tabular output of per-bin test results | Report generation |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| Wilson CI overlap | Fisher exact test per bin | More formal but slower; needs 2x2 contingency table per bin; Wilson overlap is equivalent for practical purposes |
| BH FDR | Bonferroni | Much too conservative for ~1000 tests |
| sklearn IsotonicRegression | Manual PAVA implementation | No reason to reimplement; sklearn is standard |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Load all injection CSVs | ~5s | I/O | Already cached in Phase 18/19 runs |
| Build 7 x 15x10 grids | ~2s | numpy histogram2d | Trivial |
| 1050 Wilson CI overlap tests | <1s | None | Vectorizable |
| BH FDR correction | <1ms | None | Sort + compare |
| 7 isotonic regressions per M-column (~70 total) | <1s | None | sklearn PAVA is O(n) |
| 7 direct MC alpha(h) computations | <1s | None | One sum per h-value |

**Total: ~10s. No GPU needed. No new dependencies needed (sklearn already in dev extras).**

**Installation / Setup:**
```bash
# sklearn should already be available; if not:
uv add --optional dev scikit-learn
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| IS uniform recovery (re-run) | IS estimator with w=1 matches standard | `run_uniform_weight_verification()` | max diff < 1e-14 (already passed in Phase 19) |
| Per-bin Wilson CI overlap | Uniform and weighted P_det are consistent | Compare CI bounds bin-by-bin | All bins overlap (within FDR tolerance) |
| Global alpha(h) agreement | Grid-based P_det integrates to same global P_det as direct MC | Sum grid P_det weighted by bin occupancy vs direct sum | Agreement within 1/sqrt(N) ~ 0.01 |
| Farr N_eff criterion | Sufficient effective injections for unbiased inference | `farr_criterion_check()` per h-value | N_eff > 4*N_det globally and per-bin where N_det > 0 |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| P_det at low z, high M | z -> 0, M -> 10^6 Msun | P_det -> 1 | Physical: nearby heavy MBH always detectable |
| P_det at high z, low M | z -> z_max, M -> M_min | P_det -> 0 | Physical: distant light MBH undetectable |
| P_det monotonicity in z | Fixed M, increasing z | Non-increasing | Physical: larger z -> larger d_L -> lower SNR |
| IS with w=1 | All weights equal | Exact recovery of standard estimator | Mathematical identity |
| N_eff with w=1 | All weights equal | N_eff = N exactly | Kish formula: (Nc)^2/(Nc^2) = N |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| Uniform recovery | max abs diff between standard and IS(w=1) | 1e-14 | 0.0 |
| Grid vs direct MC alpha(h) | |alpha_grid(h) - alpha_MC(h)| | 1/sqrt(N) ~ 0.007 for N~23000 | Agreement |
| Wilson CI coverage | Simulated binomial data with known P | 95.45% coverage | Wilson CI has correct coverage for n >= 10 |

### Red Flags During Computation

- If Wilson CIs from uniform and IS-weighted estimators do not overlap for bins with N_total > 100 in both methods: indicates a bug in the IS estimator, not statistical fluctuation.
- If monotonicity violations appear in bins with N_total > 50 and P_det in (0.1, 0.9): likely a waveform failure artifact or binning error, not noise.
- If global alpha(h) from grid differs from direct MC by more than 5%: indicates systematic grid bias (empty bins, edge effects).
- If N_eff < 4*N_det globally: the injection campaign is inadequate for unbiased inference (unlikely for uniform sampling but must verify).
- If P_det > 0.05 at the highest-z bin for any h-value: the z-range may be insufficient.

## Common Pitfalls

### Pitfall 1: Confusing d_L Monotonicity with z Monotonicity

**What goes wrong:** P_det is built on a (d_L, M) grid, but monotonicity must be tested in z (not d_L). While d_L is monotonically increasing in z for standard cosmology, the bin edges in d_L space correspond to non-uniform z intervals, and testing "P_det non-increasing along d_L bins" is equivalent to testing "P_det non-increasing in z" only if the d_L-to-z mapping is monotonic (which it is for LCDM with Omega_m=0.25).
**Why it happens:** The grid is in (d_L, M) space for computational convenience.
**How to avoid:** Verify that d_L(z) is monotonically increasing in the z-range used (it is for z < 5 with any reasonable cosmology). Then monotonicity in d_L implies monotonicity in z. Test directly on d_L bin indices.
**Warning signs:** Non-monotonic d_L(z) would only occur for exotic dark energy models (w < -1 with phantom crossing) which are not in scope.
**Recovery:** If concerned, convert d_L bin centers to z and test in z-space directly.

### Pitfall 2: Multiple Testing Without Correction

**What goes wrong:** Testing ~1050 bins at alpha=0.05 without correction yields ~52 false positives by chance, leading to a false conclusion that the methods disagree.
**Why it happens:** Naive per-bin testing ignores the multiplicity of comparisons.
**How to avoid:** Apply BH FDR correction at q=0.05 to all per-bin p-values or CI overlap tests.
**Warning signs:** "15 out of 700 bins show disagreement" -- this is expected by chance without correction.
**Recovery:** Re-analyze with BH correction; report the adjusted significance.

### Pitfall 3: Testing Sparse Bins and Drawing Conclusions

**What goes wrong:** Bins with N_total < 10 have Wilson CIs so wide that any P_det value is "consistent." Testing these bins is meaningless -- they will always pass, giving false confidence.
**Why it happens:** Phase 18 data has many bins with < 10 events (especially at high z or low M).
**How to avoid:** Exclude bins with N_total < 10 from formal testing. Report them separately as "insufficient statistics."
**Warning signs:** "All bins pass" when most bins have N_total < 5.
**Recovery:** Report the fraction of bins with sufficient statistics and the test results restricted to those bins.

### Pitfall 4: Ignoring Waveform Failure Bias in Monotonicity Test

**What goes wrong:** Waveform failures (30-50% of injections) may concentrate at specific (z, M) regions, creating non-physical structure in P_det that appears as monotonicity violations.
**Why it happens:** Failed waveforms are counted as non-detections (N_total incremented but not N_det), which deflates P_det in failure-prone bins.
**How to avoid:** Check if the injection CSVs include a failure/success flag. If failures concentrate in specific bins, note this as a known limitation.
**Warning signs:** Monotonicity violations at intermediate z (not at the detection boundary) in bins with high waveform failure rates.
**Recovery:** Flag violations and cross-reference with waveform failure rate per bin (if available from Phase 17 audit).

## Level of Rigor

**Required for this phase:** Statistical evidence with explicit significance levels and multiple testing correction.

**Justification:** This is a validation phase -- the outputs are pass/fail verdicts that determine whether the enhanced sampling design is trustworthy. Hand-waving "looks about right" is explicitly forbidden by the contract ("Forbidden proxies: Qualitative 'looks right' without statistical test").

**What this means concretely:**

- Every per-bin comparison must produce a quantitative CI overlap or p-value
- Multiple testing correction must be applied (BH FDR at q=0.05)
- Monotonicity violations must be flagged with bin coordinates and statistical significance
- Boundary condition checks must verify specific bins (not just "P_det is large at small z")
- The Farr criterion must be verified numerically per bin, not just globally
- VALD-02 grid vs direct MC comparison must report absolute and relative differences

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Wald CI for binomial proportions | Wilson score CI | Brown et al. (2001) | Wilson has correct coverage for small n; Wald fails near p=0 or p=1 |
| Bonferroni for multiple testing | Benjamini-Hochberg FDR | Benjamini & Hochberg (1995) | FDR is much less conservative; standard for high-dimensional testing |
| Global N_eff check only | Per-bin N_eff diagnostics | Farr (2019), Essick & Farr (2024) | Per-bin checks catch localized under-sampling that global checks miss |

**Superseded approaches to avoid:**

- **Wald CI:** Fails catastrophically for p near 0 or 1 and for small n. Wilson score is strictly superior.
- **Chi-squared goodness-of-fit on sparse grids:** Requires expected counts >= 5 in all cells; inappropriate for P_det grids with many zero-count bins.

## Open Questions

1. **Waveform failure tracking in injection CSVs**
   - What we know: Phase 17 audited injection physics; failures occur at 30-50% rate
   - What's unclear: Whether the injection CSVs contain a failure/success flag that would allow per-bin failure rate analysis
   - Impact on this phase: Monotonicity violations may be caused by failure artifacts rather than real P_det structure
   - Recommendation: Check CSV columns for failure indicators; if absent, note as a limitation and recommend adding in future campaigns

2. **Non-uniform weight validation data**
   - What we know: Phase 19 designed the IS estimator and verified w=1 recovery
   - What's unclear: No actual non-uniform weight injection data exists yet (the enhanced campaign has not been run)
   - Impact on this phase: The IS-weighted P_det comparison can only be done with simulated/synthetic weights or with w=1 (which is trivially consistent). VALD-01's "enhanced P_det" may refer to the reweighted estimator applied to uniform data (which should give identical results), or it may require actual enhanced sampling data that doesn't exist yet.
   - Recommendation: Implement the validation framework now with w=1 data. Design it so that when enhanced data becomes available, the same scripts produce the validation report. For now, the "enhanced vs uniform" comparison is a framework verification, not a new-data comparison.

3. **Grid vs direct MC comparison scope**
   - What we know: VALD-02 asks for comparison of grid-based P_det vs direct MC alpha(h)
   - What's unclear: The direct MC integral alpha(h) is a single number per h-value, while the grid is a 2D function. The comparison must be at the level of the integrated selection effect, not per-bin.
   - Impact on this phase: The comparison is: alpha_grid(h) = sum_bins(P_det(B) * f_pop(B)) vs alpha_MC(h) = (1/N) sum_i I(det). These should agree within MC uncertainty.
   - Recommendation: Compute both and compare. Agreement validates that gridding introduces no systematic bias.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Wilson CI overlap | CIs too wide to detect disagreement | Two-proportion z-test on bins with N > 30 | Minimal; add statsmodels call |
| BH FDR | Too many discoveries (real disagreement) | Investigate per-bin: are failures in boundary bins? | Diagnostic effort, not method change |
| Isotonic regression | Cannot install sklearn | Manual PAVA in 20 lines of numpy | 30 minutes implementation |
| Grid vs MC comparison | Grid and MC disagree by > 5% | Investigate edge effects, empty bins, d_L range mismatch | Debug, not method change |

**Decision criteria:** If more than 5% of eligible bins (N >= 10) show disagreement after BH correction, this indicates a real problem (likely a bug in the IS estimator or grid construction). Investigate before concluding the methods are inconsistent.

## Sources

### Primary (HIGH confidence)

- [Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133](https://projecteuclid.org/euclid.ss/1009213286) - Wilson CI properties and coverage
- [Benjamini & Hochberg (1995) JRSS-B 57:289-300](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) - FDR procedure
- [Farr (2019) arXiv:1904.10879](https://arxiv.org/abs/1904.10879) - N_eff > 4*N_det criterion
- [Mandel, Farr & Gair (2019) arXiv:1809.02063](https://arxiv.org/abs/1809.02063) - Selection effects framework, direct MC integral
- [Tiwari (2018) arXiv:1712.00482](https://arxiv.org/abs/1712.00482) - Self-normalized IS estimator for GW

### Secondary (MEDIUM confidence)

- [Essick & Farr (2024) arXiv:2404.16930](https://arxiv.org/abs/2404.16930) - Modern summary of GW selection effect recipes
- [scikit-learn IsotonicRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) - PAVA algorithm implementation
- [scipy.stats.barnard_exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html) - Alternative exact test for 2x2 tables

### Tertiary (LOW confidence)

- Phase 18 grid quality report (project-internal) - Baseline statistics
- Phase 19 sampling design report (project-internal) - VRF predictions

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH - All methods are textbook statistics (Wilson CI, BH FDR, isotonic regression, MC integration)
- Standard approaches: HIGH - Wilson CI overlap with BH correction is the standard approach for comparing binomial proportions across many bins
- Computational tools: HIGH - All tools already available in the project (numpy, scipy, astropy, analysis modules from Phases 18-19)
- Validation strategies: HIGH - Multiple independent checks (per-bin CI, monotonicity, boundary, Farr criterion, grid vs MC)

**Research date:** 2026-04-01
**Valid until:** Indefinite (statistical methods are stable; tool APIs may change with major version updates)

## Caveats and Self-Critique

1. **Assumption: Enhanced data does not yet exist.** Open Question 2 flags that the "enhanced P_det" comparison may not be fully testable until an enhanced injection campaign is run. The validation framework is real; the data to fully exercise it may not be. This does not block the phase -- the framework should be built and tested with synthetic weights and w=1 recovery.

2. **Wilson CI overlap is conservative.** Two CIs that barely overlap correspond to a ~2.5-sigma difference (the sum of two 2-sigma intervals). A formal two-sample test would be more powerful. This is acceptable because we are testing for gross disagreement, not subtle differences.

3. **Monotonicity test has low power in sparse regions.** With only 3-5 boundary bins per h-value (Phase 18 baseline), the isotonic regression test has very few points to work with. A single noisy bin can trigger a violation. The test should be interpreted as a sanity check, not a rigorous hypothesis test.

4. **The BH procedure assumes independent or positively correlated tests.** Adjacent bins in the P_det grid share the same underlying physics (nearby d_L and M values have correlated P_det). Positive correlation makes BH conservative (controls FDR at a level <= q), so the procedure remains valid but may be slightly conservative.

5. **No simpler method was dismissed.** The recommended approach (Wilson CI overlap + BH + isotonic regression + direct MC) is already the simplest valid approach. Each component addresses one success criterion directly.
