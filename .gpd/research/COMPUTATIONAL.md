# Computational Methods: P_det Grid Construction and Quality

**Physics Domain:** Gravitational wave selection effects, EMRI detection probability for LISA dark siren cosmology
**Researched:** 2026-03-31

### Scope Boundary

COMPUTATIONAL.md covers computational TOOLS, libraries, and infrastructure for P_det grid construction, interpolation, and quality assessment. It does NOT cover the physics of EMRI waveforms or the Bayesian inference formalism -- those belong in METHODS.md and PRIOR-WORK.md respectively.

---

## Open Questions

| Question | Why Open | Impact on Project | Approaches Being Tried |
|----------|---------|-------------------|----------------------|
| Optimal number of injections per bin for EMRI P_det | EMRI waveforms are expensive (~seconds each on GPU); budget constrains total injections | With 10k injections over 600 bins, many bins have < 10 events -- insufficient for precise P_det | Importance sampling near detection boundary; adaptive binning to merge sparse bins |
| Sharp P_det boundary treatment | P_det transitions from ~1 to ~0 over a narrow redshift range; linear interpolation smears this | Smeared boundary biases the VT integral and hence the H0 posterior normalization | Higher-resolution grids near boundary; sigmoid fitting; monotonicity constraints |
| h-dependence interpolation accuracy | Linear interpolation between 7 h-grid points may miss nonlinear P_det(h) dependence | Biased H0 posterior if P_det(h) is poorly approximated between grid points | Finer h-grid; cubic interpolation; analytic scaling relations |

---

## Anti-Approaches

| Anti-Approach | Why Avoid | What to Do Instead |
|---------------|-----------|-------------------|
| KDE-based P_det (old `DetectionProbability` class) | Smooths over the sharp detection boundary; bandwidth selection is arbitrary; poor in sparse 2D regions | Histogram binning with binomial confidence intervals per bin (current approach, needs refinement) |
| Unbinned Monte Carlo P_det evaluation | Requires full injection set at every likelihood evaluation; too slow for MCMC | Pre-compute binned grid, interpolate at evaluation time |
| Uniform grid in both z and M | M spans orders of magnitude (1e4 to 1e7 solar masses); uniform binning wastes resolution at high M and under-resolves at low M | Log-spaced M bins (already implemented), linear z bins |
| Cubic interpolation on noisy grids | Cubic splines overshoot and produce negative P_det or P_det > 1 when the underlying grid has noise from low statistics | Use linear interpolation with clipping, or monotone piecewise cubic (PCHIP) which preserves bounds |
| Treating empty bins as "unknown" | Tempting to interpolate across empty bins, but this can overestimate P_det in truly undetectable regions | Conservative: P_det = 0 for empty bins (current approach); flag these bins for quality assessment |

---

## Logical Dependencies

```
Injection campaign CSV data -> SimulationDetectionProbability.__init__() loads and bins
  |
  v
2D histogram (total_counts, detected_counts per bin) -> P_det grid per h-value
  |
  v
RegularGridInterpolator wraps each grid -> _interpolate_at_h() queries at (z, M)
  |
  v
Linear interpolation between h-grid brackets -> detection_probability_with_bh_mass_interpolated()
  |
  v
BayesianStatistics.single_host_likelihood() uses P_det(z, M, h) in normalization integral
```

Key constraint: The P_det grid quality directly enters the likelihood normalization. Systematic bias in P_det propagates linearly into the H0 posterior.

---

## Algorithms and Tools

### 1. Binomial Confidence Intervals Per Bin

**Problem:** Each bin contains n_total injections of which k are detected. P_det = k/n is a binomial proportion estimate. We need confidence intervals to flag unreliable bins.

**Recommended method:** Wilson score interval (not Wald, not Clopper-Pearson).

**Rationale:** The Wilson interval has good coverage properties even at small n and extreme proportions (p near 0 or 1). The Wald interval gives nonsensical results when k=0 or k=n. Clopper-Pearson is overly conservative, producing unnecessarily wide intervals that would flag too many bins as unreliable. The Jeffreys interval is a reasonable alternative but Wilson is faster and well-understood.

Reference: Brown, Cai, DasGupta (2001), "Interval Estimation for a Binomial Proportion", Statistical Science 16(2):101-133.

**Implementation:** Use `astropy.stats.binom_conf_interval(k, n, confidence_level=0.95, interval='wilson')` or `astropy.stats.binned_binom_proportion` for vectorized computation. Astropy is already a project dependency.

**Wilson score formula:**
```
p_hat = k / n
kappa = 1.96  (for 95% CI)
center = (p_hat + kappa^2 / (2n)) / (1 + kappa^2 / n)
half_width = (kappa / (1 + kappa^2/n)) * sqrt(p_hat*(1-p_hat)/n + kappa^2/(4n^2))
CI = [center - half_width, center + half_width]
```

### 2. Minimum Injections Per Bin

**The core statistical question:** How many injections n per bin are needed so that the P_det estimate has relative error below some threshold?

**Derivation from binomial statistics:**
For a binomial proportion p estimated from n trials, the standard error is sqrt(p(1-p)/n). The relative error is SE/p = sqrt((1-p)/(np)).

Setting relative error < epsilon:
```
n > (1-p) / (p * epsilon^2)
```

**Concrete thresholds (95% CI, Wilson interval):**

| True P_det | n for 10% relative error | n for 20% relative error | n for 30% relative error |
|-----------|-------------------------|-------------------------|-------------------------|
| 0.5       | 100                     | 25                      | 11                      |
| 0.3       | 233                     | 58                      | 26                      |
| 0.1       | 900                     | 225                     | 100                     |
| 0.05      | 1900                    | 475                     | 211                     |
| 0.01      | 9900                    | 2475                    | 1100                    |

**Recommendation for this project:** With ~17 injections per bin on average, accept 20-30% relative error as the quality floor. Flag bins with n < 10 as unreliable. Bins with n < 5 should be marked as having no meaningful P_det estimate.

**Farr (2019) criterion:** For the global selection function (not per-bin), the effective number of injections must exceed 4 times the number of detected events: N_eff > 4 * N_det. This is a necessary condition for unbiased hierarchical population inference. With 10,000 injections per h-value and ~20 detected EMRIs expected, this global criterion is easily satisfied. The per-bin challenge is separate and more demanding.

Reference: Farr (2019), "Accuracy Requirements for Empirically-Measured Selection Functions", RNAAS 3, 66. arXiv:1904.10879.

### 3. Grid Quality Metrics

**Metric 1: Effective sample size per bin**
```python
n_eff = n_total  # For uniform injection prior this equals total injections in bin
```
Flag: n_eff < 10 (unreliable), n_eff < 5 (discard/merge).

**Metric 2: Wilson CI half-width per bin**
Compute the 95% Wilson CI for each bin. Quality flag if half-width > 0.15 (absolute).

**Metric 3: Fraction of flagged bins**
If more than 20% of bins in the transition region (0.05 < P_det < 0.95) are flagged, the grid resolution or injection count is insufficient.

**Metric 4: Bootstrap stability**
Resample the injection set with replacement B=200 times, rebuild the grid each time, compute the standard deviation of P_det per bin. This captures correlated effects that the per-bin binomial CI misses (e.g., clustering in parameter space).

**Metric 5: Integral consistency**
Compute the marginalized detection rate integral:
```
<P_det> = integral P_det(z,M) * p(z,M) dz dM
```
This should be stable to within the Poisson uncertainty of the injection campaign (~1/sqrt(N_inj) ~ 1%).

### 4. Binning Strategies

#### Current approach: Fixed uniform-z x log-M grid (30 x 20 = 600 bins)

**Strengths:** Simple, reproducible, compatible with RegularGridInterpolator.

**Weakness:** Bins near the detection boundary (z ~ z_max for a given M) may have very few events while bins at low z (where P_det ~ 1) waste injections.

#### Recommended improvement: Adaptive binning with quality floor

**Strategy:** Start with the current 30x20 grid. After building the grid, identify bins with n_total < N_min (suggest N_min = 10). Merge adjacent under-populated bins along the z-axis (since the M-dependence is weaker) until the merged bin has n_total >= N_min. Record the effective bin edges for the merged grid.

**Implementation complexity:** Moderate. The merged grid is no longer regular, so RegularGridInterpolator cannot be used directly. Options:
1. Use `scipy.interpolate.LinearNDInterpolator` on the irregular merged grid (slower but flexible).
2. Keep the regular grid but flag unreliable bins and let downstream code handle them (simpler).
3. Use a coarser regular grid (e.g., 15x10 = 150 bins) that ensures ~67 injections/bin on average.

**Recommendation:** Option 3 (coarser grid) for the next iteration. It preserves the simple RegularGridInterpolator infrastructure, quadruples the per-bin count, and is a one-line change (`_Z_BINS = 15`, `_M_BINS = 10`). Supplement with per-bin Wilson CIs as quality metadata. Consider Option 1 only if the coarser grid fails validation.

#### Alternative: Importance-sampled injection distribution

Instead of drawing injections uniformly in (z, M, angles), oversample the detection boundary region. This is already partially implemented (high-z skip in the injection campaign). The correction factor (injection prior / population prior) must be tracked per injection.

### 5. Interpolation Methods

#### Current: `RegularGridInterpolator(method='linear')`

**Properties:**
- C0 continuous (continuous but not smooth)
- Preserves bounds if input grid is in [0,1] (with clipping)
- Cannot overshoot beyond grid values
- Fast: O(1) per query after setup

**Verdict:** Adequate for the current grid resolution. The main risk is smearing the sharp P_det boundary.

#### Alternative: `method='cubic'` (tensor-product cubic spline)

**Properties:**
- C2 continuous (smooth first and second derivatives)
- CAN overshoot: may produce P_det < 0 or P_det > 1 near sharp transitions
- Requires clipping to [0,1] after interpolation (already done)
- ~3x slower per query than linear

**Verdict:** Do NOT use for P_det grids with sharp boundaries and noisy bins. The overshooting problem is well-documented. The smoothness gain is not worth the artifact risk.

#### Alternative: `method='pchip'` (Piecewise Cubic Hermite Interpolating Polynomial)

**Properties:**
- C1 continuous
- Monotonicity-preserving: will not overshoot if the data is monotone in each axis
- Available in scipy's RegularGridInterpolator (scipy >= 1.12)
- Modest overhead vs linear

**Verdict:** Promising for P_det because P_det is approximately monotonically decreasing in z for fixed M. However, the 2D tensor-product PCHIP does not guarantee global monotonicity in both axes simultaneously. Worth testing but not a clear win over linear + higher resolution.

#### Alternative: Gaussian Process (GP) regression

**Properties:**
- Provides uncertainty estimates naturally
- Can handle irregular grids
- Captures smooth structure with few data points
- Expensive: O(n^3) for n grid points, or O(n*m^2) with sparse approximations

**Verdict:** Overkill for this application. The 600-point grid is small enough that GP training is fast (~ms), but GP inference at every likelihood evaluation adds overhead. The neural network emulator approach (Callister et al. 2024, arXiv:2408.16828) is the state-of-the-art for LIGO but requires thousands of training injections and is designed for higher-dimensional parameter spaces. For our 2D grid with 7 h-slices, the histogram + linear interpolation approach is appropriate.

Reference: Callister, Essick, Holz (2024), "A neural network emulator of the Advanced LIGO and Advanced Virgo selection function", arXiv:2408.16828.

### 6. Quality Visualization Diagnostics

Implement the following diagnostic plots:

1. **P_det heatmap per h-value:** 2D colormap of P_det(z, M) with bin edges visible. Overlay contours at P_det = {0.1, 0.5, 0.9}.

2. **Injection count heatmap:** 2D colormap of n_total per bin. Flag bins below threshold with hatching or markers.

3. **Wilson CI width heatmap:** 2D colormap of the 95% CI half-width. Highlights where the grid is unreliable.

4. **P_det(z) slices at fixed M:** 1D curves showing P_det vs z for several M values, with Wilson error bands. Verifies the transition shape.

5. **P_det(h) slices at fixed (z, M):** Verifies that linear h-interpolation is reasonable. Check for non-monotonic behavior.

6. **Bootstrap spread heatmap:** Standard deviation of P_det from B=200 bootstrap resamples.

---

## Software Tools and Libraries

| Tool | Version | Purpose | Status |
|------|---------|---------|--------|
| `scipy.interpolate.RegularGridInterpolator` | scipy >= 1.12 | 2D P_det grid interpolation | Already in use |
| `astropy.stats.binom_conf_interval` | astropy >= 5.0 | Wilson score CI per bin | Available (astropy is a dependency) |
| `astropy.stats.binned_binom_proportion` | astropy >= 5.0 | Vectorized binned binomial proportion with CI | Available |
| `numpy.histogram2d` | numpy >= 1.20 | 2D binning of injection events | Already in use |
| `pandas` | any recent | CSV loading of injection data | Already in use |
| `matplotlib` | any recent | Diagnostic plots | Already in use |

No additional dependencies required. All tools are already in the project's dependency tree.

---

## Recommended Investigation Scope

Prioritize:

1. **Add per-bin Wilson confidence intervals** to `SimulationDetectionProbability`. Store as metadata alongside the P_det grid. This is a small code addition (~30 lines) with high diagnostic value.

2. **Implement quality flag array** parallel to the P_det grid: flag bins with n_total < 10 or Wilson CI half-width > 0.15. Log summary statistics on construction.

3. **Test coarser grid (15x10)** as an alternative to 30x20. Compare the marginalized detection rate integral between the two. If they agree to within 2%, the coarser grid is sufficient and more statistically robust.

4. **Add diagnostic plots** (items 1, 2, 4 from the visualization list above) to validate grid quality after construction.

5. **Bootstrap stability test** with B=200 resamples on the production injection data. If the bootstrap standard deviation exceeds the Wilson CI in any bin, investigate correlated injection sampling effects.

Defer:
- Gaussian Process or neural network emulators: unnecessary complexity for 2D grids with 7 h-slices
- Adaptive/irregular binning: only if the coarser regular grid fails validation
- Importance-sampled injection distributions: requires modifying the injection campaign itself, which has already run

---

## Key References

| Reference | ID | Type | Relevance |
|-----------|-----|------|-----------|
| Farr (2019) | arXiv:1904.10879 | Research note | N_eff > 4*N_det criterion for selection function accuracy |
| Essick & Farr (2024) | arXiv:2404.16930 | Methods paper | Quick recipes for P_det computation; Marcum Q-function; noise effects on selection |
| Callister, Essick, Holz (2024) | arXiv:2408.16828 | Methods paper | Neural network P_det emulator; state-of-art for LIGO (overkill for our 2D case) |
| Brown, Cai, DasGupta (2001) | Stat. Sci. 16(2):101 | Review | Definitive comparison of binomial CI methods; recommends Wilson |
| Mandel, Farr, Gair (2019) | arXiv:1809.02293 | Review | Hierarchical Bayesian inference with selection effects |
| SciPy RegularGridInterpolator docs | scipy.org | Documentation | Interpolation methods: linear, cubic, pchip |
| Astropy binom_conf_interval docs | astropy.org | Documentation | Wilson score implementation |
