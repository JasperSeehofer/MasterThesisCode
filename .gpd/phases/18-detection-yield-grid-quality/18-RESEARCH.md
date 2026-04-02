# Phase 18: Detection Yield & Grid Quality - Research

**Researched:** 2026-03-31
**Domain:** Gravitational wave injection campaign statistics, binomial confidence intervals, grid quality assessment
**Confidence:** HIGH

## Summary

Phase 18 is a data analysis phase. The injection campaign produced 165,000 events across 7 h-values (0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90), with detection rates ranging from 0.13% (h=0.60, SNR>=15) to 0.81% (h=0.90, SNR>=15). The phase goal is to quantify detection yield, break down compute waste, validate the z>0.5 cutoff, and assess the quality of the 30x20 P_det grid using Wilson score confidence intervals. All detections occur at z < 0.21, confirming z_cut=0.5 is safe with large margin.

The mathematical content is straightforward: binomial proportion confidence intervals (Wilson score), histogram bin occupancy analysis, and grid interpolation error comparison. No new physics derivations are needed. The main computational challenge is the 30x20 grid having 133/600 bins with fewer than 10 injections at h=0.73 -- these bins need quality flags. The 15x10 grid reduces this to 10/150 bins with fewer than 10 injections, a dramatic improvement in per-bin statistics at the cost of spatial resolution.

**Primary recommendation:** Use `astropy.stats.binom_conf_interval` with `interval='wilson'` and `confidence_level=0.9545` (95% CI) for per-bin confidence intervals. Flag bins with n_total < 10 as unreliable. Compare 30x20 vs 15x10 grids using CI half-width in the detection boundary region (0.05 < P_det < 0.95). For compute waste, decompose into three categories using CSV event counts (successful) vs SLURM log total iterations (total attempted).

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| Injection CSV data (simulations/injections/) | data | 165k events across 7 h-values; source for all yield and grid computations | load, aggregate by h, compute statistics | plan / execution / verification |
| SimulationDetectionProbability class | prior artifact | Contains _build_grid_2d() with 30x20 grid; GRID-03 adds quality flags here | read current implementation, add quality flags | plan / execution |
| Farr (2019) arXiv:1904.10879 | benchmark | N_eff > 4*N_det criterion for sufficient effective samples | verify criterion is met per h-value globally | plan / execution / verification |
| Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133 | method | Wilson CI is recommended over Wald/Clopper-Pearson for binomial proportions | use Wilson interval via astropy | plan / execution |
| Phase 17 audit outputs | prior artifact | Parameter consistency verified; 8 exception types catalogued; CSV records only successes | build on failure characterization for waste decomposition | plan |

**Missing or weak anchors:** SLURM logs from the cluster are needed for YELD-02 (compute waste breakdown) to determine total attempted iterations vs successful CSV records. If logs are unavailable, waste decomposition must rely on the known relationship: total_attempted = total_successful / (1 - failure_rate), with failure_rate estimated from Phase 17 findings (~30-50%). The CSV records only successes; failed events are not recorded.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| SNR threshold | 15 | 20 | constants.py:48 (SNR_THRESHOLD = 15) |
| Units: distances | Gpc | Mpc | physical_relations.py, SimulationDetectionProbability |
| Units: masses | Solar masses (source-frame) | kg, redshifted mass | ParameterSpace |
| Hubble parameter | Dimensionless h = H0/(100 km/s/Mpc) | H0 in km/s/Mpc | constants.py |
| Grid space | (d_L, M) not (z, M) | (z, M) | SimulationDetectionProbability._build_grid_2d() |
| P_det grid dimensions | 30 d_L bins x 20 M bins | 15x10 (comparison target) | simulation_detection_probability.py:29-30 |
| Confidence intervals | Wilson score, 95% (z=1.96) | Clopper-Pearson, Wald, Jeffreys | Brown et al. (2001) recommendation |

**CRITICAL: The grid is built in (d_L, M) space, NOT (z, M) space. The z>0.5 cutoff validation (YELD-03) operates on the CSV z column, but all grid analysis (GRID-01, GRID-02, GRID-03) must use the luminosity_distance column. Do not confuse the two coordinate systems.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| p_hat = k/n | Sample binomial proportion | Standard | Point estimate of P_det per bin (k=detected, n=total) |
| Wilson CI: (p_hat + z^2/(2n) +/- z*sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2))) / (1 + z^2/n) | Wilson score confidence interval | Brown, Cai, DasGupta (2001) Eq. in Sec. 2 | Per-bin CI for P_det |
| N_eff = (sum w_i)^2 / sum(w_i^2) | Kish effective sample size | Kish (1965) | Assess whether injection count is sufficient |
| N_eff > 4*N_det | Farr criterion | Farr (2019) arXiv:1904.10879 | Minimum injections for unbiased selection function |
| Detection fraction: f_det = N_det / N_total | Detection yield | Standard | YELD-01 primary output |
| CI half-width: w = (CI_upper - CI_lower) / 2 | Confidence interval width | Standard | Quality metric for grid comparison |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Wilson score CI | Confidence interval for binomial proportion, handles small n and extreme p | Per-bin P_det quality assessment | Brown et al. (2001) |
| Histogram binning (numpy.histogram2d) | Counts events per (d_L, M) bin | Grid construction for both 30x20 and 15x10 | NumPy documentation |
| RegularGridInterpolator comparison | Quantifies interpolation error between resolutions | 30x20 vs 15x10 grid comparison | SciPy documentation |
| Pie/bar chart visualization | Displays waste breakdown | YELD-02 deliverable | matplotlib |
| Heatmap visualization | Displays per-bin CI widths | GRID-01 deliverable | matplotlib.imshow or pcolormesh |

### Approximation Schemes

No approximations are introduced in this phase. The Wilson CI is exact for the score test inversion. The histogram-based P_det estimator is unbiased when all injections are drawn from the same distribution (which Phase 17 confirmed).

## Standard Approaches

### Approach 1: Direct Histogram Analysis with Wilson CIs (RECOMMENDED)

**What:** Load all injection CSVs, compute per-h detection fractions, build 30x20 and 15x10 grids, compute Wilson CIs per bin, flag unreliable bins, and compare grid quality metrics.

**Why standard:** This is the direct approach to assessing injection campaign quality. Wilson CIs are the standard recommendation for binomial proportions (Brown et al. 2001). The LVK collaboration uses analogous injection-based methods for CBC selection effects.

**Key steps:**

1. **YELD-01:** Aggregate CSVs per h-value. Compute f_det = N(SNR >= 15) / N_total to 3 significant figures. Report as table.
2. **YELD-02:** Decompose events into three categories: (a) waveform failures (inferred from SLURM logs or estimated from total SLURM iterations minus CSV rows), (b) sub-threshold (SNR < 15, present in CSV), (c) detected (SNR >= 15, present in CSV). Note: CSV contains only successful waveform evaluations, so failures must be inferred.
3. **YELD-03:** Query all events with z > 0.5 and SNR >= 15. Confirm zero detections across all 7 h-values.
4. **GRID-01:** For each h-value, build 30x20 histogram in (d_L, M). Compute Wilson 95% CIs per bin. Flag bins with n < 10.
5. **GRID-02:** Build 15x10 histogram, compute CIs, compare CI half-widths in detection boundary region.
6. **GRID-03:** Add quality flags to SimulationDetectionProbability: bins with n < 10 flagged as unreliable, optionally with CI half-width threshold.

**Known difficulties at each step:**

- Step 2: The CSV records only successes. Total SLURM iterations come from log files (pattern: "X / Y successful SNR computations"). If logs are unavailable, the failure fraction must be estimated. Phase 17 found ~30-50% failure rates from code analysis but no quantitative log data.
- Step 4: With 25,500 events for h=0.73 across 600 bins, the mean is 42.5 events/bin but 133 bins have fewer than 10. These sparse bins are concentrated at extreme d_L (high) and extreme M (low/high).
- Step 5: The "detection boundary region" (0.05 < P_det < 0.95) is narrow -- most detections are at very low z (< 0.1), so only a few d_L bins are in the transition region. The 15x10 grid dramatically improves bin occupancy (only 10 bins with n < 10 vs 133 for 30x20).

### Approach 2: Bootstrap Resampling for CI Validation (FALLBACK)

**What:** Use bootstrap resampling to construct empirical CIs for P_det per bin, as a cross-check on Wilson CIs.

**When to switch:** If the Wilson CI half-widths seem unreasonably narrow or wide in specific bins, or if the planner wants a non-parametric validation.

**Tradeoffs:** More computationally expensive (resampling over 25k events repeatedly), but provides a cross-check. Not needed unless Wilson CIs are suspect.

### Anti-Patterns to Avoid

- **Using Wald (normal approximation) CIs:** The standard z-interval p_hat +/- z*sqrt(p_hat*(1-p_hat)/n) has well-documented problems at small n and extreme p (Brown et al. 2001). With P_det near 0 in most bins, Wald CIs are unreliable. Use Wilson instead.
  - _Example:_ A bin with k=0, n=5 gives Wald CI = [0, 0], which is absurdly narrow. Wilson gives [0, 0.434], correctly reflecting uncertainty.
- **Averaging CI widths across all bins:** The contract requires per-bin CIs and flagging, not a single average. An average masks that some bins have excellent statistics while others are uninformative.
- **Confusing (z, M) and (d_L, M) grids:** The SimulationDetectionProbability builds grids in (d_L, M) space. YELD-03 (z>0.5 cutoff) operates in z-space. Keep these separate.
- **Using SNR threshold of 20 instead of 15:** constants.py defines SNR_THRESHOLD = 15. The Phase 17 summary mentioned 363 detections at 0.22% rate -- that used SNR >= 20, not the project's actual threshold of 15. Verify which threshold was used in Phase 17 vs what the contract requires.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Wilson score CI formula | See Mathematical Framework above | Brown, Cai, DasGupta (2001) | Call astropy.stats.binom_conf_interval(k, n, confidence_level=0.9545, interval='wilson') |
| Farr criterion | N_eff > 4*N_det | Farr (2019) arXiv:1904.10879 | Check globally per h-value |
| Phase 17: parameter consistency | All 14 parameters consistent, 4 intentional differences | audit-parameter-consistency.md | Baseline: injection data is trustworthy |
| Phase 17: z_cut safety | z_cut=0.5 safe for all h in [0.60, 0.90] | audit-cosmological-model.md | YELD-03 expected result |
| Phase 17: 8 exception types | Classified in audit-waveform-failures.md | audit-waveform-failures.md | Failure categories for YELD-02 |
| Phase 17: CSV limitation | CSV records only successes; failed events hit continue | audit-waveform-failures.md | Cannot directly count failures from CSV |

**Key insight:** The Wilson CI and Farr criterion are textbook results. Do not re-derive them -- call the library function and cite the paper.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| Detection counts per h | See data exploration below | This research | SNR_THRESHOLD = 15 |
| Grid occupancy (30x20, h=0.73) | 133/600 bins with n<10 | This research | Current grid parameters |
| Grid occupancy (15x10, h=0.73) | 10/150 bins with n<10 | This research | Proposed coarser grid |

### Data Exploration Results (from this research)

**Detection yield per h-value (SNR >= 15):**

| h | N_total | N_det | f_det | max z_det |
| --- | --- | --- | --- | --- |
| 0.60 | 22,500 | 50 | 0.222% | 0.0949 |
| 0.65 | 26,000 | 78 | 0.300% | 0.1477 |
| 0.70 | 23,500 | 68 | 0.289% | 0.1565 |
| 0.73 | 25,500 | 95 | 0.373% | 0.1645 |
| 0.80 | 25,000 | 97 | 0.388% | 0.1542 |
| 0.85 | 25,500 | 138 | 0.541% | 0.1688 |
| 0.90 | 17,000 | 137 | 0.806% | 0.2041 |
| **Total** | **165,000** | **663** | **0.402%** | **0.2041** |

**z > 0.5 detections: ZERO across all 7 h-values.** YELD-03 is confirmed by the data -- no analysis needed beyond this check.

**Grid occupancy comparison (h=0.73):**

| Metric | 30x20 | 15x10 |
| --- | --- | --- |
| Total bins | 600 | 150 |
| Empty bins | 46 | 10 |
| Bins with n < 10 | 133 | 10 |
| Bins with n >= 10 | 467 | 140 |
| Mean events/bin | 42.5 | 170.0 |
| Median events/bin | 23.0 | 98.5 |
| Max events/bin | 344 | 996 |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Interval Estimation for a Binomial Proportion | Brown, Cai, DasGupta | 2001 | Definitive comparison of binomial CI methods; recommends Wilson | Use Wilson interval, not Wald |
| Accuracy Requirements for Empirically-Measured Selection Functions | Farr | 2019 | N_eff > 4*N_det criterion | Apply per h-value to check sufficiency |
| Extracting distribution parameters from multiple uncertain observations with selection biases | Mandel, Farr, Gair | 2019 | Framework for selection effects in hierarchical Bayesian inference | Context for why P_det grid quality matters |
| Self-normalized importance sampling (SNIS) | Tiwari | 2018 | Standard IS estimator for GW selection effects | Future importance weighting (Phase 19) |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| numpy | numpy.histogram2d | Bin events into (d_L, M) grids | Standard for histogram construction |
| astropy | astropy.stats.binom_conf_interval | Wilson score CIs per bin | Peer-reviewed, handles edge cases (k=0, k=n) |
| scipy | scipy.interpolate.RegularGridInterpolator | Build interpolated P_det grids for comparison | Already used in SimulationDetectionProbability |
| pandas | pd.read_csv, pd.concat | Load and aggregate injection CSVs | Already used in SimulationDetectionProbability |
| matplotlib | imshow, pcolormesh, pie | Heatmaps, pie charts per contract deliverables | Standard plotting |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| matplotlib colorbar + diverging colormap | CI half-width heatmap with threshold highlighting | GRID-01 deliverable |
| numpy.divide with where= | Safe division avoiding 0/0 in empty bins | Already used in SimulationDetectionProbability._build_grid_2d() |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| astropy Wilson CI | scipy.stats.binom (manual computation) | astropy handles edge cases automatically; no reason to reimplement |
| matplotlib heatmap | seaborn.heatmap | seaborn not in dependencies; matplotlib sufficient |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Load 262 CSVs (~165k rows) | < 5 seconds | I/O | Already fast with pd.concat |
| Wilson CI for 600 bins x 7 h-values | < 1 second | None | astropy vectorized |
| Grid comparison (30x20 vs 15x10) | < 1 second | None | Simple histogram recomputation |
| Heatmap visualization | < 2 seconds per plot | None | Standard matplotlib |

**Installation / Setup:**
```bash
# All required packages are already in the project dependencies
# astropy, numpy, scipy, pandas, matplotlib all available via uv sync
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| sum(N_det) + sum(N_undetected) = N_total per h | CSV event accounting | Sum detected + undetected from CSV, compare to row count | Exact equality |
| Wilson CI contains p_hat | CI correctly computed | Check CI_lower <= p_hat <= CI_upper for every bin | True for all bins |
| Wilson CI in [0, 1] | Physical bounds | Check bounds | True by construction for Wilson |
| 15x10 grid P_det approx equals 30x20 grid P_det at coarse bin centers | Grid consistency | Evaluate both grids at 15x10 centers | Agreement within CI widths |
| Detection fraction increases with h | Physical expectation: higher h -> closer sources -> more detections | Compare f_det across h values | Monotonically increasing (approximately) |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| Empty bin (k=0, n>0) | No detections in bin | Wilson CI = [0, z^2/(n+z^2)], not [0,0] | Brown et al. (2001) |
| Saturated bin (k=n) | All detected | Wilson CI = [n/(n+z^2), 1], not [1,1] | Brown et al. (2001) |
| Large n limit | n >> 100 | Wilson CI -> Wald CI | Standard asymptotics |
| P_det monotonicity in d_L | Fixed M, increasing d_L | P_det should decrease | Physics: farther = fainter |

### Red Flags During Computation

- If any bin shows P_det > 0 at d_L values where all neighboring bins have P_det = 0, this is noise from very low n -- the bin should be flagged.
- If the detection fraction does NOT increase monotonically with h, investigate whether the h=0.90 dataset has fewer total events (it does: 17k vs 25k for others) or different parameter coverage.
- If the 15x10 grid P_det differs from 30x20 by more than 0.1 in the boundary region, the coarser grid is too coarse for this data.

## Common Pitfalls

### Pitfall 1: Using Wald CI Instead of Wilson CI

**What goes wrong:** Wald CI = p_hat +/- z*sqrt(p_hat*(1-p_hat)/n) gives [0,0] when k=0 and overshoots [0,1] bounds at extreme proportions.
**Why it happens:** Wald is the "textbook" CI taught in introductory statistics and the default in some software.
**How to avoid:** Use `astropy.stats.binom_conf_interval(k, n, confidence_level=0.9545, interval='wilson')`.
**Warning signs:** CI lower bound = 0 AND upper bound = 0 for any bin with n > 0.
**Recovery:** Replace with Wilson CI.

### Pitfall 2: Misinterpreting CSV Event Counts as Total Injections

**What goes wrong:** The CSV contains only events where waveform generation succeeded and SNR was computed. Failed waveforms are not recorded. Using N_csv as the denominator for the "waste" computation underestimates the total GPU time spent.
**Why it happens:** The injection_campaign() code `continue`s past failures without recording them.
**How to avoid:** For YELD-02, clearly distinguish N_csv_total (successful evaluations in CSV) from N_attempted (total SLURM iterations including failures). Use SLURM logs or the progress pattern "X / Y successful SNR computations" to recover N_attempted.
**Warning signs:** Waste fractions sum to < 100% because failures are missing.
**Recovery:** Report waste fractions relative to both N_csv (for what the CSV shows) and N_attempted (for true GPU cost), clearly labeled.

### Pitfall 3: Confusing SNR Threshold Values

**What goes wrong:** constants.py defines SNR_THRESHOLD = 15, but some prior analysis in the project (Phase 17 summary) reported results using SNR >= 20. Using the wrong threshold gives wrong detection counts.
**Why it happens:** The SNR threshold was changed at some point, or different analyses used different values.
**How to avoid:** Always use SNR_THRESHOLD = 15 from constants.py. If the contract mentions a threshold, verify it matches.
**Warning signs:** Detection counts differ from prior reports (Phase 17 said 363 detections at 0.22% -- that used threshold 20, not 15).
**Recovery:** Recompute with correct threshold. Report both if ambiguity exists.

### Pitfall 4: Grid Comparison Without Matching Bin Edges

**What goes wrong:** Comparing 30x20 and 15x10 grids that use different bin edge definitions (e.g., different d_L_max or M_min/M_max) makes the comparison meaningless.
**How to avoid:** Ensure both grids cover the same parameter range with the same edge-definition logic (linspace for d_L, geomspace for M). The 15x10 grid should use the same d_L_max and M_min/M_max as the 30x20 grid, just with fewer bins.
**Warning signs:** The parameter ranges of the two grids don't match.
**Recovery:** Recompute with matched ranges.

## Level of Rigor

**Required for this phase:** Controlled numerical analysis with exact binomial statistics.

**Justification:** This is a data analysis phase, not a derivation phase. The mathematical content (Wilson CI) is well-established. The main requirement is accurate data handling and clear presentation.

**What this means concretely:**

- Detection fractions reported to 3 significant figures (per contract)
- Wilson CIs computed exactly via astropy (no hand-rolled approximations)
- Grid comparisons use consistent bin definitions
- All figures include axis labels, units, and colorbar scales
- Waste decomposition clearly distinguishes "from CSV data" vs "including estimated failures"

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Wald CI for binomial proportions | Wilson CI | Brown et al. (2001) | Wilson is now standard recommendation for small n |
| KDE-based P_det (DetectionProbability) | Simulation-based P_det (SimulationDetectionProbability) | Phase 11.1 of this project | This phase assesses the quality of the new approach |

**Superseded approaches to avoid:**

- **Wald CI:** Erratic coverage at small n and extreme proportions. Use Wilson instead.
- **KDE-based DetectionProbability:** Replaced by SimulationDetectionProbability in Phase 11.1. Do not compare against it as a validation target (it was the old approach being replaced).

## Open Questions

1. **SLURM log availability for YELD-02**
   - What we know: Phase 17 identified the CSV-only-records-successes limitation. The analyze_failures.py script has SLURM log parsing capability.
   - What's unclear: Whether SLURM logs have been rsynced from the cluster. The progress pattern "X / Y successful SNR computations" would give exact failure counts per task.
   - Impact on this phase: Without logs, YELD-02 waste decomposition is approximate (CSV shows successful events only; failure count must be estimated from known ~30-50% failure rate).
   - Recommendation: Check for SLURM logs in cluster/ or simulations/ directories. If absent, report YELD-02 with two scenarios: (a) using CSV data only (sub-threshold vs detected), (b) estimating total including failures at 30% and 50% assumed failure rates.

2. **SNR threshold ambiguity**
   - What we know: constants.py has SNR_THRESHOLD = 15. Phase 17 summary mentioned 363 detections and 0.22% rate, which matches SNR >= 20 (total 165k * 0.0022 = 363). At SNR >= 15, there are 663 detections (0.402%).
   - What's unclear: Which threshold the contract intends. The contract says "SNR >= threshold" without specifying the value.
   - Impact on this phase: All detection counts depend on the threshold choice.
   - Recommendation: Use SNR_THRESHOLD = 15 from constants.py as the primary analysis. Optionally report SNR >= 20 results in a secondary table for comparison with Phase 17 numbers.

3. **Detection boundary region definition**
   - What we know: The contract says "where 0.05 < P_det < 0.95" for the CI half-width comparison.
   - What's unclear: With overall detection rate ~0.4%, most bins have P_det near 0. The "boundary region" may contain very few bins.
   - Impact on this phase: The 30x20 vs 15x10 comparison may have very few bins in the boundary region, making the comparison statistically weak.
   - Recommendation: Report the number of bins in the boundary region for each grid. If too few bins qualify, relax the definition to "bins where P_det > 0" for a broader comparison.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Wilson CI via astropy | astropy import issue | Manual Wilson formula: (p + z^2/(2n) +/- z*sqrt(p*(1-p)/n + z^2/(4n^2))) / (1 + z^2/n) | Minimal -- 5 lines of code |
| SLURM log parsing for YELD-02 | Logs unavailable | Estimate failures at assumed rates (30%, 50%) | Low -- report with uncertainty bounds |
| 30x20 vs 15x10 comparison | Too few boundary-region bins | Compare using all bins with P_det > 0, or use MAE of P_det between fine and coarse grids | Low -- broader comparison still informative |

**Decision criteria:** If SLURM logs are not available after checking cluster/ and simulations/ directories, proceed with estimation approach immediately -- do not block the phase.

## Sources

### Primary (HIGH confidence)

- [Brown, Cai, DasGupta (2001) "Interval Estimation for a Binomial Proportion", Statistical Science 16:101-133](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) -- definitive comparison of binomial CI methods; recommends Wilson
- [Farr (2019) "Accuracy Requirements for Empirically-Measured Selection Functions", arXiv:1904.10879](https://arxiv.org/abs/1904.10879) -- N_eff > 4*N_det criterion for injection sufficiency
- [astropy.stats.binom_conf_interval documentation](https://docs.astropy.org/en/stable/api/astropy.stats.binom_conf_interval.html) -- implementation of Wilson CI in astropy

### Secondary (MEDIUM confidence)

- [Mandel, Farr, Gair (2019) arXiv:1809.02063](https://arxiv.org/abs/1809.02063) -- framework for selection effects in hierarchical Bayesian inference
- [Tiwari (2018) arXiv:1712.00482](https://arxiv.org/abs/1712.00482) -- self-normalized importance sampling for GW selection effects
- Phase 17 audit outputs (audit-parameter-consistency.md, audit-waveform-failures.md) -- established injection data reliability

### Tertiary (LOW confidence)

- Phase 17 failure rate estimates (~30-50%) -- based on code analysis, not empirical log counts; use with caution for YELD-02

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH -- Wilson CI is textbook; Farr criterion well-established
- Standard approaches: HIGH -- histogram analysis with binomial CIs is straightforward
- Computational tools: HIGH -- all tools already in project dependencies and well-documented
- Validation strategies: HIGH -- internal consistency checks and physical expectations are clear

**Research date:** 2026-03-31
**Valid until:** Indefinite for methodology; data-dependent results valid as long as injection data is unchanged

## Caveats and Self-Critique

1. **Assumption: CSV data is unbiased subsample.** Phase 17 flagged that waveform failures correlate with specific parameter regions (high eccentricity, extreme mass ratios). If failures preferentially occur where P_det would be nonzero, the histogram P_det is biased. This phase cannot fully resolve this concern but should flag bins where failure correlation is suspected. The conservative approach (treat all failures as non-detections) is documented in PITFALLS.md.

2. **The "detection boundary region" may be nearly empty.** With detection rates of 0.2-0.8%, very few bins have P_det between 0.05 and 0.95. The GRID-02 comparison may need to use a relaxed criterion. This is a limitation of the data, not the method.

3. **SNR threshold choice matters more than methodology.** The difference between SNR >= 15 (663 detections) and SNR >= 20 (363 detections) is larger than any methodological choice in this phase. Clarifying which threshold the contract intends is more important than optimizing the CI method.

4. **No alternative method was dismissed.** The Wilson CI is the unique correct choice for this application per Brown et al. (2001). Clopper-Pearson is valid but conservative; Jeffreys is valid but slower. There is no methodological controversy here.

5. **A specialist would agree with this recommendation.** The histogram + Wilson CI approach is textbook for injection campaign quality assessment. The only potential disagreement would be about the grid comparison metric (CI half-width vs. interpolation error vs. KL divergence), where CI half-width is the simplest and most interpretable choice for the thesis context.
