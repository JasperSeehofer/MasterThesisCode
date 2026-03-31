# Research Summary

**Project:** EMRI Parameter Estimation -- Injection Campaign Physics Analysis (v1.2.2)
**Domain:** Gravitational wave selection effects, EMRI detection probability, dark siren H0 inference
**Researched:** 2026-03-31
**Confidence:** MEDIUM

## Unified Notation

| Symbol | Quantity | Units/Dimensions | Convention Notes |
|--------|---------|-----------------|-----------------|
| P_det | Detection probability | dimensionless [0,1] | P(SNR >= threshold \| z, M, h) marginalized over extrinsic parameters |
| z | Source redshift | dimensionless | Cosmological redshift of the EMRI host |
| M | Source-frame MBH mass | M_sun | Mass of the massive black hole |
| M_z | Observer-frame (redshifted) mass | M_sun | M_z = M * (1+z) |
| h | Dimensionless Hubble parameter | dimensionless | H0 = 100h km/s/Mpc; h = 0.73 is the project fiducial |
| d_L | Luminosity distance | Gpc | From dist(z, h) using Omega_m = 0.25 (WMAP-era, known outdated) |
| w_i | Importance weight | dimensionless | w_i = p(theta_i) / q(theta_i) for sample i |
| N_eff | Effective sample size | dimensionless | Kish formula: (sum w)^2 / sum w^2 |
| p(theta) | Physical/population prior density | [theta]^{-1} | From Model1CrossCheck EMRI rate distribution |
| q(theta) | Proposal/sampling density | [theta]^{-1} | Distribution actually used to draw injections |
| SNR | Signal-to-noise ratio | dimensionless | Matched-filter SNR against LISA PSD; threshold = 20 |

**Convention notes:**
- All distances in Gpc throughout. The dist() function returns Gpc; all P_det lookups expect Gpc.
- Masses in solar masses, source-frame unless subscripted _z.
- The project uses WMAP-era cosmology (Omega_m = 0.25, h = 0.73). This is internally consistent but physically outdated (Planck 2018: Omega_m = 0.3153, h = 0.6736). No convention conflict exists across research files; all four use the same system.
- Fourier conventions are not relevant for this milestone (no waveform modifications).

## Executive Summary

The injection campaign for simulation-based P_det estimation follows the standard gravitational wave community methodology established by Mandel, Farr & Gair (2019) and operationalized by the LVK collaboration. The approach -- draw synthetic EMRIs from a population prior, compute SNR via full waveform generation, and bin detections into a (z, M) histogram per h-value -- is sound in principle. However, the current implementation has three quantifiable inefficiencies: (1) uniform sampling wastes the majority of GPU compute on events with zero detection probability, (2) the 30x20 grid produces many bins with fewer than 10 injections, degrading P_det accuracy in the transition region where it matters most, and (3) waveform generation failures at a rate of 30-50% effectively reduce the injection budget by a factor of 2-3 without any tracking of where failures concentrate.

The recommended approach is a staged improvement: first, add importance weight tracking and Wilson confidence intervals to the existing P_det grid builder (minimal code change, high diagnostic value); second, implement stratified sampling with Neyman-optimal allocation to concentrate injections near the detection boundary; third, reduce grid resolution from 30x20 to 15x10 bins to ensure adequate per-bin statistics with the available injection count. These improvements do not change the total injection budget -- they redistribute the same compute more efficiently, with an expected 2-5x reduction in P_det boundary variance. The self-normalized importance sampling estimator (Tiwari 2018) is the standard tool and is guaranteed unbiased to O(1/N).

The principal risks are: (a) waveform failure correlation with specific parameter regions creating a biased subsample (the failure rate is high enough to matter, and currently failures are silently skipped without tracking); (b) the z > 0.5 importance sampling cutoff is physically motivated but not rigorously validated for all h values; and (c) the P_det grid enters the likelihood denominator, so even modest systematic errors propagate into the H0 posterior. All three risks are addressable with the auditing and diagnostics recommended below.

## Key Findings

### Computational Approaches

The P_det grid construction pipeline (injection CSV -> histogram2d -> RegularGridInterpolator -> linear h-interpolation) is functional and uses appropriate tools (numpy, scipy, astropy all already in the dependency tree). No new dependencies are needed. [CONFIDENCE: HIGH]

**Core approach:**

- **Histogram-based P_det estimator:** Simple N_det/N_total per (z, M) bin -- correct, standard, but high variance in sparse bins. The self-normalized importance-weighted variant (sum(w_i * I_i) / sum(w_i)) generalizes this to non-uniform proposals with O(1/N) bias.
- **RegularGridInterpolator (linear):** Adequate for current grid resolution. Cannot overshoot [0,1] bounds. PCHIP and cubic alternatives rejected due to overshoot risk on noisy grids. [CONFIDENCE: HIGH]
- **Wilson score confidence intervals:** Recommended per-bin quality metric. Correctly handles small n and extreme proportions, unlike Wald intervals. Available via astropy.stats.binom_conf_interval. [CONFIDENCE: HIGH -- Brown, Cai, DasGupta 2001 is definitive]

**Deferred (overkill for current scale):** Gaussian process regression, neural network P_det emulators (Callister et al. 2024), cross-entropy adaptive importance sampling.

### Prior Work Landscape

The GW community approach to selection effects via injection campaigns is well-established for compact binary coalescences (CBCs). EMRI-specific injection campaigns are less mature -- Babak et al. (2017) used semi-analytic SNR rather than full waveform injections. This project's full-waveform approach is more accurate but more expensive. [CONFIDENCE: HIGH for methodology; MEDIUM for EMRI-specific validation]

**Must reproduce (benchmarks):**

- P_det(z, M | h) is monotonically decreasing in z at fixed M (farther = fainter) -- any non-monotonic structure beyond statistical noise indicates a bug or waveform failure artifact
- P_det = 1 at low z, high M and P_det = 0 at high z, low M -- physical boundary conditions
- Farr (2019) global criterion: N_eff > 4 * N_det (~80 effective injections needed for ~20 detections) -- easily satisfied globally but per-bin is the challenge

**Novel contributions:**

- Full-waveform EMRI injection campaign with FEW/fastlisaresponse (no published EMRI injection optimization exists)
- Importance-weighted P_det grid estimation for EMRIs -- if implemented, this is a methodological advance for the EMRI community

**Defer:**

- Neural network P_det emulators (applicable at LIGO scale, overkill for 2D grids with 7 h-slices)
- Cross-entropy rare-event estimation (only needed if P_det < 0.01 cells require precision)

### Methods and Tools

The recommended implementation order is: (1) record proposal densities and add importance-weight support to the histogram estimator, (2) add Wilson CIs and quality flags per bin, (3) implement stratified sampling with Neyman allocation for future campaigns, (4) test a coarser 15x10 grid as an alternative to the current 30x20. All methods are standard Monte Carlo techniques with well-understood convergence properties. The key insight is that improved sampling redistributes fixed compute without increasing cost.

**Major components:**

1. **Self-normalized IS estimator** -- handles arbitrary proposal distributions; enables reusing injections across different population models (Tiwari 2018)
2. **Stratified sampling with Neyman allocation** -- concentrates samples where variance is highest (P_det ~ 0.5 boundary); expected 2-5x variance reduction in boundary bins
3. **Wilson score CI per bin** -- quality diagnostic flagging bins with insufficient statistics (n < 10 or CI half-width > 0.15)
4. **Adaptive boundary refinement** -- two-stage approach using 30% pilot + 70% targeted allocation near the detection boundary

### Critical Pitfalls

1. **Waveform failure correlation with parameter regions [CRITICAL]** -- 30-50% of injections fail (timeouts, parameter bounds, solver failures), and failures are silently skipped. If failures correlate with specific (z, M, e0, a) regions, the surviving subsample is biased. Prevention: track all failures with parameters in a separate CSV or status column; quantify failure rate by (z, M) bin; apply conservative correction treating failures as non-detections.

2. **Parameter distribution mismatch between injection and evaluation [CRITICAL]** -- Injection samples from the population prior p(M, z); evaluation queries P_det at per-event posterior locations. Empty bins near detected events bias the selection-effect correction. Prevention: verify every detected event falls in a well-populated bin; overlay injection distribution with detected events.

3. **z > 0.5 cutoff as unvalidated importance sampling [MODERATE]** -- The cutoff is empirically motivated (all 24 detections at z < 0.18) but is an implicit importance sampling assumption. The cutoff validity is h-dependent: lower h reduces d_L at fixed z, potentially allowing detections beyond z = 0.5. Prevention: validate with a small no-cutoff batch at extreme h values.

4. **Low bin counts in 2D histogram [MODERATE]** -- 10,000 events / 600 bins = ~17 events/bin average, but non-uniform distribution means many bins have < 5 events. Prevention: reduce grid to 15x10 (150 bins, ~67/bin average) or merge sparse bins; add Wilson CI quality flags.

5. **d_L-to-z inversion inconsistency [MODERATE]** -- Evaluation inverts d_L to z via root-finding; any numerical precision difference or cosmological parameter mismatch vs. injection creates bin lookup errors. Prevention: add round-trip test z -> d_L -> z with tolerance 1e-4.

## Approximation Landscape

| Method | Valid Regime | Breaks Down When | Controlled? | Complements |
|--------|------------|-----------------|------------|-------------|
| Histogram P_det (N_det/N_total) | N_total > ~30 per bin, P_det not extreme | N_total < 5; dominates by Poisson noise | Yes (binomial variance = P(1-P)/N) | Importance-weighted estimator for non-uniform proposals |
| Self-normalized IS estimator | N_eff > 50 per bin; proposal has full support | Proposal has thin tails (extreme weights); N_eff < 10 | Yes, bias ~ O(1/N), variance via Kish N_eff | Stratified sampling to control per-bin N_eff |
| Stratified/Neyman allocation | Known approximate P_det from pilot; stable boundary | Pilot too small to identify boundary (<~2000 samples); boundary shifts between pilot and production | Yes (variance reduction factor calculable from pilot) | Boundary refinement for sharp transitions |
| Linear interpolation (z, M grid) | P_det varies slowly within each bin | Sharp detection boundary crosses bin interior; P_det gradient >> 1/bin_width | No formal error bound; empirical testing needed | Finer grid or PCHIP interpolation |
| Linear interpolation (h grid) | P_det(h) approximately linear between grid points | Strong curvature in P_det(h) at fixed (z, M) near the detection boundary | No formal bound; d_L(z,h) is nonlinear in h | Add intermediate h values or use cubic with clipping |
| z > 0.5 cutoff | P_det(z > 0.5) = 0 for all M, all h | Lower h values push detection horizon outward; different mass ranges | No (empirical assumption) | Run validation batch without cutoff |

**Coverage gap:** No reliable P_det estimate exists for bins with fewer than ~10 injections. With the current 30x20 grid and 10,000 injections per h, a significant fraction of bins (especially high-z, extreme-M) fall below this threshold. The coarser grid (15x10) is the simplest fix.

## Theoretical Connections

### Importance Sampling as a Unifying Framework [ESTABLISHED]

The self-normalized IS estimator unifies several currently separate operations: (1) the existing z > 0.5 cutoff is implicit importance sampling with a step-function proposal, (2) stratified sampling is IS with a piecewise-constant proposal matched to bin boundaries, (3) adaptive boundary refinement is IS with a pilot-estimated proposal. Implementing the general IS estimator (with weight tracking) from the start makes all future sampling improvements automatic -- any change to the proposal only requires updating the weight computation.

### Binomial Statistics for Quality Assessment [ESTABLISHED]

The per-bin P_det estimate is a binomial proportion. This connects directly to the well-studied problem of binomial confidence intervals (Wilson, Clopper-Pearson, Jeffreys). The Wilson interval is recommended because it performs well at small n and extreme proportions -- exactly the regime of most P_det bins.

### Hierarchical Bayesian Inference Framework [ESTABLISHED]

P_det enters the dark siren likelihood as a normalization factor: the denominator integral over the population must account for selection effects (Mandel, Farr & Gair 2019). This means P_det accuracy requirements are set by the desired H0 posterior precision, not by some abstract statistical threshold. The Farr (2019) N_eff > 4*N_det criterion quantifies this connection.

### Cross-Validation Opportunity [CONJECTURED]

The old KDE-based DetectionProbability class and the new histogram-based SimulationDetectionProbability should agree qualitatively (same boundary shape, same high/low regions). Quantitative differences are expected due to different smoothing assumptions. This comparison is a free cross-check that validates both implementations.

## Critical Claim Verification

| # | Claim | Source | Verification | Result |
|---|-------|--------|--------------|--------|
| 1 | Self-normalized IS estimator is standard for GW selection effects | METHODS.md | web_search: Tiwari 2018 arXiv:1712.00482 | CONFIRMED -- published in CQG, widely cited |
| 2 | Farr (2019) N_eff > 4*N_det criterion | PRIOR-WORK.md, COMPUTATIONAL.md | web_search: Farr 2019 arXiv:1904.10879 | CONFIRMED -- published in RNAAS, widely adopted |
| 3 | Wilson score CI recommended for small n binomial proportions | COMPUTATIONAL.md | web_search: Brown Cai DasGupta 2001 | CONFIRMED -- definitive review in Statistical Science |
| 4 | EMRI waveform failures at 30-50% rate | PITFALLS.md | Code context (CLAUDE.md known bugs, waveform hangs) | CONFIRMED by project experience -- 30s/90s timeouts documented |
| 5 | All detections at z < 0.18, z_cut = 0.5 is safe margin | PITFALLS.md | PROJECT.md confirms 24/69500 detections, all z < 0.18 | CONFIRMED from project data |

### Cross-Validation Matrix

|                        | Histogram P_det | IS-weighted P_det | Old KDE P_det | Analytic (SNR scaling) |
|-----------------------|:---:|:---:|:---:|:---:|
| **Histogram P_det**    | --- | Identical when weights=1 | Qualitative boundary shape | Boundary location z_max(M) |
| **IS-weighted P_det**  | Identical when uniform | --- | Qualitative boundary shape | Boundary location z_max(M) |
| **Old KDE P_det**      | Boundary shape | Boundary shape | --- | Boundary location |
| **Analytic (SNR~M^{5/6}/d_L)** | z_max(M) contour | z_max(M) contour | z_max(M) contour | --- |

**Reading:** The SNR scaling relation SNR ~ M^{5/6}/d_L provides an independent check on where the detection boundary should lie. The old KDE-based P_det serves as a qualitative cross-validation of the boundary shape.

### Input Quality -> Roadmap Impact

| Input File | Quality | Affected Recommendations | Impact if Wrong |
|------------|---------|------------------------|-----------------|
| METHODS.md | GOOD | Sampling strategy, estimator choice, implementation order | Would change whether IS or stratified sampling is recommended first |
| PRIOR-WORK.md | GOOD | Validation targets, benchmark criteria | Could miss a standard quality metric |
| COMPUTATIONAL.md | GOOD | Tool selection, grid resolution, interpolation choice | Would affect specific implementation details |
| PITFALLS.md | GOOD | Risk mitigation, failure tracking design, audit scope | Blind spots in pitfall coverage could lead to undetected biases |

## Implications for Roadmap

### Phase 1: Injection Physics Audit

**Rationale:** Before improving sampling or grids, verify that the existing injection pipeline is internally consistent. Mismatch between injection and evaluation parameters would invalidate all P_det grids regardless of statistical quality.
**Delivers:** Verified parameter consistency (z, M, d_L, h, cosmology) between injection_campaign() and BayesianStatistics; documented waveform failure statistics; confirmed z > 0.5 cutoff validity.
**Validates:** Round-trip d_L-to-z inversion consistency to 1e-4; M sampling range matches between pipelines; cosmological parameters identical in both paths.
**Avoids:** Pitfall 1 (parameter distribution mismatch), Pitfall 6 (d_L-to-z inconsistency).
**Methods:** Code audit, round-trip numerical tests, log parsing for failure statistics.
**Risk:** LOW -- primarily code reading and simple numerical checks.

### Phase 2: P_det Grid Quality Assessment

**Rationale:** Quantify the quality of existing P_det grids before designing improvements. Need to know which bins are reliable and which are not.
**Delivers:** Per-bin Wilson CIs; quality flag arrays; injection count heatmaps; P_det(z) slice plots with error bands; comparison between 30x20 and 15x10 grids.
**Uses:** Wilson score interval (astropy.stats.binom_conf_interval), Kish effective sample size, diagnostic visualizations.
**Builds on:** Phase 1 verification that injection data is internally consistent.
**Avoids:** Pitfall 4 (low bin counts without diagnostics).
**Risk:** LOW -- standard statistical diagnostics on existing data.

### Phase 3: Enhanced Sampling Design

**Rationale:** Design the improved sampling strategy informed by Phase 2 diagnostics showing where the grid is weak.
**Delivers:** Importance weight tracking in injection pipeline; stratified allocation algorithm for future campaigns; proposal density design for boundary-enhanced sampling.
**Uses:** Self-normalized IS estimator (Tiwari 2018); Neyman-optimal allocation; defensive mixture proposal.
**Builds on:** Phase 2 boundary identification and quality metrics.
**Avoids:** Pitfall 3 (untracked importance sampling), Pitfall 2 (failure tracking).
**Risk:** MEDIUM -- requires modifying injection_campaign() code and validating that weights are correctly computed.

### Phase 4: Validation and Cross-Checks

**Rationale:** Verify that the enhanced sampling produces P_det grids consistent with the original uniform approach, and that the H0 posterior is not sensitive to the methodological changes.
**Delivers:** Side-by-side comparison of uniform vs. stratified P_det grids; P_det monotonicity verification; marginalized detection rate integral stability check; comparison with old KDE P_det.
**Uses:** Bootstrap stability test (B=200); integral consistency check; cross-validation with analytic SNR scaling.
**Builds on:** Phase 3 enhanced sampling implementation.
**Risk:** LOW -- comparison and validation tests on existing and new data.

### Phase Ordering Rationale

- Phase 1 must come first: no point improving sampling if the injection pipeline has parameter inconsistencies that would propagate to any grid.
- Phase 2 before Phase 3: quantify current grid quality to know where improvements are needed and set the baseline for measuring improvement.
- Phase 3 depends on Phase 2: the boundary-enhanced proposal requires knowing where the boundary is from the Phase 2 analysis.
- Phase 4 after Phase 3: validation requires both the old (uniform) and new (stratified) grids to compare.

### Phases Requiring Deep Investigation

- **Phase 3 (Enhanced Sampling Design):** Requires careful implementation of importance weights, validated against the analytical result that weights=1 recovers the standard estimator. The proposal distribution design (mixture of uniform + boundary-enhanced) needs tuning parameters (alpha, sigma_z) that depend on Phase 2 results. The waveform failure tracking is a new feature requiring code changes to injection_campaign().

Phases with established methodology (straightforward execution):

- **Phase 1 (Injection Physics Audit):** Standard code audit and numerical round-trip tests. Well-defined checks.
- **Phase 2 (P_det Grid Quality):** Wilson CIs, histograms, and diagnostic plots are textbook statistics. astropy provides the implementation.
- **Phase 4 (Validation):** Comparison tests and bootstrap are standard.

## Confidence Assessment

| Area | Confidence | Notes |
|------|-----------|-------|
| Computational Approaches | HIGH | All recommended tools (numpy, scipy, astropy) are mature; histogram estimator is textbook; Wilson CI is well-studied |
| Prior Work | HIGH | GW selection effects methodology is well-established (LVK); Tiwari 2018 and Farr 2019 are widely cited and adopted |
| Methods | MEDIUM | IS estimator and stratified sampling are well-understood in general, but EMRI-specific application is novel; no published benchmark exists for EMRI injection optimization |
| Pitfalls | MEDIUM | Critical pitfalls (failure correlation, parameter mismatch) are well-identified but their quantitative impact depends on the actual injection data which has not yet been fully analyzed |

**Overall confidence:** MEDIUM -- methodology is sound and well-grounded in the GW literature, but the EMRI-specific application is novel territory without published benchmarks. The main uncertainty is quantitative (how bad are the waveform failures? how sparse are the bins in practice?) rather than methodological.

### Gaps to Address

- **Waveform failure statistics:** Unknown until injection logs are parsed or failure tracking is added. This is the single largest uncertainty -- if failures are 50% and correlated with parameter regions, the existing P_det grids may have uncontrolled bias.
- **Actual detection yield:** 24/69500 is the only data point so far. Per-h-value detection fractions needed.
- **h-interpolation accuracy:** No test of whether linear interpolation between h grid points is adequate. Requires either a test injection run at intermediate h or an analytical argument based on d_L(z,h) curvature.

## Open Questions

1. **What is the waveform failure fraction by (z, M) region?** [HIGH priority, blocks Phase 1 completion] -- determines whether existing P_det grids have parameter-correlated bias.
2. **Does the z > 0.5 cutoff remain valid at h = 0.60?** [MEDIUM priority, blocks Phase 3 for extreme h values] -- lower h increases d_L, potentially pushing the detection horizon beyond z = 0.5 for massive EMRIs.
3. **Is the 15x10 coarser grid sufficient?** [MEDIUM priority, Phase 2 deliverable] -- compare marginalized detection rate between 30x20 and 15x10; if they agree to 2%, the coarser grid is adequate.
4. **How sensitive is the H0 posterior to per-bin P_det noise?** [MEDIUM priority, Phase 4] -- determines whether the current injection count is sufficient or whether the campaign needs extension.

## Sources

### Primary (HIGH)

- [Tiwari (2018), "Weighted MC for GW sensitive volume," CQG, arXiv:1712.00482](https://arxiv.org/abs/1712.00482) -- Self-normalized IS estimator for P_det; directly applicable weight formula
- [Farr (2019), "Accuracy Requirements for Selection Functions," RNAAS, arXiv:1904.10879](https://arxiv.org/abs/1904.10879) -- N_eff > 4*N_det criterion
- [Mandel, Farr & Gair (2019), arXiv:1809.02063](https://arxiv.org/abs/1809.02063) -- Hierarchical Bayesian framework for GW population inference with selection effects
- [Brown, Cai, DasGupta (2001), Stat. Sci. 16(2):101-133](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) -- Binomial CI comparison; recommends Wilson score

### Secondary (MEDIUM)

- [Gerosa & Fishbach (2024), "Quick recipes for GW selection effects," arXiv:2404.16930](https://arxiv.org/abs/2404.16930) -- Detection probability approximations, Marcum Q-function
- [Chua & Vallisneri (2023), "Rapid EMRI sensitivity with ML," arXiv:2212.06166](https://arxiv.org/abs/2212.06166) -- Neural network SNR interpolation for EMRIs; demonstrates smoothness of SNR landscape
- [Gray et al. (2020)](https://arxiv.org/abs/1908.06050) -- Dark siren H0 inference; direct ancestor of this approach
- [Owen (2013), "Monte Carlo theory, methods and examples"](https://artowen.su.domains/mc/) -- Standard MC reference for stratified and importance sampling
- [Kish (1965), "Survey Sampling," Wiley](https://www.wiley.com/en-us/Survey+Sampling-p-9780471109495) -- Effective sample size formula

### Tertiary (LOW)

- [Speri et al. (2025), arXiv:2509.08875](https://arxiv.org/abs/2509.08875) -- Systematic errors in fast EMRI waveforms; relevant to waveform failure characterization
- [Callister, Essick, Holz (2024), arXiv:2408.16828](https://arxiv.org/abs/2408.16828) -- Neural network P_det emulator for LIGO; overkill for current application but reference for future work
- [Veach & Guibas (1995), SIGGRAPH](https://dl.acm.org/doi/10.1145/218380.218498) -- Multiple importance sampling / balance heuristic

---

_Research analysis completed: 2026-03-31_
_Ready for research plan: yes_

```yaml
# --- ROADMAP INPUT (machine-readable, consumed by gpd-roadmapper) ---
synthesis_meta:
  project_title: "EMRI Parameter Estimation -- Injection Campaign Physics Analysis (v1.2.2)"
  synthesis_date: "2026-03-31"
  input_files: [METHODS.md, PRIOR-WORK.md, COMPUTATIONAL.md, PITFALLS.md]
  input_quality: {METHODS: good, PRIOR-WORK: good, COMPUTATIONAL: good, PITFALLS: good}

conventions:
  unit_system: "mixed: natural (GW), Gpc (distances), M_sun (masses)"
  metric_signature: "N/A (no waveform modifications this milestone)"
  coupling_convention: "N/A"
  renormalization_scheme: "N/A"

methods_ranked:
  - name: "Self-normalized importance-weighted histogram estimator"
    regime: "N_eff > 50 per bin; proposal has full support over prior"
    confidence: HIGH
    cost: "O(N) per grid build (negligible vs injection cost)"
    complements: "Stratified sampling to control per-bin N_eff"
  - name: "Stratified sampling with Neyman-optimal allocation"
    regime: "Pilot run identifies boundary (>2000 pilot samples)"
    confidence: HIGH
    cost: "Same total injections, redistributed; ~3h wall time per h-value"
    complements: "IS estimator for combining pilot + targeted samples"
  - name: "Wilson score binomial confidence intervals"
    regime: "Any n >= 1; especially good for small n and extreme proportions"
    confidence: HIGH
    cost: "O(1) per bin (negligible)"
    complements: "Quality flagging for grid reliability assessment"
  - name: "Adaptive boundary refinement (two-stage)"
    regime: "Well-identified boundary from pilot; 30% pilot + 70% targeted"
    confidence: MEDIUM
    cost: "Same total injections; requires boundary estimation from pilot"
    complements: "IS estimator with defensive mixture proposal"

phase_suggestions:
  - name: "Injection Physics Audit"
    goal: "Verify parameter consistency between injection and evaluation pipelines; quantify waveform failure statistics"
    methods: ["Self-normalized importance-weighted histogram estimator"]
    depends_on: []
    needs_research: false
    risk: LOW
    pitfalls: ["parameter-distribution-mismatch", "dl-to-z-inversion-inconsistency"]
  - name: "P_det Grid Quality Assessment"
    goal: "Quantify per-bin reliability of existing P_det grids with Wilson CIs and quality flags"
    methods: ["Wilson score binomial confidence intervals"]
    depends_on: ["Injection Physics Audit"]
    needs_research: false
    risk: LOW
    pitfalls: ["low-bin-counts", "h-interpolation-accuracy"]
  - name: "Enhanced Sampling Design"
    goal: "Implement importance weight tracking and stratified allocation for future campaigns"
    methods: ["Self-normalized importance-weighted histogram estimator", "Stratified sampling with Neyman-optimal allocation", "Adaptive boundary refinement (two-stage)"]
    depends_on: ["P_det Grid Quality Assessment"]
    needs_research: false
    risk: MEDIUM
    pitfalls: ["z-cutoff-unvalidated-importance-sampling", "waveform-failure-correlation"]
  - name: "Validation and Cross-Checks"
    goal: "Verify enhanced sampling produces consistent P_det grids and stable H0 posteriors"
    methods: ["Wilson score binomial confidence intervals"]
    depends_on: ["Enhanced Sampling Design"]
    needs_research: false
    risk: LOW
    pitfalls: []

critical_benchmarks:
  - quantity: "P_det monotonicity in z at fixed M"
    value: "Strictly non-increasing (within statistical noise)"
    source: "Physical expectation (farther = fainter)"
    confidence: HIGH
  - quantity: "P_det boundary conditions"
    value: "P_det -> 1 at low z, high M; P_det -> 0 at high z, low M"
    source: "Physical expectation"
    confidence: HIGH
  - quantity: "Farr criterion for global selection function"
    value: "N_eff > 4 * N_det (~80 effective injections for ~20 detections)"
    source: "Farr (2019), arXiv:1904.10879"
    confidence: HIGH
  - quantity: "Per-bin minimum for reliable P_det"
    value: "N_total >= 10 (n >= 50 preferred for <5% relative error at P_det ~ 0.5)"
    source: "Binomial statistics; Kish (1965)"
    confidence: HIGH

open_questions:
  - question: "What is the waveform failure fraction by (z, M) region?"
    priority: HIGH
    blocks_phase: "Injection Physics Audit"
  - question: "Does the z > 0.5 cutoff remain valid at h = 0.60?"
    priority: MEDIUM
    blocks_phase: "Enhanced Sampling Design"
  - question: "Is the 15x10 coarser grid sufficient for P_det accuracy?"
    priority: MEDIUM
    blocks_phase: "P_det Grid Quality Assessment"
  - question: "How sensitive is the H0 posterior to per-bin P_det noise?"
    priority: MEDIUM
    blocks_phase: "Validation and Cross-Checks"

contradictions_unresolved: []
```
