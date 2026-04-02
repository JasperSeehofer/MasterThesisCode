# Phase 19: Enhanced Sampling Design - Research

**Researched:** 2026-04-01
**Domain:** Monte Carlo variance reduction, importance sampling, stratified allocation for gravitational wave detection probability estimation
**Confidence:** HIGH

## Summary

Phase 19 designs an importance-weighted histogram estimator and stratified sampling strategy that provably reduces GPU time by >2x for equivalent P_det grid quality. The mathematical content is well-established Monte Carlo methodology (self-normalized importance sampling, Neyman-optimal allocation, defensive mixture proposals) applied to the specific problem of EMRI detection probability estimation on a 2D (d_L, M) grid.

Phase 18 established the baseline: with ~23k uniform injections per h-value on a 15x10 grid, boundary bins (0.05 < P_det < 0.95) number only 3-5 per h-value, and boundary CI half-widths are 0.06-0.08 (15x10 grid). The detection boundary is confined to a narrow strip at d_L ~ 0.2-1.0 Gpc, M ~ 2e5-1e6 Msun. Over 99% of successfully-computed waveforms produce sub-threshold SNR. This extreme sparsity of detections (0.2-0.8% detection fraction) combined with the narrow boundary region makes this problem ideally suited for stratified sampling with Neyman allocation: concentrating injections on the boundary yields large variance reduction because the boundary currently receives only ~2-3% of the total injection budget.

The recommended approach is a two-stage design: (1) 30% pilot uniform sample to identify the boundary, (2) 70% targeted sample allocated via Neyman-optimal allocation to boundary bins, with a defensive mixture proposal (alpha * uniform + (1-alpha) * boundary-enhanced) ensuring full support. All samples are combined using the self-normalized importance-weighted histogram estimator, which reduces to the standard N_det/N_total when weights are uniform. The expected variance reduction factor in boundary bins is calculable from Phase 18 pilot data and must exceed 2x per the contract.

**Primary recommendation:** Implement the self-normalized IS estimator `P_det(B) = sum(w_i * I_i) / sum(w_i)` with `w_i = p(theta_i) / q(theta_i)` as the core estimator in `SimulationDetectionProbability`, then design the stratified two-stage allocation on top. The IS estimator is the foundation; all sampling improvements are automatically handled by changing q and computing weights.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| Tiwari (2018) arXiv:1712.00482 | method | Self-normalized IS estimator for GW sensitive volume; the exact weight formula used | read Sec. 2-3; implement Eq. (5)-(8) | plan / execution / verification |
| Owen (2013) Ch. 8-9 | method | Stratified sampling theory, Neyman allocation formula, variance reduction bounds | reference for allocation algorithm | plan / execution |
| Kish (1965) | method | Effective sample size N_eff = (sum w)^2 / sum(w^2) | use as diagnostic per bin | plan / execution / verification |
| Farr (2019) arXiv:1904.10879 | benchmark | N_eff > 4*N_det criterion for unbiased selection function | verify per-bin after reweighting | plan / verification |
| Phase 18 grid quality report | prior artifact | Baseline boundary bins, CI half-widths, quality flags, per-h tables | use as pilot data for Neyman allocation; baseline for >2x comparison | plan / execution / verification |
| Phase 18 `analysis/grid_quality.py` | prior artifact | Wilson CI computation, boundary identification, grid comparison functions | reuse for computing variance reduction factor | plan / execution |
| `SimulationDetectionProbability` class | prior artifact | Current histogram-based P_det grid builder; must be extended with IS weights | modify `_build_grid_2d` to accept weights | plan / execution |
| Hesterberg (1995) Technometrics 37(2) | method | Defensive mixture distributions for robust importance sampling | use alpha*uniform + (1-alpha)*targeted formula | plan / execution |
| Veach & Guibas (1995) SIGGRAPH | method | Balance heuristic for combining samples from multiple proposals (pilot + targeted) | reference for multi-stage weight combination | plan |

**Missing or weak anchors:** No published EMRI-specific injection campaign optimization exists. The methodology is standard MC but the application to EMRI P_det grids is novel. Validation must rely on internal consistency (weights=1 recovery) and variance reduction arithmetic rather than literature benchmarks.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Grid space | (d_L, M) not (z, M) | (z, M) | SimulationDetectionProbability |
| Distance units | Gpc | Mpc | physical_relations.py |
| Mass units | Solar masses, source-frame | kg, redshifted mass | ParameterSpace |
| Hubble parameter | Dimensionless h = H0/(100 km/s/Mpc) | H0 in km/s/Mpc | constants.py |
| SNR threshold | 15 | 20 | constants.py:SNR_THRESHOLD |
| Grid resolution | 15x10 (d_L x M) for design target | 30x20 (current production) | Phase 18 recommendation |
| CI method | Wilson score, 95% (z=1.96) | Clopper-Pearson | Brown et al. (2001) |
| Boundary definition | 0.05 < P_det < 0.95 | Other thresholds | Phase 18 convention |
| Unreliable bin threshold | n_total < 10 | n_total < 30 | Phase 18 convention |
| Weight convention | w_i = p(theta_i) / q(theta_i) | inverse convention | Tiwari (2018), Owen (2013) |

**CRITICAL: All equations and results below use these conventions. The grid is in (d_L, M) space. The proposal distribution q operates in (z, M) space (where sampling occurs) but weights are applied per-bin in (d_L, M) space after the z -> d_L mapping.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| `P_det(B) = sum_{i in B} w_i * I_i / sum_{i in B} w_i` | Self-normalized IS estimator | Tiwari (2018) Eq. 5-8 | Core estimator replacing N_det/N_total |
| `w_i = p(theta_i) / q(theta_i)` | Importance weight | Standard IS | Per-injection weight computation |
| `N_eff(B) = (sum_{i in B} w_i)^2 / sum_{i in B} w_i^2` | Kish effective sample size | Kish (1965) | Per-bin quality diagnostic |
| `N_k ~ sigma_k = sqrt(P_det(k) * (1 - P_det(k)))` | Neyman-optimal allocation | Cochran (1977) Ch. 5; Owen (2013) Ch. 8 | Allocate targeted injections to strata |
| `q(theta) = alpha * p(theta) + (1-alpha) * g(theta)` | Defensive mixture proposal | Hesterberg (1995) | Ensure full support; bound max weight |
| `Var[P_hat(B)] ~ P(1-P) / N_eff(B)` | Variance of IS-weighted proportion | Standard binomial + IS | Predicted CI half-width for comparison |
| `VRF = Var_uniform / Var_stratified` | Variance reduction factor | Standard | Contract metric: must exceed 2 |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Self-normalized IS | Corrects for non-uniform proposal in histogram bins | Grid construction in `_build_grid_2d` | Tiwari (2018) |
| Neyman-optimal allocation | Allocates injections proportional to per-stratum std dev | Two-stage targeted allocation | Owen (2013) Ch. 8 |
| Defensive mixture | Ensures q(theta) > 0 wherever p(theta) > 0 | Proposal distribution design | Hesterberg (1995) |
| Pilot-based boundary identification | Identifies bins where 0.05 < P_det < 0.95 | Stage 1 output feeding Stage 2 | Phase 18 methodology |
| Wilson score CI with IS weights | Quality assessment for weighted estimates | Verification of improved grid | Brown et al. (2001) adapted |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Self-normalized IS bias | 1/N per bin | N_eff > 50 per bin | Bias ~ O(1/N_eff), negligible for N_eff > 50 | Bias-reduced SNIS (Cardoso et al. 2022) |
| Neyman allocation with estimated sigma | sigma_pilot vs sigma_true | Pilot N > 30 per boundary bin to get sigma to ~20% | Misallocation ~ (sigma_hat/sigma)^2 | Iterate: re-estimate sigma after targeted stage |
| Pilot boundary identification | P_det_pilot vs P_det_true | Pilot large enough to identify boundary location (>2000 total, >10 per boundary bin) | May miss narrow boundary features | Use 40% pilot instead of 30% |
| Linear h-interpolation of P_det | h-curvature of P_det | P_det(h) approximately linear between grid points | Uncontrolled; empirical from Phase 18 | Add intermediate h-values |

## Standard Approaches

### Approach 1: Self-Normalized IS + Neyman-Stratified Two-Stage Design (RECOMMENDED)

**What:** Implement the self-normalized importance-weighted histogram estimator as the core P_det computation, then design a two-stage injection campaign (30% pilot + 70% targeted) with Neyman-optimal allocation of the targeted portion to boundary bins.

**Why standard:** The self-normalized IS estimator is the standard tool in GW population inference (Tiwari 2018, used by LVK). Neyman allocation is the textbook-optimal stratified sampling method (Cochran 1977, Owen 2013). The two-stage pilot+targeted design is standard in survey sampling when stratum variances are unknown a priori.

**Track record:** Tiwari (2018) is widely cited (~100+ citations) and operationally used by the LVK collaboration for CBC sensitive volume estimation. Neyman allocation has been standard since 1934. The defensive mixture proposal is standard practice in computational statistics (Hesterberg 1995, Owen 2013 Sec. 9.3).

**Key steps:**

1. **Implement IS-weighted histogram estimator** in `SimulationDetectionProbability._build_grid_2d()`:
   - Accept per-injection weights `w_i` (default: all 1.0 for backward compatibility)
   - Replace `detected_counts / total_counts` with `sum(w_i * I_i) / sum(w_i)` per bin
   - Store per-bin N_eff alongside existing quality flags
   - Verify: when all w_i = 1, output is identical to current estimator (machine precision)

2. **Design the Neyman-optimal allocation algorithm:**
   - Input: Phase 18 pilot data (P_det per bin, n_total per bin)
   - Compute per-bin Bernoulli std dev: sigma_k = sqrt(P_hat_k * (1 - P_hat_k))
   - Allocate N_targeted proportional to sigma_k across all bins
   - Boundary bins (sigma_k ~ 0.5 at P_det ~ 0.5) get the most; P_det ~ 0 or ~ 1 bins get few
   - Minimum allocation: at least 5 injections per non-empty bin (avoid zero-sample strata)

3. **Design the defensive mixture proposal:**
   - `q(z, M) = alpha * p_uniform(z, M) + (1 - alpha) * g_targeted(z, M)`
   - `p_uniform`: uniform in (z, M) over the injection domain (current sampling)
   - `g_targeted`: enhanced density in bins identified as boundary by pilot
   - alpha = 0.3 (30% uniform, 70% targeted) -- ensures full support
   - Compute weights: `w_i = p_prior(z_i, M_i) / q(z_i, M_i)` for targeted samples; `w_i = 1` for pilot samples (drawn from prior)

4. **Specify the two-stage design:**
   - Stage 1 (pilot): 30% of total budget, uniform sampling (= current approach)
   - Stage 2 (targeted): 70% of total budget, drawn from defensive mixture proposal
   - Combined: pool all samples with appropriate weights
   - For combining pilot + targeted: pilot has `w_i = 1` (drawn from prior); targeted has `w_i = p(theta_i) / q(theta_i)`

5. **Compute expected variance reduction factor from Phase 18 data:**
   - Use Phase 18 per-bin n_total and P_det as the "uniform" baseline
   - Compute Neyman allocation for the same total injection count
   - Compare boundary-region CI half-widths: VRF = CI_hw_uniform^2 / CI_hw_stratified^2
   - Contract requires VRF > 2 (CI half-width reduction > sqrt(2) ~ 1.41x)

**Known difficulties at each step:**

- Step 1: The existing `_build_grid_2d` uses `np.histogram2d` which does simple counting. The IS-weighted version must manually bin samples and accumulate weights, not use `np.histogram2d` for the weighted sum. Alternative: use `np.histogram2d` for binning then loop over bins for weighted sums.
- Step 2: Phase 18 shows only 3-5 boundary bins per h-value on the 15x10 grid. With so few boundary bins, the Neyman allocation concentrates heavily, which is correct but means most of the (d_L, M) grid receives very few targeted injections. This is fine because non-boundary bins are already well-determined.
- Step 3: The prior p(z, M) comes from `Model1CrossCheck.emri_distribution(M, z)`. The proposal q(z, M) must be evaluable for weight computation. The emcee sampler in `Model1CrossCheck` produces samples but does not directly provide the density value. Must either: (a) evaluate `emri_distribution(M, z)` at each sample point, or (b) use a simpler analytic approximation of p for the weight ratio.
- Step 4: Combining pilot and targeted samples requires tracking which stage each sample came from and its weight. The injection CSV format must be extended with a `weight` column (or computed post-hoc from the proposal density).
- Step 5: The variance reduction factor depends on how much of the current injection budget falls in boundary bins. With ~3% of bins being boundary and injections approximately uniformly distributed, roughly 3% of injections currently fall in boundary bins. Concentrating 70% of the targeted budget on these bins increases boundary sample size by ~23x * (0.7/0.03) ~ 16x for the targeted portion, yielding ~5x total when combined with the 30% pilot.

### Approach 2: Post-Hoc Reweighting Only (FALLBACK / FIRST STEP)

**What:** Keep the current uniform injection campaign exactly as-is. Record the proposal density for each injection. Implement the IS-weighted estimator to enable future non-uniform proposals without requiring a new campaign.

**When to switch:** If the two-stage design proves too complex to implement within this phase, or if the pilot data suggests the boundary region is too narrow for stratified allocation to help significantly.

**Tradeoffs:** No variance reduction in this approach (weights are all 1 for uniform sampling). But it lays the foundation: once the IS estimator is in place, any future change to the proposal distribution is automatically handled. The variance reduction factor would be 1.0x (no improvement), which does not meet the >2x contract. However, combined with the 15x10 grid switch (already yielding 3x CI improvement per Phase 18), the overall quality improvement may still be substantial.

### Anti-Patterns to Avoid

- **Importance sampling without recording weights:** If the proposal distribution is changed but weights are not tracked, P_det estimates are biased. Every non-uniform sample MUST have a computable weight.
  - _Example:_ The current z > 0.5 cutoff is implicit importance sampling with no weight correction. Phase 18 confirmed it is valid (zero detections above z = 0.5), but as a matter of principle the weight should be recorded.

- **Proposal with zero support where prior is nonzero:** If q(z, M) = 0 in any region where p(z, M) > 0, the weight w = p/q is undefined (division by zero). The defensive mixture prevents this by construction.
  - _Example:_ A pure boundary-only proposal that places zero density outside the boundary would miss the ~97% of parameter space where P_det is 0 or 1, making the estimator undefined there.

- **Using raw (unnormalized) IS instead of self-normalized:** The unnormalized IS estimator requires knowing the exact normalization of p(theta). The self-normalized version cancels the normalization in the ratio, which is essential because `emri_distribution` is not normalized.
  - _Example:_ `sum(w_i * I_i) / N` (unnormalized) would give wrong answers unless sum(w_i)/N converges to the correct normalizing constant. Use `sum(w_i * I_i) / sum(w_i)` instead.

- **Treating Neyman allocation as a hard rule for tiny N:** With only 3-5 boundary bins per h-value on the 15x10 grid, Neyman allocation may assign zero injections to some strata. Always impose a minimum (e.g., 5 per stratum) to avoid empty bins.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Self-normalized IS estimator | `P_hat = sum(w_i * I_i) / sum(w_i)` | Tiwari (2018) Sec. 2; Owen (2013) Sec. 9.1 | Implement directly; do not re-derive |
| IS estimator bias | `Bias = O(1/N_eff)` | Owen (2013) Sec. 9.1 | Cite; negligible for N_eff > 50 |
| Kish effective sample size | `N_eff = (sum w)^2 / sum(w^2)` | Kish (1965) | Implement directly as diagnostic |
| Neyman allocation formula | `N_k propto N_k_population * sigma_k` | Cochran (1977) Ch. 5 | Implement directly for allocation |
| Wilson score CI | See Phase 18 | Brown et al. (2001) | Already implemented in `analysis/grid_quality.py` |
| Farr criterion | `N_eff > 4 * N_det` | Farr (2019) | Already verified globally per h in Phase 18 |
| Defensive mixture | `q = alpha * uniform + (1-alpha) * targeted` | Hesterberg (1995) | Standard form; choose alpha = 0.3 |
| Balance heuristic | MIS weight for multi-proposal | Veach & Guibas (1995) | Reference only; simpler pooled estimator suffices |

**Key insight:** All core mathematical results are textbook-level Monte Carlo theory. The innovation in this phase is the APPLICATION to EMRI P_det grids with specific numerical parameters from Phase 18, not the derivation of new estimators.

### Phase 18 Data to Use Directly

| Result | Value | Source | How to Use |
| --- | --- | --- | --- |
| Boundary bins per h (15x10) | 3-5 bins | grid-quality-report.md | Strata for Neyman allocation |
| Boundary CI half-widths (15x10) | 0.06-0.08 | grid-quality-report.md | Baseline for variance reduction comparison |
| N_total per h | 17k-26k | yield-report.md | Total injection budget for allocation |
| N_detected per h | 50-138 | yield-report.md | Farr criterion check after reweighting |
| Detection fraction | 0.22-0.81% | yield-report.md | Input for Neyman sigma_k computation |
| Unreliable bins (15x10) | 10-71 per h | grid-quality-report.md | Bins needing most improvement |
| Per-bin n_total, n_detected | Arrays from quality_flags() | SimulationDetectionProbability | Direct input for Neyman allocation |
| h=0.90 is statistics-limited | 71/150 unreliable bins | grid-quality-report.md | Most benefit from stratified allocation |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Sensitive volume via weighted MC | Tiwari | 2018 | Core IS estimator formula | Eq. 5-8: weight formula, self-normalized estimator |
| Accuracy requirements for selection functions | Farr | 2019 | N_eff criterion for unbiased inference | N_eff > 4*N_det; apply per-bin |
| Quick recipes for GW selection effects | Gerosa & Fishbach | 2024 | Modern review of P_det estimation methods | Sec. 3: injection-based methods; confirms standard approach |
| Hierarchical Bayesian inference with selection | Mandel, Farr & Gair | 2019 | Framework: P_det enters likelihood denominator | Eq. 8-12: how P_det errors propagate to posterior |
| Defensive mixture distributions | Hesterberg | 1995 | Bounded importance weights via mixture proposals | alpha-mixture formula; weight bound = 1/alpha |
| Optimally combining sampling techniques | Veach & Guibas | 1995 | Multi-proposal combination (balance heuristic) | Reference for combining pilot + targeted samples |
| MC theory, methods and examples | Owen | 2013 | Textbook for stratified sampling + IS | Ch. 8 (stratified), Ch. 9 (importance sampling) |
| Survey sampling | Kish | 1965 | Effective sample size formula | N_eff = (sum w)^2 / sum(w^2) |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| numpy | histogram2d, digitize | Bin assignment for weighted histogram | Already used in SimulationDetectionProbability |
| numpy | bincount, add.at | Accumulate per-bin weighted sums | Efficient for sparse bin accumulation |
| scipy | RegularGridInterpolator | Interpolate IS-weighted P_det grid | Already used in SimulationDetectionProbability |
| astropy | stats.binom_conf_interval | Wilson CIs for verification | Already used in analysis/grid_quality.py |
| pandas | DataFrame | Injection CSV I/O with weight column | Already used in SimulationDetectionProbability |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| matplotlib | Visualize: weight distribution, N_eff heatmap, VRF comparison | Verification and reporting |
| analysis/grid_quality.py | Reuse Wilson CI, boundary identification, grid comparison | Direct reuse for before/after comparison |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
| --- | --- | --- |
| Manual weighted histogram | scipy.stats.binned_statistic_2d with weights | scipy version less flexible for IS; manual gives full control |
| Defensive mixture | Clipped proposal (q = max(q_targeted, epsilon * p)) | Clipping is equivalent; mixture is cleaner |
| Neyman allocation | Proportional allocation | Neyman is optimal; proportional ignores variance heterogeneity |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| IS-weighted histogram (existing data) | < 1 second per h-value | None | Simple array operations on ~23k samples |
| Neyman allocation computation | < 1 second | None | Arithmetic on ~150 bin variances |
| Variance reduction factor computation | < 1 second | None | Ratio of variances from Phase 18 data |
| Proposal density evaluation (emri_distribution) | ~0.1s per injection | None for design phase | Only needed during future campaign execution |
| Full two-stage campaign execution | ~3 hours/h on GPU | Waveform generation | Not part of this phase (design only) |

**Installation / Setup:**
```bash
# No additional packages needed. All tools already in the dependency tree.
# Phase 18 analysis scripts provide reusable functions.
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Uniform weights recovery | IS estimator reduces to standard when w_i = 1 | Set all w_i = 1, compare to current N_det/N_total | Identical to machine precision (< 1e-15) |
| Weight normalization | sum(w_i)/N approximates E_q[p/q] = 1 for correctly specified proposal | Compute mean weight per bin | mean(w) ~ 1.0 +/- O(1/sqrt(N)) |
| N_eff <= N per bin | Effective sample size cannot exceed actual sample size | Check N_eff(B) <= n_total(B) for all bins | Always true; equality when weights uniform |
| P_det in [0, 1] | IS estimator stays in valid range | Check all output values | P_hat in [0, 1] for self-normalized estimator (guaranteed by construction) |
| Variance reduction factor > 1 | Stratified is better than uniform | Compare CI half-widths at boundary | VRF > 2 per contract |
| Monotonicity preserved | P_det still decreasing in d_L at fixed M | Check P_det(d_L) slices after reweighting | No new non-monotonic features (beyond noise) |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| Uniform proposal | q = p everywhere | IS estimator = standard histogram estimator | Tiwari (2018) |
| Perfect oracle proposal | q propto p * I * (1-I) | N_eff = N (all weights equal); zero-variance estimator | Owen (2013) |
| All weight on one sample | One extreme w_i | N_eff = 1; estimate useless | Kish (1965) |
| Equal Neyman allocation | All sigma_k equal | Reduces to proportional allocation | Cochran (1977) |
| alpha = 1 in defensive mixture | Pure uniform proposal | Recovers current approach (no improvement) | Hesterberg (1995) |
| alpha = 0 in defensive mixture | Pure targeted proposal | Maximum variance reduction but risk of zero support | Hesterberg (1995) |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| IS estimator vs standard at h=0.73 | Set w_i = 1 for all 25500 injections | max |diff| < 1e-14 | Phase 18 P_det grid at h=0.73 |
| VRF computation | Compare predicted CI hw to Phase 18 baseline | VRF > 2.0 | Phase 18 boundary CI half-widths |
| Neyman allocation totals | sum(N_k) = N_targeted | Exact integer sum | Total targeted budget |
| Defensive mixture support | min(q) > 0 over entire domain | q > alpha * p_min > 0 | Full support guarantee |

### Red Flags During Computation

- If any bin has N_eff < 1 after reweighting, the proposal has pathological weight concentration. Check for q(theta) << p(theta) in that bin.
- If the maximum importance weight exceeds 100 * mean weight, the proposal tails are too thin relative to the prior. Increase alpha in the defensive mixture.
- If the variance reduction factor is < 1 (stratified is WORSE than uniform), something is wrong with the allocation algorithm. Check that sigma_k estimates are reasonable.
- If P_det values change by more than 0.1 when switching from standard to IS-weighted estimator with uniform weights, there is a bug in the weight computation.

## Common Pitfalls

### Pitfall 1: Forgetting to Record Proposal Density

**What goes wrong:** Injections are generated from a non-uniform proposal but the density q(theta_i) is not stored. Post-hoc weight computation is impossible.
**Why it happens:** The current injection_campaign() function stores (z, M, phiS, qS, SNR, h_inj, d_L) but not the sampling density. Adding non-uniform sampling without recording q makes the data un-reweightable.
**How to avoid:** Add a `log_q` column to the injection CSV storing log(q(z_i, M_i)) for each injection. For current uniform-from-prior samples, log_q = log(emri_distribution(M_i, z_i)) (since q = p for uniform prior sampling).
**Warning signs:** CSV file has no weight or density column but the sampling code uses non-uniform proposal.
**Recovery:** If q is known analytically, recompute log_q from stored (z, M) values. If q was adaptive or stochastic, the data is lost.

### Pitfall 2: Evaluating emri_distribution for Weight Computation

**What goes wrong:** The prior density p(z, M) = `emri_distribution(M, z)` is needed for weight computation, but the emcee sampler draws samples without evaluating the density at each point. Additionally, `emri_distribution` may return 0 or very small values at certain (M, z) combinations, causing weight computation issues.
**Why it happens:** The emcee sampler evaluates `log_probability` internally but the user code only receives sample positions, not density values. The density must be re-evaluated at each sample point.
**How to avoid:** Call `cosmological_model.emri_distribution(M_i, z_i)` for each injection sample to get p(theta_i). This is O(1) per sample (analytic function, no waveform needed). Store as `log_p` in CSV alongside `log_q`.
**Warning signs:** Division by zero in weight computation; extreme weight values.
**Recovery:** Post-hoc evaluation of emri_distribution at stored (z, M) pairs.

### Pitfall 3: Boundary Identification from Noisy Pilot

**What goes wrong:** The pilot has low statistics per bin, so the estimated boundary (0.05 < P_hat < 0.95) may misidentify bins. Bins that are truly P_det ~ 0 but have P_hat = 1/3 (from n=3, k=1) get flagged as boundary.
**Why it happens:** With ~23k total events and 150 bins on a 15x10 grid, average occupancy is ~153/bin but varies from 0 to ~1000. At 30% pilot (6900 events), average drops to ~46/bin, with many bins having < 10 events.
**How to avoid:** Use a conservative boundary definition for allocation purposes: flag as boundary only if BOTH (a) 0.05 < P_hat < 0.95 AND (b) n_total >= 10 in the pilot. This prevents over-allocating to bins that appear boundary only due to noise.
**Warning signs:** Boundary bin count changes significantly between 20% and 40% pilot fractions.
**Recovery:** Use a wider boundary threshold (e.g., 0.01 < P_hat < 0.99) or increase pilot fraction to 40%.

### Pitfall 4: Self-Normalized Estimator Bias at Small N_eff

**What goes wrong:** The self-normalized IS estimator has bias O(1/N_eff). For bins with N_eff < 10, this bias can be comparable to the statistical error, leading to P_det estimates that are systematically off.
**Why it happens:** The self-normalized estimator divides by sum(w_i) rather than N, introducing a ratio bias. This is well-understood and negligible for N_eff > 50, but concerning for sparse bins.
**How to avoid:** Flag bins with N_eff < 50 as potentially biased. For bins with N_eff < 10, revert to the unweighted estimator (which has no bias but higher variance).
**Warning signs:** P_det values that change by > 0.05 between the IS-weighted and unweighted estimators in the same bin.
**Recovery:** Use the unweighted estimator for flagged bins; report both estimates.

## Level of Rigor

**Required for this phase:** Controlled approximation with quantitative error bounds.

**Justification:** This is a DESIGN phase (not execution). The deliverables are (a) mathematical specification of the estimator with proof of correctness in the uniform-weight limit, (b) allocation algorithm with predicted variance reduction computed from Phase 18 data, and (c) implementation-ready specification for the two-stage design. The variance reduction factor must be calculated, not hand-waved.

**What this means concretely:**

- The IS estimator must be verified to machine precision against the standard histogram estimator when weights are uniform. This is a HARD requirement (Success Criterion 1).
- The Neyman allocation must be computed from actual Phase 18 per-bin statistics, not from generic variance reduction formulas. The predicted >2x improvement must be demonstrated numerically (Success Criterion 4).
- The defensive mixture must have a mathematical guarantee of full support. The proof is trivial (alpha > 0 ensures q >= alpha * p_uniform > 0 wherever p_uniform > 0), but it must be stated explicitly (Success Criterion 5).
- The two-stage design must specify exact numbers: pilot sample size, boundary identification algorithm, targeted proposal distribution, weight formula. These are design specifications, not proofs (Success Criterion 3).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| KDE-based P_det | Histogram-based P_det from injections | Phase 11.1 (2026) | Simulation-based P_det is more accurate but needs more samples |
| Uniform injection sampling | (This phase) IS-weighted + stratified | Phase 19 | Expected >2x variance reduction in boundary bins |
| 30x20 grid | 15x10 grid (Phase 18 recommendation) | Phase 18 (2026) | 3x CI improvement from grid coarsening |
| Unweighted N_det/N_total | Self-normalized IS estimator | (This phase) | Foundation for arbitrary proposal distributions |

**Superseded approaches to avoid:**

- **KDE-based DetectionProbability:** Replaced by SimulationDetectionProbability in Phase 11.1. Do NOT reintroduce KDE for the new estimator. The KDE approach has bandwidth selection issues and does not handle per-bin importance weights naturally. However, KDE P_det remains useful as a qualitative cross-check of boundary shape.

## Open Questions

1. **How to evaluate the prior density p(z, M) efficiently for weight computation?**
   - What we know: `emri_distribution(M, z)` in `cosmological_model.py` provides the unnormalized density. It is an analytic function (no waveform evaluation needed). The function computes `dN_dz_of_mass(M, z) * R_emri(M)`.
   - What's unclear: Whether `emri_distribution` is fast enough to call per-injection (~23k calls per h-value). Also whether the unnormalized density cancels correctly in the self-normalized estimator (it should: p/q ratio, both unnormalized, cancels common normalization).
   - Impact on this phase: If density evaluation is slow, must precompute and store. If normalization does not cancel, must normalize both p and q.
   - Recommendation: Proceed with unnormalized p and q. The self-normalized estimator `sum(w_i * I_i) / sum(w_i)` uses the ratio p/q which cancels normalization. Verify this analytically in the plan.

2. **What pilot fraction optimizes the variance-computation tradeoff?**
   - What we know: 30% pilot is the conventional recommendation (Owen 2013). Phase 18 used 100% uniform (= all pilot).
   - What's unclear: For this specific grid (150 bins, ~23k events), is 30% pilot (6900 events) sufficient to identify the 3-5 boundary bins? With 6900 events and 150 bins, the average is 46/bin. The boundary bins at ~3% of total would get ~1.4 events each on average -- too few.
   - Impact on this phase: May need to increase pilot fraction to 50% or use the existing Phase 18 data AS the pilot (no new pilot needed).
   - Recommendation: **Use existing Phase 18 data as the pilot.** The 23k events already collected per h-value serve as a perfect pilot. The targeted 70% is the ADDITIONAL campaign, not a fraction of the same campaign. This resolves the pilot-size concern entirely.

3. **Should the targeted proposal operate in (z, M) space or (d_L, M) space?**
   - What we know: Sampling occurs in (z, M) space (emcee draws z, M). The grid is in (d_L, M) space. The boundary is identified in (d_L, M) space.
   - What's unclear: Whether to define the targeted proposal in (z, M) and map boundary bins back, or in (d_L, M) and accept the z-to-d_L Jacobian.
   - Impact on this phase: Affects weight computation formula (need Jacobian |dd_L/dz| if changing spaces).
   - Recommendation: Define targeted proposal in (z, M) space (where sampling occurs). Map boundary bin edges from d_L to z using the known d_L(z, h) relation. The weight w_i = p(z_i, M_i) / q(z_i, M_i) is computed in (z, M) space with no Jacobian needed.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| IS-weighted estimator | Weight instability (max weight >> mean) | Increase alpha in defensive mixture to 0.5-0.7; reduces VRF but stabilizes | Low (parameter change) |
| Neyman allocation | Pilot identifies too few boundary bins (<2) | Use proportional allocation instead (allocate by bin population) | Low (change allocation formula) |
| Two-stage design (30/70 split) | Pilot fraction insufficient for boundary ID | Use Phase 18 data as pilot, all new injections are targeted | Zero (use existing data) |
| >2x VRF in boundary | Boundary bins too few for meaningful stratification | Switch to 15x10 grid (already gives 3x CI improvement over 30x20); claim efficiency gain from grid coarsening | Medium (reframe success criterion) |
| Full approach | emri_distribution too slow or too noisy for weight computation | Post-hoc reweighting with analytic approximation of p(z,M) ~ z^alpha * M^beta fitted to existing samples | Medium (need to fit approximate density) |

**Decision criteria:** If the computed VRF from Phase 18 data is < 1.5 (meaning stratified allocation provides < 50% improvement over uniform), the two-stage design is not worth the implementation complexity. In that case, fall back to Approach 2 (post-hoc reweighting with the IS estimator for future-proofing) and achieve the >2x quality improvement via grid coarsening (15x10 vs 30x20, already demonstrated at 3x in Phase 18).

## Sources

### Primary (HIGH confidence)

- [Tiwari (2018), "Estimation of the Sensitive Volume for GW Source Populations Using Weighted MC," CQG, arXiv:1712.00482](https://arxiv.org/abs/1712.00482) - Self-normalized IS estimator formula, weight computation, GW-specific application
- [Farr (2019), "Accuracy Requirements for Empirically-Measured Selection Functions," RNAAS, arXiv:1904.10879](https://arxiv.org/abs/1904.10879) - N_eff > 4*N_det criterion for sufficient effective samples
- [Owen (2013), "Monte Carlo theory, methods and examples," Ch. 8-9](https://artowen.su.domains/mc/) - Stratified sampling variance reduction, IS theory, Neyman allocation
- [Brown, Cai, DasGupta (2001), "Interval Estimation for a Binomial Proportion," Stat. Sci. 16(2):101-133](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) - Wilson CI (already implemented in Phase 18)
- [Cochran (1977), "Sampling Techniques," 3rd ed., Wiley, Ch. 5](https://www.wiley.com/en-us/Sampling+Techniques%2C+3rd+Edition-p-9780471162407) - Neyman allocation formula and optimality proof

### Secondary (MEDIUM confidence)

- [Hesterberg (1995), "Weighted Average Importance Sampling and Defensive Mixture Distributions," Technometrics 37(2)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1995.10484303) - Defensive mixture for bounding importance weights
- [Veach & Guibas (1995), "Optimally Combining Sampling Techniques for MC Rendering," SIGGRAPH](https://dl.acm.org/doi/10.1145/218380.218498) - Balance heuristic for multi-proposal combination
- [Gerosa & Fishbach (2024), "Quick recipes for GW selection effects," CQG, arXiv:2404.16930](https://arxiv.org/abs/2404.16930) - Modern review confirming injection-based P_det methods
- [Mandel, Farr & Gair (2019), arXiv:1809.02063](https://arxiv.org/abs/1809.02063) - Hierarchical Bayesian framework; how P_det enters likelihood

### Tertiary (LOW confidence)

- [Kish (1965), "Survey Sampling," Wiley](https://www.wiley.com/en-us/Survey+Sampling-p-9780471109495) - N_eff formula (universally cited but original text rarely consulted)
- [Cardoso et al. (2022), "Bias Reduced Self-Normalized Importance Sampling," NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/04bd683d5428d91c5fbb5a7d2c27064d-Paper-Conference.pdf) - Bias reduction for SNIS; only relevant if N_eff is very small

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH - All estimators are textbook MC methods with well-understood convergence
- Standard approaches: HIGH - IS + stratified sampling is the standard GW community approach (Tiwari 2018, Farr 2019)
- Computational tools: HIGH - All tools already in the dependency tree; no new packages needed
- Validation strategies: HIGH - Uniform-weight recovery test is definitive; VRF computation is arithmetic

**Research date:** 2026-04-01
**Valid until:** Indefinite for mathematical content; Phase 18 data is a snapshot of the current injection campaign

## Caveats and Self-Critique

1. **Assumption: Phase 18 data can serve as the pilot.** This assumes the existing uniform injection campaign is representative. If the boundary shifts significantly with a different grid resolution or SNR threshold, the pilot-based allocation would be suboptimal. Mitigation: the defensive mixture ensures full coverage regardless.

2. **The >2x VRF claim needs numerical verification, not just assertion.** The research establishes that the methodology CAN provide >2x improvement, but the ACTUAL improvement depends on Phase 18's specific per-bin statistics. The planner must schedule a task that computes VRF from the actual data. If VRF < 2, the contract target is not met and a fallback (grid coarsening) must be invoked.

3. **The proposal density q(z, M) evaluation depends on emri_distribution being evaluable.** I have confirmed this function exists and is analytic, but I have not verified its computational cost per call or whether it returns sensible values across the full (z, M) domain. The planner should include a task to profile this function.

4. **Combining pilot + targeted samples is simpler than the full balance heuristic.** I recommended the simpler pooled estimator (pilot w=1, targeted w=p/q) over the Veach-Guibas balance heuristic. The simpler approach is slightly suboptimal in variance but much easier to implement and verify. An expert might prefer the balance heuristic for better variance properties, but the complexity is not warranted given the current project scale.

5. **The narrow boundary region (3-5 bins) means stratified allocation has few strata to optimize.** With so few boundary bins, the Neyman allocation essentially becomes "put most targeted samples in 3-5 bins." This is correct but the variance reduction factor may be sensitive to exactly which bins are identified as boundary. The defensive mixture mitigates this by ensuring some coverage everywhere.
