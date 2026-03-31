# Prior Work

**Project:** Injection Campaign Physics Audit and Enhanced Sampling for EMRI Dark Siren P_det
**Physics Domain:** Gravitational wave selection effects, dark siren cosmology, Monte Carlo methods
**Researched:** 2026-03-31

## Theoretical Framework

### Governing Theory

| Framework | Scope | Key Equations | Regime of Validity |
|-----------|-------|---------------|-------------------|
| Hierarchical Bayesian inference with selection effects | Population-level inference from GW detections subject to Malmquist bias | Mandel, Farr & Gair (2019) Eq. 2-8: likelihood with P_det normalization | Applies whenever detected sample is a biased subset of the population |
| Dark siren cosmology | H0 inference from GW events without EM counterparts, using galaxy catalogs | Likelihood marginalized over galaxy catalog with P_det selection correction | Requires galaxy catalog completeness or completeness correction; z < 1 for EMRI dark sirens |
| Injection-based selection function estimation | Empirical P_det from simulated signal injections | P_det(theta) = N_found(theta) / N_total(theta) in each bin | Requires sufficient injection density per bin; subject to Poisson noise |

### The Selection Effects Problem

In dark siren cosmology, the observed sample of GW events is subject to selection bias: louder, closer, more favorably oriented events are preferentially detected. The population-level likelihood for cosmological parameters (specifically H0) must account for this via the detection probability P_det. The standard hierarchical Bayesian framework (Mandel, Farr & Gair, arXiv:1809.02063; Loredo 2004) gives:

```
L(data | Lambda) ~ prod_i p(d_i | theta_i, Lambda) / alpha(Lambda)
```

where alpha(Lambda) = integral p(theta | Lambda) P_det(theta) d(theta) is the fraction of the population that is detectable, and Lambda includes H0. Getting alpha(Lambda) wrong biases H0.

**Confidence: HIGH** -- This framework is the established standard in the field, used by LVK in all population and cosmology papers since O2.

### Mathematical Prerequisites

| Topic | Why Needed | Key Results | References |
|-------|-----------|-------------|------------|
| Inhomogeneous Poisson processes | Selection effects enter as rate modulation | Detected events follow thinned Poisson process with rate R * P_det | Mandel, Farr & Gair (2019), arXiv:1809.02063 |
| Monte Carlo integration with importance weights | Reweighting injections from proposal to target distribution | VT = sum w_i * V_i where w_i = p_target(theta_i) / p_draw(theta_i) | Tiwari (2018), arXiv:1712.00482 |
| Marcum Q-function | Semi-analytic P_det for matched-filter SNR with noise fluctuations | P_det(rho_opt) = Q_1(rho_opt, rho_th) replaces step function at threshold | Gerosa & Bellotti (2024), arXiv:2404.16930 |
| Binomial statistics in histogram bins | Uncertainty quantification for P_det = N_det / N_total per bin | Variance = P_det * (1 - P_det) / N_total; Wilson interval for small N | Standard Monte Carlo textbooks |

### Unit System and Conventions

- **Masses:** Solar masses (M_sun). Code uses M = total MBH mass in source frame; M_z = M * (1+z) is the redshifted (detector-frame) mass.
- **Distances:** Mpc for luminosity distance d_L.
- **SNR:** Dimensionless optimal matched-filter SNR. Detection threshold rho_th = 20 (current code).
- **Hubble constant:** h = H0 / (100 km/s/Mpc), with h = 0.73 as fiducial.
- **Redshift bins:** Linear in z (current code). Mass bins: logarithmic / geometric spacing (current code).

### Known Limiting Cases

| Limit | Parameter Regime | Expected Behavior | Reference |
|-------|-----------------|-------------------|-----------|
| Low z, high M | z -> 0, M >> 10^5 M_sun | P_det -> 1 (all events detected) | Physical: SNR ~ 1/d_L -> inf |
| High z | z >> 0.5 for EMRIs | P_det -> 0 (below threshold) | Confirmed: all detections at z < 0.18 in initial campaign |
| SNR >> threshold | rho_opt >> rho_th | P_det -> 1 | Marcum Q_1(rho_opt, rho_th) -> 1 |
| SNR << threshold | rho_opt << rho_th | P_det -> 0 | Marcum Q_1(rho_opt, rho_th) -> 0 |
| N_total -> inf | Many injections per bin | P_det converges; Poisson noise -> 0 | Law of large numbers |

## Key Parameters and Constants

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| SNR threshold | 20 | constants.py | Conservative; LVK uses ~8-12 for ground-based |
| z_cut for injection campaign | 0.5 | main.py:456 | All detections at z < 0.18 in initial campaign |
| P_det grid: z bins | 30 | simulation_detection_probability.py:27 | Linear spacing |
| P_det grid: M bins | 20 | simulation_detection_probability.py:28 | Log (geometric) spacing |
| Fiducial h | 0.73 | constants.py | WMAP-era; Planck 2018 gives 0.6736 |
| Waveform failure rate | ~30-50% | Operational experience | few/fastlisaresponse hangs on edge cases |

## Established Results to Build On

### Result 1: LVK Found-Injections Monte Carlo (Standard Method)

**Statement:** The selection integral alpha(Lambda) is estimated by drawing N_inj signals from a proposal distribution p_draw(theta), computing SNR for each, and forming the importance-weighted Monte Carlo sum:

```
alpha(Lambda) = (1/N_inj) * sum_{i=1}^{N_inj} [found_i] * p(theta_i|Lambda) / p_draw(theta_i)
```

where [found_i] = 1 if SNR >= threshold. Crucially, the LVK does NOT bin P_det into a grid. The selection integral is computed directly as this MC sum over all injections for each Lambda value, avoiding all gridding and interpolation issues.

**Status:** Established standard. Used in GWTC-2 (arXiv:2010.14533), GWTC-3 (arXiv:2111.03634), GWTC-4 (arXiv:2508.18083).

**Reference:** Tiwari (2018), arXiv:1712.00482; Abbott et al. (2021), arXiv:2010.14533 Sec. IV.

**Relevance to project:** The current code builds a P_det grid and interpolates via `SimulationDetectionProbability`. The LVK approach stores all (z, M, SNR) triples and sums at evaluation time. Cost is O(N_inj * N_h_values) per likelihood evaluation -- cheap for N_inj ~ 10^4. **Recommended as primary approach for alpha(Lambda); keep grid for visualization only.**

**Confidence: HIGH**

### Result 2: Importance Sampling for VT (Tiwari 2018)

**Statement:** A single injection set from a broad proposal p_draw(theta) can estimate VT for any population model via reweighting:

```
VT(Lambda) = V_max * (sum_{i: found} w_i) / (sum_{i: all} w_i)
```

where w_i = p(theta_i|Lambda) / p_draw(theta_i). Self-normalized estimator, unbiased to O(1/N_inj).

Key requirement: **common support** -- p_draw > 0 wherever p(theta|Lambda) > 0. Efficiency depends on overlap: poor overlap gives high-variance weights and low N_eff.

**Reference:** Tiwari (2018), arXiv:1712.00482.

**Relevance to project:** Current campaign draws from the astrophysical population, so w_i = 1 (proposal = target). This is correct but wasteful: ~80%+ of GPU-evaluated injections at 0.18 < z < 0.5 produce SNR << 20 and contribute nothing. The z > 0.5 cut is equivalent to truncating the proposal; since P_det = 0 in the truncated region, no bias is introduced. If the proposal is changed to concentrate near the detection boundary, importance weights w_i must be tracked and applied.

**Confidence: HIGH**

### Result 3: Semi-Analytic P_det via Marcum Q-Function

**Statement:** For a matched-filter search, the detection probability at optimal SNR rho_opt is:

```
P_det(rho_opt) = Q_1(rho_opt, rho_th)
```

where Q_1 is the first-order Marcum Q-function. The measured SNR follows a Rice distribution centered on rho_opt due to noise fluctuations. The step-function approximation is biased asymmetrically: underestimates P_det for rho_opt slightly below threshold (noise can push above) and overestimates slightly above (noise can push below).

**Reference:** Gerosa & Bellotti (2024), arXiv:2404.16930, Eqs. 3-7. Package: `pip install marcumq`.

**Relevance to project:** Current code uses sharp step function at SNR = 20. For threshold = 20, the transition region is narrow (~18-22), so the step function is a reasonable first approximation. The Marcum Q-function is a low-effort refinement: replace binary detected/not-detected with continuous p_det per injection.

**Confidence: HIGH**

### Result 4: Effective Sample Size Criterion (Farr 2019)

**Statement:** For reliable selection-effect correction, the effective sample size must satisfy:

```
N_eff = (sum w_i)^2 / (sum w_i^2) > 4 * N_det
```

where N_det is the number of observed GW events. Below this threshold, MC noise in alpha(Lambda) biases population inference.

**Reference:** Farr (2019), arXiv:1904.10879.

**Relevance to project:** With uniform proposal (w_i = 1), N_eff = N_found. Current campaign: ~24 detections from ~69500 injections. If dark siren analysis uses ~10-20 simulated detections, need N_eff > 40-80. **Current N_found = 24 is marginal. This directly motivates concentrating injections in the detectable region to increase N_found.**

**Confidence: HIGH**

### Result 5: EMRI Dark Siren H0 with LISA

**Statement:** EMRIs detected by LISA serve as dark sirens for H0 measurement. With a complete galaxy catalogue and ~10-50 detections at z < 1, H0 constrainable to a few percent.

**Reference:** MacLeod & Hogan (2008); Laghi et al. (2021) (cosmolisa code); Fang et al. (2024), arXiv:2403.04950.

**Relevance:** Directly describes the science goal. Laghi et al. use galaxy-catalogue cross-matching similar to Pipeline B. Their selection function treatment is more schematic than the injection-based approach here.

**Confidence: HIGH**

### Result 6: ML-Based EMRI Sensitivity (Wong et al. 2023)

**Statement:** Neural networks can rapidly predict EMRI SNR from source parameters and learn P_det(theta), enabling full treatment of selection effects without expensive per-event waveform generation.

**Reference:** Wong et al. (2023), arXiv:2212.06166.

**Relevance:** Not recommended for this project (2D parameter space too simple). But validates that P_det varies strongly with (z, M) and selection effects are critical for EMRI population inference. Could serve as cross-check.

**Confidence: HIGH**

### Result 7: Systematic Bias in Dark Siren P_det

**Statement:** Incorrect selection effects introduce systematic H0 bias exceeding statistical uncertainties at O4/O5 sensitivity. Sources: galaxy catalog incompleteness, incorrect host-galaxy weighting, P_det estimation errors.

**Reference:** Pierra et al. (2025), arXiv:2503.18887; LVK O4a (2025), arXiv:2603.20195.

**Relevance:** Motivates the physics audit. The /(1+z) Jacobian fix (Phase 15) was one such systematic.

**Confidence: HIGH**

## Open Problems Relevant to This Project

### Open Problem 1: Optimal EMRI Injection Distribution

**Statement:** What proposal distribution minimizes variance of alpha(Lambda) for EMRI dark siren H0 inference given GPU budget constraints?

**Why it matters:** Current campaign wastes ~80%+ of GPU time on undetectable events at z > 0.2.

**Current status:** No EMRI-specific optimization published. LVK uses broad proposals for CBC (Tiwari 2018). Initial campaign identifies the detection boundary; second pass can concentrate there.

**Key references:** Tiwari (2018), arXiv:1712.00482.

### Open Problem 2: Grid Resolution vs. Poisson Noise Trade-off

**Statement:** What (z, M) grid resolution avoids H0 bias while maintaining per-bin statistics?

**Why it matters:** 30x20 = 600 bins with ~69500 injections gives ~116/bin average, but highly non-uniform.

**Current status:** No systematic study. LVK avoids gridding entirely (direct MC integral). Guidance: N_total > 20 per bin; Wilson CIs for uncertainty.

### Open Problem 3: Waveform Failure Selection Bias

**Statement:** Does the 30-50% failure rate correlate with physical parameters, biasing P_det?

**Why it matters:** If failures correlate with (z, M, spin, eccentricity), effective proposal is modified.

**Mitigation:** Log all failures with input parameters. Compare distributions. If correlated, apply correction.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|------------|-------------|---------|
| Selection integral | Direct MC sum (LVK standard) | Grid interpolation (current) | MC sum avoids interpolation artifacts |
| P_det smoothing | Marcum Q-function | Sharp step function (current) | Step OK at rho_th=20 but Marcum Q more physical |
| P_det emulation | 2D histogram + interpolation | Neural network | NN overkill for 2D space |
| Interpolation | RegularGridInterpolator linear | Gaussian process regression | GP gives uncertainty but heavier |
| Injection proposal | Stratified sampling | Optimized importance sampling | Stratified more robust, no P_det prior needed |

## Concrete Implementable Methods

### Method A: Direct MC Integral for alpha(Lambda) -- PRIMARY RECOMMENDATION

Replace grid-based P_det interpolation with LVK-standard MC sum for the selection integral:

```python
# At evaluation time, for each h value:
found_mask = injection_df["SNR"] >= snr_threshold
alpha_h = found_mask.mean()  # uniform proposal
# Non-uniform proposal: alpha_h = np.mean(weights[found_mask])
```

**Pros:** No binning, no interpolation, no edge effects. Matches LVK standard.
**Cons:** Must load injection data at evaluation time. Less visual than P_det map.
**Effort:** Low. Modify `SimulationDetectionProbability` to use MC integral.

### Method B: Stratified Injection Sampling -- FOR NEXT CAMPAIGN

Divide (z, M) into strata; draw N_min injections per stratum:

```python
z_strata = np.linspace(0, 0.5, 10)   # 9 z-strata
M_strata = np.geomspace(M_min, M_max, 8)  # 7 M-strata -> 63 strata
N_per_stratum = 200
# Weight: w = p_astro(z, M) * bin_volume / N_per_stratum
```

**Pros:** Guarantees per-bin coverage. Detection boundary well-sampled.
**Cons:** Modify injection_campaign(). Must track importance weights.
**Effort:** Moderate.

### Method C: Marcum Q-Function Smoothing -- LOW-EFFORT REFINEMENT

Replace sharp threshold with continuous P_det per injection:

```python
from marcumq import marcumq
p_det_i = marcumq(1, snr, threshold)  # continuous [0, 1]
alpha_h = np.mean(p_det_per_injection)  # in MC integral
```

**Pros:** Physically motivated. One-line change. Handles near-threshold events.
**Cons:** Requires `pip install marcumq`. Small effect at rho_th = 20.
**Effort:** Minimal.

### Method D: Per-Bin Quality Metrics -- DIAGNOSTIC

Wilson CI for each grid bin:

```python
z_score = 1.96  # 95% CI
p_hat = n_det / n_total
denom = 1 + z_score**2 / n_total
center = (p_hat + z_score**2 / (2*n_total)) / denom
margin = z_score * np.sqrt(p_hat*(1-p_hat)/n_total + z_score**2/(4*n_total**2)) / denom
# Flag: n_total < 20 or CI width > 0.3
```

**Pros:** Quantifies grid reliability.
**Effort:** Low.

## Key References

| Reference | arXiv/DOI | Type | Relevance |
|-----------|-----------|------|-----------|
| Mandel, Farr & Gair (2019) | arXiv:1809.02063 | Paper | Foundational: hierarchical Bayesian inference with selection effects |
| Tiwari (2018) | arXiv:1712.00482 | Paper | Found-injections MC for VT; importance weighting |
| Farr (2019) | arXiv:1904.10879 | Paper | N_eff > 4*N_det criterion for reliable selection correction |
| Gerosa & Bellotti (2024) | arXiv:2404.16930 | Paper | Semi-analytic P_det via Marcum Q-function |
| Abbott et al. / LVK (2021) | arXiv:2010.14533 | Paper | GWTC-2 population; injection campaign methodology |
| Abbott et al. / LVK (2023) | arXiv:2111.03634 | Paper | GWTC-3 population; 4-pipeline injection campaign |
| Wong et al. (2023) | arXiv:2212.06166 | Paper | ML-based EMRI sensitivity for LISA |
| Laghi et al. (2021) | -- | Paper | EMRI dark siren H0; cosmolisa |
| Pierra et al. (2025) | arXiv:2503.18887 | Paper | Systematic bias in dark siren P_det |
| LVK O4a dark sirens (2025) | arXiv:2603.20195 | Paper | Latest LVK H0 from dark sirens |
| Fang et al. (2024) | arXiv:2403.04950 | Paper | EMRI formation channel and dark siren H0 |
| Talbot & Golomb (2024) | arXiv:2408.16828 | Paper | Neural network emulation of LVK selection function |
| GWTC-4 population (2025) | arXiv:2508.18083 | Paper | Latest LVK population analysis |
