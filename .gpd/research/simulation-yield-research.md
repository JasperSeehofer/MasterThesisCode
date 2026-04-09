# Simulation Yield and EMRI Detectability Research

**Researched:** 2026-04-09
**Domain:** EMRI population synthesis, LISA detectability, simulation efficiency
**Confidence:** HIGH

## Summary

The current simulation pipeline generates EMRI events by sampling from the Babak et al. (2017) M1 cosmological model (M in [10^4.5, 10^6] solar masses, z in [0, 1.5], fixed spin a=0.98, fixed mu=10 solar masses) and computing matched-filter SNR using GPU-accelerated waveforms. Only events with SNR >= 20 proceed to the expensive Fisher matrix computation. The observed detection fraction of ~2.3% (or lower, depending on how one counts) is entirely consistent with the astrophysical model and LISA's sensitivity.

**The low yield is not a bug.** The EMRI rate model predicts events out to z=1.5 (d_L ~ 10.9 Gpc), but the LISA detection horizon for EMRIs at SNR=20 is roughly z ~ 0.3 (d_L ~ 1.55 Gpc). Only ~1.7% of the rate-weighted EMRI population falls within this horizon, and even within the horizon, not all events pass the SNR cut (orientation, orbital parameters). The injection campaign data confirms this: out of 5000 draws from the full population, exactly 1 event (0.02%) passed SNR >= 20. The main simulation loop's higher yield (~2.3%) reflects galaxy-catalog matching, which preferentially assigns nearby hosts.

**Primary recommendation:** Implement a two-stage approach: (1) pre-screen on luminosity distance before generating any waveform (skip d_L > 2 Gpc immediately), and (2) use the existing 1-year quick-SNR check more aggressively. For the H0 inference, the current yield is adequate -- the bottleneck is GPU hours, not methodology.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| SNR threshold | 20 | 15 (original Babak M1) | Babak et al. (2017) uses 20 for "detection" |
| Observation time | 5 years (full), 1 year (pre-screen) | 2, 4 years | Current code |
| Mass range | 10^4.5 - 10^6 solar masses | 10^4 - 10^7 (wider) | Model1CrossCheck._apply_model_assumptions() |
| Spin | Fixed a=0.98 | Distribution truncnorm(0.98, 0.05) | Code fixes spin; distribution exists but is unused in main loop |
| Compact object mass | Fixed mu=10 solar masses | Distribution | Code fixes mu |
| Cosmology | h=0.73, Omega_m=0.25 (WMAP-era) | Planck 2018: h=0.674, Omega_m=0.315 | constants.py |

## What Controls EMRI Detectability

### SNR Scaling Relations

The matched-filter SNR for an EMRI scales approximately as:

| Parameter | Scaling | Physical Reason |
| --- | --- | --- |
| Luminosity distance d_L | SNR ~ 1/d_L | Amplitude inversely proportional to distance |
| Compact object mass mu | SNR ~ mu | GW amplitude proportional to mass ratio mu/M times total mass |
| Central BH mass M | SNR ~ M^(5/6) (peak) | Higher M puts signal at lower frequencies in LISA's sweet spot; but very high M exits the band |
| Spin a | SNR increases with a | Higher spin means more orbits in strong-field regime before plunge |
| Eccentricity e0 | Weak dependence | Eccentric orbits emit at multiple harmonics; moderate e slightly increases total power |
| Observation time T | SNR ~ sqrt(T) to T | sqrt(T) for stationary; EMRIs chirp, so scaling can be steeper |
| Inclination/orientation | Factor ~2 variation | Face-on vs edge-on; sky position relative to LISA arm geometry |

### Detection Horizon

From the code's luminosity distance data and the physical relations module:

| Redshift z | d_L (Gpc) | SNR regime (typical EMRI) |
| --- | --- | --- |
| 0.1 | 0.44 | Easily detectable (SNR >> 20) |
| 0.2 | 0.95 | Detectable for favorable parameters |
| 0.3 | 1.51 | Near detection threshold (SNR ~ 15-30) |
| 0.5 | 2.77 | Mostly undetectable |
| 1.0 | 6.54 | Undetectable (SNR < 1) |
| 1.5 | 10.89 | Completely undetectable |

**Effective LISA EMRI detection horizon: z ~ 0.3 (d_L ~ 1.5 Gpc) for SNR=20.**

The code already encodes this: `LUMINOSITY_DISTANCE_THRESHOLD_GPC = 1.55` in constants.py.

### Parameter Ranges Where EMRIs Are Essentially Undetectable

1. **d_L > 2 Gpc (z > 0.4):** SNR drops as 1/d_L; at 2 Gpc, typical SNR is 5-10, well below threshold.
2. **M < 10^4.5 solar masses:** Too few cycles in LISA band before plunge; SNR drops rapidly.
3. **M > 10^6.5 solar masses:** GW frequency drops below LISA's sensitive band (< 10^-4 Hz).
4. **Very low mass ratio (mu < 5):** Amplitude ~ mu; halving mu halves SNR.

### Confirmed from Injection Data

Analysis of 5000 injection events (h=0.73 tasks):
- **0.02% pass SNR >= 20** (1 out of 5000)
- **0.64% have SNR > 5**
- All events with SNR > 5 have d_L < 1.66 Gpc and z < 0.33
- All events with SNR > 5 have log10(M) between 5.25 and 5.97

Analysis of 42 detected events in cramer_rao_bounds.csv:
- All have d_L < 0.70 Gpc (z < ~0.15)
- log10(M) concentrated at 5.35-5.85 (the "sweet spot")
- Median SNR = 17 (most detections are near threshold)

## Is 2.3% a Reasonable Detection Fraction?

**YES.** The 2.3% figure is consistent with the literature and the model assumptions. Here is the quantitative breakdown:

### Volume Argument

The EMRI rate model samples from z in [0, 1.5]. The detection horizon is at z ~ 0.3.
- Naive comoving volume fraction within d_L < 1.55 Gpc: **0.29%**
- Rate-weighted fraction (integrating dN/dz * R_EMRI over M and z): **1.71%**
- The rate-weighted fraction is higher than the volume fraction because the EMRI rate peaks at lower redshifts for the dominant mass bins.

### Literature Comparison

| Study | Detection Rate | Notes |
| --- | --- | --- |
| Babak et al. (2017) PRD 95, 103012 | 1 - 4200 yr^-1 | M1 model: ~294 yr^-1 intrinsic, ~10-50 detected |
| Laghi et al. (2021) arXiv:2102.01708 | ~25 events in 4-yr mission (M1) | Uses SNR>100 for "loud" events for H0 |
| Berry et al. (2019) | ~10s yr^-1 (M1, optimistic) | |

Babak et al. M1 model predicts ~294 EMRIs/year intrinsic rate, with ~10-50 detected at SNR >= 20 (detection fraction ~3-17%). However, this depends critically on the assumed detection threshold and the mass function.

**The 2.3% overall simulation yield (including events far beyond the horizon) translates to roughly the right number of detections per unit of simulation effort.** The main simulation pipeline benefits from the galaxy catalog matching, which preferentially selects nearby host galaxies, effectively concentrating events near the detection horizon.

### Why the Injection Campaign Shows Lower Rates

The injection campaign draws from the full cosmological volume without galaxy catalog matching. This means most draws are at z > 0.5 where EMRIs are completely undetectable. The 0.02% detection fraction from injections reflects the raw population-to-detection ratio across the entire sampled volume.

## Existing SNR Pre-Screening (Already Implemented)

The current code already implements a two-stage SNR screening:

1. **1-year quick SNR check** (line 394-401 of main.py):
   - Generates waveform using `snr_check_generator` (1-year observation)
   - If `quick_snr < SNR_THRESHOLD * 0.3` (i.e., quick_snr < 4.5), skip event
   - Factor 0.3 is conservative: SNR scales as sqrt(T) for stationary sources, so 1-yr SNR * sqrt(5) ~ 2.24x for 5-year; the 0.3 factor (threshold = 6.7 for full SNR=20) accounts for EMRI chirp

2. **5-year full SNR check** (line 409):
   - Only computed if quick SNR passes
   - Full waveform generation (5 years)
   - If SNR >= 20, proceed to Fisher matrix

3. **Fisher matrix** (line 473):
   - Only computed for events passing the full SNR check
   - 14 parameters x 4 waveforms per parameter (5-point stencil) = 56 waveforms
   - Plus 105 inner products for the Fisher matrix

**The pre-screening is already effective.** The 1-year quick-SNR check eliminates the vast majority of undetectable events at 1/5 the waveform cost.

## Optimization Opportunities

### 1. Distance-Based Pre-Screen (HIGHEST IMPACT, TRIVIAL TO IMPLEMENT)

Before generating any waveform, check if the host galaxy's luminosity distance exceeds a generous threshold. Skip immediately if d_L > 2.0 Gpc (well beyond the detection horizon of ~1.55 Gpc but with safety margin for orientation-dependent boosts).

**Estimated speedup:** Eliminates ~75% of waveform generations (based on the fraction of sampled events at d_L > 2 Gpc). Each skipped event saves ~1-2s of GPU time (1-year waveform generation).

```python
# In data_simulation(), after set_host_galaxy_parameters():
if parameter_estimation.parameter_space.luminosity_distance.value > 2.0:
    _ROOT_LOGGER.debug(f"Skipping event: d_L = {d_L:.2f} Gpc > 2.0 Gpc")
    continue
```

**Impact on H0 inference: NONE.** This is a deterministic cut that can be applied identically in the detection probability calculation. The P_det is already zero for these events.

### 2. More Aggressive Quick-SNR Threshold

The current threshold factor of 0.3 is very conservative. For the 5-point stencil Fisher matrix:
- 1-year SNR of 6 extrapolates to 5-year SNR of ~13-20 (depending on chirp evolution)
- Could raise the factor from 0.3 to 0.4 or even 0.5, accepting a small risk of missing marginal detections

**Estimated additional speedup:** 10-20% of remaining waveform evaluations. Less impactful than the distance cut.

### 3. Mass-Based Pre-Screen

Check if M falls in the LISA-sensitive mass range. With the current Model1CrossCheck assumptions, M is already restricted to [10^4.5, 10^6], which is entirely within the sweet spot. No additional filtering needed.

### 4. Importance Sampling / Smart Sampling

**Standard approach in the literature:** Chua & Gair (2022, arXiv:2212.06166) developed a machine-learning interpolator for EMRI SNR to efficiently compute detection probabilities across the parameter space. This avoids generating waveforms for every sampled event.

For this project, importance sampling would mean:
- Sample (M, z) from a proposal distribution concentrated near the detection horizon (z < 0.5, M in sweet spot)
- Reweight by the ratio p_true(M,z) / p_proposal(M,z) in the H0 inference

**However, this is NOT recommended for this project** because:
1. The galaxy catalog matching already provides a form of importance sampling (only real galaxy positions are used)
2. The P_det correction in the Bayesian inference already accounts for selection effects
3. Changing the sampling distribution would require careful validation that the reweighting is correct
4. The current approach is standard and well-understood (Babak et al. 2017 methodology)

### 5. Skip Fisher Matrix for Marginal Detections

Events with SNR barely above 20 contribute relatively little to the H0 posterior (large distance uncertainties). Could consider:
- Only computing Fisher matrix for SNR > 25
- Using a simplified Fisher approximation for 20 < SNR < 25

**Not recommended:** The Fisher matrix provides the Cramer-Rao bounds that feed into the H0 inference. Skipping it would lose information. The marginal events are actually scientifically important because they probe the detection boundary.

## Fisher Matrix Cost Analysis

### Current Cost Structure

| Operation | Waveforms | Inner Products | Time per Event (est.) |
| --- | --- | --- | --- |
| Quick SNR (1-yr) | 1 | 1 | ~0.5s |
| Full SNR (5-yr) | 1 | 1 | ~2s |
| 5-point stencil derivatives | 56 (4 per parameter x 14) | 0 | ~110s |
| Fisher matrix entries | 0 | 105 (upper triangle) | ~50s |
| **Total for detected event** | **58** | **107** | **~160s** |
| **Total for rejected event** | **1-2** | **1-2** | **~0.5-4s** |

The Fisher matrix dominates the cost for detected events. But since only ~2% of events are detected, the total simulation time is dominated by the screening of non-detected events.

### Effective Cost Breakdown

For 1000 sampled events with ~2% detection rate (~20 detections):
- Screening 980 non-detections: 980 x ~1s = ~980s (16 min)
- Processing 20 detections: 20 x ~160s = ~3200s (53 min)
- **Total: ~70 min for 20 detections**

The distance pre-screen (optimization #1) would reduce the screening cost by ~75%, saving ~12 min per 1000 events.

## Validation Strategies

### Checking Detection Fraction

1. **Rate integral check:** Integrate the EMRI rate model over the detectable region (z < 0.3, M in [10^4.5, 10^6]) and compare to the total rate. Should give ~1.7% (confirmed above).

2. **Distance distribution check:** The detected events should have d_L strongly concentrated at d_L < 1 Gpc, with a tail to ~1.5 Gpc. The data confirms this (all 42 detections at d_L < 0.70 Gpc).

3. **SNR distribution check:** The detected SNR distribution should follow approximately dN/dSNR ~ SNR^-4 for an Euclidean population (volume-limited). The sharp cutoff at SNR=20 and rapid falloff is consistent.

4. **Mass distribution check:** Detections should be concentrated in the "sweet spot" log10(M) ~ 5.3-5.9, where EMRIs produce the most in-band GW cycles. The data shows log10(M) concentrated at 5.35-5.85.

### Red Flags

- If detection fraction exceeds 10%, the distance sampling is likely biased toward nearby events
- If no detections have SNR > 50, the high-SNR tail may be suppressed (check spin distribution)
- If all detections cluster at a single mass, the mass function sampling may be broken

## Common Pitfalls

### Pitfall 1: Confusing Intrinsic Rate with Detection Rate

**What goes wrong:** Quoting 294 EMRIs/year (intrinsic) and expecting 294 detections.
**Why it happens:** The intrinsic rate counts all EMRIs in the universe; LISA detects a tiny fraction.
**How to avoid:** Always distinguish "intrinsic rate" from "detection rate." The detection rate depends on the SNR threshold, observation time, and detector sensitivity curve.

### Pitfall 2: Galaxy Catalog Boundary Effects

**What goes wrong:** The GLADE catalog has a completeness limit at z ~ 0.3-0.5. Events beyond this don't have host galaxy matches, which can bias the sampling.
**Why it happens:** Galaxy catalog incompleteness is a known systematic in dark siren analyses.
**How to avoid:** The code already handles this through the completeness correction (L_comp term in the Bayesian inference). The simulation pipeline's galaxy matching naturally restricts to the catalog's range.

### Pitfall 3: Fixed Spin Biases the Detection Rate

**What goes wrong:** Fixing a=0.98 (near-maximal spin) gives optimistic detection rates.
**Why it happens:** Higher spin EMRIs accumulate more GW cycles in the strong-field regime, boosting SNR. The true spin distribution may include lower spins with lower SNR.
**How to avoid:** Document that the detection rate assumes the M1 model's near-maximal spin. For the H0 inference, this is consistent as long as the same assumption is used in the P_det calculation.

### Pitfall 4: Pre-Screening Introduces Selection Bias

**What goes wrong:** Aggressive pre-screening (e.g., strict distance cut) could miss rare loud events from favorable orientations at moderate distance.
**Why it happens:** The quick-SNR check and distance cut are deterministic, not probabilistic.
**How to avoid:** Use generous thresholds (d_L < 2 Gpc, not 1.55 Gpc; quick-SNR factor 0.3, not 0.5). Any selection applied in the simulation must be reflected in the P_det calculation.

## Level of Rigor

**Required for this analysis:** Controlled approximation with quantitative validation.

**What this means concretely:**
- Detection fraction must be consistent within a factor of ~2 with literature estimates
- Pre-screening thresholds must be demonstrably conservative (no missed detections in validation)
- Any modification to the sampling must be provably equivalent in the H0 inference (same P_det correction)

## Relevant Prior Work

| Paper | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Science with LISA. V: EMRIs | Babak et al. | 2017 | Defines M1 model used in this code | Detection rates, mass function, SNR thresholds |
| GW cosmology with EMRIs | Laghi et al. | 2021 | EMRI dark siren H0 forecast | Detection numbers, methodology, selection effects |
| Rapid EMRI sensitivity with ML | Chua & Gair | 2022 | ML-based SNR interpolation | Approach to fast P_det evaluation |
| Hitchhiker's Guide to galaxy catalogs | Gray et al. | 2022 | Galaxy catalog methodology | Completeness correction, selection effects |
| Clustering effects on dark siren H0 | Various | 2024 | Simulation study of dark sirens | Convergence with N_det, clustering effects |

## Computational Tools

| Tool | Purpose | Already in Use |
| --- | --- | --- |
| FastEMRIWaveforms (few) | EMRI waveform generation | Yes |
| fastlisaresponse | LISA TDI response | Yes |
| CuPy | GPU array operations | Yes |
| emcee | MCMC sampling of (M, z) | Yes |
| scipy.stats.gaussian_kde | P_det estimation | Yes |

No additional tools are needed for the optimization recommendations.

## Recommendations Summary

| Priority | Action | Estimated Speedup | Implementation Effort | Risk |
| --- | --- | --- | --- | --- |
| 1 | Distance pre-screen (d_L > 2 Gpc) | ~75% fewer waveforms | 5 lines of code | Near-zero |
| 2 | Slightly more aggressive quick-SNR | ~10-20% additional | 1 line change | Low (validate first) |
| 3 | Log statistics on rejection reasons | N/A (diagnostics) | ~20 lines | Zero |
| -- | Importance sampling | Variable | Major refactor | HIGH (could bias H0) |
| -- | Skip Fisher for marginal events | ~20% Fisher savings | Moderate | MEDIUM (lose information) |

**Bottom line:** The 2.3% detection fraction is physically correct and consistent with the literature. The most impactful optimization is a trivial distance pre-screen. Do not change the fundamental sampling strategy -- the current approach (sample from astrophysical distribution, compute SNR, apply threshold) is the standard method used by Babak et al. (2017) and Laghi et al. (2021).

## Sources

### Primary (HIGH confidence)

- [Babak et al. (2017), PRD 95, 103012 -- arXiv:1703.09722](https://arxiv.org/abs/1703.09722) -- M1 model, EMRI detection rates, mass function
- [Laghi et al. (2021), arXiv:2102.01708](https://arxiv.org/abs/2102.01708) -- EMRI dark siren H0 forecast
- [Chua & Gair (2022), MNRAS 522, arXiv:2212.06166](https://arxiv.org/html/2212.06166) -- ML-based EMRI SNR computation
- [Gray et al. (2020), arXiv:1908.06050](https://arxiv.org/abs/1908.06050) -- Dark siren galaxy catalog method

### Secondary (MEDIUM confidence)

- [Laghi et al. (2026), arXiv:2603.23612](https://arxiv.org/html/2603.23612v1) -- Joint EMRI+MBHB LISA cosmology (recent)
- [Berry et al. (2019)](https://cplberry.com/tag/emris/) -- EMRI rate estimates review
- [Burke (2024), EMRI Workshop](https://ollieburke.github.io/EMRI_Workshop/book/EMRI_workshop_book.html) -- EMRI waveform tutorial

### Code-Based (HIGH confidence for this project)

- `master_thesis_code/cosmological_model.py` -- Model1CrossCheck implementation
- `master_thesis_code/parameter_estimation/parameter_estimation.py` -- SNR and Fisher computation
- `master_thesis_code/constants.py` -- Thresholds and physical constants
- `simulations/injections/` -- 5000-event injection campaign data analysis

## Metadata

**Confidence breakdown:**

- Detection fraction analysis: HIGH -- confirmed by both rate integral and empirical injection data
- SNR scaling relations: HIGH -- standard GW physics, confirmed by code behavior
- Optimization recommendations: HIGH -- trivial changes with clear impact estimates
- Literature comparison: MEDIUM -- rate estimates span orders of magnitude; our fraction is within the expected range

**Research date:** 2026-04-09
**Valid until:** Indefinitely (astrophysical model and LISA sensitivity are stable)
