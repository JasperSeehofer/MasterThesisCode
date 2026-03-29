# Feature Landscape

**Domain:** EMRI parameter estimation physics corrections and production simulation campaign
**Researched:** 2026-03-29 (updated with detailed specifications)

## Table Stakes

Features required for v1.2 to produce scientifically meaningful results. Missing any of these undermines the thesis.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| 5-point stencil Fisher derivatives | Forward-diff is O(epsilon), produces inaccurate CRBs; Vallisneri (2008) explicitly warns about this; the 10% d_L threshold was raised from 5% specifically because of this | Medium | Existing `five_point_stencil_derivative()` method (lines 187-243); needs integration into `compute_fisher_information_matrix()` | Method exists but is never called; signature mismatch with `finite_difference_derivative()` must be resolved |
| Galactic confusion noise in PSD | Constants defined (constants.py:74-81) but never used; confusion noise dominates LISA sensitivity at 0.1-3 mHz where EMRIs live; omitting it overestimates SNR | Low | `LISA_configuration.py` PSD methods; additive S_c(f) term | Babak et al. (2023) arXiv:2303.15929 Eq. 17 |
| Production simulation campaign (100+ tasks) | Smoke test (3 tasks, 10 steps = 20 detections) is statistically meaningless for H0 inference; need hundreds of detections | Low | Existing `submit_pipeline.sh`, cluster access validated in v1.1 | Infrastructure ready; only needs parameter selection |
| Full H0 posterior sweep [0.6, 0.9] | Current code evaluates at single h_value (0.73); a posterior requires likelihood over a grid of h values | Medium | `--h_value` CLI flag, `evaluate.sbatch`, `BayesianStatistics.evaluate()` | Need sweep script or modified sbatch; output combination logic |

## Differentiators

Features that improve scientific quality beyond minimum requirements.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| Observation-time-dependent confusion noise | S_c(f) depends on T_obs; more years of observation = more DWDs subtracted = lower confusion | Low | Confusion noise implementation | Babak formula includes T_obs-dependent knee frequency; constants.py has coefficients |
| Tightened d_L error threshold (5% from 10%) | 10% threshold was loosened due to forward-diff imprecision; 5-point stencil should allow tightening | Low | 5-point stencil completion | `FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.10` in bayesian_statistics.py:58 |
| Confusion noise toggle | Allow disabling confusion noise for comparison with previous results | Low | Confusion noise implementation | Add `include_confusion_noise` parameter to PSD method |
| Campaign diagnostic plots | SNR distribution, d_L error distribution, detection rate vs redshift, posterior convergence | Medium | Production campaign completion | Plotting infrastructure exists in `master_thesis_code/plotting/` |
| Posterior combination script | Automate combining per-h JSON results into final H0 constraint with credible intervals | Low | H0 sweep completion | Standard Bayesian combination; new script in `scripts/` |
| Seed sensitivity analysis | Run 2-3 campaigns with different seeds to assess posterior stability | Low | Production campaign completion | Pure operational task |

## Anti-Features

Features to explicitly NOT build for this milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| wCDM dark energy model fix | w0, wa silently ignored but fixing changes the physics model substantially; out of scope | Document as future work; keep LCDM hardcoded |
| Planck 2018 cosmology update | Changing Omega_m=0.25->0.3153 and H=0.73->0.6736 invalidates all prior validation | Keep WMAP-era values for consistency; document discrepancy in thesis |
| MCMC posterior sampling | Grid evaluation over [0.6, 0.9] with ~100 points is fast enough | Grid evaluation with delta_h ~ 0.003 |
| Pipeline A (BayesianInference) fixes | Dev cross-check with hardcoded 10% sigma(d_L); Pipeline B is production | Focus on Pipeline B exclusively |
| Galaxy redshift uncertainty fix (1+z)^3 | LOW priority; non-standard scaling affects z > 0.14 near detection horizon | Document as systematic uncertainty |
| Automated convergence checks | Over-engineering for thesis; visual inspection of posterior shape suffices | Plot the posterior; check by eye |

## Feature Dependencies

```
5-point stencil derivatives ----+
                                |
Galactic confusion noise in PSD +---> Production campaign (100+ tasks)
                                           |
                                           v
                                      H0 posterior sweep [0.6, 0.9]
                                           |
                                           v
                                      [Optional] Tighten d_L threshold
                                      [Optional] Seed sensitivity
```

**Ordering rationale:** Physics corrections MUST precede the production campaign -- otherwise the campaign produces data with known-wrong Fisher matrices and PSD, wasting 100+ GPU-hours. The H0 sweep operates on the campaign's CRB CSV, so it follows. The dependency chain is strictly linear.

## Detailed Feature Specifications

### 1. Five-Point Stencil Fisher Derivatives

**Current state:** `compute_fisher_information_matrix()` (line 339) calls `finite_difference_derivative()` which uses O(epsilon) forward difference:
```
df/dx = (f(x+e) - f(x)) / e
```
Requires 15 waveform generations (1 base + 1 per 14 parameters).

**Target state:** Use `five_point_stencil_derivative()` (lines 187-243) which uses O(epsilon^4) central difference:
```
df/dx = (-f(x+2e) + 8f(x+e) - 8f(x-e) + f(x-2e)) / (12e)
```
Requires 56 waveform generations (4 per parameter, no base waveform needed).

**Key implementation details:**
- **Signature mismatch:** `finite_difference_derivative()` loops internally and returns `dict[str, Array]`. `five_point_stencil_derivative()` takes a single `Parameter` and returns one array. Need a new wrapper that loops over all 14 parameters.
- **Bounds checking changes:** Forward-diff checks `x + e < upper`. Stencil checks `x - 2e > lower AND x + 2e < upper`. Some parameter draws that worked before will now raise `ParameterOutOfBoundsError`. This slightly reduces detection rate.
- **GPU cost:** ~3.7x more waveform generations. At 1-5s per waveform on H100, Fisher matrix time goes from ~15-75s to ~56-280s per detection.
- **Code cleanup needed:** Existing method has `print()` statements (should be `_LOGGER`) and references `mp.current_process().name` (multiprocessing artifact from prior prototype).

**Reference:** Vallisneri (2008), arXiv:gr-qc/0703086

**Confidence:** HIGH -- method exists; needs wiring and cleanup.

### 2. Galactic Confusion Noise in PSD

**Current state:** `LISA_configuration.py` implements instrument noise only (S_OMS + S_TM). Constants for confusion noise defined in `constants.py:74-81` but never used.

**Target state:** Add S_c(f) galactic confusion noise to A/E channel PSD. From Babak et al. (2023) Eq. 17:
```
S_c(f) = A * f^(-7/3) * exp(-f^alpha + beta * f * sin(kappa * f))
         * [1 + tanh(gamma * (f_k - f))]
```
where beta = a1 * T_obs^b1, kappa = ak * T_obs^bk, gamma and f_k also depend on T_obs.

**Implementation:** Add `S_conf(frequencies, T_obs)` method to `LisaTdiConfiguration`. Modify `power_spectral_density_a_channel()` to add S_c to the existing instrument noise. The total PSD becomes:
```
S_n(f) = S_instrument(f) + S_c(f)
```

**Impact on results:**
- Higher PSD at low frequencies -> lower SNR for most EMRIs
- Wider Cramer-Rao bounds -> larger parameter uncertainties
- Fewer detections above SNR_THRESHOLD=20
- More realistic and scientifically defensible results

**Confidence:** HIGH -- additive term; constants already defined; well-documented formula.

### 3. Production Simulation Campaign

**Current state:** Smoke test: 3 tasks x 10 steps = 30 events, 20 detections, 18 passed d_L filter. Per-task runtime ~45-90 min on H100.

**Target state:** 100+ tasks with sufficient steps for O(100-1000) detections.

**Recommended parameters:**
- **Tasks:** 100 (good parallelism on gpu_h100 partition)
- **Steps per task:** 50 (each step = one EMRI event sampled and evaluated)
- **Total events:** 5000
- **Expected detections:** With physics fixes, conservatively 40-60% detection rate = 2000-3000 detections above SNR>20. Of those, ~60-80% pass d_L filter = 1200-2400 usable.
- **Wall time per task:** 50 steps x (1-5 min waveform + ~4 min Fisher matrix for detections) = ~4-8 hours. Set SLURM `--time=10:00:00`.
- **Seed:** Pick a new seed (e.g., 200 or 1000) distinct from smoke test seed 100.

**Risk:** Physics corrections (stencil + confusion noise) may change detection rate significantly. If detection count is too low, increase steps per task or add more tasks.

**Confidence:** HIGH -- infrastructure proven; this is operational scaling.

### 4. Full H0 Posterior Sweep

**Current state:** `--evaluate --h_value 0.73` evaluates p(data | h) at a single h. Output is JSON in `simulations/posteriors/`.

**Target state:** Evaluate p(data | h) for h in linspace(0.6, 0.9, 100), combine into normalized posterior.

**Implementation options:**

**Option A (recommended): Shell loop + SLURM array**
```bash
# evaluate_sweep.sbatch with SLURM_ARRAY_TASK_ID -> h_value mapping
H_MIN=0.6, H_MAX=0.9, N_STEPS=100
h = H_MIN + TASK_ID * (H_MAX - H_MIN) / (N_STEPS - 1)
```
Each evaluation is independent and takes ~1-10 min with 16 CPUs. Can run as 100-element SLURM array job on CPU partition.

**Option B: In-process loop**
Modify `BayesianStatistics.evaluate()` to accept h_min, h_max, h_steps and loop internally. Simpler but sequential.

**Post-processing:** Combine per-h JSON results into posterior array. Normalize. Compute MAP estimate and 68%/95% credible intervals. Plot.

**Confidence:** MEDIUM -- single-h evaluation works; sweep needs new scripting and output combination.

## Waveform Generation Cost Analysis

| Method | Waveforms/Fisher | Time/detection (H100) | 50-step task time |
|--------|-----------------|----------------------|-------------------|
| Forward difference (current) | 15 | 15-75s | ~25-65 min |
| 5-point stencil (target) | 56 | 56-280s | ~90-240 min |

The ~3.7x increase is acceptable:
- H100 GPUs handle it within SLURM limits
- Accuracy improvement is essential (O(e) -> O(e^4))
- Only detections (SNR > 20) trigger Fisher matrix computation

## MVP Recommendation

**Must have (blocks thesis):**
1. 5-point stencil derivatives (physics correctness)
2. Galactic confusion noise in PSD (physics correctness)
3. Production campaign (100 tasks, 50 steps)
4. H0 posterior sweep [0.6, 0.9]

**Should have (improves thesis quality):**
5. Tighten d_L threshold from 10% to 5% (evaluate after stencil)
6. Diagnostic plots for campaign results

**Defer:**
- wCDM, Planck cosmology, redshift uncertainty -- all change physics model
- Seed sensitivity -- nice-to-have, not blocking

## Sources

- [Vallisneri (2008), arXiv:gr-qc/0703086](https://arxiv.org/abs/gr-qc/0703086) -- Fisher matrix numerical derivatives, step size recommendations
- [Babak et al. (2023), arXiv:2303.15929](https://arxiv.org/abs/2303.15929) -- LISA PSD with galactic confusion noise, Eq. 17
- [Babak 2023 PDF (full paper)](https://boa.unimib.it/retrieve/23718a72-52bb-446e-8c8c-ad954a87f112/Babak-2023-J%20Cosmol%20Astropart%20Phys-VoR.pdf) -- JCAP published version
- [SLURM GRES documentation](https://slurm.schedmd.com/gres.html) -- GPU resource scheduling
- Existing codebase: `parameter_estimation.py` (lines 140-243, 335-359), `LISA_configuration.py`, `constants.py:74-81`, `bayesian_statistics.py`

---

*Feature analysis: 2026-03-29*
