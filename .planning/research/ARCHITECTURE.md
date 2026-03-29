# Architecture Patterns: v1.2 Physics Corrections & Production Campaign

**Domain:** Physics corrections to EMRI Fisher matrix and PSD, production-scale cluster runs
**Researched:** 2026-03-29

## Recommended Architecture

No new architectural components. All v1.2 work modifies existing components in-place or adds small cluster scripts alongside existing ones.

### Integration Map

```
[Existing]                          [v1.2 Changes]

parameter_estimation.py             MODIFY: compute_fisher_information_matrix()
  compute_fisher_information_matrix()    loop calls five_point_stencil_derivative()
    -> finite_difference_derivative()    instead of finite_difference_derivative()
    -> scalar_product_of_functions()

LISA_configuration.py               MODIFY: power_spectral_density_a_channel()
  power_spectral_density_a_channel()     adds S_conf(f, T_obs) term
  S_OMS(), S_TM()                   NEW: S_conf() static method

constants.py                        NO CHANGE (confusion noise coefficients
  LISA_PSD_A, LISA_PSD_ALPHA, ...        already defined at lines 74-81)

cluster/submit_pipeline.sh          MODIFY: add --h_min, --h_max, --h_steps
cluster/evaluate.sbatch             NO CHANGE (single-h template)
                                    NEW: cluster/sweep_h0.sbatch (array job)

arguments.py                        NO CHANGE (--h_value already exists)
main.py                             NO CHANGE (evaluate() already accepts h_value)
bayesian_statistics.py              NO CHANGE (already accepts h_value param)
```

### Modified Components

| Component | Change | Impact |
|-----------|--------|--------|
| `parameter_estimation.py` | Swap `finite_difference_derivative()` for per-parameter `five_point_stencil_derivative()` loop in `compute_fisher_information_matrix()` | 56 waveforms instead of 15; ~3.7x slower derivative step; inner product count unchanged (105 calls) |
| `LISA_configuration.py` | Add `S_conf()` method; sum into `power_spectral_density_a_channel()` | PSD increases at 0.1-3 mHz; SNR decreases; fewer detections expected |
| `parameter_estimation.py` | Update `_get_cached_psd` cache key to `(n, T_obs)` | Correct PSD for 1yr pre-check vs 5yr full computation |
| `cluster/submit_pipeline.sh` | Add optional `--h_min`, `--h_max`, `--h_steps` flags | Evaluate stage becomes array job over h-values when sweep flags present |

### New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `cluster/sweep_h0.sbatch` | `cluster/` | SLURM array job for H0 sweep; one task per h-value |
| `scripts/combine_posteriors.py` | `scripts/` | Combine per-h JSON posteriors into single H0 posterior curve |

### Component Boundaries (Unchanged)

The boundary between `parameter_estimation.py` (computation) and `LISA_configuration.py` (noise model) is correct. PSD is computed by LISA_configuration and consumed by parameter_estimation via `scalar_product_of_functions`. Adding confusion noise is entirely within LISA_configuration's responsibility.

## Integration Details

### 1. Five-Point Stencil Integration

**Current state:** `compute_fisher_information_matrix()` (line 339) calls `self.finite_difference_derivative()` which returns `dict[str, Any]` with all 14 derivatives, computing 1 base waveform + 14 perturbed waveforms (15 total).

**Target state:** Use the existing `five_point_stencil_derivative()` method (lines 187-243) which already implements the correct O(epsilon^4) stencil with 4 perturbed waveforms per parameter.

**Integration approach:** Loop in `compute_fisher_information_matrix()` to build the derivative dict:

```python
# In compute_fisher_information_matrix():
lisa_response_derivatives: dict[str, Any] = {}
for parameter in parameter_list:
    lisa_response_derivatives[parameter.symbol] = self.five_point_stencil_derivative(parameter)
```

This replaces the single `self.finite_difference_derivative()` call at line 339. The rest of the Fisher matrix computation (symmetric upper-triangle loop, `scalar_product_of_functions` calls) stays identical.

**Waveform count change:** 15 -> 56 waveforms. At ~2-5s per waveform on GPU, derivative step goes from ~30-75s to ~110-280s. The 30s timeout on CRB computation in `data_simulation()` (line 332: `signal.alarm(30)`) **must be increased** to at least 300s.

**Bounds check impact:** Forward difference checks `value + epsilon < upper_limit`. Five-point stencil checks `value +/- 2*epsilon` against both limits. More events will trigger `ParameterOutOfBoundsError` and be skipped. This is a slight yield reduction, handled gracefully by the existing catch-and-continue.

**Cleanup needed in existing `five_point_stencil_derivative()`:**
- Lines 199-202, 229: replace `print()` with `_LOGGER.info()`/`_LOGGER.debug()`
- Lines 203-204: the `parameter_space` override argument is unused in the new flow -- keep for backward compat but do not use

**Code change scope:**
```
parameter_estimation.py:
  MODIFY: compute_fisher_information_matrix() -- replace line 339 with per-parameter loop
  MODIFY: five_point_stencil_derivative() -- replace print() with _LOGGER calls
  KEEP: finite_difference_derivative() -- retain for comparison/fallback
main.py:
  MODIFY: signal.alarm(30) at line 332 -- increase to 300 or make configurable
```

### 2. Galactic Confusion Noise Integration

**Current state:** `power_spectral_density_a_channel()` computes only instrumental noise (S_OMS + S_TM terms). Confusion noise constants are defined in `constants.py` lines 74-81 (`LISA_PSD_A`, `LISA_PSD_ALPHA`, `LISA_PSD_F2`, `LISA_PSD_A1`, `LISA_PSD_B1`, `LISA_PSD_AK`, `LISA_PSD_BK`) but never imported or used in `LISA_configuration.py`.

**Target state:** Add `S_conf(f, T_obs)` to the A/E channel PSD per Babak et al. (2023), arXiv:2303.15929 Eq. (17):

```
S_conf(f) = A * f^(-7/3) * exp(-f^alpha + beta * f * sin(kappa * f))
            * [1 + tanh(gamma * (f_k - f))]
```

where alpha, beta, gamma, f_k depend on observation time T_obs.

**T_obs threading:** The confusion noise depends on observation time. `LisaTdiConfiguration` is currently stateless. Two options:

**Option A (recommended): Pass T_obs through the PSD call chain.**
- `power_spectral_density(frequencies, channel, T_obs=5.0)` -> `power_spectral_density_a_channel(frequencies, T_obs=5.0)`
- Default 5 years (standard LISA mission duration)
- `_get_cached_psd` passes `self.T` to PSD and uses `(n, T_obs)` as cache key

**Option B: Store T_obs on LisaTdiConfiguration.**
- Makes it stateful; couples it to ParameterEstimation
- Less flexible for the pre-check case (T=1yr)

Option A is better: keeps `LisaTdiConfiguration` stateless, handles the T=1yr/T=5yr split naturally through the cache key.

**PSD cache invalidation:** Currently keyed by `n` only (line 108: `if n not in self._psd_cache`). Must become `(n, T_obs)` because:
- SNR pre-check: `compute_signal_to_noise_ratio(use_snr_check_generator=True)` uses T=1yr
- Full computation: T=5yr
- Different T_obs means different confusion noise means different PSD

**Code change scope:**
```
LISA_configuration.py:
  IMPORT: confusion noise constants from constants.py
  NEW: S_conf(frequencies, T_obs) static method (~20 lines)
  MODIFY: power_spectral_density_a_channel(frequencies, T_obs=5.0) -- add S_conf term
  MODIFY: power_spectral_density(frequencies, channel, T_obs=5.0) -- thread T_obs

parameter_estimation.py:
  MODIFY: _get_cached_psd(n) -> _get_cached_psd(n, T_obs) with (n, T_obs) key
  MODIFY: scalar_product_of_functions / callers -- pass T_obs through
  MODIFY: compute_signal_to_noise_ratio -- pass T=1 for pre-check, T=5 for full
```

**Results impact:** Confusion noise dominates 0.1-3 mHz (low end of EMRI band):
- SNR values decrease
- Cramer-Rao bounds widen
- Detection count likely drops
- The 10% d_L error threshold from FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD may filter more events

### 3. Production Campaign (100+ Tasks)

**No architectural changes.** The array job pattern scales directly:
```bash
submit_pipeline.sh --tasks 200 --steps 50 --seed 42
```

**Scaling considerations:**

| Concern | Smoke test (3 tasks) | Production (100+ tasks) | Mitigation |
|---------|---------------------|------------------------|------------|
| Wall time per task | 2h sufficient | May need 4-6h with 5-point stencil | Benchmark after physics fixes, adjust sbatch `--time` |
| CRB timeout | 30s | 300s needed for 5-point | Increase `signal.alarm()` |
| Queue wait | Instant | Minutes to hours | Use `--array=0-99%50` throttle |
| Merge I/O | 3 CSV files | 100+ files | Already handled by `emri-merge` |
| Failed tasks | Manual | Systematic | `resubmit_failed.sh` exists |
| Workspace storage | ~1 MB | ~50 MB | Within workspace limits |

### 4. H0 Posterior Sweep Integration

**Current state:** `evaluate.sbatch` runs for a single `h_value` (defaults to 0.73). The `--h_value` CLI flag already exists and works.

**Target state:** Evaluate posterior over h in [0.60, 0.61, ..., 0.90] (31 points) in parallel.

**Recommended approach: SLURM array job (`sweep_h0.sbatch`).**

Each array task maps `SLURM_ARRAY_TASK_ID` to an h value:
```bash
H_VALUE=$(python3 -c "print(round(${H_MIN} + ${SLURM_ARRAY_TASK_ID} * (${H_MAX} - ${H_MIN}) / (${H_STEPS} - 1), 4))")
python -m master_thesis_code "$RUN_DIR" --evaluate --h_value "$H_VALUE"
```

**Why not sequential:** Each evaluation takes ~2-5 minutes with 16 CPUs. Sequential = 60-150 minutes. Array job = ~5 minutes wall time (all 31 run simultaneously).

**Why no Python changes:** The `--h_value` flag already works end-to-end. Each evaluation writes `simulations/posteriors/h_X_YZ.json` independently. No code modification needed.

**Pipeline chain update:** `submit_pipeline.sh` gains optional `--h_min`, `--h_max`, `--h_steps` flags:
- When present: chain `sweep_h0.sbatch` (array) after merge instead of `evaluate.sbatch`
- When absent: chain `evaluate.sbatch` (single h) as before (backward compat)

**Post-sweep aggregation:** Simple script reads all `posteriors/h_*.json` files, extracts likelihood values, writes combined posterior. Can be a `scripts/combine_posteriors.py` or a final `collect.sbatch` job.

**Code change scope:**
```
cluster/sweep_h0.sbatch:       NEW (~50 lines)
cluster/submit_pipeline.sh:    MODIFY -- add sweep flags, conditional chaining
scripts/combine_posteriors.py:  NEW (~30-50 lines)
```

## Data Flow: Before vs After v1.2

### Before (v1.1)
```
data_simulation()
  -> compute SNR (instrumental PSD only)
  -> if SNR >= 20: Fisher matrix (forward-diff, 15 waveforms) -> CRB -> CSV

evaluate(h=0.73)
  -> BayesianStatistics.evaluate(h_value=0.73) -> posteriors/h_0_73.json
```

### After (v1.2)
```
data_simulation()
  -> compute SNR (PSD + confusion noise)                      [CHANGED]
  -> if SNR >= 20: Fisher matrix (5-point, 56 waveforms)      [CHANGED]
     -> CRB -> CSV

for h in [0.60..0.90] (parallel SLURM array):                 [NEW]
  evaluate(h=h) -> posteriors/h_{h}.json

combine_posteriors() -> h0_posterior_combined.json              [NEW]
```

## Patterns to Follow

### Pattern 1: Physics Change Protocol
Every formula change must follow the protocol in CLAUDE.md: old formula, new formula, reference, dimensional analysis, limiting case. Both the stencil swap and confusion noise addition are physics changes.

### Pattern 2: PSD Cache Key = (n, T_obs)
When confusion noise is added, the cache key in `_get_cached_psd` must become `(n, T_obs)` to correctly handle the T=1yr pre-check vs T=5yr full computation. Without this, the pre-check and full computation would share the wrong PSD.

### Pattern 3: GPU/CPU Portability via _get_xp
The new `S_conf()` method must use `_get_xp(frequencies)` to work with both NumPy and CuPy arrays. This pattern is already established in `S_OMS()`, `S_TM()`, and `power_spectral_density_a_channel()`.

### Pattern 4: Regression Tests Before Physics Changes
Before changing any formula, add a test capturing current numerical output, so the change effect is verifiable. This is especially important for the PSD (can test that S_conf > 0 and that PSD with confusion > PSD without).

### Pattern 5: One Physics Fix Per Commit
Each physics change gets its own `[PHYSICS]` prefixed commit. Do not bundle stencil and confusion noise in the same commit. This enables clean bisection if results change unexpectedly.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Changing Multiple Physics Simultaneously
Fixing stencil AND adding confusion noise AND updating cosmological constants in one commit. Impossible to attribute result changes to any single fix. Fix stencil first, validate. Add confusion noise, validate.

### Anti-Pattern 2: Production Campaign Before Physics Fixes
Running 100+ tasks with forward-diff Fisher matrix. Wasted GPU hours producing imprecise CRBs that must be regenerated.

### Anti-Pattern 3: In-Process H0 Loop
Adding `for h in h_values:` inside `BayesianStatistics.evaluate()`. Sequential on single CPU allocation; 31x slower wall-clock than SLURM array approach.

### Anti-Pattern 4: Changing PSD Without Updating Cache Key
Adding confusion noise but keeping cache key as just `n`. The T=1yr pre-check and T=5yr computation would share incorrect PSD values.

### Anti-Pattern 5: Deleting `finite_difference_derivative()`
The forward-difference method is useful for comparison, debugging, and potentially faster pre-screening. Keep it with a docstring noting it is retained as fallback.

## Build Order (Dependency-Driven)

```
Phase 1: Confusion noise (lower risk, self-contained)
Phase 2: 5-point stencil (higher complexity, timeout adjustment)
Phase 3: Verification campaign (3 tasks, compare old vs new)
Phase 4: Production campaign (100+ tasks)
Phase 5: H0 sweep (new cluster script, runs on Phase 4 output)
```

The physics corrections are independent and could be parallelized, but sequential is safer for validation. Confusion noise first because it is an additive term with no control flow changes.

## Scalability Considerations

| Concern | Smoke (3 tasks) | Production (100 tasks) | Large (500 tasks) |
|---------|-----------------|----------------------|-------------------|
| Waveform time/detection | ~30-75s | ~110-280s (5-point) | Same per-task |
| CRB timeout | 30s | 300s needed | Same |
| SLURM queue | Instant | Minutes-hours | Hours; use throttle |
| H0 sweep | N/A | 31 tasks, ~5 min each | Same |
| Storage | ~1 MB | ~50 MB | ~250 MB |
| Detection yield | ~7/10 steps | Expect lower with confusion noise | Same rate |

## Sources

- Babak et al. (2023), arXiv:2303.15929 -- LISA sensitivity, galactic confusion noise (Eq. 17)
- Vallisneri (2008), arXiv:gr-qc/0703086 -- Fisher matrix best practices, derivative stencils
- Existing codebase analysis: `parameter_estimation.py`, `LISA_configuration.py`, `constants.py`, `bayesian_statistics.py`, `submit_pipeline.sh`, `evaluate.sbatch`, `arguments.py`
- v1.1 smoke test: 20 detections from 30 events, 18 passed d_L filter (PROJECT.md)

---

*Architecture analysis: 2026-03-29*
