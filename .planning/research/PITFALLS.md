# Domain Pitfalls: v1.2 Physics Corrections & Production Campaign

**Domain:** Numerical derivative methods, LISA noise modeling, production-scale GW simulations, H0 posterior sweeps
**Researched:** 2026-03-29
**Overall confidence:** HIGH (based on direct codebase inspection + Vallisneri 2008 + Babak et al. 2021)

## Critical Pitfalls

Mistakes that cause rewrites, corrupted results, or wasted cluster hours.

### Pitfall 1: 5-Point Stencil Bounds Check Rejects More Events

**What goes wrong:** The 5-point stencil requires evaluating waveforms at `p +/- 2*epsilon`, while the current forward-difference only needs `p + epsilon`. Switching stencils doubles the bounds-checking range. Parameters near their limits (especially `p0` with range [10, 16] and `e0` with range [0.05, 0.7]) will trigger `ParameterOutOfBoundsError` far more often, silently reducing the detection yield.

**Why it happens:** The existing `five_point_stencil_derivative()` method (line 210) already checks `p - 2*epsilon` and `p + 2*epsilon`, but the calling code (`compute_fisher_information_matrix()` at line 339) calls `finite_difference_derivative()` which only checks `p + epsilon`. Switching the call target changes the rejection rate.

**Consequences:** A production campaign may produce significantly fewer detections than the smoke-test predicted. With 100 tasks x 50 steps, if rejection rate doubles from 30% to 60%, you get half the detections expected. The H0 posterior becomes statistically weaker than planned.

**Prevention:**
1. Before switching, run a dry-run counting how many events pass bounds checks with forward-diff vs. 5-point stencil on the same seed.
2. Consider reducing `derivative_epsilon` for parameters with narrow ranges (currently all use 1e-6, which is likely fine for most, but verify `p0` and `e0` specifically).
3. Add logging that counts bounds rejections per parameter symbol so you can diagnose which parameters cause the most rejections.

**Detection:** Compare detection counts between forward-diff and 5-point runs on identical seeds and parameters. A drop greater than 20% warrants investigation.

### Pitfall 2: 5-Point Stencil Has Wrong Sign Convention in Existing Code

**What goes wrong:** The existing `five_point_stencil_derivative()` at line 233 computes:
```python
(-lisa_responses[3] + 8*lisa_responses[2] - 8*lisa_responses[1] + lisa_responses[0]) / 12 / epsilon
```
The stencil steps are `[-2, -1, +1, +2]` (indices 0-3), so `responses[0]` = f(x-2h), `responses[1]` = f(x-h), `responses[2]` = f(x+h), `responses[3]` = f(x+2h). The standard 5-point formula is:
```
f'(x) = (-f(x+2h) + 8*f(x+h) - 8*f(x-h) + f(x-2h)) / (12h)
```
This matches: `(-responses[3] + 8*responses[2] - 8*responses[1] + responses[0])`. The sign convention is **correct**.

**Why this matters:** The formula is correct, but the lack of a base waveform (f(x)) in the 5-point formula may confuse reviewers. Additionally, the method currently accepts an optional `parameter_space` argument and mutates `self.parameter_space` (line 204), which is a side effect that the forward-difference method avoids by computing a local `base_waveform`. This asymmetry is a maintenance hazard.

**Consequences:** If someone "fixes" the signs thinking they are wrong, the Fisher matrix becomes garbage. If someone passes a `parameter_space` argument, it silently replaces the instance state.

**Prevention:**
1. Add a reference comment above the formula: `# Eq. (8) in Vallisneri (2008), arXiv:gr-qc/0703086`
2. Remove the `parameter_space` mutation side effect when integrating.
3. Add a regression test that computes the derivative of a known analytic function (e.g., sin) and verifies the result matches to O(h^4) accuracy.

**Detection:** Unit test comparing 5-point derivative of `sin(x)` against `cos(x)` should match to ~1e-20 for h=1e-6 (since error is O(h^4) = O(1e-24)).

### Pitfall 3: Switching to 5-Point Stencil Doubles Waveform Generation Count Per Fisher Matrix

**What goes wrong:** Forward-difference computes 14 derivatives using 14+1 = 15 waveform evaluations (1 base + 1 per parameter). The 5-point stencil needs 4 evaluations per parameter = 56 waveforms total. The 30-second timeout (`signal.alarm(30)` at line 332 of `main.py`) was calibrated for the forward-diff workload. With 4x more waveform calls, the timeout will fire on events that would have succeeded.

**Why it happens:** Each waveform generation call takes 0.5-2 seconds on H100 GPUs. Forward-diff: ~15 calls = 7-30s. Five-point: ~56 calls = 28-112s. The 30s timeout kills events that need >30s total for all derivatives.

**Consequences:** Massive increase in timeout-skipped events. Production campaign yields far fewer CRB detections. Some parameter combinations that produced valid Fisher matrices with forward-diff will now timeout.

**Prevention:**
1. Increase the timeout from 30s to 120s for the CRB computation phase (the SNR phase can keep 30s since it only generates 2 waveforms).
2. Separate the timeout for SNR computation and CRB computation.
3. Monitor `generation_time` in the CSV output to calibrate the timeout.
4. Consider batching: the current loop in `five_point_stencil_derivative` generates waveforms one-at-a-time. If `few` supports batch evaluation, that would cut wall time.

**Detection:** Log the wall time of each `compute_Cramer_Rao_bounds()` call. If median time exceeds 50% of the timeout, the timeout is too aggressive.

### Pitfall 4: Fisher Matrix Ill-Conditioning Worsens with Better Derivatives

**What goes wrong:** The 5-point stencil is more accurate (O(h^4) vs O(h)), which paradoxically can make the Fisher matrix more ill-conditioned. The forward-difference smooths over numerical noise from the waveform generator, acting as implicit regularization. The 5-point stencil resolves finer structure, exposing near-degeneracies between parameters (especially the three phase parameters Phi_phi0, Phi_theta0, Phi_r0).

**Why it happens:** Vallisneri (2008) warns that Fisher matrices for multi-parameter waveforms are "often nearly singular" and "the inverse (covariance matrix) is plagued by numerical instabilities." Better derivatives make this worse, not better.

**Consequences:** `np.matrix(...).I` (line 365 of parameter_estimation.py) will produce covariance matrices with negative diagonal elements or astronomically large uncertainties. The `d_L_uncertainty` may become negative or exceed the luminosity distance itself, producing nonsensical `Detection` objects.

**Prevention:**
1. After inversion, check that all diagonal elements of the covariance matrix are positive. If not, flag the event.
2. Compute the condition number of the Fisher matrix before inversion. If `cond(F) > 1e12`, skip or regularize.
3. Use `np.linalg.inv()` instead of `np.matrix(...).I` -- the matrix class is deprecated and provides no numerical advantage.
4. Consider adding Tikhonov regularization: `F_reg = F + lambda * I` with a small lambda.

**Detection:** Log the condition number alongside each CRB row. A condition number > 1e10 is a warning sign.

### Pitfall 5: Confusion Noise Changes SNR Threshold Pass Rate

**What goes wrong:** Adding galactic confusion noise to the PSD increases the noise floor in the 0.1-3 mHz band. EMRIs in this frequency range (which is most of them -- EMRI signals typically span 0.1-10 mHz) will have reduced SNR. Events that previously passed the SNR >= 20 threshold may now fail, dramatically reducing the detection count.

**Why it happens:** The confusion noise PSD is additive: `S_total(f) = S_instrument(f) + S_confusion(f)`. The confusion noise dominates below ~3 mHz for the first few years of observation. Since `scalar_product_of_functions` divides by PSD, a larger denominator means smaller SNR.

**Consequences:** The production campaign may need more tasks to achieve the same number of detections. If calibrated on the old (no-confusion) PSD, the 100-task budget may be insufficient.

**Prevention:**
1. Implement confusion noise first, THEN calibrate the production campaign size.
2. Run a small test (5 tasks, same seed as smoke test) with confusion noise and compare detection counts to the no-confusion smoke test.
3. The constants in `constants.py` (lines 75-82) reference arXiv:2303.15929 Eq. 17 but actually the canonical reference is Babak et al. (2021) arXiv:2108.01167. Verify the formula against the correct source.
4. The confusion noise depends on observation time T -- use `self.T = 5` years consistently.

**Detection:** Compare SNR distributions (median, mean, fraction above threshold) with and without confusion noise on the same parameter sets.

### Pitfall 6: Confusion Noise Formula Has Observation-Time Dependence

**What goes wrong:** The galactic confusion noise PSD from Babak et al. depends on the observation duration T_obs through time-dependent exponents (the `A1`, `B1`, `AK`, `BK` coefficients in constants.py). The current PSD cache in `_get_cached_psd()` keys on waveform length `n`, not on observation time. If the 1-year SNR check generator and the 5-year full generator produce different-length waveforms that happen to hash to the same `n`, the wrong confusion noise would be used.

**Why it happens:** The knee frequency and slope of the confusion noise change with observation time because more galactic binaries are resolved and subtracted over longer observation periods. The PSD method currently receives only `frequencies` and `channel`, with no `T_obs` parameter.

**Consequences:** Using the wrong T_obs in the confusion noise formula produces incorrect SNRs. A 5-year confusion noise applied to a 1-year SNR pre-check would underestimate noise (fewer sources subtracted at 1 year), inflating the quick SNR.

**Prevention:**
1. Pass `T_obs` as a parameter to `power_spectral_density()` and to the confusion noise function.
2. Clear the PSD cache when switching between the SNR check generator (T=1 year) and the full generator (T=5 years), or key the cache on `(n, T_obs)`.
3. For a first implementation, hardcode T_obs=5 years and document that the 1-year quick check does not include confusion noise (it only needs to be a rough filter anyway).

**Detection:** Compare PSD values at 1 mHz for T=1yr vs T=5yr; they should differ significantly (factor ~2-5).

## Moderate Pitfalls

### Pitfall 7: `five_point_stencil_derivative` Uses `print()` Not `logging`

**What goes wrong:** The existing method at lines 199-202 uses `print(..., flush=True)` with `time.ctime()` for progress reporting, unlike the rest of the codebase which uses `_LOGGER`. In a production SLURM job, print output goes to stdout which may be buffered differently from the log file. Worse, the `mp.current_process().name` reference (line 230) suggests this method was designed for multiprocessing, but the Fisher matrix computation is currently single-process per GPU task.

**Prevention:** Replace `print()` calls with `_LOGGER.info()` when integrating the method. Remove the `mp.current_process().name` reference.

### Pitfall 8: H0 Sweep Runs Evaluate Once Per h-Value but Reloads Everything Each Time

**What goes wrong:** The current `evaluate()` in main.py creates a new `BayesianStatistics()` instance per call, which reads CSVs, creates `DetectionProbability` KDEs, builds multivariate Gaussians, and spawns a multiprocessing pool. For a sweep over 30+ h-values in [0.6, 0.9], this redundant setup dominates wall time.

**Why it happens:** `BayesianStatistics.__init__()` (line 106) reads all CSVs. `evaluate()` (line 117) rebuilds detection probability and Gaussian objects. The detection-independent setup is identical for every h-value.

**Consequences:** An H0 sweep that should take 30 minutes takes 10+ hours because setup is repeated 30 times. On a cluster with 1-hour time limits, the job times out.

**Prevention:**
1. Refactor `BayesianStatistics.evaluate()` to separate setup from h-dependent computation.
2. Create a `sweep(h_values: list[float])` method that does setup once, then loops over h-values.
3. Alternatively, submit each h-value as a separate SLURM array task (parallelize across h-values).
4. The simplest cluster-friendly approach: modify `evaluate.sbatch` to accept an `--array` parameter where each task evaluates one h-value from a pre-defined grid.

**Detection:** Time the setup phase vs. the per-h computation phase. If setup > 50% of total time, refactoring pays off.

### Pitfall 9: Production Campaign Symlink Collision

**What goes wrong:** `simulate.sbatch` (line 83) creates a symlink `$PROJECT_ROOT/simulations -> $RUN_DIR/simulations`. If two campaigns run simultaneously (different seeds, same project root), the symlink is overwritten by the second job, and the first job's output goes to the wrong directory.

**Why it happens:** The symlink is a workspace-level shared resource, but array tasks from different campaigns share the same `$PROJECT_ROOT`.

**Consequences:** Data from campaign A ends up in campaign B's run directory. Results are silently mixed.

**Prevention:**
1. Never run two campaigns simultaneously from the same `$PROJECT_ROOT`.
2. Add a lockfile check: if `$PROJECT_ROOT/simulations` already points to a different run directory, abort with an error.
3. Long-term: make the simulation code accept an absolute output directory instead of using relative paths from CWD.

**Detection:** Check that `readlink $PROJECT_ROOT/simulations` matches the expected `$RUN_DIR/simulations` in each sbatch script before running.

### Pitfall 10: `afterok` Dependency Kills Entire Pipeline on Single Task Failure

**What goes wrong:** `submit_pipeline.sh` (line 101) uses `--dependency="afterok:$SIM_JOB"` for the merge step. With 100+ array tasks, if even ONE task fails (timeout, OOM, node failure), the merge job is never scheduled. The entire pipeline stalls silently.

**Why it happens:** `afterok` requires ALL tasks to complete successfully. At scale, the probability of at least one failure is high. With 100 tasks and a 5% per-task failure rate, P(at least one failure) = 1 - 0.95^100 = 99.4%.

**Consequences:** Pipeline hangs indefinitely. No merge, no evaluate, no results. The user must manually check `sacct`, identify failures, resubmit, and re-chain.

**Prevention:**
1. Change merge dependency from `afterok` to `afterany` so merge runs even if some tasks fail. The merge script already handles missing files gracefully.
2. Add a monitoring step or `scontrol show job` check that alerts when tasks fail.
3. Document the `resubmit_failed.sh` workflow prominently -- it exists but users may not know to use it.

**Detection:** After submission, periodically run `sacct -j $SIM_JOB --format=JobID,State,ExitCode` to catch failures early.

### Pitfall 11: d_L Error Threshold Must Be Recalibrated After Stencil Upgrade

**What goes wrong:** The current 10% threshold (`FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.10` in bayesian_statistics.py line 58) was set because the forward-difference Fisher matrix was too imprecise for tighter thresholds. After upgrading to the 5-point stencil, d_L uncertainties should decrease, making 5% feasible. But if the threshold is tightened prematurely (before verifying the stencil works correctly), valid detections are discarded. If not tightened at all, the improved Fisher matrix is wasted.

**Why it happens:** The threshold is a downstream parameter that depends on the upstream physics change. The two changes (stencil + threshold) must be sequenced correctly.

**Consequences:** Tightening too early: fewer detections than needed for the H0 posterior. Not tightening: the 5-point stencil upgrade provides no practical benefit to the science output.

**Prevention:**
1. Run the production campaign with the 10% threshold first.
2. Analyze the distribution of `d_L_uncertainty / d_L` in the results.
3. If the median drops well below 10%, tighten to 5% in a separate step.
4. Never change the physics (stencil) and the filter (threshold) in the same phase.

**Detection:** Plot the histogram of `d_L_uncertainty / d_L` for both forward-diff and 5-point results on the same parameter sets.

## Minor Pitfalls

### Pitfall 12: `np.matrix` Is Deprecated

**What goes wrong:** Line 365 of parameter_estimation.py uses `np.matrix(cp.asnumpy(...)).I` for matrix inversion. `np.matrix` is deprecated since NumPy 1.15 and may be removed in a future version.

**Prevention:** Replace with `np.linalg.inv(cp.asnumpy(fisher_information_matrix))`. This is a one-line change with identical semantics for square matrices.

### Pitfall 13: PSD Cache Not Invalidated When Confusion Noise Is Added

**What goes wrong:** `_psd_cache` in `ParameterEstimation` caches PSD arrays keyed by waveform length `n`. Adding confusion noise changes the PSD values, but the cache does not know this. If any code path populates the cache before confusion noise is configured, stale values persist.

**Prevention:** Clear the cache when the PSD configuration changes, or compute the PSD lazily with all noise components always included.

### Pitfall 14: H0 Sweep Output File Naming Collision

**What goes wrong:** The posterior output path (bayesian_statistics.py line 270) uses `np.round(self.h, 3)` to generate filenames like `h_0_73.json`. For h-values like 0.725 and 0.7249, rounding to 3 decimals produces the same filename. The second write overwrites the first.

**Prevention:** Use more decimal places in the filename (4-5), or use the exact h-value string without rounding.

### Pitfall 15: Workspace Expiration During Long Campaigns

**What goes wrong:** bwHPC workspaces expire after 60 days. A production campaign submitted near the end of a workspace lifetime may lose results before analysis is complete.

**Prevention:** Check workspace expiration date before submitting. Copy results to persistent storage (`$HOME` or project storage) immediately after the pipeline completes.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Fisher stencil upgrade | Timeout too short (Pitfall 3), more bounds rejections (Pitfall 1), ill-conditioning (Pitfall 4) | Increase timeout to 120s, log condition numbers, validate on test set first |
| Confusion noise in PSD | SNR drop reduces detections (Pitfall 5), T_obs dependence (Pitfall 6), cache invalidation (Pitfall 13) | Implement before campaign sizing, pass T_obs, clear PSD cache |
| Production campaign (100+ tasks) | afterok stalls pipeline (Pitfall 10), symlink collision (Pitfall 9) | Switch to afterany, document single-campaign constraint |
| H0 sweep [0.6, 0.9] | Redundant setup per h-value (Pitfall 8), filename collisions (Pitfall 14) | Refactor or parallelize sweep, increase filename precision |
| d_L threshold adjustment | Premature tightening (Pitfall 11) | Sequence after stencil validation, analyze distribution first |

## Ordering Recommendations

Based on pitfall dependencies, the recommended phase ordering is:

1. **Physics corrections first** (stencil + confusion noise) -- because the production campaign results depend on these being correct. Running 100+ tasks with wrong physics wastes cluster hours.
2. **Small validation campaign** -- verify detection rates, d_L error distributions, and condition numbers before committing to 100+ tasks.
3. **Production campaign** -- with calibrated timeouts, afterany dependencies, and confusion noise.
4. **H0 sweep** -- after the campaign completes and results are validated.

Running the production campaign before fixing the physics (as a "baseline") is tempting but wasteful: the results are not scientifically usable and the cluster allocation is limited.

## Sources

- [Vallisneri (2008), "Use and abuse of the Fisher information matrix"](https://arxiv.org/abs/gr-qc/0703086) -- Fisher matrix numerical pitfalls, ill-conditioning, parameter degeneracies
- [Babak et al. (2021), "LISA Sensitivity and SNR Calculations"](https://arxiv.org/abs/2108.01167) -- canonical LISA PSD including galactic confusion noise
- [SLURM Job Array Support](https://slurm.schedmd.com/job_array.html) -- afterok vs afterany semantics, array task limits
- Direct codebase inspection of `parameter_estimation.py`, `LISA_configuration.py`, `bayesian_statistics.py`, `main.py`, `constants.py`, cluster scripts
