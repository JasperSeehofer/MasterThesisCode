# Technology Stack

**Project:** v1.2 Production Campaign & Physics Corrections
**Researched:** 2026-03-29

## Key Finding: No New Dependencies Required

All four target features (5-point stencil, confusion noise, production campaign, H0 sweep) are implementable with the existing stack. No new Python packages, no version bumps, no infrastructure additions.

This is the correct outcome for a physics-correction milestone on an established codebase.

## Recommended Stack Changes

### New Dependencies

None.

### Version Changes

None required. The existing pinned versions are sufficient:
- `numpy` -- array math for stencil and PSD, already used everywhere
- `scipy` -- integration/interpolation for Bayesian inference, already used
- `cupy-cuda12x` -- GPU arrays, already used for Fisher matrix computation
- `pandas` -- CSV I/O, already used for Cramer-Rao bounds

### Configuration Changes

| Change | File | Purpose | Why |
|--------|------|---------|-----|
| Add `--h_range` CLI arg | `arguments.py` | Accept H0 sweep bounds (e.g. `0.6 0.9`) | Current `--h_value` takes a single float; sweep needs a range |
| Add `--h_steps` CLI arg | `arguments.py` | Number of H0 grid points (e.g. 31) | Controls resolution of posterior sweep |
| Confusion noise observation time | `LISA_configuration.py` | T_obs parameter for time-dependent confusion noise | Babak et al. (2023) Eq. 17 coefficients depend on mission duration |

## What Exists and Why It's Sufficient

### 5-Point Stencil Fisher Matrix

**Already implemented** at `parameter_estimation.py:187-243` (`five_point_stencil_derivative()`). The method exists but is never called -- `compute_fisher_information_matrix()` at line 339 calls `finite_difference_derivative()` instead.

**Fix is a targeted refactor** (line 339): replace `self.finite_difference_derivative()` with a loop calling `self.five_point_stencil_derivative()` per parameter, collecting derivatives into the same dict format.

**No new math libraries needed.** The 5-point stencil formula `(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h` is already correctly implemented using CuPy array arithmetic.

**Performance note:** The stencil requires 4 waveform evaluations per parameter (vs 1 for forward-diff), so derivative computation goes from 14 waveform calls to 56. The inner product count stays at 105 (symmetric matrix). GPU time per simulation step will roughly 4x for the derivative phase, but the inner product phase (the actual bottleneck) is unchanged. On H100 GPUs with the existing 30s waveform timeout, this adds ~2-3 minutes per Fisher matrix.

**Integration concern:** The existing `five_point_stencil_derivative()` has a different signature than `finite_difference_derivative()` -- it takes a single `Parameter` and returns one derivative, while the forward-diff version loops internally and returns a dict. The refactor must bridge this gap, either by adapting the 5-point method to loop, or by calling it per-parameter and assembling the dict.

### Galactic Confusion Noise PSD

**Constants already defined** in `constants.py:74-82` (`LISA_PSD_A`, `LISA_PSD_ALPHA`, `LISA_PSD_F2`, `LISA_PSD_A1`, `LISA_PSD_B1`, `LISA_PSD_AK`, `LISA_PSD_BK`) citing arXiv:2303.15929 Eq. 17.

**Implementation location:** Add `S_conf(f, T_obs)` method to `LisaTdiConfiguration` and sum it into `power_spectral_density_a_channel()`. The confusion noise formula is:

```
S_conf(f) = A * f^(-7/3) * exp(-f^alpha + beta * f * sin(kappa * f)) * [1 + tanh(gamma * (f_k - f))]
```

where `alpha`, `beta`, `gamma`, `f_k` are functions of observation time T_obs. This uses only `xp.exp`, `xp.sin`, `xp.tanh`, `xp.power` -- all available in both NumPy and CuPy.

**No new dependencies.** Pure array math with existing `_get_xp()` pattern.

**Reference chain:** The standard LISA sensitivity curve with confusion noise follows Robson, Cornish & Liu (2019), arXiv:1803.01944. The constants in `constants.py` cite Babak et al. (2023), arXiv:2303.15929, which uses a compatible parameterization. The codebase has already chosen the Babak parameterization -- use it consistently.

**PSD cache impact:** `ParameterEstimation._psd_cache` caches PSD per waveform length `n`. Adding confusion noise changes PSD values but cache keys (based on `n`) remain valid. The cache just stores different (larger) PSD values. No invalidation needed.

### Production Campaign (100+ tasks)

**Infrastructure already exists.** `submit_pipeline.sh --tasks 100 --steps 50 --seed 42` works today. The v1.1 smoke test validated the full pipeline with 3 tasks.

**Scaling considerations (no stack changes needed):**
- SLURM array jobs scale linearly -- 100 tasks is just `--array=0-99`
- Each task is independent (no MPI, no shared state)
- Merge script handles arbitrary numbers of per-task CSVs
- Workspace storage: ~100KB per task (CSV rows) = ~10MB total, well within limits

**Potential cluster-side adjustment:** If the GPU partition limits concurrent array tasks (common on shared clusters), SLURM's `%throttle` syntax handles this: `--array=0-99%20` runs 20 at a time. This is a submission flag, not a code change.

**Timing estimate (from v1.1 smoke test):** Each task with 10 steps took ~15-20 minutes. With the 5-point stencil adding ~4x to the derivative phase, expect ~25-35 minutes per 10-step task. For 50 steps per task, budget ~2-3 hours per task within the 72h `gpu_h100` walltime limit.

### H0 Posterior Sweep

**Current state:** `--evaluate --h_value 0.73` evaluates at a single H0 point. The `BayesianStatistics.evaluate()` method processes one h-value per invocation.

**Sweep strategy -- two options, recommend Option A:**

**Option A: SLURM array over h-values (recommended)**
Submit evaluate as an array job where each task evaluates one h-value. No code changes to `BayesianStatistics` needed. A new sweep sbatch script generates h-values from the array task ID:

```bash
# evaluate_sweep.sbatch
H_MIN=0.60
H_MAX=0.90
H_STEPS=31
H=$(python3 -c "print(round($H_MIN + $SLURM_ARRAY_TASK_ID * ($H_MAX - $H_MIN) / ($H_STEPS - 1), 4))")
python -m master_thesis_code "$RUN_DIR" --evaluate --h_value $H
```

This parallelizes trivially. Each evaluation takes ~minutes on 16 CPUs. A 31-point grid (0.60 to 0.90, step 0.01) runs in the time of one evaluation.

**Option B: In-process loop**
Add `--h_range 0.6 0.9 --h_steps 31` to iterate inside `evaluate()`. Simpler submission but sequential -- 31x slower wall-clock time.

**Recommended: Option A.** It requires only a new sbatch script (`cluster/evaluate_sweep.sbatch`) and a post-processing script to combine per-h posteriors. No changes to the core Python evaluation code. The SLURM dependency chain becomes: simulate -> merge -> evaluate_sweep (array).

**Post-processing:** Combine per-h JSON posteriors into a single posterior curve. This is a simple script reading JSON files and assembling `{h: likelihood}` pairs. Uses only `json`, `numpy`, `pathlib` -- all already available.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Stencil method | 5-point (O(h^4)) | Automatic differentiation (JAX) | Overkill; 5-point stencil already implemented; AD would require rewriting waveform pipeline |
| Stencil method | 5-point (O(h^4)) | Complex-step derivative | Would require complex-valued waveform generator support, which `few` does not provide |
| Confusion noise | Analytic fit (Babak 2023) | Numerical foreground subtraction | Out of thesis scope; analytic fit is standard practice in EMRI Fisher matrix literature |
| H0 sweep | SLURM array per h-value | Dask/multiprocessing sweep | Array jobs are simpler, already proven infrastructure |
| H0 sweep | SLURM array per h-value | In-process `--h_range` loop | Sequential; wastes cluster CPU allocation |
| Production scale | 100 tasks x 50 steps | Fewer tasks with more steps | More tasks = better fault tolerance (one timeout loses fewer events) |

## What NOT to Add

| Library | Why Tempting | Why Wrong |
|---------|-------------|-----------|
| JAX / autograd | "Better derivatives" | Would require rewriting `few`/`fastlisaresponse` interface; 5-point stencil is standard for Fisher matrices in GW literature (Vallisneri 2008) |
| h5py / HDF5 | "Better data format" | CSV works fine at this scale (~10MB total); adding format complexity for a thesis is not justified |
| Dask | "Parallel H0 sweep" | SLURM array jobs are simpler and already work |
| tqdm | "Progress bars" | Logging is sufficient; adds unnecessary dependency |
| corner / chainconsumer | "Posterior plotting" | matplotlib already handles 1D posterior plots; no MCMC chains to visualize for H0 grid evaluation |
| numdifftools | "Validated numerical derivatives" | 5-point stencil is already implemented correctly; adding a dependency for one formula is not justified |

## Integration Points

### 5-Point Stencil Integration
- **Touches:** `parameter_estimation.py` (line 339: swap derivative call; refactor `five_point_stencil_derivative` to match dict return format)
- **Bounds check:** 5-point stencil needs `value +/- 2*epsilon` in bounds (already checked in existing method at line 210-215)
- **GPU memory:** 4x more waveforms generated (56 vs 14), but each is immediately processed and can be freed. The existing `del` pattern in `five_point_stencil_derivative` handles this.
- **Debug prints:** Existing method uses `print()` (lines 199, 230) instead of `_LOGGER`. Clean up during integration.
- **Test strategy:** Compare forward-diff vs 5-point Fisher matrix on a known parameter set; verify 5-point has smaller numerical error and CRB values change only modestly (same order of magnitude).

### Confusion Noise Integration
- **Touches:** `LISA_configuration.py` (add `S_conf()` method, modify `power_spectral_density_a_channel()` to sum `S_conf` into existing PSD)
- **Touches:** `constants.py` only if T_obs needs to be added (may already be sufficient with `ParameterEstimation.T`)
- **PSD cache:** Valid -- same n maps to same (now larger) PSD values. No cache invalidation needed.
- **Test strategy:** PSD with confusion noise > PSD without, especially at 0.1-3 mHz. SNR should decrease (more noise). Verify confusion noise is negligible above ~5 mHz.

### H0 Sweep Integration
- **New files:** `cluster/evaluate_sweep.sbatch`, `scripts/combine_posteriors.py`
- **Touches:** `cluster/submit_pipeline.sh` (add sweep stage after merge, replacing single-point evaluate)
- **Does NOT touch:** `bayesian_statistics.py` (each h-value evaluation is independent)
- **Output structure:** Each array task writes to `simulations/posteriors/posterior_h_0.XX.json`. Combine script reads all and produces `simulations/posteriors/h0_posterior_combined.json`.

## Installation

No changes to installation commands:

```bash
# Dev machine (no GPU)
uv sync --extra cpu --extra dev

# Cluster (GPU, CUDA 12)
uv sync --extra gpu
```

## Sources

- [Robson, Cornish & Liu (2019), arXiv:1803.01944](https://arxiv.org/abs/1803.01944) -- LISA sensitivity curves and confusion noise parameterization
- [Babak et al. (2023), arXiv:2303.15929](https://arxiv.org/abs/2303.15929) -- PSD parameterization used in constants.py
- [Vallisneri (2008), arXiv:gr-qc/0703086](https://arxiv.org/abs/gr-qc/0703086) -- Fisher matrix numerical derivatives, 5-point stencil recommendation
- Existing codebase: `parameter_estimation.py`, `LISA_configuration.py`, `constants.py`, `bayesian_statistics.py`, `submit_pipeline.sh`, `evaluate.sbatch`

---

*Stack analysis: 2026-03-29 -- v1.2 milestone (physics corrections + production campaign)*
