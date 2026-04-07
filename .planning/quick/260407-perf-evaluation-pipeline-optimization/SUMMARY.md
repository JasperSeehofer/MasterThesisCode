# Quick Task: Evaluation Pipeline Performance Optimization

**Date:** 2026-04-07
**Trigger:** Evaluate job (h=0.600) timed out at 15 min on bwUniCluster — pool spawn alone took 13+ min, zero detections processed.

## Problem

The Bayesian evaluation pipeline (`bayesian_statistics.py`) could not complete within the 15-minute SLURM wall time. Root causes:

1. **Pool spawn serialization (~12 min):** 126 workers each received 9,000 frozen scipy `multivariate_normal` objects + a `SimulationDetectionProbability` with 463k-row DataFrame via pickle
2. **Too many detections (4,455):** SNR threshold at 15 produced 90% poorly-localized events (SNR 15-20) with 50k-200k host galaxies each
3. **Module import thrashing:** 126 simultaneous Python startups importing numpy/scipy on shared cluster filesystem

## Changes (7 commits)

| Commit | Change | Impact |
|--------|--------|--------|
| `de86052` | Drop DataFrame from pickle + pre-warm P_det grid | -6 GB pickle, no redundant grid builds |
| `b354419` | Replace 9,000 scipy MVN objects with 6 numpy arrays; pass host scalars instead of objects | Pool initargs: ~100 MB → ~1.5 MB |
| `cf3cd89` | **[PHYSICS]** Restore SNR_THRESHOLD from 15 to 20 | 4,455 → 417 detections (well-localized only) |
| `97180a3` | Add SNR filter in evaluate() for pre-existing CRB files | Applies threshold at evaluation time |
| `a5f4869` | Switch to forkserver context + strip injection arrays | Workers inherit imports via COW |
| `6a39a63` | Add `set_forkserver_preload()` for numpy/scipy/pandas | Eliminates 126× module import on shared FS |
| `a0de491` | Switch CPU jobs to `cpu_il` partition (128 CPUs) | Larger partition, shorter queue times |

## Performance Results

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Pool spawn | >12 min | **1.7 min** | 7x |
| Detection processing | >2 hours (4,455 det) | **~5 min** (417 det) | ~25x |
| Total wall time | **DNF** (>15 min) | **7 min 16 sec** | 2x+ headroom |
| Gaussian precomputation | ~1 sec (scipy objects) | **0.06 sec** (numpy arrays) | 17x |

## Key Decisions

- **spawn → forkserver:** Required because `fork` deadlocks with BLAS/LAPACK thread locks in `scipy.integrate.quad`. `forkserver` is safe (forks from clean server, no computation before fork).
- **SNR threshold 15 → 20:** 90% of detections at SNR 15-20 had poor sky localization (median 6.6% d_L error, 50k-200k hosts each). These dominate compute time without meaningful posterior contribution. Standard LISA threshold is 20.
- **`set_forkserver_preload()`:** The dominant pool spawn cost was 126× Python + module imports on the cluster's shared filesystem, not data transfer. Preloading lets the forkserver import once, workers inherit via copy-on-write.

## Profiling Infrastructure Added

- `time.perf_counter()` instrumentation for gaussian precomputation, pool spawn, and per-detection timing with running average + ETA
- Logs via `_LOGGER.info()` — always-on in production runs

## Files Modified

- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — main pipeline changes
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py` — `__getstate__`/`__setstate__`
- `master_thesis_code/constants.py` — SNR_THRESHOLD 15 → 20
- `cluster/evaluate.sbatch` — cpu_il, 128 CPUs, 15 min
- `cluster/merge.sbatch` — cpu_il
- `cluster/combine.sbatch` — cpu_il

## Validated On

- **Dev partition test (h=0.730):** 7 min 16 sec total, 103s pool spawn, 417 detections processed
- **All tests passing:** 382 passed, ruff clean, mypy clean
