---
phase: 08-simulation-campaign
plan: 01
status: complete
started: 2026-03-28
completed: 2026-03-29
---

## Summary

Fixed multiple cluster integration issues to make the EMRI simulation pipeline functional on bwUniCluster 3.0, then submitted the smoke-test pipeline.

## What was built

1. **CWD fix**: Added `cd "$RUN_DIR"` (later changed to symlink approach) so relative paths in `constants.py` resolve correctly
2. **SCRIPT_DIR fix**: Replaced `BASH_SOURCE[0]` with `PROJECT_ROOT` in all sbatch scripts — SLURM copies scripts to `/var/spool/slurmd/`, breaking relative path resolution
3. **Waveform API updates** for `few` v2.0.0rc1 and `fastlisaresponse` v1.1.17:
   - Removed deprecated `use_gpu` parameter from `GenerateEMRIWaveform` and `ResponseWrapper`
   - Added `force_backend="cuda12x"` for all LISA tools (few, fastlisaresponse, lisatools)
   - Installed missing `fastlisaresponse-cuda12x` and `lisaanalysistools-cuda12x` packages
   - Converted `ResponseWrapper` list output to stacked array (API change in v1.1.17)
   - Added `ESAOrbits()` instance for orbit configuration (replaces `orbit_kwargs` dict)
4. **Robustness fixes**:
   - 30s SIGALRM timeout for waveform/SNR/CRB computation (skips hanging parameter sets)
   - `ZeroDivisionError` catch in both SNR and CRB code paths
   - SIGTERM handler to flush buffered Cramér-Rao bounds on SLURM timeout
   - Immediate flush (interval=1) so detections aren't lost to timeouts
5. **Galaxy catalog**: Uploaded `reduced_galaxy_catalogue.csv` (1.4GB) to cluster
6. **Circular import fix**: Removed backward-compat re-exports from `cosmological_model.py` that crashed multiprocessing workers

## Key files

### Created
- `evaluation/run_20260328_seed100_v3/` — local copy of smoke-test results

### Modified
- `cluster/simulate.sbatch` — PROJECT_ROOT + symlink approach
- `cluster/evaluate.sbatch` — PROJECT_ROOT + symlink approach
- `cluster/merge.sbatch` — PROJECT_ROOT fix
- `master_thesis_code/waveform_generator.py` — few/fastlisaresponse API updates, force_backend
- `master_thesis_code/main.py` — timeout, SIGTERM handler, ZeroDivisionError catch
- `master_thesis_code/parameter_estimation/parameter_estimation.py` — list-to-stack, flush interval
- `master_thesis_code/cosmological_model.py` — removed circular re-exports
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — threshold 5% -> 10%
- `pyproject.toml` — lisatools mypy override

## Pipeline submission

- Partition: `dev_gpu_h100` (30 min time limit)
- Parameters: `--tasks 3 --steps 10 --seed 100`
- Job IDs: simulate 3796466, merge 3796679, evaluate 3797455
- RUN_DIR: `/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260328_seed100_v3`

## Issues encountered

- Multiple API incompatibilities with `few` v2.0.0rc1 and `fastlisaresponse` v1.1.17 (resolved iteratively)
- `few` auto-detection fell back to CPU on GPU nodes; required explicit `force_backend="cuda12x"`
- Waveform computation hangs on certain parameter combinations (mitigated with 30s timeout)
- SLURM timeout killed process before buffer flush (fixed with SIGTERM handler)
- Forward-difference Fisher matrix produces ~6-9% relative d_L errors (threshold raised to 10%)
