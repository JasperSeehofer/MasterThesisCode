---
phase: 11-validation-campaign
plan: 02
subsystem: cluster-ops
tags: [validation, cluster, bwunicluster, comparison-report, campaign]

dependency_graph:
  requires:
    - phase: 11-01
      provides: compare_validation_runs.py script
    - phase: 09-confusion-noise
      provides: confusion noise in PSD
    - phase: 10-five-point-stencil-derivatives
      provides: five-point stencil derivatives
  provides:
    - validation campaign results on bwUniCluster (seed 100, 3 tasks, 10 steps)
    - comparison report v1.1 vs v1.2 CRBs
  affects: [12-production-campaign]

tech_stack:
  added: []
  patterns: [cluster-slurm-array, rsync-results, comparison-report]

key_files:
  created:
    - evaluation/run_v12_validation/comparison_report.md
  modified:
    - cluster/simulate.sbatch (production partition config)

key_decisions:
  - "Validation campaign confirmed corrected physics (confusion noise + 5-point stencil) produces valid CRBs"
  - "P90 fractional d_L error recommended as threshold (per comparison report)"
  - "simulate.sbatch switched from dev_gpu_h100 to gpu_h100 for production use"

requirements_completed: [SIM-01, SIM-03]

completed: 2026-04-01
tasks_completed: 3
tasks_total: 3
---

# Phase 11 Plan 02: Validation Campaign Summary

**Corrected-physics validation campaign completed on bwUniCluster; comparison report confirms v1.2 CRBs are valid and ready for production scale-up**

## Performance

- **Completed:** 2026-04-01
- **Tasks:** 3/3 (submit, wait for SLURM completion, rsync + compare)

## Accomplishments

- Pushed claudes_sidequests (with Phase 9 + 10 corrections) to cluster and updated venv
- Submitted validation campaign: 3 tasks × 10 steps, seed 100, via `submit_pipeline.sh`
- All SLURM jobs completed (simulate array, merge, evaluate chain)
- Results rsynced to local `evaluation/run_v12_validation/`
- Comparison report produced via `scripts/compare_validation_runs.py` covering detection rate, SNR distribution, CRB analysis, and pass/fail summary
- simulate.sbatch updated from `dev_gpu_h100` to `gpu_h100` production partition

## What Was Done

### Physics verification on cluster

Confirmed on cluster before submission:
- `signal.alarm(90)`: 2 occurrences in main.py ✓
- `include_confusion_noise`: present in LISA_configuration.py ✓
- `use_five_point_stencil`: present in parameter_estimation.py ✓

### Comparison report

Report at `evaluation/run_v12_validation/comparison_report.md` covering:
1. Run Metadata
2. Detection Rate (v1.1 vs v1.2)
3. SNR Distribution
4. CRB Analysis (fractional d_L error stats + percentiles)
5. d_L Threshold Recommendation (P90)
6. Wall Time Analysis
7. Error Analysis
8. Pass/Fail Summary

### Outcome

Corrected physics validated at small scale. Results sufficient to proceed to production campaign (Phase 12) with confidence.

## Self-Check: PASSED

- Validation campaign submitted and completed on bwUniCluster: ✓
- Results rsynced locally: ✓
- Comparison report produced: ✓
- simulate.sbatch updated for production partition: ✓

---
*Phase: 11-validation-campaign*
*Completed: 2026-04-01*
