---
phase: 33-pdet-grid-resolution
plan: 02
status: partial
completed_tasks: [1]
pending_tasks: [2]
---

# Plan 02 Summary: Cluster Scripts & Comparison

## Completed

### Task 1: Cluster script updates
- **evaluate.sbatch**: Added explicit `--pdet_dl_bins 60 --pdet_mass_bins 40` flags for reproducibility
- **combine.sbatch**: Added `--save_baseline` step after combining posteriors (saves baseline.json to both project root and $RUN_DIR)
- **combine.sbatch**: Added `--generate_figures` step to produce PDF figures on-cluster, avoiding need to rsync ~600 MB/h of raw posterior data

**Commit:** `4b39979` — feat(33-02): cluster scripts: 60-bin P_det grid, baseline save, on-cluster figures

## Pending

### Task 2: Human verification checkpoint
User will run the full 38-point h-sweep on the cluster and verify:
1. P_det grid coverage > 95% in evaluation logs
2. Figures generated successfully in `$RUN_DIR/simulations/figures/`
3. Baseline saved to `$RUN_DIR/baseline.json`
4. Visual comparison of posteriors (combined + single-event) against previous 30-bin results

## Deviations
- **Scope change**: User requested full cluster sweep instead of local single-h comparison. Added baseline saving and on-cluster figure generation to combine.sbatch so figures can be rsynced directly without transferring raw posteriors.
