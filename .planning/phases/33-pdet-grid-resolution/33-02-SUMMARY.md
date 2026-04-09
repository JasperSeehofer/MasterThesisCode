---
phase: 33-pdet-grid-resolution
plan: 02
status: complete
completed_tasks: [1, 2]
pending_tasks: []
---

# Plan 02 Summary: Cluster Scripts & Comparison

## Completed

### Task 1: Cluster script updates
- **evaluate.sbatch**: Added explicit `--pdet_dl_bins 60 --pdet_mass_bins 40` flags for reproducibility
- **combine.sbatch**: Added `--save_baseline` step after combining posteriors (saves baseline.json to both project root and $RUN_DIR)
- **combine.sbatch**: Added `--generate_figures` step to produce PDF figures on-cluster, avoiding need to rsync ~600 MB/h of raw posterior data

**Commit:** `4b39979` — feat(33-02): cluster scripts: 60-bin P_det grid, baseline save, on-cluster figures

### Task 2: Human verification checkpoint — COMPLETE

Full 38-point h-sweep completed on cluster (jobs 3897259 + retry 3898710 + combine 3898840).

Results:
1. 3 tasks (17/25/26) initially failed due to NFS symlink race — resubmitted sequentially (`%1`), all passed
2. Figures generated in `$RUN_DIR/simulations/figures/` and rsynced locally
3. Baseline saved: MAP h=0.740, bias=+1.4%
4. **Zero delta** vs 30-bin baseline — log-posteriors identical to 4 decimal places across all 38 h-values, both variants (with and without BH mass)

**Conclusion:** P_det grid resolution is not a source of bias. Phase 33 validated.

## Deviations
- **Scope change**: User requested full cluster sweep instead of local single-h comparison. Added baseline saving and on-cluster figure generation to combine.sbatch so figures can be rsynced directly without transferring raw posteriors.
