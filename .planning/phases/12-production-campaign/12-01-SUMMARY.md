---
phase: 12-production-campaign
plan: 01
subsystem: cluster-ops
tags: [production, cluster, bwunicluster, slurm, cramer-rao-bounds]

dependency_graph:
  requires:
    - phase: 11-validation-campaign
      provides: validated corrected physics, production partition config
    - phase: 11.1-simulation-based-detection-probability
      provides: simulation-based P_det
  provides:
    - production CRB catalog (~1000+ detections, seed 200)
    - merged cramer_rao_bounds.csv + prepared_cramer_rao_bounds.csv
  affects: [13-h0-posterior-sweep, 21-analysis-post-processing]

tech_stack:
  added: []
  patterns: [cluster-slurm-array, yield-driven-campaign, rsync-results]

key_files:
  created: []
  modified:
    - cluster/simulate.sbatch (gpu_h100 partition, 1h time limit)

key_decisions:
  - "100 tasks x 50 steps = 5000 EMRI events, yield target 1000+ detections"
  - "Seed 200 — distinct from Phase 11 (seed 100) and Phase 11.1 (seed 12345)"
  - "Yield-driven completion: accepted partial task completion once detection target met"
  - "All CRBs saved unfiltered; d_L threshold filtering deferred to evaluation time"
  - "Results stored outside repo at ~/emri-results/production-campaign-seed200/"

requirements_completed: [PROD-01]

completed: 2026-04-01
tasks_completed: 6
tasks_total: 6
---

# Phase 12 Plan 01: Production Campaign Summary

**Production-scale EMRI simulation campaign (100 tasks × 50 steps, seed 200) completed on bwUniCluster with corrected physics; 1000+ detections rsynced locally for H0 inference**

## Performance

- **Completed:** 2026-04-01
- **Campaign scale:** 100 tasks × 50 steps = 5,000 EMRI events
- **Seed:** 200 (distinct from validation campaign seed 100)

## Accomplishments

- Updated `simulate.sbatch` to `gpu_h100` production partition with 1h time limit
- Pushed code to cluster and verified corrected physics (confusion noise, five-point stencil) present
- Submitted production campaign via `submit_pipeline.sh --tasks 100 --steps 50 --seed 200`
- Monitored yield (detection count from per-task CRB CSVs); yield target ~1000+ detections met
- Merge + prepare jobs completed automatically via SLURM dependency chain
- Results rsynced to `~/emri-results/production-campaign-seed200/` (outside repo)
- Verified merged CRB CSV line count, SNR range, and run_metadata reproducibility fields

## What Was Done

### Campaign submission

Physics corrections confirmed on cluster before submission:
- five_point_stencil in parameter_estimation.py ✓
- include_confusion_noise in LISA_configuration.py ✓

Submitted: `bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 200`
Three SLURM jobs chained: simulate array → merge → evaluate

### Data transfer

CRB catalog rsynced to `~/emri-results/production-campaign-seed200/simulations/`:
- `cramer_rao_bounds.csv` — merged catalog with 1000+ detections
- `prepared_cramer_rao_bounds.csv` — prepared for Bayesian inference
- `run_metadata_*.json` — reproducibility records (git commit, seed, SLURM job IDs)

## Self-Check: PASSED

- Production campaign submitted with correct parameters: ✓
- Yield target (~1000+ detections) met: ✓
- Merged and prepared CRBs available locally: ✓
- run_metadata.json records git commit and seed: ✓
- Data stored outside repo (~/emri-results/): ✓

---
*Phase: 12-production-campaign*
*Completed: 2026-04-01*
