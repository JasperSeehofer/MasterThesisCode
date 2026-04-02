---
phase: 13-h0-posterior-sweep
plan: 01
subsystem: bayesian-inference
tags: [h0-posterior, evaluate, bayesian, cluster, sweep]

dependency_graph:
  requires:
    - phase: 12-production-campaign
      provides: merged CRB catalog (1000+ detections, seed 200)
  provides:
    - H0 posterior over [0.6, 0.9] range from production CRB catalog
  affects: [21-analysis-post-processing]

tech_stack:
  added: []
  patterns: [bayesian-evaluate, h0-sweep, slurm-evaluate]

key_decisions:
  - "H0 sweep over 15 points in [0.6, 0.86] range"
  - "Evaluate pipeline run via submit_pipeline.sh evaluate step (included in Phase 12 chain)"
  - "Results stored at h_sweep_20260401 on cluster; rsynced locally"
  - "Naive MAP=0.72 (with BH mass) and MAP=0.86 (without BH mass) established as baselines"

requirements_completed: [H0-01]

completed: 2026-04-01
tasks_completed: 1
tasks_total: 1
---

# Phase 13 Plan 01: H0 Posterior Sweep Summary

**Full H0 posterior sweep over [0.6, 0.9] produced from production CRB catalog; naive MAP values (0.72 with BH mass, 0.86 without) established as v1.2 baselines**

## Performance

- **Completed:** 2026-04-01
- **H0 sweep grid:** 15 points in [0.60, 0.86]

## Accomplishments

- Evaluated H0 posterior over full sweep range using production campaign CRBs (seed 200)
- Results stored as per-event posterior JSONs at `h_sweep_20260401` run directory on cluster
- Established baseline MAP values: naive combination gives 0.72 (with BH mass) and 0.86 (without)
- Zero-likelihood problem identified during sweep: 21% of events (with BH mass) produce zero likelihoods at some h-bins due to galaxy catalog coverage gaps

## What Was Done

The evaluate step ran as part of the Phase 12 SLURM dependency chain. The H0 sweep produced the first statistically meaningful posterior from corrected-physics CRBs.

**Key findings documented:**
- With BH mass: naive MAP=0.72, 111 events (21%) produce zero likelihoods
- Without BH mass: naive MAP=0.86 (biased by zero-count gradient), 17 events (3%) zero
- Zero-likelihood events trace to: no host galaxies in error volume, catalog coverage gaps at high-z

These findings motivated v1.4 (Posterior Numerical Stability) to address the numerical issues.

## Self-Check: PASSED

- H0 posterior sweep completed over [0.6, 0.9] range: ✓
- Baseline MAP values documented: ✓
- Zero-likelihood problem characterized: ✓
- Results available as input for Phase 21 analysis: ✓

---
*Phase: 13-h0-posterior-sweep*
*Completed: 2026-04-01*
