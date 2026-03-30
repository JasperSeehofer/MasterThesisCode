---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Production Campaign & Physics Corrections
status: executing
stopped_at: Phase 11.1 context gathered
last_updated: "2026-03-30T21:04:38.474Z"
last_activity: 2026-03-30 -- Phase 11 execution started
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 4
  completed_plans: 3
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-29)

**Core value:** The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramer-Rao bounds for statistically meaningful Hubble constant posteriors.
**Current focus:** Phase 11 — validation-campaign

## Current Position

Phase: 11 (validation-campaign) — EXECUTING
Plan: 1 of 2
Status: Executing Phase 11
Last activity: 2026-03-30 -- Phase 11 execution started

Progress: [░░░░░░░░░░] 0% (v1.2: 0/5 phases)

## Performance Metrics

**Velocity:**

- Total plans completed: 13 (v1.0: 9, v1.1: 4)
- Average duration: ~5 min
- Total execution time: ~1 hour

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1]: 10% d_L threshold was workaround for forward-diff -- revisit after stencil upgrade
- [v1.1]: 30s waveform timeout -- must increase for 5-point stencil
- [v1.2 Roadmap]: Confusion noise before stencil -- lower risk, enables independent validation
- [Phase 11]: Phase 10 merged with --no-ff; comparison script uses P90 d_L threshold recommendation

### Roadmap Evolution

- Phase 11.1 inserted after Phase 11: Simulation-Based Detection Probability (URGENT) — large-scale SNR-only campaign to build P_det(z,h) grid, replacing KDE-based DetectionProbability

### Pending Todos

None.

### Blockers/Concerns

- CRB timeout (30s) will fire on nearly every event after stencil upgrade -- must increase in Phase 10
- Confusion noise will reduce detection yield -- campaign size must be calibrated in Phase 11
- Fisher matrix ill-conditioning may emerge with better derivatives -- condition number monitoring needed
- **H0 posterior bias** — Debug session confirmed formula is correct; residual bias is from P_det=1 (disabled). Phase 11.1 will build simulation-based P_det to resolve this.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260330-oaf | Diagnostic bias fix: remove /d_L factor and disable P_det in Pipeline B likelihood | 2026-03-30 | ae118d4 | [260330-oaf-diagnostic-bias-fix-remove-d-l-factor-an](./quick/260330-oaf-diagnostic-bias-fix-remove-d-l-factor-an/) |
| 260330-otu | Condense CLAUDE.md from ~846 to ~426 lines | 2026-03-30 | b47b46e | [260330-otu-condense-claude-md-remove-redundancy-tri](./quick/260330-otu-condense-claude-md-remove-redundancy-tri/) |
| 260330-ojq | Re-run evaluation with diagnostic fix: peak shifted h=0.600->0.678, 60% bias reduction | 2026-03-30 | 8013749 | [260330-ojq-re-run-evaluation-pipeline-with-h-value-](./quick/260330-ojq-re-run-evaluation-pipeline-with-h-value-/) |
| 260330-twe | Re-run evaluation with BH mass Gaussian index fix: no change (delta-function approx nullifies fix) | 2026-03-30 | ab77e70 | [260330-twe-re-run-h-value-sweep-evaluation-with-bh-](./quick/260330-twe-re-run-h-value-sweep-evaluation-with-bh-/) |

## Session Continuity

Last session: 2026-03-30T21:04:38.473Z
Stopped at: Phase 11.1 context gathered
Resume file: .planning/phases/11.1-simulation-based-detection-probability/11.1-CONTEXT.md
