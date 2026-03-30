---
gpd_state_version: 1.0
milestone: v1.2.1
milestone_name: "With BH Mass" Likelihood Bias Audit
status: defining-objectives
last_updated: "2026-03-30"
last_activity: 2026-03-30 -- Milestone initialization
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-03-30)

**Core research question:** Why does the "with BH mass" likelihood channel produce an H0 posterior biased to h=0.600, nearly 3x worse than the "without BH mass" channel?
**Current focus:** Defining research objectives

## Current Position

Phase: — (milestone initialization)
Status: Defining objectives
Last activity: 2026-03-30 -- Milestone initialization

Progress: [░░░░░░░░░░] 0%

## Accumulated Context

### Decisions

- [v1.2.1]: Keep analytic M_z marginalization (commit 15b49a3) — correct physics
- [v1.2.1]: P_det out of scope — handled in Phase 11.1
- [v1.2.1]: Derive from d_L-only literature + extend to M_z

### Blockers/Concerns

- `/(1+z)` on line 679 is suspected double-counted Jacobian — needs derivation to confirm
- Sky localization weight placement flagged by two TODOs — needs audit
- "With BH mass" denominator uses MC sampling while numerator uses quadrature — potential inconsistency

## Session Continuity

Last session: 2026-03-30
Stopped at: Milestone initialization — defining objectives
