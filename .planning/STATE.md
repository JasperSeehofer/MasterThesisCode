---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: Posterior Numerical Stability
status: defining_requirements
stopped_at: Milestone started, defining requirements
last_updated: "2026-04-02T16:00:00.000Z"
last_activity: 2026-04-02
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Fix posterior combination numerical instability and deploy before pending cluster evaluation jobs run
**Current focus:** Defining requirements for v1.4

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-04-02 — Milestone v1.4 started

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 22 (v1.0: 9, v1.1: 4, v1.2: ~9)
- Average duration: ~5 min
- Total execution time: ~2 hours

## Accumulated Context

### Decisions

- [v1.4]: "With BH mass" naive MAP=0.72 is the best current result; "without BH mass" MAP=0.86 is biased by zero-count gradient
- [v1.4]: Option 2 (per-event floor) overcorrects — both variants peak at h=0.60
- [v1.4]: Option 1 (exclude zero events) gives MAP=0.66/0.68 — clean but loses 3-21% of events
- [v1.4]: STAT-8 added to TODO.md documenting the full issue
- [v1.3]: matplotlib-only stack; sole new dependency is `corner` for parameter posteriors
- [v1.3]: `text.usetex=False` default; opt-in LaTeX toggle for final thesis figures only

### Pending Todos

None.

### Blockers/Concerns

- **Time-sensitive:** 22 simulation tasks remain on cluster, then merge, then evaluate — must deploy before evaluate starts
- **Physics code change:** `bayesian_statistics.py` modifications require `/physics-change` protocol if formulas change
- **"With BH mass" has 111 zero-events (21%)** — more than "without BH mass" (17 events, 3%)

## Session Continuity

Last session: 2026-04-02
Stopped at: Milestone v1.4 started, defining requirements
Resume file: —
