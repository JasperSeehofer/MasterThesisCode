---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: Posterior Numerical Stability
status: ready_to_plan
stopped_at: Roadmap created, ready to plan Phase 21
last_updated: "2026-04-02T17:00:00.000Z"
last_activity: 2026-04-02
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Fix posterior combination numerical instability and deploy before pending cluster evaluation jobs run
**Current focus:** Phase 21 — Analysis & Post-Processing

## Current Position

Phase: 21 of 23 (Analysis & Post-Processing)
Plan: Not started
Status: Ready to plan
Last activity: 2026-04-02 — Roadmap created for v1.4

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
- [v1.4]: Analysis largely done in conversation — Phase 21 formalizes it

### Pending Todos

None.

### Blockers/Concerns

- **Time-sensitive:** 22 simulation tasks remain on cluster, then merge, then evaluate — must deploy before evaluate starts
- **Physics code change:** NFIX-02 (`bayesian_statistics.py` floor) requires `/physics-change` protocol
- **"With BH mass" has 111 zero-events (21%)** — more than "without BH mass" (17 events, 3%)

## Session Continuity

Last session: 2026-04-02
Stopped at: Roadmap created for v1.4, ready to plan Phase 21
Resume file: ---
