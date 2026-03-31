---
gpd_state_version: 1.0
milestone: v1.2.1
milestone_name: "With BH Mass" Likelihood Bias Audit
status: ready-to-execute
last_updated: "2026-03-31"
last_activity: 2026-03-31 -- Phase 14 planned (2 plans)
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-03-30)

**Core research question:** Why does the "with BH mass" likelihood channel produce an H0 posterior biased to h=0.600, nearly 3x worse than the "without BH mass" channel?
**Current focus:** Phase 14 -- First-Principles Derivation

## Current Position

**Current Phase:** 14
**Current Phase Name:** First-Principles Derivation
**Total Phases:** 3 (14, 15, 16)
**Current Plan:** --
**Total Plans in Phase:** 2
**Status:** Ready to execute
**Last Activity:** 2026-03-31
**Last Activity Description:** Phase 14 planned (2 plans in 2 waves)

**Progress:** [░░░░░░░░░░] 0%

## Intermediate Results

- "Without BH mass" posterior peak: h=0.678 (P_det=1 baseline)
- "With BH mass" posterior peak: h=0.600 (P_det=1 baseline)
- Single-galaxy likelihood peaks at h=0.73 (formula correct in isolation) -- debug session 2026-03-30
- BH mass Gaussian index [0] vs [1] has no effect under delta-function approximation -- quick task 260330-twe
- Analytic M_z marginalization (commit 15b49a3) did not change peak location

## Open Questions

- [v1.2.1] Is the `/(1+z)` at line 679 a double-counted Jacobian after the analytic marginalization refactor?
- [v1.2.1] Is the sky localization weight placed correctly (in exactly one likelihood factor)?
- [v1.2.1] Is the "with BH mass" denominator consistent with the numerator?
- [v1.2.1] Does the analytic M_z marginalization (commit 15b49a3) introduce any other issues?

## Accumulated Context

### Decisions

- [v1.2.1]: Keep analytic M_z marginalization (commit 15b49a3) -- correct physics
- [v1.2.1]: P_det out of scope -- handled in Phase 11.1
- [v1.2.1]: Derive from d_L-only literature + extend to M_z
- [v1.2.1]: Use "without BH mass" channel (h=0.678) as reference baseline

### Blockers/Concerns

- `/(1+z)` on line 679 is suspected double-counted Jacobian -- needs derivation to confirm
- Sky localization weight placement flagged by two TODOs -- needs audit
- "With BH mass" denominator uses MC sampling while numerator uses quadrature -- potential inconsistency

## Session Continuity

**Last session:** 2026-03-30
**Stopped at:** Roadmap created -- ready to plan Phase 14
