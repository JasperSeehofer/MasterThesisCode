---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Visualization Overhaul
status: executing
stopped_at: Completed 19-01-PLAN.md
last_updated: "2026-04-02T19:53:00.000Z"
last_activity: 2026-04-02 -- Plan 19-01 complete
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 7
  completed_plans: 6
  percent: 88
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Publication-quality thesis visualizations with consistent style and comprehensive uncertainty display
**Current focus:** Phase 19 — campaign dashboards & batch generation

## Current Position

Phase: 19 (campaign-dashboards) — EXECUTING
Plan: 1 of 2 (complete)
Status: Executing Phase 19
Last activity: 2026-04-02 -- Plan 01 complete

Progress: [█████████░] 88%

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
- [Phase 21]: Used StrEnum for CombinationStrategy (Python 3.13 ruff compliance)
- [Phase 21]: Physics-floor strategy falls back to exclude with logged warning (Phase 22 placeholder)
- [Phase 21]: NaN distinguishes missing events from zero-likelihood events in the array
- [Phase 21]: Lazy import of combine_posteriors inside if-block matches existing generate_figures pattern
- [Phase 21]: Integration tests use absolute path fallback for campaign data access from worktrees
- [Phase 22]: Physics floor uses min(nonzero) directly per event (not /100); all-zero events excluded
- [Phase 22]: check_overflow removed entirely (dead code); log-space accumulation handles stability
- [Phase 23-deploy-validate]: Physics-floor MAP=0.66 equals exclude MAP=0.66 (diff=0.00 < 0.05 threshold) on h_sweep_20260401 campaign — PASS
- [Phase 23-deploy-validate]: Deferred with-BH-mass validation: posteriors_with_bh_mass/ not present in h_sweep_20260401 campaign
- [Phase 23-deploy-validate]: Fast-forward merge of claudes_sidequests into main preserved linear history; cluster at 5793f70 with phases 21+22 fixes before evaluate jobs ran
- [Phase 17]: Used np.trapezoid instead of deprecated np.trapz for NumPy 2.x compatibility
- [Phase 17]: Extracted _plot_detection_heatmap private helper to share logic between P_det coordinate variants
- [Phase 17]: Widened make_colorbar type from AxesImage to ScalarMappable for contourf support
- [Phase 17]: Used list[float] for bin_edges to satisfy mypy Axes.hist type expectations
- [Phase 17]: LisaTdiConfiguration works on CPU (guarded cupy import) so decompose test runs without GPU marker
- [Phase 18]: Widened make_colorbar mappable type from AxesImage to ScalarMappable for scatter/contourf support
- [Phase 18]: Added corner to mypy ignore_missing_imports (no py.typed marker)
- [Phase 18]: astropy.stats type stubs available -- no type: ignore needed for binom_conf_interval
- [Phase 18]: Log-sum-exp used for numerical stability in posterior combination

- [Phase 19]: Used constrained_layout instead of tight_layout for subplot_mosaic with colorbars
- [Phase 19]: Figure height 5.25in (width*0.75) for Mollweide vertical space

### Pending Todos

None.

### Blockers/Concerns

- **Time-sensitive:** 22 simulation tasks remain on cluster, then merge, then evaluate — must deploy before evaluate starts
- **Physics code change:** NFIX-02 (`bayesian_statistics.py` floor) requires `/physics-change` protocol
- **"With BH mass" has 111 zero-events (21%)** — more than "without BH mass" (17 events, 3%)

## Session Continuity

Last session: 2026-04-02T19:53:00Z
Stopped at: Completed 19-01-PLAN.md
Resume file: .planning/phases/19-campaign-dashboards/19-01-SUMMARY.md
