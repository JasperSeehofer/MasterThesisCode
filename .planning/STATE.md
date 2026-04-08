---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Paper
status: planning
stopped_at: Phase 30 context gathered
last_updated: "2026-04-08T09:37:27.384Z"
last_activity: 2026-04-08 — Roadmap defined (5 phases, 11 requirements mapped)
progress:
  total_phases: 9
  completed_phases: 1
  total_plans: 1
  completed_plans: 1
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-08)

**Core value:** Measure H0 from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.
**Current focus:** v2.1 H0 Bias Resolution — diagnose and fix per-event posterior bias (MAP h=0.66 vs true h=0.73), testing each fix in isolation.

## Current Position

Phase: 30 — Baseline & Evaluation Infrastructure
Plan: —
Status: Ready to plan
Last activity: 2026-04-08 — Roadmap defined (5 phases, 11 requirements mapped)

## Performance Metrics

**Velocity:**

- Total plans completed: 43 (v1.0: 9, v1.1: 4, v1.2: 12, v1.3: 11, v1.4: 5, v1.5: 2)
- Total phases: 33 (28 complete across v1.0-v1.5 + v2.0 Phase 26; 5 planned for v2.1 BiasRes)

## Accumulated Context

### Pending Todos

- Fix galaxy catalog unconditional init blocking generate-figures (`main.py:49`)

### Blockers/Concerns

- v2.0 Paper paused: posterior bias must be resolved before production run results are meaningful
- v2.1 Publication Figures paused (phases 35-38): depends on bias resolution

### Key Context for v2.1 H0 Bias Resolution

- **Bias diagnosis:** MAP h=0.66 (without BH mass) / h=0.68 (with BH mass), true h=0.73. 531 events, bias compounds exponentially.
- **Root cause hypothesis:** Completion term L_comp dominates (79% weight, GLADE completeness ~21% at EMRI distances). dVc/dz volume prior in L_comp introduces systematic low-h preference.
- **P_det fix partial:** Commit 44d5358 fixed fill_value=0.0->None, reduced bias -9.2%->-6.9%. Residual remains.
- **Debug artifacts:** `.planning/debug/h0-inference-worsening.md`, `.gpd/debug/h0-posterior-bias-worsening.md`
- **Strategy:** Test fixes ONE AT A TIME with before/after posterior comparison (MAP h, width, bias %).
- **Phase 32 is physics:** Completion term prior change requires `/physics-change` protocol and GPD execution.

### Phase Notes (v2.1 H0 Bias Resolution)

**Phase 30 (Baseline & Evaluation Infrastructure) — READY:**

- Capture MAP h, 68% CI, bias % from current pipeline as baseline JSON
- Build before/after comparison report (human + machine readable)
- Store baseline in `.planning/debug/`

**Phase 31 (Catalog-Only Diagnostic) — PENDING Phase 30:**

- Run with f_i=1.0 to confirm L_comp hypothesis
- Add per-event diagnostic logging (L_cat, L_comp, f_i, log-likelihood per h)

**Phase 32 (Completion Term Fix) — PENDING Phase 31, GPD/physics:**

- Replace dVc/dz with EMRI-rate-weighted prior
- Invoke `/physics-change` before implementation

**Phase 33 (P_det Grid Resolution) — PENDING Phase 30:**

- Increase 30->60 d_L bins, make configurable
- Validate 4-sigma coverage for >95% of events

**Phase 34 (Fisher Matrix Quality) — PENDING Phase 30:**

- Remove allow_singular=True
- Regularize or exclude near-singular matrices, log condition numbers

### Phase Notes (v2.0 — paused)

**Phase 26 (Paper Draft) — COMPLETE:**

- All sections drafted: Introduction, Method (12 equations), Results (4 equations, 4 figure placeholders), Discussion, Conclusions, Appendix A
- 25 RESULT PENDING markers awaiting production run

**Phase 27 (Production Run & Figures) — PAUSED:**

- Blocked on bias resolution (v2.1) — results are meaningless with 7% bias

**Phase 28 (Review & Submission) — PAUSED:**

- Depends on Phase 27

## Session Continuity

Last session: 2026-04-08T09:37:27.383Z
Stopped at: Phase 30 context gathered
Resume file: .planning/phases/30-baseline-evaluation-infrastructure/30-CONTEXT.md

## Quick Tasks Completed

| Date | Task | Commits | Summary |
|------|------|---------|---------|
| 2026-04-07 | Evaluation pipeline performance | de86052..a0de491 (7 commits) | Pool spawn 12 min->1.7 min, total 7:16 per h-value. forkserver+preload, numpy arrays, SNR filter, cpu_il partition. |
| 2026-04-07 | Add interactive Plotly figures to GitHub Pages | 8b47b5f..33e1c86 (2 commits) | 4 Plotly HTML figures (posterior, sky map, Fisher ellipses, convergence), --generate_interactive CLI flag, CI Pages deployment, landing page. |
