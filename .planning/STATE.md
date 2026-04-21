---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Pipeline Correctness
status: defining_requirements
stopped_at: Milestone started — requirements next
last_updated: "2026-04-21T00:00:00.000Z"
last_activity: 2026-04-21
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.
**Current focus:** v2.2 Pipeline Correctness — remediation of 10 pre-batch audit findings before next cluster campaign.

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-04-21 — Milestone v2.2 started following pre-batch audit

## Performance Metrics

**Velocity:**

- Total plans completed: 55 (v1.0: 9, v1.1: 4, v1.2: 12, v1.3: 11, v1.4: 5, v1.5: 2, v2.1: 10, v2.0 paused)
- Total phases completed: 33 (28 across v1.0–v1.5 + v2.0 Phase 26 + v2.1 Phases 30–34)

## Accumulated Context

### Pending Todos

- Fix galaxy catalog unconditional init blocking generate-figures (`main.py:49`) — carried from v2.1

### Blockers/Concerns

- v2.0 Paper paused: posterior re-evaluation required under v2.2 corrected frame before production run figures
- v2.1 Publication Figures paused (originally phases 35–38): depends on v2.2 correctness gate

### Key Context for v2.2 Pipeline Correctness

- **Audit date:** 2026-04-21
- **Audit axes:** statistical/Bayesian, physical/coordinate, HPC/visualization (3 parallel Explore agents + direct code verification)
- **Plan artifact:** `~/.claude/plans/i-want-a-last-elegant-feather.md`
- **Memory artifacts:** `project_coordinate_bugs.md`, `project_audit_2026_04_21.md`

**Critical findings:**

1. **Coordinate frame (CRITICAL):**
   - `galaxy_catalogue/handler.py:286–291, 307–309`: BallTree uses latitude embedding `(cos θ cos φ, cos θ sin φ, sin θ)` on polar-angle data → all ecliptic-equator galaxies collapse to `(0, 0, 1)`.
   - `handler.py:486–492`: GLADE J2000 equatorial (RA/Dec) stored as ecliptic `qS/phiS` with no rotation. Waveform treats them as ecliptic SSB → up to 23.4° bias.
   - Self-consistent within sim+eval but geometrically broken.

2. **Statistical:**
   - L_cat uses `ΣN_g/ΣD_g` not Gray Eq. 24-25's `(1/N)Σ(N_g/D_g)` (`bayesian_statistics.py:929-944`).
   - P_det numerator NN-fill, denominator zero-fill asymmetry (`simulation_detection_probability.py:454–463` vs `:672–717`; `bayesian_statistics.py:119`).

3. **Parameter Estimation:**
   - `set_host_galaxy_parameters` hardcodes h=0.73 via `dist()` default (`parameter_space.py:148`).
   - `derivative_epsilon=1e-6` uniform across 14 params of wildly different scale.

4. **Hygiene / HPC:**
   - `LamCDMScenario.Omega_m`: lower=0.5, upper=0.04 (swapped).
   - `SNR_THRESHOLD=20` vs `snr_threshold=15` vs `0.3×15=4.5` in different places.
   - `parameter_estimation.py` has unguarded `cp.*` → not CPU-importable.
   - `_crb_flush_interval=1` + FFT cache cleared every iteration.

**User decisions (from AskUserQuestion):**
- Coord: fix + re-evaluate existing CRBs (no re-simulation).
- L_cat: prove equivalence (or fix if not equivalent).
- Injection: staged — densify M×z×d_L first, then sky-dependent P_det.
- HPC/viz: safe wins only.

**Abort gate before cluster submission:** if re-evaluated MAP at h=0.73 shifts >5%, pause and investigate.

### Phase Notes (v2.1 — shipped)

Phase 30 (Baseline/comparison), Phase 31 (Catalog-only diagnostic), Phase 32 (L_comp full-volume D(h), GPD, /physics-change), Phase 33 (P_det 60-bin validation), Phase 34 (Fisher condition-number gate) — all shipped 2026-04-09. MAP 0.73, bias 0.0% at N=59 with h=0.73 injections.

### Phase Notes (v2.0 — paused)

**Phase 26 (Paper Draft) — COMPLETE:**
- All sections drafted: Introduction, Method, Results, Discussion, Conclusions, Appendix A
- 25 RESULT PENDING markers awaiting post-v2.2 re-evaluation

**Phases 27–28 — PAUSED:** blocked on v2.2 correctness gate.

## Session Continuity

Last session: 2026-04-21
Stopped at: Milestone v2.2 started — awaiting roadmap
Resume file: None

## Quick Tasks Completed

| Date | Task | Commits | Summary |
|------|------|---------|---------|
| 2026-04-07 | Evaluation pipeline performance | de86052..a0de491 (7 commits) | Pool spawn 12 min→1.7 min, total 7:16 per h-value. forkserver+preload, numpy arrays, SNR filter, cpu_il partition. |
| 2026-04-07 | Add interactive Plotly figures to GitHub Pages | 8b47b5f..33e1c86 (2 commits) | 4 Plotly HTML figures (posterior, sky map, Fisher ellipses, convergence), --generate_interactive CLI flag, CI Pages deployment, landing page. |
| 2026-04-09 | Add with-BH-mass variant to plot_posterior_convergence | 1af4487 | Both variants shown on convergence plot; outdated delta-function assumption removed. |
