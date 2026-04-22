---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: milestone
status: completed
stopped_at: Phase 39 context gathered
last_updated: "2026-04-22T21:30:01.845Z"
last_activity: 2026-04-22 — Phase 38 complete; all 5 SCs pass; state advanced to Phase 39
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 14
  completed_plans: 12
  percent: 86
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.
**Current focus:** Phase 39 — HPC & Visualization Safe Wins

## Current Position

Phase: 39 — NEXT, READY TO EXECUTE
Plan: TBD (plans not yet written)
Status: In-progress (Phase 38 COMPLETE 2026-04-22; Phase 39 queued)
Last activity: 2026-04-22 — Phase 38 complete; all 5 SCs pass; state advanced to Phase 39

**Milestone phase map:**

| # | Phase | Routing | REQ-IDs (count) |
|---|-------|---------|-----------------|
| 35 | Coordinate Bug Characterization | GSD | COORD-01 (1) |
| 36 | Coordinate Frame Fix | GPD | COORD-02, 03, 04 (3) |
| 37 | Parameter Estimation Correctness | GSD+GPD | COORD-05, PE-01..05 (6) |
| 38 | Statistical Correctness | GSD+GPD | STAT-01..04 (4) |
| 39 | HPC & Visualization Safe Wins | GSD+GPD | HPC-01..05, VIZ-01..02 (7) |
| 40 | Verification Gate | GSD+GPD | VERIFY-01..05 (5) |
| 41 | Stage 1 Injection Campaign | GSD (conditional) | CAMP-01 (1) |
| 42 | Stage 2 Sky-Dependent Injection | GSD (conditional) | CAMP-02 (1) |

**Abort gate:** Phase 40 VERIFY-02 — if re-evaluated MAP at h=0.73 shifts >5% from v2.1 baseline MAP=0.73, pause before CAMP-* phases.

## Performance Metrics

**Velocity:**

- Total plans completed: 58 (v1.0: 9, v1.1: 4, v1.2: 12, v1.3: 11, v1.4: 5, v1.5: 2, v2.1: 10, v2.0 paused)
- Total phases completed: 35 (28 across v1.0–v1.5 + v2.0 Phase 26 + v2.1 Phases 30–34 + v2.2 Phases 35–36)

## Accumulated Context

### Pending Todos

- Fix galaxy catalog unconditional init blocking generate-figures (`main.py:49`) — carried from v2.1

### Blockers/Concerns

- v2.0 Paper paused: posterior re-evaluation required under v2.2 corrected frame before production run figures
- v2.1 Publication Figures paused (originally phases 35–38): depends on v2.2 correctness gate. **Number conflict note:** v2.2 Phase 35 (Coordinate Bug Characterization) collides with the existing Phase 35 (Unified Pipeline & Paper Figures) already shipped in v2.1 PubFigs. Since v2.2 is the active milestone and the user specified "Start at Phase 35," v2.1 PubFigs phases 36-38 will need renumbering if/when they resume post-v2.2.

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

### Phase Notes (v2.2 — active)

**Phase 38 (Statistical Correctness) — COMPLETE 2026-04-22:**
STAT-01..STAT-04 all resolved. STAT-01: L_cat fixed to (1/N)Σ(N_g/D_g) per Gray et al. (2020) Eq. 24-25 (commit 005e792). STAT-02: test_l_cat_equivalence.py 3/3 pass with counterexample (2/3 ≠ 3/4). STAT-03: single_host_likelihood now uses zero-fill P_det matching D(h) denominator (commit a70d1a2). STAT-04: quadrature_weight_outside_grid WARNING infrastructure in place; event 2 shows numerator=1.000 (injection grid coverage gap — Phase 41 trigger). SC-1..SC-5 all PASS. 524 tests GREEN. Smoke run exits 0; posterior finite.
**Phase 40 VERIFY-02 note:** P_det zero-fill change (STAT-03) may shift posterior relative to Phase 37 baseline. Phase 40 must use post-Phase-38 evaluation as its reference. Event 2's 100% off-grid numerator weight strongly suggests Phase 41 injection campaign will trigger.

**Phase 37 (Parameter Estimation Correctness) — COMPLETE 2026-04-22:**
COORD-05, PE-01..PE-05 all resolved. PE-01: h_inj threaded into set_host_galaxy_parameters — Fisher CRBs now self-consistent at injected h. PE-02: per-parameter derivative_epsilon under Vallisneri 2008 protocol (Fisher det change < 1% on 4 seeds). SC-1..SC-7 all PASS. Phase 36 roundtrip regression (9 tests) GREEN throughout.
**Phase 40 VERIFY-02 note:** PE-02 epsilon change may perturb fisher_sky_2x2 values relative to the Phase 36 regression pickle (36-superset-regression.pkl). Phase 40 should use post-Phase-37 CRB values as its reference baseline, not the Phase 36 pickle.

### Phase Notes (v2.1 — shipped)

Phase 30 (Baseline/comparison), Phase 31 (Catalog-only diagnostic), Phase 32 (L_comp full-volume D(h), GPD, /physics-change), Phase 33 (P_det 60-bin validation), Phase 34 (Fisher condition-number gate) — all shipped 2026-04-09. MAP 0.73, bias 0.0% at N=59 with h=0.73 injections.

### Phase Notes (v2.0 — paused)

**Phase 26 (Paper Draft) — COMPLETE:**

- All sections drafted: Introduction, Method, Results, Discussion, Conclusions, Appendix A
- 25 RESULT PENDING markers awaiting post-v2.2 re-evaluation

**Phases 27–28 — PAUSED:** blocked on v2.2 correctness gate.

## Session Continuity

Last session: --stopped-at
Stopped at: Phase 39 context gathered
Resume file: --resume-file
Next command: `/gpd:execute-phase 38` (run 38-01 first, then 38-02, then 38-03)

## Quick Tasks Completed

| Date | Task | Commits | Summary |
|------|------|---------|---------|
| 2026-04-07 | Evaluation pipeline performance | de86052..a0de491 (7 commits) | Pool spawn 12 min→1.7 min, total 7:16 per h-value. forkserver+preload, numpy arrays, SNR filter, cpu_il partition. |
| 2026-04-07 | Add interactive Plotly figures to GitHub Pages | 8b47b5f..33e1c86 (2 commits) | 4 Plotly HTML figures (posterior, sky map, Fisher ellipses, convergence), --generate_interactive CLI flag, CI Pages deployment, landing page. |
| 2026-04-09 | Add with-BH-mass variant to plot_posterior_convergence | 1af4487 | Both variants shown on convergence plot; outdated delta-function assumption removed. |

**Planned Phase:** 35 (Coordinate Bug Characterization) — 3 plans — 2026-04-21T21:29:40.875Z
| Phase 36 P03 | 230 | 4 tasks | 4 files |
| Phase 36-coordinate-frame-fix P04 | 2 | 3 tasks | 1 files |
| Phase 37 P02 | 18 | 4 tasks | 4 files |
