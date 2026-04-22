# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-04-07)

**Core research question:** What is the Hubble constant H0 as measured by dark siren inference from LISA EMRI detections?
**Current focus:** v2.1 Publication Figures — defining objectives

## Current Position

**Current Phase:** 32 (complete)
**Current Phase Name:** Completion Term Fix
**Current Plan:** 02 (complete)
**Total Phases:** TBD (v2.0 Phases 26-28 remain; v2.1 phases to be added)
**Status:** Phase 32 complete. Full-volume D(h) denominator fix validated.
**Last Activity:** 2026-04-22
**Last Activity Description:** Phase 32 complete. Full-volume D(h) denominator for L_comp eliminates H0 posterior bias: MAP 0.60→0.73 (both channels), bias -17.8%→0.0% at 59 events (SNR≥20). Cluster production run pending for definitive validation at 531 events.

**Progress:** [██████████] 100% (Phase 32 complete)

## Active Calculations

None.

## Intermediate Results

- "Without BH mass" posterior peak: h=0.678 (P_det=1 baseline)
- "With BH mass" posterior peak: h=0.652 (P_det=1 baseline, post-all-fixes, still biased)
- Detection yield: 0.22-0.81% per h at SNR>=15 (v1.2.2)
- IS estimator VRF: 11.8-24.9x in boundary bins (v1.2.2)
- Validation: VALD-01 + VALD-02 PASS (v1.2.2)
- **Production (completeness-corrected, Phase 27):**
  - Without BH mass: MAP h=0.66, σ<0.014 (grid UB), precision ≤2.1%, bias -9.6%, 531 events
  - With BH mass: MAP h=0.68, σ<0.014 (grid UB), precision ≤2.0%, bias -6.8%, 527 events
  - Both CIs grid-limited (15-pt h-grid, spacing 0.02)
  - Completeness correction worsened bias vs thesis baseline (0.712/0.742 → 0.66/0.68)
  - SNR_THRESHOLD = 15, convergence slope -0.71 (consistent with N^{-1/2})
- **Post-completion-term-fix (Phase 32, local validation):**
  - Without BH mass: MAP h=0.73, bias 0.0%, 59 events (SNR≥20)
  - With BH mass: MAP h=0.73, bias 0.0%, 59 events (SNR≥20)
  - Fix: full-volume D(h) denominator per Gray et al. (2020) Eq. A.19
  - Bias-vs-N: monotonically decreasing, 0.0% at N=59
  - Cluster production run needed for definitive 531-event comparison

## Open Questions

- [v1.2.1] What causes the remaining "with BH mass" low-h bias after /(1+z) fix? -- Diagnosis in quick-2: Sources A (p_det mismatch) + F (zero-likelihood clustering) are primary candidates. Fix requires: (1) P_det=1 production run, (2) zero-likelihood logging, (3) Physics Change Protocol for Source A fix.
- [v1.2.1] "Without BH mass" h=0.86 overshoot with real P_det -- Diagnosis in quick-2: Source E (P_det normalization). Confirmed by P_det=1 cross-check (h=0.678). Action: P_det=1 production run.
- [v1.2.1] p_det in numerator uses detection.M rather than galaxy M*(1+z) at trial z -- Source A in quick-2 analysis. Physics Change Protocol required before fix.

## Accumulated Context

### Decisions

See `.gpd/milestones/v1.2.2-ROADMAP.md` for full v1.2.2 decision log.

Key carry-forward decisions:
- 15x10 grid recommended over 30x20 for current injection counts
- IS estimator backward-compatible; Neyman allocation ready for production
- alpha=0.3 defensive mixture bounds max weight at 3.33
- [Phase quick-2]: Quick task 2: analyze evolution of residual bias across milestones: what bias sources were eliminated and what remains to be investigated — Ad-hoc task completed outside planned phases
- [Phase quick-3]: Quick task 3: Literature research on galaxy catalog completeness correction for dark siren likelihood — Root cause of H0 bias identified as GLADE incompleteness; research produced implementation specification based on Gray et al. (2020) framework
- [Phase 24]: Completeness estimation — f(z,h) interface and comoving_volume_element delivered. Old get_completeness() backward compat disregarded per user. Data provenance (number vs luminosity completeness) to be verified in Phase 25.
- [Phase 25]: Likelihood correction — Gray et al. (2020) Eq. 9 combination formula: p_i = f_i * L_cat + (1-f_i) * L_comp. Completion term uses 'without BH mass' 3D Gaussian for both variants (uncataloged host has no galaxy mass info). Integration limits match catalog term (4-sigma d_L).
- [Phase quick-4]: Quick task 4: Physics audit of PrepareDetections — sigma chain correct, independent sampling non-standard but defensible — Ad-hoc task completed outside planned phases
- [Phase quick-5]: Quick task 5: SNR rescaling refactor — literature confirms single-h injection is standard (Gray+2020, Laghi+2021, Finke+2021); SimulationDetectionProbability refactored to pool all injection data and compute P_det via exact SNR~1/d_L rescaling — Eliminates interpolation artifacts, pools 463k injection events, enables exact P_det at any h
- [Phase 0]: Started milestone v2.1: Publication Figures — New milestone cycle — unify visualization, modern style, galaxy-level plots, interactive figures
- [Phase 32]: Completion term fix — Full-volume D(h) denominator per Gray et al. (2020) Eq. A.19 eliminates H0 posterior bias (MAP 0.60→0.73, bias -17.8%→0.0%). L_comp > 1 is physically expected (p_GW is a probability density).
- [Phase quick-6]: Quick task 6: PE-02 — Per-parameter derivative_epsilon for 14 EMRI parameters in ParameterSpace — Per-parameter epsilons derived from Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11) 5-point stencil optimal step h* ≈ 3.3e-4 × |x|. Committed as [PHYSICS] PE-02. SC-3 regression tests pass (521 passing).

### Active Approximations

- Gaussian GW measurement errors (Fisher matrix, valid for SNR >= 20)
- Gaussian galaxy mass distribution (SMBH scaling relations)
- Galaxy catalog completeness (GLADE+ complete to z ~ 0.1)

**Convention Lock:**

- Metric signature: mostly-plus
- Fourier convention: physics
- Natural units: SI
- Coordinate system: spherical
- Index positioning: Einstein

### Propagated Uncertainties

None yet.

### Pending Todos

None.

### Blockers/Concerns

- **"With BH mass" posterior still biased low** -- Dominant mechanisms: Source A (p_det(M_det) mismatch) + Source F (21% zero-likelihood events, h-bin-correlated). Phase 16 blocked until diagnosis confirmed via P_det=1 production run and zero-likelihood logging.
- **"Without BH mass" h=0.86 overshoot with real P_det** -- Source E (P_det normalization). Fix path: P_det=1 production run to confirm, then investigate IS estimator h-dependent coverage at high d_L.
- p_det in numerator uses detection.M (Source A, Physics Change Protocol required for fix)

## Session Continuity

**Last session:** 2026-04-08
**Stopped at:** Phase 32 (Completion Term Fix) complete. Full-volume D(h) denominator validated locally (MAP 0.60→0.73). Ready for cluster production run to confirm at 531 events.
**v2.0 status:** Phase 27 Plans 01-03 complete. Plan 04 deferred. Phase 32 complete. New evaluation run submitted on cluster (finer h-grid, more detections).
**Resume file:** .gpd/phases/32-completion-term-fix/validation/map_comparison.json
**Pending cluster action:** Re-run evaluation with Phase 32 fix on cluster (531 events, SNR≥15) for definitive production comparison.
**v2.1 routing:** Implementation phases tracked in GSD (`.planning/`), not GPD. GPD holds research survey + requirements only.
