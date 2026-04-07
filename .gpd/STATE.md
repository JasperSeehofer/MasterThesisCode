# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-04-01)

**Core research question:** What is the Hubble constant H0 as measured by dark siren inference from LISA EMRI detections?
**Current focus:** v2.0 Phase 26 complete — PRD paper draft delivered, awaiting production run (Phase 27)

## Current Position

**Current Phase:** 27
**Current Phase Name:** Production Run & Figures
**Current Plan:** 3/4 complete (Plans 01-03 done, Plan 04 postponed)
**Total Phases:** 3 (Phases 26-28, v2.0 Paper)
**Status:** Phase 27 partially complete — data validated, results extracted, figures generated. Paper marker fill (Plan 04) deferred pending pipeline validation.
**Last Activity:** 2026-04-07
**Last Activity Description:** Plans 01-03 executed: pipeline bugs fixed, production data validated (23+7 files, 538 events), MAP/CI/precision extracted (h=0.66/0.68, both grid-limited), 3 figures generated + 1 placeholder. Plan 04 (paper markers) deferred.

**Progress:** [██████░░░░] 60% (Phase 27 partial, 28 not started)

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

**Last session:** 2026-04-07
**Stopped at:** Phase 27 Plans 01-03 complete. Plan 04 (paper marker fill) deferred — user wants pipeline validation first.
**Resume file:** .gpd/phases/27-production-run-figures/extracted_results.json
**Pending cluster action:** Transfer CRB CSV files from bwUniCluster to cluster_results/eval_corrected_full/ for SNR distribution figure. Then re-run plot_snr_distribution() from paper_figures.py.
**Pending pipeline validation:** Completeness correction worsened bias (MAP 0.66/0.68 vs thesis 0.712/0.742). Investigate before filling paper markers.
