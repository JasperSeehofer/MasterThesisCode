# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-04-01)

**Core research question:** What is the Hubble constant H0 as measured by dark siren inference from LISA EMRI detections?
**Current focus:** v1.3 planning -- bias evolution analysis complete (quick-2)

## Current Position

**Current Phase:** --
**Current Phase Name:** --
**Total Phases:** 7 (Phases 14-20, across v1.2.1 + v1.2.2)
**Status:** Between milestones
**Last Activity:** 2026-04-04
**Last Activity Description:** Quick task 2 complete — bias evolution analysis (quick-2/bias-evolution-analysis.md)

**Progress:** [██████████] 100%

## Active Calculations

None.

## Intermediate Results

- "Without BH mass" posterior peak: h=0.678 (P_det=1 baseline)
- "With BH mass" posterior peak: h=0.652 (P_det=1 baseline, post-all-fixes, still biased)
- Detection yield: 0.22-0.81% per h at SNR>=15 (v1.2.2)
- IS estimator VRF: 11.8-24.9x in boundary bins (v1.2.2)
- Validation: VALD-01 + VALD-02 PASS (v1.2.2)

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

**Last session:** 2026-04-02
**Stopped at:** Quick task 2 complete — bias evolution analysis written and committed (7e6731c)
**Resume file:** .gpd/quick/2-analyze-evolution-of-residual-bias-ac/bias-evolution-analysis.md
