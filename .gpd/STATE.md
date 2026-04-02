# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-04-01)

**Core research question:** What is the Hubble constant H0 as measured by dark siren inference from LISA EMRI detections?
**Current focus:** v1.2.2 complete -- planning next milestone

## Current Position

**Current Phase:** --
**Current Phase Name:** --
**Total Phases:** 7 (Phases 14-20, across v1.2.1 + v1.2.2)
**Status:** Between milestones
**Last Activity:** 2026-04-01
**Last Activity Description:** v1.2.2 milestone archived (Injection Campaign Physics Analysis)

**Progress:** [██████████] 100%

## Active Calculations

None.

## Intermediate Results

- "Without BH mass" posterior peak: h=0.678 (P_det=1 baseline)
- "With BH mass" posterior peak: h=0.600 (P_det=1 baseline, post-fix, still biased)
- Detection yield: 0.22-0.81% per h at SNR>=15 (v1.2.2)
- IS estimator VRF: 11.8-24.9x in boundary bins (v1.2.2)
- Validation: VALD-01 + VALD-02 PASS (v1.2.2)

## Open Questions

- [v1.2.1] What causes the remaining "with BH mass" low-h bias after /(1+z) fix? -- Phase 16 scope, blocked on P_det data
- [v1.2.1] p_det in numerator uses detection.M rather than galaxy M*(1+z) at trial z -- known approximation

## Accumulated Context

### Decisions

See `.gpd/milestones/v1.2.2-ROADMAP.md` for full v1.2.2 decision log.

Key carry-forward decisions:
- 15x10 grid recommended over 30x20 for current injection counts
- IS estimator backward-compatible; Neyman allocation ready for production
- alpha=0.3 defensive mixture bounds max weight at 3.33

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

- **"With BH mass" posterior still biased low** -- Phase 16 blocked on P_det data
- p_det in numerator uses detection.M (known approximation, Phase 15)

## Session Continuity

**Last session:** 2026-04-01
**Stopped at:** v1.2.2 milestone completed and archived
**Resume file:** --
