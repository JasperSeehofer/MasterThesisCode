# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-04-24)

**Core research question:** What is the Hubble constant H0 as measured by dark siren inference from LISA EMRI detections?
**Current focus:** v2.2 Phase 43 — Posterior Calibration Fix (SC-3 MAP=0.860 root cause diagnosis)

## Current Position

**Current Phase:** 43
**Current Phase Name:** Posterior Calibration Fix
**Total Phases:** —
**Current Plan:** 3/3 — all plans complete; verification in progress
**Total Plans in Phase:** 3
**Status:** Complete — all plans executed; MAP=0.730 confirmed; Phase 42 deferred
**Last Activity:** 2026-04-27
**Last Activity Description:** Plan 43-03 complete: post-fix MAP=0.730 (VERIFY-03 SC-3 PASS); Q3 anisotropy resolved; Phase 42 DEFERRED.

**Progress:** [██████████] 100%

## Active Calculations

None yet.

## Intermediate Results

- Phase 40 VERIFY-03 SC-3: MAP=0.860 (expected 0.73±0.01) — extract_baseline sums log-L without D(h)
- Phase 40 VERIFY-02: MAP shift 0.0% (abort gate NOT fired)
- Phase 40 VERIFY-04: stage_2_trigger=true (Q3 |ΔMAP|=0.020 >> σ=0.0037)
- Phase 40 VERIFY-05: mean_lb=0.041 (borderline; Phase 41 SKIPPED per user Q2)
- v2.2 CRB re-evaluation: 60 events at SNR≥20 (vs v2.1 417 events)
- v2.2 MAP=0.73 confirmed by --evaluate comparison in VERIFY-02
- **Phase 43 COMPLETE:** MAP=0.730 post-fix (VERIFY-03 SC-3 PASS); H2 CRB ecliptic migration + H1 extract_baseline deprecation; Phase 42 DEFERRED (Q3 anisotropy resolved)

## Open Questions

- Is D(h) denominator missing from _combine_posteriors/extract_baseline? (FIX-01)
- Do CRBs in simulations/prepared_cramer_rao_bounds.csv store qS/phiS in equatorial or ecliptic frame? (FIX-02)
- If CRBs are equatorial (pre-COORD): does a new simulation run need to be triggered? (scope-expander)
- Will VERIFY-04 anisotropy resolve after D(h)/frame fix, or is Phase 42 still needed?
- Does --evaluate with equatorial CRBs give MAP≈0.73 or MAP≠0.73?
- If H2 causes --evaluate MAP≠0.73: transform CRBs or re-run simulation?
- Does VERIFY-04 anisotropy resolve after H1/H2 fix?

## Performance Metrics

| Label | Duration | Tasks | Files |
| ----- | -------- | ----- | ----- |
| -     | -        | -     | -     |

## Accumulated Context

### Decisions

- [Phase 40]: Q1: Insert fix phase for VERIFY-03 SC-3 (angle audit + D(h) in --combine) — MAP=0.860 from extract_baseline; root cause: D(h) missing OR CRB frame mismatch post-COORD fix
- [Phase 40]: Q2: Skip Phase 41 (Stage 1 Injection Campaign) — VERIFY-05 mean_lb=0.041 accepted as known limitation; 19 off-grid events (3.5%) documented
- [Phase 40]: Q3: Defer Phase 42 until Phase 43 Q1 resolved — Anisotropy may be a symptom of same D(h)/COORD bug, not genuine sky-dependent P_det

### Active Approximations

None yet.

**Convention Lock:**

- Metric signature: mostly-plus
- Fourier convention: physics
- Natural units: SI
- Coordinate system: spherical
- Index positioning: Einstein

*Custom conventions:*
- Sky Angles: qS = ecliptic colatitude (polar angle from north ecliptic pole), phiS = ecliptic longitude; GLADE equatorial RA/Dec rotated to ecliptic via astropy BarycentricTrueEcliptic (Phase 36 fix)
- H0 Units: dimensionless ratio (H0 / 100 km/s/Mpc)
- Posterior Normalization: p(h|data) ∝ sum_i log[L_i(h)] - N*log[D(h)], where D(h) is the completeness-corrected detectable volume

### Propagated Uncertainties

None yet.

### Pending Todos

None yet.

### Blockers/Concerns

None

## Session Continuity

**Last session:** 2026-04-24
**Stopped at:** Phase 43 planning started
**Resume file:** —
