# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-03-30)

**Core research question:** Why does the "with BH mass" likelihood channel produce an H0 posterior biased to h=0.600, nearly 3x worse than the "without BH mass" channel?
**Current focus:** Phase 14 complete — ready for Phase 15 (Code Audit & Fix)

## Current Position

**Current Phase:** 15
**Current Phase Name:** Code Audit & Fix
**Total Phases:** 3 (14, 15, 16)
**Current Plan:** --
**Total Plans in Phase:** TBD
**Status:** Ready to plan
**Last Activity:** 2026-03-31
**Last Activity Description:** Phase 14 complete — derivation verified (7/7 contract targets, 10/10 physics checks)

**Progress:** [███░░░░░░░] 33%

## Active Calculations

None.

## Intermediate Results

- "Without BH mass" posterior peak: h=0.678 (P_det=1 baseline)
- "With BH mass" posterior peak: h=0.600 (P_det=1 baseline)
- Single-galaxy likelihood peaks at h=0.73 (formula correct in isolation) -- debug session 2026-03-30
- BH mass Gaussian index [0] vs [1] has no effect under delta-function approximation -- quick task 260330-twe
- Analytic M_z marginalization (commit 15b49a3) did not change peak location
- **[Phase 14] /(1+z) at line 646 is SPURIOUS** — Jacobian absorbed by Gaussian rescaling (Eq. 14.21), verified numerically to rtol=1e-10
- **[Phase 14] Denominator (lines 656-685) is correct** — no changes needed
- **[Phase 14] Sky localization weight correctly inside 3D GW Gaussian** — not a separate factor
- **[Phase 14] σ_Mz → ∞ limiting case recovers d_L-only** — verified numerically (CV < 3e-10)

## Open Questions

- ~~[v1.2.1] Is the `/(1+z)` at line 679 a double-counted Jacobian?~~ **RESOLVED: YES, spurious (Phase 14)**
- ~~[v1.2.1] Is the sky localization weight placed correctly?~~ **RESOLVED: YES, inside 3D GW Gaussian (Phase 14)**
- ~~[v1.2.1] Is the "with BH mass" denominator consistent?~~ **RESOLVED: YES, correct as-is (Phase 14)**
- ~~[v1.2.1] Does the analytic M_z marginalization introduce Jacobian issues?~~ **RESOLVED: NO, correctly implemented (Phase 14)**
- [v1.2.1] Does removing /(1+z) shift the posterior toward h=0.678? (Phase 16 will test)
- [v1.2.1] p_det in numerator uses detection.M rather than galaxy M*(1+z) at trial z — investigate in Phase 15

## Performance Metrics

| Label | Duration | Tasks | Files |
| ----- | -------- | ----- | ----- |
| 14-01 | ~13min   | 2     | 2     |
| 14-02 | ~11min   | 2     | 2     |

## Accumulated Context

### Decisions

- [Phase 14]: /(1+z) at line 646 is SPURIOUS — Jacobian absorbed by Gaussian rescaling (Eq. 14.21)
- [Phase 14]: Denominator (lines 656-685) is correct — no changes needed
- [Phase 14]: Sky localization weight correctly inside 3D GW Gaussian (not separate factor)
- [Phase 14]: Analytic M_z marginalization correctly implemented (matches Bishop 2006)
- [v1.2.1]: Keep analytic M_z marginalization (commit 15b49a3) -- correct physics
- [v1.2.1]: P_det out of scope -- handled in Phase 11.1
- [v1.2.1]: Derive from d_L-only literature + extend to M_z
- [v1.2.1]: Use "without BH mass" channel (h=0.678) as reference baseline

### Active Approximations

- Gaussian GW measurement errors (Fisher matrix approximation, valid for SNR >= 20)
- Gaussian galaxy mass distribution (reasonable for SMBH mass estimates from scaling relations)
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

- ~~/(1+z) on line 679 is suspected double-counted Jacobian~~ **CONFIRMED spurious (Phase 14)**
- ~~Sky localization weight placement flagged by two TODOs~~ **RESOLVED (Phase 14)**
- "With BH mass" denominator uses MC sampling while numerator uses quadrature — methodology asymmetry noted but mathematically correct
- p_det in numerator uses fixed detection.M rather than galaxy M*(1+z) at trial z — INFO-level observation for Phase 15

## Session Continuity

**Last session:** 2026-03-31
**Stopped at:** Phase 14 complete, verified — ready for Phase 15 planning
**Resume file:** —
