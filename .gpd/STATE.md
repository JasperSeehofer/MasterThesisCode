# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-03-31)

**Core research question:** Can we improve injection campaign detection yield and P_det grid resolution through enhanced sampling?
**Current focus:** v1.2.2 Injection Campaign Physics Analysis -- Phase 17 ready to plan

## Current Position

**Current Phase:** 17
**Current Phase Name:** Injection Physics Audit
**Total Phases:** 4 (Phases 17-20)
**Current Plan:** --
**Total Plans in Phase:** TBD
**Status:** Ready to plan
**Last Activity:** 2026-03-31
**Last Activity Description:** Roadmap created with 4 phases (Audit -> Yield & Grid -> Enhanced Sampling -> Validation)

**Progress:** [░░░░░░░░░░] 0%

## Active Calculations

None.

## Intermediate Results

- "Without BH mass" posterior peak: h=0.678 (P_det=1 baseline, product-based) / h=0.730 (sum-based)
- "With BH mass" posterior peak: h=0.600 (P_det=1 baseline, pre-fix)
- **[Phase 15] Post-fix "with BH mass" posterior still monotonically decreasing** -- /(1+z) fix changed scale but not shape
- Single-galaxy likelihood peaks at h=0.73 (formula correct in isolation) -- debug session 2026-03-30
- BH mass Gaussian index [0] vs [1] has no effect under delta-function approximation -- quick task 260330-twe
- Analytic M_z marginalization (commit 15b49a3) did not change peak location
- **[Phase 14] /(1+z) at line 646 is SPURIOUS** -- Jacobian absorbed by Gaussian rescaling (Eq. 14.21), verified numerically to rtol=1e-10
- **[Phase 14] Denominator (lines 656-685) is correct** -- no changes needed
- **[Phase 14] Sky localization weight correctly inside 3D GW Gaussian** -- not a separate factor
- **[Phase 14] sigma_Mz -> infinity limiting case recovers d_L-only** -- verified numerically (CV < 3e-10)

## Open Questions

- ~~[v1.2.1] Is the `/(1+z)` at line 679 a double-counted Jacobian?~~ **RESOLVED: YES, spurious (Phase 14)**
- ~~[v1.2.1] Is the sky localization weight placed correctly?~~ **RESOLVED: YES, inside 3D GW Gaussian (Phase 14)**
- ~~[v1.2.1] Is the "with BH mass" denominator consistent?~~ **RESOLVED: YES, correct as-is (Phase 14)**
- ~~[v1.2.1] Does the analytic M_z marginalization introduce Jacobian issues?~~ **RESOLVED: NO, correctly implemented (Phase 14)**
- ~~[v1.2.1] Does removing /(1+z) shift the posterior toward h=0.678?~~ **RESOLVED: NO -- scale changed but shape unchanged (Phase 15)**
- [v1.2.1] p_det in numerator uses detection.M rather than galaxy M*(1+z) at trial z -- documented as known approximation (Phase 15)
- [v1.2.1] What causes the remaining "with BH mass" low-h bias after /(1+z) fix? -- Phase 16 scope
- [v1.2.2] Are injection parameter distributions consistent with the simulation pipeline?
- [v1.2.2] What is the detection yield (detected / total injections)?
- [v1.2.2] What fraction of compute is wasted on waveform failures vs undetectable events?
- [v1.2.2] Can importance sampling improve P_det grid quality with fewer injections?
- [v1.2.2] Does the z > 0.5 cutoff introduce any bias in P_det estimates?

## Performance Metrics

| Label | Duration | Tasks | Files |
| ----- | -------- | ----- | ----- |
| 14-01 | ~13min   | 2     | 2     |
| 14-02 | ~11min   | 2     | 2     |

## Accumulated Context

### Decisions

- [Phase 14]: /(1+z) at line 646 is SPURIOUS -- Jacobian absorbed by Gaussian rescaling (Eq. 14.21)
- [Phase 14]: Denominator (lines 656-685) is correct -- no changes needed
- [Phase 14]: Sky localization weight correctly inside 3D GW Gaussian (not separate factor)
- [Phase 14]: Analytic M_z marginalization correctly implemented (matches Bishop 2006)
- [Phase 15]: /(1+z) removed from line 655 and testing function line 870 -- reference comments added
- [Phase 15]: Denominator confirmed correct per Eq. (14.33), comments only added
- [Phase 15]: MC convergence ~1% relative error at N=10000, acceptable
- [Phase 15]: p_det(detection.M) documented as known approximation, not changed
- [Phase 15]: Quick validation FAIL -- posterior still monotonically decreasing after fix; additional bias sources exist
- [v1.2.1]: Keep analytic M_z marginalization (commit 15b49a3) -- correct physics
- [v1.2.1]: P_det out of scope -- handled in Phase 11.1
- [v1.2.1]: Derive from d_L-only literature + extend to M_z
- [v1.2.1]: Use "without BH mass" channel (h=0.678) as reference baseline
- [v1.2.2]: Analyze injection physics before next campaign -- GPU time is expensive

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

- **"With BH mass" posterior still biased low after /(1+z) fix** -- additional bias sources must be investigated (Phase 16, blocked on P_det data)
- **Injection CSVs may not yet be rsynced from cluster** -- Phase 18 (yield analysis) requires this data
- p_det in numerator uses fixed detection.M rather than galaxy M*(1+z) at trial z -- documented as known approximation (Phase 15)
- "With BH mass" denominator uses MC sampling while numerator uses quadrature -- methodology asymmetry documented, mathematically correct (Phase 15)

## Session Continuity

**Last session:** 2026-03-31
**Stopped at:** v1.2.2 roadmap created -- ready to plan Phase 17
**Resume file:** --
