# Research State

## Project Reference

See: .gpd/PROJECT.md (updated 2026-03-31)

**Core research question:** Can we improve injection campaign detection yield and P_det grid resolution through enhanced sampling?
**Current focus:** v1.2.2 Injection Campaign Physics Analysis -- Phase 20 complete (milestone complete)

## Current Position

**Current Phase:** 20
**Current Phase Name:** Validation
**Total Phases:** 4 (Phases 17-20)
**Current Plan:** 2/2 complete
**Total Plans in Phase:** 2
**Status:** Complete (verified 2026-04-01)
**Last Activity:** 2026-04-01
**Last Activity Description:** Phase 20 verified (7/7 contract targets passed) -- VALD-01 PASS (zero BH discoveries, 916 bins), VALD-02 PASS (alpha_grid = alpha_MC exactly)

**Progress:** [██████████] 100%

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
| 17-01 | ~4min    | 2     | 3     |
| 17-02 | ~5min    | 2     | 2     |
| 18-01 | ~5min    | 2     | 4     |
| 18-02 | ~10min   | 2     | 5     |
| 19-01 | ~4min    | 2     | 2     |
| 19-02 | ~7min    | 2     | 3     |
| 20-01 | ~7min    | 2     | 3     |
| 20-02 | ~3min    | 2     | 2     |

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
- [Phase 17-01]: All 14 EMRI parameters consistent between injection and simulation (4 intentional differences per D-01/D-04)
- [Phase 17-01]: dist() defaults identical across pipelines; only h varies intentionally per D-04
- [Phase 17-01]: d_L round-trip max relative error 2.18e-13 (threshold 1e-4) -- numerically exact
- [Phase 17-01]: z_cut=0.5 safe for all h in [0.60, 0.90] -- SNR ~ 3.2 at z=0.5 for heaviest EMRI
- [Phase 17-01]: get_distance() is dead code, never called anywhere in codebase
- [Phase 17-02]: 8 exception types in injection_campaign() traced to few library, fastlisaresponse, SIGALRM timeout
- [Phase 17-02]: CSV records only successes -- failed events hit continue without recording
- [Phase 17-02]: injection_campaign() missing AssertionError catch present in data_simulation()
- [Phase 17-02]: Timeout message says 90s but actual is 30s (_TIMEOUT_S=30)
- [Phase 17-02]: Detection rate 0.22% -- all detections at z < 0.2, M ~ 10^5.5-10^6
- [Phase 18-01]: SNR threshold = 15 (from constants.py); 663 detections vs 363 at SNR >= 20 (Phase 17)
- [Phase 18-01]: Failure rate bracketed at 30%/50% (SLURM logs unavailable for exact measurement)
- [Phase 18-01]: f_det non-monotonicity at h=0.70 attributed to Poisson noise (within 1-sigma)
- [Phase 18-01]: z_cut=0.5 safe with >=2.5x margin; all detections below z=0.204
- [Phase 18-01]: Farr criterion satisfied (N_total/N_det >= 124 for all h)
- [Phase 18-02]: 15x10 grid recommended over 30x20 for ~23k events/h -- 3.2x smaller CIs, 93% reliable bins
- [Phase 18-02]: h=0.90 needs more injections (47% unreliable bins even in 15x10)
- [Phase 18-02]: Detection boundary confined to d_L < 1 Gpc, M > 2e5 Msun -- target for importance sampling
- [Phase 18-02]: Quality flags added as metadata to SimulationDetectionProbability (no interpolation change)
- [Phase 19-01]: Preserved original np.histogram2d code path when weights=None for bit-for-bit backward compatibility
- [Phase 19-01]: Used np.digitize + np.add.at for weighted IS accumulation (single pass)
- [Phase 19-01]: 'reliable' mask stays based on integer n_total >= 10, not N_eff
- [Phase 19-01]: IS estimator backward compatible: max |diff| = 0.0 for all 7 h-values (decisive test)
- [Phase 19-02]: VRF computed from actual Phase 18 per-bin counts (not generic formulas)
- [Phase 19-02]: Targeted budget = 70% of pilot per h-value; alpha=0.3 for defensive mixture
- [Phase 19-02]: VRF 11.8-24.9x for boundary bins across all h-values (contract target >2.0)
- [Phase 19-02]: CI half-width improvement 3.4-4.6x in boundary bins
- [Phase 19-02]: Full support proof: q >= 0.3*p > 0; max weight bounded at 1/alpha = 3.33
- [Phase 20-01]: Boundary condition threshold adjusted from P_det > 0.8 to 'detections in lowest-d_L row' (EMRI max P_det ~ 0.4)
- [Phase 20-01]: VALD-01 overall PASS: zero BH discoveries across 916 bins, zero monotonicity violations, Farr global pass
- [Phase 20-01]: Farr per-bin WARN for h >= 0.73 is expected (high-P_det bins); not a failure
- [Phase 20-02]: alpha_grid = alpha_MC exactly (algebraic identity for unweighted estimator) -- round-trip grid pipeline verified
- [Phase 20-02]: h=0.70 non-monotonicity in alpha(h) within 1-sigma Poisson noise (WARN, not FAIL)
- [Phase quick-1]: Quick task 1: Investigate quick SNR check scaling in main.py — Ad-hoc investigation — callback *5 is wrong (should be √5), threshold *0.2 is conservative but defensible for chirping EMRIs

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
