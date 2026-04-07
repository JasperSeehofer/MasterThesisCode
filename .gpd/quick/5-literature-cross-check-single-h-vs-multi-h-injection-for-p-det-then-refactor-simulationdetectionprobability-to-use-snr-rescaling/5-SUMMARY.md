# SUMMARY: SNR Rescaling Literature Cross-Check and SimulationDetectionProbability Refactor

**Plan:** quick-5-01
**Status:** completed
**Commit:** 8161533

## One-Liner

SimulationDetectionProbability refactored from per-h grid interpolation to pooled injection data with exact SNR rescaling (SNR ~ 1/d_L), confirmed by dark siren literature (Gray+2020, Laghi+2021, Finke+2021) and validated by identity/monotonicity/numerical ratio tests.

## Literature Findings

### Papers Consulted

1. **Gray et al. (2020), arXiv:1908.06050** -- Section III.B-C: P_det(H0) enters the dark siren likelihood denominator (Eq. 8). The H0 dependence is captured entirely through d_L(z, H0). They do NOT run separate injection campaigns per H0 -- they inject at fixed source parameters and evaluate detectability at each H0 by computing d_L(z, H0). This is exactly the SNR rescaling approach.

2. **Laghi et al. (2021), arXiv:2102.01708** -- Section III.A: EMRI dark siren study with LISA. Uses the fact that GW strain h(t) ~ 1/d_L, so SNR ~ 1/d_L. P_det(H0) is computed from a single injection catalog by rescaling the distance to each source via d_L(z, H0). Single-injection rescaling is their standard approach.

3. **Finke et al. (2021), arXiv:2101.12660** -- Section II.C: Follows the same methodology as Gray et al. GW events are simulated once, and P_det at each H0 is computed by rescaling d_L. No per-H0 injection campaigns.

### Physics Validation of SNR ~ 1/d_L

The rescaling is **exact** (not an approximation) for the following reasons:

1. **Source-frame parameters are intrinsic.** The EMRI waveform in the source frame is determined by (M, mu, a, p0, e0, Y0) -- intrinsic source properties independent of cosmology.

2. **Redshift z is H0-independent.** A source at a fixed spacetime location has a fixed cosmological redshift z. Changing H0 changes the distance-redshift relation d_L(z, H0), not z itself.

3. **Detector-frame waveform is H0-independent.** The observed frequencies f_obs = f_source/(1+z) depend on z (fixed) and source-frame parameters (fixed), not H0. The LISA TDI response depends on f_obs, so it is also H0-independent.

4. **Only the amplitude changes.** The GW strain amplitude scales as 1/d_L, so SNR = sqrt(<h|h>) ~ 1/d_L. Since d_L(z, H0) ~ 1/H0, we get SNR(h_target) = SNR_raw * d_L(z, h_inj) / d_L(z, h_target).

### Subtleties Checked

- **Population model dependence on H0:** The EMRI event rate per comoving volume is astrophysical, not cosmological. The comoving volume element dVc/dz depends on H0, but this affects the expected number of detections (likelihood numerator), not P_det (a per-source quantity). No re-weighting needed.

- **Selection effects beyond SNR:** For EMRIs in the LISA band, the SNR threshold is the primary selection criterion. No H0-dependent frequency cuts or observation time effects apply.

### Conclusion

**GO.** Single-h injection with SNR rescaling is standard practice in the dark siren literature and is physically exact for EMRIs. The approach eliminates interpolation artifacts, pools all injection data for better statistics, and enables exact P_det evaluation at any h.

## Refactoring Summary

### Old Approach
- Per-h injection data loaded separately
- Pre-computed P_det grids for each h in the h_grid
- Linear interpolation between h grid points via `_interpolate_at_h`
- Artifacts from interpolation between coarse h grid points

### New Approach
- ALL injection CSVs pooled into a single DataFrame regardless of h_inj
- P_det grids built lazily at query time via `_get_or_build_grid(h)`
- SNR rescaling: `SNR(h) = SNR_raw * d_L(z, h_inj) / d_L(z, h_target)`
- LRU cache (max 20 entries) for performance
- `h_grid` parameter deprecated (emits DeprecationWarning)

### Key Results

| Check | Result | Confidence |
|-------|--------|------------|
| Literature supports rescaling | YES (3/3 papers) | [CONFIDENCE: HIGH] |
| SNR ~ 1/d_L is exact | YES (physics argument) | [CONFIDENCE: HIGH] |
| Identity test (h = h_inj) | PASS (rtol=1e-10) | [CONFIDENCE: HIGH] |
| Monotonicity (higher h -> higher P_det) | PASS | [CONFIDENCE: HIGH] |
| Numerical ratio matches d_L ratio | PASS (rtol=1e-10) | [CONFIDENCE: HIGH] |
| Public API backward compatible | YES | [CONFIDENCE: HIGH] |
| Pickle safety preserved | YES | [CONFIDENCE: HIGH] |
| All 14 tests pass | YES | [CONFIDENCE: HIGH] |
| ruff + mypy clean | YES | -- |
| Full test suite (378 tests) passes | YES | -- |

### Conventions

| Convention | Value |
|------------|-------|
| Units | SI (Gpc for distances, solar masses for M, km/s/Mpc for H0) |
| Cosmology | flat LCDM, Omega_m=0.25, H=0.73 (project fiducial) |
| SNR scaling | SNR proportional to 1/d_L (exact for GW amplitude) |
| h definition | h = H0 / (100 km/s/Mpc) |

### Deviations

None.

### Contract Results

```yaml
plan_contract_ref: quick-5-01
contract_results:
  claims:
    claim-snr-rescaling-valid:
      status: confirmed
      evidence: "Literature (Gray+2020, Laghi+2021, Finke+2021) all use single-injection rescaling. Physics argument confirms SNR ~ 1/d_L is exact for fixed source-frame parameters. Numerical tests verify identity and ratio."
    claim-backward-compat:
      status: confirmed
      evidence: "Public API unchanged. All 6 original tests pass. Pickle roundtrip works. h_grid parameter accepted with deprecation warning."
  deliverables:
    deliv-literature-note:
      status: produced
      path: ".gpd/quick/5-literature-cross-check-single-h-vs-multi-h-injection-for-p-det-then-refactor-simulationdetectionprobability-to-use-snr-rescaling/5-SUMMARY.md"
    deliv-refactored-class:
      status: produced
      path: "master_thesis_code/bayesian_inference/simulation_detection_probability.py"
    deliv-tests:
      status: produced
      path: "master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py"
  acceptance_tests:
    test-literature-support:
      outcome: pass
      evidence: "3 papers consulted, all support SNR rescaling approach"
    test-numerical-consistency:
      outcome: pass
      evidence: "Identity test (h=h_inj gives SNR_raw, rtol=1e-10), ratio test (SNR matches d_L ratio, rtol=1e-10)"
    test-api-compat:
      outcome: pass
      evidence: "14 tests pass including 6 original tests adapted for new internals. Public method signatures unchanged."
  forbidden_proxies:
    fp-untested-rescaling:
      status: rejected
      evidence: "Three numerical tests verify rescaling: identity, monotonicity, and exact ratio."
    fp-literature-skip:
      status: rejected
      evidence: "Three papers consulted with specific section references."
  must_surface_refs:
    ref-gray2020:
      status: completed
      actions: [read, cite]
    ref-laghi2021:
      status: completed
      actions: [read, cite]
```

## Self-Check: PASSED

- [x] simulation_detection_probability.py exists and compiles
- [x] test_simulation_detection_probability.py exists and all 14 tests pass
- [x] Commit 8161533 exists
- [x] ruff clean
- [x] mypy clean
- [x] Full test suite (378 tests) passes
