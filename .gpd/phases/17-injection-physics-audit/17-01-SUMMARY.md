---
phase: 17-injection-physics-audit
plan: 01
depth: full
one-liner: "Verified injection campaign parameters and cosmological model are consistent with simulation pipeline; d_L round-trip accurate to 2e-13"
subsystem: [validation, analysis]
tags: [injection-campaign, cosmology, parameter-audit, round-trip, luminosity-distance]

requires:
  - phase: 11.1-simulation-based-detection-probability
    provides: "Design decisions D-01 through D-09 defining intentional injection/simulation differences"
provides:
  - "All 14 EMRI parameters traced through injection and simulation paths with line citations"
  - "dist() call-site audit confirming identical cosmological defaults except intentional h-handling"
  - "d_L round-trip accuracy verified to 2e-13 relative error across 7 h-values x 100 z-points"
  - "z_cut=0.5 safety confirmed via SNR scaling argument"
  - "get_distance() confirmed unused in injection path"
affects: [17-02, 18-yield-grid-analysis]

methods:
  added: [code-path-tracing, round-trip-numerical-test, SNR-scaling-argument]
  patterns: [line-by-line-parameter-audit]

key-files:
  created:
    - ".gpd/phases/17-injection-physics-audit/audit-parameter-consistency.md"
    - ".gpd/phases/17-injection-physics-audit/audit-cosmological-model.md"
    - ".gpd/phases/17-injection-physics-audit/test_round_trip.py"

key-decisions:
  - "All 4 parameter differences (M, d_L, phiS, qS) categorized as intentional per D-01/D-04"
  - "z_cut=0.5 is safe: SNR(z=0.5) ~ 3.2 << threshold 15 for all h in [0.60, 0.90]"

conventions:
  - "SI units (distances in Gpc, masses in solar masses, h dimensionless)"
  - "Flat LambdaCDM: Omega_m=0.25, Omega_DE=0.75, w0=-1, wa=0 (WMAP-era)"
  - "Dimensionless h = H0/(100 km/s/Mpc), fiducial h=0.73"

plan_contract_ref: ".gpd/phases/17-injection-physics-audit/17-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-param-identical:
      status: passed
      summary: "All 14 EMRI parameters traced line-by-line. 4 differences (M, d_L, phiS, qS) all intentional per D-01/D-04. Galaxy catalog intermediary fully traced."
      linked_ids: [deliv-param-report, test-param-linebyline, ref-design-decisions, ref-injection-code, ref-simulation-code]
    claim-cosmo-consistent:
      status: passed
      summary: "dist() called with identical defaults (Omega_m=0.25, Omega_DE=0.75) in both pipelines. Only h differs, intentional per D-04."
      linked_ids: [deliv-cosmo-report, test-cosmo-defaults, test-h-intentional, ref-design-decisions, ref-dist-function]
    claim-roundtrip-accurate:
      status: passed
      summary: "Round-trip z->d_L->z_rec accurate to 2.18e-13 relative error (9 orders of magnitude below 1e-4 threshold) across all 700 test points."
      linked_ids: [deliv-roundtrip-test, test-roundtrip-accuracy, ref-dist-function, ref-hogg1999]
    claim-zcut-safe:
      status: passed
      summary: "z_cut=0.5 safe: SNR(z=0.5) ~ 3.2 for sources barely detectable at z=0.1, well below threshold 15. d_L(z=0.5, h=0.60) = 3.3645 Gpc >> LISA horizon 1.55 Gpc."
      linked_ids: [deliv-cosmo-report, test-zcut-safety, ref-dist-function]
  deliverables:
    deliv-param-report:
      status: passed
      path: ".gpd/phases/17-injection-physics-audit/audit-parameter-consistency.md"
      summary: "Complete line-by-line comparison of 14 parameters with specific code line citations, summary table, galaxy catalog intermediary traced."
      linked_ids: [claim-param-identical, test-param-linebyline]
    deliv-cosmo-report:
      status: passed
      path: ".gpd/phases/17-injection-physics-audit/audit-cosmological-model.md"
      summary: "All 20 dist() call sites documented. h-handling difference verified per D-04. Impact assessment with numerical d_L values. z_cut safety analysis."
      linked_ids: [claim-cosmo-consistent, claim-zcut-safe, test-cosmo-defaults, test-h-intentional, test-zcut-safety]
    deliv-roundtrip-test:
      status: passed
      path: ".gpd/phases/17-injection-physics-audit/test_round_trip.py"
      summary: "Standalone Python script testing 7 h-values x 100 z-points, plus limiting cases and Hubble law. All tests pass."
      linked_ids: [claim-roundtrip-accurate, test-roundtrip-accuracy]
  acceptance_tests:
    test-param-linebyline:
      status: passed
      summary: "All 14 parameters have entries in comparison table with specific line numbers. Every difference categorized against D-01 through D-09. Galaxy catalog intermediary explicitly traced."
      linked_ids: [claim-param-identical, deliv-param-report, ref-design-decisions, ref-injection-code, ref-simulation-code]
    test-cosmo-defaults:
      status: passed
      summary: "All 20 dist() call sites inspected. None override Omega_m or Omega_DE. Only h varies between injection (explicit) and simulation (default 0.73)."
      linked_ids: [claim-cosmo-consistent, deliv-cosmo-report, ref-dist-function]
    test-h-intentional:
      status: passed
      summary: "Injection uses dist(z, h=h_value) at main.py:500. Simulation uses dist(z) at parameter_space.py:148. Matches D-04 exactly."
      linked_ids: [claim-cosmo-consistent, deliv-cosmo-report, ref-design-decisions]
    test-roundtrip-accuracy:
      status: passed
      summary: "Max relative error 2.18e-13 across all 700 test points. Worst case at z=0.001, h=0.70. No edge case failures."
      linked_ids: [claim-roundtrip-accurate, deliv-roundtrip-test, ref-dist-function]
    test-zcut-safety:
      status: passed
      summary: "SNR scaling argument: d_L(z=0.1)/d_L(z=0.5) = 0.16 independent of h. SNR(z=0.5) ~ 3.2 for threshold-detected z=0.1 source. Well below SNR_THRESHOLD=15."
      linked_ids: [claim-zcut-safe, deliv-cosmo-report, ref-dist-function]
  references:
    ref-design-decisions:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "D-01 through D-09 read and compared against implementation. All intentional differences verified."
    ref-injection-code:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "injection_campaign() (main.py:396-577) fully read and traced."
    ref-simulation-code:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "data_simulation() (main.py:188-383) fully read and traced."
    ref-dist-function:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "dist() and dist_to_redshift() (physical_relations.py) fully audited. All 20 call sites documented."
    ref-hogg1999:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Hogg (1999) cited as reference for d_L(z) formula and limiting case dist(z=0)=0."
  forbidden_proxies:
    fp-looks-similar:
      status: rejected
      notes: "Report provides specific line numbers for every parameter comparison. No vague similarity claims."
    fp-skip-galaxy-catalog:
      status: rejected
      notes: "Galaxy catalog intermediary fully traced through handler.py:553-592, including find_closest_galaxy_to_coordinates()."
    fp-roundtrip-qualitative:
      status: rejected
      notes: "Quantitative errors reported per h-value: max 2.18e-13 relative error. Full numerical output included."
  uncertainty_markers:
    weakest_anchors:
      - "Design decisions D-01 through D-09 may not cover every intentional difference -- undocumented intentional differences would be flagged as bugs"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: claim-roundtrip-accurate
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-dist-function
    comparison_kind: benchmark
    metric: relative_error
    threshold: "<= 1e-4"
    verdict: pass
    recommended_action: "No action needed. Round-trip is machine-precise."
    notes: "Actual max error 2.18e-13, 9 orders of magnitude below threshold."
  - subject_id: claim-zcut-safe
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-dist-function
    comparison_kind: baseline
    metric: snr_ratio
    threshold: "SNR(z=0.5) << SNR_THRESHOLD=15"
    verdict: pass
    recommended_action: "No action needed. z_cut=0.5 is well above detection horizon."
    notes: "SNR(z=0.5) ~ 3.2, factor 4.7x below threshold. Empirically all detections at z < 0.18."

duration: 4min
completed: 2026-03-31
---

# Phase 17-01: Injection Physics Audit -- Parameter & Cosmological Model Consistency

**Verified injection campaign parameters and cosmological model are consistent with simulation pipeline; d_L round-trip accurate to 2e-13**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-31T21:25:35Z
- **Completed:** 2026-03-31T21:29:32Z
- **Tasks:** 2
- **Files created:** 3

## Key Results

- All 14 EMRI parameters traced through both code paths. 4 differences (M, d_L, phiS, qS) are all intentional per D-01/D-04. [CONFIDENCE: HIGH]
- dist() called with identical cosmological defaults in both pipelines. Only h varies (intentional per D-04). [CONFIDENCE: HIGH]
- d_L round-trip max relative error: 2.18e-13 (threshold: 1e-4). Machine-precise inversion. [CONFIDENCE: HIGH]
- z_cut=0.5 is safe: SNR(z=0.5) ~ 3.2 << threshold 15. [CONFIDENCE: HIGH]
- get_distance() method is defined but never called. [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Line-by-line parameter distribution comparison** - `c688b51` (validate)
2. **Task 2: Cosmological model consistency and d_L round-trip test** - `d703c60` (validate)

## Files Created/Modified

- `.gpd/phases/17-injection-physics-audit/audit-parameter-consistency.md` -- 14-parameter comparison with line citations
- `.gpd/phases/17-injection-physics-audit/audit-cosmological-model.md` -- dist() call-site audit, h-handling, z_cut safety
- `.gpd/phases/17-injection-physics-audit/test_round_trip.py` -- Standalone round-trip test (7 h x 100 z)

## Next Phase Readiness

- Parameter consistency confirmed: injection campaign data is safe for P_det grid construction
- Cosmological model consistency confirmed: no hidden parameter mismatches
- Round-trip inversion is numerically exact: no accuracy concerns for z-reconstruction from d_L
- Plan 17-02 can proceed with analysis of the simulation pipeline likelihood computation

## Contract Coverage

- Claim IDs advanced: claim-param-identical -> passed, claim-cosmo-consistent -> passed, claim-roundtrip-accurate -> passed, claim-zcut-safe -> passed
- Deliverable IDs produced: deliv-param-report -> passed, deliv-cosmo-report -> passed, deliv-roundtrip-test -> passed
- Acceptance test IDs run: test-param-linebyline -> passed, test-cosmo-defaults -> passed, test-h-intentional -> passed, test-roundtrip-accuracy -> passed, test-zcut-safety -> passed
- Reference IDs surfaced: ref-design-decisions -> read/compare, ref-injection-code -> read/compare, ref-simulation-code -> read/compare, ref-dist-function -> read/compare, ref-hogg1999 -> cite
- Forbidden proxies rejected: fp-looks-similar, fp-skip-galaxy-catalog, fp-roundtrip-qualitative

## Validations Completed

- dist(z=0, h) = 0 exactly for all 7 h-values (limiting case)
- Low-z Hubble law: dist(0.001, h) within 0.08% of c*z/(1e5*h) Gpc
- Round-trip: max rel_error = 2.18e-13 across 700 test points (9 orders of magnitude below threshold)
- z_cut=0.5 safety: d_L(z=0.5, h=0.60) = 3.3645 Gpc >> LISA horizon 1.55 Gpc
- All 20 dist() call sites verified: none override Omega_m or Omega_DE
- get_distance() never called in codebase (grep confirmed)

## Decisions & Deviations

None -- plan executed exactly as written.

## Key Quantities and Uncertainties

| Quantity | Symbol | Value | Uncertainty | Source | Valid Range |
|----------|--------|-------|-------------|--------|-------------|
| Round-trip max relative error | rel_err | 2.18e-13 | N/A | test_round_trip.py | z in [0.001, 0.5], h in [0.60, 0.90] |
| d_L at z=0.5, h=0.60 | d_L | 3.3645 Gpc | exact (analytic) | dist() hypergeometric | flat LambdaCDM |
| d_L at z=0.5, h=0.90 | d_L | 2.2430 Gpc | exact (analytic) | dist() hypergeometric | flat LambdaCDM |
| SNR ratio z=0.5 vs z=0.1 | ratio | 0.16 | N/A | d_L scaling | h-independent |

## Open Questions

- None from this plan. All questions resolved.

---

_Phase: 17-injection-physics-audit_
_Completed: 2026-03-31_
