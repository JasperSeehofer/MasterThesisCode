---
phase: 17-injection-physics-audit
plan: 02
depth: full
one-liner: "Cataloged 8 exception types in injection_campaign() with source tracing, documented CSV design limitation, and analyzed 165k injection events showing 0.22% detection rate concentrated at z<0.2"

subsystem: [validation, analysis]
tags: [emri, waveform-failures, injection-campaign, detection-probability]

requires:
  - phase: 11.1-simulation-based-detection-probability
    provides: injection_campaign() implementation (D-01 through D-09 design decisions)
provides:
  - Exception catalog for injection_campaign() with 8 exception types and source tracing
  - CSV design limitation documentation (failed events not recorded)
  - Quantitative injection data analysis (165k events, 363 detections, z/M distributions)
  - Failure analysis script (analyze_failures.py) for future data parsing
  - Future tracking proposal for structured failure recording
affects: [18-yield-grid-analysis, 19-enhanced-sampling]

methods:
  added: [code-path-audit, injection-csv-analysis]
  patterns: [exception-catalog-with-source-tracing, data-availability-graceful-degradation]

key-files:
  created:
    - .gpd/phases/17-injection-physics-audit/audit-waveform-failures.md
    - .gpd/phases/17-injection-physics-audit/analyze_failures.py
  modified: []

key-decisions:
  - "Cannot compute waveform failure rate from CSV alone -- CSV records only successes"
  - "SLURM logs not available locally -- failure-by-type breakdown deferred to log rsync"
  - "injection_campaign() missing AssertionError catch that data_simulation() has"
  - "Timeout message says 90s but actual timeout is 30s (misleading but correct behavior)"

patterns-established:
  - "Exception catalog format: exception type, caught-at line, source file:line, physical trigger, susceptible parameter region"
  - "Data analysis script with 3-scenario graceful degradation (CSV / logs / no-data)"

conventions:
  - "SI units (distances in Gpc, masses in solar masses, h dimensionless)"
  - "Flat LambdaCDM: Omega_m=0.25, Omega_DE=0.75, H=0.73"

plan_contract_ref: ".gpd/phases/17-injection-physics-audit/17-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-failure-categorized:
      status: passed
      summary: "All 8 exception types in injection_campaign() cataloged with source file:line, physical trigger conditions, and expected (z,M,e0) susceptibility regions"
      linked_ids: [deliv-failure-report, test-failure-code-audit, ref-injection-code, ref-param-estimation]
    claim-failure-by-region:
      status: partial
      summary: "Code-level analysis of expected failure regions completed. Data-driven failure-by-region correlation blocked by missing SLURM logs (CSV does not record failures). Detection distribution by (z,M) bin available from CSV."
      linked_ids: [deliv-failure-report, deliv-analysis-script, test-failure-breakdown, ref-injection-code, ref-slurm-logs]
    claim-csv-limitation:
      status: passed
      summary: "CSV design limitation fully documented: 8 continue statements (L516,519,525,531,538,543,551,554) skip results.append(); _INJECTION_COLUMNS has no failure status field; simulation_steps controls successes not attempts"
      linked_ids: [deliv-failure-report, test-csv-gap-documented, ref-injection-code]
  deliverables:
    deliv-failure-report:
      status: passed
      path: ".gpd/phases/17-injection-physics-audit/audit-waveform-failures.md"
      summary: "Complete report with exception catalog, source tracing, CSV limitation, data analysis results, and future tracking proposal"
      linked_ids: [claim-failure-categorized, claim-failure-by-region, claim-csv-limitation]
    deliv-analysis-script:
      status: passed
      path: ".gpd/phases/17-injection-physics-audit/analyze_failures.py"
      summary: "Analysis script handles all 3 scenarios; ran successfully against local CSV data (Scenario A)"
      linked_ids: [claim-failure-by-region]
  acceptance_tests:
    test-failure-code-audit:
      status: passed
      summary: "All 8 except clauses in injection_campaign() documented with exception type, source function/line, and physical trigger condition"
      linked_ids: [claim-failure-categorized, deliv-failure-report, ref-injection-code, ref-param-estimation]
    test-failure-breakdown:
      status: partial
      summary: "Code-level analysis of expected failure regions provided. Quantitative breakdown by exception type blocked by missing SLURM logs. Detection-by-(z,M) bin breakdown provided from CSV data."
      linked_ids: [claim-failure-by-region, deliv-failure-report, deliv-analysis-script]
    test-csv-gap-documented:
      status: passed
      summary: "8 continue statements cited with line numbers. CSV column list documented. Absence of failure status column confirmed."
      linked_ids: [claim-csv-limitation, deliv-failure-report, ref-injection-code]
  references:
    ref-injection-code:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "injection_campaign() (main.py:396-577) read line-by-line; all try/except blocks identified and traced"
    ref-param-estimation:
      status: completed
      completed_actions: [read]
      missing_actions: []
      summary: "parameter_estimation.py read for exception sources: ParameterOutOfBoundsError at L167,219; generate_lisa_response, compute_signal_to_noise_ratio call chain traced"
    ref-slurm-logs:
      status: missing
      completed_actions: []
      missing_actions: [read]
      summary: "SLURM logs not available locally. Cannot parse for per-exception failure counts or progress messages. Recommend rsync from bwUniCluster."
  forbidden_proxies:
    fp-aggregate-rate:
      status: rejected
      notes: "Report breaks down by exception type (8 categories), by z bin (8 bins), and by log10(M) bin (6 bins). No single aggregate failure rate reported."
    fp-assume-random:
      status: rejected
      notes: "Code-level analysis documents specific parameter regions susceptible to each failure type (high e0, extreme M, near-separatrix orbits). Detection distribution shows strong z and M dependence."
  uncertainty_markers:
    weakest_anchors:
      - "SLURM logs not rsynced -- failure-by-type quantitative breakdown impossible from current data"
      - "CSV does not record spin (a) or eccentricity (e0) -- failure correlation with these parameters cannot be assessed even with failure CSV"
    unvalidated_assumptions:
      - "Code-level failure susceptibility analysis assumes few library failure modes are dominated by e0, M, p0 based on general knowledge; actual distribution may differ"
    competing_explanations: []
    disconfirming_observations: []

duration: 5min
completed: 2026-03-31
---

# Phase 17, Plan 02: Waveform Failure Characterization Summary

**Cataloged 8 exception types in injection_campaign() with source tracing, documented CSV design limitation, and analyzed 165k injection events showing 0.22% detection rate concentrated at z<0.2**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-31T21:26:54Z
- **Completed:** 2026-03-31T21:31:40Z
- **Tasks:** 2
- **Files created:** 2

## Key Results

- **8 exception types** cataloged in injection_campaign() with source tracing to `few` library, `fastlisaresponse`, and SIGALRM timeout
- **CSV design limitation confirmed:** All 8 exception handlers execute `continue` without appending to the results list; no failure status column exists in CSV
- **165,000 events analyzed** across 7 h-values (0.6-0.9): only 363 detections (0.22% rate)
- **Detection horizon:** All detections at z < 0.2, with 93% at z < 0.1; concentrated at M ~ 10^5.5 - 10^6
- **Data predates z_cut = 0.5:** 34% of events at z > 0.5 produce zero detections (wasted compute)
- **Missing: `AssertionError` catch** in injection_campaign() that exists in data_simulation() -- potential crash risk

## Task Commits

1. **Task 1: Code-level waveform failure characterization** - `60fe875` (analyze)
2. **Task 2: Data-driven failure analysis script** - `3e9a9f2` (analyze)

**Plan metadata:** (this commit)

## Files Created/Modified

- `.gpd/phases/17-injection-physics-audit/audit-waveform-failures.md` - Exception catalog, source tracing, CSV limitation, data analysis results, future tracking proposal
- `.gpd/phases/17-injection-physics-audit/analyze_failures.py` - Analysis script with 3-scenario graceful degradation

## Next Phase Readiness

- Exception catalog and detection distribution ready for Phase 18 (yield analysis) and Phase 19 (enhanced sampling design)
- Failure-by-type quantitative breakdown requires SLURM log rsync (documented gap)
- Detection concentration at z < 0.2, M ~ 10^5.5-10^6 directly informs enhanced sampling strategy
- Concrete future tracking proposal ready for implementation in next injection campaign

## Contract Coverage

- Claim IDs advanced: claim-failure-categorized -> passed, claim-failure-by-region -> partial (code-level done, data gap on SLURM logs), claim-csv-limitation -> passed
- Deliverable IDs produced: deliv-failure-report -> passed, deliv-analysis-script -> passed
- Acceptance test IDs run: test-failure-code-audit -> passed, test-failure-breakdown -> partial, test-csv-gap-documented -> passed
- Reference IDs surfaced: ref-injection-code -> completed, ref-param-estimation -> completed, ref-slurm-logs -> missing (not available locally)
- Forbidden proxies rejected: fp-aggregate-rate -> rejected (breakdown by type/z/M provided), fp-assume-random -> rejected (parameter-region analysis provided)

## Validations Completed

- All 8 except clauses in injection_campaign() (L510-554) verified against code
- Line numbers cross-checked: _INJECTION_COLUMNS at L386, results init at L451, _TIMEOUT_S at L462
- Analysis script ran successfully in Scenario A (CSV data available)
- CSV column list matches _INJECTION_COLUMNS: z, M, phiS, qS, SNR, h_inj, luminosity_distance (no failure field)
- Comparison with data_simulation() exception handling: identified missing AssertionError catch and different timeout values

## Decisions & Deviations

None - followed plan as specified. Data availability matched Scenario A (CSVs available, SLURM logs not).

## Open Questions

- What is the quantitative failure rate per exception type? (Requires SLURM log rsync)
- Does the failure rate correlate with specific (z, M, e0) regions? (Requires failure CSV or log parsing)
- Should z_cut be tightened from 0.5 to 0.2 given zero detections above z=0.2? (Risk: h-dependent horizon)
- Should AssertionError catch be added to injection_campaign()? (Low effort, prevents rare crash)

## Key Quantities and Uncertainties

| Quantity | Symbol | Value | Uncertainty | Source | Valid Range |
|---|---|---|---|---|---|
| Total injection events | N_total | 165,000 | exact | CSV count | All h-values |
| Total detections | N_det | 363 | exact | SNR >= 20 count | All h-values |
| Detection rate | f_det | 0.220% | exact (sampling variance ~5%) | N_det/N_total | Pre-z_cut data |
| Detection z horizon | z_max_det | 0.155 | exact | Max detected z | This campaign |
| Detection M range | log10(M) | [4.97, 6.00] | exact | Min/max detected M | This campaign |

## Issues Encountered

- SLURM logs not available locally, blocking quantitative failure-by-type analysis. Documented as data gap with rsync recommendation.

---

_Phase: 17-injection-physics-audit, Plan 02_
_Completed: 2026-03-31_
