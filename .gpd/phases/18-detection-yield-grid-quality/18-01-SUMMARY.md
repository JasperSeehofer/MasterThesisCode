---
phase: 18-detection-yield-grid-quality
plan: 01
depth: full
one-liner: "Computed per-h detection yield (0.22-0.81%), decomposed GPU waste into failures/sub-threshold/detected, confirmed z>0.5 cutoff safe and Farr criterion satisfied across all 7 h-values"
subsystem: [analysis, validation]
tags: [injection-campaign, detection-probability, yield-analysis, gpu-efficiency, farr-criterion]

requires:
  - phase: 17-injection-physics-audit
    provides: "Injection parameter consistency, CSV column schema, 8 exception types, 165k events confirmed"
provides:
  - "Per-h detection yield table (f_det to 3 significant figures for 7 h-values)"
  - "2-way CSV-exact and 3-way estimated (30%/50% failure) waste decompositions"
  - "z > 0.5 cutoff validated with zero detections above z=0.5"
  - "Farr (2019) criterion satisfied: N_total/N_det >= 124 for all h"
  - "Standalone analysis script (analysis/injection_yield.py) for reproducible yield computation"
  - "Waste breakdown figure (figures/injection_yield_waste_breakdown.pdf)"
affects: [18-02, 19-enhanced-sampling]

methods:
  added: [csv-aggregation, waste-decomposition, farr-criterion-check]
  patterns: [per-h-grouped-analysis, 2-way-and-3-way-waste-split]

key-files:
  created:
    - "analysis/injection_yield.py"
    - ".gpd/phases/18-detection-yield-grid-quality/yield-report.md"
    - "figures/injection_yield_waste_breakdown.pdf"

key-decisions:
  - "SNR threshold = 15 (from constants.py), not 20 (Phase 17 used 20 for comparison)"
  - "Failure rates estimated at 30% and 50% to bracket unknown true rate (SLURM logs unavailable)"
  - "Minor f_det non-monotonicity at h=0.70 attributed to Poisson sampling noise (within 1-sigma)"

conventions:
  - "SI units (distances in Gpc, masses in solar masses, h dimensionless)"
  - "Flat LambdaCDM: Omega_m=0.25, Omega_DE=0.75, H=0.73"
  - "SNR_THRESHOLD = 15"

plan_contract_ref: ".gpd/phases/18-detection-yield-grid-quality/18-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-yield-per-h:
      status: passed
      summary: "Detection fraction computed to 3 significant figures for all 7 h-values: 2.22e-3 (h=0.60) to 8.06e-3 (h=0.90). Total 663 detections from 165,000 events."
      linked_ids: [deliv-yield-script, deliv-yield-report, test-yield-3sf, test-yield-monotonic, ref-injection-csvs, ref-phase17-summary]
    claim-waste-decomposition:
      status: passed
      summary: "2-way CSV-exact decomposition (detected + sub-threshold = 100%) and 3-way estimated decomposition at 30% and 50% failure rates computed per h-value. All fractions verified to sum to 1.0."
      linked_ids: [deliv-yield-script, deliv-waste-figure, deliv-yield-report, test-waste-sum-100, test-waste-csv-exact, ref-injection-csvs, ref-phase17-failures]
    claim-zcutoff-safe:
      status: passed
      summary: "Zero detections above z=0.5 for all 7 h-values at SNR >= 15. Max detected z = 0.204 (h=0.90). z_cut=0.5 provides >= 2.5x safety margin."
      linked_ids: [deliv-yield-script, deliv-yield-report, test-zcutoff-zero, ref-injection-csvs, ref-phase17-zcutsafe]
  deliverables:
    deliv-yield-script:
      status: passed
      path: "analysis/injection_yield.py"
      summary: "Standalone script with load_injection_data(), compute_yield(), compute_waste(), validate_zcutoff(), check_farr_criterion(), plot_waste_breakdown(). Convenience wrappers detection_fraction(), waste_breakdown(), zcutoff_check() provided."
      linked_ids: [claim-yield-per-h, claim-waste-decomposition, claim-zcutoff-safe]
    deliv-yield-report:
      status: passed
      path: ".gpd/phases/18-detection-yield-grid-quality/yield-report.md"
      summary: "Complete yield report with detection yield table, 3 waste decomposition scenarios, z-cutoff validation, Farr criterion, and key findings."
      linked_ids: [claim-yield-per-h, claim-waste-decomposition, claim-zcutoff-safe]
    deliv-waste-figure:
      status: passed
      path: "figures/injection_yield_waste_breakdown.pdf"
      summary: "Two-panel figure: stacked bar chart of waste decomposition (30% failure scenario) and detection fraction vs h."
      linked_ids: [claim-waste-decomposition]
  acceptance_tests:
    test-yield-3sf:
      status: passed
      summary: "All 7 h-values have f_det reported to 3 significant figures: 2.22e-3, 3.00e-3, 2.89e-3, 3.73e-3, 3.88e-3, 5.41e-3, 8.06e-3"
      linked_ids: [claim-yield-per-h, deliv-yield-report]
    test-yield-monotonic:
      status: passed
      summary: "f_det(h=0.90) = 8.06e-3 > f_det(h=0.60) = 2.22e-3. Minor non-monotonicity at h=0.70 (2.89e-3 vs 3.00e-3 at h=0.65) is within Poisson noise (1-sigma)."
      linked_ids: [claim-yield-per-h, deliv-yield-report]
    test-waste-sum-100:
      status: passed
      summary: "CSV 2-way fractions sum to 1.0 exactly (assert abs < 1e-12). Estimated 3-way fractions sum to 1.0 (assert abs < 1e-10). Both verified by runtime assertions."
      linked_ids: [claim-waste-decomposition, deliv-yield-script]
    test-waste-csv-exact:
      status: passed
      summary: "N_sub_threshold + N_det = N_total exactly for all 7 h-values (integer equality assertion)."
      linked_ids: [claim-waste-decomposition, deliv-yield-script]
    test-zcutoff-zero:
      status: passed
      summary: "Explicit query of z > 0.5 AND SNR >= 15 returns 0 events for all 7 h-values."
      linked_ids: [claim-zcutoff-safe, deliv-yield-script]
  references:
    ref-injection-csvs:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "All 262 CSVs loaded and aggregated. 165,000 total events confirmed. CSV columns (z, M, phiS, qS, SNR, h_inj, luminosity_distance) match Phase 17 audit."
    ref-phase17-summary:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Phase 17-01 confirmed injection parameters consistent with simulation pipeline. d_L round-trip accurate to 2e-13."
    ref-phase17-failures:
      status: completed
      completed_actions: [read]
      missing_actions: []
      summary: "Phase 17-02 cataloged 8 exception types. CSV-only-records-successes limitation documented. Failure rate estimated but not quantified from data."
    ref-phase17-zcutsafe:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Phase 17-01 confirmed z_cut=0.5 safe via SNR scaling argument. This plan re-validated with actual data at SNR >= 15."
  forbidden_proxies:
    fp-estimated-yield:
      status: rejected
      notes: "All 262 CSVs loaded and yield computed from actual event counts. No estimation or sampling used."
    fp-average-waste:
      status: rejected
      notes: "Waste breakdown reported per h-value in all 3 scenarios (CSV-only, 30% failure, 50% failure), not just global averages."
    fp-zcutoff-assumed:
      status: rejected
      notes: "z > 0.5 AND SNR >= 15 explicitly queried from data per h-value. Not assumed from Phase 17."
  uncertainty_markers:
    weakest_anchors:
      - "SLURM logs unavailable -- failure fraction bracketed at 30% and 50% assumed rates rather than measured"
    unvalidated_assumptions:
      - "Failure rate uniform across h-values (assumed in 3-way decomposition; likely varies)"
    competing_explanations: []
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: claim-yield-per-h
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-injection-csvs
    comparison_kind: baseline
    metric: detection_count
    threshold: "~663 total detections at SNR >= 15"
    verdict: pass
    recommended_action: "No action. Total matches expectation from plan."
    notes: "663 detections from 165,000 events (f_det = 4.02e-3 overall)."
  - subject_id: claim-zcutoff-safe
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-phase17-zcutsafe
    comparison_kind: benchmark
    metric: detection_count_above_zcutoff
    threshold: "0 detections above z=0.5"
    verdict: pass
    recommended_action: "No action. z_cut=0.5 safe with >= 2.5x margin."
    notes: "Max detected z = 0.204 (h=0.90), well below z=0.5."

duration: 5min
completed: 2026-04-01
---

# Phase 18-01: Injection Campaign Yield Analysis

**Computed per-h detection yield (0.22-0.81%), decomposed GPU waste into failures/sub-threshold/detected, confirmed z>0.5 cutoff safe and Farr criterion satisfied across all 7 h-values**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-01T11:11:39Z
- **Completed:** 2026-04-01T11:16:30Z
- **Tasks:** 2
- **Files created:** 3

## Key Results

- Detection yield ranges from f_det = 2.22e-3 (h=0.60) to 8.06e-3 (h=0.90), a 3.6x increase. Total: 663 detections from 165,000 events at SNR >= 15. [CONFIDENCE: HIGH]
- Over 99% of successfully-computed waveforms are sub-threshold. With 30% estimated failure rate, waveform failures dominate GPU waste (~30% of all compute). [CONFIDENCE: MEDIUM -- failure rate estimated, not measured]
- Zero detections above z = 0.5 for all 7 h-values. Max detected z = 0.204 (h=0.90). z_cut = 0.5 provides >= 2.5x safety margin. [CONFIDENCE: HIGH]
- Farr (2019) criterion satisfied: N_total/N_det >= 124 for all h (minimum at h=0.90). [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Compute detection yield and waste breakdown** - `a6ae4bd` (analyze)
2. **Task 2: Generate yield report and waste visualization** - `2daebdf` (docs)

## Files Created/Modified

- `analysis/injection_yield.py` -- Standalone analysis script with yield computation, waste decomposition, z-cutoff validation, Farr criterion, and waste figure generation
- `.gpd/phases/18-detection-yield-grid-quality/yield-report.md` -- Complete yield report with per-h tables, waste breakdown, z-cutoff confirmation, Farr criterion
- `figures/injection_yield_waste_breakdown.pdf` -- Two-panel figure: stacked bar waste breakdown + detection fraction vs h

## Next Phase Readiness

- Per-h yield data ready for Phase 18-02 (grid quality analysis) and Phase 19 (enhanced sampling design)
- Waveform failure dominance identifies the primary efficiency improvement target
- Detection concentration at z < 0.2 and M ~ 10^5-10^6 (from Phase 17) directly informs importance sampling design
- Farr criterion confirms current injection count is sufficient for P_det estimation

## Contract Coverage

- Claim IDs advanced: claim-yield-per-h -> passed, claim-waste-decomposition -> passed, claim-zcutoff-safe -> passed
- Deliverable IDs produced: deliv-yield-script -> passed, deliv-yield-report -> passed, deliv-waste-figure -> passed
- Acceptance test IDs run: test-yield-3sf -> passed, test-yield-monotonic -> passed, test-waste-sum-100 -> passed, test-waste-csv-exact -> passed, test-zcutoff-zero -> passed
- Reference IDs surfaced: ref-injection-csvs -> read/compare, ref-phase17-summary -> cite, ref-phase17-failures -> read, ref-phase17-zcutsafe -> cite
- Forbidden proxies rejected: fp-estimated-yield, fp-average-waste, fp-zcutoff-assumed
- Decisive comparison verdicts: claim-yield-per-h -> pass, claim-zcutoff-safe -> pass

## Validations Completed

- Integer accounting: N_det + N_sub_threshold = N_total exactly for all 7 h-values (runtime assertion)
- Total events: 165,000 -- matches Phase 17 report
- Total detections: 663 at SNR >= 15 -- matches plan expectation
- Waste fractions sum to 1.0 (within 1e-12 for 2-way, 1e-10 for 3-way)
- z > 0.5 cutoff: zero detections for all h -- consistent with Phase 17 SNR scaling argument
- Farr criterion: all ratios >> 4 (min 124.1 at h=0.90)
- f_det overall trend monotonically increasing with h (minor h=0.70 dip within Poisson noise)

## Decisions & Deviations

None -- plan executed as specified.

## Key Quantities and Uncertainties

| Quantity | Symbol | Value | Uncertainty | Source | Valid Range |
|----------|--------|-------|-------------|--------|-------------|
| Total injection events | N_total | 165,000 | exact | CSV count | All h |
| Total detections (SNR>=15) | N_det | 663 | exact | SNR threshold | All h |
| Overall detection fraction | f_det | 4.02e-3 | Poisson ~4% | N_det/N_total | All h |
| Detection fraction (h=0.60) | f_det(0.60) | 2.22e-3 | Poisson ~14% | 50/22500 | h=0.60 |
| Detection fraction (h=0.90) | f_det(0.90) | 8.06e-3 | Poisson ~8.5% | 137/17000 | h=0.90 |
| Max detected redshift | z_max_det | 0.204 | exact | max(z | SNR>=15) | h=0.90 |
| Min Farr ratio | N_total/N_det | 124.1 | exact | 17000/137 | h=0.90 |

## Figures Produced

| Figure | File | Description | Key Feature |
|--------|------|-------------|-------------|
| Fig. 18.1 | `figures/injection_yield_waste_breakdown.pdf` | Waste decomposition + detection fraction | Left: stacked bars (30% failure scenario); Right: f_det increasing with h |

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|--------------|-----------|----------------|----------------|
| Uniform failure rate across h | Failure rate similar for all h | Unknown | If failure rate varies strongly with h (e.g., h=0.90 has different M distribution) |
| Failure rate bracket [30%, 50%] | True rate within range | Upper/lower bound | If true rate < 30% or > 50% |

## Open Questions

- What is the actual waveform failure rate? (Requires SLURM log rsync from cluster)
- Does the failure rate vary by h-value? (Assumed uniform; may differ)
- Should z_cut be tightened from 0.5 to 0.3 given max detection at z=0.204? (Risk: h-dependent horizon shift)
- What is the optimal sampling strategy for Phase 19? (This plan provides the efficiency baseline)

---

_Phase: 18-detection-yield-grid-quality, Plan 01_
_Completed: 2026-04-01_
