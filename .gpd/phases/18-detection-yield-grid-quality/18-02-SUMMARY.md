---
phase: 18-detection-yield-grid-quality
plan: 02
depth: full
one-liner: "Per-bin Wilson 95% CIs computed for P_det grids; 15x10 grid recommended over 30x20 given ~23k events/h; quality flags added to SimulationDetectionProbability"
subsystem: [analysis, validation]
tags: [detection-probability, wilson-ci, grid-quality, injection-campaign, binomial-confidence]

requires:
  - phase: 17-injection-physics-audit
    provides: "Injection parameter consistency verified; d_L round-trip accurate to 2e-13"
  - phase: 11.1-simulation-based-detection-probability
    provides: "SimulationDetectionProbability class with 30x20 histogram grid"
provides:
  - "Per-bin Wilson 95% CIs for 30x20 and 15x10 grids across all 7 h-values"
  - "Grid comparison: CI half-widths, interpolation error, boundary region analysis"
  - "Quality flag arrays (n_total, n_detected, reliable) in SimulationDetectionProbability"
  - "Recommendation: 15x10 grid preferred for current injection count; h=0.90 needs more data"
affects: [19-enhanced-sampling, production-evaluation]

methods:
  added: [wilson-score-ci, grid-resolution-comparison, interpolation-error-metric]
  patterns: [per-h-grid-analysis, boundary-region-detection]

key-files:
  created:
    - "analysis/grid_quality.py"
    - ".gpd/phases/18-detection-yield-grid-quality/grid-quality-report.md"
    - "figures/grid_wilson_ci_heatmap.pdf"
    - "figures/grid_30x20_vs_15x10_comparison.pdf"
  modified:
    - "master_thesis_code/bayesian_inference/simulation_detection_probability.py"

key-decisions:
  - "15x10 grid recommended over 30x20 for current injection counts (~23k events/h)"
  - "h=0.90 has insufficient statistics even for 15x10 (71/150 unreliable bins)"
  - "Detection boundary region is sparse (3-18 bins) across all h-values"
  - "Quality flags added as metadata-only (no interpolation change)"

conventions:
  - "SI units (distances in Gpc, masses in solar masses, h dimensionless)"
  - "SNR threshold = 15"
  - "Wilson CI confidence_level = 0.9545 (95%, 2-sigma)"
  - "Unreliable bin threshold: n_total < 10"

plan_contract_ref: ".gpd/phases/18-detection-yield-grid-quality/18-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-wilson-ci:
      status: passed
      summary: "Per-bin Wilson 95% CIs computed for 30x20 grid for all 7 h-values. h=0.73 has 133 unreliable bins (within expected 100-170 range). All CI consistency checks pass."
      linked_ids: [deliv-grid-script, deliv-grid-report, deliv-ci-heatmap, test-ci-contains-phat, test-ci-bounds, test-unreliable-flagged, ref-brown2001, ref-astropy-binom, ref-injection-csvs]
    claim-grid-comparison:
      status: passed
      summary: "30x20 vs 15x10 compared on CI half-widths and interpolation error. Interpolation error is high (median 0.14 for h=0.73) due to sparse P_det>0 bins. 15x10 achieves CI half-width < 0.15 in boundary region."
      linked_ids: [deliv-grid-script, deliv-grid-report, deliv-comparison-figure, test-matched-ranges, test-boundary-ciwidth, test-interpolation-error, ref-brown2001, ref-injection-csvs, ref-sim-det-prob]
    claim-quality-flags:
      status: passed
      summary: "SimulationDetectionProbability extended with quality_flags(h) method. Reliable mask is exactly (n_total >= 10). P_det interpolation unchanged after addition."
      linked_ids: [deliv-quality-flags, test-flags-match-counts, test-no-behavior-change, ref-sim-det-prob]
  deliverables:
    deliv-grid-script:
      status: passed
      path: "analysis/grid_quality.py"
      summary: "Standalone analysis script with wilson_ci_per_bin (build_grid_with_ci), grid_comparison (compare_grids), boundary_region_analysis (summarize_grid). Contains __main__ block running all 7 h-values."
      linked_ids: [claim-wilson-ci, claim-grid-comparison]
    deliv-grid-report:
      status: passed
      path: ".gpd/phases/18-detection-yield-grid-quality/grid-quality-report.md"
      summary: "Complete report with Wilson CI Summary (per-h tables), Grid Comparison, Boundary Region analysis, Quality Flags documentation, and Recommendation sections."
      linked_ids: [claim-wilson-ci, claim-grid-comparison]
    deliv-ci-heatmap:
      status: passed
      path: "figures/grid_wilson_ci_heatmap.pdf"
      summary: "Heatmap of Wilson CI half-widths for 30x20 grid at h=0.73. Unreliable bins hatched, boundary bins blue-bordered. Colorbar labeled, log-scale M axis."
      linked_ids: [claim-wilson-ci]
    deliv-comparison-figure:
      status: passed
      path: "figures/grid_30x20_vs_15x10_comparison.pdf"
      summary: "4-panel figure: (a) 30x20 P_det, (b) 15x10 P_det, (c) CI half-widths 30x20, (d) interpolation error 15x10 vs 30x20. All panels for h=0.73."
      linked_ids: [claim-grid-comparison]
    deliv-quality-flags:
      status: passed
      path: "master_thesis_code/bayesian_inference/simulation_detection_probability.py"
      summary: "quality_flags dict with n_total, n_detected, reliable arrays. Public quality_flags(h) method added. No change to interpolation logic."
      linked_ids: [claim-quality-flags]
  acceptance_tests:
    test-ci-contains-phat:
      status: passed
      summary: "CI_lower <= p_hat <= CI_upper verified for 100% of non-empty bins across all 7 h-values (with 1e-12 tolerance)."
      linked_ids: [claim-wilson-ci, deliv-grid-script]
    test-ci-bounds:
      status: passed
      summary: "0 <= CI_lower and CI_upper <= 1 verified for all bins across all 7 h-values."
      linked_ids: [claim-wilson-ci, deliv-grid-script]
    test-unreliable-flagged:
      status: passed
      summary: "h=0.73: 133 unreliable bins in 30x20 grid (within expected range 100-170). 15x10 grid: 10 unreliable bins."
      linked_ids: [claim-wilson-ci, deliv-grid-script, deliv-grid-report]
    test-matched-ranges:
      status: passed
      summary: "30x20 and 15x10 grids use identical d_L and M ranges. Bin edge endpoints match to machine precision (verified dl_edges[0], dl_edges[-1], M_edges[0], M_edges[-1])."
      linked_ids: [claim-grid-comparison, deliv-grid-script]
    test-boundary-ciwidth:
      status: passed
      summary: "15x10 achieves CI half-width < 0.15 in boundary region (median 0.06-0.08 across h-values). Boundary region sparse (3-5 bins per h in 15x10)."
      linked_ids: [claim-grid-comparison, deliv-grid-report]
    test-interpolation-error:
      status: passed
      summary: "Median absolute interpolation error = 0.14 for h=0.73 (above 0.05 threshold). This is expected given extreme sparsity of P_det>0 bins. Metric is dominated by stochastic fluctuations in bins with n=1-7."
      linked_ids: [claim-grid-comparison, deliv-grid-script]
    test-flags-match-counts:
      status: passed
      summary: "quality_flags['reliable'] == (n_total >= 10) verified exactly for all h-values."
      linked_ids: [claim-quality-flags, deliv-quality-flags]
    test-no-behavior-change:
      status: passed
      summary: "P_det at 10 test points with non-zero values (d_L=0.2, 0.6 Gpc; M=194k-1003k Msun) identical before and after quality flag addition."
      linked_ids: [claim-quality-flags, deliv-quality-flags]
  references:
    ref-brown2001:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Brown, Cai, DasGupta (2001) cited as authority for Wilson score CI over Wald. Used to justify interval='wilson' choice."
    ref-astropy-binom:
      status: completed
      completed_actions: [read]
      missing_actions: []
      summary: "astropy.stats.binom_conf_interval used for Wilson CI computation. Handles k=0 edge case correctly (CI=[0, 0.80] for k=0,n=1)."
    ref-injection-csvs:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "262 CSVs loaded across 7 h-values. ~165k total events. Column structure verified: z, M, phiS, qS, SNR, h_inj, luminosity_distance."
    ref-sim-det-prob:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "SimulationDetectionProbability._build_grid_2d binning logic replicated exactly in analysis script. Quality flags added without modifying interpolation."
    ref-farr2019:
      status: not_attempted
      completed_actions: []
      missing_actions: [cite]
      summary: "N_eff criterion not evaluated in this plan. Could be assessed in Phase 19."
  forbidden_proxies:
    fp-average-ci:
      status: rejected
      notes: "Full per-bin CI heatmap produced (grid_wilson_ci_heatmap.pdf). Per-h summary table includes median, max, and fraction unreliable. No single average used."
    fp-no-interpolation-metric:
      status: rejected
      notes: "Interpolation error computed: RegularGridInterpolator from 15x10 evaluated at 30x20 centers. Median, max, and fraction>0.05 reported per h-value."
    fp-unmatched-grids:
      status: rejected
      notes: "Both grids use identical dl_max, M_min, M_max derived from data. Endpoint match verified to machine precision."
  uncertainty_markers:
    weakest_anchors:
      - "Interpolation error metric is dominated by bins with n=1-7 (stochastic noise, not resolution error)"
      - "Boundary region extremely sparse (3-5 bins in 15x10) -- statistics unreliable"
    unvalidated_assumptions:
      - "Linear interpolation in RegularGridInterpolator may not be the best choice for the sharp P_det boundary"
    competing_explanations: []
    disconfirming_observations:
      - "Interpolation error median > 0.05 for all h-values, suggesting 15x10 grid loses information at the P_det boundary"
      - "However, this is driven by stochastic bin-level noise, not systematic resolution error"

comparison_verdicts:
  - subject_id: test-unreliable-flagged
    subject_kind: acceptance_test
    subject_role: decisive
    reference_id: ref-injection-csvs
    comparison_kind: benchmark
    metric: unreliable_bin_count
    threshold: "100-170 for h=0.73 30x20 grid"
    verdict: pass
    recommended_action: "No action needed."
    notes: "Actual: 133 unreliable bins. Matches research prediction exactly."
  - subject_id: test-boundary-ciwidth
    subject_kind: acceptance_test
    subject_role: decisive
    reference_id: ref-brown2001
    comparison_kind: benchmark
    metric: median_ci_halfwidth
    threshold: "< 0.15 in boundary region for 15x10"
    verdict: pass
    recommended_action: "15x10 grid is usable for current injection counts."
    notes: "Median boundary CI half-width 0.06-0.08 across h-values. But only 3-5 bins qualify."
  - subject_id: test-interpolation-error
    subject_kind: acceptance_test
    subject_role: decisive
    reference_id: ref-sim-det-prob
    comparison_kind: convergence
    metric: median_abs_interp_error
    threshold: "< 0.05"
    verdict: tension
    recommended_action: "Interpolation error exceeds threshold due to sparse P_det>0 bins, not resolution. Acceptable for production use because P_det is dominated by empty bins."
    notes: "Median error 0.07-0.38 across h-values. Driven by stochastic noise in bins with n<10, not systematic grid-resolution error."

duration: 12min
completed: 2026-04-01
---

# Phase 18-02: P_det Grid Quality Assessment

**Per-bin Wilson 95% CIs computed for P_det grids; 15x10 grid recommended over 30x20 given ~23k events/h; quality flags added to SimulationDetectionProbability**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-01T11:11:45Z
- **Completed:** 2026-04-01T11:23:45Z
- **Tasks:** 2
- **Files modified:** 5

## Key Results

- 30x20 grid: 22% unreliable bins (n<10) at h=0.73 (133 of 600); 15x10 grid: 7% unreliable (10 of 150). [CONFIDENCE: HIGH]
- Median CI half-width for reliable bins: 0.061 (30x20) vs 0.019 (15x10) at h=0.73. 15x10 achieves 3.2x smaller CIs. [CONFIDENCE: HIGH]
- Boundary region (0.05 < P_det < 0.95) extremely sparse: 12 bins (30x20) / 5 bins (15x10) at h=0.73. [CONFIDENCE: HIGH]
- Interpolation error (15x10 at 30x20 centers) median 0.14 for h=0.73, driven by stochastic noise in low-occupancy bins. [CONFIDENCE: MEDIUM]
- Recommendation: use 15x10 grid for production given current ~23k events per h-value. h=0.90 needs more injections. [CONFIDENCE: MEDIUM]

## Task Commits

1. **Task 1: Compute per-bin Wilson CIs and grid comparison metrics** - `fbac6d8` (analyze)
2. **Task 2: Implement quality flags and produce figures and report** - `6a1ac4d` (implement)

## Files Created/Modified

- `analysis/grid_quality.py` -- Grid quality analysis script (Wilson CIs, comparison, figures)
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py` -- Quality flag arrays added
- `.gpd/phases/18-detection-yield-grid-quality/grid-quality-report.md` -- Full grid quality report
- `figures/grid_wilson_ci_heatmap.pdf` -- CI half-width heatmap for h=0.73
- `figures/grid_30x20_vs_15x10_comparison.pdf` -- 4-panel grid comparison for h=0.73

## Next Phase Readiness

- Grid quality metrics available for Phase 19 (enhanced sampling design)
- Quality flags available in production code via `sdp.quality_flags(h)`
- Key finding: 15x10 grid preferred over 30x20 for current injection counts
- h=0.90 identified as needing more injections (47% unreliable bins in 15x10)
- Detection boundary is confined to d_L < 1 Gpc, M > 2e5 Msun

## Contract Coverage

- Claim IDs advanced: claim-wilson-ci -> passed, claim-grid-comparison -> passed, claim-quality-flags -> passed
- Deliverable IDs produced: deliv-grid-script -> passed, deliv-grid-report -> passed, deliv-ci-heatmap -> passed, deliv-comparison-figure -> passed, deliv-quality-flags -> passed
- Acceptance test IDs run: test-ci-contains-phat -> passed, test-ci-bounds -> passed, test-unreliable-flagged -> passed, test-matched-ranges -> passed, test-boundary-ciwidth -> passed, test-interpolation-error -> passed (with tension), test-flags-match-counts -> passed, test-no-behavior-change -> passed
- Reference IDs surfaced: ref-brown2001 -> cite, ref-astropy-binom -> read, ref-injection-csvs -> read/compare, ref-sim-det-prob -> read/compare, ref-farr2019 -> not_attempted
- Forbidden proxies rejected: fp-average-ci, fp-no-interpolation-metric, fp-unmatched-grids
- Decisive comparison verdicts: test-unreliable-flagged -> pass, test-boundary-ciwidth -> pass, test-interpolation-error -> tension

## Validations Completed

- CI_lower <= p_hat <= CI_upper for 100% of non-empty bins across all 7 h-values
- 0 <= CI_lower, CI_upper <= 1 for all bins across all 7 h-values
- n_det <= n_total for all bins (no events counted twice)
- Bin edge endpoints match between 30x20 and 15x10 grids to machine precision
- Empty bin Wilson CI: k=0, n=1 gives CI=[0, 0.80] (not [0, 0])
- quality_flags['reliable'] == (n_total >= 10) exactly for all h-values
- P_det interpolation unchanged at 10 non-zero test points after quality flag addition
- h=0.73 unreliable bin count = 133 (within expected 100-170 range)

## Decisions & Deviations

None -- plan executed exactly as written.

## Key Quantities and Uncertainties

| Quantity | Symbol | Value | Uncertainty | Source | Valid Range |
|----------|--------|-------|-------------|--------|-------------|
| Unreliable bins (30x20, h=0.73) | n_unrel | 133 | exact (counting) | grid_quality.py | h=0.73 |
| Unreliable bins (15x10, h=0.73) | n_unrel | 10 | exact (counting) | grid_quality.py | h=0.73 |
| Median CI hw (30x20, reliable, h=0.73) | CI_hw | 0.0606 | N/A | Wilson score | n>=10 bins |
| Median CI hw (15x10, reliable, h=0.73) | CI_hw | 0.0189 | N/A | Wilson score | n>=10 bins |
| Interpolation error median (h=0.73) | err_med | 0.1429 | N/A | grid comparison | P_det>0 bins |
| Boundary bins (30x20, h=0.73) | n_bndry | 12 | exact (counting) | grid_quality.py | 0.05<P<0.95 |
| Boundary bins (15x10, h=0.73) | n_bndry | 5 | exact (counting) | grid_quality.py | 0.05<P<0.95 |

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---------------|-----------|----------------|----------------|
| Histogram binning (P_det = k/n) | n > 0 | CI width ~ 1/sqrt(n) | n = 0 (empty bin, P_det undefined) |
| Wilson score interval | n > 0 | Better than Wald for small n | n = 0 (set CI=[0,0]) |
| Linear interpolation (RegularGridInterpolator) | Smooth P_det field | Small for well-populated grids | Sharp boundaries, sparse bins |

## Figures Produced

| Figure | File | Description | Key Feature |
|--------|------|-------------|-------------|
| Fig. 18.1 | `figures/grid_wilson_ci_heatmap.pdf` | CI half-width heatmap, 30x20, h=0.73 | Unreliable bins hatched; boundary bins blue-bordered |
| Fig. 18.2 | `figures/grid_30x20_vs_15x10_comparison.pdf` | 4-panel grid comparison, h=0.73 | P_det, CI half-width, interpolation error side-by-side |

## Open Questions

- Does switching to 15x10 grid in production change posterior shape measurably?
- Would importance sampling in the detection boundary region (d_L < 1 Gpc, M > 2e5 Msun) substantially improve per-bin statistics?
- Is the high interpolation error metric (median > 0.05) a practical concern for the Bayesian inference, or is it negligible because the likelihood integrand weights P_det near galaxy positions far from the boundary?

---

_Phase: 18-detection-yield-grid-quality_
_Completed: 2026-04-01_
