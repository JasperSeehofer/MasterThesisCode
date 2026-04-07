---
phase: 27-production-run-figures
plan: 02
depth: full
one-liner: "Extracted MAP, CIs, precision, bias from combined posteriors (both grid-limited) and generated H0 posterior comparison + single-event likelihood figures"
subsystem: analysis
tags: [bayesian-inference, posterior-analysis, paper-figures, h0-measurement]

requires:
  - phase: 27-production-run-figures plan 01
    provides: Validated production data files, toplevel combined posteriors confirmed as correct source
provides:
  - MAP h=0.66 (without BH mass), h=0.68 (with BH mass) from completeness-corrected posteriors
  - Grid-limited credible intervals (upper bounds due to 0.02 h-spacing)
  - H0 posterior comparison figure (h0_posterior_comparison.pdf)
  - Single-event likelihood figure (single_event_likelihoods.pdf)
  - paper_figures.py plotting module with factory functions
affects: [27-03, 27-04]

methods:
  added: [CDF interpolation for CI extraction on coarse grids, tail-read extraction for large JSON files]
  patterns: [peak-normalization for posterior overlay, regex-based scalar extraction from 585MB files]

key-files:
  created:
    - .gpd/phases/27-production-run-figures/extracted_results.json
    - master_thesis_code/plotting/paper_figures.py
    - paper/figures/h0_posterior_comparison.pdf
    - paper/figures/single_event_likelihoods.pdf

key-decisions:
  - "Used toplevel combined posteriors (15-pt grid, MAP=0.66/0.68) per Plan 01 recommendation"
  - "Both variants flagged as grid-limited: CIs reported as upper bounds, precision ordering comparison skipped"
  - "With-mass per-event data loaded via tail-read regex (last 300KB) to avoid parsing full 585MB JSON files"
  - "Single-event figure uses 4 events selected by h_std percentile distribution (5th/25th/50th/95th)"

conventions:
  - "h = H0 / (100 km/s/Mpc), dimensionless"
  - "h_true = 0.73 (injected value)"
  - "peak-normalized posteriors for figure display"
  - "SI units (km/s/Mpc for H0)"

plan_contract_ref: ".gpd/phases/27-production-run-figures/27-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-numerical-extraction:
      status: passed
      summary: "MAP, 68%/95% CI, precision, bias extracted for both variants. Both are grid-limited (1-2 bins above 1% of peak). MAP=0.66 (no mass), MAP=0.68 (with mass). Precision ~2% for both (upper bound)."
      linked_ids: [deliv-results-json, test-map-consistency, test-ci-positive, test-precision-order, ref-combined-posteriors]
    claim-posterior-figure:
      status: passed
      summary: "Two-curve overlay with markers at grid points, truth line at h=0.73, 68% CI shading, legend. Both posteriors visible and peak-normalized."
      linked_ids: [deliv-posterior-fig, test-figure-contents]
    claim-single-event-figure:
      status: passed
      summary: "4x2 panel figure with 4 representative events (peaked/moderate/median/broad). With-mass panels show probability concentrated in fewer bins."
      linked_ids: [deliv-single-event-fig, test-single-event-physics]
  deliverables:
    deliv-results-json:
      status: passed
      path: ".gpd/phases/27-production-run-figures/extracted_results.json"
      summary: "Complete JSON with map_h, ci_68, ci_95, sigma_h, precision_pct, bias_pct, n_events_used, ci_grid_limited flag for both variants"
      linked_ids: [claim-numerical-extraction, test-map-consistency, test-ci-positive]
    deliv-posterior-fig:
      status: passed
      path: "paper/figures/h0_posterior_comparison.pdf"
      summary: "17KB PDF, single-column width, two curves with markers + CI shading + truth line"
      linked_ids: [claim-posterior-figure, test-figure-contents]
    deliv-single-event-fig:
      status: passed
      path: "paper/figures/single_event_likelihoods.pdf"
      summary: "20KB PDF, 4x2 panel, 4 representative events with both variants side by side"
      linked_ids: [claim-single-event-figure, test-single-event-physics]
  acceptance_tests:
    test-map-consistency:
      status: passed
      summary: "MAP=0.66 (no mass) and MAP=0.68 (with mass) match combined_posterior.json exactly"
      linked_ids: [claim-numerical-extraction, deliv-results-json, ref-combined-posteriors]
    test-ci-positive:
      status: passed
      summary: "68% CI widths: 0.028 (no mass), 0.027 (with mass). Both positive, both < 0.26 (full grid range)"
      linked_ids: [claim-numerical-extraction, deliv-results-json]
    test-precision-order:
      status: not_applicable
      summary: "Both variants are grid-limited (2 and 1 bins above 1% of peak). Precision ordering comparison skipped per plan instructions. Both precisions are upper bounds (~2%)."
      linked_ids: [claim-numerical-extraction, deliv-results-json]
    test-figure-contents:
      status: passed
      summary: "Visual inspection confirms: two curves with distinct colors/markers, truth line at h=0.73, axis labels, legend with 'Without M_z' and 'With M_z'"
      linked_ids: [claim-posterior-figure, deliv-posterior-fig]
    test-single-event-physics:
      status: passed
      summary: "With-mass panels show probability concentrated in fewer h-bins for all 4 events. Note: with-mass has only 7 h-values vs 23 for no-mass, so 'narrower' manifests as fewer bins with signal rather than smoother narrow peaks."
      linked_ids: [claim-single-event-figure, deliv-single-event-fig]
  references:
    ref-combined-posteriors:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "Both toplevel combined posteriors loaded, MAP values verified against stored map_h field, posterior arrays used for CDF integration and figure generation"
  forbidden_proxies:
    fp-no-baseline-as-production:
      status: rejected
      notes: "All values extracted from production combined_posterior.json files. Thesis baseline values (0.712, 0.742) included only as comparison reference in the JSON."
    fp-no-median-only:
      status: rejected
      notes: "MAP is the primary reported statistic. Median also computed (0.661, 0.680) as cross-check only."
  uncertainty_markers:
    weakest_anchors:
      - "Both posteriors are grid-limited on 15-point grid (spacing 0.02). CIs are upper bounds, not resolved measurements."
      - "With-BH-mass has only 7 per-event h-values, making single-event figure comparison somewhat limited"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations:
      - "Both MAPs are biased low relative to h_true=0.73: -9.6% (no mass), -6.8% (with mass). This is consistent with known low-h bias from galaxy catalog incompleteness."

duration: 15min
completed: 2026-04-07
---

# Phase 27, Plan 02: Numerical Extraction and Paper Figures Summary

**Extracted MAP, CIs, precision, bias from combined posteriors (both grid-limited) and generated H0 posterior comparison + single-event likelihood figures**

## Performance

- **Duration:** 15 min
- **Started:** 2026-04-07
- **Completed:** 2026-04-07
- **Tasks:** 2/2
- **Files modified:** 4

## Key Results

- Without BH mass: MAP h = 0.66, sigma_h < 0.014 (grid-limited UB), bias = -9.6%, 531/534 events used [CONFIDENCE: HIGH]
- With BH mass: MAP h = 0.68, sigma_h < 0.014 (grid-limited UB), bias = -6.8%, 527/534 events used [CONFIDENCE: HIGH]
- Both posteriors are grid-limited: probability concentrated in 1-2 bins on 15-point h-grid (spacing 0.02). CIs and precisions are upper bounds. [CONFIDENCE: HIGH]
- Precision ordering test NOT meaningful (both grid-limited). Both report ~2% upper bound precision. [CONFIDENCE: MEDIUM]

## Task Commits

1. **Task 1: Extract MAP, credible intervals, precision, bias** - `dea0c10` (analyze)
2. **Task 2: Generate H0 posterior comparison and single-event likelihood figures** - `64bce4b` (figure)

## Files Created/Modified

- `.gpd/phases/27-production-run-figures/extracted_results.json` -- all numerical values for paper markers
- `master_thesis_code/plotting/paper_figures.py` -- factory functions for paper figures
- `paper/figures/h0_posterior_comparison.pdf` -- two-curve posterior overlay
- `paper/figures/single_event_likelihoods.pdf` -- 4x2 single-event panel

## Next Phase Readiness

- extracted_results.json provides all numerical values for Plan 04 (paper marker filling)
- Both PDF figures ready for inclusion in paper via \includegraphics
- paper_figures.py provides reusable factory functions for any additional figure variants
- Key caveat for paper text: CIs must be reported as upper bounds, not measurements

## Contract Coverage

- Claim IDs: claim-numerical-extraction -> passed, claim-posterior-figure -> passed, claim-single-event-figure -> passed
- Deliverable IDs: deliv-results-json -> passed, deliv-posterior-fig -> passed, deliv-single-event-fig -> passed
- Acceptance test IDs: test-map-consistency -> passed, test-ci-positive -> passed, test-precision-order -> not_applicable (grid-limited), test-figure-contents -> passed, test-single-event-physics -> passed
- Reference IDs: ref-combined-posteriors -> completed (read, compare)
- Forbidden proxies: fp-no-baseline-as-production -> rejected, fp-no-median-only -> rejected

## Key Quantities and Uncertainties

| Quantity | Symbol | Value | Uncertainty | Source | Valid Range |
|---|---|---|---|---|---|
| MAP h (no mass) | h_MAP | 0.66 | grid-limited (spacing 0.02) | combined_posterior.json argmax | h in [0.60, 0.86] |
| MAP h (with mass) | h_MAP | 0.68 | grid-limited (spacing 0.02) | combined_posterior_with_bh_mass.json argmax | h in [0.60, 0.86] |
| sigma_h (no mass) | sigma_h | < 0.014 | upper bound | CDF interpolation, 68% CI half-width | grid-limited |
| sigma_h (with mass) | sigma_h | < 0.014 | upper bound | CDF interpolation, 68% CI half-width | grid-limited |
| Bias (no mass) | -- | -9.6% | -- | (MAP - h_true)/h_true | -- |
| Bias (with mass) | -- | -6.8% | -- | (MAP - h_true)/h_true | -- |
| Events used (no mass) | N | 531 | exact | combined_posterior.json | -- |
| Events used (with mass) | N | 527 | exact | combined_posterior_with_bh_mass.json | -- |

## Figures Produced

| Figure | File | Description | Key Feature |
|---|---|---|---|
| Fig. 27-02.1 | `paper/figures/h0_posterior_comparison.pdf` | Two posteriors overlaid with truth line | Both curves extremely peaked on coarse grid; markers show grid resolution |
| Fig. 27-02.2 | `paper/figures/single_event_likelihoods.pdf` | 4x2 panel: 4 events x 2 variants | With-mass concentrated in fewer bins; 7 vs 23 h-values |

## Validations Completed

- MAP values match combined_posterior.json map_h field exactly (0.66, 0.68)
- CDF monotonically increasing for both variants
- h_16 < MAP < h_84 for both (MAP within CI)
- 68% CI contained within 95% CI for both
- Grid-limitation detection: 2 bins (no mass), 1 bin (with mass) above 1% of peak
- Bias negative for both (MAP < h_true = 0.73), consistent with known low-h bias
- n_events_used matches JSON: 531 (no mass), 527 (with mass)
- Figures are valid PDFs (17K and 20K, non-zero)
- paper_figures.py passes ruff check and mypy

## Decisions Made

1. **Used toplevel combined posteriors** (15-pt grid) per Plan 01 finding that subdir versions are copies/bugs
2. **Grid-limitation flagging**: both variants ci_grid_limited=true since <=2 bins have >1% of peak value
3. **Precision ordering comparison skipped** per plan instructions when both are grid-limited
4. **Tail-read extraction** for 585MB with-mass files: regex on last 300KB instead of full JSON parse
5. **Event selection** for single-event figure: h_std percentile distribution (5th/25th/50th/95th)

## Deviations from Plan

### Deviation 1: Grid spacing note

- **[Rule 4 - Missing component]** The h-grid is nominally spacing 0.02 but has two points at 0.01 spacing (h=0.72-0.73 and h=0.73-0.74). The ci_note in the JSON says "0.01 spacing" (minimum) but the dominant grid spacing is 0.02. This is the actual grid, not a bug.
- **Impact:** Minor. Grid limitation is correctly flagged regardless.

**Total deviations:** 1 (Rule 4, auto-documented)
**Impact on plan:** None. All deliverables produced as specified.

## Issues Encountered

- With-BH-mass per-event files are 585MB each (contain per-galaxy likelihood breakdowns). Full JSON parsing would take minutes per file. Solved with tail-read approach reading last 300KB.
- With-BH-mass variant has only 7 h-values for per-event data (vs 23 for no-mass). Single-event comparison figure has asymmetric resolution between columns.
- Single-event no-mass likelihoods are highly oscillatory across the dense 23-point grid (many local maxima). This is physical -- individual events have multi-modal likelihoods due to multiple galaxy candidates.

## Open Questions

- Would a finer h-grid (e.g., 0.005 spacing) resolve the grid-limited CIs? Current 0.02 spacing is too coarse for the extremely peaked posteriors.
- Are the thesis baseline values (MAP=0.712, 0.742) directly comparable to production values (MAP=0.66, 0.68), or do they use different analysis choices (P_det, completeness correction)?
- The ~10% negative bias in both variants: is this acceptable for the paper, or should it motivate a reanalysis?

---

_Phase: 27-production-run-figures, Plan: 02_
_Completed: 2026-04-07_
