---
phase: 45-p-det-first-bin-asymptote-fix
plan: 04
depth: full
one-liner: "Hybrid 4c patch shipped: anchored interpolator at (0, 0.7931) AND (0.05, 1.0) in _build_grid_1d. 509 CPU tests pass; pre/post probes confirm +0.33 lift at d_L=0.05 (h=0.73). Plan 45-05 cluster re-eval pending."
subsystem: [computation, validation]
tags: [hybrid-anchor, intermediate-anchor, regular-grid-interpolator, detection-probability, bias-correction, hubble-constant, p-det]

requires:
  - phase: 45-03
    provides: "Plan 45-02 single-anchor was sub-discrete-grid-step on cluster (mean(bootstrap_MAP) shifted only -0.0047; discrete MAP unchanged at 0.7650 vs target [0.72, 0.74])"
provides:
  - "Hybrid intermediate anchor at d_L=0.05 wired into _build_grid_1d (commit pending)"
  - "Pre-fix and post-fix interpolator probe JSONs (55 rows each) with row-level deltas — confirms +0.331 lift at d_L=0.05 (h=0.73)"
  - "7 new TestPhase45EmpiricalAnchor regression tests; 4 amended Plan 45-02 tests for hybrid layout (fixture-independent)"
  - "[PHYSICS] commit on simulation_detection_probability.py with old-formula/new-formula/limiting-cases protocol"
affects: [45-05, future-cluster-evals, paper-numerics]

methods:
  added: ["hybrid two-anchor grid layout: prepend BOTH (0, P_MAX) AND (D_INTERMEDIATE, P_INTERMEDIATE) to histogram grid"]
  patterns: ["fixture-independent linear-interp identity assertions for hybrid two-segment formula"]

key-files:
  modified:
    - master_thesis_code/bayesian_inference/simulation_detection_probability.py
    - master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py
    - scripts/bias_investigation/probe_interp_values.py
  created:
    - scripts/bias_investigation/outputs/phase45/pre_hybrid_interp_probe.json
    - scripts/bias_investigation/outputs/phase45/post_hybrid_interp_probe.json
    - .gpd/phases/45-p-det-first-bin-asymptote-fix/45-04-SUMMARY.md

key-decisions:
  - "Used fixed physical position d_L=0.05 Gpc for the intermediate anchor (NOT c_0(h)/2). Rationale: empirical asymptote was derived in fixed physical d_L bins (16/16 detected at d_L<0.10 Gpc), not at fractional positions of c_0(h). Fixed position is h-independent and trivially passes the h-spread regression."
  - "Used point estimate 1.0 for _P_INTERMEDIATE_EMPIRICAL (not Wilson LB 0.806). The conservative-bound role is filled by the d_L=0 anchor 0.7931; the intermediate's job is to capture the empirical asymptote, not provide a second conservative bound."
  - "Kept _P_MAX_EMPIRICAL_ANCHOR = 0.7931 unchanged from Plan 45-02. The two anchors are independent."
  - "Added c_0(h) ≤ 0.05 fallback branch to preserve grid monotonicity (rare; only at very high h). Test coverage included via synthetic small-dl_max fixture."
  - "Did NOT run local --evaluate proxy MAP check; cluster re-eval (Plan 45-05) is the binding gate."
  - "Did NOT modify _build_grid_2d (with_bh_mass channel) — out of Phase 45 scope."

patterns-established:
  - "Pattern: when discrete-grid MAP doesn't move under a sub-step continuous shift, escalate to a stronger lift via an additional anchor at a fixed physical position. Use mean(bootstrap_MAP) and posterior peak height as continuous diagnostics, not just argmax(posterior)."

conventions:
  - "Two h-independent module-level constants control the hybrid: _D_INTERMEDIATE_ANCHOR_GPC = 0.05 (position), _P_INTERMEDIATE_EMPIRICAL = 1.0 (value)."
  - "p_det dimensionless ∈ [0, 1] preserved (clip in calling function)"
  - "d_L in Gpc"
  - "h = H_0 / (100 km/s/Mpc) dimensionless"
  - "SNR_THRESHOLD = 20.0"

plan_contract_ref: ".gpd/phases/45-p-det-first-bin-asymptote-fix/45-04-PLAN.md#/contract"
contract_results:
  claims:
    claim-hybrid-applied:
      status: passed
      summary: "_build_grid_1d now prepends (0.0, 0.7931) AND (0.05, 1.0) when c_0(h) > 0.05; falls back to single-anchor when c_0 ≤ 0.05. Verified: interp(0)=0.7931, interp(0.05)=1.0 exactly, interp(c_0)=p̂(c_0) unchanged. Edge case fallback covered by synthetic-fixture test."
      evidence: "scripts/bias_investigation/outputs/phase45/post_hybrid_interp_probe.json"
    claim-h-independence-preserved:
      status: passed
      summary: "h-spread at d_L=0.05 across {0.65, 0.70, 0.73, 0.80, 0.85} is exactly 0.0 (intermediate is a fixed scalar). Phase 44 regression test_zero_fill_no_h_dependent_step_for_close_dL still passes."
    claim-docstring-and-comment-updated:
      status: passed
      summary: "Inline comment block in _build_grid_1d updated; docstring of _build_grid_1d rewritten to describe the hybrid + fallback; docstring of detection_probability_without_bh_mass_interpolated_zero_fill updated. test_docstring_states_linear_and_anchor extended to assert 'hybrid' or 'intermediate' present."
    claim-quality-gate-green:
      status: passed
      summary: "uv run pytest -m 'not gpu and not slow' --no-cov: 509 passed, 6 skipped, 15 deselected. mypy clean (113 source files). ruff check + format clean on all modified files."
  acceptance_tests:
    test-intermediate-value-at-005-equals-constant:
      status: passed
      summary: "interp(d_L=0.05; h) == 1.0 exactly for h ∈ {0.65, 0.70, 0.73, 0.80, 0.85} (rel-tol 1e-12)."
    test-intermediate-h-independent:
      status: passed
      summary: "h-spread at d_L=0.05 is exactly 0 across 5 h values (the intermediate is a module-level scalar at fixed physical position)."
    test-anchor-at-zero-unchanged:
      status: passed
      summary: "interp(d_L=0; h) == 0.7931 exactly for all 5 h values (Plan 45-02 invariant under hybrid)."
    test-interp-at-c0-unchanged-by-intermediate:
      status: passed
      summary: "interp(c_0(0.70); h=0.70) == p̂(c_0) reproduced from histogram (Plan 45-02 invariant under hybrid; the linear interp on [0.05, c_0] passes through both endpoints)."
    test-below-intermediate-strict-lift:
      status: passed
      summary: "interp(0.025; h=0.73) on production fixture == 0.5*(0.7931+1.0)=0.89655 (linear-interp midpoint identity); strictly exceeds pre-hybrid value 0.7309."
    test-above-intermediate-strict-lift:
      status: passed
      summary: "interp(0.075; h=0.73) on production fixture matches the linear-interp prediction on segment [0.05, c_0] (0.771468 vs predicted 0.7715); strictly exceeds pre-hybrid value 0.6065."
    test-edge-case-small-c0-skips-intermediate:
      status: passed
      summary: "Synthetic fixture with c_0 ≈ 0.0069 → 0.05 NOT in grid axis; no exception raised; value remains in [0, 1]. Fallback branch verified."
    test-zero-fill-boundary-suite-still-passes:
      status: passed
      summary: "All 4 TestZeroFillBoundaryConvention tests pass. test_zero_fill_below_first_bin_is_nonzero_for_valid_dL was amended to use the hybrid two-segment linear-interp formula (still fixture-independent, handles c_0/2 ≷ 0.05 and the fallback branch)."
    test-no-h-step-still-passes:
      status: passed
      summary: "test_zero_fill_no_h_dependent_step_for_close_dL passes unchanged."
    test-cpu-suite-pass:
      status: passed
      summary: "509 passed, 6 skipped, 15 deselected — full 'not gpu and not slow' suite green. (Plan 45-04 added 7 new tests; pre-Plan-45-04 baseline was 502 passed.)"
    test-mypy-ruff-clean:
      status: passed
      summary: "mypy: Success, no issues found in 113 source files. ruff check + format: All checks passed."
    test-pre-post-delta-matches-prediction:
      status: passed
      summary: "post_hybrid_interp_probe.json delta column matches linear-interp predictions to rel-tol 1e-3 at all 6 test points for h=0.73 (see Pre/Post-Hybrid Table below)."
  forbidden_proxies:
    fp-wilson-lb-intermediate:
      status: rejected
      notes: "_P_INTERMEDIATE_EMPIRICAL = 1.0 (point estimate), NOT 0.806 (Wilson LB)."
    fp-fractional-c0:
      status: rejected
      notes: "_D_INTERMEDIATE_ANCHOR_GPC = 0.05 (fixed physical position), NOT c_0(h)/2."
    fp-grid-2d-edits:
      status: rejected
      notes: "git diff master_thesis_code/bayesian_inference/simulation_detection_probability.py | grep '_build_grid_2d' is empty. with_bh_mass channel untouched."
    fp-call-site-edits:
      status: rejected
      notes: "git diff master_thesis_code/bayesian_inference/bayesian_statistics.py is empty. All 6 zero_fill production call sites preserved."
    fp-cluster-eval:
      status: rejected
      notes: "Plan 45-04 ran ONLY local pytest + interpolator probes. Plan 45-05 will run the cluster re-eval with the same acceptance criteria as Plan 45-03."
    fp-no-physics-prefix:
      status: rejected
      notes: "Commit subject begins '[PHYSICS] Phase 45 Plan 45-04: hybrid intermediate anchor at (0.05, 1.0)' per CLAUDE.md /physics-change protocol."
    fp-throwaway-scripts:
      status: rejected
      notes: "Extended scripts/bias_investigation/probe_interp_values.py (added 3 d_L probe points) per feedback_no_adhoc_scripts.md. No sibling scripts created."
    fp-weakened-anchor:
      status: rejected
      notes: "_P_MAX_EMPIRICAL_ANCHOR = 0.7931 unchanged. test_anchor_at_zero_unchanged confirms d_L=0 anchor invariant under hybrid."
  uncertainty_markers:
    weakest_anchors:
      - "Linear interpolation introduces a slope discontinuity at d_L=0.05 (continuous value, discontinuous slope). The slope jumps from +4.138/Gpc on [0, 0.05] to roughly -9.1/Gpc on [0.05, c_0(0.73)]. Acceptable per RESEARCH.md §9 risk register."
      - "Intermediate anchor at d_L=0.05 with value 1.0 violates monotonicity of p_det in d_L on [0, 0.05] (rises from 0.7931 to 1.0). Justified empirically: Wilson LB at d_L=0 is below empirical asymptote; integrand on L_comp remains conservatively lower-bounded."
      - "c_0(h) ≤ 0.05 fallback is rare (very high h) but covered by synthetic fixture test."
    unvalidated_assumptions:
      - "Production cluster posterior MAP shift prediction is −0.025 to −0.05 (continuous), targeting discrete MAP ∈ {0.745, 0.755}. Not validated until Plan 45-05 cluster re-eval."
      - "with_bh_mass channel (untouched by Plan 45-02 and Plan 45-04) sits at MAP=0.7450 (Plan 45-03). It will likely move under Plan 45-04 if and only if the same anchor logic is applied to _build_grid_2d (Phase 46 scope)."
    disconfirming_observations:
      - "If Plan 45-05 cluster MAP > 0.74 (still under-corrected): the hybrid lift wasn't enough; revisit the anchor at d_L=0 (point estimate 0.8873) or move intermediate position to d_L=0.025 to widen the high-lift segment."
      - "If Plan 45-05 cluster MAP < 0.72 (over-corrected): the hybrid is too aggressive; revisit either (a) intermediate value 0.95 instead of 1.0 to soften, or (b) move intermediate position to d_L=0.075 to narrow the [0, 0.05] high-slope segment."

comparison_verdicts:
  - subject_id: claim-hybrid-applied
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-plan45-03-summary
    comparison_kind: cross_method
    metric: lift_at_d_L_005_h_073
    threshold: ">= +0.30"
    verdict: pass
    recommended_action: "Proceed to Plan 45-05 cluster re-eval."
    notes: "Post-hybrid lift at d_L=0.05 (h=0.73) is exactly +0.331277 (1.0 − 0.668723), a 25× larger lift than Plan 45-02 alone."

duration: 90min
completed: 2026-05-01
---

# Phase 45-04 SUMMARY — Hybrid 4c Intermediate Anchor Wired In

**Hybrid 4c patch shipped: anchored interpolator at `(0, 0.7931)` AND `(0.05, 1.0)` in `_build_grid_1d`. 509 CPU tests pass; pre/post probes confirm +0.331 lift at d_L=0.05 (h=0.73). Plan 45-05 cluster re-eval pending.**

## Performance

- **Duration:** ~90 min (research read + edit + tests + probes + SUMMARY)
- **Started:** 2026-05-01 (after Plan 45-03 close)
- **Completed:** 2026-05-01
- **Tasks:** 5 (all completed atomically per Plan 45-04)
- **Files modified:** 3 production/test/script + 2 JSON outputs + 1 SUMMARY

## Constants

- **`_P_MAX_EMPIRICAL_ANCHOR = 0.7931`** — unchanged from Plan 45-02 (Wilson 95% LB).
- **`_D_INTERMEDIATE_ANCHOR_GPC = 0.05`** — NEW; fixed physical position; h-INDEPENDENT.
- **`_P_INTERMEDIATE_EMPIRICAL = 1.0`** — NEW; empirical point estimate from 16/16 detected at d_L<0.10 Gpc; h-INDEPENDENT.

## What changed

- `master_thesis_code/bayesian_inference/simulation_detection_probability.py`:
  - 2 new module-level constants (`_D_INTERMEDIATE_ANCHOR_GPC = 0.05`, `_P_INTERMEDIATE_EMPIRICAL = 1.0`) with full provenance comment block above the constant definitions.
  - `_build_grid_1d` body now prepends BOTH `(0.0, _P_MAX_EMPIRICAL_ANCHOR)` AND `(_D_INTERMEDIATE_ANCHOR_GPC, _P_INTERMEDIATE_EMPIRICAL)` to the histogram grid when `c_0(h) > _D_INTERMEDIATE_ANCHOR_GPC`. Fallback to Plan 45-02 single-anchor layout when `c_0(h) ≤ 0.05`.
  - `_build_grid_1d` docstring updated to describe the hybrid layout + fallback.
  - `detection_probability_without_bh_mass_interpolated_zero_fill` boundary docstring updated (the L860+ region) to describe the piecewise hybrid: `linear_interp((0, P_MAX), (0.05, P_INTERMEDIATE))` on `[0, 0.05]` + `linear_interp((0.05, P_INTERMEDIATE), (c_0, p̂(c_0)))` on `[0.05, c_0]`.

- `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py`:
  - 7 new tests in `TestPhase45EmpiricalAnchor` (intermediate value/h-independence/anchor-at-zero/c_0-unchanged/midpoint identity/above-intermediate prediction/c_0≤0.05 fallback).
  - 1 amended assertion in `TestZeroFillBoundaryConvention.test_zero_fill_below_first_bin_is_nonzero_for_valid_dL` (hybrid two-segment formula; fixture-independent; handles fallback).
  - 3 amended Plan 45-02 tests (`test_interp_at_c0_unchanged_by_anchor`, `test_interp_below_c0_strictly_lifted`, `test_docstring_states_linear_and_anchor`) — fixed `grid[0][1]` indexing under hybrid layout.

- `scripts/bias_investigation/probe_interp_values.py`: extended `DL_VALUES_GPC` to include 3 new probe points (0.005, 0.025, 0.075) for the hybrid acceptance tests. Pre-existing `--output-name` and `--label` CLI flags reused; no new CLI surface added.

## Pre-Hybrid → Post-Hybrid Probe (h=0.73)

| d_L (Gpc) | Pre (Plan 45-02) | Post (Plan 45-04 hybrid) | Δ | Note |
|---|---|---|---|---|
| 0.000 | 0.793100 | 0.793100 | +0.000000 | Plan 45-02 anchor unchanged ✓ |
| 0.001 | 0.790612 | 0.797238 | +0.006626 | Linear interp from 0.7931 → 1.0 |
| 0.005 | 0.780662 | 0.813790 | +0.033128 | Linear interp on [0, 0.05] |
| 0.010 | 0.768225 | 0.834480 | +0.066255 | Linear interp on [0, 0.05] |
| 0.025 | 0.730911 | **0.896550** | **+0.165639** | Midpoint of [0.7931, 1.0] ✓ |
| 0.050 | 0.668723 | **1.000000** | **+0.331277** | Intermediate anchor ✓ |
| 0.075 | 0.606534 | 0.771468 | +0.164934 | Linear interp on [0.05, c_0] |
| 0.100 | 0.544437 | 0.544437 | +0.000000 | Plan 45-02 c_0 invariant ✓ |
| 0.150 | 0.441805 | 0.441805 | +0.000000 | Untouched region |
| 0.200 | 0.339172 | 0.339172 | +0.000000 | Untouched region |
| 0.300 | 0.134825 | 0.134825 | +0.000000 | Untouched region |

**Window-average lift on [0, 0.10]:** ≈ +0.166 (trapezoid integration on the deltas / 0.10) — an order of magnitude larger than Plan 45-02's window-average lift of +0.013–0.06.

**h-independence at d_L=0.05:** spread across h ∈ {0.65, 0.70, 0.73, 0.80, 0.85} = 0.0 exactly.

## Predicted MAP shift on production set

Per Plan 45-03 SUMMARY, the bin-0.755 vs bin-0.765 posterior probability gap is ~0.10. The hybrid window-average lift +0.166 is the right order of magnitude to close that gap.

- **Predicted continuous MAP shift:** −0.025 to −0.05 (downward toward truth h=0.73).
- **Predicted discrete MAP target:** {0.745, 0.755} (vs Plan 45-02 outcome 0.7650).
- **Acceptance window for Plan 45-05 (locked from Plan 45-03 ACCEPTANCE.md §5):**
  - MAP ∈ [0.72, 0.74]
  - 68% bootstrap interval contains 0.73
  - 4-strategy MAP invariance preserved
  - σ_MAP ≤ 0.025

## Plan 45-02 invariants preserved

- **d_L=0 anchor unchanged:** interp(0; h) = 0.7931 for all h (test_anchor_at_zero_unchanged).
- **d_L=c_0 value unchanged:** interp(c_0(h); h) = p̂(c_0) for all h (test_interp_at_c0_unchanged_by_intermediate).
- **4 TestZeroFillBoundaryConvention tests pass:** one assertion amended for hybrid formula; still fixture-independent. Other 3 unchanged.
- **6 zero_fill call sites in `bayesian_statistics.py`:** untouched (test_zero_fill_symmetry_invariant).
- **Phase 44 zero-above-dl_max invariant:** preserved (test_zero_fill_above_dl_max_remains_zero).
- **`_P_MAX_EMPIRICAL_ANCHOR = 0.7931`:** unchanged.
- **`_build_grid_2d`:** untouched.

## Test counts

- **Pre-Plan-45-04 baseline:** 502 passed (collected 508 with 6 skipped, "not gpu and not slow").
- **Post-Plan-45-04:** 509 passed (collected 515; +7 new tests).
- **Plan 45-04 only test file:** 39 passed (32 baseline + 7 new).

## Open follow-ups

- **Plan 45-05** — clone `submit_phase45_eval.sh` (or rerun with new date — `run_dir` auto-namespaces by date), submit cluster re-eval, evaluate against the 4 acceptance criteria from Plan 45-03 ACCEPTANCE.md §5. Acceptance window unchanged.
- **with_bh_mass channel (Phase 46+):** Plan 45-04 only patches `_build_grid_1d` (without_bh_mass channel). The 2D variant `_build_grid_2d` is unchanged; the with_bh_mass MAP=0.7450 from Plan 45-03 will be unaffected by this patch. If the paper's primary channel is with_bh_mass, the Plan 45-04 patch may not move that MAP.

## Self-Check: PASSED

All deliverables exist:
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py` (modified, hybrid + fallback)
- `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` (modified, +7 tests, 4 amendments)
- `scripts/bias_investigation/probe_interp_values.py` (modified, +3 probe points)
- `scripts/bias_investigation/outputs/phase45/pre_hybrid_interp_probe.json` (55 rows)
- `scripts/bias_investigation/outputs/phase45/post_hybrid_interp_probe.json` (55 rows + delta column)
- `.gpd/phases/45-p-det-first-bin-asymptote-fix/45-04-SUMMARY.md` (this file)

Manual limiting-case checks verified post-implementation. Automated regression tests verified (509 pass / 6 skip / 15 deselect; +7 new). mypy clean (113 source files); ruff check + format clean on all modified files.

---

_Phase: 45-p-det-first-bin-asymptote-fix_
_Plan: 04_
_Completed: 2026-05-01_
_Status: SHIPPED LOCAL — awaiting Plan 45-05 cluster re-eval (binding gate)_
