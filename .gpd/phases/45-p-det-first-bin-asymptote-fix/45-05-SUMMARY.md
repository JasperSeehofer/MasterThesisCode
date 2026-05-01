---
phase: 45-p-det-first-bin-asymptote-fix
plan: 05
depth: full
one-liner: "Plan 45-04 hybrid cluster re-eval: MAP shifted 0.7650 → 0.7550 (one discrete grid step toward truth h=0.73); still outside [0.72, 0.74] by 0.015. Branch B (UNDER-CORRECTION) again — patch worked but conservative-bound + point-estimate anchor combo undershoots."
subsystem: [cluster-execution, validation, acceptance]
tags: [cluster-eval, bootstrap, acceptance-gate, under-correction-improved, plan-45-06-pending]

requires:
  - phase: 45-04
    provides: "Hybrid intermediate anchor (0.05, 1.0) layered on top of (0, 0.7931) anchor; +0.331 lift at d_L=0.05 (h=0.73) per pre/post probes"
provides:
  - "Cluster re-eval posterior under hybrid (results/phase45_v2_posteriors/combined_posterior.json — MAP=0.7550, n=412)"
  - "with_bh_mass posterior (results/phase45_v2_posteriors_with_bh_mass/combined_posterior.json — MAP=0.7450, slightly tightened vs Plan 45-03)"
  - "Bootstrap summary: post_hybrid_bootstrap_summary.json — std=0.0099, interval_68=[0.7450, 0.7650]"
  - "Empirical evidence: Plan 45-04 hybrid ≈2× as effective as Plan 45-02 alone but still ~3-7× short of acceptance window"
affects: [45-06-future, paper-numerics, with-bh-mass-channel]

methods:
  added: []
  patterns: ["empirically-calibrated escalation rounds: each anchor revision predicts a continuous MAP shift; discrete MAP either flips or doesn't; bootstrap interval is the binding gate"]

key-files:
  created:
    - results/phase45_v2_posteriors/combined_posterior.json
    - results/phase45_v2_posteriors_with_bh_mass/combined_posterior.json
    - scripts/bias_investigation/outputs/phase45/post_hybrid_bootstrap_summary.json
    - .gpd/phases/45-p-det-first-bin-asymptote-fix/45-05-SUMMARY.md

key-decisions:
  - "Did NOT mark Phase 45 complete: MAP=0.7550 fails primary acceptance gate [0.72, 0.74]; though improved by one discrete grid step from Plan 45-03 outcome 0.7650."
  - "Did NOT close any GitHub issues: Phase 45 outcome is still escalate."
  - "Reused cluster/submit_phase45_eval.sh unchanged (auto-namespaces by date; same source CRBs); cluster's run_phase45_20260501 was overwritten with Plan 45-05 results (Plan 45-03 results preserved in committed local results/phase45_posteriors/)."
  - "Confirmed via grep on cluster source that hybrid constants _D_INTERMEDIATE_ANCHOR_GPC=0.05 and _P_INTERMEDIATE_EMPIRICAL=1.0 are present, ruling out Branch D (NO-EFFECT)."

patterns-established:
  - "Pattern: when continuous Δ MAP per round is ≈ −0.005 and discrete grid step is 0.005, each round either moves discrete MAP or doesn't; ~3 rounds of similar magnitude were needed to cross from 0.7650 to ≤0.74. The Plan 45-04 hybrid achieved ~2× the lift of Plan 45-02 alone (one discrete step crossed); a third step needs another ≈+0.05 average lift on [0, c_0]."

conventions:
  - "without_bh_mass channel uses _build_grid_1d (1D interpolator, ANCHORED with hybrid in Plan 45-04)"
  - "with_bh_mass channel uses _build_grid_2d (2D interpolator, NOT anchored — out of scope for Plan 45)"

contract_results:
  acceptance_tests:
    test-pre-flight-commit-match:
      status: passed
      summary: "Cluster HEAD descends from Plan 45-04 [PHYSICS] commit 4a260e9; verified via git merge-base --is-ancestor."
    test-pre-flight-tests-pass:
      status: passed
      summary: "Cluster pytest on test_simulation_detection_probability.py: 39 passed (including 4 TestZeroFillBoundaryConvention + 13 TestPhase45EmpiricalAnchor tests)."
    test-map-in-range:
      status: failed
      summary: "without_bh_mass MAP=0.7550 (Plan 45-03 was 0.7650). Outside the locked acceptance window [0.72, 0.74] by 0.015. The hybrid moved the MAP by exactly one discrete grid step (Δh=0.005) toward truth; mean(bootstrap_MAP) shifted 0.7603→0.7551 (Δcontinuous=-0.005)."
    test-bootstrap-coverage:
      status: failed
      summary: "interval_68=[0.7450, 0.7650]; truth h=0.73 OUTSIDE. Verdict: systematic. (Bootstrap interval narrowed slightly: pre-Plan-45-04 was [0.7450, 0.7650], same width but mass redistributed within the interval.)"
    test-zero-strategies-equal:
      status: passed
      summary: "All 4 strategies produce MAP=0.7550 with 412 events used, 0 excluded. Phase 44 STAT-03 invariant preserved."
    test-bootstrap-sigma-stable:
      status: passed
      summary: "post-fix std=0.0099 (pre-fix Step 0 baseline 0.0114; Plan 45-03 0.0117). Slight TIGHTENING — within stability gate."

duration: 25min
completed: 2026-05-01
---

# Phase 45-05 SUMMARY — Hybrid Cluster Re-Eval ESCALATE-AGAIN (Branch B improved)

**Plan 45-04 hybrid cluster re-eval: MAP shifted 0.7650 → 0.7550 (one discrete grid step toward truth). Still outside [0.72, 0.74] by 0.015. Branch B (UNDER-CORRECTION) again — patch worked but conservative-bound + point-estimate anchor combo undershoots.**

## Executive Summary

The hybrid 4c patch (Plan 45-04) achieved its predicted local-probe behaviour exactly — interp(0.05; h=0.73) lifted from 0.6687 to 1.0, +0.331 — but on the 412-event production posterior the MAP only moved by one discrete grid step (0.005), from 0.7650 to 0.7550. Both `without_bh_mass` and `with_bh_mass` channels remain just outside the acceptance window:

- without_bh_mass MAP=0.7550 (Δ-truth = +0.025, ~2.5σ_boot from h=0.73)
- with_bh_mass MAP=0.7450 (Δ-truth = +0.015; unchanged from Plan 45-03 because `_build_grid_2d` was not patched)

Acceptance gate: 2/4 PASS, 2/4 FAIL → Branch B (UNDER-CORRECTION). User decision required for next escalation.

## Cluster Job Metadata

- **Submitted:** 2026-05-01 (Plan 45-04 commit 4a260e9 deployed)
- **EVAL_JOB:** 4190271 (38-element array, h ∈ {0.60, …, 0.86} step 0.005, all COMPLETED, ~4–5 min each)
- **COMBINE_JOB:** 4190272 (afterok, 3:17 wallclock, ExitCode 0:0)
- **NEW_RUN:** `/pfs/work9/workspace/scratch/st_ac147838-emri/run_phase45_20260501` (overwritten from Plan 45-03 — auto-namespace by date)
- **Total wallclock:** ~7 minutes end-to-end (matches Plan 45-03 baseline)
- **HEAD on cluster:** descends from `4a260e9 [PHYSICS] Phase 45 Plan 45-04: hybrid intermediate anchor at (0.05, 1.0)`

## Acceptance Criteria

| # | Criterion | Plan 45-03 | Plan 45-05 | Verdict |
|---|---|---|---|---|
| 5.1 | `map_h ∈ [0.72, 0.74]` | 0.7650 (FAIL) | **0.7550** (FAIL) | **FAIL** |
| 5.2 | bootstrap `interval_68` ∋ 0.73 | [0.7450, 0.7650] (FAIL) | [0.7450, 0.7650] (FAIL) | **FAIL** |
| 5.3 | 4-strategy MAP invariance | all = 0.7650 (PASS) | all = 0.7550 (PASS) | **PASS** |
| 5.4 | bootstrap σ_MAP ≤ 0.025 | 0.0117 (PASS) | 0.0099 (PASS) | **PASS** |

**Final acceptance: ESCALATE (Branch B — UNDER-CORRECTION, improved by one grid step).**

## Posterior Comparison: Plan 45-03 (single-anchor) vs Plan 45-05 (hybrid)

### without_bh_mass channel (1D interpolator — patched twice)

| h | Plan 45-03 P(h) | Plan 45-05 P(h) | Δ |
|---|---|---|---|
| 0.720 | 0.00000 | 0.00003 | +0.00003 |
| 0.725 | 0.00007 | 0.00123 | +0.00116 |
| 0.730 | 0.00033 | 0.00314 | +0.00280 |
| 0.735 | 0.00189 | 0.00955 | +0.00766 |
| 0.740 | 0.01061 | 0.03445 | +0.02384 |
| 0.745 | 0.09569 | **0.20465** | **+0.10896** |
| 0.750 | 0.13851 | 0.19389 | +0.05538 |
| 0.755 | 0.24136 | **0.24747** | +0.00611 (NEW MAP) |
| 0.760 | 0.12246 | 0.09018 | -0.03228 |
| 0.765 | **0.34710** | 0.19877 | -0.14833 (was MAP) |
| 0.770 | 0.02489 | 0.01083 | -0.01406 |
| 0.775 | 0.01603 | 0.00553 | -0.01050 |

**MAP transition:** discrete argmax moved from h=0.765 to h=0.755. Posterior peak height dropped 0.347 → 0.247 (broader). Mass at h=0.745 jumped +0.109; mass at h=0.730 was 0.00033, now 0.00314 (~10× more, but still tiny). Total leftward mass migration: ~0.20.

### with_bh_mass channel (2D interpolator — NOT patched by Plan 45-04)

- Plan 45-03: MAP=0.7450, peak=0.5562
- Plan 45-05: MAP=0.7450, peak=0.5315

The 2D channel's MAP is unchanged (as expected — `_build_grid_2d` was not touched). Slight peak narrowing reflects downstream coupling through `bayesian_statistics.py`'s likelihood combination, but the discrete MAP did not move.

## Bootstrap (1000 resamples, RNG seed 20260429)

```
n_events_used     = 412
n_events_excluded = 0
strategy          = physics-floor
mean(MAP)         = 0.7551   (Plan 45-03: 0.7603; Δ = -0.0052)
std(MAP)          = 0.0099   (Plan 45-03: 0.0117; tighter posterior)  → PASS gate 5.4
median(MAP)       = 0.7550
interval_68       = [0.7450, 0.7650]   (truth 0.73 OUTSIDE → systematic)
interval_95       = [0.7450, 0.7650]   (= 68% because h-grid is discrete)
verdict           = systematic
```

**σ_MAP improved**: post-hybrid 0.0099 vs pre-hybrid 0.0117. Sharper posterior, but still excludes truth.

## 4-Strategy Invariance

```
strategy           MAP
naive             0.7550
exclude           0.7550
per-event-floor   0.7550
physics-floor     0.7550
```

Phase 44 STAT-03 invariant preserved.

## Diagnostic: Why one grid step and not two?

The hybrid lift at d_L=0.05 was +0.331 (the linear-interp probe at h=0.73). Window-average lift over [0, 0.10] ≈ +0.166. This translated to a continuous mean(bootstrap_MAP) shift of −0.005 — essentially the same magnitude as Plan 45-02's continuous shift (−0.005). The hybrid did NOT scale linearly with the local lift.

**Possible reasons:**

1. **L_comp gradient saturates.** Beyond a certain anchor lift, the L_comp denominator's response to interpolator changes plateaus. The first +0.04 of mean lift (Plan 45-02) and the next +0.16 of mean lift (Plan 45-04 hybrid) produced similar continuous MAP shifts because the integrand on [0, c_0] contributes <X% of the total D(h) integral.

2. **Window proximity skew.** The 26/60 events with windows crossing [0, c_0] integrate weighted by their detection-likelihood at the window centroid, not the d_L=0.05 anchor. Plan 45-04's biggest lifts are at d_L=0.025-0.075, which may not align with the window centroids of the production events.

3. **with_bh_mass channel is the bigger lever.** The without_bh_mass channel has MAP=0.755 (3.4% from truth); the with_bh_mass channel has MAP=0.745 (2.0% from truth). Patching `_build_grid_2d` (Phase 46 scope under current plan) would likely move the with_bh_mass channel into [0.72, 0.74] — possibly making it the primary paper result.

## Escalation Options

Per Plan 45-04 SUMMARY's pre-flagged escalations and the empirical evidence above:

| Option | Description | Predicted effect | Cost |
|---|---|---|---|
| **A** | Increase d_L=0 anchor: `_P_MAX_EMPIRICAL_ANCHOR = 0.8873` (point estimate, not LB) | Adds +0.094 at d_L=0; window-average lift ~+0.05 | Single-line change; small risk of overshoot |
| **B** | Move intermediate to d_L=0.025 | Lifts [0, 0.025] high; lowers [0.025, 0.05] (net ~zero on window integral) | Single-line change; rejected — empirically wrong direction |
| **C** | Add a third anchor at d_L=0.075 with value (e.g.) 0.85 | Extends high-lift segment further; +0.05 mean lift on [0.05, c_0] | Three-anchor layout; more code surface |
| **D** | Patch `_build_grid_2d` (with_bh_mass channel) with same hybrid logic | with_bh_mass MAP likely moves 0.7450 → 0.7350 or 0.7400 | Larger refactor; was Phase 46 scope |
| **E** | Accept current state: document MAP=0.755 (without_bh_mass) and MAP=0.745 (with_bh_mass) as the final paper result with quantified bias | None on physics; paper deadline pragmatic | Zero technical cost; physics integrity question |

**Recommended:** Option A (increase d_L=0 anchor) as the next minimum-touch escalation. If still under-corrects, Option D becomes the paper-blocker fix.

**Rejected:** Option B (counter-productive direction); Option C (more code surface for similar effect to A).

## Plan 45-02 / 45-04 Invariants Preserved

- `_P_MAX_EMPIRICAL_ANCHOR = 0.7931` unchanged.
- `_D_INTERMEDIATE_ANCHOR_GPC = 0.05` unchanged.
- `_P_INTERMEDIATE_EMPIRICAL = 1.0` unchanged.
- 4-strategy MAP invariance preserved.
- `_build_grid_2d` untouched (with_bh_mass channel unchanged at MAP=0.7450).
- 6 zero_fill call sites in `bayesian_statistics.py` untouched.

## Files Created

- `results/phase45_v2_posteriors/` (38 per-h JSONs + combined + diagnostics, ~468KB)
- `results/phase45_v2_posteriors_with_bh_mass/` (~750MB; only combined + tables committed)
- `scripts/bias_investigation/outputs/phase45/post_hybrid_bootstrap_summary.json`
- `.gpd/phases/45-p-det-first-bin-asymptote-fix/45-05-SUMMARY.md` (this file)

## Cluster Operations Log

```
SSH bwunicluster                                                    OK
git stash + git pull (cluster)                                      OK (FF eae4388 → 4a260e9)
Cluster pytest pre-flight                                           OK (39 passed)
bash cluster/submit_phase45_eval.sh                                 OK (EVAL=4190271, COMBINE=4190272)
sacct (after ~7 min)                                                ALL COMPLETED, ExitCode 0:0
rsync posteriors/ + posteriors_with_bh_mass/ + logs/                OK (~750 MB total)
inline bootstrap (412 events, B=1000, RNG=20260429)                 OK
4-strategy MAP comparison                                            ALL = 0.7550 (PASS invariance)
```

## Self-Check: PASSED (acceptance gate is what's escalated; the plan executed correctly)

All deliverables exist:
- `results/phase45_v2_posteriors/combined_posterior.json` (rsynced, MAP=0.7550, 412 events, git_commit field None as in Plan 45-03)
- `results/phase45_v2_posteriors_with_bh_mass/combined_posterior.json` (MAP=0.7450)
- `scripts/bias_investigation/outputs/phase45/post_hybrid_bootstrap_summary.json` (B=1000, σ=0.0099, interval=[0.7450, 0.7650])
- `.gpd/phases/45-p-det-first-bin-asymptote-fix/45-05-SUMMARY.md` (this file)

STATE.md will NOT be updated to "Phase 45 complete" — Phase 45 stays at Plan 45-05 with status `Escalate` until either Plan 45-06 closes the bias OR the user accepts current state for the paper.

---

_Phase: 45-p-det-first-bin-asymptote-fix_
_Plan: 05_
_Completed: 2026-05-01_
_Status: ESCALATE — Branch B (UNDER-CORRECTION, improved); user decision needed for Plan 45-06 path_
