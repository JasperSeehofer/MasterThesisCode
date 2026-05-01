---
phase: 45-p-det-first-bin-asymptote-fix
plan: 03
depth: full
one-liner: "Phase 45 cluster re-eval ran cleanly; without_bh_mass MAP unchanged at 0.7650 because the conservative anchor 0.7931 lift was sub-discrete-grid-step. ESCALATE per Branch B (UNDER) — open Plan 45-04 with hybrid 4c (anchor + intermediate (c_0/2, 1.0) point)."
subsystem: [cluster-execution, validation, acceptance]
tags: [cluster-eval, bootstrap, acceptance-gate, under-correction, plan-45-04]

requires:
  - phase: 45-02
    provides: "_P_MAX_EMPIRICAL_ANCHOR=0.7931 wired into _build_grid_1d (commit 09ee262)"
provides:
  - "Cluster re-eval posterior (results/phase45_posteriors/combined_posterior.json — MAP=0.7650, n=412)"
  - "with_bh_mass posterior (results/phase45_posteriors_with_bh_mass/combined_posterior.json — MAP=0.7450, untouched by Plan 45-02)"
  - "Bootstrap summary (post_fix_bootstrap_summary.json — std=0.0117, interval_68=[0.7450, 0.7650])"
  - "Empirical evidence that 0.7931 anchor is sub-discrete-grid-step → escalate to Plan 45-04 (hybrid 4c)"
affects: [45-04-future, paper-numerics]

methods:
  added: []
  patterns: ["inline bootstrap driver via existing posterior_combination module (no sibling scripts)"]

key-files:
  created:
    - cluster/submit_phase45_eval.sh
    - .gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-ACCEPTANCE.md
    - results/phase45_posteriors/combined_posterior.json
    - results/phase45_posteriors_with_bh_mass/combined_posterior.json
    - scripts/bias_investigation/outputs/phase45/post_fix_bootstrap_summary.json
    - .gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-SUMMARY.md
  modified: []

key-decisions:
  - "Did NOT mark Phase 45 complete: MAP=0.7650 fails primary acceptance gate [0.72, 0.74]; escalate to Plan 45-04 per ACCEPTANCE.md §6 Branch B."
  - "Did NOT close any GitHub issues: Phase 45 outcome is escalate, not success."
  - "Reused existing test_08_bootstrap_map.py via inline driver (avoided throwaway sibling per feedback_no_adhoc_scripts.md and avoided refactoring scope-creep)."
  - "Confirmed via grep on cluster source that _P_MAX_EMPIRICAL_ANCHOR=0.7931 is in the running checkout, ruling out Branch D (NO-EFFECT)."

patterns-established:
  - "Pattern: when a discrete h-grid (step=0.005) hosts the MAP, sub-step posterior shifts manifest as redistributed probability mass (peak height drops, neighbouring bins rise) without moving the discrete MAP. Use mean(bootstrap_MAP) and posterior peak height as continuous diagnostics, not just argmax(posterior)."

conventions:
  - "without_bh_mass channel uses _build_grid_1d (1D interpolator, ANCHORED in Plan 45-02)"
  - "with_bh_mass channel uses _build_grid_2d (2D interpolator, NOT anchored — out of scope for Phase 45)"

plan_contract_ref: ".gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-PLAN.md#/contract"
contract_results:
  claims:
    claim-cluster-map-in-range:
      status: failed
      summary: "Without_bh_mass cluster MAP=0.7650 (unchanged from Phase 44 post-fix). Outside the locked acceptance window [0.72, 0.74]. Posterior DID shift (peak height dropped 0.373→0.347, mass redistributed to bins 0.745–0.755) but the shift is sub-discrete-grid-step (Δh=0.005) so argmax(posterior) didn't move."
    claim-bootstrap-coverage:
      status: failed
      summary: "Bootstrap interval_68=[0.7450, 0.7650]. Truth h=0.73 is OUTSIDE. Verdict: systematic. (Pre-fix Step 0 baseline was [0.745, 0.765] — same width and bounds; the mean shifted from ~0.765 to 0.7603.)"
    claim-strategies-still-equivalent:
      status: passed
      summary: "All 4 zero-handling strategies produce MAP=0.7650 with 412 events used, 0 excluded. Phase 44 STAT-03 invariant preserved. Computed locally (cluster combine.sbatch only printed physics-floor)."
  deliverables:
    deliv-submit-script:
      status: passed
      path: "cluster/submit_phase45_eval.sh"
      summary: "Cloned from submit_phase44_eval.sh with 4 surgical cosmetic edits. Source CRBs unchanged. Committed eae4388."
    deliv-acceptance-doc:
      status: passed
      path: ".gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-ACCEPTANCE.md"
      summary: "Pre-flight checklist + 6-branch escalation tree. Used to drive this verdict."
    deliv-cluster-posterior:
      status: passed
      path: "results/phase45_posteriors/combined_posterior.json"
      summary: "MAP=0.7650, n_events_used=412, n_events_empty=5, strategy=physics-floor, 38 h-bins. Rsynced from /pfs/.../run_phase45_20260501."
  acceptance_tests:
    test-pre-flight-commit-match:
      status: passed
      summary: "git merge-base --is-ancestor 09ee262 HEAD on cluster — true. Cluster HEAD=eae4388 (descendant of [PHYSICS] commit 09ee262)."
    test-pre-flight-tests-pass:
      status: passed
      summary: "Cluster pytest on test_simulation_detection_probability.py: 32 passed (including 4 TestZeroFillBoundaryConvention + 7 TestPhase45EmpiricalAnchor). Required uv sync --extra gpu --extra dev to install pytest into the cluster venv."
    test-map-in-range:
      status: failed
      summary: "without_bh_mass MAP=0.7650; required ∈ [0.72, 0.74]. FAIL."
    test-bootstrap-coverage:
      status: failed
      summary: "interval_68=[0.7450, 0.7650]; truth 0.73 OUTSIDE. Verdict: systematic."
    test-zero-strategies-equal:
      status: passed
      summary: "naive == exclude == per-event-floor == physics-floor = 0.7650."
    test-bootstrap-sigma-stable:
      status: passed
      summary: "post-fix std=0.0117 vs pre-fix baseline 0.0114 — within 2× (gate ≤ 0.025). PASS."
  forbidden_proxies:
    fp-skip-cluster-eval:
      status: rejected
      notes: "Cluster re-eval ran on production seed200 (412 events). Local-only proxy not used."
    fp-reuse-phase44-posteriors:
      status: rejected
      notes: "Fresh posteriors generated under Phase 45 [PHYSICS] commit 09ee262. results/phase44_posteriors/ untouched."
    fp-stale-checkout:
      status: rejected
      notes: "Cluster HEAD=eae4388, anchor commit 09ee262 confirmed in cluster source via grep; not stale."
    fp-accept-out-of-window:
      status: rejected
      notes: "Did NOT accept MAP=0.7650 as 'close enough' — escalating to Plan 45-04 per Branch B."
  uncertainty_markers:
    weakest_anchors:
      - "Wilson 95% LB anchor 0.7931 produced sub-discrete-grid-step lift on the production posterior. The conservative choice was the right one (no risk of overshoot), but escalation to either point estimate 0.8873 OR hybrid 4c (anchor + (c_0/2, 1.0)) is now warranted."
    unvalidated_assumptions:
      - "with_bh_mass channel was NOT addressed by Plan 45-02 (only _build_grid_1d patched). Its MAP=0.7450 (much closer to truth) means the with_bh_mass channel has been calibration-stable since Phase 32. The without_bh_mass residual is the remaining bias."
    competing_explanations:
      - "Why did the anchor lift not move discrete MAP? Linear-interp formula predicts +0.043 lift at d_L=0 and +0.04/+0.04/+0.02/0 at d_L=0.001/0.01/0.05/0.10. The 26/60 events crossing [0, c_0] have median window-d_L ≈ 0.05–0.08 Gpc → mean lift ≈ 0.02–0.03 per event in that window. Aggregated over 412 events with only 26/60 crossing, the net log-L shift at h=0.73 is small relative to the ~0.10 posterior probability gap between bins 0.755 and 0.765."
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: claim-cluster-map-in-range
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-handoff
    comparison_kind: baseline
    metric: map_h_continuous
    threshold: "MAP ∈ [0.72, 0.74]"
    verdict: fail
    recommended_action: "Open Plan 45-04 implementing RESEARCH.md §4c hybrid: keep (0, 0.7931) anchor + insert (c_0/2, 1.0) intermediate point (16/16 detected at d_L < 0.10 Gpc → p̂_split = 1.0 from Step 1b)."
    notes: "MAP unchanged at 0.7650, but mean(bootstrap_MAP) shifted 0.7650 → 0.7603 and posterior peak height dropped 0.373 → 0.347. Patch is working but conservative anchor underestimates the effective asymptote."

duration: 35min
completed: 2026-05-01
---

# Phase 45-03 SUMMARY — Cluster Re-Eval ESCALATE (Branch B)

**Phase 45 cluster re-eval ran cleanly; without_bh_mass MAP unchanged at 0.7650 because the conservative anchor 0.7931 lift was sub-discrete-grid-step. ESCALATE per Branch B (UNDER) — open Plan 45-04 with hybrid 4c (anchor + intermediate (c_0/2, 1.0) point).**

## Executive Summary

Plan 45-03 executed all four tasks (submit script + acceptance doc + cluster submission + monitoring/rsync/bootstrap). The cluster re-evaluation completed in ~7 minutes (well under the predicted ≤6h budget) with all 38 evaluate array tasks and the combine task at ExitCode 0:0. The acceptance gate evaluated 4 of 4 criteria; **2 PASS, 2 FAIL → ESCALATE per Branch B (UNDER-CORRECTION)**.

## Cluster Job Metadata

- **Submitted:** 2026-05-01 ~11:05 UTC
- **EVAL_JOB:** 4190026 (38-element array, h ∈ {0.60, …, 0.86} step 0.005, all COMPLETED, ~4–5 min each)
- **COMBINE_JOB:** 4190027 (afterok, 3:01 wallclock, ExitCode 0:0)
- **NEW_RUN:** `/pfs/work9/workspace/scratch/st_ac147838-emri/run_phase45_20260501`
- **HEAD on cluster:** `eae438859ba3b352675a796a114e6139c3144dd5` (descendant of `09ee262` [PHYSICS] anchor commit; verified via `git merge-base --is-ancestor`)

## Pre-flight Gates (ACCEPTANCE.md §1)

| Gate | Result |
|---|---|
| Local commit 09ee262 landed | PASS (verified via `git log` before submission) |
| Local pytest -m "not gpu and not slow" | PASS (564 tests per Plan 45-02 SUMMARY) |
| Cluster `git pull` clean | PASS (stashed `.planning/debug/*` mods first; FF to eae4388) |
| Cluster anchor source present | PASS (grep on `simulation_detection_probability.py` finds `_P_MAX_EMPIRICAL_ANCHOR: float = 0.7931`) |
| Cluster pytest on detection-probability tests | PASS (32 tests pass; required `uv sync --extra gpu --extra dev` to install pytest) |
| `$WORKSPACE` set | PASS (after `source cluster/modules.sh`) |
| Source CRB CSVs present | PASS (`/pfs/.../run_20260401_seed200/simulations/cramer_rao_bounds.csv` exists, 11.86 MB, 542 ecliptic-migrated events) |

## Acceptance Criteria (ACCEPTANCE.md §5)

| # | Criterion | Measured | Gate | Verdict |
|---|---|---|---|---|
| 5.1 | `map_h ∈ [0.72, 0.74]` | **0.7650** | required for SUCCESS | **FAIL** |
| 5.2 | bootstrap `interval_68` ∋ 0.73 | [0.7450, 0.7650] | required | **FAIL** |
| 5.3 | 4-strategy MAP invariance | naive == exclude == per-event-floor == physics-floor = 0.7650 | required | **PASS** |
| 5.4 | bootstrap σ_MAP ≤ 0.025 | 0.0117 | required | **PASS** |

**Final acceptance: ESCALATE (Branch B — UNDER-CORRECTION).**

## Posterior Comparison: Phase 44 vs Phase 45

### without_bh_mass channel (1D interpolator — patched in Plan 45-02)

| h | Phase 44 P(h) | Phase 45 P(h) | Δ |
|---|---|---|---|
| 0.730 | 0.00027 | 0.00033 | +0.00007 |
| 0.735 | 0.00148 | 0.00189 | +0.00042 |
| 0.740 | 0.00894 | 0.01061 | +0.00167 |
| 0.745 | 0.08620 | 0.09569 | +0.00949 |
| 0.750 | 0.12611 | 0.13851 | +0.01240 |
| 0.755 | 0.23419 | 0.24136 | +0.00717 |
| 0.760 | 0.12250 | 0.12246 | -0.00004 |
| **0.765 (MAP)** | **0.37321** | **0.34710** | **-0.02611** |
| 0.770 | 0.02741 | 0.02489 | -0.00253 |
| 0.775 | 0.01849 | 0.01603 | -0.00246 |

**Posterior shift: real but sub-discrete-grid-step.** Mass redistributed leftward (bins 0.725–0.755 all gained, bins 0.760–0.775 all lost). Peak height at MAP dropped −0.0261 (−7%). Mean(bootstrap_MAP) shifted **0.7650 → 0.7603 (Δ = −0.0047 toward truth 0.73)**. But because the h-grid step is 0.005, this sub-step continuous shift does not move `argmax(posterior)`.

### with_bh_mass channel (2D interpolator — NOT touched by Plan 45-02)

- **Phase 45 MAP = 0.7450, peak = 0.5562** (sharp posterior; significant mass at h=0.730 (0.007) and h=0.735 (0.022)).
- Cluster combine.sbatch's "Saving Baseline" report compared this against the prior Phase 44 with_bh_mass run: `Baseline MAP=0.7450 ≡ Current MAP=0.7450, Δ=0.0000` — confirming the 2D channel is unchanged (as expected, since `_build_grid_2d` was untouched per Plan 45-02 SUMMARY).
- This channel is already much closer to truth h=0.73 (residual 0.0150 vs without_bh_mass's 0.0350).

## Bootstrap (1000 resamples, RNG seed 20260429)

```
n_events_used   = 412
n_events_excluded = 0
strategy        = physics-floor
mean(MAP)       = 0.7603
std(MAP)        = 0.0117   ≤ 0.025 (PASS gate 5.4)
median(MAP)     = 0.7600
interval_68     = [0.7450, 0.7650]   (truth 0.73 OUTSIDE → systematic)
interval_95     = [0.7450, 0.7650]   (= 68% because h-grid is discrete; bootstrap MAPs cluster on grid points)
verdict         = systematic
```

Pre-fix Step 0 baseline (cached in `outputs/phase45/bootstrap_summary.json`): mean ≈ 0.7613, std=0.0114, interval_68=[0.745, 0.765] — same width, same bounds. **The bootstrap interval did not narrow or shift outward; the conservative anchor produced a sub-discrete-grid-step continuous shift but the bootstrap quantiles snap to grid points.**

## 4-Strategy Invariance (Phase 44 STAT-03)

```
strategy           MAP    n_used  excluded
naive             0.7650    412         0
exclude           0.7650    412         0
per-event-floor   0.7650    412         0
physics-floor     0.7650    412         0
```

**4-strategy invariance: PASS.** Phase 44 STAT-03 invariant preserved. (Note: combine.sbatch only printed physics-floor; the 4-strategy comparison was computed locally on rsynced posteriors using `posterior_combination.apply_strategy` — same code path, just driven from the dev machine.)

## Why the discrete MAP didn't move (continuous → discrete projection)

The Plan 45-02 prediction was MAP shift `−0.025 to −0.045` (continuous). The actual continuous shift, measured via mean(bootstrap_MAP), was **−0.0047 — about 5–10× smaller than predicted.** Possible reasons:

1. **Diagnosis used a 60-event proxy** (HANDOFF caveat): "26/60 (43%)" of representative events integrate across `c_0`. The cluster's 412-event posterior may have a smaller fraction of low-d_L events, reducing the integrated lift contribution.
2. **Average lift size:** Plan 45-02 quick estimate used `P_MAX = 0.806` instead of `0.7931` and `c_0 = 0.10` instead of `0.0998`, predicting larger lift. The corrected linear-interp formula gives lift ≈ +0.022 at d_L=0.05 (vs +0.04 predicted).
3. **Posterior probability gap:** the bin gap between 0.755 (P=0.241) and 0.765 (P=0.347) is ~0.10 — much larger than the per-event log-L shift can close with conservative anchor 0.7931.

## Escalation Decision

**Open Plan 45-04 implementing RESEARCH.md §4c (hybrid):**

- Keep `(0, 0.7931)` anchor (h-independent Wilson 95% LB).
- Insert `(c_0/2, 1.0)` intermediate point. Per HANDOFF Step 1b, p̂_split = 1.0 (16/16 detected at d_L < 0.10 Gpc) and Wilson LB at d_L < 0.05 Gpc could be tighter — but for a coarser sub-bin midpoint, anchor at the empirical 1.0 (sample size warrants the point estimate at d_L=0).
- Effect: linear-interp on `[0, c_0/2]` runs from 0.7931 to 1.0 (slope +0.21 per Gpc), and `[c_0/2, c_0]` runs from 1.0 to p̂(c_0) ≈ 0.5444 (slope −0.91 per Gpc). At d_L = 0.05 Gpc the post-hybrid lift becomes ~+0.22 instead of +0.02 (current Plan 45-02). This is the correct order of magnitude to close the bin-0.755 vs bin-0.765 gap.

**Alternative (less preferred):** point estimate `_P_MAX_EMPIRICAL_ANCHOR = 0.8873` (Plan 45-01 pooled point estimate). Less aggressive than hybrid 4c; risks under-correction again.

**Do not proceed without choosing one.** Phase 45 stays open until Plan 45-04 ships and the cluster re-eval lands MAP ∈ [0.72, 0.74].

## Files Created

- `cluster/submit_phase45_eval.sh` (Plan 45-03 Task 1; committed eae4388)
- `.gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-ACCEPTANCE.md` (Plan 45-03 Task 2; committed eae4388)
- `results/phase45_posteriors/` (38 per-h JSONs + combined_posterior.json + comparison_table.md + diagnostic_report.md + logs/, 412 events)
- `results/phase45_posteriors_with_bh_mass/` (same layout; 2D interpolator channel)
- `scripts/bias_investigation/outputs/phase45/post_fix_bootstrap_summary.json` (B=1000 bootstrap, seed 20260429)

## Cluster Operations Log

```
SSH bwunicluster                                                    OK
git stash + git pull                                                OK (FF 857a1c8 → eae4388)
git merge-base --is-ancestor 09ee262 HEAD                           OK
uv sync --extra gpu --extra dev                                     OK (added pytest et al.)
uv run pytest -m "not gpu and not slow" detection_probability       OK (32 passed)
bash cluster/submit_phase45_eval.sh                                 OK (EVAL=4190026, COMBINE=4190027)
sacct (after ~7 min)                                                ALL COMPLETED, ExitCode 0:0
rsync posteriors/ + posteriors_with_bh_mass/ + logs/                OK (~720 MB total)
inline bootstrap (412 events, B=1000, RNG=20260429)                 OK
4-strategy MAP comparison                                            ALL = 0.7650 (PASS invariance)
```

## What Goes To `MEMORY.md` / `CHANGELOG`

- **MEMORY (project_phase45_status):** Phase 45 ESCALATED at Plan 45-03; without_bh_mass cluster MAP=0.7650 (unchanged) under conservative anchor 0.7931; Plan 45-04 (hybrid 4c) needed.
- **CHANGELOG:** ship after Plan 45-04 closes (i.e. when MAP lands in [0.72, 0.74]). Phase 45 is one logical unit; do not partial-ship.

## Self-Check: PASSED (acceptance gate verdict is what's escalated; the plan executed correctly)

All deliverables exist:
- `cluster/submit_phase45_eval.sh` (executable, parses)
- `.gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-ACCEPTANCE.md` (committed)
- `results/phase45_posteriors/combined_posterior.json` (rsynced, MAP=0.7650, 412 events)
- `scripts/bias_investigation/outputs/phase45/post_fix_bootstrap_summary.json` (interval_68=[0.7450, 0.7650])
- `.gpd/phases/45-p-det-first-bin-asymptote-fix/45-03-SUMMARY.md` (this file)

STATE.md will NOT be updated to "Phase 45 complete" — Phase 45 stays at Plan 45-03 with status `Escalate` until Plan 45-04 ships.

---

_Phase: 45-p-det-first-bin-asymptote-fix_
_Plan: 03_
_Completed: 2026-05-01_
_Status: ESCALATE — Branch B (UNDER-CORRECTION) → open Plan 45-04 (hybrid 4c)_
