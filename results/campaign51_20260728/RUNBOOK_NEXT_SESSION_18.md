# Runbook — next session (written 2026-08-17, supersedes RUNBOOK_NEXT_SESSION_17)

**Read first:** `results/run_20260817_fusion_counterfactual/CAMPAIGN_REPORT_20260817.md`
(the comprehension-first readout; §10 is the pending decision table), then the prereg's
VERDICT section in the same directory.

## 0. State (end of the 2026-08-17 session, part 2)

- **The item-4 fusion counterfactual is EXECUTED and READ OUT** (164/164 tasks, ~170 of
  270 CPU-h, all NULLs PASS, off twin bit-identical to the run of record on 1D).
  Results: [P2] 1D tilt +24.6/+22.7 chord (≡ N-2 of record to 3 decimals), [P1] 2D tilt
  +1.2/−3.3 (near-inert, in the prior bracket — MAJOR-1 regime call confirmed), zero MAP
  motion anywhere, M-4 skew = median +0.02–0.03 (max +0.20) catalogue-share gain confined
  to the ~10% catalogue-bearing events.
- **Two [RULE]s are IN FRONT OF THE AUTHOR** (report §10): (1) M-4 materiality → keeps or
  reopens the [P3] catalogue-leg deferral (row #117 item 2's condition); (2) campaign-
  re-run scope given zero M-3 motion (options: none / targeted / full). **No ledger row
  is written until the author rules** (rows quote verbatim rulings).
- Operational deviation of record: sidecar `parent_csv` stale-path repair (hash-verified,
  path-only; cluster-skill gotcha 10 added).
- Fusion itself: landed at `2b10b8b8` (see runbook 17 §0 for that chain).

## 1. Next tasks

1. Author rules on report §10 items 1–2 → ledger row #119 quoting the ruling; then the
   flagged claim status for the fusion-magnitude numbers.
2. If re-run scope granted: its own prereg (A8-v2) before anything runs.
3. Carried: #66/#67 production calibration harness (pp_coverage mass channel, TO-BUILD —
   the likeliest disappointment path, flagged in report §9); book ch14 (the whole
   L4→L6→fusion→counterfactual arc); Gray-convention paper task (row #110) now holding
   M-4 as input; veto-flagged rows #109–#111; author WIP (3 book files) uncommitted.

## 2. Standing constraints

Unchanged from runbook 17 §2 (append-only, A8-v2, top-tier cap, branch calls presented,
A3). Workspace expires 2026-09-23 (0 extensions); counterfactual outputs are retrieved
locally (13G untracked; readout+report+prereg committed).

## 3. Resume recipe

1. `git log --oneline -3` — expect the readout commit at HEAD or a descendant.
2. Read the report §10. 3. Author rules; write ledger row #119; proceed per §1.
