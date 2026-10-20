# Commit plan 5 -- tree 2, T1.1 + T2.1 + T2.2 + PA-HIER-32 + A14 housekeeping

Launched under row #255 -- tree 2 node COMMIT_PLAN_5. Mechanical record only; no git operations
performed by this node (the orchestrator commits). Branch fix/p32d-classg-venue-repair, HEAD
ecd33336 at write time. Quality gate re-run this node (report only, all foreground, all under
600 s): ruff check darksiren_emri/ clean; ruff format --check darksiren_emri/ 70 files already
formatted; mypy darksiren_emri/ Success, no issues in 70 source files; pytest -m "not gpu and
not slow" split in two halves -- non-validation half 1514 passed / 15 skipped / 28 deselected,
validation half 401 passed / 2 deselected, combined 1915 passed / 15 skipped / 30 deselected
(matches the T2.2 baseline of record exactly).

## Filtered git status (source/docs/test files relevant to this node's commits; the campaign
results-run scratch directories -- ca_rhs_work, p3_work, p3_b0_work, wbhzero_work,
hier_s0_registered_run, kwq1_*_run, wave2_20260829/c4, run_2026*, results/prod2d_closure_20260818,
simulations, and similar per-seed/per-node artefact trees -- are pre-existing untracked
campaign-compute output, not part of this node's diff, and are omitted here; see the raw git
status output for the complete listing)

Modified (source):
- darksiren_emri/arguments.py
- darksiren_emri/bayesian_inference/bayesian_statistics.py
- darksiren_emri/main.py
- darksiren_emri/validation/correspondence_1d.py
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py

Modified (docs, append-only or corrective):
- docs/derivations/population_mismatch_dark_score.md
- docs/gates/G7_systematics_budget.md
- docs/gates/PHYSICS-GATE-LEDGER.md
- results/campaign51_20260728/RUNBOOK_NEXT_SESSION_37.md
- results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md
  (PA-HIER-32 block)
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_HIER_RECORD.md
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/B7_2_TWIN_CF_READOUT_RECORD.md
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md
- results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
  (rows #256-#263)

New (test):
- darksiren_emri_test/bayesian_inference/test_theta_phi_divisor.py
- darksiren_emri_test/integration/test_candidate_dump_instrumentation.py

New (docs, tree-2):
- results/campaign51_20260728/realistic_20260729/tree2_20260830/TREE2_CHARTER_20260830.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/PHYSICS_CHANGE_THETA_DIVISOR_20260830.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/T1_1_DIVISOR_IMPLEMENTATION_RECORD.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/T1_1_DIVISOR_VERIFIER_REPORT.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/T2_2_CANDIDATE_HOOK_RECORD.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/README.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/COMMIT_PLAN_5.md (this file)
- results/campaign51_20260728/realistic_20260729/tree2_20260830/t1_1_gate_work/ (scratch)
- results/campaign51_20260728/realistic_20260729/tree2_20260830/t1_1_verifier_work/ (scratch,
  includes the smoke_run/ live-catalogue cell)

New (docs, unrelated to this node -- flag for the orchestrator to route separately):
- docs/CLAUDE_SCIENCE_ABSTRACT.md
- docs/CLAUDE_SCIENCE_BRIEF.md

## Proposed commits

### (a) [PHYSICS] theta-consistent no-BH divisor + sky-cone-radius flag (code + tests + gate rows)

Files: darksiren_emri/arguments.py, darksiren_emri/bayesian_inference/bayesian_statistics.py,
darksiren_emri/main.py, darksiren_emri/validation/correspondence_1d.py,
darksiren_emri_test/bayesian_inference/test_theta_phi_divisor.py,
docs/gates/PHYSICS-GATE-LEDGER.md (the three 2026-08-30 "presented"/"implemented"/"verified"
rows), results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
(rows #259-#260).

Message shape: "[PHYSICS] theta-consistent no-BH global-selection divisor (site 2.3phi) +
sky-cone-radius flag" -- new theta_phi_divisor {off,on} flag (default off, byte-identical) and
sky_cone_k flag (default 1.5, byte-identical); registered form Sigma_phi_reg(theta;h) =
Sigma_phi_point(h) x rho(theta;h); panel-clean gate presentation, independently verified
(must_fix: none); 19 new tests, 85-test regression group clean, full suite 1915/15/27 (baseline
1896 + 19 net-new, zero regressions). Launched under row #255 -- tree 2 node T1.1.

### (b) instrumentation: per-candidate p_Di dump hook (code + tests + gate row)

Files: darksiren_emri/bayesian_inference/bayesian_statistics.py (the T2.2 hunks -- distinct from
(a)'s hunks; both land in the same file but at disjoint line ranges per the implementation
record's file lists), darksiren_emri/arguments.py, darksiren_emri/main.py,
darksiren_emri/validation/correspondence_1d.py,
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py,
darksiren_emri_test/integration/test_candidate_dump_instrumentation.py,
docs/gates/PHYSICS-GATE-LEDGER.md (the 2026-08-30 "instrumentation" row),
results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md (row #262).

Note for the orchestrator: (a) and (b) both touch bayesian_statistics.py, arguments.py, main.py
and correspondence_1d.py -- if a single working tree cannot be split cleanly hunk-by-hunk between
the two commits, (a) then (b) in sequence (a) first, since (b)'s per-h reset block sits inside
code (a) also touches) is the safe order; do not attempt a hunk-level split if git add -p proves
ambiguous, and land both together as a single commit with two paragraphs instead rather than risk
an inconsistent intermediate state.

Message shape: "instrumentation: per-candidate p_Di dump hook (A10, row #255)" -- opt-in
candidate_dump_dir kwarg, default None byte-identical; GATE BI + GATE SCHEMA new tests (3,
slow-marked); no physics-trigger formula/constant changed, A10 instrumentation-guard route used
in place of the full physics-change gate. Launched under row #255 -- tree 2 node T2.2.

### (c) docs: tree-2 charter, derivation, PA-HIER-32, housekeeping, ledger

Files: results/campaign51_20260728/realistic_20260729/tree2_20260830/ (entire directory:
TREE2_CHARTER_20260830.md, PHYSICS_CHANGE_THETA_DIVISOR_20260830.md,
T1_1_DIVISOR_IMPLEMENTATION_RECORD.md, T1_1_DIVISOR_VERIFIER_REPORT.md,
B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md, T2_2_CANDIDATE_HOOK_RECORD.md, README.md,
COMMIT_PLAN_5.md, t1_1_gate_work/, t1_1_verifier_work/),
results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md
(PA-HIER-32), docs/derivations/population_mismatch_dark_score.md, docs/gates/G7_systematics_
budget.md, results/campaign51_20260728/RUNBOOK_NEXT_SESSION_37.md,
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_HIER_RECORD.md,
B7_2_TWIN_CF_READOUT_RECORD.md, B8_2_HARNESS_DESIGN_20260829.md,
PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md, PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md
(the A14 housekeeping citation fixes, filed under row #258, already committed material being
carried forward if not yet landed -- orchestrator should check whether row #258's housekeeping
edits are already committed before including them here to avoid a duplicate/empty diff),
results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md (rows
#256-#263, or the residual subset not already covered by (a)/(b) above).

Message shape: "docs: tree-2 charter (T1.1 gate+build+verify, T2.1 derivation, T2.2 hook design),
PA-HIER-32 registration amendment, A14 housekeeping citation fixes; ledger rows #256-#263."
Launched under row #255 -- tree 2 nodes 0, T1.1, T2.1, A5, A14, PA-HIER-32.

## Not covered by this plan (flagged, not commit-ready)

- docs/CLAUDE_SCIENCE_ABSTRACT.md and docs/CLAUDE_SCIENCE_BRIEF.md are untracked and unrelated to
  tree 2's T1/T2 work; not included in any proposed commit above -- orchestrator to route.
- The large untracked campaign-compute directories (ca_rhs_work, p3_work, p3_b0_work,
  wbhzero_work, hier_s0_registered_run, kwq1_parity_run, kwq1_registered_run,
  wave2_20260829/c4, run_2026* under results/, results/prod2d_closure_20260818 additions,
  scripts/bridge_closure/outputs/f4_specz_decomposition.json, and the top-level simulations/
  directory) are pre-existing scratch/run output, not this node's diff; per the A13/A14
  housekeeping convention (row #255) these are routed to the existing archive/gitignore handling,
  not swept into any commit here.
- T1.2 (S0-A re-certification) is not yet built and has no diff to commit.
