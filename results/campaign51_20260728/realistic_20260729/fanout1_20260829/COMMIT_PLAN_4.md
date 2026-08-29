# Commit Plan 4 — synthesis docket 2 filing (2026-08-29/30)

`launched under rows #222/#223 — charter node: wave-2 synthesis chair (docket 2)`

No `git commit`/`add`/`reset`/`stash` was run by this node (standing rule; append-only, foreground
only, no `ssh`, no source edits). This is a proposal for the orchestrator's own commit pass.

## 1. Filtered `git status --short`

Filtered to the paths this pass touched or created (`results/campaign51_20260728/realistic_20260729/fanout1_20260829/`
and `results/campaign51_20260728/realistic_20260729/gate_b_20260730/`). Excludes everything already
landed at HEAD `60f9996e` (wave-2 GAP-closure + wave-2 registration end + the B7.3 adoption commit
`d4765539` are already in the tree; `COMMIT_PLAN_3.md`'s split is not re-listed here) and excludes
the unrelated pre-existing untracked material noted in `COMMIT_PLAN_3.md` §1
(`docs/CLAUDE_SCIENCE_*.md`, `scripts/bridge_closure/outputs/f4_specz_decomposition.json`,
`simulations/`, `p3_*_work/`, `wbhzero_work/`, etc.) plus `ca_rhs_work/` under `realistic_20260729/`.

```
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/README.md
 M results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_S0A_DEFECT_FORENSIC_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/DOCKET2_PACKAGE_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_END_VERIFIER_PASS_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/SYNTHESIS_DOCKET_2_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMMIT_PLAN_4.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/b1_1_forensic_work/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_full_output.json
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_seed900101/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_seed900102/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_seed900103/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_seed900104/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0c_seed900101/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_registered_run/s0a_full_output.json
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_registered_run/s0a_seed900101/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_registered_run/s0a_seed900102/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_registered_run/s0a_seed900103/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_registered_run/s0a_seed900104/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_parity_run/
```

**Counts:** 18 status lines — 3 modified, 4 new docket/record files (+ this plan), 11 new
run-artifact directories/files.

## 2. `du` of new run-artifact directories (2026-08-30)

```
1.6M   b1_1_forensic_work/        (forensic scripts + f1-f21 JSON twin/decomposition outputs)
155M   hier_s0_registered_run/s0a_seed900101/
108M   hier_s0_registered_run/s0a_seed900102/
109M   hier_s0_registered_run/s0a_seed900103/
162M   hier_s0_registered_run/s0a_seed900104/
854M   hier_s0_registered_run/s0c_seed900101/
24M    hier_s0_work/
401M   kwq1_registered_run/
28M    kwq1_parity_run/
```

Total new run-artifact footprint this pass ≈ **1.84 GB**, almost entirely `simulations/`
subtrees (per-node `posteriors_with_bh_mass/h_0_*.json` waveform/Fisher intermediates — 48
files found over 4MB, several tens of MB each, e.g.
`hier_s0_registered_run/s0a_seed900104/node_b_plus_sites2.2_nosmear/simulations/posteriors_with_bh_mass/h_0_73.json`).
Consistent with `COMMIT_PLAN_3.md` §2's ruling on the wave-2 run dirs, this pass proposes the
**same exclusion policy**: keep only `*.log` files and per-node `diagnostics/event_likelihoods.csv`
files (all individually well under the 4MB cap) from each of `b1_1_forensic_work/`,
`hier_s0_registered_run/`, `hier_s0_work/`, `kwq1_registered_run/`, `kwq1_parity_run/`; exclude
every `simulations/` intermediate subtree beneath them. (`b1_1_forensic_work/` at 1.6M has no
`simulations/` subtree — its scripts and JSON outputs are small and can be added whole.)

## 3. Quality gate

**Not re-run this pass.** No source (`darksiren_emri/`) or test (`darksiren_emri_test/`) files
were touched — this pass is append-only ledger entries, two new markdown records, a forensic
report + its scripts/JSON, and read-only run artifacts (Stage-0 P0/S0-C reruns, KW-Q1 registered
run + parity check). Gate history on file: `COMMIT_PLAN_3.md` §3 (HEAD `dd63fe0c` + diff: ruff
clean, mypy clean, 1889 passed / 15 skipped / 27 deselected, coverage 73.21%); the B7.3 adoption
pass separately reported 1896 passed / 15 skipped / 27 deselected
(`B7_3_ADOPTION_VERIFIER_REPORT.md`). Nothing in this pass's diff can regress either result.

## 4. Proposed commit split

Single docs commit — everything filed by the docket-2 chair is documentation/records/read-only
run diagnostics, no code:

### `docs: fan-out 1 wave-2 registration-end synthesis (docket 2) — chair re-derivations, tree state, F4 reconciliation, 7 RULEs to author; B1.1-F forensic; ledger row #252`

```
results/campaign51_20260728/realistic_20260729/fanout1_20260829/SYNTHESIS_DOCKET_2_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/DOCKET2_PACKAGE_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_END_VERIFIER_PASS_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_S0A_DEFECT_FORENSIC_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/README.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMMIT_PLAN_4.md
results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/b1_1_forensic_work/
```

Plus, from `hier_s0_registered_run/`, `hier_s0_work/`, `kwq1_registered_run/`, `kwq1_parity_run/`:
**only** `*.log` files and per-node `diagnostics/event_likelihoods.csv` files, per the exclusion
policy in §2. As `COMMIT_PLAN_3.md` §4(c) notes for the same pattern: a `git add` for this slice
should use explicit filtered paths (`find ... -name '*.log' -o -name event_likelihoods.csv`) or a
`.gitignore`-style include list, never a bare `git add <dir>/`, which would sweep in the ≈1.84 GB
of `simulations/` intermediates this plan excludes.

No `[PHYSICS]` tag: this pass touched no physics-trigger file and changed no computed value —
it is a synthesis/filing pass over already-produced numbers plus one forensic report that itself
made no source edits (`B1_1_S0A_DEFECT_FORENSIC_20260829.md` panel note: `bayesian_statistics.py`
/`correspondence_1d.py`/`hier_s0_driver.py` unedited this session).

## 5. Path count summary

- docs (files): 8
- docs (whole small dir): 1 (`b1_1_forensic_work/`)
- filtered run-artifact slices (logs + small CSVs only): 5 directories
  (`hier_s0_registered_run/`, `hier_s0_work/`, `kwq1_registered_run/`, `kwq1_parity_run/`, plus
  their subdirectories), excluding ≈1.84 GB of `simulations/` intermediates

**Return:** this plan at
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMMIT_PLAN_4.md`.
