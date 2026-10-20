# Commit Plan 3 — wave-2 GAP-CLOSURE Records node (2026-08-29)

`launched under rows #222/#223 — charter node Records (wave-2 GAP-CLOSURE)`

No `git commit`/`add`/`reset`/`stash` was run by this node (standing rule). This is a proposal
for the orchestrator's own commit pass. Report-only quality gate (§3) is likewise not a fix pass.

## 1. Filtered `git status --short`

Excludes (per dispatch instruction): `results/**/ca_rhs_work`, `simulations/`,
`docs/CLAUDE_SCIENCE_*.md`, `scripts/bridge_closure/outputs/f4_specz_decomposition.json`, and the
pre-existing untracked dirs `p3_2d_work/`, `p3_2d_work_m2z/`, `p3_b0_work/`, `p3_work/`,
`wbhzero_work/`, `cb_null_pinning_output.json`, `head_readout_extraction_20260827.md`,
`ledger_row_collision_map_20260827.md`, `rule1_sweep_complete_20260827.md`, `results/run_*`,
`results/prod2d_closure_*` (none produced by this pass).

```
 M darksiren_emri/arguments.py
 M darksiren_emri/main.py
 M darksiren_emri_test/test_arguments.py
 M docs/derivations/population_mismatch_dark_score.md
 M docs/gates/G7_systematics_budget.md
 M docs/gates/PHYSICS-GATE-LEDGER.md
 M results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_HIER_RECORD.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/B3_1_POP_RECORD.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_CMEM_A1_20260829.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/README.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/SYNTHESIS_DOCKET_1_20260829.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
 M results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
?? cluster/WAVE2_SUBMISSION_NOTE_20260829.md
?? cluster/submit_wave2.sh
?? cluster/wave2_c0_baseline.sbatch
?? cluster/wave2_c1_s0b_TEMPLATE.sbatch
?? cluster/wave2_c3_win_k3.sbatch
?? cluster/wave2_c4_twin_mz_sel.sbatch
?? darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py
?? darksiren_emri_test/test_theta_cli_forwarding.py
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_2_DRIVER_EXTENSION_NOTE.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B3_2_POP_FLAG_RECORD.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B4_2_KWQ1_RUN_FORM_NOTE.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B5_2_PULL_READ_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B7_2_FALSIFIER_I_RECORD.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMMIT_PLAN_2.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/P6_THETA_CLI_PLUMBING_RECORD.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_C0_BASELINE_GATE_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/WAVE2_REGISTRATION_CHECK_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/b5_pull_read.json
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/b5_pull_read.py
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_score.py
```

**Counts:** 41 status lines total — 16 modified, 23 new files, 2 new run-artifact directories.

## 2. `du` of new run-artifact directories

```
70M  hier_s0_registered_run/   (141 files total under both dirs combined)
24M  hier_s0_work/
```

Logs + `event_likelihoods.csv` diagnostics only (the material this pass is instructed to keep):
`hier_s0_registered_run/` logs+diagnostics = **396K**; `hier_s0_work/` logs+diagnostics = **52K**
(every `event_likelihoods.csv` found is 4K-32K, well under the 4MB cap — none excluded on size).
The remaining ~93.5MB across both directories is `simulations/` subtrees (waveform/Fisher
intermediates per run node) — **excluded** from the commit per the dispatch instruction
("EXCLUDING `hier_s0_registered_run/` and `hier_s0_work/` run outputs except logs and per-node
`diagnostics/event_likelihoods.csv` under 4 MB").

## 3. Quality gate (REPORT ONLY — no fixes applied by this node)

Run foreground, 2026-08-29, at HEAD `dd63fe0c` + this pass's uncommitted diff:

- `uv run ruff check darksiren_emri/` → **All checks passed!**
- `uv run ruff format --check darksiren_emri/` → **70 files already formatted**
- `uv run mypy darksiren_emri/` → **Success: no issues found in 70 source files**
- `uv run pytest -m "not gpu and not slow" -q -p no:cacheprovider` → **1889 passed, 15 skipped,
  27 deselected**, 0 failed, 169.56s, coverage 73.21% (gate 25%)

All four gates green. No regressions from the P6 plumbing change or any doc-only work.

## 4. Proposed commit split

### (a) `cli: theta-hook flags on the production CLI (identity defaults) — charter B1.2 P6`

```
darksiren_emri/arguments.py
darksiren_emri/main.py
darksiren_emri_test/test_arguments.py
darksiren_emri_test/test_theta_cli_forwarding.py
```

Adds `--theta_b`/`--theta_s`/`--theta_sites` to the CLI, forwarding unmodified into
`BayesianStatistics.evaluate()`'s pre-existing (row #216, `d40fe5c8`) theta-hook parameters.
Byte-identical defaults (`theta_b=0.0`, `theta_s=1.0`, `theta_sites="all"`); no physics-trigger
file touched. Not a `[PHYSICS]` commit — it is a CLI-surface addition to already-landed, already-
gated physics code (GATE T-ID literal-skip identity enforced inside `evaluate()` itself, unchanged
here). 13 new/modified tests, all green (§3).

### (b) `test: 2D-twin S_4D-homogeneity falsifier (i)`

```
darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py
```

Regression test against the *existing* `catalogue_numerator_survival_2d` flag (landed wave 1,
`0b308828`) — a new falsifier, not a new physics change. 4 test functions, 52/52 passed
including pre-existing suite members (row #236).

### (c) `docs: fan-out 1 wave-2 GAP-closure — B3 PREMISE-REFUTED, registrations C0/C3/C4/PA-HIER-31, pull read, B8.2 design, chair check, archive scheduling`

Everything else:

```
docs/derivations/population_mismatch_dark_score.md
docs/gates/G7_systematics_budget.md
docs/gates/PHYSICS-GATE-LEDGER.md
results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_HIER_RECORD.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_2_DRIVER_EXTENSION_NOTE.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B3_1_POP_RECORD.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B3_2_POP_FLAG_RECORD.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B4_2_KWQ1_RUN_FORM_NOTE.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B5_2_PULL_READ_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B7_2_FALSIFIER_I_RECORD.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMMIT_PLAN_2.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMMIT_PLAN_3.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/P6_THETA_CLI_PLUMBING_RECORD.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_CMEM_A1_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/README.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_C0_BASELINE_GATE_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/SYNTHESIS_DOCKET_1_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/WAVE2_REGISTRATION_CHECK_20260829.md
results/campaign51_20260728/realistic_20260729/fanout1_20260829/b5_pull_read.json
results/campaign51_20260728/realistic_20260729/fanout1_20260829/b5_pull_read.py
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_score.py
results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
cluster/WAVE2_SUBMISSION_NOTE_20260829.md
cluster/submit_wave2.sh
cluster/wave2_c0_baseline.sbatch
cluster/wave2_c1_s0b_TEMPLATE.sbatch
cluster/wave2_c3_win_k3.sbatch
cluster/wave2_c4_twin_mz_sel.sbatch
```

Plus, from `hier_s0_registered_run/` and `hier_s0_work/`: **only** `*.log` files and per-node
`diagnostics/event_likelihoods.csv` files (396K + 52K, all individually well under 4MB) — every
`simulations/` intermediate subtree beneath those two directories is excluded per the dispatch
instruction. A `git add` for this slice should use explicit `-name`-filtered paths or a
`.gitignore`-style include list rather than `git add hier_s0_registered_run/ hier_s0_work/`,
which would sweep in the excluded ~93.5MB of simulation intermediates.

Note: `B3_2_POP_FLAG_RECORD.md` and the `docs/gates/PHYSICS-GATE-LEDGER.md` `dispatch-declined`
row are records of a *non-event* (B3.2 implementation declined, no code) — they belong in this
docs split, not in (a), consistent with `COMMIT_PLAN_2.md`'s reasoning for the same file.

## 5. Path count summary

- (a) cli: 4 files
- (b) test: 1 file
- (c) docs: 30 tracked/untracked files + 2 filtered run-artifact subtrees (logs + small CSVs only)

**Return:** this plan at
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMMIT_PLAN_3.md`.
