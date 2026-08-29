# Commit Plan 2 — wave-2 PREP Records node (2026-08-29)

`launched under rows #222/#223 — charter node Records (wave-2 PREP)`

No git command was run by this node (mechanical Records dispatch; standing rule "NO git
commit/add/reset/stash"). This is a proposal for the orchestrator's own commit pass.

## `git status --short`, filtered

Filtered to exclude (per dispatch instruction): `results/**/ca_rhs_work`, `simulations/`,
`docs/CLAUDE_SCIENCE_*.md`, `scripts/bridge_closure/outputs/f4_specz_decomposition.json`, and
pre-existing untracked results dirs (`p3_2d_work/`, `p3_2d_work_m2z/`, `p3_b0_work/`, `p3_work/`,
`wbhzero_work/`, `prod2d_closure_20260818/`, `run_20260620_seed500_phase50/`,
`run_20260628_seed600_figures/`, `run_20260804_frozeng/`, `run_20260804_postfix/`,
`run_20260805_d1/`, `run_20260805_n2sel1d/`, `run_20260817_fusion_counterfactual/`, and the
2026-08-27-dated standalone files `cb_null_pinning_output.json`,
`head_readout_extraction_20260827.md`, `ledger_row_collision_map_20260827.md`,
`rule1_sweep_complete_20260827.md` — none produced by this wave-2 PREP dispatch).

```
 M docs/gates/PHYSICS-GATE-LEDGER.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_HIER_RECORD.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/B3_1_POP_RECORD.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_CMEM_A1_20260829.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/README.md
 M results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
 M results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
?? darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B3_2_POP_FLAG_RECORD.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B5_2_PULL_READ_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B7_2_FALSIFIER_I_RECORD.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/WAVE2_REGISTRATION_CHECK_20260829.md
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/b5_pull_read.json
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/b5_pull_read.py
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work/
?? results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_score.py
```

24 paths total (9 modified, 15 new — 1 new test file, 13 new docs/data files, 2 new run-artifact
directories).

## Quality gate (REPORT ONLY — no fixes applied by this node)

- `uv run ruff check darksiren_emri/` — **All checks passed!**
- `uv run ruff format --check darksiren_emri/` — **70 files already formatted**
- `uv run mypy darksiren_emri/` — **Success: no issues found in 70 source files**
- `uv run pytest -m "not gpu and not slow" -q -p no:cacheprovider` — **1875 passed, 15 skipped,
  27 deselected**, 146.85 s, coverage 73.17 % (gate 25 %)

All four gates green at HEAD `dd63fe0c` + this pass's uncommitted diff. No regressions.

## Proposed commit split

### (a) `[PHYSICS]` B3.2 flag (code + tests + gate rows)

**N/A this pass — nothing to commit under this heading.** B3.2's dispatch to *implement* the
`completion_population_prior` flag was **declined** (`B3_2_POP_FLAG_RECORD.md`): the cited
presentation (`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md`) is a STOP, not a green light —
its own §13 item 2 says "No code under this presentation." `git status`/`git diff` confirm no
file under `darksiren_emri/` is touched by this flag; no `completion_population_prior` symbol
exists anywhere in the tree. The only artifact from B3.2 is the `dispatch-declined` row appended
to `docs/gates/PHYSICS-GATE-LEDGER.md` (2 lines, append-only) — that lands in split (b) below as
a docs/records change, not a physics change, since it records a non-event (no code written).

If the orchestrator instead wants a `[PHYSICS]` commit for this wave, the only candidate is the
B7.2-pre falsifier (i) test file (`test_survival_2d_homogeneity_falsifier.py`) — but it adds a
regression test against the *existing* `catalogue_numerator_survival_2d` flag (already committed
in wave 1, `0b308828`), not a new physics change, so it is proposed under (b) as a test addition
rather than a `[PHYSICS]` commit in its own right.

### (b) docs/records (proposed single commit)

All 24 paths above, split into two commits if the orchestrator prefers granularity:

**(b1) test addition** (1 file):
- `darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py` — 4 new
  CPU-only unit tests (B7.2-pre falsifier (i)); 52/52 passing including 48 pre-existing;
  ruff/mypy clean.

**(b2) wave-2 PREP records + ledger** (23 files):
- `docs/gates/PHYSICS-GATE-LEDGER.md` (M) — B3.2 `dispatch-declined` append-only row
- `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md` (M) —
  rows #234–#238 appended (B3.2, B5.2-pre, B7.2-pre, B8.2, wave-2 registration check)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/COMPUTE_LEDGER.md` (M) — wave-2
  cost refinements section appended (C2 struck, C1 corrected, revised cluster total, new B8.2
  local row, P0/P1′/P2/P6 pre-wave refinements)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/README.md` (M) — wave-2 PREP
  file index + ledger-row cross-reference appended
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/B1_1_HIER_RECORD.md`,
  `B3_1_POP_RECORD.md`, `PREREGISTRATION_CMEM_A1_20260829.md`, `PROPOSAL_2D_TWIN_ADOPTION_20260829.md`
  (M — pre-existing wave-1 records with wave-2 append-only notes/corrections from other wave-2
  PREP workers, e.g. F-A findings, falsifier (i) result table §13; not authored by this node)
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py` (M — driver
  build progression, not authored by this node)
- New node records: `B3_2_POP_FLAG_RECORD.md`, `B5_2_PULL_READ_20260829.md`,
  `B7_2_FALSIFIER_I_RECORD.md`, `B8_2_HARNESS_DESIGN_20260829.md`
- New registrations/presentations: `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md`,
  `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md`
- New chair document: `WAVE2_REGISTRATION_CHECK_20260829.md`
- New scripts/data: `b5_pull_read.py`, `b5_pull_read.json`, `kwq1_score.py`
- New run-artifact directories: `hier_s0_registered_run/`, `hier_s0_work/` (S0-A/S0-C logs +
  per-node `event_likelihoods.csv` outputs)

Suggested commit message (docs only, no `[PHYSICS]` tag since no formula/constant changed):
`docs: wave-2 PREP records — B3.2 STOP declined, B5.2-pre pull read, B7.2-pre falsifier (i),
B8.2 design note, wave-2 registration check (rows #234–#238)`.

## Notes

- No physics-trigger file (`bayesian_statistics.py`, `handler.py`, `constants.py`,
  `LISA_configuration.py`, `parameter_estimation.py`, `cosmological_model.py`,
  `simulation_detection_probability.py`, `physical_relations.py`) appears in the filtered status
  above — none was edited by any wave-2 PREP node this pass.
- `hier_s0_registered_run/` and `hier_s0_work/` contain S0-A/S0-C run logs and per-node CSVs; if
  these are large, consider `.gitignore`-ing the raw run dirs and committing only the driver
  script + summary JSON, per the repo's existing convention for `*_work/` directories (most of
  which are excluded above as pre-existing untracked). Left in the proposed commit as-is;
  orchestrator's call.
