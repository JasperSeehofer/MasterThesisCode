# b-s4-harness-repair -- EXECUTION RECORD

Node: `b-s4-harness-repair` (type: build), Research Graph 1, Branch A.

## Authorization

Quoted verbatim from `results/campaign51_20260728/realistic_20260729/graph1_20260901/RESEARCH_GRAPH_1_PROPOSAL_20260901.md`
decisions-table row 3 (row #290 ratified):

> `| 3 | b-s4-harness-repair (Branch A) | DO | Approved | the row #288 (a)-(c) repairs;
> r-b82-s4 registration authoring; m-s3 launches only after d-s4-review and a green design gate |
> everything in row 2's NOT-covered cell, plus the d-calibration ruling |`

And the node's own row (§1.1):

> `| b-s4-harness-repair | build | the three S4 defects of row #288 (a)-(c): seed-population
> separation, missing cell-T aggregation, wall-limited stop rule | authorized-by
> d-batch1-charter AND d-s3-rerun (Yes disposition); feeds from row #288 pilot record |
> g-byte-id on untouched code paths; g-population lint on the repaired aggregator | 0
> mismatches at N >= 1e5 pairs (infra 2.5); 0 mixed rows | cheap | sonnet / medium |`

Scope note honored: this build implements the harness repair only. It does **not** define a
stop rule (that is `r-b82-s4` registration content, a separate top-tier node) -- it only makes
the harness *emit* wall-limited-vs-completion-limited status so that registration can define a
rule over it.

## Defect source

- `results/campaign51_20260728/realistic_20260729/tree2_20260830/B8_2_S3_PILOT_READOUT_RECORD.md`
  §4 (caveats 4.1-4.3), quoted and reproduced below.
- Ledger row #288, `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`.
- Design of record: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md`
  (line 233: the cell-T/cell-S SD ratio registered as an S4 input; §2.3: cell T carries no
  coverage/PIT claim by design -- only its SD is the like-for-like comparand).
- Harness code of record: `results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py`
  (the only file touched by this build).

## Files changed

- **Modified**: `results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py`
  (+227/-7 lines; `git diff --stat` confirmed no other tracked file touched).
- **Added** (new, not yet committed -- chair commits): `results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py`
  (211 lines, 9 regression tests, CPU-only, colocated with the harness so `THIS_DIR`-relative
  paths inside the module under test resolve the same way as a direct invocation).

No physics-trigger file (`physical_relations.py`, `constants.py`, `LISA_configuration.py`,
`parameter_estimation.py`, `bayesian_statistics.py`, `simulation_detection_probability.py`,
`cosmological_model.py`) was touched -- confirmed via `git diff --name-only` against that file
list (empty result). `b8_cal_harness.py` is a results-directory driver/aggregator script, not
one of the trigger files; the `/physics-change` gate does not apply to this build and was not
invoked.

## The three defects and their fixes

### (a) seed-population separation

**Defect** (pilot record §4.1): `score_only()`'s glob (`universe_seed*_{cell}.json`) does not
distinguish `n_draw_requested` (the population/N tag), so the cell-S score-only pass pooled 3
N-ladder timing seeds (N=106, 400, 1588) with the 63 N=200 pilot seeds into one contaminated
`n_universes=66` aggregate.

**Fix** (`b8_cal_harness.py:1454-1465` `PopulationMixError` + `_population_tag`;
`:1470-1531` `score_only()` body): `score_only(work_root, cell, population=None)` now:
1. Reads every matched checkpoint's `n_draw_requested` as its population tag.
2. If `population` is omitted and more than one population is present among the matched files,
   it **refuses** -- raises `PopulationMixError` naming every population and its count, rather
   than silently pooling (the g-population lint). If exactly one population is present (the
   common case), it is used automatically -- no behavior change for unmixed data.
3. If `population` is passed explicitly, it aggregates only checkpoints at that population and
   reports every excluded row (`excluded_other_population`, with file path + its
   `n_draw_requested`) and the full `populations_present_before_filter` list in the output --
   so 0 rows are ever silently mixed, and exclusions are auditable, not just dropped.
4. New CLI flag `--population INT` (`:1848-1855`) threads this through; a
   `PopulationMixError` caught in `main()` (`:1886-1890`) prints the lint refusal and exits 1
   instead of crashing with a traceback.

Verified against the banked pilot data (`b8_cal_harness_work_ladder/`, 86 checkpoint files):
`score_only(wr, "S")` with no population now raises `PopulationMixError` listing
`{106: 1, 200: 63, 400: 1, 1588: 1}`; `score_only(wr, "S", population=200)` gives the clean
`n_universes=63` cell-S aggregate (F_no_bh=7.450, F_with_bh=11.38 -- close to, but correctly
distinct from, the row #288 contaminated 66-universe numbers 7.426/11.35, since 3 fewer,
differently-scaled universes are now excluded rather than pooled).

### (b) missing cell-T aggregation

**Defect** (pilot record §3.2-3.3): the 20 completed cell-T checkpoints
(`universe_seed902000_T.json` .. `universe_seed902019_T.json`) existed on disk but no
`--score-only --cell T` invocation was ever run, so no coverage/PIT/F numbers and no T0/T-vs-S
ratio (design line 233) could be quoted.

**Fix**: `score_only()` was never cell-special-cased in a way that blocked T (`--cell` already
accepted `choices=["S","T"]`); the actual gap was operational (no invocation), not a code
defect in the per-cell path itself -- confirmed by running `score_only(wr, "T")` on the banked
20-checkpoint set, which succeeds cleanly (single population, no lint refusal needed:
`n_universes=20`, F_no_bh=11.27). What **was** added: `score_ratio_t_over_s()`
(`b8_cal_harness.py:1707-1733`), the T0/T-vs-S control read design line 233 registers as an S4
input -- it runs `score_only` for both cells at a shared `population`, and reports the
`sigma_h_harness_median_sd` ratio per channel (`T_over_S`), with an explicit `reason` string
(not a silent NaN) if either cell has zero matching universes. New CLI flag
`--score-only-ratio-t-s` (`:1858-1864`) exposes this from the command line, printing both
per-cell reports plus the ratio block. Per design §2.3, this function computes **only** the SD
ratio, not a coverage/PIT verdict for cell T -- its docstring says so explicitly, matching the
design's "no coverage claim from cell T" rule.

Verified against banked data: `score_ratio_t_over_s(wr, population=200)` gives
`no_bh: T_over_S=1.517`, `with_bh: T_over_S=0.9984` (S=63 universes, T=20 universes, both at
N=200).

### (c) wall-limited stop rule (information only -- no rule invented)

**Defect** (pilot record §4.2): both pilot cells stopped on `--max-wall-s`, not completion (S:
63/100, T: 20/25), but this fact lived only in log text, not in any machine-readable output the
aggregator could report per cell.

**Fix**:
- `run_status_path(work_root, cell)` (`b8_cal_harness.py:1131-1143`) names a per-cell sidecar
  file `_run_status_{cell}.json` under `--work-root`.
- The driver loop in `main()` now tracks `stopped_reason` (`:1917` init to
  `"exhausted_n_universes"`, `:1920` set to `"wall_limited"` if the wall-limit break fires) and,
  at the end of every non-`--score-only` invocation, writes/overwrites the sidecar
  (`:1990-2012`) with `stopped_reason`, requested vs. done counts for this invocation, the
  cumulative checkpoint count for the cell, `max_wall_s`, and elapsed wall time. The file
  explicitly states it is a FACT record, not a stop RULE.
- `score_only()` now reads this sidecar if present and folds it into the output as
  `run_status` (`b8_cal_harness.py`, after the `count_audit` block): `available`,
  `stopped_reason`, `wall_limited` (bool), invocation counts, `max_wall_s`,
  `wall_elapsed_s_this_invocation`. If the sidecar is absent (e.g. checkpoints predate this
  repair, or the driver was never invoked for that cell), `run_status.available = False` with
  an explicit `reason` string -- never silently omitted.
- `print_score_only_report()` prints the `run_status` line so a human reading the console output
  sees wall-limited-vs-completion-limited status directly.

No stop rule is defined, applied, or implied anywhere in this build -- confirmed by grep: the
only occurrences of "stop rule" in the diff are in docstrings/comments explicitly disclaiming
that this build defines one, routing that content to `r-b82-s4`.

Verified against banked data: since the pilot predates this repair, no `_run_status_*.json`
sidecar exists yet for `b8_cal_harness_work_ladder/` -- `score_only(wr, "S", population=200)`
correctly reports `run_status.available=False` with reason "no `_run_status_S.json` sidecar
found ... -- either the driver was never invoked for this cell under this work-root, or these
checkpoints predate the row #288 S4 defect (c) repair". This is the expected/correct behavior
for pre-repair data; a fresh S4 registration run under the repaired harness will populate the
sidecar automatically.

## g-byte-id check on untouched code paths

**Check plan**: every checkpoint-producing code path (`run_one_universe`, `_channel_stats`,
`_score_at_truth_by_class`, the count-audit per-bin sums, `sigma_floor_for`, `my_ks_uniform`,
`binom_bands`, the driver's per-universe checkpoint write) is untouched by this build -- only
`score_only()`'s glob-to-aggregate wiring, the CLI, and `main()`'s post-loop status write were
edited. The cheap local check available (no cluster/GPU access, no re-run of the ~100k-s pilot
compute): re-run the (unchanged) per-checkpoint numeric computation inside `score_only()` on the
existing 86 banked checkpoint JSONs under both the OLD (pre-repair, `git show HEAD:...`) and
NEW code, for the one case where both are directly comparable without the fix changing scope --
cell T, which is already single-population (20/20 checkpoints at N=200), so old
`score_only(wr,"T")` and new `score_only(wr,"T",population=200)` aggregate the identical
checkpoint set.

**Result**: loaded both module versions via `importlib` (old = `git show HEAD:...` written
in-place, restored after the check; new = the repaired file) and diffed every JSON-serializable
key present in both outputs (`json.dumps(..., sort_keys=True)` equality per key):

```
old-only keys: set()
new-only keys: {'populations_present_before_filter', 'excluded_other_population', 'run_status', 'population'}
total shared-key mismatches: 0
```

0 mismatches on every pre-existing field (`no_bh`, `with_bh`, `count_audit`, `n_universes`,
`cell`, `files`) -- the repair is additive-only on the untouched single-population path. This
is a full-population comparison (n=20, the entire banked cell-T set), not a stochastic N>=1e5
Monte Carlo pairing -- the harness has no such large deterministic-pair test fixture available
locally (its own N>=1e5 g-byte-id instrument, `--no-draw-weight-cache`/`--no-precompute-cache`,
requires re-running the ~100k-s generative pipeline, out of scope for a cheap/medium-effort
build with no cluster access this session). The mixed-population cell-S case (66 vs. 63
universes) is **not** a byte-identity comparison target -- that divergence *is* the bug being
fixed, not a regression.

## Test results (verbatim)

New regression test file:
`results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py`
(9 tests, CPU-only, synthetic minimal checkpoints -- no GPU, no real generative context, no
cluster). Note: this file is **not** under `darksiren_emri_test/` (`pyproject.toml`
`testpaths = ["darksiren_emri_test"]`), so it is not picked up by a bare `uv run pytest` and
does not affect the repo-wide `fail_under = 25` coverage gate; it must be pointed at explicitly,
as done below and as any S4/S5 follow-on node should do.

```
$ uv run pytest results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py -v --no-cov
============================= test session starts ==============================
platform linux -- Python 3.13.13, pytest-9.0.2, pluggy-1.6.0 -- /home/jasper/Repositories/darksiren-emri/.venv/bin/python3
cachedir: .pytest_cache
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/jasper/Repositories/darksiren-emri
configfile: pyproject.toml
plugins: benchmark-5.2.3, anyio-4.12.1, cov-7.0.0
collecting ... collected 9 items

results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_only_refuses_mixed_population PASSED [ 11%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_only_explicit_population_excludes_other_rows PASSED [ 22%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_only_single_population_needs_no_explicit_arg PASSED [ 33%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_only_aggregates_cell_t PASSED [ 44%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_ratio_t_over_s PASSED [ 55%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_ratio_t_over_s_missing_cell_reports_reason PASSED [ 66%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_only_reports_run_status_when_present PASSED [ 77%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_score_only_reports_run_status_absent_explicitly PASSED [ 88%]
results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py::test_run_status_path_completion_limited PASSED [100%]

============================== 9 passed in 2.21s ===============================
```

```
$ uv run ruff check results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py
All checks passed!

$ uv run ruff format --check results/campaign51_20260728/realistic_20260729/tree2_20260830/test_b8_s4_harness_repair.py results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py
1 file already formatted, 1 file already formatted   (after `ruff format` was applied once to b8_cal_harness.py to match repo style)

$ uv run mypy results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py
Success: no issues found in 1 source file
```

(An earlier invocation of pytest pointed directly at the single test file without `--no-cov`
triggered the repo-wide `fail_under=25` coverage gate on `darksiren_emri/` -- expected and
inert, since this file sits outside `testpaths` and a single-file coverage slice cannot reach
25% of the whole package; not a defect in this build. All 9 tests still reported PASSED in that
run too.)

## What could NOT be fixed here (routed onward, not silently dropped)

- **The stop rule itself** (what counts as an acceptable wall-limited partial run, whether a
  registered n_U can be satisfied incrementally across resumed invocations) is explicitly out
  of scope per the task's own instruction and the graph's decisions-table row 3 scope note --
  routed to `r-b82-s4` (registration node) as intended.
- **g-byte-id at N>=1e5 pairs** (the graph node's own acceptance criterion, "infra 2.5") was not
  run -- it requires the `--no-draw-weight-cache`/`--no-precompute-cache` real-generative-context
  comparison the harness itself documents (§8/§10 of the harness docstring), which needs
  cluster-scale compute this session did not have access to. The cheap local substitute
  performed (full-population, n=20, exact-match diff on every shared JSON key across old/new
  code) is reported above as what was actually checked, not conflated with the N>=1e5 criterion.
- **No re-run of the S3 pilot itself** was performed (would consume the ~100k-s wall budget
  again); this build only repairs the aggregator and driver status-emission code, per its
  "build" (not "measure") node type.
- **A cell-T coverage/PIT verdict** was deliberately NOT produced -- design §2.3 states no
  coverage claim is made from cell T (PIT degenerate by construction); `score_ratio_t_over_s`
  reports only the SD ratio, consistent with that design constraint.

## Commit status

Not committed -- per instruction, the chair commits.
