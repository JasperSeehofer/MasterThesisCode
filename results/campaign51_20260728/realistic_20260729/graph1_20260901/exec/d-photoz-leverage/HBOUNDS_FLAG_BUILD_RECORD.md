# `--h-bounds` flag build record

Builder task: add a CLI flag to `hier_s0_driver.py` so `H_BOUNDS` (module
constant `(0.50, 0.86)`, the sole decisive discriminator identified in
`DRIVER_VENUE_FORENSICS.md` section 5) can be overridden without editing the
driver. No cluster access; no edits under `darksiren_emri/`.

## What changed

`results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`:

- New `parse_h_bounds(spec: str) -> tuple[float, float]` helper (NumPy
  docstring) parsing `"LO,HI"`, requiring `LO < HI`.
- New `--h-bounds LO,HI` CLI flag, `default=H_BOUNDS` (the existing
  `(0.50, 0.86)` module constant) — so every existing invocation that omits
  the flag is byte-identical.
- `h_bounds` threaded as a parameter through the full call chain that
  previously read the module constant directly: `run_theta_node` (the
  S0-A/S0-R/FT/b0i/iiib path), `run_arm_seed_s0a`, `run_arm_seed_s0r`,
  `run_seed_s0c` (the S0-C path, `c1d.run_mirror_seed_inprocess` call), the
  `_run_one_seed_worker` picklable-args tuple, and `run_arm`. All resolve to
  the CLI-supplied value (or `H_BOUNDS` when unset); no other constant or
  default value was touched.
- Provenance: `run_arm`'s output payload (`<arm>_{smoke,full}_output.json`)
  now carries `"h_bounds": [lo, hi]` at top level; `run_seed_s0c`'s per-seed
  dict (S0-C's `payload["per_seed"]` entries) carries the same key. Applies
  to all three arms (S0-A, S0-R, S0-C) and all three `--config` venues (b0i,
  ft, iiib). `--score-only` is unchanged — it never calls `evaluate()` (reads
  existing on-disk CSVs only), so `h_bounds` has no effect there and was not
  added to its output for that reason (nothing to record: it forwards to no
  node).

`git diff --stat` (only the driver changed):

```
 .../fanout1_20260829/hier_s0_driver.py             | 72 +++++++++++++++++++++-
 1 file changed, 70 insertions(+), 2 deletions(-)
```

Full diff: see `git diff -- results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`.

## Quality gate

- `uv run ruff check hier_s0_driver.py` — All checks passed.
- `uv run ruff format --check hier_s0_driver.py` — clean (after one
  `ruff format` pass to add the blank line the formatter wants after the new
  top-level `parse_h_bounds` function).
- `uv run mypy hier_s0_driver.py` — 15 errors, all pre-existing
  (`**cat_num_surv_2d_kwargs` dict-unpacking against `run_mirror_seed_inprocess`'s
  keyword-only signature; unrelated to this change). Confirmed identical
  error count/lines via `git stash` + re-run against the unmodified file.

## Byte-identity proof

Ran the driver's own `--score-only` (cheapest read-only mode; no `evaluate()`
call, no venue construction) against the retrieved S0-B run
`results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/s0b_run_20260902`,
before (git-stashed original) and after (this change), on fresh copies of
the retrieved data:

```
--arm S0-A --score-only --seeds 900101 --nodes truth,s_plus,s_minus \
  --theta-sites 2.2 --smear off --config iiib --out-root <copy>
```

Both runs exited 0. Results:

```
$ diff beforerun/s0a_score_output.json afterrun/s0a_score_output.json
(no output — identical)
$ diff beforerun/s0a_score.md afterrun/s0a_score.md
(no output — identical)
```

`stdout` differs only in the echoed output-file *paths* (different scratch
directories for the before/after copies) — the JSON payload embedded in
stdout is otherwise identical. `--h-bounds` was not passed in either run
(testing the default path), confirming the flag's default is a true no-op.

## `cluster/graph1_s0b_truth_hbounds060.sbatch`

Written, **not submitted** (no cluster access in this task). Exact copy of
`cluster/graph1_m_s0b_production.sbatch`'s driver invocation, restricted to
the **truth node only** via the production sbatch's own node-selection
mechanism (`NODES=(...)` indexed by `$SLURM_ARRAY_TASK_ID` — here
`NODES=(truth)` with `--array=0-0`, instead of the production script's
5-element array), with `--h-bounds 0.60,0.86` appended (production's class
default, per `DRIVER_VENUE_FORENSICS.md`'s "discriminating single-job test").
Same resources (`cpus-per-task=16`, `time=03:00:00`, partition `cpu_il`),
job name `graph1-s0b-hb060`, out-root
`$WS/graph1_s0b_truth_hbounds060_20260904` (under
`exec/d-photoz-leverage/`, this task's own working area), and the same
ancestor-pin pattern (`git merge-base --is-ancestor`) as the production
sbatch — `EXPECTED_COMMIT="PIN_COMMIT"` is a placeholder for the ops agent to
fill in with the commit that lands this `--h-bounds` change (or a
descendant) before submitting.

Node-selection flag used: the production sbatch has no single-node CLI flag
on the driver itself — node selection is the sbatch's own `NODES[$SLURM_ARRAY_TASK_ID]`
array-indexing convention over the driver's `--nodes` argument. Restricting
to truth-only is therefore a single-element `NODES` array plus `--array=0-0`,
not a driver-side change.
