# Launching jobs on bwUniCluster — the complete guide

Everything you need to run the EMRI pipeline (or a one-off test) on the cluster.
Read this once; it replaces guessing. **Always run `cluster/preflight.sh` first**
(see `cluster/SKILL.md` / the `/cluster` skill). Paths live in `cluster/cluster.env`.

---

## 1. The mental model (internalize this — it explains every gotcha)

There are **two distinct directories**, and confusing them causes most failures:

| | What | Where |
|---|---|---|
| **PROJECT_ROOT** | the **code** + the **galaxy catalog** | `~/MasterThesisCode` (the one repo) |
| **RUN_DIR** | this run's **output** (logs, CSVs, posteriors) | `$WORKSPACE/run_YYYYMMDD_seedS/` |

The package uses **relative paths from the current working directory**:
- it reads the catalog from `./master_thesis_code/galaxy_catalogue/` (handler.py:24),
- it reads/writes `./simulations/…`.

So every batch job runs from a **private per-run CWD** (TC-03): it `cd`s into
`$RUN_DIR/cwd/`, which holds two symlinks —
`simulations → $RUN_DIR/simulations` and
`master_thesis_code → $PROJECT_ROOT/master_thesis_code`. Code and catalog come
from the one repo; output lands in `$RUN_DIR/simulations/`. Because each run
owns its CWD, **concurrent runs with different RUN_DIRs are safe** — there is
no shared `$PROJECT_ROOT/simulations` symlink to fight over anymore.
(`merge.sbatch` needs no CWD tricks — it uses absolute `--workdir` paths.)

**Env threading:** submit wrappers pass everything the sbatch needs via
`sbatch --export=ALL,RUN_DIR=…,BASE_SEED=…`. The sbatch validates them and falls
back to `PROJECT_ROOT=${PROJECT_ROOT:-$HOME/MasterThesisCode}`.

**Modules:** `.venv/bin/python` is linked to the module `libpython3.13.so`. Every
job `source cluster/modules.sh` (which also exports `$WORKSPACE`, `$PROJECT_ROOT`,
`$VENV_PATH`) then `source "$VENV_PATH/bin/activate"`.

---

## 2. Partition cheat-sheet

| Partition | Use | Limits / anchors (2026-07-03) |
|---|---|---|
| `gpu_h100_short` | production GPU sim (tasks are time-capped, backfills fast) | 30-min wall, 1 GPU/task |
| `gpu_a100_short` | injection campaigns (`inject.sbatch`) + GPU smoke tests | `inject.sbatch` requests a 30-min wall; the smoke test uses 5 min |
| `cpu,cpu_il` | inference / merge / combine | evaluate: **56–76 min per h-value @ 3355 events / 16 cpus** (jobs 5732036, volume_deconv; 6h pre-smoke budget, re-size after smoke); combine: **~20 min posteriors** + 90-min budget (job 5735965 anchor); figures rendered locally (`RENDER_FIGURES=0`) |
| `dev_gpu_h100` / `dev_*` | quick queue for testing | short wall, fast start |

Seed convention everywhere: **per-task seed = `BASE_SEED + SLURM_ARRAY_TASK_ID`**
(reproducible; resubmits reproduce). Recorded in `run_metadata_<task>.json`.

### 2a. Node topology & packing — size the sbatch before you submit

Full detail + evidence in `cluster/SKILL.md` gotchas 5-9 (node-topology findings from
the venue-transfer perf pass, `results/venue_transfer_20260811/perf/PERF_ROADMAP.md` §4).
Summary for anyone writing or resizing a new sbatch:
- `cpu_il` nodes are **128-core**; plain `cpu` nodes are **192-core** — don't assume one
  when sizing against the other. `dev_cpu_il` QOS is MaxSubmit 4 / MaxRunning 1 / 30-min.
- **Match `--cpus-per-task` to the actual worker count** (pool size vs. seed count in the
  task's range) — an oversized reservation idles cores for the whole run; either shrink
  the reservation or use a finer grain (e.g. `--grain h`) to fill them.
- **Packing >2 tasks/node measurably slows each task** (~1.7× per-seed, memory-bandwidth
  contention) — size `--time` against the *contended* anchor whenever packing is tight,
  not the uncontended one.
- **`sbatch --test-only` is not a walltime prediction tool** for short wide jobs (ignores
  backfill, off by orders of magnitude, EXP-61) — log probe-vs-actual instead of trusting it.
- **A walltime kill on a JSON-at-end instrument loses the whole task's work**, and
  `scontrol` walltime extensions are denied to regular users — size with margin or plan
  to resubmit, not extend.

---

## 3. The three regular pipelines (exact commands)

Run all of these **from `~/MasterThesisCode` after `source cluster/modules.sh`.**

### 3a. Simulation → CRB (+ auto merge → evaluate → combine)
One command chains simulate (GPU) → merge (CPU) → evaluate (CPU) → combine:
```bash
bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42 \
    --injection_pool "$WORKSPACE/injection_<date>_seed<seed>/simulations/injections"
# creates $WORKSPACE/run_YYYYMMDD_seed42/ ; prints all job IDs + a sacct line
```
- `--tasks` = GPU array size, `--steps` = EMRI iterations/task, `--seed` = base seed.
- `--injection_pool` (required unless `--no_injections`) links the pool's
  `injection_h_*.csv` into `RUN_DIR/simulations/injections/` at submit time, so
  evaluate's p_det grid uses exactly the intended pool (see `cluster/datasets.yaml`).
- `--h_true V` sets the injected truth for closure runs (default 0.73); a
  non-default truth is embedded in the run-dir name (`run_YYYYMMDD_seedS_h0p67`).
- Dependency chain: simulate → merge (`afterany`, tolerates task timeouts) →
  evaluate (`afterok`, h-grid parsed from `evaluate.sbatch` — currently 41
  points 0.60–0.86) → combine (`afterok`).
- **Test small first:** `--tasks 2 --steps 10`.

### 3b. Injection campaign → P_det pool
Builds the detection-probability pool for `SimulationDetectionProbability`:
```bash
bash cluster/submit_injection.sh --tasks_per_h 80 --steps 900 --seed 12345
# => 80 tasks × 900 events = 72,000 pooled samples at h=0.73 (single-h default)
```
- **A single h suffices** — the survival p_det is h-invariant (`d_hor = SNR·d_L/thr`
  cancels h). Multi-h (`--h_values "0.60,0.70,…"`) only adds pooled samples.
- Output: `$WORKSPACE/injection_YYYYMMDD-HHMMSS_seed12345/simulations/injections/`.

### 3c. Inference → H0 posterior (standalone, on an existing CRB)
Usually part of 3a, but to (re-)evaluate an existing run:
```bash
RUN=$WORKSPACE/run_20260516_seed400_phase50
# --array must match the H_VALUES count in evaluate.sbatch (currently 41 → 0-40)
sbatch --parsable --array=0-40 \
  --output="$RUN/logs/evaluate_%A_%a.out" --error="$RUN/logs/evaluate_%A_%a.err" \
  --export=ALL,RUN_DIR="$RUN" cluster/evaluate.sbatch
# then combine:
sbatch --dependency=afterok:<EVAL_JOB> \
  --output="$RUN/logs/combine_%j.out" --error="$RUN/logs/combine_%j.err" \
  --export=ALL,RUN_DIR="$RUN" cluster/combine.sbatch
```
Needs `$RUN/simulations/prepared_cramer_rao_bounds.csv` + an injection pool
symlinked/present. **A standalone run-dir also needs the raw
`$RUN/simulations/cramer_rao_bounds.csv` present alongside the `prepared_*.csv`**
— it's not just the prepared file; `evaluate.sbatch` reads both, and a run-dir
assembled by hand (rather than produced by the full 3a pipeline) is easy to
leave missing the raw CSV. Outputs `posteriors/` + `posteriors_with_bh_mass/` + `combined_posterior.json`.

---

## 4. Testing on the cluster

### 4a. GPU smoke test (prove GPU actually runs, 1 event, ~5 min)
```bash
bash cluster/submit_gpu_smoke.sh          # → $WORKSPACE/.../gpu_smoke/ traces + nvidia-smi
```

### 4b. Run the unit tests on the cluster
```bash
source cluster/modules.sh
uv run pytest -m "not gpu and not slow"   # fast subset
uv run pytest -m gpu                       # GPU-only tests (cluster has CUDA)
uv run pytest -m slow                      # closures / long tests
```

### 4c. Invoke pipeline functions DIRECTLY (ad-hoc checks / new test scaffolds)
The pipeline is a normal importable package (`import few`, NOT `fastemriwaveforms`).
After `source cluster/modules.sh`, call functions straight from the venv python:
```bash
.venv/bin/python - <<'PY'
from master_thesis_code.physical_relations import dist_vectorized, comoving_volume_element
from master_thesis_code.emri_rate import R_eff_per_mbh, mbh_mass_function
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler
# e.g. sanity-check a relation at a known limit:
print("dist(z=0) =", dist_vectorized([0.0], h=0.73))   # must be ~0
PY
```
Console entry points (defined in `pyproject.toml [project.scripts]`):
`emri-merge` (merge per-task CRB CSVs), `emri-prepare` (SNR-filter → prepared CSV),
`emri-merge-injections`. Run under `uv run` or the activated venv.

Test conventions (mirror these when adding tests): files
`master_thesis_code_test/test_<module>.py`; `@pytest.mark.gpu` for CUDA-needing
tests; the `xp` fixture parametrizes numpy/cupy — thread `use_gpu=(xp.__name__=="cupy")`.
Physical-correctness tests must NOT require a GPU (`dist(0)==0`, `psd>0`, `p_det∈[0,1]`).

### 4d. A throwaway test job (interactive)
For a quick interactive GPU/CPU shell instead of a batch script:
```bash
salloc --partition=dev_gpu_h100 --gres=gpu:1 --time=00:20:00
# then, on the allocated node:
cd ~/MasterThesisCode && source cluster/modules.sh && source .venv/bin/activate
python -m master_thesis_code /tmp/testrun --simulation_steps 2 --seed 1 --use_gpu
```
For a batch one-off, copy `cluster/JOB_TEMPLATE.sbatch` and submit with a small
inline wrapper that makes `RUN_DIR` and passes `--export`.

---

## 5. CLI flag reference (`python -m master_thesis_code <working_dir> <flags>`)

| Flag | Purpose |
|---|---|
| `--simulation_steps N` | EMRI iterations (sim/inject) |
| `--simulation_index I` | task index (→ output file suffix); = `SLURM_ARRAY_TASK_ID` |
| `--seed S` | random seed (always set it for reproducibility) |
| `--h_value V` | injected truth (sim/inject) or eval point (evaluate) |
| `--use_gpu` | enable CuPy/GPU (never hardcoded; cluster only) |
| `--injection_campaign` | build a P_det injection pool |
| `--evaluate` | run Bayesian inference → posterior |
| `--combine` | combine per-h posteriors → `combined_posterior.json` |
| `--snr_analysis` | SNR-only pass |
| `--catalog_only` | catalog-likelihood ablation |
| `--generate_figures` / `--generate_interactive` | plots |
| `--pdet_dl_bins 60` / `--pdet_mass_bins 40` | P_det grid resolution (pin for reproducibility) |
| `--pdet_estimator` / `--strategy` | estimator / strategy selectors |
| `--num_workers N` | inference pool size (default: cgroup CPUs − 2) |
| `--fisher_cond_threshold` · `--save_baseline` · `--compare_baseline` · `--log_level` | misc |

---

## 6. Re-run safety & idempotency

- **Skip-if-output** per unit (evaluate/combine already do this): guard on the
  target file so resubmits don't redo finished work. `evaluate.sbatch` exits 0
  per-task if its `h_<label>.json` posteriors already exist.
- **Archive-then-write**: `submit_pipeline.sh` archives existing `posteriors*/`
  to `simulations/archive/eval_<ts>/` **at submit time on the login node**
  (moved out of evaluate.sbatch task 0, which raced with sibling tasks) — so a
  re-eval never silently overwrites without a copy.
- **`afterany` vs `afterok`**: merge follows simulate with `afterany` (GPU tasks
  are time-capped and "time out" by design); evaluate/combine use `afterok`.
- **Resubmit only failures:**
  `bash cluster/resubmit_failed.sh [--include-timeout] [--force] <JOBID> <RUN_DIR> <BASE_SEED> <STEPS> [H_VALUE]`.
  - Default states: `FAILED,NODE_FAIL,OUT_OF_MEMORY`. `--include-timeout` adds
    `TIMEOUT` — opt-in because TIMEOUT is the *expected* terminal state on
    `gpu_h100_short` (time-capped by design).
  - `H_VALUE` is optional: recovered from `run_metadata_*.json` if omitted
    (conflicting explicit values abort); falls back to 0.73 with a loud warning
    only if neither source exists.
  - The script **refuses to run** if `simulations/cramer_rao_bounds.csv` already
    exists (merge appends → duplicate events); archive/remove the merged CSVs or
    `scancel` the pending merge first, or pass `--force`.
- A failed parent leaves children `DependencyNeverSatisfied` (zombie) — preflight
  flags them; clear with `scancel`.

---

## 7. Pre-launch checklist

1. `ssh bwunicluster 'bash -s' < cluster/preflight.sh` → `VERDICT: READY ✓`.
2. Right **branch/commit** (`git -C ~/MasterThesisCode log -1`)? Catalog **8-col**?
3. `source cluster/modules.sh` (so `$WORKSPACE` resolves).
4. **Test small first** (`--tasks 2 --steps 10`, or the smoke test).
5. Set `--seed`. Note the printed `RUN_DIR` + job IDs.
6. Monitor: `squeue -u $USER` · `sacct -j <IDs> --format=JobID,State,Elapsed,MaxRSS,ExitCode`.
7. Retrieve: `rsync -avz bwunicluster:$RUN_DIR/ ./results/…`. Copy finals off the
   workspace before it expires (`ws_list`, `ws_extend emri 60`).
