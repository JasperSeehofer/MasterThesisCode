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
- it reads the catalog from `./master_thesis_code/galaxy_catalogue/`,
- it reads/writes `./simulations/…`.

So every job does the same dance: **`cd $PROJECT_ROOT`** then
**`ln -sfn $RUN_DIR/simulations $PROJECT_ROOT/simulations`** — run from the code,
but redirect `./simulations` to this run's output. Output therefore lands in
`$RUN_DIR/simulations/`. (Because there is one shared symlink, avoid running two
jobs from the same PROJECT_ROOT with different RUN_DIRs *interactively* at once;
SLURM tasks each re-point it at start, and all point to the same RUN_DIR per job.)

**Env threading:** submit wrappers pass everything the sbatch needs via
`sbatch --export=ALL,RUN_DIR=…,BASE_SEED=…`. The sbatch validates them and falls
back to `PROJECT_ROOT=${PROJECT_ROOT:-$HOME/MasterThesisCode}`.

**Modules:** `.venv/bin/python` is linked to the module `libpython3.13.so`. Every
job `source cluster/modules.sh` (which also exports `$WORKSPACE`, `$PROJECT_ROOT`,
`$VENV_PATH`) then `source "$VENV_PATH/bin/activate"`.

---

## 2. Partition cheat-sheet

| Partition | Use | Limits |
|---|---|---|
| `gpu_h100_short` | production GPU sim (tasks are time-capped, backfills fast) | 30-min wall, 1 GPU/task |
| `gpu_a100_short` | GPU smoke tests | 5-min wall |
| `cpu_il` | inference / merge / combine | up to 128 cpus, ~15 min/h-value |
| `dev_gpu_h100` / `dev_*` | quick queue for testing | short wall, fast start |

Seed convention everywhere: **per-task seed = `BASE_SEED + SLURM_ARRAY_TASK_ID`**
(reproducible; resubmits reproduce). Recorded in `run_metadata_<task>.json`.

---

## 3. The three regular pipelines (exact commands)

Run all of these **from `~/MasterThesisCode` after `source cluster/modules.sh`.**

### 3a. Simulation → CRB (+ auto merge → evaluate → combine)
One command chains simulate (GPU) → merge (CPU) → evaluate (CPU) → combine:
```bash
bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42
# creates $WORKSPACE/run_YYYYMMDD_seed42/ ; prints all job IDs + a sacct line
```
- `--tasks` = GPU array size, `--steps` = EMRI iterations/task, `--seed` = base seed.
- Dependency chain: simulate → merge (`afterany`, tolerates task timeouts) →
  evaluate (`afterok`, 38-point h-grid 0.60–0.86) → combine (`afterok`).
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
sbatch --parsable --array=0-37 \
  --output="$RUN/logs/evaluate_%A_%a.out" --error="$RUN/logs/evaluate_%A_%a.err" \
  --export=ALL,RUN_DIR="$RUN" cluster/evaluate.sbatch
# then combine:
sbatch --dependency=afterok:<EVAL_JOB> \
  --output="$RUN/logs/combine_%j.out" --error="$RUN/logs/combine_%j.err" \
  --export=ALL,RUN_DIR="$RUN" cluster/combine.sbatch
```
Needs `$RUN/simulations/prepared_cramer_rao_bounds.csv` + an injection pool
symlinked/present. Outputs `posteriors/` + `posteriors_with_bh_mass/` + `combined_posterior.json`.

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
  target file so resubmits don't redo finished work.
- **Archive-then-write**: `evaluate.sbatch` task 0 archives existing
  `posteriors*/` to `simulations/archive/eval_<ts>/` before a fresh sweep — so a
  re-eval never silently overwrites without a copy.
- **`afterany` vs `afterok`**: merge follows simulate with `afterany` (GPU tasks
  are time-capped and "time out" by design); evaluate/combine use `afterok`.
- **Resubmit only failures:** `bash cluster/resubmit_failed.sh <JOBID> <RUN_DIR> <BASE_SEED> <STEPS>`.
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
