---
name: cluster
description: >
  How to work with the bwUniCluster 3.0 EMRI environment: the canonical paths,
  the always-run-first preflight readiness gate, submit/monitor/retrieve recipes,
  the dataset inventory, and the known gotchas. Use whenever a task involves the
  cluster — before submitting jobs, checking state, or retrieving results.
disable-model-invocation: true
argument-hint: [preflight | status | submit | pull | inventory]
allowed-tools: Bash, Read
---

## EMRI Cluster Operations Guide

The single source of operational truth for the cluster. **Do not rediscover this
each session** — read it, and keep it current when the cluster layout changes.

### Golden rule
**Run the preflight before submitting or evaluating anything on the cluster:**
```bash
ssh bwunicluster 'bash -s' < cluster/preflight.sh
```
It is read-only and prints one block: repo branch/commit/tag, venv usability,
catalog schema, workspace expiry, queue (+ zombie jobs), and a live dataset scan.
Only proceed when it says `VERDICT: READY ✓`. Paths/expectations live in
`cluster/cluster.env`.

### Canonical facts (verify with preflight, don't assume)
- **SSH alias:** `bwunicluster` (works non-interactively / `BatchMode`).
- **ONE repo:** `~/MasterThesisCode`. Do **not** make separate clones/worktrees for
  parallel or "frozen" work — instead branch and **tag** (e.g. `commission-base`);
  archaeology returns via `git checkout <tag>`. (A redundant `mtc-eval` worktree
  was retired 2026-07-01 for exactly this reason.)
- **Workspace:** `ws_find emri` → `/pfs/work9/workspace/scratch/st_ac147838-emri`.
  bwHPC workspaces **expire** (extend: `ws_extend emri 60`); copy finals off before then.
- **uv:** `~/.local/bin/uv`. First-time env: `cluster/setup.sh`.

### ⚠️ Gotchas (each has bitten before)
1. **venv needs modules.** `.venv/bin/python` is linked to the module-provided
   `libpython3.13.so`. Always `source cluster/modules.sh` before `.venv/bin/python`
   or `uv run` — otherwise "cannot open shared object libpython3.13.so". The sbatch
   scripts already `module load`; interactive use does not.
2. **Catalog schema.** `reduced_galaxy_catalogue.csv` must be **8 columns**
   (redshift-flag retained, commit 479afdd+). A 6/7-col copy is stale and read
   silently. It is gitignored (~1.6 GB); stage from the dev box:
   `rsync -avz --partial master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv bwunicluster:MasterThesisCode/master_thesis_code/galaxy_catalogue/`
   (GLADE+.txt is not on the cluster; the reduced csv cannot self-rebuild there.)
3. **Zombie jobs.** A failed parent leaves dependents `DependencyNeverSatisfied` —
   they never run and clog the view. Preflight flags them; clear with `scancel <id>`.
4. **Seed convention.** Per-task seed = `BASE_SEED + SLURM_ARRAY_TASK_ID`
   (reproducible; resubmits reproduce). Recorded in `run_metadata_<task>.json`.

### The pipeline & where artifacts land
```
submit_pipeline.sh --tasks N --steps M --seed S
   simulate.sbatch (GPU array) → merge.sbatch (CPU) → evaluate.sbatch (CPU) [→ combine.sbatch]
Output root:  $WORKSPACE/run_YYYYMMDD_seedS/
   logs/            simulate_/merge_/evaluate_/combine_ .out/.err
   simulations/
     cramer_rao_bounds.csv            # STAGE 2 raw (after merge)
     prepared_cramer_rao_bounds.csv   # STAGE 2 SNR-filtered (after prepare)
     injections/                      # STAGE 1 P_det pool (if an injection campaign)
     posteriors/  posteriors_with_bh_mass/   # STAGE 3 h_*.json + combined_posterior.json
   run_metadata_<task>.json           # git_commit, seed, timestamp, args  ← provenance
```

### Recipes
- **Preflight / status:** `ssh bwunicluster 'bash -s' < cluster/preflight.sh`
- **Submit a campaign** (from cluster, after `source cluster/modules.sh`):
  `bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42`
  Test small first: `--tasks 2 --steps 10`. Injection campaign: `cluster/submit_injection.sh`.
- **Monitor:** `squeue -u $USER` · `sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode`
  · live: `tail -f $WORKSPACE/run_*/logs/*.out`
- **Resubmit only failed array tasks:** `bash cluster/resubmit_failed.sh <JOBID> <RUN_DIR> <BASE_SEED> <STEPS>`
- **Retrieve results:**
  `rsync -avz bwunicluster:$(ssh bwunicluster 'ws_find emri')/run_YYYYMMDD_seedS/ ./results/`
- **Local evaluate** (dev box, CPU): `uv run python -m master_thesis_code <dir> --evaluate --h_value 0.73 --num_workers N`

### Launching jobs & writing new ones
Full guide with exact commands for every case: **`cluster/LAUNCHING_JOBS.md`**.
- **Mental model:** PROJECT_ROOT (`~/MasterThesisCode`, code+catalog) vs RUN_DIR
  (`$WORKSPACE/run_*`, output). Jobs `cd $PROJECT_ROOT` then
  `ln -sfn $RUN_DIR/simulations $PROJECT_ROOT/simulations` (the pipeline uses
  relative paths from CWD). Env is threaded via `sbatch --export=ALL,RUN_DIR=…`.
- **Regular pipelines:** simulation `cluster/submit_pipeline.sh --tasks N --steps M --seed S`
  (chains simulate→merge→evaluate→combine) · injection/P_det `cluster/submit_injection.sh --tasks_per_h N --steps M --seed S`
  (single-h suffices — p_det is h-invariant) · standalone inference = `evaluate.sbatch`+`combine.sbatch` on an existing RUN_DIR.
- **New job:** copy **`cluster/JOB_TEMPLATE.sbatch`** (encodes the module/venv load,
  symlink dance, seed derivation, idempotency skip-if-output).
- **Testing:** GPU smoke `cluster/submit_gpu_smoke.sh` · tests `uv run pytest -m "not gpu and not slow"`
  (add `-m gpu` on the cluster) · call functions directly from `.venv/bin/python`
  after `source cluster/modules.sh` · interactive `salloc --partition=dev_gpu_h100`.
  Entry points: `emri-merge` / `emri-prepare` / `emri-merge-injections`.
- **Always test small first** (`--tasks 2 --steps 10`) and set `--seed`.

### Dataset inventory (which run produced what, and is it still aligned?)
- **Live ground truth:** the `[DATASETS]` block of `preflight.sh`.
- **Semantic map + provenance chain:** `cluster/datasets.yaml`
  (injection pool → P_det → CRB seed → posteriors; retirement status).
- **Full history + Pipeline-Change staleness tiers:** `DATA_INVENTORY.md`.
- **Alignment check:** a dataset matches current code only if its
  `run_metadata.json:git_commit` is an ancestor of the eval commit **and** no
  Pipeline-Change trigger file changed since. When unsure, re-run.

### More detail
`cluster/README.md` — full quickstart, troubleshooting (OOM/timeout/CUDA), script reference.
