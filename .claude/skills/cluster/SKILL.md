---
name: cluster
description: >
  How to work with the bwUniCluster 3.0 EMRI environment: the canonical paths,
  the always-run-first preflight readiness gate, submit/monitor/retrieve recipes,
  the dataset inventory, and the known gotchas. Use whenever a task involves the
  cluster — before submitting jobs, checking state, or retrieving results.
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
catalog schema, workspace expiry, queue (+ zombie jobs), and a live dataset scan
(including an unregistered/dangling registry cross-check, gotcha 11). Only proceed
when it says `VERDICT: READY ✓` — a trailing `(WARN: ...)` is not a blocker, but
resolve it before the next campaign compounds the backlog. Paths/expectations live
in `cluster/cluster.env`.

### Canonical facts (verify with preflight, don't assume)
- **SSH alias:** `bwunicluster` (works non-interactively / `BatchMode`).
- **ONE repo:** `~/darksiren-emri`. Do **not** make separate clones/worktrees for
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
   `rsync -avz --partial darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv bwunicluster:darksiren-emri/darksiren_emri/galaxy_catalogue/`
   (GLADE+.txt is not on the cluster; the reduced csv cannot self-rebuild there.)
3. **Zombie jobs.** A failed parent leaves dependents `DependencyNeverSatisfied` —
   they never run and clog the view. Preflight flags them; clear with `scancel <id>`.
4. **Seed convention.** Per-task seed = `BASE_SEED + SLURM_ARRAY_TASK_ID`
   (reproducible; resubmits reproduce). Recorded in `run_metadata_<task>.json`.
5. **Node topology: `cpu_il` is 128-core, plain `cpu` is 192-core.** Size
   `--cpus-per-task`/packing assumptions against the partition you actually requested —
   a 64-core reservation on `cpu_il` leaves less headroom than the same number on `cpu`.
   `dev_cpu_il` QOS caps are tight: **MaxSubmit 4 / MaxRunning 1 / 30-min wall** — fine
   for a smoke test, not for anything you intend to leave running.
6. **Cross-task memory-bandwidth contention is real and measured, not theoretical.**
   ~25-core tasks packed 5/node run **~1.7× slower per seed** than the same task packed
   2/node (venue-transfer campaign anchor, ~4h uncontended → ~7h contended per seed).
   Size `--time` against the **contended** per-seed anchor, not the uncontended one,
   whenever more than 2 tasks/node are packed — sizing against uncontended timing under
   tight packing risks a mid-run SLURM kill.
7. **Grain vs. walltime: match `--cpus-per-task` to the actual worker count.**
   `mp.Pool(processes=N)` only has work for as many workers as there are seeds in the
   task's range — reserving 64 cores for a 25-seed pool idles up to 39 cores/task for
   the whole run. Either shrink the reservation to match the seed count (accepting
   contention per gotcha 6, and resizing `--time` by ~1.7×), or use a finer parallel
   grain (e.g. `--grain h`) to give the reserved cores work *within* a seed instead of
   leaving them idle across seeds.
8. **`sbatch --test-only` ignores backfill.** Its predicted start times are off by
   orders of magnitude for short, wide jobs (EXP-61) — do not size a campaign's
   submission plan around them. Log probe-vs-actual pairs at each submission instead of
   trusting the estimate.
9. **Instruments that write JSON only at run end lose everything on a walltime kill** —
   there is no partial-progress checkpoint to resume from. Size `--time` with margin
   against the *contended* anchor (gotcha 6); `scontrol` walltime extensions are denied
   for regular users, so a job sized too tight has no recovery path but resubmitting
   from scratch.
10. **Realization sidecars record ABSOLUTE paths that go stale when the repo moves.**
   `observed_catalogue_seed*.meta.json` stores `parent_csv` as an absolute path; the
   2026-08-17 fusion counterfactual lost both joint_r1 arrays (~2 CPU-h) to the
   pre-rename `MasterThesisCode/` path. Safe repair: verify the file at the new path
   hashes to the sidecar's `parent_csv_sha256`, then rewrite `parent_csv` (keep a
   `.bak` copy). The observed-CSV hash check is separate and unaffected. Prefer
   checking sidecar paths in any run that passes `OBSERVED_CATALOGUE` after a repo
   move.
11. **Register the dataset when the run finishes, not later.** A month of campaigns
    (~30 dirs, ~250 GB — csg_pilot, o4_shards, p3_2d/b0-identity/bat/cf/massab,
    seed61000-65000, `realizations_20260729`) went uninventoried because "update the
    inventory" was a remembered convention, not a check — see
    `cluster/WORKSPACE_ARCHIVAL_TRIAGE_20260827.md`. `preflight.sh`'s `[DATASETS]`
    block now cross-checks the live workspace listing against `cluster/datasets.yaml`
    + `DATA_INVENTORY.md` and WARNs on anything unregistered — but the WARN only
    fires on your *next* preflight run, i.e. after the gap already exists. Add the
    registry entry as part of a run's completion, same commit/session as banking
    the result.
12. **`run_metadata*.json` (git_commit/seed/timestamp/args) is written by
    `main.py`'s `_write_run_metadata`, ONLY when a job runs through
    `python -m darksiren_emri`.** Every bespoke harness driver invoked directly
    (`python <driver>.py` or `python -m darksiren_emri.validation.<module>`) —
    which is what every post-2026-07-28 campaign in gotcha 11 actually did —
    bypasses that entry point and gets no provenance file at all unless the
    sbatch script calls it explicitly. Fix: `source cluster/write_provenance.sh`
    then `write_provenance "$OUT_DIR" "<note>"` (one line, fail-soft, JSON out)
    — already wired into `JOB_TEMPLATE.sbatch` and `p3_2d_fleet.sbatch`/
    `p3_2d_rhs2.sbatch`/`venue_transfer.sbatch`. Copy the same line into any new
    bespoke sbatch script; do not assume the template's sample
    `python -m darksiren_emri` call at the bottom is what you're actually running.

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
- **Local evaluate** (dev box, CPU): `uv run python -m darksiren_emri <dir> --evaluate --h_value 0.73 --num_workers N`

### Launching jobs & writing new ones
Full guide with exact commands for every case: **`cluster/LAUNCHING_JOBS.md`**.
- **Mental model:** PROJECT_ROOT (`~/darksiren-emri`, code+catalog) vs RUN_DIR
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
