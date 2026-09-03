# Batch 2 Cluster Ops Record

Session started 2026-09-03. Agent: cluster-ops batch2.

## Step 1: Preflight

```
[REPO]  /home/st/st_us-403333/st_ac147838/darksiren-emri
        branch=fix/p32d-classg-venue-repair head=081b1f28  ahead=0 behind=19  dirty=607 files
        tag 'commission-base' -> b593f021
[VENV]  import-ok numpy 2.4.3 +few
[CATALOG] reduced_galaxy_catalogue.csv: 1.6G cols=8 [OK]; row1 z=0.001733 — z_cmb frame ✓
[WORKSPACE] 'emri' path=/pfs/work9/workspace/scratch/st_ac147838-emri remaining=19 days
[QUEUE] running=0 pending=0 dependency-dead=0
[DATASETS] 76 unregistered dataset dir(s) — WARN (non-blocking)
VERDICT: READY ✓ (WARN: 1 issue(s))
  • 76 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```

Local HEAD: 40509193 (descendant of required 0c4b1dec). Cluster repo behind by 19 commits (at 081b1f28), dirty=607 files (untracked results/ dirs, expected). Proceeding to sync.

## Step 2: Sync cluster repo

`git fetch && git checkout fix/p32d-classg-venue-repair && git pull --ff-only` — fast-forward 081b1f28..40509193 (19 commits), clean.
Cluster HEAD now: `40509193` (matches local).
uv.lock/pyproject.toml unchanged in this range — `uv sync` skipped per instructions.
Untracked file found: `cluster/graph1_c0prime_byteid_postdecouple_gate.sbatch` (Sep 2, pre-existing) — no filename collision with new sbatch files planned below; left in place.

## Step 3: GATE-ACC reporting-only relaunch

Read exec/v-falsifier-ii-classG/ADJUDICATING_READ_RECORD.md §6/§8 (2026-09-02 login-node `--stage
gates` run died: libpython3.13 not found, modules not sourced — an interactive/bare invocation,
not a batch job) and LAUNCH_RECORD.md. Driver: `results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py`
(`--stage gates --arm {bt,bc} --seeds 900101,...,900133 --out-root $WS/p3_2d_fleet_aprime_20260902`).
`--stage gates` takes the full seed set in one call (not per-seed like `--stage fleet`), so this is
a single-node CPU job, not an array — chains arm=bt then arm=bc sequentially.

Wrote `cluster/graph1_gateacc_relaunch.sbatch`, modeled on `cluster/p3_2d_fleet.sbatch` (closest
working comparand for this driver): `cpu_il`, 16 cpus-per-task, `--time=08:00:00`, single task,
`source cluster/modules.sh` + venv activate + `write_provenance`. rsync'd to cluster and submitted.

**Job ID: 6790465**

## Step 4: seed61000 timeout logs fetch (READ only)

Located `run_20260729_seed61000` under `$WS` (`ws_find emri` ->
`/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed61000`, 30G total dir).
Log-type files only (`*.log`,`*.out`,`*.err`,`run_metadata_*.json`): 2194 files, 730M — under the
2GB abort threshold, fetch proceeded. `grep -rl "Skip tally"` over the full dir returned 0 extra
matches (any such text would already be inside the .out files already included).

rsync'd (include-filtered, NOT injections/posteriors) to
`results/campaign51_20260728/realistic_20260729/seed61000/cluster_logs_fetch_20260904/`.
Local result: 2194 files, 728M. md5 manifest written to
`results/campaign51_20260728/realistic_20260729/seed61000/cluster_logs_fetch_20260904_MANIFEST.md5`
(2194 lines, one per file).

## Step 5: R4 comparand job (h=0.73, theta_sites=2.2)

Found `cluster/graph1_headrebaseline_iiib.sbatch` (the registered CoR-P production CLI for the
m-head-rebaseline iiib run, exec/m-head-rebaseline/LAUNCH_RECORD.md). Confirmed flag names via
`uv run python -m darksiren_emri --help`: `--theta_sites {all,2.1,2.2,2.3}`, `--theta_b`,
`--theta_s`, `--catalogue_leg_1d_mass_aware {auto,off,on}`, `--smear_global_selection` (store_true,
default off). Cross-checked against `REGISTERED_RESOLVED_FLAGS` in
`darksiren_emri/validation/correspondence_1d.py:3154` (theta_b=0.0, theta_s=1.0, theta_sites="all"
baseline — confirms the headrebaseline CLI is literally the registered production set).

Wrote `cluster/graph1_r4_comparand_sites22.sbatch`: single h=0.73 task (not the H_GRID_41 array),
identical CLI to graph1_headrebaseline_iiib.sbatch except `--theta_sites 2.2` (was `all`) and added
explicit `--catalogue_leg_1d_mass_aware off` (production default is `auto`, which resolves `on`
post-flip, so this is a real override, not a no-op). `--theta_b 0.0 --theta_s 1.0` and no
`--smear_global_selection` already matched the task's requirement in the baseline. Same resources
(cpu_il, 16 cpus, 00:45:00) and dataset pins (CRB md5 9a1f2a14384a9281c97ca3be312ddaab, catalogue
md5 c52c13b5cab61f6b3f04bbe202550969) as one headrebaseline task.

Submitted with `RUN_DIR=$WS/graph1_r4_comparand_sites22_20260904`.

**Job ID: 6790708**

Outputs NOT interpreted per instructions.

## Step 5 (continued): R4 comparand retrieval

Job 6790708 COMPLETED, exit 0, elapsed 00:06:31. .err content is matplotlib font-subsetting noise
(PDF generation), not an error.

rsync -aL of `$WS/graph1_r4_comparand_sites22_20260904/` to
`results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/r4_comparand_sites22_20260904/`.
First pass (excluding only `injections`) pulled in 1.9G because `-aL` dereferenced the job's
`cwd/darksiren_emri` symlink (points at `$PROJECT_ROOT/darksiren_emri`, including the 1.6G
`reduced_galaxy_catalogue.csv`) and a duplicate `cwd/simulations` symlink — neither is a run
output. Deleted the local `cwd/` subtree (it duplicates the top-level `simulations/` + code, not
part of the run's own output) — final retrieved set: 136M, 12 files + 1 symlinked-CRB
duplicate-content file (`prepared_cramer_rao_bounds.csv`, md5 matches the registered CRB pin
9a1f2a14384a9281c97ca3be312ddaab, confirming the correct dataset was used).

md5 manifest (13 files incl. `cramer_rao_bounds.csv`/`prepared_cramer_rao_bounds.csv`) written to
`results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/r4_comparand_sites22_20260904_MANIFEST.md5`
— **all 13 md5s verified byte-identical against the remote copy.** No output content interpreted.

Files retrieved:
- GIT_COMMIT_AT_RUN.txt, run_metadata.json
- darksiren_emri_20260903_223503_h_0_73.log
- logs/{graph1_r4_comparand_sites22_6790708.out,.err, provenance_6790708_none.json}
- simulations/cramer_rao_bounds.csv, prepared_cramer_rao_bounds.csv
- simulations/diagnostics/event_likelihoods.csv
- simulations/fisher_quality.csv, fisher_quality_diagnostic.pdf
- simulations/posteriors/h_0_73.json, posteriors_with_bh_mass/h_0_73.json

## Step 3 (continued): GATE-ACC job status at report time

Polled job 6790465 for ~10 min (3 checks, 150s apart): still **PENDING** (reason: Priority — normal
fairshare/backfill queueing, cluster/SKILL.md gotcha 13, not an error). No time bound was specified
for this job in the launch task (8h budget, reporting-only), so not blocking further on it here.

## Step 6: Final summary

- **Preflight:** VERDICT: READY ✓ (WARN: 76 unregistered dataset dirs — non-blocking, pre-existing
  backlog, not caused by this session).
- **Cluster HEAD:** synced clean fast-forward to `40509193` (matches local). uv.lock/pyproject.toml
  unchanged in range — `uv sync` skipped per instructions.
- **Job 1 — GATE-ACC relaunch (`cluster/graph1_gateacc_relaunch.sbatch`):** job **6790465**, last
  state **PENDING** (queued, Priority/fairshare — normal). Out-root
  `$WS/p3_2d_fleet_aprime_20260902` (gates_bt.json then gates_bc.json expected on completion).
- **Job 2 — R4 comparand (`cluster/graph1_r4_comparand_sites22.sbatch`):** job **6790708**,
  **COMPLETED**, exit 0:0, elapsed 00:06:31.
- **seed61000 log fetch:** 2194 files, 730M remote / 728M local — under the 2GB abort threshold,
  fetched in full. md5 manifest at
  `cluster_logs_fetch_20260904_MANIFEST.md5`.
- **R4 comparand retrieval:** 13 files, 136M (after excluding the job's own `cwd/darksiren_emri`
  symlink dereference, which is not run output). **All 13 md5s verified identical to the remote
  copy**, including the two CRB-pin symlinked files.
- **Deviations from instructions:**
  1. First retrieval rsync pass (`-aL` with only `injections` excluded) pulled in 1.9G because it
     dereferenced the job's `cwd/darksiren_emri` symlink into the full package + 1.6G galaxy
     catalogue — not run output. Corrected by deleting the local `cwd/` subtree post-transfer;
     final retrieved set is the intended 12 run-output files (+1 CRB-symlink duplicate). No
     re-fetch needed, no data lost.
  2. GATE-ACC job (6790465) had not left PENDING after ~10 min of polling at report time; not
     blocked on further since no completion time bound was specified for it (8h budget,
     reporting-only per F12). Left running for the author/next session to pick up.
  3. No untracked-sbatch collision required moving aside (pre-existing untracked file had a
     different name from both new sbatch scripts).

No science/read interpretation performed on R4 comparand outputs, per instructions. No cluster
deletions or moves performed.
