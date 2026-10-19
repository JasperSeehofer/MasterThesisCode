# S0-B production retrieval + GATE-ACC addendum status + cluster health

**Date:** 2026-09-03 (retrieval agent, mechanical/medium effort — no science read, no interpretation)
**Repo branch at time of run:** `fix/p32d-classg-venue-repair`

---

## Task 1 — S0-B production run (job 6779532, 5/5) retrieval

### Remote source
```
/home/st/st_us-403333/st_ac147838/darksiren-emri/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-s0b-production/s0b_run_20260902
```
(925 MB per `du -sh` on remote.)

### Local target
```
results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/s0b_run_20260902/
```

### Symlinks found and excluded (row #311 lesson)

`ssh bwunicluster "find <remote> -type l -printf '%p -> %l\n'"` found **5 symlinks**, all named
`simulations/injections` under each of the 5 node dirs in `s0a_seed900101/`, all resolving to the
same shared injection pool **outside** the run dir:

```
s0a_seed900101/node_s_plus_iiib_sites2.2_nosmear/simulations/injections
s0a_seed900101/node_s_minus_iiib_sites2.2_nosmear/simulations/injections
s0a_seed900101/node_b_minus_re_iiib_sites2.2_nosmear/simulations/injections
s0a_seed900101/node_truth_iiib_sites2.2_nosmear/simulations/injections
s0a_seed900101/node_b_plus_re_iiib_sites2.2_nosmear/simulations/injections
```
→ all point to:
```
/pfs/data6/home/st/st_us-403333/st_ac147838/darksiren-emri/results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728
```

All 5 were **excluded** from the rsync (pattern `--exclude='**/simulations/injections'`) and from
both md5 manifests (`-not -path '*/simulations/injections/*'` on the remote side; the local copy
never had them since rsync skipped them). This is the shared injection/seed pool called out in row
#311 — it must not be duplicated per node.

### Commands used

```bash
mkdir -p results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/s0b_run_20260902

REMOTE="/home/st/st_us-403333/st_ac147838/darksiren-emri/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-s0b-production/s0b_run_20260902"
LOCAL="results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/s0b_run_20260902/"

rsync -aL --info=stats2,progress0 \
  --exclude='**/simulations/injections' \
  "bwunicluster:${REMOTE}/" "$LOCAL"

# remote md5 (excluding injection symlink targets)
ssh bwunicluster "find '$REMOTE' -type f -not -path '*/simulations/injections/*' -exec md5sum {} +" \
  | sed "s|$REMOTE/||" | sort -k2 > s0b_remote_md5.txt

# local md5
find "$LOCAL" -type f -exec md5sum {} + | sed "s|$LOCAL/||" | sort -k2 > s0b_local_md5.txt

diff s0b_remote_md5.txt s0b_local_md5.txt
```

### Verification result

**MD5 VERDICT: MATCH** — `diff` between remote and local md5 manifests produced no output.

- Remote file count (excluding injection symlinks): **57**
- Local file count: **57**
- rsync summary: `Number of files: 85 (reg: 57, dir: 28)`, `Total transferred file size: 969,571,608 bytes`
- Local `du -sh`: **925M**

### Top-level contents (local, post-retrieval)

```
logs/                                  (5 SLURM task .out + 5 .err files)
provenance_6779532_4.json
provenance_6779535_0.json
provenance_6779536_1.json
provenance_6779537_2.json
provenance_6779538_3.json
s0a_full_output.json
s0a_seed900101/                        (5 node dirs: node_b_minus_re, node_b_plus_re,
                                         node_s_minus, node_s_plus, node_truth
                                         — each _iiib_sites2.2_nosmear — plus es_null_det.csv)
```

Each `node_*` dir contains:
- `selection_tables_h_0_73.json` (189 B)
- `simulations/fisher_quality_diagnostic.pdf`, `fisher_quality.csv`
- `simulations/cramer_rao_bounds.csv`, `prepared_cramer_rao_bounds.csv` (~4.2 MB each)
- `simulations/diagnostics/event_likelihoods.csv` (~456 KB)
- `simulations/posteriors/h_0_73.json` (~48 KB)
- `simulations/posteriors_with_bh_mass/h_0_73.json` (~180–192 MB — dominates the 925 MB total)

### JSON top-level keys (existence/shape only, not interpreted)

- `s0a_full_output.json`: `arm, seeds_requested, jobs, cpu_per_job, wall_s, n_seeds_ok,
  n_seeds_error, errors, theta_sites, smear, config, h_values, score_h, theta_phi_divisor,
  sky_cone_k, catalogue_leg_1d_mass_aware, theta_zwindow, z_window_k, node_dir_suffix,
  per_seed_summary, note`
- `provenance_*.json` (all 5, identical schema): `git_commit, git_branch,
  tree_dirty_file_count, slurm_job_id, slurm_array_job_id, slurm_array_task_id, seed,
  hostname, start_timestamp_utc, command, note`
- `node_*/selection_tables_h_0_73.json` (all 5 nodes, identical schema): `h, beta_G_phi,
  beta_Gbar_phi, sigma_phi, sigma_4d, r_Malm`
- `node_*/simulations/posteriors/h_0_73.json`: per-event numeric-string keys (`"0"`…)
- `node_*/simulations/posteriors_with_bh_mass/h_0_73.json`: `galaxy_likelihoods,
  additional_galaxies_without_bh_mass`, plus per-event numeric-string keys

### SLURM logs

Already present inside the retrieved run dir at `logs/` (the sbatch `#SBATCH --output` for
`graph1_m_s0b_production*.sbatch` writes directly into the run dir's `logs/`, not into a separate
workspace or `$HOME/darksiren-emri/cluster/logs` location — no separate fetch was needed):

```
m_s0b_production_task_0_6779535.out / .err
m_s0b_production_task_1_6779536.out / .err
m_s0b_production_task_2_6779537.out / .err
m_s0b_production_task_3_6779538.out / .err
m_s0b_production_task_4_6779532.out / .err
```

### `sacct -j 6779532 --format=JobID,State,Elapsed,MaxRSS,TotalCPU -X`

```
JobID             State    Elapsed     MaxRSS   TotalCPU
------------ ---------- ---------- ---------- ----------
6779532_0     COMPLETED   00:07:36              00:00:00
6779532_1     COMPLETED   00:07:31              00:00:00
6779532_2     COMPLETED   00:07:31              00:00:00
6779532_3     COMPLETED   00:07:27              00:00:00
6779532_4     COMPLETED   00:07:21              00:00:00
```
(MaxRSS/TotalCPU blank as reported by `sacct -X` for this account's accounting config — not a
retrieval error, just what `-X` returns without `--format` step-level rows.) All 5/5 array tasks
**COMPLETED**, elapsed ~7.2–7.6 minutes each. Confirms the 5/5 COMPLETED status.

---

## Task 2 — GATE-ACC addendum status (`p3_2d_fleet_aprime_20260902`)

Workspace: `WS=/pfs/work9/workspace/scratch/st_ac147838-emri`
Target dir: `$WS/p3_2d_fleet_aprime_20260902/`

### File check

| File | Status |
|---|---|
| `gates_bt.json` | **MISSING** |
| `gates_bc.json` | **MISSING** |
| `gates_33seed.DONE` | **MISSING** |
| `gates_bt_33seed.log` | exists, 376 B, last modified Sep 2 14:30 |
| `gates_bc_33seed.log` | exists, 196 B, last modified Sep 2 13:52 |

Since **both JSONs do not exist**, per task instructions **no rsync/retrieval was performed** for
this stage (the "if both JSONs exist" condition was not met).

### Full log contents (verbatim, not interpreted)

`gates_bt_33seed.log`:
```
Injection pool spans 2 code revisions (a9f29e82, f6449051) — legitimate for straggler resubmits after a non-physics fix, but verify none of them changed SNR semantics.
Event -1: 171 host(s) have f_k == 0 across the whole host-z window (empty/ZoA pixel) — host-z kernel falls back to the w_pop-only form for those hosts (further occurrences in this worker are suppressed).
```

`gates_bc_33seed.log`:
```
/pfs/data6/home/st/st_us-403333/st_ac147838/darksiren-emri/.venv/bin/python3: error while loading shared libraries: libpython3.13.so.1.0: cannot open shared object file: No such file or directory
```

### Still-running check

- Current login node (`uc3n991`): `ps aux | grep -i 'p3_2d_fleet\|gates_b' | grep -v grep` →
  **no matching process**.
- Attempted `ssh -o BatchMode=yes` from the login node to `uc3n990`, `uc3n991`, `uc3n992`,
  `uc3n993` to check other login nodes: `uc3n990`, `uc3n991` (self), `uc3n993` all returned
  `Permission denied (publickey,keyboard-interactive)`; `uc3n992` returned `No route to host`.
  **Inner cross-node ssh is not permitted for this account** — only the current login node
  (`uc3n991`) could be checked, and no relevant process is running there.

No interpretation offered beyond the above facts; the library-load error in `gates_bc_33seed.log`
and the two logs' small size/stale mtimes are reported verbatim for the author/chair to assess.

---

## Task 3 — Cluster health (verbatim)

### `squeue -u $USER`
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
```
(empty — no jobs currently queued or running under this account)

### `ws_list`
```
id: emri
     workspace directory  : /pfs/work9/workspace/scratch/st_ac147838-emri
     remaining time       : 19 days 22 hours
     creation time        : Sat Jul 25 18:19:41 2026
     expiration date      : Wed Sep 23 18:19:41 2026
     filesystem name      : pfs7wor9
     available extensions : 0
```

### `df -h $WS`
```
Filesystem                                                                     Size  Used Avail Use% Mounted on
...(pfs7wor9 lustre mount, multi-OST path)...:/pfs7wor9                       4.6P  2.7P  1.7P  62% /pfs/work9
```

---

## Summary for the chair

- **S0-B (job 6779532) retrieval: DONE, MD5 MATCH, 57/57 files, 925 MB local, sacct confirms 5/5
  COMPLETED.** 5 shared-injection-pool symlinks correctly excluded (row #311 lesson applied).
- **GATE-ACC addendum (`p3_2d_fleet_aprime_20260902`): NOT complete** — `gates_bt.json`,
  `gates_bc.json`, and the `gates_33seed.DONE` sentinel are all absent; only two small, stale logs
  exist (last touched Sep 2, ~13:52/14:30). `gates_bc_33seed.log` shows a `libpython3.13.so.1.0`
  load failure (consistent with cluster gotcha 1 — module/venv not sourced). No process is running
  on the reachable login node; other login nodes could not be checked (cross-node ssh denied). No
  retrieval was performed since neither JSON exists.
- **Cluster health:** queue empty, workspace `emri` has 19d22h remaining (expires 2026-09-23, 0
  extensions left), `/pfs/work9` at 62% used (1.7P available).
