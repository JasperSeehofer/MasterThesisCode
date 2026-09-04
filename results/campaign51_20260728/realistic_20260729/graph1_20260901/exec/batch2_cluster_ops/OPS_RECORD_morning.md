# Morning batch cluster-ops record — 2026-09-04

One cluster-ops agent, batch 2 morning tasks. All cluster access via
`cluster/agent_ssh.sh run/poll` per the ControlMaster discipline
(SKILL.md, ledger row #357/#359). Never closed the master; no parallel
ssh fan-out; no destructive cluster ops.

## 1. sacct — 6790794 (S0-C), 6790859 (sealed m1), 6790465 (GATE-ACC)

```
JobID                             JobName      State    Elapsed ExitCode
------------ ---------------------------- ---------- ---------- --------
6790465           graph1-gateacc-relaunch    PENDING   00:00:00      0:0
6790794_0..9              graph1-s0c-hgrid  COMPLETED   ~00:07-08:xx   0:0   (10/10)
6790859_0..40    graph1-sealed-m1-headstack     FAILED   00:00:11-04:03   1:0   (41/41)
```

Full table captured verbatim in the session transcript; all 10 S0-C tasks COMPLETED,
all 41 sealed-m1 tasks FAILED at exit 1:0, GATE-ACC (6790465) PENDING throughout this
session (never observed running).

## 2. Sealed m1 (6790859) — STOP cause

Task 0 err (`graph1_sealed_m1_iiib_task0_6790869.err`, under
`$WS/graph1_sealed_m1_iiib_20260904/logs/`):

```
STOP: the 0.67 run's own injections link resolves to /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed64000_h0p67/simulations/injections, not the canonical pool /pfs/work9/workspace/scratch/st_ac147838-emri/injection_pool_mix200k_20260728 -- pool mismatch, report
```

Root cause: the sbatch guard (`cluster/graph1_sealed_m1_headstack.sbatch`) does
`readlink -f "$CRB_SRC/simulations/injections"` on the 0.67 run's own
`simulations/injections` path and requires it to resolve to the canonical pool
`injection_pool_mix200k_20260728`. Checked directly:
`/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed64000_h0p67/simulations/injections`
is a **real directory** (`drwxr-xr-x`, not a symlink), so `readlink -f` returns the
path unchanged — it can never equal the canonical-pool path regardless of which pool
was actually used to build that run's CRB. The 0.67 run predates the mix200k pool
convention (injections materialized in place, not linked), so the guard's
"link resolves to canonical pool" check is structurally unsatisfiable for this input —
not a data mismatch, a guard precondition violated by an older run's directory layout.
This matches every task's ~11-15s failure (fast STOP, before any compute); the ~2-4 min
outliers (tasks 15-24) are queue/module-load variance, same STOP line.

**Not resubmitted** per instruction — reporting only.

## 3. S0-C (6790794) retrieval

All 10/10 tasks COMPLETED. Retrieved both per-h out-roots with
`rsync -aL --exclude='injections'` (excludes the `simulations/injections` symlinks,
which all resolve outside the run dir to
`.../gate_b_20260730/injection_pool_mix200k_20260728`, already local in-repo) to
`results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/s0c_hgrid_20260904/`:

- `h_0p665/`: 62 files, 965,654,700 bytes
- `h_0p780/`: 62 files, 972,678,895 bytes
- Total: 124 files

**MD5 verdict: MATCH — all 124 files byte-identical** (remote `find -type f
-not -path '*/injections/*' | md5sum` vs local, sorted by relative path, `diff` clean;
remote and local file counts both 124).

## 4. GATE-ACC (6790465)

State: **PENDING** throughout the session (queued, never observed RUNNING). Not
retrieved — nothing to retrieve yet. No `gates_bt.json`/`gates_bc.json`/
`gates_33seed.DONE`/`gates_{bt,bc}_33seed.log` fetch attempted.

## 5. R4b (graph1_r4b_comparand_sites22_2doff)

- Cluster repo HEAD confirmed ancestor of 40509193 before touching anything
  (`git merge-base --is-ancestor 40509193 HEAD` → OK, cluster was at `06a12422`).
- Cluster repo was behind local HEAD `a898f464`; `git pull --ff-only` blocked by one
  locally-modified tracked file (`cluster/graph1_sealed_m1_headstack.sbatch`) and three
  untracked collisions (`cluster/graph1_gateacc_relaunch.sbatch`,
  `cluster/graph1_r4_comparand_sites22.sbatch`,
  `cluster/graph1_sealed_m1_headstack.sbatch.bak_local20260904`). Moved all four aside
  into `cluster/_pull_collisions_20260904/` (not deleted), then `git pull --ff-only`
  fast-forwarded cleanly `06a12422..8f933e7b` (72 files, brought in
  `cluster/graph1_r4b_comparand_sites22_2doff.sbatch` itself via the same pull —
  diffed byte-identical against the local copy in this repo, so no separate rsync of
  the sbatch was needed).
- Created `$WS/graph1_r4b_comparand_sites22_2doff_20260904`.
- Submitted: **job 6794207** (`sbatch --export=ALL,RUN_DIR=... cluster/graph1_r4b_comparand_sites22_2doff.sbatch`).
- Polled via `cluster/agent_ssh.sh poll 6794207 60`; RUNNING through at least 00:03:11
  elapsed at last poll snapshot in this record (see live poll for terminal state/retrieval —
  update below once terminal).

<!-- R4b terminal state + retrieval appended below once the poll completes -->

**Terminal state:** job 6794207 COMPLETED, 00:05:53 elapsed, ExitCode 0:0.

**Retrieval:** rsync -aL, excluding `injections`, `cramer_rao_bounds.csv`,
`prepared_cramer_rao_bounds.csv` (all three symlinks resolving outside the run dir, to
`run_20260729_seed61000`) and `cwd/` (the `cwd/darksiren_emri` code link plus a
`cwd/simulations` link back inside the run dir that would double-copy real output),
to `results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/r4b_comparand_sites22_2doff_20260904/`:
11 regular files, 133,198,845 bytes. **MD5 verdict: MATCH — all 11 files byte-identical**
(remote vs local, sorted by relative path, `diff` clean).

**R4b-vs-S0-B-truth diff** (`simulations/diagnostics/event_likelihoods.csv`, h=0.73 rows,
joined on `event_idx`, 1588/1588 rows matched; S0-B truth =
`retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/.../event_likelihoods.csv`):

| column | max_abs_diff | max_rel_diff | n(rel>1e-9) |
|---|---|---|---|
| combined_no_bh | 2.105473e-03 | 4.232196e-01 | 896 |
| combined_with_bh | 1.089342e-03 | 6.227209e-01 | 735 |
| L_cat_no_bh | 1.300574e-02 | 1.000000e+00 | 1083 |
| L_cat_with_bh | 1.757943e-02 | 1.000000e+00 | 1081 |
| B_num | 1.415610e-07 | 1.775069e-14 | 0 |
| B_num_wbh | 5.587935e-08 | 1.784888e-14 | 0 |
| D_tilde_phi | 0.0 | 0.0 | 0 |
| alpha_G_phi | 0.0 | 0.0 | 0 |
| den_log_term | 0.0 | 0.0 | 0 |
| num_log_term_no_bh | 5.502937e-01 | 3.581346e-02 | 784 |
| num_log_term_with_bh | 9.747701e-01 | 6.806873e-02 | 594 |

**Verdict: NOT byte-identical.** `D_tilde_phi`, `alpha_G_phi`, `den_log_term` are exact
(0 diff), and `B_num`/`B_num_wbh` match to float noise only (~1e-14 relative, 0 rows over
the 1e-9 threshold) — the non-catalogue legs are unaffected. The catalogue-leg columns
(`L_cat_no_bh`, `L_cat_with_bh`, `num_log_term_{no_bh,with_bh}`, and the downstream
`combined_{no_bh,with_bh}`) diverge on the majority of rows (594-1083 of 1588 rows over
the 1e-9 threshold, up to relative 1.0 on `L_cat_*`) — consistent with R4b's driver-pinned
deviation (`catalogue_numerator_survival_2d=off`, plus the other R4b legs) actually
changing the catalogue numerator computation relative to S0-B truth; the discriminating
test isolates the catalogue leg as the source of the divergence.

## 6. Pool build logs (r-timeout-selection §8 item A)

`REGISTRATION_DRAFT.md` §8 item A confirmed (approved [DO]): fetch
`injection_pool_mix200k_20260728` build logs. The pool directory itself
(`$WS/injection_pool_mix200k_20260728`, 707 CSV files) carries no logs — per
`docs/campaign_redesign_51_design.md:124` ("Delivered pool ... stack `a9f29e8`/
`f644905`"), the pool is a stack of 7 dated `injection_20260728-*` campaign run dirs
(seeds 51000, 51100, 53000, 54000, 54100, 55000, 56000), each with its own `logs/`
and `run_metadata_*.json`.

Fetched log-type files only (`*.out`, `*.err`, `*.log`, `run_metadata_*.json`;
directory structure preserved) from all 7 source dirs to
`results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728_buildlogs_fetch_20260904/`:

- 1070 `.err` + 1070 `.out` + 1070 `run_metadata_*.json` + 299 `master_thesis_code_*.log` = 3509 files
- Total size: **41 MB** (well under the 2 GB abort threshold)
- `MANIFEST.md5` written alongside (3510 lines incl. itself... excluded from own hash — see file)

## Deviations from plan

- GATE-ACC never left PENDING — task 3 deliverable (gates_*.json fetch) not done this
  session; flag for the next cluster-ops pass.
- Sealed m1 root-caused but not fixed/resubmitted (out of scope per instruction).
