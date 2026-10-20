# OPS_RECORD_hbounds.md — S0-B truth-node h-bounds=0.60,0.86 test

Session: single cluster-ops agent, all access via `cluster/agent_ssh.sh` (ControlMaster discipline).
Date: 2026-09-04.

## 1. Repo sync

- Tracked collision `cluster/graph1_sealed_m1_headstack.sbatch` (locally modified) and untracked
  `cluster/graph1_c0prime_byteid_postdecouple_gate.sbatch` / `.pre_guard_correction.bak` moved
  into `cluster/_pull_collisions_20260904/` before pulling (pattern from prior sessions).
- `git fetch && git pull --ff-only` on `fix/p32d-classg-venue-repair`: fast-forwarded
  `8f933e7b` → `d9e50179`. Confirmed `git rev-parse HEAD` = `d9e501790fdc2f451664d6c3027f0ed774d35ab8`
  (short `d9e50179`) on the cluster.

## 2. PIN_COMMIT edit + submit

- `sed -i 's/PIN_COMMIT/d9e50179/' cluster/graph1_s0b_truth_hbounds060.sbatch` run BOTH locally
  and on the cluster (post-pull). md5 of the edited file matches on both sides:
  `9d38c63638a840456ef8d27f448648db`.
- Local edit is **uncommitted** (per task instructions — commit not required for the test).
- Out-root note / deviation: the sbatch as committed at `d9e50179` hardcodes
  `OUT_ROOT="$PROJECT_ROOT/.../graph1_20260901/exec/d-photoz-leverage/graph1_s0b_truth_hbounds060_20260904"`
  (repo-relative, matching the header comment and the `m-s0b-production` sibling's convention),
  **not** a `$WS`-rooted path — there is no `OUT_ROOT` env override in the script. Per task
  instructions I additionally created `$WS/graph1_s0b_truth_hbounds060_20260904`
  (`/pfs/work9/workspace/scratch/st_ac147838-emri/graph1_s0b_truth_hbounds060_20260904`), but the
  job as submitted writes to the repo-relative path, not that workspace dir — I did not edit the
  committed sbatch's `OUT_ROOT` (out of scope: would be a script-behavior change beyond the
  PIN_COMMIT edit authorized by this task). Flagging for author/orchestrator awareness.
- Submitted: **job 6794615**, `--array=0-0`, single truth node, `--h-bounds 0.60,0.86`.

## 3. Poll — STOPPED at budget (3 calls, ~9 min each)

All three `poll 6794615 60` calls (~27 min elapsed) returned `PENDING 00:00:00` — the job has not
started running yet (fairshare floor, gotcha 13: our account sits at the bottom of the priority
ranking, so short array jobs queue behind backfill). Task budget (`≤ 9 min per call, ≤ 3 calls`)
is exhausted. **Steps 3–5 (retrieve, diff vs R4/R4b, conditional production submission) are NOT
DONE** — job 6794615 has not produced output yet.

## 4. sacct — other jobs (task item 6)

```
JobID          JobName                     State      Elapsed   ExitCode
6790465        graph1-gateacc-relaunch     PENDING    00:00:00  0:0
6794421_0..12  graph1-sealed-m1-headstack  COMPLETED  ~00:05:xx 0:0   (13 of ≥15 tasks done)
6794421_13     graph1-sealed-m1-headstack  RUNNING    00:01:19  0:0
6794421_[14+   graph1-sealed-m1-headstack  PENDING    00:00:00  0:0
```

- **6794421** (sealed m1): array is mid-run — 13 tasks COMPLETED cleanly (exit 0:0, ~5 min each),
  task 13 RUNNING, task(s) 14+ still PENDING. No failures so far.
- **6790465** (GATE-ACC relaunch): still PENDING, 00:00:00 elapsed — has not started.

## 5. Next action (not taken this session — budget exhausted)

Re-poll job 6794615 (and 6790465, 6794421 tail) in a fresh cluster-ops turn once queue movement is
expected; then proceed with retrieval (tar-through-wrapper test, fallback scp), the R4/R4b column
diff, and the conditional `graph1_s0b_production_window.sbatch` build+submit per the original task
spec.

## 6. Status-check readout (chair, 2026-09-04, single foreground sacct — no wait)

One additional `sacct -j 6794615,6790465,6794421 -X` snapshot (no poll/sleep loop):

```
6790465        graph1-gateacc-relaunch     PENDING    00:00:00  0:0
6794421_0..12  graph1-sealed-m1-headstack  COMPLETED  ~00:05:xx 0:0
6794421_13     graph1-sealed-m1-headstack  RUNNING    00:01:55  0:0
6794421_[14+   graph1-sealed-m1-headstack  PENDING    00:00:00  0:0
6794615_[0]    graph1-s0b-hb060            PENDING    00:00:00  0:0
```

- **6794615** (truth-node h-bounds=0.60,0.86 test): still PENDING, has not started. NOT retrieved,
  NOT md5-verified. R4b/R4 diff **NOT DONE** — no output exists yet to diff. Production-window
  job **NOT SUBMITTED** (gated on the diff verdict, which is unavailable).
- **6794421** (sealed m1): unchanged from §4 — 13 COMPLETED, task 13 RUNNING (00:01:55), rest
  PENDING, no failures.
- **6790465** (GATE-ACC relaunch): unchanged — PENDING, 0 elapsed.

Session stops here in the foreground per instruction (never end a turn waiting on a job).
