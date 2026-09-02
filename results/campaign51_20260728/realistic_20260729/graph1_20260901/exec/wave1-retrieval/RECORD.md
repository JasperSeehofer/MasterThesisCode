# Wave 1 retrieval — RECORD

Research Graph 1, wave 1. Retrieval only — no reads, no interpretation of posterior/score content.
This record is the archive-of-record manifest + transfer verification for the four completed
SLURM jobs (6764460 C0-prime x2, 6764461 iiib HEAD x41, 6764462 joint_r1 HEAD x41, 6764463 T5 Arm S
x16), launched per:
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-head-rebaseline/LAUNCH_RECORD.md`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-t5-armS/LAUNCH_RECORD.md`

`$WS` = `/pfs/work9/workspace/scratch/st_ac147838-emri`.

## 1. sacct verdict (verbatim)

```
JobID|JobName|State|ExitCode
6764460_0|graph1-c0prime-headrebaseline|COMPLETED|0:0
6764460_0.batch|batch|COMPLETED|0:0
6764460_0.extern|extern|COMPLETED|0:0
6764460_1|graph1-c0prime-headrebaseline|COMPLETED|0:0
6764460_1.batch|batch|COMPLETED|0:0
6764460_1.extern|extern|COMPLETED|0:0
6764461_{0..40}|graph1-headrebaseline-iiib|COMPLETED|0:0            (41 tasks, .batch/.extern all COMPLETED 0:0)
6764462_{0..40}|graph1-headrebaseline-joint_r1|COMPLETED|0:0        (41 tasks, .batch/.extern all COMPLETED 0:0)
6764463_{0..15}|graph1-t5-armS-iiib|COMPLETED|0:0                   (16 tasks, .batch/.extern all COMPLETED 0:0)
```

Aggregate check: `sacct -j 6764460,6764461,6764462,6764463 --format=JobID,State,ExitCode
--parsable2 --noheader | awk -F'|' '{print $2, $3}' | sort | uniq -c` →

```
    300 COMPLETED 0:0
```

300 = every job/step/batch/extern record across all four jobs. Zero non-`COMPLETED 0:0` rows.
No exceptions to report.

## 2. Run-dir existence, file counts, sizes

Three-valued existence check performed before any transfer — all five run dirs found EXISTS
(not ABSENT, not unreachable) on `$WS` at submission-verified paths:

| Run dir | Remote `du -sLh` (dedup, symlink-aware) | Remote file count (`find -L -type f`) | 10 GB gate |
|---|---|---|---|
| `run_20260902_graph1_c0prime_headrebaseline_iiib` | 1.8G | 1545 | under — proceed |
| `run_20260902_graph1_c0prime_headrebaseline_joint_r1` | 1.8G | 1549 | under — proceed |
| `run_20260902_graph1_headrebaseline_iiib` | 6.8G | 1945 | under — proceed |
| `run_20260902_graph1_headrebaseline_joint_r1` | 6.5G | 1949 | under — proceed |
| `run_20260902_graph1_t5_armS_iiib` | 3.2G | 6300 | under — proceed |

No run dir exceeded 10 GB; no STOP was triggered; all five were transferred.

**Note on local vs. remote size (not an error):** each run dir's top level contains a
`cwd/simulations` path that is a symlink back into the same physical `simulations/` tree
(the standard `ln -sfn $RUN_DIR/simulations $PROJECT_ROOT/simulations` pipeline convention).
`du -sLh` on the cluster deduplicates by device+inode when it dereferences, so the table above
undercounts unique-content-times-two directories. `rsync -aL` (used per task instruction —
"dereference symlinks, standing gotcha") does **not** dedup: it materializes the symlinked
subtree as an independent copy, so the local retrieved directories are correspondingly larger.
This was anticipated, verified against the remote file count (which also does not dedup, via
`find -L -type f`), and matches exactly — see §3. It is not data loss, drift, or a transfer defect.

## 3. Local retrieval — file counts and sizes (post-transfer)

| Run dir | Local file count | Matches remote manifest line count | Local `du -sh` |
|---|---|---|---|
| `run_20260902_graph1_c0prime_headrebaseline_iiib` | 1545 | yes (1545) | 2.0G |
| `run_20260902_graph1_c0prime_headrebaseline_joint_r1` | 1549 | yes (1549) | 1.9G |
| `run_20260902_graph1_headrebaseline_iiib` | 1945 | yes (1945) | 12G |
| `run_20260902_graph1_headrebaseline_joint_r1` | 1949 | yes (1949) | 12G |
| `run_20260902_graph1_t5_armS_iiib` | 6300 | yes (6300) | 9.7G |

Total: 13,288 files, ~37.6 GB local (raw sum of `du -sh`; matches the deref'd-manifest world, not
the dedup'd remote-`du` world in §2).

Local retrieval root:
`results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/<run_dir_basename>/`

C0-prime eval copies already present under
`exec/m-head-rebaseline/c0prime_eval/` (a partial task-21-equivalent extract) were left untouched;
the full run dirs above are the archive of record and were retrieved regardless, per task
instruction.

## 4. Manifest build + verification method

- **Manifests built on the cluster, before transfer**, one per run dir:
  `(cd $WS/<dir> && find -L . -type f -print0 | sort -z | xargs -0 md5sum) >
  $WS/graph1_wave1_retrieval_manifests/<dir>.md5`
  (`-L`: dereference symlinks, matching the transfer mode.)
- Manifests copied to
  `exec/wave1-retrieval/manifests/<dir>.md5` (verbatim, unmodified).
- Manifest line counts (cluster-side, confirmed post-completion — an earlier read of the largest
  manifest mid-write showed a transient partial count due to shell redirection creating the target
  file before population; re-checked after the ssh session's own completion and it matched the
  fresh `find -L` count exactly):

  | Manifest | Lines |
  |---|---|
  | `run_20260902_graph1_c0prime_headrebaseline_iiib.md5` | 1545 |
  | `run_20260902_graph1_c0prime_headrebaseline_joint_r1.md5` | 1549 |
  | `run_20260902_graph1_headrebaseline_iiib.md5` | 1945 |
  | `run_20260902_graph1_headrebaseline_joint_r1.md5` | 1949 |
  | `run_20260902_graph1_t5_armS_iiib.md5` | 6300 |

- **Transfer:** `rsync -aL --stats bwunicluster:$WS/<dir>/ <local_retrieved>/<dir>/` for each of the
  five dirs. (Two transfer attempts were interrupted mid-run by the harness's background-task
  lifetime cap, not by any network/data fault; each resumed rsync run is idempotent and simply
  completed the remaining files — confirmed by the final file counts in §3 matching the manifests
  exactly.)
- **Verification, after transfer, locally:** `(cd <local_retrieved>/<dir> && md5sum -c
  <manifest>.md5 --quiet)` for each of the five dirs.

## 5. Verification results

```
=== verifying run_20260902_graph1_c0prime_headrebaseline_iiib ===
exit=0
=== verifying run_20260902_graph1_c0prime_headrebaseline_joint_r1 ===
exit=0
=== verifying run_20260902_graph1_headrebaseline_iiib ===
exit=0
=== verifying run_20260902_graph1_headrebaseline_joint_r1 ===
exit=0
=== verifying run_20260902_graph1_t5_armS_iiib ===
exit=0
```

`md5sum -c --quiet` prints nothing and exits 0 only when every checksum in the manifest matches
and every listed file was found — silence + `exit=0` on all five dirs confirms **0 mismatches, 0
missing files, across 13,288 checksummed files**.

## Verdict

All four SLURM jobs (300/300 job+step records) COMPLETED 0:0. All five run dirs retrieved
verbatim (rsync -aL) to
`results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/`. Manifest-verified
locally with 0 mismatches. No run dir exceeded the 10 GB pre-transfer gate. Retrieval is the
archive of record for wave-1 banking; no science content was read or interpreted as part of this
task.
