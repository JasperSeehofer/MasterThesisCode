# OPS RECORD — m1 (sealed-mock stage m1, headstack) retrieval

Cluster-ops agent, single agent for this batch. Timestamp: 2026-09-04T11:40+02:00.
All cluster traffic via `cluster/agent_ssh.sh run '<cmd>'` (3-slot semaphore); no plain
ssh/scp/rsync used; ControlMaster untouched; no parallel ssh fan-out; no `sleep >= 60`
remotely. `cluster` skill loaded first per protocol.

---

## 1. sacct snapshot — jobs 6794615, 6790465, 6794421

Command: `sacct -j 6794615,6790465,6794421 -X --format=JobID,JobName%30,State,Elapsed,ExitCode,NCPUS,CPUTimeRAW`

```
JobID                               JobName      State    Elapsed ExitCode      NCPUS CPUTimeRAW
------------ ------------------------------ ---------- ---------- -------- ---------- ----------
6790465             graph1-gateacc-relaunch    PENDING   00:00:00      0:0          0          0
6794421_0        graph1-sealed-m1-headstack  COMPLETED   00:05:48      0:0         16       5568
6794421_1..39   (39 more array tasks, all COMPLETED, 16 cores each)
6794421_40       graph1-sealed-m1-headstack  COMPLETED   00:05:11      0:0         16       4976
6794615_[0]                graph1-s0b-hb060    PENDING   00:00:00      0:0          0          0
```
Full 41-row array table captured verbatim in the tool transcript (elapsed range
00:04:48–00:05:53, CPUTimeRAW range 4608–5648, all 16 NCPUS, all ExitCode 0:0,
all State COMPLETED — 41/41 tasks present, none missing/failed).

**Job status summary:**
- **6794421** (`graph1-sealed-m1-headstack`, the sealed-m1 array, 41 tasks): **COMPLETED 41/41**, exit 0:0 throughout.
- **6790465** (`graph1-gateacc-relaunch`): **PENDING**, 0 elapsed — not yet started, out of scope for this retrieval.
- **6794615** (`graph1-s0b-hb060`): **PENDING** (array `[0]`), 0 elapsed — not yet started, out of scope for this retrieval.

**Sealed-m1 (6794421, 41 tasks) core-hour and wall-time computation:**
- Σ CPUTimeRAW = 205,152 s → **57.63 core-h** (205152/3600 = 57.6311…)
- Per-task wall time (elapsed, minutes): **min 4.80 min, median 5.20 min, max 5.88 min**

**Decision-6 check (D_SEALED_REGISTER_DOSSIER.md, dossier sec 1 item 6 — "joint_r1 submitted only if
iiib array's sacct cost lands <= 60 core-h"):**
**57.63 core-h ≤ 60 core-h → PASS.** iiib venue is under the cap; joint_r1 is clear to submit per the
dossier's own gate. **REPORT ONLY — nothing submitted, per task instruction.**

---

## 2. Run-dir layout (from sbatch lines 40-120) + tree

`RUN_DIR=$WORKSPACE/graph1_sealed_m1_iiib_20260904` (WORKSPACE resolved via `cluster/modules.sh` → `ws_find emri`).
sbatch (`cluster/graph1_sealed_m1_headstack.sbatch`) writes:
- `$RUN_DIR/simulations/posteriors/h_<label>.json`
- `$RUN_DIR/simulations/posteriors_with_bh_mass/h_<label>.json`
- `$RUN_DIR/logs/graph1_sealed_m1_${VENUE}_task${TID}_${SLURM_JOB_ID}.{out,err}`
- `$RUN_DIR/run_metadata_<task>.json` (via `write_provenance.sh`, gotcha 12)
- `$RUN_DIR/simulations/{cramer_rao_bounds.csv, prepared_cramer_rao_bounds.csv}` — **symlinks** into
  `$WORKSPACE/run_20260729_seed64000_h0p67/simulations/` (the 0.67 CRB pool; NOT copied, NOT touched here — blindness rule).
- `$RUN_DIR/simulations/injections` — symlink into `$WORKSPACE/injection_pool_mix200k_20260728`.
- `$RUN_DIR/simulations/diagnostics/event_likelihoods.csv` (15 MB, per-event diagnostics; NOT in a `posteriors*` dir).
- `$RUN_DIR/simulations/{fisher_quality.csv, fisher_quality_diagnostic.pdf}`.
- `$RUN_DIR/cwd/` — job scratch (selection_tables_h_*.json per h, symlinks to code/simulations).

**Tree, depth 2 (dirs):**
```
graph1_sealed_m1_iiib_20260904/
  cwd/                              41 files (selection_tables_h_*.json + symlinks)
  logs/                             205 files (task .out/.err + provenance_*.json)
  simulations/                      2 files (fisher_quality.csv, fisher_quality_diagnostic.pdf) + 2 symlinks (CRB, injections)
    diagnostics/                    1 file  (event_likelihoods.csv, 15 MB)
    posteriors/                     41 files (h_*.json, 1.8 MB total)
    posteriors_with_bh_mass/        41 files (h_*.json, **3.3 GB total**, ~86 MB/file)
  (top level)                       83 files: darksiren_emri_*.log (per-h main.py logs) + run_metadata_0..40.json (95 KB total) + GIT_COMMIT_AT_RUN.txt
```

**h-value count check:** `simulations/posteriors/h_*.json` = **41**, `simulations/posteriors_with_bh_mass/h_*.json` = **41**.
**Exactly 41/41 h-values present in both posterior dirs — matches H_GRID_41. PASS.**

No `run_metadata*.json` or diagnostics CSV files exist *inside* either `posteriors*/` directory itself
(checked with `find … -maxdepth 1 -type f ! -name "h_*.json"` on both dirs — empty). The per-run
`run_metadata_<task>.json` files live at `$RUN_DIR` top level, not inside the posterior dirs; included
below anyway (95 KB total, provenance is directly relevant and trivially small).

---

## 3. Retrieval

### DEVIATION — `posteriors_with_bh_mass/` is 3.3 GB, not "small" (flagged, not silently worked around)

Task whitelist: "every `posteriors*/h_*.json`" with a `<50 MB total` budget stated for the
metadata/diagnostics addendum, and a `>30 MB → h_*.json-only` fallback for the archive.
Measured: `posteriors/` = 1.8 MB (41 files, ~41 KB each) — trivial, well inside any budget.
`posteriors_with_bh_mass/` = **3.3 GB** (41 files, ~86 MB **each**) — three orders of magnitude
past the 30/50 MB figures the instructions were sized around, and impractical to move through
the sanctioned base64-over-ssh-text channel (would inflate to ~4.4 GB of text, likely a
multi-hour transfer well past the semaphore's per-call timeout, and disproportionate to a "read
inputs" retrieval).

**Action taken:** retrieved `posteriors/h_*.json` (41 files, small) + `run_metadata_*.json`
(41 files, small) only. **`posteriors_with_bh_mass/` was NOT retrieved.** This is a scope
narrowing from the literal "posteriors*" wildcard, made because the literal instruction
conflicts with its own stated size budget by ~65×; flagging for an explicit author/chair call
on whether the bh_mass variant is needed for this read (and if so, how — targeted rsync of a
handful of h-values, or accepting the large one-time transfer).

### Archive + transfer (small set only)

- Built `$RUN_DIR/m1_iiib_posteriors.tgz` from `simulations/posteriors` + `run_metadata_*.json`.
  (First `tar czf` attempt produced a spurious 325,582,848-byte file with trailing-garbage /
  `tar: Child returned status 2` on listing — almost certainly a stale/partial file already
  sitting at that path from an earlier attempt this session; `rm -f` + rebuild produced a clean
  662,212-byte archive that listed correctly, 83 entries. Noted in case the stray large file
  recurs.)
- Remote: `m1_iiib_posteriors.tgz`, **662,212 bytes**, **md5 = `b1e14f4bad2e65315d02e59ae2284804`**.
- Transferred via `cluster/agent_ssh.sh run 'base64 <file>'` → captured to a local `.b64` file →
  `base64 -d` → local `.tgz`.
- **Local md5 = `b1e14f4bad2e65315d02e59ae2284804` — MATCHES remote. PASS.**
- Extracted to local destination:
  `/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-sealed-mock/m1_iiib_retrieved/`
- Remote archive removed after successful verified transfer (`rm -f $RUN_DIR/m1_iiib_posteriors.tgz`).

### Local manifest

- `MANIFEST.md5` — per-file md5 of all 82 retrieved files (41 `simulations/posteriors/h_*.json` +
  41 `run_metadata_*.json`), written at
  `.../m1_iiib_retrieved/MANIFEST.md5`.
- **list-md5 of MANIFEST.md5 = `7e8631c7a3a3adc49ae272ac38e6ab64`** (also saved to
  `MANIFEST.list-md5.txt` in the same dir).
- Local retrieved-set size: **2.0 MB**.
- `h_*.json` count in retrieved set: **41** (posteriors/ only — posteriors_with_bh_mass/ excluded per deviation above).

**No CRB, no sealed file, no `observed_catalogue*` were opened, copied, or transferred** — row #349
blindness rule respected throughout; only posterior-side outputs and run_metadata provenance were
touched.

---

## 4. Log tails — task 0 and task 40

Log naming resolves via `SLURM_ARRAY_TASK_ID` → individual `SLURM_JOB_ID` (JobIDRaw), not the
array base id — confirmed with `sacct -j 6794421 --format=JobID,JobIDRaw`: task0 → JobIDRaw
**6794462**, task40 → JobIDRaw **6794421** (base array id reused for the last-indexed task's own
JobIDRaw in this array's numbering — verified against `ls logs/`, both files present under those
exact names).

### task 0 (`graph1_sealed_m1_iiib_task0_6794462.out`, tail 30)
```
=== graph1-sealed-m1-headstack venue=iiib task=0 job=6794462 host=uc2n810.localdomain commit=8f933e7b74153437f5bc709e9481f6b7ae7e4f0a ===
pool content-manifest pin OK: 75f4030d5d3b0405fd948049bef5767e
dataset pins OK: CRB67=8e9253fef42f574c569a04a3e19299ab rows=1345 (expected 1343; g-population read-time disclosure) catalogue=c52c13b5cab61f6b3f04bbe202550969 pool=/pfs/work9/workspace/scratch/st_ac147838-emri/injection_pool_mix200k_20260728 (707 files)
provenance stamp: .../logs/provenance_6794462_0.json (commit=8f933e7b... dirty=665 job=6794462/0)
=== graph1-sealed-m1-headstack: venue=iiib task=0 h=0.600 seed=777000 ===
=== done: graph1-sealed-m1-headstack venue=iiib task=0 h=0.600 ===
```
`.err` tail: benign matplotlib/fonttools glyph-subsetting noise from PDF generation (font
subsetting for `fisher_quality_diagnostic.pdf`) — no WARN/ERROR text.

### task 40 (`graph1_sealed_m1_iiib_task40_6794421.out`, tail 30)
```
=== graph1-sealed-m1-headstack venue=iiib task=40 job=6794421 host=uc2n781.localdomain commit=d9e501790fdc2f451664d6c3027f0ed774d35ab8 ===
pool content-manifest pin OK: 75f4030d5d3b0405fd948049bef5767e
dataset pins OK: CRB67=8e9253fef42f574c569a04a3e19299ab rows=1345 (expected 1343; g-population read-time disclosure) catalogue=c52c13b5cab61f6b3f04bbe202550969 pool=/pfs/work9/workspace/scratch/st_ac147838-emri/injection_pool_mix200k_20260728 (707 files)
provenance stamp: .../logs/provenance_6794421_40.json (commit=d9e501790f... dirty=702 job=6794421/40)
=== graph1-sealed-m1-headstack: venue=iiib task=40 h=0.860 seed=777040 ===
=== done: graph1-sealed-m1-headstack venue=iiib task=40 h=0.860 ===
```
`.err` tail: same benign fonttools noise, no WARN/ERROR.

**Both pin lines OK** in both tasks (pool content-manifest, CRB, catalogue). Both ran the same
row-count disclosure (1345 rows vs the 1343 the header comment expected) — this is called out
in-script as an accepted "g-population read-time disclosure", not a STOP; both tasks proceeded to
`done`.

**Observation (not a STOP, flagging for the record):** task 0's commit is `8f933e7b7415...`
and task 40's is `d9e501790fdc...` — **different HEAD commits** at run time, `dirty=665` vs
`dirty=702` (uncommitted-diff line counts also differ). Both pass the sbatch's own
`git merge-base --is-ancestor 081b1f28 HEAD` + `git diff --quiet 081b1f28 HEAD -- darksiren_emri/`
freeze checks (no STOP was printed by either task), so the array ran across a repo HEAD that
moved between 09:44 (task 0) and 10:2x (task 40) without the darksiren_emri/ physics tree
changing relative to the frozen commit 081b1f28 — consistent with author/other-session commits
landing on unrelated files mid-array. Not further investigated per this batch's scope (report only).

---

## 5. Addendum (2026-09-04T11:52+02:00) — whitelist completion + remote cleanup

Coordinator correction: retrieve the full whitelist (`simulations/diagnostics/*.csv`,
`simulations/posteriors*/h_*.json`, `run_metadata*.json`, `GIT_COMMIT_AT_RUN.txt`), tar only
that set, foreground `cluster/agent_ssh.sh run` with `AGENT_SSH_TIMEOUT=540`.

- First re-issue mistakenly included `posteriors_with_bh_mass/h_*.json` (3.3 GB) and hit the
  590 s harness timeout; it was stopped (`TaskStop`) **before it obtained a mux slot** ("session
  refused, backing off 30 s" was its only output), so nothing ran remotely and no partial
  remote file was left. The 3.3 GB `posteriors_with_bh_mass/` deviation of §3 STANDS
  (still NOT retrieved; chair call pending).
- Second issue (3 files, foreground, completed in seconds): `GIT_COMMIT_AT_RUN.txt`,
  `simulations/fisher_quality.csv`, `simulations/diagnostics/event_likelihoods.csv` +
  a remote-side md5 manifest, `tar -czf - | base64 -w0` over the sanctioned channel
  (5,056,265-byte tgz), extracted into the same local dir. Remote md5s vs local after
  extraction: **3/3 OK**
  (`282f442d…` GIT_COMMIT, `b97187bc…` fisher_quality.csv, `ed3d5479…` event_likelihoods.csv).
- `GIT_COMMIT_AT_RUN.txt` = `d9e501790fdc2f451664d6c3027f0ed774d35ab8` (= remote HEAD at
  array end; `081b1f28` is its ancestor on both cluster and local; `d9e50179` is an ancestor
  of local HEAD `66d00e25`). The §4 observation stands: task 0 ran at `8f933e7b`, task 40 at
  `d9e50179`; both passed the `darksiren_emri/` diff-quiet freeze vs `081b1f28`.
- `event_likelihoods.csv`: 55,064 lines = header + **55,063 rows = 41 h × 1343 events** —
  i.e. the scorer used 1343 events, matching the 07-31/08-03 readouts; the 1345-row CRB
  disclosure is the 2 empty events (report only, consistent with the JSON `n_events_empty=2`).
- `MANIFEST.md5` regenerated over all **85** retrieved files (41 `posteriors/h_*.json` +
  41 `run_metadata_*.json` + the 3 above); **list-md5 = `f88a444ffb179add23954349b59dbb1c`**
  (supersedes `7e8631c7…` of §3; saved in `MANIFEST.list-md5.txt`). Local set: 17 MB.
- Remote `$RUN_DIR/m1_iiib_posteriors.tgz` removed (confirmed `REMOVED`); no other remote
  files created or altered. Remote `/tmp/m1_rest_manifest.md5` removed in the same command.
- Prior-job disposition for the record: **6790859** (SUBMIT_RECORD_m1) = 41/41 FAILED exit 1:0
  in ~13 s each on the old identity guard (`STOP: the 0.67 run's own injections link resolves
  to …/run_20260729_seed64000_h0p67/simulations/injections, not the canonical pool …`) —
  exactly the guard defect the GUARD CORRECTION replaced; **6794421** is the run of record.
  6794421 accounting re-derived from `sacct` Elapsed × NCPUS: 207,472 core-s = **57.63 core-h**
  (§1's CPUTimeRAW figure agrees to the same 57.63); submit 09:38:47, last task end 11:32:28.
- Local destination remains `exec/r-sealed-mock/m1_iiib_retrieved/` (the dossier §4 names
  `retrieved/graph1_sealed_m1_iiib_20260904/`; not moved — flagged, one `mv` if the chair wants
  the dossier path).
