# BYTEID_EVIDENCE_LAUNCH — byte-identity gate for the h-decoupling [PHYSICS] commit

Research Graph 1, Branch B (`b-hprior-fix`). Launched 2026-09-02 by the byte-identity evidence
launcher agent.

## Authorization

Rows #301 item 4(a) / #308 (the ratified physics change, commit `a26959b4` or its immediate docs
successor on `fix/p32d-classg-venue-repair`). The design's g-byte-id gate requires evidence that
in-bound results are byte-identical after the change:

- `a26959b4` — `[PHYSICS] decouple h grid-admissibility from the host-window bound
  (rows #293/#301/#304/#308-#309)`
- Docs successor confirmed as an ancestor of the syncing commit (see below):
  `8f3a52a4 docs: gate-ledger rows point at the landed [PHYSICS] commit a26959b4`

## Plan of record (chair-set, disclosed substitute for design §2.6)

The design's own §2.6 specifies a 41-node full-grid re-baseline (~70 CPU-h). The plan actually
executed here is a **disclosed cheaper substitute**, set by the chair for this launch: re-run only
the C0-prime gate pair (2 tasks, h=0.730, both venues — `iiib` and `joint_r1`) at the post-change
commit and byte-compare against the just-banked **pre-change** C0-prime outputs from job
`6764460` (`$WS/run_20260902_graph1_c0prime_headrebaseline_{iiib,joint_r1}`,
`$WS = /pfs/work9/workspace/scratch/st_ac147838-emri`). This gives identical code-path coverage
to the full plan at ~2-3 CPU-h instead of ~70. The full-population diff (1588 events x all
columns x 2 venues, comfortably >> 1e5 values) satisfies the design's N>=1e5 byte-comparison
criterion. **The comparison read itself is a later step, not performed by this launch.**

## Steps executed

### 1. Local HEAD contains the [PHYSICS] commit

```
$ git log --oneline -3
dcb2c470 docs: docket items 6-8 — rows #308 addendum + Option A' ratification ask (item 8)
8f3a52a4 docs: gate-ledger rows point at the landed [PHYSICS] commit a26959b4
a26959b4 [PHYSICS] decouple h grid-admissibility from the host-window bound (rows #293/#301/#304/#308-#309)
```
Local HEAD at launch time: `dcb2c470` (a26959b4 four commits back, confirmed an ancestor).
Branch: `fix/p32d-classg-venue-repair`.

### 2. Preflight (per `.claude/skills/cluster/SKILL.md`)

```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 72 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
The unregistered-dataset WARN is a pre-existing, unrelated backlog (gotcha 11) — not a blocker,
not addressed by this launch.

### 3. Sync cluster checkout to local HEAD (bundle+scp+ff-only pattern)

Direct push was **not** attempted (denied per task brief). Instead:

1. `git bundle create sync.bundle dcc75352..HEAD` (ranged from the cluster's known ancestor
   commit to local HEAD; 91 KB vs. 144 MB for a full-history bundle).
2. `scp sync.bundle bwunicluster:~/sync_byteid.bundle`
3. On cluster: `git fetch ~/sync_byteid.bundle HEAD:refs/bundle-tmp` — cluster checkout had
   0 modified tracked files (`git status --porcelain | grep -vc '^??'` = 0), so an ff-only merge
   was safe: `git merge-base --is-ancestor HEAD refs/bundle-tmp` → `FF_OK`.
4. First merge attempt aborted: one untracked file (`cluster/graph1_t5_armR_c0prime.sbatch`,
   left over from a prior untracked-script convention push) collided with a file the incoming
   history now tracks. Moved it aside, merged (`Fast-forward: dcc75352..dcb2c470`, 18 files
   changed including the h-decoupling commit's physics/test files and the docket docs), restored
   it after (content preserved, now tracked — matches local).
5. Cleaned up the temporary ref and bundle file on the cluster.

**Post-sync cluster state:** `HEAD = dcb2c470` (byte-identical to local HEAD at launch),
0 tracked modifications, 539 untracked scratch/log files (pre-existing, unrelated to this sync).

### 4. Submit the C0-prime template with a fresh out-root

Template: `cluster/graph1_c0prime_headrebaseline_gate.sbatch` (the pre-change gate script,
job 6764460). Copied to `cluster/graph1_c0prime_byteid_postdecouple_gate.sbatch` on the cluster,
changing **only**: job-name, out-root name (`run_20260902_graph1_c0prime_headrebaseline_*` ->
`run_20260902_graph1_c0prime_byteid_postdecouple_*`), log-name prefix, header/comment text, and
the provenance-note string. Diffed byte-for-byte against the template to confirm: the `#SBATCH`
resource lines, the CLI invocation (all `python -m darksiren_emri ... --evaluate` flags), the
seed derivation (`EVAL_SEED=777000`, `TASK_INDEX=21` -> seed 777021 for both tasks), and the
dataset-pin STOP-gates (CRB md5 `9a1f2a14384a9281c97ca3be312ddaab`, catalogue md5
`c52c13b5cab61f6b3f04bbe202550969`, joint_r1 observed-catalogue sha256
`e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`) are unchanged.

Pre-submission checks:
- Fresh out-roots (`run_20260902_graph1_c0prime_byteid_postdecouple_{iiib,joint_r1}`) verified
  absent from the workspace — no idempotency collision.
- Pre-change comparand outputs confirmed present and banked:
  `$WS/run_20260902_graph1_c0prime_headrebaseline_{iiib,joint_r1}/simulations/posteriors/h_0_73.json`
  (job 6764460, both venues, 48848/48849 bytes, timestamped 2026-09-02 00:12).

### 5. Submission

```
$ git rev-parse HEAD
dcb2c470472f2f1f912c166ab48c3890a410c42c
$ sbatch cluster/graph1_c0prime_byteid_postdecouple_gate.sbatch
Submitted batch job 6768603
```

## Job ID and working directories

| Job | SLURM ID | Array | Working dir | Expected wall time |
|---|---|---|---|---|
| C0-prime byte-identity gate | `6768603` | 0-1 (0=iiib, 1=joint_r1) | `$WS/run_20260902_graph1_c0prime_byteid_postdecouple_{iiib,joint_r1}` | <= 1:30:00/task (measured analogue, job 6764460: ~6.5 min/task) |

`$WS = /pfs/work9/workspace/scratch/st_ac147838-emri`. Cluster HEAD at submission: `dcb2c470`.
Estimated cost: single-digit CPU-h (2 tasks x ~6.5 min anchor).

## What this launch is and is not

- This produces posteriors + diagnostics only, at the post-decoupling commit. The g-byte-id
  verdict (max_abs diff on all shared columns across both venues' full posterior/diagnostics
  output, N >= 1e5 values, comparand = job 6764460) is evaluated by the orchestrator at readout,
  not by this launch.
- No commits were made as part of this launch (per task brief).
- Chair monitors completion; this agent does not poll.
