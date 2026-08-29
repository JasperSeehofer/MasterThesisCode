# Wave-3 cluster submission note (2026-08-30)

**Launched under rows #222/#223 — charter wave 3 / node B7.3 readout.** BUILDER deliverable: the
two sbatch scripts + submission wrapper + this note. No `git commit`/`add` was run by this node;
no `ssh`/`sbatch` was run by this node (cluster access is down for this pass; `submit_wave3.sh`
defaults to `DRY_RUN=1` and only prints commands regardless). Nothing here is an approval request
— it is the mechanical artifact the orchestrator reviews before flipping `DRY_RUN=0`.

**What this readout is (and is not):** this is the ONE wave-3 blind full-grid HEAD readout that
amendment F2 designates — batched, one blind reading, per-change attribution reserved for a
separate arm. It does **not** by itself compute the per-change delta or adjudicate the A14
falsifier (§8 below). Registrations:
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md`
  §0.2 (F2 statement), §8 (A14 falsifier, `T_mat` = 0.008), §9 (ledger rows / commit message).
- `results/campaign51_20260728/realistic_20260729/MEASUREMENT_HEAD_READOUT_20260827.md` (grid,
  seeding convention, dataset pins, the registered-structural-blindness precedent this readout
  mirrors).
- `results/campaign51_20260728/realistic_20260729/headreadout_20260827/{iiib,joint_r1}/run_metadata_21.json`
  (CoR-P CLI source).

Scripts delivered (all under `cluster/`):
- `wave3_headreadout_iiib.sbatch` — 41-task array over the full `H_GRID_41`, one h per task
  (task index = canonical H41 index; task 21 = h 0.730). CoR-P CLI verbatim from
  `headreadout_20260827/iiib/run_metadata_21.json` PLUS the explicit post-wave-2 defaults
  (`--mass_filter_geometry linear --mass_filter_k 1.5 --theta_b 0.0 --theta_s 1.0
  --theta_sites all`). Passes **nothing** for `catalogue_numerator_survival_2d`/`_center` —
  deliberate blindness to the row-#223 adoption (documented at length in the script header).
- `wave3_headreadout_joint_r1.sbatch` — identical, joint_r1 venue: adds
  `--observed_catalogue $WS/realizations_20260729/observed_catalogue_seed900001.csv` and its own
  sha256 STOP-gate.
- `submit_wave3.sh` — prints (does not execute unless `DRY_RUN=0`) the two sbatch lines below,
  plus an 8-item pre-launch checklist.

Both out-roots use the exact names `results/_archive/archive_run_wave2.sh`'s newly-appended
"wave 3" `ITEMS` block expects (`run_20260830_wave3_headreadout_{iiib,joint_r1}`).

---

## 1. Tasks, `--time`, CPU-h estimate

| venue | tasks | h-grid | `--time` | measured wave-2 per-task anchor (16 cpus, iiib) |
|---|---:|---|---|---|
| **iiib** | 41 (array 0-40) | full `H_GRID_41` | 00:45:00 | C0 6:28, C3 4:34–4:50, C4 6:10–6:38 |
| **joint_r1** | 41 (array 0-40) | full `H_GRID_41` | 01:30:00 | historically ≥ 2.2× the iiib anchor (`cluster/datasets.yaml` / `MEASUREMENT_HEAD_READOUT_20260827.md` §9/§F) |

**`--time` sizing:** iiib's slowest measured wave-2 anchor is 6:38 (398 s); call the working
figure "≈7 min". 7 min × 1.3 assumed packing/contention overhead (gotcha 6) ≈ 9.1 min;
`--time=00:45:00` is a ≈5× margin over that — generous, matching the task brief's own framing.
joint_r1 at 2.2–3× that same anchor puts the expected slow end at roughly 9.1 min × 3 ≈ 27.3 min;
`--time=01:30:00` keeps a >3× margin over that. Both scripts size against the *contended* anchor
per gotcha 6, and neither instrument has a partial-progress checkpoint (gotcha 9) — a walltime
kill loses the whole task, so headroom is deliberate, not slack.

**CPU-h estimate, from the measured anchors (41 tasks × 16 cpus × per-task time)**, using the
iiib anchor range [4:34, 6:38] = [274 s, 398 s] and the joint_r1 2.2–3× multiplier applied to that
same range:

```
iiib:      41 x 16 x [274, 398] s / 3600  =  49.9  –  72.5   CPU-h
joint_r1:  iiib range x [2.2, 3.0]        = 109.8  – 217.6   CPU-h
-----------------------------------------------------------------
TOTAL (82 tasks, both venues):                159.8  – 290.1 CPU-h
```

This is the **measured-anchor** estimate, i.e. what the readout is expected to actually cost.
The `--time` **budget ceiling** (if every task used its full walltime, which none should under
normal operation) is far higher: iiib 41 × 16 × 45 min = 492 CPU-h, joint_r1 41 × 16 × 90 min =
984 CPU-h — quoted only as the worst-case bound a stuck/looping task could reach, not as the
planning number.

**Backfill-friendly shape (SKILL.md gotcha 13):** two 41-task arrays of short, 16-cpu tasks on
`cpu_il` — no monolithic job, one h-value per task, matching the repo-wide convention.

---

## 2. Exact submission lines (from `submit_wave3.sh`, `DRY_RUN=1` output)

```bash
# 1. iiib — blind HEAD readout, full H_GRID_41 array
sbatch --parsable --array=0-40 \
    --export=ALL,RUN_DIR=$WS/run_20260830_wave3_headreadout_iiib \
    cluster/wave3_headreadout_iiib.sbatch

# 2. joint_r1 — blind HEAD readout, full H_GRID_41 array
sbatch --parsable --array=0-40 \
    --export=ALL,RUN_DIR=$WS/run_20260830_wave3_headreadout_joint_r1 \
    cluster/wave3_headreadout_joint_r1.sbatch
```

No inter-dependency: both arrays can submit together, and there is no STEP-2-smoke-then-array
split here (unlike wave-2's C4) because both scripts' per-h CLI is already measured at production
scale via the wave-2 C0/C3/C4 anchors — there is no new, unmeasured code path being probed at
h=0.730 first.

**Pre-launch checklist** (also printed by `submit_wave3.sh` itself, 8 items):
1. The row-#223 `[PHYSICS]` adoption commit is HEAD on this branch and pushed (`git log -1`
   subject starts `[PHYSICS] adopt the with-BH catalogue-leg twin`); `git status --porcelain`
   empty (A22 dirty-state stamp).
2. The cluster checkout has **pulled that exact commit** — verify
   `ssh bwunicluster 'git -C ~/darksiren-emri rev-parse HEAD'` matches local HEAD byte-for-byte.
   Submitting against a cluster checkout that predates the adoption commit defeats the entire
   point of this readout (F2 reads the *adopted* default).
3. `ssh bwunicluster 'bash -s' < cluster/preflight.sh` → `VERDICT: READY ✓`.
4. Archive-scheduled: confirm `results/_archive/archive_run_wave2.sh`'s "wave 3" `ITEMS` block
   (appended this pass, §5 below) will actually run post-retrieval — workspace expires
   2026-09-23, 0 extensions remaining.
5. Dataset pins (also re-checked by each sbatch's own STOP-gate at run start): CRB
   `prepared_cramer_rao_bounds.csv` md5 `9a1f2a14384a9281c97ca3be312ddaab`; reduced galaxy
   catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`; joint_r1 observed-catalogue
   `observed_catalogue_seed900001.csv` sha256
   `e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`.
6. Gotcha #10 (realization sidecar staleness): verify
   `$WS/realizations_20260729/observed_catalogue_seed900001.csv.meta.json`'s `parent_csv`
   absolute path still resolves on the cluster (repair per gotcha #10 if the repo checkout has
   moved since the sidecar was written) **before** submitting the joint_r1 array — a stale
   sidecar fails every joint_r1 task at run start, not just one.
7. Falsifier band of record for this readout's eventual use, stated so a downstream reader
   cannot infer a different threshold: **A14, `T_mat` = 0.008 on |Δmean_h| (2D channel), on BOTH
   venues**, evaluated against a **separate** `--catalogue_numerator_survival_2d off`
   counterfactual arm at the same wave-3 commit (not part of this delivery) —
   `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §8.
8. Fresh out-roots verified absent on the cluster immediately before submitting (no idempotency
   collision) — this checklist was authored without cluster access, so re-verify live.

---

## 3. Retrieval (post-run)

```bash
WS=$(ssh bwunicluster 'ws_find emri')
mkdir -p results/campaign51_20260728/realistic_20260729/wave3_20260830/{iiib,joint_r1}
rsync -avz bwunicluster:$WS/run_20260830_wave3_headreadout_iiib/     results/campaign51_20260728/realistic_20260729/wave3_20260830/iiib/
rsync -avz bwunicluster:$WS/run_20260830_wave3_headreadout_joint_r1/ results/campaign51_20260728/realistic_20260729/wave3_20260830/joint_r1/
```

**Then archive** (Option A, MUST-ARCHIVE tier, workspace expires 2026-09-23, 0 extensions):
```bash
bash results/_archive/archive_run_wave2.sh   # the "wave 3" ITEMS block now also rsyncs these two out-roots
```

---

## 4. Combine step

Per the pipeline (`CLAUDE.md` "Bayesian Inference Pipeline" / `cluster/combine.sbatch`), each
venue's 41 per-h posteriors are combined into the joint H0 posterior once its array completes:

```bash
python -m darksiren_emri "$WS/run_20260830_wave3_headreadout_iiib" --combine
python -m darksiren_emri "$WS/run_20260830_wave3_headreadout_joint_r1" --combine
```

or, on the cluster, via the existing `cluster/combine.sbatch` (4 cpus, 90 min budget — see that
script's own header):

```bash
sbatch --export=ALL,RUN_DIR=$WS/run_20260830_wave3_headreadout_iiib     cluster/combine.sbatch
sbatch --export=ALL,RUN_DIR=$WS/run_20260830_wave3_headreadout_joint_r1 cluster/combine.sbatch
```

`RENDER_FIGURES` stays off by default (per the 2026-07-03 ops decision baked into
`combine.sbatch`); render locally from the rsynced posteriors if figures are wanted.

---

## 5. Dataset-registration lines (add at completion, per SKILL.md gotcha 11 — register when the
run finishes, not later; same commit/session as banking the result)

**`cluster/datasets.yaml`** (append, following the `run_20260827_headreadout_iiib`/`joint_r1`
entries' form):

```yaml
  run_20260830_wave3_headreadout_iiib:
    git_commit: "<wave-3 commit hash, filled at run completion -- must be the row-#223 [PHYSICS]
      adoption commit or later>"
    note: "Wave-3 charter node B7.3 — the ONE blind HEAD readout, iiib venue, full H_GRID_41
      (41 nodes), EVAL_SEED 777000. CoR-P CLI verbatim (headreadout_20260827/iiib/
      run_metadata_21.json) + post-wave-2 defaults (mass_filter_geometry=linear/k=1.5,
      theta_b=0.0/theta_s=1.0/theta_sites=all). Deliberately passes NO
      catalogue_numerator_survival_2d flag (blind to the row-#223 adoption, F2). CRB set
      run_20260729_seed61000 (md5 9a1f2a14384a9281c97ca3be312ddaab). Registration:
      results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md
      §0.2/§8/§9. Falsifier verdict (A14, T_mat=0.008, vs the separate off-arm): <fill in>."
  run_20260830_wave3_headreadout_joint_r1:
    git_commit: "<wave-3 commit hash>"
    note: "Wave-3 charter node B7.3 — the ONE blind HEAD readout, joint_r1 venue (observed
      catalogue realization observed_catalogue_seed900001.csv, sha256
      e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751), full H_GRID_41,
      EVAL_SEED 777000. Same CLI/config as the iiib sibling plus --observed_catalogue. Same
      registration + falsifier reference. Falsifier verdict: <fill in>."
```

**`DATA_INVENTORY.md`** (append, following the `run_20260827_headreadout_iiib` row's form):

```
| `run_20260830_wave3_headreadout_iiib` | `<wave-3 commit>` | FULLY RECOVERABLE | Wave-3 charter
  node B7.3 — the ONE blind HEAD readout, iiib venue, full H_GRID_41. Registration:
  `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §0.2/§8/§9. <falsifier verdict, fill in>. |
| `run_20260830_wave3_headreadout_joint_r1` | `<wave-3 commit>` | FULLY RECOVERABLE | Wave-3
  charter node B7.3 — the ONE blind HEAD readout, joint_r1 venue. Registration: same. <falsifier
  verdict, fill in>. |
```

---

## 6. Notes for the orchestrator

- This node did not run `ssh`, `sbatch`, `git commit`, or `git add` (cluster access is down for
  this pass; `submit_wave3.sh` also defaults to `DRY_RUN=1` regardless).
- Neither script passes `--catalogue_numerator_survival_2d` or `--catalogue_numerator_survival_2d_center`
  — this is intentional (F2 blindness), not an omission. Do not "complete" it by adding the flag;
  that would defeat the readout's purpose.
- The falsifier check (A14, `T_mat` = 0.008) needs a **second** pair of arrays run with an
  explicit `--catalogue_numerator_survival_2d off` at the same wave-3 commit — that pair is not
  part of this delivery and is presumably a separate charter node/build.
- Each script's dataset-pin STOP-gates run on the compute node at job start, not at submission
  time — a mismatch fails the SLURM task with a clear message instead of silently scoring
  against a stale input.
- `results/_archive/archive_run_wave2.sh` is gitignored; the "wave 3" `ITEMS` block was appended
  locally this pass (§5's out-root names) and is not itself part of any commit.
