# Wave-3 cluster submission note (2026-08-30)

**Launched under rows #222/#223 — charter wave 3 / node B7.3 readout.** BUILDER deliverable: the
three sbatch scripts + submission wrapper + this note. No `git commit`/`add` was run by this node;
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
- `wave3_c0prime_off_gate.sbatch` — 2-task array (task 0 = iiib, task 1 = joint_r1), h = 0.730
  only. Same CoR-P CLI + post-wave-2 defaults as the two readout scripts below, **plus the
  explicit pre-adoption pair** `--catalogue_numerator_survival_2d off
  --catalogue_numerator_survival_2d_center unset`. This is the A14 falsifier **baseline gate**
  ("C0′") — see §1a below.
- `wave3_headreadout_iiib.sbatch` — 41-task array over the full `H_GRID_41`, one h per task
  (task index = canonical H41 index; task 21 = h 0.730). CoR-P CLI verbatim from
  `headreadout_20260827/iiib/run_metadata_21.json` PLUS the explicit post-wave-2 defaults
  (`--mass_filter_geometry linear --mass_filter_k 1.5 --theta_b 0.0 --theta_s 1.0
  --theta_sites all`). Passes **nothing** for `catalogue_numerator_survival_2d`/`_center` —
  deliberate blindness to the row-#223 adoption (documented at length in the script header).
- `wave3_headreadout_joint_r1.sbatch` — identical, joint_r1 venue: adds
  `--observed_catalogue $WS/realizations_20260729/observed_catalogue_seed900001.csv` and its own
  sha256 STOP-gate.
- `submit_wave3.sh` — prints (does not execute unless `DRY_RUN=0`) all three sbatch lines below
  (c0prime_off_gate first), plus a 9-item pre-launch checklist.

All four out-roots use the exact names `results/_archive/archive_run_wave2.sh`'s newly-appended
"wave 3" `ITEMS` block expects (`run_20260830_wave3_headreadout_{iiib,joint_r1}`,
`run_20260830_wave3_c0prime_off_{iiib,joint_r1}`).

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

## 1a. C0′ off-gate

**Purpose.** The A14 falsifier's delta read (§ intro above; `T_mat` = 0.008 on |Δmean_h|,
2D channel, both venues) needs a **pre-adoption baseline**. The obvious way to get one — a second
82-task array (both venues × full `H_GRID_41`) with an explicit
`--catalogue_numerator_survival_2d off --catalogue_numerator_survival_2d_center unset` — costs
another ≈160–290 CPU-h (the same order as §1's own estimate) and is not needed if the
**already-banked** 2026-08-27 readouts (`headreadout_20260827/{iiib,joint_r1}`) still reproduce
bit-for-bit under that same explicit-off CLI at the wave-3 commit: the code paths touched by the
four post-`d04d9dc9` commits are, by construction, supposed to be byte-identical to pre-flag
behaviour at their default/literal-skip values (gate-ledger rows 2026-08-28/29), so an explicit
`off` at the wave-3 commit is predicted to reproduce the banked rows exactly. `wave3_c0prime_off_gate.sbatch`
certifies that prediction cheaply — one h-value (0.730, the same canonical task-21 point already
used as the C0 baseline gate's own anchor), both venues, 2 tasks total — instead of assuming it.
This mirrors `REGISTRATION_C0_BASELINE_GATE_20260829.md`'s own C0 gate (same single-h economy,
same gate mechanics), applied here to the row-#223 flag's **pre-adoption** value rather than to
the four intervening estimator commits (F4: size the falsifier-baseline check against what is
actually needed, not the full grid).

**Cost.** ≈2 CPU-h (iiib, task 0: ~7 min × 16 cpus, same per-h anchor as §1's iiib row) + ≈4 CPU-h
(joint_r1, task 1: 2.2–3× that anchor) ≈ **6 CPU-h total** for both tasks — negligible next to
§1's 160–290 CPU-h estimate for the two 41-node blind arrays.

**Gate band** (per `REGISTRATION_C0_BASELINE_GATE_20260829.md` §3/§13, the same form used for the
prior C0 gate): max |relative difference| **≤ 1e-12** on the 14 diagnostic
`event_likelihoods.csv` columns (`w_G, w_G_legacy, w_tilde_G, alpha_G_phi, r_Malm, D_tilde_phi,
L_cat_no_bh, L_cat_with_bh, B_num, B_num_wbh, g_frac, L_comp, combined_no_bh, combined_with_bh`)
at h = 0.730, per venue, against the corresponding banked
`headreadout_20260827/{iiib,joint_r1}/event_likelihoods.csv` task-21 rows; **plus** md5-identical
`posteriors/h_0_73.json` and `posteriors_with_bh_mass/h_0_73.json` (md5 is a strictly stronger
identity claim than a parsed max-abs-diff, per §13's own precedent).

**PASS** (both venues, band satisfied) ⇒ the banked 2026-08-27 readouts **are** the pre-adoption
baseline for the A14 delta read — no separate off-array is needed, and the full readout (§0
above) can be reported against them directly.

**FAIL** (either venue) ⇒ the explicit-off CLI at the wave-3 commit does not reproduce the banked
bytes: something in the four post-`d04d9dc9` commits (or the wave-3 commit itself) moved
behaviour even at the flag's own pre-adoption value. In that case the full 82-task
`--catalogue_numerator_survival_2d off` array (both venues, full `H_GRID_41`) becomes necessary,
and the h = 0.730 per-column diff produced by this gate is diagnosed FIRST (which commit owns it)
before that larger array is launched — do not launch the 82-task array blind to which commit
broke reproduction.

---

## 2. Exact submission lines (from `submit_wave3.sh`, `DRY_RUN=1` output)

```bash
# 0. c0prime_off_gate — A14 falsifier baseline gate, both venues, h=0.730 only (array 0-1)
sbatch --parsable --array=0-1 \
    cluster/wave3_c0prime_off_gate.sbatch

# 1. iiib — blind HEAD readout, full H_GRID_41 array
sbatch --parsable --array=0-40 \
    --export=ALL,RUN_DIR=$WS/run_20260830_wave3_headreadout_iiib \
    cluster/wave3_headreadout_iiib.sbatch

# 2. joint_r1 — blind HEAD readout, full H_GRID_41 array
sbatch --parsable --array=0-40 \
    --export=ALL,RUN_DIR=$WS/run_20260830_wave3_headreadout_joint_r1 \
    cluster/wave3_headreadout_joint_r1.sbatch
```

No inter-dependency at the SLURM level: all three arrays can submit together (the c0prime gate
does not block the blind readout's own compute). There is no STEP-2-smoke-then-array split here
(unlike wave-2's C4) because all three scripts' per-h CLI is already measured at production scale
via the wave-2 C0/C3/C4 anchors — there is no new, unmeasured code path being probed at h=0.730
first. The dependency that DOES matter is at the **reporting** stage, not submission: do not read
the blind readout's delta against the banked 2026-08-27 baseline until the c0prime gate's own
PASS/FAIL (§1a, checklist item 9 below) is known.

**Pre-launch checklist** (also printed by `submit_wave3.sh` itself, 9 items):
1. The row-#223 `[PHYSICS]` adoption commit is HEAD on this branch and pushed (`git log -1`
   subject starts `[PHYSICS] adopt the with-BH catalogue-leg twin`); `git status --porcelain`
   empty (A22 dirty-state stamp).
2. The cluster checkout has **pulled that exact commit** — verify
   `ssh bwunicluster 'git -C ~/darksiren-emri rev-parse HEAD'` matches local HEAD byte-for-byte.
   Submitting against a cluster checkout that predates the adoption commit defeats the entire
   point of this readout (F2 reads the *adopted* default).
3. `ssh bwunicluster 'bash -s' < cluster/preflight.sh` → `VERDICT: READY ✓`.
4. Archive-scheduled: confirm `results/_archive/archive_run_wave2.sh`'s "wave 3" `ITEMS` block
   (appended this pass, §5 below) lists all four out-roots and will actually run
   post-retrieval — workspace expires 2026-09-23, 0 extensions remaining.
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
   venues**, evaluated against the pre-adoption baseline — the banked 2026-08-27 readouts if item
   9 PASSES, else a **separate** `--catalogue_numerator_survival_2d off` counterfactual array at
   the same wave-3 commit (not part of this delivery) —
   `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §8.
8. Fresh out-roots verified absent on the cluster immediately before submitting (no idempotency
   collision) — this checklist was authored without cluster access, so re-verify live.
9. `c0prime_off_gate`'s own PASS/FAIL (§1a), checked **after** it completes and **before** the
   blind-readout delta is reported: max |relative difference| ≤ 1e-12 on the 14 diagnostic
   `event_likelihoods.csv` columns + md5-identical posterior JSONs, at h=0.730, both venues, vs.
   `headreadout_20260827/{iiib,joint_r1}` task 21 — `REGISTRATION_C0_BASELINE_GATE_20260829.md`
   §3/§13. FAIL on either venue means the full 82-task off-array becomes necessary (§1a); diagnose
   the diff before launching it.

---

## 3. Retrieval (post-run)

```bash
WS=$(ssh bwunicluster 'ws_find emri')
mkdir -p results/campaign51_20260728/realistic_20260729/wave3_20260830/{iiib,joint_r1,c0prime_off_iiib,c0prime_off_joint_r1}
rsync -avz bwunicluster:$WS/run_20260830_wave3_headreadout_iiib/     results/campaign51_20260728/realistic_20260729/wave3_20260830/iiib/
rsync -avz bwunicluster:$WS/run_20260830_wave3_headreadout_joint_r1/ results/campaign51_20260728/realistic_20260729/wave3_20260830/joint_r1/
rsync -avz bwunicluster:$WS/run_20260830_wave3_c0prime_off_iiib/     results/campaign51_20260728/realistic_20260729/wave3_20260830/c0prime_off_iiib/
rsync -avz bwunicluster:$WS/run_20260830_wave3_c0prime_off_joint_r1/ results/campaign51_20260728/realistic_20260729/wave3_20260830/c0prime_off_joint_r1/
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
- Neither blind-readout script passes `--catalogue_numerator_survival_2d` or
  `--catalogue_numerator_survival_2d_center` — this is intentional (F2 blindness), not an
  omission. Do not "complete" it by adding the flag; that would defeat the readout's purpose.
  `wave3_c0prime_off_gate.sbatch` is the one script that DOES pass the pair explicitly (at the
  pre-adoption value `off`/`unset`) — that is also intentional (§1a), not an inconsistency.
- The A14 falsifier delta read (`T_mat` = 0.008) needs a pre-adoption baseline. This delivery's
  `wave3_c0prime_off_gate.sbatch` (§1a) is the cheap (≈6 CPU-h) attempt to certify the banked
  2026-08-27 readouts as that baseline without a full second 82-task off-array; **if it FAILS**,
  the full off-array (both venues, full `H_GRID_41`, explicit
  `--catalogue_numerator_survival_2d off`) becomes necessary and is presumably a separate charter
  node/build at that point — diagnose the gate's own h=0.730 diff first (§1a) before launching it.
- Each script's dataset-pin STOP-gates run on the compute node at job start, not at submission
  time — a mismatch fails the SLURM task with a clear message instead of silently scoring
  against a stale input.
- `wave3_c0prime_off_gate.sbatch` computes its own per-task `RUN_DIR` from `$WORKSPACE` + a
  case-on-`SLURM_ARRAY_TASK_ID` venue name (task 0 = iiib, task 1 = joint_r1) rather than reading
  an exported `RUN_DIR` — a plain 2-task array shares one `--export=ALL` environment across both
  tasks, so the two venues' distinct out-roots (and CoR-P CLIs — joint_r1 adds
  `--observed_catalogue`) are selected inside the script, not from `submit_wave3.sh`'s export.
- `wave3_c0prime_off_gate.sbatch` uses a single uniform `--time=01:30:00` for both tasks (the
  larger of the two venues' own per-script times) — plain `sbatch --array` has no per-task-index
  walltime, so the ceiling is sized to the slower venue (joint_r1); iiib finishes at a small
  fraction of it. This is a deliberate builder-level deviation from "00:45:00 / 01:30:00 per
  venue" read literally — see the script's own header for the full reasoning.
- `results/_archive/archive_run_wave2.sh` is gitignored; the "wave 3" `ITEMS` block was appended
  locally this pass (§5's out-root names, now including the two `c0prime_off` out-roots) and is
  not itself part of any commit.
