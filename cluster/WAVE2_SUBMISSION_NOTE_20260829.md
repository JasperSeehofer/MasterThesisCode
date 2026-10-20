# Wave-2 cluster submission note (2026-08-29)

**Launched under rows #222/#223 — charter nodes C0/B5.2/B7.2/B1.2.** BUILDER deliverable: the
sbatch scripts + submission wrapper + this note. No `git commit`/`add` was run by this node; no
`sbatch` was run by this node (`submit_wave2.sh` defaults to `DRY_RUN=1` and only prints
commands). Nothing here is an approval request — it is the mechanical artifact the orchestrator
reviews before flipping `DRY_RUN=0`.

Registrations implemented:
- C0 — `results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_C0_BASELINE_GATE_20260829.md`
- C3 — `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md`
- C4 — `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.2 + §13.3
- C1 (template, not submitted) — `results/campaign51_20260728/realistic_20260729/fanout1_20260829/WAVE2_REGISTRATION_CHECK_20260829.md` §2 (PA-HIER-31 skeleton) + `P6_THETA_CLI_PLUMBING_RECORD.md`

Scripts delivered (all under `cluster/`):
- `wave2_c0_baseline.sbatch` — single task, h=0.730, CoR-P + explicit `mass_filter_geometry=linear,k=1.5`.
- `wave2_c3_win_k3.sbatch` — 4-task array (H4 grid), arm T only (`mass_filter_geometry=log,k=3.0`).
- `wave2_c4_twin_mz_sel.sbatch` — 4-task array (H4 grid, h=0.730 at index 0), arm T only
  (`catalogue_numerator_survival_2d=mz_sel,center=eff`); designed for a split submission
  (index 0 alone, then 1-3 as a dependent array).
- `wave2_c1_s0b_TEMPLATE.sbatch` — 4-task array (theta nodes at fixed h=0.730). **Template only,
  not runnable as-is** — theta CLI flags are commented out pending P6's commit; guarded by the
  header block. Not wired into `submit_wave2.sh`'s default submission.
- `submit_wave2.sh` — prints (does not execute unless `DRY_RUN=0`) the exact sbatch lines below,
  in the required order (C0 + C3 + C4-smoke in one set, C4-remainder dependent, C1 commented out).

All four out-roots use the exact names `results/_archive/archive_run_wave2.sh` already expects
(`run_20260829_wave2_c{0,1,3,4}_iiib`) — verified against that script's `ITEMS[]` array.

---

## 1. Per-arm tasks, `--time`, CPU-h estimate

| arm | tasks | h-grid | `--time` | CPU-h estimate (tasks × 16 cpus × time, worked below) |
|---|---:|---|---|---:|
| **C0** | 1 | {0.730} | 03:00:00 | 15–23 (registration §7: single h-point × 14.93–22.9 CPU-h/h-point anchor; 1 task × 16 cpus × up to ~1:25:52 contended anchor ≈ 22.9 CPU-h upper) |
| **C3** | 4 | {0.660, 0.665, 0.670, 0.730} | 03:00:00 | 44–137 (4 nodes × 14.93–22.9 CPU-h/node × 0.73–1.5 measured p95 candidate-growth factor, `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` §7) |
| **C4** | 4 (1 smoke + 3 dependent) | {0.730, 0.660, 0.665, 0.670} | 03:00:00 | 59.7–105 (4 nodes × 14.93–20.27 CPU-h/node × 1.0–1.3 assumed `mz_sel` overhead; ceiling at the 1.3× upper bound, proposal §6.2/§13.3 table) |
| **C1** (template, NOT submitted) | 4 | {0.730} × 4 theta-nodes | 03:00:00 | 60–92 (unsmeared/`"2.2"` form only; the 81–113 CPU-h smeared band is withdrawn — `WAVE2_REGISTRATION_CHECK_20260829.md` §0 F-A / `COMPUTE_LEDGER.md` row C1 revision) |

**`--time=03:00:00` sizing, all four scripts, against the CONTENDED anchor** (HEAD-readout
off-arm slowest task 1:25:52 = 5152 s, `MEASUREMENT_HEAD_READOUT_20260827.md` §F — this repo's
gotcha-6/9 convention: size against the contended anchor, not the uncontended 56–76 min one):

- **C0**: 1 task, no packing at all — 03:00:00 gives ~2.1× margin over 5152 s even with zero
  scaling factor.
- **C3**: scale by the measured **p95 candidate-growth factor 1.5** (registered as 1.498,
  `b5_window_count.json:growth_factor_iii_vs_i.p95`): 5152 s × 1.498 = 7718 s = 2:08:38.
  03:00:00 (10800 s) gives ~40% margin over that product.
- **C4**: scale by the **1.0–1.3 assumed `mz_sel` overhead** (not yet measured at production
  scale — the h=0.730 task IS the measurement, `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §13.2):
  5152 s × 1.3 = 6698 s = 1:51:38. 03:00:00 gives ~1:08 margin (registration's own "~2.1× the
  slowest observed task" sizing, §6.2/§13.3).
- **C1** (template): PA-HIER-31 item 1 itself specifies `--time=03:00:00`; carried unchanged.

**Backfill-friendly shape (SKILL.md gotcha 13):** every arm is a small array of short,
16-cpu tasks on `cpu_il` — no monolithic job, one h-value (or theta-node) per task, matching
the repo-wide convention and the fairshare-floor guidance (many short array tasks win on
backfill; argue size on scientific need, not headroom).

**Total wave-2 cluster CPU-h (C0 + C3 + C4, 9 tasks):**

```
C0:  15   –  23   CPU-h
C3:  44   – 137   CPU-h
C4:  59.7 – 105   CPU-h
-------------------------
TOTAL: 118.7 – 265 CPU-h   (9 tasks)
```

Plus C1 template (60–92 CPU-h, 4 tasks) if/when PA-HIER-31 + P6 land — **not counted in the
submitted total**, since it is not part of this submission set. Conditional fallback: **+120–173
CPU-h** if the C0 gate FAILS (C3 re-runs its own 4-node baseline at +59.7–91.6 CPU-h, C4 at
+59.7–81.1 CPU-h — `WAVE2_REGISTRATION_CHECK_20260829.md` §0 item 7 / `COMPUTE_LEDGER.md` row
"C0-FAIL fallback").

---

## 2. Exact submission order (from `submit_wave2.sh`, `DRY_RUN=1` output)

```bash
# 1. C0 — shared baseline gate task (single task, no array)
sbatch --parsable --export=ALL,RUN_DIR=$WS/run_20260829_wave2_c0_iiib \
    cluster/wave2_c0_baseline.sbatch

# 2. C3 — log k=3 window counterfactual, arm T (array 0-3)
sbatch --parsable --array=0-3 --export=ALL,RUN_DIR=$WS/run_20260829_wave2_c3_iiib \
    cluster/wave2_c3_win_k3.sbatch

# 3. C4 — PROD-CF-2D mz_sel/eff, arm T, STEP-2 smoke (h=0.730 only, array 0-0)
C4_SMOKE_JOBID=$(sbatch --parsable --array=0-0 \
    --export=ALL,RUN_DIR=$WS/run_20260829_wave2_c4_iiib \
    cluster/wave2_c4_twin_mz_sel.sbatch)

# 4. C4 — remaining 3 tasks, DEPENDENT array (afterok on the smoke)
sbatch --parsable --array=1-3 --dependency=afterok:$C4_SMOKE_JOBID \
    --export=ALL,RUN_DIR=$WS/run_20260829_wave2_c4_iiib \
    cluster/wave2_c4_twin_mz_sel.sbatch

# 5. C1 — COMMENTED OUT this wave (PA-HIER-31 unauthored, P6 not committed):
#   sbatch --parsable --array=0-3 --export=ALL,RUN_DIR=$WS/run_20260829_wave2_c1_iiib \
#       cluster/wave2_c1_s0b_TEMPLATE.sbatch
```

Steps 1–3 form "one submission set" per the task brief (C0 + C3 + C4-smoke, no
inter-dependency between them — C3's arm-T tasks do not wait on C0's result, only the
*baseline-reuse decision* for C3/C4 depends on C0's PASS/FAIL). Step 4 is deliberately gated on
step 3's job ID via `--dependency=afterok`, so the STEP-2 smoke's observed walltime is known
before the other three C4 tasks' budget is committed (task-brief instruction; also protects
against the disclosed 1.0–1.3× overhead being an under-measurement).

**Pre-launch checklist** (also printed by `submit_wave2.sh` itself):
1. `ssh bwunicluster 'bash -s' < cluster/preflight.sh` → `VERDICT: READY ✓`.
2. Wave-2 commit exists; `git status` clean at the cluster checkout (A22 dirty-state stamp —
   `WAVE2_REGISTRATION_CHECK_20260829.md` §0/§5 item 1: the tree was dirty at registration-check
   time, so this commit does not exist yet as of this builder pass).
3. `COMPUTE_LEDGER.md` archive-scheduled cells for C0/C1/C3/C4 read **yes** — confirmed already
   at `COMPUTE_LEDGER.md:99-102` (GAP-6 closure append, 2026-08-29); the earlier `pending` cell
   at line 42 is stale-by-design (append-only convention) and cross-references the later row.
4. Dataset pins (also re-checked by each sbatch's own STOP-gate at run start, per CLAUDE.md):
   CRB `prepared_cramer_rao_bounds.csv` md5 `9a1f2a14384a9281c97ca3be312ddaab`
   (`run_20260729_seed61000`); reduced galaxy catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`.

---

## 3. Retrieval (post-run)

```bash
WS=$(ssh bwunicluster 'ws_find emri')
mkdir -p results/campaign51_20260728/realistic_20260729/wave2_20260829/{c0,c3,c4}
rsync -avz bwunicluster:$WS/run_20260829_wave2_c0_iiib/ results/campaign51_20260728/realistic_20260729/wave2_20260829/c0/
rsync -avz bwunicluster:$WS/run_20260829_wave2_c3_iiib/ results/campaign51_20260728/realistic_20260729/wave2_20260829/c3/
rsync -avz bwunicluster:$WS/run_20260829_wave2_c4_iiib/ results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/
```

(C1's `wave2_c1_iiib` line is omitted here since C1 is not submitted this wave; add it verbatim
once C1 runs, matching the same pattern.)

**Then archive** (Option A, MUST-ARCHIVE tier, workspace expires 2026-09-23):
```bash
bash results/_archive/archive_run_wave2.sh   # rsyncs the same 4 out-roots into results/_archive/
```

---

## 4. Dataset-registration lines (add at completion, per SKILL.md gotcha 11 — register when the
run finishes, not later; same commit/session as banking the result)

**`cluster/datasets.yaml`** (append, following the existing `run_20260827_headreadout_iiib`
entry's form):

```yaml
  run_20260829_wave2_c0_iiib:
    git_commit: "<wave-2 commit hash, filled at run completion>"
    note: "Wave-2 charter node C0 — shared baseline gate task, iiib venue, h=0.730 only,
      seed 777021 (paired to headreadout_20260827/iiib task 21). CoR-P CLI + explicit
      mass_filter_geometry=linear/mass_filter_k=1.5. CRB set run_20260729_seed61000
      (md5 9a1f2a14384a9281c97ca3be312ddaab). Gate: all numeric event_likelihoods.csv columns
      + both posteriors vs the banked d04d9dc9 rows at <=1e-12 relative. Registration:
      results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_C0_BASELINE_GATE_20260829.md.
      PASS/FAIL: <fill in>."
  run_20260829_wave2_c3_iiib:
    git_commit: "<wave-2 commit hash>"
    note: "Wave-2 charter node C3 (B5.2) — log k=3 mass-window geometry counterfactual, arm T
      only (mass_filter_geometry=log, mass_filter_k=3.0), iiib venue, H4 grid
      {0.660,0.665,0.670,0.730}. Baseline B reused at zero compute from
      run_20260827_headreadout_iiib conditional on C0 PASS. Registration:
      results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md.
      Verdict: <fill in>."
  run_20260829_wave2_c4_iiib:
    git_commit: "<wave-2 commit hash>"
    note: "Wave-2 charter node C4 (B7.2) — PROD-CF-2D with-BH catalogue-leg twin
      (catalogue_numerator_survival_2d=mz_sel, center=eff), arm T only, iiib venue, H4 grid
      {0.730,0.660,0.665,0.670} (0.730 = STEP-2 smoke, task 0). Baseline B reused at zero
      compute from run_20260827_headreadout_iiib conditional on C0 PASS. Registration:
      results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md
      §6.2/§13.3. mz_sel overhead measured: <fill in>. Verdict: <fill in>."
```

**`DATA_INVENTORY.md`** (append, following the `run_20260827_headreadout_iiib` row's form):

```
| `run_20260829_wave2_c0_iiib` | `<wave-2 commit>` | FULLY RECOVERABLE | Wave-2 charter node C0
  — shared baseline gate (h=0.730 only, seed 777021). CRB set `run_20260729_seed61000` (md5
  `9a1f2a14384a9281c97ca3be312ddaab`). Registration: `REGISTRATION_C0_BASELINE_GATE_20260829.md`.
  <PASS/FAIL, fill in>. |
| `run_20260829_wave2_c3_iiib` | `<wave-2 commit>` | FULLY RECOVERABLE | Wave-2 charter node C3
  (B5.2) — log k=3 window counterfactual, arm T, H4 grid. Registration:
  `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md`. <verdict, fill in>. |
| `run_20260829_wave2_c4_iiib` | `<wave-2 commit>` | FULLY RECOVERABLE | Wave-2 charter node C4
  (B7.2) — PROD-CF-2D mz_sel/eff twin, arm T, H4 grid (0.730 = STEP-2 smoke). Registration:
  `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.2/§13.3. <verdict, fill in>. |
```

(A fourth `run_20260829_wave2_c1_iiib` row is added only once C1 actually runs — not this wave.)

---

## 5. Notes for the orchestrator

- This node did not run `sbatch`, `git commit`, or `git add`. `submit_wave2.sh` defaults to
  `DRY_RUN=1`; flipping to `DRY_RUN=0` is the orchestrator's decision, made after the pre-launch
  checklist above is satisfied.
- The wave-2 commit hash is unknown at authoring time (dirty tree, per
  `WAVE2_REGISTRATION_CHECK_20260829.md` §0/§5 item 1) — every sbatch script stamps
  `git rev-parse HEAD` into `$RUN_DIR/GIT_COMMIT_AT_RUN.txt` and into the provenance JSON at run
  START, so the A22 stamp is whatever HEAD is when `sbatch` actually runs, not a value baked in
  by this builder pass. The orchestrator is responsible for verifying that HEAD, at submission
  time, is the intended wave-2 commit (A22).
- Each script's dataset-pin STOP-gate (CRB md5 + catalogue md5) runs on the COMPUTE NODE at job
  start, not at submission time — a mismatch fails the SLURM task with a clear message rather
  than silently scoring against a stale input.
- C1 is deliberately not part of `submit_wave2.sh`'s default set; its template exists so the
  moment PA-HIER-31 is ratified and P6 lands (commit, not just "implemented"), the orchestrator
  can uncomment the guarded `--theta_*` lines in `wave2_c1_s0b_TEMPLATE.sbatch` and add its
  submission line to `submit_wave2.sh`.
