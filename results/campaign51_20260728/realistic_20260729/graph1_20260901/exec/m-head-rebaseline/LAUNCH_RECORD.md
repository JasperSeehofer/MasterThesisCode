# m-head-rebaseline — LAUNCH RECORD

Research Graph 1, Branch B. Launched 2026-09-02 (wave-1 fan-out) by the cluster launcher agent.

## Authorization

Ledger row #290 (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`),
decisions table row 4, quoted verbatim from the graph docket ratification:

> **Author ruling (verbatim): "all is ratified from the graph and the new graph structure looks
> awesome! thank you"** — ... **rows 3–11 [DO] APPROVED** — branch heads A–I trigger their first
> items (S4 harness repair; **post-flip HEAD re-baseline**; joint_r1 transform derivation; ...)

Graph spec: `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.2, decisions row 4:
"C0-prime check then blind HEAD arrays under the post-flip default (wave-3 pattern, rows
#279/#281/#283)"; row 8 flags that only decision-table row 4 authorizes this node (row 8 is the
separate T5 authorization, reported in the sibling LAUNCH_RECORD).

State candidate: `STATE_AND_CANDIDATES_20260901.md` item 11 ("Post-flip HEAD re-baseline").

## Preflight verdict (verbatim)

First run (before the fast-forward pull described below):
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 65 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
but the [REPO] line read `head=38cc0f58 ahead=0 behind=8` — 8 commits behind origin, and commit
`5e7fda16` (the row #286 flip, `[PHYSICS] flip catalogue_leg_1d_mass_aware production default to
'auto'`) was confirmed NOT an ancestor of that HEAD
(`git merge-base --is-ancestor 5e7fda16 HEAD` → `NOT_ANCESTOR`). Per the task brief's requirement
that "commit 5e7fda16 must be what the cluster runs", this blocked submission until repaired.

**Repair:** `git status --porcelain` on the cluster showed all 438 dirty entries as untracked
(`??`) job-log/scratch files, zero modified tracked files, so a fast-forward pull was safe:
`git pull --ff-only origin fix/p32d-classg-venue-repair` → `Updating 38cc0f58..1ec9514d,
Fast-forward`. Post-pull: `head=1ec9514d ahead=0 behind=0`; `git merge-base --is-ancestor 5e7fda16
HEAD` → `YES_ANCESTOR`.

Re-run preflight (post-pull, the verdict this launch was actually submitted under):
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 65 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
The 65-dir WARN is a pre-existing, unrelated backlog (gotcha 11) — not a blocker, not addressed by
this launch.

**Lustre OST 5 blocker (2026-08-31) — confirmed cleared:** `lctl get_param
osc.*OST0005*.active` on the cluster returned `active=1` for all three filesystems reporting an
OST5 (`pfs6wor8`, `pfs7dat6`, `pfs7wor9`); `lfs df -h` showed all `pfs7work9` OSTs mounted and at
~30% use, no degraded/inactive entries.

## Cluster repo state at submission

- Branch: `fix/p32d-classg-venue-repair`
- Commit: `1ec9514d` (docs: row #290 — Research Graph 1 RATIFIED...)
- `5e7fda16` confirmed an ancestor (the flip commit runs)
- Tag `commission-base` → `b593f021` (unaffected, unmoved)

## Configs launched

Two new sbatch scripts (copied to `~/darksiren-emri/cluster/` on the cluster via rsync, NOT
committed to git — matches existing untracked-script convention on the cluster checkout):

1. **`cluster/graph1_c0prime_headrebaseline_gate.sbatch`** — 2-task array (task 0=iiib,
   task 1=joint_r1), h=0.730 only (H41 index 21, seed 777021), explicit
   `--catalogue_numerator_survival_2d off --catalogue_numerator_survival_2d_center unset`
   (reproduces the wave-3 banked blind-readout row bit-for-bit at the post-flip commit; the
   g-c0-baseline instrument — max_abs = 0 on shared columns, md5 match — is evaluated by the
   orchestrator at readout against `wave3_20260830/{iiib,joint_r1}` task-21 outputs, not computed
   by this job).
2. **`cluster/graph1_headrebaseline_iiib.sbatch`** — 41-task array (full H_GRID_41), CoR-P CLI,
   BLIND (no `--catalogue_numerator_survival_2d`/`_center` flag; `catalogue_leg_1d_mass_aware`
   left at its own `auto` default, which engages post-flip).
3. **`cluster/graph1_headrebaseline_joint_r1.sbatch`** — same, joint_r1 venue
   (`--observed_catalogue` + its own sha256 STOP-gate).

All three CLIs are byte-identical to the wave-3 templates
(`cluster/wave3_c0prime_off_gate.sbatch`, `cluster/wave3_headreadout_{iiib,joint_r1}.sbatch`)
except run-dir names, job names, and provenance-note text (authorization line updated to row #290
decisions row 4). No CLI flags were changed — nothing scientific was improvised.

## Seeding

Per-task seed = `EVAL_SEED (777000) + H41 index` (cluster/SKILL.md gotcha 4). C0-prime gate uses
H41 index 21 (h=0.730) for both tasks, seed 777021. Blind HEAD arrays use `SLURM_ARRAY_TASK_ID`
as the H41 index directly (task N = H41 index N), seeds 777000–777040.

## Dataset checksum pins (evidence)

Both STOP-gated in every task (verified in-script, not just claimed):
- CRB set `run_20260729_seed61000`, `prepared_cramer_rao_bounds.csv` md5
  `9a1f2a14384a9281c97ca3be312ddaab`
- `reduced_galaxy_catalogue.csv` md5 `c52c13b5cab61f6b3f04bbe202550969`
- joint_r1 additionally: `observed_catalogue_seed900001.csv` sha256
  `e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`

**Realization-sidecar check (gotcha 10), performed before submission:**
`observed_catalogue_seed900001.meta.json:parent_csv` =
`/home/st/st_us-403333/st_ac147838/darksiren-emri/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`,
`parent_csv_sha256` = `7af3f4f4a2d51de8fbeb6583e9fa8d825f66ca95817e23d728a969277e4bd7d9`. Verified
that path exists on the cluster and its live sha256 matches the sidecar exactly — no repair
needed.

**Fresh out-roots verified absent** immediately before submission (no idempotency collision):
`run_20260902_graph1_c0prime_headrebaseline_{iiib,joint_r1}`,
`run_20260902_graph1_headrebaseline_{iiib,joint_r1}` all absent from the workspace.

## Job IDs and working directories

| Job | SLURM ID | Array | Working dir | Expected wall time |
|---|---|---|---|---|
| C0-prime gate | `6764460` | 0-1 | `$WS/run_20260902_graph1_c0prime_headrebaseline_{iiib,joint_r1}` | ≤ 1:30:00/task (measured wave-3 analogue: ~6.5 min/task) |
| Blind HEAD, iiib | `6764461` | 0-40 | `$WS/run_20260902_graph1_headrebaseline_iiib` | ≤ 0:45:00/task (measured wave-3 analogue: ~7 min/task, ~4.5 CPU-h total) |
| Blind HEAD, joint_r1 | `6764462` | 0-40 | `$WS/run_20260902_graph1_headrebaseline_joint_r1` | ≤ 1:30:00/task (measured wave-3 analogue: ≥2.2x iiib) |

`$WS` = `/pfs/work9/workspace/scratch/st_ac147838-emri`. Total estimated cost: single-digit CPU-h
(84 tasks × ~6.5 min wave-3 anchor, per the graph spec's own cost line for state candidate 11).

## Notes / what this launch is and is not

- This produces posteriors + diagnostics only. The g-c0-baseline gate verdict and the delta read
  against the wave-3 comparand are evaluated by the orchestrator at readout, not by this launch.
- Submitted without a SLURM `--dependency` between the gate and the blind arrays, matching the
  wave-3 precedent (submit_wave3.sh submitted all three concurrently) — the blind arrays are cheap
  and useful regardless of the gate's own PASS/FAIL; reporting discipline (no delta read until the
  gate result is known) is an orchestrator-side rule, not an enforced SLURM dependency.
- Chair monitors completion; this agent does not poll.
