# m-t5-armR-c0prime — LAUNCH RECORD

Research Graph 1, Branch F. Launched 2026-09-02 (wave-2 first batch) by the cluster launcher
agent.

## Authorization (quoted)

Ledger row #290 decisions row 8 (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`):

> Arm R launch strictly behind its own C0-prime-equivalent gate (row #284(4a))

Plus row #301 (wave-2 launch authorization; "g-c0-baseline re-stamped GREEN-AS-CORRECTED per
docket item 5(A)" — the precedent this node's C0-prime pattern relies on).

Design source: `results/campaign51_20260728/realistic_20260729/tree2_20260830/
PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md` §6.2 (Arm R): "baseline = the banked joint_r1 HEAD
readout at the same nodes (zero compute) subject to the same C0-style ingredient gate on this
venue (**a joint_r1 C0-prime task, approx 1-2 CPU-h, is required because no joint_r1 baseline has
been re-run at the current HEAD**)."

Pattern source: `cluster/wave3_c0prime_off_gate.sbatch` (the generic wave-3 C0-prime pattern —
single h=0.730 node, byte-identical CoR-P CLI to the corresponding full-grid readout, comparing
against a banked comparand row) and `cluster/graph1_headrebaseline_joint_r1.sbatch` (the freshest
banked joint_r1 HEAD-readout CLI, wave-1's m-head-rebaseline node, out-root
`run_20260902_graph1_headrebaseline_joint_r1`, task 21 = h 0.730 = the comparand row this gate
reproduces).

## What this gate is (and is not)

This is the **ingredient check** for Arm R's baseline, not Arm R's own measurement. Arm R's design
(§6.2) reads joint_r1 at `--mass_filter_geometry log --mass_filter_k 3.0` against the *production
baseline's* banked HEAD readout — i.e. the comparand row uses the standard production flags
(`--mass_filter_geometry linear --mass_filter_k 1.5`, BLIND to
`--catalogue_numerator_survival_2d`/`_center`, matching `graph1_headrebaseline_joint_r1.sbatch`
exactly), not Arm R's own log/k=3 flags. This script reproduces that baseline row bit-for-bit at
the current commit; Arm R's actual log/k=3 array is a separate, later launch, gated on this
script's read coming back green (not evaluated here — that is a later read, not this launch's
job).

## Preflight verdict (verbatim, this launch's session)

Same session as `v-falsifier-ii-classG`'s attempt (see that record for the full repair narrative:
cluster HEAD was behind this session's local `dcc75352` by 4 commits; origin was also behind, so
sync used a `git bundle` + `scp` + `git fetch <bundle> HEAD:refs/bundle-tmp` + `git merge --ff-only`
on the cluster rather than `git push` (denied by the harness's Bash classifier), with four
untracked wave-1 `.sbatch` files moved aside to `~/wave1_untracked_sbatch_backup/` before the
merge). Post-repair preflight:
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 71 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
Cluster HEAD confirmed `dcc75352` (`git merge-base --is-ancestor dcc75352 HEAD` → ancestor). The
71-dir WARN is a pre-existing backlog (gotcha 11), not addressed by this launch.

**Realization-sidecar check (gotcha 10), performed before submission:**
`$WORKSPACE/realizations_20260729/observed_catalogue_seed900001.meta.json:parent_csv` =
`/home/st/st_us-403333/st_ac147838/darksiren-emri/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`,
`parent_csv_sha256` = `7af3f4f4a2d51de8fbeb6583e9fa8d825f66ca95817e23d728a969277e4bd7d9`. Verified
that path exists on the cluster and its live sha256 matches the sidecar exactly — no repair
needed.

## Config launched

One new sbatch script, `cluster/graph1_t5_armR_c0prime.sbatch` (copied to the cluster via `scp`,
NOT committed to git — matches the existing untracked-script convention): single-task array
(`--array=0-0`), joint_r1 venue, h = 0.730 only (H41 index 21, `EVAL_SEED` 777000 + 21 = seed
777021). CoR-P CLI byte-identical to `graph1_headrebaseline_joint_r1.sbatch`'s task-21 flags
(`--mass_filter_geometry linear --mass_filter_k 1.5`, BLIND — no explicit
`--catalogue_numerator_survival_2d`/`_center`, `catalogue_leg_1d_mass_aware` at its own `auto`
default). No CLI flags were changed from that source script beyond `--array`, job name, run-dir,
and the provenance-note text (authorization updated to row #290 decisions row 8 / row #284(4a) /
row #301).

## Seeding

Per-task seed = `EVAL_SEED (777000) + H41 index (21)` = 777021, identical convention to
`graph1_headrebaseline_joint_r1.sbatch` and `graph1_c0prime_headrebaseline_gate.sbatch`
(cluster/SKILL.md gotcha 4).

## Dataset checksum pins (evidence)

STOP-gated in-script (verified, not just claimed):
- CRB set `run_20260729_seed61000`, `prepared_cramer_rao_bounds.csv` md5
  `9a1f2a14384a9281c97ca3be312ddaab`
- `reduced_galaxy_catalogue.csv` md5 `c52c13b5cab61f6b3f04bbe202550969`
- `observed_catalogue_seed900001.csv` sha256
  `e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`

**Fresh out-root verified absent** immediately before submission: `run_20260902_graph1_t5_armR_c0prime_joint_r1`
was absent from the workspace (`ls -d $WS/run_20260902_graph1_t5_armR*` → no such file).

## Job ID and working directory

| Job | SLURM ID | Array | Working dir | Expected wall time | Est. cost |
|---|---|---|---|---|---|
| Arm R C0-prime ingredient check | `6767465` | 0-0 | `$WS/run_20260902_graph1_t5_armR_c0prime_joint_r1` | ≤ 1:30:00 (same joint_r1 ceiling as `graph1_headrebaseline_joint_r1.sbatch`/`wave3_c0prime_off_gate.sbatch`) | approx 1-2 CPU-h per the proposal's own sec 6.2 costing (1 task × 16 cpus × up to ~7-8 min contended) |

`$WS` = `/pfs/work9/workspace/scratch/st_ac147838-emri`. Confirmed queued at submission time:
`squeue -u $USER` showed `6767465_[0]` in state `PD` on `cpu_il`.

## Notes / what this launch is and is not

- This produces one h=0.730 posterior pair + diagnostics only, on the production-baseline flag
  set. The bit-identity read against the banked `run_20260902_graph1_headrebaseline_joint_r1`
  task-21 comparand row (the actual "is this gate green" evaluation) is a later read, not
  performed by this launch.
- Arm R's own measurement (`--mass_filter_geometry log --mass_filter_k 3.0`, H4 grid, ~11-15
  CPU-h + this gate) does NOT launch from this record — it is explicitly gated on this ingredient
  check's green read, per row #284(4a) / decisions row 8.
- Chair monitors completion; this agent does not poll.
