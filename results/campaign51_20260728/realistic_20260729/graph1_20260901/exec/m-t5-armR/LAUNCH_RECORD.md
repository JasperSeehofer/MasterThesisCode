# m-t5-armR — LAUNCH RECORD

Research Graph 1, Branch F. Launched 2026-09-02 by the cluster launcher agent. **Arm R's own
measurement** (log-symmetric mass window k=3.0 vs the linear k=1.5 baseline, joint_r1 venue) — the
array named in the design's §6.2 as distinct from, and gated behind, the C0-prime ingredient check.

## Authorization

Ledger row #290 (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`),
decisions row 8: "Arm R launch strictly behind its own C0-prime-equivalent gate" (row #284(4a)),
plus row #301 (wave-2 launch authorization).

Design of record: `results/campaign51_20260728/realistic_20260729/tree2_20260830/
PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md` §6.2 (Arm R):

> "Form: the joint_r1 HEAD-readout CLI (headreadout_20260827/joint_r1/run_metadata_21.json:cli_args,
> observed_catalogue seed 900001) with mass_filter_geometry = log, mass_filter_k = 3.0, on the H4
> grid; baseline = the banked joint_r1 HEAD readout at the same nodes (zero compute) subject to the
> same C0-style ingredient gate on this venue ... Cost: approx 11-15 CPU-h (docket 2 section 7 rank
> 5; joint_r1 HEAD cost >= 2.2x iiib) + the C0-prime."

## Gate precondition (satisfied — quoted)

`exec/m-t5-armR-c0prime/eval/GATE_RECORD.md`:

> "**GREEN — g-c0-baseline-equivalent (Arm R) identity holds.**" ... "Bit-identity holds on every
> shared column of every named comparand artifact (both posterior JSONs by md5, the diagnostics CSV
> by max_abs=0 across 17 columns × 1588 rows). **GREEN.** No numbers are banked as a delta because
> there is no delta to bank." ... "If GREEN: Arm R's launch precondition (row #290 decisions row 8 /
> row #284(4a)) is **satisfied** by this record. The launch of Arm R's own measurement
> (`--mass_filter_geometry log --mass_filter_k 3.0`, H4 grid) is the chair's to dispatch, not this
> record's."

sacct evidence quoted in that record: job `6767465_0` — `COMPLETED 0:0`, single task, h=0.730, seed
777021, 0 non-`COMPLETED 0:0` records. This launch dispatches on that green.

## Cluster repo state at submission

- Local (session) HEAD: `dcb2c470472f2f1f912c166ab48c3890a410c42c` (descendant of `dcc75352` and of
  the `a26959b4` `[PHYSICS]` commit — "decouple h grid-admissibility from the host-window bound").
- Cluster HEAD checked before any action: `ssh bwunicluster 'git rev-parse HEAD'` → **already**
  `dcb2c470472f2f1f912c166ab48c3890a410c42c` (identical to local). No lock files found
  (`.git/*.lock` absent, no lock newer than 10 minutes), no sync in progress — **no pull/push
  performed by this launch.**
- Note on `a26959b4`: that `[PHYSICS]` commit decouples h-grid admissibility from the *host-window*
  bound — a 1D-guard change. It is unrelated to, and cannot affect, this 2D mass-window arm (Arm R
  varies `mass_filter_geometry`/`mass_filter_k`, a 2D catalogue-selection flag, not the 1D h-grid
  admissibility guard the commit touches). Using the newer commit is therefore safe for this launch.

## Preflight verdict (verbatim)

```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 72 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
(pre-existing backlog, unrelated to this launch — same WARN class as every prior graph-1 launch
record.)

## Config launched

New sbatch script `cluster/graph1_t5_armR.sbatch` (scp'd to the cluster, NOT committed to git —
matches the existing untracked-script convention for graph-1 exec scripts).

- CLI verbatim source: `cluster/graph1_headrebaseline_joint_r1.sbatch` (the joint_r1 HEAD-readout
  CLI the design's §6.2 names) / `cluster/graph1_t5_armR_c0prime.sbatch` (the gate script that
  reproduced its task-21 row bit-for-bit at the current commit). Every flag copied verbatim except
  `--mass_filter_geometry` (`linear` → `log`) and `--mass_filter_k` (`1.5` → `3.0`), which are
  exactly the design's registered arm variables. No flag not named in the design was changed.
- Full flag set: `--strategy physics-floor --pdet_dl_bins 60 --pdet_mass_bins 40
  --pdet_estimator local_linear --pdet_z_resolved --fisher_cond_threshold 1e16 --host_z_kernel
  volume_deconv --host_mass_kernel auto --normalization_mode absolute_marginal
  --selection_in_completion_numerator fused --catalogue_mass_overlap production
  --catalogue_mass_error_scale 1.0 --completion_b_scale derived --eddington_m on
  --sigma4d_mass_kernel point --completion_event_measure ratio --catalogue_global_selection phi
  --mass_filter_geometry log --mass_filter_k 3.0 --theta_b 0.0 --theta_s 1.0 --theta_sites all
  --observed_catalogue $WS/realizations_20260729/observed_catalogue_seed900001.csv`. BLIND to
  `--catalogue_numerator_survival_2d`/`_center` (left at CLI default), matching the
  headrebaseline/C0-prime convention exactly — no post-flip 2D default flag introduced that isn't
  in the design's cited CLI.
- Array shape: 4 tasks, H4 grid `{0.660, 0.665, 0.670, 0.730}` = H41 indices `{7, 8, 9, 21}` —
  identical H4 convention to `cluster/graph1_t5_armS_iiib.sbatch`. This is the grid the design's
  §6.2 text specifies ("on the H4 grid"); no other grid choice was available in the design text, so
  none was improvised.
- Baseline: **not re-run** (design: "baseline = the banked joint_r1 HEAD readout at the same nodes
  (zero compute)"). Comparand = `run_20260902_graph1_headrebaseline_joint_r1` tasks 7/8/9/21,
  code-state identity certified at task 21 by the C0-prime gate (`GATE_RECORD.md`); tasks 7/8/9
  share the identical CLI/code path and are not independently re-gated — this is the gate's own
  stated scope (code-state identity, not per-h numerics) and matches how the C0-prime gate itself
  was scoped (single h=0.730 node "ingredient check").

## Seeding

Per-task seed = `EVAL_SEED (777000) + H41 index`, H41 index ∈ `{7, 8, 9, 21}` (cluster/SKILL.md
gotcha 4) — identical convention to `graph1_headrebaseline_joint_r1.sbatch` and
`graph1_t5_armS_iiib.sbatch`.

## Dataset checksum pins (STOP-gated in-script, evidence)

- CRB set `run_20260729_seed61000`, `prepared_cramer_rao_bounds.csv` md5
  `9a1f2a14384a9281c97ca3be312ddaab`
- `reduced_galaxy_catalogue.csv` md5 `c52c13b5cab61f6b3f04bbe202550969`
- `observed_catalogue_seed900001.csv` sha256
  `e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`

**Fresh out-root verified absent** immediately before submission:
```
$ ssh bwunicluster 'ls -d $WORKSPACE/run_20260902_graph1_t5_armR_joint_r1'
ls: cannot access '.../run_20260902_graph1_t5_armR_joint_r1': No such file or directory
```
(Only the sibling C0-prime out-root, `run_20260902_graph1_t5_armR_c0prime_joint_r1`, pre-existed —
different name, already retrieved/gated, not reused.)

## Cost recomputation before submission

Design's registered estimate: "approx 11-15 CPU-h" for Arm R's own measurement (excluding the
C0-prime). Measured anchor from the same venue/code-state, this session: the C0-prime gate task
(`6767465_0`, single h=0.730 task, same joint_r1 CLI family) ran `00:06:37` wall on 16 cpus
(`sacct -j 6767465 --format=Elapsed,NCPUS`) = 0.1103 h × 16 = **1.77 CPU-h/task**. Four H4 tasks at
comparable per-task cost (log k=3.0 exercises the same window-evaluation code path as linear k=1.5
at comparable cost) → **≈7.1 CPU-h**, below the design's own 11-15 CPU-h band and well under the
15 CPU-h STOP threshold. No STOP triggered; launched at the design's specified `--time=01:30:00`
ceiling per task for margin (same ceiling as the C0-prime gate and the headrebaseline joint_r1
array).

## Job ID and working directory

| Job | SLURM ID | Array | Working dir | Est. cost |
|---|---|---|---|---|
| Arm R own measurement (log k=3.0, H4 grid) | `6768608` | 0-3 | `$WS/run_20260902_graph1_t5_armR_joint_r1` | ≈7.1 CPU-h (measured anchor); design band 11-15 CPU-h |

`$WS` = `/pfs/work9/workspace/scratch/st_ac147838-emri`. Confirmed queued at submission time:
`squeue -u $USER` showed `6768608_[0-3]` in state `PD` on `cpu_il`.

## Notes / what this launch is and is not

- This produces 4 posterior pairs + diagnostics (H4 grid, log k=3.0). The
  MATERIAL/INTERMEDIATE/IMMATERIAL-CONSISTENT-WITH-HB classification (design §6.2's registered
  predictions: true-host recovery gain among the 73 in-catalogue events, expected +12 to +16 hosts;
  Δmean_h,pred on the three-way map; the [+8, +20]-host falsifier band) and the bit-identity read of
  the unchanged 1D channel (R6) are a later read, not performed by this launch.
- No commits made by this launch. No git pull/push performed (cluster HEAD already matched local
  HEAD; verified before acting, not assumed).
- Chair monitors completion; this agent does not poll.
