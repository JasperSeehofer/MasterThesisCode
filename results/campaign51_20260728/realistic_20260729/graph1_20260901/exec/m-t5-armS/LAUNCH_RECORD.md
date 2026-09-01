# m-t5-armS — LAUNCH RECORD

Research Graph 1, Branch F. Launched 2026-09-02 (wave-1 fan-out) by the cluster launcher agent.
**Arm S only** — Arm R is explicitly out of scope for this launch, gated behind its own
C0-prime-equivalent ingredient check per decisions row 8's own text (not built here).

## Authorization

Ledger row #290 (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`),
decisions table row 8, quoted verbatim from the graph docket ratification:

> **Author ruling (verbatim): "all is ratified from the graph and the new graph structure looks
> awesome! thank you"** — ... **rows 3–11 [DO] APPROVED** — branch heads A–I trigger their first
> items (...; **T5 arms S and gated R**; ...)

Graph spec row 8 (`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.6):
"Arm S launch (design ratified: rows #278(2)/#284(4)); Arm R launch strictly behind its own
C0-prime-equivalent gate (row #284(4a))".

Design of record: `results/campaign51_20260728/realistic_20260729/tree2_20260830/
PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md` §6.1 (Arm S). Ratified per rows #278(2) ("ALIGNED →
RATIFIED: the 78.9% retention figure is retired as a design input") and #284(4) restating the
same. The design doc was found and read in full before building anything — no configuration was
improvised.

## Preflight verdict (verbatim)

Shared with m-head-rebaseline (same session, same cluster state at submission time):
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 65 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
(post the fast-forward pull to `1ec9514d` documented in the m-head-rebaseline LAUNCH_RECORD; the
65-dir WARN is a pre-existing, unrelated backlog — not a blocker). Lustre OST 5 (2026-08-31
blocker) confirmed active on all filesystems — see m-head-rebaseline's record for the full
`lctl`/`lfs df` evidence, identical for this launch.

## Cluster repo state at submission

- Branch: `fix/p32d-classg-venue-repair`, commit `1ec9514d`, `5e7fda16` an ancestor.
- Note: the ratified Arm S design deliberately holds `--catalogue_numerator_survival_2d off`
  EXPLICIT (see below) — Arm S is NOT reading the post-flip 2D default; only the mass-window
  geometry is under test here. This is a disclosed choice from the design doc itself (§6.1), not
  an oversight.

## Config launched

One new sbatch script (copied to `~/darksiren-emri/cluster/` via rsync, NOT committed to git):

**`cluster/graph1_t5_armS_iiib.sbatch`** — 16-task array (4 k-values × 4 H4 nodes).

- Form: verbatim the C3 form (`cluster/wave2_c3_win_k3.sbatch` CLI — CRB_SRC
  `run_20260729_seed61000`; catalogue md5 `c52c13b5...`; CRB md5 `9a1f2a14...`; EVAL_SEED
  777000 + H41 index {7, 8, 9, 21}; H4 grid `{0.660, 0.665, 0.670, 0.730}`).
- `--mass_filter_geometry log`, `--mass_filter_k` swept over `{2.0, 2.5, 3.5, 1000000.0}`.
  - k ∈ {2.0, 2.5, 3.5}: the three REQUIRED points of the ratified k-scan.
  - k = 1000000.0: the OPTIONAL k=∞ (no-window) anchor named in the graph spec and the design
    doc §6.1 ("k = infinity is an optional fifth point ... implemented via mass_filter_k = 1e6,
    both geometries converge to the all-True mask, invariant 2 of the physics-change doc R7").
    Included because it is cheap (one more 4-task H4 set) and explicitly registered.
  - k = 3.0 is **NOT** re-run — it is already banked from wave-2 job C3
    (`cluster/wave2_c3_win_k3.sbatch`, the +0.003523 point cited by the design doc as "a valid
    fourth point on the same curve").
- `--catalogue_numerator_survival_2d off` — HELD EXPLICIT at the C3 value (design doc §6.1: "so
  that the banked k=3 point ... is a valid fourth point on the same curve"; disclosed as a choice
  against the post-flip production default). `mass_filter_sigma` stays at the production
  "symmetric" default (not varied, per §8's A10 invariants).
- All other flags (host_z_kernel, host_mass_kernel, normalization_mode,
  selection_in_completion_numerator, catalogue_mass_overlap, completion_b_scale, eddington_m,
  sigma4d_mass_kernel, completion_event_measure, catalogue_global_selection) are the CoR-P /
  A10-invariant values, byte-identical to `wave2_c3_win_k3.sbatch`.

## Seeding

Per-task seed = `EVAL_SEED (777000) + H41 index`, H41 index ∈ {7, 8, 9, 21} for the H4 grid
(cluster/SKILL.md gotcha 4). TID → (k_index = TID/4, h_index = TID%4); each (k, h) pair writes to
its own `$BASE_RUN_DIR/<k_label>/` subdirectory to avoid posterior-filename collisions across k
values at the same h.

## Dataset checksum pins (evidence)

STOP-gated in every task:
- CRB set `run_20260729_seed61000`, `prepared_cramer_rao_bounds.csv` md5
  `9a1f2a14384a9281c97ca3be312ddaab`
- `reduced_galaxy_catalogue.csv` md5 `c52c13b5cab61f6b3f04bbe202550969`
- No observed-catalogue realization (iiib venue only; no `--observed_catalogue` flag, matching the
  C3 form).

**Fresh out-root verified absent** immediately before submission:
`run_20260902_graph1_t5_armS_iiib` absent from the workspace.

## Job ID and working directory

| Job | SLURM ID | Array | Working dir | Expected wall time |
|---|---|---|---|---|
| T5 Arm S k-scan | `6764463` | 0-15 | `$WS/run_20260902_graph1_t5_armS_iiib/{k2_0,k2_5,k3_5,kinf}/` | ≤ 3:00:00/task (C3 measured anchor: 4.97 CPU-h for 4 H4 tasks at k=3) |

`$WS` = `/pfs/work9/workspace/scratch/st_ac147838-emri`. Total estimated cost: ~20 CPU-h (4
k-sets × ~5 CPU-h/set; design doc §6.1's own estimate is "3 (or 4) x approx 5 CPU-h ... expected
total 15-20 CPU-h" — the 4th (k=∞) set nudges this to the high end of that range, still within the
graph spec's "approx 15-20 CPU-h" line for state candidate 6 to within the design's own disclosed
margin).

## Notes / what this launch is and is not

- Arm R (joint_r1, decisive k=3) is explicitly NOT launched. It requires its own
  C0-prime-equivalent ingredient check first (the wave-3 generic C0-prime, job 6746274, covered
  the 2D-twin check, not this arm's own configuration) — that gate is wave-2's responsibility, not
  this launch.
- This produces posteriors + diagnostics only per k-node. The MATERIAL/INTERMEDIATE/
  IMMATERIAL-CONSISTENT-WITH-HB classification per k (design doc §6.1's registered bands) and the
  scan-level verdict are computed by the orchestrator at readout, not by this launch.
- Chair monitors completion; this agent does not poll.
