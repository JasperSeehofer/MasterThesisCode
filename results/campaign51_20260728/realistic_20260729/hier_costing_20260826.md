# [HIER] COSTING RE-DERIVATION (D6) + CLUSTER ARRAY DESIGN (D4) — 2026-08-26

**Author:** subagent, orchestrator-directed (`[OPUS-ORCH 2026-08-26]` D1–D7) · **Status:**
DESIGN ONLY — zero compute run, nothing submitted. Supersedes the stale `[ORCH-COST]` line
in `docs/derivations/PROPOSAL_HIER_SELFCAL_20260825.md:49-55` per rule [A11]. Every numeric
claim below carries `{value, source (file:line), date}`; anything not directly quotable is
marked **NOT FOUND**, never estimated silently.

---

## 0. The stale line being replaced

`docs/derivations/PROPOSAL_HIER_SELFCAL_20260825.md:51` (dated 2026-08-25):

> "25 θ-nodes × the 12-seed mirror fleet ≈ **~50–100 CPU-h** as one cluster array"

This treats a (θ,seed) cell as costing ≈ 50–100/300 = **0.167–0.333 CPU-h/cell**, i.e. it
prices ONE likelihood evaluation per cell. §2 of the same document requires evaluating the
**h × θ grid** per event ("evaluate the existing per-event likelihood on an (h × θ) grid …
giving per-event L(h, θ) cubes") — an h-sweep is intrinsic to the instrument, not optional.
The stale line never priced that sweep. §2 below re-derives with the sweep priced in.

---

## 1. Measured timing anchors (per rule [A11]: {value, source, date})

| # | anchor | value | source | date |
|---|---|---|---|---|
| A | Production `evaluate()`, per h-value, full production catalogue | 56–76 min @ 3355 events / 16 cpus | `cluster/LAUNCHING_JOBS.md:47` | anchors dated 2026-07-03 (`LAUNCHING_JOBS.md:43`) |
| B | **Mirror-venue** `evaluate()`-equivalent, SINGLE arm = single h, 200 events | 64.996 s / 62.944 s (bc/bt arms) @ `--cpus-per-task=16` | `cluster/p3_2d_rhs2.sbatch:15-16`, sourced from `p3_2d_work/b{c,t}_900101_meta.json` `wall_time_s` | measured 2026-08-25/26 (P3-2D fleet run) |
| C | Same anchor, corroborating comment | "~64 s/arm (single-h) uncontended; 2 arms/task ≈ 128 s" | `cluster/p3_2d_fleet.sbatch:31-33` | 2026-08-25 |
| D | Per-task fixed overhead (host-pool/catalogue build), one-time | ~10–30 s | `cluster/p3_2d_rhs2.sbatch:38-39` | 2026-08-26 |
| E | Partition/walltime actually used, recent 2D-family arrays | `cpu_il`, `--cpus-per-task=16`, `--time=02:00:00` (24-task array); `cpu_il`, `--cpus-per-task=16`, `--time=01:00:00` (32-task array) | `cluster/p3_2d_fleet.sbatch:53-57`; `cluster/p3_2d_rhs2.sbatch:61-66` | 2026-08-25 / 2026-08-26 |
| F | Cross-task contention factor when packed >2 tasks/node on `cpu_il` | ~1.7× slower per task | `.claude/skills/cluster/SKILL.md` gotcha 6 (line ~62-65) | venue-transfer campaign anchor |
| G | Largest arrays actually submitted in this repo (precedent, not a hard limit) | `--array=0-48` (49 tasks); `--tasks_per_h 80` (80 tasks) | `cluster/venue_transfer.sbatch:19`; `cluster/LAUNCHING_JOBS.md:99-101` | undated (existing scripts) |
| H | Workspace expiry | 2026-09-23 | `results/campaign51_20260728/realistic_20260729/HANDOFF_20260730.md:179`; corroborated by launch-task's own "verified 2026-08-26" claim | 2026-07-30 doc / 2026-08-26 verification |
| I | Queue wait, any registered value | **NOT FOUND** — only qualitative "queue-wait banked per row #185" mentions with no attached number | `PREREGISTRATION_P3_2D_20260825.md:99-100`; `PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md:122,185` | — |
| J | Site `MaxArraySize`/`MaxSubmitJobs` for the regular `cpu_il` partition | **NOT FOUND** anywhere in `cluster/preflight.sh`, `cluster/SKILL.md`, `cluster/LAUNCHING_JOBS.md`, `cluster/README.md`, `cluster/cluster.env` | grep swept all five | — (only the *dev* QOS is documented: `dev_cpu_il` MaxSubmit 4/MaxRunning 1/30-min wall, `.claude/skills/cluster/SKILL.md` gotcha 5 — does not bound the regular `cpu_il` partition Stage F would use) |
| K | Row #185 exact wording | verbatim author grant: *"there is no reason for me not to use the cluster, but if we find that we constantly hit the fair share blockade. if thats not the case we can use it as much as possible"* — a qualitative grant, **not** a numeric "~2 CPU-h" threshold | `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md:2766` | 2026-08-24 |

**Derived per-h-point cost (mirror-venue regime, the correct proxy — HIER explicitly runs
"on the mirror venue, where truth-θ is known"**, `PROPOSAL_HIER_SELFCAL_20260825.md:17-18`,
same synthetic-universe/200-event regime as P3-2D's fleet, anchor B/C above):

```
cost_per_h_point_per_cell = 63.97 s (avg of 64.996/62.944) × 16 cpus / 3600 = 0.2843 CPU-h
```

**Derived per-h-point cost (production regime, anchor A — cited only as an upper-bound
sanity check, NOT the design proxy, since it prices a 3355-event production catalogue
HIER does not use):**

```
cost_per_h_point_production = (56..76 min / 60) × 16 cpus = 14.93..20.27 CPU-h
```

The two anchors are ~50-70× apart per h-point (n_events 200 vs 3355, ratio 16.8×) —
consistent with fixed per-h overhead (galaxy-catalogue BallTree lookups, host-pool build)
dominating at small n_events rather than pure linear scaling; not investigated further here
(out of scope for a costing pass).

---

## 2. Re-derived cost, both stages (D1)

### 2a. The undetermined inner multiplier — h-grid size

**NOT FOUND.** No document (`PROPOSAL_HIER_SELFCAL_20260825.md`, `STAGE_L_HIER_20260825.md`,
`STAGE_L_HIER_V86_READING_20260825.md`) specifies how many h-grid points the (h,θ) instrument
evaluates per θ-node. This is the single largest cost driver and is an open pre-launch
decision, not a costing-agent estimate. Per the launch task's own fallback instruction, cost
is given **per h-point** below; the author/reviewer must pin `n_h` before either stage is
priced as a final number.

### 2b. Cost formula (both stages), including the measured per-task overhead (anchor D)

```
cells(P) = 3×3 θ-grid × 4 seeds  = 36
cells(F) = 5×5 θ-grid × 12 seeds = 300

CPU-h(stage, n_h) = cells(stage) × 16 cpus × (n_h × 63.97 s + 30 s) / 3600
                   = cells(stage) × 0.2843·n_h + cells(stage) × 0.1333   [CPU-h]

Stage P: CPU-h(P, n_h) = 10.24·n_h +  4.80
Stage F: CPU-h(F, n_h) = 85.29·n_h + 40.00
```

### 2c. Illustrative totals at representative `n_h` (labeled illustrative — not measured;
`n_h` is not registered anywhere)

| n_h | meaning | Stage P (36 cells) | Stage F (300 cells) |
|---|---|---|---|
| 1 | one likelihood eval/cell — the stale line's IMPLICIT assumption (0.167–0.333 CPU-h/cell matches this regime almost exactly, cf. anchor B/C's 0.284 CPU-h) | 15.0 CPU-h | 125.3 CPU-h |
| 9 | a coarse h-subgrid (illustrative only) | 96.9 CPU-h | 807.6 CPU-h |
| 15 | mid illustrative point | 158.3 CPU-h | 1319.4 CPU-h |
| 41 | reuse of the full production evaluate grid (`cluster/evaluate.sbatch:9,56`, 41 points 0.60–0.86) | 424.4 CPU-h | 3537.0 CPU-h |

**Reading:** even at the cheapest non-trivial reading (`n_h=1`), Stage F alone (125.3 CPU-h)
already exceeds the stale proposal's full 50–100 CPU-h envelope — because the stale line
priced Stage F's 300-cell count against a per-cell cost with NO overhead term and (per
its own implicit-`n_h`=1 reading) still landed at the bottom of the same range only by
omitting the fixed per-task overhead this re-derivation adds (anchor D). At any h-grid
resolution actually adequate to characterize a posterior shape (`n_h≥9`), Stage F costs
**8–35× the stale estimate**. **This confirms D6: the proposal's line is stale and the
gap is fully explained — it never priced the h-sweep.**

### 2d. Per-task walltime and sbatch sizing (both stages, as a function of `n_h`)

```
per_task_wall_uncontended(n_h) = n_h × 63.97 s + 30 s
per_task_wall_contended(n_h)   = per_task_wall_uncontended(n_h) × 1.7   [anchor F, >2 tasks/node]
```

| n_h | uncontended | contended (×1.7, anchor F) |
|---|---|---|
| 1 | 94 s | 160 s |
| 9 | 606 s (10.1 min) | 1030 s (17.2 min) |
| 41 | 2653 s (44.2 min) | 4510 s (75.2 min) |

`--time=02:00:00` (the existing `p3_2d_fleet.sbatch:55` precedent) covers every `n_h ≤ 41`
even under full contention, with margin — reuse it unchanged rather than re-deriving a new
walltime per `n_h`.

---

## 3. D4 array design — flattened 1-D (theta-node, seed) index

### 3a. Task count and decode

```
N_THETA(P) = 9   (3×3 grid, row-major flat index over the θ=(b,s) grid)
N_SEED(P)  = 4
TASKS(P)   = 36            → --array=0-35

N_THETA(F) = 25  (5×5 grid)
N_SEED(F)  = 12
TASKS(F)   = 300           → --array=0-299 (see §3c on chunking)

theta_idx = SLURM_ARRAY_TASK_ID // N_SEED
seed_idx  = SLURM_ARRAY_TASK_ID %  N_SEED
(theta_b, theta_s) = THETA_GRID_FLAT[theta_idx]     # precomputed row-major (b,s) list
SEED = BASE_SEED_HIER + seed_idx                    # NOT + SLURM_ARRAY_TASK_ID — see 3b
```

### 3b. Why this decode is collision-free BY CONSTRUCTION (and is NOT a recurrence of
PA-2D-8's F8 defect)

- **The (theta_idx, seed_idx) mapping is a bijection on `[0, N_THETA·N_SEED)`** by
  elementary integer div/mod — trivially true for any `SLURM_ARRAY_TASK_ID` in range, no
  additional argument needed. This gives one array task per (θ,seed) cell, no two tasks
  ever decode to the same cell.
- **`SEED = BASE_SEED_HIER + seed_idx` deliberately repeats the SAME physical seed across
  every θ-node** (`seed_idx` ranges only 0..3 / 0..11, independent of `theta_idx`). This
  is correct BY DESIGN, not a bug: the (h,θ) grid instrument must score the SAME realized
  mirror-universe draw under every θ hypothesis for the θ-comparison to be meaningful
  (`PROPOSAL_HIER_SELFCAL_20260825.md:34`, "Truth-θ = (0, 1) on the mirror venue by
  construction" — a fixed draw, varying kernel). No accumulator ever sums contributions
  from two different `theta_idx` under one seed into a shared statistic — each `(theta_idx,
  seed_idx)` cell writes its own independent output JSON.
- **The PA-2D-8 / F8 defect (`PREREGISTRATION_P3_2D_20260825.md:284-286`,
  `cluster/p3_2d_rhs2.sbatch:76-80`) was a different failure mode entirely**: RHS2 ran
  MULTIPLE independent stochastic draw-chunks per task (`chunk i uses SEED+i inside the
  task`), and adjacent TASK_IDs' chunk-seed ranges overlapped because the stride between
  tasks (originally ×1) was smaller than the span each task's own internal chunk loop
  covered — the SAME nominal seed was drawn into TWO different tasks' accumulated sums,
  double-counting draws and invalidating the SE. **HIER's design has no internal
  multi-chunk draw loop** — one task performs exactly one deterministic mirror-fleet
  realization at its assigned `seed_idx`, then sweeps `theta` (already fixed per task by
  `theta_idx`) and the h-grid on top of that single fixed realization. There is no
  accumulator spanning multiple seeds within a task, so there is no F8-shaped collision
  class to stride against. (If a future revision adds within-task chunking — e.g. to hit
  a target N via repeated draws — the ×100-stride fix and the >100-chunk STOP from F8 must
  be re-applied; noted here so it is not silently reintroduced.)

### 3c. Array-size limits and chunking (D4 + task item 3)

**Site `MaxArraySize`/`MaxSubmitJobs` for the regular `cpu_il` partition: NOT FOUND**
(anchor J). The largest arrays actually run in this repo are 49 tasks
(`cluster/venue_transfer.sbatch:19`) and 80 tasks (`cluster/LAUNCHING_JOBS.md:99-101`) —
neither confirms nor bounds 300. **Recommendation: query the live limit before Stage F
submission** (`sacctmgr show qos cpu_il` or `scontrol show config | grep -i
MaxArraySize` during the mandatory `/cluster` preflight) rather than assume either way.

Pending that check, **chunk Stage F into 5 sub-arrays of 60 tasks each**, split cleanly
along the θ-grid's own row structure (5 θ-rows × 12 seeds = 60 tasks/sub-array,
`--array=0-59` per submission with a `THETA_ROW` env var offsetting `theta_idx` by
`ROW×5`) — each sub-array sits comfortably inside the 49/80-task precedent band (anchor
G), avoiding an untested 300-task single submission regardless of what the site limit
turns out to be. Stage P (36 tasks) needs no chunking — it is already inside the 49-task
precedent.

### 3d. sbatch header (both stages — reuse the `p3_2d_fleet.sbatch` precedent unchanged)

```
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-35        # Stage P
#SBATCH --array=0-59        # Stage F, per 60-task chunk (×5 submissions)
```
Justification: `--cpus-per-task=16` matches the measured mirror-fleet anchor's own
reservation (anchor B/C, `evaluate()`'s internal pool auto-scales to
`os.sched_getaffinity(0) - 2`, `cluster/p3_2d_fleet.sbatch:17-20`); `--time=02:00:00`
covers every `n_h ≤ 41` even fully contended (§2d).

### 3e. BASE_SEED_HIER — tentative, disjoint-checked against every registered range found

| range in use | owner |
|---|---|
| 900101–900124 | P3-2D fleet/companion (`cluster/p3_2d_fleet.sbatch:67`) |
| 960001 | `ca_rhs_scorer.py:296` `DEFAULT_SCORE_BASE_SEED` |
| 970001 | `ca_rhs_scorer.py:297` `DEFAULT_MC_BASE_SEED` |
| 980001–~983104 | RHS2 (`cluster/p3_2d_rhs2.sbatch:76-80`, capped `< 990001`) |
| ≥990001 | `F10C_REFERENCE_BASE_SEED` (`ca_rhs_scorer.py:1774`) |

Propose **`BASE_SEED_HIER = 940001`** (Stage F max seed = 940001+11 = 940012) — sits in
the unused gap between 900124 and 960001. **This is a proposal only, checked against the
ranges this grep sweep found — it is not a substitute for a full disjoint-seed-range
audit before commit**, matching the discipline the F8/`ca_rhs_scorer.py:1773` comments
already apply to every other registered range.

---

## 4. D3 — smear_sigma_z reconciliation (blocking; status of the check, not a resolution)

`smear_sigma_z: bool = False` (`darksiren_emri/bayesian_inference/bayesian_statistics.py:2664`)
gates a per-galaxy catalogue-column redshift smear: when `True` it requires and reads the
catalogue's `REDSHIFT_MEASUREMENT_ERROR` column into `z_err_all`
(`bayesian_statistics.py:2790-2798`) on a "point-evaluated path"; when `False`, `z_err_all`
is zeroed and unused (`bayesian_statistics.py:2799-2800`). This is a **per-galaxy, catalogue-
column-driven** mechanism. θ's proposed `s` (scatter-scale, `σ_z → s·σ_z`,
`PROPOSAL_HIER_SELFCAL_20260825.md:32`) is a **global multiplicative transform applied at
evaluate-time to every event's kernel**, not a per-galaxy catalogue column read. On the code
evidence alone these are mechanistically distinct (different input source, different
application point) — but whether they interact when both are live in the same evaluate()
call (e.g. does `s` multiply the already-smeared `z_err_all`, or does turning on `s`
implicitly require `smear_sigma_z=True`/`False` to avoid a silent double-application) is
**not resolved by this grep-only pass**. Per D3 this is a **PRE-LAUNCH BLOCKING** code-level
check (the PA-2D-6 stale-flag-pattern precedent — `PREREGISTRATION_P3_2D_20260825.md`
PA-2D-6/PA-2D-1 F7) and must be closed in code, with an instrument-level test, before either
array is submitted. Not attempted here — zero-compute scope.

---

## 5. Does Stage F fit the workspace lifetime? (item 4)

Workspace `emri` expires **2026-09-23**, zero extensions verified (anchor H) — **28 days**
from today (2026-08-26). Per-task duration is bounded (§2d, ≤ ~75 min contended even at
`n_h=41`), so no single task risks the deadline. Total CPU-h is what could — and at `n_h=41`
Stage F alone is ~3537 CPU-h (§2c), which at even a generous continuous 200-CPU concurrent
footprint (~12–13 nodes' worth of 16-cpu tasks running simultaneously, itself unconfirmed —
anchor I, no queue-wait/throughput data exists to certify achievable concurrency) would take
~18 wall-hours of CPU time — but **actual wall-clock depends on fair-share queue depth,
which is NOT FOUND/unmeasured** (anchor I). At `n_h≤9` (807.6 CPU-h) the budget is close to
the PA-2D-8 RHS2 precedent (72.8 CPU-h, `PREREGISTRATION_P3_2D_20260825.md:293`, ×~11) —
that precedent turned around within roughly a session-to-day cycle by the ledger's own
adjacent-row dating, so `n_h≤9` reads comfortably fittable inside 28 days. **At `n_h=41` the
fit is not certified — this needs a live queue-depth read at submission time, not an a
priori claim.** Recommendation: pin `n_h` low enough (≤9-15) for Stage F, or run Stage F in
the 5 chunked sub-arrays (§3c) sequentially with queue-time logged per chunk (per SKILL.md
gotcha 8's "log probe-vs-actual" discipline) so a mid-campaign fit/no-fit call can be made
before committing the remaining chunks.

---

## 6. Local-vs-cluster (item 5, row #185)

Row #185 (anchor K) is a qualitative "use the cluster freely, watch for fair-share" grant,
not a numeric threshold — the launch task's own "~2 CPU-h" framing is a paraphrase of that
grant's spirit, applied here as the working rule since no sharper numeric rule exists in the
ledger. Both stages clear it by a wide margin regardless of `n_h`: Stage P's floor
(`n_h=1`) is already 15.0 CPU-h and Stage F's floor is 125.3 CPU-h (§2c) — both are
cluster-first by construction, matching D1's design and the P3-2D/RHS2 precedent this design
reuses wholesale (§3d). A local run is not costed as a serious alternative: even ignoring
per-task overhead, Stage F's 300 cells run serially on a dev box at `n_h=1` alone would be
300 × 64 s ≈ 5.3 wall-hours single-threaded before any h-sweep is added, and the dev-box
`evaluate()` pool anchor (§1, anchor A) shows the same workload is far cheaper in CPU-hours
when parallelized across `cpu_il`'s per-task pool — there is no regime here where local beats
cluster.

---

## 7. Open items this costing surfaces (not resolved here — zero-compute scope)

1. **`n_h` (h-grid size per θ-node) is undetermined and must be pinned by the author/
   reviewer before either stage's CPU-h number is final** — the single largest lever in
   this whole costing (§2a, §2c).
2. **D3's smear_sigma_z/θ-s reconciliation is unresolved in code** (§4) — blocking per the
   orchestrator's own D3 designation.
3. **Site `MaxArraySize` is NOT FOUND** — needs a live check at submission time (§3c).
4. **Queue-wait/achievable-concurrency data does not exist anywhere in this repo** (anchor
   I) — Stage F's wall-clock-vs-workspace-deadline fit at `n_h≥15` cannot be certified
   without it (§5).
