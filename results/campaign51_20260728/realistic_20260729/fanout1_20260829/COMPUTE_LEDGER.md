# Compute Ledger (amendment F4)

Launched under rows #222/#223 — charter node Stamp. Append-only: new rows/waves append, do not
edit existing rows once measured values are filled.

Deadline gate: workspace expires **2026-09-23**. Any arm whose outputs would still be needed after
that date must be archive-scheduled (Option A rsync) before or immediately after it lands.

| Wave | Node | Arm | Estimate CPU-h | Measured CPU-h | Venue | Authorization | Archive-scheduled? | Deadline check (2026-09-23) |
|------|------|-----|----------------|-----------------|-------|----------------|---------------------|------------------------------|
| 1 | B1.1 | S0-A + registered S0-R + S0-C ([HIER] prereg §7.2 ceiling) | 35 | ~11.2 CPU-h consumed at report time, PARTIAL/IN-PROGRESS (S0-A seed900101 setup+truth+b_plus done, b_minus running; S0-C seed900101 running; S0-R not started, both remaining S0-A off-truth nodes and 3 more seeds not started) -- see `B1_1_HIER_RECORD.md` sec 2.5/6 for the measured per-node costs and the re-derived ~5.9h serial full-grid projection this session could not complete | local | rows #222/#223, ledger row #224 | Option A (in flight) | OK — local, no cluster workspace exposure |
| 1 | B2.1 | A1 | — | | — | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
| 1 | B3.1 | zero-compute measure-first read (re-derived M1-vs-comoving prediction, no cluster) | — | ~0.001 (local, ~5s wall) | local | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
| 1 | B4.1 | intake (top-tier) | — | | — | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
| 1 | B4.1 [IMP] part 1 | decomposition on banked artifacts (zero `evaluate()` calls; pandas + log parsing only) | 0 | ~0.02 (single-core script, <2 min wall) | local | rows #222/#223 | Option A (in flight) | OK |
| 1 | B5.1 | flag + zero-compute count | 0 | | local | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
| 1 | B6.1 | lands first | — | | — | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
| 1 | B7.1 | proposal (top-tier) | — | | — | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
| 1 | B8.1 | closed-form + numeric Fisher floor on the real seed61000 CRB (N=1588), no `evaluate()` calls, deterministic (no RNG) | 0 | ~0.01 (single-core script, ~5 runs incl. debug, <2 min wall total) | local | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
| **1** | **wave total** | | **≈35 (local); 0 cluster** | | | ledger row #224 | | 25 days remaining at launch |

Notes:
- Per-node arm granularity for B2.1–B8.1 was not itemized with individual CPU-h estimates in the
  row #224 launch text (only the wave total ≈35 CPU-h local, zero cluster, was stated); those cells
  are left blank rather than invented. Fill in as each node's own record specifies its own arm cost.
- "Measured CPU-h" columns are blank at Stamp-node time (mechanical/records task only); runner
  agents for each node append their measured value into this same row, never overwrite the
  estimate column.
- Source: ledger row #224, `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md:3022`.

| 1 | B4.1 [IMP] part 2 | stage-0 intake + stage-1 forecast free reads (zero `evaluate()`; four pandas passes over banked mirror/production diagnostics CSVs) | 0 | ~0.1 (single-core, ≈6 min wall) | local | rows #222/#223 | Option A (in flight) | OK — nothing new on the cluster; KW-Q1 (B4.2) estimated 8.4 CPU-h local, registered in CLAIM_IMPOSTOR_DRAG_20260829.md §1.3 |
| 1 | B6.1 | ALIGN implement + targeted/full suite (local, no `evaluate()` production run) | — | ~0.05 (targeted 27/27 + full suite 1851 passed / 15 skipped, single-core wall) | local | rows #222/#223, ledger row #230 | Option A (in flight) | OK |
| 1 | B7.1 | proposal authoring (top-tier) + panel review (0 rounds) | — | negligible (no `evaluate()` calls; document + review only) | local | rows #222/#223, ledger row #231 | Option A (in flight) | OK |
| 1 | B8.1 | closed-form + numeric Fisher floor, chair re-run | 0 | ~0.01 (single-core, chair re-run byte-identical) | local | rows #222/#223, ledger row #232 | Option A (in flight) | OK |
| **1** | **wave total (measured)** | | **≈35 (local); 0 cluster** | **≈11.4 CPU-h measured at report time (B1.1 ~11.2 partial/in-progress + B2.1/B3.1/B4.1/B6.1/B7.1/B8.1 negligible; B5.1 zero-compute)** | local | rows #222/#223, ledger rows #225–#233 | Option A (in flight) | 25 days remaining at launch |

## Wave 2 (estimates only — not yet launched; from SYNTHESIS_DOCKET_1_20260829.md §4)

| Wave | Node | Arm | Estimate CPU-h | Measured CPU-h | Venue | Authorization | Archive-scheduled? | Deadline check (2026-09-23) |
|------|------|-----|----------------|-----------------|-------|----------------|---------------------|------------------------------|
| 2 | pre-wave | P0–P5 (S0-A/S0-C completion, θ_sites equivalence gate, KW-Q1, B5 mass-pull read, B7 S_4D-homogeneity test, B5.1/B6.1 `[PHYSICS]` commits) | ≈20–40 (local) | | local | row #233 (info only; not yet authorized to launch) | pending | OK — local only |
| 2 | C0 | shared baseline gate task (serves B3.2/B5.2/B7.2), h=0.730, iiib | 15–23 | | cluster (iiib) | row #233 (info only) | pending | must be set before sbatch (F4) |
| 2 | C1 | S0-B production θ-score (B1.2), 4 h-nodes (b±, s± at 0.730) | 60–113 (60–92 unsmeared / 81–113 smeared) | | cluster (iiib) | row #233 (info only) | pending | must be set before sbatch |
| 2 | C2 | M1-prior arm (B3.2), 3 h-nodes {0.720, 0.730, 0.740} | 45–69 | | cluster (iiib) | row #233 (info only) | pending | must be set before sbatch |
| 2 | C3 | log k=3 counterfactual (B5.2), H4 grid | 44–137 | | cluster (iiib) | row #233 (info only) | pending | must be set before sbatch |
| 2 | C4 | PROD-CF-2D `mz_sel`/`eff` (B7.2), H4 grid | 60–105 | | cluster (iiib) | row #233 (info only) | pending | must be set before sbatch |
| **2** | **wave-2 total (estimate)** | | **≈244–487 CPU-h (≈20–40 local + ≈224–447 cluster)** | | | not yet authorized | all pending | 25 d at wave-1 launch; wave 2 not yet scheduled |

Notes (wave 2): below the charter's 350–650 CPU-h band since B2.2 (105–265) is not triggered (B2 parked at
depth 1, row #226) and Stage P is not in wave 2. Not included: S0-R (disarmed), joint_r1 arms (≥2.2× cost),
B7 falsifier (ii) (208–286 CPU-h, returns separately per row #220), G27/G41 grids. This section is
information only per row #233 — the orchestrator decides whether/when to launch wave 2.
