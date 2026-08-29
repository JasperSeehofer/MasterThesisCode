# Compute Ledger (amendment F4)

Launched under rows #222/#223 — charter node Stamp. Append-only: new rows/waves append, do not
edit existing rows once measured values are filled.

Deadline gate: workspace expires **2026-09-23**. Any arm whose outputs would still be needed after
that date must be archive-scheduled (Option A rsync) before or immediately after it lands.

| Wave | Node | Arm | Estimate CPU-h | Measured CPU-h | Venue | Authorization | Archive-scheduled? | Deadline check (2026-09-23) |
|------|------|-----|----------------|-----------------|-------|----------------|---------------------|------------------------------|
| 1 | B1.1 | S0-A + registered S0-R + S0-C ([HIER] prereg §7.2 ceiling) | 35 | ~11.2 CPU-h consumed at report time, PARTIAL/IN-PROGRESS (S0-A seed900101 setup+truth+b_plus done, b_minus running; S0-C seed900101 running; S0-R not started, both remaining S0-A off-truth nodes and 3 more seeds not started) -- see `B1_1_HIER_RECORD.md` sec 2.5/6 for the measured per-node costs and the re-derived ~5.9h serial full-grid projection this session could not complete | local | rows #222/#223, ledger row #224 | Option A (in flight) | OK — local, no cluster workspace exposure |
| 1 | B2.1 | A1 | — | ~0.017 (single-core, wall 59.6 s vs 1800 s budget) | local | rows #222/#223, ledger row #224 | Option A (in flight) | OK |
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

## Notes (appended, 2026-08-29 — wave-2 PREP Notes worker, rows #222/#223)

- B2.1 measured wall time (59.6 s, `B2_1_CMEM_A1_RECORD.md:12`) filled into the wave-1 table above
  (single-core → ≈0.017 CPU-h); B5.1/B6.1/B7.1/B8.1 measured cells were already filled by an
  earlier append (rows further down this file) and are left unchanged here.
- Wave-2 pre-wave rows P0–P5 and cluster rows C0–C4 (from `SYNTHESIS_DOCKET_1_20260829.md` §4.3)
  are already present above under "Wave 2 (estimates only — not yet launched)", each carrying
  "archive-scheduled: pending" and the row #233 information-only authorization. No new rows added
  by this pass — this note confirms the table already satisfies the wave-2 row requirement rather
  than duplicating it.

## Wave 2 cost refinements (appended, 2026-08-29 — Records node, rows #222/#223)

Standing rule 1 (append-only) applies: the estimate cells in the "Wave 2 (estimates only)" table
above are left untouched; refinements land as new rows / this note, per arm, each citing the
document that produced the refinement.

| Wave | Node | Arm | Refined estimate | Note |
|------|------|-----|-------------------|------|
| 2 | C2 | M1-prior arm (B3.2) | **STRIKE — 0 CPU-h (do not run)** | `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §13 item 2 + `B3_2_POP_FLAG_RECORD.md`: premise REFUTED at zero compute (§F, generator provenance); PRESENTED WITH A STOP (`docs/gates/PHYSICS-GATE-LEDGER.md:98-99`); no code written, no `completion_population_prior` flag exists. `WAVE2_REGISTRATION_CHECK_20260829.md` row 3 concurs ("DEVIATE: strike C2; accept the STOP"). Original estimate 45–69 CPU-h superseded by this strike, cell above left as-is (append-only). |
| 2 | C1 | S0-B production θ-score (B1.2) | **60–92 CPU-h (unsmeared/`"2.2"` form only)** | `WAVE2_REGISTRATION_CHECK_20260829.md` §0 finding F-A + §11: `theta_sites="2.2"` unsmeared is the CoR-P-faithful form (`combined_no_bh` max_rel 7.45e-3 vs the smeared `"all"` form — NOT bit-identical, so P1's equivalence premise is refuted-in-part); the 81–113 CPU-h smeared band is **withdrawn** (item 6, "removed from the ledger as an option"). Original range 60–113 CPU-h above superseded for the registered form; cell left as-is (append-only). |
| 2 | C0+C1+C3+C4 | wave-2 cluster total (revised, C2 struck) | **179–357 CPU-h, 13 tasks** (was 224–447, 16 tasks) | `WAVE2_REGISTRATION_CHECK_20260829.md` §0 item 7: C0 15–23 + C1 60–92 + C3 44–137 + C4 60–105 = 179–357 CPU-h. Conditional **+120–173 CPU-h** if the shared baseline gate task C0 FAILS (C3/C4 each re-run their own 4-node baseline: +59.7–91.6 and +59.7–81.1 CPU-h respectively; `WAVE2_REGISTRATION_CHECK_20260829.md:107`, proposal §6.2 "H4 with full baseline re-run 119.4–162.2"). |
| 2 | pre-wave | P0 (S0-A/S0-C completion) | **≈5 CPU-h / 40 min** (re-scoped to the 2.2/unsmeared form; was 5–11 + ≤15 ceiling smeared) | `WAVE2_REGISTRATION_CHECK_20260829.md` §12/item (b): "re-scope P0 to the same 2.2/unsmeared form... instead of the smeared 11 CPU-h." Registered smeared b_plus node → REPORTED-ONLY, not the driver of P0's cost. |
| 2 | pre-wave | P1′ (new, optional) | **0.33 CPU-h** (one (0,1)-smeared node, ≈20 min local) | `WAVE2_REGISTRATION_CHECK_20260829.md` line 118: attributes the −12% `alpha_G_phi` shift to θ-form vs the smear switch; informational, not blocking. Not in the original P0–P5 list. |
| 2 | pre-wave | P2 (KW-Q1, [IMP] 4.2 read) | **8.4 CPU-h recommended** (`"2.2"`/unsmeared; was 8.4–13.7 range, `"all"`/smeared primary) | `WAVE2_REGISTRATION_CHECK_20260829.md` §0 item 9: `D̃^φ`/`α_G^φ` cancel identically in the KW-Q1 statistic `s_imp`, so F-A does not reach it — chair recommends the cheaper `"2.2"`/unsmeared form (8.4 CPU-h) with a run-record note that `s_imp` is form-invariant, rather than `CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3's `"all"`/smeared PRIMARY (13.7 CPU-h). |
| 2 | pre-wave | P6 (new, blocking) | not costed (θ CLI plumbing) | `WAVE2_REGISTRATION_CHECK_20260829.md` item (b): added as a blocking pre-wave item ahead of S0-B; no CPU-h estimate given in source (code-plumbing task, not a compute arm). |
| 2 (local, new) | B8.2 | two-channel calibration harness design → build ([A3]) | **130–475 CPU-h local** (was ≈6 CPU-h per 24-arm sweep in the docket) | `B8_2_HARNESS_DESIGN_20260829.md` bottom line + `WAVE2_REGISTRATION_CHECK_20260829.md` item 8: the docket's "≈6 CPU-h per sweep" anchor is a mirror-N (≈106–200 scored events) number; at production N=1588 with ≥100 universes the mandatory cells are ≈130–475 CPU-h local (13–46 h wall at 14 cores), bracketed because per-`evaluate()`-call N-scaling is UNMEASURED. This is a **new local row**, not previously in the wave-2 table; no deadline exposure (local, no cluster workspace). |

**Revised wave-2 total (informational, supersedes row #233's ≈244–487 CPU-h estimate at the cell
level only — no existing row edited):** pre-wave local ≈13.7+ CPU-h (P0 5 + P1′ 0.33 + P2 8.4,
excluding un-costed P6) + cluster 179–357 (+120–173 conditional) + the new B8.2 local row 130–475
⇒ **local ≈144–489 CPU-h, cluster 179–357 (+120–173 conditional)**, all figures sourced from
`WAVE2_REGISTRATION_CHECK_20260829.md` and `B8_2_HARNESS_DESIGN_20260829.md` as cited per row
above, 2026-08-29. Launch decision remains the orchestrator's (row #222 form (ii), information
only).

## Wave 2 archive-scheduled / GAP-6 closure (appended 2026-08-29 — wave-2 GAP-CLOSURE archive/notes
worker, launched under rows #222/#223 — charter node: NODE archive+minor-notes)

Per `WAVE2_REGISTRATION_CHECK_20260829.md` §5 item 6 (F4 forbids launch while an archive cell reads
"pending"). Standing rule 1 (append-only) applies; the estimate/measured cells of the tables above
are left untouched.

| item | value | source |
|---|---|---|
| C0 archive-scheduled | **yes** (`results/_archive/archive_run_wave2.sh`) | this note; script created this pass, run AFTER retrieval, not run by this node |
| C1 archive-scheduled | **yes** (`results/_archive/archive_run_wave2.sh`) | same |
| C3 archive-scheduled | **yes** (`results/_archive/archive_run_wave2.sh`) | same |
| C4 archive-scheduled | **yes** (`results/_archive/archive_run_wave2.sh`) | same |
| C1 smeared band | **struck** — only the 60–92 CPU-h unsmeared/`"2.2"` form is registrable; the 81–113 CPU-h smeared band (originally in the wave-2 estimate table row C1, and repeated in the "Wave 2 cost refinements" section above) is withdrawn per `WAVE2_REGISTRATION_CHECK_20260829.md` §0 finding F-A / §3 item 6 ("removed from the ledger as an option — it was priced on P1's now-refuted equivalence"). | `WAVE2_REGISTRATION_CHECK_20260829.md` §0 F-A, §3 item 6, §5 item 6 |

## Cross-reference note (appended 2026-08-29 — wave-2 GAP-CLOSURE worker, rows #222/#223, charter node C0 revision pass)

A refuter panel on `REGISTRATION_C0_BASELINE_GATE_20260829.md` flagged that the C0 row in the
original wave-2 estimates table above (line 42, Archive-scheduled = `pending`) and the C0 row in
the "Wave 2 archive-scheduled / GAP-6 closure" table above (line 99, Archive-scheduled = `yes`)
read as contradictory to a reader who finds line 42 first. Both rows are correct and unedited
(append-only, standing rule 1) — line 99 is the later, dated status-of-record for C0 (and
C1/C3/C4); line 42's `pending` reflects the estimate-only state at row #233 launch and is
superseded **in effect, not in text** by line 99. This note exists solely so a reader of either
row can find the other without a doc-wide scan; see `REGISTRATION_C0_BASELINE_GATE_20260829.md`
§11.4 for the full disposition.

New rows (conditional / local fallback costs named in §1.1/§5 item 6 of the registration check but
not previously itemized as ledger rows):

| Wave | Node | Arm | Estimate CPU-h | Measured CPU-h | Venue | Authorization | Archive-scheduled? | Deadline check (2026-09-23) |
|------|------|-----|----------------|-----------------|-------|----------------|---------------------|------------------------------|
| 2 | C0-FAIL fallback | conditional re-run of C3's and C4's own 4-node baselines if the shared C0 gate task FAILS at 1e-12 (C0 §1.1 GAP item; not a separate arm unless triggered) | **+120–173** (C3 +59.7–91.6, C4 +59.7–81.1) | | cluster (iiib) | rows #222/#223, this note | pending (only materializes on a C0 FAIL) | conditional; within budget if triggered once |
| 2 (local) | B8.2 | two-channel calibration harness build/design, S1–S3 local (overlaps wave 2, no cluster) | **130–475** (13–46 h wall at 14 cores) | | local | rows #222/#223, this note | n/a — local only, no cluster workspace exposure | OK — local, no deadline exposure |

Sources: `WAVE2_REGISTRATION_CHECK_20260829.md` §1.1 ("Fallback cost not in the ledger... add a
conditional row"; §3 item 7's +120–173 CPU-h figure with C3/C4 split from proposal §6.2 "H4 with
full baseline re-run 119.4–162.2") and §5 item 6 ("add... the B8.2 local row (130–475 CPU-h)");
`B8_2_HARNESS_DESIGN_20260829.md` §6 ("mandatory total ≈ 130–475 CPU-h local, 0 cluster, 13–46 h
wall"). Stamped: launched under rows #222/#223 — charter node NODE archive+minor-notes (GAP 6),
2026-08-29.

## Node B3 closure — C2 STRUCK (orchestrator decision 2026-08-29, charter node B3)

**Launched under rows #222/#223 — charter node B3.** Append-only; nothing above this note is
altered; no row is deleted or edited in place. This confirms and formally closes, from the B3
branch-verdict side, the C2 status already entered by the "Wave 2 cost refinements" table above
(row: `2 | C2 | M1-prior arm (B3.2) | STRIKE — 0 CPU-h (do not run)`) and by the
`WAVE2_REGISTRATION_CHECK_20260829.md` GAP-5 disposition.

**STATUS: STRUCK.** Row `2 | C2 | M1-prior arm (B3.2), 3 h-nodes {0.720, 0.730, 0.740} | 45–69 |
… | pending | must be set before sbatch` (Wave-2 estimates table, "M1-prior arm" line) and its
refinement-table counterpart ("STRIKE — 0 CPU-h") are both marked **STRUCK** by this note; neither
row is edited or removed (standing rule 1). C2 does not launch in wave 2.

**Reason (branch verdict of record):** **B3 CLOSED — PREMISE-REFUTED (provenance, zero compute)**
— `B3_1_POP_RECORD.md` superseding note (2026-08-29) and
`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §F/§13: the generating commit `03cfe80` draws
production dark hosts from `(1−f)·dVc/dz/(1+z)`, byte-identical to the estimator's own completion
prior (`bayesian_statistics.py:1203`), so the `completion_population_prior` flag the C2 arm would
exercise has no premise to test — building it costs 45–69 CPU-h for a NULL-BY-CONSTRUCTION or
generator-inconsistent counterfactual, not an instrument that closes a real mismatch. No code for
the flag exists (`B3_2_POP_FLAG_RECORD.md`; gate ledger row "PRESENTED WITH A STOP … NO CODE").

**Wave-2 cluster total after this strike (unchanged from the existing refinement-table row, restated
here for the closure record):** **C0 + C1 + C3 + C4 = 13 tasks, 179–357 CPU-h**
(C0 15–23 + C1 60–92 unsmeared-form + C3 44–137 + C4 60–105), **+120–173 CPU-h conditional** on a
C0 FAIL (already itemized above as the "C0-FAIL fallback" row). Source:
`WAVE2_REGISTRATION_CHECK_20260829.md` §3 item 7 / §4 row 9;
`SYNTHESIS_DOCKET_1_20260829.md` "L-lines re-cut" note, 2026-08-29.

REPORTED.

## Wave 2 — launch note, appended 2026-08-29

Launched under rows #222/#223 — charter nodes C0 / B5.2 (C3) / B7.2 (C4). Append-only; the
estimate rows in the "Wave 2 (estimates only...)" table above are not edited. Wave-2 commit of
record: `ff230621`. Submitted 2026-08-29 20:55 CEST.

| Arm | Job ID(s) | Tasks | Out-root |
|---|---|---|---|
| C0 | `6738998` | 1 (h=0.730) | `$WS/run_20260829_wave2_c0_iiib` |
| C3 | `6738999` | array 0-3 (H4 grid {0.660, 0.665, 0.670, 0.730}) | `$WS/run_20260829_wave2_c3_iiib` |
| C4 (smoke) | `6739000` | array 0 (h=0.730) | `$WS/run_20260829_wave2_c4_iiib` |
| C4 (remainder) | `6739001` | array 1-3, `--dependency=afterok:6739000` | `$WS/run_20260829_wave2_c4_iiib` |
| C1 | not submitted | — | held for P0 completion (driver defect, `hier_s0_driver.py:647`, `--jobs>1` node loop) |

C2 remains STRUCK (row above, "Node B3 closure"). Deadline check: workspace expires 2026-09-23
(~24 days out at launch) — OK. Archive-scheduled: yes for C0/C1/C3/C4 (unchanged from the
GAP-6 closure append above). Source: `cluster/WAVE2_SUBMISSION_NOTE_20260829.md`; ledger row
#245.

## C0 measured — appended 2026-08-29

Launched under rows #222/#223 — charter node C0. Append-only; the estimate rows above
(§ "Wave 2 (estimates only...)" table, C0 row: 15–23 CPU-h) are not edited.

**C0 measured: 1.7 CPU-h** (SLURM job `6738998`, Elapsed 00:06:28 × 16 cpus/task, 1 task,
COMPLETED 2026-08-29), against the 15–23 CPU-h estimated pre-launch.

**Anchor correction [A11].** The pre-launch estimate was built from the 56–76 min/h-value
anchor in `cluster/LAUNCHING_JOBS.md:47`, which is measured on the **3355-event** production
set. This run (and the iiib venue generally, at 1588 events) is not that population: job
`6725283`'s (the banked HEAD readout, same iiib venue, same 1588-event population) own per-task
Elapsed ranged 00:00:18 (task 21, h=0.730 — the exact point C0 reused) … 00:42:26 (task 13)
across its 41-task array, i.e. the 56–76 min anchor is roughly 8–140x the actual per-task
elapsed observed on this population. C0's realized 6.5 min/1.7 CPU-h is consistent with the
low end of that observed range, not with the 3355-event anchor.

**Consequence for C3/C4/C1 estimates.** The C3 (44–137 CPU-h), C4 (59.7–105 CPU-h), and C1
(60–92 CPU-h) figures elsewhere in this ledger (Wave-2 estimates / refinement tables, and row
#245's launch note) are built from the same 56–76 min/h-value, 3355-event anchor and should be
treated as loose upper bounds, not tight estimates, for the 1588-event iiib venue — they are not
retracted or re-computed here (each arm has its own per-h-value multiplicity and flag surface
that C0's single-h read does not cover), but a reader costing the remaining wave-2 arms should
re-derive from `6725283`'s per-task Elapsed range rather than from `LAUNCHING_JOBS.md:47` alone.

Source: `REGISTRATION_C0_BASELINE_GATE_20260829.md` §13 (RESULT RECORD); ledger row #246.

## C3 measured — appended 2026-08-29

Launched under rows #222/#223 — charter node B5.2. Append-only; the estimate rows above
(§ "Wave 2 (estimates only...)" table, C3 row: 44–137 CPU-h) are not edited.

**C3 measured: 4.97 CPU-h** (SLURM job `6738999`, array 0-3, all 4 tasks COMPLETED
2026-08-29, exit 0:0) — per-task: h=0.660 job `6739003` elapsed 00:04:50 (1.289 CPU-h),
h=0.665 job `6739004` elapsed 00:04:39 (1.240 CPU-h), h=0.670 job `6739005` elapsed 00:04:36
(1.227 CPU-h), h=0.730 job `6738999` elapsed 00:04:34 (1.218 CPU-h), all at 16 cpus/task —
against the 44–137 CPU-h estimated pre-launch: **9×–28× below estimate**.

Consistent with the [A11] anchor correction already on record (row #246 / "C0 measured"
section above): the 56–76 min/h-value anchor (`cluster/LAUNCHING_JOBS.md:47`) is drawn from
the 3355-event set, not the 1588-event iiib venue this arm actually ran on; C3's realized
~4.5–4.8 min/h-value is consistent with the low end of the per-task Elapsed range observed on
job `6725283` (the banked HEAD readout, same iiib venue).

Source: `fanout1_20260829/b5_2_readout.json` `cost_F4`; `fanout1_20260829/B5_2_WIN_K3_READOUT_RECORD.md`
§5; `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` §13; `BIAS_HISTORY_LEDGER.md` row #247.

## C4 measured — appended 2026-08-29

Launched under rows #222/#223 — charter node B7.2. Append-only; the estimate rows above
(§ "Wave 2 (estimates only...)" table, C4 row: 60–105 CPU-h) are not edited.

**C4 measured: 6.8 CPU-h** (SLURM job `6739000` task 0, h=0.730, Elapsed 00:06:25; job
`6739001` tasks 1–3, h=0.660/0.665/0.670, Elapsed 00:06:38/00:06:17/00:06:10, all 4 tasks
COMPLETED 2026-08-29, 16 cpus/task) — against the 59.7–105 CPU-h estimated pre-launch:
**~9×–15× below estimate**. STEP-2 overhead pin: task-0 (h=0.730) 385 s vs C0's baseline
388 s (same h, same venue, `off` arm) ⇒ measured overhead factor ≈ 0.99×, inside the
registered 1.0–1.3× assumed band.

Consistent with the [A11] anchor correction already on record (row #246 / "C0 measured"
section above): the 56–76 min/h-value anchor (`cluster/LAUNCHING_JOBS.md:47`) is drawn from
the 3355-event set, not the 1588-event iiib venue this arm actually ran on; C4's realized
~6.3–6.6 min/h-value is consistent with the low end of the per-task Elapsed range observed on
job `6725283` (the banked HEAD readout, same iiib venue).

Source: `fanout1_20260829/b7_2_readout.json`; `fanout1_20260829/B7_2_TWIN_CF_READOUT_RECORD.md`
§6; `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §15; `BIAS_HISTORY_LEDGER.md` row #248.

## P2 (KW-Q1) measured — appended 2026-08-29

Launched under rows #222/#223 — charter node B4.2 (independent reader). Append-only; the
estimate rows above (§ "Wave 2 cost refinements" table, P2 row: 8.4 CPU-h recommended
`"2.2"`/unsmeared) are not edited.

**P2 (KW-Q1) measured: 6.152 CPU-h** — main registered run (4 seeds × 3 s-nodes × 2 h,
`--jobs 1`) wall 1417.786 s × 14 cpus = 5.514 CPU-h; T-ID/PARITY re-evaluation (1 seed, truth
node) wall 164.070 s × 14 cpus = 0.638 CPU-h; total 6.152 CPU-h — against the registered
8.4 CPU-h (`"2.2"`/unsmeared) estimate: **≈27 % below estimate** (0.73×), local, no cluster
exposure. A first scorer invocation (path mismatch, `node_*_ft/` instead of the actual
`node_*_ft_sites2.2_nosmear/` directories) found 0 rows and is excluded from this cost as a
runner-side invocation error, not a measurement.

Source: `fanout1_20260829/hier_s0_registered_run/logs/runner3_wave2pre_20260829.log`
(`wall_s`/`cpu_per_job` JSON blocks under `2026-08-29T22:49:07+02:00 START KW-Q1 ft ...` and
`2026-08-29T23:12:46+02:00 START KW-Q1 parity ...`); `fanout1_20260829/b4_2_readout.json`
`cost_measured`; `fanout1_20260829/B4_2_KWQ1_READOUT_RECORD.md` §8;
`BIAS_HISTORY_LEDGER.md` row #249.

## P0 (S0-A/S0-C) measured — appended 2026-08-29 (row #250/#251 companion)

Launched under rows #222/#223 — charter node B1.1. Append-only; entries above (rows 11/41/76)
not edited.

**P0 measured (S0-A full pass, `hier_s0_registered_run`): 2960 s wall x 14 cpu_per_job = 11.5
CPU-h** (`s0a_full_output.json` `elapsed_s`; 4 seeds, sites2.2_nosmear, "bc" numerator, 20 cells
mean ~65 s/cell). **S0-C (seed 900101, 41-h grid, 12 cpu): 3125.1 s wall = 10.42 CPU-h**
(`s0c_full_output.json`); S0-C marginal cost quoted at 24.4 s/h-node in the B1.1 stage-0 record
and this pass's authorization text (not independently re-derived from the 41-h-grid timing
above in this pass -- flagged for reconciliation, not treated as a discrepancy).

**C1 (S0-B) NOT LAUNCHED.** Per row #250's ORCHESTRATOR PATH DECISION: B1 stops at 1.1 pending
the B1.1-F forensic; `cluster/wave2_c1_s0b_TEMPLATE.sbatch` remains a template (theta flags
commented out); no SLURM submission occurred for C1 this pass. The forensic (row #251) localises
the S0-A defect to VENUE-LAW/INSTRUMENT-FORM but does not lift the B0-A' STOP and does not
license C1 launch.

Source: `fanout1_20260829/B1_1_HIER_STAGE0_RECORD.md`; `fanout1_20260829/B1_1_S0A_DEFECT_FORENSIC_20260829.md`
sec 7; `BIAS_HISTORY_LEDGER.md` rows #250-#251.

**B7.3 adoption housekeeping (row #253, 2026-08-30):** [PHYSICS] adoption commit `d4765539` (catalogue_numerator_survival_2d="mz_sel", center="eff") ledger row filed retroactively (docket 2 §7 item 7); zero compute, no new run. Wave 3 (commit of record `60f9996e`, sbatch set built) remains BLOCKED on cluster SSH (down since ~21:15 on 2026-08-29) — not submitted.

## Local run archive (2026-08-30, orchestrator; verifier F4 item)

The registered local Stage-0 (S0-A/S0-C) and KW-Q1 run directories (incl. the 41 gitignored *.log / simulations/ files) are archived locally as `results/_archive/local_runs/fanout1_stage0_kwq1_runs_20260830.tgz` (gitignored; sha256 recorded in the shell log of this session). Not in git; not on the cluster.
