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
