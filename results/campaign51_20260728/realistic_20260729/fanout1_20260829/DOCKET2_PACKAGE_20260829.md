# Docket 2 Package (2026-08-29 / retrieved 2026-08-30)

**Launched under rows #222/#223 — charter fan-out 1 wave 3 prep, NODE package assembly.**
Citation-only index; no interpretation offered here. Nothing in this docket is an approval
request (row #222 form (ii)): the orchestrator chooses paths; the end-of-fan-out verifier
(registered: `fanout1_20260829/REGISTRATION_END_VERIFIER_PASS_20260829.md`) is the author check.
Repo `darksiren-emri`, branch `fix/p32d-classg-venue-repair`, HEAD `60f9996e` (wave-3 commit of
record); adoption commit `d4765539`.

---

## 1. Ledger rows #245–#251 (`gate_b_20260730/BIAS_HISTORY_LEDGER.md`)

| Row | Headline (first 200 chars) | Record file(s) cited |
|---|---|---|
| **#245** | "WAVE 2 CLUSTER SET LAUNCHED (fan-out 1; [FABLE-ORCH]) — C0 + C3 + C4 at ff230621: **submitted to bwUniCluster 3.0** (orchestrator, ~20:55 CEST) under charter nodes C0 / B5.2 (C3) / B7.2 (C4), launched under rows #222/#223." | none cited inline (launch-note row) |
| **#246** | "**C0 BASELINE GATE PASS (bit-identical) at `ff230621`** — banked HEAD readout is the wave-2 baseline; costing anchor corrected. SLURM job `6738998` COMPLETED (Elapsed 00:06:28, ExitCode 0:0), iiib venue, h=0.730 only. Gate" | none cited inline (see `COMPUTE_LEDGER.md` "C0 measured" section, `REGISTRATION_C0_BASELINE_GATE_20260829.md` §13) |
| **#247** | "Wave 2, charter node B5.2 [WIN] C3 log-k3 counterfactual READ OUT: INTERMEDIATE (Δmean_h,pred = +0.003523 via the I_HEAD stencil; between IMMATERIAL ≤ 0.003 and T_mat 0.008) — REPORTED, adoption gate NOT granted; R1 retenti[on]" | `fanout1_20260829/B5_2_WIN_K3_READOUT_RECORD.md`; `fanout1_20260829/b5_2_readout.json`; `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` §13; `fanout1_20260829/SYNTHESIS_DOCKET_1_20260829.md` (L10); `fanout1_20260829/COMPUTE_LEDGER.md`; `cluster/datasets.yaml` + `DATA_INVENTORY.md` (`run_20260829_wave2_c3_iiib`) |
| **#248** | "Wave 2, charter node B7.2 [2D-TWIN] C4 PROD-CF-2D READ OUT: IMMATERIAL-PREDICTED (Δmean_h,pred = +0.0025057 via the I_HEAD stencil, at or below T_mat/2 = 0.004; Δℓ′(0.665) = +7.429 nats per unit h; Δℓ″ = −30.3, far below 29[...]" | `fanout1_20260829/B7_2_TWIN_CF_READOUT_RECORD.md` §6; `fanout1_20260829/b7_2_readout.json`; `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §15; `fanout1_20260829/COMPUTE_LEDGER.md`; `cluster/datasets.yaml` + `DATA_INVENTORY.md` (`run_20260829_wave2_c4_iiib`) |
| **#249** | "Wave 2 pre-wave, charter node B4.2 [IMP] KW-Q1 READ OUT (independent reader): **KERNEL-WIDTH-INERT, REPORTED-ONLY** (\|R\| = 0.084812 ≤ the 0.2 INERT ceiling, an order of magnitude below the 0.5 OWNS floor; not adopted — carr[ies]" | `fanout1_20260829/B4_2_KWQ1_READOUT_RECORD.md`; `fanout1_20260829/b4_2_readout.json`; `CLAIM_IMPOSTOR_DRAG_20260829.md` §5; `fanout1_20260829/COMPUTE_LEDGER.md` |
| **#250** | "Wave 2 pre-cluster, charter node B1.1 [HIER] Stage 0 COMPLETE: S0-A returns B0-A-prime INSTRUMENT-DEFECT (Z_b = -3.68, Z_s = -7.08 no-BH; with-BH Z_b +0.38, Z_s -2.03; n = 461, 4 seeds; ENG pass; PARITY not exact) — STOP pe[nding]" | `fanout1_20260829/B1_1_HIER_STAGE0_RECORD.md`; panel verification (sonnet/high, independent recomputation from banked CSVs/JSON only) |
| **#251** | "B1.1-F forensic filed: LOCALISATION VENUE-LAW / INSTRUMENT-FORM, not a hook-arithmetic defect. Independent numpy twin of the no-BH catalogue leg reproduces `L_cat_no_bh` at the truth node to 9.2e-13 max \|Delta ln L\| and the" | `fanout1_20260829/B1_1_S0A_DEFECT_FORENSIC_20260829.md` (secs 0, 5, 6, 7); panel verification (sonnet/high) |

Row #251 is the current last row in the ledger (verified: no row #252+ present at retrieval).

---

## 2. Record files created or appended under `fanout1_20260829/` since 18:00 (`ls -lt`, top level)

| Modified (local) | Size (bytes) | File |
|---|---:|---|
| 2026-08-30 00:27 | 26182 | `COMPUTE_LEDGER.md` |
| 2026-08-30 00:19 | 26750 | `B1_1_S0A_DEFECT_FORENSIC_20260829.md` |
| 2026-08-30 00:15 | (dir) | `b1_1_forensic_work/` |
| 2026-08-30 00:04 | 43755 | `REGISTRATION_END_VERIFIER_PASS_20260829.md` |
| 2026-08-29 23:53 | 15871 | `B7_3_ADOPTION_VERIFIER_REPORT.md` |
| 2026-08-29 23:40 | 5005 | `B7_3_ADOPTION_IMPLEMENTATION_RECORD.md` |
| 2026-08-29 23:39 | 76962 | `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` |
| 2026-08-29 23:34 | (dir) | `__pycache__/` |
| 2026-08-29 23:34 | 72942 | `hier_s0_driver.py` |
| 2026-08-29 23:23 | 41392 | `CLAIM_IMPOSTOR_DRAG_20260829.md` |
| 2026-08-29 23:23 | 4734 | `b4_2_readout.json` |
| 2026-08-29 23:22 | 11008 | `B4_2_KWQ1_READOUT_RECORD.md` |
| 2026-08-29 23:15 | (dir) | `kwq1_registered_run/` |
| 2026-08-29 23:15 | (dir) | `kwq1_parity_run/` |
| 2026-08-29 22:57 | 21883 | `B1_1_HIER_STAGE0_RECORD.md` |
| 2026-08-29 22:00 | 38636 | `B1_2_DRIVER_EXTENSION_NOTE.md` |
| 2026-08-29 21:38 | (dir) | `hier_s0_registered_run/` |
| 2026-08-29 21:31 | 19906 | `B7_2_TWIN_CF_READOUT_RECORD.md` |
| 2026-08-29 21:30 | 62662 | `PROPOSAL_2D_TWIN_ADOPTION_20260829.md` |
| 2026-08-29 21:29 | 3298 | `b7_2_readout.json` |
| 2026-08-29 21:25 | 13879 | `b7_2_readout.py` |
| 2026-08-29 21:23 | 54822 | `SYNTHESIS_DOCKET_1_20260829.md` |
| 2026-08-29 21:19 | 39388 | `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` |
| 2026-08-29 21:19 | 8930 | `B5_2_WIN_K3_READOUT_RECORD.md` |
| 2026-08-29 21:18 | 22814 | `b5_2_readout.json` |
| 2026-08-29 21:16 | (dir) | `hier_s0_smoke_fix/` |
| 2026-08-29 21:05 | 22135 | `REGISTRATION_C0_BASELINE_GATE_20260829.md` |
| 2026-08-29 20:45 | 10627 | `COMMIT_PLAN_3.md` |
| 2026-08-29 20:44 | 6923 | `README.md` |
| 2026-08-29 20:14 | 10477 | `P6_THETA_CLI_PLUMBING_RECORD.md` |
| 2026-08-29 20:11 | 45489 | `B8_2_HARNESS_DESIGN_20260829.md` |
| 2026-08-29 20:11 | 4052 | `B4_2_KWQ1_RUN_FORM_NOTE.md` |
| 2026-08-29 20:10 | 33888 | `B1_1_HIER_RECORD.md` |
| 2026-08-29 20:09 | 23345 | `B3_1_POP_RECORD.md` |
| 2026-08-29 20:03 | 8184 | `COMMIT_PLAN_2.md` |
| 2026-08-29 19:56 | 44475 | `WAVE2_REGISTRATION_CHECK_20260829.md` |
| 2026-08-29 19:38 | 53573 | `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` |
| 2026-08-29 19:38 | 7440 | `B3_2_POP_FLAG_RECORD.md` |
| 2026-08-29 19:22 | (dir) | `hier_s0_work/` |
| 2026-08-29 19:22 | 19423 | `kwq1_score.py` |
| 2026-08-29 19:21 | 4936 | `B7_2_FALSIFIER_I_RECORD.md` |
| 2026-08-29 19:18 | 18480 | `B5_2_PULL_READ_20260829.md` |
| 2026-08-29 19:16 | 24089 | `b5_pull_read.json` |
| 2026-08-29 19:16 | 9417 | `b5_pull_read.py` |
| 2026-08-29 19:12 | 33682 | `PREREGISTRATION_CMEM_A1_20260829.md` |
| 2026-08-29 19:03 | 8432 | `COMMIT_PLAN.md` |
| 2026-08-29 18:46 | 15557 | `b8_information_floor.json` |
| 2026-08-29 18:40 | (dir) | `verify_b51/` |
| 2026-08-29 18:37 | 6172 | `b3_pop_prediction.json` |

(Nested run-tree outputs under `hier_s0_registered_run/`, `hier_s0_work/`, `kwq1_registered_run/`,
`kwq1_parity_run/`, `hier_s0_smoke_fix/`, `b1_1_forensic_work/`, `cmem_a1_work/` are not
individually itemized here — they are per-node simulation artifacts referenced by the node
records above, not separate docket entries.)

---

## 3. Compute ledger totals (`fanout1_20260829/COMPUTE_LEDGER.md`)

**Wave 1 (measured):** wave total ≈35 CPU-h local estimated; **≈11.4 CPU-h measured** at report
time (B1.1 ~11.2 partial/in-progress + B2.1/B3.1/B4.1/B6.1/B7.1/B8.1 negligible; B5.1
zero-compute). Source: ledger rows #225–#233.

**Wave 2 (estimate → measured, per arm):**

| Arm | Estimate | Measured | Ratio | Source |
|---|---|---|---|---|
| C0 (baseline gate) | 15–23 CPU-h | **1.7 CPU-h** | job `6738998`, Elapsed 00:06:28 | ledger row #246 |
| C3 (log-k3) | 44–137 CPU-h | **4.97 CPU-h** | 9×–28× below estimate | ledger row #247 |
| C4 (PROD-CF-2D) | 60–105 CPU-h | **6.8 CPU-h** | ~9×–15× below estimate | ledger row #248 |
| C2 (M1-prior) | 45–69 CPU-h | **STRUCK — 0 CPU-h** | premise refuted at zero compute | "Node B3 closure" note |
| P2 (KW-Q1) | 8.4 CPU-h ("2.2"/unsmeared) | **6.152 CPU-h** | ≈27% below estimate | ledger row #249 |
| P0 (S0-A/S0-C) | ≈5 CPU-h / 40 min | **S0-A 11.5 CPU-h; S0-C 10.42 CPU-h** | S0-C marginal quoted 24.4 s/h-node, flagged for reconciliation, not re-derived this pass | ledger rows #250–#251 |
| C1 (S0-B) | 60–92 CPU-h (unsmeared) | **NOT LAUNCHED** | held pending B1.1-F forensic (row #250 path decision); `wave2_c1_s0b_TEMPLATE.sbatch` remains a template | ledger rows #250–#251 |

**Anchor correction [A11]:** the pre-launch 56–76 min/h-value anchor (`cluster/LAUNCHING_JOBS.md:47`)
was built from the **3355-event** production set, not the **1588-event** iiib venue actually run;
the banked HEAD readout job `6725283` (same iiib venue) shows per-task Elapsed 00:00:18–00:42:26
across 41 tasks — the anchor is roughly **8×–140×** too high for this venue. Consequence: C3/C4/C1
wave-2 estimates elsewhere in the ledger are loose upper bounds, not tight estimates, for iiib;
not retracted or re-computed. Source: `REGISTRATION_C0_BASELINE_GATE_20260829.md` §13; ledger row
#246 ("C0 measured" section).

**Revised wave-2 total (informational, cell-level only):** local ≈144–489 CPU-h, cluster 179–357
(+120–173 conditional on a C0 FAIL); B8.2 harness build 130–475 CPU-h local (revised up from ≈6
CPU-h/sweep after N=1588 re-scoping). C0+C1+C3+C4 launched total after C2 strike: 13 tasks,
179–357 CPU-h estimated pre-launch — realized (C0+C3+C4 measured, C1 not launched): **13.47
CPU-h** (1.7 + 4.97 + 6.8).

---

## 4. Wave-3 plan (`cluster/WAVE3_SUBMISSION_NOTE_20260830.md`)

**Status:** BUILDER deliverable only — no `ssh`/`sbatch` run by this node (cluster access down for
this pass); `submit_wave3.sh` defaults to `DRY_RUN=1`. Not an approval request.

**Tasks:** two 41-task arrays (full `H_GRID_41`), 16 cpus/task, `cpu_il`:
- **iiib**: `--time=00:45:00`; CPU-h estimate from measured wave-2 anchors [4:34, 6:38] range →
  **49.9–72.5 CPU-h**.
- **joint_r1**: `--time=01:30:00`; iiib range × 2.2–3.0 multiplier → **109.8–217.6 CPU-h**.
- **TOTAL (82 tasks, both venues): 159.8–290.1 CPU-h** (measured-anchor estimate; the `--time`
  budget ceiling if fully consumed is 492 + 984 = 1476 CPU-h, worst case only).

**Blind falsifier band of record:** **A14, `T_mat` = 0.008** on |Δmean_h| (2D channel), on BOTH
venues, evaluated against a **separate** `--catalogue_numerator_survival_2d off` counterfactual
arm at the same wave-3 commit (not part of this delivery). Source:
`PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §8; note §1 item 7 of the pre-launch checklist.

**Baseline:** the banked 2026-08-27 HEAD readout, certified by **C0 PASS bit-identical** (ledger
row #246; `REGISTRATION_C0_BASELINE_GATE_20260829.md` §13) — commit `d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5`,
2026-08-27T19:40:20.

**Deliberate blindness (F2):** neither sbatch script passes `--catalogue_numerator_survival_2d`
or `_center` — the readout is blind to the row-#223 adoption by design; not to be "completed" by
adding the flag.

**8-item pre-launch checklist** (§1 of the note): (1) row-#223 `[PHYSICS]` commit is HEAD + pushed,
clean `git status`; (2) cluster checkout has pulled that exact commit (`ssh` HEAD match); (3)
`preflight.sh` → `VERDICT: READY ✓`; (4) archive-scheduled confirmation (`archive_run_wave2.sh`
"wave 3" `ITEMS` block); (5) dataset pins (CRB md5 `9a1f2a14384a9281c97ca3be312ddaab`, galaxy
catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`, joint_r1 observed-catalogue sha256
`e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`); (6) gotcha #10 realization
sidecar staleness check; (7) falsifier band stated (A14/`T_mat`=0.008); (8) fresh out-roots
verified absent pre-submit.

---

## 5. SSH outage status

Per task-level briefing (this docket assembled with **no ssh** available; cluster access has been
down since ~21:15 on 2026-08-29 and had not been restored as of retrieval, 2026-08-30). Consistent
with `WAVE3_SUBMISSION_NOTE_20260830.md`'s own statement ("no `ssh`/`sbatch` was run by this node
— cluster access is down for this pass") and `README.md`'s C1 sbatch note ("C1 sbatch is a
template only — θ flags commented out pending the P6 commit"). Blocked as a result:
- **C4 provenance extras** — not completed this pass (no cluster access to verify/append).
- **C1 template** — remains a template only, not converted to a live submission (P6 commit not
  yet landed on a cluster checkout that could be verified).
- **Wave-3 submission** — `submit_wave3.sh` stays in `DRY_RUN=1`; no `sbatch` executed; checklist
  item 2 (verify cluster HEAD matches local HEAD) and item 3 (`preflight.sh` READY) are both
  unverifiable while the outage persists.

---

## 6. Git state since `a794404c`

```
git log --oneline a794404c..HEAD
```

```
60f9996e docs: wave-2 local reads — S0-A B0-A′ INSTRUMENT-DEFECT (Z_b −3.68, Z_s −7.08; C1 not
         submitted), S0-C marginal 24.4 s/node, KW-Q1 KERNEL-WIDTH-INERT R = +0.085 (row #249;
         B4 does not merge into B1), driver §8/§8.1, wave-3 readout sbatch set
d4765539 [PHYSICS] 2D catalogue leg: adopt catalogue_numerator_survival_2d="mz_sel" (center "eff")
         as the production default — the with-BH catalogue-leg twin (charter B7.3, row #223)
0d0eb691 docs: wave-2 readouts + records — C0 gate PASS bit-identical (row #246), C3 log-k3
         INTERMEDIATE +0.0035 / retention transfer falsified (row #247), C4 2D-twin IMMATERIAL
         +0.0025 all gates PASS (row #248), B7.3 adoption gate presented (panel-clean), launch
         stamp (row #245), driver fix §8/§8.1, P1 full-N result, dataset registrations
ff230621 docs: fan-out 1 wave-2 prep — B3 PREMISE-REFUTED (provenance), registrations C0/C3/C4 +
         PA-HIER-31, pull read (L9), B8.2 design, chair check, depth-2 path decisions of record
a713c8b8 cluster: wave-2 sbatch set (C0 baseline gate, C3 log-k3 window, C4 2D-twin mz_sel; C1
         S0-B template) — CoR-P CLI verbatim, md5 STOP-gates, provenance, H4 arrays; archive list
         in results/_archive/archive_run_wave2.sh (gitignored, local)
cc305748 test: 2D-twin S_4D-homogeneity falsifier (i) — twin degree-0 under uniform S_4D
         rescaling, coded arrangement not; can fail (double-applied survival) — charter B7.2-pre
fb9d8aff cli: θ-hook flags --theta_b/--theta_s/--theta_sites on the production CLI (identity
         defaults, byte-identical) — charter B1.2 P6
dd63fe0c docs: fan-out 1 wave 1 — rows #224–#233, F1–F5 adopted, eight node records, synthesis
         docket 1, HIER Stage-0 driver
901653a1 validation: θ-hook + mass-window passthrough in run_mirror_seed_inprocess (identity
         defaults, byte-identical) — [HIER] Stage-0 driver plumbing, charter B1.1/B5.1
0b308828 [PHYSICS] mass window: instrument flag mass_filter_geometry ∈ {linear,log} +
         mass_filter_k (default byte-identical) — row #221 F-ii / row #223, charter B5.1
1f003da6 [PHYSICS] θ-hook: align s placement to [HIER] §1.2 — scale σ_z,raw before the σ_pv fold
         (row #221 item 4 / row #223, charter B6.1)
```

11 commits since `a794404c` (exclusive) through `60f9996e` (HEAD).

---

*Assembled 2026-08-30, no `ssh`, no `git commit`/`add`, append-only. Every number above carries
its own {value, source file:line, date} in the section it appears under.*
