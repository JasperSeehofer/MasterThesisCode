# Fan-out 1 — wave 1 records (2026-08-29) `[FABLE-ORCH]`

Charter: artifact `500fef3e` (mirror: `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_37.md` §2/§5).
Authorization: ledger rows #221 (depth-1 grant), #222 ([STANDING] all depths, verifier at the end),
#223 (production-default flips inside the tree). Every node record in this directory is append-only
and carries the stamp "launched under rows #222/#223 — charter node <X>".

Node records: `B<n>_<m>_<TAG>_RECORD.md` · registrations/proposals/gate presentations by their usual
`PREREGISTRATION_*` / `PROPOSAL_*` / `PHYSICS_CHANGE_*` names · compute ledger: `COMPUTE_LEDGER.md` (F4)
· wave synthesis: `SYNTHESIS_DOCKET_1_20260829.md`.

Ledger rows for wave 1: #225 (B1.1) · #226 (B2.1) · #227 (B3.1) · #228 (B4.1) · #229 (B5.1) ·
#230 (B6.1) · #231 (B7.1) · #232 (B8.1) · #233 (synthesis docket 1 filed, information only).

Ledger rows for wave-2 PREP (appended 2026-08-29 — Records node): #234 (B3.2 gate
presented + dispatch-declined) · #235 (B5.2-pre pull read + L9 reconciliation) · #236
(B7.2-pre falsifier (i) implemented + PASS, proposal note appended) · #237 (B8.2 design note
filed) · #238 (wave-2 registration completeness check filed, information only).

Ledger rows for wave-2 GAP-CLOSURE (appended 2026-08-29 — Records node, this pass): #239
(orchestrator path decisions of record, verbatim, filed for reference) · #240 (B3 CLOSED
PREMISE-REFUTED, C2 struck) · #241 (C0 registration filed, panel {"refuted":false,"rounds":1})
· #242 (PA-HIER-31 appended, panel {"refuted":false,"rounds":2}) · #243 (P6 θ CLI plumbing
implemented, not committed) · #244 (archive scheduling delivered + remaining GAP items 6-12
dispositioned).

## File index (as of the Records node, 2026-08-29)

**Node records**
- `B1_1_HIER_RECORD.md`, `B1_1_HIER_BUILD_NOTE.md` — [HIER] S0-A
- `B2_1_CMEM_A1_RECORD.md` — [CMEM] A1
- `B3_1_POP_RECORD.md` — [POP]
- `B4_1_IMP_RECORD.md`, `B4_1_IMP_DECOMPOSITION.md` — [IMP]
- `B5_1_WIN_RECORD.md` — [WIN]
- `B6_1_ALIGN_RECORD.md` — [ALIGN]
- `B7_1_TWIN_RECORD.md` — [2D-TWIN]
- `B8_1_CAL_FLOOR_RECORD.md` — [CAL]

**Registrations / proposals / gate presentations**
- `PREREGISTRATION_CMEM_A1_20260829.md`
- `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md`
- `PROPOSAL_2D_TWIN_ADOPTION_20260829.md`
- `CLAIM_IMPOSTOR_DRAG_20260829.md`

**Wave synthesis / ledgers**
- `SYNTHESIS_DOCKET_1_20260829.md` — wave-1 verdict table, depth-2 recommendations, wave-2 batch proposal
- `COMPUTE_LEDGER.md` — measured wave-1 CPU-h + wave-2 estimates
- `COMMIT_PLAN.md` — Records node's proposed commit split (git status, du, 3 file lists)

**Wave-2 PREP node records / registrations (appended 2026-08-29, rows #234–#238)**
- `B3_2_POP_FLAG_RECORD.md` — [POP] B3.2 dispatch: implementation DECLINED, STOP already on
  record (`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §13 item 2); no code written
- `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` — gate presentation, PRESENTED WITH A STOP
- `B5_2_PULL_READ_20260829.md`, `b5_pull_read.py`, `b5_pull_read.json` — [WIN] B5.2-pre true-host
  mass pull distribution + L9 reconciliation (zero-compute read)
- `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` — [WIN] B5.2 k=3 log-window counterfactual
  registration (stage 2, no CPU-h spent)
- `B7_2_FALSIFIER_I_RECORD.md` — [2D-TWIN] B7.2-pre falsifier (i) implementation + run (PASS,
  52/52 tests); proposal §13 note appended
- `B8_2_HARNESS_DESIGN_20260829.md` — [CAL] B8.2 design note for the two-channel calibration
  harness ([A3]); input to a stage-2 registration, not itself a registered band
- `WAVE2_REGISTRATION_CHECK_20260829.md` — wave-2 PREP chair registration-completeness check
  (information only); source of the cost refinements folded into `COMPUTE_LEDGER.md`
- `kwq1_score.py` — KW-Q1 statistic implementation (B4.2 read instrument)
- `hier_s0_registered_run/`, `hier_s0_work/` — [HIER] S0-A/S0-C run artifacts (P0/registered-run
  logs and per-node `event_likelihoods.csv` outputs)

**Scripts + data artifacts (per node)**
- B3: `b3_1_pop_measure.py`, `b3_pop_prediction.json`
- B4: `build_b4_imp_decomposition.py`, `decomp_analysis_eta2.py`, `b4_imp_stage1_forecast.py`,
  `b4_imp_stage1_production_o2.py`, `b4_imp_stage1_split.py`, and their JSON/CSV outputs
  (`b4_imp_decomposition.csv`, `b4_imp_eta2_by_seed.csv`, `b4_imp_recovery_by_arm.csv`,
  `b4_imp_stage1_events.csv`, `b4_imp_stage1_forecast.json`, `b4_imp_stage1_production_o2.json`,
  `b4_imp_stage1_split.json`)
- B5: `b5_window_count.py`, `b5_window_count_arm_jackknife.py` + their `.json` outputs
- B8: `b8_information_floor.py`, `b8_information_floor.json`
- B2: `cmem_a1.py`, `cmem_a1_result.json`, `cmem_a1_work/` (breakdown/gates/result JSON)
- B1: `hier_s0_driver.py`, `hier_s0_registered_run/` (logs + s0a/s0c seed900101 run dirs),
  `hier_s0_work/` (run_logs + s0a seed900101 dir)
- `verify_b51/` — refuter-report staging dir for B5.1 (empty at docket time, per §0 disclosure)
- `__pycache__/` — build artifact, not tracked

## Wave-2 GAP-CLOSURE additions (2026-08-29, rows #239-#244)

- `REGISTRATION_C0_BASELINE_GATE_20260829.md` — [registration] C0 shared baseline gate, 19-field
  (18 numeric) ≤1e-12 reproduction gate vs the banked `d04d9dc9` HEAD readout; §11 revision note
  addresses a 6-item refuter round (column count corrected 16→19 fields). Not yet launched
  (A22 stamp requires the wave-2 commit).
- `P6_THETA_CLI_PLUMBING_RECORD.md` — [HIER instrument path] `--theta_b`/`--theta_s`/
  `--theta_sites` CLI plumbing (`darksiren_emri/arguments.py`, `darksiren_emri/main.py`),
  byte-identical defaults, GATE T-ID; IMPLEMENTED, not committed.
- `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md` —
  PA-HIER-31 appended (S0-B registration; θ form, b-nodes ±0.033, bands, F3 predictions) +
  REVISION NOTE 1 (5-item refuter round, addressed append-only).
- `cluster/WAVE2_SUBMISSION_NOTE_20260829.md` + `cluster/wave2_c{0,1,3,4}_*.sbatch` +
  `cluster/submit_wave2.sh` — cluster submission wrapper (DRY_RUN=1 default); C1 sbatch is a
  template only (θ flags commented out pending the P6 commit).
- `darksiren_emri_test/test_theta_cli_forwarding.py`,
  `darksiren_emri_test/bayesian_inference/test_survival_2d_homogeneity_falsifier.py` — new test
  files (P6 forwarding tests; B7.2-pre falsifier (i), already counted in row #236).
- B3 [POP] arm CLOSED as PREMISE-REFUTED; C2 struck from wave 2 (row #240); docket L1/L4
  re-cut by an appended note on `B3_1_POP_RECORD.md` §3.
- Quality gate (this pass, foreground, HEAD `dd63fe0c` + uncommitted diff): ruff check clean,
  ruff format clean (70 files), mypy clean (70 files), pytest 1889 passed / 15 skipped / 27
  deselected, 0 failed, 169.56s, coverage 73.21%.
- `COMMIT_PLAN_3.md` — this pass's proposed 3-way commit split (cli / test / docs).
