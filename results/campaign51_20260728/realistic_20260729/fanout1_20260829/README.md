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
