# SYNTHESIS DOCKET 2 — fan-out 1, wave 2 cluster set + local Stage-0/KW-Q1 reads + the B7.3 adoption (2026-08-29 / filed 2026-08-30)

**Launched under rows #222/#223 — charter node: wave-2 synthesis chair (docket 2).**
**Purpose: INFORMATION ONLY (row #222 governance form (ii)). Nothing here is an approval request;
the orchestrator chooses paths; the registered end-of-fan-out verifier
(`REGISTRATION_END_VERIFIER_PASS_20260829.md`) is the author's check. Every number carries
{value; source file:line; date}. Append-only.** Repo `darksiren-emri`, branch
`fix/p32d-classg-venue-repair`, HEAD `60f9996e` (wave-3 commit of record, in sync with
`origin/fix/p32d-classg-venue-repair` at chair time); adoption commit `d4765539`. Chair
constraints honoured: foreground only, no `ssh`, no `git commit`/`add`, no source edits.

Chair's scoped package: `DOCKET2_PACKAGE_20260829.md` and every record it indexes —
`SYNTHESIS_DOCKET_1_20260829.md` (+ L-lines re-cut, L10), `WAVE2_REGISTRATION_CHECK_20260829.md`,
`REGISTRATION_C0_BASELINE_GATE_20260829.md` §1–§13, `B5_2_WIN_K3_READOUT_RECORD.md`,
`B7_2_TWIN_CF_READOUT_RECORD.md`, `B1_1_HIER_STAGE0_RECORD.md`, `B1_1_S0A_DEFECT_FORENSIC_20260829.md`,
`B4_2_KWQ1_READOUT_RECORD.md` (+ `B4_2_KWQ1_RUN_FORM_NOTE.md`, `CLAIM_IMPOSTOR_DRAG_20260829.md`
§1.3–§1.4/§5), `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §8/§9/§13,
`B7_3_ADOPTION_IMPLEMENTATION_RECORD.md`, `B7_3_ADOPTION_VERIFIER_REPORT.md` (+ addendum), commit
`d4765539` (`git show --stat`), `B3_1_POP_RECORD.md` superseding note + `B3_2_POP_FLAG_RECORD.md` +
`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §F/§12/§13, `B2_1_CMEM_A1_RECORD.md`,
`B8_2_HARNESS_DESIGN_20260829.md` §0/§5/§6/§8, `PREREGISTRATION_HIER_HTHETA_20260826.md`
PA-HIER-27..31 + REVISION NOTES 1–2 + P1 full-N note + STAGE-0 RESULT RECORD (`:1838-2683`),
`COMPUTE_LEDGER.md` (incl. the uncommitted P0 section), `cluster/WAVE3_SUBMISSION_NOTE_20260830.md`,
`cluster/wave3_headreadout_iiib.sbatch`, `B5_2_PULL_READ_20260829.md`, `B7_2_FALSIFIER_I_RECORD.md`,
`B8_1_CAL_FLOOR_RECORD.md` §3–§4, `README.md`, `COMMIT_PLAN_3.md` §4–§5, ledger rows #245–#251
(`gate_b_20260730/BIAS_HISTORY_LEDGER.md:3064-3094`, rows #250–#251 uncommitted), runbook 37 §2/§5,
`docs/RESEARCH_CYCLE.md:628-652` (F1–F5), `docs/gates/PHYSICS-GATE-LEDGER.md` tail, the
`hier_s0_registered_run/logs/runner{,2,3}_wave2pre_20260829.log` timestamps, `git reflog`.

---

## 0. Chair re-derivations (foreground, local, two pandas/numpy scripts in the session scratchpad; zero `evaluate()` calls; ≈ 2 CPU-min)

Where a record and a refuter (or a later record) disagree, the ONE decisive number was re-derived
from the raw artifact, not from the record's restatement.

| # | object | chair value | record value | match | source re-read |
|---|---|---|---|---|---|
| (a) | S0-A pooled Z (no-BH, N = 461, own script, dedupe keep-last, ln of `combined_no_bh`) | score_b −1.616461 ± 0.439682, **Z −3.676431**; score_s −0.086253 ± 0.012185, **Z −7.078607**; with-BH +0.379259 / −2.026806; dark class n = 5, max\|score\| = **0.0** | −3.6764 / −7.0786 / +0.3793 / −2.0268 / 0.0 | bit-level | `hier_s0_registered_run/s0a_seed9001{01..04}/node_*_sites2.2_nosmear/…/event_likelihoods.csv` (2026-08-30) |
| (b) | Forensic **E11** — the registered b-statistic with the no-BH divisor made θ-dependent by the per-node scalar ρ(θ) (`f12_out.json:rho_pool200k_robust`: b+ 0.953788, b− 1.043486, s+ 0.989305, s− 1.005931), exact per-event `(βL_cat/ρ + B_num)/D̃`, βL_cat = combined·D̃ − B_num, matched events only | score_b **−0.268151 ± 0.431469, Z −0.621484** (n 456); score_s −0.072758 ± 0.012178, Z −5.974341 | −0.268 ± 0.431 (Z −0.62); −0.0728 ± 0.0122 (Z −5.97) | all printed digits | same CSVs + `b1_1_forensic_work/f12_out.json` (2026-08-30) |
| (c) | KW-Q1 R from the scorer's S-values | (−0.9591134 − (−1.0456670))/1.0205308 = **+0.0848123** | +0.084812 | exact | `kwq1_registered_run/kwq1_score_output.json` (the readout record's own from-CSV re-derivation is the independent leg; the chair checked the ratio, the 0.2/0.5 bands and the per-seed max 0.156) |
| (d) | B5.2 stencil from the record's 4-dp Δℓ(h) (+0.5442 / +0.5972 / +0.6486) | Δℓ′ 10.44, Δℓ″ −64.0, **Δmean_h,pred +0.003521** | +0.003523 (full precision), Δℓ″ −63.7 | rounding-consistent; INTERMEDIATE either way | `B5_2_WIN_K3_READOUT_RECORD.md` §4; `b5_2_readout.json:primary_reading_delta_mean_h_pred` |
| (e) | B7.2 stencil from `stencil.per_node_delta_ell` (−3.030674 / −2.993148 / −2.956381) | Δℓ′ 7.42930, Δℓ″ −30.36, **Δmean_h,pred +0.0025057** | +0.0025057 | exact | `b7_2_readout.json:stencil` |
| (f) | C0 bit-identity re-diff (h = 0.73, 1588 rows, own join on `event_idx`) | max\|Δ\| = **0.0** on all 14 shared numeric columns; c0 carries 18 numeric columns (+3 OAT) vs 15 banked | same | exact | `wave2_20260829/c0/diagnostics/event_likelihoods.csv` vs `headreadout_20260827/iiib/event_likelihoods.csv` |
| (g) | S0-C marginal per h-node (seed 900101, 41 nodes, 12 cpu) | mean of the 40 post-first `per_h_delta_s` = **24.374 s**; (wall 2680.995 − first-h 1704.272)/40 = 24.418 s; S0-C 3125.111 s × 12 = **10.417 CPU-h**; S0-A 2959.625 s × 14 = **11.510 CPU-h** | 24.37 s; 10.42; 11.51 | ✓ — closes the compute ledger's "flagged for reconciliation" | `hier_s0_registered_run/s0c_full_output.json`, `s0a_full_output.json` |
| (h) | F4 totals (§4) | cluster 1.724 + 4.973 + 6.800 = **13.50 CPU-h**; local wave-2 6.152 + 11.510 + 10.417 = **28.08**; wave-1 11.4 ⇒ **≈ 53.0 CPU-h fan-out to date** | 13.47 (ledger, rounded per-arm) | ✓ | sacct Elapsed × cpus as cited in `COMPUTE_LEDGER.md` C0/C3/C4/P2/P0 sections |
| (i) | wave-3 estimate | 41×16×[274, 398] s/3600 = 49.93–72.52 (iiib); ×[2.2, 3.0] = 109.84–217.57 (joint_r1); **159.77–290.10** total; `--time` ceilings 492 / 984 | 49.9–72.5 / 109.8–217.6 / 159.8–290.1 / 492 + 984 | ✓ | `cluster/WAVE3_SUBMISSION_NOTE_20260830.md` §1 |
| (j) | centering band | 3 × 0.0017470584 = **0.0052412** | 0.0052 | ✓ | `b8_information_floor.json:oneD/GLADE_photo/closed_form/sigma_h_floor` |

---

## 1. Verdict table — every node touched since docket 1

Registration vocabulary; caps carried verbatim. "Rule outcome" = the charter's depth rule (runbook 37 §2).

| node | verdict of record (caps) | decisive number {value; source; date} | refuter / verifier state | charter rule outcome |
|---|---|---|---|---|
| **C0** shared baseline gate (job `6738998`, `ff230621`, iiib, h = 0.730) | **PASS — bit-identical.** Banked HEAD readout `headreadout_20260827/iiib/` (`d04d9dc9`) = the zero-compute L5 baseline for C3/C4 and the θ = (0,1) truth node for C1. Fallback (+120–173 CPU-h) NOT triggered. **PARTIAL for C1** until the §11.2 identity check on the 3 OAT columns (`den_log_term`, `num_log_term_*`) runs — it has NOT run. | max\|Δ\| **0.000** on 14 columns, 1588 rows; both posteriors md5-identical (`563ef45b…`, `2b4fb3e0…`); Elapsed 00:06:28 ⇒ **1.7 CPU-h** vs 15–23 {`REGISTRATION_C0_BASELINE_GATE_20260829.md` §13; row #246; chair (f)} | registration panel round 1: **6 must-fix** (column count 16 → 19, OAT coverage, C1 row, archive cite, A8 scoping, length) — all addressed in §11; result not refuted; chair re-diff exact | L5 honoured; anchor correction [A11] filed (§4) |
| **B5.2 [WIN] C3** log k = 3 counterfactual (job `6738999`, H4) | **INTERMEDIATE — REPORTED, not adjudicated; adoption NOT granted** (rule 5.2 conditions 2 and 3 satisfied, condition 1 open). Sign UP (toward 0.73), ≈ 2.3× HB's +0.0015. **R1 retention falsifier FALSIFIED informatively**: production true-host retention identical between arms; the window's collapse is 100 % dark/impostor-class. R6 PASS, R2 PASS, R5 PASS (no G27). R3 not computable (no 0.725/0.735 nodes); R1 growth sub-check UNDETERMINED. | Δmean_h,pred = **+0.003523** (Δℓ′ 10.444 / I_HEAD 2965; chair (d) +0.003521 from 4-dp inputs); retention **66/76 = 0.8684** both arms vs band [0.762, 0.816]; 621/1588 events lose all with-BH support, all `host_galaxy_index = −1`, 0/76 in-catalogue events change; R6 max rel 2.67e-14; R2 0.968 (951/982); \|Δℓ″\| 63.7 (2.1 % of I_HEAD); cost **4.97 CPU-h** vs 44–137 {`B5_2_WIN_K3_READOUT_RECORD.md` §2–§5; `b5_2_readout.json`; row #247; 2026-08-29} | independent reader ≠ builder ≠ runner; no separate refuter dispatched on the readout (verifier item 8 covers it). Open flag: provenance `tree_dirty_file_count=296` vs the A22 "clean" stamp — resolved by C0 §13 as the cluster checkout's UNTRACKED-file count (tracked tree clean), not re-checked on C3's own stamp. | charter 5.1 "else return with numbers" ⇒ **F-ii [RULE] returns to the author** (row #247 path decision); no k = 3 in the wave-3 readout; L10 (B5.2 → B8.2) appended |
| **B7.2 [2D-TWIN] C4** PROD-CF-2D `mz_sel`/`eff` (jobs `6739000`/`6739001`, H4) | **IMMATERIAL-PREDICTED** (≤ T_mat/2 = 0.004); gates R1/R2/R6 PASS; no G27 escalation; **PROVISIONAL** on (i) provenance extras not retrieved (SSH outage) and (ii) attribution — falsifier (ii) unrun (row #220). Not the adoption's H₀ verdict (F2). | Δmean_h,pred = **+0.0025057** (Δℓ′ +7.4294, Δℓ″ −30.3; chair (e) exact); R1 **0/6352** violations (2424 empty-set equalities); R2 **982/982** (1.0); R6 max_abs **0.0** at all 4 nodes; secondary 4-node Δmean +0.000192, ΔMAP 0.0; sign census 0 positive-tilt events at every node; STEP-2 overhead **0.99×** (385 s vs 388 s); cost **6.8 CPU-h** vs 59.7–105 {`B7_2_TWIN_CF_READOUT_RECORD.md` §6; `b7_2_readout.json`; row #248} | independent reader; disclosed operationalizations ("≪" → < 0.1·I_HEAD; "bit-identical" → ≤ 1e-12, observed exactly 0.0); falsifier (i) PASS (rel. dev. 2.6e-16 twin / 1.500 coded / A15 probe 0.600; 52 tests) {`B7_2_FALSIFIER_I_RECORD.md` §2} | 7.2 → **7.3 adoption gate OPENED** (row #248 path decision); the ONLY wave-2 adoption candidate |
| **B7.3 [2D-TWIN] adoption** (`[PHYSICS]` commit `d4765539`) | **ADOPTED as production default** (`catalogue_numerator_survival_2d="mz_sel"`, `_center="eff"`), STRUCTURAL-CONSISTENCY change, no bias claim; explicit `"off"`/`"unset"` = the counterfactual, byte-identical by test; kernel bodies `:6231-7723` untouched; **pending the wave-3 blind readout + a separate `off` arm for the A14 T_mat = 0.008 falsifier; author ratification is a fresh [RULE]** (§6 item 4). | gate presentation PANEL-CLEAN 0 rounds; 12 decisive pin tests green; full suite **1896 passed / 15 skipped / 27 deselected** (baseline 1889; +7); ruff/mypy clean; 8 Class-A sites explicit; Class-B B3 (`hier_s0_driver.py`) pinned at 3 call sites; five archived `scripts/*.py` pinned `"off"`/`"unset"` (`mass_trunc_ab.py:151-152`, `volume_trunc_ab.py:150-151`, `eddington_m_impact.py:164-165`, `ablation_cube_seed600.py:155-156`, `quick_validation_15.py:84-85`) {`B7_3_ADOPTION_VERIFIER_REPORT.md`; `B7_3_ADOPTION_IMPLEMENTATION_RECORD.md`; `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §13; `git show --stat d4765539`: 16 files, 1048+/115−} | gate-ledger rows presented / implemented / **verified (builder-run smoke, disclosed)**; then an **independent verifier PASS** with one disclosed completeness gap (5 scripts) closed same day by addendum. Two residuals found by this chair: (i) the INFO log line's citation was changed by the orchestrator before commit from "ACTIVE (row #249)" (as recorded in §13.1 and the verifier report) to "ACTIVE (adopted under row #223, charter B7.3)" (`bayesian_statistics.py:3728`) — records not amended; (ii) a `(row #<adoption>)` placeholder survives at `bayesian_statistics.py:3274`; (iii) the `BIAS_HISTORY_LEDGER.md` adoption row the presentation's §9.2 asked the orchestrator to file ("#249 at authoring") was **never filed** — no ledger row cites `d4765539` (`grep`, 2026-08-30). | F2: batched into the ONE wave-3 blind readout; "at most one adoption wave between HEAD readouts" satisfied (B5 returned, B3 struck, B2 parked) |
| **B4.2 [IMP] KW-Q1** (local, `kwq1_registered_run`, FT config, `sites2.2_nosmear`, 4 seeds) | **KERNEL-WIDTH-INERT, REPORTED-ONLY** (carried with the θ-driver INSTRUMENT-DEFECT disclosure). A14 falsifier NOT withdrawn (q1 localisation reconfirmed). **B4 does NOT merge into B1.** | **R = +0.084812** (chair (c)); S(1/√2) −1.0456670, S(1) −1.0205308, S(√2) −0.9591134 (n 191); per-seed R +0.1563 / +0.0386 / +0.1105 / +0.0516 (max 0.156 < 0.2); q1 share **92.25 %** ≥ 50 %; GATE I 7.6e-8 (tol 2e-6); GATE ENG 486/486; T-ID/PARITY max\|Δ\| 0.0 (174 events, both h); cost **6.152 CPU-h** vs 8.4 {`B4_2_KWQ1_READOUT_RECORD.md` §1–§8; `b4_2_readout.json`; row #249} | independent reader re-derived S/R from the node CSVs (not the scorer); form-invariance of `s_imp` argued by algebra (`B4_2_KWQ1_RUN_FORM_NOTE.md`); forensic E21 quantifies the divisor-gap contamination of R at ≈ +0.02–0.03 (§5 item 2) — VERIFIER-SCOPE | 4.2 INERT ⇒ **4.3** = mixture-weight/catalogue-depth h-slope derivation + per-candidate instrumented run (3.4 CPU-h; needs a non-physics-hook ruling); merge clause not triggered (row #249 path decision) |
| **B1.1 [HIER] Stage 0** — S0-A remainder (P0, runner-3, 4 seeds × 5 nodes, `sites2.2_nosmear`, `bc`/b0i) + S0-C | **B0-A′ → INSTRUMENT-DEFECT → STOP** (prereg §4.5), REPORTED-ONLY (PA-HIER-28 item 9). GATE ENG PASS; registered §3.3 GATE PARITY vindicated at review; the driver's informal parity vs the 2026-08-23 bank NOT exact (now explained, next row). Scoped to sites 2.1/2.2 under this run's flags; the CoR-M contradiction (R2′) binds this exact run. | **Z_b −3.676, Z_s −7.079** (no-BH, N 461; chair (a) bit-level); with-BH +0.379 / −2.027; per-seed score_s Z −3.94 / −4.29 / −2.64 / −3.13 (same sign, all 4); dark class n = 5 exactly 0.0 (the PA-HIER-31(d) identity check PASSES); ENG 0.98858; informal parity `ln_L_no_bh` max abs 5.716e-4 (rel 4.9e-5); **S0-C marginal 24.37 s/h-node** after a 1704.3 s first-h table build (chair (g)); P0 **11.51 CPU-h**, S0-C **10.42 CPU-h** {`B1_1_HIER_STAGE0_RECORD.md` §2–§5; prereg STAGE-0 RESULT RECORD `:2526-2683`; row #250 (uncommitted)} | panel (sonnet/high): every headline reproduced bit-for-bit; must_fix none. Duplicate-row artifact disclosed (seed 900101 b_plus CSV 212 rows / 106 events, halves identical). | depth-1 band \|Z\| ≤ 3 fails ⇒ STOP; **C1 (S0-B) NOT submitted**, template only; the S0-B question returns (row #250 path decision) |
| **B1.1-F forensic** (zero re-evaluation; numpy twin + banked CSVs + pool) | **LOCALISATION: VENUE-LAW / INSTRUMENT-FORM, not a hook-arithmetic defect.** (i) **HOOK-PLACEMENT gap**: the no-BH catalogue divisor Σ^φ carries no θ in any built form (`bayesian_statistics.py:2906/:2916` branch order; `:5187-5191` consumer) ⇒ the registered score at truth-θ is ⟨c⟩·∂_θ ln Σ^φ ≠ 0 by construction; (ii) candidate-ball truncation (sky 1.5σ_max, z ±3σ_d ±1σ_g) drops the true host for 16.1 % of events and leaves an impostor-dominated mixture (median 278 candidates, true-host share median 0.006); (iii) the ±ln√2 secant carries an intrinsic O(Δ²) bias +0.0455/event (PA-HIER-4 class). `bc`-flag hypothesis REFUTED (E8). **GATE PARITY 5.7e-4 residual RESOLVED**: the `_B0I_ZTRUE_GRID_N` 401 → 4001 hardening in `d40fe5c8` moved `z_true` ≤ 1.06e-5 / `obs_d_L` ≤ 6.1e-5 (E19); with-BH residual = the symmetric mass filter `cf4f8a2a`. Does NOT lift the STOP; does NOT license S0-B/C1/Stage P. REPORTED-ONLY. | E7 twin: `L_cat_no_bh` max\|Δ ln L\| **9.2e-13**; secants 3.0e-12 (b) / 8.4e-13 (s), corr 1.000000; **E11: score_b −1.634 ± 0.444 (Z −3.68) → −0.268 ± 0.431 (Z −0.62)** (chair (b) exact); E12 c-weighted s after both corrections **−0.005 ± 0.011 (Z −0.5)**; E10 C_b −2.20/−2.25, C_s −0.026/−0.024; E13 +0.0455 ± 0.0005; E20 edge case 15,618 / 20,834,171 pool rows (0.075 %), 3/800 drawn hosts; E21 KW-Q1 contamination +0.039 per unit catalogue share ⇒ ≈ +0.02–0.03 in R {`B1_1_S0A_DEFECT_FORENSIC_20260829.md` §0–§6; `b1_1_forensic_work/f*_out.json`; row #251 (uncommitted)} | panel (sonnet/high): not refuted, must_fix none; **one overclaim flagged** — E7's "own GL-50 quadrature" reuses production `_GL_NODES_50`/`_host_pixels`/`get_possible_hosts_from_ball_tree` outside the hook scope (does not undermine 9.2e-13) | fix routing per prereg §4.5: divisor θ-dependence → `/physics-change` (trigger file); secant form + z-binning read → registration amendment / fresh [RULE]; comparand for the informal parity → instrumentation (regenerate the bank at the current grid) |
| **B3 [POP]** closure (B3.2 dispatch declined; superseding note on B3.1) | **CLOSED — PREMISE-REFUTED (provenance, zero compute).** Production dark hosts are drawn from `(1−f)·dVc/dz/(1+z)` = the estimator's own completion prior (byte-identical); the "M1-vs-comoving" term is zero by construction; B3.1's 98.5 %/103.9 % coverage re-read as the estimator's algebraic response to a prior swap. No `completion_population_prior` code exists; C2 STRUCK (0 CPU-h). Gate ledger: PRESENTED WITH A STOP + dispatch-declined row. | `git show 03cfe80:…/dark_siren_injection.py:328`; CRB md5 `9a1f2a14…`: **1514 dark / 76 in-catalogue** of 1590; F(0.73, 1.5) = 0.01754; HEAD dark-class score −0.4668 ± 0.0162 / −0.3938 ± 0.0207 vs row #138's −0.635 / −0.565 (7.16σ / 5.95σ, STALE) {`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §F/§14; `B3_1_POP_RECORD.md` superseding note; `b3_pop_prediction.json`} | builder declined under the approval-scope rule (correct: the STOP postdates the dispatch's inputs); registration-check chair concurs ("DEVIATE: strike C2; accept the STOP"); L1/L4 re-cut by appended note | coverage rule fired on a statistic that cannot bear the causal reading ⇒ branch closed at depth 1; **G7 row 16 re-grade → [RULE]** (§6 item 5); the +0.11-high pure-completion finding (B4.1 C5) is now WITHOUT a competing population explanation |
| **B2 [CMEM]** (no new record since docket 1) | **PARKED with the bound**: R2c NOT-DISTINGUISHED, p = 0.0358 ≥ 0.01, deficit direction; C-STRUCTURAL-ONLY (row #220) stands; A2 not triggered; k_sky = 1.5 an invariant of C3 (L3). | T = −0.12311, p = **0.0358** (10 000 perms); census 380/2336 = 0.16267; power ≈ 68 % at −16 % {`B2_1_CMEM_A1_RECORD.md`; row #226} | bit-for-bit re-execution of the sha1-pinned instrument; inherited cap-citation error (row #219, not #216 item 4) — must-fix, appended | closes at depth 1; the pooled-observation note and the R2c bank-vs-follow-up word go to the author (§6 items 7–8) |
| **B8.2 [CAL] design** + **L10** | **DESIGNED, NOT BUILT** (S1–S5 unstarted): the estimator is production's `evaluate()` via `run_mirror_seed_inprocess`; truth = the estimator's own generative mixture (b0i + bsel at w̃_G); measures F; refuses "starved"; SHARED-FILTER referent re-pointed to S4 after C2 was struck. **L10**: mirror-derived retention/growth predictions are hypotheses to check against production, not design inputs (the mirror's linear-Gaussian mass law did not transfer). | cost **130–475 CPU-h local, 13–46 h wall** (bracketed: per-`evaluate()` N-scaling UNMEASURED, 1.0–3.8 CPU-h per universe at N = 1588); n_U = 100 (S) + 25 (T); PIT–KS band 0.134 {`B8_2_HARNESS_DESIGN_20260829.md` bottom line, §4.1, §6, §8, §10} | design note, no band registered (verifier item 14) | 8.2 designed; 8.3 (coverage + count audit → stop/continue) blocked on S1–S5 |
| **PA-HIER-31** + REVISION NOTES 1–2 + P1 full-N + P6 CLI (`fb9d8aff`) | S0-B registration text complete (θ form `"2.2"`/unsmeared; b-nodes ±0.033; `score_lns` relabel; by class + z-bin; B0-B/B0-M/B0-P bands; L2 profile prediction; 60–92 CPU-h unsmeared, smeared band STRUCK). Two **OPEN CONTRADICTIONS** registered for fresh author [RULE]s: R1 (PA-HIER-10 vs PA-HIER-31(b) for CoR-P) and R2′ (the same pair for CoR-M / S0-A). R3′ downgrades P0's "certifies the instrument" to sites 2.1/2.2 only. R4′ mirror P1′ recommended, unexecuted. | F-A at full N (seed 900101 b_plus): `L_cat_no_bh` max_abs **0.0**; `combined_no_bh` max_rel **7.447e-3**; `D_tilde_phi` 7.503e-3; `alpha_G_phi` **13.66 %** {prereg `:2501-2510`; row #245} | two refuter rounds (5 + 4 must_fix), all addressed append-only; REVISION NOTE 1's own R1 scope was refuted by REVISION NOTE 2 (R1′) | C1 NOT launched (STOP); registration stands for the next tree; the two contradictions + item (f)'s now-resolved residual → §6 item 2 |
| **B6.1 / B5.1 commits** (`1f003da6`, `0b308828`) | landed before every wave-2 arm (L8 ✓; C0 certifies both at production scale, bit-identical). | C0 max\|Δ\| 0.0 (chair (f)) | — | closed |
| **Wave-3 sbatch set** (`60f9996e`; `cluster/wave3_headreadout_{iiib,joint_r1}.sbatch`, `submit_wave3.sh` DRY_RUN=1) | BUILDER deliverable; F2 blindness verified by the chair: neither script passes `--catalogue_numerator_survival_2d`/`_center` (`wave3_headreadout_iiib.sbatch:13-19,158-159`); explicit `--mass_filter_geometry linear --mass_filter_k 1.5 --theta_b 0.0 --theta_s 1.0 --theta_sites all`. NOT submitted (SSH down). | 82 tasks; **159.8–290.1 CPU-h** (chair (i)) | — | wave 3 = the ONE blind readout; the A14 `off` arm is a separate build |

---

## 2. Tree state per branch after wave 2

- **B1 [HIER] — STOPPED at 1.1 with a LOCALISED HOOK-DEFECT.** S0-A returned B0-A′ (Z_b −3.68, Z_s −7.08); the forensic localises it to (i) the θ-free no-BH divisor Σ^φ (a hook-placement gap, not arithmetic — the twin reproduces the hook to 9.2e-13), (ii) the candidate-ball truncation of the b0i mixture, (iii) the secant's O(Δ²) bias. **The fix = a θ-consistent divisor Σ^φ(θ) = Σ_g w_g S̃_g(θ)** (site 2.3 extended to the phi-table branch, or a per-node scalar ρ(θ) applied post hoc) — a `/physics-change` node in `bayesian_statistics.py`; predicted effect Z_b → −0.62 (E11, chair-reproduced), Z_s → ≈ −6 until the sky-cone radius (`:4869`, hardcoded 1.5) becomes a flag (then Z_s → −0.5 ± 1, E12). **S0-B (C1) and the P6-plumbed template are UNLAUNCHED**; the forensic predicts an S0-B non-null of order ⟨c⟩·C_b ≈ −1.3 per unit b by construction on production (§7 of the forensic) — it must be subtracted or the divisor hooked before any LEVER-LIVE read. **Stage P re-cost with the measured 24.4 s marginal** (unsmeared, mirror, N ≈ 106): one θ-cell over `H_GRID_41` = 1704.3 s first-h + 40 × 24.4 s ≈ 2681 s wall ≈ **8.9 CPU-h at 12 cpu**; the registered 3×3 × 4-seed grid (36 cells, `PREREGISTRATION_HIER_HTHETA_20260826.md:596`) ⇒ ≈ **320 CPU-h (12-cpu billing), ≈ 27 h serial wall** vs the registered 424.4 — the first-h table build is 64 % of every cell; in the smeared form each θ-engaged h-node costs 1190.93 s single-core (docket 1 L6) ⇒ ≈ 13.6 h wall per cell, out of reach. Stage P is moot until the divisor fix lands and S0-A re-certifies. Open [RULE]s: §6 items 2, 3, 6.
- **B2 [CMEM] — PARKED** at depth 1 with the bound (p = 0.0358, ≈ 68 % power); A2 not triggered. Two open author words (§6 items 7–8).
- **B3 [POP] — CLOSED, PREMISE-REFUTED** (provenance, 0 CPU-h). Deliverables returned: the provenance finding (rows #137–#139 re-read), the G7 row 16 sensitivity number (−0.60 on bins 2–5 per shape swap), the docs-only "two M1s" cross-reference. G7 row 16 re-grade → [RULE].
- **B4 [IMP] — → 4.3** (INERT at 4.2; no merge into B1): the mixture-weight/catalogue-depth h-slope derivation (s_β = −3.2891/h, ≈ 63 % of the impostor-leg score; s_L = −27.08/h on active events) + the per-candidate instrumented run (3.4 CPU-h, contingent on a non-physics-hook ruling) + the enlarged-ball counterfactual as the s-follow-up (forensic §6 item 2). The 1D-rail decomposition stands: impostor leg NECESSARY (0.6077 → 0.7134 dark-only), pure completion +0.11 high (0.8396), kernel width INERT.
- **B5 [WIN] — INTERMEDIATE returned with numbers** (+0.0035; retention transfer FALSIFIED: 66/76 unchanged, the collapse is 621/1588 dark-class events; L10 to B8.2). F-ii design [RULE] to the author (§6 item 1). No adoption; no k = 3 in wave 3.
- **B6 [ALIGN] — CLOSED** (`1f003da6`; bit-identical while `SIGMA_V_PEC_KM_S = 0.0`; C0-certified at production scale).
- **B7 [2D-TWIN] — ADOPTED** (`d4765539`) pending the wave-3 blind readout and its separate `off` arm (A14, T_mat = 0.008, both venues); falsifier (ii) unrun; ratification = fresh [RULE] (§6 item 4). Housekeeping: the missing `BIAS_HISTORY_LEDGER.md` adoption row; the `row #<adoption>` placeholder.
- **B8 [CAL] — 8.2 DESIGNED; S1–S5 NOT BUILT.** F unmeasured; the stop rule not evaluable (§3).

---

## 3. Wave-3 plan as it stands, and the stop/continue logic

**Plan (BUILDER deliverable, not submitted):** two 41-task arrays, 16 cpus/task, `cpu_il`
(`--time` 00:45:00 iiib / 01:30:00 joint_r1), full `H_GRID_41`, EVAL_SEED 777000, CoR-P CLI
verbatim from `headreadout_20260827/iiib/run_metadata_21.json` + the explicit post-wave-2
defaults; **deliberately blind** to the row-#223 adoption (no 2D flag passed — F2). **82 tasks,
159.8–290.1 CPU-h at the corrected anchor** (chair (i)); `--time` ceilings 492 + 984 CPU-h are the
worst case only. **Baseline = the banked 2026-08-27 HEAD readout** (`d04d9dc9`, 2026-08-27T19:40:20),
certified by C0 PASS bit-identical. **Falsifier of record:** A14, **T_mat = 0.008** on \|Δmean_h\|
(2D channel), BOTH venues, evaluated against a **separate** `--catalogue_numerator_survival_2d off`
arm at the same commit — that arm is NOT part of the delivery (another 82 tasks of the same
shape if built at full grid; ≈ 160–290 CPU-h more). Registered point prediction (REPORTED-ONLY,
iiib): Δmean_h ≈ +0.0025 upward, toward truth, from the HEAD 2D offset −0.066653 (iiib) /
−0.066987 (joint_r1) {`PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §8}. Instrument-defect
falsifiers at full grid: R1 (eventwise inequality) and R6 (1D bit-identical) ⇒ any violation
REVERTS the default to `"off"`.

**Blocked on SSH (down since ≈ 21:15 on 2026-08-29):** checklist items 2 (cluster HEAD = local
HEAD), 3 (`preflight.sh` READY), 8 (out-roots absent); the C4 provenance extras (2 × 130 MB
`posteriors_with_bh_mass` JSONs at h = 0.67/0.73, 4 `run_metadata_*.json`, `logs/`,
`GIT_COMMIT_AT_RUN.txt`); C0's `sacct` re-pull; the C1 template → live conversion; the Option-A
archive run for the wave-2 out-roots; the wave-3 `datasets.yaml`/`DATA_INVENTORY.md` stamps.

**Stop/continue logic against B8.1's numbers — stated plainly: the stop condition CANNOT be
evaluated yet, and its centering clause is already known to fail.**
- Centering clause: \|⟨h⟩ − 0.73\| ≤ 3σ_floor = **0.0052** (chair (j)). HEAD 2D ⟨bias⟩ = −0.0668
  (38.2 floor-σ), 1D −0.1190 (68.1 floor-σ) {`B8_1_CAL_FLOOR_RECORD.md` §3}. The only production
  change riding wave 3 is predicted to move the 2D mean by +0.0025 (C4) — i.e. from ≈ 38 to ≈ 37
  floor-σ. A wave-3 centering PASS is therefore excluded by numbers already in hand; the readout's
  job is the A14 falsifier and the G41 baseline for the next tree, not a stop verdict.
- Width clause: σ_h,measured ≤ F·σ_floor with **F UNMEASURED** (B8.2 S1–S5 unbuilt; placeholder
  F = 10 puts HEAD 2D at 10.6× — "right at the line" — and is not a registered number). Per B8.2 §5
  the width clause is meaningful only after a centering PASS; per the stage-5 rule, anything with a
  centering FAIL routes to the branch tree (B1/B4/B5 own it), not to a stop.
- Consequence for the operating cycle: wave 3 closes fan-out 1's verifier scope (item 20) and
  seeds the next tree; it does not close the campaign.

---

## 4. Compute ledger reconciliation (F4)

**Estimate vs measured, every arm (chair-recomputed from the cited sacct/log values):**

| arm | estimate (CPU-h) | measured (CPU-h) | ratio | source |
|---|---|---|---|---|
| C0 baseline gate | 15–23 | **1.72** (00:06:28 × 16) | 9–13× below | row #246; C0 §13 |
| C3 log-k3 | 44–137 | **4.97** (4:50 + 4:39 + 4:36 + 4:34 = 1114 s × 16) | 9–28× below | row #247; `b5_2_readout.json:cost_F4` |
| C4 PROD-CF-2D | 59.7–105 (74.7–101.4 incl. C0; ceiling 132) | **6.80** (385 + 398 + 377 + 370 = 1530 s × 16) | 9–15× below | row #248; `B7_2_TWIN_CF_READOUT_RECORD.md` §1 |
| C2 M1-prior | 45–69 | **0 — STRUCK** | — | `COMPUTE_LEDGER.md` "Node B3 closure" |
| C1 S0-B | 60–92 (unsmeared; 81–113 smeared band STRUCK) | **NOT LAUNCHED** | — | rows #245/#250 |
| P2 KW-Q1 (local) | 8.4 (registered 2.2/unsmeared form) | **6.152** (1417.786 s + 164.070 s, × 14) | 0.73× | row #249; runner-3 log |
| P0 S0-A remainder (local) | ≈ 5 (re-scoped) / 11 (smeared, docket 1) | **11.51** (2959.6 s × 14, `--jobs 1`, 20 cells at ≈ 65 s + 4 venue builds) | 2.3× the re-scoped figure — the "5 CPU-h / 40 min" assumed 5-way parallelism the driver cannot deliver (daemonic-pool limit, §8) | `s0a_full_output.json`; chair (g) |
| S0-C (local) | ≤ 15 ceiling; marginal UNMEASURED | **10.42** (3125.1 s × 12); **marginal 24.37 s/h-node** after a 1704.3 s first-h build | inside ceiling; marginal now measured | `s0c_full_output.json`; chair (g) |
| B8.2 harness (local, not started) | 130–475 (revised from ≈ 6/sweep) | 0 | — | B8.2 §6 |

**Corrected anchor [A11].** The 56–76 min/h-value figure (`cluster/LAUNCHING_JOBS.md:47`) is a
3355-event-set number; on the 1588-event iiib venue the measured per-task Elapsed is **4:34–6:38**
at 16 cpus (C0/C3/C4; job `6725283` ranged 00:00:18–00:42:26 across 41 tasks). Working anchor
for iiib: **≈ 1.2–1.8 CPU-h per h-point** (16-cpu billing); joint_r1 ≈ 2.2–3.0× that. Every
wave-2 cluster estimate was 9–28× high; the C1 60–92 band re-costs to **≈ 7–27 CPU-h**
(4 nodes × 1.7–6.8) if S0-B is ever launched; B7 falsifier (ii)'s 208–286 (24–33 tasks) re-costs
to roughly **≈ 40–60 CPU-h** at the same ratio (chair estimate, not registered).

**Fan-out total to date (chair (h)):** cluster **13.50 CPU-h** (C0 + C3 + C4; 9 tasks) + local
wave-2 **28.08** (P2 6.15 + P0 11.51 + S0-C 10.42) + wave-1 local ≈ 11.4 ⇒ **≈ 53 CPU-h**,
against a pre-launch wave-2 estimate of 179–357 cluster + ≈ 144–489 local. **Unbanked, not in the
ledger:** the runner-1 P0 attempt (20:28:21 → 20:46:40, 18.3 min) and the runner-2 P0 attempt
(21:39:52 → 21:58:20, 18.5 min) at a 14-cpu allocation ≈ **8.6 CPU-h nominal** (log timestamps,
`runner_wave2pre_20260829.log:766,806`, `runner2_wave2pre_20260829.log:2,315`), plus two aborted
KW-Q1 launches (< 2 min each). **Wave-3 estimate:** 159.8–290.1 CPU-h (82 tasks; chair (i)); with
the A14 `off` arm ≈ 320–580 total. **Deadline gate:** workspace expires 2026-09-23 (0 extensions);
24 days at wave-2 launch; every wave-2 out-root archive-scheduled = yes
(`results/_archive/archive_run_wave2.sh`, gitignored, run AFTER retrieval — **not yet run**, SSH).
**Ledger hygiene:** the "P0 (S0-A/S0-C) measured" section of `COMPUTE_LEDGER.md` and rows
#250/#251 are **uncommitted** at chair time (`git diff HEAD`: +21 / +10 lines).

---

## 5. Findings that are themselves new information (valued outputs, including refuted / undetermined)

1. **The [HIER] instrument has a θ-divisor defect, not a hook-arithmetic defect** (B1.1-F,
   chair-reproduced E11). θ enters the site-2.2 numerator kernel exactly as registered (twin 9.2e-13)
   but the no-BH global-selection divisor Σ^φ is point-evaluated and θ-free in every built form
   (`bayesian_statistics.py:2906` phi-table branch precedes `:2916 elif smear_sigma_z`; consumed at
   `:5187-5191`). Hence E[∂_θ ln L] at truth-θ = ⟨c⟩·∂_θ ln Σ^φ(θ) ≠ 0 **by construction of the
   instrument**: C_b = −2.20 ± 0.04 per unit b (sign: S̄_φ falls with z, ⟨∂_z ln S̄_φ⟩ = −2.4).
   Restoring the θ-dependence turns Z_b −3.68 into **−0.62**; the s-axis residual (Z −6) is ≈ 75 %
   candidate-ball truncation (E9) and the rest the secant's own O(Δ²) bias (+0.0455/event, E13) —
   the B0-A s-band is mis-formed at N ≈ 461 even on a perfect venue (predicted Z ≈ +3.8). What it
   says about the instrument: every no-BH θ-read to date (S0-A, the smeared partial run, KW-Q1)
   measured a numerator-only θ against a θ-inert divisor; **S0-B on production inherits the same
   gap verbatim** and would return a non-null ≈ ⟨c⟩·C_b ≈ −1.3 per unit b with no lever behind it.
   Also new: a real b-hook edge case at b = −0.02 (negative kernel centre for 0.30 % of pool rows;
   inverted window for 15,618 rows whose `Z_g ≤ 0` guard silently substitutes 1.0) — immaterial
   here, a defect nonetheless. And the informal GATE-PARITY residual is **explained**: a
   generator-side comparand delta (401 → 4001 grid, `d40fe5c8`), not an estimator path.
2. **Does the divisor defect cancel in KW-Q1's R? Argued from the forensic — mostly, not
   exactly (VERIFIER-SCOPE).** The registration-check chair showed `D̃^φ` and `α_G^φ` cancel
   identically in `s_imp,i = Δ_h ln[(βL_cat,i + B_num,i)/B_num,i]` and `β_G^φ` is θ/smear-inert
   — that cancellation is of the *h-dependent global normalisers*, and it holds. The forensic's E21
   identifies the residual: the un-normalised factor ρ(s; h) multiplies L_cat(s) inside the bracket,
   so its h-derivative survives — Δ_h[ln ρ(√2; h) − ln ρ(1/√2; h)] = +0.039 per unit catalogue
   share at h ∈ {0.725, 0.735} (C_s(0.725) = −0.02380, C_s(0.735) = −0.02324). Against
   \|S(1)\| ≈ 0.80–1.02 that is **≈ +0.02–0.04 in R**, on a measured R = +0.085 with a 0.2 INERT
   ceiling: the verdict is unchanged in either direction (the correction would move R toward
   +0.05–0.06, deeper inside INERT; even added with the wrong sign it stays < 0.13). Two caveats
   the verifier should check: (i) E21 was computed on a 200k-row pool subsample for FT's "phi"
   numerator, not re-run through the scorer; (ii) the paired within-run design cancels the seed
   level shifts (SD(R) 0.055 vs SD(S) 0.106) but not a form-common multiplicative ρ(s) — so
   "form-invariant" (`B4_2_KWQ1_RUN_FORM_NOTE.md`) is exact for the global normalisers and
   first-order-small for the divisor gap. Attribution note carried from the forensic §6: an OWNS
   verdict (had one occurred) would have referred to the *truncated impostor mixture's* width
   response, not to a photo-z error misstatement — INERT is unaffected.
3. **The production-vs-mirror retention non-transfer (mirror mass law).** The mirror fleet
   predicted a 17–21 pp true-host loss under log k = 3; production lost **0/76** (66/76 recovered
   in both arms). The pull read explains why the mirror's number was never a production number:
   the mirror draws `M_true = m_eff + BH_MASS_ERROR·Z` — a **linear** Gaussian truncated at
   `M > 0` at fleet-median CV = 1.018 — so no symmetric log-window keyed to a Gaussian tail table
   retains 99.7 % of *those* hosts (empirical \|pull\| ≤ 3 = 78.8 % ≈ the mirror's 78.9 %
   retention), while production's 76 in-catalogue hosts sit where the window is vacuous. The
   window's whole production effect is on the **impostor / dark-class pool** (621/1588 events lose
   all with-BH support) — a candidate-set-composition object, which is why the H₀ effect is small
   and upward (+0.0035). L10 is the design consequence: mirror-derived retention/growth
   predictions are hypotheses for B8.2, not inputs. **L9 resolved:** `BH_MASS_ERROR/BH_MASS` IS the
   ln-space R&V15 budget since `555f0186`; B8.1's "0.19 current" was a pre-fix (2026-06-30) figure.
4. **The B3 provenance refutation.** Production's dark class is drawn from the estimator's own
   completion prior; the two "M1"s in the code (the `p0 = 1` constant-comoving `emri_rate` law
   used by the generator vs `Model1CrossCheck`'s extracted z-evolution used only by the p_det
   pool) were conflated by row #138's memo and B3.1 alike. Consequences: (a) the dark-class tilt is
   an internal/selection object (rows #140–#144, B1/B4), now WITHOUT a competing population
   explanation; (b) the +0.11-high pure-completion leg (B4.1 C5) has no population fix available;
   (c) the historical −0.635/−0.565 baseline is STALE by 7.16σ/5.95σ; (d) a real, large, paper-
   facing systematic is quantified without compute: a population-shape swap of the size between
   the two M1s moves the dark-class score by −0.60 on bins 2–5 (≈ −290 nats per unit h over 484
   events) — H₀ from completion-dominated EMRI dark sirens is O(1)-degenerate with the population's
   redshift evolution on real data.
5. **The 1D-rail decomposition (B4) now reads:** impostor leg **NECESSARY** (removing it un-rails
   1D to 0.7134 ± 0.0277 at the model's own class composition), **pure completion +0.11 high**
   (0.8396, MAP at the 0.86 edge — now un-explained by population), **kernel width INERT**
   (R = +0.085), localisation to z_true < 0.358 reconfirmed at **92.25 %** of the impostor-leg
   score, ≈ 63 % of which rides the global mixture-weight h-slope (s_β = −3.2891/h). The mechanism
   fork left standing is normalisation/mixture-weight vs in-ball depth skew (B4.3). The forensic
   adds an independent b0i observation in the same direction: on the mirror, low-z impostors want
   b < 0 because the volume prior lifts every candidate's kernel mean ≈ 1σ above its listed z
   (E15: +0.91 at the lowest z_g bin), and the pooled score is a cancellation of two opposite-sign
   classes (c < 0.2: +2.40; c > 0.95: −18.7, E17).
6. **The anchor correction.** Every wave-2 cluster estimate was built from a 3355-event anchor and
   came in 9–28× high; the iiib venue costs ≈ 1.2–1.8 CPU-h per h-point. This changes the yield
   calculus of the next tree: an S0-B arm is ≈ 7–27 CPU-h, a full-grid per-change arm ≈ 50–73
   CPU-h (iiib) — cluster reads are no longer the expensive option; the local `evaluate()`-bound
   items (B8.2 at 130–475 CPU-h; KW-Q1 at 6 CPU-h per 24 cells) are.
7. **UNDETERMINED (valued):** the net H₀ sign of the k = 3 window beyond INTERMEDIATE (needs the
   G41 read or a ruling); the mechanism behind the +0.11 pure-completion excess; the magnitude of
   the CoR-M smear-form confound (R2′; the mirror P1′ is the 0.33 CPU-h resolving measurement);
   F (B8.2); the S0-B production θ-score (never measured).
8. **REFUTED (valued):** the docket-1 P1 equivalence (in part: `L_cat_no_bh` identical,
   `combined_no_bh` not — α_G^φ −12 %/13.66 %); the GATE-PARITY batch-order hypothesis (F-B) and
   then the code-delta hypothesis's estimator half (E19: comparand only); the `bc`-flag cause (E8);
   B3's premise; the mirror retention transfer (R1); the "3σ log window retains 99.7 %" reading of
   ε = 2Φ(−3) as a retention statement; the "5 CPU-h / 40 min" P0 re-scope (parallelism the driver
   cannot deliver); the 3355-event cost anchor for iiib.

---

## 6. [RULE]s that return to the author (each with inputs and the exact question)

Tags per CLAUDE.md. None is pre-approved by rows #221–#223: every item's inputs post-date the grant.

1. **[RULE] F-ii — the mass-window design, with C3's numbers.** Inputs: Δmean_h,pred = +0.003523
   (INTERMEDIATE, 0.003 < x < 0.008, sign up); production true-host retention unchanged (66/76 both
   arms); the entire candidate collapse is dark-class (621/1588); the mirror's 78.9 % retention was
   a linear-Gaussian-law artefact (pull read, CV median 1.018); R6 1D untouched; cost 4.97 CPU-h.
   Question: treat the INTERMEDIATE read as (a) adopt log k = 3 as a documented design choice
   (structural, not bias-motivated), (b) keep linear k = 1.5 and document the one-sided geometry
   as deliberate, or (c) commission a k-scan / a window keyed to the mass law before ruling — and
   is a joint_r1 arm required? **Folds in row #220's still-open "WGEOM §9 F-ii consequence
   ruling"** (ε-derived truncation vs documented design choice; `PREREGISTRATION_MKER_WGEOM_20260828.md:245`):
   B5.1/B5.2 are exactly that instrument + measurement; the chair's reading is that one ruling can
   close both, but whether they are the same object is itself the author's call.
2. **[RULE] PA-HIER-31's open items.** (a) REVISION NOTE 1 R1: PA-HIER-10's unconditional
   `smear_sigma_z=True`-at-every-node pin vs PA-HIER-31(b)'s `smear_global_selection=False` for
   **CoR-P** — which is authoritative, given F-A (`combined_no_bh` 7.45e-3, `alpha_G_phi` −12 %/
   13.66 %)? (b) REVISION NOTE 2 R2′: the same pair for **CoR-M / S0-A** — the P0 run of record
   used the narrowed form; its STOP is scoped to sites 2.1/2.2 (R3′); the resolving measurement
   (mirror P1′, ≈ 0.33 CPU-h) is registered but unexecuted (R4′). (c) Item (f)'s "diagnose, not
   accept" disposition for the 5.718e-4 residual: the forensic diagnosed it at zero compute
   (generator grid 401 → 4001) — ratify that as the disposition and retire the "one re-run of the
   bank" step (or order the bank regenerated at the current grid as the instrumentation fix).
   (d) Registration amendment items the forensic routes to the author: the ±ln√2 secant form
   (intrinsic +0.0455/event bias ⇒ the s-band is mis-formed at this N) and the z-binned θ-score
   read (E16: binning on z_true selects events whose z_true fell low in their kernel — a selection
   artefact, not a mechanism). **Delta for the verifier:** `REGISTRATION_END_VERIFIER_PASS` item 16
   lists "the original register's author item 3 (physics-change scope) … explicitly deferred" as
   still open — it is not: PA-HIER-28 (2026-08-28, author verbatim "exactly as recommended by
   you") resolved items 3 = GATE, 4 = GATE, 5 = FALLBACK, 9 = AFFORDABLE; the chair counts the
   pre-existing open [HIER] items as the two contradictions (a)/(b) plus the two unexecuted P1′
   measurements, not a deferred item 3.
3. **[DO] + [RULE] The θ-consistent divisor fix as a physics-change proposal.** Inputs: E10/E11
   (C_b −2.20/−2.25, C_s −0.026/−0.024; corrected Z_b −0.62, chair-reproduced); the E20 edge case;
   the sky-cone radius hardcoded at `bayesian_statistics.py:4869` (needed for the s-axis, E12);
   NEEDS-CODE in a trigger file; cost band 35–60 min wall (b-axis) + ≈ 1.5 h wall (s-axis),
   UNMEASURED. Question: authorize the gate presentation for Σ^φ(θ) (site 2.3 extended to the
   phi-table branch, byte-identical at θ = (0,1)) and the sky-cone flag as the first node of the
   next tree — row #223's standing grant covered production changes *inside* fan-out 1's tree; B1
   is STOPPED, so this is a new node and returns as a fresh [DO].
4. **[RULE] B7.3 ratification after the wave-3 readout.** Inputs: `d4765539` (structural
   consistency; C4 +0.0025 IMMATERIAL-PREDICTED; R1/R2/R6 PASS; falsifier (i) PASS; suite 1896;
   independent verifier PASS; five archived scripts pinned); falsifier (ii) unrun (class-G fleet,
   ≈ 208–286 CPU-h at the old anchor); the A14 T_mat = 0.008 read requires a separate `off` arm not
   yet built; the ×2.25–2.35 identity residual (row #211) disclosed; the adoption's
   `BIAS_HISTORY_LEDGER.md` row not filed. Question: after the blind readout + `off` arm land,
   ratify `mz_sel`/`eff` as the production default (or revert to `"off"` pending falsifier (ii)).
5. **[RULE] G7 row 16 re-grade.** Inputs: §F (the mock's population prior is the generator's own
   law — no population-shape systematic of the row-#138 kind exists in the mock); the HEAD dark-
   class score −0.4668/−0.3938 vs the STALE −0.635/−0.565 (7.16σ/5.95σ); the real-data sensitivity
   −0.60 on bins 2–5 per M1-(i)↔M1-(ii) shape swap. Question: re-grade row 16 from "MEASURED,
   calibration-affecting" to "mock: zero by construction; real data: O(1) degeneracy with the
   population's z-evolution, hierarchical marginalisation required" — and retire rows #137/#138 as
   citations.
6. **[RULE] The S0-B question.** Inputs: the forensic's by-construction prediction of an S0-B
   non-null ≈ ⟨c⟩·C_b ≈ −1.3 per unit b on production; C1 re-costed ≈ 7–27 CPU-h; P6 landed; the
   two open contradictions (item 2). Question: launch S0-B only after the divisor fix (chair's
   reading of prereg §4.5 and forensic §7), or launch now as a REPORTED-ONLY read with the post-hoc
   ρ(θ) subtraction disclosed — one word, launch-after-fix / launch-now.
7. **[RULE] CMEM pooled-observation note (awareness, not a recommendation).** Two independent
   fleets read deficit-direction at p = 0.0152 (row #219) and p = 0.0358 (A1, 68 % power at −16 %);
   no pooled statistic was computed or banked (the record honours that). Question: is the author
   content to leave this unregistered, or should a registration be opened?
8. **[RULE] The R2c bank-vs-follow-up word** (row #220 "one word still required", not answered by
   A1's non-trigger of A2). Inputs: A1 NOT-DISTINGUISHED; a ≥ 90 %-power registration at the
   −11.6 % effect at α = 0.01 needs ≈ 1.7× smaller SE ⇒ ≈ 3× the strata (≈ 30 new mirror seeds ×
   2 arms; ≈ 15 CPU-h local, chair estimate). Question: bank-and-park / higher-power follow-up.

---

## 7. NEXT TREE candidates (operating cycle: tree → verify → plan next tree), ranked by expected yield per cost

| rank | candidate | cost (at the corrected anchor) | expected yield | gate / prerequisite |
|---|---|---|---|---|
| 1 | **θ-consistent divisor fix + S0-A re-certification + S0-B** — Σ^φ(θ) at the phi-table branch (or a per-node ρ(θ) scalar, E11's operation) + the sky-cone-radius flag; re-run the 20 S0-A cells (registered prediction: Z_b → −0.62 ± 0.43, Z_s → −0.07 ± 0.012 without the cone flag, → −0.5 ± 1 with it); then C1 on iiib | fix: physics gate + ≈ 1–3 min/node divisor pass (UNMEASURED); S0-A re-run ≈ 11.5 CPU-h local (or ≈ 6 if the venue builds are cached); S0-B ≈ 7–27 CPU-h cluster | the only route to a LEVER-LIVE/DEAD read on production; converts the STOP into a certified instrument; the falsifiable predictions are already registered by the forensic | [DO]/[RULE] items 3, 6, 2; the CoR-M P1′ (0.33 CPU-h) first |
| 2 | **B4.3** — mixture-weight h-slope derivation (s_β = −3.2891/h) + the per-candidate instrumented run + the enlarged-ball counterfactual (sky 3σ, z ±4σ_g; median candidates 278 → 1729) | derivation 0 CPU-h; instrumented run 3.4 CPU-h local; enlarged-ball cells ≈ 3–6× a normal cell | names the mechanism of the NECESSARY cause of the 1D rail; the forensic's E9/E14/E17 already show the sign flip under ball enlargement on b0i | non-physics-hook ruling for the instrumented run; ball radius flag = the same edit as rank 1's s-axis item |
| 3 | **B8.2 S1–S5** — the two-channel harness; F measured; the count audit | 130–475 CPU-h local, 13–46 h wall (S3 measures the N-scaling first) | makes the width clause of the stop rule evaluable; the acceptance census (§2.4) is itself a production-venue check | S4 registration (top tier) reviewed by the end verifier before S5 |
| 4 | **B7 falsifier (ii)** — class-G fleet, Option A′, rung 1 repaired (24–33 tasks) + the wave-3 `off` arm (82 tasks) | ≈ 40–60 CPU-h (chair re-cost from 208–286) + ≈ 160–290 CPU-h | discharges "attribution provisional" on the one production adoption; the A14 T_mat read needs the `off` arm regardless | after the blind readout lands (F2 ordering) |
| 5 | **Mass-law-consistent window design** from the pull read — a k-scan {2, 2.5, 3.5} on iiib (H4 each) and/or a window keyed to the linear-truncated law; joint_r1 at k = 3 | ≈ 5 CPU-h per k-node set (C3 measured 4.97); joint_r1 ≈ 11–15 | bounded by INTERMEDIATE (+0.0035): decides whether any window geometry is MATERIAL; the design object is the impostor pool, not the true host | [RULE] item 1 first — do not spend before the F-ii word |
| 6 | **CMEM higher-power registration** (≥ 90 % at −11.6 %) | ≈ 15 CPU-h local (≈ 30 new mirror seeds × 2 arms), chair estimate | structural class, REPORTED-ONLY cap; low H₀ yield | [RULE] item 8 first |
| 7 | **Zero-compute housekeeping** (do before any tree): file the B7.3 adoption row in `BIAS_HISTORY_LEDGER.md` (next free = #252 after #250/#251 commit); commit rows #250/#251 + the P0 ledger section; replace the `row #<adoption>` placeholder (`bayesian_statistics.py:3274`) and append a note reconciling §13.1's quoted log text with the committed one; run the C0 §11.2 OAT-column identity check on the retrieved c0 CSV; the two P1′ nodes (0.33 CPU-h each); the "two M1s" docstring cross-reference; the driver's duplicate-row assertion; retrieve C4's provenance extras and run the archive script once SSH returns | ≈ 1 CPU-h total | closes verifier items 12/15/16/18/19 cleanly | SSH for the last two |

Not recommended for the next tree: Stage P (moot until rank 1; ≈ 320 CPU-h unsmeared, wall-bound);
B2.2 (A2 not triggered); any k_sky change (B2.2's 4× kernel-cost scaling, runbook 37 §5).

---

## 8. Governance incidents since docket 1 (disclosed; none hidden) and the verifier-scope delta

1. **runner-1 → runner-2 → runner-3 chain and the daemonic-pool limitation.** runner-1: P1 (rc 0,
   20:19:53–20:28:20) → P0 `--jobs 2` **crashed rc 1** at 20:46:40 (`compute_scores()` →
   `pd.concat` "No objects to concatenate", `hier_s0_driver.py:647`; per-seed node results not
   collected across workers) → S0-C (rc 0, 20:46:42–21:38:48) → a KW-Q1 launch (21:38:48) that died
   by `BrokenPipeError` when runner-2 started. runner-2 (fixed driver, `--jobs 2`): P0 **crashed
   differently** at 21:58:20 — `AssertionError: daemonic processes are not allowed to have
   children` (the outer seed pool cannot spawn `evaluate()`'s inner per-event pool; leaked loky
   semaphores) → its KW-Q1 launch (21:58:21) also died. runner-3 (`--jobs 1`): P0 21:59:44–22:49:05
   rc 0 = run of record; KW-Q1 22:49:07–23:12:46 rc 0. Consequences: `--jobs > 1` is dead in this
   driver (documented in `B1_2_DRIVER_EXTENSION_NOTE.md` §8/§8.1); the "5 CPU-h / 40 min at 5
   parallel nodes" P0 re-scope was unattainable; ≈ 8.6 CPU-h nominal of crashed attempts are
   unbanked and absent from the ledger (§4); runner-3's echoed stage label says "jobs2" while the
   flag was `--jobs 1` (cosmetic, disclosed). The 2026-08-20 "never end a turn to wait" rule was
   honoured (runs launched `nohup`, results read from JSON).
2. **Scorer invocation error (KW-Q1):** the first scorer call looked for `node_*_ft/` instead of
   `node_*_ft_sites2.2_nosmear/`, found 0 rows, rc 1 at 23:15:32; re-invoked with `--theta-sites
   2.2 --smear off`; excluded from cost; disclosed in the record and row #249.
3. **SSH outage** (control session expired ≈ 21:15; `Permission denied (publickey,keyboard-
   interactive)` under `BatchMode=yes`; retries stopped on instruction): C4's `posteriors_with_bh_mass`
   h = 0.67/0.73, `run_metadata_*.json`, `logs/`, `GIT_COMMIT_AT_RUN.txt` not retrieved
   (C4 verdict PROVISIONAL on provenance; the gate/stencil numbers are diagnostics-CSV-only and
   complete); C0's sacct not re-pulled for C3's wall comparison; C1 template not converted; wave 3
   not submitted (`DRY_RUN=1`); the wave-2 archive script not run. Only the author can re-authenticate.
4. **GitHub 100 MB rejection and the gitignore.** `git reflog`: the wave-2 readouts commit was
   amended twice (`8520114a` → `cc7f407a` → `0d0eb691`); `8520114a` and `cc7f407a` both included
   `wave2_20260829/c0/posteriors_with_bh_mass/h_0_73.json` (the per-event 2D posterior class —
   c4's siblings are 130,590,466 / 130,602,224 bytes); the final `0d0eb691` drops it and adds
   `.gitignore:101-102` (`results/…/wave*_2026*/*/posteriors_with_bh_mass/`). The rejection is
   evidenced only by the reflog + the ignore rule; **no ledger row or record names the event** — a
   one-line note is recommended (verifier item 18(c)). The branch is in sync with origin at
   `60f9996e`; `COMMIT_PLAN_3.md`'s path-filtered add for the ≈ 93.5 MB of `hier_s0_*` simulation
   intermediates was applied (the wave-2/3 commits carry logs, small CSVs and JSON only).
5. **Log-message citation fix by the orchestrator.** The B7.3 presentation §13.1 and the verifier
   report quote the INFO line as `"[PHYSICS] … ACTIVE (row #249)"`; row #249 was consumed by the
   KW-Q1 readout, and the committed text reads `ACTIVE (adopted under row #223, charter B7.3)`
   (`bayesian_statistics.py:3728`, `d4765539`). Correct in substance; the two records were not
   amended (append-only) and now differ from the code by that string; a `(row #<adoption>)`
   placeholder remains at `:3274`. Rule breach: none (comment/log text; kernel range untouched).
6. **The adoption's `BIAS_HISTORY_LEDGER.md` row is missing.** Presentation §9.2 asked the
   orchestrator to file it; no row cites `d4765539`; the gate ledger has its three rows. Not a
   rule-6 breach (gate presented before code, ledger rows filed, verifier run) — a record gap the
   end verifier's item 12 will hit.
7. **Rows #250/#251 and the P0 compute-ledger section are uncommitted** at chair time (`git diff
   HEAD`: BIAS_HISTORY_LEDGER +10, COMPUTE_LEDGER +21). Docket 1 was filed into a commit the same
   evening; docket 2 should travel with these.
8. **Provenance `tree_dirty_file_count=296` vs A22 "clean":** flagged by C3's reader, resolved by
   C0 §13 (untracked-file count on the cluster checkout; tracked tree clean) — not re-verified on
   C3/C4's own stamps by any independent agent; the launch-stamp wording ("tree clean") should say
   "tracked tree clean". Not a breach.
9. **B7.3 "verified" gate-ledger row is a builder-run smoke pass** (disclosed; the B5.1 precedent);
   the independent verifier was dispatched afterwards and PASSED; its one disclosed gap (5 scripts)
   was closed in the same commit. Rule 2 discharged after the fact, not before.
10. **The `[PHYSICS]` commit carries a 218+/41− reformat of `hier_s0_driver.py`** (ruff format of
    pre-existing long lines around the Class-B pin) — disclosed by the verifier as whitespace-only.
11. **Forensic E7 overclaim** ("own GL-50 quadrature" reuses production helpers outside the hook
    scope) — flagged by the panel, not corrected in the record (append-only).
12. **B3.2 dispatch text** instructed "IMPLEMENT the flag exactly as presented … authorized to edit
    bayesian_statistics.py" against a presentation and gate-ledger row that already said STOP; the
    builder declined correctly under the approval-scope rule. An orchestration drafting error, caught.
13. **Verifier-registration staleness:** item 16's "author item 3 deferred" (resolved by PA-HIER-28);
    item 2's "no forensic root-cause file exists yet" (it now does).
14. **No rule-6 breach found by this chair:** the only physics-trigger edits since `a794404c` are
    the gated commits `1f003da6`, `0b308828`, `d4765539` (presentation before code, ledger rows,
    row #223 in the APPROVED column, verifier run for B7.3); `cc305748`/`fb9d8aff`/`901653a1` are
    test/CLI/harness plumbing with byte-identical defaults. No node ran `git`; every record carries
    the rows #222/#223 stamp. Top-tier cap: ≤ 3 per wave respected (forensic, this chair, the B7.3
    presentation author).
15. **This chair:** two local re-derivation scripts (§0), no `evaluate()`, no `ssh`, no git, no
    source edits; this file is the only write.

**Verifier-scope delta (items to add to `REGISTRATION_END_VERIFIER_PASS_20260829.md`'s 20):**
(a) B1.1-F's decisive numbers — E7 9.2e-13, E11 Z −0.62, E19's 401 → 4001 explanation, E21's KW-Q1
contamination — re-derived from `b1_1_forensic_work/` and the node CSVs (chair (b) is one leg);
(b) the KW-Q1 form-invariance claim narrowed to "exact for D̃^φ/α_G^φ, first-order for the divisor
gap" (§5 item 2); (c) the missing adoption ledger row, the `row #<adoption>` placeholder and the
§13.1-vs-code log-text mismatch; (d) F2 blindness of the two wave-3 sbatch scripts (chair-checked)
and, when the `off` arm is built, that it is a separate registered arm; (e) rows #250/#251 committed
before the verifier runs; (f) the S0-C marginal reconciliation (chair (g)); (g) the C0 §11.2
OAT-column identity check status; (h) the GitHub-rejection incident's absence from the record;
(i) the unbanked ≈ 8.6 CPU-h of crashed runner attempts vs the ledger; (j) the two P1′ measurements
and the two open contradictions as the [HIER] return set (item 16 corrected); (k) this docket's
§6 list against §3's seeded minimum set — new here: items 3 (divisor fix [DO]) and 6 (S0-B word)
carry the forensic's inputs, item 1 folds in the WGEOM §9 F-ii ruling, item 2(c)/(d) are new.

---

*Chair: inherit-tier subagent, scoped package, filed 2026-08-30 (work dated 2026-08-29/30).
Nothing in this docket is an approval request; all path choices are the orchestrator's; every gate,
verdict and choice above goes to the registered end-of-fan-out verifier. Append-only.*

## Appended correction note (2026-08-30, orchestrator; from the docket-2 refuter, severity minor)

1. §4 F4 table, C3 row: the intermediate task-time sum reads "1114 s"; the correct sum is 290 + 279 + 276 + 274 = **1119 s** (the quoted 4.97 CPU-h already matches the per-task values in `b5_2_readout.json`: 4.9734 CPU-h). SUPERSEDES the "1114 s" figure only.
2. §0 item (e): Δℓ″ = −30.36 was re-differenced from 6-decimal per-node values; the full-precision source is `b7_2_readout.json: delta_ell_doubleprime_at_0.665 = −30.311`. Rounding-consistent; the decisive Δmean_h,pred = +0.0025057 and the validity check (|Δℓ″| ≈ 1 % of I_HEAD = 2965) are unchanged.
3. §4 wording: the C0 ratio "9–13× below" is 8.7–13.3× from the cited numbers. Cosmetic.
