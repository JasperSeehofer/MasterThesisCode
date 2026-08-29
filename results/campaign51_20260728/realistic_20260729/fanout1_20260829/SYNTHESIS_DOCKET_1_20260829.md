# SYNTHESIS DOCKET 1 — fan-out 1, wave 1 (2026-08-29)

**Launched under rows #222/#223 — charter node: wave-1 synthesis chair (docket 1).**
**Purpose: INFORMATION ONLY (row #222 governance form (ii)). No approval is requested here; the
orchestrator chooses depth-2 paths on its own judgement; the end-of-fan-out verifier is the author's
check. Every number below carries {value; source; date}. Append-only.**

Chair's context package: runbook 37 §2/§5 (`results/campaign51_20260728/RUNBOOK_NEXT_SESSION_37.md:43-100`),
ledger rows #221–#224 (`gate_b_20260730/BIAS_HISTORY_LEDGER.md:3016-3022`), the eight node records in this
directory with their refuter reports as supplied by the orchestrator (refuter reports for B5.1-implementation,
B6.1 and B8.1 were NOT in the package — `verify_b51/` was empty at 18:43 CEST; the chair substituted its own
spot-checks, disclosed in §7), `docs/RESEARCH_CYCLE.md` stages 5–6 (`:331-402`) and amendments F1–F5 (`:628-652`).

Chair re-derivations performed this session (foreground, all local, < 1 CPU-min total):
(a) B8 instrument re-run `b8_information_floor.py` → JSON byte-identical except `elapsed_s` (0.564 vs 0.564 s);
(b) B3 five-bin n-weighted coverage recomputed from `b3_pop_prediction.json:venues.*.bins`;
(c) direct source reads of `bayesian_statistics.py:3587` (θ/site-2.3 guard), `:3735-3757` (`catalogue_global_selection="auto"`
→ `"phi"` under `absolute_marginal`), `:5187-5191` (no-BH denominator ternary), `:4156/:4166/:4212` (the only three
consumers of `smear_global_selection`), `constants.py:95` (`SIGMA_V_PEC_KM_S = 0.0`);
(d) `PREREGISTRATION_HIER_HTHETA_20260826.md:40-60` (§1.2 `z_centre` symbol) against B6.1's judgment call;
(e) orphan-process state of B1.1's two background runs (`ps` clean; `node_b_minus/simulations/` has no `diagnostics/`;
`s0c_seed900101/.../posteriors/` empty; both logs last written 18:27–18:29 CEST).

---

## 1. Wave-1 verdict table

Registration vocabulary; caps carried verbatim. "Rule outcome" = the charter's depth-1 → depth-2 rule (runbook 37 §2).

| node | verdict of record (caps) | decisive number {value; source; date} | refuter state | rule outcome |
|---|---|---|---|---|
| **B1.1 [HIER] S0-A** | **B0-A / B0-A′ UNDETERMINED** — registered 4-seed × 5-node pooled Z_b/Z_s not computable (1 seed, 2 of 5 nodes complete). GATE ENG **PASS** (b_plus vs truth, 106/106 events move, median rel 0.0198). GATE T-ID (registered, unit-level) **PASS** 20/20. Driver's informal GATE PARITY **NOT EXACT** (`combined_no_bh` max_rel 5.718e-4, undiagnosed). S0-R NOT RUN (already out of scope by PA-HIER-28 item 5). S0-C NOT COMPLETED (no h-point written). All under the **REPORTED-ONLY** cap (PA-HIER-28 item 9). | θ-engaged (smeared) node `evaluate()` = **1190.93 s** vs truth node **64.73 s** — 18.6× the registered §7.1 anchor 63.97 s {`hier_s0_registered_run/logs/s0a_seed900101_full.log`; 2026-08-29 18:27 CEST}. Smeared quadrature **single-core-bound** (94–103 % CPU on a 14-core pin) {`ps` observation in `B1_1_HIER_RECORD.md` §1 item 5}. Site 2.3 **structurally inert for the no-BH channel under `catalogue_global_selection="phi"`** {`bayesian_statistics.py:5187-5191`, chair-confirmed}. | not refuted; **minor** — 4 must-fix citation items (pool-scaling lines 4533-4536/4562 not 4490-4495; ternary 5187-5191; ln-transform of `combined_*` unstated; cap attribution PA-HIER-28 item 9 not "C3 absent") | Band `\|Z\| ≤ 3` **not evaluable**. Instrument partially certified (ENG, T-ID). Depth-2 (S0-B) requires: S0-A completion + an appended amendment (§2 B1). |
| **B2.1 [CMEM] A1** | **R2c NOT-DISTINGUISHED (parked)** — primary equal-weight p = 0.0358 ≥ α = 0.01; direction deficit-consistent. **C-STRUCTURAL-ONLY (row #220) remains the verdict of record; A2 NOT triggered.** REPORTED-ONLY / structural class; single-h; zero H₀-space claim. | T = **−0.12311** ln (outside/inside ≈ 0.884, ≈ 11.6 % deficit), perm p = **0.0358** (10 000 perms, seed 20260829); secondary T_w = −0.10828, p = 0.0522; census bc 190/1168, bt 190/1168, pooled 380/2336 (0.16267) {`cmem_a1_work/cmem_a1_result.json`, `cmem_a1_gates.json`; 2026-08-29}. Pre-registered power at the original −16 % effect: ≈ 68 % {`PREREGISTRATION_CMEM_A1_20260829.md` §8}. | not refuted; **none** — bit-for-bit independent re-execution of the sha1-pinned instrument; one inherited citation looseness (REPORTED-ONLY cap is row #219, not "#216 item 4") | DISPLACED? **No** ⇒ **park with the bound**; B2 closes at depth 1 (no B2.2, no B2.3). |
| **B3.1 [POP]** | **"3.2 warranted"** on both venues — row #138's M1-vs-comoving population-mismatch prediction, independently re-derived, covers the current dark-class score-at-truth. Historical −0.635/−0.565 baselines **STALE**. Zero-compute, no band cap (measure-first read). | Coverage bins 2–5 (z ≥ 0.392): **98.5 % (iiib) / 103.9 % (joint_r1)**; five bins (chair-recomputed from the table's own rows, n = 605/491): **113.1 % / 125.9 %**; the record's "all 5 bins" 114.3 %/129.9 % is the all-dark-event figure (n = 606/493) and silently includes 1/2 events below the bottom bin edge {`b3_pop_prediction.json:venues.*.{dark_ensemble, dark_ensemble_bins2to5_only_robustness, bins, n_underflow_below_bottom_edge}`; 2026-08-29}. HEAD dark-class 1D score **−0.4668 ± 0.0162 (iiib, n = 606) / −0.3938 ± 0.0207 (joint_r1, n = 493)** vs row #138's −0.635 ± 0.017 / −0.565 ± 0.020: **7.16σ / 5.95σ** {same JSON `head_vs_historical`; historical `BIAS_HISTORY_LEDGER.md:1347-1348`, `hier_provenance_stamps_20260826.md:150`}. | not refuted; **minor** — 3 must-fix (mislabelled "all 5 bins" row; cross-check "within 4 %" is 3.9 %/**7.8 %**; CRB md5 attributed to `run_metadata_21.json`, which carries no md5 — correct source `MEASUREMENT_HEAD_READOUT_20260827.md:42-43`) | Coverage ≥ 50 % ⇒ **3.2**: M1-consistent population-prior flag (physics gate) + score-at-truth read riding B1.2's arm. |
| **B4.1 [IMP]** | **NOT EXONERATED; remainder NOT DIFFUSE; a DEFECT (survives at the model's own class composition); NECESSARY cause of the production 1D rail (iiib, ASSUMPTION-JOIN), sufficiency NOT shown; mechanism UNDETERMINED** (kernel width / mixture-weight h-slope / in-ball depth skew). **4.2 read NAMED: "KW-Q1".** Merge into B1 **CONDITIONAL** (declared per charter 4.3). All `[LOCAL]` forecast inputs, no bands. | FT remainder **+0.12274 ± 0.00774** (80.8 % of the coded-leg drag +0.15181; un-rails 12/12 → 0/12) {`b4_imp_stage1_forecast.json:arms.ft.fleet`; 2026-08-29}; lowest-z quartile (z_true < 0.358) carries **91.7 % (ft) / 86.2 % (fc)** of the impostor-leg score; catalogue-share r ≈ −0.77; SNR η² 0.009 {`…covariates.*`}; production iiib full 0.6077 → pure-dark-only **0.7134 ± 0.0277** (c68 TRUE) {`b4_imp_stage1_production_o2.json:iiib`}; O2 reproduced to 4e-17. | not refuted; **minor** — all 13 decisive numbers re-derived from source and matched; exoneration table's 17 citations resolve | 4.2 read named within the ≤ 20 CPU-h envelope: registered **8.4 CPU-h** (`CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3) — chair re-cost §4 (≈ 13.7 CPU-h if run smeared; 8.4 if the §2-B1 "2.2"-equivalence gate passes). |
| **B5.1 [WIN]** | **IMPLEMENTED (not committed)**: `mass_filter_geometry ∈ {"linear","log"}` (default `"linear"`) + `mass_filter_k` (default 1.5), byte-identical default (unit tests + 100 000-pair independent script, 0 mismatches; full suite 1871 passed). Gate ledger rows: presented / presented (revised) / implemented / verified. Zero-compute count: **log k = 3 REDUCES the aggregate candidate count** and **drops true-host retention** — contradicting runbook 37 §5's "cannot add more than 4.2 %" framing. | pass fraction (i) linear k1.5 **0.95768** (gate 0.9577 PASSED) vs (iii) log k3 **0.69509**; true-host retention **0.9567 → 0.7890**; per-event growth (iii)/(i): mean 0.814, median 0.949, p95 1.498, max 10.0; 24-arm jackknife: retention(iii) 0.7898 ± 0.0455 (SE 0.0093), drop ≈ 18 arm-SE {`b5_window_count.json`, `b5_window_count_arm_jackknife.json`; `PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md` §7 + R2/R5; 2026-08-29}. | gate-doc panel **REFUTED the first count** (`gw_window()` used the linear formula under "log"), fixed + re-run by a different agent, every headline unchanged ≤ 1.3e-6 (R1/R2). Implementation-record refuter report **not in the chair's package**; chair spot-checked the JSON against the record (match). | 5.2 counterfactual at k = 3 is warranted **but its arm shape changes** (§2 B5): a 17-point true-host loss is the object, not a candidate-count growth; a zero-compute pull-distribution pre-read is recommended before cluster CPU-h. |
| **B6.1 [ALIGN]** | **IMPLEMENTED (not committed)**: `s` scales the RAW catalogue error BEFORE the PV fold at sites 2.1/2.2/2.3; `b` unchanged. Bit-identical today (`SIGMA_V_PEC_KM_S = 0.0`, `constants.py:95`). 3 gate-ledger rows filed (presented / implemented / verified, `docs/gates/PHYSICS-GATE-LEDGER.md` uncommitted diff). | targeted 27/27; full suite 1851 passed / 15 skipped; θ=(0,1) identity pins green; discriminators at σ_pv = 200 km/s, s = 1.4142 match the pre-fold closed form at rtol 1e-9 {`B6_1_ALIGN_RECORD.md` §5–6; 2026-08-29}. Judgment call: `sigma_z_pv` from the UNSHIFTED host z (prose), not the appended note's z̃ formula literal — **chair check: consistent with the registered §1.2, whose `sigma_z_pv = (1 + z_centre)·σ_v/c` uses the pre-shift symbol** {`PREREGISTRATION_HIER_HTHETA_20260826.md:44-56`}. | refuter report **not in the chair's package**; chair verified prereg consistency (above) and that `SIGMA_V_PEC_KM_S = 0.0` still holds. | **CLOSED at depth 1** pending the orchestrator's `[PHYSICS]` commit (charter: must land before S0-B). Judgment call → end verifier. |
| **B7.1 [2D-TWIN]** | **PROPOSAL complete** (`PROPOSAL_2D_TWIN_ADOPTION_20260829.md`, 568 lines): adopt `catalogue_numerator_survival_2d="mz_sel"`, centre **`eff`** (decided in-proposal; numerically inert at production precision, σ_cond p50 = 8.8e-8); ×2.25–2.35 residual disclosed; C₂\* 2D identity NOT closed (calibration status `supported`, capped). `"auto"` value not yet in code (future gate item). | Wave-2 arm PROD-CF-2D, H4 grid {0.660, 0.665, 0.670, 0.730}: **74.7–101.4 CPU-h** (twin 59.7–81.1 + baseline gate task), ceiling ×1.3 ≈ 132; G27 escalation 418–568 (conditional); T_mat = **0.008** {proposal §6.2 table; `MEASUREMENT_HEAD_READOUT_20260827.md:268-285`; 2026-08-29}. | panel **clean after 0 rounds**: builder-report + verifier-report both non-refuting, minor, no must-fix; §1.5 S-homogeneity bookkeeping not re-derived by either (deferred to falsifier (i), zero compute). | 7.2 counterfactual arm, one venue (iiib), H4 — inside the charter's 50–130 envelope. |
| **B8.1 [CAL]** | **F5 information floor at the production venue (N = 1588) computed**: single-known-host, no-impostor Fisher floor; **with-BH channel adds no rescue at any literature-realistic σ_M** (confirms F5 at the actual N); measured HEAD posteriors ≈ 11× wider (2D) and the 2D centre misses truth by ≈ 38 floor-σ. Stop condition stated (centering ≤ 3 σ_floor; width ≤ F·σ_floor, F unmeasured → B8.2). [INFO-STARVATION] (register §13, OVERTURNED) explicitly NOT resurrected — chair read item 13 and concurs. Builder smoke-test status. | σ_h,floor(1D, σ_z = 0.035) = **0.001747** (0.239 % of h); 2D at σ_M = 1.99 (0.55 dex) = **0.001747** (identical to 4 s.f.); 2D at the informational σ_M = 0.02: 0.001295; spec-z σ_z = 0.0017: 0.000560 {`b8_information_floor.json:oneD/twoD.*.closed_form.sigma_h_floor`; **chair re-run 2026-08-29, byte-identical**}. Route A (numeric FD) unstable at photo-z: 0.000371, n_eff ≈ 5 — a documented negative result. Measured 2D ⟨σ_h⟩ 0.01847, ⟨bias⟩ −0.0668 {`head_readout_extraction_20260827.md`; 2026-08-27}. | refuter report **not in the chair's package**; `b8_information_floor.json` mtime 18:35 (after the 17:21 record) indicates a re-run by another agent; chair's own re-run reproduces every number (deterministic, no RNG). | 8.2: build the two-channel calibration harness ([A3]) — local, no cluster. |

---

## 2. Depth-2 path recommendations (chair's reasoning; the orchestrator decides)

### B1 [HIER] — is 1.2 (S0-B, production θ-score at truth) warranted under REPORTED-ONLY?

**Recommendation: YES, sequenced and re-scoped — not as registered.** Four reasons and four pre-conditions.

Why yes: (1) S0-B is the only read in the tree of the coherence hypothesis *where the tilt lives*; the S0-A
null is a control by construction (prereg §2.1 D7 refinement) and can never carry the finding. (2) Two other
branches consume the same driver (B4.2 KW-Q1; B3.2's score read rides B1.2's arm) — the instrument is now
shared (F3), which raises its yield per CPU-h. (3) Under REPORTED-ONLY the S0-B read is still the decisive
*fork* for B1's depth 3: |Z_b|, |Z_s| ≤ 3 on production ⇒ LEVER-DEAD-AT-N bound ⇒ 1.3b (park, redirect);
either > 3 ⇒ 1.3a (Stage P + C3 build) — a ruling the charter already anticipates. (4) B4.1's C2 localises the
impostor remainder to z_true < 0.358 with catalogue share as the strongest correlate; S0-B "by z + class" reads
directly against that profile (registered cross-branch line L2, §3).

Pre-conditions (all local, before any sbatch):
- **P0 — complete S0-A** (the instrument certification the registration requires before S0-B). Real cost is
  now known: 16 remaining θ-engaged nodes × 1190.93 s single-core + 3 setups × ~458 s + truth/per-event loops
  ≈ **11 CPU-h**, ≈ 2 h wall if run 5 nodes in parallel (the smeared phase is single-core-bound, so
  parallelism across nodes is free) — or ≈ 5 CPU-h / 40 min wall if P1 passes and the remaining nodes run
  unsmeared. Also complete **S0-C** (registered ceiling 15 CPU-h; measured marginal still UNKNOWN — see §6 item 4).
- **P1 — `theta_sites` equivalence gate (new, ~0.2 CPU-h):** re-run seed 900101 b_plus with `theta_sites="2.2"`,
  `smear_global_selection=False` and compare `combined_no_bh` against the banked `theta_sites="all"` b_plus
  CSV. Chair's source read predicts bit-identity: under `"phi"` the no-BH denominator is the θ-inert phi table
  (`bayesian_statistics.py:5187-5191`, `:4212` called with `smear_sigma_z=False` and no θ), and
  `smear_global_selection` has no other no-BH consumer (`:4156` feeds the discarded table; `:4166` is with-BH
  only). If bit-identical, **every no-BH-channel θ-read in the tree (S0-A remainder, KW-Q1, S0-B) can run
  unsmeared at the 65-s anchor**, and Stage P/F (1.3a) re-costs back to the registered anchor rather than 18.6×
  it. Production resolves `catalogue_global_selection="auto"` → `"phi"` under `absolute_marginal`
  (`:3735-3757`), so this applies to S0-B exactly as to the mirror. Site 2.3's θ-dependence stays live only in
  the with-BH channel (secondary/diagnostic in the registration).
- **P2 — appended amendment PA-HIER-31 (registration before running, F3):** (a) b-node: adopt the as-built
  ±0.02 with disclosure OR re-derive ±0.033 from b_max = 0.0661 (PA-HIER-29) — the chair recommends the
  re-derived node for S0-B and a disclosed "as-built" label for the S0-A remainder (paired within arm, so mixing
  is not allowed); (b) register the site-2.3 no-BH inertness under `"phi"` as the instrument's operating fact
  (GATE ENG certifies sites 2.1/2.2 only); (c) relabel `score_s` (linear secant) vs the registered `score_lns`
  (Z identical; magnitudes not comparable to ln-s bands); (d) S0-B read design: 5 θ-nodes × h = 0.73, iiib,
  **by z-bin (B3.1's registered edges) and by class (C-A/C-B/C-C)**, both branches' predicted profiles
  registered first (§3 L1); (e) GATE PARITY residual 5.7e-4: accept as below-band with the batch-order
  hypothesis recorded, or diagnose (one re-run of the banked bc CSV at the current commit decides it).
- **P3 — B6.1's commit lands first** (charter ordering; numerically inert today).

Cost of S0-B itself (§4): 4 θ-nodes at h = 0.730 on iiib (the truth node is the shared baseline gate task):
**60–92 CPU-h** unsmeared, **81–113 CPU-h** if smeared (+4 × 1190.93 s × 16 cpus = +21 CPU-h billed for an
idle allocation). The registered "74.7–101.4" priced 5 nodes at the unsmeared anchor and is superseded by
this band ([A11]).

### B2 [CMEM] — DISPLACED ⇒ 2.2, else park

**Park with the bound.** p = 0.0358 ≥ 0.01 on the registered primary; verdict map prereg §9. A2 (k_sky
1.5 → 3 counterfactual, 105–265 CPU-h) is **not** triggered; B2 closes at depth 1 with C-STRUCTURAL-ONLY
(row #220) as the verdict of record and an ≈ 11.6 % outside-cone deficit, NOT-DISTINGUISHED at α = 0.01 with
≈ 68 % pre-registered power at the original effect size. For the record, not as a recommendation: two
independent fleets now read deficit-direction at p = 0.0152 (row #219) and p = 0.0358 (this node); a pooled
meta-read is **not registered and would be post-hoc** — the end verifier should note it as an unregistered
observation, and any future re-open needs a fresh registration with power ≥ 90 % at the −11.6 % effect.
Consequence for wave 2: the k_sky confound with B5.2 (§3 L3) is removed; k_sky = 1.5 is an invariant of B5.2.

### B3 [POP] — coverage rule

**3.2 warranted: build the M1-consistent population-prior flag through its own /physics-change gate
(presentation before code; approval cites row #223) and register the score-at-truth read on the shared
S0-B instrument.** The chair's conditions: (a) the presentation must state **which legs the M1 prior touches**
— the completion denominator's `dV_c/dz/(1+z)` (`bayesian_statistics.py:1169-1216`) only, or also the
catalogue leg's `w_pop` (G2b) — and register a prediction **per leg**; (b) predictions to register before
running (F3): under the M1-consistent prior, the dark-class score at truth on bins 2–5 moves from −0.612
(iiib) / −0.574 (joint_r1) by ≈ +0.60 (the re-derived predicted term −0.603/−0.597), leaving a residual whose
two-sided band is the open completion-leg-defect object (rows #140–#144 registered ≥ 0.073 at the old
baseline); a residual |·| ≥ 0.2 confirms a second, larger defect; |·| ≤ 0.1 closes the dark-class tilt to the
population term; (c) B4.1's C5 pure-completion arm (production iiib pure-all 0.8396, MAP at the 0.86 edge)
is the second registered readout: the M1 prior should move it toward 0.73 — register the direction and a
band; (d) **exoneration boundary, disclosed for the verifier:** register item 5 [WPOP-TUNING] ("adjusting the
population-rate prior weighting to absorb the residual", NEGLIGIBLE ≤ +0.0004, row 64) is a *tuned weight*;
B3.2 is a *generator-consistent prior replacing a constant-comoving assumption* on the completion leg. The
chair holds these distinct (B3.1's refuter reached the same reading) but the boundary is close enough that
the gate presentation must quote item 5 verbatim and argue the distinction; (e) paper-facing caveat: on real
data the EMRI population is unknown — the flag is a mock-consistency instrument; the residual population
systematic stays in `docs/gates/G7_systematics_budget.md` row 16. Cost: the M1-prior arm at 3 h-nodes
{0.720, 0.730, 0.740} on iiib, **45–69 CPU-h**; the comoving baseline is the banked HEAD readout (zero
compute) if the shared baseline gate task reproduces the banked columns.

### B4 [IMP] — the named 4.2 read

**Run KW-Q1 as registered (`CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3), local, behind P1.** θ = (0, s),
s ∈ {1/√2, 1, √2}, h ∈ {0.725, 0.735}, 4 B-SEL realisations at the FT configuration, frozen q1 set
(z_true < 0.358, ≈ 45 events/seed), R = [S(√2) − S(1/√2)]/|S(1)|, bands OWNS |R| ≥ 0.5 / INERT ≤ 0.2 / MIXED.
Chair's additions: (a) the driver currently hardcodes the b0i/bc configuration
(`hier_s0_driver.py:94-97`: `catalogue_numerator_survival="off"`, `catalogue_global_selection="phi"`); KW-Q1
needs the FT configuration (`"phi"` twin, fused, HEAD Σ^φ) — a **builder adaptation** (builder ≠ runner ≠
B1.1's driver author, rule 2), with its own GATE PARITY against a fresh same-commit FT re-evaluation (not the
2026-08-23 banked CSV, which predates Σ^φ); (b) cost: 8.4 CPU-h as registered **only if P1 passes**
(unsmeared `"2.2"`); otherwise 4 seeds × 2 s-nodes × 2 h × 1190.93 s single-core = +5.3 CPU-h ⇒ ≈ 13.7 CPU-h
(local, wall parallelisable); (c) GATE ENG must be scored on the catalogue leg (`L_cat_no_bh` differs across
s-nodes on ≥ 99 % of active rows) — under P1's finding this is exactly where θ acts, so the gate is
non-vacuous; (d) the merge clause stands as declared: OWNS ⇒ B4 merges into B1 at depth 3 (B1.3a inherits
the q1 localisation and the absorption prediction); INERT ⇒ B4.3 = mixture-weight h-slope derivation
(s_β = −3.2891/h) + the per-candidate instrumented run (3.4 CPU-h, needs a non-physics hook ruling).
The two zero-compute secondaries (validated event_idx → CRB join for the production dark-only read; HEAD-basis
Σ^φ re-anchor of the FT remainder) get a fresh sonnet runner.

### B5 [WIN] — counterfactual arm shape

**5.2 is warranted (author-ratified design, row #221 F-ii), but the object has changed and the registration
must say so before any CPU-h.** B5.1 measured that log k = 3 is *not* a widening: aggregate candidates fall
to 0.726× linear (heavy-side cut re-introduced where linear's negative lower edge was vacuous at CV > 2/3,
≈ 99.6 % of the catalogue) and **true-host retention drops 95.7 % → 78.9 %** (18 arm-SE). Chair's arm shape:
- Venue iiib, wave-2 HEAD commit, arm T = (`"log"`, k = 3.0) vs baseline (`"linear"`, 1.5), H4 grid
  {0.660, 0.665, 0.670, 0.730} (same stencil logic as B7.2: Δmean_h,pred = Δℓ′(0.665)/I_HEAD, I_HEAD = 2965,
  validity |Δℓ″| ≪ I_HEAD; a 0.0015 shift ⇔ Δℓ′ = 4.4 nats per unit h — resolvable, since reads are paired
  deterministic recomputations with ≤ 8.5e-15 reproducibility floor). Band: ΔMAP/Δmean vs HB's **+0.0015**
  (`CLAIM_WGEO_20260827.md` §4.1), two-sided, plus IMMATERIAL/MATERIAL at the HEAD T_mat = 0.008 as a
  secondary edge.
- Registered zero-compute reads through the **production flags** at h = 0.730 (closing R4 falsifier 2):
  per-event candidate growth (predicted median 0.949, p95 1.498, max 10.0), true-host retention on iiib
  (predicted 0.789 ± 0.009 from the mirror census — a genuine two-sided prediction on a different fleet),
  class migration C-A/C-B → C-C (events whose true host leaves the window become dark-class by construction),
  and the R6-style 2D/1D channel split (the mass window is 1D-irrelevant only where the 1D leg ignores mass —
  register which columns must be bit-identical).
- **Pre-read before sbatch (zero compute, local): the true-host mass pull distribution** on the mirror fleet —
  ln(M_z/(1+z_host)) − ln(BH_MASS) over BH_MASS_ERROR/BH_MASS per event (an 20-line extension of
  `b5_window_count.py`). Reason: a "3σ" log window that drops 21 % of true hosts is not a 3σ window of the
  realized scatter (a correctly-budgeted log-normal σ retains 99.7 %); either BH_MASS_ERROR understates the
  realized generator scatter, or the injected M is not tied to the host's catalogue mass (a population-vs-
  catalogue object, B3's class), or the "small-error correspondence" σ_lnM = BH_MASS_ERROR/BH_MASS is being
  applied at median CV ≈ 0.86 where ln(1 + CV) = 0.62 ≠ 0.86 (the gate's own §2 states the correspondence
  holds for ε ≪ 1). The pull read decides which, at zero cost, and the ε = 2Φ(−3) = 0.27 % rationale of F-ii
  rests on it. The k = 3 arm still launches in wave 2 (authorized); the pull read is what makes its H₀ read
  interpretable. A second k-node is a depth-3 option, not wave 2 ("not more because you can").
- Cost: 4 nodes × 14.93–22.9 CPU-h × candidate factor (0.73–1.5) ≈ **44–137 CPU-h** + the shared baseline
  task; SLURM per-task time from the p95 growth factor, not the mean (16 events go from 0 to > 0 candidates).

### B6 [ALIGN] — closed?

**Yes, at depth 1, subject to (i) the orchestrator's `[PHYSICS]` commit carrying the three ledger rows and
(ii) the end verifier's reading of the disclosed judgment call.** The chair checked that call against the
registered arithmetic: prereg §1.2 writes `sigma_z_pv = (1 + z_centre)·σ_v/c` with `z_centre` the *pre-shift*
symbol (the bias line is `z_centre → z_centre + b(1 + z_centre)`), so B6.1's "raw-z σ_pv" implementation is
the registered form and the appended note's z̃ formula literal was the divergent text. Numerically inert
while `SIGMA_V_PEC_KM_S = 0.0`; the new b-order regression test pins it. No depth 2/3 exists for B6.

### B7 [2D-TWIN] — arm

**7.2 = PROD-CF-2D on iiib, H4 grid, 74.7–101.4 CPU-h (ceiling 132), as proposed in §6.2**, with three
chair conditions: (a) falsifier (i) — the S_4D-homogeneity regression test (zero compute) — runs and passes
BEFORE submission; (b) the h = 0.730 task is the STEP-2 smoke that pins the assumed 1.0–1.3 `mz_sel` overhead
(the batch accessor scales as n_cand × 50 × 24 at production candidate counts); (c) falsifier (ii) — the
class-G fleet re-run, 208–286 CPU-h — is **not** a wave-2 arm (row #220: WGEOM forks return separately).
G27 escalation (418–568 CPU-h) only on an AMBIGUOUS H4 read and only with its own launch summary. R1 (eventwise
inequality), R2 (engagement ≥ 0.95), R6 (1D channel bit-identical) are instrument-defect gates; the verdict
map is two-sided at T_mat = 0.008. Adoption stays serialized into the one wave-3 blind HEAD readout (F2).

### B8 [CAL] — 8.2 build with cost

**Build the two-channel calibration harness ([A3]) locally; zero cluster.** Starting asset:
`darksiren_emri/validation/pp_coverage.py` (G4b synthetic-universe P–P harness). What B8.2 must add, per
B8.1 §4: the *actual* candidate-count-vs-z density of the production venue (the thing the single-host floor
omits) so that the dilution factor F in `σ_h,measured ≤ F·σ_h,floor` becomes a measured number rather than the
placeholder 10; both channels (1D, 2D at σ_M = 1.99); coverage AND the absolute-count audit (stage-5 rule:
SBC alone cannot catch a filter both sides share). Cost: builder effort (top-tier design, sonnet build) +
local CPU of order the mirror fleet (24 arms × 65 s × 14 cores ≈ 6 CPU-h per harness sweep). The stop
condition it feeds: centering |⟨h⟩ − 0.73| ≤ 3σ_floor = **0.0052** (already fails by 38–68 floor-σ on HEAD,
so no width-only stop is meaningful before the bias budget is found), width ≤ F·σ_floor with F from the
harness. Design constraint: the harness measures F for an *internally consistent* estimator — it must not
be built or read as a starvation claim (register §13, OVERTURNED).

---

## 3. Cross-branch dependency lines (F1: who inherits which number from whom)

| line | from → to | inherited object {value; source} | governing clause |
|---|---|---|---|
| **L1** | B1.2 ↔ B3.2 | **shared instrument** = the S0-B production θ-driver at iiib, h = 0.730, CoR-P (`normalization_mode="absolute_marginal"` ⇒ `"phi"`). B3.2 inherits the truth node's configuration and B3.1's registered z-bin edges (0.075, 0.392, 0.559, 0.659, 0.753, 1.018) and HEAD dark-class profile (iiib: +0.081, −0.332, −0.562, −0.701, −0.855; `b3_pop_prediction.json:venues.iiib.bins`); B1.2 reads its θ-score "by z + class" on the same bins. **F3: both predicted profiles registered before the first sbatch** — B1's (θ-score null/non-null by class; the impostor-class prediction from L2) and B3's (the −0.60 population term removed under the M1 prior, per leg). | F3, F1 |
| **L2** | B1.1 → B4.2 | the θ-driver (`hier_s0_driver.py`, sha1 `5313c3198f84e3b7e90840d63356851a46677adb`), its GATE T-ID (20/20) and PARITY status (5.7e-4 residual, undiagnosed), **finding 4** (site 2.3 inert for no-BH under `"phi"`) ⇒ P1 equivalence gate ⇒ KW-Q1 unsmeared; the q1 anchor |S(1)| = 0.798 ± 0.042 (ft; `b4_imp_stage1_forecast.json:covariates.ft.z_true`). **Merge clause:** OWNS |R| ≥ 0.5 ⇒ B4 merges into B1 at depth 3; INERT ≤ 0.2 ⇒ B4.3 derivation; MIXED ⇒ both reported. | charter 4.3; rule 2 (three distinct agents) |
| **L3** | B2.1 → B5.2 | A2 NOT triggered ⇒ **k_sky = 1.5 is an invariant of B5.2** (no cone widening in wave 2), so B5.2's candidate factor is attributable to the mass geometry alone; B2.1's census (outside-cone 380/2336 = 0.16267, `cmem_a1_gates.json`) is the reference for any class migration B5.2 reports; B4.1's "18.4 % not-recovered (1D)" class is presumably B2's outside-cone class — recorded, not adjudicated. | F1 |
| **L4** | B4.1 → B3.2 | C5: the dark-class catalogue leg is NECESSARY for the 1D rail (iiib 0.6077 → 0.7134 dark-only) while the **pure completion leg alone is +0.11 high** (pure-all 0.8396, MAP at the 0.86 edge; `b4_imp_stage1_production_o2.json:iiib`) — B3.2 must register a prediction on the pure-completion arm, not only on the score. | F3 |
| **L5** | B3.2, B5.2, B7.2 → one baseline | **one shared baseline gate task** at h = 0.730 on iiib at the wave-2 commit (the row-#201 PROD-A0 ingredient gate: banked per-event columns reproduced to ≤ 1e-12 ⇒ the banked HEAD readout is the zero-compute baseline for all three); all wave-2 arms at the **same commit** (A22 dirty-state stamp = clean). | F2, A22 |
| **L6** | B1.1 → all costing | the 18.6× smeared-cell factor and the single-core-bound finding re-cost every θ-engaged cell (Stage P 424.4 CPU-h registered is under-costed for its 32 smeared cells unless P1 holds); P1 is the switch. | F4, [A11] |
| **L7** | B8.1 → all stop conditions | 3σ_floor = 0.0052 (from σ_h,floor = 0.001747) is the centering band every branch's wave-3 blind readout is read against; F (width dilution) is B8.2's deliverable. | stage 5 stop rule |
| **L8** | B6.1 → B1.2, B4.2 | the s-placement commit precedes any s ≠ 1 node that banks (A22 stamp); numerically inert today. | charter B6 row |
| **L9** | B5.1 ↔ B8.1 | **inconsistent descriptions of `BH_MASS_ERROR`'s content** (B5 §7: the 0.55-dex R&V15 σ_int is "the DOMINANT term in BH_MASS_ERROR"; B8 §0: σ_M = 0.19 is "the code's current fit-only estimate, a known 3–7× under-estimate") — one of these is stale (commit `555f018` added the scatter). Does not change either verdict (B8's headline holds at every σ_M); must be reconciled in B5.2's registration and by the verifier. | rule 3 |

**L2 re-pin (appended note, 2026-08-29 — wave-2 GAP-CLOSURE archive/notes worker, launched under
rows #222/#223 — charter node: NODE archive+minor-notes, GAP 10).** L2's driver sha1
`5313c3198f84e3b7e90840d63356851a46677adb` above is STALE (append-only: the table row is left as
written). The driver's blob will be re-pinned to whatever it is inside the wave-2 commit once that
commit exists: **sha1sum at launch: `<fill>`**. As read now, before any wave-2 commit — today's
`sha1sum` of `hier_s0_driver.py`:

```
9f831b9f7d6b8fed820d547bbe8cd64ff00873e3  results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
```

{command run: `sha1sum results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`, 2026-08-29}

This value may change before the wave-2 commit lands (this file is owned by another agent per the
task's standing rules and this node made no edits to it) — the "sha1sum at launch" line above is
the one that must actually gate any A22 stamp citing L2; today's read is informational only.
{source: `WAVE2_REGISTRATION_CHECK_20260829.md` §1.2 line 127, §5 item 10; 2026-08-29}

---

## 4. Wave-2 batch proposal (F4: CPU-h per arm, argued size, total, archive, deadline, registrations first)

**Anchors used ([A11]):** mirror unsmeared cell 64.73 s @ 14 cpus (≈ 0.25 CPU-h; `s0a_seed900101_full.log`);
mirror θ-engaged smeared cell 1190.93 s single-core (≈ 0.33 CPU-h real; 5.3 CPU-h billed at 16 cpus idle);
**S0-C per-h marginal NOT MEASURED** (the registered read did not complete — disclosed; the wave-3 blind readout
will re-measure it for free); production iiib per h-point **14.93–22.9 CPU-h** (instructed 14.93–20.27 from
`cluster/LAUNCHING_JOBS.md:47`, 3355 events, 2026-07-03; fresher HEAD-config bracket (a) 1.27 sel=off lower
bound, (b) 22.9 HEAD-config off-arm slowest task 1:25:52 — `MEASUREMENT_HEAD_READOUT_20260827.md` §9/§F,
2026-08-27/28). joint_r1 ≥ 2.2× iiib — excluded from wave 2.

### 4.1 Pre-wave, local (no cluster exposure), in this order

| item | what | CPU-h (local) | wall |
|---|---|---|---|
| P1 | `theta_sites="2.2"` unsmeared vs banked `"all"` b_plus, seed 900101 — bit-identity of `combined_no_bh` | ≈ 0.2 | ≈ 10 min (setup-dominated) |
| P0 | complete S0-A (3 seeds × 5 nodes + seed 900101's b_minus/s_plus/s_minus); complete S0-C | ≈ 11 (smeared) / ≈ 5 (if P1 passes); S0-C ≤ 15 (ceiling) | ≈ 2 h / 40 min at 5 parallel nodes; S0-C unknown |
| P2 | KW-Q1 (B4.2) after the driver's FT-config adaptation | 8.4 (P1 passes) / ≈ 13.7 (smeared) | ≈ 1 h parallel |
| P3 | B5 true-host mass pull-distribution read | ≈ 0.01 | minutes |
| P4 | B7 falsifier (i) S_4D-homogeneity regression test | ≈ 0.01 | minutes |
| P5 | B6.1 + B5.1 `[PHYSICS]` commits (orchestrator), clean commit for every wave-2 arm | — | — |
| **pre-wave total** | | **≈ 20–40 CPU-h local** | |

### 4.2 Registrations that must be authored BEFORE the first sbatch (F3)

1. **PA-HIER-31** (appended to `PREREGISTRATION_HIER_HTHETA_20260826.md`): b-node re-anchor decision;
   site-2.3 no-BH inertness under `"phi"` as an operating fact; P1 equivalence result and the resulting
   `theta_sites` for every no-BH read; `score_s` relabel; S0-B by z + class with A15 operating characteristics
   at N = 1588 (per-event ln L secant SEM from S0-A's measured per-event scatter — available once P0 lands),
   A10 invariants + blindness sentence, A14 falsifier, two-sided bands (B0-B ≡ B0-A's |Z| ≤ 3, plus per-class).
2. **B3.2 physics-gate presentation** (M1-consistent population prior flag; 5 items; approval column cites
   row #223) **and** its registered predicted profile on the shared instrument (per leg; pure-completion arm
   band; [WPOP-TUNING] boundary argued verbatim).
3. **PREREGISTRATION_WIN_K3_COUNTERFACTUAL** (B5.2): arm T/baseline, H4 stencil, HB +0.0015 band two-sided,
   retention prediction 0.789 ± 0.009 on iiib, class-migration read, pull-distribution result (P3) and the
   `BH_MASS_ERROR` content reconciliation (L9); A15 at N = 1588 (paired deterministic ⇒ materiality bands).
4. **B7.2 PROD-CF-2D** — already registered in proposal §6.2; add the STEP-2 smoke item (overhead pin) and the
   falsifier-(i) pass record (P4) as an appended note.
5. **B8.2 harness design note** (local; no arm).
6. **Compute-ledger rows** for every arm below (F4) with the archive-scheduled column filled BEFORE launch.

### 4.3 Cluster batch (one submission set, iiib only, HEAD wave-2 commit, `cpu_il`, 16 cpus/task, `--time=03:00:00`, backfill-friendly arrays)

| arm | node | tasks (h-nodes) | CPU-h estimate | argued size |
|---|---|---|---|---|
| **C0** shared baseline gate task | B3.2/B5.2/B7.2 (L5) | 1 (0.730) | **15–23** | one task serves three arms; PROD-A0 ingredient gate ⇒ banked HEAD readout = baseline at zero compute |
| **C1** S0-B production θ-score | B1.2 | 4 (b±, s± at 0.730) | **60–92** (unsmeared) / **81–113** (smeared, +21 billed-idle) | 4 nodes, not a 3×3 grid; one h (score, not posterior); truth node = C0 |
| **C2** M1-prior arm | B3.2 | 3 (0.720, 0.730, 0.740) | **45–69** | central-difference score only; the full posterior under the prior comes free in wave 3 if adopted |
| **C3** log k = 3 counterfactual | B5.2 | 4 (H4) | **44–137** | H4 not G41; candidate factor 0.73–1.5 applied; per-task time from p95 |
| **C4** PROD-CF-2D `mz_sel`/`eff` | B7.2 | 4 (H4) | **60–105** | proposal §6.2 nominal 59.7–81.1, ceiling 105; G27 only on AMBIGUOUS, separate launch summary |
| **wave-2 cluster total** | | **16 tasks** | **≈ 224–447 CPU-h** (upper bound takes C1 smeared) | below the charter's 350–650 because B2.2 (105–265) is not triggered and Stage P is not in wave 2 |

Not in wave 2, by design: S0-R (DISARMED, PA-HIER-28 item 5); B2.2 (A1 not DISPLACED); Stage P (B1.3a, fork
pending S0-B); B7 falsifier (ii) 208–286 CPU-h (returns separately, row #220); joint_r1 arms (≥ 2.2× cost;
T_mat conservative there); G27/G41 grids (wave 3 delivers G41 for every adopted change).

**Fairshare / queue:** the author's rule of record is no CPU-h cap, size on need (row #222); fairshare is at
the floor (skill gotcha 13) — 16 tasks × ≤ 3 h at 16 cpus is backfill-shaped. **Deadline (F4):** workspace
expires **2026-09-23**; 25 days at launch (row #224); a 16-task batch completes in hours-to-days of queue —
comfortable. **Archive-before-run:** Option A in flight, 7 items OK at launch (row #224); every wave-2
out-root is MUST-ARCHIVE tier — the orchestrator confirms the archive status line in the launch summary
before `sbatch`, and each arm's ledger row carries "archive-scheduled: yes" before launch (F4), else the arm
does not launch.

---

## 5. What the end-of-fan-out verifier must inspect

Every gate presentation, verdict and path choice, in this order:

1. **B1.1**: the two code-level findings (site-2.3 inertness under `"phi"`, `bayesian_statistics.py:5187-5191`
   and `:4212`; single-core-bound smeared quadrature, `:1708-1720`) — independently re-read; the P1
   equivalence result; the GATE PARITY 5.7e-4 residual's disposition; the four must-fix citation corrections
   (appended, not edited); the orphaned background runs (§7).
2. **B2.1**: the verdict map application (prereg §9) and the correct non-triggering of A2; the two-fleet
   near-band observation is unregistered (§2 B2).
3. **B3.1**: the integration-measure choice (d log₁₀M, emcee coordinates, `cosmological_model.py:230-292`);
   the table relabel (113.1 %/125.9 % five-bin vs 114.3 %/129.9 % all-dark); the 7.8 % joint_r1 cross-check;
   the STALE-baseline finding (§6 item 1). **B3.2 gate**: legs touched, per-leg predictions, the [WPOP-TUNING]
   boundary argument, row #223 citation in the APPROVED column, three ledger rows.
4. **B4.1**: the exoneration table's 17 rows (mechanism-grepped, both layers); the ASSUMPTION-JOIN (76/1588);
   the cross-fleet composite (−0.146 at w̃_G = 0.062); KW-Q1's registration (A10/A14/A15) and the three-agent
   independence for its run.
5. **B5.1**: the refuted-then-corrected zero-compute count (R1/R2, different agent); the implementer's
   `mass_filter_sigma` interaction resolution (§3 of the record) against the presentation's §2 sketch;
   byte-identity evidence (100 000 pairs); the four ledger rows; **the retention paradox and L9** (§6 item 3)
   before B5.2's sbatch.
6. **B6.1**: the σ_pv judgment call vs prereg §1.2 (chair: consistent) and the appended-note formula literal;
   three ledger rows; commit ordering vs S0-B.
7. **B7.1**: the `eff` centering decision; the §1.5 S-homogeneity bookkeeping (falsifier (i) result); the
   cost band derivation; that `"auto"` is not claimed to exist in code.
8. **B8.1**: the Fisher algebra (`closed_form_fisher`, the `h²` Jacobian placement, the record's own
   caught-and-fixed slip); Route A's rejection; register §13 respected; the L9 `BH_MASS_ERROR` question.
9. **Path choices made by the orchestrator after this docket** (which of §2's recommendations were taken,
   with what deviation) and every wave-2 registration in §4.2 for A8/A10/A14/A15 completeness.
10. **Tree state**: that B1.1's measurements were taken on an uncommitted working tree (B6.1 edits 17:29–17:35,
    B5.1 edits ≤ 17:53, S0-A start 17:58) — byte-identity of both edits at the measured nodes (s = 1;
    linear/1.5 defaults) is argued, not stamped; and that the wave-2 commit is clean (A22).

---

## 6. Findings that are themselves new information (valued outputs, including refuted/undetermined)

1. **The dark-class 1D score at truth has moved 6–7σ since row #138** — HEAD −0.4668 ± 0.0162 (iiib) /
   −0.3938 ± 0.0207 (joint_r1) vs −0.635 ± 0.017 / −0.565 ± 0.020 (7.16σ / 5.95σ; `b3_pop_prediction.json`).
   The predicted population term is essentially unchanged (−0.533/−0.512 vs the memo's −0.555), so the
   coverage rose arithmetically (98.5 %/103.9 % on bins 2–5) because the total shrank under the intervening
   estimator fixes. Row #137/#138 numbers are STALE for citation; G7 row 16 needs re-grading (a [RULE] the
   verifier surfaces to the author, not this docket). Reconciliation with `MEASUREMENT_HEAD_READOUT_20260827.md`'s
   "2D channel MATERIALLY GROWN" is open — different statistics (dark-class 1D score slope vs full-sample 2D
   posterior mean); not contradictory; unreconciled.
2. **[HIER] instrument facts** (code-level, chair-confirmed): (a) under `catalogue_global_selection="phi"` —
   the production default resolved from `"auto"` under `absolute_marginal` — θ's site-2.3 effect is discarded
   for the no-BH channel; every no-BH θ-read certifies sites 2.1/2.2 only; (b) the smeared global-selection
   quadrature is single-core-bound: a θ-engaged cell costs 1190.93 s wall regardless of `--cpus-per-task`,
   18.6× the registered anchor, and Stage P's 424.4 CPU-h is under-costed for its 32 smeared cells unless the
   no-BH read is run unsmeared (P1). The registered S0-C marginal is **UNDETERMINED** (no h-point written in
   > 800 s at n_h = 41 — a shared per-construction precompute the anchor does not price).
3. **The mass window's true-host loss** (B5.1): log k = 3 retains 78.9 % of true hosts vs linear k1.5's 95.7 %
   and *reduces* aggregate candidates to 0.726×; the runbook 37 §5 performance framing ("cannot add more than
   4.2 %") is **refuted in direction**. A "3σ" window losing 21 % of true hosts means the candidate-side
   σ_lnM = BH_MASS_ERROR/BH_MASS is not the realized log-normal σ of the mock (or the injected M is not tied to
   the host's catalogue mass) — decidable at zero compute (§2 B5 pull read); the ε = 2Φ(−3) design rationale
   depends on the answer. **L9**: B5.1 and B8.1 describe `BH_MASS_ERROR`'s content inconsistently.
4. **The F5 floor at the actual production N** (B8.1, chair-reproduced): σ_h,floor = 0.001747 (0.24 % of h);
   the with-BH channel adds no rescue at any literature-realistic σ_M (2D floor = 1D floor to 4 s.f. for
   σ_M ≥ 0.60; 0.5 % apart at σ_M = 0.19); spec-z alone (0.000560) beats every photo-z + mass combination; measured HEAD 2D width is
   10.6× the floor and its centre 38 floor-σ off — the pipeline extracts ≈ 1 % of the single-host Fisher
   information; the gap is an estimator-consistency budget, not starvation (register §13). Route A's
   finite-difference instability at low z is itself evidence that the single-host marginal is degenerate
   under GLADE photo-z (h–z compensation), consistent with row #98's DS-8 T1 rail.
5. **[CMEM] R2c twice near-band, twice NOT-DISTINGUISHED** (p = 0.0152 row #219; p = 0.0358 here, different
   fleets, ≈ 68 % power) — an unregistered pattern, recorded for the verifier; C-STRUCTURAL-ONLY stands.
6. **[IMP] localisation**: the impostor-drag remainder is a low-z (z_true < 0.358), catalogue-share-correlated,
   SNR-blind object; ≈ 63 % of its per-event score rides the global mixture-weight h-slope
   (s_β = −3.2891/h), ≈ 37 % the per-event catalogue-vs-completion slope; removing the dark-class catalogue
   leg alone un-rails production 1D to 0.713 ± 0.028 (assumption-join) while the pure completion leg is
   +0.11 high — **the 1D posterior is a balance of two wrong legs** (B4 and B3 objects respectively).
7. **UNDETERMINED (valued):** B0-A/B0-A′ (S0-A pooled null) — instrument works, grid unfinished at the real
   cost; S0-C marginal; the impostor mechanism (three candidates, KW-Q1 decides the merge); the net sign of
   the k = 3 window's H₀ effect (17-point host loss vs contamination reduction).
8. **REFUTED (valued):** the first B5 zero-compute count's GW-side window formula (gw_window bug, corrected,
   numbers unchanged ≤ 1.3e-6); runbook 37 §5's B5 performance note (direction); B3.1's "within 4 %"
   cross-check on joint_r1 (7.8 %); the B1.1 record's "C3 absent" cap attribution (it is PA-HIER-28 item 9).

---

## 7. Governance incidents (disclosed; none hidden)

1. **B1.1 — registered scope not completed and two orphaned background processes.** 1 of 4 seeds; 2 of 5
   nodes; S0-C not completed. The runner launched S0-A b_minus and S0-C in the background and the session
   ended with both running (record §2.5/§4); both died without output (b_minus: no `diagnostics/`; S0-C: no
   `posteriors/h_*.json`; logs last written 18:27–18:29 CEST) — ≈ 20–30 CPU-min consumed with nothing banked.
   The 2026-08-20 "never end a turn to wait" rule was followed in letter; the unbanked partial compute is the
   cost. Compute ledger B1.1: ≈ 11.2 CPU-h measured vs 35 estimated (partial).
2. **B1.1 — framing**: S0-R "not run" presented as the runner's scope decision; it was already out of scope by
   PA-HIER-28 item 5 (refuter). Four must-fix citation errors remain uncorrected (must be appended as a dated
   note; rule 1 forbids editing the record).
3. **B1.1 — measurements on a dirty tree.** S0-A ran (17:58 →) on a working tree carrying B6.1's (17:29–17:35)
   and B5.1's (≤ 17:53) uncommitted production edits. Both are argued byte-identical at the measured nodes
   (s = 1 ⇒ no-op; linear/1.5 default ⇒ 100 000-pair identity) — argued, not stamped. Wave 2 must run on a
   clean commit (A22).
4. **B5.1 — a refuted instrument, corrected by a different agent** (rule 2 discharged; append-only respected:
   §7 stands with SUPERSEDED markers, R1–R9 appended). The implementer resolved a presentation design gap
   (`mass_filter_sigma` × geometry interaction) by judgment, disclosed. The implementation smoke test is
   builder-run (disclosed). **No refuter report for the implementation reached the chair.**
5. **B6.1 — judgment call** (prose over the appended note's formula literal), disclosed in three places;
   chair finds it matches the registered §1.2. No refuter report reached the chair.
6. **B8.1 — builder smoke-test status**; the chair (a different agent) re-ran the deterministic instrument
   byte-identically — a reproduction, not a panel verification. No refuter report reached the chair.
7. **B3.1 — three must-fix items open** (mislabelled table row; overstated cross-check; wrong md5 source).
   To be appended, not edited.
8. **B2.1 — inherited citation looseness** (REPORTED-ONLY cap cited to row #216 item 4; correct row #219),
   introduced by the prereg text, not the runner.
9. **Compute ledger (F4)**: measured columns for B2.1 (≈ 0.035 CPU-h), B5.1, B6.1, B7.1 left blank; the
   ledger's own note says runners fill them — not done for four nodes. Minor.
10. **Refuter coverage gap**: refuter outputs for B5.1-implementation, B6.1 and B8.1 were absent from the
    chair's package (`verify_b51/` empty at 18:43 CEST). The chair's spot-checks (§0 items a–e) are
    chair-level and are flagged as such for the end verifier.
11. **No rule-6 breach found**: HEAD is still `a794404c`, no stash, no node ran git; physics-trigger and
    `handler.py` edits exist only under the two gated nodes (B5.1, B6.1) with their presentations authored
    before code and ledger rows filed (row #223 form).
12. **Charter cap on top-tier agents** (3 per wave) respected per row #224: B4.1 intake, B7.1 proposal, chair.

---

*Chair: inherit-tier subagent, scoped package, 2026-08-29 evening. Nothing in this docket is an approval
request; all path choices are the orchestrator's; all gates and choices go to the end-of-fan-out verifier.*

---

## L-lines re-cut (orchestrator decision 2026-08-29)

**Launched under rows #222/#223 — charter node B3.** Append-only; nothing above this note is
altered (standing rule 1). B3 is CLOSED as PREMISE-REFUTED (provenance, zero compute;
`B3_1_POP_RECORD.md` superseding note, `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §F/§13).
This closes `WAVE2_REGISTRATION_CHECK_20260829.md` GAP 5 ("§1.3 C2 … L-lines … GAP … the docket
lines must be re-cut by an appended note"). This note re-cuts §3's dependency table (L1, L4) and
§4.3's cluster batch (C2) accordingly; §3's table rows above are left as-is.

- **L1 = B1.2 only** (`from → to`: was "B1.2 ↔ B3.2"). B3.2 is struck — no `completion_population_prior`
  instrument exists and none will be built (§F refutes the premise at zero compute; the C2 arm is
  struck below). The shared instrument's *second registrant* on the S0-B truth node is no longer
  B3's population-term prediction; it becomes **B4's impostor-class prediction via L2**
  (`CLAIM_IMPOSTOR_DRAG_20260827.md` §1.3 / this docket §4.2 item 1's L2 profile prediction,
  `PA-HIER-31` item 10, second bullet "B4 [IMP] (L2)"). B1.2's own F3 registration (θ-score
  null/non-null by class) is unaffected.
- **L4 struck as a dependency** (`from → to`: was "B4.1 → B3.2"). B4.1's C5 finding — the pure
  completion leg alone is **+0.11 high** (`pure-all mean_h = 0.8396` vs 0.73, MAP at the 0.86
  edge; `b4_imp_stage1_production_o2.json:iiib`) while the dark-class catalogue leg is necessary
  for the 1D rail (0.6077 → 0.7134 dark-only) — stays **open** under B4/B1 alone; it no longer has
  a B3.2 arm to register a competing prediction against. **Explicitly noted:** this +0.11-high
  pure-completion finding is now **WITHOUT a competing population explanation** — §F shows the
  production dark-class prior is already the generator's own law, so a population-shape swap
  cannot be invoked to move the pure-completion posterior toward 0.73 (and the physics-change
  presentation's own §6.2 sign argument shows a generator-consistent M1 swap would in any case
  move that posterior further UP, away from 0.73, not toward it — contradicting the docket's own
  §2 B3 condition (c) as originally written). The +0.11 residual is carried forward as an open
  object of the B4/B1 completion-leg thread (rows #140–#144's internal-misnormalization /
  completion-leg-defect line), not attributed to population mismatch.
- **C2 struck from §4.3**, per `WAVE2_REGISTRATION_CHECK_20260829.md` §0 item 3 / §4 row 3
  ("DEVIATE: strike C2; accept the STOP") and `COMPUTE_LEDGER.md`'s wave-2 cost-refinement row
  (C2 → STRUCK, 0 CPU-h). **New wave-2 cluster total per the chair's own re-derivation**
  (`WAVE2_REGISTRATION_CHECK_20260829.md` §3 item 7): **C0 + C1 + C3 + C4 = 13 tasks,
  179–357 CPU-h** (C0 15–23 + C1 60–92 unsmeared-form + C3 44–137 + C4 60–105), superseding this
  docket's own §4.3 table total of 16 tasks / 224–447 CPU-h at the cell level only (that table's
  rows are left unedited, append-only); **+120–173 CPU-h conditional** on the shared baseline gate
  task C0 FAILing (C3 and C4 would each then re-run their own 4-node baseline).

- **§0(c) / §6 item 2(a) corrected by appended note** (`WAVE2_REGISTRATION_CHECK_20260829.md` §5
  item 9, `B1_1_HIER_RECORD.md`'s own appended-note section). This docket's chair re-derivation
  §0(c) ("direct source reads of `bayesian_statistics.py:3587`... `:5187-5191` (no-BH denominator
  ternary)...") and §6 item 2(a) ("under `catalogue_global_selection='phi'`... θ's site-2.3 effect
  is discarded for the no-BH channel; every no-BH θ-read certifies sites 2.1/2.2 only") together
  with B1.1 record "finding 4" are **REFUTED-IN-PART** by the registration-check chair's F-A
  finding: site 2.3 is inert for `L_cat_no_bh` (confirmed — §0(c)/item 2(a) stand for that
  quantity) but **NOT** inert for `combined_no_bh` (max_rel 7.45e-3 via `alpha_G_phi`/`D_tilde_phi`
  under `"all"`+smeared vs `"2.2"`+unsmeared, mechanism: Σ^4D → r_Malm → α_G^φ → D̃^φ,
  `bayesian_statistics.py:2440-2500,4160-4171,5770`). `combined_no_bh` — not `L_cat_no_bh` — is the
  quantity both the P1 equivalence gate and the driver's score consume, so the practical
  consequence (site 2.3 is NOT fully inert for the no-BH read used downstream) reverses the
  headline claim even though the narrower `L_cat_no_bh` sub-claim survives. The GATE-PARITY
  "batch-order" hypothesis for the 5.718e-4 driver-vs-banked-CSV residual is separately
  **REFUTED** (F-B: 9-vs-106-event truth nodes are bit-identical on all 17 columns; a code/config
  delta or process/thread-count effect remains the live hypothesis). See `B1_1_HIER_RECORD.md`'s
  "Chair findings appended, verbatim" section for the full F-A/F-B text.
  {source: `WAVE2_REGISTRATION_CHECK_20260829.md` §0, §5 item 9; verified 2026-08-29}

Stamped: launched under rows #222/#223 — charter node NODE archive+minor-notes (GAP 9),
appended-only, 2026-08-29.

REPORTED.
