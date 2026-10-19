# REGISTRATION — END-OF-FAN-OUT INDEPENDENT VERIFIER PASS (fan-out 1)

`Launched under rows #222/#223 — the author's stated check on the Fan-out Charter.` This is a
**fresh registration** (not an amendment to any node record; nothing under any other file's
divider is edited by this document).

**Registered:** 2026-08-29 · **Runs:** after docket 2 and the wave-3 blind HEAD readout — this
document is authored now, ahead of that data, per amendment F5's requirement that the verifier
pass itself be *registered* (predictions/scope fixed before the panel sees results). **Class:**
process/governance registration for a verification Workflow, not itself a physics measurement.
No H₀-space claim is made or adjudicated here.

## 0. Why this document exists, and what it is not

Row #222 (`BIAS_HISTORY_LEDGER.md:3018`, author verbatim): *"you can continue through each
consecutive task of the branches without waiting for my approval. It will be double checked at
the end. I ratify the entire tree and trust your judgement on which path to take in the branches
based on the results, but you will be checked with a verifier afterwards."* Row #223
(`:3020`, verbatim): *"everything that is part of the tree can be decided including production
changes. It will be checked afterwards."* Amendment F5 (`docs/RESEARCH_CYCLE.md`, ADOPTED
2026-08-29 with the row #222 substitution) names the concrete replacement for per-wave depth
gates: *"(i) append-only records at every node exactly as before, (ii) one synthesis docket per
wave for information (not for approval), (iii) a registered end-of-fan-out VERIFIER pass
(independent panel; verdicts 'refuted'/'undetermined' explicitly valued)."* Runbook 37 §5 gives
the panel's shape directly: *"register it as its own workflow: sonnet panel + ≤1 top-tier
adjudicator; 'refuted' and 'undetermined' are valued outputs."* This document is that
registration.

**This is the author's check, not a rubber stamp.** Every item below is written so that a
"refuted" or "undetermined" verdict is a first-class, expected, valued outcome — not a defect in
the verifier's work. Per CLAUDE.md's binding approval-scope rule, no item here is pre-approved by
any of rows #216–#223: every gate/adoption inside the tree was pre-*authorized* to be *decided*
without waiting, but the decisions themselves — and, separately, the physics-change ADOPTIONS
that flipped production defaults — are explicitly named by row #223 as being "in the end-of-fan-out
verifier's mandatory scope." This registration is how that scope is discharged.

**Tiering (CLAUDE.md, author mandate 2026-08-07):** mechanical/formulaic stages → `sonnet`;
adversarial verification → up to the top-tier hard cap. This workflow: **one `sonnet` verifier per
inspection item (fresh context each), one dedup/conflict `sonnet` pass, ONE top-tier
adjudicator.** That is 1 top-tier agent for the whole pass — three under the ≤3-per-workflow hard
cap — because panel redundancy (20 independent re-derivations) substitutes for model tier, per
the mandate's own text: *"Every fanned-out stage ... runs `sonnet` regardless of its adversarial
label — panel redundancy substitutes for model tier."* Effort: `high` for each sonnet verifier
(this is adversarial re-derivation of a decisive number, not a mechanical lookup); `xhigh` for the
top-tier adjudicator only (synthesis + judgment-call adjudication across 20 independent reports).

---

## 1. The mandatory inspection list

Twenty items. Each gets **one independently-launched `sonnet` verifier, fresh context** (no
prior involvement in the node being checked — the same "builder ≠ runner" independence rule the
fan-out itself used throughout). Each verifier receives, as its dispatch: the record path(s), the
one decisive number to re-derive from source (not to trust from the record), the source file:line
citation to re-open, what a REFUTED finding would look like for that specific item, and the cap
the item's verdict already carries (so the verifier does not accidentally upgrade a REPORTED-ONLY
or structural-class finding into something it is not). Falsification brief (per amendment A20,
carried into every verifier's dispatch verbatim): **try to refute it. Re-execute the decisive
computation from the raw source artifact the record cites — not the record's own restatement of
the number — and return exactly one of `refuted` / `confirmed` / `undetermined`, with the
evidence for whichever you return.** A verifier that returns `confirmed` without independently
re-running the computation has not done the job.

### Item 1 — B1.1 (wave-1 record): code-level findings + GATE PARITY residual + must-fix citations

**Record:** `B1_1_HIER_RECORD.md` (+ `B1_1_HIER_BUILD_NOTE.md`); ledger row #225.
**Re-derive:** the two code-level claims at their cited line numbers — site 2.3 structural
inertness for the no-BH channel under `catalogue_global_selection="phi"`
(`bayesian_statistics.py:5187-5191`) and the smeared-quadrature single-core-bound cost claim
(θ-engaged `evaluate()` = 1190.93 s vs the truth node's 64.73 s, 18.6× the registered §7.1 anchor
63.97 s, `hier_s0_registered_run/logs/s0a_seed900101_full.log`). **What refuted looks like:** the
ternary at `:5187-5191` actually does route site 2.3 into the no-BH denominator under `"phi"`
(this would directly contradict F-A's later finding at item 3 below and must be reconciled, not
silently banked alongside it); or the 1190.93 s figure is not reproducible from the log (e.g. it
includes queueing/setup time not attributable to the quadrature itself). **Cap:** REPORTED-ONLY
(PA-HIER-28 item 9); minor, 4 disclosed must-fix citation corrections (pool-scaling line numbers,
ternary line numbers, ln-transform of `combined_*` unstated, cap attribution) — verify these were
appended, not silently edited into the original record (rule 1).

### Item 2 — B1.1 Stage-0: S0-A/S0-C independent-reader verdict, B0-A′ INSTRUMENT-DEFECT

**Record:** `B1_1_HIER_STAGE0_RECORD.md`. **Re-derive:** `Z_b = −3.676`, `Z_s = −7.079` (pooled,
N=461 event-instances, 4 seeds, registered primary channel `ln_L_no_bh`), from the raw
`s0a_seed<seed>/node_*/simulations/diagnostics/event_likelihoods.csv` files under
`hier_s0_registered_run/` — this is a from-scratch score/GATE-ENG/GATE-PARITY re-implementation,
not a read of the driver's own JSON. **What refuted looks like:** the pooled `|Z|` does not clear
3.0 on independent re-computation (the STOP would not have been warranted); or the dark-class
(`L_cat_no_bh==0`) 5-event subset does *not* return exactly zero on both axes (this is the
`PA-HIER-31(d)` instrument-identity check — a nonzero here would itself be a second,
independently-discovered defect). **Cap:** REPORTED-ONLY (PA-HIER-28 item 9); this is a valued
STOP, not a claim being refuted — a verifier confirming Z_b/Z_s does not "pass" anything, it
confirms the instrument correctly halted. **No forensic root-cause file exists yet
(`B1_1_S0A_DEFECT_FORENSIC_20260829.md` is not on disk at registration time)** — the verifier's
job is to confirm the STOP is real and correctly scoped (sites 2.1/2.2 only, under
`theta_sites="2.2"`/`smear off`), not to diagnose the defect. **Returns to the author:** whether
S0-B/C1 may launch before the forensic lands (the record itself states explicitly: "this finding
does NOT license a Stage-P or Stage-F launch, an S0-B launch, or a C1/C3 build").

### Item 3 — B2.1: [CMEM] A1, R2c NOT-DISTINGUISHED, parked

**Record:** `B2_1_CMEM_A1_RECORD.md`; ledger row #226. **Re-derive:** `T = −0.12311` (ln
outside/inside ≈ 0.884), permutation `p = 0.0358` (10,000 perms, seed 20260829, N_out pooled
380/2336) from `cmem_a1_work/cmem_a1_result.json` + `cmem_a1_gates.json` — bit-for-bit
re-execution of the sha1-pinned instrument, not a re-read of the JSON. **What refuted looks
like:** `p < 0.01` on independent re-run (this would mean A1 was mis-scored as
NOT-DISTINGUISHED and A2's cone-widening trigger condition was wrongly left unfired); or the
census (bc 190/1168, bt 190/1168) does not match the banked fleet's own event counts. **Cap:**
REPORTED-ONLY, structural class, zero H₀-space claim (row #219's cap, correctly cited here —
note the record's own text carries an inherited citation error attributing the cap to row #216
item 4, which the verifier should flag as a must-fix, not re-litigate). **Returns to the
author:** the k_sky/CMEM pooled-observation note — two independent fleets now read
deficit-direction at p=0.0152 (row #219) and p=0.0358 (this node); the record itself states a
pooled meta-read "would be post-hoc" and is explicitly *not* a recommendation. The verifier
confirms this framing was honored (no pooled p-value was computed or banked) and surfaces the
observation to the adjudicator's returns-to-author table.

### Item 4 — B3: coverage read (B3.1) + STOP declined (B3.2) + closure PREMISE-REFUTED

**Records:** `B3_1_POP_RECORD.md` (+ `b3_pop_prediction.json`, `b3_1_pop_measure.py`);
`B3_2_POP_FLAG_RECORD.md`; `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` §F; ledger rows
#227, #234, #240. **Re-derive — the §F provenance chain, three independently reproducible reads:**
(1) the production dark-host draw law `(1−f)·dVc/dz/(1+z)` at `dark_siren_injection.py:328`
(commit `03cfe80`) is byte-identical in functional form to the estimator's own constant-comoving
completion prior; (2) the HEAD dark-class 1D score at truth, −0.4668±0.0162 (iiib, n=606) /
−0.3938±0.0207 (joint_r1, n=493), from `b3_pop_prediction.json:head_vs_historical`, compared
against row #138's historical −0.635±0.017/−0.565±0.020 (7.16σ/5.95σ move — confirm this
divergence independently, it is the reason the historical baseline is STALE); (3) the five-bin
chair-recompute 113.1%/125.9% (n=605/491) vs the record's own flagged mislabel "all 5 bins"
114.3%/129.9% (n=606/493, which silently includes 1–2 sub-bottom-edge events) — re-derive both
numbers from `venues.*.{dark_ensemble,dark_ensemble_bins2to5_only_robustness,bins,
n_underflow_below_bottom_edge}` in the same JSON. **What refuted looks like:** the dark-host draw
law is NOT byte-identical to the completion prior on closer reading of `dark_siren_injection.py`
(this would resurrect the C2 M1-prior counterfactual arm that was struck at zero compute); or the
7.16σ/5.95σ divergence does not reproduce (this would mean row #138 is not actually STALE and G7
row 16 should not be re-graded). **Cap:** B3.1 zero-compute, no band; B3.2 gate PRESENTED WITH A
STOP, dispatch-to-implement correctly DECLINED per the approval-scope rule (an approval — "implement
the flag exactly as presented" — cannot propagate to a decision whose own inputs, the STOP itself,
postdate it); B3 closure PREMISE-REFUTED, C2 struck, 0 CPU-h spent, L1/L4 re-cut. **Returns to
the author:** G7 row 16 re-grade (docket §6 item 1) — a [RULE] the verifier surfaces, not itself
adjudicates.

### Item 5 — B4.1: [IMP] claim card, NOT EXONERATED

**Records:** `B4_1_IMP_RECORD.md` (+ `B4_1_IMP_DECOMPOSITION.md`, `CLAIM_IMPOSTOR_DRAG_20260829.md`);
ledger row #228. **Re-derive:** FT remainder +0.12274±0.00774 (80.8% of the coded-leg drag
+0.15181; un-rails production 1D from 12/12→0/12 railed) from
`b4_imp_stage1_forecast.json:arms.ft.fleet`; the production HEAD pure-dark-only figure 0.7134
(MAP 0.70, σ 0.0277, c68 TRUE) against the full-posterior 0.6077, from
`b4_imp_stage1_production_o2.json:iiib` — re-run O2 to the record's own claimed 4e-17
reproduction tolerance. **What refuted looks like:** the remainder IS exonerated on a fresh grep
of both exoneration layers against the mechanism (not the tag — see the standing gotcha
"rule-1 exoneration checks are insufficient"); or the ASSUMPTION-JOIN (event_idx==CRB row order,
0.04786=76/1588) does not actually match production's known in-catalogue fraction, invalidating
the pure-dark-only decomposition. **Cap:** all `[LOCAL]` forecast inputs, no bands; the verdict
itself (NOT EXONERATED, NECESSARY but sufficiency NOT shown, mechanism UNDETERMINED) is the
valued output — re-verify all 17 exoneration-table rows resolve against their citations, not just
a sample.

### Item 6 — B4.2: KW-Q1 readout, KERNEL-WIDTH-INERT

**Record:** `B4_2_KWQ1_READOUT_RECORD.md` (+ `b4_2_readout.json`); ledger row #249. **Re-derive:**
`|R| = 0.084812` (pooled, 4 seeds 900101–900104) against the 0.2 INERT ceiling and the 0.5 OWNS
floor, from `S(1/√2)=−1.0456670`, `S(1)=−1.0205308`, `S(√2)=−0.9591134` in
`kwq1_registered_run/kwq1_score_output.json`; per-seed R = +0.1563/+0.0386/+0.1105/+0.0516 —
confirm none individually crosses 0.2. **What refuted looks like:** R crosses 0.2 on independent
re-computation (KERNEL-WIDTH would OWN, not be INERT, materially changing B4's merge-vs-derivation
path decision at row #249's own trailing orchestrator note). **Cap:** REPORTED-ONLY, with an
explicit, carried, *unresolved* instrument disclosure — the same `hier_s0_driver.py` family
returned the item-2 B0-A′ INSTRUMENT-DEFECT on a different (score-at-truth null) test; the record
argues the two designs differ enough that the defect is not automatically inherited, but states
plainly the instrument "is not yet certified clean." The verifier must not treat KW-Q1's own GATE
I (7.613e-8, tol 2e-6, PASS) and GATE ENG (486/486, PASS) as certifying the shared driver overall —
only as certifying this specific paired-comparison design. Also verify the A14 falsifier is NOT
withdrawn: q1 share of Σ s_imp at truth = 92.25% (≥ the 50% floor) from the same JSON.

### Item 7 — B5.1: [WIN] gate implemented, byte-identity

**Records:** `B5_1_WIN_RECORD.md` (+ `b5_window_count.py`, `b5_window_count_arm_jackknife.py`,
`PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md`); ledger row #229. **Re-derive:** pass fraction
linear-k1.5 = 0.95768 (gate 0.9577, PASS) vs log-k3 = 0.69509; true-host retention 0.9567→0.7890;
24-arm jackknife retention(log-k3) 0.7898±0.0455 from `b5_window_count.json` +
`b5_window_count_arm_jackknife.json` — independently re-run the 100,000-pair byte-identity script
against the default (`mass_filter_geometry="linear"`, `mass_filter_k=1.5`), expecting 0
mismatches. **What refuted looks like:** a nonzero mismatch count on the byte-identity re-run
(the default flip would not be byte-identical, contradicting the gate's central claim); or the
log-k3 retention figure does not reproduce (this directly feeds item 9's C3 readout and item 5's
L9 reconciliation). **Cap:** implemented-not-committed at registration time; the record discloses
a refuted-then-corrected first count (the `gw_window()` formula bug) — the verifier confirms the
correction is real (headline numbers changed ≤1.3e-6, per R1/R2) and that runbook 37 §5's
"cannot add more than 4.2%" performance framing is REFUTED IN DIRECTION by this node's own
numbers (log-k3 *reduces* aggregate candidates to 0.726×, it does not add).

### Item 8 — B5.2: C3 log-k3 counterfactual READ OUT INTERMEDIATE + pull-read/L9

**Records:** `B5_2_WIN_K3_READOUT_RECORD.md` (+ `b5_2_readout.json`); `B5_2_PULL_READ_20260829.md`
(+ `b5_pull_read.py`, `b5_pull_read.json`); ledger rows #235, #247. **Re-derive — the one decisive
number:** `Δmean_h,pred = +0.003523` via the I_HEAD stencil (`Δℓ'(0.665)=10.444` nats/h,
`I_HEAD=2965`, from `b5_2_readout.json:stencil.*`) — recompute the central-difference stencil from
the raw `Δℓ(h)` values at h={0.660,0.665,0.670,0.730} (job `6738999`, all COMPLETED) and confirm
it falls strictly between IMMATERIAL (≤0.003) and T_mat (0.008) — INTERMEDIATE, not adopted.
**Separately re-derive the R1 falsifier:** true-host retention 66/76 = 86.842% identical between
arm T and baseline B (source: `logs/wave2_c3_task3_6738999.err:9917`), FALSIFYING the mirror-fleet
prediction of a 17–21pp retention loss — and confirm the documented mechanism (all 621 collapsed
events are dark-class, 0/76 in-catalogue events change membership) from the
`event_likelihoods.csv`/`prepared_cramer_rao_bounds.csv` join. **For the pull-read (L9):**
re-derive the empirical `|pull|≤3` fraction = 78.8% under `σ_lnM = BH_MASS_ERROR/BH_MASS` from
`b5_pull_read.json`, and confirm it matches item 7's independently-measured 78.9% true-host
retention to within 0.2 points — this is the reconciliation that resolves the apparent B5.1-vs-B8.1
contradiction over what `BH_MASS_ERROR` means. **What refuted looks like:** the stencil
recomputation lands outside INTERMEDIATE (either clearing T_mat, which would argue for adoption,
or falling at/below IMMATERIAL, which would argue against ever re-litigating); or the R1
mechanism does not hold on the independent join (would mean the retention paradox is
unexplained, not resolved). **Cap:** REPORTED, adoption gate NOT granted; the record itself flags
an unresolved contradiction (the provenance stamps' `tree_dirty_file_count=296` vs the A22 stamp's
"clean" claim) that the verifier should re-check, not wave through. **Returns to the author:** the
F-ii design question with these numbers — row #247's own orchestrator path decision states this
explicitly: "the F-ii design question returns with these numbers to the end-of-fan-out
verifier/author as a fresh [RULE]."

### Item 9 — B6.1: [ALIGN] θ-hook `s`-placement [PHYSICS] commit

**Record:** `B6_1_ALIGN_RECORD.md`; ledger row #230; confirmed landed as one of the four
post-`d04d9dc9` commits cited in the C0 registration (`REGISTRATION_C0_BASELINE_GATE_20260829.md`):
`1f003da6`. **Re-derive:** targeted suite 27/27 + full suite 1851 passed/15 skipped; the
discriminator match at σ_pv=200 km/s, s=1.4142 against the pre-fold closed form at rtol 1e-9 —
independently re-run the targeted test file and the closed-form comparison script. **What
refuted looks like:** the discriminator does not match the closed form at the claimed tolerance
(the `s`-before-PV-fold ordering would be wrong); or `SIGMA_V_PEC_KM_S` is not actually 0.0 in
`constants.py` today (the byte-identity claim would be false — this is directly checkable). **Cap:**
CLOSED at depth 1, `[PHYSICS]` commit landed; the record discloses a judgment call (prose over
the appended note's z̃ formula literal) that the chair found consistent with the registered §1.2 —
the verifier independently checks this consistency rather than trusting the chair's finding, since
no refuter report reached the chair for this node (a disclosed governance gap, item 18 below).

### Item 10 — B7.1: [2D-TWIN] proposal, `eff` centering decision

**Records:** `B7_1_TWIN_RECORD.md` (+ `PROPOSAL_2D_TWIN_ADOPTION_20260829.md`); ledger row #231.
**Re-derive:** σ_cond p50 = 8.8e-8 (numerically-inert centering claim) from the proposal's own
identity check; the cost band 74.7–101.4 CPU-h (twin 59.7–81.1 + baseline gate task) against
`cluster/LAUNCHING_JOBS.md:47`'s per-h-point anchor. **What refuted looks like:** the `eff`
centering is NOT numerically inert at production precision (this would mean the "decided
in-proposal" design choice was not actually free, and the ×2.25–2.35 residual disclosed in the
proposal deserves more scrutiny than a footnote). **Cap:** proposal complete, C₂* 2D identity NOT
closed, calibration status "supported, capped." Confirm the refuter panel's "0 rounds, clean"
claim by independently checking the §1.5 S-homogeneity bookkeeping item the record discloses as
"not re-derived by either" builder or verifier, deferred to falsifier (i) — item 11's falsifier
(i) PASS is the actual discharge of this deferral; the verifier here should confirm that
deferral chain, not assume it resolved itself.

### Item 11 — B7.2: C4 PROD-CF-2D READ OUT IMMATERIAL-PREDICTED

**Records:** `B7_2_TWIN_CF_READOUT_RECORD.md` (+ `b7_2_readout.json`); `B7_2_FALSIFIER_I_RECORD.md`;
ledger rows #236, #248. **Re-derive:** `Δmean_h,pred = +0.0025057` (at or below T_mat/2 = 0.004)
via `Δℓ'(0.665) = +7.429`, `Δℓ'' = −30.3`, `I_HEAD=2965`, from `b7_2_readout.json:stencil.*` —
recompute the stencil from the raw per-h `Δℓ(h)` values (jobs `6739000`/`6739001`, all
COMPLETED). Independently re-run falsifier (i) (`test_survival_2d_homogeneity_falsifier.py`, 4
tests) and confirm: twin `combined_wbh` invariant under S_4D rescaling (rel. dev. 2.60e-16 at
c=0.4) — PASS; coded form NOT invariant (rel. dev. 1.500) — correctly discriminates. **What
refuted looks like:** the stencil recomputation does not land at/below T_mat/2 (this would move
the readout out of IMMATERIAL-PREDICTED and into the same INTERMEDIATE territory as item 8,
changing the adoption calculus in item 12); or the falsifier does not reproduce PASS on
independent re-run (homogeneity would not hold, undermining the entire adoption's premise).
**Cap:** PROVISIONAL — provenance extras pending retrieval (the SSH outage interrupted retrieval
before falsifier (ii)'s attribution could complete, row #220's registered falsifier stays unrun).
Verify R1 (0/6352 violations), R2 (982/982 engaged), R6 (1D bit-identical, max_abs 0.0) all
independently from the raw job logs, not from the JSON's own gate-summary fields.

### Item 12 — B7.3: [2D-TWIN] `[PHYSICS]` adoption (presentation + implementation + verifier)

**Records:** `PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` (gate presentation);
`B7_3_ADOPTION_IMPLEMENTATION_RECORD.md`; `B7_3_ADOPTION_VERIFIER_REPORT.md` (an
already-completed independent verification, itself a check on the implementation);
`[PHYSICS]` commit `d4765539`. **Re-derive:** independently confirm the default flip is confined
to exactly the nine declaration sites the presentation names — re-run the twelve decisive pin
tests listed in the existing verifier report (§4), especially `test_off_matches_the_pre_flag_
golden_across_modes` (three parametrizations) and `test_kernel_default_pair_bit_identical_to_
explicit_mz_sel_eff` — and independently re-run the full fast suite, expecting **1896 passed, 15
skipped, 27 deselected** (net +7 over the pre-adoption baseline of 1889). **Re-check the five
archived-script pins** (`scripts/mass_trunc_ab.py:151-152`, `volume_trunc_ab.py:150-151`,
`eddington_m_impact.py:164-165`, `ablation_cube_seed600.py:155-156`,
`quick_validation_15.py:84-85`) — confirm each now pins `catalogue_numerator_survival_2d="off"`
explicitly (the disclosed completeness gap the existing verifier report closed in its own
addendum) and that `ruff`/`mypy`/`py_compile` are clean on all five. **What refuted looks like:**
the flip is NOT confined to the declared sites (a value changed somewhere the presentation did
not name — this is the single most consequential possible finding in the entire inspection list,
since it is a production default flip); or the suite counts do not reproduce; or any of the five
script pins is missing or incorrect on re-inspection. **Cap:** this is the ONE production-default
flip this fan-out has made so far (row #248's orchestrator path decision: "it is the ONLY
adoption candidate of wave 2"); per row #223 it was pre-authorized to be *decided* without
waiting, but the gate itself is explicitly named as being in this pass's "mandatory scope." A
`refuted` verdict here does not undo the commit (this document has no authority to touch code or
git) — it returns the finding to the author as the highest-priority item in the adjudicator's
table. **Note the existing verifier report is evidence, not authority** (standing rule, this
repo): the item-12 sonnet verifier must independently re-run the pin tests and the full suite
itself, not accept the prior report's PASS table at face value — it may of course cite and cross-check
against it.

### Item 13 — B8.1: [CAL] F5 information floor computed

**Record:** `B8_1_CAL_FLOOR_RECORD.md` (+ `b8_information_floor.json`,
`b8_information_floor.py`); ledger row #232. **Re-derive:** `σ_h,floor(1D, σ_z=0.035) =
0.001747` (0.239% of h) via the closed-form Fisher floor; confirm the 2D floor at σ_M=1.99 (0.55
dex) equals 0.001747 to 4 s.f. (i.e., the with-BH channel adds no rescue at any
literature-realistic σ_M); confirm Route A's numeric finite-difference instability at photo-z
(0.000371, n_eff≈5) is a genuine negative result, not a coding error. **What refuted looks
like:** the with-BH channel DOES measurably tighten the floor at a literature-realistic σ_M (this
would reopen the with-BH-mass-channel rescue question the F5 forecast closed); or the closed-form
Fisher algebra (the `h²` Jacobian placement the record discloses as a "caught-and-fixed slip")
does not reproduce on independent derivation. **Cap:** [INFO-STARVATION] (register §13,
OVERTURNED) explicitly NOT resurrected by this finding — the verifier must not re-open that
closed question even if the floor numbers look severe; the gap between the 0.001747 floor and
the measured HEAD 2D width (≈11× wider, centre 38 floor-σ off) is registered as "an
estimator-consistency budget, not starvation."

### Item 14 — B8.2: [CAL] two-channel calibration harness design note

**Record:** `B8_2_HARNESS_DESIGN_20260829.md`; ledger row #237. **Re-derive:** the honest cost
correction — 130–475 CPU-h local (not the docket's "≈6 CPU-h per sweep" anchor, a 20–80×
correction) — by checking the arithmetic against the stated production N=1588, ≥100 universes,
and the disclosed UNMEASURED per-`evaluate()`-call N-scaling. **What refuted looks like:** the
cost correction itself does not hold up (e.g., the N-scaling assumption is wrong in the other
direction, meaning the true cost is closer to the original docket anchor after all). **Cap:**
design note only (top-tier, no code, no run) — nothing here is a registered band; the verifier
confirms no band or A15 power claim was smuggled into a document that is explicitly scoped as
input to a future stage-2 registration.

### Item 15 — C0: baseline gate PASS + costing-anchor correction

**Records:** `REGISTRATION_C0_BASELINE_GATE_20260829.md`; ledger row #246. **Re-derive:** the
bit-identical gate result (max_abs 0.000 on all 14 non-trivial shared numeric columns, SLURM job
`6738998`, Elapsed 00:06:28) directly from `wave2_20260829/c0/diagnostics/event_likelihoods.csv`
vs the banked `headreadout_20260827/iiib/event_likelihoods.csv`; independently recompute the
**costing-anchor correction** — job's own per-task Elapsed ≈6.5 min × 16 cpus ≈ 1.7 CPU-h vs the
15–23 CPU-h estimate (a 9–13× overestimate), and confirm the 56–76 min/h-value anchor
(`cluster/LAUNCHING_JOBS.md:47`) is indeed sourced from the unrelated 3355-event set, not this
1588-event iiib venue. **What refuted looks like:** the gate is NOT bit-identical on independent
column-by-column re-diff (this would mean the banked HEAD readout is not a valid zero-compute
baseline for C3/C4, invalidating items 8 and 11's cost claims and possibly their readouts); or the
costing correction itself does not reproduce from the raw `sacct` record. **Cap:** none — this is
a hard PASS/FAIL gate at ≤1e-12 relative tolerance; a `refuted` verdict here is maximally
consequential (it would mean C3/C4's baseline was never actually validated).

### Item 16 — B1.2 PA-HIER-31: S0-B registration + open items routed as fresh [RULE]s

**Record:** `PREREGISTRATION_HIER_HTHETA_20260826.md:1951-2211+` (PA-HIER-31 + REVISION NOTE 1 +
REVISION NOTE 2); source skeleton `WAVE2_REGISTRATION_CHECK_20260829.md` §2; ledger row #242.
**Re-derive:** the F-A finding's own numbers (seed 900101, 9 shared events, b=+0.02, h=0.73):
`L_cat_no_bh` bit-identical (max_rel 0.0) between `"all"`+smeared and `"2.2"`+unsmeared forms,
but `combined_no_bh` diverges at max_rel **7.45e-3**, driven by `alpha_G_phi` (−12.0%) and
`D_tilde_phi` (−0.745%) — re-run this 9-event comparison directly from the two forms' raw
event-likelihood CSVs. **Confirm the open items routed to the author as fresh [RULE]s**: at
minimum, (a) REVISION NOTE 1's registered OPEN CONTRADICTION (`PA-HIER-10`'s unconditional
`smear_sigma_z=True`-at-every-node CoR-M pin vs `PA-HIER-31(b)`'s CoR-P-scoped narrowing) —
R1′/R2′ in REVISION NOTE 2 found this narrowing claim itself CONFIRMED-material-wrong on its own
terms and registered a *second*, distinct open contradiction for the CoR-M instance; (b) the
original register's author item 3 (`docs`/prereg §"(ii) Genuinely OPEN-FOR-AUTHOR" item 3: whether
the θ-hook requires the full `/physics-change` protocol or the lighter instrumentation guard) —
this pass's own text states it "deliberately declines to make on the author's behalf." The item-16
verifier's job includes **completing this count** against the full "(ii) Genuinely OPEN-FOR-AUTHOR"
list and the two REVISION NOTEs — the exact number of still-open pre-existing items is not
independently confirmed at registration time and must be enumerated, not assumed, by the
verifier. **What refuted looks like:** the F-A divergence does not reproduce (the CoR-P-faithful
θ form decision for S0-B would lose its stated justification); or an item believed closed by
PA-HIER-29/30 turns out still open (undercounting the return-to-author list). **Cap:**
registration text, no band; PA-HIER-31 itself is not a measurement, it is the design S0-B will
run under — nothing here is CPU-h spent yet (C1 has "NOT submitted" status at last ledger entry,
row #245).

### Item 17 — Path choices + tree state (docket §5 items 9–10)

**Record:** `SYNTHESIS_DOCKET_1_20260829.md` §5 items 9–10; row #239's verbatim orchestrator
dispatch block; row #238 (wave-2 registration check, F-A chair re-derivation). **Re-derive:**
independently confirm which of docket §2's ten recommendations were taken as-is vs deviated from
(row #238 §4's table: AGREE on 7 of 10 rows, DEVIATE on B1/B3/the wave-2 batch), and that each
deviation is argued with a citable number, not asserted. **Confirm the dirty-tree finding**:
B1.1's S0-A measurements ran (17:58→) on a working tree carrying B6.1's (17:29–17:35) and B5.1's
(≤17:53) uncommitted production edits — the argument that both are byte-identical at the measured
nodes (s=1 no-op; linear/1.5 default) is *argued, not stamped*, per docket §7 item 3's own
disclosure. **What refuted looks like:** one of the "AGREE" rows in row #238's table does not
actually hold up against the cited numbers (the deviation table itself would need correction); or
the dirty-tree byte-identity argument is wrong (S0-A's numbers would be contaminated by
mid-flight production edits — a serious provenance defect requiring a re-run). **Cap:** none —
provenance integrity is a hard requirement (`docs/RESEARCH_CYCLE.md` A22/G4b-adjacent rule: "no
registered measurement executes while a concurrent workstream mutates the working tree").

### Item 18 — Governance incidents (docket §7 + runner-1/2 + SSH outage + >100MB gitignore risk)

**Records:** `SYNTHESIS_DOCKET_1_20260829.md` §7 (12 disclosed incidents);
`B1_1_HIER_STAGE0_RECORD.md` §1 (runner-1/runner-2/runner-3 chain);
`B7_2_TWIN_CF_READOUT_RECORD.md` §2 (SSH outage interrupting provenance retrieval);
`COMMIT_PLAN_3.md` §4–5 (the ~93.5MB simulation-intermediate exclusion). **Re-derive/re-check:**
(a) the runner-1→runner-2→runner-3 chain — runner-1 crashed with `pd.concat ValueError: No
objects to concatenate` (per-seed results not collected across `--jobs>1` workers); runner-2
crashed differently with `AssertionError: daemonic processes are not allowed to have children`
(a nested `multiprocessing.Pool` inside a pool worker); runner-3 (`--jobs 1`) is the run of
record — confirm this diagnosis from the three raw logs, not the record's narrative; (b) confirm
the SSH outage genuinely interrupted C4's provenance-extras retrieval (not silently dropped
data that should have been retrieved); (c) confirm the commit-hygiene risk: `hier_s0_registered_run/`
and `hier_s0_work/` together carry ~93.5MB of `simulations/` intermediates that COMMIT_PLAN_3.md
explicitly excludes via path-filtered `git add` (only `*.log` + small per-node
`event_likelihoods.csv`, individually <4MB) rather than a blanket directory add — verify this
filtering was actually applied to whatever commit ultimately lands the wave-2 tree (GAP item 1),
since GitHub's single-file size behavior makes a blanket add of that subtree a real risk, not a
hypothetical one. **What refuted looks like:** any of the three crash diagnoses is wrong (the
actual root cause differs from what's logged — this would matter for whether the driver is safe
to reuse in a future tree); or the wave-2 commit, once it exists, is found to have swept in the
excluded intermediates anyway. **Cap:** governance/process findings, not physics claims — no
band applies; disclosure completeness is the standard (docket §7's own framing: "disclosed; none
hidden").

### Item 19 — Compute ledger totals and F4 compliance

**Record:** `COMPUTE_LEDGER.md` (all sections); cross-reference every measured-cost append cited
in items 8/11/15 above plus row #249's KW-Q1 figure (6.152 CPU-h measured vs 8.4 estimated).
**Re-derive:** sum the measured wave-2 cluster total (C0 1.7 + C3 4.97 + C4 6.8 ≈ 13.5 CPU-h
measured, against the registered 179–357 CPU-h estimate band — confirm this ~13–26× favorable
miss is consistent across every arm's own `sacct`-sourced number, not just the ones already
cross-checked in items 8/11/15) and confirm the F4 deadline gate (workspace expiry 2026-09-23) is
still comfortably clear. **What refuted looks like:** a measured-cost figure does not match its
cited `sacct`/log source (the ledger would be unreliable as the F4 deadline-gate instrument
itself). **Cap:** none — F4 is a hard compliance check ("no arm launches inside a workspace-expiry
window unless its outputs are archive-scheduled").

### Item 20 — Wave-3 blind HEAD readout (pending) and its T_mat=0.008 falsifier

**Status at registration:** NOT YET RUN. This item is registered now, before the data exists, per
F5's requirement and per this document's own stamp ("runs after docket 2 and the wave-3 readout").
**What the verifier must inspect once the readout lands:** (a) that F2 (serialized adoption) was
honored — B7.3's `mz_sel`/`eff` flip is the *only* production-default change riding this readout,
per row #248's own path decision, and no second adoption slipped in alongside it; (b) that the
readout's own Δmean_h reading against B7.3's flip is compared to the *same* T_mat=0.008
materiality threshold (`MEASUREMENT_HEAD_READOUT_20260827.md:268-285`, ratified row #213) used
throughout items 8/11/12, and that any per-change attribution is drawn only from the registered
C4 arm (row #248), never from the blind readout's own delta directly (F3/F2's explicit
prohibition: "any per-change attribution comes only from registered arms, never from the
readout's delta"); (c) that the falsifier (ii) still-unrun status (row #220) is either
discharged or explicitly disclosed as still provisional in whatever claim the readout supports.
**What refuted looks like:** a second, un-registered production change is found riding the same
readout (an F2 violation); or per-change attribution is drawn from the blind delta rather than
the registered arm. **Cap:** none — F2 compliance is a hard governance requirement, not a
physics band.

---

## 2. Dedup + conflict stage

After all 20 sonnet verifiers report, one further `sonnet` pass runs **dedup + conflict
detection** (not adjudication) over the 20 reports:

1. **Cross-item consistency.** Item 1's site-2.3 inertness claim and item 16's F-A finding are
   about the *same code path* read two different ways (informal driver check vs a chair-registered
   9-event comparison) — flag if the two verifiers' independent re-derivations disagree, since row
   #225 and PA-HIER-31 already flagged this as "REFUTED-IN-PART" and any residual disagreement
   is itself a finding. Similarly cross-check items 7/8 (B5.1 retention 78.9% vs B5.2/pull-read
   78.8%, item 8) and items 9/16 (B6.1's `1f003da6` ordering vs PA-HIER-31 item 12's "`1f003da6`
   precedes ✓ (L8)" claim).
2. **Duplicate must-fix items.** Several nodes (B1.1, B3.1, B2.1) carry inherited citation errors
   the docket already disclosed — the dedup pass confirms these are the *same* disclosed items
   reappearing, not new ones, and does not double-count them in the adjudicator's severity
   ranking.
3. **Collect every "returns to the author" item** verbatim from the 20 reports into one list,
   deduplicated, for the adjudicator's table (§3 below gives the known minimum set; the dedup
   pass's job is to confirm nothing was dropped and nothing was duplicated across items 3, 4, 8,
   16, and 20).

Output: one consolidated conflict/dedup memo, `sonnet`, `high` effort, handed to the adjudicator
alongside the 20 raw reports (not in place of them — the adjudicator reads both).

## 3. The top-tier adjudicator

**One inherit-tier agent.** Writes the verifier report: a verdict table (one row per item, 1–20,
columns: verdict [refuted/confirmed/undetermined], decisive number re-derived, must-fix count,
cap), and — separately and prominently — **the list of every [RULE] that returns to the author**,
collected across all 20 items and the dedup pass. The known minimum set to seed this table (do
not treat as exhaustive — the adjudicator's job is to confirm completeness against the 20 raw
reports):

- **F-ii design question with C3's numbers** (item 8; row #247's own orchestrator note names this
  explicitly).
- **PA-HIER-31's open items** (item 16): REVISION NOTE 2's R1′/R2′ open contradictions
  (CoR-P narrowing correction; the separate CoR-M-side contradiction) plus the original register's
  author item 3 (physics-change scope ruling, explicitly deferred).
- **The B7.3 `[PHYSICS]` adoption ratification** (item 12) — pre-authorized to be *decided*
  under row #223, but its gate is explicitly named as mandatory verifier scope; the author's
  ratification of the flip itself is a fresh [RULE] this pass returns, not something row #223
  already settled.
- **G7 row 16 re-grade** (item 4; docket §6 item 1 — the dark-class score's 6–7σ move against a
  now-STALE row #138 baseline).
- **The S0-B question after the (as-yet-unfiled) forensic** (item 2) — whether S0-B/C1 may launch
  once the B0-A′ INSTRUMENT-DEFECT is diagnosed, and under what scoping.
- **The k_sky/CMEM pooled-observation note** (item 3) — two independent near-band reads
  (p=0.0152, p=0.0358); explicitly not a recommendation, but flagged for the author's awareness.
- **The higher-power R2c follow-up vs bank-and-park** — carried forward from row #220's own "NOT
  covered" list; item 3's A1 parked outcome (p=0.0358 ≥ 0.01) answers the *immediate* trigger
  question (A2 not fired) but does not itself re-rule on the standing "one word still required"
  item from row #220 — the adjudicator confirms whether this is now moot or still open.
- **The WGEOM §9 F-ii consequence ruling** (carried from row #220's "NOT covered" list; a distinct
  F-ii from B5's, predating this fan-out — the adjudicator checks whether it was addressed by any
  wave-1/2 node or remains untouched).

The adjudicator does **not** re-run any computation itself — its inputs are the 20 verifier
reports plus the dedup memo. Its output is the single reviewable artifact this pass exists to
produce, per the repo convention that decision-gating material goes in a persistent, re-readable
document, not a chat summary.

---

## 4. Fan-out arithmetic

**20 items × 1 sonnet verifier each = 20 sonnet agents, fresh context, launched in parallel** (no
inter-item dependency prevents parallel launch — each verifier reads only its own node's records
and re-executes only its own decisive computation). **+ 1 sonnet dedup/conflict pass** (sequenced
after all 20 land) **+ 1 top-tier adjudicator** (sequenced after the dedup pass). **Total: 22
agents, 1 of which is top-tier** — comfortably inside the CLAUDE.md hard cap of ≤3 top-tier agents
per workflow. This fan-out's shape (N items × 1 sonnet, computed before launch, per the tiering
mandate's requirement that "any phase whose fan-out depends on an earlier phase's output must
carry an explicit cap in the script") has no earlier-phase dependency to cap — the 20-item list
is fixed by this registration, not discovered at runtime.

## 5. A15-style operating characteristic — expected single-pass miss rate

This fan-out's own wave-1/wave-2 refuter-panel rounds give a direct, in-repo empirical base rate
for how often a single sonnet pass catches a real issue on its first look, rather than an assumed
number:

| refuter-panel round (this fan-out) | result |
|---|---|
| B5.1 zero-compute count, round 1 (`b5_window_count.py`) | REFUTED the first count — found the `gw_window()` linear-formula-under-"log" bug, corrected by a different agent, all headlines ≤1.3e-6 unchanged after the fix |
| B7.1 [2D-TWIN] proposal panel | clean at 0 rounds — no must-fix |
| C0 registration panel (row #241) | round 1 returned **6 must-fix items** (column-count corrections, missing coverage for 3 new columns, C1 consumption row, archive-cell citation, A8 scoping) |
| PA-HIER-31 panel (row #242) | round 1 returned **5 must-fix items**; round 2 (chair re-check) found no residual |
| B7.3 independent verifier (existing, item 12's predecessor) | found **1 disclosed gap** (5 archived scripts not on the site list) — closed same-day by an addendum |

**Of 5 completed refuter-panel rounds this fan-out, 4 found at least one must-fix item on first
pass (80%).** None of the 4 reversed a verdict — every catch was a citation, completeness, or
formula-scope correction, not a wrong headline number. This is the operating characteristic to
carry into this registration's own panel: **expect roughly 1 in 5 of the 20 items to sail through
clean on a single sonnet pass with zero findings, and expect the other ~4 in 5 to surface at
least one correction** — most of them minor/citation-class, some (as B5.1's was) load-bearing.
A verifier item returning zero findings is not itself evidence of a weak pass; conversely, an
item 12 finding of **any** kind (given its production-default-flip stakes) should be weighted far
above a citation correction on B1.1 or B3.1. The two governance-coverage gaps this fan-out already
disclosed on its own (docket §7 item 10: "refuter outputs for B5.1-implementation, B6.1, and B8.1
were absent from the chair's package") are a direct argument for why *this* pass insists on fresh,
independent sonnet dispatches for every item rather than reusing any prior panel's output as
sufficient.

## 6. Cost

All 20 items are foreground reads + small independent re-computations against already-banked
CSVs/JSONs/logs — **no cluster CPU-h**, no `sbatch`. The heaviest per-item cost is item 12's full
fast-suite re-run (~2 minutes wall) and item 2's/item 8's from-scratch score re-derivation from
banked per-event CSVs (seconds to low minutes, no `evaluate()` calls). Total wall time is
dominated by 20 parallel agent turns, not compute; there is no CPU-h line for this registration in
`COMPUTE_LEDGER.md` because no cluster or GPU resource is touched. Token/agent cost: 20 sonnet ×
`high` effort + 1 sonnet dedup × `high` + 1 top-tier × `xhigh` — the only `xhigh` in this workflow,
justified per the tiering mandate's own rule ("uniform-xhigh workflows are a drift smell; justify
each xhigh") by the adjudicator's job being synthesis and judgment-call resolution across 20
independent reports plus the standing-decision-worthy returns-to-author table, not a mechanical
rollup.

---

**Registered 2026-08-29 under rows #222/#223; runs after docket 2 and the wave-3 blind HEAD
readout.**
