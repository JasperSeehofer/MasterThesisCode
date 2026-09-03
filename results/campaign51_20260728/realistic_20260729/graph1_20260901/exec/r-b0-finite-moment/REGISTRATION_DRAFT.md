# r-b0-finite-moment — REGISTRATION DRAFT: the finite-moment b0 identity statistic at HEAD

Date: 2026-09-03 (night). Node: r-b0-finite-moment (Research Graph 1 ADDENDUM, Branch N [B0FM], §1.5) — **DRAFT**.
Author of record for all scientific decisions: Jasper Seehofer. Prereg author: top-tier subagent (xhigh),
the same identity as r-sealed-mock (addendum §5.3). Status: **PROPOSED THROUGHOUT.** Authorization: addendum
§3 row A-N1 (authoring + the zero-compute B-R design gate on banked CSVs, under row #325). Band content and
any launch return as fresh RULE **d-b0fm-band**. max_revisions 2 (ORCHESTRATOR-DERIVED; addendum §1.7 / parent
§1.13 provisional default). Research-cycle stages 0–2 applied; the B-R design gate is banked in
`BR_CONTROL_RECORD.md` (this directory). A21-B0-C (row #177 verdict review) binds the bands. Append-only after commit.

## 0. Existence contract

| input | state | evidence |
|---|---|---|
| rows #177–#180 (UNDISCRIMINATING; item-3 grant; DRAFT; Gate-B adjudication) | present | `gate_b_20260730/BIAS_HISTORY_LEDGER.md:2590-2689` |
| `CLAIM_B0_FINITE_MOMENT_20260824.md` (F-0; C-A/C-B/C-C candidates; conditioned targets B-T ≈ 1.59, B-C ≈ 0.52) | present | `realistic_20260729/` |
| `PREREGISTRATION_B0_IDENTITY_20260823.md` (§4 bands, §7 blindness, §8 costing lines 173–174, BAND FREEZE, VERDICT) | present | `realistic_20260729/` |
| `PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md` + PA-CA-1…11 + banked verdict | present | rows #181, #186 (banked), #187 (RATIFIED) |
| 24 banked b0i CSVs + selection JSONs (`3bd6b564`, FULLY RECOVERABLE) | present | `p3_b0_work/`; `DATA_INVENTORY.md:280` |
| C-A RHS synthetic chunks (225; 0–4 contaminated, fresh re-score dirs absent locally) | present (220 clean) | `ca_rhs_work/` |
| scorers | present | `p3_b0_identity_test.py` (2 031 lines), `ca_rhs_scorer.py` (2 577 lines) |
| b0i generator (`catalogue_selected`) | present; hosts ∝ w_g·S̃_φ,g, NO per-galaxy mass weighting | `darksiren_emri/validation/correspondence_1d.py:1246-1268, 446-475` |
| b0i2d generator (`catalogue_selected_2d`, venue mass law) | present as code; the [P3-2D] thread is **PARKED** | rows #198–#211 (row #211 "PARKED at UNATTRIBUTED-bounded"); `STUCK_P3_2D_SYMPTOM_CARD_20260826.md` |
| fresh HEAD b0i pairs | absent (not run; nothing launched tonight) | — |

## 1. Provenance chain and what is genuinely open (stage 0) [DOC]

Row #177: the mean-of-odds statistic is UNDISCRIMINATING (B-R passes the vacuous band; k̂ up to 2.7).
Row #178 item 3: the finite-moment redesign is granted. Row #179/#180: the DRAFT derives F-0 (the intake filter
`distance_relative_error < 0.10` removes 41.8 % of drawn b0i events, 1 397/2 400, class-asymmetric) and ranks
C-A (bounded-transform identity, φ = w) first. **Row #181: C-A is REGISTERED; row #186: C-A EXECUTED and
banked TWIN-CALIBRATED (T_w(B-T) = −0.001294 ± 0.001223; coded −17.4σ from its own model value, at its
twin-law-derived displacement −0.86σ; GATE B-R at predicted value; C-TCI in band); row #187: RATIFIED.**
The addendum's §0 correction 1 ("no mention of finite-moment after row #186") is literally true because the
redesign was completed under the name C-A before row #186. Hence the branch's genuinely open content is not
"register the redesign" but three narrower things:

1. **The POWER form of the B-R control.** C-A's GATE B-R is a scorer-validation gate (the identity holds
   exactly for the rescaled arrangement by construction, PA-CA-4); the addendum demands a control in which the
   refuted arrangement must FAIL the band. That control is defined and run in `BR_CONTROL_RECORD.md` §2/§4:
   **DISCRIMINATING** (|T| = 0.0192 = 3.8 × band, |Z| = 16.6; every C-TCI member ≥ 3.1σ; Λ > 40σ) —
   on the stale basis, verdict-free by construction.
2. **The basis.** The banked fleet and C-A pre-date `cf4f8a2a`, A18, A14 (rows #202/#286/#284); the
   with-BH columns of the banked fleet are bit-identical across arms (row #189: "ZERO twin-2D information").
3. **The blindness disclosure the brief needs.** C-A calibrates the estimator against its OWN generator
   (PA-2 aligned premise: hosts drawn from the twin's law) — "self-consistency of the pipeline's own mixture,
   not a real-universe claim" (row #186 item 1). No statistic on b0i can lift that; §6 says so.

## 2. The registered statistic, null and bands (carried from C-A; nothing re-derived) [DOC]

Identity (row #180 adjudicated): C\*·E_{q_G}[φ(w)·R] = E_{q_Ḡ}[φ(w)], φ = w ⇔ C\*·E_G[1−w] = E_Ḡ[w].
- **Primary T_w(a) = LHS(a) − RHS(a)**, LHS_s = (C\*/200)·Σ_acc(1−w_e) (drawn-count normalised, F-0 inside
  the design, dead rows included), RHS = (1/N_syn)·Σ_ALL-draws w_a·1_acc from the completion-class predictive
  scorer — PA-CA-1. Variance ≤ C\*²/4n (proved); k̂ retained as an invariant (finding if > 0.7).
- **Null:** T_w(a) = 0 for the correctly arranged mixture. **Band:** |T_w| ≤ max(3σ_comb, 0.005), t₁₁ quantiles
  (PA-CA-7a); realized σ_comb 1.22e-3 ⇒ the 0.005 anchor dominates. A21-B0-C trigger carried verbatim
  (UNDETERMINED iff k̂_max > 0.7 AND (raw/PSIS disagree OR 3·SEM > 10·ε)).
- **Controls, both mandatory:** (i) GATE B-R exact (scorer): |LHS_BR − RHS_BR| ≤ 0.005; (ii) **GATE B-R
  POWER (new):** |LHS_BR − r·RHS_BR| > 0.005 with |Z| > 3 — the band must reject the refuted arrangement on
  the run's own data; failure = VACUOUS ⇒ no verdict, the addendum's park branch.
- **Verdict map** = C-A §4 as amended (PA-CA-2 coherence slope κ̂): TWIN-CALIBRATED / TWIN-MISCALIBRATED /
  VENUE-MISSPEC / CONTROL-FAIL; plus VACUOUS (power control fails).
- **Robustness twins:** C-TCI indicator profile τ ∈ {30, 100, 300, 1000} (PA-CA-7b; asymmetric power
  disclosed); C-B Λ̄ REPORT-ONLY unless its nulls separate ≥ 3σ (PA-CA-7c).
- **Blindness list amended (addendum requirement):** the F-0 intake filter is now INSIDE the statistic
  (drawn-count normalisation) and named in §6; [F0-SEL] is RESOLVED-BOUNDED for production (row #189:
  0.13–0.59 %) — the 41.8 % is venue physics (donor resampling), not a production selection.

## 3. Why "fresh HEAD pairs at the production default" is not a like-for-like rerun [DOC]/[INFER]

- The 1D catalogue leg at HEAD is `catalogue_leg_1d_mass_aware="auto"` → engaged "on" under phi/phi/θ-off
  (row #286). Its per-candidate factor is S_4D(d_L(z;h), M_g(1+z)) — the galaxy's own with-BH survival
  (`bayesian_statistics.py:7076-7110`). The b0i generator draws hosts ∝ w_g·S̃_φ,g with "no per-galaxy mass
  weighting" (`correspondence_1d.py:1246-1268`). ⇒ With mass-aware ON the PA-2 aligned-generator premise is
  broken by construction: the identity would read VENUE-MISSPEC whatever the leg's correctness. The aligned
  generator for a mass-aware leg is the b0i2d venue mass law (`catalogue_selected_2d`), whose thread is
  PARKED (row #211). This is why the [HIER] driver pins the flag "off" at all 7 sites (row #287;
  `fanout1_20260829/hier_s0_driver.py:518,526,559`).
- With mass-aware pinned OFF (a warned COUNTERFACTUAL at HEAD, row #286), the no-BH path at HEAD reproduces
  the banked b0i CSVs to the E19 comparand floor: no_bh max_rel 2.03e-5–4.88e-5 at the T1.2 truth node
  (addendum §1.0 q-parity-growth, `hier_s0_registered_run/s0a_score_output.json`); RHS-F at HEAD 0.0 both
  arms (PA-CA-11). The Σ^φ slot was already "phi" in every banked meta (PA-CA-6). ⇒ **fresh 1D pairs with the
  flag off carry ≈ zero information beyond row #186** (a byte-identity stamp, not a measurement).
- The 2D channel: with-BH columns at HEAD differ across arms (A14 mz_sel/eff) but the 2D identity needs the
  b0i2d RHS machinery (`ca_rhs_scorer.py` `stage_rhs2`, [P3-2D] `ca_rhs_work2d/`) — PARKED with a STUCK card.

Consequently the measure node has three admissible shapes; the choice is the author's (§7):

| shape | arm pair | information | cost (sourced) |
|---|---|---|---|
| **N1 byte-id stamp** | HEAD, mass-aware OFF (twin default vs coded explicit off), 12 seeds, H_GRID_FULL 46 nodes ([P3-HGRID] rows #182–#184: the h-grid is NOT a free choice) | HEAD-vs-banked no-BH parity (E19-class); the identity numbers re-read = row #186 within the floor | 24 × 0.478–0.9 CPU-h = 11.5–21.6 (`PREREGISTRATION_B0_IDENTITY_20260823.md:173-174`; banked bc_900101 wall 2 003.8 s); RHS re-score NOT needed if the byte-id precheck passes (RHS-F at HEAD = 0.0) |
| **N2 mass-aware identity** | HEAD, mass-aware ON, generator extended to draw ∝ w_g·S̃_4D,g (b0i2d revival) + new RHS pass + new registration | the first identity read of the production 1D leg as flipped | UNSCOPED: b0i2d revival (rows #198–#211 PARKED; the repair fleet `p3_2d_fleet_repair_20260827`, 48 arm-seed pairs), RHS pass realized 6.9 wall-h / registered 20–90 CPU-h fallback (PA-CA-7d; `ca_rhs_score_output.json` elapsed_s 24 703) |
| **N3 close on C-A** | no fresh pairs | q-b0-finite-moment settled at the self-consistency level by rows #186/#187 with the §6 blindness recorded; the brief's "awaits a catalogued-host identity test" line updated to cite C-A; the b0 identity of the mass-aware leg deferred to the [P3-2D] revival | 0 |

RECOMMENDATION (flagged, not a ruling): **N3**, with N1 only if the chair wants a HEAD parity stamp on the
dossier (it is a g-byte-id gate, not a science arm). N2 is the real next question but is a [P3-2D]-class
registration outside this addendum's cap.

## 4. Measure-node registration (applies to whichever shape the author picks; N2 needs its own prereg)

Venue b0i, fused cell, Σ^φ slot "phi" (explicit, never "auto" — PA-CA-6), 12 seeds 900101–900112 (paired
with the banked fleet ⇒ a HEAD-minus-banked per-seed delta is a free secondary), H_GRID_FULL (46 nodes,
`correspondence_1d.py:353-360`), read at h = 0.73. Arms: B-T = production default (`catalogue_numerator_survival`
auto→phi), B-C = explicit "off" (coded), B-R = the r-rescale of B-T (r = 1.515548762178686, registered literal
`ca_rhs_scorer.py:285`, re-derived per run by `_r_h_gen`). Every flag stamped RESOLVED in the metas (A22).

Gates (fail ⇒ NO-READ unless A21-amended): **g-byte-id precheck** (1 seed, HEAD vs banked `bc_900101`
no-BH columns ≤ 1e-12 or the E19 floor with the mechanism named; ≈ 0.5–0.9 CPU-h, BEFORE the fleet) ·
GATE ACC (12 realized n_kept inside the 99.6 % binomial band [98, 137], `ca_rhs_acceptance_output.json`) ·
GATE RHS-F (scorer reproduces the venue's own L_cat/B_num ≤ 1e-6) · GATE B-R exact · **GATE B-R POWER** ·
GATE W (closure ≤ 1e-6) · k̂ invariant · g-znorm (E-B0(a): the Σ^φ slot enters as exactly 1/r_φ(h),
1.128688 at 0.73, CV ~3e-15 — row #177) · g-population (0 mixed rows; seeds listed) · g-precision
(full-precision columns; `w_G`/`w_tilde_G` NEVER read — PA-CA-1).

Disposition (every row a fresh RULE): TWIN-CALIBRATED / TWIN-MISCALIBRATED / VENUE-MISSPEC / CONTROL-FAIL
per C-A §4+PA-CA-2; **VACUOUS** (power control inside the band) ⇒ park bounded-undetermined with the reason;
**BYTE-ID-ONLY** (N1 with parity ≤ floor) ⇒ nothing new banked beyond the stamp; NOT-EVALUABLE ⇒ inputs named.

## 5. Costs (ORCHESTRATOR-DERIVED where marked)

| item | cost | source |
|---|---|---|
| B-R design gate (tonight) | ≈ 2 min single core, 0 evaluate() | `BR_CONTROL_RECORD.md` |
| N1 fleet | 11.5–21.6 CPU-h (+ 0.5–0.9 precheck) | prereg lines 173–174 anchor 0.478–0.9 CPU-h/seed; cap **25 CPU-h** ORCHESTRATOR-DERIVED (20 + precheck + 15 % margin) |
| RHS re-score at HEAD (only if the precheck shows the no-BH path moved) | 6.9 wall-h realized / 20–90 CPU-h registered fallback | `ca_rhs_score_output.json` elapsed_s; PA-CA-7d |
| N2 | unscoped — returns with its own registration | rows #198–#211 |

## 6. Invariants and structural blindness ([A10])

Invariants: the b0i registries (`ARM_SPECS/ARM_HOST_MODE`) · catalogue md5 `c52c13b5…` and the 0.73 pool ·
H_GRID_FULL · C\* and companions from the fresh leaf build cross-checked against row #177 (0.170472, ρ 0.987771)
· r literal · `PRODUCTION_FLAGS` verbatim with the three survival/slot flags stamped resolved · mass-aware OFF
(N1) — audited 2026-09-03 here. **Structural blindness (binding on any verdict of this family):** (i) the PA-2
aligned-generator premise — the venue realises the twin's own law, so a PASS is self-consistency, not
real-universe correctness; (ii) the S̄_φ-table common mode (five instruments) cancels in T_w; (iii) the RHS
scorer's own code beyond RHS-F/B-R coverage; (iv) one venue, one node (0.73); (v) N1 cannot see the
mass-aware leg at all; (vi) the F-0 acceptance is inside the statistic — a defect in the filter's venue
implementation enters both sides coherently.

## 7. Open questions routed to d-b0fm-band (RECOMMENDATIONs flagged, none binding)

1. Which measure shape: N3 (RECOMMENDED) · N1 as a parity stamp only · N2 as a new [P3-2D]-class registration.
2. If N1: accept the E19 floor (2–5e-5 rel) as the parity criterion, or demand ≤ 1e-12 same-machine byte-id
   (row #325 item 3 semantics: byte-id is same-machine; the banked fleet was 4 local + 8 cluster, PA-16).
3. Whether the POWER control result on the stale basis (`BR_CONTROL_RECORD.md`, DISCRIMINATING at 16.6σ)
   closes the addendum's q-b0-finite-moment design-gate question, so that the claim node
   c-twin-correct-on-catalogued-host's status line is updated from "UNDISCRIMINATING (row #177)" to
   "TWIN-CALIBRATED at the self-consistency level (rows #186/#187), blindness (i) attached" — a wording
   ruling on existing evidence, not a new verdict.
4. Whether the brief's line 82–83 ("the only venue able to adjudicate catalogue-leg correctness") should carry
   the (i) caveat: b0i adjudicates self-consistency of the arrangement, not the law.
5. Cap for N1 (25 CPU-h ORCHESTRATOR-DERIVED) and venue (local vs cluster; row #185 standing: cluster-first).

Cost tonight: 0 evaluate(). Tiering: N1 arrays sonnet/low; any N2 registration top-tier/xhigh (own node).
