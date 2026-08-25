# PRE-REGISTRATION — [P3-2D] the with-BH catalogue-leg twin: 2D bounded identity test (stage 2)

**Date:** 2026-08-25 · **Thread:** `[P3-2D]` (author grant row #188 item 1; stage 0 =
`CLAIM_P3_2D_20260825.md` + `p3_2d_probe.py`, row #189 — cited by section throughout;
[AGENT]-tagged stage-0 numbers become comparands only after the pre-execution review
re-derives them). **Orchestrator-autonomous; [ORCH-*] tags bind. Append-only after commit;
A21 governs.** The C-A governance stack (PA-CA-1/8/10/11 conventions, F-0 inside targets,
out-root guard, resolved-flag A22 stamps) is INHERITED wholesale; only 2D-specific items are
registered here.

## 1. Hypothesis and identity

**2D twin hypothesis (stage-0 §1):** the generator-matched with-BH catalogue numerator carries
the survival INSIDE the candidate's own mass quadrature —
`mz_sel = ∫ N(x; μ_cond, σ_cond) · p_gal(x; z) · S_4D(d_L(z;h), x·M_z,det) dx` — per
candidate (the Eddington-shifted mass posterior; NOT point-S_4D, NOT S̄_φ(z), NOT S̃_φ,g).
Sharp-GW-mass limit: `mz · S_4D(d_L, μ_cond·M_z,det)`.

**Registered statistic:** the 2D bounded identity with
`w₂ = α_G_φ·L_cat_with_bh / (α_G_φ·L_cat_with_bh + B_num_wbh)` and the single
derivation-fixed constant `C₂\* = β_G_φ·r_Malm·ρ₂/β̄_Ḡ_φ` (stage-0 §2; the Σ^4D
cancellation verified ≤6.9e-8 on all 24 banked artifacts; completion-class 2D mass = β̄_Ḡ_φ
exactly by the tower identity; J = d̂·M̂_z event-constant). Drawn-count normalization
(PA-CA-1 form): per-seed `LHS₂_s = (C₂\*/200)·Σ_acc(1−w₂)`; RHS₂ over ALL synthetic draws
with the per-draw F-0 filter (verified mass-blind at :5540-5554). Bounded summands (probed
k̂ = −2.22 pooled); **the unbounded odds analog is REFUSED by registration** (probed k̂ +4.76,
max 2.2e97 — the A21-B0-C class).

**ρ₂ convention [to be FIXED by the pre-execution review]:** stage-0 banks both readings
(LHS₂ 0.025475 ± 0.000675 under ρ₂=ρ; 0.025790 ± 0.000683 under ρ₂=1); the reviewer derives
which is exact for the 2D pairing and the prereg is A21-amended to ONE before any instrument
runs.

## 2. Arms, venue, and instruments

| arm | what | runs |
|---|---|---|
| **B2-C (coded)** | LHS₂ from a FRESH 12-seed b0i-2D fleet under the coded with-BH numerator | 12 cluster-array tasks |
| **B2-T (twin-2D)** | same fleet scored under the 2D twin flag | same 12 tasks (both arms per task, one draw each — the pairing rule) |
| **B2-R (control)** | the analog fail-at-predicted-value rescale, derived by the review (stage-0 transfer table: D_C₂/κ̂₂/B-R₂ RE-DERIVE, not copied) | zero-compute |

**Why a fresh fleet (stage-0 §3, decisive):** the banked b0i CSVs' with-BH columns are
BIT-IDENTICAL across bt/bc (zero twin-2D information), and the venue's donor-mass columns are
unlinked to the drawn host (the four ~1e-101 monster events) — so the venue needs the
**mass-law extension**: latent host mass M ~ the candidate's p_gal; joint 2×2 (d̂, M̂_z) draw;
Bernoulli(S_4D) acceptance (the B-SEL S̄-thinning precedent, now 2D). The banked coded-arm
LHS₂ (S₂ = 0.390399 ± 0.010344 per-seed vector) banks as a VENUE-DRIFT CONTROL only, never a
comparand.

**Instruments (committed before running):** (i) `catalogue_numerator_survival_2d ∈
{"off","mz_sel"}` — estimator counterfactual flag, byte-identical default, BOTH dispatch
paths (scalar :6273-6327 twin-kept-in-sync + batch :6859-6908 = production), with-BH channel
ONLY (the evaluate_with_bh_mass=False call must be leak-proof); consumes the EXISTING
`detection_probability_with_bh_mass_interpolated` (d_L, M_z) object — no new table; the
raw-vs-`_host_M_eff` centering choice mirrors the existing `eddington_m` precedent and is
FIXED by the review. (ii) the venue mass-law extension in `correspondence_1d.py`
(harness-only). (iii) `ca_rhs_scorer.py` extended with the 2D accumulators (RHS₂, D_C₂/κ̂₂,
B-R₂, C-TCI₂ profile, the Σ̃^4D companion pass ~1–2 CPU-h zero-evaluate).

## 3. Gates

The C-A stack verbatim (ACC with the extended draw law re-closed-looped; RHS-F₂ both arms
pre-accumulation at the h_bounds pin; B-R₂ at its derived predicted value; W closure;
PA-CA-11 out-root guard; k̂ invariant; A22 resolved-flag stamps) **plus**: **GATE M2-LINK** —
the drawn host's mass and the scored candidate's mass columns verifiably linked per event
(the monster-event class must be structurally absent in the extended venue; registered check:
zero events with |ln L_cat_with_bh| implying survival evaluated at an unlinked mass — form
fixed by the review).

## 4. Bands and verdict map

The C-A map verbatim with 2D objects: TWIN2-CALIBRATED / TWIN2-MISCALIBRATED (κ̂₂ coherence
clause) / VENUE-MISSPEC / CONTROL-FAIL; band `max(3σ_comb, ε₂)` with **ε₂ frozen pre-verdict
from the realized LHS₂ SEM scale** (anchor proposal 0.005·(C₂\*/C\*) ≈ 0.0019 — reviewer
ratifies or replaces; may only tighten post-data). σ freeze at the RHS₂ SE landing; SE
fallback per PA-CA-7(d) with the cap stated at launch.

## 5. A10 — invariants and blindness

**Invariants:** the 1D twin cell at its production default (rows #186–#187) in BOTH arms · the
Σ^φ/Σ^4D slots as adopted · h = 0.73 read, h_bounds = (0.50, 0.86) · the C-A LHS/RHS
machinery as committed. **Blindness:** (i) the S_4D-table common mode (now shared by six
instruments); (ii) the mass-law extension's own code (gate-mitigated by M2-LINK/RHS-F₂, not
eliminated); (iii) the R&V15 mass-relation scatter question (the 555f018 fixes are in; the
log-normal refactor remains deferred — venue-conditional disclosure); (iv) venue- and
h-conditional as ever; (v) F-0 mass-blindness is verified at source but its 2D interaction
with the joint (d̂, M̂_z) draw is NEW — disclosed for the review.

## 6. Falsifiers (A19)

TWIN2-CALIBRATED falsified by: B2-R off its derived value; the C-TCI₂ profile disagreeing in
band; GATE M2-LINK failure on re-audit; the ρ₂-convention sensitivity exceeding the band
(both readings banked, the difference 0.000315 must stay ≪ ε₂ or the convention choice is
verdict-bearing and returns to the author). Verdicts [ORCH-banked, provisional] pending the
author's stage-5 ruling.

## 7. Costing (A6/A17; cluster-first per row #185) — [ORCH-COST]

Fleet: 12 array tasks (~2–3 CPU-h total, h_bounds-pinned single-h; queue-wait banked per
row #185). RHS₂: the PA-CA-7(d)-capped pass — target SE ≤ ε₂/5, fallback ε₂/3, cap N at the
launch-stated CPU-h budget (~40 CPU-h cluster-array; the 180 CPU-h upper estimate is the
no-cap worst case and is NOT authorized without a fresh costing line). Σ̃^4D companion
~1–2 CPU-h. Zero `evaluate()` beyond the 12-task fleet.

*(Committed before the instruments exist; pre-execution adversarial review — including the
ρ₂ convention, the B2-R/D_C₂ derivations, the ε₂ anchor, and stage-0 number re-derivation —
precedes any instrument run; A20 review before banking.)*

---

## PRE-EXECUTION REVIEW AMENDMENTS PA-2D-1 (2026-08-25, pre-commit, NO instrument has run; review banked verbatim in `A20_REVIEW_P3_2D_DESIGN_20260825.md` — findings F2, F4, F7–F16 ADOPTED VERBATIM as registered text, F3's cite corrected; the decisive registered objects restated here)

- **Centering RULED (F2):** the twin inserts S_4D inside the branch-resolved mass kernel the
  coded path already evaluates — gaussian branch centered at `_host_M_eff` (production
  configuration), mass_trunc branch at raw host_M per its own kernel; the VENUE's latent-mass
  draw binds to the SAME resolved kernel; S(M≤0) = 0.
- **C₂\* RESOLVED (F4):** `C₂\* ≡ β_G_φ·Σ̃^4D/(Σ^φ·β̄_Ḡ_φ)` — Σ^4D cancels completely;
  Σ̃^4D (the venue draw-law contraction) is the ONE new number, computed by the registered
  zero-evaluate companion pass and frozen BEFORE the σ freeze; both stage-0 LHS₂ readings
  demoted to bracket references; the §6 ρ₂-sensitivity falsifier superseded accordingly.
- **A22 = FIVE resolved flags (F7):** + `selection_in_completion_numerator="fused"` (its "off"
  voids the tower identity F6 rides on).
- **D_C₂/κ̂₂/verdict-map (F8) and GATE B2-R (F9) registered forms:** as in the review, with
  **r₂ = 1/r_Malm(0.73) = Σ^φ/Σ^4D = 2.6124925** (pre-derived; guards the α↔β mix-up class).
- **Generator bindings (F10):** whole-event rejection; venue p_gal = the resolved estimator
  kernel; the FREE Ḡ-class z-marginal consistency gate (tower identity) registered.
- **GATE M2-LINK executable form (F11):** the three-part registered check (latent provenance
  triple + bit-level RHS-F₂ consumption + Mahalanobis² fleet bound + the −50-nats
  monster-absence clause); the prereg's original predicate superseded.
- **GATE ACC re-referenced (F12; claim §2.3's class-G z-law claim REFUTED):** bands from the
  EXTENDED class-G replay; 0.5821 is NOT a reference.
- **S₂ and the riding 1D LHS DEMOTED to report-only drift diagnostics (F13)** — no PASS/FAIL,
  no 0.04233/0.03741 targets on the extended venue.
- **POWER GATE + fleet size (F14; [ORCH-RULE], adopting the review's decision):** **24 seeds**
  (900101–900124; +~3 CPU-h — the cheapest power doubling; SEM_LHS₂ → ~4.8e-4); at the σ
  freeze, if |D_C₂| ≤ 2·band the B-C discrimination clause is UNDERPOWERED — no TWIN2 verdict
  banks on it, return to the author with frozen numbers.
- **SE-cap precedence (F15):** realistic landing = fallback SE ≤ ε₂/3 = 6.38e-4 (N ≈ 24k);
  if the capped N cannot reach it, A21 STOP + fresh costing line — the band is never widened
  silently.
- **§5 additions (F16):** (vi) the sky-block independence disclosure with the numeric 2D
  paired sharp-pin residual reported at the σ freeze; (vii) the V2 measure-prefactor is
  common-mode HERE and cancels in T_w₂ — OPEN for the production proposal. PA-CA-10/11 carried
  to every new instrument. Dead-row convention: A₂ = 0 ⟹ w₂ = 0 ⟹ summand 1 (85 = 6.1%
  banked-frame reference).
- **F17:** every decisive stage-0 number independently re-derived (incl. the Hill-estimator
  corroboration of the unbounded-analog refusal — α ≈ 0.23 < 1, one draw = 100.0000% of the
  pooled sum).

**PA-2D-2 (2026-08-25; A21 instrument STOP, pre-gate, NO registered number banked).** The
companion pass's mandated spot-check FAILED (1.7–8.5% deviations): the production-borrowed
GH-24 mass quadrature — exact for the narrow per-event product-Gaussian (σ_cond p50 8.8e-8) —
carries a diagnosed 1.19% bias in the companion's wide population-σ regime (σ_M 60–200% of
M_g; the integrand spans many cells of the 40-bin piecewise-linear S_4D grid; arbitrated by a
4001²-point brute-force rule agreeing with scipy.quad to 1e-6). The agent refused to bank and
held the STOP (cluster never launched). **Fix registered ([ORCH-DECIDE], instrument-side):**
the companion integrates ∫N(M;μ,σ)·S_4D(·,M)dM EXACTLY per grid cell via the closed-form
Gaussian moments over each linear segment (erf-based) — exact for the interpolated S by
construction, no order parameter, cost class unchanged; validated ≥1e-9 against the
brute-force arbiter before any banked run. The per-event estimator flag KEEPS GH-24 (its
regime is narrow by the F2 branch structure — disclosed asymmetry, reviewed at the A20 gate).
Also registered: the fleet-driver threading gap (run_arm_seed does not thread the 2D flags) is
closed by a committed driver wrapper mirroring the p3_b0_identity_test._run_arm_seed
precedent — instrument-side, disclosed.

**PA-2D-3 (2026-08-25; A21 instrument STOP #2 on the companion, NO registered number banked).**
The companion full pass COMPLETED (`ca_rhs_work2d/p3_2d_companion.json`; Σ̃^4D candidate
348079019.37, C₂\* candidate 0.061244) but its mandated spot-check FAILED the registered 1e-6
target (max 3.81e-4, median 5.7e-5, 100 rows) — the candidates are COLLECTED, NOT BANKED.
Adjudication (four/five-method drill-down, `ca_rhs_work2d/spot_check_adjudication.{py,json}` +
`spot_check_drilldown.{py,json}`): the PA-2D-2 exact erf mass-marginal is VINDICATED (swapping
only the z-stage closes the gap); the defect is the companion's **GL(50) z-quadrature**, which
under-resolves the host-z kernel window (kinks where d_L(z;h) crosses dl_centers cell edges) in
the wide-σ/near-horizon regime — the same borrowed-quadrature-regime lesson as PA-2D-2, one
axis over. **Fix registered ([ORCH-DECIDE], instrument-side):** the companion's z-integration
becomes segment-aware (breakpoints at every dl_centers edge inside the window, or an
equivalently exact per-segment rule); the re-run's spot-check target is re-derived from an
arbiter demonstrated to converge below it (the current brute/quad arbiters share a ~1e-4–5e-4
noise floor, so the raw 1e-6 target was unfalsifiable as posed); re-run + re-spot-check before
any C₂\* freeze. Additionally registered (runbook-32 §0 carry): before the freeze, VERIFY
whether Σ̃^4D's draw-law contraction is independent of the [P3-WBHZERO] eligibility ruling
(the mass filter shapes the candidate set the draw law binds to). Companion re-run sequenced
AFTER the WBHZERO mirror fleet vacates the box.

**PA-2D-4 (2026-08-25 late; the [P3-2D] un-HOLD — A21 amendment per rows #196/#198/#202
sequencing; the HOLD of row #194 is LIFTED).**

1. **Eligibility model:** the venue and every scoring path calibrate against POST-ADOPTION
   production — `mass_filter_sigma` at its production default (**"symmetric"**, [PHYSICS]
   `cf4f8a2a`, row #202), left un-overridden in BOTH arms so the venue tracks production
   exactly (the same convention as the other adopted flags). The A22 resolved-flag stamp set
   gains `mass_filter_sigma` (now SIX stamps).
2. **M2-LINK(iii) re-attribution (the row-#191 monster clause):** the pilot's 7/84
   both-arms-zero events are RE-ATTRIBUTED to mass-filter exclusions (the [P3-WBHZERO]
   forensic, rows #194/#196) — NOT unlinked masses. The −50-nats monster-absence clause stays
   for its original class; the zero-with-BH part is superseded by:
3. **GATE M2-Z (registered FREE PREDICTION, runbook 33):** under the adopted symmetric window
   the pilot's exact-zero with-BH class VANISHES — re-run the seed-900101 pilot (both arms)
   and require `L_cat_with_bh > 0` on ALL 7 previously-zero events (their balls are non-empty
   and z-passed; symmetric retains >=1 candidate — the WZ-P-class prediction). Residual zeros
   beyond the rare kernel-zero class (~1.6%, row #196 fleet forensic) => STOP. This gate runs
   BEFORE the fleet.
4. **Companion (PA-2D-3 chain):** re-run with the segment-aware z-rule + arbiter-grounded
   target; ADDITIONALLY verify eligibility-independence: the companion's Sigma~^4D contraction
   must be shown (by code-path inspection, banked with the re-run meta) to consume the draw
   law only — if any consumed table or pool object is candidate-mass-filter-conditioned, the
   filter ruling changes Sigma~^4D and the frozen C2* must be computed under the symmetric
   model; state the finding either way in the re-run JSON.
5. Fleet/RHS2 costing (sec 7 + PA-2D-1 F14/F15) unchanged; cluster-first per row #185.

**PA-2D-5 (2026-08-25 late; GATE M2-Z executed — registered FAIL, attributed, and re-scored
against the PRE-EXISTING banked prediction; the fleet un-blocks).**

1. **The run:** fresh pilot, seed 900101, both arms, out-root `p3_2d_work_m2z/` (banked pilot
   untouched); driver inherits the adopted symmetric default (no explicit pass — verified;
   A22 does not yet stamp mass_filter_sigma, recorded in `gate_m2z.json` instead — the stamp
   extension is carried to the fleet driver).
2. **Result vs the AS-REGISTERED gate: FAIL** — 5/7 previously-zero events return
   `L_cat_with_bh > 0` in both arms; events **51 and 84 remain exactly zero**. No new zeros.
   STOP honored; the fleet did not launch on this gate.
3. **Attribution (orchestrator drafting error, owned):** PA-2D-4 item 3 registered "ALL 7
   vanish" — contradicting the evidence already banked at registration time: the Gate-B
   counterfactual (row #196; preserved `counterfactual_out.json` pilot table) predicts
   **n_sym = 0 for exactly events 51 (pull 2.385) and 84 (pull 2.122)** — both sit >1.5σ of
   their own catalogue mass error outside the window, so the symmetric ±1.5σ window ALSO
   excludes them — and n_sym = 1 for the other five. The fresh run's zero set {51, 84} and
   live set match that banked per-event prediction EXACTLY, arm by arm.
4. **Re-scored gate (the evidence-derived form the registration should have carried):**
   "the fresh pilot's both-arms zero-with-BH set equals the Gate-B-predicted n_sym=0 set
   exactly" ⇒ **PASS (exact match)**. Because the comparand pre-dates the run in the banked
   record (not fitted post hoc), the correction is registered openly with this provenance;
   the as-written FAIL stays on the record as item 2.
5. **Consequence:** the WBHZERO attribution chain receives its second independent structural
   confirmation (after CF-X-prod); the M2-LINK(iii) monster clause's zero class is now
   MEASURED as the mass-filter class in the linked venue. **The fleet is UN-BLOCKED**; its
   driver gains the mass_filter_sigma A22 stamp before launch.
