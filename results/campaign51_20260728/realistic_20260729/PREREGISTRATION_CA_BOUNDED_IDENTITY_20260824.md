# PRE-REGISTRATION — [P3-IMP] the C-A bounded identity test (the b0 successor; stage 2)

**Date:** 2026-08-24 · **Thread:** `[P3-IMP]` · **Grant:** row #178 item 3 ([DO], author) as
scoped by the banked Gate-B adjudication §4 (`GATE_B_ADJUDICATION_FINITE_MOMENT_20260824.md`,
row #180). **Orchestrator-autonomous session; [ORCH-*] tags bind.** **Append-only after
commit; A21 governs.** Vocabulary per the primer.

## 1. The registered identity and statistic

For the class mixture at h_true, with φ(w) = w and R = (1−w)/w the class-odds ratio, the
bounded-transform identity (adjudicated, row #180):

    C* · E_{d~q_G}[ φ(w)·R ] = E_{d~q_Ḡ}[ φ(w) ]   ⇔   C* · E_G[1−w] = E_Ḡ[w]

— LHS: the venue side, a mean of the [0,1]-bounded (1−w) over catalogue-class draws, already
BANKED (the 24 b0i CSV sets; no new venue runs). RHS: the model side, E of the bounded w under
the mixture's OWN completion-class predictive — computable by a synthetic-event scorer with NO
Ḡ-class venue draws (adjudicated distinct from the refuted reciprocal form). Both sides are
means of variables bounded by max(1, C\*): **variance ≤ C\*²/4n; the A21-B0-C band-vacuity
mechanism cannot recur** (k̂ check retained as an invariant; its firing would itself be a
finding). **Verdict statistic per arm a: T_w(a) = C\*·E_G-banked[1−w]_a − RHS_model(a)**, at
h = H_TRUE = 0.73, **acceptance-conditioned on BOTH sides (F-0 inside the design)**.

**Conventions (adjudication-fixed):** (i) dead rows INCLUDED — the LHS runs over ALL accepted
events (w = 0 rows contribute 1−w = 1; removes the PA-6(a) ambiguity and the unregistered
live-row conditioning; banked all-rows LHS: B-T 0.07279, B-C 0.06435, paired Δ = +0.008447);
(ii) C\* and every model-side moment from the SAME leaf builds as the RHS scorer; (iii) the
acceptance model (the F-0 σ/d̂ < 0.10 ∧ SNR ≥ 20 filter) is computed by the SAME synthetic
scorer and gate-checked closed-loop (§3).

## 2. Arms and instrument

| arm | LHS (banked, zero-compute) | RHS (new instrument) |
|---|---|---|
| **B-T** | C\*·mean(1−w) over the 12 bt CSVs, all accepted rows | RHS_model(twin arrangement) |
| **B-C** | same over the 12 bc CSVs | RHS_model(coded arrangement) — the coded displacement becomes a DERIVED number, not order-of-magnitude |
| **B-R (scorer control)** | the r_h-rescaled transform of the bt frame (cat_scale = R(0.73) = 1.5155) | must land at its derived transform of RHS(twin) within the registered tolerance — fail-at-predicted-value, the row-#177-validated pattern |

**Instrument `ca_rhs_scorer.py` (committed before it runs):** draws synthetic events from the
mixture's completion-class predictive (the generator law the O6/O7-confirmed chain defines),
applies the F-0 acceptance, evaluates w under each arrangement via the SAME leaf functions the
production estimator uses (imported, never reimplemented), and accumulates E_Ḡ[w] to
SE ≤ 5e-4 (~10 CPU-h; LHS-limited combined σ thereafter). **C-B rides along** (~1 CPU-h): the
coded-null pinning pass for the paired log-LR Λ̄ (banked probe −0.02516 ± 0.00454) — the free
corroborator; registered as REPORTED-WITH-VERDICT, not verdict-bearing.

## 3. Gates (fail ⇒ VOID unless A21-amended)

- **GATE ACC (closed-loop acceptance):** the scorer's acceptance model predicts each banked
  seed's n_kept; PASS iff all 12 realized counts (106…131) fall within the model's central
  binomial bands AND the fleet-level predicted P_G is within its stated error of 0.5821. The
  current standalone model's 7% class-G error is the failure to beat; failure ⇒ STOP, amend to
  a filter-aware acceptance model before any verdict.
- **GATE RHS-F (generator fidelity):** the RHS scorer, run on the VENUE'S OWN drawn events,
  reproduces the banked `L_cat_no_bh`/`B_num` columns to the CSV storage floor (≤1e-6 rel).
- **GATE B-R (scorer control):** as tabled; tolerance 0.005 on the T_w scale (registered here,
  pre-data).
- **GATE W (closure):** the W-B0 closure reuse on every consumed CSV (≤1e-6, storage floor).
- **k̂ invariant:** per-arm k̂ of the LHS summand distribution reported; k̂ > 0.7 would be a
  FINDING (structurally unexpected for a bounded summand), not a band modifier.
- **A22 as amended:** stamps written before the scorer runs; both flag values + the F-0
  acceptance-model version in every meta.

## 4. Bands and verdict map (formulas frozen now; σ numbers frozen when the RHS SE lands, pre-verdict)

σ_comb(a)² = SEM_LHS(a)² + SE_RHS(a)². With the banked LHS SEMs (~1.3e-3) and RHS SE ≤ 5e-4,
σ_comb ≈ 1.4e-3; the anchor 0.005 ≈ 3.6σ_comb.

- **TWIN-CALIBRATED:** |T_w(B-T)| ≤ max(3σ_comb, 0.005) AND B-C lands at its DERIVED
  displacement (|T_w(B-C) − D_C| ≤ max(3σ_comb, 0.005), D_C = the pre-frozen derived coded
  displacement) AND the B-R control at predicted value ⇒ the twin's per-candidate S̄_φ
  arrangement is the calibrated one; the production catalogue-leg physics-change proposal
  proceeds (author-gated).
- **TWIN-MISCALIBRATED:** |T_w(B-T)| > the band with B-C at its derived value ⇒ the twin
  carries a genuine residual mis-calibration; magnitude banked; the thread returns to stage 0
  with the candidate demoted.
- **VENUE-MISSPEC:** B-C NOT at its derived displacement ⇒ the F-0/acceptance model is the
  first suspect (GATE ACC notwithstanding); no arm verdict.
- **CONTROL-FAIL:** B-R off its predicted value ⇒ scorer defect; no verdict; A21 STOP.
- C-B reported alongside per its banked closure decomposition; band-free.

## 5. A10 — invariants and structural blindness

**Invariants:** the banked b0i fleet CSVs (sha256-manifested; consumed read-only) · the Σ^φ
production slot (adopted row #178; the banked CSVs are phi-slot by construction) · H_TRUE node
read · C\* and companions from the row-#177-banked mass companion, cross-checked against the
fresh leaf build · R(0.73) = 1.5155487621… (rescore JSON).

**Structural blindness:** (i) the PA-2 aligned-generator premise (unchanged); (ii) the
four-instrument S̄_φ-table common mode (now five); (iii) the RHS scorer's own code — mitigated
by GATE RHS-F and the B-R control, not eliminated; (iv) the acceptance model's residual error
after GATE ACC (budget stated numerically in the verdict); (v) one venue, one h — venue- and
node-conditional throughout.

## 6. Falsifiers (A19)

TWIN-CALIBRATED is falsified by: GATE ACC failure on re-audit; the C-TCI τ-profile robustness
twin (winsorized members of the φ family across τ, exact truncation corrections per the
adjudicated derivation) disagreeing in band; the C-B corroborator's coded-null pass landing
inconsistent with its KL decomposition. TWIN-MISCALIBRATED is falsified by a VENUE-MISSPEC
signature emerging in the τ-profile. Verdicts are [ORCH-banked, provisional] pending the
author's stage-5 ruling.

## 7. Costing (A6/A17) — [ORCH-COST]

LHS: zero-compute (banked). RHS: ~10 CPU-h local (SE 5e-4). C-B pinning: ~1 CPU-h. Gates:
≤1 CPU-h. **Total ≤ ~12 CPU-h, ZERO fresh `evaluate()` fleets.** Cluster optional, not needed.

*(Committed before the instrument exists; pre-execution adversarial review before the scorer
runs; σ freeze pre-verdict; A20 review before banking.)*

---

## PRE-EXECUTION REVIEW AMENDMENTS PA-CA-1…PA-CA-9 (2026-08-24, pre-commit, NO registered instrument has run; review banked verbatim in `A20_REVIEW_CA_DESIGN_20260824.md`, verdict BLOCKED → amended as prescribed)

**PA-CA-1 (Finding 1, FATAL — the conditioned identity).** §1's T_w dropped the class-asymmetry
factor (P̄_G = 0.5821 ≠ P̄_Ḡ = 0.9269): a per-side acceptance-conditional mean breaks the
identity by ×1.592 (a guaranteed ~+0.03 ≈ 19–22σ false TWIN-MISCALIBRATED). REGISTERED FORM
(drawn-count normalization; exact under F-0; the reviewer's replacement text verbatim):
per-seed **LHS_s(a) = (C\*/200)·Σ_{accepted rows}(1−w_e)**, fleet mean over 12 seeds;
**RHS_model(a) = (1/N_syn)·Σ_{ALL synthetic draws} w_a(d_j)·1_acc(d_j)** — normalized by ALL
completion-class draws with the F-0 filter applied exactly per draw (σ_dL < 0.10·d̂; the SNR
clause inactive by donor construction — NO acceptance MODEL enters the verdict statistic).
**Banked LHS under this normalization: B-T 0.04233 ± 0.00108 · B-C 0.03741 ± 0.00095 · paired
Δ = +0.004919 ± 0.000146 (12/12).** w registered operationally: w_e = A_e/(A_e+B_e),
A_e = β_G_φ(0.73)·L_cat_no_bh (β from the per-seed selection JSON), B_e = B_num; **the CSVs'
`w_G`/`w_tilde_G` columns are NOT this w and must not be read.**

**PA-CA-2 (Finding 2 — D_C derived + the coherence slope).** D_C = E_Ḡ[(W̃−1)·w_BC·1_acc]
(W̃ = L_cat^BT/L_cat^BC ≤ 1; model-side, same synthetic set, no circularity; sign < 0 ✓).
Criterion registered in COLLAPSED form: |LHS(B-C) − E_Ḡ[W̃·w_BC·1_acc]| ≤ band. Verdict map
amended per the reviewer's reachability derivation: register the pre-frozen coherence slope
κ̂ = E_Ḡ[W̃w_BC·1_acc]/E_Ḡ[w_BT·1_acc]; **TWIN-MISCALIBRATED ⇔ |T_w(B-T)| > band AND
|(T_w(B-C)−D_C) − κ̂·T_w(B-T)| ≤ max(3σ_comb, 0.005); VENUE-MISSPEC ⇔ the B-C deviation
inconsistent with that coherence relation** (the original both-cells map had an unreachable
TWIN-MISCALIBRATED for violations ≳2× the band).

**PA-CA-3 (Finding 3 — GATE ACC rewritten, reviewer's text verbatim).** The scorer replays the
class-G b0i draw law (kernel-smeared z; per-seed donor context as drawn) with the exact F-0
filter per draw. PASS iff (i) every realized n_kept,s in the central binomial(200, p̄_s) band
at per-seed coverage 99.6% (joint false-STOP ≈ 5%), AND (ii) |P_model − 0.5821| ≤ 2σ_P,
σ_P² = 0.5821·0.4179/2400 + SE_model². Registered overdispersion note: realized seed sd 8.84
vs common-p 6.98 (p ≈ 0.09); clause-(i) failure with (ii) passing = a FINDING on seed
conditioning — STOP, A21-amend. With PA-CA-1, GATE ACC is venue-fidelity only; no acceptance-
model number enters T_w; §5(iv) restated accordingly.

**PA-CA-4 (Finding 4 — GATE B-R exact form).** Banked side (deterministic):
LHS_BR = (C\*/200)·Σ_acc(1−w_e)/(1+(r−1)w_e) over the 12 bt frames = **0.03571 ± 0.00093**
(review-computed; frozen). Model side: RHS_BR = E_Ḡ[w/(1+(r−1)w)·1_acc] on the same synthetic
set. PASS iff |LHS_BR − RHS_BR| ≤ 0.005. C\* un-rescaled both sides (registered; the C\*′ = rC\*
alternative differs ×1.52 — this convention chosen once here). Scalar transforms of RHS(twin)
are NOT the control (Jensen gap up to ~0.008 > tolerance). Synergy: B-R + RHS-F jointly pin
the PA-CA-6 slot hazard.

**PA-CA-5 (Finding 5).** Dead-row inclusion CONFIRMED coherent both sides; all §1 quoted
LHS numbers replaced by the PA-CA-1 drawn-count set.

**PA-CA-6 (Finding 6 — slot hazard).** All 24 banked metas ran `"phi"` explicitly; the current
bare-class fallback is `"s3d"` — an s3d-defaulting scorer skews RHS ~5σ_comb. A22 clause
tightened: all THREE flag values stamped as RESOLVED values (never "auto"); the scorer passes
`catalogue_global_selection="phi"` EXPLICITLY. GATE RHS-F runs on BOTH arms (≥1 full seed per
arm), h = 0.73 rows, BEFORE any RHS accumulation; dead rows reproduce as exact zeros.

**PA-CA-7 (Finding 7).** (a) σ set: σ_comb(B-T) ≈ 1.19e-3; convention: t₁₁ quantiles (3σ →
3.35·SEM). (b) C-TCI registered as the INDICATOR member only: (C\*/200)Σ_acc R_e·1{R_e≤τ} vs
E_Ḡ[1{R≤τ}·1_acc], τ ∈ {30, 100, 300, 1000}; power asymmetry disclosed (per-τ bands 4–18× the
anchor: agreement weak evidence; only >3σ_τ discrepancy falsifies). (c) C-B verdict-map role:
enters §6 as a falsifier ONLY if its pinned coded-null and twin-null separate ≥3σ_null AND the
measured Λ̄ lies ≥3σ closer to the coded-null; else REPORT-ONLY. (d) Costing corrected: the
5e-4 RHS SE needs ~20–90 CPU-h; REGISTERED FALLBACK: SE ≤ 1e-3 acceptable (σ_comb → 1.47e-3,
anchor ≥ 3.4σ, still LHS-limited), N capped, realized SE stated at the σ freeze. (e) The
"sha256-manifested" claim was FALSE for 4/24 CSVs (local seeds 900102/900103 uncovered) — the
complete 24-CSV manifest is generated in the registration commit.

**PA-CA-8 ([ORCH-RULE] — the implementer's open item 1, the donor-marginal convention).** The
identity's E_G side was realized by the venue's own law (200-event realizations,
without-replacement donor draws within a realization); the exact conditioned identity holds
under THAT law. REGISTERED: both GATE ACC and the RHS accumulate over repeated INDEPENDENT
200-draw realizations of the venue's own convention (chunk = 200, the implementer's fixed
scale; the smoke-found 1200-chunk bias — P_G 0.4555 vs 0.5821, 19.8 SE — is the banked
evidence for the rule). i.i.d.-with-replacement resampling is NOT the registered law.

**PA-CA-9 ([ORCH-DO] — cost structure).** The per-chunk full-catalogue draw-weight recompute
(the implementer's open item 2) is removed by CACHING the class-G draw weights across chunks
(deterministic in the pool + tables at fixed h_true; a pure memoization, no math change) —
implemented with the PA-CA fold-in and covered by the determinism test. GATE ACC n_mc and the
RHS N run under the PA-CA-7(d) cap.

*(This block + the banked review + the amended instruments + the 24-CSV manifest commit
together BEFORE the registered instruments run; round-2 focused review on the amendments
precedes the commit per the established pattern.)*
