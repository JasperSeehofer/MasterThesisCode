# r-b0-finite-moment — B-R CONTROL RECORD (design gate, stale basis, zero compute, VERDICT-FREE)

Date: 2026-09-03 (night). Node: r-b0-finite-moment design gate (Research Graph 1 ADDENDUM, Branch N [B0FM]).
Author of record for all scientific decisions: Jasper Seehofer. Run by: the addendum prereg author (top-tier).
**Scope statement (binding on every quotation of this record):** the 24 banked b0i CSVs (commit `3bd6b564`,
2026-08-23) PRE-DATE [P3-WBHZERO] `cf4f8a2a`, the twin flip `bac48696`, A18 (row #286) and A14 (row #284).
This is a **statistic-power test on a stale basis**. It can say whether the registered band is vacuous for
the refuted arrangement; it can NEVER yield a twin/coded identity verdict — none is stated, implied, or
recommended here. B-T/B-C were NOT blind on this basis (already read at rows #181/#186); disclosed.

## 1. Inputs (three-valued existence contract)

| input | state | path |
|---|---|---|
| 12 × B-T + 12 × B-C banked CSVs (seeds 900101–900112) | present | `p3_b0_work/{bt,bc}_<seed>_work/seed<seed>/simulations/diagnostics/event_likelihoods.csv` (46 h-nodes each; 2 794 rows at h = 0.73 = 2 × 1 397, the F-0 intake count) |
| per-seed selection tables (β_G_φ, β̄_Ḡ_φ at 0.73) | present, identical across seeds (β_G_φ = 153 322 758.616) | `…/selection_tables_h_0_73.json` |
| mass companion (ρ, Σ's) | present | `p3_b0_identity_test_output.json` → `mass_companion_at_h_gen` |
| C-A synthetic RHS chunks 5–224 (twin + coded) | present, clean (≤ 200 rows/chunk, no duplicate event_idx) | `ca_rhs_work/score_chunk<k>_{twin,coded}_work/simulations/diagnostics/event_likelihoods.csv` |
| C-A chunks 0–4 | present but CONTAMINATED (368–376 rows; A20_REVIEW_CA_VERDICT F3); the fresh re-score dirs are **absent locally** | excluded here; the banked full-225 numbers are quoted for comparison |
| registered literal r = R(0.73) | present | `ca_rhs_scorer.py:285` = 1.515548762178686 |
| banked comparands | present | `ca_rhs_work/ca_verdict_banked_full225.json`; `PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md` PA-CA-1/PA-CA-4/PA-CA-11 |

Instrument: `br_control.py` (this directory; stdlib + numpy/pandas; reads only; writes `br_control_output.json`).
Wall ≈ 2 min single-core. No `evaluate()` call, no cluster, no source edit.

## 2. Definitions (exactly the C-A registered forms; nothing new except the POWER control)

w_e = A_e/(A_e + B_e), A_e = β_G_φ(0.73)·L_cat_no_bh, B_e = B_num, rows at h = 0.73, dead rows included
(w = 0 ⇒ 1 − w = 1) — PA-CA-1. C\* = β_G_φ·ρ/β̄_Ḡ_φ. Drawn-count normalisation 200 per seed / per chunk.

- LHS_s(a) = (C\*/200)·Σ_acc(1 − w_e); RHS(a) = (1/N_syn)·Σ_acc w_a — PA-CA-1.
- **B-R exact form (C-A GATE B-R, scorer validation):** LHS_BR = (C\*/200)·Σ(1 − w)/(1 + (r−1)w) on the
  B-T frames; RHS_BR = E_Ḡ[w/(1 + (r−1)w)·1_acc] — PA-CA-4. Identity-exact for ANY r ⇒ tests the scorer.
- **B-R POWER form (this node's design gate, the addendum's "B-R must FAIL the band"):** the refuted
  arrangement's own responsibility w′ = rA/(rA + B) plugged into BOTH sides as an analyst who believed it
  would: T_w^naive(B-R) = C\*·mean(1 − w′) − E_Ḡ[w′·1_acc] = LHS_BR − r·RHS_BR. Under the true identity its
  value is RHS_BR·(1 − r) ≠ 0 (derivation: C\*·E_G[1−w′] = E_Ḡ[w′]/r exactly). This is the same construction
  as C-A's banked "naive coded displacement". Pre-stated gate criterion (written before the numbers were
  read): DISCRIMINATING iff |T_w^naive(B-R)| > 0.005 AND |Z| > 3; VACUOUS iff |T_w^naive(B-R)| ≤ 0.005.
- C-TCI indicator member per τ ∈ {30, 100, 300, 1000} (PA-CA-7b): LHS = (C\*/200)·Σ R_e·1{R_e ≤ τ},
  RHS = E_Ḡ[1{R ≤ τ}·1_acc], R = B/A; power analog with R′ = R/r on both sides.
- C-B Λ̄ (paired live rows, + ln(Σ_w/Σ̃^φ) = +0.254207): the B-R analog is a deterministic shift +ln r.

## 3. Reproduction of the banked C-A numbers (rule 2: reproduce before use) [LOCAL]

| quantity | this record | banked | source of banked |
|---|---|---|---|
| C\* | 0.1704717 | 0.170472 | row #177 mass companion |
| LHS(B-T) | 0.042330 ± 0.001082 | 0.04233 ± 0.00108 | PA-CA-1 |
| LHS(B-C) | 0.037411 ± 0.000951 | 0.03741 ± 0.00095 | PA-CA-1 |
| paired Δ(B-T − B-C) | +0.004919 ± 0.000146 (12/12) | +0.004919 ± 0.000146 | PA-CA-1 |
| LHS_BR | 0.035707 ± 0.000931 | 0.03571 ± 0.00093 | PA-CA-4 |
| RHS_w(twin) [220 clean chunks] | 0.043650 ± 0.000579 | 0.0436235 ± 0.0005711 [225] | PA-CA-11 |
| RHS_w(coded) | 0.057593 ± 0.000669 | 0.0575733 ± 0.0006607 | PA-CA-11 |
| RHS_BR | 0.036217 ± 0.000450 | 0.0362018 ± 0.0004442 | PA-CA-11 |
| D_C comparand E_Ḡ[W̃·w_BC·1_acc] | 0.038356 ± 0.000504 | 0.0383376 ± 0.0004967 | PA-CA-11 |
| Λ̄ (B-T, 12 seeds) | −0.025157 ± 0.004543 | −0.02516 ± 0.00454 | row #180 |
| accepted synthetic events (220 chunks) | 40 375 | 41 307 (225) | PA-CA-11 |

All LHS quantities agree to the printed digit; the 220-chunk RHS agree with the 225-chunk banked values
within 0.05 SE. The banked basis is intact.

## 4. Design-gate results (three-valued against the §2 criterion; NO identity verdict)

| statistic | value | σ_comb | Z | vs band 0.005 |
|---|---|---|---|---|
| G-BR-exact: LHS_BR − RHS_BR (scorer gate) | −0.000510 | 0.001034 | −0.49 | inside (as banked: −0.000495, −0.48σ) |
| **G-BR-POWER: LHS_BR − r·RHS_BR** | **−0.019182** | 0.001154 | **−16.6** | **outside, 3.8 × band**; identity prediction −0.018672 (agrees within 0.44σ) |
| reference: coded naive LHS(B-C) − RHS(coded) | −0.020182 | 0.001162 | −17.4 | outside (banked −0.020162) |
| reference: T_w(B-T) | −0.001320 | 0.001227 | −1.08 | inside (banked −0.001294) — NOT a verdict here |
| reference: LHS(B-C) − D_C | −0.000945 | 0.001076 | −0.88 | inside (banked −0.000927) |

C-TCI power analog, B-R naive vs B-T (T, Z): τ=30 −0.0616 (−8.6) vs −0.0058 (−0.66); τ=100 −0.0734 (−5.6)
vs +0.0212 (+1.14); τ=300 −0.0848 (−3.3) vs +0.0113 (+0.46); τ=1000 −0.0996 (−3.1) vs +0.0386 (+0.79).
C-B: Λ̄(B-R) = Λ̄(B-T) + ln r = −0.02516 + 0.41578 = +0.3906 (SEM 0.0045) vs the twin-law null ≈ −0.022 ± 0.010
(row #180): excluded by > 40σ.

**Gate outcome (pre-stated criterion): DISCRIMINATING.** The refuted arrangement fails the registered
0.005 band at |Z| = 16.6 in the primary (φ = w) member, at |Z| ≥ 3.1 in every C-TCI member, and by > 40σ in
Λ. The A21-B0-C vacuity mechanism (one tail event inflating the SEM to self-pass) cannot operate: the
per-seed LHS_BR spread is 0.0307–0.0417 (SD 0.0032; seed 900108 = 0.0408, mid-pack), k̂ of the bounded
summand sub-threshold (banked −0.42, PA-CA-11).

## 5. What this record does and does not say

- Says: the finite-moment band of the C-A family has power against the refuted arrangement on this venue;
  the proposal's "B-R passes again → statistic vacuous → park" branch is NOT the branch on the stale basis.
- Does not say: anything about the twin's or the coded leg's correctness on b0 (the T_w(B-T)/B-C rows are
  quoted only as reproductions of row #186's banked arithmetic, which carries its own ratified scope and
  blindness clauses: rows #186/#187).
- Does not say: anything about HEAD. The basis is `3bd6b564`; see REGISTRATION_DRAFT.md §3 for why fresh
  HEAD pairs with the production default are not a like-for-like rerun.

RECOMMENDATION (flagged, not a ruling): treat this gate as GREEN for the design and route the launch
question (which HEAD arm, if any) to d-b0fm-band per the registration draft §7.
