# Stage-1 information forecast — r-highz-completion (Graph 2 mechanism node, R15)

Date: 2026-09-04. Author D (top-tier prereg author). Verdict-free; every number is a forecast INPUT with
its source. No registered statistic (term-freeze Δ_t, shares, per-term scores, harness pooled reads) has been
computed by anyone. Companion to `REGISTRATION_DRAFT.md` and `MECHANISM_NOTE.md`.

## 1. What a perfect analysis of the banked data can say (rule 9)

The question "which TERM of the zero-candidate likelihood carries the high-z pull" is fully decidable from
`event_likelihoods.csv` (41 h × 1588 events, both venues) because the per-event identity
`ln L = ln B_num + ln g − ln D̃_φ` is exact for zero-candidate events (MECHANISM_NOTE §2, closure 2.7e-15) and
the denominator is event-common. The harness replicate (67 universes × 41 h, same columns) decides whether
the same term structure and magnitude appear in a self-consistent universe. The ≤ 2 CPU-h cap is a ceiling on
nothing (≈ 0.05 CPU-h).
What the data CANNOT say: which FACTOR inside `B_num` — GW distance × volume (legitimate) vs completeness
(1−f_k) × survival S̄_φ (estimator-internal candidates) — carries a T_B-owned pull; that split needs the
integrand re-evaluated (conditional follow-up `b-highz-bnum-factor`, draft §9). It also cannot separate an
estimator-internal cause from a production-venue mismatch (real catalogue vs the estimator's completeness
model) when the harness is NULL — that distinction is the follow-up's, not this node's.

## 2. Forecast inputs

| input | value | source | tag |
|---|---|---|---|
| K (top-z decile, 159) leave-out | Δmean_h = +0.086106; null CI99 [−0.009089, +0.010751] (1000 draws, seed 20260904) | END_VERIFICATION BATCH 2; row #363 | [DOC] |
| oracle: leave out S (k = 82) | +0.046234 | same | [DOC] |
| leave out the whole exact-zero dark class (606) | +0.155678 (0.666 → 0.822, past truth); leave out the 982 hosted: −0.0655 | same | [DOC] |
| S composition | 67/82 zero-candidate (82 % vs bulk 36 %); median z 0.849 vs 0.481; C11 SNR AUC 0.23 | rows #362/#363 | [DOC] |
| dark-class completion-leg matched-channel score | T_prod = −0.19664 ± 0.01944 per dark event (Z −10.1); T_harn = −0.0505 ± 0.0073; ρ = 0.257 → 74 % production-only | row #347 | [DOC] |
| harness class localisation | S3 cell S: catalogue-hosted (truth label) Z 9.76/7.15 OUTSIDE; dark 1.26/1.76 INSIDE; rail fraction 14.9 %/20.9 % at h = 0.86 | row #335 | [DOC], harness venue |
| pre-flip z-resolution of the dark-class score | ≈ 0 below z ≈ 0.4, −1.08 by z ≈ 0.9 (score −0.635, 37σ) | row #137 / RETROSPECTIVE_D1 | [DOC], STALE ([A11], pre-flip 1D leg) — forecast input only |
| g_frac h-slope | carrier of the 2D residual in the 08-04 frozen-g evaluate (row #91); "correct physics" by AUTHOR RULING R-A (ledger §5, 2026-08-05); closed-loop Δ2 = +0.011 ± 0.004 (row #92) | ledger §2 rows 91/92, §5 | [DOC] |
| cone loss | immaterial (φ 0.4 %); zero-candidate events have no host to lose | row #344 | [DOC] |
| population counts (iiib) | P_dark = 606 (= C2 False = C3c censored = C7 == 0); K = 159; K_dark = 144; K_hosted = 15; R (low-z half of P_dark \ K) = 231, z ≤ 0.541; K z ≥ 0.736; P_dark median z 0.62 | `covariate_table_iiib.csv` | [LOCAL] counts |
| population counts (joint_r1) | P_dark = 493; K = 159 (same set as iiib — same CRB); K_dark = 111; K_hosted = 48; R = 191, z ≤ 0.483 | `covariate_table_joint_r1.csv` | [LOCAL] counts |
| harness populations (67 n200 universes, cell S) | scored 173–192 (Σ 12,060); zero-candidate 53–85 (Σ 4,826, all-h zero ≡ h=0.73 zero, 67/67); top-z decile 17–19 (Σ 1,207) of which zero-candidate 14–19 (Σ 1,148); median z 0.45 (all) / 0.58 (zero-candidate) | harness CSVs | [LOCAL] counts |
| information scale | I_HEAD(iiib) = 2965 nats/h² (σ_h 0.018366); harness N = 200 → per-event Δ ≈ 8× larger for the same score | row #302 / MEASUREMENT_HEAD_READOUT §C.1 | [DOC] |

## 3. Expected outcome and why (author D's reading — not a measurement)

**Production term shares.** T_B (ln B_num) is expected to OWN the freeze total (share ≳ 0.7): the GW distance
likelihood, the completeness weight and the survival table all live in B_num, and the 1D channel (no g term)
carries an offset of the same size as 2D (0.6670 vs 0.6659), so the z-differential pull cannot be mostly a
2D-only object. T_g's share is expected small but non-zero and possibly sign-opposed (the frozen-g evaluate
moved 2D MAP by −0.12…−0.16 in the pre-fusion era, but post-fusion 1D≈2D). T_D ≡ 0 by identity.
**Kill criterion reachability.** With two separable terms whose shares sum to 1 − r (r = non-additivity of
mean_h), "every term < 0.2" can fire only if |r| ≥ 0.6, i.e. when the freeze is not decomposable at the term
level at all. Expected |r| ≲ 0.1 (per-event changes of O(0.5 nats) against a 3000-nats/h² posterior;
row #344's 18× non-linearity was a heavy-tailed 10-event leave-out, a different regime). Forecast: the kill
criterion does NOT fire; the node returns TERM-OWNS(T_B) on production.
**Harness control (the decisive read).** Under calibration, E[∂_h ln L_e | z] = 0 at truth for every z, so the
self-consistent universe predicts a NULL z-differential score (S_F^harn ≈ 0, |Z| ≤ 3). Row #335 (dark class
INSIDE) and row #347 (ρ = 0.26) both point that way. Forecast: HARNESS-NULL or ρ_S ≤ 0.2 → outcome
PRODUCTION-ONLY (draft §5 row 2) — the pull is not what the estimator does on its own universe. If instead the
harness resolves the same T_B-owned differential at ρ_S ≥ 0.5, the estimator produces the high-z pull by
itself (ESTIMATOR-INTERNAL candidate) and `b-highz-bnum-factor` becomes mandatory.
**Direction.** All 82 S events pull DOWN (row #362); K leave-out is positive → the freeze Δ_F for K_dark is
expected positive (high-z zero-candidate profiles tilt lower than the low-z reference). A NEGATIVE Δ_F would mean
the high-z decile pulls LESS per event than the low-z dark bulk and the +0.086 is a class effect, not a z
effect — a live, un-forecast branch (draft §5 "Z-DIFFERENTIAL-NULL").

## 4. Power (first-order, from banked scales only)

Freeze of 144 events: Δ_F ≈ 144·S_F/I_HEAD. To resolve Δ_F at the null CI99 half-width (≈ 0.010 for 159-event
draws) needs |S_F| ≳ 0.2 nats/h per event — the same size as T_prod (−0.197). The production read is expected
POWERED. Harness: S_F^harn pooled over 1,148 events with a between-universe SE; if the per-event score spread is
comparable to production's (SE_prod 0.019 at n = 1512 → per-event SD ≈ 0.76 nats/h), SE_harn ≈ 0.76/√1148 ≈
0.022 nats/h → a harness S_F of |0.07| is resolvable at 3σ; ρ_S ≥ 0.5 vs ≤ 0.2 is discriminable if
|S_F^prod| ≳ 0.2. Rail caveat: 10–14 harness universes sit at the upper rail (row #335); their mean_h-based Δ
is a bound, but the score-based S_F (three-node stencil at 0.725/0.730/0.735) is rail-free — hence the transfer
statistic is S_F, not Δ_F (draft §2.3).
