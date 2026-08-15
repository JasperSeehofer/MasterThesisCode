# STAGE-2 READOUT — A-M2′ term ablation + A-NULL specificity control

**Date:** 2026-08-15 · **Prereg:** `PREREGISTRATION_M2PRIME_ABLATION.md` (registered `092b121b`) ·
**Scorer:** `score_m2prime_stage2.py` (frozen pre-data, `191b0db7`) · **Data:** `AM2P_…seeds0_25.json`,
`ANULL_…seeds0_15.json` (job 6315313, both COMPLETED 0:0; AM2P 1:21:25, ANULL 0:42:16) ·
**Status: PRESENTED, NOT ADJUDICATED.** The branch call is the author's ([RULE]); the
execution-completeness clause is satisfied (both registered arms ran; none withdrawn).

## 1. Validity — the study stands

- **DS-N1 PASS, at full strength.** All 15 paired seeds, both channels: per-seed MAP grid indices
  **exactly equal** MN0X's stored records, and the floor-aware integer shift law holds with
  **m(h_MAP) = 982 = N at every seed's MAP** — the ×1.7 perturbation was exactly argmax-inert and
  exactly accounted. Abort (d′) does not fire. The specificity control the stage-1 design lacked
  is now on the record and green.
- A-NULL's own DS-M1 row reproduces the paired seeds' base result identically
  (+0.034667 ± 0.001579, the MN0 value — as it must, since the argmax is unchanged per seed).
- MN0X cross-check: recomputed b_ref = +0.037250, |Δ| = 3.5e-17 vs the registered reference.
- Zero rails, zero non-finite `ln_post`, both arms, both channels.

## 2. The measurement

| arm | ch | bias | SE | HPD 50/68/90 | post_sd med | DS-M1 class |
|---|---|---|---|---|---|---|
| **A-M2′** | 1D | **+0.019200** | 0.000746 | 0.000/0.000/0.000 | 0.003953 | **TERM-PARTIAL** |
| **A-M2′** | 2D | **+0.021400** | 0.000737 | 0.000/0.000/0.000 | 0.004071 | **TERM-PARTIAL** |
| A-NULL | 1D | +0.034667 | 0.001579 | 0.000/0.000/0.000 | 0.004265 | TERM-INNOCENT |
| A-NULL | 2D | +0.037000 | 0.001604 | 0.000/0.000/0.000 | 0.004315 | TERM-INNOCENT |

**Restoring the measure/Jacobian inside the z-integral removes Δb(1D) = −0.018050 ± 0.000894
(vs b_ref +0.037250) — 48.5 % of the bias — and leaves +0.019200 ± 0.000746, still 25.7σ from zero
and still confidently wrong (coverage 0/25 at every level).** No channel split: both channels
classify TERM-PARTIAL, so split-precedence does not route to branch 5.

**Against the registered weak expectation (§2, non-branch-carrying, two-sided):** predicted
b ≈ +0.011 ± 0.010, i.e. [+0.001, +0.021]. **1D +0.019200 is inside** (upper quarter);
**2D +0.021400 is outside by +0.0004** — reported as a miss of the weak expectation on the 2D
channel; it carries no branch weight by registration.

**Post-hoc observation (NOT registered, labeled as such):** the §2 arithmetic over-predicted the
total tilt-driven bias by the factor 0.0527/0.0373 = 1.41. Applying that same reconciliation factor
to the predicted M2′ effect, −0.0259 / 1.41 = **−0.0183**, against the measured **−0.0181 (1D)** —
agreement to 2 × 10⁻⁴. The tilt × curvature arithmetic, scaled once by its known overall factor,
accounts for the M2′ share almost exactly. This is stated for the author's information and binds
nothing.

## 3. The registered branch

> **Branch 3 — M2′-PARTIAL** (satisfying arm A-M2′; both channels TERM-PARTIAL).
> Registered meaning: *M2′ contributes but does not own; the registered follow-up is the M6
> composite decomposition (§2), starting from its L0 obligations — no repair is proposed from a
> partial read.*

**M6-L0 state, already in hand** (`M6_L0_KILLTESTS_20260814.md`, presented not adjudicated):
M6-as-registered is **killed on the broadest reading** — tilt not dose-invariant across all
f_h > 0 cells (max dev 37.9 % vs ±10 %; grand mean 3251 nats/h vs 2739 predicted) and α-share
42.9 % vs 52.7 ± 5 pp — while test (ii) (bias/σ²_post constant within factor 2) **survives**, and
the commission's numbers reproduce on the narrower f_i = 1.0 column (2643–2720 nats/h, α-share
52.1 %). Coherent summary offered without adjudication: the bias decomposes as a tilt × curvature
structure whose tilt has (a) the α piece (correct physics, ~1394 nats/h), (b) the missing-J piece
(now **measured** at −0.018 in h via A-M2′), and (c) a residual, *impostor-dose-dependent* tilt
component the pure σ_z-blind form does not capture — the next candidate structure to formalize.

## 4. What is now licensed, and what is not

- LICENSED: "M2′ (missing measure/Jacobian inside the z-integral) **contributes ≈ half** of the
  +1×σ_z displacement; it does not own it." The estimator-term register {M2′} is now *measured*,
  not exhausted-by-assertion; the open set after this readout is the residual account (M6-revised,
  M7).
- NOT LICENSED: any repair (barred from a partial read); any "the register is exhausted" language;
  any quotation of the 2D weak-expectation agreement (it missed by +0.0004).
- The `/physics-change` new-formula slot remains **EMPTY and author-gated**: a J-restoration alone
  demonstrably does not restore calibration (coverage still 0/25), so it is not by itself a
  candidate repair.

## 5. Decisions for the author

1. **[RULE]** Ratify branch 3 (M2′-PARTIAL) as the branch of record, and the DS-N1 PASS + this
   readout as verdicts of record.
2. **[RULE]** The 2D weak-expectation miss (+0.0004 outside a non-branch-carrying window): accept
   as recorded, or order a note in the prereg's expected-nulls register.
3. **[DO]** Authorize the M6-revision L0 work (formalize the dose-dependent tilt residual from the
   committed 20-cell + 2-arm data; CPU-minutes) and the M7-L0 derivation — both required before any
   further arm; **2 L1 slots now used of the stage's 2 — any further instrument arm needs a fresh
   registration.**

*Bands locked at registration and unchanged; scorer frozen pre-data; raw vectors rescored, the
`aggregate` block used only as a labeled cross-check. This document is append-only from its
registering commit.*
