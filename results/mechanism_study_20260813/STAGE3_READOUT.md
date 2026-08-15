# STAGE-3 READOUT — A-JREN joint repair

**Date:** 2026-08-15 · **Prereg:** `PREREGISTRATION_A_JREN_STAGE3.md` (registered `5b58dab7`) ·
**Scorer:** `score_stage3.py` (pre-committed in the registering commit) · **Data:**
`AJREN_h0p730_results_seeds0_25.json` (job 6320397, COMPLETED 0:0, wall 1:29:25) ·
**Status: PRESENTED, NOT ADJUDICATED.**

## 1. The measurement

| arm | ch | bias | SE | HPD 50/68/90 | post_sd med | DS-M1 | DS-J1 restored |
|---|---|---|---|---|---|---|---|
| **A-JREN** | 1D | **+0.017800** | 0.000712 | 0.000/0.000/0.000 | 0.005939 | **TERM-PARTIAL** | **NO** |
| **A-JREN** | 2D | **+0.022200** | 0.000712 | 0.000/0.000/0.000 | 0.005936 | **TERM-PARTIAL** | **NO** |

MN0X cross-check exact (|Δ| = 3.5e-17). Zero rails, zero non-finite. No DS-M1 class split;
per the registered F3 note the outcome adjudicates through **branch 5** (author call).

## 2. Against the registered F2 window — additivity confirmed (1D), mild 2D sub-additivity

- **1D:** measured +0.017800 vs the additive prediction **+0.0173 ± 0.012 — INSIDE, 0.0005 from
  dead center.** In differences: Δb(joint) = −0.019450 ± 0.000873 vs additive sum
  −0.01805 (J) − 0.0019 (REN) = −0.01995 — agreement within **0.6σ**. **The two located defects
  repair additively; outcome (c) of the adopted synthesis: T_res is REAL and promoted to a
  first-class target.**
- **2D:** Δb = −0.017550 vs additive −0.02025 — a **+0.0027 (≈3.8σ) sub-additive residual on the
  2D channel only**, reported raw (the F2 window is 1D-registered; no branch weight).
- **Coverage is NOT restored** (0/25 at every level, both channels) — repairing both located terms
  does not recalibrate the estimator. Notable diagnostic: **bias/post_sd drops from 8.49 to
  3.00** — the renormalization broadens the claimed width (post_sd 0.00439 → 0.00594) while the
  displacement shrinks, i.e. the joint repair moves *both* factors of the displacement law but
  eliminates neither.

## 3. What this settles

1. **The mechanism ledger is now fully measured at the single-term level:** of the original
  +0.037250, the missing Jacobian carries −0.01805, the unrenormalized truncation −0.0019 (its
  small toy prediction confirmed in the joint arm by additivity), and **+0.0178 remains — the
  α-tilt (correct physics, still uncancelled) plus T_res, exactly the unlocated object the
  synthesis promoted.** The 1D additivity kills the "T_res is a bookkeeping artifact of
  non-commuting ablations" reading: the residual is not an interaction term (at 1D precision).
2. **No repair candidate is complete.** The `/physics-change` slot stays empty on the registered
  bar (partial read). The located terms are necessary but demonstrably not sufficient.
3. **The next target is sharply constrained:** T_res is dose-decaying (+699→−62 nats/h),
  decelerating, f_h-independent, survives J+REN repair, is not M1's quadratic (refuted), not M7's
  edge flux (closed), not REN (wrong shape), and now not an ablation interaction (1D). The 2D-only
  sub-additive +0.0027 is the one new lead this arm produced.

## 4. Decisions for the author

1. **[RULE]** Ratify this readout and the branch-5 record (additivity confirmed at 1D; T_res
   promoted; coverage not restored).
2. **[RULE]** A-REN (conditional single-term arm, seeds reserved): with its −0.0019 single-term
   effect now confirmed inside the joint arm by additivity, the marginal value of running it alone
   is low (a ~2.7σ detection at N = 25). Recommend **withdraw by [RULE]**; running it remains
   registered and available.
3. **[DO/next]** Stage-4 direction: an L0-first hunt for T_res under the §3 constraint set, with
   the 2D sub-additivity as a discriminator — proposal to follow as a reviewable artifact if
   granted.

*Bands locked at registration; scorer pre-committed; raw vectors rescored; append-only from its
registering commit.*
