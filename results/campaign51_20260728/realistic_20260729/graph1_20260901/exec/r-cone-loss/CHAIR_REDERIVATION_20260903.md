# m-cone-loss — chair re-derivation + booking (2026-09-03 ~21:45, Fable 5.1 chair)

Read of record: READ_RECORD_rev2.md + cone_result_rev2_read.json (disjoint sonnet reader, real mode
once, all gates GREEN: G-1 pins, G-2 double anchor, G-3 join 66/76 = P6 counter, G-4 KS p = 0.87 and
binomial envelope p = 1.0). Blindness: COMPROMISED (row #340) — thresholds frozen before the leak.

## Re-derived by the chair from the JSON intermediates (MATCH)
- SE(Δh_cone,1D) = sd_MAD · √(n_OUT + n_OUT²/n_IN) / I_1D = 0.8401 · √(10 + 100/66) / 3256 = 8.755e-4 ✓ (JSON 8.756e-4)
- φ_1D = |Δh|/0.063 = 0.000273/0.063 = 0.00433 ✓ · Z = −0.000273/0.000876 = −0.312 ✓ · M = 0.008/0.000876 = 9.14 ✓
- 2D: Δh −0.000309 ± 0.000906, φ 0.48 %, Z −0.34, M 8.8 — same picture.
- Heavy tails disclosed: plain SD / MAD-scaled SD = 8.5 (1D), 7.5 (2D); 2-outlier sensitivity median 1.00 / 0.86.

## Disposition rows (draft §4, as the script evaluated them)
| row | outcome |
|---|---|
| IMMATERIAL-FLOOR-SHARE (|Δh| < T_mat ∧ φ < 0.2 ∧ M ≥ 3) | TRUE |
| CONE-OWNS-FLOOR | FALSE |
| INTERMEDIATE-UNPOWERED | FALSE (powered, M ≈ 9) |
| INTERMEDIATE (… or linear-vs-leave-out disagree > 2·SE) | TRUE — leave-out Δmean_h = 0.662083 − 0.666987 = −0.004904 vs linear −0.000273; residual 0.00463 > 2·SE = 0.00175 |
Two rows fire → the table is not mutually exclusive on this run → **booked INTERMEDIATE (fresh RULE
to the author)**, chair-derived, with the primary fact: the linear cone-loss statistic is IMMATERIAL
at 0.4 % of the −0.063 rail (powered), i.e. q-cone-loss's kill criterion ("confirms the floor within
band → irreducible geometry") is NOT met in the direction the charter anticipated: the cones that
cannot contain the true host do NOT own the floor.

## Facts for the decider (no ruling)
1. Removing the 10 outside-cone events LOWERS mean_h by 0.0049 (from 0.66699 to 0.66208): these
   events pull the estimate toward truth, not away from it. The −0.063 rail is therefore carried by
   the 1578 events whose true host IS inside the cone (consistent with row #342 K: ~3–6 % of events
   carry the offset, and row #335: the S3 defect localizes to the catalogue-hosted class).
2. The 18× gap between the linear score estimate and the leave-out re-marginalisation is a
   non-linearity of the posterior mean in the removed events' likelihoods — a registered
   cross-check outcome, not a defect; the INTERMEDIATE row exists for it.
3. Harness replicate: Δs (OUT − IN) = +0.32 ± 0.20 over 48 usable universes (19 had no OUT event or
   NaN stencil) — not significant.
4. Cost: ≈ 0.1 CPU-h, local, zero cluster.

## ERRATUM (end-verification D4, 2026-09-03 ~22:10) — booking CORRECTED
The draft's own §2 rule reads: "Disagreement beyond 2·SE flags the linear response as non-linear and
the read is booked on the leave-out number with the flag." The chair had booked INTERMEDIATE on the
two-rows-fire literal reading and omitted this resolution rule. Applying it: leave-out Δmean_h =
−0.004904 → φ_leave-out = 0.0049/0.063 = 7.8 % < 0.2, |Δh| < T_mat = 0.008 → **IMMATERIAL-FLOOR-SHARE,
booked on the leave-out number, non-linearity flag attached** (chair-derived; returns as fresh RULE).
The verifier also shows the "18×" gap was a mismatched comparison (Δh_cone is the excess over s̄_IN;
leave-out removes OUT wholesale); like-for-like −Σ_OUT s_e/I_1D = −0.00347 ± 0.00082 vs −0.00490 is
1.75·SE — inside 2·SE. The non-linearity claim in "facts for the decider" item 2 is WITHDRAWN.
