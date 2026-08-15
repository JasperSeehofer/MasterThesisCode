# STAGE-3 L0 SYNTHESIS — the defect reduced to one object

**Date:** 2026-08-15 · **Covers:** L0-REN-A/B, L0-SB, L0-LIT (ledger row #105) + adversarial
verification (GO; amendments applied as addenda to the underlying notes) · **Status: PRESENTED,
NOT ADJUDICATED.** Decision table in §4.

## 1. What the L0 wave established (verifier-confirmed, zero computational defects)

1. **The displacement law is confirmed, parameter-free.** bias = T/Ā (tilt at truth over actual
   log-posterior curvature) holds at **1.15 ± 0.13** across 16 distinct cells/arms spanning both
   studies and the campaign venue, zero counterexamples on a ±3× band. The former 0.749 "closure
   factor" is dissolved (it was the post_sd/curvature mismatch).
2. **The overconfidence is not a separate anomaly.** post_sd ≈ σ_A: the posterior width is
   approximately *correctly calibrated to local curvature*; the 8.5× displacement-to-width ratio
   is algebraically (T/Ā)/σ_A — fully accounted for by item 1. **The single remaining unexplained
   object in the entire defect is the tilt T itself.** (Open detail, carried: the seed-to-seed MAP
   scatter exceeds the naive sandwich prediction by 2–3×.)
3. **T's ledger:** T(full dose) = 2625 ± 19 = α **+1393.6** (analytic, correct physics, M4) +
   missing-J **+1133 ± 36** (measured via A-M2′, on-prediction incl. the ln D′ term) + residual
   **−62 … +699** (dose-decaying, unlocated).
4. **H-REN split verdict:** the unrenormalized truncated kernel is a **real defect** (R1 LIVE —
   implied shift order 1e-3, sign +, with a population-transfer caveat on precision) but **not
   T_res's owner** (R2 WRONG-SHAPE: T_REN grows with dose, +21/+92, where T_res decays,
   −550/−212) and in budget tension (R3, standing primarily on the sign mismatch).
5. **Non-additivity is live (the derivation's pre-stated fork (ii)):** REN's tilt is itself a
   near-cancellation (+475 boost vs −324 weight-shift in the verifier's exact decomposition), so
   single-term ablations plausibly do not commute — and therefore **T_res may not be a locatable
   term at all, but a bookkeeping residual of an additive decomposition whose ablations don't
   commute.** The joint arm distinguishes these two readings; no single-term arm can.
6. **Literature (full-text):** neither Gray 2020 nor Gray 2023 carries the |d d_L/dz| measure in
   the event term (likelihood-of-z convention; Gray 2020's kernel equation never exercised, mocks
   at σ_z = 0); Gray 2023's own conditional escape clause for unrenormalized truncation requires an
   identical truncated expression in numerator and selection term — a condition a per-candidate
   window with a global α(h) does not obviously meet. Our venue's measured J and REN effects are
   therefore **not contradicted by published practice, and the convention question they raise is
   apparently untested in the literature at σ_z > 0.**

## 2. The state of the mechanism hunt, in one paragraph

The bias is a displacement, not a width failure: a correctly-normalized-looking posterior sits at
the pseudo-true point T/Ā of a misspecified likelihood. Of the tilt T, 53 % is α (correct physics,
uncancelled because the event term is misspecified), 43 % is the measured missing-Jacobian term,
and the remainder is a dose-decaying residual that no single candidate (M1-quadratic, M7, REN)
matches in shape — with live evidence that "residual" may be an artifact of assuming ablations
add. The two located defects (J, REN) are exactly the two places where the event term's
convention (likelihood-of-z, truncated, unrenormalized) diverges from a generative-model density —
and the joint repair is the first configuration that could zero T rather than shave it.

## 3. What A-JREN measures (the draft registration is ready to fill)

`DRAFT_PREREGISTRATION_A_REN.md` (`97a9a11a`): A-REN and the conditional A-JREN, seeds verified
disjoint, A8-v2 throughout, bands TBD-pending-L0 — now fillable from this wave. The R3
BUDGET-TENSION read has fired A-JREN's registered trigger. A-JREN (J + renormalization jointly)
answers, on the instrument at ~25 CPU-h: does the joint repair (a) zero the tilt (bias in-band,
branch M-OWNS-JOINTLY), (b) reduce it beyond the single-term sum (non-additive, partial), or
(c) match the additive sum (T_res is a real third term, hunt continues)? Each outcome is
decisive: (a) ends the mechanism hunt and hands the `/physics-change` gate a complete candidate;
(b) redirects to the interaction structure; (c) re-opens the register with T_res promoted to a
first-class target.

## 4. Decisions for the author

| # | decision | tag |
|---|---|---|
| 1 | Adopt this synthesis (with the verifier amendments) as the L0 wave's record | **[RULE]** |
| 2 | Fill the draft registration's bands from this wave, **register** A-REN + A-JREN (run A-JREN first, A-REN only if its single-term measurement is still needed after the joint result), and run on the cluster (~25–50 CPU-h total, fresh seed blocks +54000/+54100) | **[DO]** |
| 3 | Whether the Gray-convention finding (§1 item 6) enters the paper's scope now or after A-JREN | **[RULE], timing** |

**Tiering if item 2 is granted:** band-filling + registration finalization — orchestrator (bands
are derivations); instrument-switch implementation — sonnet/high (the `estimator_variant` pattern
is established); one inherit/xhigh pre-registration verifier (the M2P-stage precedent caught a
CRITICAL both times it ran — it is not optional); cluster ops — orchestrator via /cluster.

*No repair is proposed here; the `/physics-change` slot remains empty until a joint-arm result
gives it a complete candidate. Append-only from this commit.*
