# L4-T1 — post-repair tilt composition (A-JREN, full dose)

**Date:** 2026-08-15 · **Authorized:** ledger row #108 · **Status: PRESENTED, NOT
ADJUDICATED.** Numbers only; the author adjudicates.

**Script:** `l4_t1_tilt_composition.py` · **Output:** `L4_T1_output.json` · ruff / ruff-format /
mypy clean.

**Data:** `AJREN_h0p730_results_seeds0_25.json` (joint J+REN repair, `estimator_variant:
jacobian_and_kernel_renorm`, n=25 seeds) and `AM2P_h0p730_results_seeds0_25.json` (J-only repair,
`estimator_variant: m2prime_jacobian`, n=25 seeds), both committed under
`results/mechanism_study_20260813/`. L0-REN-B toy and L0-SB reference numbers are quoted from
`L0_REN_B_TOY_RESULTS_20260815.md` and `L0_SB_DIAGNOSTIC_20260815.md` (committed, not recomputed
here). **Never reads a file's `aggregate` block** — all `T` values below are the raw-vector
`ln_post_1d`/`ln_post_2d` grid-neighbour central difference at `h_true`, verbatim
`venue_transfer._slope_at_truth` / `m6r_l0_decomposition._slope_at_truth_per_seed` geometry.

## 1. T(AJREN) vs the analytic alpha-tilt

| channel | T(AJREN), n=25 seeds | alpha-tilt (analytic) | T − alpha | significance |
|---|---|---|---|---|
| 1D | **+514.52 ± 16.76** nats/h | +1393.63 nats/h | **−879.11 ± 16.76** | **52.5σ from zero** |
| 2D | **+643.17 ± 16.77** nats/h | +1393.63 nats/h | **−750.46 ± 16.77** | **44.8σ from zero** |

T(AJREN) is roughly a third of the analytic alpha-tilt in 1D (37%) and under half in 2D (46%), not
equal to it within errors. **The "T_res(full dose) ≈ 0 post-repair, remaining bias = pure
uncancelled alpha" hypothesis does not hold on this reading**: the residual T(AJREN) − alpha is
large, negative, and channel-dependent (1D and 2D residuals differ by ~129 nats/h, ~5.4σ combined),
i.e. a real T_res survives the joint repair at full dose, on top of whatever alpha-share is present
in T(AJREN).

## 2. Instrument REN tilt vs the L0-REN-B toy

| arm | T1 (1D) | T2 (2D) |
|---|---|---|
| AJREN (J+REN) | +514.52 ± 16.76 | +643.17 ± 16.77 |
| AM2P (J-only) | +1492.01 ± 30.67 | +1623.51 ± 30.69 |
| **T(AJREN) − T(AM2P)** = instrument REN tilt | **−977.49 ± 34.95** | **−980.34 ± 34.97** |

| | full-dose (f_i=1.0) value | uncertainty |
|---|---|---|
| Instrument REN tilt (this arm, both channels ≈ −978 to −980) | −977 to −980 | ±35 (SE) |
| L0-REN-B toy, full dose | **+99.52** | ±18.43 (SE) / ±52.13 (seed-scatter std) |

Instrument-minus-toy separation: 27.3σ (1D, toy SE), 27.3σ (2D, toy SE); 17.2σ / 17.2σ using the
toy's seed-scatter std instead. **The instrument REN tilt is opposite in sign and roughly an order
of magnitude larger in magnitude than the toy's production-population-transfer prediction** —
not a confirmation of the toy read at full dose on the real instrument. (Toy caveat on record:
L0-REN-B is a 500-event toy at its own z_median, not the production n(z); the toy result was
always flagged as a transfer check, not a point prediction.)

## 3. Displacement-law closure on AJREN (out-of-sample, parameter-free)

Using AJREN's **own** local-quadratic-fit curvature Ā at truth (verbatim
`l0_sb_diagnostic._local_quadratic_fit` geometry, half-window=2) as the reference curvature:

| channel | bias (measured) | T/Ā (predicted bias) | ratio = bias / (T/Ā) | L0-SB headline ratio | Δ from headline |
|---|---|---|---|---|---|
| 1D | +0.017800 ± 0.000712 | +0.017995 | **0.9891** | 1.147 ± 0.132 | −0.158 (1.2σ) |
| 2D | +0.022200 ± 0.000712 | +0.022409 | **0.9907** | 1.164 ± 0.134 | −0.173 (1.3σ) |

The displacement law is **stronger, not weaker, on AJREN than the fitted headline itself**: the
ratio closes to within ~1% of exact unity in both channels, sitting ~1.2–1.3σ below the S-cell
headline's own 1.147/1.164 ± 0.132/0.134 — inside the headline's own scatter band by any
conventional threshold looser than 1σ, and closer to the textbook Laplace value (ratio = 1) than
the pre-repair cells that calibrated the headline. This is a genuine out-of-sample point (AJREN was
not part of the 16-S-cell + prior-arm fit set): the law transfers.

## 4. 2D channel — same three numbers, checked against the Stage-3 2D-only sub-additivity (+0.0027)

| quantity | 1D | 2D | 2D − 1D |
|---|---|---|---|
| T(AJREN) | +514.52 ± 16.76 | +643.17 ± 16.77 | **+128.65 ± 23.72** (5.4σ) |
| T − alpha | −879.11 | −750.46 | +128.65 (same) |
| instrument REN tilt | −977.49 ± 34.95 | −980.34 ± 34.97 | −2.85 ± 49.5 (~0.06σ, no excess) |
| displacement-law ratio | 0.9891 | 0.9907 | +0.0016 (no material excess) |

**T(AJREN) itself is ~129 nats/h larger in 2D than 1D (5.4σ)** — a real, channel-asymmetric excess
in the raw tilt, consistent in direction with the Stage-3 finding that the joint repair is mildly
sub-additive specifically in 2D (+0.0027, ≈3.8σ, `STAGE3_READOUT.md` §2). However, **the instrument
REN tilt (T(AJREN) − T(AM2P)) shows no corresponding 2D-only excess** (−977.5 vs −980.3, difference
consistent with zero) and **the displacement-law ratio is essentially channel-identical** (0.989 vs
0.991). So on this reading the 2D-only excess lives in T(AJREN) directly (and, by extension, in
T(AM2P), since T(AM2P) also differs by channel: +1492.0 vs +1623.5, a 131.5-nat 2D excess of its
own — nearly identical in size to AJREN's 128.65), not in the REN-specific difference. The
2D-vs-1D gap looks like a property carried by the estimator's tilt broadly (present already at
J-only dose) rather than something introduced newly by the kernel-mass renormalization step.

## 5. Summary table (all numbers, both channels)

| quantity | 1D | 2D |
|---|---|---|
| T(AJREN), n=25 | +514.52 ± 16.76 | +643.17 ± 16.77 |
| T(AM2P), n=25 | +1492.01 ± 30.67 | +1623.51 ± 30.69 |
| alpha-tilt (analytic) | +1393.63 | +1393.63 |
| T(AJREN) − alpha | −879.11 ± 16.76 | −750.46 ± 16.77 |
| instrument REN tilt = T(AJREN) − T(AM2P) | −977.49 ± 34.95 | −980.34 ± 34.97 |
| toy REN tilt (full dose) | +99.52 ± 18.43 (SE) / ±52.13 (std) | (channel-agnostic toy) |
| displacement-law ratio (bias / (T/Ā), own Ā) | 0.9891 | 0.9907 |
| L0-SB headline displacement ratio (reference) | 1.147 ± 0.132 | 1.164 ± 0.134 |

## 6. Constants used

- Analytic alpha-tilt: `ALPHA_SLOPE_COEFF * ALPHA_N / ALPHA_H = 1.036 * 982 / 0.730 = 1393.63`
  nats/h (`PREREGISTRATION_M2PRIME_ABLATION.md` Sec.2, reused verbatim from
  `m6r_l0_decomposition.py`).
- L0-REN-B toy, full dose: +99.52 nats/h, seed-scatter std 52.13, SE 18.43 (n=8 seeds),
  `L0_REN_B_TOY_RESULTS_20260815.md` §2 (production-scaled to N=982).
- L0-SB headline displacement-law ratio: 1.147 ± 0.132 (1D) / 1.164 ± 0.134 (2D),
  `L0_SB_DIAGNOSTIC_20260815.md` line 103 / `L0_SB_output.json`
  `section3_predictions.ratio_stats_Abar_{1d,2d}`.
- Stage-3 2D-only sub-additivity: +0.0027 (≈3.8σ), `STAGE3_READOUT.md` §2.
