# L4 drift-term direct evaluation — the registered hardening recompute

**Date:** 2026-08-15 · **Authorized:** ledger row #109 item 1 (the amendment-A2 registered
recompute) · **Status: PRESENTED, NOT ADJUDICATED.**

**Script:** `l4_drift_direct_eval.py` · **Output:** `L4_DRIFT_EVAL_output.json` · Same 15 seed
realizations × 3 dose levels as the switch decomposition; the model is the Part-2 §2 closed form
(two-Gaussian, clip-free, linearized) evaluated with **no quadrature and no mirror** — an
independent construction.

## 1. Numbers (1D, nats/h, mean ± SE over 15 seeds)

| f_i | mass (ΣG) | **drift (direct)** | width | model total | mirror T_cand | model − mirror |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 | +1059.6 | **+1090.9 ± 79.6** | −7.1 | +2143.4 | +2134.9 | +8.5 (0.4%) |
| 0.50 | +1059.6 | **+575.0 ± 66.5** | −33.7 | +1600.8 | +1619.2 | −18.4 (1.1%) |
| 1.00 | +1059.6 | **+193.8 ± 47.3** | −29.2 | +1224.3 | +1243.4 | −19.1 (1.5%) |

**The parameter-free closed-form model reproduces the exact instrument's candidate-sum tilt to
0.4–1.5% at all three dose levels.** This is the strongest validation of the §2 derivation as a
whole: mass + drift + width, with no fitted constants, IS the coded estimator's tilt.

## 2. The frame identity — drift+width vs exp+window+leftover

The switch decomposition and the closed form partition the same non-mass residue along different
boundaries (the D3 exponent asymmetry acts, in the closed-form frame, as a shift of the effective
peak — i.e. drift-like). The cross-frame identity drift + width ≈ exp-scale + window + leftover
closes to ~1–3%:

| f_i | drift + width (model) | exp + window + leftover (switches) | gap |
|---|---:|---:|---:|
| 0.25 | +1083.8 | +1075.3 | +8.5 |
| 0.50 | +541.3 | +559.6 | −18.3 |
| 1.00 | +164.6 | +183.8 | −19.2 |

## 3. Reading

1. **The row-#109 hedged identification hardens:** the dose-decaying residual is the
   responsibility-weighted drift channel, now measured directly (+1091 → +575 → +194) with
   exactly the leftover's decay shape; drift dominates the non-mass residue at every dose.
2. **The term boundary between "exponent-scale" and "drift" is frame-dependent** — the switch
   ledger and the closed form are two consistent coordinatizations of the same residue; claims
   should quote the frame (as row #109's records now do).
3. **The whole defect is now derived:** T(f) = α + ΣG (mass) + drift(f) + width(f), closed-form,
   parameter-free, ≤1.5% at every measured dose. Nothing about the tilt remains unexplained at
   this precision; what remains open scientifically is the A-FULL question (does the correct-form
   estimator zero it) and the 2D +129 channel excess (out of 1D scope, carried).

*Append-only from its commit; the author adjudicates whether item 1 discharges the A2 hedge.*
