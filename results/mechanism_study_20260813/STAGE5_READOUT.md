# STAGE-5 READOUT — A-FULL (FULL-F, the correct-form estimator)

**Date:** 2026-08-16 · **Prereg:** `PREREGISTRATION_A_FULL_STAGE5.md` (registered `dec62032`) ·
**Scorer:** `score_stage5.py` (pre-committed in the registering commit; output
`score_stage5_output.json`) · **Data:** `AFULL_h0p730_results_seeds0_25.json` (job 6327889,
COMPLETED 0:0, wall 1:17:52, 15 workers, seeds +54200…+54224) ·
**Status: PRESENTED, NOT ADJUDICATED.**

## 1. The measurement (registered reads, verbatim from the scorer)

| DS | read | verdict |
|---|---|---|
| **DS-F1** (primary) | T(1D) = **+22.0 ± 29.2** nats/h, band [−131.5, +192.7]; 0.16σ from the mirror prediction +30.6 | **PASS** |
| DS-F2 (weak) | bias(1D) = **+0.0010 ± 0.0011** (consistent with zero; original defect +0.0373) | MET |
| **DS-F3** (coverage) | 1D hpd50/68/90 = **0.640 / 0.760 / 0.960** vs nominal 0.50/0.68/0.90 (binomial bands) | **RESTORED** |
| DS-F4 (2D, descriptive) | T(2D) − T(1D) = **+135.7** (coded-form reference +129 ± 24) | persists |
| DS-F5 | zero rails, zero non-finite, per-seed sd 146.1 | clean |

2D channel (report, non-branch-carrying): T(2D) = +157.7 ± 29.2; bias(2D) = +0.0076 ± 0.0012;
coverage 0.360/0.480/0.760 — **not restored in 2D**.

**Branch (§5, mechanical): 1 — DS-F1 PASS + DS-F3 RESTORED → M-OWNED-CLOSED candidate.**

## 2. What this settles (pending ratification)

1. **The venue mechanism account is validated end-to-end on the instrument.** The correct-form
   estimator — d_obs-density GW factor × selected-population prior w_pop·S̄_φ/α × leave-one-out
   impostor weight; no Jacobian, no kernel renormalization — zeroes the 1D tilt (+2644 → +22 ±
   29), zeroes the 1D bias (+0.0373 → +0.0010 ± 0.0011), and **restores 1D coverage from 0/25 to
   nominal-consistent** — the first configuration in the entire thread to do so. Prediction-to-
   measurement: 0.16σ (tilt), with coverage restoration predicted qualitatively (P3).
2. **The 1D H₀-bias mechanism is therefore closed at the venue level:** α-pairing (D1+D4 one
   broken pairing) + GW z-mass growth + exponent scale + LOO weight, each derived, each priced,
   each repaired, ledger closing to a zero-consistent remainder.
3. **The 2D channel is now the single remaining located defect:** its +135.7 excess tilt and
   +0.0076 ± 0.0012 bias (~6σ) survive the full 1D-correct form — the defect lives in the
   mass-channel factor g_i (`completion_mass_factor_g` machinery), not in anything the 1D repair
   touches. This was the carried residual of rows #109/#110; it is now isolated with an
   order-of-magnitude-cleaner background (everything else is zero).
4. Stated residuals honored as pre-registered: the low-dose (+169 at f_i = 0.25) prediction was
   not probed (full dose only); the pool-vs-model prior mismatch stands as a population-model
   systematic (KS D = 0.085), untouched by this arm.

## 3. Decisions for the author

| # | decision | tag |
|---|---|---|
| 1 | Ratify this readout and the branch-1 record: the 1D venue mechanism thread **CLOSES**, M-OWNED by the correct-form account | **[RULE]** |
| 2 | Open the **production `/physics-change` proposal** for `bayesian_statistics.py` (the same broken pairing + ratio-form event term exists in production; this arm is the evidence base; full 5-step gate, its own reviewable artifact) | **[DO]** |
| 3 | The 2D mass-channel defect: open a targeted investigation (L0-first: derive the g_i term's own tilt in the convolution frame, then a mirror pre-measurement — the stage-4/5 method transfers directly), or carry it as a stated 2D residual | **[RULE]** |

*Bands locked at registration; scorer pre-committed; raw vectors rescored; append-only from its
registering commit.*
