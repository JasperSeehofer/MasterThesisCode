# generator_marginal 7-point probe (seed1000, deep venue) — measured vs pre-registered predictions

**Date:** 2026-07-26 · **Commit under test:** 8fbb21e (`[PHYSICS] generator_marginal`, 4d_exact
primary, point/point σ_z pairing) · **Baseline:** `../v1_probe` (`absolute_marginal`, 49b9ade-era
stack) · **Events:** 3454 (1996 fallback / 1458 host-found; 307 A-dominated at h=0.73).
**Command:** fused `--h_values 0.60,0.65,0.70,0.73,0.76,0.80,0.86 --seed 1000
--normalization_mode generator_marginal` (identical inputs to v1_probe via symlinked
`prepared_cramer_rao_bounds.csv` / `injections/`).

Pre-registered predictions: `../DERIVATION_GENERATOR_CONSISTENT_NORM.md` §6.3 + §9 gate 2.
Metric: total 1D ln-likelihood Σ_i ln p_i(h) (reproduces the baseline's +54.24 gap exactly).

## Headline

| Quantity | absolute_marginal (measured) | packet prediction (4D) | generator_marginal (measured) |
|---|---|---|---|
| 1D MAP (grid) | **0.86 RAIL** | still rails HIGH | **0.73 (= truth)** |
| 2D MAP (grid) | 0.86 RAIL | — | **0.73 (= truth)** |
| 1D gap 0.73→0.86 (ln) | +54.24 | ≈ +52 | **−898.8** |
| 2D gap 0.73→0.86 (ln) | +128.2 | — | **−735.4** |

1D lnL relative to h=0.73: −1475 (0.60), −1143 (0.65), −890 (0.70), 0, −874 (0.76),
−897 (0.80), −899 (0.86) — a sharp, broad-based truth peak (138 events contribute >1 ln
to the 0.73-vs-0.76 difference, 88 events >5 ln, largest single contribution 18.2 ln;
zero-likelihood events: none at any h — the sums are clean).

**The deep-venue HIGH rail is cured; the posterior peaks at the injected truth in both
channels.** Per the packet's own pre-registered gate (§9.2: "a measured gap far BELOW the
prediction falsifies the §6.4 attribution and would re-open FIX-3 as a rail candidate"),
this measurement FALSIFIES the packet's §6.4 claim that the rail is not
composition-normalization-sourced: the full generator-consistent estimator (normalization
substitutions + point/point σ_z pairing) de-rails the deep venue by itself.

## Mechanism decomposition (why the gap prediction missed)

The packet's §6.3 gap extrapolation held the numerator kernel N_g FIXED at the σ_z form
and modeled only the n̄_w→n̂_w and D→D_gen substitutions. Every quantity attributable to
those substitutions matches the packet to 3–4 digits:

| Quantity | packet | measured |
|---|---|---|
| n̂_w(0.73) | 2.7317 | 2.7317 |
| d ln n̂_w/dh | 3/h = +4.1096 | +4.1116 (0.70→0.76 secant) |
| d ln D_gen/dh (4D) | ≈ −1.49 | −1.490 |
| P̂(cat\|det, 0.73) | 0.113 (gate [0.10, 0.19]) | 0.1133 |
| fallback Δ(d ln p_i/dh), D→D_gen4 only | −0.028 | **−0.027** |
| W_cat / V_f(0.73) anchors | 6.3477e8 / 2.3237e8 | rel 1.9e-16 / 2.5e-11 |

The deviation is carried entirely by the **point/point σ_z leg** (approved author decision
§7.2, hard-verified generator-exact): with N_g = p(x|z_g) point-evaluated on scatter-free
mock redshifts, catalogue-matched events become sharply h-informative
(d_L(z_g;h) pinned against d̂_L at Fisher precision), which the kernel-smoothed
extrapolation could not represent:

- A-dominated Δ(d ln p_i/dh) at truth: measured **−2.87** vs −0.23±0.05 predicted
  (kernel-N_g convention) — the extra ≈ −2.5/h is the point-numerator sharpening.
- Host-found mean slope: +0.701 → +0.393 (packet, kernel-N_g 3D convention: → ≈ +0.81).
- B-dominated host-found: +0.634 → +1.008 (their A-terms also sharpen toward truth).
- ALL-events mean slope at truth: +0.223 → **+0.077** (≈ 3× closer to zero-mean score).

## Honest caveats

1. Slopes above are 0.70→0.76 secants; the truth peak is narrower than the grid step, so
   per-class "slopes at truth" mix peak and plateau. A dense grid around 0.73 is needed to
   resolve the peak width / local curvature (σ(h) estimate).
2. Nothing was tuned (per instruction). The result is the exact assembly of the approved
   Eqs. (3)–(5) + point/point; the normalization layer reproduces every packet anchor.
3. The seed600 shallow p_det→1 A/B gate (§9.1) has not been re-run yet; §5(d)'s algebraic
   identity holds for the normalization but NOT for the point-N_g leg (the packet's §5d
   assumed kernel-N_g), so the seed600 gate should be re-registered accordingly.
4. FIX-2 (z-resolved survival) is no longer needed to explain a HIGH rail on this venue;
   its status should be re-assessed against this stack.

Analysis script: session scratchpad `analyze_genmarg_probe.py` (metric locked against the
baseline's +54.24). Raw outputs in `simulations/` (posteriors both channels, diagnostics
CSV, probe.log).
