# FIX-2 z-resolved survival probes (seed1000, deep venue) — stacked + FIX-2-alone

**Date:** 2026-07-26 · **Commit under test:** `a608c4f` (`[PHYSICS] z-resolved detection
survival S(d_L|z)`, gated `--pdet_z_resolved`, on top of FIX-3 `8fbb21e`).
**Inputs:** identical to `../v1_probe*` (seed1000 `prepared_cramer_rao_bounds.csv`, canonical
`data/injections` pool, 3454 events). Metric: total ln-likelihood Σ_i ln p_i(h) per channel
(same statistic as `../v1_probe_genmarg/PROBE_RESULTS.md`); physics-floor combine both channels.

**Interpretation baseline (per coordinator update 2026-07-26):** the FIX-3-alone probe
(`fb361e8`) falsified the packet's +92 ln baseline (MAP at truth, gap −898.8/−735.4), so the
stacked probe tests whether FIX-2 PRESERVES the truth peak — the packet's −21 ln stacked
number is obsolete as a gate. The FIX-2-alone −69 ln prediction (on the unchanged
absolute_marginal baseline, `../v1_probe` +54.24) remains a valid two-sided mechanism test.

## Probe A — stacked (generator_marginal + --pdet_z_resolved), 7-point

| h | 1D lnL rel. 0.73 | 2D lnL rel. 0.73 |
|---|---|---|
| 0.60 | −1429.9 | −1488.3 |
| 0.65 | −1109.5 | −1093.7 |
| 0.70 | −877.6 | −813.0 |
| 0.73 | **0** | **0** |
| 0.76 | −888.4 | −789.2 |
| 0.80 | −937.2 | −805.2 |
| 0.86 | −994.8 | −831.4 |

- **1D MAP = 0.73 (truth); 2D MAP = 0.73 (truth). Physics-floor combine: MAP h = 0.7300 in
  BOTH channels.** The FIX-3 truth peak is PRESERVED (not displaced) and the anti-0.86 margin
  deepens: 1D gap 0.73→0.86 = **−994.8 ln** (FIX-3-alone: −898.8), 2D **−831.4** (−735.4).
  Peak shape stays sharp and broad-based (every off-truth point ≥ 789 ln below).
- Mechanism (4d_exact assembly): n̂_w table IDENTICAL to FIX-3-alone (2.7317 at 0.73;
  d ln n̂_w/dh = +4.11) — FIX-2 correctly touches no draw-side normalizer.
  D_gen_zres(0.73) = 7.5947e8; d ln D_gen_zres/dh (0.70→0.76 secant) = **−1.360** vs −1.490
  FIX-3-alone (+0.13, the survival-shape tilt);
  P̂(cat|det, 0.73) = **0.1372** vs 0.1133 FIX-3-alone (β_Ḡ ×0.803 while Σ_glob_wbh unchanged).
  β_Ḡ_zres(0.73) = 6.5524e8 = 0.8033 × pooled (packet: 0.803 ✓).
- Zero-likelihood events: none at any h, either channel ("empty 2" in the 2D combine = the
  same two structurally empty events as the v1 probes).

## Probe B — FIX-2 alone (absolute_marginal + --pdet_z_resolved), 3-point

| h | 1D lnL rel. 0.73 | 2D lnL rel. 0.73 |
|---|---|---|
| 0.60 | −77.5 | −177.7 |
| 0.73 | 0 | 0 |
| 0.86 | **−68.8** | +9.7 |

- **1D gap 0.73→0.86 = −68.75 ln vs the packet's predicted −69 ln (§6 "FIX-2 alone on current
  stack") — hit to within 0.3 ln.** Baseline (pooled absolute_marginal, `../v1_probe`): +54.24.
  The predicted LOW overshoot is real and quantitatively confirmed: FIX-2 alone replaces the
  HIGH rail with a comparable LOW tilt, exactly the compensation structure the packet derived.
  Combined with Probe A this is the two-sided verification that the survival-shape mechanism
  (D→D_zres bookkeeping, −0.26/h → here −123 ln swing over 3454 events) is real, not accidental.
- 2D channel: +9.7 ln toward 0.86 (grid MAP 0.86 on the coarse 3-point grid; near-flat).
  The 2D channel keeps the M_z-conditioned survival unchanged by design, so FIX-2 enters it
  only through D = β_G + β_Ḡ; no packet-level 2D prediction was pre-registered.

## Verdict

- FIX-2 stacked on FIX-3 **preserves the truth peak in both channels** and does not displace
  the MAP (0.73 → 0.73); the margin against the old rail deepens by ~96 ln.
- The FIX-2-alone −69 ln prediction is reproduced at −68.75 ln — the packet's survival-shape
  mechanism is quantitatively verified on the real event set.
- **No tuning was performed under any outcome** (per instruction).

## Anchor reproduction (production code path, real 50k pool, 41-h grid)

| Quantity @0.73 | pooled | z-resolved | packet |
|---|---|---|---|
| dlogD/dh (sky-aware) | −1.5198 | −1.2989 | −1.5176 / −1.26 |
| dlogD/dh (isotropic) | −1.5199 | −1.3055 | — |
| dlogβ_Ḡ/dh | −1.2023 | −0.9753 | −1.200 / −0.929 |
| D_zres/D | — | 0.8021 | 0.801 |
| β_Ḡ_zres/β_Ḡ | — | 0.8033 | 0.803 |

The z-resolved slope −1.30 vs the packet's −1.26 headline is NOT an implementation
difference: queried on the packet script's own 3000-point one-sided-rounded d_L grid the
implementation returns **−1.27252, bit-matching the packet estimator in the same integral
harness** — the −1.26 headline carries ~0.03–0.04 of the z2 script's own d_L rounding and
interpolated cosmology tables. The exact-in-d_L production implementation shipped unchanged.

Raw compact artifacts: `genmarg_stacked/` and `absmarg_fix2_alone/` (1D per-h posteriors,
combined posteriors both channels, run metadata). Full logs + 2D per-h JSONs (158 MB/h) in the
session scratchpad probe dirs (re-runnable from the commands in each `run_metadata.json`).
