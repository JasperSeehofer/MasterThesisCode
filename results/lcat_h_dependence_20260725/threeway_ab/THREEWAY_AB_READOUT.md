# Three-way per-leg attribution A/B (math-review F2) — readout 2026-07-26

**Setup:** seed-1000 deep venue, identical inputs to `../v1_probe*` (symlinked
`prepared_cramer_rao_bounds.csv` + `injections/` from `v1_probe_smeared`),
fused 7-point grid {0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86}, 3454 events,
no zero-likelihood events at any h in any cell. New cell run at `906284c0b`
(= PR #48 flag + PR #45 log dedup, local merge branch `probe/threeway-ab`)
with `--normalization_mode generator_marginal --host_z_kernel volume_deconv`
— the #40a decomposition flag's first production-scale use. Metric: total
ln-likelihood Σ_i ln p_i(h) − Σ_i ln p_i(0.73) from
`simulations/diagnostics/event_likelihoods.csv` (combined_no_bh /
combined_with_bh), matching `../v1_probe_genmarg/PROBE_RESULTS.md`.

## The three cells (gap ≡ lnL(0.86) − lnL(0.73))

| Cell | numerator kernel | normalization | 1D MAP | 1D gap | 2D MAP | 2D gap |
|---|---|---|---|---|---|---|
| A `absolute_marginal` (baseline) | volume_deconv | n̄_w, D | **0.86 RAIL** | +54.2 | **0.86 RAIL** | +128.2 |
| B `generator_marginal` + `host_z_kernel=volume_deconv` (NEW) | volume_deconv | n̂_w, D_gen | **0.73 = truth** | **−85.4** | 0.80 (interior, +29.4 over truth) | +13.4 |
| C `generator_marginal` (production) | δ (point) | n̂_w, D_gen | **0.73 = truth** | −898.8 | **0.73 = truth** | −735.4 |

1D lnL−lnL(0.73) per h, cell B: −72.3 (0.60), −19.5 (0.65), −2.1 (0.70), 0,
−5.9 (0.76), −29.6 (0.80), −85.4 (0.86).
2D, cell B: −204.4, −98.2, −29.4, 0, +20.2, +29.4, +13.4.

## Attribution (A→B = normalization legs alone; B→C = δ-kernel alone)

| Channel | total movement A→C | normalization legs (A→B) | δ-kernel (B→C) | δ-kernel share |
|---|---|---|---|---|
| 1D | −953.0 ln | −139.6 ln | −813.4 ln | **85.3%** |
| 2D | −863.6 ln | −114.8 ln | −748.8 ln | **86.7%** |

## Findings

1. **The redteam's F2 magnitude claim is confirmed with a refinement**: the
   δ-kernel numerator carries ~85–87% of the total ln movement (redteam
   estimate: ~95%). The paper attribution sentence should say ~85%.
2. **Qualitative refinement of F2 — the de-railing and the sharpness have
   different owners.** The normalization substitutions ALONE (cell B) de-rail
   the 1D channel completely: MAP at truth, monotone fall-off both sides.
   The δ-kernel's ln-share buys peak *depth/information*, not the 1D cure.
3. **The 2D (with-BH-mass) channel is the exception**: with the broadened
   kernel numerator it retains an interior HIGH tilt (MAP 0.80, +29.4 ln over
   truth; +13.4 at 0.86). Only the δ-kernel brings the 2D MAP to truth. So
   the point/point pairing is load-bearing for the 2D channel specifically —
   directly relevant to the real-data kernel derivation (§3.6 of
   `docs/derivations/hostz_pv_photoz_kernel.md`): with a broadened kernel on
   real data, expect the 2D channel to need the re-derived mass-kernel
   treatment, not just wider windows. The residual 2D tilt with kernel
   numerator is consistent in sign with the known 2D +0.025 residual history
   (mass-kernel truncation thread, EXP-45).
4. Cell B is a diagnostic combination (generator-consistent normalization is
   *derived* for the point/point generator pairing — mixing it with a
   broadened numerator kernel is per-leg attribution instrumentation, not a
   candidate production mode).

## Provenance

- Cell A: `../v1_probe/` (absolute_marginal, 49b9ade-era stack).
- Cell C: `../v1_probe_genmarg/PROBE_RESULTS.md` (8fbb21e).
- Cell B: `../v1_probe_genmarg_vdkernel/` (906284c0b, 2026-07-26T22:26).
- Flag: PR #48 (`resolve_host_z_kernel`, issue #40a).
