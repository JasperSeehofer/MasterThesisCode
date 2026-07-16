# FINDING — Part 1 `volume_trunc` FALSIFIED at the seed600 decisive gate

**Date:** 2026-07-12 · **Branch:** `physics/zero-host-completion-fallback` ·
**Verdict:** ❌ The approved Part 1 shallow-venue kernel correction (`volume_trunc`)
**does not fix the shallow bias — it makes it substantially worse.** Do NOT deploy
to the campaign. The production default `volume_deconv` is untouched (byte-identical).

## What was tested

Per `.planning/HANDOFF-VOLUME-TRUNC-EXEC-20260712.md` / scoping §7b:
`volume_trunc` = the calibrated volume kernel with (i) the lower z-limit floored at
0 instead of 1e-6 and (ii) the in-catalogue **numerator** integrated over the
per-host galaxy window `[z_g−4σ, z_g+4σ]` (shared with `Z_g`/`D_g`) instead of the
event-level GW window. Decisive gate: seed600 494-event shallow-venue A/B,
`volume_trunc` vs `volume_deconv`, same 7-point grid as the N-5 / Eddington driver.
Driver: `scripts/volume_trunc_ab.py`. Raw result: `gate_result.json`.

## Result (truth h = 0.73)

| channel | mode | MAP | mean | edge | posterior shape |
|---|---|---|---|---|---|
| 1D | `volume_deconv` (baseline) | 0.73 | **0.7450** | 0.000 | 0.53 @0.73, 0.44 @0.76 |
| 1D | `volume_trunc` | 0.80 | **0.8000** | 0.000 | **0.999 @0.80** |
| 2D | `volume_deconv` (baseline) | 0.76 | **0.7681** | 0.003 | 0.73 @0.76, 0.22 @0.80 |
| 2D | `volume_trunc` | 0.80 | **0.8000** | 0.000 | **0.9998 @0.80** |

Δ(trunc − deconv): **1D mean +0.0549, 2D mean +0.0319** — moved AWAY from truth,
in the WRONG direction, and by ~4× the +0.013 residual we were trying to remove.

**Baseline validation:** the `volume_deconv` arm reproduces the established seed600
subsample reference exactly (1D mean 0.745, 2D mean 0.768) → the driver + data are
sound; the divergence is entirely the `volume_trunc` kernel.

## Mechanism (two compounding effects; `quadrature_diagnostic.py`)

For a representative shallow host (z_g = 0.05, σ_z = 0.033, σ_z/z ≈ 0.66) the host
window is wide (`[0, 0.18]` in z) while the GW likelihood peak in z is narrow
(~0.003, from the 5% d_L localization):

| h | numerator, GW window (n=50) | numerator, host window (n=50) | numerator, host window (exact quad) |
|---|---|---|---|
| 0.60 | 0.0003 | **0.0000** | 0.2417 |
| 0.73 | 0.0005 | **0.0000** | 0.4314 |
| 0.86 | 0.0007 | **0.0000** | 0.6537 |

1. **Quadrature aliasing (dominant).** The shared `fixed_quad(n=50)` — correct for
   the *narrow, peak-centred* GW window — is numerically invalid over the *wide*
   host window: the sparse Gauss-Legendre nodes straddle the narrow GW peak and
   miss it (n=50 → 0.0 vs exact 0.24–0.65). Which nodes happen to catch a peak
   depends on the host and on h, so the per-host numerators are erratic and
   h-dependent → the combined posterior collapses onto whichever grid point (0.80)
   the aliasing favours. This is an implementation-adequacy failure, not a clean
   test of the physics.
2. **Genuine high-h tilt.** Even the *exact* host-window numerator is monotonically
   INCREASING in h (0.24 → 0.65). Unifying the numerator support integrates the GW
   likelihood over the full host redshift range, which in the shallow regime rewards
   higher h. So the "unified numerator support" idea itself appears biased high here,
   independent of quadrature.

Both effects push H0 high → the observed collapse to h = 0.80.

## Conclusion & implication for the production kernel fix

- **The numerator-window unification is NOT the +0.013 shallow lever.** As specified
  (reuse `fixed_quad(n=50)` over the host window) it is numerically broken; and its
  exact form tilts high. Reject Part 1 as-designed.
- To even *evaluate* the physics of a shared numerator support one would need a
  **peak-aware / adaptive / much-higher-order** quadrature for the numerator over the
  wide window (the narrow GW peak must be resolved). That is a different, larger
  change than Part 1 scoped.
- The shallow +0.0132 attribution stands ([L8]: σ_z/z-at-low-z truncated-volume-kernel
  Eddington effect), but its **cure is not the numerator window.** Candidate B
  (photo-z-marginalized soft membership) and the distance-error coupling ([L7]) remain
  the open directions — now with the added constraint that any wide-window numerator
  integral must be quadrature-robust.

## Status of the code

`volume_trunc` is implemented behind an isolated `normalization_mode` (scalar +
batched kernels bit-identical; `volume_deconv`/`local_ratio` byte-identical — golden
regen was additions-only; full CPU suite 889 passed). It is retained as an
**experimental / FALSIFIED** mode (like `volume_global`) so this finding is
reproducible and a quadrature-robust reimplementation can build on the wiring. It is
NOT wired into the CLI and MUST NOT be used for production/campaign runs.
