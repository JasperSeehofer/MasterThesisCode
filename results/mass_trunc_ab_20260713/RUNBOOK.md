# EXP-45 RUNBOOK — mass_trunc production A/B on the seed600 shallow venue

**Written BEFORE the run** (pre-registration discipline; the volume_trunc lesson —
do not force a result). Fill the OUTCOME section only after the A/B completes.

## What is being tested

The 2D (with-BH-mass) channel's host-mass prior is replaced from the production
linear-Gaussian G2d moment match (`eddington_shifted_host_mass`) to the truncated
lognormal × R_eff prior on `[M_MIN, M_MAX]` (`normalization_mode="mass_trunc"`;
Gauss-Hermite numerator, Gauss-Legendre-in-lnM denominator). Motivation:
`results/mass_kernel_truncation_20260713/FINDINGS.md` — the toy showed the linear
kernel biases the 2D channel **HIGH** by +0.016…+0.02 at the real shallow-shell
photo-z leverage (σ_z/z ~ 0.5–0.65), a candidate for the venue's **2D +0.025**
residual and the info-monotonicity violation (2D bias > 1D bias).

Venue: archived seed600 shallow venue (494-event subsample; hosts z < 0.12,
injection pool z_max = 0.5) — the same rung `volume_trunc_ab` used, so the
`volume_deconv` baseline arm reproduces the established seed600 subsample means.
7-point grid `[0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86]`. Truth h = 0.73.

## Baseline (volume_deconv) reference numbers (from the N-5 / volume_trunc A/B)

- 1D: mean ≈ **0.745**, MAP ≈ 0.73
- 2D: mean ≈ **0.768**, MAP ≈ 0.73–0.76 (venue 2D residual +0.025 over full-venue 0.7546)

## Pre-registered predictions

**HARD CORRECTNESS GATE (must hold, else a bug):**
- **1D channel byte-identical.** `mass_trunc` shares the `volume_deconv` host-z
  kernel and touches ONLY the 4D mass term, so the 1D combined posterior must be
  **exactly unchanged**: `Δ1D_mean == 0` and `Δ1D_MAP == 0` (bit-level; the golden
  test already pins per-event 3D byte-equality). Any 1D drift ⇒ STOP, bug.

**PHYSICS HYPOTHESIS (what the A/B decides):**
- **CALIBRATED** (mass-kernel truncation IS a real driver of the 2D residual):
  `mass_trunc` LOWERS the 2D estimate toward truth — `Δ2D_mean ≤ −0.005`
  (production − correct is HIGH, so correct is lower), plausibly in the toy's
  `[−0.025, −0.005]` band; 2D MAP unchanged or one grid step lower. Direction is
  the load-bearing claim: **2D mean must move DOWN**.
- **BIASED / NULL** (truncation is more correct but NOT the +0.025 lever, or the
  toy over-stated it in isolation): `Δ2D_mean > −0.003` (negligible) OR moves UP
  (wrong sign). Then `mass_trunc` is retained as a more-correct kernel but is not
  the production explanation for the 2D residual — a real finding to REPORT, not to
  force (exactly as `volume_trunc` was reported falsified).

**Quadrature-robustness watch (the volume_trunc failure mode):** if BOTH arms'
posteriors collapse onto a grid edge or the 2D mean jumps implausibly (e.g. ≥ 0.05),
suspect the Gauss-Hermite/GL quadrature, not the physics — check that the baseline
arm still reproduces ≈0.745/0.768 first (it must, `volume_deconv` is byte-identical).

## Command

```
uv run python scripts/mass_trunc_ab.py \
    --crb_dir ~/data-backups/seed600_local_derail_20260702/crux_ws \
    --injections_dir ~/data-backups/seed600_local_derail_20260702/simulations/injections \
    --scratch_dir /tmp/mass_trunc_ab [--workers N]
```
Writes `.planning/gate/mass_trunc_ab.json`.

## OUTCOME (2026-07-13, run in 707 s; gate_result.json)

- baseline (volume_deconv): 1D mean=0.7450 MAP=0.73 | 2D mean=0.7681 MAP=0.76 edge=0.003
- mass_trunc:               1D mean=0.7450 MAP=0.73 | 2D mean=0.7710 MAP=0.76 edge=0.010
- Δ (mass_trunc − volume_deconv): 1D mean=+0.0000 MAP=0 | 2D mean=+0.0029 MAP=0
- 1D byte-identical? **YES** (correctness gate PASSED).
- **Verdict: BIASED-NULL.** Δ2D_mean = +0.0029 (> −0.003 AND wrong sign) → the
  mass-kernel truncation is EXONERATED as the 2D +0.025 residual driver. Baseline
  reproduced the reference exactly and no quadrature artefact (edge stays ~0.01, no
  rail), so this is the physics, not the numerics. The isolated numerator toy
  over-stated the effect; the selection denominator cancels it in the full pipeline.
  Full analysis: `FINDING.md`.
