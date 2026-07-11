# pp_coverage p_det-in-numerator probe — VERDICT (2026-07-11)

**Provenance:** quick task `260711-27m-pdet-in-numerator` (code `0d08992`, executed
inline on `physics/zero-host-completion-fallback`); RUNBOOK.md in this directory
(hypothesis + pre-registered predictions written before the runs). Follows
260711-07n (gray/conditioned STILL BIASED), 260711-117 (exact mode; σ_z-dependent
leak removed; σ_z-independent floor +0.002…+0.005 remains), 260711-1ps (floor
persistent under grid/quadrature refinement; prior-tilt lever arm negligible).

## VERDICT: hypothesis REFUTED — the persistent floor is NOT the latent-detection p_det-inside-numerator factor

Pre-registered prediction (i) **fails**: with `--pdet-in-numerator` the exact-mode
deep cells are statistically UNCHANGED (Δbias ≤ +0.0006, cov68 shifts within
binomial noise); the floor +0.0025…+0.0060 survives intact. Prediction (ii)
**fails in the opposite direction**: on the untruncated controls the factor
FLIPS the small negative control bias (−0.0030…−0.0018) to a small positive one
(+0.0028…+0.0063) and degrades cov68 at the 0.72/0.84 truths (0.675 → 0.550,
0.675 → 0.575). Pre-registered branch (iii) therefore applies: the floor remains
the open item, and the probe adds the sharp finding that the p_det-inside form —
although it is the formally exact conditional for this latent-thresholded
generative model — measures WORSE than the Mandel–Farr–Gair no-p_det-inside form
on the calibrated controls.

## Per-cell comparison (flag ON vs flag-off baselines; n=120, truths per row)

Exact mode, deep cells (baselines: `results/pp_coverage_exactmode_20260711/`):

| zs | σ_z | h_true | bias off | bias ON | Δ | cov68 off | cov68 ON | 2·SEM |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | +0.0023 | +0.0025 | +0.0002 | 0.692 | 0.708 | 0.0013 |
| 0.2 | 0.015 | 0.72 | +0.0034 | +0.0035 | +0.0001 | 0.550 | 0.550 | 0.0015 |
| 0.2 | 0.015 | 0.84 | +0.0042 | +0.0043 | +0.0001 | 0.608 | 0.608 | 0.0019 |
| 0.2 | 0.035 | 0.62 | +0.0026 | +0.0028 | +0.0002 | 0.708 | 0.692 | 0.0015 |
| 0.2 | 0.035 | 0.72 | +0.0046 | +0.0047 | +0.0001 | 0.575 | 0.575 | 0.0019 |
| 0.2 | 0.035 | 0.84 | +0.0042 | +0.0044 | +0.0002 | 0.517 | 0.525 | 0.0022 |
| 0.3 | 0.015 | 0.62 | +0.0003 | +0.0031 | +0.0028 | 0.700 | 0.625 | 0.0007 |
| 0.3 | 0.015 | 0.72 | +0.0024 | +0.0037 | +0.0013 | 0.625 | 0.583 | 0.0008 |
| 0.3 | 0.015 | 0.84 | +0.0047 | +0.0052 | +0.0005 | 0.525 | 0.508 | 0.0011 |
| 0.3 | 0.035 | 0.62 | −0.0010 | +0.0032 | +0.0042 | 0.708 | 0.700 | 0.0010 |
| 0.3 | 0.035 | 0.72 | +0.0023 | +0.0042 | +0.0019 | 0.633 | 0.533 | 0.0011 |
| 0.3 | 0.035 | 0.84 | +0.0054 | +0.0060 | +0.0006 | 0.483 | 0.408 | 0.0013 |

Two-branch untruncated/inert controls at σ_z=0.035 (baselines:
`results/pp_coverage_deepvenue_20260710/`):

| zs | h_true | bias off | bias ON | Δ | cov68 off | cov68 ON |
|---|---|---|---|---|---|---|
| 0.5 | 0.62 | −0.0030 | +0.0028 | +0.0058 | 0.758 | 0.775 |
| 0.5 | 0.72 | −0.0017 | +0.0044 | +0.0061 | 0.675 | 0.550 |
| 0.5 | 0.84 | −0.0010 | +0.0063 | +0.0073 | 0.692 | 0.450 |
| 1.0 | 0.62 | −0.0030 | +0.0028 | +0.0058 | 0.758 | 0.775 |
| 1.0 | 0.72 | −0.0018 | +0.0043 | +0.0061 | 0.675 | 0.550 |
| 1.0 | 0.84 | −0.0024 | +0.0042 | +0.0066 | 0.675 | 0.575 |

Pattern: the factor's net effect is a nearly uniform **+0.006 shift in
kernel-dominated (host-branch) events** and **≈ nothing in completion-dominated
events** — the opposite of what a completion-branch floor mechanism requires.

## Interpretation (for the ledger)

1. **Formal-vs-effective inverse:** for this generative model (detection decided
   on true z before the noise draws) the p_det-inside conditional is the
   mathematically exact inverse — yet it measures worse. The reconciliation is
   that the harness inference carries a second O(σ_f²) approximation: the GW
   likelihood is evaluated as `N(dL_obs; A(z)/h, σ_f·dL_obs)` with a constant,
   observed-distance σ, while the generative noise is `σ_f·dL_true` (z-dependent
   along the integral, with the accompanying 1/σ(z) normalization variation).
   Empirically the no-p_det + constant-σ combination nearly cancels on the
   controls (bias −0.002); inserting p_det alone breaks that cancellation
   (+0.004…+0.006). Magnitude check: σ_f² = 0.0025 → ~0.002–0.004 in h — the
   scale of both the floor and the flag shift.
2. **Sharpened floor candidate (open item, next session):** the
   σ(dL_obs)-vs-σ(dL_true) noise-model approximation — σ_z-independent ✓,
   prior-tilt-insensitive ✓, grid-insensitive ✓, O(σ_f²) scale ✓. A decisive
   probe needs the inference σ inside the integral (`σ_f·A(z)/h`, with the
   1/σ(z) prefactor), run with and without p_det-inside — 2×2 with the flag.
   Also worth a cheap n_events scaling check (does the floor scale as a skewed
   MAP-statistic artifact, given calibrated controls carry −0.002…−0.003 MAP
   offsets of the same magnitude and cov68 is largely in-band?).
3. **Practical weight:** at +0.003…+0.005 in h the floor is at/below the
   campaign per-seed σ_boot (~0.005) and an order of magnitude below the leak
   term it survived (up to +0.037 two-branch / +0.123 gray). For production
   the deep-incompleteness story stays: dominant mechanism = membership-support
   kernel leak (260711-117); the floor is a harness-level model-approximation
   residual until proven otherwise.
4. **Production correction candidates unchanged** (from 260711-117): soft
   (photo-z-marginalized) membership weighting of in-catalogue kernels —
   /physics-change + literature pass (Gray 2020; Chen–Fishbach–Holz 2018;
   Mastrogiovanni et al. ICAROGW). The latent-vs-data-thresholded distinction
   measured here is a REQUIRED input to that pass: production also thresholds
   SNR on the noiseless injected waveform (latent-thresholded class), and this
   probe shows the naive "add p_det inside" move can degrade calibration when
   other O(σ²) approximations are present. Do not cargo-cult it.

## Carried caveats

1. 1D-channel only; single effective host per event; hard truncation (vs
   production's soft M_BH-prune) — as in the three predecessor SUMMARYs.
2. The flag's +0.006 control shift is measured at σ_z=0.035/σ_f=0.05 for this
   venue (D50=1.85 Gpc); it will scale with how much of the kernel/GW support
   sits on the p_det roll-off.
