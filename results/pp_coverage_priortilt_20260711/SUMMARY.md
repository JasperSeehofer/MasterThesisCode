# pp_coverage prior-tilt ladder (N-3) + residual-floor discriminator — VERDICT (2026-07-11)

**Provenance:** quick task `260711-1ps-prior-sensitivity`; handoff item **N-3**
(prior-sensitivity probe, feeds decision D1) + the **residual-floor
discriminator** for the σ_z-independent +0.002…+0.005 completion-branch floor
isolated by 260711-117 (`results/pp_coverage_exactmode_20260711/SUMMARY.md`);
`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`. Code at `e5b8383` on
`physics/zero-host-completion-fallback` (`inference_wpop_tilt` γ knob —
inference-side w_pop × exp(γ·z), strict γ==0.0 gate, generative truth draw
never tilted — plus `--h-step`). RUNBOOK.md in this directory (grid, commands,
BOTH pre-registered predictions — committed at `c78c2f5` BEFORE any run,
followed as recorded). γ=0 baselines cited, not re-run:
`results/pp_coverage_deepvenue_20260710/pp_zs0.2_sz0.035_volume.json`
(two_branch) and
`results/pp_coverage_exactmode_20260711/pp_exact_zs0.2_sz0.035.json` (exact) —
identical grid/seed/realizations (120 × 250 × truths {0.62, 0.72, 0.84},
seed 20260701, h_step 0.004, n_z_quad 160, zs 0.2, σ_z 0.035).

**Anti-repetition:** gray/conditioned (adjudicated STILL BIASED, 260711-07n)
and the σ_z-dependent kernel-support leak (adjudicated removed, 260711-117)
are NOT re-litigated here.

## VERDICT

1. **N-3 lever arm: prior sensitivity is REAL, monotone, ~linear — and
   NEGLIGIBLE in magnitude.** d(map_mean)/dγ = +0.0001…+0.0017 across all
   truths and both modes. The headline D1 number, **Δh(γ_10%) ≤ +0.0004 in h
   (≤ +0.05% of truth; exact mode ≤ +0.015%)**, is 10–100× SMALLER than the
   completion-branch floor it was designed to probe (+0.0026…+0.0046).
   Pre-registered prediction (i) held on direction/linearity but FAILED on its
   headline expectation: the deep completion-dominated regime is **NOT
   meaningfully population-prior-driven** under exp(γ·z)-family
   misspecification. Structural reason: the tilt enters every inference factor
   in RATIO form (B_num/D shares the tilted w_pop measure; the volume kernel
   is renormalized per event by Z_i), so a smooth multiplicative prior error
   largely cancels. The σ_z-independent floor CANNOT be attributed to
   w_pop-shape misspecification of this family.
2. **Floor discriminator: PERSISTENT — a genuine composition residual, not a
   grid/quadrature artifact.** The exact-mode γ=0 floor at the primary truths
   moves by ≤ 0.0002 (0.62: +0.0026→+0.0027; 0.72: +0.0046→+0.0044) under a
   4× finer H0 grid (h_step 0.004→0.001) and a 2× finer per-event z-quadrature
   (n_z_quad 160→320), never shrinking toward 0, and stays significant against
   2·SEM (0.0015–0.0019). Pre-registered outcome (persistent) applies: the
   floor must be quantified against the campaign SEM before any depth-1.5 +
   fallback closure claim.

## 1. Lever arm d(map_mean)/dγ per truth per mode (5-point ladder incl. γ=0 baseline)

OLS slope over γ ∈ {−0.2, −0.1, 0, +0.1, +0.2}; central differences shown for
the linearity check. comp_frac column shows the completion-fraction dependence.

| mode | h_true | comp_frac | d(map_mean)/dγ (OLS) | central (γ±0.2) | central (γ±0.1) | per-0.1-segment slopes |
|---|---|---|---|---|---|---|
| two_branch | 0.62 | 0.709 | +0.00030 | +0.00033 | +0.00017 | +0.0003, +0.0000, +0.0003, +0.0007 |
| two_branch | 0.72 | 0.787 | +0.00170 | +0.00158 | +0.00217 | +0.0010, +0.0013, +0.0030, +0.0010 |
| two_branch | 0.84 | 0.848 | +0.00047 | +0.00050 | +0.00033 | +0.0007, +0.0003, +0.0003, +0.0007 |
| exact | 0.62 | 0.709 | +0.00010 | +0.00008 | +0.00017 | +0.0000, +0.0000, +0.0003, +0.0000 |
| exact | 0.72 | 0.787 | +0.00023 | +0.00025 | +0.00017 | +0.0007, +0.0000, +0.0003, +0.0000 |
| exact | 0.84 | 0.848 | +0.00070 | +0.00075 | +0.00050 | +0.0013, +0.0000, +0.0010, +0.0007 |

- Direction: map_mean INCREASES with γ everywhere (more prior weight at high z
  ⇒ higher h preferred) — matches the direction measured at implementation
  time in the monotonicity test.
- comp_frac dependence: no clean monotone trend in comp_frac — the largest
  two_branch arm sits at the middle truth (0.72), and the two_branch 0.84 cell
  is rail-dominated (rail 0.90–0.94, map_mean pinned near the 0.86 grid edge),
  so its arm is compressed and should be read as secondary.
- Resolution note (honesty): map_mean over 120 realizations moves in quanta of
  h_step/120 ≈ 3.3e-5; the ensemble mean is an unbiased estimator of the
  continuous shift (realization peaks are ~uniform relative to grid nodes), and
  an independent h_step=0.001 measurement (the committed
  `test_tilt_monotonic_map_mean` config) confirms strict per-node monotonicity
  at γ = ±0.1. The tiny arms are real measurements, not dead grid.

## 2. Headline D1 number: Δh for a ±10%-across-completion-domain prior misspecification

γ_10% = ln(1.1)/(0.95 − 0.2) = **0.12708** (an exp(γ·z) tilt that accumulates
to a 10% prior error across the completion domain [0.2, 0.95]). Linear
interpolation between the γ=+0.1 and γ=+0.2 ladder rungs:

| mode | h_true | Δh(γ=+0.1) | Δh(γ=+0.2) | **Δh(γ_10%)** | % of h_true |
|---|---|---|---|---|---|
| two_branch | 0.62 | +0.00003 | +0.00010 | **+0.00005** | +0.008% |
| two_branch | 0.72 | +0.00030 | +0.00040 | **+0.00033** | +0.045% |
| two_branch | 0.84 | +0.00003 | +0.00010 | **+0.00005** | +0.006% |
| exact | 0.62 | +0.00003 | +0.00003 | **+0.00003** | +0.005% |
| exact | 0.72 | +0.00003 | +0.00003 | **+0.00003** | +0.005% |
| exact | 0.84 | +0.00010 | +0.00017 | **+0.00012** | +0.014% |

**The honest "how population-prior-driven is the deep regime" number for D1:
a 10%-across-domain w_pop misspecification moves the MAP by at most +0.0004 in
h (+0.05% of truth) in the leak-carrying two_branch composition and at most
+0.0001 (+0.015%) in the exact composition** — far below the +0.002…+0.005
floor, below 2·SEM everywhere, and ~2 orders below the deep-venue truncation
bias (+0.014…+0.039) the campaign actually faces.

## 3. Composition sensitivity: two_branch vs exact

Tilted two_branch DOES respond more strongly than tilted exact (OLS arms
+0.0003/+0.0017/+0.0005 vs +0.0001/+0.0002/+0.0007; at the interior 0.72 truth
the contrast is ~7×): the σ_z kernel-support leak that two_branch still
carries amplifies prior sensitivity — the un-truncated host numerator's
above-edge mass is NOT renormalized against the same tilted measure, so the
tilt cancels less completely. The exact composition, whose branches tile
[0, Z_MAX_POP] exactly, is the more prior-robust of the two. But both arms are
negligible in absolute terms, so composition matters far less than the leak
itself (adjudicated in 260711-117); this contrast separates composition
sensitivity from pure prior sensitivity without changing any conclusion.

## 4. Floor verdict: PERSISTENT (not a grid/quadrature artifact)

Exact mode, γ=0, zs=0.2, σ_z=0.035. Baseline row from the cited 260711-117
JSON; SEM = map_std/√120. PRIMARY readout = 0.62 and 0.72 truths (0.84 sits
near the 0.86 grid edge — secondary).

| grid | h_true | map_mean | map_bias | 2·SEM | cov68 | rail |
|---|---|---|---|---|---|---|
| h_step=0.004, n_z_quad=160 (baseline) | 0.62 | 0.622633 | +0.0026 | 0.0015 | 0.708 | 0.000 |
| h_step=0.002, n_z_quad=160 | 0.62 | 0.622683 | +0.0027 | 0.0015 | 0.692 | 0.000 |
| h_step=0.001, n_z_quad=160 | 0.62 | 0.622683 | +0.0027 | 0.0015 | 0.675 | 0.000 |
| h_step=0.004, n_z_quad=320 | 0.62 | 0.622600 | +0.0026 | 0.0015 | 0.700 | 0.000 |
| h_step=0.004, n_z_quad=160 (baseline) | 0.72 | 0.724567 | +0.0046 | 0.0019 | 0.575 | 0.000 |
| h_step=0.002, n_z_quad=160 | 0.72 | 0.724400 | +0.0044 | 0.0019 | 0.583 | 0.000 |
| h_step=0.001, n_z_quad=160 | 0.72 | 0.724400 | +0.0044 | 0.0019 | 0.592 | 0.000 |
| h_step=0.004, n_z_quad=320 | 0.72 | 0.724367 | +0.0044 | 0.0019 | 0.575 | 0.000 |
| h_step=0.004, n_z_quad=160 (baseline) | 0.84 | 0.844233 | +0.0042 | 0.0022 | 0.517 | 0.192 |
| h_step=0.002, n_z_quad=160 | 0.84 | 0.844283 | +0.0043 | 0.0022 | 0.542 | 0.183 |
| h_step=0.001, n_z_quad=160 | 0.84 | 0.844350 | +0.0044 | 0.0022 | 0.567 | 0.183 |
| h_step=0.004, n_z_quad=320 | 0.84 | 0.843900 | +0.0039 | 0.0023 | 0.533 | 0.192 |

- The floor does NOT shrink toward 0: every variation is ≤ 0.0002 in |Δbias|
  (primary truths ≤ 0.0002; secondary 0.84 ≤ 0.0003), an order below the floor
  itself and far inside 2·SEM — while the floor stays SIGNIFICANT against
  2·SEM at both primary truths in every grid (+0.0026 vs 0.0015; +0.0044 vs
  0.0019). ⇒ **persistent composition residual.**
- 2·SEM caveat (pre-registered): ±0.002-scale conclusions live at the SEM
  boundary; the 0.62-truth floor (+0.0026 vs 2·SEM 0.0015) is barely 1.7σ-of-
  the-mean above zero at 120 realizations. The grid-to-grid variations share
  the same seed/realizations, so their stability is a same-noise comparison
  (stronger than independent 2·SEM would suggest), but the ABSOLUTE floor size
  at 0.62 needs more realizations for a sharper significance claim.

## 5. Decision mapping (D1 — depth-1.5 + fallback; NOT re-deciding, user's call)

Per the handoff outcome→decision map:

- **The prior-sensitivity escape hatch for the floor is CLOSED:** the
  σ_z-independent completion-branch floor is not explained by w_pop
  misspecification of the exp(γ·z) family, and — the D1-relevant half — a
  plausibly-sized population-prior error does NOT destabilize the deep regime
  (≤ +0.05% of truth at a 10%-across-domain tilt). Any statistical-siren
  framing of the deep venue does NOT inherit a first-order population-prior
  systematic from this family; the floor itself (+0.3…+0.6% of truth at 71–85%
  completion fraction) is now the binding residual, and it is PERSISTENT.
- **Supports the 260711-117 production-correction candidate:** the exact
  (membership-truncated) composition is both the least biased AND the most
  prior-robust — the floor is its only remaining defect and is grid-converged,
  so an estimator-level correction route (soft membership-truncated kernels,
  /physics-change + literature) is not blocked by prior sensitivity.
- **What the floor now needs** (if depth-1.5 + fallback is pursued): a
  quantification against the campaign SEM (at campaign event counts the
  +0.003…+0.005 floor may or may not be resolvable) and/or a mechanism probe
  that is NOT prior-shape (e.g. the finite ±5σ GW window of B_num, or the
  host/completion counterweight asymmetry flagged by the tilt diagnostics in
  260711-117). Truncation (option b) remains the robustness bound.

## Pre-registered prediction evaluation (honesty ledger)

- **(i) Lever arm:** direction and ~linearity CONFIRMED (monotone ascending in
  γ, segment slopes consistent within grid quanta); the implicit magnitude
  expectation ("the deep regime is population-prior-driven") REFUTED — the
  measured arm is negligible. Reported as found.
- **(ii) Floor:** PERSISTENT outcome applies (pre-stated); grid/quadrature
  artifact ruled out.

## Carried caveats (verbatim)

1. **1D-channel only** — the 2D (+0.025 remaining) question is NOT covered by
   this harness.
2. **Single-host clean limit** — production host-found events carry the full
   in-catalogue galaxy sum; this harness's exact mode truncates a single
   effective host kernel.
3. **Hard truncation** (`z_support` step) vs production's soft M_BH-prune
   truncation of the effective catalogue — and (from N-2d) hard truncation
   under observed-z membership is itself misspecified; production analogs need
   the soft form.

Plus one new caveat: the tilt family is exp(γ·z) — smooth, monotone,
sign-definite in d/dz. Prior errors OUTSIDE this family (e.g. non-monotone
merger-rate evolution, sharp features) are not bounded by this probe.
