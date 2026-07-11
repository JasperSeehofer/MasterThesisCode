# RUNBOOK — pp_coverage σ(dL_obs)-vs-σ(dL_true) noise-model floor probe, 2026-07-11

**Provenance:** quick task `260711-hx1-floor-noise-model`; floor decomposition
[L7] item (d) (`.planning/BIAS-INVESTIGATION-20260710.md`); the sharpened
candidate from `results/pp_coverage_pdetnum_20260711/SUMMARY.md` §2. Code adds
`sigma_dl_model_in_likelihood` (config + `--sigma-model-in-likelihood`) to
`master_thesis_code/validation/pp_coverage.py`. Baselines reused VERBATIM (same
grid/seed/realizations): const-σ / p_det-off = `results/pp_coverage_exactmode_20260711/`
+ `results/pp_coverage_deepvenue_20260710/`; const-σ / p_det-on =
`results/pp_coverage_pdetnum_20260711/`.

**Purpose.** The remaining σ_z-independent floor (+0.002…+0.005 in the exact deep
cells; −0.002…−0.003 offset on the inert controls) is hypothesised to be the
inference noise-model approximation: the GW likelihood is evaluated with a
CONSTANT observed-distance σ = σ_f·dL_obs, while the generative noise was
σ = σ_f·dL_true (z-dependent inside the integral, with its 1/σ(z) normalization).
The new `model-σ` path evaluates the likelihood as `N(dL_obs; A(z)/h, σ_f·A(z)/h)`
— the z-dependent true/model-distance σ with the accompanying 1/σ(z) prefactor
(carried automatically by `_norm_pdf`). With `--pdet-in-numerator` ON, model-σ is
the FULLY-CONSISTENT exact conditional for this latent-thresholded generative
model (the 27m probe showed p_det-inside alone breaks the accidental cancellation
of the const-σ approximation; here we remove the approximation it was cancelling).

**Anti-repetition (ledger, do NOT re-litigate):** gray/conditioned mixture (07n,
STILL BIASED), prior tilt (1ps, NEGLIGIBLE lever arm), p_det-inside ALONE (27m,
REFUTED as the floor). This probe tests a DIFFERENT factor (the inference σ model),
2×2 with the p_det flag; the const-σ columns are the committed 07n/117/27m JSONs.

**Harness-only, no /physics-change.** Production soft-f(z)-kernel correction stays
user-gated.

---

## Pre-registered predictions (written BEFORE any run — falsifiable per branch)

**Hypothesis H_σ:** the σ_z-independent floor IS the σ(dL_obs)-vs-σ(dL_true)
noise-model approximation.

Criteria: cov68 within ±0.085 of 0.68; SEM = map_std/√120; deep cells = the 12
truncated exact rows zs∈{0.2,0.3}×σ_z∈{0.015,0.035}×3 truths.

- **P1 — CALIBRATED ⇒ H_σ CONFIRMED.** model-σ collapses the deep-cell floor toward
  zero: **|map_bias| < 2·SEM on ≥ 7/12 deep cells** (vs 1/12 for const-σ exact), AND
  the inert controls' const-σ −0.002…−0.003 offset moves toward 0 (|Δ| ≥ 0.0015
  toward zero). The **model-σ + p_det-inside** cell (fully-consistent exact
  conditional) is the closest-to-unbiased of the four 2×2 combinations. Continuous
  check: net tilt at h_true (dlogL_dh_host + dlogL_dh_completion) shrinks toward 0
  vs the const-σ baseline.

- **P2 — REFUTED ⇒ H_σ FALSE.** model-σ leaves the deep floor statistically intact
  (**Δ|bias| ≤ SEM** on the deep cells) ⇒ the residual is NOT the σ-model
  approximation. Report which cells move and by how much; hand off to P3.

- **P3 — n_events scaling (orthogonal to P1/P2).** On the representative deep cell
  zs=0.3/σ_z=0.035, run n_events∈{250,1000,4000} for const-σ AND model-σ.
  Pre-registered reading: a residual that is an **asymptotic bias stays FLAT in n**;
  a residual that is a **finite-sample MAP-estimator skew shrinks ∝ 1/√n** (≈ halve
  250→1000, quarter 250→4000). This adjudicates "real small bias" vs "estimator
  artifact" regardless of P1/P2. (The controls already carry a −0.002…−0.003 MAP
  offset at nominal cov68 — the skew signature.)

- **Fine-grid confirm (debrief lesson #2):** the coarse h-grid (h_step=0.004)
  quantizes; re-run the key deep cell at `--h-step 0.001` for const-σ and model-σ so
  the reported Δbias is not a grid-quantization artifact. Also read the continuous
  net-tilt diagnostic (grid-step-independent) as the primary signal.

---

## Common settings

All runs (unless noted): `--kernel volume --n-realizations 120 --n-events 250
--truths 0.62 0.72 0.84 --seed 20260701` (117/27m conventions). Output dir:
`results/pp_coverage_noisemodel_20260711/`.

## Set (a) — 2×2 NEW model-σ columns, exact deep cells + controls

Grid `ZS ∈ {0.2, 0.3, 0.5, 1.0} × SZ ∈ {0.015, 0.035}`, exact mode, for each of
{model-σ, model-σ + p_det-inside}. `zs∈{0.5,1.0}` = inert controls.

```bash
# model-σ, p_det OFF
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
  --mixture-mode exact --sigma-model-in-likelihood \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_noisemodel_20260711/pp_modelsig_exact_zs{ZS}_sz{SZ}.json

# model-σ, p_det ON (fully-consistent exact conditional)
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
  --mixture-mode exact --sigma-model-in-likelihood --pdet-in-numerator \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_noisemodel_20260711/pp_modelsig_pdet_exact_zs{ZS}_sz{SZ}.json
```

(16 JSONs: 8 cells × 2 variants.)

## Set (b) — n_events scaling (zs=0.3, σ_z=0.035)

`N ∈ {250, 1000, 4000}` for const-σ (baseline) AND model-σ (n=250 const-σ is the
committed exactmode cell; run the other five):

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events {N} --sigma-z 0.035 --z-support 0.3 \
  --mixture-mode exact [--sigma-model-in-likelihood] \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_noisemodel_20260711/pp_nscale_{constsig|modelsig}_n{N}.json
```

## Set (c) — fine-grid confirm (zs=0.3, σ_z=0.035, h_step=0.001)

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.3 \
  --mixture-mode exact [--sigma-model-in-likelihood] --h-step 0.001 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_noisemodel_20260711/pp_finegrid_{constsig|modelsig}.json
```

## Regression guard

`--mixture-mode exact` WITHOUT `--sigma-model-in-likelihood` must reproduce the
committed `results/pp_coverage_exactmode_20260711/pp_exact_zs0.3_sz0.035.json`
byte-for-byte (default-off bit-identity).
