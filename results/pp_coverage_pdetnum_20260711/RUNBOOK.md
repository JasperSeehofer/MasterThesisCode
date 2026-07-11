# pp_coverage p_det-in-numerator probe — RUNBOOK (2026-07-11)

**Provenance:** floor-mechanism probe, continuation of the N-2 decomposition
(`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`); quick task
`260711-27m-pdet-in-numerator` (code at `0d08992` on
`physics/zero-host-completion-fallback`, executed inline). Follows quick tasks
260711-07n (gray/conditioned adjudicated STILL BIASED), 260711-117 (exact mode —
σ_z-dependent leak removed, σ_z-independent floor +0.002…+0.005 remains) and
260711-1ps (floor PERSISTENT under grid/quadrature refinement; prior-tilt lever
arm negligible).

## Hypothesis under test

The harness generative model decides detection on the TRUE z
(`_sample_detected_redshifts` draws z from `w_pop * p_det` BEFORE the
dL_obs/z_gal noise draws), so detection ⫫ data | z and the exact conditional is

    p(data, G | detected, h)
        = ∫ 1_G(z) p_GW(dL_obs|z,h) [N(z; z_gal, σ_z)] p_det(A(z)/h) w_pop(z) dz / D(h)

— with `p_det(A(z)/h)` INSIDE the numerator integrals. The Mandel–Farr–Gair
(2019, arXiv:1809.02063) no-p_det-inside form applies when detection is a
deterministic function of the OBSERVED data; for latent-thresholded detection
the factor stays inside. The completion branch integrates deep into the p_det
roll-off (D50 ≈ z 0.35 at h=0.72), where the missing factor overweights
undetectable volume with a positive h-tilt — σ_z-independent (no kernel) and
insensitive to a w_pop-only tilt (the p_det factor is h-dependent, unlike the
γ perturbation) — matching every measured property of the 260711-117/1ps floor.

## Pre-registered predictions (written BEFORE the runs)

(i) **exact + `--pdet-in-numerator`** ("full exact inverse") at the deep cells:
floor REMOVED — `|map_bias| < 2·SEM` AND cov68 within 0.68 ± 0.085 for the
0.62/0.72 truths (0.84 carries the grid-edge caveat), at BOTH σ_z 0.015/0.035.

(ii) **two_branch + flag** at the untruncated control (z_support=1.0,
σ_z=0.035): the mild control-level undercoverage (cov68 0.55–0.68 in the
exactmode/priortilt-era controls) moves TOWARD nominal — same missing factor,
small because p_det ≈ 1 over most kernels.

(iii) If (i) fails ⇒ the floor is NOT the latent-detection factor — report
honestly; the residual becomes the open item.

## Grid and commands

Volume kernel, n_realizations=120, n_events=250, seed=20260701, truths
[0.62, 0.72, 0.84], default h-grid [0.600, 0.860] step 0.004, n_z_quad=160.
Baselines (flag off) are NOT re-run — cited from
`results/pp_coverage_exactmode_20260711/` (exact) and
`results/pp_coverage_deepvenue_20260710/` + `results/pp_coverage_graymix_20260711/`
(two_branch/controls).

```bash
# (a) exact + flag, deep cells (4 runs)
for ZS in 0.2 0.3; do for SZ in 0.015 0.035; do
  uv run python -m master_thesis_code.validation.pp_coverage \
    --n-realizations 120 --n-events 250 --sigma-z $SZ --kernel volume \
    --z-support $ZS --mixture-mode exact --pdet-in-numerator \
    --output results/pp_coverage_pdetnum_20260711/pp_pdetnum_exact_zs${ZS}_sz${SZ}.json
done; done

# (b) two_branch + flag, untruncated/inert controls (2 runs)
for ZS in 0.5 1.0; do
  uv run python -m master_thesis_code.validation.pp_coverage \
    --n-realizations 120 --n-events 250 --sigma-z 0.035 --kernel volume \
    --z-support $ZS --mixture-mode two_branch --pdet-in-numerator \
    --output results/pp_coverage_pdetnum_20260711/pp_pdetnum_tb_zs${ZS}_sz0.035.json
done
```

Criteria as in the prior sweeps: cov68 band 0.68 ± 0.085 (±2·SE at n=120);
bias criterion |Δmap_mean vs truth| < 2·SEM, SEM = map_std/√120.
