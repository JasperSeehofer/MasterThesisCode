# RUNBOOK — pp_coverage Gray-mixture (`mixture_mode`) sweep, 2026-07-11

**Provenance:** EXP-41 / handoff item N-1 (`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`);
quick task `260711-07n-pp-coverage-gray-mixture`; code at `0f6f914` on
`physics/zero-host-completion-fallback` (adds `mixture_mode` gray/conditioned +
per-branch tilt diagnostics to `master_thesis_code/validation/pp_coverage.py`).
Baseline for the A/B comparison: the L-A two-branch sweep
`results/pp_coverage_deepvenue_20260710/` (grid reused VERBATIM).

**Purpose:** test whether the faithful Gray et al. (2020, arXiv:1908.06050,
Eqs. 29+32) mixture `(beta_G*L_cat_i + B_num)/D` for host-found events —
with the per-host selection denominator `D_g_i` of Eqs. A.9/A.10 (production
commit `713fbd1` analog) — restores calibration at 60–95% incompleteness,
where the clean two-branch limit was found BIASED HIGH (L-A verdict). This
adjudicates the N-1 fork: clean-limit artifact vs production-composition
defect. The `conditioned` contrast (N-2b) separates `w_G(h) = beta_G/D`
bookkeeping from the completion integral itself.

---

## A. Gray sweep grid — 8 cells

Grid: `z_support` in `{0.2, 0.3, 0.5, 1.0}` x `sigma_z` in `{0.015, 0.035}`,
`--mixture-mode gray`, `kernel=volume`, defaults otherwise
(`n_realizations=120`, `n_events=250`, `truths=[0.62, 0.72, 0.84]`,
`seed=20260701`).

`z_support=1.0` is `> Z_MAX_POP=0.95`, i.e. the **untruncated CONTROL** at
each `sigma_z`. NOTE: in gray mode the zs=1.0 control degenerates to the
per-host local-ratio estimator `N_i/D_g_i` (B_num window is empty and
`beta_G == D` cancels), NOT to the two-branch `N_i/D` control — it validates
the gray in-catalogue machinery in the complete-catalogue limit.

The 8 concrete commands (`ZS` in `{0.2,0.3,0.5,1.0}` x `SZ` in `{0.015,0.035}`):

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs{ZS}_sz{SZ}.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs{ZS}_sz{SZ}.log
```

Concretely:

```bash
# zs=0.2, sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 0.2 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs0.2_sz0.015.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs0.2_sz0.015.log

# zs=0.2, sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.2 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs0.2_sz0.035.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs0.2_sz0.035.log

# zs=0.3, sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 0.3 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs0.3_sz0.015.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs0.3_sz0.015.log

# zs=0.3, sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.3 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs0.3_sz0.035.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs0.3_sz0.035.log

# zs=0.5, sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 0.5 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs0.5_sz0.015.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs0.5_sz0.015.log

# zs=0.5, sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.5 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs0.5_sz0.035.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs0.5_sz0.035.log

# zs=1.0 (CONTROL, untruncated -> local-ratio N_i/D_g_i), sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 1.0 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs1.0_sz0.015.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs1.0_sz0.015.log

# zs=1.0 (CONTROL, untruncated -> local-ratio N_i/D_g_i), sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 1.0 \
  --mixture-mode gray --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_gray_zs1.0_sz0.035.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_gray_zs1.0_sz0.035.log
```

## B. Conditioned contrast — 4 deepest cells (N-2b)

`z_support` in `{0.2, 0.3}` x `sigma_z` in `{0.015, 0.035}` (the 4 deepest
cells of the 8-cell grid), `--mixture-mode conditioned`, all 3 truths.
(The plan text's "x sigma_z 0.035" would give only 2 cells; its own verify
gate requires 4 `pp_cond_*.json` files — the 4-deepest-cells reading is
authoritative and covers the sigma_z axis of the contrast.)

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
  --mixture-mode conditioned --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_graymix_20260711/pp_cond_zs{ZS}_sz{SZ}.json \
  2>&1 | tee results/pp_coverage_graymix_20260711/pp_cond_zs{ZS}_sz{SZ}.log
```

for `(ZS, SZ)` in `(0.2, 0.015), (0.2, 0.035), (0.3, 0.015), (0.3, 0.035)`.

## C. Runtime

Each 120x250x3 cell runs in ~5–6 s on the dev machine (32 cores; the harness
is single-process numpy) — 12 cells ~70 s total. No background
parallelization was needed (plan's 15-min threshold not approached).

## D. Anchor bit-identity check (two_branch no-op guarantee)

The committed sigma_z=0.10 anchor config (`n_realizations=250`,
`n_events=250`, `sigma_z=0.10`, `kernel=volume`, `seed=20260701`, no
`z_support`, default `mixture_mode=two_branch`) was re-run under the new code
and its `.results` diffed against
`results/pp_coverage_sigmaz_scan_20260703/pp_sigmaz0.10_volume.json`:
**PASS — every pre-existing key byte-identical**; the only differences are
the additive schema keys (`completion_fraction` from 260710-sjm, plus the
new `dlogL_dh_host_mean` / `dlogL_dh_completion_mean`). Also enforced in-tree
by `test_z_support_none_golden_pin` and the full-dict
`test_z_support_at_zmax_pop_matches_untruncated_limiting_case`.

---

## Pre-registered verdict criteria (from the plan, design pin #7)

- **CALIBRATED** <= `cov68` within `+/-0.085` of nominal 0.68 AND
  `|Delta map_mean vs truth| < 2*SEM` (`SEM = map_std/sqrt(120)`) across the
  truncated cells (`zs` in `{0.2, 0.3}`)
  => L-A bias is a clean-limit artifact; depth+fallback safe at the estimator
  level; EXP-40 becomes a confirmation; D1 can keep depth 1.5.
- **STILL BIASED** <= otherwise
  => production composition suspect at deep incompleteness; report which
  regime (which zs/sigma_z/truth cells fail); N-2 corners it.

N-2b contrast mapping: if `conditioned` calibrates where `gray` does not,
the defect is `w_G(h) = beta_G/D` bookkeeping, not the completion integral.

See `SUMMARY.md` in this directory for the verdict and tables.
