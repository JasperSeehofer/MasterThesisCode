---
phase: quick-260711-1ps
plan: 01
subsystem: validation
tags: [pp-coverage, prior-sensitivity, N-3, floor-discriminator, dark-siren, H0]
requires:
  - quick-260711-117 (exact mixture mode, --n-z-quad, sigma_z-independent floor isolation)
  - results/pp_coverage_deepvenue_20260710 (two_branch gamma=0 baseline)
provides:
  - inference_wpop_tilt (gamma) knob in pp_coverage harness (inference-only exp(gamma*z) w_pop tilt)
  - --inference-wpop-tilt and --h-step CLI flags
  - N-3 lever-arm measurement + D1 headline Dh(gamma_10%)
  - floor discriminator verdict (persistent vs artifact)
affects:
  - decision D1 (issue #30 depth-vs-truncation, user's call)
  - production-correction candidate (membership-truncated kernel route)
tech-stack:
  added: []
  patterns: [pre-registered RUNBOOK before runs, strict ==0.0 default gate + golden-pin guard]
key-files:
  created:
    - results/pp_coverage_priortilt_20260711/RUNBOOK.md
    - results/pp_coverage_priortilt_20260711/SUMMARY.md
    - results/pp_coverage_priortilt_20260711/ (8 tilt + 3 floor JSONs, 11 logs untracked per *.log gitignore)
  modified:
    - master_thesis_code/validation/pp_coverage.py
    - master_thesis_code_test/validation/test_pp_coverage.py
decisions:
  - "Tilt gate is strict (tilt == 0.0 returns the untilted weight object) so the default path is bit-identical; guarded by unmodified golden pins"
  - "Monotonicity test runs at h_step=0.001: the default 0.004 grid quantizes the tiny gamma=+-0.1 MAP shift to exact ties (measured); deterministic harness => stable"
  - "Logs left untracked (project-wide *.log gitignore) — identical convention to the deepvenue/exactmode results dirs"
metrics:
  duration: "~12 min"
  completed: "2026-07-11"
  tasks: 2
  commits: 3
---

# Quick Task 260711-1ps: Prior-Sensitivity Probe (N-3) + Floor Discriminator Summary

**Inference-side w_pop tilt knob (gamma) + --h-step added to the G4b harness; 11-run sweep shows the deep completion-dominated regime is nearly INSENSITIVE to exp(gamma*z) prior misspecification (D1 headline Dh(gamma_10%) <= +0.0004 in h, <= +0.05% of truth) and the sigma_z-independent +0.0026...+0.0046 exact-mode floor is PERSISTENT (grid/quadrature artifact ruled out).**

## What was done

- **Task 1 (`e5b8383`, TDD):** `_inference_population_weight(z, tilt)` = w_pop(z)·exp(tilt·z) with a strict `tilt == 0.0` early-return (bit-identical default; all existing golden pins pass UNMODIFIED); threaded through all four inference-side call sites (host volume kernel, B_num, D(h), beta_G) — the generative truth draw `_sample_detected_redshifts` is never tilted; `--inference-wpop-tilt` + `--h-step` CLI flags; 5 new tests (RED verified before implementation: bit-identity + golden-pin guard, inequality, determinism, --h-step round-trip/grid-size, strict monotonicity with direction measured = ascending).
- **Task 2 (`c78c2f5` pre-registration, `724fc29` results):** RUNBOOK with both predictions committed BEFORE any run; 8-run tilt ladder (gamma ∈ {−0.2,−0.1,+0.1,+0.2} × {two_branch, exact}, gamma=0 anchored by the cited committed baselines) + 3-run floor discriminator (h_step 0.002/0.001, n_z_quad 320); SUMMARY verdict at `results/pp_coverage_priortilt_20260711/SUMMARY.md`.

## Key results

1. **Lever arm (N-3):** d(map_mean)/dgamma = +0.0001…+0.0017 (real, monotone ascending, ~linear) — two_branch 0.62/0.72/0.84: +0.0003/+0.0017/+0.0005; exact: +0.0001/+0.0002/+0.0007 (comp_frac 0.709/0.787/0.848).
2. **D1 headline:** Dh(gamma_10% = ln(1.1)/0.75 = 0.127) = +0.00003…+0.00033 absolute (+0.005…+0.045% of h_true) — 10–100× below the floor, below 2·SEM everywhere. The prior-sensitivity escape hatch for the floor is CLOSED; pre-registered magnitude expectation ("deep regime is population-prior-driven") honestly REFUTED (ratio structure self-cancels the tilt).
3. **Composition:** leak-carrying two_branch is ~7× more prior-sensitive than exact at the interior 0.72 truth; both negligible; exact is the most prior-robust composition.
4. **Floor verdict: PERSISTENT.** Exact gamma=0 floor at primary truths (+0.0026 at 0.62, +0.0046 at 0.72) moves ≤ 0.0002 under h_step 0.004→0.002→0.001 and n_z_quad 160→320; stays significant vs 2·SEM (0.0015/0.0019). Genuine composition residual — quantify against campaign SEM before any depth-1.5+fallback closure claim.

## Deviations from Plan

**1. [Rule 1 - Bug] Monotonicity test grid resolution**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** On the plan's tiny config at default h_step=0.004, map_mean at gamma ∈ {−0.1, 0, +0.1} quantizes to exact ties (the true shift is ~1e-4-scale) — the strict-monotonicity assertion cannot resolve it.
- **Fix:** Test config uses h_step=0.001 (documented in the test docstring); direction measured (ascending), not assumed, per plan intent.
- **Files modified:** master_thesis_code_test/validation/test_pp_coverage.py
- **Commit:** e5b8383

No other deviations — plan executed as written (logs untracked follows the pre-existing project *.log gitignore and prior results-dir convention).

## Verification

- Full fast suite green twice (854 passed, 6 skipped), golden pins UNMODIFIED; ruff + mypy clean; both flags in --help.
- Task-2 automated check passed: 11 JSONs + RUNBOOK + SUMMARY grep.

## Commits

- `e5b8383` feat(260711-1ps): inference-side w_pop prior-tilt knob (gamma) + --h-step CLI flag
- `c78c2f5` results(260711-1ps): pre-register prior-tilt ladder + floor-discriminator RUNBOOK (before runs)
- `724fc29` results(260711-1ps): prior-tilt ladder + floor discriminator — lever arm NEGLIGIBLE, floor PERSISTENT

## Self-Check: PASSED
