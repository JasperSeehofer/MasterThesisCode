# RUNBOOK — pp_coverage exact membership-truncated-kernel sweep, 2026-07-11

**Provenance:** EXP-41-exact / handoff items N-2c + N-2d
(`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`); quick task
`260711-117-pp-coverage-exact-kernel`; code at `6a3c8ab` on
`physics/zero-host-completion-fallback` (adds `mixture_mode="exact"` +
`--n-z-quad` to `master_thesis_code/validation/pp_coverage.py`).
Baselines for the A/B comparisons: the L-A two-branch sweep
`results/pp_coverage_deepvenue_20260710/` and the EXP-41 gray/conditioned
sweep `results/pp_coverage_graymix_20260711/` (grids reused VERBATIM).

**Purpose:** the exact mode is the last untested composition. Under the
harness generative model (Mandel, Farr & Gair 2019, arXiv:1809.02063)
detection is conditioned ONCE via `1/D(h)` with NO p_det inside the
numerator, and catalogue membership `G = 1[z_true < z_support]` is part of
the observed data — so the exact host-event likelihood is the volume-kernel
numerator TRUNCATED at the catalogue support edge `z_support`, removing the
above-edge kernel leak every prior mode carried. Zero-host events keep
`B_num(h)/D(h)` (Gray et al. 2020, arXiv:1908.06050, Eqs. 29+32, the
completion mixture whose support the two branches tile exactly). This
adjudicates the N-2 mechanism decomposition: is the deep-incompleteness
high bias a membership-support LEAK in the host numerator (removed by exact
truncation) or a deeper composition defect (persists)?

**Anti-repetition (ledger):** gray and conditioned were adjudicated STILL
BIASED in quick task 260711-07n (`results/pp_coverage_graymix_20260711/SUMMARY.md`,
12/12 truncated cells fail both criteria; gray WORSE than the clean limit).
They are NOT re-litigated here — this sweep reuses their committed JSONs for
the side-by-side deltas only.

**Count reconciliation:** design pin #4a says "12 JSONs" but its own naming
pattern `pp_exact_zs{ZS}_sz{SZ}.json` over `ZS ∈ {0.2,0.3,0.5,1.0} × SZ ∈
{0.015,0.035}` yields 8 files. The "12" is the 12 truncated cell×truth
VERDICT ROWS (4 truncated cells × 3 truths), mirroring graymix's "12/12
cells" language — NOT the JSON count. Set (a) = 8 JSONs, set (b) = 8,
set (c) = 8 ⇒ **24 JSONs total** (~3 min at ~6 s/cell; no parallelization).

---

## Pre-registered prediction (written BEFORE any run)

> exact mode is CALIBRATED at all completion fractions (cov68 within ±0.085 of
> 0.68 AND |map_bias| < 2·SEM, SEM = map_std/√120, across the truncated cells
> zs ∈ {0.2, 0.3}) — because the only difference vs the two-branch clean limit
> is removal of the spurious above-edge kernel mass, the last remaining
> discrepancy from the exact inverse. CALIBRATED ⇒ mechanism IDENTIFIED
> (membership-support leak in the host-event numerator); production-correction
> candidate = f(z)-weighted in-catalogue kernel integrands → /physics-change +
> literature (Gray 2020; Chen–Fishbach–Holz 2018; Mastrogiovanni/ICAROGW),
> NOT this task. NOT CALIBRATED ⇒ mechanism deeper than membership bookkeeping;
> report which cells fail and how.

---

## Common settings

All runs: `--kernel volume --n-realizations 120 --n-events 250
--truths 0.62 0.72 0.84 --seed 20260701` (graymix conventions).

## Set (a) — exact 8-cell sweep

Grid: `ZS ∈ {0.2, 0.3, 0.5, 1.0} × SZ ∈ {0.015, 0.035}`. `zs ∈ {0.5, 1.0}`
are the untruncated/near-empty CONTROLS (completion_fraction ≈ 0; at
zs=1.0 > Z_MAX_POP=0.95 the truncation clamp `z_hi → min(z_hi, 1.0)` is
inert above the population ceiling, so exact degenerates to the two-branch
control). Per cell:

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
  --mixture-mode exact --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_exactmode_20260711/pp_exact_zs{ZS}_sz{SZ}.json \
  2>&1 | tee results/pp_coverage_exactmode_20260711/pp_exact_zs{ZS}_sz{SZ}.log
```

(8 JSONs: `pp_exact_zs0.2_sz0.015.json`, `pp_exact_zs0.2_sz0.035.json`,
`pp_exact_zs0.3_sz0.015.json`, `pp_exact_zs0.3_sz0.035.json`,
`pp_exact_zs0.5_sz0.015.json`, `pp_exact_zs0.5_sz0.035.json`,
`pp_exact_zs1.0_sz0.015.json`, `pp_exact_zs1.0_sz0.035.json`.)

## Set (b) — N-2c σ_z ladder (zs = 0.2, modes two_branch AND exact)

`σ_z ∈ {0.005, 0.015, 0.035}` at default `n_z_quad=160`, plus `σ_z = 0.002`
with `--n-z-quad 480`. At σ_z=0.002 the default 160-point window
under-samples the host-z Gaussian; 480 restores ≳4 quadrature points per σ
over the truncated support. σ_z=0 is NOT runnable (divide-by-zero in the
Gaussian kernel), so σ_z=0.002 probes the σ_z→0 limit. This answers whether
the deep-venue bias vanishes as σ_z→0 (kernel-leak signature) or persists
(composition signature).

```bash
# sigma_z = 0.002 cells add --n-z-quad 480; drop it for 0.005/0.015/0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.002 --z-support 0.2 --n-z-quad 480 \
  --mixture-mode {MODE} --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_exactmode_20260711/pp_ladder_{MODE}_sz0.002.json \
  2>&1 | tee results/pp_coverage_exactmode_20260711/pp_ladder_{MODE}_sz0.002.log
```

(8 JSONs: `MODE ∈ {two_branch, exact} × SZ ∈ {0.002, 0.005, 0.015, 0.035}`.
Sanity cross-check: the two_branch sz=0.015/0.035 ladder cells must
reproduce the L-A deep-venue zs=0.2 values in
`results/pp_coverage_deepvenue_20260710/pp_zs0.2_sz{SZ}_volume.json` —
same config, same seed.)

## Set (c) — N-2d observed-membership probe (modes gray AND exact)

The 4 deepest cells `zs ∈ {0.2, 0.3} × σ_z ∈ {0.015, 0.035}`, with
`--membership-on-observed` (membership decided on the observed `z_gal`
instead of the true `z_host` — production's BallTree sees measured
redshifts):

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
  --mixture-mode {MODE} --membership-on-observed \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_exactmode_20260711/pp_obsmem_{MODE}_zs{ZS}_sz{SZ}.json \
  2>&1 | tee results/pp_coverage_exactmode_20260711/pp_obsmem_{MODE}_zs{ZS}_sz{SZ}.log
```

(8 JSONs: `MODE ∈ {gray, exact} × (ZS, SZ) ∈ {(0.2, 0.015), (0.2, 0.035),
(0.3, 0.015), (0.3, 0.035)}`.)

## Verdict criteria (pre-registered, identical to graymix design pin #7)

- **CALIBRATED** ⇐ `cov68` within `±0.085` of nominal 0.68 AND
  `|map_bias| < 2·SEM` (`SEM = map_std/√120`) across the 12 truncated
  cell×truth rows (`zs ∈ {0.2, 0.3}` × 3 truths).
- **STILL BIASED** ⇐ otherwise; report which cells fail and how.

See `SUMMARY.md` in this directory for the verdict, tables, and the
decision-tree mapping to `.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`.
