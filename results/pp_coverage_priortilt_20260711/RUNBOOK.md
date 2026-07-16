# RUNBOOK — pp_coverage prior-tilt ladder (N-3) + residual-floor discriminator, 2026-07-11

**Provenance:** quick task `260711-1ps-prior-sensitivity`; handoff item **N-3**
(prior-sensitivity probe, feeds decision D1) plus the **residual-floor
discriminator** for the σ_z-independent +0.002…+0.005 completion-branch bias
floor isolated by quick task 260711-117
(`results/pp_coverage_exactmode_20260711/SUMMARY.md`);
`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`. Code at **`e5b8383`** on
`physics/zero-host-completion-fallback` (adds
`PPCoverageConfig.inference_wpop_tilt` = γ — inference-side w_pop × exp(γ·z),
strict γ==0.0 gate, generative truth draw never tilted — plus the `--h-step`
CLI flag to `master_thesis_code/validation/pp_coverage.py`).

**γ=0 baselines (already committed at the IDENTICAL grid/seed/realizations —
NOT re-run, cited for the finite-difference lever arm):**

- two_branch γ=0, σ_z=0.035, zs=0.2:
  `results/pp_coverage_deepvenue_20260710/pp_zs0.2_sz0.035_volume.json`
- exact γ=0, σ_z=0.035, zs=0.2:
  `results/pp_coverage_exactmode_20260711/pp_exact_zs0.2_sz0.035.json`

Both: n_realizations=120, n_events=250, seed=20260701, truths
[0.62, 0.72, 0.84], h_step=0.004, n_z_quad=160. All ladder runs below MUST
match this grid.

**Anti-repetition (ledger):** gray/conditioned modes were adjudicated STILL
BIASED in 260711-07n (`results/pp_coverage_graymix_20260711/SUMMARY.md`) and
the σ_z-DEPENDENT kernel-support leak mechanism was adjudicated in 260711-117
(removed exactly by the membership-truncated kernel). Neither is re-litigated
here. This task probes ONLY (i) the inference-prior sensitivity of the
completion-dominated regime and (ii) the σ_z-INDEPENDENT completion-branch
residual floor.

---

## Pre-registered predictions (written BEFORE any run)

> **(i) Exact-mode lever arm (N-3):** completion-branch prior sensitivity is
> REAL and roughly linear in γ across the ladder — `B_num` integrates the
> population prior w_pop over the out-of-catalogue volume by construction, so
> the deep regime is population-prior-driven. The MAGNITUDE is UNKNOWN; that
> is the measurement. (Direction observed during test implementation on a
> tiny config: map_mean ascending in γ — more prior weight at high z pushes
> the posterior toward higher h.)
>
> **(ii) Floor prediction — UNKNOWN, a genuine discriminator.** Both outcomes
> and their consequences, stated in advance:
> - **(artifact)** If the +0.002…+0.005 exact-mode γ=0 floor SHRINKS toward 0
>   with finer h_step (0.004 → 0.002 → 0.001) and/or finer z-quadrature
>   (n_z_quad 160 → 320), it is MAP-grid/quadrature discretization ⇒ exact
>   mode is fully calibrated and the production-correction candidate
>   (membership-truncated / completeness-weighted kernel, 260711-117 item 2)
>   GAINS strength.
> - **(persistent)** If the floor is STABLE under finer grids, it is a genuine
>   composition residual of the completion-dominated regime, to be quantified
>   against the campaign SEM before any depth-1.5 + fallback closure claim.

---

## Common settings

All 11 runs: `--kernel volume --n-realizations 120 --n-events 250
--truths 0.62 0.72 0.84 --seed 20260701 --z-support 0.2 --sigma-z 0.035`
(deep-venue conventions; run via
`uv run python -m master_thesis_code.validation.pp_coverage ... 2>&1 | tee <log>`).

## Set (a) — tilt ladder (8 runs)

Default `h_step=0.004` and `n_z_quad=160` so the γ=0 baselines above complete
the 5-point ladder. Grid: `MODE ∈ {two_branch, exact} × GAMMA ∈ {-0.2, -0.1,
0.1, 0.2}`:

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --kernel volume --n-realizations 120 --n-events 250 --truths 0.62 0.72 0.84 --seed 20260701 \
  --z-support 0.2 --sigma-z 0.035 --mixture-mode {MODE} --inference-wpop-tilt {GAMMA} \
  --output results/pp_coverage_priortilt_20260711/pp_tilt_{MODE}_g{GAMMA}.json \
  2>&1 | tee results/pp_coverage_priortilt_20260711/pp_tilt_{MODE}_g{GAMMA}.log
```

(8 JSONs: `pp_tilt_two_branch_g-0.2.json`, `pp_tilt_two_branch_g-0.1.json`,
`pp_tilt_two_branch_g0.1.json`, `pp_tilt_two_branch_g0.2.json`,
`pp_tilt_exact_g-0.2.json`, `pp_tilt_exact_g-0.1.json`,
`pp_tilt_exact_g0.1.json`, `pp_tilt_exact_g0.2.json`.)

## Set (b) — floor discriminator (3 runs, exact mode, γ=0)

Finer h_step (2 runs, `HS ∈ {0.002, 0.001}`, default n_z_quad=160):

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --kernel volume --n-realizations 120 --n-events 250 --truths 0.62 0.72 0.84 --seed 20260701 \
  --z-support 0.2 --sigma-z 0.035 --mixture-mode exact --h-step {HS} \
  --output results/pp_coverage_priortilt_20260711/pp_floor_hstep{HS}.json \
  2>&1 | tee results/pp_coverage_priortilt_20260711/pp_floor_hstep{HS}.log
```

Finer z-quadrature (1 run, default h_step=0.004):

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --kernel volume --n-realizations 120 --n-events 250 --truths 0.62 0.72 0.84 --seed 20260701 \
  --z-support 0.2 --sigma-z 0.035 --mixture-mode exact --n-z-quad 320 \
  --output results/pp_coverage_priortilt_20260711/pp_floor_nzq320.json \
  2>&1 | tee results/pp_coverage_priortilt_20260711/pp_floor_nzq320.log
```

Floor comparison anchor: the exact γ=0 run at default h_step/n_z_quad is the
cited baseline `pp_exact_zs0.2_sz0.035.json` (biases +0.0026/+0.0046/+0.0042
at truths 0.62/0.72/0.84).

## Analysis plan (SUMMARY.md)

1. **Lever arm** d(map_mean)/dγ per truth per mode, finite-differenced across
   the 5-point ladder {−0.2, −0.1, 0 (baseline), +0.1, +0.2}, with each
   truth's comp_frac (0.71 → 0.85) alongside.
2. **Headline D1 number:** Δh(γ_10%) for a ±10%-across-completion-domain prior
   misspecification, γ_10% = ln(1.1)/(0.95 − 0.2) ≈ 0.127, linearly
   interpolated between γ=+0.1 and γ=+0.2; absolute Δh AND % of h_true, per
   truth per mode.
3. **Composition sensitivity:** two_branch (σ_z leak still present) vs exact
   lever-arm contrast.
4. **Floor verdict:** exact γ=0 floor at h_step 0.004 vs 0.002 vs 0.001 and
   n_z_quad 160 vs 320; PRIMARY readout on the 0.62/0.72 truths (0.84 sits
   near the 0.86 grid edge — secondary); 2·SEM column (SEM = map_std/√120).
5. **Decision mapping** to D1 per the handoff outcome→decision map — WITHOUT
   re-deciding D1 (user's call).

Carried caveats (verbatim): 1D-channel only; single-host clean limit; hard
z_support truncation vs production's soft M_BH prune.
