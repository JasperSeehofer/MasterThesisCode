# RUNBOOK — pp_coverage `mixture_mode="absolute"` calibration campaign, 2026-07-26

**Provenance:** `results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md`
Variant 1 (Sec 6, validation gate 1) — the absolute-mass marginal
`p_i(h) = [A_i(h) + B_num_i(h)] / D(h)` with NO self-normalization of the
catalogue term. Harness change: `master_thesis_code/validation/pp_coverage.py`,
new `mixture_mode="absolute"`.

**Purpose:** measure whether the harness's independent synthetic-universe P-P/
coverage instrument shows the `two_branch` (current default) estimator's
completion-regime miscalibration cured by `absolute` mode, in the same three
regimes the derivation and prior harness campaigns
(`results/pp_coverage_deepvenue_20260710/`, `results/pp_coverage_graymix_20260711/`)
established: a complete-catalogue shallow cell, an intermediate completion-
governed cell, and a deep completion-governed cell (22-85% of events routed
to the completion branch).

---

## Regression check (existing modes byte-identical)

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.3 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output /tmp/regression_zs0.3_sz0.035_volume.json
```

Diffed against `results/pp_coverage_deepvenue_20260710/pp_zs0.3_sz0.035_volume.json`
(2026-07-10, `two_branch` default). Every shared field (`coverage`, `map_mean`,
`map_std`, `map_median`, `map_bias`, `rail_fraction`, `completion_fraction`)
is bit-identical; the only diff is that the new JSON carries the
`dlogL_dh_host_mean` / `dlogL_dh_completion_mean` diagnostic keys the 2026-07-10
harness predates (unrelated to this change — those keys were added by a later,
independent commit). Confirms this change does not perturb `two_branch`,
`gray`, `conditioned`, or `exact`.

## Cell definitions

| Cell | `--z-support` | `--sigma-z` | Completion fraction @ h=0.62/0.72/0.84 (two_branch) |
|---|---|---|---|
| shallow (complete catalogue) | `1.0` (> `Z_MAX_POP=0.95`, untruncated control) | `0.035` | 0.00 / 0.00 / 0.00 |
| intermediate | `0.3` | `0.035` | 0.22 / 0.39 / 0.55 |
| deep | `0.2` | `0.035` | 0.71 / 0.79 / 0.85 |

Common config: `--n-events 250 --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume`.
`n_realizations` per cell noted below (120 for the first smoke pass, 500 for
the final reported numbers — see SUMMARY.md; both are archived).

## Commands (final, n_realizations=500)

```bash
OUT=results/pp_coverage_absolute_20260726

# Shallow / complete catalogue
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 500 --n-events 250 --sigma-z 0.035 --z-support 1.0 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --mixture-mode two_branch --output "$OUT/pp_shallow_two_branch.json"

uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 500 --n-events 250 --sigma-z 0.035 --z-support 1.0 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --mixture-mode absolute --output "$OUT/pp_shallow_absolute.json"

# Intermediate
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 500 --n-events 250 --sigma-z 0.035 --z-support 0.3 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --mixture-mode two_branch --output "$OUT/pp_intermediate_two_branch.json"

uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 500 --n-events 250 --sigma-z 0.035 --z-support 0.3 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --mixture-mode absolute --output "$OUT/pp_intermediate_absolute.json"

# Deep
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 500 --n-events 250 --sigma-z 0.035 --z-support 0.2 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --mixture-mode two_branch --output "$OUT/pp_deep_two_branch.json"

uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 500 --n-events 250 --sigma-z 0.035 --z-support 0.2 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --mixture-mode absolute --output "$OUT/pp_deep_absolute.json"
```

Runtime: ~17s/cell at `n_realizations=120`; ~70s/cell at `n_realizations=500`
(single CPU core, no GPU). Full 6-run campaign at `n_realizations=500`
completes in well under 10 minutes — far inside the 2-4h budget; no need to
scale further given the near-null effect size found (SUMMARY.md).

## Deliverables

- `SUMMARY.md` — coverage tables (old vs new), bias per cell, verdict.
- `pp_{shallow,intermediate,deep}_{two_branch,absolute}.json` — raw per-cell
  results (small: coverage/bias/completion_fraction summary statistics only,
  not per-realization traces).
- `pp_*.log` — stdout of each run (one-line-per-truth summary).
