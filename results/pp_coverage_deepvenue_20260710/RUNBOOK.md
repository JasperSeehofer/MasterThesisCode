# RUNBOOK — pp_coverage deep-venue (`z_support`) sweep, 2026-07-10

**Provenance:** handoff item L-A (`.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md`,
lines 23-35); production analog issue #29 (`bayesian_statistics.py` commit
`8db6c6e`, zero-host pure-completion fallback `p_i = B_num/D`); quick task
`260710-sjm-pp-coverage-deepvenue-mode`.

**Purpose:** measure P-P coverage + MAP bias of the issue-#29 pure-completion
fallback estimator at 60-95% catalogue incompleteness in a from-scratch
synthetic universe, using the `z_support` catalogue-support-truncated mode
added to `master_thesis_code/validation/pp_coverage.py` by this quick task.
If well-calibrated here, the eventual cluster re-eval (EXP-40) is a
confirmation rather than a first look.

This runbook is the single source the ORCHESTRATOR follows after this plan
merges. **The executor does NOT run the sweep** — cells are ~120x250
realizations x events and take minutes each.

---

## A. Sweep grid — 8 cells

Grid: `z_support` in `{0.2, 0.3, 0.5, 1.0}` x `sigma_z` in `{0.015, 0.035}`,
`kernel=volume`, defaults otherwise (`n_realizations=120`, `n_events=250`,
`truths=[0.62, 0.72, 0.84]`, `seed=20260701`).

`z_support=1.0` is `> Z_MAX_POP=0.95`, i.e. the **untruncated CONTROL** at
each `sigma_z` (`completion_fraction` should be identically `0`).

Per-cell command template:

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z {SZ} --z-support {ZS} \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs{ZS}_sz{SZ}_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs{ZS}_sz{SZ}_volume.log
```

The 8 concrete commands (`zs` in `{0.2,0.3,0.5,1.0}` x `sz` in `{0.015,0.035}`):

```bash
# zs=0.2, sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 0.2 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs0.2_sz0.015_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs0.2_sz0.015_volume.log

# zs=0.2, sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.2 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs0.2_sz0.035_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs0.2_sz0.035_volume.log

# zs=0.3, sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 0.3 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs0.3_sz0.015_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs0.3_sz0.015_volume.log

# zs=0.3, sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.3 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs0.3_sz0.035_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs0.3_sz0.035_volume.log

# zs=0.5, sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 0.5 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs0.5_sz0.015_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs0.5_sz0.015_volume.log

# zs=0.5, sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 0.5 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs0.5_sz0.035_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs0.5_sz0.035_volume.log

# zs=1.0 (CONTROL, untruncated), sz=0.015
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.015 --z-support 1.0 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs1.0_sz0.015_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs1.0_sz0.015_volume.log

# zs=1.0 (CONTROL, untruncated), sz=0.035
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 120 --n-events 250 --sigma-z 0.035 --z-support 1.0 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_zs1.0_sz0.035_volume.json \
  2>&1 | tee results/pp_coverage_deepvenue_20260710/pp_zs1.0_sz0.035_volume.log
```

Each command writes `pp_zs{ZS}_sz{SZ}_volume.json` + `.log` under
`results/pp_coverage_deepvenue_20260710/`.

---

## B. Anchor bit-identity re-run

Reproduce the committed anchor config (`n_realizations=250`, `n_events=250`,
`sigma_z=0.10`, `kernel=volume`, `seed=20260701`, **no** `z_support`) and
diff its `results` object against the pre-existing anchor from
`results/pp_coverage_sigmaz_scan_20260703/pp_sigmaz0.10_volume.json`:

```bash
uv run python -m master_thesis_code.validation.pp_coverage \
  --n-realizations 250 --n-events 250 --sigma-z 0.10 \
  --truths 0.62 0.72 0.84 --seed 20260701 --kernel volume \
  --output results/pp_coverage_deepvenue_20260710/pp_sigmaz0.10_volume_rerun.json

diff <(jq -S .results results/pp_coverage_deepvenue_20260710/pp_sigmaz0.10_volume_rerun.json) \
     <(jq -S .results results/pp_coverage_sigmaz_scan_20260703/pp_sigmaz0.10_volume.json)
```

**Note:** the `.results` block MUST be byte-identical (proves `z_support=None`
is a no-op). The `.config` block legitimately gains the `sigma_z_pv` and
`z_support` keys (added since the anchor was generated) — that difference is
**EXPECTED, not a regression**, so diff `.results` only, never the raw file.

---

## C. SUMMARY.md verdict format

The orchestrator writes `results/pp_coverage_deepvenue_20260710/SUMMARY.md`
after running sections A and B, using this template:

### Per-cell x truth table

One row per (`z_support`, `sigma_z`, `h_true`) triple (8 cells x 3 truths =
24 rows), columns:

| z_support | sigma_z | h_true | cov50 | cov68 | cov90 | rail_fraction | MAP mean | MAP bias | completion_fraction |
|---|---|---|---|---|---|---|---|---|---|

### Control comparison

For each truncated cell (`zs` in `{0.2, 0.3, 0.5}`), compare against its
`z_support=1.0` control **at the same `sigma_z`** (i.e. 6 comparisons: 3
`zs` values x 2 `sigma_z` values, each against its matching-`sigma_z`
`zs=1.0` row).

### Verdict criteria

- **Coverage collapse** — flag if `cov68` falls outside `+/-2*SE` of the
  control's `cov68`, with `SE ~= 0.085` for `n_realizations=120`
  (`2*sqrt(0.68*0.32/120) ~= 0.085`).
- **Bias flag** — flag if `|Delta map_mean vs control| > 2*SEM`, with
  `SEM = map_std / sqrt(120)` (using the truncated cell's own `map_std`).

### Carried caveats (state verbatim in the SUMMARY)

1. **1D-channel only** — the 2D (+0.057) question is NOT covered by this
   harness.
2. **Single-host clean limit** — production host-found events ALSO carry a
   `B_num` admixture in the mixture; this harness omits that, so ONLY the
   zero-host branch is the exact production analog.
3. **Hard truncation** (`z_support` step) vs production's soft M_BH-prune
   truncation of the effective catalogue.

---

## Do NOT

- Do NOT run any sweep command as part of the executor's plan task — only
  this runbook is authored here. The orchestrator runs section A/B post-merge
  and writes the SUMMARY per section C.
