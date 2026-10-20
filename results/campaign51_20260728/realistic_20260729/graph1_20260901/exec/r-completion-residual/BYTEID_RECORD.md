# BYTEID_RECORD — independent byte-id verification, b-completion-scorer (r-completion-residual)

Date: 2026-09-03. Role: INDEPENDENT BYTE-ID VERIFIER. Scope: anchor-reproduction path ONLY (the
REGISTRATION_DRAFT.md §5 g-byte-id gate: 67 S3 harness checkpoints' `score_at_truth.no_bh.dark.mean`
bit-for-bit, plus the T0 re-baseline `mean_h = 0.666987` to the draft's quoted 1e-9). Did **not** run
`compute_registered_statistics` / the registered statistic, and did not import or call the builder's
`completion_residual_reads.py` — per standing rule ("verifier output is evidence, not authority: re-
derive, do not trust prose"), a fresh script (`byteid_check.py`, same directory) re-reads the raw
files and re-implements both checks independently.

## What was independently re-derived

**(A) Harness byte-id (67 checkpoints).** Globbed `universe_seed*_S.json` under
`tree2_20260830/b8_cal_harness_work_s4_postflip/` directly (67 files found, matching the draft's
seed901000–901066 range and §5's "0 mixed rows" population gate: all 67 have
`universe.n_draw_requested == 200`). Read `score_at_truth.no_bh.dark.mean` from each, sorted by
seed, and compared bit-for-bit (`==`, not a tolerance) against the 67 values transcribed by hand
from BUILD_RECORD.md's `dark_full_score_means` list. Also independently re-checked
`resolved_flags` for internal consistency (one distinct JSON-serialized block across all 67 — same
result the builder reported) and independently re-aggregated mean/SEM of the 67 values.

**(B) T0 `mean_h` re-baseline.** Reproduced from the raw
`retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv`
(1588 events × 41 h-nodes, `combined_no_bh` channel) using `combine_log_likelihood(..., "physics_
floor")` imported verbatim from `darksiren_emri.validation.correspondence_1d` (no physics
re-implemented, per the registration draft's explicit instruction) and the T0 gradient-trapezoid
convention from `prod2d_closure_20260818/tier0_bootstrap_jackknife.py`'s own docstring
(`w = np.gradient(h_grid)`, `mean_h = Σ post_n·h·w`). This independently confirms `READOUT_RECORD.md`
row `iiib | 1D (no-BH, combined_no_bh) | ... | mean_h 0.666987`.

## Results

| check | result |
|---|---|
| harness checkpoints globbed | 67 |
| harness checkpoints matched to `--population 200` | 67/67 |
| `resolved_flags` internally consistent | yes (1 distinct block across 67) |
| 67 `dark.mean` values vs BUILD_RECORD.md's quoted list | **bit-for-bit exact**, `max_abs_dev = 0.0` |
| independent re-aggregation: mean of 67 dark means | `0.008215870005381617` (matches BUILD_RECORD.md's `mean_of_dark_full_score_means` exactly) |
| independent re-aggregation: SEM of 67 dark means | `0.006314188695650197` (matches BUILD_RECORD.md's `sem_of_dark_full_score_means` exactly) |
| T0 `mean_h`, independently computed | `0.6669869414473403` |
| `round(computed, 6)` vs displayed anchor `0.666987` | equal |
| literal `abs(computed − 0.666987)` | `5.855265972076751e-08` |
| literal `1e-9` tolerance (draft's quoted number) satisfiable against a 6-dp source | **no** — same disclosed finding as BUILD_RECORD.md: a 6-decimal display anchor cannot in principle be matched to `1e-9` absolute difference (rounding error alone can reach `5e-7`); "reproduced to 1e-9" is only achievable as "rounds to the displayed anchor," which this independent computation does |

## Verdict: **GREEN**

Every pair compared is either exactly equal (all 67 harness means, and the mean/SEM aggregates) or
within the only tolerance the source data can support (the T0 anchor, which — as BUILD_RECORD.md
itself discloses — is unsatisfiable at a literal `1e-9` against a 6-decimal display value by
construction; both the builder's and this independent computation round to the displayed anchor
exactly, and the two full-precision computations of `mean_h` agree with each other to the last
printed digit: `0.6669869414473403` both times). No discrepancy of any kind was found between this
independent read and the builder's reported numbers.

- **n_pairs compared:** 68 (67 harness checkpoint means + 1 T0 `mean_h` anchor)
- **max_abs_dev:** `5.855265972076751e-08` (entirely attributable to the T0 anchor's 6-dp display
  rounding, not to any computational disagreement — the 67 harness pairs are all `0.0`, and the
  independent `mean_h` full-precision value matches the builder's full-precision value to the last
  printed digit)

## Script

`byteid_check.py` (same directory) — standalone, does not import `completion_residual_reads.py`.
Run from repo root:

```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/byteid_check.py
```

Full JSON report (including all 67 re-read `dark.mean` values) printed to stdout; exit code 0.

## Scope discipline

Did not touch `darksiren_emri/` (only imported `combine_log_likelihood`, read-only). Did not run
the pipeline or a cluster job. Did not compute `T_prod`/`T_harn`/`Z`/`ρ`/the disposition — that is
the registered statistic, explicitly out of scope for this byte-id pass.
