# venue_transfer.py — Counterfactual Equivalence + Post-Swap Re-Profile

DATA ONLY — no recommendations (see `PERF_ROADMAP.md` for interpretation/roadmap). Nothing
under `master_thesis_code/` was touched by this task; RAILS were `results/venue_transfer_20260811/perf/`
only. No commits made.

**Machine:** `thinkpadseehofer`, Linux 6.18.40-2-lts x86_64 (dev machine, CPU-only, no GPU).
**Date:** 2026-08-11 (run timestamps below).
**Repo state:** branch `perf/realistic-venue`, `dark_mass_density_per_mass` in
`master_thesis_code/bayesian_inference/bayesian_statistics.py` carries the uncommitted swap
described in its docstring (default path = cached two-segment affine evaluation of `ln phi`;
`exact=True` = pre-swap verbatim `emri_rate.py` chain). Instrument code
(`master_thesis_code/validation/venue_transfer.py`) unmodified.

Same smoke parameters throughout, matching `profile_smoke.py`: cell `Tc`, `h_true=0.730`,
`balls="real_k"`, `sigma_mode="glade"`, `n_events_cap=30`, seed `VT_BASE_SEED + 44000`
(`20260808+44000`, inside the registered Tc(0.730) block), grain=seed, 1 worker (in-process).
Context: `n_events=30`, `sum_K=20024`, `max_K=14586`.

---

## 1. Counterfactual equivalence (`counterfactual_smoke.py`)

Context built ONCE, seed run TWICE against the same context: run A as-committed (affine
default), run B with the module-global `dark_mass_density_per_mass` name rebound to
`functools.partial(<original>, exact=True)` for the duration of the call (then restored;
restoration verified by identity assertion in-script). Both per-seed records serialised
through the identical deterministic JSON encoder (`sort_keys=True`, numpy leaves cast to
native Python types) and compared byte-for-byte, then leaf-by-leaf.

Raw output: `counterfactual_smoke.json` (full records + per-leaf diff table).

### Wall times

| run | path | wall (s) |
|---|---|---|
| A | affine default (as-committed) | 86.304 |
| B | exact (pre-swap, forced via module-global rebind) | 130.011 |

Ratio B/A = 1.506× (run B, exact, is ~1.5× slower than run A, affine, on this smoke). For
reference, `profile_smoke.py`'s originally-measured pre-swap baseline (separate run, same
parameters, cProfile overhead included) was 124.92 s — run B (130.011 s, no profiler overhead)
is consistent with that anchor.

### Equivalence verdict

- **Byte-identical serialisation: NO** (`byte_identical=False`).
- **11 leaves differ**, all at floating-point-noise scale:
  - **max abs diff: 2.842171e-12**
  - **max rel diff: 5.150040e-09**

Per-leaf diff table (all 11 differing leaves, from `counterfactual_smoke.json`):

| leaf | run A (affine) | run B (exact) | abs diff | rel diff |
|---|---|---|---|---|
| `edge_mass_2d` | 2.155755962915588e-08 | 2.1557559518133576e-08 | 1.110223e-16 | 5.150040e-09 |
| `ln_post_2d[2]` | -692.9570583387931 | -692.957058338793 | 1.136868e-13 | 1.640604e-16 |
| `ln_post_2d[5]` | -685.0833558360796 | -685.0833558360795 | 1.136868e-13 | 1.659460e-16 |
| `ln_post_2d[16]` | -675.9657761230719 | -675.9657761230718 | 1.136868e-13 | 1.681843e-16 |
| `ln_post_2d[22]` | -674.4010552721818 | -674.4010552721817 | 1.136868e-13 | 1.685745e-16 |
| `ln_post_2d[26]` | -675.1275941620166 | -675.1275941620165 | 1.136868e-13 | 1.683931e-16 |
| `ln_post_2d[33]` | -678.707261506095 | -678.7072615060949 | 1.136868e-13 | 1.675050e-16 |
| `map_2d_refined` | 0.734396507893643 | 0.7343965078936445 | 1.554312e-15 | 2.116448e-15 |
| `mean_2d` | 0.7352467055258997 | 0.7352467055258999 | 2.220446e-16 | 3.020001e-16 |
| `pit_2d` | 0.3869108406969245 | 0.3869108406969185 | 5.995204e-15 | 1.549505e-14 |
| `sum_dlog_gfrac_dh` | 3.7179597309332704 | 3.7179597309361125 | 2.842171e-12 | 7.644437e-13 |

All other leaves of the per-seed record (including `map_1d`, all 1D-channel fields, and the
non-differing 2D-channel fields) are exactly identical between run A and run B. The largest
relative difference (5.15e-9, `edge_mass_2d`) and largest absolute difference (2.84e-12,
`sum_dlog_gfrac_dh`) are both consistent with accumulated double-precision rounding from a
different but analytically-equivalent evaluation order (the docstring's "residual is O(few
ULP)" claim), not with a functional discrepancy between the affine and exact chains.

---

## 2. Post-swap re-profile (`profile_smoke.py`, default affine path)

Independent dedicated profiling run (separate process from the counterfactual script, same
parameters). Output files: `profile_smoke_postswap.pstats`, `profile_smoke_postswap_top.txt`
(baseline `profile_smoke.pstats` / `profile_smoke_top.txt` — the pre-swap, `exact=True`-era
committed files — copied aside before this run and restored byte-identical afterward; verified
via `md5sum` match and empty `git diff --stat` on both tracked files).

```
context_build_s=20.437 seed_wall_s=88.031
137679 function calls (136283 primitive calls) in 88.031 seconds
```

### Top-10 hotspots, post-swap (n=30 smoke), vs. pre-swap baseline (§1.1 of `PERF_ROADMAP.md`, wall 124.92 s)

| rank (postswap, by cumtime) | function (file:line) | ncalls | postswap cumtime (s) | postswap % of seed wall | baseline cumtime (s) | baseline % of seed wall |
|---|---|---|---|---|---|---|
| 1 | `_channel_terms_at_h` (venue_transfer.py:1061) | 41 | 88.025 | 100.0% | 124.92 | 100.0% |
| 2 | `_g_ball_capped` (venue_transfer.py:905) | 82 | 85.094 | 96.7% | 122.04 | 97.7% |
| 3 | `completion_mass_factor_g` (bayesian_statistics.py:2001) | 1,394 | 80.221 | 91.1% | 116.28 | 93.1% |
| 4 | `dark_mass_density_per_mass` (bayesian_statistics.py:1794) | 1,394 | 57.860 | 65.7% | 95.18 | 76.2% |
| 5 | `scipy.stats.pdf` (GL-candidate kernel) | 164 | 1.522 | 1.7% | 1.52 (`norm.pdf`) | 1.2% |
| 6 | `dist_vectorized` (physical_relations.py:226) | 82 | 1.099 | 1.2% | 1.09 | 0.9% |
| 7 | `scipy.interpolate._interpolate.__call__` | 82 | 0.951 | 1.1% | — (not in baseline top-10) | — |
| 8 | `scipy.interpolate._interpolate._evaluate` | 82 | 0.944 | 1.1% | — (not in baseline top-10) | — |
| 9 | `scipy.stats._continuous_distns._pdf` | 164 | 0.448 | 0.5% | — | — |
| 10 | `scipy.stats._continuous_distns._norm_pdf` | 164 | 0.447 | 0.5% | — | — |

Baseline ranks 5-10 (`dark_mass_log10_density_unnormalised`, `R_eff_per_mbh`,
`duty_cycle_Gamma`, `mbh_mass_function`, `R0_per_mbh`, `kappa_cap` — the `emri_rate.py` chain,
59.8%/42.8%/16.6%/15.2%/11.6%/10.6% of baseline wall) do **not** appear anywhere in the
post-swap top-25 (below the reduction cutoff): the affine default path does not call
`dark_mass_log10_density_unnormalised` or any `emri_rate.py` function — that entire chain is
compiled out of the per-call cost.

By internal (leaf, `tottime`) time, post-swap:

| rank | function | ncalls | tottime (s) | % of seed wall |
|---|---|---|---|---|
| 1 | `dark_mass_density_per_mass` (bayesian_statistics.py:1794) | 1,394 | 57.850 | 65.7% |
| 2 | `completion_mass_factor_g` (bayesian_statistics.py:2001) | 1,394 | 21.914 | 24.9% |
| 3 | `_g_ball_capped` (venue_transfer.py:905) | 82 | 4.823 | 5.5% |
| 4 | `scipy.interpolate._interpolate._evaluate` | 82 | 0.944 | 1.1% |
| 5 | `scipy.stats.pdf` | 164 | 0.591 | 0.7% |

`dark_mass_density_per_mass` remains the single largest leaf cost post-swap (65.7% of seed
wall, tottime), despite no longer calling the `emri_rate.py` chain.

### Wall-clock: baseline vs. post-swap

| quantity | value |
|---|---|
| Pre-swap baseline (`profile_smoke.py`, cProfile-wrapped, from `PERF_ROADMAP.md` §1.1) | 124.92 s |
| Post-swap (`profile_smoke.py`, cProfile-wrapped, this run) | 88.031 s |
| Reduction | 36.89 s (29.5%) |
| Speedup ratio (baseline / post-swap) | 1.419× |

Cross-check against the non-profiled counterfactual runs (§1, no cProfile overhead, same
context/seed): run A (affine) 86.304 s vs. run B (exact) 130.011 s, ratio 1.506×; run B is
consistent with the 124.92 s cProfile-wrapped baseline anchor.

---

## Files

- `results/venue_transfer_20260811/perf/counterfactual_smoke.py` — counterfactual driver (this task).
- `results/venue_transfer_20260811/perf/counterfactual_smoke.json` — full per-seed records (run A, run B) + diff table.
- `results/venue_transfer_20260811/perf/profile_smoke_postswap.pstats` — raw cProfile dump, post-swap (default affine) path.
- `results/venue_transfer_20260811/perf/profile_smoke_postswap_top.txt` — top-25 cumulative + top-15 tottime tables, post-swap.
- `results/venue_transfer_20260811/perf/profile_smoke.pstats`, `profile_smoke_top.txt` — pre-swap baseline, restored byte-identical to the committed version (verified `md5sum` + `git diff --stat`).
- `results/venue_transfer_20260811/perf/COUNTERFACTUAL_SMOKE.md` — this document.

---

## Route 1 (2026-08-12)

DATA ONLY. `completion_mass_factor_g` (`master_thesis_code/bayesian_inference/bayesian_statistics.py`)
carries an uncommitted second swap on top of the phi-swap above: `adaptive: bool = True` default —
per-row adaptive Gauss-Hermite order (fast `n=8` unless a row triggers a fallback condition, in
which case pinned `n=64`); `adaptive=False` restores the phi-swap-era pinned-`n=64` convention.
Same smoke parameters throughout: cell `Tc`, `h_true=0.730`, `balls="real_k"`, `sigma_mode="glade"`,
`n_events_cap=30`, seed `VT_BASE_SEED + 44000`, grain=seed, 1 worker (in-process). Context:
`n_events=30`, `sum_K=20024`, `max_K=14586`.

### Namespace note

`master_thesis_code/validation/venue_transfer.py` imports `completion_mass_factor_g` directly
(`from master_thesis_code.bayesian_inference.bayesian_statistics import
completion_mass_factor_g`, venue_transfer.py:198-199); the production caller `_g_ball_capped`
(venue_transfer.py:969) invokes it as a bare name resolved in **venue_transfer's own module
namespace**, not via `bs.completion_mass_factor_g` attribute access. `route1_counterfactual_smoke.py`
therefore rebinds `vt.completion_mass_factor_g` (not `bs.completion_mass_factor_g`) for run B, via a
wrapper that force-sets `kw["adaptive"] = False` on every call (a `functools.partial` was rejected
because it cannot safely override a keyword-only arg a caller might re-supply).

### 1. Counterfactual equivalence (`route1_counterfactual_smoke.py`)

Context built ONCE, seed run TWICE: run A as-committed (adaptive default), run B with
`vt.completion_mass_factor_g` rebound to force `adaptive=False` for the duration of the call (then
restored; restoration verified by identity assertion in-script). Raw output:
`route1_counterfactual_smoke.json`.

#### Wall times

| run | path | wall (s) |
|---|---|---|
| A | adaptive default (as-committed) | 12.782 |
| B | pinned n=64 (`adaptive=False`, forced via `vt`-namespace rebind) | 86.768 |

Ratio B/A = 6.789× (run B, pinned n=64, is ~6.8× slower than run A, adaptive, on this smoke).

#### Equivalence verdict

- **Byte-identical serialisation: NO** (`byte_identical=False`).
- **5 leaves differ**, all at floating-point-noise scale:
  - **max abs diff: 1.136868e-13**
  - **max rel diff: 1.262560e-14**

Per-leaf diff table (all 5 differing leaves, from `route1_counterfactual_smoke.json`):

| leaf | run A (adaptive) | run B (pinned n=64) | abs diff | rel diff |
|---|---|---|---|---|
| `ln_post_2d[10]` | -680.2165286525411 | -680.216528652541 | 1.136868e-13 | 1.671333e-16 |
| `ln_post_2d[23]` | -674.4569633219951 | -674.456963321995 | 1.136868e-13 | 1.685606e-16 |
| `map_2d_refined` | 0.7343965078936405 | 0.734396507893643 | 2.442491e-15 | 3.325847e-15 |
| `mean_2d` | 0.7352467055258998 | 0.7352467055258997 | 1.110223e-16 | 1.510001e-16 |
| `pit_2d` | 0.3869108406969294 | 0.3869108406969245 | 4.884981e-15 | 1.262560e-14 |

All other leaves (including `map_1d` and all 1D-channel fields) are exactly identical between
run A and run B. Magnitudes (max abs 1.14e-13, max rel 1.26e-14) are consistent with
accumulated double-precision rounding from a different but analytically-equivalent per-row
node-count/summation order (fast `n=8` vs. pinned `n=64` contraction), not with a functional
discrepancy — same order of magnitude as, and smaller than, the phi-swap counterfactual's
residual (§1 above, max rel 5.15e-9).

### 2. Route 1 re-profile (`profile_smoke.py`, adaptive default)

Independent dedicated profiling run (separate process from the counterfactual script, same
parameters). Output files: `profile_smoke_route1.pstats`, `profile_smoke_route1_top.txt`
(committed baseline `profile_smoke.pstats`/`profile_smoke_top.txt` and the postswap
`profile_smoke_postswap.pstats`/`profile_smoke_postswap_top.txt` were untouched; the
run target files were copied aside before this run, restored byte-identical afterward,
verified via `md5sum` match and empty `git diff --stat` on both tracked files).

```
context_build_s=20.693 seed_wall_s=13.466
194833 function calls (193437 primitive calls) in 13.466 seconds
```

#### Top-10 hotspots, Route 1 (n=30 smoke, by cumtime), vs. post-swap (§2 above, wall 88.031 s)

| rank (route1, by cumtime) | function (file:line) | ncalls | route1 cumtime (s) | route1 % of seed wall | postswap cumtime (s) | postswap % of seed wall |
|---|---|---|---|---|---|---|
| 1 | `_channel_terms_at_h` (venue_transfer.py:1061) | 41 | 13.459 | 100.0% | 88.025 | 100.0% |
| 2 | `_g_ball_capped` (venue_transfer.py:905) | 82 | 10.265 | 76.2% | 85.094 | 96.7% |
| 3 | `completion_mass_factor_g` (bayesian_statistics.py:2012) | 1,394 | 10.083 | 74.9% | 80.221 | 91.1% |
| 4 | `_contract_group` (bayesian_statistics.py:2101, Route 1 helper) | 1,394 | 8.996 | 66.8% | — (function did not exist pre-Route 1) | — |
| 5 | `dark_mass_density_per_mass` (bayesian_statistics.py:1805) | 1,394 | 6.255 | 46.5% | 57.860 | 65.7% |
| 6 | `scipy.stats.pdf` (GL-candidate kernel) | 164 | 1.534 | 11.4% | 1.522 | 1.7% |
| 7 | `dist_vectorized` (physical_relations.py:226) | 82 | 1.129 | 8.4% | 1.099 | 1.2% |
| 8 | `scipy.interpolate._interpolate.__call__` | 82 | 1.041 | 7.7% | 0.951 | 1.1% |
| 9 | `scipy.interpolate._interpolate._evaluate` | 82 | 1.035 | 7.7% | 0.944 | 1.1% |
| 10 | `scipy.stats._continuous_distns._pdf` | 164 | 0.476 | 3.5% | 0.448 | 0.5% |

By internal (leaf, `tottime`) time, Route 1:

| rank | function | ncalls | tottime (s) | % of seed wall |
|---|---|---|---|---|
| 1 | `dark_mass_density_per_mass` (bayesian_statistics.py:1805) | 1,394 | 6.248 | 46.4% |
| 2 | `_contract_group` (bayesian_statistics.py:2101) | 1,394 | 2.508 | 18.6% |
| 3 | `completion_mass_factor_g` (bayesian_statistics.py:2012) | 1,394 | 1.017 | 7.6% |
| 4 | `scipy.interpolate._interpolate._evaluate` | 82 | 1.035 | 7.7% |
| 5 | `scipy.stats.pdf` | 164 | 0.587 | 4.4% |

Post-swap the same rank-1 leaf cost was `dark_mass_density_per_mass` at 65.7% of seed wall
(57.850 s); Route 1 collapses that to 46.4% (6.248 s) in absolute terms while the *relative*
share stays roughly the largest single leaf — the adaptive order reduces `dark_mass_density_per_mass`
call volume/argument size (via the fast-order fallback split in `_contract_group`) rather than
eliminating the leaf.

### 3. Seed-wall progression and cumulative speedup

| stage | driver | seed wall (s) | speedup vs. previous stage | cumulative speedup vs. baseline |
|---|---|---|---|---|
| baseline (pre-swap, exact `dark_mass_density_per_mass`) | `profile_smoke.py`, `PERF_ROADMAP.md` §1.1 | 124.92 | — | 1.000× |
| phi swap (affine `dark_mass_density_per_mass` default) | `profile_smoke_postswap.pstats` (§2 above) | 88.031 | 1.419× | 1.419× |
| Route 1 (adaptive Gauss-Hermite order, `completion_mass_factor_g`) | `profile_smoke_route1.pstats` (this section) | 13.466 | 6.538× | 9.278× |

All three numbers are `profile_smoke.py`-measured (cProfile-wrapped, includes profiler
overhead), same smoke parameters, directly comparable.

### Files (Route 1 addendum)

- `results/venue_transfer_20260811/perf/route1_counterfactual_smoke.py` — Route 1 counterfactual driver.
- `results/venue_transfer_20260811/perf/route1_counterfactual_smoke.json` — full per-seed records (run A, run B) + diff table.
- `results/venue_transfer_20260811/perf/profile_smoke_route1.pstats` — raw cProfile dump, Route 1 (adaptive default) path.
- `results/venue_transfer_20260811/perf/profile_smoke_route1_top.txt` — top-25 cumulative + top-15 tottime tables, Route 1.
