# Route 1 — adaptive Gauss-Hermite order for `completion_mass_factor_g`: measurement study

PERF/MEASUREMENT ONLY. No file under `master_thesis_code/` was touched. No
cluster/SSH use. This document reports what was measured; it makes no design
recommendation — the (n_low, t_tol) choice, if any, is the orchestrator's call.

## Scope

`completion_mass_factor_g` (`master_thesis_code/bayesian_inference/bayesian_statistics.py:2001`)
evaluates, per call, `g_i(z) = (1/sqrt(pi)) * SUM_j w_j * phi_x(mu_cond + sqrt(2) sigma_cond t_j) * scale`
at a fixed Gauss-Hermite order `n_hermite = 64` (`_G_I_HERMITE_NODES`), where
`phi_x(x) = dark_mass_density_per_mass(x*scale)*scale` is a two-segment
power law in the source-frame mass `M = x*scale` with band edges/kink at
`M in {1e4, 1e5, 1e7}` (`M_SOURCE_FRAME_MIN`, the `kappa_cap` surrogate
turn-over, `M_SOURCE_FRAME_MAX`).

Approved plan "Route 1": use a lower order `n_low` by default, falling back
to `n=64` for z-nodes whose Gaussian conditional straddles a breakpoint
within `|t| <= t_tol`, tested as

```
[mu_cond - sqrt(2)*sigma_cond*t_tol, mu_cond + sqrt(2)*sigma_cond*t_tol] * scale
    crosses any of {1e4, 1e5, 1e7} ?
```

This study harvests the realistic venue query distribution ONCE and answers,
offline, for the candidate grid `n_low in {8, 12, 16, 24}` x
`t_tol in {4, 5, 6}`: convergence error, fallback fraction, projected
speedup, and acceptance against two tolerance criteria. No re-run of the
instrument was needed for the sweep.

## Methods

### 1. Harvest (`harvest_route1.py`)

```
uv run python results/venue_transfer_20260811/perf/route1_study/harvest_route1.py
```

Builds the venue context via `master_thesis_code.validation.venue_transfer`:
`VenueConfig(cell="Tc", h_true=0.730, balls="real_k", sigma_mode="glade",
n_events_cap=30)`, then runs ONE seed (`vt.VT_BASE_SEED + 44000 = 20304808`,
the same seed as `counterfactual_smoke.py`/`profile_smoke.py`) with
`bayesian_statistics.completion_mass_factor_g` (and the same symbol as
imported into the `venue_transfer` namespace — it is a bare-name import, so
both bindings were patched) rebound to a logging wrapper that records, per
call, `det_M_z`, `proj_d_L_to_M`, `sigma_cond_M`, `n_hermite`, and the
per-node `(z_nodes, d_L_fraction)` arrays passed in, then calls the original
function unchanged. The module-global rebind is restored (and the
restoration asserted) after the run, following the pattern of
`results/venue_transfer_20260811/perf/counterfactual_smoke.py`.

The per-call scalars are broadcast over their node arrays and flattened into
one long per-node table, which is what lets `mu_cond(z) = 1 + proj_d_L_to_M
*(d_L_fraction - 1)` and `scale(z) = det_M_z/(1+z)` be recomputed for every
harvested z-node offline, without re-running the instrument.

Total harvested rows exceeded the 10M row cap, so the per-node table was
stratified-subsampled by call ID (each of the 1394 calls contributes the
same ~24.4% fraction of its own rows, `numpy.random.default_rng(20260808)`)
down to the cap. Output: `route1_harvest.npz`.

### 2-5. Offline analysis (`analyze_route1.py`)

```
uv run python results/venue_transfer_20260811/perf/route1_study/analyze_route1.py
```

For every harvested z-node, computes `g_i` at
`n in {8, 12, 16, 24, 64, 128, 256}` via
`numpy.polynomial.hermite.hermgauss` and the production
`dark_mass_density_per_mass` (default/affine path — the as-committed
production convention, never re-typed), chunked 200k rows at a time to
bound memory. `n=256` is the reference. The straddle test (exact spec
above) is evaluated per row for each `t_tol` candidate. `§3` additionally
computes a split-interval Gauss-Legendre reference (64 GL nodes per
sub-interval split at the interior breakpoint, Gaussian weight written
explicitly rather than absorbed into the `e^{-t^2}` kernel) restricted to
the straddling subset, and `§4` applies the profiled cost model
(g_i leg = 91.1% of seed wall, `results/venue_transfer_20260811/perf/PERF_ROADMAP.md`)
to the average node count implied by each `(n_low, t_tol)` pair.

**Correction during this study**: the first pass of `straddles()` compared
the dimensionless `mu_cond +- sqrt(2) sigma_cond t_tol` window directly
against the raw mass breakpoints `{1e4, 1e5, 1e7}`, omitting the `* scale`
the approved spec requires — a bug caught before results were used, not
after. The script was corrected to build the window in `x_M` space and then
multiply by the per-row `scale` before comparing to the breakpoints, and
the full sweep was rerun. **The corrected numbers are bit-for-bit identical
to the pre-fix numbers reported below** (same `fallback_fraction=0.0` at
every `t_tol`, same 12 acceptance winners, same speedups) — on this
harvested distribution the two implementations happen to agree because
`mu_cond` sits within `~5e-7` of 1 (dimensionless) while the nearest
breakpoint in `x_M` units is `>=0.03` away for every harvested row (see
Harvest statistics below), so the tiny incorrect window and the correct
`scale`-multiplied window both fail to reach any breakpoint. The corrected
script (`analyze_route1.py` as currently on disk) is what produced every
table in this document; nothing here is stale.

## Harvest statistics

- Seed: 20304808 (`VT_BASE_SEED + 44000`), `n_events_cap=30`.
- `build_venue_context`: 18.21 s. `n_events=30`, `sum_K=20024`, `max_K=14586`.
- Instrumented seed run wall time: 78.85 s.
- **Calls to `completion_mass_factor_g`: 1394.**
- **Total flattened z-nodes across all calls: 41,049,200.**
- Rows analyzed after stratified subsampling: 9,999,859 (24.37% of the total, stratified per call ID).

Quantiles (min / p1 / p10 / p50 / p90 / p99 / max) over the analyzed rows:

| Quantity | min | p1 | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| `sigma_cond_M` (dimensionless, x_M units) | 3.896e-09 | 1.356e-08 | 3.800e-08 | 6.276e-07 | 6.276e-07 | 6.276e-07 | 6.276e-07 |
| `scale = det_M_z/(1+z)` (M_sun) | 2.333e+05 | 2.841e+05 | 2.923e+05 | 3.096e+05 | 1.106e+06 | 1.420e+06 | 1.437e+06 |
| `mu_cond` (dimensionless) | 0.99999954 | 0.99999954 | 0.99999958 | 1.00000000 | 1.00000042 | 1.00000046 | 1.00000046 |

`sigma_cond_M` clustering at a single value for p50/p90/p99 reflects that
`sigma_cond` is event-level (constant across all z-nodes/quadrature nodes of
one event) and the row-weighted distribution is dominated by a few
high-node-count events. `mu_cond` sits within `~5e-7` of 1 for the entire
harvested distribution — the conditional Gaussian in `x_M` is extremely
narrow and centered essentially exactly on `x_M=1`, while the nearest
breakpoint (the kink at `M=1e5`) sits at `x_M = 1e5/scale in [0.070, 0.429]`
over the harvested `scale` range — roughly 5-6 orders of magnitude further
out in `x_M` than `sigma_cond_M` at `t_tol<=6`.

## Table 2 — convergence sweep vs. n=256, split by straddling / non-straddling

No harvested z-node straddles a breakpoint at any tested `t_tol` (see
Table on fallback fraction below) — the `straddling` group is empty
(`n_rows=0`, error reported as N/A) for all `(n, t_tol)` combinations. The
`non_straddling` group is therefore the full 9,999,859-row sample at every
`t_tol`. Values below are identical across `t_tol in {4,5,6}`.

| n | max rel. err vs n=256 (non-straddling) | P99.9 rel. err vs n=256 (non-straddling) | n_rows |
|---|---|---|---|
| 8   | 1.305e-15 | 1.108e-15 | 9,999,859 |
| 12  | 1.270e-15 | 9.331e-16 | 9,999,859 |
| 16  | 1.290e-15 | 9.410e-16 | 9,999,859 |
| 24  | 1.136e-15 | 9.018e-16 | 9,999,859 |
| 64  | 9.482e-16 | 7.514e-16 | 9,999,859 |
| 128 | 1.080e-15 | 7.565e-16 | 9,999,859 |
| 256 | 0.0 (reference) | 0.0 | 9,999,859 |

All errors are at or near float64 machine epsilon (~2.2e-16) times a few
ULP — i.e. on this harvest even `n=8` is converged to double-precision
noise, because every harvested Gaussian conditional sits deep inside one
smooth affine segment of `ln phi`, nowhere near the kink.

### Fallback fraction and average node count per (t_tol)

| t_tol | fallback_fraction (share of z-nodes routed to n=64) |
|---|---|
| 4 | 0.000000 |
| 5 | 0.000000 |
| 6 | 0.000000 |

### Average node count per z-node for each (n_low, t_tol)

Since `fallback_fraction=0.0` for every `t_tol`, average node count equals
`n_low` exactly, for every `t_tol`:

| n_low | t_tol | fallback_fraction | avg nodes/z-node |
|---|---|---|---|
| 8  | 4, 5, 6 | 0.0 | 8.0 |
| 12 | 4, 5, 6 | 0.0 | 12.0 |
| 16 | 4, 5, 6 | 0.0 | 16.0 |
| 24 | 4, 5, 6 | 0.0 | 24.0 |

## Table 3 — n=64 self-convergence at straddling nodes

**Empty result: zero harvested z-nodes straddle any breakpoint at
`t_tol in {4, 5, 6}`**, so there is no straddling subset to report
n=64-vs-n=256 or n=64-vs-split-Gauss-Legendre errors on. All three rows
(one per `t_tol`) have `n_rows=0`, `vs_n256=null`, `vs_split_gl=null`. This
is a direct consequence of the harvest statistics above: the harvested
`mu_cond` is within `~5e-7` of 1 while the nearest breakpoint sits `>=0.07`
away in `x_M` units, so even `t_tol=6` (`half_width = sqrt(2)*6.28e-7*6
~= 5.3e-6` at the widest observed `sigma_cond_M`) is roughly four orders of
magnitude short of reaching a breakpoint. The pinned n=64 convention's own
quadrature defect at the kink is therefore not exercised anywhere in this
harvested realistic-venue distribution, and this study cannot report a
non-vacuous self-convergence number for it.

## Table 4 — projected speedup

`speedup = 1/(0.089 + 0.911*(avg_nodes/64))`, applied to the average node
counts above (identical across `t_tol` since `fallback_fraction=0.0`
everywhere):

| n_low | avg nodes/z-node | projected seed-wall speedup |
|---|---|---|
| 8  | 8.0  | 4.9291x |
| 12 | 12.0 | 3.8489x |
| 16 | 16.0 | 3.1571x |
| 24 | 24.0 | 2.3222x |

(Identical for `t_tol in {4, 5, 6}`.)

## Table 5 — acceptance scan

Criteria: non-straddling max rel. error < 1e-12 (vs n=256) AND overall
(with fallback applied) max rel. error vs the n=64 convention < 1e-10.

| n_low | t_tol | max rel err non-straddling vs n=256 | max rel err overall vs n=64 convention | pass non-straddling <1e-12 | pass overall <1e-10 | accept |
|---|---|---|---|---|---|---|
| 8  | 4 | 1.305e-15 | 1.644e-15 | true | true | **true** |
| 12 | 4 | 1.270e-15 | 1.297e-15 | true | true | **true** |
| 16 | 4 | 1.290e-15 | 1.282e-15 | true | true | **true** |
| 24 | 4 | 1.136e-15 | 1.148e-15 | true | true | **true** |
| 8  | 5 | 1.305e-15 | 1.644e-15 | true | true | **true** |
| 12 | 5 | 1.270e-15 | 1.297e-15 | true | true | **true** |
| 16 | 5 | 1.290e-15 | 1.282e-15 | true | true | **true** |
| 24 | 5 | 1.136e-15 | 1.148e-15 | true | true | **true** |
| 8  | 6 | 1.305e-15 | 1.644e-15 | true | true | **true** |
| 12 | 6 | 1.270e-15 | 1.297e-15 | true | true | **true** |
| 16 | 6 | 1.290e-15 | 1.282e-15 | true | true | **true** |
| 24 | 6 | 1.136e-15 | 1.148e-15 | true | true | **true** |

All 12 candidate `(n_low, t_tol)` pairs pass both criteria, by a margin of
roughly 3 orders of magnitude on the non-straddling bound (~1e-15 vs the
1e-12 threshold) and 5 orders of magnitude on the overall bound (~1e-15 vs
the 1e-10 threshold). This is a direct restatement of Table 2: since no
harvested node ever falls back, "overall vs n=64 convention" reduces to
"n_low vs n=64" on the full sample, which Table 2 already showed to be
converged to float64 noise for every candidate `n_low`.

## Artifacts

- `harvest_route1.py` — instrumented single-seed harvest (writes `route1_harvest.npz`)
- `analyze_route1.py` — offline convergence/acceptance sweep (writes the JSON tables below)
- `route1_harvest.npz` — harvested per-node data (9,999,859 rows, stratified-subsampled from 41,049,200)
- `harvest_stats.json`, `table2_convergence_sweep.json`, `table3_n64_self_convergence.json`,
  `table4_projected_speedup.json`, `table5_acceptance_scan.json`, `table_nodecount.json` — raw JSON backing every table above
