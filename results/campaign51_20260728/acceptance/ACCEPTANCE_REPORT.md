# #51 campaign injection pool — pre-registered acceptance scoring

Date: 2026-07-28. Pool: `results/campaign51_20260728/pool_mix200k/`
(707 CSVs, 200,100 rows, mix3_50_25_25 stratified measure, h_inj = 0.73,
z_cut = 1.5). Scored offline, CPU-only, against the pre-registered criteria of
`docs/campaign_redesign_51_design.md` §4 / `SIZING_ANALYSIS.md` §6.

Method: the joint (u = ln(1+z), m = log10 M_z) with-BH survival grid was
rebuilt from the delivered pool by the **production estimator**
(`SimulationDetectionProbability`, `pdet_z_resolved=True`,
`pdet_wbh_z_resolved=True`, `expected_z_max=1.5`, `snr_threshold=20`); the
per-node Kish ESS and K5 shrinkage weights were read from its internal
`_wbh_ess` / `_wbh_w` tables. Catalogue weights are the GLADE+ profile
`W_z_lm` (`zres_survival/catalog_zw_profile.json`, 300 z x 60 lm cells),
projected onto the grid with the identical conventions of the sizing analysis
(`campaign_sizing_20260728/s1_sizing.py`: cell-centre queries, grid-box clamp
on m, bilinear ESS interpolation, w = ESS/(ESS + n0), n0 = 10).
Script: `score_acceptance.py`; all numbers: `acceptance_numbers.json`;
estimator build log: `build_log.txt`. No source files modified.

Measure-match confirmation (from the build log): the joint S(d_L | u, m)
kernel/ESS was built from **ALL 200,100 rows** (all strata); the pool-marginal
legs (pooled/1D survival, 2D S(d_L|M_z), sky bands, FIX-2 S(d_L|z), the K5
m-marginal shrinkage target) used the **99,014 stratum-'a' rows only** — as
ratified ([RATIFY-Z1], SIZING_ANALYSIS.md §4).

## Verdict — PRE-REGISTERED CRITERIA (reachable catalogue weight, as-built grid)

| # | Criterion | Threshold | Measured | Verdict |
|---|-----------|-----------|----------|---------|
| 1 | Catalogue-weighted median ESS | >= 1000 | **9088** | **PASS** |
| 2 | Catalogue weight-fraction on ESS < 500 | <= 1 % | **0.077 %** | **PASS** |
| 3 | Reachable-weight w-bar = E_W[ESS/(ESS+10)] | >= 0.99 | **0.99841** | **PASS** |

Unreachable-ridge weight (m > 7 + log10(1+z), source M > 1e7 — structurally
uncoverable, **exempt by pre-registration and reported separately, not
hidden**): **5.041 %** of total catalogue weight (sizing analysis predicted
5.04 % / "reachable 94.96 %" — exact agreement).

All three criteria also pass on the all-clamped weight set (median 8838,
W<500 = 0.102 %, w-bar 0.99838) and on the design-projected 61x69 grid (see
below). The estimator's *actual shipped* K5 weights give
w-bar(reachable) = 0.99838 — consistent with the ESS-derived value.

## Grid deviation vs the sizing design (flagged)

The sizing analysis assumed a **61 u x 69 m** grid at 0.05 dex on
m in [4, 7.398]. The delivered estimator builds **61 u x 31 m**
(`_WBH_ZRES_M_NODES = 31`) with m-nodes spanning the pool's own range
[4.0017, 7.3946] → **0.1131 dex spacing**. Since the grids differ materially
in noding, BOTH were scored; the design-projected rebuild reuses the
estimator's own bandwidths and Abramson pilot (sigma_u = 0.03873,
sigma_m = 0.09592) on the same 200,100-row sample. (Cross-check: a standalone
replication of the as-built 61x31 ESS matches the estimator's internal table
to max rel. dev. 1.8e-13 — convention parity is exact.)

| Metric (reachable catalogue weight) | as-built 61x31 | design 61x69 | sizing prediction (N=200k, mix3) |
|---|---|---|---|
| w-bar | 0.99841 | 0.99836 | 0.9985 |
| median ESS | 9088 | 8831 | 8160 |
| W-frac ESS < 500 | 0.077 % | 0.077 % | 0.01 % |
| W-frac ESS < 10 / < 100 | 0 / 0 | 0 / 0 | — |
| min ESS at query cells | 371 | 394 | — |
| grid-wide reachable min ESS | 151.3 | 150.7 | 172 |

Assessment of the deviation:
- **ESS/w-bar are noding-insensitive** (as the design doc predicted): every
  acceptance number agrees between the two grids to < 0.6 % relative.
  The acceptance verdict is grid-independent.
- **Interpolation-fidelity intent NOT met by the shipped noding**: the design
  wanted spacing (0.05 dex) below sigma_m (0.096); the shipped 31-node grid has
  0.113 dex > sigma_m. This is a **repo-constant deviation**
  (`_WBH_ZRES_M_NODES` still 31, m-range pool-derived instead of fixed
  [4, 7.398]), **not a pool deficiency** — the pool itself covers the full
  design m-range (delivered [4.0017, 7.3946] vs design [4, 7.398]). If the
  0.05-dex fidelity intent is to be honored, the constant needs a follow-up
  change (out of scope here; flagged for the author).
- Grid-wide reachable min ESS came in at 151 vs the synthetic-draw prediction
  172 (−12 %) — the expected real-campaign attrition/seed variation; still
  15x above the n0 = 10 floor, and the estimator log reports shrunk fraction
  (w < 0.5) = 0.000.

## §3.4-style table (as-built grid, 61 u x 31 m, 1753/1891 nodes reachable)

| Quantity | Value |
|---|---|
| node ESS min (all nodes / reachable nodes) | 90.6 / 151.3 |
| node ESS median (reachable nodes) | 2082 |
| reachable-node fraction ESS < 10 | 0.0 % |
| reachable-node fraction ESS < 100 | 0.0 % |
| reachable-node fraction ESS < 500 | 5.70 % |
| catalogue-weighted median ESS (reachable / all-clamped) | 9088 / 8838 |
| catalogue W-frac ESS < 10 | 0.0 % |
| catalogue W-frac ESS < 100 | 0.0 % |
| catalogue W-frac ESS < 500 (reachable / all-clamped) | 0.077 % / 0.102 % |
| catalogue W-frac ESS < 1000 (reachable) | 1.43 % |
| catalogue-weighted w-bar (reachable / all-clamped) | 0.99841 / 0.99838 |
| unreachable-ridge weight-fraction (exempt, reported separately) | 5.041 % |
| estimator-internal node ESS min/median (build-log line) | 90.61 / 1971.8 |

(Design-projected 61x69 equivalents in `acceptance_numbers.json`
`results.design_projected`.)

## Pool-delivery facts

| Fact | Value |
|---|---|
| rows / files | 200,100 / 707 |
| strata rows | a = 99,014; b = 50,947; c = 50,139 |
| SNR >= 20 detected fraction (overall) | 23.23 % |
| detected fraction per stratum | a: 7.62 %, b: 47.15 %, c: 29.72 % |
| detector-frame m = log10 M_z range | [4.0017, 7.3946] (design support [4, 7.398]) |
| z range | [6.5e-5, 1.49998]; depth gate at expected_z_max = 1.5 passed |
| h_inj / z_cut | 0.73 (single) / 1.5 (single — provenance gate passed) |
| code_rev | a9f29e82, f6449051 (2 revs — estimator warns; straggler-resubmit pattern) |
| t_plunge_yr present | 194,100 rows (6,000 pilot rows from task_0-59 lack it) |

Notes / surprises (reported as measured, no massaging):
1. The pool's second code revision is **f6449051**, not the `acaa0af`
   named in the tasking note as part of the generation stack. Two revisions in
   one pool is the documented straggler-resubmit pattern and the estimator
   accepted it with a warning; whether f6449051 vs a9f29e8 changed SNR
   semantics was NOT verified here and should be confirmed from the git log.
2. a-stratum count is 99,014, not the nominal 100,000 (≈1 % attrition), b/c
   are 50,947/50,139 vs nominal 50,000 — the delivered mix is 49.5/25.5/25.1 %,
   effectively on-design.
3. W-frac ESS<500 (0.077 %) is ~8x the synthetic prediction (0.01 %) but
   remains 13x under the 1 % criterion.
4. FIX-2 marginal leg (a-rows only) built clean: 121 u-nodes, node ESS
   min/median 388/6761, 0/726 sky-band cells below the ESS floor.

**Overall: the delivered pool PASSES all three pre-registered acceptance
criteria, on both the as-built and the design-projected grid.** The single
material deviation is the estimator's m-noding (31 nodes / pool-derived range
instead of the 69-node / fixed-range design grid) — acceptance-irrelevant but
flagged for the interpolation-fidelity follow-up.
