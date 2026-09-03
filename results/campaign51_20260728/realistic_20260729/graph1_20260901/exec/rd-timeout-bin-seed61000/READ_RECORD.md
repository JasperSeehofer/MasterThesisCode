# READ_RECORD — rd-timeout-bin-seed61000

Node: `rd-timeout-bin-seed61000` (Graph 1 addendum Branch L, item 2; docket R9 chair-approved).
Type: **read**, verdict-free. Precedent: `../rd-timeout-bin-seed3000/READ_RECORD.md`. Full machine
output: `READ_RECORD.json`, `design_gate_bin_edges.json`, `rate_table_*.csv`, `comparison_*.csv`,
`selection_effect_note.csv`. Script: `analyze.py` (re-derivable).

## Existence contract

| input | status | detail |
|---|---|---|
| `.../seed61000/prepared_cramer_rao_bounds.csv` | **PRESENT, md5 VERIFIED** | pin `9a1f2a14384a9281c97ca3be312ddaab`, matches actual; 1590 rows (kept, production H0-inference pool) |
| `.../cluster_logs_fetch_20260904_MANIFEST.md5` | **PRESENT** | 2194 lines, `md5sum -c` verified clean against the fetched tree |
| `logs/simulate_6088772_*.err` (simulate array, 100 tasks) | **PRESENT** | 100/100 files; timeout param-dicts (both `Waveform/SNR ...` and `Cramér-Rao ...`) confirmed to live here, NOT in `.out` or the top-level `master_thesis_code_*.log` app logs |
| SNR-stage timeout records | **PRESENT, full params** | 820 |
| CRB-stage timeout records | **PRESENT, full params** | 2 |
| `Skip tally` summary lines | **ABSENT** | 0/100 tasks (all cancelled at walltime before printing one) — worse completeness than seed3000 (33/100), disclosed |
| other skip categories (`ParameterOutOfBoundsError`, `ZeroDivisionError`, ...) | present as bare error strings, **no per-event params logged** | not binnable, excluded by construction (same convention as seed3000) |

The fetch (2194 files) also contains `real_r1..r5`, `sig0_control`, `zoom`, `estimatorB_2x2`
subtrees and top-level `master_thesis_code_*.log` files belonging to the same
`run_20260729_seed61000` workspace but to *other* jobs/evaluate variants — **not used**; only job
`6088772` (the seed61000 simulate array itself) is analyzed here.

## Design gate — bin edges frozen BLIND, before any rate was read

**Not** a verbatim reuse of seed3000's numeric edges on all axes. Reason: seed3000's `p0` prior was
the now-**retired** `[10, 16]` SNAPSHOT-mode bound (few's Pn5AAK input domain); production seed61000
draws `p0` via the unclamped plunge-window convention (`HIGHM_AUDIT.md` item 1, 2026-07-28 flip) — no
upper clamp. Observed seed61000 `p0` support is **[3.68, 87.22]**, not [10,16]. The injected ranges
genuinely differ, so per the node's own instruction the *same blind rule* (log-spaced M / quantile
e0 / quantile p0, 5 bins each) was applied **fresh** to the seed61000 union(kept, timeout)
population. `M`'s edges are additionally reported reusing seed3000's edges **verbatim**, for a direct
cross-run comparison table (raw injection CSVs were excluded from the fetch per OPS_RECORD step 4,
so both runs use the kept+timeout union as the population proxy — same convention).

- seed61000-native **M**: log-spaced, [1.147e4, 1.377e7] M☉ (5 bins)
- seed61000-native **e0**: quintile, [0.0503, 0.1997] (5 bins)
- seed61000-native **p0**: quintile, [3.68, 87.22] (5 bins)
- **M (comparison)**: seed3000's edges reused verbatim, [3.55e4, 1.77e6] M☉

Both edge sets recorded in `design_gate_bin_edges.json`.

## SNR-stage timeout rate — aggregate

**820 / 2412 = 0.3400** (Garwood 95% [0.3171, 0.3641]). Denom = 1590 kept + 2 crb-stage timeouts +
820 snr-stage timeouts.

## By axis (seed61000-native edges), Poisson (Garwood) 95%

### M (log-spaced)

| bin | range (M☉) | n_kept | n_to | denom | rate | Garwood 95% |
|---|---|---|---|---|---|---|
| 0 | 1.15e4–4.74e4 | 0 | 206 | 206 | 1.000 | [0.868,1.146]† |
| 1 | 4.74e4–1.96e5 | 9 | 302 | 311 | 0.971 | [0.865,1.087]† |
| 2 | 1.96e5–8.07e5 | 1279 | 216 | 1495 | 0.144 | [0.126,0.165] |
| 3 | 8.07e5–3.33e6 | 304 | 81 | 385 | 0.210 | [0.167,0.261] |
| 4 | 3.33e6–1.38e7 | 0 | 15 | 15 | 1.000 | [0.560,1.649]† |

†Garwood upper bound on the count can exceed denom at low n; reported uncapped per exact-Poisson
convention.

**Max adjacent-bin gradient: 14.35σ, bins 1→2** (0.971 → 0.144).

### e0 (quintile) — flat

| bin | range | n_kept | n_to | denom | rate | Garwood 95% |
|---|---|---|---|---|---|---|
| 0 | 0.0503–0.0815 | 317 | 166 | 483 | 0.344 | [0.293,0.400] |
| 1 | 0.0815–0.1089 | 305 | 177 | 482 | 0.367 | [0.315,0.425] |
| 2 | 0.1089–0.1392 | 324 | 158 | 482 | 0.328 | [0.279,0.383] |
| 3 | 0.1392–0.1699 | 320 | 162 | 482 | 0.336 | [0.286,0.392] |
| 4 | 0.1699–0.1997 | 326 | 157 | 483 | 0.325 | [0.276,0.380] |

**Max adjacent-bin gradient: 1.02σ.** Flat within Poisson.

### p0 (quintile) — steep gradient, top bin saturated

| bin | range | n_kept | n_to | denom | rate | Garwood 95% |
|---|---|---|---|---|---|---|
| 0 | 3.68–11.34 | 371 | 112 | 483 | 0.232 | [0.191,0.279] |
| 1 | 11.34–12.86 | 436 | 46 | 482 | 0.095 | [0.070,0.127] |
| 2 | 12.86–14.49 | 442 | 40 | 482 | 0.083 | [0.059,0.113] |
| 3 | 14.49–19.96 | 343 | 139 | 482 | 0.288 | [0.242,0.341] |
| 4 | 19.96–87.22 | 0 | 483 | 483 | 1.000 | [0.913,1.093]† |

**Max adjacent-bin gradient: 13.58σ, bins 3→4.** Bin 4 (p0 > ~20) is 100% timeout — this bin did not
exist in seed3000's bounded [10,16] prior; the unclamped plunge-window `p0` tail is entirely lost.

### 2-D (M, p0) grid (native edges)

Max adjacent-cell gradient **5.97σ**, cells (M-bin2,p0-bin2)↔(M-bin2,p0-bin3) — i.e. within the
dominant M-bin-2 slice, `p0` still shows a real secondary tilt (0.071→0.231) unlike seed3000 where
the marginal p0 axis was flat. Full table `rate_table_2d_M_p0.csv`.

## CRB-stage timeouts — descriptive only (n=2)

2/1592 = 0.00126, Garwood 95% [0.00006, 0.00417].

## Population depth (z, kept pool, h=0.73)

| n | min z | median z | p95 z | max z | HOST_DRAW_Z_MAX | depth fraction |
|---|---|---|---|---|---|---|
| 1590 | 0.0164 | 0.4902 | 0.8065 | 1.1097 | 1.5 | **0.740** |

seed61000's kept population reaches 74.0% of nominal depth (vs seed3000's 68.3%) — deeper but still
a partial; 0.0% of events exceed 90% of the ceiling.

## Direct comparison: seed3000 vs seed61000, per seed3000's frozen M bins

| M bin | range (M☉) | seed3000 rate (n_to/denom) | seed61000 rate (n_to/denom) | diff σ |
|---|---|---|---|---|
| 0 | 3.55e4–7.76e4 | 0.873 (103/118) | 1.000 (309/309) | 1.20 |
| 1 | 7.76e4–1.70e5 | 0.785 (401/511) | 0.970 (163/168) | 2.13 |
| 2 | 1.70e5–3.70e5 | 0.271 (378/1395) | 0.458 (153/334) | **4.63** |
| 3 | 3.70e5–8.10e5 | 0.109 (245/2238) | 0.082 (99/1202) | 2.45 |
| 4 | 8.10e5–1.77e6 | 0.264 (69/261) | 0.241 (96/399) | 0.57 |

The dominant M-bin-1→2-style cliff reproduces in both runs at the same location; bin 2 (the
mid-mass shoulder) is itself 4.6σ higher in seed61000 than seed3000 at the same M edges — the
production pool loses a larger fraction of its mid-mass draws than the seed3000 partial suggested.
Bins 0/1 (low-N in seed3000, saturated in both) and bin 4 are consistent within ~1–2σ.

## Three-valued read against the registered band

Band: *flat within Poisson → NON-ISSUE with bound; >3σ gradient → new systematic.*

| axis | max gradient | band call |
|---|---|---|
| M | 14.35σ | **NEW-SYSTEMATIC-CANDIDATE** |
| e0 | 1.02σ | NON-ISSUE-WITH-BOUND |
| p0 | 13.58σ | **NEW-SYSTEMATIC-CANDIDATE** |
| 2-D (M, p0) | 5.97σ | **NEW-SYSTEMATIC-CANDIDATE** |
| **overall** | **14.35σ** | **NEW-SYSTEMATIC-CANDIDATE**, now carried by BOTH M and p0 (unlike seed3000, where p0 was individually flat) |

This upgrades the seed3000 read: at seed3000's bounded p0 prior, p0 alone was NON-ISSUE; at
seed61000's true (unbounded) production prior, p0 independently clears the 3σ band by a wide margin
(13.6σ), driven entirely by the p0>~20 tail (100% timeout, N=483) that the seed3000 prior could not
sample.

## Selection-effect note (FACTS only)

| M bin | frac of bin LOST to timeout | frac of kept (H0-inference) population in bin |
|---|---|---|
| 0 (1.15e4–4.74e4) | 100.0% | 0.0% |
| 1 (4.74e4–1.96e5) | 97.1% | 0.6% |
| 2 (1.96e5–8.07e5) | 14.4% | 80.4% |
| 3 (8.07e5–3.33e6) | 21.0% | 19.1% |
| 4 (3.33e6–1.38e7) | 100.0% | 0.0% |

The kept H0-inference population is overwhelmingly concentrated in M-bin 2 (80.4% of all 1590 kept
events), which is also the bin where the timeout loss is smallest (14.4%). The two extreme M bins
(0 and 4) contribute 0% of the kept population and lose 100% of their draws — the inference sample
never sees the very-low-mass or very-high-mass tails at all, by construction of this selection
effect, not by any property of those masses' astrophysical rate.

## Gaps

1. Raw injected-population CSVs excluded from the fetch (OPS_RECORD step 4, "NOT injections") — bin
   edges use kept+timeout union, same convention as seed3000, disclosed not substituted silently.
2. p0 prior changed between runs (retired [10,16] bound vs production unclamped plunge-window) — see
   Design gate; only M edges reused verbatim across runs, p0/e0 use fresh seed61000-native edges.
3. 0/100 simulate tasks logged a final "Skip tally" line (all hit walltime first) — worse than
   seed3000's 33/100; does not affect the binned rate (built from raw records).
4. Denominator excludes all non-timeout skip categories (no per-event params logged for them) — rate
   answers "of {timed out, succeeded}, what fraction timed out", not "of every draw".
5. CRB-stage timeout sample (n=2) descriptive only, not binned.
6. Other subtrees in the fetch (`real_r1..r5`, `sig0_control`, `zoom`, `estimatorB_2x2`, top-level
   mixed-stage app logs) belong to the same workspace but different jobs/variants — not analyzed.

## RECOMMENDATION (labelled as such — not a ruling)

Feed to the chair's disposition as: the production seed61000 pool **confirms and sharpens** the
seed3000 partial's M-axis finding (14.3σ vs 12.2σ; the M-bin-1→2 cliff reproduces at the same
location, with the mid-mass shoulder itself 4.6σ higher in production) and adds a **second
independent axis** — p0 — that clears the 3σ band only once the true unclamped production prior is
used (13.6σ, saturated 100% loss above p0≈20, N=483); seed3000's p0 axis read NON-ISSUE only because
its retired [10,16] prior could not sample that tail. Recommend NOT closing G7 row 8 as NON-ISSUE:
this is now a two-axis systematic candidate on the deeper (74% vs 68% z-depth), still-partial
production pool, with a disclosed selection-effect fact (the kept H0-inference sample is 80.4%
concentrated in the one M-bin with the lowest loss rate, and 0% in the two saturated-loss extreme
M-bins) worth carrying into any downstream systematics-budget write-up.

## CHAIR RE-DERIVATION + NOTE (2026-09-04 ~02:15)
Re-derived from the rate tables with binomial SEs (Garwood in the reader's tables; same direction,
larger σ here): M-axis gradients 3.0 / 62.8 / 2.9 / 38.0 σ across bins 0→4 (rates 1.00 / 0.971 /
0.144 / 0.210 / 1.00); p0-axis 5.8 / 0.7 / 8.5 / 34.5 σ (rates 0.232 / 0.095 / 0.083 / 0.288 / 1.00);
e0 flat (max 1.3σ). MATCH in direction and band (both > 3σ).
Facts for the decider (no ruling): the H0-inference population is the drawn population truncated
to M ∈ [~2e5, ~3.3e6] M☉ and p0 ≲ 20 by waveform timeouts (97–100 % loss outside; 80 % of kept
events in one M bin). Whether this biases H0 depends on the detection-probability model
(SimulationDetectionProbability builds p_det from the injection pool; if timeouts are dropped
rather than counted as non-detections, p_det is estimated on the truncated population) and on any
M/p0 dependence of the per-event H0 information — both are registrable questions (Graph 2
candidates: q-timeout-selection-pdet, q-timeout-population-mismatch). G7 row 8 cannot be closed
NON-ISSUE; it is now a two-axis systematic candidate on the production pool.
