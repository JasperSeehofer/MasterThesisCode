# VERIFY-05: P_det Quadrature-Weight-Outside-Grid Summary

**Timestamp:** 20260423T172607Z
**Phase:** 40 Wave 3
**Requirement:** VERIFY-05

**Phase 41 trigger rule (D-18, W1 3-way verdict):**
  `mean_{h=0.73}(numerator)` lower bound (unlogged=0) vs borderline band:
  - `< 0.03` → NOT-TRIGGERED
  - `[0.03, 0.05]` → PHASE-41-TRIGGER-BORDERLINE (user decides)
  - `> 0.05` → PHASE-41-TRIGGERED

## Phase 41 Trigger Verdict

- `mean_{h=0.73}` all-events lower bound (unlogged=0) = **0.0409**
- Borderline band: [0.03, 0.05]
- **Verdict: PHASE-41-TRIGGER-BORDERLINE**
- Rationale: mean_{h=0.73}(numerator) lower bound = 0.0409 is inside borderline band [0.03, 0.05] — user decides

## Interpretation Note

STAT-04 logs a WARNING once per *host-galaxy call* where the d_L integration window
extends outside the P_det injection grid, only when numerator > 0.05 OR denominator > 0.05.
Events below that threshold are NOT captured. This means:

- **`n_rows_logged`**: total WARNING rows (one per host-galaxy call above threshold),
  NOT one per event. Events with many potential hosts generate many rows.
- **`logged_mean_numerator`**: mean over WARNING rows — biased HIGH (all rows are >5%).
- **`per_event_max_mean_numerator_lb`**: mean of per-event max(numerator) / N_events_total.
  Per-event max collapses all host-galaxy calls for one event into a single value.
  Unlogged events (all host calls <= 5%) contribute 0.0 — a conservative LOWER BOUND.
  **This is the D-18 decision variable.**

## Per-h Summary

| h | n_unique_events_num | n_rows_logged | logged_mean_num | logged_max_num | per_ev_max_mean_num_lb | per_ev_max_mean_den_lb |
|---|---------------------|---------------|-----------------|----------------|------------------------|------------------------|
| 0.6000 | 24 | 26053 | 0.3434 | 1.0000 | 0.0442 | 0.0524 |
| 0.6100 | 24 | 26053 | 0.3434 | 1.0000 | 0.0440 | 0.0524 |
| 0.6200 | 24 | 26053 | 0.3434 | 1.0000 | 0.0439 | 0.0524 |
| 0.6300 | 24 | 26053 | 0.3433 | 1.0000 | 0.0437 | 0.0524 |
| 0.6400 | 24 | 26053 | 0.3433 | 1.0000 | 0.0435 | 0.0524 |
| 0.6500 | 24 | 26053 | 0.3417 | 1.0000 | 0.0433 | 0.0524 |
| 0.6600 | 24 | 26053 | 0.3397 | 1.0000 | 0.0431 | 0.0524 |
| 0.6700 | 24 | 26053 | 0.3377 | 1.0000 | 0.0429 | 0.0524 |
| 0.6800 | 24 | 26053 | 0.3359 | 1.0000 | 0.0427 | 0.0524 |
| 0.6850 | 24 | 26053 | 0.3349 | 1.0000 | 0.0426 | 0.0524 |
| 0.6900 | 24 | 26053 | 0.3340 | 1.0000 | 0.0425 | 0.0524 |
| 0.6950 | 24 | 26053 | 0.3331 | 1.0000 | 0.0424 | 0.0524 |
| 0.7000 | 24 | 26053 | 0.3322 | 1.0000 | 0.0423 | 0.0524 |
| 0.7050 | 24 | 26053 | 0.3314 | 1.0000 | 0.0422 | 0.0524 |
| 0.7100 | 23 | 26053 | 0.3299 | 1.0000 | 0.0421 | 0.0524 |
| 0.7150 | 23 | 26053 | 0.3260 | 1.0000 | 0.0418 | 0.0524 |
| 0.7200 | 23 | 26053 | 0.3219 | 1.0000 | 0.0415 | 0.0524 |
| 0.7250 | 23 | 26053 | 0.3179 | 1.0000 | 0.0412 | 0.0524 |
| 0.7300 | 23 | 26053 | 0.3139 | 1.0000 | 0.0409 | 0.0524 |
| 0.7350 | 23 | 26053 | 0.3100 | 1.0000 | 0.0406 | 0.0524 |
| 0.7400 | 23 | 26053 | 0.3060 | 1.0000 | 0.0403 | 0.0524 |
| 0.7450 | 23 | 26053 | 0.3020 | 1.0000 | 0.0399 | 0.0524 |
| 0.7500 | 23 | 26053 | 0.2976 | 1.0000 | 0.0395 | 0.0524 |
| 0.7550 | 23 | 26053 | 0.2932 | 1.0000 | 0.0391 | 0.0524 |
| 0.7600 | 23 | 26053 | 0.2889 | 1.0000 | 0.0387 | 0.0524 |
| 0.7650 | 23 | 26053 | 0.2846 | 1.0000 | 0.0383 | 0.0524 |
| 0.7700 | 23 | 26053 | 0.2804 | 1.0000 | 0.0378 | 0.0524 |
| 0.7750 | 23 | 26053 | 0.2761 | 1.0000 | 0.0374 | 0.0524 |
| 0.7800 | 23 | 26053 | 0.2720 | 1.0000 | 0.0370 | 0.0524 |
| 0.7850 | 23 | 26053 | 0.2679 | 1.0000 | 0.0366 | 0.0524 |
| 0.7900 | 23 | 26053 | 0.2639 | 1.0000 | 0.0361 | 0.0524 |
| 0.8000 | 22 | 26053 | 0.2559 | 1.0000 | 0.0350 | 0.0524 |
| 0.8100 | 22 | 26053 | 0.2474 | 1.0000 | 0.0340 | 0.0524 |
| 0.8200 | 22 | 26053 | 0.2380 | 1.0000 | 0.0330 | 0.0524 |
| 0.8300 | 22 | 26053 | 0.2287 | 1.0000 | 0.0320 | 0.0524 |
| 0.8400 | 21 | 26052 | 0.2196 | 1.0000 | 0.0309 | 0.0524 |
| 0.8500 | 20 | 26052 | 0.2119 | 1.0000 | 0.0300 | 0.0524 |
| 0.8600 | 19 | 26051 | 0.2044 | 1.0000 | 0.0293 | 0.0524 |

## All-h Aggregate (Across All WARNING Rows, All h-Values)

- N_rows_logged = 990010
- Mean numerator across all rows = 0.2999
- Max numerator across all rows  = 1.0000
- Mean denominator across all rows = 0.2934
- Max denominator across all rows  = 1.0000
- N_events_total (from CRB CSV) = 542

## Histogram of Numerator Weights (10 bins on [0, 1], All WARNING Rows, All h)

| Bin range          | Count |
|--------------------|-------|
| [0.00, 0.10) | 655747 |
| [0.10, 0.20) | 4313 |
| [0.20, 0.30) | 2283 |
| [0.30, 0.40) | 5234 |
| [0.40, 0.50) | 12010 |
| [0.50, 0.60) | 14694 |
| [0.60, 0.70) | 20678 |
| [0.70, 0.80) | 17832 |
| [0.80, 0.90) | 21427 |
| [0.90, 1.00) | 235792 |

## Top-10 Dominant Events at h=0.73 (by Max Numerator Across Host Calls)

| event_idx | max_numerator | max_denominator | n_host_calls |
|-----------|---------------|-----------------|--------------|
| 2 | 1.0000 | 1.0000 | 49 |
| 90 | 1.0000 | 0.8380 | 7 |
| 161 | 1.0000 | 0.7220 | 43 |
| 105 | 1.0000 | 0.6220 | 83 |
| 107 | 1.0000 | 1.0000 | 36 |
| 115 | 1.0000 | 1.0000 | 2464 |
| 113 | 1.0000 | 0.5440 | 64 |
| 112 | 1.0000 | 1.0000 | 65 |
| 110 | 1.0000 | 0.4120 | 5 |
| 109 | 1.0000 | 0.4190 | 1 |

**Event 2 spotlight:** max_numerator=1.0000, max_denominator=1.0000, n_host_calls=49.
Event 2 was previously flagged in STATE.md Phase 38 as having 100% off-grid
numerator weight (injection grid coverage gap). Every potential host galaxy for
this event has its d_L integration window entirely outside the P_det grid.
If Phase 41 triggers, the densified injection grid should prioritize the
M x z x d_L region around this event's parameters.

## Artifacts

- Raw CSV: `.planning/debug/pdet_quadrature_raw_20260423T172607Z.csv`
- Parser driver: `.planning/debug/pdet_quadrature_parser_20260423T172607Z.py`
- Aggregator driver: `.planning/debug/pdet_quadrature_aggregator_20260423T172607Z.py`
- Machine-readable: `.planning/debug/pdet_quadrature_summary_20260423T172607Z.json`
