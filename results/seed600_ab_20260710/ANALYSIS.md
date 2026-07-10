# seed600 frozen-venue A/B — ANALYSIS (2026-07-10 → 07-11 overnight)

Handoff item L-B; plan of record [L2]. Inputs identical across all three code eras
(see README.md): seed600 prepared CRBs (3,375 rows), 80-CSV shallow pool
(`--allow_low_pdet_coverage`), live z_cmb catalogue, 17-pt grid 0.725–0.805, seed
600999, volume_deconv. Reference: the `562918ef` PV-test `run_live` artifacts
(2026-07-03/04). Combines: `--combine --allow_low_pdet_coverage` with each arm's own
code (combine needs the escape flag too — it rebuilds the D(h) survival grid).

## Headline numbers

| | 1D MAP | 1D mean | 2D MAP | 2D mean | n_events |
|---|---|---|---|---|---|
| run_live @`562918ef` (07-03) | 0.7450 | 0.74320 | 0.7850 | 0.78704 | 3,342 |
| run_A @`fc45d1f` (perf tip) | 0.7450 | **0.74320** | 0.7550 | **0.75455** | 3,342 |
| run_B @`f29a5e7` (#29 fallback) | 0.7450 | 0.74352 | 0.7550 | 0.75498 | **3,355** |

## Verdict 1 — code-drift gate (A vs run_live): 1D PASS exactly

- Combined 1D posterior: MAP and mean identical to 5 decimals; combine-level
  `D_h_per_h` identical to 1.1e-10 (both channels).
- Per-event 1D likelihoods (56,831 scalars = 3,342 events × 17 h): worst rel diff
  **2.64e-08** (h=0.755, event 1552); only 28 scalars (0.05%) above 1e-9 —
  consistent with the spline-table d_L (`afc59e9`) value-preservation tolerance.
  Key sets identical; the 13 zero-host events appear as empty entries in BOTH.

## Verdict 2 — the 2D difference is the DOCUMENTED `713fbd1` physics fix, not drift

`713fbd1` ([PHYSICS], Category B, on the perf branch / PR #31) replaced the with-BH-mass
MC selection denominator D_g (1–5% noise; up to **+54% wrong** for low-z wide-photo-z
hosts — it integrated unphysical z<0 mass) with an exact windowed semi-analytic erf-sum;
goldens were re-pinned in that commit (PIN_VD_BH_DEN 0.9131→0.9427).

**Measured venue-level impact (this A/B, identical inputs): 2D MAP 0.785 → 0.755, mean
0.78704 → 0.75455.** The open "2D +0.057" residual on this venue is **+0.0246 under
current code — the D_g fix removes 57% of it**, and the 2D MAP is now interior (the
0.805 grid-clip caveat weakens). The per-galaxy `galaxy_likelihoods` diagnostics differ
accordingly (that is where D_g lives); this is expected value movement, not corruption.

## Verdict 3 — #29 fallback real-data footprint (B vs A): exactly as designed

- **Hosts-present events bit-identical** in both channels (max rel diff 0.0 over all
  shared non-empty entries, all 17 h) — the #29 "hosts-present undisturbed" claim and
  the #30 z-caps production no-op both confirmed on the full venue.
- **Exactly 221 = 13 × 17 empty→filled flips per channel**: the 13 zero-host events
  (0.4% of 3,355) now contribute the pure-completion likelihood. The per-h host-lookup
  yield metric (first real-data appearance) logged `3342/3355 events with catalogue
  hosts, 13 pure-completion (zero-host) fallbacks` at every h.
- Combined-posterior footprint: **1D MAP unchanged (0.745), mean +0.00032** (6% of the
  0.005 grid step); 2D MAP unchanged (0.755), mean +0.00043. Sanity bound satisfied:
  at 0.4% completion fraction the fallback is a negligible perturbation — consistent
  with the L-A synthetic sweep, where bias onsets at completion fractions ≳0.2
  (`results/pp_coverage_deepvenue_20260710/SUMMARY.md`).
- Combine bookkeeping (for the Paper A 3,343-events correction): arm A reports
  `n_events_total=3342, n_events_empty=13` (1D; 2D shows 15 = 13 zero-host + 2
  empty-BH-mass entries). Arm B reports `n_events_total=3355, n_events_used=3353,
  n_events_excluded=2, n_events_empty=0` — i.e. of the 13 restored events, 2 are
  subsequently excluded by the combine's zero-handling (physics-floor) because their
  pure-completion likelihood underflows on this grid; the net gain is 11 contributing
  events. The z_cmb PV-config zero-host count is exactly 13 (yield metric, all 17 h).

## Provenance

- run_metadata.json in each arm dir stamps the exact code commit (verified:
  `fc45d1fd…` / `f29a5e77…`), grid, seed 600999, 8 workers.
- Runs executed from detached worktrees via `cwd/master_thesis_code` symlinks
  (worktrees removed after analysis; the stamped commits are the reference).
- Heavy artifacts (per-h JSONs ~8.5 GB/arm in the 2D channel, eval logs ~8 GB/arm)
  stay untracked per repo convention; committed here: README, ANALYSIS, run_metadata,
  and the four combined posteriors.
- Ω_m era reminder: this venue is A/B-only; its absolute residuals carry the (now
  measured, negligible) −0.0006 era term — see `results/seed600_omega_m_era_20260710/`.
