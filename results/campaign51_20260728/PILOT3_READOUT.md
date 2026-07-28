# Campaign #51 pilot #3 readout — corrected stack (2026-07-28)

**Setup.** Job 6073215, 60 A100 tasks × 100 events = 6,000 mixture
injections, seeds 54000–54059, stack `a9f29e8` (= bounds `ecb56d6` + mixture
`e110234`-era + PSD transfer fix `49251f3` + plunge-window ICs `e419062` +
separatrix-sign skip `a9f29e8`). All 60 tasks COMPLETED, zero tracebacks;
skips: 18 separatrix-sign (0.3 %), 50 timeouts (0.8 %); strata a/b/c =
2909/1562/1529; ≤ 8:06 wall/task. (Pilot #2, job 6073027, seed 53000, lost
22/60 tasks to the unhandled separatrix-sign ValueError — retired; pilot #1
readout remains QUARANTINED for physics.) NB pilot #3 CSVs lack the
`t_plunge_yr`/`p0` provenance columns (writer bug fixed `acaa0af`; optional
columns — rows remain valid pool members).

## Corrected-physics horizon structure (MEASURED)

| detector m bin | n | det | max d_hor [Gpc] | p90 d_hor |
|---|---|---|---|---|
| 5.8–6.0 | 513 | 141 | 7.08 | 5.36 |
| 6.0–6.2 | 498 | 199 | 7.77 | 5.35 |
| 6.2–6.4 | 409 | 151 | 6.52 | 4.47 |
| 6.4–6.6 | 486 | 200 | 5.32 | 3.16 |
| 6.6–6.8 | 538 | 122 | 2.81 | 1.17 |
| 6.8–7.0 | 448 | 16 | 1.37 | 0.61 |
| 7.0–7.2 | 133 | 0 | 0.63 | 0.47 |
| 7.2–7.4 | 39 | 0 | 0.33 | 0.24 |

1,426/6,000 detections; last detection at m = **6.960** (pilot #1's
artifact wall was 6.143). The old five-decade cliff is gone: the horizon
declines smoothly by ~1.4 decades over the last dex — genuine LISA band
physics. Cross-validation: measured p90 d_hor = 0.47 Gpc at 7.0–7.2 vs the
independent S_eff estimate 0.35–0.48 Gpc at M_z = 1e7
(`plunge_window/snr_seff_measurements.json`) — two routes agree. Low-M
detectability also rises strongly (19/39 det at 4.0–4.2 vs 1/34 in pilot
#1): plunge-window events are near-plunge and loud at every mass — the
selection function is globally new, as the convention change predicts.

## Pre-registered narrowing rule: **NOT VERIFIED — full band stands**

No candidate lg\* ∈ [6.0, 7.0] passes (every 0.2-dex bin above any
candidate still has max d_hor ≥ 0.33 Gpc ≫ the 2.5e−5 Gpc threshold).
Detections to m = 6.96 confirm Babak's "detectable to 10⁷ M☉ (spinning)"
on our own pipeline. The pilot #1 "narrowing verified at 10^6.5" was an
artifact of the two now-fixed problems, exactly as suspected in the
quarantine note.

## Campaign consequences

- Full band [1e4, 1e7] source frame FINAL (measured, not assumed).
- Bulk released on this stack: 3 sequential arrays × ~216 tasks × 300
  events (300, not 400: pilot walltime 8 min/100 events ⇒ ~22–24 min/task
  against the 30-min cap) → +194.4k rows → ~200k pool incl. pilot #3.
- The 81.4 % of catalogue rate-weight above the old support is now firmly
  inside the measured detectable band up to m ≈ 7.0 — the (g1) support
  limitation is resolved by data.

Provenance: `pilot3/injections/` (60 CSVs, local-only), `pilot3_decision.json`.
