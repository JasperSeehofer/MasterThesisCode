# G-1 / G-2 per-event diagnostics — rescued from `/tmp`, 2026-08-21

**Why these exist here.** Ledger row #145 item 8 records the G-1 STOP gate's historical PASS as
**UNSUPPORTED** — "no G-1 posterior, JSON or `event_likelihoods.csv` is banked anywhere under
`results/`". That was true of the repo, but three runs' diagnostics were still sitting in `/tmp`
scratch (1.7 GB of surrounding working files, since discarded). They are the **only** artifacts
from which G-1 can be recomputed without a full re-run, and `/tmp` does not survive a reboot.

| file | source | note |
|---|---|---|
| `g1_seed900001_event_likelihoods.csv` | `/tmp/correspondence_1d_g1/seed900001/simulations/diagnostics/` | **the G-1 null gate itself** (unity-completeness shim) |
| `g2_seed900101_event_likelihoods.csv` | `/tmp/correspondence_1d_g2/seed900101/simulations/diagnostics/` | G-2 cost pilot, b0 configuration |
| `g2_seed900102_event_likelihoods.csv` | `/tmp/correspondence_1d_g2/seed900102/simulations/diagnostics/` | G-2 cost pilot, b0 configuration |

Written 2026-08-19 22:22–23:28 by `correspondence_1d.py --stage g1 / --stage g2`. Note the grid is
**41 nodes** (`H_GRID_41`), not the 46-node `H_GRID_FULL` the arm fleet banks.

## What was already computed from them (rows #145, #146)

- **G-2 seed900101 reproduced the sentinel defect end-to-end on real per-event data:** exactly 2 of
  69 events are zero at *every* h-node, so all 41 nodes carry sentinel multiplicity k=2, `sum_log_l`
  is exactly constant, and the run reports the grid-midpoint artefact. This is the seed that
  confirmed the mechanism against banked `b0_seed900101.json` (`log_posterior` ≡ `-2e+300`).
- **G-1 seed900001 recomputed under production's registered strategies** (as-run vs corrected):

  | combine | n | mean_h | bias |
  |---|---|---|---|
  | as-run (`-1e300` sentinel) | 69 | 0.7300 | +0.0000 ← the recorded "PASS" |
  | `physics_floor` | 65 | 0.7350 | +0.0050 |
  | `per_event_floor` | 69 | 0.7350 | +0.0050 |
  | `exclude` | 27 | 0.6701 | −0.0599 |

  **42 of its 69 events are identically zero at every h** — with `f ≡ 1` the completion leg vanishes,
  so `g_frac` is undefined for 100% of B-F1/G-1 events (row #147 item 3) and any event without
  catalogue support is impossible under the arm's own model.

## Why G-1 is still UNSUPPORTED, not resolved

Its as-run signature (`map_h = 0.730`, `sigma_h = 0.0000`) is the **partial-mask** mode — the
posterior collapsed onto a single node — not the flat mode behind B-F1. So the "same mechanism as
B-F1" inference is unsupported, and these files are a *local* re-run of unknown correspondence to
whatever produced the historical G-1 verdict. **They enable a recomputation; they do not settle the
gate.** Treat any number derived from them as provenance-limited.
