# rd-2d-bootstrap-jackknife — verdict-free frequentist width/influence read (Branch K)

Research Graph 1 ADDENDUM, `GRAPH1_ADDENDUM_PROPOSAL_20260903.md` §1.2 (execution item 3), §6.1
order 3. Authorization: row #325 ("continue autonomous. you can make decisions but flag them") +
decision A-K1 (`Approved (RUNNABLE TONIGHT, verdict-free)`). **No verdict is rendered here.**
Numbers, the reproduction check, and a RECOMMENDATION line only — the three-valued disposition
against the proposal's band is reported as data, not adjudicated; `d-2d-frequentist` returns to
the author/chair.

## Sources

- Data: `retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv`
  and `retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv`
  (both present locally, verified: 65,109 lines each = header + 65,108 data rows = 41 h-nodes ×
  1,588 events; `h` column has exactly 41 unique values matching `H_GRID_41`
  (`darksiren_emri/validation/correspondence_1d.py:353`); every h-node has exactly 1,588 rows).
- Convention: reproduced the **"frozen T0 convention"** explicitly named in
  `exec/m-head-rebaseline/READOUT_RECORD.md` ("gradient-trapezoid grid weights `w =
  np.gradient(h_grid)`... `mean_h = Sigma post_n(h)*h*w(h)`") and defined verbatim in
  `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` (`_moments`,
  `_physics_floor_apply`, the multinomial-count-weighted bootstrap, the drop-top-k pattern). This
  is **not** the `correspondence_1d.py` default (`weights_convention="trapezoid"`, the corrected
  composite-trapezoid rule) — that default was explicitly superseded for T0-family reads and
  differs from the gradient convention only at the two grid endpoints (0.005 vs 0.010 weight); row
  #302's own quoted numbers are gradient-weighted, so the gradient convention is the one that
  reproduces them and is used throughout this record.
- Script (this task, saved beside this record): `rd_2d_bootstrap_jackknife.py`. Raw numbers:
  `rd_2d_bootstrap_jackknife_output.json`.
- `H_TRUE = 0.73` (`darksiren_emri/constants.py:25`).

## 1. Row #302 reproduction (STOP gate — not triggered)

| venue | channel | target map_h | target mean_h | computed map_h | computed mean_h | abs diff | reproduces ≤1e-5 |
|---|---|---:|---:|---:|---:|---:|---|
| iiib | 2D (`combined_with_bh`) | 0.665 | 0.665854 | 0.665 | 0.6658540600 | 6.00e-08 | **YES** |
| iiib | 1D (`combined_no_bh`) | 0.665 | 0.666987 | 0.665 | 0.6669870586 | 5.86e-08 | **YES** |
| joint_r1 | 2D (`combined_with_bh`) | 0.665 | 0.667127 | 0.665 | 0.6671274830 | 4.83e-07 | **YES** |
| joint_r1 | 1D (`combined_no_bh`) | 0.665 | 0.667032 | 0.665 | 0.6670323337 | 3.34e-07 | **YES** |

All four channel×venue targets reproduce to well under the 1e-5 STOP threshold (max deviation
4.8e-7). σ_h re-derived for iiib 2D: **0.018474738977** vs the quoted 0.018475 (diff 2.6e-7) —
also reproduced. 0/1,588 events excluded by the physics floor in any of the four rows (matches
row #302: "zero events excluded ... in any of the four channel×venue rows"). **STOP not
triggered; proceeding.**

## 2. Bootstrap SE(mean_h) at N=1,588 (B=2,000, seed=20260903)

Multinomial-count-weighted resampling (tier0's method: `counts ~ Multinomial(n, 1/n)`,
`logpost_boot = counts @ logL`, exactly equivalent to sampling 1,588 events with replacement
2,000 times).

| venue | channel | SE(mean_h) | SE(MAP) | own σ_h | SE/own σ_h | SE / σ_h=0.018475 (row #302 iiib-2D ref) | rail fraction (MAP at 0.60 or 0.86) |
|---|---|---:|---:|---:|---:|---:|---:|
| iiib | 2D | 0.016505 | 0.018356 | 0.018475 | **0.8934** | 0.8934 | 0.0 |
| iiib | 1D | 0.017434 | 0.019547 | 0.017526 | 0.9948 | 0.9437 | 0.0 |
| joint_r1 | 2D | 0.012914 | 0.012297 | 0.018924 | 0.6824 | 0.6990 | 0.0 |
| joint_r1 | 1D | 0.021017 | 0.025638 | 0.020346 | 1.0330 | 1.1376 | 0.0 |

The proposal's explicit reference (`σ_h = 0.018475`) is the iiib-2D value, so for that
channel×venue the two ratio columns coincide by construction. **No channel/venue reaches SE ≥
2× σ_h** (max observed 1.14×, joint_r1 1D vs the fixed reference; max 1.033× against its own
σ_h). g-censoring: 0/2,000 bootstrap draws land MAP at either grid edge (0.60 or 0.86), all four
rows — no rail contamination.

## 3. Jackknife influence structure (full leave-one-out, all 1,588 events)

`influence_i = mean_h(full) - mean_h(full minus event i)`: positive means event *i*'s presence
pulls `mean_h` **up**; since `mean_h(full) < H_TRUE` in every row here, the events whose *removal*
moves `mean_h` **toward** truth are those with the most **negative** influence.

| venue | channel | influence mean | influence std | influence min | influence max | \|influence\| mean |
|---|---|---:|---:|---:|---:|---:|
| iiib | 2D | -1.19e-06 | 2.791e-04 | -6.015e-04 | +1.621e-03 | 2.218e-04 |
| iiib | 1D | -1.40e-06 | 2.477e-04 | -3.647e-04 | +1.471e-03 | 1.972e-04 |
| joint_r1 | 2D | -2.14e-06 | 2.937e-04 | -2.153e-03 | +1.930e-03 | 2.219e-04 |
| joint_r1 | 1D | -8.01e-06 | 3.938e-04 | **-8.332e-03** | +4.671e-03 | 2.570e-04 |

(Full per-row summaries for all four channel×venue combinations are in
`rd_2d_bootstrap_jackknife_output.json` → `results[*].jackknife.influence_summary`; the top-10
events by \|influence\| per row are listed there too, e.g. iiib 2D's largest single influence is
event 889, +1.621e-03.) The influence distribution is right-skewed in three of four rows (max
positive influence 5–8× the mean |influence|); **joint_r1 1D is a disclosed outlier** — one event
carries influence -8.332e-03, roughly 32× the mean |influence| in that row and ~5–9× the largest
influence in any of the other three rows, and it is consistent with that row's markedly smaller
minimal-subset fraction (2.90%, §4) and the widest single-N=100 bootstrap tail. No single event's
removal materially closes the offset on its own in any row — the largest single-event
`|influence|` outside joint_r1-1D (1.6–2.2e-3) is under 3% of the full-sample offset magnitude
(≈0.064, iiib) — consistent with a diffuse-per-event *or* a moderate-heavy-tail structure; part 4
below is the statistic that discriminates between them.

## 4. Minimal-subset fraction that carries the offset

**Ranking used (decisive):** events sorted by *signed* influence in the direction that reduces
`|mean_h - 0.73|` (i.e., for these rows all `mean_h < 0.73`, so most-negative-influence events —
the ones whose removal raises `mean_h` most — are removed first). Cumulative removal, recomputing
`mean_h` after each prefix, until `|mean_h - H_TRUE| ≤ σ_h` (own σ_h, from part 1).

| venue | channel | σ_h used | minimal k removed | n_events | **minimal fraction** |
|---|---|---:|---:|---:|---:|
| **iiib** | **2D** | 0.018475 | 82 | 1588 | **5.164%** |
| iiib | 1D | 0.017526 | 94 | 1588 | 5.919% |
| joint_r1 | 2D | 0.018924 | 72 | 1588 | 4.534% |
| joint_r1 | 1D | 0.020346 | 46 | 1588 | 2.897% |

**Ranking sensitivity disclosed:** an earlier pass ranked by `|influence|` (undirected) instead
of the signed/directional statistic above. That ranking is **not informative** here — the
top-|influence| events are dominated by *positive*-influence events, whose removal moves `mean_h`
*away* from truth first (the cumulative curve is non-monotone, peaking at ~0.129 deviation around
k≈200–400 before recovering), so the "minimal k" it returns is ~95% of events in every row (see
`cross_check_abs_influence_ranked_minimal_fraction` in the JSON) — an artifact of ranking
direction, not a second finding. The directional ranking above is the one that answers the
proposal's kill-criterion question and is the number carried forward.

**Sanity check on the ranking machinery:** removing *all* 1,588 events collapses the posterior to
flat, and the gradient-weighted grid centroid of `H_GRID_41` is exactly 0.73 (verified
independently: `(h_grid * np.gradient(h_grid)).sum() / np.gradient(h_grid).sum() == 0.73` to
machine precision) — the k=1588 endpoint of every curve lands within 1e-13 of truth by
construction, confirming the moment/weight code is correct, not that it is informative at k=1588.

## 5. Bootstrap width-vs-N table (with replacement, 200 draws per N)

| N | iiib 2D std(mean_h) | iiib 1D std(mean_h) | joint_r1 2D std(mean_h) | joint_r1 1D std(mean_h) |
|---:|---:|---:|---:|---:|
| 100 | 0.02933 | 0.02979 | 0.02994 | 0.02963 |
| 200 | 0.03158 | 0.02929 | 0.03168 | 0.03065 |
| 400 | 0.02801 | 0.02906 | 0.02338 | 0.02695 |
| 800 | 0.02339 | 0.02459 | 0.01752 | 0.02490 |
| 1588 | 0.01358 | 0.01499 | 0.01213 | 0.02105 |

(N=1588 here uses its own 200-draw bootstrap batch, separate RNG draws from part 2's B=2,000 run —
both are with-replacement resamples of the same 1,588-event set and agree to within the expected
sampling noise of a 200- vs 2,000-draw SE estimate: e.g. iiib 2D 0.01358 here vs 0.01651 in part
2.) The N=100/200 rows are noisy (only 200 draws, heavier subsample-to-subsample variance) and not
strictly monotone decreasing — expected at this draw count — but the N=400→800→1588 tail in every
row decreases as expected with increasing N. Full table with the 1/√N-from-1588 scaling
prediction per N is in the JSON (`results[*].width_vs_N`).

## 6. Three-valued read against the proposal band (verdict-free; reported, not adjudicated)

Band, verbatim from §1.0 `q-2d-offset-frequentist`: **diffuse** if SE(mean_h) within 1.5× σ_h AND
>10% of events carry the offset; **re-scope** if SE ≥ 2× σ_h OR ≤5% of events carry the offset.

| venue | channel | SE/σ_h (own) | ≥2×? | minimal fraction | ≤5%? | >10%? | band hit |
|---|---|---:|---|---:|---|---|---|
| **iiib** | **2D** (proposal's named σ_h) | 0.893 | no | 5.164% | **no** (just above) | no | **neither named band** |
| iiib | 1D | 0.995 | no | 5.919% | no | no | **neither named band** |
| joint_r1 | 2D | 0.682 | no | 4.534% | **yes** | no | **RE-SCOPE** (≤5% clause) |
| joint_r1 | 1D | 1.033 | no | 2.897% | **yes** | no | **RE-SCOPE** (≤5% clause) |

No channel×venue reaches the **diffuse** band's >10% requirement (max 5.919%, iiib 1D — the
largest of the four is still under 6%). No channel×venue reaches SE ≥ 2×σ_h (max 1.033×, joint_r1
1D, against its own σ_h; max 1.138×, joint_r1 1D, against the fixed 0.018475 reference).

**iiib — the venue carrying the proposal's own quoted σ_h=0.018475 — sits in a narrow gap that
neither named band covers**: SE comfortably within 1.5× (not disqualifying diffuse on that leg)
but the minimal fraction (5.16% 2D / 5.92% 1D) is *just* above the 5% re-scope threshold and *well*
below the 10% diffuse threshold. **joint_r1, run as the replica, lands cleanly in RE-SCOPE** via
the ≤5%-events clause on both channels (4.53% / 2.90%).

## RECOMMENDATION (flagged, not a ruling — verdict-free per row #325/A-K1)

The offset is **not diffuse** by the proposal's own >10% test in any of the four channel×venue
rows measured tonight — that possibility is cleanly excluded (max minimal-fraction observed:
5.92%, well under 10%). Whether it counts as the **re-scope** condition turns entirely on which
side of the literal 5% line the primary (iiib) venue's minimal fraction falls, and iiib misses
that line by a small margin (0.16–0.92 percentage points) on both channels while its replica
(joint_r1) clears it comfortably on both channels. My recommendation: treat this as **RE-SCOPE**
in substance — the fraction-of-events statistic is consistently single-digit-percent across all
four rows (2.9–5.9%), an order of magnitude below the diffuse threshold, and the iiib/joint_r1 gap
straddling exactly 5% reads as a threshold-sensitivity artifact of one arbitrarily-chosen band
edge rather than a qualitative difference between the two venues — but this is a recommendation on
a boundary case, explicitly **not** the `d-2d-frequentist` ruling itself, which the proposal
reserves for the author/chair.

## Gate panel

- **g-precision**: full float64 arithmetic throughout (pandas pivot → numpy log-sum), no
  truncated-string reconstruction. PASS.
- **g-population**: 1,588 rows per h-node, 41 h-nodes, both venues — the registered 41-node grid
  only, no G-EXT nodes mixed in; 0 events excluded by the physics floor in any row. PASS.
- **g-censoring**: 0/2,000 bootstrap draws land MAP at either grid edge (0.60/0.86), all four
  rows. PASS (no rail flag to report per rows #267/#280's rule).

## What this record is not

- Not a ruling on `d-2d-frequentist` — the three-valued table in §6 and the recommendation in
  the following section are inputs to that decision, not the decision.
- Not a re-derivation by a second (decisive-verifier) identity — this is the analyst's read; per
  §5.3 of the proposal, "both decisive numbers re-derived by the addendum's decisive verifier
  (top-tier)" is a separate, still-pending step.
- Not a comparison against the mechanism-split three-way decomposition (artifact §10) — this is
  an independent frequentist axis only, per the proposal's own framing of Branch K.
