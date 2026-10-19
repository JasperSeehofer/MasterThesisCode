# m-head-rebaseline — verdict-free HEAD readout (both venues)

Research Graph 1, wave 2. Authorization: ledger row #301 (author ruling, verbatim: *"all
ratified, + 4a and 5a, did i miss some decisions?"*), item 5 option (A) — lifting the wave-1 read
embargo (see `c0prime_eval/GATE_RECORD.md` "RE-STAMP" section, this task's companion record).
**No verdicts, no band calls, no in/out-of-band adjudication.** Numbers and gate stamps only.

## Gate stamps consumed

- **g-c0-baseline: GREEN-AS-CORRECTED**, both venues, per `c0prime_eval/GATE_RECORD.md` RE-STAMP
  section (docket item 5(A), row #301, basis = row #299 chair re-verification: with-BH columns
  ndiff 0/1588 both venues against the flag-matched `c0prime_off` comparand; no-BH deltas are the
  registered `catalogue_leg_1d_mass_aware` flip on candidate-bearing events only, iiib 982/1588,
  joint_r1 1095/1588).
- **g-znorm:** not evaluated on this data. The flipped-leg identity check (`d-rphi-retire`, row
  #292) was performed on the 1D catalogue leg's divisor construction directly in code, not on a
  fresh run's diagnostics CSV; the retrieved head-rebaseline output does not carry a
  `global_denom_no_bh`/`global_denom_with_bh` pair of columns to re-derive the identity from. No
  g-znorm evaluation is offered here.

## Sources

- Data: `retrieved/run_20260902_graph1_headrebaseline_{iiib,joint_r1}/simulations/diagnostics/
  event_likelihoods.csv` (41 h-nodes × 1588 events each venue; commit `1ec9514d`, post-flip
  default; retrieval verified 0 mismatches / 13,288 files, row #300).
- Scorer: the frozen T0 convention (`results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py`
  docstring, P7-2a/P7-2b/P7-2c; same convention used for the wave-3 corrected readout,
  `wave3_20260830/WAVE3_A14_DELTA_READ_20260831.md` "CORRECTION NOTE"): per-event physics-floor
  zero handling (no-op here — zero zero-entries in either channel, both venues), `log(L_e(h))`,
  `Sigma_e log L_e(h)` per h (uniform prior), gradient-trapezoid grid weights
  `w = np.gradient(h_grid)` on the non-uniform 41-node H_GRID_41, `mean_h = Sigma post_n(h)*h*w(h)`,
  `MAP = argmax_h logpost(h)`.
- Re-derived directly from the retrieved CSVs by this task (not copied from any prior record).

## Readout table

| venue | channel | n_h | n_events | map_h | mean_h | sigma_h | posterior floor (min nonzero L, both channels) |
|---|---|---|---:|---:|---:|---:|---:|
| iiib | 2D (with-BH, `combined_with_bh`) | 41 | 1588 | 0.665 | 0.665854 | 0.018475 | 5.162188e-07 |
| iiib | 1D (no-BH, `combined_no_bh`) | 41 | 1588 | 0.665 | 0.666987 | 0.017526 | 4.036829e-06 |
| joint_r1 | 2D (with-BH, `combined_with_bh`) | 41 | 1588 | 0.665 | 0.667127 | 0.018924 | 8.898730e-07 |
| joint_r1 | 1D (no-BH, `combined_no_bh`) | 41 | 1588 | 0.665 | 0.667032 | 0.020346 | 5.060752e-06 |

No events excluded by the physics floor in any row (0/1588, all four channel×venue
combinations) — every per-event likelihood vector has at least one strictly positive entry across
the 41-node grid.

## Context note (not adjudication)

The ratified `d-jr1-band` design (docket item 3a, row #301) registers `map_h ∈ [0.64, 0.70] AND
mean_h ∈ [0.64, 0.70]` for the joint_r1 arm. Both joint_r1 numbers above (`map_h = 0.665`,
`mean_h = 0.667127`) fall inside that interval; both iiib numbers do as well. **This record makes
no in-band/out-of-band call** — that disposition belongs to `d-calibration` and the registered
`m-jr1-massaware` arm, which runs its own grid (H_GRID_41, conditionally G-EXT) and its own scorer
invocation, not this task's re-derivation from the head-rebaseline retrieval.

## What this record is not

- Not a comparison to the wave-3 blind HEAD readout or any other prior baseline — that is a
  delta-read, explicitly out of scope here (the embargo lift authorizes reading this data, not
  interpreting a delta against a comparand).
- Not an evaluation of `m-joint-r1-mass-aware`'s own registered band/verdict — that arm has not
  run under this task.
- No code edited, no commits, no cluster jobs, nothing awaited.
