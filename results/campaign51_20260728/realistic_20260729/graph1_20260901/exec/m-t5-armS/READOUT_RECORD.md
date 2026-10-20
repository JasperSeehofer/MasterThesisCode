# m-t5-armS — verdict-free k-scan readout (iiib)

Research Graph 1, wave 2, Branch F. Authorization: ledger row #301 (docket item 5(A) lifts the
wave-1 embargo that this arm's comparand check was blocked behind). Design of record:
`tree2_20260830/PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md` §6.1 (Arm S). Launch:
`exec/m-t5-armS/LAUNCH_RECORD.md` (job 6764463, 16-task array). **No verdicts beyond the design's
own pre-registered, mechanical band assignment (IMMATERIAL-CONSISTENT-WITH-HB /
INTERMEDIATE / MATERIAL, and the scan-level MATERIAL-AT-SOME-k / ALL-SUB-MATERIAL rule) — those
are assigned per the design's own stated rules, not chair interpretation. Window adoption itself
stays with `d-t5-window`.**

## Gate stamps consumed

- **g-c0-baseline: GREEN-AS-CORRECTED**, per `m-head-rebaseline/c0prime_eval/GATE_RECORD.md`
  RE-STAMP section. Arm S's own comparand-check requirement (design §6.1: *"Baseline B: reused
  from the C0 gate task at zero compute [if] per-event `L_cat_with_bh`/`combined_with_bh`
  reproduced to <= 1e-12 relative... otherwise baseline is re-run at this arm's own nodes"*) is
  satisfied by the re-stamp: the flag-matched C0-prime comparand (`c0prime_off`) reproduces the
  banked `wave3_20260830/iiib` blind-HEAD baseline's with-BH channel to ndiff 0/1588 (exact),
  confirmed at row #299/row #301. Baseline B below is therefore the banked
  `wave3_20260830/iiib` HEAD readout at the H4 nodes {0.660, 0.665, 0.670, 0.730}, reused at zero
  compute, matching the same baseline C3 (the banked k=3 point) used.
- **g-znorm:** not evaluated. Same reasoning as the m-head-rebaseline record — the identity check
  operates on `global_denom_no_bh`/`global_denom_with_bh`, which are not columns in
  `event_likelihoods.csv`; no fresh evaluation is offered here.

## Sources

- Data: `retrieved/run_20260902_graph1_t5_armS_iiib/{k2_0,k2_5,k3_5,kinf}/simulations/
  diagnostics/event_likelihoods.csv` (each: 4 H4-node × 1588-event, 6352 rows; commit `1ec9514d`,
  `--catalogue_numerator_survival_2d off` held explicit per the design).
- Baseline B: `wave3_20260830/iiib/simulations/diagnostics/event_likelihoods.csv`, filtered to the
  same 4 H4 nodes (commit `1e092e82`, pre-flip, same run C3 itself used as baseline).
- Banked k=3.0 point: wave-2 job C3 (`wave2_20260829/c3/`), independently read out in
  `fanout1_20260829/B5_2_WIN_K3_READOUT_RECORD.md` — cited, not re-derived, per the design's
  explicit statement that k=3 "is a valid fourth point on the same curve" and is "NOT re-run"
  (`LAUNCH_RECORD.md`).
- Stencil formula and `I_HEAD = 2965`: design §6.1, reproduced from the C3 readout's own worked
  numbers (`Δℓ'(0.665) = 10.444` nats/h → `Δmean_h,pred = +0.003523`, verified by this task's
  script against the same baseline/stencil convention before being applied to the four fresh
  k-arms).

## Per-k table

`Δℓ(h) = Σ_events ln(combined_with_bh^armS(h) / combined_with_bh^baseline(h))` over events with
both baseline and arm-S values > 0; central-difference stencil over {0.660, 0.665, 0.670} at
spacing 0.005; `Δmean_h,pred = Δℓ'(0.665) / I_HEAD` (`I_HEAD = 2965`).

| k | Δℓ(0.660) | Δℓ(0.665) | Δℓ(0.670) | Δℓ(0.730) | Δℓ'(0.665) | Δℓ''(0.665) | Δmean_h,pred | band (design's own mechanical rule) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2.0 | −2.4108 | −2.3501 | −2.2603 | −0.1420 | 15.0420 | 1160.34 | **+0.005073** | INTERMEDIATE (0.003 < 0.005073 < 0.008) |
| 2.5 | +1.6563 | +1.7122 | +1.7664 | +2.3538 | 11.0095 | −69.10 | **+0.003713** | INTERMEDIATE (0.003 < 0.003713 < 0.008) |
| 3.0 (banked, C3) | +0.5442 | +0.5972 | +0.6486 | +1.2143 | 10.4440 | −63.7 | **+0.003523** | INTERMEDIATE (0.003 < 0.003523 < 0.008) |
| 3.5 | +5.2320 | +5.2146 | +5.1973 | +5.0470 | −3.4663 | +2.46 | **−0.001169** | IMMATERIAL-CONSISTENT-WITH-HB (\|Δ\| ≤ 0.003) |
| ∞ (no window) | +7.7316 | +7.6597 | +7.5886 | +6.7953 | −14.3026 | +33.32 | **−0.004824** | INTERMEDIATE (0.003 < 0.004824 < 0.008) |

**Scan-level disposition (design §6.1's own rule, applied mechanically):** no point reaches
`|Δmean_h,pred| ≥ T_mat = 0.008` → **ALL-SUB-MATERIAL**.

**Shape read (REPORTED-ONLY, per design §6.1):** `Δmean_h,pred(k)` is monotone decreasing across
the five points in ascending k order (2.0: +0.00507 → 2.5: +0.00371 → 3.0: +0.00352 → 3.5:
−0.00117 → ∞: −0.00482), consistent with the design's own zero-compute prediction of a monotone
decreasing dark-class-collapse effect.

## Gates (as the design requires; reported, not adjudicated)

**R6 — 1D bit-identity across k arms.** `combined_no_bh` compared pairwise across the four fresh
k arms (k=2.0 as reference), all 4 H4 nodes, 1588 events each: max_abs = **1.006e-16** for
k=2.5/3.5/∞ vs k=2.0 (floating-point noise, ≪ the 1e-12 registered band) — **PASS**, mass-window
geometry does not touch the 1D channel, confirmed directly (not merely assumed) across all four
fresh arms.

Note (disclosed, not adjudicated): `combined_no_bh` in each fresh k-arm differs from the
pre-flip `wave3_20260830/iiib` baseline by up to max_abs 0.011987 at h=0.730 (982/1588 events) —
this is the same row #286 `catalogue_leg_1d_mass_aware` flip signature already documented in the
GATE_RECORD RE-STAMP (Arm S runs at commit `1ec9514d`, post-flip; the baseline predates the flip).
It is orthogonal to the k-scan itself (R6 above shows the 1D channel is flat across k) and is
reported here for completeness, not folded into any band call.

**R2 — engagement (fraction of baseline-non-empty with-BH events with a changed `L_cat_with_bh`
at h = 0.730).** Baseline non-empty count: 982/1588. Measured, all four k arms:

| k | changed / non-empty | fraction |
|---|---|---:|
| 2.0 | 982/982 | 1.0000 |
| 2.5 | 982/982 | 1.0000 |
| 3.5 | 982/982 | 1.0000 |
| ∞ | 982/982 | 1.0000 |

All four exceed the registered ≥0.90 threshold. Disclosed caveat: because the arm-S runs are
post-flip and the baseline is pre-flip, the row #286 `catalogue_leg_1d_mass_aware` coupling (§6a
of `FORENSICS_RECORD.md`: the with-BH host batch's slot-0 return is also a no-BH-leg numerator,
flag-sensitive) means every candidate-bearing event's `L_cat_with_bh` slot moves for a reason
unrelated to the mass-window geometry as well — so 100% engagement at every k does not by itself
discriminate the window effect from the flip. This is reported as a measurement, not adjudicated.

**R5 — stencil validity (`|Δℓ''(0.665)| ≪ I_HEAD = 2965`).**

| k | Δℓ'' | \|Δℓ''\| / I_HEAD |
|---|---:|---:|
| 2.0 | 1160.34 | 39.1% |
| 2.5 | −69.10 | 2.3% |
| 3.0 (banked) | −63.7 | 2.1% |
| 3.5 | +2.46 | 0.08% |
| ∞ | +33.32 | 1.1% |

k=2.0's ratio (39.1%) is far larger than the other four points (≤2.3%) — reported as a plain
measurement; the design's own escalation rule (G27) on an ambiguous R5 read is not invoked or
adjudicated by this record.

## What this record is not

- Not a ruling on window adoption — that is `d-t5-window`'s scope, explicitly reserved by the
  design (§6.1: "adoption of any k returns to the author with the curve").
- Not an evaluation of Arm R (joint_r1) — out of scope for this launch (`LAUNCH_RECORD.md`: "Arm R
  ... explicitly NOT launched").
- No code edited, no commits, no cluster jobs, nothing awaited.
