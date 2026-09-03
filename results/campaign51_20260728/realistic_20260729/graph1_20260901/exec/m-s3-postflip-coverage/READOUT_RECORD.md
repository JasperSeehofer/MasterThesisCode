# rd-s3-readout — READOUT RECORD (verdict-free)

Reader role: this record reports band arithmetic only. It contains no PASS/FAIL beyond
reporting whether an observed value falls inside/outside a stated band, and no
recommendation. All rulings return to the author.

## 0. Header

| field | value |
|---|---|
| n_U, cell S | **67** (registered target 100; n_U_min floor for WALL-LIMITED-VALID = 60) |
| n_U, cell T | **25** (registered target 25 — COMPLETE) |
| stopped_reason, cell S | `wall_limited` (invocation 1 sidecar; row #333(3): wall 87016 s, no surviving process) |
| stopped_reason, cell T | `exhausted_n_universes` (row #326: invocation 2 completed the remaining 20 after invocation 1's 5-universe OOM crash, row #321) |
| Author ruling, row #333(4), quoted | **"Never invoke cell S again. Clean single-machine provenance, no 24h re-run, and it unblocks rd-s3-readout immediately. Uses 1 of 3 allowed invocations total."** — orchestrator-derived consequence: "cell S is CLOSED at n_U = 67 on a single machine; invocations 2 and 3 are forfeit by ruling, not exhausted; `rd-s3-readout` ... must report n_U = 67 (not the 100 of the registration's design target) with `stopped_reason: wall_limited` disclosed — the read is valid under the registered floor, not under the design target." |
| git HEAD (this readout) | `79c446083d2d6f5b19203efa0adbf76fbe42e7d3` (matches the HEAD recorded in `AGGREGATION_RECORD.md` §0; no commits since aggregation) |
| catalogue md5 pin status | `AGGREGATION_RECORD.md` §0: `md5sum darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv = c52c13b5cab61f6b3f04bbe202550969` — **"matches the required pin exactly. PROCEED."** (stated by the aggregation record; not independently re-derived against a separate pin source by this reader) |

## 1. g-population lint (checkpoint-level, this reader's own re-derivation)

Directly enumerated all 92 checkpoint files under
`tree2_20260830/b8_cal_harness_work_s4_postflip/`:

- **Cell S**: 67 files, seeds contiguous **901000–901066** (67 values, `901000 + i` for
  `i=0..66`). All 67 seeds fall inside the registered block 901000–901099. **0 seeds** at or
  above the falsifier reservation (901100+). **0 seeds** from the pilot ladder work root
  (`b8_cal_harness_work_ladder/`, a different directory entirely — no path collision possible).
- **Cell T**: 25 files, seeds contiguous **902000–902024** (25 values). All inside the
  registered block 902000–902024. **0 out-of-block seeds.**
- **Mixed rows**: **0.** Per-file `universe.cell` field checked against the file-name suffix
  for all 92 files: 0 mismatches.
- **Population (N=200) confirmed in every checkpoint**: `universe.n_draw_requested` read from
  all 92 files; the set of distinct values is `{200}` for both S and T — **no other population
  value present**.

**Lint result: CLEAN. 0 mixed rows.**

## 2. Table 1 — per cell × channel

Harness values are the `aggregate_{S,T}.log` / `aggregate_ratio_TS.log` outputs (re-quoted);
"reader-recomputed" values are independently re-derived by this reader directly from the 92
checkpoint JSONs (`posterior.{channel}.pit` for the exact-KS re-derivation; SD/floor values
re-multiplied for the F confirmation, §4 below).

### 2.1 Cell S (n_U = 67)

| stat | no_bh | with_bh |
|---|---|---|
| σ_h, harness (median SD) | 0.059361 | 0.0590479 |
| σ_h, floor (B8.1) | 0.00518915 | 0.00518889 |
| F = SD/floor (harness-quoted) | 11.44 | 11.38 |
| F, reader-recomputed to 4 s.f. (§4) | 11.4394 | 11.3797 |
| PIT-KS D (reader-recomputed from 67 `pit` values, exact match to harness's quoted D) | 0.321731 | 0.334016 |
| **Exact 2-sided Kolmogorov critical value at n_U=67, α=0.05** (`scipy.stats.kstwo.isf(0.05, 67)`, two-sided since D = max(D⁺,D⁻) per `my_ks_uniform`) | **0.163221** | 0.163221 |
| D exceeds exact critical value? | **YES** (0.3217 > 0.1632) | **YES** (0.3340 > 0.1632) |
| exact two-sided p-value (`scipy.stats.kstwo.sf(D, 67)`) | 1.14e-06 | 3.63e-07 |
| harness informational line replaced | was `<= 0.134 at n_U=100` — not applicable at realized n_U=67; superseded by the row above | same |
| HPD50 observed (k/n_U) | 0.537 (36/67) | 0.373 (25/67) |
| HPD50 registered-orientation band (r-b82-s4 §2.1, "at n_U=100" column, quoted): [0.402, 0.598] | **not directly applicable — different n_U**; see §3 rescale discussion | same |
| HPD50 harness normal-approx 2σ band at n_U=67 (`binom_bands`, NOT the registered "exact Binomial" label — Design Gate Check 1(b) caveat) | [0.3778, 0.6222], in_band=**True** | [0.3778, 0.6222], in_band=**False** |
| HPD50 reader-computed exact-discrete-binomial 2σ-equivalent band at n_U=67 (`scipy.stats.binom.ppf` at the 2.28%/97.72% quantiles — an independent construction, see §3) | [0.3731, 0.6269], **INSIDE** | [0.3731, 0.6269], **boundary/INSIDE** (0.373 = lower edge) |
| HPD68 observed | 0.582 (39/67) | 0.463 (31/67) |
| HPD68 normal-approx band | [0.5660, 0.7940], **True** | [0.5660, 0.7940], **False** |
| HPD68 exact-discrete band | [0.5672, 0.7910], **INSIDE** | [0.5672, 0.7910], **OUTSIDE** |
| HPD90 observed | 0.866 (58/67) | 0.806 (54/67) |
| HPD90 normal-approx band | [0.8267, 0.9733], **True** | [0.8267, 0.9733], **False** |
| HPD90 exact-discrete band | [0.8209, 0.9701], **INSIDE** | [0.8209, 0.9701], **OUTSIDE** |
| HPD95 observed | 0.910 (61/67) | 0.896 (60/67) |
| HPD95 normal-approx band | [0.8967, 1.0033], **True** | [0.8967, 1.0033], **False** (barely; 0.896 < 0.8967) |
| HPD95 exact-discrete band | [0.8955, 1.0], **INSIDE** | [0.8955, 1.0], **boundary/INSIDE** (0.896 ≈ 0.8955) |
| mean(MAP) − h_true | 0.04187 | 0.05022 |
| Z (mean-MAP) | 5.89 | 6.48 |
| score-zero Z, catalogue_hosted | 9.76, \|Z\|≤3 = **False** | 7.15, \|Z\|≤3 = **False** |
| score-zero Z, dark | 1.26, \|Z\|≤3 = **True** | 1.76, \|Z\|≤3 = **True** |
| score-zero Z, all | 4.93, \|Z\|≤3 = **False** | 4.26, \|Z\|≤3 = **False** |

### 2.2 Cell T (n_U = 25)

| stat | no_bh | with_bh |
|---|---|---|
| σ_h, harness (median SD) | 0.0589684 | 0.0595289 |
| σ_h, floor (B8.1) | 0.00520363 | 0.00520337 |
| F = SD/floor (harness-quoted) | 11.33 | 11.44 |
| F, reader-recomputed to 4 s.f. | 11.3322 | 11.4405 |
| PIT-KS D (reader-recomputed from 25 `pit` values, exact match to harness) | 0.306030 | 0.345870 |
| **Exact 2-sided Kolmogorov critical value at n_U=25, α=0.05** (`scipy.stats.kstwo.isf(0.05, 25)`) | **0.264041** | 0.264041 |
| D exceeds exact critical value? | **YES** (0.3060 > 0.2640) | **YES** (0.3459 > 0.2640) |
| exact two-sided p-value | 1.42e-02 | 3.56e-03 |
| HPD50 observed | 0.400 (10/25) | 0.400 (10/25) |
| HPD50 normal-approx band | [0.3, 0.7], **True** | [0.3, 0.7], **True** |
| HPD50 exact-discrete band | [0.32, 0.68], **INSIDE** | [0.32, 0.68], **INSIDE** |
| HPD68 observed | 0.560 (14/25) | 0.480 (12/25) |
| HPD68 normal-approx band | [0.4934, 0.8666], **True** | [0.4934, 0.8666], **False** (barely) |
| HPD68 exact-discrete band | [0.48, 0.84], **INSIDE** | [0.48, 0.84], **boundary/INSIDE** (0.48 = lower edge) |
| HPD90 observed | 0.880 (22/25) | 0.840 (21/25) |
| HPD90 normal-approx band | [0.78, 1.02], **True** | [0.78, 1.02], **True** |
| HPD90 exact-discrete band | [0.76, 1.0], **INSIDE** | [0.76, 1.0], **INSIDE** |
| HPD95 observed | 0.880 (22/25) | 0.880 (22/25) |
| HPD95 normal-approx band | [0.8628, 1.0372], **True** | [0.8628, 1.0372], **True** |
| HPD95 exact-discrete band | [0.84, 1.0], **INSIDE** | [0.84, 1.0], **INSIDE** |
| mean(MAP) − h_true | 0.0388 | 0.0488 |
| Z (mean-MAP) | 2.82 | 3.58 |
| score-zero Z, catalogue_hosted | 3.48, \|Z\|≤3 = **False** | 2.94, \|Z\|≤3 = **True** |
| score-zero Z, dark | 0.871, \|Z\|≤3 = **True** | 1.92, \|Z\|≤3 = **True** |
| score-zero Z, all | 2.49, \|Z\|≤3 = **True** | 3.1, \|Z\|≤3 = **False** |

**Cell T reminder (registration §2.2, carried verbatim):** no coverage/PIT/verdict claim is
ever registered from cell T — the HPD/KS rows above for T are reported for completeness
(they are computable, and the harness prints them) but were never registered bands for T; T's
one registered read is the SD (median) and the T/S ratio (Table 2).

**Count-audit per z-bin (both cells)**: all bins, both cells, `in_3sigma=True` — no anomaly
(quoted in full in `aggregate_S.log`/`aggregate_T.log` §5.1/§5.2 of `AGGREGATION_RECORD.md`).

## 3. The exact-KS and "exact Binomial" caveats — what changed vs. the harness's own line

Per `DESIGN_GATE_RECORD.md` Check 1 (routed to this readout, not consuming a revision):

1. **PIT-KS**: the harness's `pit_ks_ban_informational: 0.134` line is explicitly the fixed
   n_U=100 constant, invalid at n_U=67/25. This readout computed the genuine exact critical
   value at each realized n_U via `scipy.stats.kstwo.isf(0.05, n_U)` — the *two-sided*
   Kolmogorov distribution (`kstwo`, not `ksone`), matching `my_ks_uniform()`'s construction
   `D = max(D⁺, D⁻)`. Result: **D exceeds the exact critical value in all four cell×channel
   readings** (§2.1/§2.2 above), decisively (p ≤ 1.4e-2 in every case, ≤ 3.6e-7 in three of
   four).
2. **"Exact Binomial" HPD label**: `binom_bands()` is a documented normal approximation
   (`p ± 2·sqrt(p(1-p)/n)`), and its output at n_U=67/25 is what the harness printed and used
   for its `in_band` flags (reproduced exactly by this reader). The registration's own
   n_U=100 "orientation" column (e.g. HPD50 [0.402, 0.598]) does **not** reproduce from
   `binom_bands(0.5, 100)` = [0.400, 0.600] (Design Gate Check 1, quoted) — the generating
   method for that orientation column is undocumented and this reader could not identify it
   (tried: plain normal-approx — reproduces `binom_bands` exactly, not the orientation column;
   discrete-binomial 2.28%/97.72% quantile — also does not reproduce it, see below). Because
   the true "registered" band-generation method is unreproducible from anything in the
   repository, this reader supplies, alongside the harness's normal-approx band, an
   **independently constructed exact-discrete-binomial 2σ-equivalent band**
   (`scipy.stats.binom.ppf` at the 2.28%/97.72% quantiles, divided by n) at each realized
   n_U, labeled explicitly as reader-computed, not as a reproduction of the registration's
   orientation numbers. Sanity check: at n=100 this reader's discrete-quantile method gives
   HPD50 [0.40, 0.60] — matching `binom_bands(0.5,100)` to 2 d.p., and *still* not matching
   the registration's [0.402, 0.598] orientation column. **The registration's own n_U=100
   orientation numbers remain unreproduced by any formula in the harness or by this reader's
   attempt** — this is an open documentation gap, not resolved by this readout, carried
   forward exactly as the design gate flagged it.
3. Both caveats are read-time, not launch-time (Design Gate: "affects rd-s3-readout... not
   the generative launch").

## 4. g-precision — F = SD/floor reproduction to 4 s.f.

| cell/channel | SD | floor | F (recomputed) | harness-quoted F | match? |
|---|---|---|---|---|---|
| S no_bh | 0.059361 | 0.00518915 | 11.4394 | 11.44 | ✓ (rounds to harness's 2-d.p. figure) |
| S with_bh | 0.0590479 | 0.00518889 | 11.3797 | 11.38 | ✓ |
| T no_bh | 0.0589684 | 0.00520363 | 11.3322 | 11.33 | ✓ |
| T with_bh | 0.0595289 | 0.00520337 | 11.4405 | 11.44 | ✓ |

No rounding discrepancy beyond the harness's own 2-d.p. display truncation; all four confirm.

## 5. Table 2 — T/S ratio read vs. registered control band

| channel | S σ_h (median) | T σ_h (median) | T/S (harness) | registered control band | outcome |
|---|---|---|---|---|---|
| no_bh | 0.059361 | 0.0589684 | 0.9934 | **REPORTED-ONLY** — registration §2.2: "no-BH T/S: REPORTED-ONLY... the flip changes this channel unpredictably — banding it would smuggle pre-flip calibration back in." No band exists to compare against. | **NOT-EVALUABLE (by design — no band was ever registered for this channel)** |
| with_bh | 0.0590479 | 0.0595289 | 1.008 | §2.3: **pinned to reproduce the pre-flip pilot value 0.9984 exactly, contingent on the byte-identity check (§2.3) being green**, evaluated "on the completed-seed overlap" — i.e. this is not a free-standing band on the *n_U=67/25* aggregate T/S value, it is a byte-level per-seed identity claim | **NOT-EVALUABLE as stated** — the aggregate T/S=1.008 computed here is over the full 67 S / 25 T checkpoint sets, not restricted to "the completed-seed overlap" the pin actually specifies, and — per §6 below — no byte-pin comparison script exists and none was run. The observed 1.008 vs. pre-flip 0.9984 (Δ=0.0096) is reported as a plain fact, not scored against the pin, which was never executed. |

Pre-flip comparands, for context only (never a calibration reference per registration §0):
row #288 pilot (contaminated cell-S pool, n=63+3 mixed): F_no_bh=7.43, F_with_bh=11.35; row
#291 repaired clean read (n=63 S / n=20 T, N=200 population): F_no_bh(S)=7.450,
F_with_bh(S)=11.38, F_no_bh(T)=11.27, T/S ratios no_bh=1.517, with_bh=0.9984.

## 6. Every registered band condition from r-b82-s4 §2/§3 — observed value + outcome

| # | registered condition (r-b82-s4) | observed | outcome |
|---|---|---|---|
| 1 | §2.1 PIT-KS D (PRIMARY), cell S no_bh, "≤ exact 5% Kolmogorov critical value at realized n_U" | D=0.3217 vs exact crit 0.1632 (n_U=67) | **OUTSIDE** |
| 2 | same, cell S with_bh | D=0.3340 vs 0.1632 | **OUTSIDE** |
| 3 | §2.1 HPD50/68/90/95, cell S no_bh, "within exact Binomial(n_U,level) 2σ bands" | all 4 levels inside the normal-approx band (harness) and inside the reader's exact-discrete band (§2.1 table) | **INSIDE** (both constructions agree) — caveat: the "exact Binomial" registered formula itself is unreproduced, §3 |
| 4 | same, cell S with_bh | normal-approx: 4/4 OUTSIDE (HPD50 fails by 0.005, HPD68 fails by 0.10, HPD90 fails by 0.02, HPD95 fails by 0.001); exact-discrete: HPD50/HPD95 boundary-INSIDE, HPD68/HPD90 OUTSIDE | **OUTSIDE (normal-approx, harness-native reading); MIXED (2 boundary-inside/2 outside under the reader's exact-discrete construction)** — construction-dependent, flagged |
| 5 | §2.1 mean(MAP)−h_true, \|Z\|≤3, cell S no_bh | Z=5.89 | **OUTSIDE** |
| 6 | same, cell S with_bh | Z=6.48 | **OUTSIDE** |
| 7 | §2.1 score-zero \|Z\|≤3, cell S, class=catalogue_hosted, both channels | no_bh Z=9.76; with_bh Z=7.15 | **OUTSIDE** (both) |
| 8 | same, class=dark | no_bh Z=1.26; with_bh Z=1.76 | **INSIDE** (both) |
| 9 | same, class=all | no_bh Z=4.93; with_bh Z=4.26 | **OUTSIDE** (both) |
| 10 | §2.1 count-audit per z-bin, \|Z\|≤3 (harness instrument test) | all 5 bins, both cells, all \|Z\|<1.2 | **INSIDE** (all bins, both cells) |
| 11 | §2.1 F sanity flag, "F outside [1,25] → anomalous read, STOP" | S: 11.44/11.38; T: 11.33/11.44 | **INSIDE [1,25]** — no STOP triggered, all four values (§2.1/§2.2 F rows) |
| 12 | §2.2 cell-T coverage/PIT/verdict | (registration: "No coverage/PIT/verdict claim from cell T ever") | **NOT-EVALUABLE — no band was ever registered; this is a standing non-claim, not a gap** |
| 13 | §2.2 no_bh T/S ratio | T/S=0.9934 | **NOT-EVALUABLE — REPORTED-ONLY by design (Table 2)** |
| 14 | §2.3 with_bh T/S byte-pin (identity to pre-flip 0.9984 on the seed overlap, contingent on §2.3 byte-identity green) | aggregate T/S=1.008 (not restricted to seed-overlap; byte-pin itself never run) | **NOT-EVALUABLE — the underlying byte-identity comparison (§2.3/row 15) was never executed; no comparison script exists (Design Gate Check 4, confirmed still true at this readout — §7)** |
| 15 | §2.3 with-BH byte-identity pin itself (per-seed `ln_post`/`sd`/`map_h`/`pit`/`hpd*` vs. `b8_cal_harness_work_ladder/` reference checkpoints, byte-identical same-machine) | reference files confirmed present on disk (63 S: 901000–901062; 20 T: 902000–902019 — this reader independently re-confirmed both ranges, §0 of this record's working notes) | **NOT-EVALUABLE — no comparison was performed by any prior record; this readout does not perform it either (out of the reader's mechanical-comparison scope as instructed — flagged as an open item, not silently passed)** |
| 16 | §3.1 completion test | S: `stopped_reason=wall_limited`, 67 checkpoints (< registered 100, ≥ n_U_min=60) — WALL-LIMITED-VALID by rule, then CLOSED by row #333 author ruling; T: `stopped_reason=exhausted_n_universes`, 25/25 | S: **WALL-LIMITED-VALID (rule), then author-CLOSED at 67 (row #333, overriding further resume)**; T: **COMPLETE** |
| 17 | §3.3 n_U_min floor (60 S / 16 T) | S: 67 ≥ 60; T: 25 ≥ 16 | **INSIDE (floor cleared)**, both cells |
| 18 | §3.5 `run_status.available` | not False on either cell (both sidecars written and read; AGGREGATION_RECORD §4) | **INSIDE (no INSTRUMENT-DEFECT trigger)** |
| 19 | §1 g-population lint (launch precondition, standing) | 0 mixed rows, 0 `PopulationMixError`, `excluded_other_population` not printed (§1 of this record) | **INSIDE (green)** |
| 20 | §6 g-znorm spot check (`global_denom_no_bh == global_denom_with_bh` under resolved "on") | checkpoint JSON schema searched for any `*denom*` field: only `precompute_completion_denominator` / `precompute_missing_completion_denominator` are present — **no `global_denom_no_bh`/`global_denom_with_bh` pair exists in these checkpoints** | **NOT-EVALUABLE — the field this spot check needs is not emitted by the checkpoint schema** (same absence pattern independently noted for m-head-rebaseline, row #302) |
| 21 | §6 g-censoring rail-fraction disclosure (>10% at grid rail → bound, not measurement) | `h_grid` has 41 nodes per checkpoint (h_bounds present); a rail-fraction count was not computed by this reader — checkpoints carry `map_h` per universe but this record did not tabulate how many `map_h` values sit at the grid's first/last node | **NOT COMPUTED BY THIS READOUT — flagged, not evaluated** (scope: the task's explicit deliverable list did not name this statistic; noted here so a decider knows it is missing, not that it passed) |

## 7. Existence contract

| file/artifact relied on | status |
|---|---|
| `exec/r-b82-s4/REGISTRATION_DRAFT.md` | **present** |
| `exec/r-b82-s4/DESIGN_GATE_RECORD.md` | **present** |
| `gate_b_20260730/BIAS_HISTORY_LEDGER.md` rows #301, #303, #326, #333 (and #288, #291, #321 for comparand/context) | **present** |
| `exec/m-s3-postflip-coverage/AGGREGATION_RECORD.md` | **present** |
| `exec/m-s3-postflip-coverage/aggregate_S.log` | **present** |
| `exec/m-s3-postflip-coverage/aggregate_T.log` | **present** |
| `exec/m-s3-postflip-coverage/aggregate_ratio_TS.log` | **present** |
| `tree2_20260830/b8_cal_harness_work_s4_postflip/universe_seed{901000..901066}_S.json` (67 files) | **present, all 67, contiguous** |
| `tree2_20260830/b8_cal_harness_work_s4_postflip/universe_seed{902000..902024}_T.json` (25 files) | **present, all 25, contiguous** |
| `tree2_20260830/b8_cal_harness_work_ladder/universe_seed{901000..901062}_S.json` (63 files, byte-pin reference) | **present, all 63** |
| `tree2_20260830/b8_cal_harness_work_ladder/universe_seed{902000..902019}_T.json` (20 files, byte-pin reference) | **present, all 20** |
| `tree2_20260830/b8_cal_harness.py` (`my_ks_uniform`, `binom_bands` source) | **present** |
| `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` (catalogue pin) | **present** (pin match asserted by `AGGREGATION_RECORD.md`, not independently re-derived here) |
| byte-pin comparison script/output (§2.3 of the registration) | **absent** (never built — Design Gate Check 4 stated this explicitly at design time; still absent at this readout) |
| a documented derivation for the registration's n_U=100 HPD orientation column | **unreachable** — this reader attempted two candidate reconstructions (plain normal-approx, exact-discrete-binomial quantile) and neither reproduces the quoted [0.402/0.598, 0.589/0.771, 0.841/0.959, 0.907/0.993] figures; no third method is named anywhere in the repository |
| `global_denom_no_bh` / `global_denom_with_bh` fields (g-znorm spot check, §6 of registration) | **absent from the checkpoint schema** (confirmed by direct key search across the checkpoint JSON) |

## 8. What a decider would need to know (facts only, no recommendation)

1. Cell S is closed by explicit author ruling at n_U=67 (row #333), below the registered
   design target of 100 but above the WALL-LIMITED-VALID floor of 60; cell T is complete at
   the full registered 25.
2. On the PRIMARY registered statistic (PIT-KS D vs. the exact critical value at realized
   n_U), **both channels of both cells are OUTSIDE the band**, decisively: D exceeds the
   exact 5% critical value by roughly 2× in cell S (both channels) and by a smaller but still
   significant margin in cell T (p-values 1.1e-6 / 3.6e-7 / 1.4e-2 / 3.6e-3).
3. The HPD coverage picture is **channel-split**: cell S's no_bh channel sits inside its HPD
   bands at all 4 levels (both the harness's normal-approx band and this reader's
   exact-discrete-binomial reconstruction agree); cell S's with_bh channel sits outside 2–4
   of 4 levels depending on which band construction is used (harness normal-approx: outside
   at all 4; reader's exact-discrete reconstruction: 2 boundary-inside, 2 outside). Cell T
   shows all 8 (4 levels × 2 channels) inside under the normal-approx reading, all 8 inside
   or boundary-inside under the exact-discrete reading.
4. The registration's own literal band-generation formula for the n_U=100 HPD orientation
   numbers could not be reproduced by this reader from anything in the harness source; the
   in/out calls above rest on two reader-supplied constructions (harness normal-approx,
   reader exact-discrete-binomial), both disclosed and distinguished, neither confirmed as
   "the" registered method.
5. mean(MAP)−h_true is significantly non-zero (Z = 2.82 to 6.48 across all four cell×channel
   combinations) — every one of these Z-tests is OUTSIDE the registered \|Z\|≤3 band except
   cell T no_bh (Z=2.82, inside).
6. score-zero at truth fails (\|Z\|>3) for the catalogue_hosted and all classes in cell S
   (both channels) and for catalogue_hosted-with_bh / all-no_bh in cell T; the dark class
   passes in every case.
7. The F sanity flag (range [1,25]) does not trigger in any of the four cell×channel
   readings — F ranges 11.33–11.44, tightly clustered, close to (not identical to) the
   pre-flip pilot's with_bh anchors (11.35/11.27/11.38) but structurally different from the
   pre-flip no_bh anchor (7.43/7.450) since the registration's no-BH band is null-referenced,
   not pilot-referenced.
8. The with-BH channel's structural byte-identity claim (§2.3) — the mechanism the
   registration uses to argue the with-BH channel is a control, not a fresh measurement — was
   never executed as a comparison. Reference files for it exist on disk (verified present,
   83 files total) but no diff was run, by this record or any prior one. The with_bh T/S
   ratio observed here (1.008) differs from the pre-flip pilot's with_bh T/S (0.9984) by
   0.0096, but this comparison is over different, non-overlapping-by-construction aggregate
   populations (n_U=67/25 here vs. n_U=63/20 pre-flip), not the seed-matched byte-level
   comparison the registration actually specifies.
9. The g-znorm spot check named in registration §6 cannot be evaluated from these checkpoints
   — the fields it needs are not present in the checkpoint schema.
10. The g-population lint is clean: 0 mixed-population rows, correct seed blocks, N=200
    confirmed in all 92 checkpoints.
11. Disposition-table language (registration §5) is not applied by this record — it is the
    author's/d-calibration's to apply, not this reader's.

---
*Prepared by rd-s3-readout (reader role). No verdict rendered. All PASS/FAIL-shaped language
above reports registered-band arithmetic only, as instructed.*
