# r-b82-s4 -- DESIGN-VALIDITY GATE RECORD

Node: `r-b82-s4` design gate. Research Graph 1, Branch A, wave 1.
Author of record for all scientific decisions: Jasper Seehofer.

## Authorization and scope

Ledger row #301: docket item 2 (d-s4-review) ratified -- `r-b82-s4`'s bands + stop rule are
**FROZEN as drafted** (`REGISTRATION_DRAFT.md`, unchanged by this record). Per graph §1.1
(`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` row `r-b82-s4`): "design gate red -> STOP m-s3
launch; stop-rule content returns as part of the d-s4-review fresh RULE." Per row 3 (row
#290): "m-s3 launches only after d-s4-review and a green design gate." This record is that
gate. **Blind by construction**: no post-flip `m-s3-postflip-coverage` output exists yet and
none was read to produce this record -- only the frozen registration text, the repair record,
the harness source (`tree2_20260830/b8_cal_harness.py`), the design of record
(`fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md`), and the graph spec were consulted.

Inputs read:
- `graph1_20260901/exec/r-b82-s4/REGISTRATION_DRAFT.md` (frozen)
- `graph1_20260901/exec/b-s4-harness-repair/RECORD.md`
- `tree2_20260830/b8_cal_harness.py` (repaired harness, row #291 state)
- `fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md`
- `graph1_20260901/RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.1 / §2

## Check 1 -- Executability: **GREEN, with two non-blocking documentation caveats**

The realized-n_U inputs a verdict needs (`n_universes`/`n_u`, per-channel `sd`, `pit`, `map_h`,
`hpd50/68/90/95`, `n_events_scored`, `score_at_truth.*`, `count_audit.per_bin.z`, `run_status`)
are all present in `score_only()`'s output (`b8_cal_harness.py:1470-1660`). Every §2.1 clause
is computable:

| §2.1 clause | harness source | verdict |
|---|---|---|
| PIT-KS D | `out[channel]["pit_ks_d"]` (`my_ks_uniform`, `:278-287`) | emitted directly |
| KS critical value at realized n_U | **not emitted as a general-n function** -- the harness hardcodes `pit_ks_band_informational: 0.134` (`:1622`), explicitly labeled "n_U=100" (comment) and matching the design doc's own fixed n=100 value (`B8_2_HARNESS_DESIGN_20260829.md:326`). At any n_U != 100 (the expected case under WALL-LIMITED-VALID, n_U in [60,99] for S / [16,24] for T) this field is **not the exact critical value** and must not be used as one. | **caveat, non-blocking**: `pit_ks_d` and `n_universes` (hence realized n_U) are both emitted; the exact critical value is a standard closed-form (e.g. `scipy.stats.kstwobign`/`kstwo`), external to but computable from these two harness outputs. Named per instruction: the harness itself emits no general-n exact-KS critical-value function -- only the fixed n=100 constant, informational. rd-s3-readout must compute the exact value externally at realized n_U rather than reuse `pit_ks_band_informational` off n=100. |
| HPD coverage 50/68/90/95, "exact Binomial(n_U, level) 2σ" | `binom_bands(level, n)` (`:290-293`) | computable at any n -- **but** `binom_bands()`'s own docstring names it a "2-sigma/3-sigma **normal-approximation**", not an exact discrete-binomial quantile. Reproducing the registration's own "at n_U=100" orientation column (0.402/0.598, 0.589/0.771, 0.841/0.959, 0.907/0.993) against a literal call `binom_bands(level,100)` gives (0.400/0.600, 0.5867/0.7733, 0.840/0.960, 0.9064/0.9936) -- close but **not identical** (offsets of 0.001-0.008), consistent with the orientation numbers having been computed by an exact/quantile method rather than by the cited `binom_bands()`. By contrast the §3 n_U_min=60 sanity claim ("68% band at n=60 is [0.56,0.80]") **does** reproduce `binom_bands(0.68,60)` = [0.5596,0.8004] to the quoted precision. So the same document mixes two different band-generation methods under one label. |
| `\|Z\|` mean(MAP) | `out[channel]["z_map"]`, `sem_map` (`:1554-1560`) | emitted directly; the registration's prose "SEM = σ̄_post/√n_U" is a slight mislabel -- the harness's `sem_map` is `std(MAP across universes, ddof=1)/√n_U` (an empirical MAP-scatter SEM), not `median(posterior SD)/√n_U`. The ready-made `z_map` field is what a reader should actually use; no code gap. |
| score-zero \|Z\| by class | `score_zero[cls]["z"]`, `pass_abs_z_le_3` (`:1587-1610`) | emitted directly, three classes present |
| count-audit per z-bin | `count_audit["per_bin"][b]["z"]`, `in_3sigma_informational` (`:1650`) | emitted directly |
| `F_no_bh = SD/floor(200)` | `out[channel]["F_dilution"]`, `sigma_floor_for()` (`:172-218`) | emitted; the quoted floor(200)=0.00518915 reproduces `sigma_h_floor[no_bh]=0.001747058397810697` rescaled by `sqrt(1588/median_n_events_scored)` (median_n≈180, not the raw N=200 draw count) -- verified consistent with `B8_2_S3_PILOT_READOUT_RECORD.md` line 207-208, which is exactly how `sigma_floor_for(channel, n_events)` is invoked (on the realized scored-event median, not the nominal N). Not a discrepancy. |

**Net**: nothing in §2.1 is uncomputable -- every clause has a source function callable at the
realized n_U. The two caveats above (KS critical value, and the "exact Binomial" label vs. the
normal-approximation `binom_bands()` actually implemented) are labeling/documentation-fidelity
issues in the frozen text, not missing statistics. They affect **rd-s3-readout**, which runs
after m-s3 generates checkpoints -- not the generative launch itself, which touches none of
these band functions. Routed as an action item for rd-s3-readout, not a launch blocker.

## Check 2 -- Stop-rule implementability: **GREEN**

Sidecar (`run_status_path`, `b8_cal_harness.py:1131-1143`, written `:1990-2012`, read back into
`score_only()`'s `run_status` block `:1697-1714`) actual field names:
`stopped_reason` (`"exhausted_n_universes"` | `"wall_limited"`), `n_universes_requested_this_invocation`,
`n_done_this_invocation`, `n_checkpoints_total_under_work_root`, `max_wall_s`,
`wall_elapsed_s_this_invocation`, plus `run_status.available` (bool) and `.wall_limited` (bool)
added by `score_only()`.

- §3.1 completion test (`stopped_reason == "exhausted_n_universes"` OR cumulative checkpoints >=
  registered n_U) -- both fields present, decidable.
- §3.2 resume-to-complete -- confirmed by code: `if ckpt_file.is_file(): skip` in the driver
  loop (`main()`), so re-running the same command is checkpoint-safe. The "≤3 invocations per
  cell" cap is **operator-tracked**, not harness-tracked (the sidecar is overwritten, not
  append-only, so it records only the latest invocation) -- expected and adequate: this is a
  launcher-script bookkeeping item, not a harness gap.
- §3.3 n_U_min floor (60/16) -- decidable from `n_checkpoints_total_under_work_root` (or
  `score_only`'s own `n_universes`) against the fixed thresholds.
- §3.5 `run_status.available == False` -> INSTRUMENT-DEFECT -- the field exists and is always
  set (`True` with data, or `False` with an explicit `reason` string, `:1710-1714`). One
  reading ambiguity, non-blocking: `available=False` is also the correct, benign state for a
  cell that has genuinely never been invoked yet (not yet launched), which is not itself a
  defect -- the clause is decidable either way from the emitted field, but the registration
  should be read as applying to a cell that *was* launched under this work root and shows no
  sidecar (crash before the final write), not to a not-yet-started cell.

## Check 3 -- Population/launch preconditions: **GREEN**

`--population INT` (`b8_cal_harness.py:1585-1591`), `PopulationMixError` +
`_population_tag()`(`:1454-1531`) confirmed present and exercised (RECORD.md: refuses the
86-file mixed pilot glob correctly, cleanly isolates `population=200`). All registration launch
flags map onto real, present CLI options (`argparse` block `:1803-1864`): `--work-root`,
`--N` (dest `n_draw`), `--cell {S,T}`, `--seed-block`, `--n-universes`, `--max-wall-s`,
`--population`, `--score-only`, `--score-only-ratio-t-s`. Seed math checks out: S block
901000-901099 is exactly 100 seeds (registered n_U=100); T block 902000-902024 is exactly 25
seeds (registered n_U=25); the falsifier reservation 901100+ does not overlap either block.
The fresh work root (`b8_cal_harness_work_s4_postflip/`, does not yet exist) discharges the
flag-half of the population tag by construction (no pre-flip checkpoint can appear there), so
the §1 open PROPOSED `_population_tag` amendment (item 4, routed to d-s4-review) is not a
launch precondition -- it is a hygiene improvement for the general case, not required here.

## Check 4 -- Byte-pin well-formedness (§2.3): **GREEN**

Checkpoint schema (`run_one_universe`, `:1170`, `:1426`) stores, per channel, `ln_post` (the
full ln-posterior vector), `sd`, `map_h`, `pit`, `hpd50/68/90/95`, `n_events_scored` -- i.e.
every quantity §2.3 names is present per-checkpoint, per-seed. Reference files exist and are
addressable: `tree2_20260830/b8_cal_harness_work_ladder/universe_seed{901000..901062}_S.json`
(63 files, pre-flip cell-S pilot) and `universe_seed{902000..902019}_T.json` (20 files,
pre-flip cell-T pilot) -- confirmed on disk, seed ranges match the registration's re-use
election exactly (cell S completed 901000-901062 of the registered 901000-901099 block; cell T
completed the full 902000-902019 of the registered 902000-902024 block). Comparison procedure
implied and mechanical: for each shared seed, diff the post-flip checkpoint's
`posterior.with_bh.{ln_post,sd,map_h,pit,hpd50,hpd68,hpd90,hpd95}` against the pre-flip
checkpoint of record at the same seed under `b8_cal_harness_work_ladder/`, same field paths.
Volume arithmetic checks out: (63+20) universes x ~200 events... 41 h-grid points per ln_post
vector -> 83 x ~200 x 41 ~= 6.8e5, matching the registration's own count. No comparison script
exists yet (this is readout-stage work, correctly not built by this repair per its own
scope note), but nothing about the procedure is under-specified -- reference files, fields, and
tolerance (byte-identical same-machine; <=1e-13 relative cross-machine, an amendment note) are
all named precisely enough to implement mechanically.

## Check 5 -- Blindness: **GREEN**

`REGISTRATION_DRAFT.md` §0 premise holds on inspection: every concrete number in the document
(F_no_bh=7.450, F_with_bh=11.38, F_no_bh(T)=11.27, T/S ratios 1.517/0.9984, cost figures
43712s/14602s) is sourced to row #288/#291 pre-flip pilot data or the design of record, and is
explicitly labeled "motivation and instrument anchors only, never calibration" for the no-BH
channel, or "structural, not a result" for the with-BH invariance argument. No post-flip
`m-s3-postflip-coverage` number appears anywhere in the draft (none exist to leak). No
result-dependent hedging or foreshadowing language was found in §§1-8.

## Check 6 -- Internal consistency: **GREEN, one minor completeness note**

§5's disposition table covers: all-in-band (COVERAGE-USABLE), any-out-of-band
(DEFECT-SIGNATURE), mixed/marginal (INTERMEDIATE), with-BH byte-pin red (INSTRUMENT-DEFECT),
INCOMPLETE-RUN (no read), g-population red (INSTRUMENT-DEFECT, re-aggregate). §3's
WALL-LIMITED-VALID / INCOMPLETE-RUN / INSTRUMENT-DEFECT trichotomy composes cleanly with §5 (a
WALL-LIMITED-VALID read still enters the §2.1 clause evaluation and lands in one of §5's rows;
INCOMPLETE-RUN and the two INSTRUMENT-DEFECT triggers are terminal and pre-empt §2.1 entirely).
No two clauses were found to conflict. Minor note (non-blocking): §5 does not state an explicit
precedence order when more than one gating condition could apply simultaneously (e.g. g-population
red discovered at the same time as a byte-pin check) -- the natural reading (instrument-level
gates such as byte-pin/g-population resolve before any §2.1 science clause is evaluated) is
consistent with every other section but is implicit rather than stated. Does not block launch.

## Overall verdict: **GREEN -- m-s3-postflip-coverage may launch**

No failing check. Two Check-1 caveats (KS critical value at realized n_U must be computed
externally, not read off the harness's fixed n=100 constant; the coverage-band label "exact
Binomial" does not match the normal-approximation `binom_bands()` the registration cites as its
source) are routed to **rd-s3-readout** as action items, since they affect readout-time band
computation, not the generative launch. They do not consume a revision of `r-b82-s4` (the bands
and stop rule are frozen as drafted per row #301; nothing here contradicts the frozen text or
requires it to change before data collection can start) -- they are implementation-fidelity
notes for whoever runs the §2.1 comparison.

## Launch parameter block (transcribed verbatim from the frozen registration; zero fresh choices)

Harness: `results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py`
(row #291 repaired state, chair-committed). Local CPU only, zero cluster (design §6 convention;
registration §7).

**Cell S** (no-BH primary + with-BH byte-pin leg):
```
python b8_cal_harness.py \
  --work-root b8_cal_harness_work_s4_postflip/ \
  --N 200 \
  --cell S \
  --seed-block 901000 \
  --n-universes 100 \
  --max-wall-s 86400
```
registered n_U = 100; n_U_min (WALL-LIMITED-VALID floor) = 60; falsifier block 901100+ stays
reserved/untouched; up to 3 invocations of the above (resume-to-complete; checkpoints skip
automatically).

**Cell T** (width-only + with-BH byte-pin leg):
```
python b8_cal_harness.py \
  --work-root b8_cal_harness_work_s4_postflip/ \
  --N 200 \
  --cell T \
  --seed-block 902000 \
  --n-universes 25 \
  --max-wall-s 86400
```
registered n_U = 25; n_U_min = 16; up to 3 invocations.

**Aggregation / readout (after generation, not part of launch)**:
```
python b8_cal_harness.py --work-root b8_cal_harness_work_s4_postflip/ --score-only --cell S --population 200
python b8_cal_harness.py --work-root b8_cal_harness_work_s4_postflip/ --score-only --cell T --population 200
python b8_cal_harness.py --work-root b8_cal_harness_work_s4_postflip/ --score-only --score-only-ratio-t-s --population 200
```

**Byte-pin reference set** (§2.3): compare against
`tree2_20260830/b8_cal_harness_work_ladder/universe_seed{seed}_{S,T}.json` for every shared
completed seed (S: 901000-901062, 63 files; T: 902000-902019, 20 files), fields
`posterior.with_bh.{ln_post,sd,map_h,pit,hpd50,hpd68,hpd90,hpd95}`.

**Not this launch's decision** (per registration §4/§8): PROD-A0 engagement-gate re-run,
comparand banking, and the population-tag `_population_tag` amendment all route elsewhere
(d-calibration / a small follow-on build item) and do not gate this launch.
