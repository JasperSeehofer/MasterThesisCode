# B8.2 S3 -- N-ladder + pilot cells (S, T) + score-only pass -- ARTIFACT READOUT RECORD

Reader-node record (no verdict, no adjudication). Author is Jasper Seehofer; this file was
produced by a READER agent whose job was to inventory artifacts and quote registered numbers
VERBATIM from the run output. Every number below is copied from either (a) the score-only
console block written into the run log, or (b) a checkpoint/verification JSON on disk, or (c)
a per-seed done-line in the log. Where a registered quantity is not present in any artifact it
is marked ABSENT -- no substitute was computed by this record.

Work root: results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder/
Harness: results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py
Run log: results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_s3_runner9_20260830.log
Design note (registration input): results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md
Floor JSON: results/campaign51_20260728/realistic_20260729/fanout1_20260829/b8_information_floor.json

---

## 1. Run provenance / stage timeline (from the log)

Stamps read directly from the "=== START/END" lines in b8_s3_runner9_20260830.log:

| Stage | Start | End | Wall (invocation total, from the log's own "total wall this invocation" line) |
|---|---|---|---|
| S3 ladder N=106 | 2026-08-30T22:00:51+02:00 | 2026-08-31T02:33:56+02:00 | 16383.5s |
| S3 ladder N=400 | 2026-08-31T02:33:56+02:00 | 2026-08-31T04:53:16+02:00 | 8358.4s |
| S3 ladder N=1588 | 2026-08-31T04:53:16+02:00 | 2026-08-31T11:40:42+02:00 | 24444.1s |
| S3 pilot cell S, n_U=100 requested, N=200 | 2026-08-31T11:40:42+02:00 | 2026-08-31T23:49:17+02:00 | 43712.5s |
| S3 pilot cell T, n_U=25 requested, N=200 | 2026-08-31T23:49:17+02:00 | 2026-09-01T03:52:41+02:00 | 14602.3s |
| S3 score-only | 2026-09-01T03:52:41+02:00 | 2026-09-01T03:52:42+02:00 | (aggregation only, ~1s) |

Every stage line reports "END rc=0" in the log.

Grid-split verification artifact (results/.../b8_cal_harness_work_ladder/_gridsplit_check_verified.json),
recorded at seed 900302 (the N=1588 ladder point), quoted verbatim:

    "verified_at_seed": 900302, "cell": "S", "bit_identical": true, "max_abs_diff": 0.0
    stamp.commit: b3f1767431b4715e301e78512a66122b1354a064
    stamp.branch: fix/p32d-classg-venue-repair

---

## 2. Artifact inventory

All paths relative to results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder/.

### 2.1 Top-level checkpoint JSONs (per-universe score inputs)

    universe_seed<seed>_<cell>.json   -- one per completed universe; glob pattern consumed by
                                         the harness's own score_only(work_root, cell).

Count: 86 files total.

| Seed range | Cell tag | N (n_draw_requested) | Count | Role |
|---|---|---|---|---|
| 900300 | S | 106 | 1 | N-ladder timing point |
| 900301 | S | 400 | 1 | N-ladder timing point |
| 900302 | S | 1588 | 1 | N-ladder timing point (also the grid-split-verified seed) |
| 901000-901062 | S | 200 | 63 | Pilot cell S universes (100 requested, 63 completed before --max-wall-s) |
| 902000-902019 | T | 200 | 20 | Pilot cell T universes (25 requested, 20 completed before --max-wall-s) |

CAVEAT (artifact-inventory fact, not a verdict): the harness's score_only() glob is
universe_seed*_{cell}.json, which does not distinguish N. The score-only pass that ran in
this log (cell S, default --cell) therefore pooled the 3 N-ladder timing seeds (N=106, 400,
1588) together with the 63 pilot seeds (N=200) into a single n_universes=66 aggregate. See
Section 4 caveats.

### 2.2 Per-universe working subdirectories

    seed<seed>_<cell>/harness.log
    seed<seed>_<cell>/selection_tables_h_0_<h>.json   (41 files per universe, one per H_GRID_41 point)

83 such subdirectories (63 seed90100x_S + 20 seed90200x_T; the 3 ladder-point seeds 900300/301/302
also have their own seed<seed>_S/ subdirectories at the top level, holding the same
harness.log + 41 selection-table structure).

Grid-split cross-check subdirectories (verification-only, seeds 900300/301/302, "_split" and
"_whole" variants): 6 directories, each containing harness.log + 41 selection_tables_h_0_<h>.json.

### 2.3 Shared caches

    draw_weight_cache/draw_weights_8aae9dfa6115f66ec6f173179595b658.npz   (1 file)
    precompute_cache/                                                     (22 entries)

### 2.4 Per-universe CSVs

ABSENT. No per-universe CSV output was found under the work root; per-universe results are
carried entirely in the universe_seed<seed>_<cell>.json checkpoints and the selection_tables_*
JSONs, not CSVs.

### 2.5 Score-only aggregate output

No separate JSON/file was written for the score-only pass; its only artifact is the console
block captured in b8_s3_runner9_20260830.log (lines ~27168-27210), reproduced verbatim in
Section 3. No cell=T score-only aggregate was run in this log (see Section 3.2 -- ABSENT).

---

## 3. Registered readout quantities, quoted verbatim

Registered definitions (source: B8_2_HARNESS_DESIGN_20260829.md and b8_cal_harness.py):
credible levels {50%, 68%, 90%, 95%} (HPD), PIT-KS D vs Uniform(0,1), mean(MAP) - h_true with
its Z, the score-zero-at-truth test by class {catalogue_hosted, dark, all}, F = sigma_h,harness
/ sigma_h,floor, and the absolute-count audit (n_pred vs n_real per z-bin). Registered
production-N (N=1588) floor values (b8_information_floor.json, dated 2026-08-29):
sigma_h_floor[no_bh] = 0.001747058397810697, sigma_h_floor[with_bh] = 0.001746970592930231
(the "0.001747" figure named in the task). The pilot ran at N=200, not N=1588, so
sigma_floor_for() reports an analytically rescaled floor (sqrt(N_ref/n_events)) for this pilot,
NOT the raw 0.001747 -- both numbers are quoted below, with the rescaling flagged as an
approximation per the function's own docstring, not a re-measurement.

### 3.1 Cell S -- score-only console block (n_universes = 66; SEE CAVEAT 4.1 -- mixed N)

    B8.2 [CAL] harness -- score-only aggregate (INFORMATIONAL, no verdict; launched under rows #255/#268 -- tree 2 node B8.2.S2)
    n_universes = 66  cell = S

    channel = no_bh
      sigma_h,harness (median SD)  = 0.0385344
      sigma_h,floor (B8.1)         = 0.00518915
      F = SD/floor                 = 7.426
      PIT-KS D                     = 0.8045  (informational band: <= 0.134 at n_U=100)
      coverage hpd50 = 0.015 (1/66), 2sigma band [0.37690850902066725, 0.6230914909793327], in_band=False
      coverage hpd68 = 0.015 (1/66), 2sigma band [0.5651615099895563, 0.7948384900104438], in_band=False
      coverage hpd90 = 0.061 (4/66), 2sigma band [0.8261451054124004, 0.9738548945875997], in_band=False
      coverage hpd95 = 0.121 (8/66), 2sigma band [0.8963456630011339, 1.003654336998866], in_band=False
      mean(MAP) - h_true = -0.12, Z = -52.3
      score-zero[catalogue_hosted]: Z = 0.124, pass(|Z|<=3) = True
      score-zero[dark]: Z = -30, pass(|Z|<=3) = False
      score-zero[all]: Z = -26.6, pass(|Z|<=3) = False

    channel = with_bh
      sigma_h,harness (median SD)  = 0.0588748
      sigma_h,floor (B8.1)         = 0.00518889
      F = SD/floor                 = 11.35
      PIT-KS D                     = 0.3313  (informational band: <= 0.134 at n_U=100)
      coverage hpd50 = 0.364 (24/66), 2sigma band [0.37690850902066725, 0.6230914909793327], in_band=False
      coverage hpd68 = 0.470 (31/66), 2sigma band [0.5651615099895563, 0.7948384900104438], in_band=False
      coverage hpd90 = 0.803 (53/66), 2sigma band [0.8261451054124004, 0.9738548945875997], in_band=False
      coverage hpd95 = 0.894 (59/66), 2sigma band [0.8963456630011339, 1.003654336998866], in_band=False
      mean(MAP) - h_true = 0.0478, Z = 6.12
      score-zero[catalogue_hosted]: Z = 7.77, pass(|Z|<=3) = False
      score-zero[dark]: Z = 0.709, pass(|Z|<=3) = True
      score-zero[all]: Z = 3.51, pass(|Z|<=3) = False

    absolute-count audit (n_pred vs n_real, harness-universe instrument test):
      z in (0.075, 0.392]: n_real=6072 n_pred=6082.05 Z=-0.129 in_3sigma=True
      z in (0.392, 0.559]: n_real=4807 n_pred=4728.80 Z=1.14 in_3sigma=True
      z in (0.559, 0.659]: n_real=1894 n_pred=1851.94 Z=0.977 in_3sigma=True
      z in (0.659, 0.753]: n_real=1058 n_pred=1068.44 Z=-0.319 in_3sigma=True
      z in (0.753, 1.018]: n_real=739 n_pred=740.50 Z=-0.0552 in_3sigma=True

Note printed by the harness itself, quoted verbatim: "the above are band OUTCOMES for the
chair's/S4's own read -- this script does NOT emit a PASS/FAIL verdict (design Sec.8 S2
acceptance, rule 2)."

n_U completed vs requested (cell S pilot only, seeds 901000-901062): 63 completed / 100
requested. The log line, quoted verbatim: "--max-wall-s (43200.0s) reached after 63
universe(s) this invocation; re-run the same command to resume (checkpoints already written
are skipped)." / "total wall this invocation: 43712.5s; 63 universe(s) scored."

### 3.2 Cell T -- score-only console block

ABSENT. The run log shows no invocation of --score-only --cell T; only --score-only
(default --cell S) was run (log lines ~27168 "START S3 score-only" through ~27210 "END rc=0",
reproduced whole in Section 3.1). The 20 completed cell-T checkpoints
(universe_seed902000_T.json .. universe_seed902019_T.json) exist on disk and could feed a
score_only(work_root, cell="T") aggregation, but no such aggregation was executed or recorded
by runner-9, so no coverage/PIT/F/score-zero numbers for cell T can be quoted. Per the task's
own instruction, no substitute aggregate was computed by this reader record.

n_U completed vs requested (cell T pilot, seeds 902000-902019): 20 completed / 25 requested.
The log line, quoted verbatim: "--max-wall-s (14400.0s) reached after 20 universe(s) this
invocation; re-run the same command to resume (checkpoints already written are skipped)." /
"total wall this invocation: 14602.3s; 20 universe(s) scored."

### 3.3 T0 / cell-T-as-control numbers

ABSENT for the same reason as 3.2. The design note (B8_2_HARNESS_DESIGN_20260829.md line 233)
registers that "cell T's ratio to cell S is reported so the production..." as an intended S4
input, but that ratio requires the cell-T score-only aggregate, which was not run.

### 3.4 N-ladder cost points (from per-seed done-lines in the log)

| Seed | N (n_draw_requested = n_realized_draw) | n_scored (no_bh) | n_catalogue_hosted | Wall time for this universe |
|---|---|---|---|---|
| 900300 | 106 | 94 | 9 | 16324.2s |
| 900301 | 400 | 352 | 18 | 8297.3s |
| 900302 | 1588 | 1339 | 95 | 24383.5s |

Quoted log lines, verbatim:

    seed 900300 cell S: done in 16324.2s -> .../universe_seed900300_S.json (n_scored no_bh=94, n_catalogue_hosted=9)
    seed 900301 cell S: done in 8297.3s -> .../universe_seed900301_S.json (n_scored no_bh=352, n_catalogue_hosted=18)
    seed 900302 cell S: done in 24383.5s -> .../universe_seed900302_S.json (n_scored no_bh=1339, n_catalogue_hosted=95)

CAVEAT: wall time is NOT monotonic in N (N=400 took less wall time, 8297.3s, than N=106,
16324.2s). This record does not attribute a cause (e.g. cache warm-up, system contention
between separate invocations) -- REPORTED-ONLY, no diagnosis performed.

### 3.5 Registered production-N floor values (context for the F candidate; not pilot output)

From b8_information_floor.json, quoted verbatim:

    oneD.GLADE_photo.closed_form.sigma_h_floor      = 0.001747058397810697   (no_bh channel)
    twoD.GLADE_photo.total_predictive_0.55dex.closed_form.sigma_h_floor = 0.001746970592930231  (with_bh channel)

These are the floors at N_ref=1588 (b8_cal_harness.py _FLOOR_N_REF = 1588). The pilot ran at
N=200, so the F values in Section 3.1 use the analytically rescaled floor (0.00518915 /
0.00518889, per sigma_h_floor(1588/median_n)^0.5), not the raw 0.001747 figures -- this
rescaling is flagged by the harness's own provenance note as "an ANALYTIC i.i.d.-Fisher-
information rescaling, not a re-measurement at this N" (b8_cal_harness.py:214-217).

---

## 4. Caveats (REPORTED-ONLY -- no verdict, no band comparison beyond quoting registered bands next to measured numbers)

4.1 **Mixed-N contamination in the cell-S score-only aggregate.** The score-only glob
universe_seed*_S.json pooled the 3 N-ladder timing universes (N=106, 400, 1588) together
with the 63 N=200 pilot universes into the single n_universes=66 aggregate quoted in Section
3.1. The coverage/PIT/F numbers in that block are therefore NOT a clean N=200 pilot-cell-S
readout; they mix four different N values. This is an artifact-provenance fact, not an
adjudication of whether it matters.

4.2 **Neither pilot cell reached its registered n_U.** Cell S completed 63/100 requested
universes; cell T completed 20/25 requested universes; both runs stopped on --max-wall-s, not
on completion. The design note's own operating-characteristics table (Section 4.1 of
B8_2_HARNESS_DESIGN_20260829.md: PIT-KS band <=0.134, coverage bands at [0.589,0.771] for 68%
etc.) is registered AT n_U=100. The coverage bands actually printed in Section 3.1 are
recomputed by binom_bands() at the REALIZED n (66, mixed-N as per 4.1), not at the registered
n_U=100 -- so the printed "band_2sigma" values differ from the design note's Section-4.1 table
values quoted above for the same nominal levels.

4.3 **Cell T aggregate and any T-vs-S ratio are ABSENT**, as detailed in 3.2/3.3. No
substitute was computed by this record.

4.4 Per the runbook-39 discipline named in the task: this record is REPORTED-ONLY -- bands are
quoted next to measured numbers strictly for orientation, no PASS/FAIL or CONSISTENT/DEFECT
verdict is stated anywhere in this file. The S4 registration review (per
B8_2_HARNESS_DESIGN_20260829.md Section 8, row "S4 registration") precedes any S5 production-N
launch; nothing in this record authorizes or blocks that review.

4.5 This record was produced by a reader-node agent (no source edits, no commits, no new
computation beyond reading existing JSON/log artifacts and one verbatim floor-JSON lookup via
python for confirmation of the two sigma_h_floor values quoted in 3.5).
