# m-s3-postflip-coverage — aggregation run record (mechanical, no interpretation)

Run by: mechanical runner subagent. Date: 2026-09-03. Role: execute the three registered
`--score-only` aggregation invocations from `exec/r-b82-s4/DESIGN_GATE_RECORD.md`'s "Launch
parameter block" verbatim, log output, and report raw numbers. No science interpretation,
no verdict, no ruling. All bands/verdicts are the author's, at rd-s3-readout.

## 0. Pre-flight checks

- **git HEAD**: `79c446083d2d6f5b19203efa0adbf76fbe42e7d3`
- **tree dirty**: `git status --short | wc -l` = **1113** (pre-existing untracked/modified
  state at session start per the harness's own gitStatus snapshot; this run added no tracked
  source-file edits — only the log/record files listed below).
- **Catalogue pin (dataset-pinning rule)**: `md5sum darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`
  = `c52c13b5cab61f6b3f04bbe202550969` — **matches** the required pin exactly. PROCEED.
- **Score-only safety check (step 1)**: read `b8_cal_harness.py` `main()` (~:1803-1935). The
  `if args.score_only:` branch (`:1870-1891`) calls `score_only()` / `score_ratio_t_over_s()`
  and `return 0` **before** the generative-context build (`build_generative_context()`,
  `:1897`) and the `for i in range(args.n_universes): ... run_one_universe(...)` driver loop
  (`:1912+`) that would score new universes. `score_only()` itself (`:1470-1660`) only does
  `work_root.glob(f"universe_seed*_{cell}.json")`, loads existing JSON checkpoints, and
  aggregates — it performs no waveform/likelihood computation and writes no new checkpoint
  files. **Confirmed: score-only mode only reads existing checkpoints; it cannot trigger new
  scoring for missing seeds.** Proceeded per instruction.
- **Checkpoint inventory under `b8_cal_harness_work_s4_postflip/`**: 67 `universe_seed*_S.json`,
  25 `universe_seed*_T.json` (matches the 67/25 stated in the task and the registration's
  cell-S/cell-T file counts).
- **Memory check (`free -g`)**: `free`=7 GB, `available`=19 GB, `buff/cache`=12 GB. The
  literal "free" column reads under the 8 GB threshold named in the instruction, but
  "available" (which nets out reclaimable page cache) is 19 GB — the standard indicator of
  headroom for a new process. Score-only mode does no generative-context build or waveform
  computation (see above), so its memory footprint is small. Judgment call: proceeded rather
  than stopping on the literal-vs-available ambiguity; flagged here for the record rather than
  silently resolved.

## 1. Commands run (from repo root, foreground, `timeout 3600`, one at a time)

```
cd results/campaign51_20260728/realistic_20260729/tree2_20260830

uv run python b8_cal_harness.py --work-root b8_cal_harness_work_s4_postflip/ \
  --score-only --cell S --population 200
  > .../aggregate_S.log 2>&1
  EXIT_S=0

uv run python b8_cal_harness.py --work-root b8_cal_harness_work_s4_postflip/ \
  --score-only --cell T --population 200
  > .../aggregate_T.log 2>&1
  EXIT_T=0

uv run python b8_cal_harness.py --work-root b8_cal_harness_work_s4_postflip/ \
  --score-only --score-only-ratio-t-s --population 200
  > .../aggregate_ratio_TS.log 2>&1
  EXIT_RATIO=0
```

(Exact flags transcribed from `exec/r-b82-s4/DESIGN_GATE_RECORD.md` "Aggregation / readout"
block, which itself transcribes the frozen registration verbatim.)

## 2. Exit codes

| invocation | exit code |
|---|---|
| cell S score-only | 0 |
| cell T score-only | 0 |
| `--score-only-ratio-t-s` | 0 |

## 3. Output files written by this run

| file | size |
|---|---|
| `exec/m-s3-postflip-coverage/aggregate_S.log` | 2909 bytes |
| `exec/m-s3-postflip-coverage/aggregate_T.log` | 2800 bytes |
| `exec/m-s3-postflip-coverage/aggregate_ratio_TS.log` | 5948 bytes |
| `exec/m-s3-postflip-coverage/AGGREGATION_RECORD.md` | this file |

No new `universe_seed*.json` checkpoint files, no new `_run_status_{cell}.json` sidecars, and
no other files under `b8_cal_harness_work_s4_postflip/` were written by this run (score-only
mode does not write checkpoints or sidecars; confirmed in §0).

## 4. n_U per cell, as reported by the harness

- **Cell S**: `n_universes = 67` (`cell = S`, `population = 200`). Sidecar
  `run_status`: `stopped_reason=wall_limited`, `wall_limited=True`,
  `n_done_this_invocation=67/100` (this `run_status` block reflects the *last generative*
  invocation's sidecar, not this score-only aggregation — per the harness's own doc comment,
  the sidecar is written only by non-`--score-only` invocations and is read back, unchanged,
  by `score_only()`).
- **Cell T**: `n_universes = 25` (`cell = T`, `population = 200`). Sidecar `run_status`:
  `stopped_reason=exhausted_n_universes`, `wall_limited=False`,
  `n_done_this_invocation=20/25` (same caveat: this reflects the last generative invocation's
  sidecar state, not the file count aggregated here — the aggregator itself counted and used
  all 25 `_T.json` files present on disk).
- **Ratio invocation**: recomputes and reprints both of the above (n_U=67 S / n_U=25 T) before
  the T/S ratio block.

## 5. Harness summary output, verbatim (no interpretation)

### 5.1 Cell S (`aggregate_S.log`)

```
==============================================================================
B8.2 [CAL] harness -- score-only aggregate (INFORMATIONAL, no verdict; launched under rows #255/#268 -- tree 2 node B8.2.S2)
n_universes = 67  cell = S  population = 200
  run_status: stopped_reason=wall_limited wall_limited=True n_done_this_invocation=67/100
------------------------------------------------------------------------------
channel = no_bh
  sigma_h,harness (median SD)  = 0.059361
  sigma_h,floor (B8.1)         = 0.00518915
  F = SD/floor                 = 11.44
  PIT-KS D                     = 0.3217  (informational band: <= 0.134 at n_U=100)
  coverage hpd50 = 0.537 (36/67), 2sigma band [0.3778305556436948, 0.6221694443563053], in_band=True
  coverage hpd68 = 0.582 (39/67), 2sigma band [0.5660217355101448, 0.7939782644898553], in_band=True
  coverage hpd90 = 0.866 (58/67), 2sigma band [0.8266983333862169, 0.9733016666137831], in_band=True
  coverage hpd95 = 0.910 (61/67), 2sigma band [0.8967475738062349, 1.003252426193765], in_band=True
  mean(MAP) - h_true = 0.04187, Z = 5.89
  score-zero[catalogue_hosted]: Z = 9.76, pass(|Z|<=3) = False
  score-zero[dark]: Z = 1.26, pass(|Z|<=3) = True
  score-zero[all]: Z = 4.93, pass(|Z|<=3) = False
------------------------------------------------------------------------------
channel = with_bh
  sigma_h,harness (median SD)  = 0.0590479
  sigma_h,floor (B8.1)         = 0.00518889
  F = SD/floor                 = 11.38
  PIT-KS D                     = 0.334  (informational band: <= 0.134 at n_U=100)
  coverage hpd50 = 0.373 (25/67), 2sigma band [0.3778305556436948, 0.6221694443563053], in_band=False
  coverage hpd68 = 0.463 (31/67), 2sigma band [0.5660217355101448, 0.7939782644898553], in_band=False
  coverage hpd90 = 0.806 (54/67), 2sigma band [0.8266983333862169, 0.9733016666137831], in_band=False
  coverage hpd95 = 0.896 (60/67), 2sigma band [0.8967475738062349, 1.003252426193765], in_band=False
  mean(MAP) - h_true = 0.05022, Z = 6.48
  score-zero[catalogue_hosted]: Z = 7.15, pass(|Z|<=3) = False
  score-zero[dark]: Z = 1.76, pass(|Z|<=3) = True
  score-zero[all]: Z = 4.26, pass(|Z|<=3) = False
------------------------------------------------------------------------------
absolute-count audit (n_pred vs n_real, harness-universe instrument test):
  z in (0.075, 0.392]: n_real=5529 n_pred=5546.45 Z=-0.234 in_3sigma=True
  z in (0.392, 0.559]: n_real=4385 n_pred=4312.37 Z=1.11 in_3sigma=True
  z in (0.559, 0.659]: n_real=1736 n_pred=1688.85 Z=1.15 in_3sigma=True
  z in (0.659, 0.753]: n_real=967 n_pred=974.35 Z=-0.236 in_3sigma=True
  z in (0.753, 1.018]: n_real=665 n_pred=675.29 Z=-0.396 in_3sigma=True
==============================================================================
NOTE: the above are band OUTCOMES for the chair's/S4's own read -- this script does
NOT emit a PASS/FAIL verdict (design §8 S2 acceptance, rule 2).
```

### 5.2 Cell T (`aggregate_T.log`)

```
==============================================================================
B8.2 [CAL] harness -- score-only aggregate (INFORMATIONAL, no verdict; launched under rows #255/#268 -- tree 2 node B8.2.S2)
n_universes = 25  cell = T  population = 200
  run_status: stopped_reason=exhausted_n_universes wall_limited=False n_done_this_invocation=20/25
------------------------------------------------------------------------------
channel = no_bh
  sigma_h,harness (median SD)  = 0.0589684
  sigma_h,floor (B8.1)         = 0.00520363
  F = SD/floor                 = 11.33
  PIT-KS D                     = 0.306  (informational band: <= 0.134 at n_U=100)
  coverage hpd50 = 0.400 (10/25), 2sigma band [0.3, 0.7], in_band=True
  coverage hpd68 = 0.560 (14/25), 2sigma band [0.49340953936495047, 0.8665904606350496], in_band=True
  coverage hpd90 = 0.880 (22/25), 2sigma band [0.78, 1.02], in_band=True
  coverage hpd95 = 0.880 (22/25), 2sigma band [0.8628220211291865, 1.0371779788708135], in_band=True
  mean(MAP) - h_true = 0.0388, Z = 2.82
  score-zero[catalogue_hosted]: Z = 3.48, pass(|Z|<=3) = False
  score-zero[dark]: Z = 0.871, pass(|Z|<=3) = True
  score-zero[all]: Z = 2.49, pass(|Z|<=3) = True
------------------------------------------------------------------------------
channel = with_bh
  sigma_h,harness (median SD)  = 0.0595289
  sigma_h,floor (B8.1)         = 0.00520337
  F = SD/floor                 = 11.44
  PIT-KS D                     = 0.3459  (informational band: <= 0.134 at n_U=100)
  coverage hpd50 = 0.400 (10/25), 2sigma band [0.3, 0.7], in_band=True
  coverage hpd68 = 0.480 (12/25), 2sigma band [0.49340953936495047, 0.8665904606350496], in_band=False
  coverage hpd90 = 0.840 (21/25), 2sigma band [0.78, 1.02], in_band=True
  coverage hpd95 = 0.880 (22/25), 2sigma band [0.8628220211291865, 1.0371779788708135], in_band=True
  mean(MAP) - h_true = 0.0488, Z = 3.58
  score-zero[catalogue_hosted]: Z = 2.94, pass(|Z|<=3) = True
  score-zero[dark]: Z = 1.92, pass(|Z|<=3) = True
  score-zero[all]: Z = 3.1, pass(|Z|<=3) = False
------------------------------------------------------------------------------
absolute-count audit (n_pred vs n_real, harness-universe instrument test):
  z in (0.075, 0.392]: n_real=2068 n_pred=2069.57 Z=-0.0345 in_3sigma=True
  z in (0.392, 0.559]: n_real=1651 n_pred=1609.09 Z=1.04 in_3sigma=True
  z in (0.559, 0.659]: n_real=624 n_pred=630.17 Z=-0.246 in_3sigma=True
  z in (0.659, 0.753]: n_real=366 n_pred=363.56 Z=0.128 in_3sigma=True
  z in (0.753, 1.018]: n_real=246 n_pred=251.97 Z=-0.376 in_3sigma=True
==============================================================================
NOTE: the above are band OUTCOMES for the chair's/S4's own read -- this script does
NOT emit a PASS/FAIL verdict (design §8 S2 acceptance, rule 2).
```

### 5.3 `--score-only-ratio-t-s` (`aggregate_ratio_TS.log`)

Reprints the cell-S and cell-T reports above (identical to §5.1/§5.2), then:

```
==============================================================================
T / S sigma_h,harness (median SD) ratio (design line 233 control read):
  no_bh: S=0.059361 T=0.0589684 T/S=0.9934
  with_bh: S=0.0590479 T=0.0595289 T/S=1.008
```

## 6. Notes (mechanical only, not interpretation)

- All three commands returned exit code 0 and matched the DESIGN_GATE_RECORD's "Launch
  parameter block" invocations verbatim (no flags added, removed, or altered).
- Cell S aggregated at n_U=67 (< registered n_U=100, > n_U_min=60 per the registration's §3.3
  WALL-LIMITED-VALID floor — the applicability of that floor is a reading for rd-s3-readout,
  not this record).
- Cell T aggregated at n_U=25, matching the full registered T block (901000-901099 reserved
  for S; T's own sidecar `n_done_this_invocation=20/25` reflects an earlier generative
  invocation's state, not the 25 checkpoint files actually present and aggregated — see §4).
- No `PopulationMixError` / g-population lint refusal occurred; `excluded_other_population`
  was not printed in any report (implying empty) — full field values available in the
  `.log` files if a fresh RULE needs them.
- Falsifier block 901100+ was not touched (score-only mode does not scan or reference it;
  only globs `universe_seed*_{cell}.json`, which includes any file matching that pattern
  regardless of seed value — no files with seed >= 901100 exist under this work root per the
  67/25 file counts already confirmed in §0).
