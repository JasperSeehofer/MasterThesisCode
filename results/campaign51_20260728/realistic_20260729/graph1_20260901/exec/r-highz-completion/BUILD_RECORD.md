# BUILD_RECORD.md -- r-highz-completion, builder b-highz-decomp

Sonnet/medium builder, 2026-09-04. Implements `REGISTRATION_DRAFT.md` + `MECHANISM_NOTE.md` exactly
in `highz_decomp_reads.py`, per `DESIGN_GATE_computability.md` (GREEN verdict, 3 AMBER findings, all
resolved below). `INFORMATION_FORECAST.md` was not opened (forbidden per task). This builder never
ran the script outside `--dry-run` (real inputs) and `--synth` (the <=10-row synthetic fixture) --
**real mode (`--out` written) was never executed by this builder**, per the task instruction; it is
reserved for the disjoint reader (`REGISTRATION_DRAFT.md` Sec.3/Sec.8).

## 1. Design-gate findings resolved

- **Finding A (AMBER, harness manifest sha256 underspecified):** reproduced by trying variants of
  the "per-file md5 manifest" construction against the real 67x2 files. The match: sorted
  `"{seed} {md5}"` lines of the **`event_likelihoods.csv` md5 only** (the CRB file's md5 is NOT part
  of the manifest), joined by `"\n"`, no trailing newline, sha256'd. Verified byte-exact against the
  pinned `6a06063dd5...adb1c0a2` on the real tree. Implemented as `harness_manifest_hash()` (script
  line ~216) and exercised in `--dry-run` (see Sec.4 below) -- resolves the AMBER with a working,
  checked-in construction rather than prose.
- **Finding B (AMBER, no `--nonadditivity-max` CLI flag):** added `--nonadditivity-max` (default
  `0.6`, matching the frozen prose value) to the argparser, alongside the other four band flags.
  The launch block in `REGISTRATION_DRAFT.md` Sec.8 does not pass it explicitly, so the script's
  default carries the frozen `0.6` -- visible in the run's own argv/metadata as the other four bands
  already are, per the design gate's recommendation.
- **Finding C (informational, float precision):** `load_term_columns()` and `_five_row_slice_closure()`
  read the full-precision columns with `pd.read_csv(..., float_precision="round_trip")`. The dry-run's
  5-row real-slice closure residual reproduced **2.665e-15** -- byte-identical to
  `DESIGN_GATE_computability.md` Sec.3's own reproduction (max 2.665e-15), confirming the exact route.
- **Finding D (informational, resolved-flags equality):** `verify_g3d_resolved_flags()` does NOT diff
  against production's raw `cli_args` (which would spuriously fail on `catalogue_numerator_survival`
  and `mass_filter_sigma`, per the design gate). It instead asserts internal agreement of the 13
  `resolved_flags` tokens across all 67 harness checkpoints -- exactly what "13 tokens, 67/67" means
  as a **re-assertion from the checkpoints** (`REGISTRATION_DRAFT.md` Sec.6 G-3d wording), not a
  cross-venue CLI-string comparison.

## 2. Own-motion bug found and fixed during the build

`discover_harness_universes()` initially took the "actually-scored" count from the checkpoint's
`universe.n_scored` field. That field is identically `200` for every universe here (it mirrors
`n_draw_requested`/`n_realized_draw`, not the number of events that survived to get a scored row).
The correct count is the diagnostics CSV's own unique `event_idx` count (173-192 per universe,
matching the checkpoint's `posterior.no_bh.n_events_scored`). Caught because the first dry-run printed
`Sigma n_scored=13400` against the G-2(iv) anchor of `12,060` (`200*67=13400` is the tell). Fixed to
read `event_idx.nunique()` from the CSV; the second dry-run reproduced `12,060` exactly (see Sec.4).

## 3. CHECKLIST TABLE (draft item -> function/line, `highz_decomp_reads.py`)

| draft item | function(s) | ~line |
|---|---|---|
| CLI paths + pins, every input (Sec.1/Sec.8) | `build_argparser`, `LOGL_MD5`/`TABLE_SHA256`/`HARNESS_MANIFEST_SHA256`/`POPULATION_SHA256`/`POPULATION_N` constants | 79-227 |
| hard INSTRUMENT-DEFECT pre-flight | `InstrumentDefect`, `preflight` | 172, 266 |
| pin verification, STOP on mismatch | `verify_file_pins` | 274 |
| population construction (P_dark, K, K_dark, K_hosted, R) | `Populations`, `construct_populations` | 292, 308 |
| population sha256 g-byteid gate | `verify_population_pins` | 344 |
| G-3a set-identity (C7==0 == C2==False == C3c_censored) | `verify_g3a_set_identity` | 375 |
| G-3b (K identical both venues) | `main` (equality check post-construction) | ~1312 |
| harness universe discovery (`n_draw_requested==population` filter, 67) | `HarnessUniverse`, `discover_harness_universes` | 389, 399 |
| harness manifest sha256 (Finding A resolved) | `harness_manifest_hash`, `verify_harness_manifest` | 216, 441 |
| G-3d resolved-flags equality (13 tokens, 67/67; Finding D resolved) | `verify_g3d_resolved_flags` | 450 |
| harness population construction per universe (P_dark,u / K_u / K_dark,u / R_u) | `_harness_z`, `construct_harness_populations` | 1184, 1195 |
| G-2(iv) harness pooled byte-id anchors (12,060/4,826/1,207/1,148) | `main` (`pooled_sizes` + `HARNESS_POOLED_ANCHORS` check) | ~1377 |
| G-1 closure gate (a)-(e) | `gate_g1_closure` | 484 |
| term profiles T_B, T_g (full-precision columns only) | `load_term_columns`, `compute_term_profiles` | 472, 531 |
| T_D (event-common, identity Delta_D=0) | `compute_T_D`; `TermFreezeResult.delta_D` fixed at 0.0 | 549, 598-605 |
| centered profiles t_hat_e(h) | `center_profile` | 567 |
| reference profile t_bar(h) = median over R | `reference_profile` | 573 |
| term-freeze counterfactual Lambda_t, Delta_t, Delta_F, closure r, shares s_t | `term_freeze_lambda`, `delta_t`, `run_term_freeze`, `TermFreezeResult` | 578-643 |
| null draws (1000, seed 20260904, CI99 of Delta_F) | `null_draw_ci99` | 645 |
| per-term score excess (stencil 0.725/0.730/0.735, Welch SE) | `stencil_slope`, `welch_se`, `score_excess` | 677-707 |
| harness pooled S_t/S_F with between-universe jackknife SE | `harness_pool_score`, `HarnessPoolResult` | 709-751 |
| production ownership disposition (3-valued + Z-DIFFERENTIAL-NULL) | `production_ownership_disposition` | 753 |
| harness outcome disposition (5-valued) | `harness_outcome_disposition` | 773 |
| g-censoring: MAP rail flags | `is_railed`, `map_h_of` | 804, 808 |
| SYNTH fixture (<=10 rows, one term carries all tilt) | `make_synth_fixture` | 817 |
| SYNTH check: closure, every disposition row, INSTRUMENT-DEFECT path | `run_synth_check` | 850 |
| 5-row real-slice closure (design-gate computability check) | `_five_row_slice_closure` | 993 |
| production family driver (Sec.4.1/4.2, per venue/channel) | `ProductionFamilyResult`, `run_production_family` | 1008-1146 |
| K_hosted (15/48) reported-only leave-out (Sec.4.4) | `run_K_hosted_leaveout` | 1149 |
| per-universe harness read (Sec.4.3) | `HarnessUniverseRead`, `run_harness_universe` | 1168-1289 |
| main(): dry-run vs real-mode wiring, `--out` JSON with every intermediate | `main` | 1292 |
| every disposition row 1:1 in code | `run_synth_check` exercises all 4 production + all 6 harness rows | 940-970 |

## 4. `--dry-run` on the REAL inputs (Sec.8 launch block, `--dry-run` appended)

Exit code **0**. No `--out` file written (verified: `highz_result_read.json` absent after the run).
No registered aggregate (`Delta_F`, `Delta_t`, share, `S_t`, `S_F`, harness pooled ln-likelihood value)
computed -- only pins, population counts/hashes, and the closure residual, per Sec.3's `--dry-run`
contract.

```
[pin OK] logl-iiib md5: 8e6a2c18dc5838dd1d52641589243672
[pin OK] logl-jr1 md5: 745954a0fdee5f10878fb5e622a06144
[pin OK] table-iiib sha256: 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0
[pin OK] table-jr1 sha256: fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a
[pop pin OK] iiib P_dark: n=606 sha256=5e7f0cf51f0d4f8a312414edd88a31594a5d07886316e7b559e85e831bd2b1e5
[pop pin OK] iiib K: n=159 sha256=c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f
[pop pin OK] iiib K_dark: n=144 sha256=50ae82c30142dc8ad7a2622fea56a29e9fce1b44ac48c5182b0b1be7e977d6ce
[pop pin OK] iiib R: n=231 sha256=f7f494ce8e7d15a91d33b9a54cfc0e334a474929611496fc4a30a0565bbea6aa
[pop pin OK] jr1 P_dark: n=493 sha256=14ad8c17dfccb3d598e6014951595907bcde3f5fd4b9cbd00390395c50940258
[pop pin OK] jr1 K: n=159 sha256=c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f
[pop pin OK] jr1 K_dark: n=111 sha256=cb1def75e3f06f2f703e09d169c4ab2203f188c4a2484427177f807dc65d698b
[pop pin OK] jr1 R: n=191 sha256=db7cbbb97a57f529d4ced1a14f02611e2fee8944befdb1f49f9c664bda4ee2a8
[pin OK] harness manifest sha256: 6a06063dd56aae74ee1cc8bbc63f7da8207ff3e3fc705290a81a2675adb1c0a2 (67 universes)
[gate OK] G-3d: 13 resolved_flags tokens identical, 67/67 universes
[counts] iiib: n=1588 P_dark=606 K=159 K_dark=144 K_hosted=15 R=231
[counts] jr1:  n=1588 P_dark=493 K=159 K_dark=111 K_hosted=48 R=191
[counts] harness: universes=67 Sigma n_scored(CSV event_idx)=12060 (anchor 12060)
[gate G-1] 5-row real-slice max closure residual: 2.665e-15 (band 1e-9)
[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path
[dry-run] gates + byte-id anchors only, no --out written, no registered aggregate computed.
```

Population counts **606/144/231** (iiib) match the task's required values exactly; the harness
manifest matched on the first listed pin (`6a06063dd5...`); all 67 universes discovered and their
resolved_flags agree 13/13 tokens. No `PIN CORRECTION (build)` was needed -- every file named in the
launch block exists at the given path.

## 5. `--synth` (<=10-row synthetic fixture)

```
[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path
```

`make_synth_fixture()` builds 6 events x 5 h-nodes with `T_B` carrying an event-specific linear tilt
for 2 "K_dark" events and `T_g`/`T_D` exactly flat (no h-dependence) for every event. `run_synth_check`
asserts `s_B == 1.0` (to 1e-9), `s_g == 0.0` (to 1e-6), closure `r == 0.0` (to 1e-9), and that the
resulting disposition is `TERM-OWNS(B)`; it then hand-exercises the remaining 3 production-side rows
(`DIFFUSE-IN-TERMS`, `Z-DIFFERENTIAL-NULL`, `INTERMEDIATE`) and all 6 harness-outcome rows
(`ESTIMATOR-INTERNAL candidate`, `FLOOR-CONSISTENT`, `PRODUCTION-ONLY` x2 trigger paths,
`INTERMEDIATE`, `UNPOWERED-CONTROL`), and the G-1 closure gate's pass and INSTRUMENT-DEFECT-raising
fail paths on a 3-row hand-built table.

## 6. What was NOT run

Real mode (`--out` written; `Delta_F`, `Delta_t`, shares, `S_t`, `S_F`, harness pooled values, MAP
rail flags on the actual population) was never invoked by this builder -- per the task instruction
("Do NOT run real mode") and the node's blindness discipline (`REGISTRATION_DRAFT.md` Sec.3: the
builder writes and dry-run-tests the script; the disjoint reader runs Sec.8 once in real mode). The
real-mode code path (`run_production_family`, `run_harness_universe`, `harness_pool_score`, the
disposition functions, and `main`'s real-mode branch) was verified by (a) static review against every
draft formula in Sec.2/Sec.4/Sec.5, (b) `ruff`/`mypy` clean, and (c) the SYNTH fixture, which
exercises the identical call graph (`run_term_freeze` -> disposition) end-to-end on synthetic data.

## 7. Quality gate

`uv run ruff check` -- All checks passed. `uv run ruff format` -- 1 file left unchanged (already
formatted). `uv run mypy highz_decomp_reads.py` -- Success: no issues found in 1 source file.
