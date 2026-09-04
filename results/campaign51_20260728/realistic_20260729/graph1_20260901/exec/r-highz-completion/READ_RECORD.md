# READ_RECORD.md — r-highz-completion (q-highz-completion), disjoint reader

Reader: disjoint agent, real-mode single execution, no script edits. Gates confirmed GREEN before
launch: `DESIGN_GATE_computability.md` (final verdict "GREEN" not explicit as a single word but §7
confirms nothing pre-computed and no defect blocking launch was raised) and `DESIGN_GATE_formula_rev3.md`
(§5 "**GREEN.** ... The disjoint reader may run §8 in real mode."). Both gate files inspected in full
before this run; not re-derived here.

## Pre-launch pin verification (independent of the script's own checks, run before §8)

All four file-level pins reproduced exactly against `REGISTRATION_DRAFT.md` §1 before invoking the
launch block:

| object | expected | observed | match |
|---|---|---|---|
| `event_likelihoods.csv` (iiib) md5 | `8e6a2c18dc5838dd1d52641589243672` | `8e6a2c18dc5838dd1d52641589243672` | YES |
| `event_likelihoods.csv` (joint_r1) md5 | `745954a0fdee5f10878fb5e622a06144` | `745954a0fdee5f10878fb5e622a06144` | YES |
| `covariate_table_iiib.csv` sha256 | `90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0` | (matches) | YES |
| `covariate_table_joint_r1.csv` sha256 | `fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a` | (matches) | YES |

Harness root `results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip`
present, 193 entries (67 universes + auxiliary cache/check directories) — consistent with the draft's
"67 universes" figure; not independently reduced here (the script's own pin check covers it, see below).

Argparse of `highz_decomp_reads.py` (lines 235–261 as read) matches the §8 launch block flag-for-flag
(`--logl-iiib`, `--logl-md5-iiib`, `--logl-jr1`, `--logl-md5-jr1`, `--table-iiib`, `--table-sha256-iiib`,
`--table-jr1`, `--table-sha256-jr1`, `--harness-root`, `--harness-population`, `--harness-cell`,
`--harness-manifest-sha256`, `--h-true`, `--decile`, `--stencil` (nargs=3), `--null-draws`, `--null-seed`,
`--share-own`, `--share-diffuse`, `--rho-hi`, `--rho-lo`, `--z-gate`, `--se-unpowered`, `--out`,
`--dry-run`), plus one undeclared-in-§8 default (`--nonadditivity-max`, default `0.6`, annotated
"DESIGN_GATE finding B" — not passed explicitly in the launch block, so the script's own default (0.6,
matching the draft's §5 replicate-table `|r|/|Δ_F| ≤ 0.6` band) applies. This is a documented default,
not a script modification.

## Exact command executed (repo root, once, real mode — §8 verbatim, no `--dry-run`)

```
cd /home/jasper/Repositories/darksiren-emri
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-highz-completion/highz_decomp_reads.py \
  --logl-iiib results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
  --logl-md5-iiib 8e6a2c18dc5838dd1d52641589243672 \
  --logl-jr1 results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
  --logl-md5-jr1 745954a0fdee5f10878fb5e622a06144 \
  --table-iiib results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_iiib.csv \
  --table-sha256-iiib 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0 \
  --table-jr1 results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_joint_r1.csv \
  --table-sha256-jr1 fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a \
  --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip \
  --harness-population 200 --harness-cell S \
  --harness-manifest-sha256 6a06063dd56aae74ee1cc8bbc63f7da8207ff3e3fc705290a81a2675adb1c0a2 \
  --h-true 0.73 --decile 0.10 --stencil 0.725 0.730 0.735 \
  --null-draws 1000 --null-seed 20260904 \
  --share-own 0.5 --share-diffuse 0.2 --rho-hi 0.5 --rho-lo 0.2 --z-gate 3.0 --se-unpowered 0.1 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-highz-completion/highz_completion_result_read.json
```

Run once. Not re-run, not edited, not retried with different flags.

## Exit code

**1**

## Full stdout (verbatim)

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
[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path, Findings A-D counter-examples, Finding H K-vs-K_dark leaveout, Finding I channel term selection, Finding J replicate-rule pass/miss
```

## Full stderr (verbatim — the entire error output; no Python traceback was printed)

```
INSTRUMENT-DEFECT: G-1d (iiib/1D P_dark full): |den_log_term - ln D_tilde_phi| 4.407e-07 > 1e-8
```

There is no `Traceback (most recent call last)` block — the script raised its own typed
`INSTRUMENT-DEFECT` condition (defined at `highz_decomp_reads.py:179-188`, a `SystemExit`-based hard
gate, exit code 1 by design) rather than an unhandled Python exception. Per instructions this counts
as the crash/halt case: pasted above verbatim; the reader stops here and does not retry, diagnose, or
re-run.

## `--out` file

**Not written.** Confirmed by `ls` on
`results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-highz-completion/highz_completion_result_read.json`
after the run: no such file. The run halted before reaching the `--out` write step.

## Pin/gate status at the point of halt

| pin/gate | status | value |
|---|---|---|
| `--logl-iiib` md5 | PASS | `8e6a2c18dc5838dd1d52641589243672` |
| `--logl-jr1` md5 | PASS | `745954a0fdee5f10878fb5e622a06144` |
| `--table-iiib` sha256 | PASS | `90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0` |
| `--table-jr1` sha256 | PASS | `fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a` |
| iiib P_dark sha256 (n=606) | PASS | `5e7f0cf51f0d4f8a312414edd88a31594a5d07886316e7b559e85e831bd2b1e5` |
| iiib K sha256 (n=159) | PASS | `c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f` |
| iiib K_dark sha256 (n=144) | PASS | `50ae82c30142dc8ad7a2622fea56a29e9fce1b44ac48c5182b0b1be7e977d6ce` |
| iiib R sha256 (n=231) | PASS | `f7f494ce8e7d15a91d33b9a54cfc0e334a474929611496fc4a30a0565bbea6aa` |
| joint_r1 P_dark sha256 (n=493) | PASS | `14ad8c17dfccb3d598e6014951595907bcde3f5fd4b9cbd00390395c50940258` |
| joint_r1 K sha256 (n=159) | PASS | `c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f` (same set as iiib K, as §1 states) |
| joint_r1 K_dark sha256 (n=111) | PASS | `cb1def75e3f06f2f703e09d169c4ab2203f188c4a2484427177f807dc65d698b` |
| joint_r1 R sha256 (n=191) | PASS | `db7cbbb97a57f529d4ced1a14f02611e2fee8944befdb1f49f9c664bda4ee2a8` |
| harness manifest sha256 (67 universes) | PASS | `6a06063dd56aae74ee1cc8bbc63f7da8207ff3e3fc705290a81a2675adb1c0a2` |
| G-3d (13 resolved_flags tokens, 67/67 universes) | PASS | identical, 67/67 |
| population counts iiib (§1 anchors) | PASS | n=1588, P_dark=606, K=159, K_dark=144, K_hosted=15, R=231 |
| population counts joint_r1 (§1 anchors) | PASS | n=1588, P_dark=493, K=159, K_dark=111, K_hosted=48, R=191 |
| harness pooled Σ n_scored | PASS | 12,060 (matches §1 anchor exactly) |
| G-1 closure, 5-row real slice (design-gate's own check, re-run here) | PASS | max residual 2.665e-15 (band 1e-9) |
| SYNTH fixture (all sub-checks the script bundles under `[SYNTH OK]`) | PASS | closure identity, disposition rows (4 production + 6 harness), G-1 pass/fail path, Findings A–D counter-examples, Finding H, Finding I, Finding J |
| **G-1d, FULL TABLE (iiib, 1D channel, P_dark population, all 606 events × 41 nodes)** | **FAIL — INSTRUMENT-DEFECT** | `\|den_log_term − ln D_tilde_phi\| = 4.407e-07`, band `≤ 1e-8` |
| G-2(i) full-sample `mean_h` byte-id anchors (4 values) | NOT REACHED | halted before this stage |
| G-2(ii) `Δ_K` leave-out anchor (+0.086106) | NOT REACHED | halted before this stage |
| G-2(iii) 0 physics-floor exclusions | NOT REACHED | halted before this stage |
| G-3a/b/c/e (remaining population checks) | NOT REACHED beyond what printed above | population sha256s above are the full set the script printed; no further G-3 sub-checks appeared in stdout |
| g-precision, g-censoring, g-byteid (remaining) | NOT REACHED | halted before these stages |

Note: the item this table calls "G-1 closure, 5-row real slice" is the same 5-row real-slice check
`DESIGN_GATE_formula_rev3.md` reported (residual `2.665e-15`, same value) — it re-passed identically
here. The failure is a **different, wider check**: `G-1d` evaluated over the **full** `iiib` table,
**1D channel**, **all 606 `P_dark` events across all 41 h-nodes** — a scope the design-gate transcripts
(both `_computability` and `_formula_rev3`) explicitly did not cover (they state "5-row real slice
ONLY").

## Registered statistics — NOT COMPUTED (run halted before §4 was reached)

The script's own `[SYNTH OK]` line indicates disposition-row *code paths* were exercised only on the
synthetic fixture (per `b-highz-decomp`'s own test design in §3), not on real data. No downstream
production or harness computation occurred on real inputs before the halt. All of the following are
**NOT COMPUTED** in this read — no value exists to report, correctly or by omission:

- `Δ_F` (all-terms freeze total): NOT COMPUTED
- `Δ_K` (159-event leave-out, iiib 2D): NOT COMPUTED (the +0.086106 anchor in §1/G-2(ii) is a
  **prior, already-registered** value quoted in the draft itself — not reproduced by this run)
- `Δ_K,dark` (144-event leave-out): NOT COMPUTED
- Each `Δ_t` (`Δ_B`, `Δ_g`, `Δ_D`) and share `s_t`: NOT COMPUTED
- All-at-once `Δ_F` and non-additivity residual `r`: NOT COMPUTED
- Stencil score excess per term (`S_B`, `S_g`, `S_F`) with SE/Z: NOT COMPUTED
- Harness control per channel — pooled `Δ`, `S_F^harn`, shares `s_t^harn`, `ρ_S`, jackknife SE,
  `Z_harn`: NOT COMPUTED
- Replicate venue/family reads (iiib 2D/1D, joint_r1 2D/1D) and the replicate-rule outcome
  (`same_t_replicate`, `sign_ok_iiib`, `sign_ok_jr1`, any downgrade): NOT COMPUTED
- Null draws (1000× `Δ_F` CI99): NOT COMPUTED
- **Production term-ownership disposition row** (§5 table 1: TERM-OWNS(t) / DIFFUSE-IN-TERMS /
  INTERMEDIATE, or the Z-DIFFERENTIAL-NULL precondition outcome): **NOT REACHED — no outcome to report**
- **Harness-control disposition row** (§5 table 2: ESTIMATOR-INTERNAL candidate / PRODUCTION-ONLY /
  FLOOR-CONSISTENT / INTERMEDIATE / UNPOWERED-CONTROL): **NOT REACHED — no outcome to report**

## Summary (verdict-free)

The launch block, executed exactly as written in `REGISTRATION_DRAFT.md` §8, in real mode, once, from
the repo root: all file-level pins, all population sha256 pins, the harness manifest pin, G-3d, and the
5-row real-slice G-1 closure check (the same one both design-gate transcripts exercised) **passed
exactly as pinned**. The run then halted with exit code 1 on a G-1d closure check run over the **full**
`iiib` table for the **1D channel** — a check scope neither design-gate transcript ran (`DESIGN_GATE_
computability.md` and `DESIGN_GATE_formula_rev3.md` both restrict themselves to "the fixture and a
5-row real slice ONLY", per `REGISTRATION_DRAFT.md` §3). No registered statistic in §4 was computed.
No `--out` file was written. The script was not modified, re-run, or retried with different arguments.
