# READ_RECORD_rev4.md — r-highz-completion (q-highz-completion), disjoint reader

Reader: disjoint agent, real-mode single execution, no script edits, no manual inspection of the
input data files. Gates confirmed GREEN before launch: `DESIGN_GATE_computability.md`,
`DESIGN_GATE_formula_rev3.md` (§5 "GREEN … the disjoint reader may run §8 in real mode"), and
`DESIGN_GATE_formula_rev4.md` (§5 "GREEN … the disjoint reader may run §8 in real mode" — scope:
PIN CORRECTION 4, `--g1d-tol`). All three gate files inspected in full before this run; not
re-derived here. This supersedes `READ_RECORD.md`, which halted with `INSTRUMENT-DEFECT: G-1d …
4.407e-07 > 1e-8` on the pre-PIN-CORRECTION-4 launch block (no `--g1d-tol` flag, no `--out` written)
— that halt is the reason PIN CORRECTION 4 exists. This run adds `--g1d-tol 1e-6` per the
correction and per the task instruction.

## Exact command executed (repo root, once, real mode)

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
  --g1d-tol 1e-6 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-highz-completion/highz_completion_result_read.json
```

Run once. Not re-run, not edited, not retried with different flags. Argparse of
`highz_decomp_reads.py` matches this block flag-for-flag (`DESIGN_GATE_formula_rev4.md` §1–§2
already re-derived this; not re-checked here beyond the successful parse). One undeclared default
applies as documented in the design gates: `--nonadditivity-max 0.6` (matches the §5 replicate-table
`|r|/|Δ_F| ≤ 0.6` band).

## Exit code

**0**

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
[gate G-1d] resolved --g1d-tol: 1.000e-06
[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path,
  Findings A-D counter-examples, Finding H K-vs-K_dark leaveout, Finding I channel term selection,
  Finding J replicate-rule pass/miss
[written] results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-highz-completion/highz_completion_result_read.json
```

## Full stderr (verbatim)

Empty. No output on stderr; no traceback, no `INSTRUMENT-DEFECT`.

## `--out` file

**Written.** `highz_completion_result_read.json`, 60,261 bytes, at the pinned path. Not present
before this run (confirmed by `ls` immediately before launch).

---

## Pins and gates

### Input-file pins (§1, pre-registered)

| object | expected | observed | match |
|---|---|---|---|
| `event_likelihoods.csv` (iiib) md5 | `8e6a2c18dc5838dd1d52641589243672` | `8e6a2c18dc5838dd1d52641589243672` | YES |
| `event_likelihoods.csv` (joint_r1) md5 | `745954a0fdee5f10878fb5e622a06144` | `745954a0fdee5f10878fb5e622a06144` | YES |
| `covariate_table_iiib.csv` sha256 | `90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0` | matches | YES |
| `covariate_table_joint_r1.csv` sha256 | `fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a` | matches | YES |
| harness manifest sha256 (67 universes) | `6a06063dd56aae74ee1cc8bbc63f7da8207ff3e3fc705290a81a2675adb1c0a2` | matches | YES |

### Population sha256s (§1)

| population | venue | n | sha256 | anchor match |
|---|---|---|---|---|
| P_dark | iiib | 606 | `5e7f0cf51f0d4f8a312414edd88a31594a5d07886316e7b559e85e831bd2b1e5` | YES |
| K | iiib | 159 | `c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f` | YES |
| K_dark | iiib | 144 | `50ae82c30142dc8ad7a2622fea56a29e9fce1b44ac48c5182b0b1be7e977d6ce` | YES |
| R | iiib | 231 | `f7f494ce8e7d15a91d33b9a54cfc0e334a474929611496fc4a30a0565bbea6aa` | YES |
| P_dark | jr1 | 493 | `14ad8c17dfccb3d598e6014951595907bcde3f5fd4b9cbd00390395c50940258` | YES |
| K | jr1 | 159 | `c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f` (identical to iiib K) | YES |
| K_dark | jr1 | 111 | `cb1def75e3f06f2f703e09d169c4ab2203f188c4a2484427177f807dc65d698b` | YES |
| R | jr1 | 191 | `db7cbbb97a57f529d4ced1a14f02611e2fee8944befdb1f49f9c664bda4ee2a8` | YES |

K_hosted (reported-only, no term decomposition): iiib 606−144=... n=15 (=159−144); jr1 n=48
(=159−111) — matches `|K_dark|+|K_hosted|=159` in both venues (G-3c).

### G-1 closure (both venues, both channels)

Closure identity band (G-1a/b, `|ln combined − (ln B_term − den_log_term)| ≤ 1e-9`): PASS in all
four production families — `max_g1_closure_residual` reported by the script as
`3.552713678800501e-15` identically for iiib_2D, iiib_1D, jr1_2D, jr1_1D (well inside the 1e-9
band). The separate 5-row real-slice closure check (same gate, run at start of `main()`) reported
`2.665e-15` (band 1e-9).

**G-1d (`|den_log_term − ln D_tilde_phi| ≤ g1d_tol=1e-6`, PIN CORRECTION 4):** PASS — the run
completed with exit 0 and raised no `INSTRUMENT-DEFECT`, which is the gate's only pass/fail signal;
`gate_g1_closure()` (`highz_decomp_reads.py:493`) computes `resid_den.max()` per call but returns
only `{"max_closure_residual", "max_g_frac_rel_residual"}` — the G-1d `ln D_tilde_phi` residual
value itself is **not persisted** in stdout or in `--out` for the 4 production-family calls (called
once per venue/channel at line 1446: `iiib/2D P_dark full`, `iiib/1D P_dark full`, `jr1/2D P_dark
full`, `jr1/1D P_dark full`). The only printed resolution is the tolerance itself: `[gate G-1d]
resolved --g1d-tol: 1.000e-06`. The pre-existing disclosed values from the design-gate's own
standalone gate-only report (`DESIGN_GATE_formula_rev4.md` §3–§4, `BUILD_RECORD.md` FIX 4; not
computed in this run) are `4.407370e-7` (iiib) and `4.102515e-7` (jr1), both `< 1e-6`, over the same
full P_dark tables. `gate_g1_closure` is not called per-harness-universe in this script (only for
the 5-row slice and the 4 production families); the harness's own physics-floor check is G-2(iii),
not G-1.

G-1e (event-independence of `D_tilde_phi`/`den_log_term` per node): PASS (no raise). G-1c (`Δ_D`
identity = 0): confirmed `delta_D_identity = 0.0` exactly in all four families (see §"Δ_D identity"
below).

### G-2 byte-id anchors

| item | anchor | observed | Δ | within band | verdict |
|---|---|---|---|---|---|
| (i) mean_h, iiib 2D | 0.6658540600 | 0.6658540599535224 | 4.6e-11 | 1e-9 | PASS |
| (i) mean_h, iiib 1D | 0.6669869414 | 0.6669869414473403 | 4.7e-11 | 1e-9 | PASS |
| (i) mean_h, jr1 2D | 0.6671265168 | 0.6671265168140829 | 1.4e-11 | 1e-9 | PASS |
| (i) mean_h, jr1 1D | 0.6670323337 | 0.6670323337269477 | 2.7e-11 | 1e-9 | PASS |
| (ii) Δ_K (159), iiib 2D | +0.086106 | 0.08610643739130941 | 4.4e-7 | 1e-6 | PASS |
| (iii) physics-floor exclusions | 0 | 0 (no raise at any of the 6 call sites: 4 production families + K_hosted × 2 channels + 67 harness universes × 2 channels) | — | — | PASS |
| (iv) population sha256s | (table above) | (table above) | — | exact | PASS |
| (iv) harness pooled sizes | 12,060 / 4,826 / 1,207 / 1,148 | `n_scored=12060, P_dark=4826, K=1207, K_dark=1148` (independently re-summed from `per_universe`: identical) | — | exact | PASS |
| (v) k=1588 endpoint = 0.73 | 0.73 | `run_metadata.h_true = 0.73` (CLI value, float literal; no distinct 1e-12 gate function found in the script) | — | — | consistent with input, not independently re-derived by a separate gate |
| (vi) SYNTH fixture s_t=1, r=0 | 1e-12 | reported in stdout as `[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path, Findings A-D counter-examples, Finding H K-vs-K_dark leaveout, Finding I channel term selection, Finding J replicate-rule pass/miss` — no numeric residual printed for this real (non-`--dry-run`) invocation | — | — | ran and passed (no raise); the design gates (`rev`, `rev2`, `rev3`) already reproduced the numeric `s_t=1, r=0 to 1e-12` under `--dry-run` |

G-2(ii) note (per `gate_g1_closure`'s own docstring at `highz_decomp_reads.py:916-931`): the anchor
is asserted **only** for `venue == "iiib"` and only against `delta_K_leaveout` (the 159-event K
leave-out), never against `delta_K_dark_leaveout` (144-event, reported-only). Confirmed by reading:
`assert_g2ii_delta_k_anchor` returns immediately for `venue != "iiib"`.

### G-3 population

| item | check | result |
|---|---|---|
| (a) set-identity, C7==0 ≡ C2==False ≡ C3c_censored | internal assertion (`highz_decomp_reads.py:376-390`), no print, no raise | PASS |
| (b) K identical event set both venues | iiib K sha256 == jr1 K sha256 (`c8ce89931d7659a4c34e99f2c64b43a05f045c0504de16077c276375c7f9241f`, both printed) | PASS |
| (c) `|K_dark|+|K_hosted|=159` both venues | iiib 144+15=159; jr1 111+48=159 | PASS |
| (d) 67 universes, 41 nodes incl. 0.73, both files, resolved-flags equality (13 tokens) | printed `[gate OK] G-3d: 13 resolved_flags tokens identical, 67/67 universes` | PASS |
| (e) seed blocks 901000–901066 only | independently re-derived from `per_universe`: 67 seeds, min=901000, max=901066, all in range | PASS |

---

## Δ_F (all-terms freeze total) and non-additivity residual r

| family | Δ_F | r (non-additivity) | \|r\|/\|Δ_F\| | non-additivity band 0.6 |
|---|---|---|---|---|
| iiib 2D | 0.0643200415769164 | 0.002823527436671691 | 0.043898097194094744 | within |
| iiib 1D | 0.057338686748139356 | 0.0 | 0.0 | within |
| jr1 2D | 0.04714053860120426 | 0.0016814103192377416 | 0.03566803369520237 | within |
| jr1 1D | 0.0531949676077188 | 0.0 | 0.0 | within |

Δ_D identity (§2.3, must be 0 to 1e-12): `delta_D_identity = 0.0` exactly, all four families.

**Null CI99 precondition** (§5: `Δ_F` must be outside its null CI99 for the production ownership
table to apply, else Z-DIFFERENTIAL-NULL):

| family | Δ_F | null CI99 | outside CI99 |
|---|---|---|---|
| iiib 2D | 0.0643200415769164 | (0.006855066293588341, 0.015302720315848482) | YES |
| iiib 1D | 0.057338686748139356 | (0.005298095419729137, 0.012365118573390751) | YES |
| jr1 2D | 0.04714053860120426 | (0.0038352574471763002, 0.013192378566185646) | YES |
| jr1 1D | 0.0531949676077188 | (0.0037834342033177455, 0.015335216319287616) | YES |

## Δ_K (159, K leave-out) and Δ_K,dark (144, K_dark leave-out), and their concordance

Populated only for the 2D channel per family (`delta_K_leaveout`/`delta_K_dark_leaveout` are `nan`
for both 1D families — the script gates this computation on `channel == "combined_with_bh"`, i.e.
2D only; see `highz_decomp_reads.py:1519-1530`).

| family | Δ_K (159) | Δ_K,dark (144) | Δ_K,dark/Δ_K (concordance, reported-only) |
|---|---|---|---|
| iiib 2D | 0.08610643739130941 | 0.07891109551111852 | 0.9164366556301503 |
| iiib 1D | nan | nan | nan |
| jr1 2D | 0.08061025721445969 | 0.0512686545556712 | 0.636006586845064 |
| jr1 1D | nan | nan | nan |

K_hosted leave-out (§4.4, reported-only, no term decomposition): iiib `0.004693094606469939`;
jr1 `0.016253244189115224`.

## Term deltas Δ_t and shares s_t

| family | term | Δ_t | s_t = Δ_t/Δ_F |
|---|---|---|---|
| iiib 2D | B | 0.059584600535121046 | 0.9263768970650846 |
| iiib 2D | g | 0.0019119136051236607 | 0.029725005740820615 |
| iiib 1D | B | 0.057338686748139356 | 1.0 |
| jr1 2D | B | 0.04376520703843789 | 0.9283985363145566 |
| jr1 2D | g | 0.0016939212435286288 | 0.035933429990241045 |
| jr1 1D | B | 0.0531949676077188 | 1.0 |

## Stencil score excess per term, with SE and Z = S/SE (production; §2.4)

| family | term | S_t (nats/h) | SE (Welch) | Z = S/SE |
|---|---|---|---|---|
| iiib 2D & 1D | B | −0.7466687801968894 | 0.026964277156416572 | −27.691036398474633 |
| iiib 2D | g | −0.03686499482053292 | 0.0009785093693730129 | −37.67464673757231 |
| jr1 2D & 1D | B | −0.8372157026766636 | 0.03563901274910937 | −23.491551479567455 |
| jr1 2D | g | −0.040404469075480426 | 0.0009108447056572746 | −44.35933900095973 |

(B's `S`/`SE` values are identical between the 2D and 1D entries of the same venue because `S_t` is
a per-term stencil statistic computed once per term, not per channel-total — this matches the
draft's "T_D cancels exactly" / channel-transferable definition, §2.4.)

## Harness control (67 universes; pooled by event-weighting over Σ K_dark,u and Σ R_u; delete-one-universe jackknife SE)

Pooled sizes (re-verified by independent re-summation over the 67 `per_universe` rows): Σ n_scored =
12,060; Σ P_dark,u = 4,826; Σ K_u = 1,207; Σ K_dark,u = 1,148 — exact match to the §1 anchors.

### 2D channel (both terms; this is the disposition-machinery channel per DESIGN_GATE_formula_rev3)

| term | pooled Δ (`pooled_S`, nats/h) | jackknife SE | Z = S/SE | s_t^harn |
|---|---|---|---|---|
| B | −0.7756912677567085 | 0.009842507742506735 | −78.81032842948538 | 0.9564751354621941 |
| g | −0.03529820703175443 | 0.0005315974161711163 | −66.40026071983816 | 0.04352486453780619 |
| **F (total)** | **S_F_harn = −0.8109894747884627** | **SE_F_harn = 0.009980222782597031** | **Z_harn = −81.2596564680522** | s_t_harn_defined = true |

`ρ_S = S_F^harn / S_F^prod = 1.03504086313373`; per-term `ρ_S_terms`: B = `1.0388692929576702`,
g = `0.9574993080453161`.

Per-universe: `n_universes = 67`; `n_universes_railed = 14` (2D `rail_full_u` count, independently
re-summed and confirmed = 14); row #335 expected 10–14 universes railed at 0.86 — this figure falls
inside that expectation, disclosed without further reading.

### 1D channel (B only; reported, not disposition-machinery per DESIGN_GATE_formula_rev3)

| term | pooled S | jackknife SE | Z = S/SE |
|---|---|---|---|
| B | −0.7756912677567085 | 0.009842507742506735 | −78.81032842948538 |

`channel_1D_n_universes_railed = 10` (also inside the row #335 10–14 expectation).

---

## Replicate families and the replicate-rule outcome (raw vs booked)

Registered replicate rule (§5): TERM-OWNS(t) must hold with the same t in joint_r1 2D; the 1D
families must show `Δ_B^1D` of the same sign as `Δ_B^2D` (both venues); a miss downgrades the
**booked** disposition to INTERMEDIATE.

`--out.replicate_rule` (computed by the script, family = the primary iiib 2D):

| field | value |
|---|---|
| family | iiib_2D |
| raw_disposition | TERM-OWNS(B) |
| booked_disposition | TERM-OWNS(B) |
| downgraded | False |
| reasons | [] |

Underlying replicate facts (re-derived from the individual family results above, not a separate
`--out` field): same owning term (B) in iiib 2D and jr1 2D; `Δ_B` sign in both 1D families
(iiib `+0.057338686748139356`, jr1 `+0.0531949676077188`) matches the sign of `Δ_B` in the
corresponding 2D families (iiib `+0.059584600535121046`, jr1 `+0.04376520703843789`) — all four
positive, all four consistent.

---

## Three-valued disposition outcomes (both tiers), exactly as §5 of the draft defines them

Below: for each tier, every row of the pre-registered table with its trigger condition, evaluated
mechanically against the numbers above (TRUE = trigger fired / FALSE = trigger did not fire). This
reproduces exactly what `--out.disposition` / `--out.harness.disposition` already report — presented
here row-by-row for the pre-registered table, not as an added interpretation.

### Production term ownership (per family; precondition already confirmed above: Δ_F outside null CI99 in all four, so none is Z-DIFFERENTIAL-NULL)

| family | TERM-OWNS(t) [s_t≥0.5, largest, \|r\|/\|Δ_F\|≤0.6] | DIFFUSE-IN-TERMS [every \|s_t\|<0.2] | INTERMEDIATE [otherwise] | `--out` disposition field |
|---|---|---|---|---|
| iiib 2D | TRUE (t=B; s_B=0.926, s_g=0.030, r-ratio=0.044) | FALSE | FALSE | TERM-OWNS(B) |
| iiib 1D | TRUE (t=B; s_B=1.0, r-ratio=0.0) | FALSE | FALSE | TERM-OWNS(B) |
| jr1 2D | TRUE (t=B; s_B=0.928, s_g=0.036, r-ratio=0.036) | FALSE | FALSE | TERM-OWNS(B) |
| jr1 1D | TRUE (t=B; s_B=1.0, r-ratio=0.0) | FALSE | FALSE | TERM-OWNS(B) |

### Harness control → mechanism outcome (single evaluation over the pooled 2D statistics, per §5)

Inputs: `Z_harn = −81.2596564680522` (`|Z_harn| = 81.26 > 3`); `ρ_S = 1.03504086313373`;
production-owning term (booked, from replicate_rule) = B; `s_t_harn[B] = 0.9564751354621941 ≥ 0.5`;
sign of production pooled S_B (`−0.7466687801968894`) matches sign of harness pooled S_B
(`−0.7756912677567085`) — both negative; `SE_F_harn = 0.009980222782597031`.

| outcome row | trigger (§5) | fired |
|---|---|---|
| ESTIMATOR-INTERNAL candidate | \|Z_harn\|>3 AND ρ_S≥0.5 AND production-owning term has s_t^harn≥0.5, same sign | TRUE |
| PRODUCTION-ONLY | \|Z_harn\|≤3, OR (\|Z_harn\|>3 AND ρ_S≤0.2) | FALSE |
| FLOOR-CONSISTENT | \|Z_harn\|≤3 AND production Δ_F inside its null CI99 | FALSE |
| INTERMEDIATE | \|Z_harn\|>3 and 0.2<ρ_S<0.5, or same magnitude but a different owning term | FALSE |
| UNPOWERED-CONTROL | harness SE>0.1 nats/h | FALSE (SE_F_harn=0.00998, well under 0.1) |

`--out.harness.disposition` field: `"ESTIMATOR-INTERNAL candidate"` — matches the row-by-row
evaluation above exactly.

---

## Provenance note (disclosed, not evaluated)

`highz_decomp_reads.py`, `BUILD_RECORD.md`, and `REGISTRATION_DRAFT.md` show as locally modified
(uncommitted) in `git status` at the time of this run — this is the PIN CORRECTION 4 fix already
reviewed GREEN by `DESIGN_GATE_formula_rev4.md`'s independent `git diff --stat` check (3 files, no
production/cluster/`darksiren_emri/` paths touched). Not re-diffed here; the reader ran the script
as it stood on disk, per instruction ("do not modify any script").
