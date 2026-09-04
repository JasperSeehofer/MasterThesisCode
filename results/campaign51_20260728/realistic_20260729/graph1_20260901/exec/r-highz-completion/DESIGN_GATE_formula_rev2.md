# DESIGN_GATE_formula_rev2.md — r-highz-completion, FRESH formula/integration re-review (round 3)

Reviewer: fresh formula/integration reviewer, no prior context, spawned specifically to (a) build
my own enumeration of `REGISTRATION_DRAFT.md` + `MECHANISM_NOTE.md` before opening any prior
design-gate file, and (b) confirm whether `DESIGN_GATE_formula.md`'s findings A–D (the round-2
RED/AMBER review that `BUILD_RECORD.md` §8 "FIX 2" claims to have closed) are **actually** closed
in the code, not merely addressed in prose. `INFORMATION_FORECAST.md` was not opened (forbidden).
No registered aggregate (`Δ_F`, `Δ_t`, share, `S_t`, `S_F`, harness pooled value, null CI99 of
`Δ_F`) was computed by me over `K_dark`/`R`/`K`/`P_dark` or any harness universe's real
`event_likelihoods.csv` — every number below is from the SYNTH fixture, the `--dry-run` console
output (real inputs, gates/pins/counts only), or a source-line citation.

## Method

1. Read `REGISTRATION_DRAFT.md` and `MECHANISM_NOTE.md` in full and built my own enumeration of
   every registered statistic (§4.1–§4.4), gate (G-1 a–e, G-2 i–vi, G-3 a–e, g-precision,
   g-censoring, g-byteid), disposition row (both 3/5-valued tables incl. every named INTERMEDIATE
   sub-case), and reported-only output — **before** opening `DESIGN_GATE_formula.md`,
   `DESIGN_GATE_computability.md`, or `BUILD_RECORD.md`.
2. Read `highz_decomp_reads.py` top to bottom and mapped every enumerated item to a function/line.
3. Ran `--dry-run` on the real, pinned Sec.8 launch block myself (reproduced below).
4. Hand-verified the SYNTH fixture's construction and closure arithmetic by re-deriving `Δ_B`,
   `Δ_F`, `r`, `s_B`, `s_g` from `make_synth_fixture()`'s stated construction (linear tilt on 2 of 6
   events, flat `T_g`/`T_D`) independently of the script's own functions.
5. Only then opened `DESIGN_GATE_formula.md` and checked each of its four findings (A–D) against my
   own enumeration and the current code — not against its own prose claims.
6. Confirmed the T0-import contract (`_load_matrix`, `_physics_floor_apply`, `_moments`) by reading
   `exec/r-offset-subset/build_influence_vector.py` directly and checking signature/behavior match.

## Verdict: **RED**

Findings A, C, D, E of `DESIGN_GATE_formula.md` are genuinely fixed — confirmed independently below.
**Finding B is only half-fixed: the G-2(i) `mean_h` anchor is correctly wired, but the G-2(ii)
`Δ_K` anchor check that FIX 2 added asserts the WRONG population's leave-out** — a new, code-level
defect introduced by the fix itself, not present (because unimplemented) at the time
`DESIGN_GATE_formula.md` was written, and not caught by that review. This alone blocks GREEN: a
real-mode run would either halt on a false `INSTRUMENT-DEFECT` at the very anchor gate meant to
catch a wrong population→CSV join, or (if it happened not to raise) validate the wrong statistic
under the registered anchor's name. Two further completeness gaps (missing 1D harness reads;
missing cross-family "Replicate rule" disposition downgrade) are also confirmed, independent of the
formula.md review, and are listed as their own findings.

---

## 1. My own enumeration → code mapping (built before opening any prior gate file)

### 1.1 Registered statistics (§4.1–§4.4)

| item | formula (draft) | code | line | status |
|---|---|---|---|---|
| `T_B, T_g, T_D` (I-2D/I-1D identity) | §2.1 | `compute_term_profiles`, `compute_T_D` | 531, 549 | ✓ correct |
| centered profiles `t̂_e(h)` | §2.1 | `center_profile` | 567 | ✓ correct |
| reference profile `t̄(h)` = median over R | §2.2 | `reference_profile` | 573 | ✓ correct (mean also returned) |
| `Λ_t`, `Δ_t`, `Δ_F`, `r`, `s_t` | §2.3 | `term_freeze_lambda`, `delta_t`, `run_term_freeze` | 578, 592, 608 | ✓ correct, hand-verified on SYNTH (§2 below) |
| `Δ_D = 0` identity | §2.3/G-1c | `TermFreezeResult.delta_D` hardcoded | 599–606 | ✓ correct |
| `Δ_K,dark` plain leave-out | §2.3 | `run_production_family` (`delta_K_dark_leaveout`) | 1263–1269 | ✓ correctly computed **but misused as the G-2(ii) anchor input — see Finding B below** |
| null draws + CI99 of `Δ_F` | §2.5 | `null_draw_ci99` (pool = `P_dark\K`, seed arg, 0.5/99.5 pctile) | 645 | ✓ correct |
| stencil slope `t'_e`, Welch SE, `S_t` | §2.4 | `stencil_slope`, `welch_se`, `score_excess` | 677, 688, 698 | ✓ correct |
| harness pooled `S_t^harn`, delete-one-universe jackknife SE | §4.3 | `harness_pool_score` | 717 | ✓ correct algorithm; **but only ever invoked for the 2D channel — see Finding "1D-harness-gap" below** |
| `ρ_S`, `s_t^harn` (defined iff `\|Z_harn\|>3`) | §2.4/§4.3 | `main()` (`rho_S`, `s_t_harn`, `s_t_harn_defined`) | ~1585–1598 | ✓ value correct; `s_t_harn_defined` flag present (draft's "defined only if" honored as a flag, not a NaN) |
| K_hosted (15/48) reported-only leave-out | §4.4 | `run_K_hosted_leaveout` | 1289 | ✓ correct |
| concordance ratios `Δ_F/Δ_K,dark`, `Δ_K,dark/Δ_K` (reported-only) | §2.3 | *(none)* | — | gap, but explicitly reported-only/non-gating (draft's own wording) — informational only |
| stencil-consistency `Δ_t ≈ \|K_dark\|·S_t/I_HEAD` (reported-only) | §2.4 | *(none)* | — | gap, non-gating; `I_HEAD` is undefined anywhere in the draft, so this cannot be built without a fresh author definition — informational only |

### 1.2 Gates (G-1, G-2, G-3, g-precision, g-censoring, g-byteid)

| gate | code | line | status |
|---|---|---|---|
| G-1(a) bit-exact zero | `gate_g1_closure` | 484 | ✓ correct |
| G-1(b) 1e-9 closure identity | `gate_g1_closure` | 484 | ✓ correct, dry-run reproduces `2.665e-15` |
| G-1(c) `Δ_D=0` | `TermFreezeResult.delta_D` | 599–606 | ✓ identity, not measured — correct per draft |
| G-1(d) 7-s.f. consistency (`g_frac`, `den_log_term`) | `gate_g1_closure` | 484 | ✓ correct |
| G-1(e) `D_tilde_phi`/`den_log_term` single-valued per node | `gate_g1_closure` | 484 | ✓ correct |
| G-2(i) full-sample `mean_h` anchor, 1e-9 | `assert_g2i_mean_h_anchor`, called in `run_production_family` | 812, 1203 | ✓ correct — **Finding B(i) genuinely closed** |
| G-2(ii) `Δ_K` (leave-out of **K, 159**) anchor, 1e-6 | `assert_g2ii_delta_k_anchor`, called in `run_production_family` | 822, 1269 | ✗ **wrong population — see Finding B(ii) below; NOT closed** |
| G-2(iii) 0 physics-floor exclusions | `assert_g2iii_no_physics_floor_exclusion`, called at all 3 `_load_matrix` sites | 796; 1197, 1294, 1371 | ✓ correct, all three sites, before any `event_idx`-keyed lookup — **Finding C genuinely closed** |
| G-2(iv) harness pooled sizes (12,060/4,826/1,207/1,148) | `main()` `pooled_sizes` vs `HARNESS_POOLED_ANCHORS` | ~1531 | ✓ correct, dry-run confirms `n_scored` sub-anchor |
| G-2(v) k=1588 endpoint = 0.73 | `[counts]` print, `n_total` | main | ✓ correct (visible in dry-run transcript) |
| G-2(vi) SYNTH fixture `s_t=1, r=0` to 1e-12 | `run_synth_check` | 910 | ✓ correct, hand-verified independently (§2 below) |
| G-3(a) set-identity `C7==0 ≡ C2==False ≡ C3c_censored` | `verify_g3a_set_identity` | 375 | ✓ correct |
| G-3(b) K identical both venues | `main()` equality check | 1455 | ✓ correct |
| G-3(c) `\|K_dark\|+\|K_hosted\|=159` | `verify_population_pins` (partition check) | 344, 366–372 | ✓ correct (implicit via partition-of-K assertion) |
| G-3(d) harness `n_draw_requested==200` filter, 67 universes, 13-token resolved-flags equality | `discover_harness_universes`, `verify_g3d_resolved_flags` | 399, 450 | ✓ correct, dry-run confirms 67/67 |
| G-3(e) seed blocks 901000–901066 only | `discover_harness_universes` (`range(901000, 901067)`) | 405 | ✓ correct |
| g-precision (full-precision columns, `float_precision="round_trip"`, stencil nodes present) | `load_term_columns` | 472 | ✓ correct for the columns; stencil-node *exact-match* is not asserted (nearest-node lookup) — informational, non-blocking (matches `DESIGN_GATE_formula.md` Finding G, unchanged) |
| g-censoring (MAP rail flags, full + every freeze, per-universe) | `is_railed`, `map_h_of`, `map_rails_freeze`, `rail_full_u` | 864, 868, 1213–1215, 1420 | ✓ correct; harness pooled statistic used for the disposition is `S_F` (score-based), never `Δ`, matching the draft's rail-free requirement |
| g-byteid (`--dry-run` passes G-1…G-3 + G-2(vi) before real mode) | `main()` dry-run branch | ~1490 | ✓ confirmed live, reproduced below |

### 1.3 Disposition rows

**Production ownership (§5 table 1), `production_ownership_disposition` (line 753):**

| row | code path | status |
|---|---|---|
| Z-DIFFERENTIAL-NULL | `null_lo <= delta_F <= null_hi` | ✓ |
| TERM-OWNS(t) | `top_share >= share_own and r_over_abs_delta_F <= nonadditivity_max` | ✓ — `nonadditivity_max` now a genuine parameter (Finding A, confirmed below) |
| DIFFUSE-IN-TERMS | `all(abs(s) < share_diffuse ...)` | ✓ |
| INTERMEDIATE — `0.2 ≤ max s_t < 0.5` (literal fall-through) | final `return "INTERMEDIATE"` | ✓ |
| INTERMEDIATE — both shares ≥ 0.5 with `r < 0` (named carve-out) | `both_ge_share_own_r_negative` | ✓ — Finding D, confirmed below |
| INTERMEDIATE — sign-opposed terms, `\|s\|>1` each (named carve-out) | `sign_opposed_gt1` | ✓ — Finding D, confirmed below |
| **Replicate rule**: TERM-OWNS(t) must hold with the same t in joint_r1 2D; 1D families' `Δ_B` same sign as 2D; "a miss → INTERMEDIATE" | *(none)* | ✗ **new gap — see "Replicate-rule gap" below** |

**Harness outcome (§5 table 2), `harness_outcome_disposition` (line 833):** all 6 rows
(`ESTIMATOR-INTERNAL candidate`, `PRODUCTION-ONLY` ×2 trigger paths, `FLOOR-CONSISTENT`,
`INTERMEDIATE`, `UNPOWERED-CONTROL`) map 1:1 to code branches and are exercised by
`run_synth_check`'s six hand-built assertions (lines 1054–1080, all six reproduced in the dry-run's
`[SYNTH OK]` line) — ✓ correct, matches my own reading of the table.

### 1.4 T0-import contract

`_import_t0_module()` (line 59) loads `exec/r-offset-subset/build_influence_vector.py` by path via
`importlib.util.spec_from_file_location` + `exec_module`, then pulls `_load_matrix`,
`_physics_floor_apply`, `_moments` off the live module (lines 76–78). I read
`build_influence_vector.py` directly: `_load_matrix(csv_path, channel) -> (h_grid, event_idx, logL,
n_excluded)` (line 145), `_physics_floor_apply` (line 123, per-row zero→min-nonzero, all-zero row
excluded), `_moments` (line 163, gradient-trapezoid weights, log-sum-exp normalisation) — all three
match `REGISTRATION_DRAFT.md` §1's citation and the signatures `highz_decomp_reads.py` assumes.
Genuine import, not re-implementation. ✓ correct.

### 1.5 `InstrumentDefect` contract

`InstrumentDefect(SystemExit)` (line 172) — every gate/pin miss raises it, giving a nonzero exit
without a Python traceback dump; confirmed live by running with a nonexistent `--logl-iiib` path
(dry-run): exits 1 with `INSTRUMENT-DEFECT: missing registered input file(s): [...]`, raised by
`preflight()` (line 266) before any pin/hash attempt. ✓ correct, matches the node-lesson memory
citation in the class docstring.

---

## 2. Independent hand-check of the SYNTH fixture (not importing the script's functions)

`make_synth_fixture()` (line 877): 6 events × 5 nodes `{0.720, 0.725, 0.730, 0.735, 0.740}`,
`rng = default_rng(0)`. `T_B` = event base (`rng.normal(0,1,6)`) + slope×(h−0.73), slope
`[0,0,0,0,5,5]` (only events 4,5 tilted); `T_g` flat per event (`rng.normal(0,0.1,6)`, tiled across
h); `T_D ≡ 0`. `K_dark = {4,5}`, `R = {0,1,2}`.

Because `T_g` and `T_D` are exactly flat, centering (`t̂ = t − t(0.73)`) makes both **identically
zero** at every node for every event — so any freeze of `T_g` (replacing target profiles with
`t̄_g`, itself zero) changes nothing: `Δ_g = 0` exactly, and the entire freeze effect is carried by
`T_B`. Re-deriving by hand: `t̄_B(h) = median_{e∈{0,1,2}} T̂_{B,e}(h) = 0` at every node (events 0–2
have slope 0, so `T̂_B` is identically 0 for them — the median of three zeros is zero). Freezing
`K_dark`'s `T̂_B` (slope 5 events) to this all-zero `t̄_B` removes the entire h-dependence
contributed by events 4,5 from the joint log-posterior, i.e. `Λ_B(h) = Λ_full(h) − (T̂_{B,4}(h) +
T̂_{B,5}(h))`, which is linear in `(h−0.73)` with slope `−10` — a pure tilt subtraction. Since
`Λ_full` is itself linear in `h` around 0.73 (flat `T_g`,`T_D`; `T_B` linear by construction) and the
node grid is symmetric, the weighted-mean-h shift `Δ_B = mean_h(Λ_B) − mean_h(Λ_full)` is a clean,
non-zero number of the sign of the removed tilt; `Δ_g = 0` (freezing an already-zero profile to an
already-zero reference changes nothing); `Δ_F = Δ_B + Δ_g = Δ_B` exactly, so `r = Δ_F − (Δ_B+Δ_g) =
0` to float precision and `s_B = 1, s_g = 0`. This matches the script's own asserted values
(`s_B==1.0` to 1e-9, `s_g==0` to 1e-6, `r==0` to 1e-9, lines 963–965) and confirms the closure
identity `Λ_t = Λ_full − Σt̂ + n·t̄` (line 578–586) and `Δ_t = mean_h(Λ_t) − mean_h(Λ_full)` (line
592–596) are wired exactly as §2.3 specifies, for the one-term-owns case.

---

## 3. `--dry-run` on the real, pinned Sec.8 launch block (run by me, verbatim CLI, appending `--dry-run`)

Exit code **0**; no `--out` file materialized. Reproduces `BUILD_RECORD.md` §4's transcript
byte-for-byte:

```
[pin OK] logl-iiib md5: 8e6a2c18dc5838dd1d52641589243672
[pin OK] logl-jr1 md5: 745954a0fdee5f10878fb5e622a06144
[pin OK] table-iiib sha256: 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0
[pin OK] table-jr1 sha256: fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a
[pop pin OK] iiib P_dark: n=606 ...  [pop pin OK] iiib K: n=159 ...
[pop pin OK] iiib K_dark: n=144 ...  [pop pin OK] iiib R: n=231 ...
[pop pin OK] jr1 P_dark: n=493 ...   [pop pin OK] jr1 K: n=159 ...
[pop pin OK] jr1 K_dark: n=111 ...   [pop pin OK] jr1 R: n=191 ...
[pin OK] harness manifest sha256: 6a06063dd5...adb1c0a2 (67 universes)
[gate OK] G-3d: 13 resolved_flags tokens identical, 67/67 universes
[counts] iiib: n=1588 P_dark=606 K=159 K_dark=144 K_hosted=15 R=231
[counts] jr1:  n=1588 P_dark=493 K=159 K_dark=111 K_hosted=48 R=191
[counts] harness: universes=67 Sigma n_scored(CSV event_idx)=12060 (anchor 12060)
[gate G-1] 5-row real-slice max closure residual: 2.665e-15 (band 1e-9)
[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path, Findings A-D counter-examples
[dry-run] gates + byte-id anchors only, no --out written, no registered aggregate computed.
```

Population counts (606/144/231 iiib; 493/111/191 jr1) match §1 exactly. No registered aggregate
computed — consistent with the task's `--dry-run` constraint on reviewers.

---

## 4. `DESIGN_GATE_formula.md` findings A–D: closure status

### Finding A (RED — `--nonadditivity-max` dead flag) → **CLOSED, confirmed**

`production_ownership_disposition` (line 753) now takes `nonadditivity_max: float = 0.6` as a real
parameter, replacing the hardcoded literal in the `TERM-OWNS` test. `run_production_family` (line
1174) accepts and forwards it; `main()`'s call site passes `args.nonadditivity_max` (I traced the
call chain: `main` → `run_production_family(..., nonadditivity_max=args.nonadditivity_max)` →
`production_ownership_disposition(..., nonadditivity_max=nonadditivity_max)`). `run_synth_check`
(lines 1040–1052) demonstrates the behavioral difference directly: the same `s_B=0.9,s_g=0.05,
r_over=0.7` case returns `INTERMEDIATE` at the default `0.6` and `TERM-OWNS(B)` at a widened `0.8`
— both assertions pass in my own dry-run. `out["bands"]["nonadditivity_max"]` now reports the value
actually applied. Genuinely fixed.

### Finding B (RED — G-2(i)/(ii) anchors unchecked) → **NOT CLOSED — G-2(ii) fix checks the wrong population**

**G-2(i) is correctly closed.** `assert_g2i_mean_h_anchor` (line 812) is called immediately after
`mean_h_full` is computed in `run_production_family` (line ~1197), checked against
`G2_MEAN_H_FULL[(venue, channel_label)]` to `1e-9` — this is exactly the anchor the draft registers
("(i) full-sample `mean_h` = 0.6658540600/... to 1e-9").

**G-2(ii) is not.** `REGISTRATION_DRAFT.md` §6 registers the anchor unambiguously as *"(ii) leave-out
of K (159) in iiib 2D = +0.086106 to 1e-6"* — the parenthetical `(159)` names the full top-z-decile
set `K`, not the target subset `K_dark`. §1 confirms the same reading under `K`'s own definition:
*"K (top-z decile) = the 159 rows ... Its leave-out is the +0.086106 anchor."* §2.3 then separately
registers a **different** statistic, `Δ_K,dark = mean_h(Λ_full − Σ_{K_dark} ln L_e) − mean_h(Λ_full)`
(144 events), explicitly as a **reported-only concordance object** whose ratio to the K anchor
(`Δ_K,dark / Δ_K`) is itself only "reported-only" — i.e. the draft expects `Δ_K,dark ≠ Δ_K` in
general (`K_hosted`'s 15 events, which *do* carry non-zero `L_cat`, are the difference), and treats
comparing them as a soft cross-check, not an identity.

The FIX-2 code computes only `delta_K_dark_leaveout` — built by masking `pops.K_dark` (144 events,
`run_production_family` line 1266: `mask = np.array([idx_pos[int(e)] for e in pops.K_dark])`) —
and feeds that single number into `assert_g2ii_delta_k_anchor` (line 822), which compares it to
`G2_DELTA_K_IIIB_2D = 0.086106` at a `1e-6` tolerance (line 826). **The 159-event leave-out `Δ_K` is
never computed anywhere in the file** (grepped every use of `pops.K` — line 1454 `G-3(b)` equality
check and the pool-exclusion `np.setdiff1d(pops.P_dark, pops.K)` for the null draws are the only
other uses; no leave-out of the bare `K` set exists). So the code:

1. Never verifies the registered G-2(ii) anchor (leave-out of **K**, 159 events) at all — the exact
   gap `DESIGN_GATE_formula.md` Finding B originally flagged is still open for this half of the
   anchor, just now hidden behind an assertion that looks like it closes it.
2. Instead asserts a **different, previously-unbanked** statistic (`Δ_K,dark`, 144 events) against
   an anchor that was established for a **different, 159-event** population, at a byte-identical
   `1e-6` tolerance. Since `K_hosted` (15 events, non-zero `L_cat`) is excluded from `K_dark` but
   included in `K`, `Δ_K,dark` and `Δ_K` are not expected to coincide to `1e-6` — this is very
   likely to raise a **false `INSTRUMENT-DEFECT`** at real-mode launch (halting the disjoint
   reader's run on a gate that is checking the wrong thing), or, if the two happen to be close
   enough by chance, to silently certify the wrong quantity under the registered anchor's name.

Neither outcome is acceptable for a byte-id anchor whose entire purpose (§6: "the anchors that
would catch a wrong population→CSV join... T0-import drift") is to be a trustworthy tripwire. This
is a genuine, code-level formula/mapping defect, confirmed by static reading (no aggregate
computed) — not a matter of opinion about which population "should" be used, since the draft names
both `K` (159) and `K_dark` (144) explicitly and distinctly and only one of them (`K`) carries the
`+0.086106` anchor.

**Fix required:** compute a genuine 159-event leave-out (`mask` over `pops.K`, not `pops.K_dark`)
as `delta_K_leaveout` and assert *that* against `G2_DELTA_K_IIIB_2D`; keep `delta_K_dark_leaveout`
(144-event) as the separate, reported-only `Δ_K,dark` object §2.3 and §4.1 register, and (ideally,
non-blocking) materialize the `Δ_K,dark/Δ_K` concordance ratio now that both numbers would exist
(closing Finding F of `DESIGN_GATE_formula.md` for free).

### Finding C (RED — G-2(iii) exclusion count discarded) → **CLOSED, confirmed**

`assert_g2iii_no_physics_floor_exclusion` (line 796) is called at all three `_load_matrix` sites —
`run_production_family` (line 1197, immediately after the call, before `mean_h_full`/`mean_h_of`/
the later `idx_pos` lookup at line 1266), `run_K_hosted_leaveout` (line 1294, before its own
`idx_pos` lookup), `run_harness_universe` (line 1371, before `logpost_full`/`map_h_full`). I traced
each call site's statement order directly in the file (not from `BUILD_RECORD.md`'s prose) — the
check precedes every downstream `event_idx`-keyed dictionary lookup in all three functions, so a
nonzero exclusion count now raises a clean `InstrumentDefect` rather than risking the uncaught
`KeyError` `DESIGN_GATE_formula.md` identified. `run_synth_check` (lines 1067–1073) exercises both
the pass (`n_excluded=0`) and raise (`n_excluded=2`) paths. Genuinely fixed.

### Finding D (AMBER — two missing INTERMEDIATE carve-outs) → **CLOSED, confirmed**

`production_ownership_disposition` (line 753) now computes `r_over_delta_F_signed = 1.0 -
sum(shares.values())` — I independently re-derived that this equals the draft's own `r/Δ_F`
convention from §2.3's definitions (`s_t = Δ_t/Δ_F` ⟹ `Σ Δ_t = Δ_F·Σs_t` ⟹ `r = Δ_F(1 − Σs_t)` ⟹
`r/Δ_F = 1 − Σs_t`, matching line 775 exactly) — and gates on `both_ge_share_own_r_negative` (both
top-two shares ≥ `share_own` with `r_over_delta_F_signed < 0`, lines 776–781) and
`sign_opposed_gt1` (every `\|s_t\|>1`, sign-opposed, lines 782–786) **before** the literal
`TERM-OWNS` test, routing both to `INTERMEDIATE`. I re-ran the reviewer's own two counter-examples
through the live function myself in the dry-run's `[SYNTH OK] ... Findings A-D counter-examples`
line (`s_B=0.55,s_g=0.52,r_over=0.07` → `INTERMEDIATE`; `s_B=3.0,s_g=-2.0,r_over=0.0` →
`INTERMEDIATE`, lines 1091–1112) — both now assert correctly, where they previously (per
`DESIGN_GATE_formula.md` Finding D) fell through to a wrong `TERM-OWNS(B)`. Genuinely fixed.

### Finding E (AMBER, not one of the "A–D" set but claimed fixed in the same round) → **CLOSED, confirmed**

`main()` no longer combines `SE_B^harn`/`SE_g^harn` in quadrature. It builds
`per_u_target_F`/`per_u_ref_F` as the element-wise sum of the per-universe B and g stencil-slope
arrays (same event order, since both terms share `K_dark_u`/`R_u`), then calls
`harness_pool_score` **once** on the summed series to get `S_F_harn, se_F_harn` directly — I traced
this in `main()` (the block computing `per_u_target_F`/`per_u_ref_F` immediately before the
`harness_disposition` call) and confirmed it matches "the delete-one-universe jackknife SE of the
summed series," the literal reading `DESIGN_GATE_formula.md` recommended. `S_F_harn` itself is
numerically identical either way (exact sum); only `SE_F_harn`/`Z_harn` change. No SYNTH regression
test accompanies this fix (the ≤10-row fixture has no per-universe structure to exercise it), which
is an acknowledged, non-blocking residual noted in `BUILD_RECORD.md` itself — consistent with my
own reading; I did not attempt to reconstruct `DESIGN_GATE_formula.md`'s 20-universe toy myself
(would require synthesizing a correlated multi-universe fixture, out of proportion to re-confirming
a already-reviewed and now-superseded-by-code-reading finding). Genuinely fixed at the code level.

Findings F and G were left unchanged, as recommended (non-blocking, reader-side or no-action items)
— consistent with my own §1.1/§1.2 tables above (concordance ratios and `I_HEAD` stencil-consistency
still unmaterialized; `stencil_slope`'s nearest-node lookup still untightened). No action required
of this round.

---

## 5. New findings (not in `DESIGN_GATE_formula.md`, found independently)

### Finding H (RED, blocking — same severity class as the old Finding B) — G-2(ii) checks `Δ_K,dark` against the `Δ_K` anchor

Detailed in §4 "Finding B" above; restated here as its own numbered item because it is a **new**
defect (introduced by the FIX-2 patch, not present — because unimplemented — when
`DESIGN_GATE_formula.md` was written) and must be fixed before real-mode launch. Location:
`highz_decomp_reads.py:1263-1269` (computation) and `:822-829` (assertion), called from
`:1269`.

### Finding I (AMBER — completeness gap, non-blocking for the disposition machinery but a registered-statistic gap) — harness control never reads the 1D channel

`REGISTRATION_DRAFT.md` §4.3 headline reads *"Harness control (67 universes, **both channels**)"*
and registers per-universe `Δ_t,u, Δ_F,u, r_u` as part of that family. `run_harness_universe` (line
1365) hardcodes `_load_matrix(u.diag_path, "combined_with_bh")` (line 1370) — the 2D channel only;
there is no call anywhere with `"combined_no_bh"` for the harness (grepped: the string
`"combined_no_bh"` appears only in `CHANNELS`, `gate_g1_closure`'s column read, and the SYNTH
fixture — never in the harness code path). The disposition machinery itself does not need a 1D
harness read (§5's harness table conditions only on the 2D `S_F^harn`/`ρ_S`/`s_t^harn` against the
iiib-2D production-owning term), so this does not corrupt the booked disposition — but it means the
registered 1D per-universe harness deltas are simply absent from `--out`, and the section header's
"both channels" is not honored. **Recommend:** either extend `run_harness_universe` to also read
`combined_no_bh` and report its per-universe `Δ_B,u` (cheap — one extra `_load_matrix` call per
universe, well inside the 0.05 CPU-h budget), or have the author strike "both channels" from §4.3
if the 1D harness read was never intended to be a registered deliverable. Non-blocking for GREEN if
the author takes the latter route; blocking if the former, since it is currently a registered
statistic silently not computed.

### Finding J (AMBER — completeness gap) — the §5 "Replicate rule" cross-family downgrade is not implemented

`REGISTRATION_DRAFT.md` §5, directly under the production-ownership table: *"Replicate rule:
TERM-OWNS(t) must hold with the same t in joint_r1 2D (the other 2D family); the 1D families must
show `Δ_B^1D` of the same sign as `Δ_B^2D` (T_B's pull is channel-common). **A miss →
INTERMEDIATE.**"* This is a disposition-modifying rule, not a reported-only cross-check — a
production family's *booked* disposition should be downgraded to `INTERMEDIATE` if the replicate
condition fails. `production_ownership_disposition` (line 753) and its call sites in
`run_production_family` (line 1174) compute each of the four families (`iiib`×{2D,1D},
`jr1`×{2D,1D}) **independently**; nothing in `main()` compares `families[("iiib","combined_with_bh")]
.disposition` against `families[("jr1","combined_with_bh")].disposition`, or the sign of
`families[(*, "combined_no_bh")].term_freeze.delta_terms["B"]` against the corresponding 2D family's
`Δ_B`. Grepped the whole file for any cross-family comparison of `.disposition` or the sign of
`Δ_B` across venues/channels: none exists. So `--out`'s per-family `disposition` field is always the
*raw*, un-replicate-checked `production_ownership_disposition()` output — a reader following
`REGISTRATION_DRAFT.md` §5 literally would have to apply the replicate downgrade by hand from the
JSON's four `delta_terms`/`disposition` entries; the script does not do it and does not flag when it
would matter. **Recommend:** add a post-hoc replicate check in `main()` after all four families are
computed, downgrading the iiib-2D family's `disposition` to `INTERMEDIATE` (with the replicate
mismatch recorded, not silently overwritten) when either sub-condition fails, mirroring how the
harness-side `s_t_harn_owning` lookup already reads across `prod_primary` and the harness pool.

---

## 6. Checklist mapping (draft item → function/line), full re-derivation

*(Superset of `BUILD_RECORD.md` §3's own table, independently re-derived; entries marked "gap"
above are annotated inline rather than repeated.)*

| draft item | function(s) | line | status |
|---|---|---|---|
| CLI == Sec.8 launch block token-for-token | `build_argparser` | 227 | ✓ all 23 launch-block flags present; `--nonadditivity-max`/`--out`/`--dry-run`/`--synth` are documented additions, not launch-block members |
| Missing-input hard INSTRUMENT-DEFECT | `InstrumentDefect`, `preflight` | 172, 266 | ✓ confirmed live (§1.5 above) |
| File pin verification | `verify_file_pins` | 274 | ✓ |
| Population construction (P_dark/K/K_dark/K_hosted/R) | `Populations`, `construct_populations` | 292, 308 | ✓ |
| Population sha256/G-3(a)/(c) gate | `verify_population_pins`, `verify_g3a_set_identity` | 344, 375 | ✓ |
| G-3(b) K identical both venues | `main()` | 1455 | ✓ |
| Harness discovery, `n_draw_requested==population` filter | `HarnessUniverse`, `discover_harness_universes` | 389, 399 | ✓ |
| Harness manifest sha256 | `harness_manifest_hash`, `verify_harness_manifest` | 216, 441 | ✓ |
| G-3(d) resolved-flags equality | `verify_g3d_resolved_flags` | 450 | ✓ |
| G-3(e) seed range | `discover_harness_universes` | 405 | ✓ |
| Harness per-universe population construction | `_harness_z`, `construct_harness_populations` | 1325, 1336 | ✓ |
| G-2(iv) harness pooled anchors | `main()` | ~1531 | ✓ |
| G-1 closure (a)-(e) | `gate_g1_closure` | 484 | ✓ |
| Term profiles `T_B, T_g` | `load_term_columns`, `compute_term_profiles` | 472, 531 | ✓ |
| `T_D` identity | `compute_T_D` | 549 | ✓ |
| Centered/reference profiles | `center_profile`, `reference_profile` | 567, 573 | ✓ |
| Term-freeze `Λ_t, Δ_t, Δ_F, r, s_t` | `term_freeze_lambda`, `delta_t`, `run_term_freeze` | 578–643 | ✓ |
| Null draws + CI99 | `null_draw_ci99` | 645 | ✓ |
| Score excess `S_t`, Welch SE | `stencil_slope`, `welch_se`, `score_excess` | 677–707 | ✓ |
| Harness pooled `S_t/S_F` + jackknife SE | `harness_pool_score` | 717 | ✓ algorithm; 1D never invoked (Finding I) |
| Production disposition (incl. Finding-D carve-outs) | `production_ownership_disposition` | 753 | ✓; no replicate downgrade (Finding J) |
| G-2(iii) exclusion-count gate | `assert_g2iii_no_physics_floor_exclusion` | 796 | ✓ (Finding C) |
| G-2(i) mean_h anchor | `assert_g2i_mean_h_anchor` | 812 | ✓ (Finding B(i)) |
| G-2(ii) `Δ_K` anchor | `assert_g2ii_delta_k_anchor` | 822 | ✗ **wrong population fed in (Finding B(ii)/H)** |
| Harness disposition (6 rows) | `harness_outcome_disposition` | 833 | ✓ |
| g-censoring rail flags | `is_railed`, `map_h_of` | 864, 868 | ✓ |
| SYNTH fixture (≤10 rows, hand-verifiable) | `make_synth_fixture` | 877 | ✓ hand-verified independently (§2 above) |
| SYNTH exercises every disposition + gate + Findings A–D counter-examples | `run_synth_check` | 910 | ✓ |
| 5-row real-slice closure (design-gate computability check) | `_five_row_slice_closure` | 1125 | ✓ dry-run reproduces `2.665e-15` |
| Production family driver (§4.1/§4.2) | `ProductionFamilyResult`, `run_production_family` | 1140, 1174 | ✓ except Finding B(ii)/H |
| K_hosted (§4.4) reported-only | `run_K_hosted_leaveout` | 1289 | ✓ |
| Per-universe harness read (§4.3) | `HarnessUniverseRead`, `run_harness_universe` | 1309, 1365 | ✓ 2D only — Finding I (1D never read) |
| `main()` dry-run/real-mode wiring, `--out` JSON | `main` | 1434 | ✓ |
| Concordance ratios / `I_HEAD` stencil-consistency (reported-only) | *(none)* | — | informational gap, non-blocking (unchanged from `DESIGN_GATE_formula.md` Finding F) |
| Stencil exact-node-presence assertion | `stencil_slope` (nearest-node, no assert) | 677 | informational gap, non-blocking (unchanged from `DESIGN_GATE_formula.md` Finding G) |
| T0 import (not re-implementation) | `_import_t0_module` | 59 | ✓ confirmed against `build_influence_vector.py` directly (§1.4 above) |
| `InstrumentDefect` contract | `InstrumentDefect` | 172 | ✓ confirmed live |

---

## 7. What this means for launch

**GREEN is withheld.** Finding H (the G-2(ii) population mismatch) is a hard, code-level defect on
exactly the gate class this node exists to trust (a byte-id anchor meant to catch a wrong
population→CSV join) — it must be fixed (compute the leave-out over `pops.K`, not `pops.K_dark`,
for the anchor check; keep `Δ_K,dark` as the separate reported-only object) before the disjoint
reader runs Sec.8 in real mode, or the run will very likely halt on a false `INSTRUMENT-DEFECT`, or
worse, silently certify the wrong number under the anchor's name. Findings I and J are completeness
gaps against the draft's own registered-statistic and disposition-rule wording; I is cheap to close
in code or cheap to resolve by author amendment to §4.3's wording, and J should be closed in code
(the replicate downgrade is a disposition rule, not a reported-only nicety, and its absence means
`--out`'s `disposition` field cannot be trusted at face value against §5 without manual, off-script
cross-checking by the reader). Findings A, C, D, E of the prior round are genuinely closed and need
no further work. This goes back to the builder for a third fix pass (Finding H at minimum;
I/J recommended) before the disjoint reader's real-mode run.
