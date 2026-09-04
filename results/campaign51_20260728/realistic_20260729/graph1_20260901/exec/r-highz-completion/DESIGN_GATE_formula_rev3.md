# DESIGN_GATE_formula_rev3.md — r-highz-completion, FRESH formula/integration re-review (round 4)

Reviewer: fresh formula/integration reviewer, no prior context, spawned specifically to (a) build my
own enumeration of `REGISTRATION_DRAFT.md` + `MECHANISM_NOTE.md` before opening any prior design-gate
file, and (b) confirm whether `DESIGN_GATE_formula_rev2.md`'s three findings — **B-half/Finding H**
(G-2(ii) anchored the wrong population), **Finding I** (1D harness channel never read), **Finding J**
(§5 Replicate rule not implemented) — that `BUILD_RECORD.md` §9 "FIX 3" claims to have closed, are
**actually** closed in the code, not merely addressed in prose. `INFORMATION_FORECAST.md` was **not
opened** (forbidden). No registered aggregate (`Δ_F`, `Δ_t`, share, `S_t`, `S_F`, harness pooled
value, null CI99 of `Δ_F`) was computed by me over `K_dark`/`R`/`K`/`P_dark` or any harness universe's
real `event_likelihoods.csv` — every number below is from the SYNTH fixture, an independent
re-implementation of the leave-out formula run on that same fixture, the `--dry-run` console output
(real inputs, gates/pins/counts only), or a source-line citation.

## Method

1. Read `REGISTRATION_DRAFT.md` and `MECHANISM_NOTE.md` in full and built my own enumeration of the
   populations (§1), the term identity (§2.1, (I-2D)/(I-1D)), the term-freeze/score-excess objects
   (§2.3/§2.4), the disposition tables and every named carve-out (§5, both tables), the gates (§6),
   and the launch CLI (§8) — **before** opening `DESIGN_GATE_formula.md`, `DESIGN_GATE_formula_rev2.md`,
   `DESIGN_GATE_computability.md`, or `BUILD_RECORD.md`.
2. Read `highz_decomp_reads.py` top to bottom (all 2015 lines) and mapped every enumerated item to a
   function/line, independently of any prior gate's own checklist.
3. Ran `--dry-run` on the real, pinned §8 launch block myself (reproduced in §3 below).
4. Only then opened `DESIGN_GATE_formula.md` and `DESIGN_GATE_formula_rev2.md` and checked, code-line
   by code-line, whether rev2's three open items (Finding H / the G-2(ii) half of Finding B, Finding
   I, Finding J) are genuinely closed — not against `BUILD_RECORD.md`'s own prose claims.
5. Hand-verified the specific numeric claim FIX 3's SYNTH extension makes for Finding H (`K` vs
   `K_dark` leave-outs on the extended fixture differ by `> 1e-5`) with a **from-scratch,
   independently written** re-implementation of `leaveout_delta_mean_h`'s formula (not importing the
   script's function), run on the same fixture construction (`make_synth_fixture`'s stated
   parameters: 6 events × 5 nodes, seed 0, slopes `[0,0,0,3,5,5]`, `K_idx={3,4,5}`, `K_dark_idx={4,5}`).
6. Confirmed the T0-import contract and the population/pin machinery are unchanged from rev2's already
   -confirmed-closed findings (A, C, D, E) — read the relevant call sites again rather than trusting
   the prior "closed" verdict, since a later fix round could in principle have regressed them.

## Verdict: **GREEN**

All three items this round was asked to verify are genuinely closed in the code, with the correct
formula and the correct population fed to each. `DESIGN_GATE_formula.md`'s Findings A, C, D, E and
`DESIGN_GATE_formula_rev2.md`'s Finding H (the G-2(ii) population fix), Finding I (1D harness read),
and Finding J (replicate-rule downgrade) are all confirmed CLOSED by independent code reading, hand
arithmetic on the extended SYNTH fixture, and a live `--dry-run` on the real, pinned inputs. Findings
F and G (both explicitly reported-only / non-blocking in every prior round) remain open exactly as
before, by the author's own prior "non-blocking" disposition — they do not gate GREEN.

---

## 1. My own enumeration → code mapping (built before opening any prior gate file)

### 1.1 Populations (§1) and the K vs K_dark distinction

| object | draft definition | code | status |
|---|---|---|---|
| `P_dark` | `C7_log10_n_cand_1d == 0.0` | `construct_populations` (`dark_mask`, line 322) | ✓ |
| `K` (159, both venues) | top decile by `C4_z_gw.rank(method="first")`; **the +0.086106 leave-out anchor is registered on THIS set** (§1: *"Its leave-out is the +0.086106 anchor"*; §6 G-2(ii): *"leave-out of K (159)"*) | `construct_populations` (`k_set`, line 327) | ✓ |
| `K_dark = K ∩ P_dark` (144 iiib / 111 jr1) | the registered TARGET set for the term-freeze machinery; its leave-out `Δ_K,dark` is explicitly `§2.3` **reported-only concordance**, not the G-2(ii) anchor object | `construct_populations` (`k_dark`, line 331) | ✓ |
| `K_hosted = K \ P_dark` (15/48) | reported-only, `L_cat ≠ 0` | `construct_populations` (`k_hosted`, line 332) | ✓ |
| `R` | lower half by rank of `P_dark \ K` | `construct_populations` (`r_set`, line 337) | ✓ |

`load_covariate_table` (line 309) sets `event_idx` as the DataFrame index, so `table.index` used
throughout `construct_populations` is genuinely `event_idx`, not a row-position accident. Dry-run
reproduces every population sha256 and count exactly (§3 below): `iiib` 606/159/144/15/231, `jr1`
493/159/111/48/191 — matching `REGISTRATION_DRAFT.md` §1 to the byte.

### 1.2 Term identity, disposition tables, gates — unchanged from prior rounds, re-verified

I independently re-read `compute_term_profiles`/`compute_T_D` (line 531/549, the (I-2D)/(I-1D)
identity), `term_freeze_lambda`/`delta_t`/`run_term_freeze` (578–643, the `Λ_t`/`Δ_t`/`Δ_F`/`r`/`s_t`
machinery), `production_ownership_disposition` (753, all four branches including the two Finding-D
carve-outs), `harness_outcome_disposition` (948, all six rows), and `gate_g1_closure` (490, G-1a–e) —
all match `REGISTRATION_DRAFT.md` §2/§5/§6 exactly, consistent with `DESIGN_GATE_formula_rev2.md`'s
own re-derivation. I did not find any regression against the prior rounds' confirmed-closed findings
(A, C, D, E) — the `nonadditivity_max` parameter is still threaded through
(`production_ownership_disposition`'s signature, line 767, forwarded from `run_production_family`
line 1414 and `main`'s call site), the two Finding-D carve-outs (`both_ge_share_own_r_negative`,
`sign_opposed_gt1`, lines 782–792) still precede the literal TERM-OWNS test, and
`assert_g2iii_no_physics_floor_exclusion` (881) is still called at all three `_load_matrix` sites
(`run_production_family` line 1426, `run_K_hosted_leaveout`, `run_harness_universe` line 1622) before
any `event_idx`-keyed lookup.

### 1.3 CLI: §8 launch block token-for-token

`build_argparser` (line 233) defines exactly the 23 required flags of §8's launch block, plus the
documented additions `--nonadditivity-max` (default `0.6`), `--out`, `--dry-run`, `--synth`. No flag
renamed or removed relative to the launch block. ✓

---

## 2. This round's target: rev2's Finding H, Finding I, Finding J

### 2.1 Finding H — G-2(ii) must anchor the **159-event K** leave-out, not `K_dark` → **CLOSED, confirmed**

`assert_g2ii_delta_k_anchor` (line 907) now takes a parameter explicitly named `delta_K_leaveout`
(not `delta_K_dark_leaveout`) and its docstring states the distinction in the exact terms
`DESIGN_GATE_formula_rev2.md` demanded: *"the registered anchor... is the leave-out of the FULL
top-z-decile set K (159 events in iiib), not the 144-event K_dark subset."*

Tracing `run_production_family` (line 1403), for `channel == "combined_with_bh"` only (the anchor is
iiib-2D-only, matching the registered scope):

```
1500  delta_K_leaveout = leaveout_delta_mean_h(
1501      logpost_full, logL, event_idx_full, h_grid, weights, mean_h_full, pops.K
1502  )
1503  assert_g2ii_delta_k_anchor(delta_K_leaveout, venue)
1504  delta_K_dark_leaveout = leaveout_delta_mean_h(
1505      logpost_full, logL, event_idx_full, h_grid, weights, mean_h_full, pops.K_dark
1506  )
```

`pops.K` (159 events) is fed to the anchor-gated call; `pops.K_dark` (144 events) is computed
separately via the same shared helper (`leaveout_delta_mean_h`, line 926) and is **never** passed to
`assert_g2ii_delta_k_anchor`. Both are returned on `ProductionFamilyResult` (`delta_K_leaveout`,
`delta_K_dark_leaveout`, lines 1383–1384) and both land in `--out`'s `production_families.*` entries
as **separate, distinctly-named fields** (`main`, lines 1930–1936), alongside a new
`concordance_K_dark_over_K_reported_only` ratio — closing `DESIGN_GATE_formula.md`'s old Finding F
for this pair, for free, as the rev2 reviewer recommended (non-blocking bonus, not required for
GREEN). This is exactly the fix rev2 specified: the byte-id anchor now gates the object the draft
actually names, and the previously-conflated 144-event object survives as its own, clearly-labeled,
non-gating reported field — I confirmed by reading both the computation and the `--out` assembly that
no code path silently substitutes one for the other.

**Hand-verification on the extended SYNTH fixture (independent re-implementation, not importing the
script's function):** `make_synth_fixture` (line 992) was extended with event 3 — a `K_hosted`-style
event (`slope_B[3] = 3.0`), present in `K_idx = {3,4,5}` but absent from `K_dark_idx = {4,5}` — so
`K`'s leave-out and `K_dark`'s leave-out are constructed to differ. I wrote a fresh Python snippet
(not importing `highz_decomp_reads.py`) that reconstructs the fixture from the stated parameters
(base_B from `default_rng(0).normal(0,1,6)`, slopes `[0,0,0,3,5,5]`, flat `T_g`, `T_D≡0`,
`weights=np.gradient(h_grid)`, `mean_h_of` as the log-sum-exp weighted mean, `leaveout_delta_mean_h`'s
formula as stated: `mean_h(Λ_full − Σ_{events} ln L_e) − mean_h(Λ_full)`) and evaluated both
leave-outs independently:

```
delta_K      = -0.0006488129478564586   (events {3,4,5} removed)
delta_K_dark = -0.0004988275708949219   (events {4,5} removed)
abs diff     =  0.0001499853769615367   (1.5e-4, > the script's asserted 1e-5 threshold
                                          and > G2_DELTA_K_TOL = 1e-6 by two orders of magnitude)
```

This matches, to the digit, the values the in-file SYNTH assertion (lines 1246–1270) computes via the
script's own `leaveout_delta_mean_h` and asserts `abs(delta_K_synth - delta_K_dark_synth) > 1e-5` —
confirming both that the assertion is non-vacuous (the gap is real, not float noise) and that the
formula the script uses is the one the draft specifies, independently reproduced outside the script.
The live `--dry-run` transcript (§3 below) confirms this assertion actually runs and passes today
(`[SYNTH OK] ... Finding H K-vs-K_dark leaveout ...`).

### 2.2 Finding I — harness control must read **both channels** → **CLOSED, confirmed**

`run_harness_universe` (line 1602) now takes a `channel: str = "combined_with_bh"` parameter (default
preserves old 2D-only call sites) and selects its separable terms via `_separable_terms_for_channel`
(line 157: `("B","g")` for `combined_with_bh`, `("B",)` for `combined_no_bh`) — the same helper
`run_production_family` uses, so "which terms does this channel have" has one source of truth across
both the production and harness code paths.

In `main()` (line 1783):

```
1783  harness_reads_by_channel: dict[str, list[HarnessUniverseRead]] = {
1784      channel: [
1785          run_harness_universe(u, args.h_true, args.decile, stencil, channel) for u in universes
1786      ]
1787      for channel in CHANNELS
1788  }
1789  harness_reads = harness_reads_by_channel["combined_with_bh"]
```

`CHANNELS = ("combined_no_bh", "combined_with_bh")` (line 156) — the dict comprehension genuinely
iterates both channel strings, over all 67 universes each (134 `_load_matrix` calls total, confirmed
by grep: `"combined_no_bh"` now appears in this loop's channel argument path, not only in `CHANNELS`/
`gate_g1_closure`/the SYNTH fixture as rev2 found). The disposition-critical quantities
(`pooled_sizes`, `S_F_harn`, `SE_F_harn`, `Z_harn`, `ρ_S`, `harness_disposition`) are, as intended and
as the launch instructions require, still derived from `harness_reads` (the 2D channel) only — I
traced every one of these names forward from `harness_reads_by_channel["combined_with_bh"]` and found
no code path that accidentally mixes in the 1D reads. The 1D channel's pooled `S_B^harn`/`SE_B^harn`
(delete-one-universe jackknife via the same `harness_pool_score`, lines 1824–1840) is computed
separately and reported under its own `--out` keys (`harness.channel_1D_pooled_S`,
`channel_1D_pooled_SE_jackknife`, `channel_1D_n_universes_railed`, lines 1984–1986) — genuinely
materializing the registered-but-previously-absent 1D per-universe harness statistic §4.3 names,
without touching the booked disposition machinery. `_separable_terms_for_channel`'s two return values
are asserted directly in `run_synth_check` (line 1273–1274, no CSV needed — this is the pure
channel→term-set mapping, exercised without opening a real per-universe file, an honest scope match
for a `<=10`-row SYNTH fixture).

### 2.3 Finding J — the §5 Replicate rule must downgrade the booked disposition and record why → **CLOSED, confirmed**

`apply_replicate_rule` (line 817) is called once in `main()` (line 1776), after all four production
families (`iiib`/`jr1` × 2D/1D) are computed. I re-derived the registered condition from
`REGISTRATION_DRAFT.md` §5 myself before reading the function: *"TERM-OWNS(t) must hold with the same
t in joint_r1 2D... the 1D families must show `Δ_B^1D` of the same sign as `Δ_B^2D`... A miss →
INTERMEDIATE."* The code:

- `_owning_term` (802) parses `"TERM-OWNS(X)"` → `"X"` (`None` for any other disposition string).
- If the iiib/2D family itself is not `TERM-OWNS`, the check is vacuous — booked = raw, no downgrade
  (lines 842–848), correctly matching "nothing to replicate-check" rather than forcing a spurious
  downgrade on a family that never claimed ownership.
- Otherwise: `same_t_replicate = (t_iiib == t_jr1)` (850) — correctly `False` if `jr1`/2D is not
  itself `TERM-OWNS` (`t_jr1 = None ≠ t_iiib`), not just on an explicit different-term mismatch; the
  two 1D-vs-2D sign checks (`_same_sign`, 854, applied per venue at 862/867) use `np.sign` equality on
  `Δ_B`, matching "the same sign" literally.
- Any one of the three sub-conditions failing sets `downgraded=True`, `booked_disposition =
  "INTERMEDIATE"`, and appends a **human-readable reason string per failing sub-condition** (852, 864,
  869) to `ReplicateRuleResult.reasons` — so a miss is never silent.

`--out`'s `replicate_rule` section (lines 1941–1952) records, distinctly, `family`, `raw_disposition`
(the untouched, per-family value — still visible in `production_families.iiib_2D.disposition`),
`booked_disposition`, `downgraded`, and `reasons` — a reader following §5 does not have to reconstruct
the replicate check by hand from the four families' raw JSON fields, and the raw value is preserved
alongside the booked one rather than being overwritten (so nothing is lost if the replicate rule
itself is later questioned). Critically, `main()`'s harness-side owning-term lookup was rewired to use
the **booked**, not raw, disposition (line 1882: `ordered[0][0] if
replicate_rule.booked_disposition.startswith("TERM-OWNS") else None`) — so a replicate-rule downgrade
correctly propagates into `prod_owning_term`/`s_t_harn_owning` and therefore into
`harness_outcome_disposition`'s `ESTIMATOR-INTERNAL candidate` gate, exactly as it must (a term whose
"ownership" claim just failed cross-family replication should not be handed to the harness comparison
as if it were trustworthy).

`run_synth_check` (1308–1339) exercises all four cases with hand-built `ProductionFamilyResult`
stand-ins (no CSV, no aggregate — an honest SYNTH-fixture scope for a cross-family logic check): a
passing 4-family set (booked unchanged, `downgraded=False`); a same-t miss (`jr1`/2D owns `g` instead
of `B` → `INTERMEDIATE`, non-empty `reasons`); a sign miss (`iiib`/1D `Δ_B` negative against a
positive `iiib`/2D `Δ_B` → `INTERMEDIATE`); and the vacuous case (`iiib`/2D itself `DIFFUSE-IN-TERMS`
→ raw passes through unchanged). I read each assertion and confirm all four are logically exhaustive
of the cases §5's rule can encounter and match the draft's stated condition exactly — no case is
missing (e.g., a `jr1`-1D-only sign miss is symmetric to the `iiib`-1D-only case tested and uses the
identical `_same_sign` call, so it is covered by the same code path, not by a separate untested
branch).

---

## 3. `--dry-run` on the real, pinned §8 launch block (run by me, verbatim CLI, appending `--dry-run`)

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
[SYNTH OK] closure identity, disposition rows (production 4 + harness 6), G-1 pass/fail path,
  Findings A-D counter-examples, Finding H K-vs-K_dark leaveout, Finding I channel term selection,
  Finding J replicate-rule pass/miss
[dry-run] gates + byte-id anchors only, no --out written, no registered aggregate computed.
```

Exit code **0**. Population counts (606/144/231 iiib; 493/111/191 jr1) match §1 exactly. `ls` on the
`--out` path after the run confirmed no file was written. No registered aggregate computed —
consistent with the task's `--dry-run` constraint on reviewers. This reproduces
`DESIGN_GATE_formula_rev2.md`'s §3 transcript byte-for-byte, with the same three new `[SYNTH OK]`
suffixes (`Finding H`/`Finding I`/`Finding J`) that `BUILD_RECORD.md` §9 FIX 3 claims to have added —
confirmed genuinely present and passing, not just claimed.

---

## 4. Checklist mapping (my own enumeration; superset check against `BUILD_RECORD.md` §9's FIX 3 table)

| draft item | code | status |
|---|---|---|
| G-2(ii) anchors the 159-event `K` leave-out, not `K_dark` | `assert_g2ii_delta_k_anchor` fed `delta_K_leaveout` (masks `pops.K`) | ✓ CLOSED (§2.1) |
| `Δ_K,dark` (144-event) kept as a separate, non-gated, reported-only object | `ProductionFamilyResult.delta_K_dark_leaveout`, `--out` field of the same name | ✓ CLOSED (§2.1) |
| `Δ_K,dark / Δ_K` concordance ratio materialized (bonus, non-blocking) | `concordance_K_dark_over_K_reported_only` | ✓ present |
| Harness control reads BOTH channels (§4.3 "both channels") | `harness_reads_by_channel` over `CHANNELS`, `run_harness_universe(..., channel=...)` | ✓ CLOSED (§2.2) |
| Disposition machinery stays 2D-only (unchanged, as the launch instructions require) | `harness_reads = harness_reads_by_channel["combined_with_bh"]`; `pooled_sizes`/`S_F_harn`/`Z_harn`/`ρ_S`/`harness_disposition` all trace to it | ✓ confirmed unchanged |
| 1D channel pooled `S`/`SE` reported in `--out` | `harness.channel_1D_pooled_S`, `channel_1D_pooled_SE_jackknife`, `channel_1D_n_universes_railed` | ✓ present |
| §5 Replicate rule: same owning term iiib/2D vs jr1/2D | `apply_replicate_rule` → `same_t_replicate` | ✓ CLOSED (§2.3) |
| §5 Replicate rule: `Δ_B^1D` same sign as `Δ_B^2D`, both venues | `apply_replicate_rule` → `sign_ok_iiib`, `sign_ok_jr1` | ✓ CLOSED (§2.3) |
| A replicate miss downgrades the **booked** disposition to `INTERMEDIATE`, reasons recorded | `ReplicateRuleResult.downgraded`/`.reasons`; `--out.replicate_rule` | ✓ CLOSED (§2.3) |
| Booked (not raw) disposition feeds the harness-side owning-term comparison | `prod_owning_term` keyed off `replicate_rule.booked_disposition` | ✓ CLOSED (§2.3) |
| Prior-round Findings A, C, D, E (nonadditivity_max threading, G-2(iii) exclusion gate, the two Finding-D carve-outs, joint harness jackknife) | unchanged, re-verified this round | ✓ still closed, no regression found |
| Findings F (concordance/stencil-consistency ratios), G (stencil exact-node-presence) | unchanged | informational, non-blocking — author's own prior disposition, not re-litigated |
| `--dry-run` on the real §8 launch block: exit 0, counts 606/144/231, manifest matched, 3 new `[SYNTH OK]` suffixes present | reproduced live, §3 above | ✓ confirmed |
| `--out` not written by `--dry-run` | confirmed (`ls` after run) | ✓ |

---

## 5. What this means for launch

**GREEN.** Every item this round was tasked to verify — the G-2(ii) anchor now checking the
159-event `K` leave-out with `Δ_K,dark` reported separately, both harness channels being read (with
the booked disposition still correctly 2D-only), and the §5 Replicate rule genuinely able to downgrade
the booked disposition with the reason recorded — has a correct, traced code path, confirmed by
independent reading, independent hand arithmetic on the extended SYNTH fixture (not importing the
script's own function), and a live `--dry-run` on the real, pinned inputs. No new defect was found in
this pass beyond the two already-disclosed, author-dispositioned non-blocking informational items (F,
G) carried unchanged from `DESIGN_GATE_formula.md`. The disjoint reader may run §8 in real mode.
