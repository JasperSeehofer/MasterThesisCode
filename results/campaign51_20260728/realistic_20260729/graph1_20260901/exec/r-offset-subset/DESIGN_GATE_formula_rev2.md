# r-offset-subset — DESIGN GATE (rev2): formula-match review of `offset_subset_reads.py`

Reviewer: fresh session, formula-match lens only, second pass. Did **not** open
`DESIGN_GATE_formula.md` (the round-1 review this fix responds to — forbidden by task) and did
**not** open `INFORMATION_FORECAST.md` (standing forbidden per the arm). Did **not** compute any
registered aggregate (AUC/OR/p/Δ_strat) over the registered 1588-event population: `influence_iiib.csv`,
`influence_joint_r1.csv`, `covariate_table_iiib.csv`, `covariate_table_joint_r1.csv` were opened only
for their header line (`head -1`) to confirm column names, never for a row or a statistic. Every
number below is either (a) hand arithmetic re-derived from `REGISTRATION_DRAFT.md`'s own formulas
against the `SYNTH_*` synthetic fixtures already committed by the builder, independently
re-executed (`uv run python SYNTH_make_synth.py`, output reproduced byte-for-byte against the
committed `SYNTH_fix2_output.json`), or (b) a source-line citation from `offset_subset_reads.py`
against `REGISTRATION_DRAFT.md` §statistic/§4/§5/§6.

**Overall: RED.** Three formula-match defects reach the registered read: one §5 disposition-table
row has no implementing code path at all (§B below), one §6 gate (`g-population`, the join
completeness clause) is never checked or disclosed (§C), and the WEAK verdict is computed off the
*raw* p-value rather than the Holm-adjusted p the draft's own §4.1 prose defines "significant"
against, corrupting the mandatory R14 evidentiary line in a reachable real-data regime (§D). None of
these were touched by FIX 2 (which addressed Findings A–D of the round-1 review only) or by the
extended synthetic checks, whose four hand-verified assertions are otherwise confirmed correct
(§A). Two further registered-but-reported-only statistics are entirely unimplemented (§E) — lower
severity (they never gate the disposition) but still a registration/implementation mismatch. Launch
block ↔ CLI parser and the sha256 refusal both check out clean (§F, §G).

## A. Extended synthetic checks (FIX 2) — hand-verified, all four confirmed correct

Re-ran `uv run python SYNTH_make_synth.py` from repo root; console output and
`SYNTH_fix2_output.json` reproduce byte-for-byte what `BUILD_RECORD_B3.md` "FIX 2" already
committed. Independent hand arithmetic below (not a re-statement of the build record's own
arithmetic):

- **Finding A (binary enrichment direction).** `C_A`: S = events 0–5, `covA[:6]=[T,T,F,F,F,F]` →
  a=2 True/S, b=4 False/S; bulk = events 6–29, `covA[6:]` has 2 True (indices 6,7) of 24 → c=2,
  d=22. Haldane OR = (2+0.5)(22+0.5) / ((4+0.5)(2+0.5)) = (2.5·22.5)/(4.5·2.5) = 56.25/11.25 =
  **5.0** — matches `or_a=5.0`. `s_bool.mean()` within S = 2/6 = **0.333** (minority) — matches. The
  fixed rule (`enriched_level = sep.effect >= 1.0`, `offset_subset_reads.py:410`) picks the True
  level (OR=5.0≥1.0) → stratum = every True row in the *full* 30-row table = 4 (2 in S + 2 in bulk)
  — matches `n_stratum_fixed=4`. The pre-fix majority-of-S rule would have picked False
  (0.333<0.5) → stratum = 26 False rows — matches `n_stratum_old_buggy_would_be=26`. Confirmed:
  code at `offset_subset_reads.py:404-412` implements the OR-direction rule exactly as described,
  and it is symmetric with the continuous branch's `sep.effect >= 0.5` (AUC direction, line 427).
- **Finding B (NOT-TESTED → INTERMEDIATE).** With every `HOLM_FAMILY` covariate at verdict `NULL`
  except `C10b` at `NOT-TESTED`, `disposition_for()` (`offset_subset_reads.py:557-595`): `separators`
  is empty (no `SEPARATES`) → falls to the `if not separators:` branch (line 585) → `not_tested_gate
  = [cov for cov in ("C8","C10b") if verdict=="NOT-TESTED"]` = `["C10b"]` (non-empty) → returns
  `("INTERMEDIATE", [])` before ever reaching the `DIFFUSE-IN-COVARIATES` return on line 594.
  Matches `SYNTH_fix2_output.json`'s `finding_B.disposition == "INTERMEDIATE"` and
  `REGISTRATION_DRAFT.md` §5's verbatim row: *"C8 or C10b NOT-TESTED and no other covariate
  separates"* → INTERMEDIATE.
- **Finding C (NaN exclusion from the decile tail).** `C_C`: 5 NaN + 15 increasing reals (0..14),
  `decile=0.10`. `valid_n = 20-5 = 15`. `n_tail = max(1, round(15*0.10)) = round(1.5)`. Python 3
  banker's rounding sends 1.5 to the nearest **even** integer, 2 — so `n_tail=2`, matching
  `mat_c.n_stratum==2`. `rank(method="first", na_option="keep")` (line 429) leaves the 5 NaN rows as
  NaN (never assigned a rank), so `stratum_mask = (ranked > 13) & ~nan_mask` selects exactly ranks
  14–15 = the two highest **real** values (13, 14 at indices 18, 19) — no NaN row enters the
  stratum. Matches `n_missing=5`, `n_stratum=2`.
- **Finding D (g-censoring null-rail wiring).** `C_D`, 4 events, only event 3 informative
  (`logl_d[3] = [-100,-100,100,-100,-100]`, others all-zero/flat), `decile=0.75` → `n_tail =
  max(1, round(4*0.75)) = 3`. Full-sample logpost = event 3's row (others contribute 0) → MAP at
  index 2 (h=0.73), interior → `map_rail_full=False`, matches. Enumerating all C(4,3)=4 possible
  size-3 leave-out draws by hand: the one draw that excludes event 3 ({0,1,2}) leaves the
  informative row intact → interior MAP, no rail. The other three draws ({0,1,3},{0,2,3},{1,2,3})
  each *include* event 3 in the drawn-and-removed set, so `draw_logpost = full - logl[draw]` is
  exactly the all-zero flat array (event 3's contribution cancels, others were already 0) →
  `argmax` of an all-equal array returns index 0 (the first max) → `map_h = h_grid[0] = 0.60` →
  rails. Expected rail fraction = 3/4 = **0.75** exactly; the 400-draw run measured **0.775**
  (sampling noise, within the script's asserted `[0.65, 0.85]` tolerance) → `censoring_gate_red =
  True` (`>= CENSORING_NULL_RAIL_RED_FRACTION=0.5`, line 105/469). The record's own
  `censoring_red_covariates` replication (`SYNTH_make_synth.py:252-253`) matches
  `build_report()`'s actual wiring at `offset_subset_reads.py:683-691`, which folds any red
  covariate into `instrument_note` and — via the unconditional `if instrument_note is not None:
  report["disposition"]["value"] = "INSTRUMENT / NO-READ"` at line 753-754 — actually overrides the
  disposition. Confirmed reachable, not merely computed-and-discarded.

All four FIX 2 assertions are correct and the code they exercise matches the draft. **Neither FIX 2
nor the extended synthetic checks touch the three findings below** — they were out of FIX 2's scope
(it responded only to round-1 Findings A–D) and no synthetic exercise in `BUILD_RECORD_B3.md`
constructs a case with more than one family's separation results compared against each other, nor a
raw-significant/Holm-non-significant covariate, nor an unmatched-join scenario.

## B. §5 disposition table ↔ code branch — one row unimplemented

| §5 row (verbatim trigger) | code branch | verdict |
|---|---|---|
| SUBSET-IDENTIFIED: ≥1 SEPARATES ∧ MATERIAL ∧ replicate-consistent | `disposition_for()` lines 570-582: `mat.material` check + `n_consistent>=2` over `REPLICATE_FAMILIES` → `identified` list | maps 1:1 |
| DIFFUSE-IN-COVARIATES: no covariate SEPARATES in the primary family | line 594, reached only past the `not_tested_gate` guard | maps 1:1 |
| INTERMEDIATE — "a covariate SEPARATES but no stratum is MATERIAL" | lines 566-569: `mat is None or not mat.material` → `intermediate.append` | maps 1:1 |
| INTERMEDIATE — "MATERIAL but not replicate-consistent" | lines 576-579: `n_consistent<2` → `intermediate.append` | maps 1:1 |
| INTERMEDIATE — "C8 or C10b NOT-TESTED and no other covariate separates" | lines 591-593 (`not_tested_gate`) | maps 1:1 (Finding B, §A) |
| INTERMEDIATE — **"primary 2D and 1D iiib families disagree in disposition"** | **none found** | **NO CODE PATH** |

`iiib_1d` appears in `offset_subset_reads.py` in exactly four places: the `FAMILIES` tuple, the
`REPLICATE_FAMILIES` tuple, `REGISTERED_K`, and as a key of `per_family_sep` (computed and stored,
but only ever *read back* inside `disposition_for()`'s replicate-consistency loop, which checks
whether an *individual covariate's separation verdict/sign* agrees with the primary family — never
whether iiib_1d's own **overall disposition** agrees or disagrees with iiib_2d's). `grep -n
"iiib_1d\|disagree\|PRIMARY_FAMILY\b" offset_subset_reads.py` returns only those four structural
uses — no function anywhere computes an iiib_1d-as-primary disposition, and `disposition_for()` is
called exactly once in `build_report()` (line 671), against the iiib_2d primary family only.

This is a distinct requirement from "MATERIAL but not replicate-consistent" (which is per-covariate
sign agreement, already correctly implemented) — the draft lists them as two separate semicolon-
delimited triggers in the same table cell, so they cannot be read as duplicates of each other. As
written, the script can **never** return INTERMEDIATE via this route: a real run where iiib_2d
reads SUBSET-IDENTIFIED while iiib_1d (run through the same §4.1-4.2 logic) would show e.g.
DIFFUSE-IN-COVARIATES or a materially different named-covariate set will silently bank
SUBSET-IDENTIFIED with no trace of the disagreement in the JSON. This is exactly the class of defect
the task asked me to flag: a registered disposition row with no implementing branch, reachable on
real data, that can make the banked read wrong relative to §5 as written.

*(Note: computing an actual iiib_1d disposition would need iiib_1d's own materiality, which needs
its own `logL_h*` columns — `BUILD_RECORD_B3.md` §1 item 4 already documents that the materiality
data contract only requires `logL_h*` for the **primary** family. This is a genuine open question for
`influence_vectors.csv`'s schema — flagged for the launch reviewer, not invented by me — but the
current code doesn't even attempt a degenerate/separation-only version of the check; it is simply
absent.)*

## C. §6 gates ↔ code branch

| §6 gate | owner per draft | code branch | verdict |
|---|---|---|---|
| G-1 pins | phase A | n/a (out of phase C scope, correctly) | not phase C's job |
| G-2 byte-id anchors | phase B | n/a (out of phase C scope, correctly) | not phase C's job |
| G-3 joins (log→event_idx, in_catalog count, f_cat bounds) | phase A | n/a (out of phase C scope, correctly) | not phase C's job |
| G-4 blindness hash | phase C (explicit) | `check_table_hash()`, called first in `main()` before `load_table()` | maps 1:1 (§G below) |
| g-population — 1588 rows × 41 nodes; **"every table row joined (0 unmatched)"**; n/n_NaN disclosed; C10b n≥10 disclosed | unlabeled — by elimination, phase C, the only phase permitted to open both files (module docstring line 25-29) | `n_nan` disclosure: yes (`SeparationResult.n_nan`, populated in `separation_for_covariate`). C10b n≥10: yes (`c10b_testable`, line 515). **Row-count / join-completeness ("0 unmatched"): no check anywhere** | **partially unimplemented** |
| g-precision | phase A/B | n/a (out of phase C scope, correctly) | not phase C's job |
| g-censoring (rail disclosure, MAP flags) | phase C | `map_rail_full`/`map_rail_stratum`/`null_rail_fraction`/`censoring_gate_red`, wired into `instrument_note` (Finding D, §A) | maps 1:1 |

`grep -n "1588\|len(table)\|len(infl)\|unmatched" offset_subset_reads.py` finds `len(table)`/
`len(infl)` used only for row-count *printing* (`--dry-run`, `meta.n_events`) — never compared
against 1588, and never compared against each other. The actual join in `separation_for_covariate`
(lines 273-274) is `col.loc[col.index.intersection(s_index)]` / `...intersection(b_index)` — `pandas
.intersection()` silently drops any `event_idx` present in one side but not the other, with no error,
no count, no field in the output JSON recording how many rows (if any) were dropped. `n_total = len(table)`
is computed in `run_family_separation` (line 513) but is dead code — assigned to `_` at line 547
and never used for anything, including a population-size check.

`BUILD_RECORD_B3.md`'s FIX 2 section explicitly claims g-population is satisfied ("g-population's
n/n_NaN disclosure was already present via `SeparationResult.n_nan`") — that statement is correct as
far as it goes but addresses only the NaN-disclosure half of the g-population bullet, not the
join-completeness half ("every table row joined (0 unmatched)"), which remains unchecked and
undisclosed. On real production data this means a silent schema mismatch between phase A's table and
phase B's influence vectors (e.g. the exact kind of schema divergence `BUILD_RECORD_B3.md` §"Post-hoc
note" already flagged as observed between B2's actual output and the draft's phase-B contract) would
not be caught by any gate — it would just quietly shrink `n_s`/`n_b` for whichever covariates lose
rows, changing AUC/OR/p without any disclosure that the join was incomplete.

## D. WEAK verdict computed off raw p, not Holm-adjusted p — corrupts the mandatory R14 line

`REGISTRATION_DRAFT.md` §4.1: *"a covariate SEPARATES iff Holm-adjusted p < 0.05 AND effect outside
the practical-null band... Both conditions; a **significant**-but-small effect... is reported as
WEAK, never as SEPARATES."* The word "significant" in that sentence has just been defined one clause
earlier as Holm-adjusted p < 0.05 — the same test used for SEPARATES — so WEAK's intended condition
is `holm_significant AND NOT band_pass`.

`holm_correct()` (`offset_subset_reads.py:338-358`) instead branches:
```python
if r.holm_significant and r.band_pass:
    r.verdict = "SEPARATES"
elif (r.p_raw < alpha) and not r.band_pass:      # <- raw p, not r.holm_significant
    r.verdict = "WEAK"
else:
    r.verdict = "NULL"
```
Hand check with a realistic m=10 family: sorted raw p-values `[0.001, 0.006, 0.011, 0.02, 0.03,
0.04, 0.3, 0.4, 0.5, 0.6]`, Holm multipliers `(10-i)` for i=0..9 = `10,9,8,...,1`, running-max
adjustment: rank-2 (`p_raw=0.006`) → `p_holm = max(0.01, 9*0.006) = 0.054`. `p_holm=0.054 ≥ 0.05` →
`holm_significant=False` (correctly NOT significant after multiplicity correction). But `p_raw=0.006
< 0.05` is still true, so if this covariate's effect misses the band (`band_pass=False`), the `elif`
branch fires and it is labelled **WEAK** — even though it is not, by the draft's own definition,
"significant" at all. The intended/correct verdict here is **NULL**.

Because `p_holm ≥ p_raw` always (the Holm multiplier is ≥1), every covariate that reaches
`holm_significant=True` also has `p_raw<alpha`, so the *SEPARATES* path (which the disposition and
materiality logic actually key off) is unaffected — this is why `disposition_for()`,
`materiality_for_covariate()`, and the four §5 dispositions are not corrupted by this bug. What *is*
corrupted is the verdict string itself wherever it is surfaced verbatim: the per-family `separation`
block in the output JSON, and — more importantly — `class_label_line()` (line 598), which is the
**mandatory R14 measurement** the draft requires for every disposition ("(a) C2, (b) C3, (c) C3c...
AUC/OR, Holm p, SEPARATES / WEAK / NULL"). A class label that is raw-significant-but-not-Holm-significant
and below the effect band will be recorded as WEAK in the R14 line when the registered definition
says NULL — misstating the evidence for a decision (R14) the draft explicitly says is adjudicated
from this exact line ("This line is evidence for R14, not the R14 ruling"). This is a realistic
regime for the actual family size (m=10/11, n_S≈82) — not a synthetic-only edge case — and none of
`BUILD_RECORD_B3.md`'s FIX 2 exercises construct a raw-significant/Holm-non-significant case, so it
was never caught.

## E. §4.1/§2 registered secondary statistics — entirely unimplemented (lower severity, non-gating)

`grep -in "spearman\|composition\|secondary"` and a search for a 2×2 cross-tab both return nothing
in `offset_subset_reads.py`. Two registered items are absent from the code and from
`build_report()`'s output entirely:

- §4.1 secondary: *"Spearman ρ between d_e and each continuous covariate over all 1588 events; the
  class composition of S as raw counts (C1/C2/C3 tables)."*
- §2 (class-label paragraph): *"C1 vs C2/C3 disagreement (truth-hosted events labelled dark and vice
  versa) is reported as a 2×2 table per label, reported-only."*

These are marked reported-only and do not feed `disposition_for()`, so their absence cannot flip a
disposition — lower severity than §B/§C/§D. But they are registered statistics under §4 (the section
I was asked to check), and phase C is the only phase with both C1/C2/C3 and `d_e`/S-membership in
scope to compute them (phase A never sees influence, phase B never sees covariates) — so by the same
elimination argument as §C, this is phase C's gap, not another phase's. A real run's
`offset_subset_result.json` would be missing content the draft pre-registered, silently.

## F. Launch block ↔ CLI parser — clean match

`REGISTRATION_DRAFT.md` §8 phase C line: `--table --table-sha256 --influence --alpha 0.05 --auc-band
0.20 --or-band 3.0 --t-mat 0.008 --decile 0.10 --null-draws 1000 --null-seed 20260904 --out
[--dry-run]`. `parse_args()` (`offset_subset_reads.py:763-779`) defines exactly these eleven flags
with identical names and identical defaults (`0.05`/`0.20`/`3.0`/`0.008`/`0.10`/`1000`/`20260904`),
plus four optional `--k-<family>` sanity flags (not present in the launch block) whose defaults
(`82`/`94`/`72`/`46`) exactly match `REGISTERED_K` — harmless additions, not a launch-block mismatch,
since the launch block omitting them means the registered defaults are used unmodified. Clean.

## G. sha256 refusal (G-4) — clean match

`main()` (line 782) calls `check_table_hash(args.table, args.table_sha256)` as the **first**
statement, before `load_table()` or `load_influence()` are ever called — so a hash mismatch raises
`SystemExit` before either input file's contents (table or influence) are read for any covariate or
statistic. Matches the draft's G-4 text exactly ("phase C refuses to run unless..."). Confirmed by
`BUILD_RECORD_B3.md` Exercise 3's own test and independently re-derivable from the code's line order
without re-running anything. Clean.

## Summary for the launch reviewer

RED on: (B) the "primary 2D vs 1D disagree" INTERMEDIATE trigger has no code, (C) g-population's
join-completeness clause is unchecked/undisclosed, (D) WEAK is keyed to raw p instead of
Holm-adjusted p, corrupting the mandatory R14 line in a realistic regime. (E) is a real but
lower-severity registration/implementation gap (reported-only secondaries missing entirely). (A),
(F), (G) are clean. None of B–E were exercised by FIX 2's four assertions, which are otherwise
correctly implemented and correctly hand-verified.
