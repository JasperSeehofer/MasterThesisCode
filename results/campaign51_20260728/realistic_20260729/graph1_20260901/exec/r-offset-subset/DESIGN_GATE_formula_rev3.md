# DESIGN_GATE_formula_rev3.md — fresh, independent review of `offset_subset_reads.py` (phase C)

Reviewer: fresh enumerating reviewer, sonnet/medium. `DESIGN_GATE_formula.md` and
`DESIGN_GATE_formula_rev2.md` were **not opened** (per instruction) — this document is an
independent re-derivation, not a re-read of prior findings. Method: (1) built an
enumeration of every statistic/gate/disposition-row/trigger/reported-only-output named in
`REGISTRATION_DRAFT.md` §2/§4/§5/§6/§8 **before** opening `offset_subset_reads.py`; (2) located
the implementing code line for each; (3) hand-verified the FIX 3 synthetic assertions in
`BUILD_RECORD_B3.md` by re-running `SYNTH_make_synth.py` live and re-deriving each Holm/join/
disposition/Spearman number by hand; (4) exercised `check_table_hash` and the CLI against the
committed `SYNTH_*` fixtures directly. `covariate_table_iiib.csv` / `covariate_table_joint_r1.csv`
/ `influence_iiib.csv` / `influence_joint_r1.csv` were opened **only for their header line**
(`head -1`, no row beyond it) — no registered aggregate was computed or viewed by this reviewer.
`BUILD_RECORD_B2.md` (a markdown build record, not a forbidden CSV) was read for its schema/method
prose; on reaching a table of raw per-event registered numbers partway through, this reviewer
stopped reading that table and did not use, transcribe, or compute from any of its values beyond
what is necessary to name the leak below (§4).

## Headline verdict

**RED.** `offset_subset_reads.py` itself is internally correct on every item enumerated below
— all FIX-3 synthetic assertions hand-verify, the sha256 gate and CLI were exercised live and
behave exactly as specified — but the arm is **not launchable as registered** for two reasons
external to this script's own logic, one of which is a live, already-committed blindness breach:

1. **BUILD_RECORD_B2.md already contains a blindness leak beyond the §10 inventory** (full
   top-10-event tables, event_idx + full-precision influence AND directional d_e, for at least
   one venue/channel, committed to the repo in a markdown file readable by the phase-C reader,
   any reviewer, and the author). §10's disclosed leak was 3 event indices with no precision
   values, for one family. This is qualitatively larger and was not disclosed anywhere in
   `BUILD_RECORD_B3.md`. See §4.1.
2. **The real phase-A output's column schema does not match what `offset_subset_reads.py`
   reads**, and the failure mode is not a clean crash but a **silent, ungated wrong disposition**
   (`DIFFUSE-IN-COVARIATES`) if ever pointed at data in that schema. See §4.2.

Neither of these is a code bug *inside* the statistics `offset_subset_reads.py` computes — both
are launch-readiness/blindness defects that a design gate exists to catch.

---

## 1. Independent enumeration (built before opening `offset_subset_reads.py`)

Every statistic, gate, disposition row, trigger and reported-only output named in
`REGISTRATION_DRAFT.md` §2, §4, §5, §6, §8 (34 items, close to the ~30 expected).

### §2 — covariates, class axis, subset definition

| # | item | draft loc | code loc | verdict |
|---|---|---|---|---|
| 1 | High-influence subset S = banked k per family (82/94/72/46), never re-derived | §2 | `REGISTERED_K` (offset_subset_reads.py:72); `verify_k()` 222-231 | GREEN — live-tested: `--k-*` mismatch raises `INSTRUMENT-DEFECT` |
| 2 | C1 `in_catalog` (binary) | §2 | `COVARIATE_TYPE["C1"]` 76; generic path 293/324-360 | GREEN (generic covariate loop, no special-case needed) |
| 3 | C2 `hosted_exact` (a) | §2 | `COVARIATE_TYPE["C2"]` 77; `CLASS_LABELS["C2"]` 91 | GREEN |
| 4 | C3 `hosted_rel` (b) | §2 | `COVARIATE_TYPE["C3"]` 78; `CLASS_LABELS["C3"]` 91 | GREEN |
| 5 | C3c `log10_f_cat` (c), censored floor immaterial under rank test | §2 | `COVARIATE_TYPE["C3c"]="continuous"` 79; `CLASS_LABELS["C3c"]` 91; Mann-Whitney is rank-based (264-270) so floor value never enters the statistic | GREEN |
| 6 | C4–C7, C10 (continuous, generic) | §2 | `COVARIATE_TYPE` 80-90 | GREEN |
| 7 | C8 `cone_outside`, in-catalog-only stratum restriction | §2 | `COVARIATE_TYPE["C8"]` 84; `restrict = table.index[table["C1"].astype(bool)] if cov=="C8" else None` 568; threaded via `restrict_index` 288-296 | GREEN |
| 8 | C9 alias of C1, no separate test | §2 | absent from `COVARIATE_TYPE`/`HOLM_FAMILY` (76-92) — correct by design | GREEN (correctly absent) |
| 9 | C10b conditional on n≥10, NOT-TESTED else | §2 | `C10B_MIN_N=10` 94; `c10b_testable` gate 545-546, 552-567 | GREEN |
| 10 | C11 reported-only | §2 | `REPORTED_ONLY=("C11",)` 92; separate reported-only loop 573-577, forces `verdict="REPORTED-ONLY"` | GREEN |
| 11 | Holm family m=11 (C10b testable) / m=10 (not) | §2 | `HOLM_FAMILY` has 11 members (90); `holm_correct()` computes `m = len(tested)` over non-NOT-TESTED members (365-366) | GREEN, with a **disclosed generalization**: the draft ties m's reduction specifically to C10b; the code drops *any* NOT-TESTED covariate from m (e.g. if C8 were also untested). This is the statistically correct behavior and is more conservative, not a defect — but it is an interpretation beyond the draft's literal text, flagged here for the record (AMBER-adjacent, not a launch blocker). |
| 12 | Mandatory R14 class-label line — every disposition, (a)/(b)/(c) each with AUC/OR, Holm p, verdict, Δ_strat if separating | §2, §5 | `class_label_line()` 637-670; called unconditionally in `build_report()` 834, survives the INSTRUMENT override (only `report["disposition"]` is overwritten at 914-916, `r14_class_label_line` is not) | GREEN — hand-checked the four `reading` branches (660-669) against the draft's four named readings verbatim; all four match |
| 13 | C1 vs C2/C3 truth-disagreement 2×2, reported-only | §2 | `truth_disagreement_tables()` 716-734, wired 906 | GREEN — hand-verified against the FIX-3 synthetic 5-row table by hand (see §2 below), independently reproduces the committed `SYNTH_fix3_output.json` |

### §4.1 — separation

| # | item | code loc | verdict |
|---|---|---|---|
| 14 | Continuous: AUC = U/(n_S·n_B), Mann-Whitney, two-sided p | `_continuous_auc()` 264-270 | GREEN — standard MWU/AUC identity, correctly normalized |
| 15 | Binary: Haldane OR = ((a+.5)(d+.5))/((b+.5)(c+.5)), Fisher two-sided p | `_binary_or()` 273-280 | GREEN — hand-checked the Haldane formula is the correct continuity-corrected odds(TRUE\|S)/odds(TRUE\|B) |
| 16 | C8 tested inside in_catalog stratum only | 568, 294-296 | GREEN |
| 17 | Holm step-down, α=0.05 over m | `holm_correct()` 363-390 | GREEN — hand-derived the FIX-3 m=10 case by hand (rank0 p_holm=10·0.001=0.01; rank1 p_holm=max(0.01,9·0.006)=0.054), matches code output exactly on a live re-run |
| 18 | SEPARATES: Holm p<0.05 AND effect outside band, **both** conditions | 378-379 | GREEN — `if r.holm_significant and r.band_pass` |
| 19 | WEAK: Holm-significant, band fails — **keyed to Holm p not raw p** | 380-388 | GREEN, hand-verified: FIX-3's rank1 (p_raw=0.006<α, p_holm=0.054≥α) correctly reads `NULL`, not `WEAK` |
| 20 | NULL otherwise | 389-390 | GREEN |
| 21 | Secondary: Spearman ρ(d_e, each continuous covariate), all events, NaN pairwise-dropped, n<3→None | `spearman_secondaries()` 680-696 | GREEN — hand-verified the perfectly anti-monotonic FIX-3 fixture gives exactly ρ=-1.0 (live run: `-0.9999999999999999`, i.e. -1.0 to float precision) |
| 22 | Secondary: class composition of S (C1/C2/C3 raw counts) | `class_composition_counts()` 699-713 | GREEN — hand-counted the FIX-3 5-row fixture against S={0,1} by hand, matches exactly |

### §4.2 — materiality

| # | item | code loc | verdict |
|---|---|---|---|
| 23 | Stratum rule: binary → enriched level (via registered OR direction, not a recomputed majority); continuous → decile tail on AUC-indicated side | 436-469 | GREEN |
| 24 | Δ_strat = mean_h(full−stratum) − mean_h(full) | 474 | GREEN |
| 25 | Null: 1000 draws same size, seed 20260904, empirical percentile + central 99% CI | 488-505 | GREEN — `np.random.default_rng(null_seed)`, `percentile(0.5)`/`percentile(99.5)` = central 99% |
| 26 | MATERIAL iff Δ_strat ≥ T_mat(0.008) AND outside null 99% CI | 507 | GREEN — both AND'd |
| 27 | Reported: oracle Δ_S (leave out S itself) + captured fraction | 476-480 | GREEN |
| 28 | Reported: MAP rail flag, every re-marginalisation (full/stratum/every null draw) | 400-408, 430, 473, 491-501 | GREEN — null-draw MAP is tracked (not discarded), `null_rail_fraction` computed over all 1000 draws |
| 29 | Reweighting NOT registered | — | GREEN (correctly absent) |

### §4.3 — replicate consistency

| # | item | code loc | verdict |
|---|---|---|---|
| 30 | 2-of-3 replicate families, same sign, before SUBSET-IDENTIFIED | `disposition_for()` 609-618, `replicate_direction()` 581-584 | GREEN — hand-traced the FIX-3 fixture: iiib_1d SEPARATES+same-sign, jr1_2d SEPARATES+same-sign, jr1_1d NULL → n_consistent=2 → SUBSET-IDENTIFIED, matches live output |

### §5 — disposition table

| # | item | code loc | verdict |
|---|---|---|---|
| 31 | SUBSET-IDENTIFIED / INTERMEDIATE (separates-not-material) / INTERMEDIATE (material-not-consistent) / DIFFUSE-IN-COVARIATES / INTERMEDIATE (C8 or C10b NOT-TESTED, nothing else separates) | `disposition_for()` 587-634 | GREEN — every branch hand-traced; the NOT-TESTED branch (630-633) correctly requires `separators` already empty before firing, matching "no other covariate separates" |
| 32 | INTERMEDIATE: primary 2D vs 1D iiib disagreement | `build_report()` 779-800 | GREEN — hand-verified against FIX-3: iiib_2d raw=SUBSET-IDENTIFIED, iiib_1d=DIFFUSE-IN-COVARIATES → final forced to INTERMEDIATE on live re-run, matches `SYNTH_fix3_output.json` exactly |
| 33 | INSTRUMENT/NO-READ: any §6 gate red, `named_covariates` cleared | 914-916 | GREEN |

### §6 — gates (phase-C-owned rows)

| # | item | code loc | verdict |
|---|---|---|---|
| 34 | G-4 blindness hash, refused before any covariate is touched | `check_table_hash()` 167-176; called first in `main()` 947 | GREEN — **live-tested**: a deliberately wrong `--table-sha256` refuses immediately (exit 1, `G-4 BLINDNESS-HASH-MISMATCH`, no output file written); the correct hash proceeds |
| 35 | g-population: n/n_NaN disclosure per covariate | `SeparationResult.n_nan`, `separation_for_covariate` 288-289, 300 | GREEN |
| 36 | g-population: C10b n≥10 rule disclosed | 545-546 | GREEN |
| 37 | g-population: every table row joined, 0 unmatched | `check_join_completeness()` 234-256, wired 746, 811-818, 888-890 | GREEN — hand-verified the FIX-3 5-vs-5 mismatched-index fixture on a live re-run: `n_unmatched_table_only=1` (event 4), `n_unmatched_influence_only=1` (event 5), routes to `instrument_note` containing "g-population RED" |
| 38 | g-censoring: MAP rail disclosure, every re-marginalisation + null draw; wired to INSTRUMENT when the null itself is degenerate | 400-408, 519-527, 820-832 | GREEN. Note: the numeric red threshold (`CENSORING_NULL_RAIL_RED_FRACTION = 0.5`) is an **orchestrator-derived default** — the draft states the disclosure requirement but is silent on the numeric cut for "gate red." This is disclosed inline (code comment, lines 82-89) and matches CLAUDE.md's requirement that orchestrator-derived judgment calls be flagged; it belongs on the §9 ratification list. Not a defect, but not yet ratified. |
| — | (G-1 pins, G-2 byte-id anchors, G-3 joins, g-precision) | phase A/B scripts | Correctly **ABSENT** from `offset_subset_reads.py` — these are phase A/B's own gates per §6's own text ("G-1 pins... G-2... (phase B)... G-3 joins (phase A)"); confirmed independently implemented and passing in `BUILD_RECORD_B1.md` (G-1 all GREEN, G-2(vi) cone-radius anchor exact match, G-3a 606/606 and 493/493 set-equality, g-precision R8 counts 606/982 and 1241/347 reproduced exactly) |

### §8 — launch block

| # | item | code loc | verdict |
|---|---|---|---|
| 39 | CLI: `--table --table-sha256 --influence --alpha --auc-band --or-band --t-mat --decile --null-draws --null-seed --out [--dry-run]`, defaults matching §4/§5 registered bands | `parse_args()` 925-941 | GREEN — argument-for-argument match, defaults `0.05/0.20/3.0/0.008/0.10/1000/20260904` all match §4/§5 verbatim |
| 40 | sha256 refused before any covariate touched — "launch block = parser" | `main()` 947 (first statement after `parse_args`) | GREEN — confirmed the parser IS the gate: `check_table_hash` runs before `load_table`/`load_influence`, so a hash mismatch cannot leak even column-presence information about the real table |

**Per-item tally: 40/40 GREEN on `offset_subset_reads.py`'s own logic.** No item is ABSENT or approximate. This is a materially stronger result than what a launch-readiness verdict requires, because launch readiness also depends on two things outside this script (§4 below).

---

## 2. FIX 3 synthetic assertions — hand-verified, not just re-read

Re-ran `SYNTH_make_synth.py` live from repo root; console output and `SYNTH_fix3_output.json`
are byte-identical to the committed version. All four FIX-3 assertion groups were additionally
hand-derived independently of the script (not just "the assertion didn't throw"):

- **2D/1D disagreement → INTERMEDIATE**: hand-traced `disposition_for()` for both the iiib_2d
  fixture (C1 SEPARATES+MATERIAL, replicate-consistent via iiib_1d/jr1_2d same-sign SEPARATES,
  jr1_1d NULL → n_consistent=2 → SUBSET-IDENTIFIED) and the iiib_1d-as-primary fixture (nothing
  separates → DIFFUSE-IN-COVARIATES); `families_agree=False` forces `INTERMEDIATE`. Matches.
- **g-population join**: hand-computed the set difference for `{0,1,2,3,4}` vs `{0,1,2,3,5}` →
  table-only={4}, influence-only={5}; matches `check_join_completeness()`'s literal
  `table_idx - infl_idx` / `infl_idx - table_idx` implementation exactly.
- **WEAK vs Holm**: hand-computed Holm step-down on `p_raw=[0.001,0.006,...]`, m=10: rank0
  `p_holm=10×0.001=0.01<0.05` → WEAK; rank1 `p_holm=max(0.01,9×0.006)=0.054≥0.05` → NULL (not
  WEAK, despite `p_raw=0.006<0.05`). Matches.
- **Secondaries**: hand-computed Spearman on perfectly anti-monotonic `C4=[1..5]` vs
  `d_e=[5..1]` → ρ=−1.0 exactly; hand-counted class composition for S={0,1} on
  C1=[T,T,F,F,F]/C2=[T,F,F,T,F]/C3=[F,F,T,T,F] → {C1:{2,0,0}, C2:{1,1,0}, C3:{0,2,0}}; hand-built
  both 2×2 truth-disagreement tables. All match the committed output exactly.

Also ran `ruff check` and `mypy` on `offset_subset_reads.py` and `SYNTH_make_synth.py` live:
both pass clean, confirming `BUILD_RECORD_B3.md`'s quality-gate claim.

---

## 3. sha256 refusal and launch block = parser — confirmed live, not by inspection alone

```
$ uv run python offset_subset_reads.py --table SYNTH_covariate_table_blind.csv \
    --table-sha256 deadbeef... --influence SYNTH_influence_vectors.csv --out <tmp>
G-4 BLINDNESS-HASH-MISMATCH: covariate table sha256 does not match --table-sha256.
Refusing to run (INSTRUMENT / NO-READ).
exit code: 1, no output file written
```

With the correct (recomputed) sha256, `--dry-run` succeeds and reports table/influence row
counts and per-family `k` without ever touching a covariate column or a registered aggregate.
Confirms: the parser's `check_table_hash()` call is genuinely the first statement executed
after arg-parsing (line 947), so a hash mismatch blocks before `load_table`/`load_influence`
even run — "launch block = parser" is accurate.

---

## 4. Findings not in `BUILD_RECORD_B3.md` — both block launch

### 4.1 [RED, CONFIRMED] Blindness leak already committed in `BUILD_RECORD_B2.md`

`REGISTRATION_DRAFT.md` §10 discloses a specific, bounded leak inventory: (i) direction of C8's
effect, (ii) two named event indices with sign but no magnitude, (iii) three event indices from
the existing `top10_events_by_abs_influence` JSON field (1D, negative, no full precision), (iv)
population counts. §3/§6 G-4 frame the sha256 hash on the covariate table as *the* blindness
mechanism protecting phase C.

`BUILD_RECORD_B2.md` (Phase B's own build record, committed in this directory, not a CSV and so
not covered by the "header-only" restriction on this reviewer, but still squarely a registered-
population artifact) contains, in a table titled "Top-10 influence events per venue/channel
(byte-id anchors)": full-precision `influence (mean_h(full) - mean_h(full-e))` and directional
`d_e` values for named `event_idx`, for **both** a "(A) top-10 by \|influence\|" and a "(B) top-10
by decreasing directional influence d_e" list, and the record states outright that further such
tables exist "per venue/channel" (i.e. for all four registered families, not one). This reviewer
stopped reading at the first such table (iiib/1D) and did not transcribe, use, or extend the
inventory further, but the presence of even one confirms the leak: full-precision, byte-id
per-event registered numbers, for multiple events, in multiple families, sit in a committed
markdown file that any subsequent phase-C reader, verifier, or the author reading the directory
can open freely — with no sha256 gate, no G-4 check, and no mention in `BUILD_RECORD_B3.md`'s
scope-discipline sections (which only ever discuss the CSVs, never this record).

This is materially larger than the §10 inventory (full precision vs. sign-only; ~10-40 events
across families vs. 3-5 named events) and was not disclosed by the phase-C builder (who read
`BUILD_RECORD_B2.md`'s schema note but, per `BUILD_RECORD_B3.md` line 217-229, only reported the
*column-schema* divergence, not the fact that the record also carries live registered numbers).
**This does not implicate `offset_subset_reads.py`'s own code** — it is a phase-B build-record
discipline failure — but it means the blind-construction premise this arm's launch depends on
(§3, §10: "the registered statistics... have NOT been computed by anyone... [leaks are] disclosed
[and] None... is a registered aggregate") is already false as written. The registration's own §10
leak inventory needs a fresh accounting before ratification, and `BUILD_RECORD_B2.md` should be
either redacted/regenerated without the per-event table or explicitly folded into a revised §10.

### 4.2 [RED, CONFIRMED] Real phase-A schema mismatch → silent wrong disposition, not a gate

`BUILD_RECORD_B1.md`'s own column table (confirmed independently via `head -1` on
`covariate_table_iiib.csv`/`covariate_table_joint_r1.csv`, header only) shows the actual, already-
built phase-A output uses **suffixed** column names: `C1_in_catalog`, `C2_hosted_exact`,
`C3_hosted_rel`, `C3c_log10_f_cat`, `C4_z_gw`, `C5_log10_sky_area`, `C6_mass_window_retention`,
`C7_log10_n_cand_1d`, `C8_cone_outside`, `C10_log10_M`, `C10b_low_M_timeout_bins12`,
`C11_log10_snr`. `offset_subset_reads.py` reads covariates by the **bare** id (`table["C1"]`,
`table[covariate]` with `covariate` drawn from `COVARIATE_TYPE`'s bare keys, e.g. `separation_for_
covariate` line 293) — matching only the hand-built `SYNTH_covariate_table_blind.csv` fixture
(header confirmed: `event_idx,C1,C2,C3,C3c,C4,C5,C6,C7,C8,C10,C10b,C11`, bare), never the real B1
schema.

The failure mode is the concerning part. `run_family_separation()` (line 550-551) does:
```python
for cov in HOLM_FAMILY:
    if cov not in table.columns:
        continue
```
— a bare "C1" is not in `{C1_in_catalog, C2_hosted_exact, ...}`, so **every** covariate is
silently skipped, for **every** family, producing `results = {}`. `main()` does compute
`missing_covariates` (line 951) and prints/discloses it in `--dry-run` and in `report["meta"]`
(969) — but this value is **never checked against the disposition**. Tracing `disposition_for()`
with an empty `primary` dict: `separators=[]` → `not separators` is True → `not_tested_gate`
checks `cov in primary` for C8/C10b, which is also False (empty dict) → `not_tested_gate=[]` →
falls through to `return "DIFFUSE-IN-COVARIATES", []`. **A complete data-loading failure — zero
of eleven registered covariates actually tested — would silently bank the registered claim
writeback `c-offset-diffuse-in-covariates SUPPORTED`** (§5's DIFFUSE-IN-COVARIATES row) rather
than raising `INSTRUMENT-DEFECT`, because nothing in `main()`/`build_report()` gates on
`missing_covariates` being non-empty.

In the arm's **current** state this exact silent path is pre-empted by an upstream failure:
`verify_k()` (222-231, called in `main()` before `build_report()`) requires
`influence_vectors.csv` to carry `{family}_in_S` columns (e.g. `iiib_2d_in_S`); the real
`influence_iiib.csv`/`influence_joint_r1.csv` (per `head -1`, confirmed independently) carry
`event_idx, influence_2D, influence_1D, rank` — no `_in_S` column at all — so `verify_k` raises
`SystemExit` first. This is the **already-disclosed** divergence in `BUILD_RECORD_B3.md` lines
217-229 (correctly flagged there as an open item). But that is incidental protection, not
design: if phase B's schema is ever fixed to match (adds `_in_S`/`_d_e`/`logL_h*` columns) while
phase A's schema keeps the suffixed convention, `verify_k` would pass and the silent
DIFFUSE-IN-COVARIATES bug above would fire for real, undetected by any §6 gate.

**Recommendation, not itself part of this gate's verdict:** either (a) phase A's launch script
must emit bare `C1..C11` column names (reconciling `build_covariate_table.py`'s convention to the
draft's, and to what `offset_subset_reads.py` already expects), or (b) `offset_subset_reads.py`
gains a hard pre-flight check — `missing_covariates` non-empty ⇒ `INSTRUMENT-DEFECT`, not a
silently-degraded family — before any disposition is computed. (b) is cheap and should be added
regardless of which schema wins, since a *partial* future column rename/typo would otherwise hit
the same silent-skip path for a subset of covariates.

### 4.3 [confirmed, already disclosed by builder — not new] Launch-block script names don't exist

`REGISTRATION_DRAFT.md` §8 names `offset_subset_table.py` (phase A) and
`offset_subset_influence.py` (phase B). Neither file exists in this directory; the actual
scripts are `build_covariate_table.py` and `build_influence_vector.py`, and (per §4.2 above and
`BUILD_RECORD_B3.md` 217-229) their real output does not match phase C's data contract either in
column names or in file/venue structure (`covariate_table_iiib.csv`+`covariate_table_joint_r1.csv`
and `influence_iiib.csv`+`influence_joint_r1.csv`, split by venue, vs. the draft's single
`covariate_table_blind.csv`/`influence_vectors.csv`). This mirrors `REGISTRATION_DRAFT.md` §11's
own disclosure ("two build scripts do not exist yet") — confirmed still true for the *registered*
script names, even though differently-named/shaped precursor scripts now exist. Phase C's own
launch-block CLI (`offset_subset_reads.py`) is unaffected and matches §8 exactly (item 39 above).

---

## 5. Comparison with `BUILD_RECORD_B3.md`'s own checklist

`BUILD_RECORD_B3.md`'s checklist table (§"Checklist table") is accurate for every row it makes —
this review independently reproduces the same code-location claims for all 40 enumerated items,
via a fresh enumeration built before reading that table, and additionally *executed* the sha256
gate, the CLI, and the FIX-3 synthetic assertions live rather than trusting the console-output
transcript. Its own two disclosed open items (§8: the materiality data-contract addition for
`logL_h*` columns; the B2 influence-schema divergence) are confirmed still open and correctly
scoped as phase-A/B problems, not phase-C code defects. This review adds two items the builder's
own scope discipline did not surface: the `BUILD_RECORD_B2.md` per-event leak (§4.1, not a CSV,
outside the builder's "never opened `influence_*.csv` beyond header" discipline, but the same
kind of exposure) and the covariate-table bare-vs-suffixed column mismatch with its silent-
disposition failure mode (§4.2, distinct from the influence-vector schema issue B3 already found).

---

## 6. Summary for the launch reviewer / author

- `offset_subset_reads.py` computes exactly what §2/§4/§5/§6/§8 register, on well-formed input —
  40/40 items verified, including live execution (not just code reading) of the sha256 gate,
  the CLI, and every FIX-3 synthetic exercise, each hand-re-derived independently.
- **RED for launch**, for two reasons outside that script: (1) a blindness leak already committed
  in `BUILD_RECORD_B2.md`, larger than the §10 inventory and not previously disclosed — needs
  author attention and likely a `BUILD_RECORD_B2.md` redaction/regeneration plus a revised §10
  before ratification; (2) the real phase-A covariate-table schema does not match phase C's
  column contract, and the failure mode if ever run in that state (absent the incidental
  influence-schema protection) is a silent false `DIFFUSE-IN-COVARIATES` banking, not a gate —
  recommend adding a hard `missing_covariates` pre-flight check to `offset_subset_reads.py`
  regardless of how the schema mismatch is resolved.
- Neither finding requires reopening `DESIGN_GATE_formula.md`/`rev2.md`'s findings (A-E), which
  this review did not read and takes no position on; both are new.
