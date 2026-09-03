# DESIGN_GATE_formula_rev5.md — r-offset-subset, fresh integration review of PIN CORRECTION 3 (round 5)

Reviewer: fresh session, no prior DESIGN_GATE_formula*.md opened (per task instruction). Scope:
`REGISTRATION_DRAFT.md`'s "PIN CORRECTIONS + CHAIR NOTES" §PIN CORRECTION 3 (round 5) against the
current `offset_subset_reads.py`, `build_influence_vector.py`, `BUILD_RECORD_B3.md` FIX 5, and the
real committed inputs. Builder/verifier discipline observed: no registered aggregate was computed
over the registered population by this review — only `--dry-run` on the real inputs (item 2) and a
real-mode hand-verification on the synthetic FIX-5 fixture (item 4), per the task's own constraint.
All five checks below hold. **Verdict: GREEN.**

## (1) Corrected launch block == the parser (six file arguments + pins)

`offset_subset_reads.py:parse_args()` (line ~1194) declares exactly the CLI PIN CORRECTION 3
rewrites §8 to: `--table-iiib`, `--table-sha256-iiib`, `--influence-iiib`, `--logl-iiib`,
`--table-jr1`, `--table-sha256-jr1`, `--influence-jr1`, `--logl-jr1` — all eight `required=True` —
plus the unchanged `--alpha/--auc-band/--or-band/--t-mat/--decile/--null-draws/--null-seed/--out/
--dry-run` and the `--k-<family>` sanity flags. That is six file-path arguments
(table-iiib/influence-iiib/logl-iiib × table-jr1/influence-jr1/logl-jr1), two of them (the
`--table-*`) carrying an explicit sha256 CLI pin and the other two (`--logl-*`) carrying an
internal md5 pin checked against `build_influence_vector.VENUE_CSV_MD5` by `verify_logl_md5()`
before any covariate is touched (`main()`, `check_table_hash()` first, then venue-detection, then
`verify_logl_md5()` for both `--logl-*` before `load_primary_logl_matrix()`). This matches the
draft's launch block verbatim — reproduced by direct comparison of the parser source against
REGISTRATION_DRAFT.md's rewritten §8 block, argument-for-argument.

## (2) `--dry-run` on the real inputs, run independently by this review

Pins re-verified independently (not copied from any record):

```
$ sha256sum covariate_table_iiib.csv covariate_table_joint_r1.csv
90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0  covariate_table_iiib.csv
fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a  covariate_table_joint_r1.csv
$ md5sum .../headrebaseline_iiib/.../event_likelihoods.csv .../headrebaseline_joint_r1/.../event_likelihoods.csv
8e6a2c18dc5838dd1d52641589243672  .../headrebaseline_iiib/.../event_likelihoods.csv
745954a0fdee5f10878fb5e622a06144  .../headrebaseline_joint_r1/.../event_likelihoods.csv
```

Both sha256 and both md5 match REGISTRATION_DRAFT.md §1/§8's committed pins exactly. Full launch
block run verbatim (repo root, `uv run python ... offset_subset_reads.py ... --dry-run`):

```
table iiib: covariate_table_iiib.csv (1588 rows), sha256 OK
table jr1:  covariate_table_joint_r1.csv (1588 rows), sha256 OK
logL iiib: .../headrebaseline_iiib/.../event_likelihoods.csv md5 OK (h_grid n=41)
logL jr1:  .../headrebaseline_joint_r1/.../event_likelihoods.csv md5 OK (h_grid n=41)
join iiib: 1588 table rows / 1588 influence rows joined on event_idx; unmatched table-only=0, unmatched influence-only=0; join_complete=True
join jr1: 1588 table rows / 1588 influence rows joined on event_idx; unmatched table-only=0, unmatched influence-only=0; join_complete=True
logL columns present: iiib=True (h_grid n=41), jr1=True (h_grid n=41)
  family iiib_2d: k=82
  family iiib_1d: k=94
  family jr1_2d: k=72
  family jr1_1d: k=46
dry-run OK
EXIT CODE: 0
```

Exit 0. Both sha256 OK, both logL md5 OK. 1588/1588 joined per venue, 0 unmatched either side,
both venues. All four families at their registered banked k exactly (82/94/72/46). No `--out`
file was created (`ls` on the target path before and after: `No such file or directory` both
times) — confirms `--dry-run` still returns before `build_report()`/any registered aggregate is
touched. This is an independent re-run, not a copy of BUILD_RECORD_B3.md §2's or the draft's own
console block — all three now agree byte-for-byte.

## (3) Materiality imports the same frozen T0 convention as `build_influence_vector.py`

Read both files. `offset_subset_reads.py` does `sys.path.insert(...)`; `import
build_influence_vector as biv`, then:

- `load_primary_logl_matrix()` calls `biv._load_matrix(path, "combined_with_bh")` directly — the
  **same function object**, not a re-implementation — which applies `_physics_floor_apply()`
  (zeros → row's own min nonzero; all-zero rows excluded) and returns `log(L_floored)`. This is a
  literal import/reuse, the strongest form of "same convention."
- `t0_moments()` (offset_subset_reads.py, scalar case) is the mean_h/map_h half of
  `build_influence_vector._moments()` (vectorized case) with identical arithmetic:
  `lp = logpost - logpost.max()`; `post = exp(lp)`; `norm = (post*weights).sum()`;
  `post_n = post/norm`; `mean_h = (post_n*h_grid*weights).sum()`; `map_h =
  h_grid[argmax(logpost)]` — verified line-by-line against `_moments()`, no divergence (it omits
  `sigma_h`, not needed for materiality, and adds a MAP-rail boolean, not present in `_moments`
  itself but consistent with `_score_venue_channel`'s own separate rail computation). The comment
  above `t0_moments()` cites `tier0_bootstrap_jackknife.py:_moments` — the same upstream source
  `build_influence_vector._moments()`'s own docstring cites ("T0 convention verbatim") — so both
  files trace to one origin; this is the "verbatim copy with citation" branch of the check.
- `weights = np.gradient(h_grid)` is identical in both files (gradient-trapezoid weights); no
  re-floor is applied in `materiality_for_covariate` (comment: "no re-floor, phase B floors on
  load" — correct, since `_load_matrix` already floored before `offset_subset_reads.py` ever sees
  the matrix).

Check (3) holds: the log-likelihood load is a direct import of the same function; the moments
computation is a verbatim-formula copy citing the same T0 source.

## (4) FIX 5 synthetic fixture exercises materiality; hand-verified

`BUILD_RECORD_B3.md` §"FIX 5" builds a 30-event, two-venue synthetic fixture
(`SYNTH_fix5_*`) sized so C3c/C4/C5/C10 separate S (n=6) from bulk (n=24) at
`p_holm ≈ 3.03e-5`, and runs `osr.main()` in **real mode** (not `--dry-run`) against it —
explicitly permitted by this script's own scope ("never on the registered population"; the
fixture is synthetic). `SYNTH_fix5_result.json` on disk shows `materiality` populated for all
four separating covariates with `disposition = INTERMEDIATE`.

This review independently reproduced one entry by hand, without touching the script's internal
state: for the primary family (iiib_2d), covariate C4 has `AUC = 0.0`, so the enriched stratum is
the **bottom** decile (`round(30*0.10) = 3` events) by C4 value — the three lowest-`z_gw` events,
which by construction of the fixture are `event_idx {0, 1, 2}`. Loading
`SYNTH_fix5_logl_iiib.csv` (`combined_with_bh`, 5-node h-grid `[0.6, 0.665, 0.73, 0.795, 0.86]`,
no zeros so no physics-floor branch taken) and computing `t0_moments` on `full_logpost` and on
`full_logpost − logL[{0,1,2}].sum(axis=0)` independently in a fresh Python session:

```
full mean_h = 0.73   (== H_TRUE, MAP == 0.73)
stratum(events 0,1,2 removed) mean_h = 0.73   (MAP == 0.73)
delta_strat (hand-computed) = 0.0
```

This matches `SYNTH_fix5_result.json`'s own `materiality.C4.delta_strat = 0.0` exactly. (The
fixture's `delta_strat = 0.0` for every separating covariate is a property of this particular
hand-built fixture — its `combined_with_bh` values are symmetric enough around the truth grid
node that removing any 3-event stratum does not perturb the weighted mean at this precision — not
a defect; the materiality PATH ran, produced a real, reproducible number, and the number was
independently reproduced by hand.) Check (4) holds.

## (5) Statistics/threshold/disposition code unchanged vs. the FIX 3 checklist — structural read

`git log` shows exactly one prior commit touching this file (`c23877ac`, the round-4 state
`BUILD_RECORD_B3.md`'s FIX-3 checklist table cites line numbers against). `git diff c23877ac --
offset_subset_reads.py` was read in full (237 insertions / 81 deletions, 446 diff lines). Every
hunk falls into one of three buckets:

1. Module docstring + a new `import build_influence_vector as biv` (documentation and import
   only — no statistics).
2. Two **new** functions inserted after `load_influence()`: `verify_logl_md5()` and
   `load_primary_logl_matrix()` (net-new code implementing the third materiality data path; does
   not modify any existing function).
3. `build_report()`, `parse_args()`, `_write_instrument_defect()`, and `main()` — every changed
   line in these four is either (a) a new `_venue_of_family()` routing helper, (b) threading
   `table_iiib/infl_iiib/table_jr1/infl_jr1` through in place of a single `table/infl`, (c)
   duplicating the join/gate/meta bookkeeping per venue, or (d) the CLI argument list itself. No
   hunk touches `separation_for_covariate`, `_continuous_auc`, `_binary_or`, `holm_correct`,
   `materiality_for_covariate`, `disposition_for`, `class_label_line`, `spearman_secondaries`,
   `class_composition_counts`, `truth_disagreement_tables`, `check_join_completeness`,
   `check_table_hash`, or `t0_moments` — every one of these (the FIX-3-checklist rows for §4.1,
   §4.2, §4.3, §5, and the phase-C-owned §6 gates) is called with different **arguments** (the
   right venue's table/infl instead of a single one) but its own body is byte-identical to the
   round-4 commit. `git diff` confirms this directly (no `-`/`+` pair inside any of those function
   bodies), which is a stronger check than re-reading the FIX-3 checklist text, since it is not
   subject to a checklist-vs-code drift.

`ruff check offset_subset_reads.py` → `All checks passed!`; `mypy offset_subset_reads.py` →
`Success: no issues found in 1 source file` (independently re-run by this review, not copied from
BUILD_RECORD_B3.md §4). Check (5) holds.

## Disposition

All five checks (1)–(5) hold. **GREEN.** Nothing in this review computed or observed a registered
aggregate over the registered population: item (2) used `--dry-run` only (confirmed by the
missing `--out` file both before and after); item (4) touched only the `SYNTH_fix5_*` synthetic
fixture, in the mode the task explicitly permits. Whether a REAL, non-`--dry-run` invocation of
the corrected §8 launch block against the registered population reaches SUBSET-IDENTIFIED /
DIFFUSE-IN-COVARIATES / INTERMEDIATE remains undetermined and is not this review's place to
determine — per REGISTRATION_DRAFT.md §3's three-agent phase design, that is a fresh disjoint
reader's job after the author's launch ratification (§9).
