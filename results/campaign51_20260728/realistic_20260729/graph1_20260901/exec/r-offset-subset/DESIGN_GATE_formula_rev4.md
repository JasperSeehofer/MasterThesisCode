# DESIGN_GATE_formula_rev4.md — r-offset-subset, FRESH integration review

Reviewer: fresh integration reviewer (rev4). **Did not open** `DESIGN_GATE_formula.md`,
`DESIGN_GATE_formula_rev2.md`, or `DESIGN_GATE_formula_rev3.md` — this review is
independently re-derived from `REGISTRATION_DRAFT.md` (incl. the PIN CORRECTIONS
section), the BUILD_RECORD_B1/B2/B3 records, the built CSVs' own headers, and the
scorer source (`offset_subset_reads.py`). Quotes of prior findings below are taken only
from `BUILD_RECORD_B3.md`'s own self-description of what it fixed and why — never from
the design-gate documents themselves.

**Constraints observed:** no registered aggregate was computed over the registered
1588-event population at any point in this review. Real CSVs (`covariate_table_{iiib,
joint_r1}.csv`, `influence_{iiib,joint_r1}.csv`) were opened only for `head`/row-count/
sha256 purposes, plus `--dry-run` invocations of the actual scorer (which load, hash,
join, and count — no statistic computed, per the script's own `--dry-run` contract).
One full statistical run was performed, but only on `SYNTH_make_synth.py`'s ≤10-row
synthetic fixture, which is explicitly permitted. Every number quoted below carries its
source path.

## Verdict: **GREEN** on all four assigned checks, with one pre-existing, still-open,
non-blocking-to-this-gate item carried forward for the launch ratification record (§5).

---

## 1. Column mapping vs. real headers and BUILD_RECORD_B1/B2 definitions

**Real headers (confirmed by `head -1`, this review):**

```
covariate_table_iiib.csv / covariate_table_joint_r1.csv:
event_idx,C1_in_catalog,C4_z_gw,C5_log10_sky_area,C8_cone_outside,C10_log10_M,
C10b_low_M_timeout_bins12,C11_log10_snr,C2_hosted_exact,C3_hosted_rel,
C3c_log10_f_cat,C3c_censored,C6_mass_window_retention,C7_log10_n_cand_1d

influence_iiib.csv / influence_joint_r1.csv:
event_idx,influence_2D,influence_1D,rank
```

`offset_subset_reads.py`'s `COVARIATE_COLUMN_MAP` (lines 111–124) maps all twelve bare
ids it declares (C1, C2, C3, C3c, C4, C5, C6, C7, C8, C10, C10b, C11 — i.e. `HOLM_FAMILY`
∪ `REPORTED_ONLY`) onto exactly these real column names, 1:1, with no typos or
transpositions — checked by direct string comparison against the `head -1` output above,
not by re-reading the code's own claim. C9 is correctly absent from the map (§2 of the
draft: alias of C1, no production column), and the module-level `assert` at line 125
enforces that the map's key set never silently drifts from the registered family.

Cross-checked against `BUILD_RECORD_B1.md`'s "Column definitions (exact, as
implemented)" table: every `(id, real column, definition, source)` row there agrees with
`COVARIATE_COLUMN_MAP` and with `COVARIATE_TYPE` (binary/continuous split: C1/C2/C3/C8/
C10b binary; C3c/C4/C5/C6/C7/C10/C11 continuous — matches the draft §2 table exactly).

**C3c and per-family influence/d_e**, specifically:
- `C3c_log10_f_cat` is read as continuous with no special-casing of the censored floor
  (−320.0) beyond what `BUILD_RECORD_B1.md` documents (floor applied at load, in Phase
  A, not in the reader) — correct, since Mann-Whitney is rank-based and the draft's own
  §2 gate note says the floor value is immaterial as long as it's below every finite
  ratio; `C3c_censored` exists as a disclosure flag but is (correctly) not itself a
  registered covariate.
- `influence_2D`/`influence_1D` in the real files **are already the directional
  statistic d_e**, not the raw `mean_h(full) − mean_h(full−e))` the column names would
  suggest — confirmed independently by reading `build_influence_vector.py` lines 277–309
  (`"influence_2D": d_e_2d, "influence_1D": d_e_1d`) rather than trusting
  `BUILD_RECORD_B2.md`'s prose claim of the same. `FAMILY_D_E_SOURCE_COL` (lines
  143–148) aliases these correctly per venue/channel, and `load_influence()` **recomputes
  its own rank from the correct source column per family** (line 359,
  `df[src_col].rank(...)`) rather than reusing the CSV's single `rank` column — material,
  because `BUILD_RECORD_B2.md` itself states that column is "by decreasing
  `influence_2D`" only, i.e. it would be the *wrong* ranking if reused verbatim for the
  1D families. The reader avoids that trap.
- `REGISTERED_K` (82/94/72/46 for iiib_2d/iiib_1d/jr1_2d/jr1_1d) matches
  `BUILD_RECORD_B2.md`'s `banked_k` column exactly for all four families.

**Result: PASS.** No mismatch found between the scorer's column mapping and either the
real, built headers or the two builders' own definitions.

## 2. Missing-column pre-flight is a hard INSTRUMENT-DEFECT

Exercised independently (not reusing `BUILD_RECORD_B3.md`'s own fixtures): took the real
`covariate_table_iiib.csv`, dropped `C4_z_gw`, renamed the file to preserve venue
detection (`my_bad_schema_iiib.csv`), recomputed its sha256, and invoked the scorer in
both `--dry-run` and real mode:

```
$ uv run python offset_subset_reads.py --table my_bad_schema_iiib.csv \
    --table-sha256 8a405f84d6f5071ab21d7de9ba9d9774e47713ec14549b37bfd504b38a6428f2 \
    --influence influence_iiib.csv ... --out /tmp/should_not_write_bad2.json --dry-run
INSTRUMENT-DEFECT: covariate table missing required column(s) for registered covariate(s): C4 -> C4_z_gw
exit code: 1
# /tmp/should_not_write_bad2.json NOT written
```

Real mode (no `--dry-run`) raises the identical `InstrumentDefectError`, exits 1, and
writes `{"disposition": {"value": "INSTRUMENT-DEFECT", ...}}` to `--out` — never a
silent skip that would let a missing registered covariate through to
`disposition_for()` undetected (the specific failure mode
`check_covariate_schema()`'s docstring names). The pre-flight (`load_table` →
`check_covariate_schema`) runs before any covariate value is touched, and `main()`
re-checks `missing_covariates` again after the schema mapping as a belt-and-suspenders
regression guard (lines 1144–1149).

**Result: PASS**, independently reproduced (not merely re-reading
`BUILD_RECORD_B3.md`'s FIX 4 exercise, though that exercise's own printed lines —
`FIX 4 / missing covariate column: ... C4 -> C4_z_gw` — match this review's output
byte-for-byte).

## 3. §8 launch block, `--dry-run`, on the real inputs — both venues

```
iiib:      sha256 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0 (recomputed, matches BUILD_RECORD_B1.md)
           venue: iiib; table 1588 rows, sha256 OK; influence 1588 rows
           join: 1588/1588 rows joined; unmatched table-only=0, unmatched influence-only=0; join_complete=True
           family iiib_2d: k=82   family iiib_1d: k=94
           dry-run OK; exit 0

joint_r1:  sha256 fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a (recomputed, matches BUILD_RECORD_B1.md)
           venue: jr1; table 1588 rows, sha256 OK; influence 1588 rows
           join: 1588/1588 rows joined; unmatched table-only=0, unmatched influence-only=0; join_complete=True
           family jr1_2d: k=72   family jr1_1d: k=46
           dry-run OK; exit 0
```

Both k values match the registered banked k (§2) exactly; no `--out` file was written by
either invocation (confirmed by `ls` failing afterward). No registered aggregate was
computed by either run — `--dry-run` returns before `build_report()` is ever called.

**Result: PASS.** Exit 0, sha256 match, 1588/1588 joins, on both venues, re-run by this
reviewer independently of `BUILD_RECORD_B3.md`'s own "Exercise 4" record of the same
commands (which reports identical counts).

## 4. Statistics code byte-identical to the rev3-reviewed version, except the schema layer

`offset_subset_reads.py` is untracked (no git history to diff against directly — it was
never committed). Two independent lines of evidence stand in for a direct diff:

**(a) Structural check.** Reading the full 1185-line file, the only additions relative
to what `BUILD_RECORD_B3.md`'s FIX 2/FIX 3 sections describe are: the "PIN CORRECTIONS"
block (`COVARIATE_COLUMN_MAP`, `FAMILY_D_E_SOURCE_COL`, `VENUE_FAMILIES`,
`REQUIRED_INFLUENCE_COLUMNS`, lines 96–167), `detect_venue()`, `check_covariate_schema()`,
`check_influence_base_schema()`, the rewritten `load_table()`/`load_influence()` (schema
adapter + per-family rank/in_S derivation), and the `main()` pre-flight wiring
(`InstrumentDefectError` handling, `missing_covariates` regression guard). All of these
are schema/loading-layer code. Every statistics function this review inspected —
`_continuous_auc`, `_binary_or`, `separation_for_covariate`, `holm_correct`,
`t0_moments`, `materiality_for_covariate` (incl. Findings A/C/D's binary-direction,
NaN-exclusion, and null-rail-fraction logic), `disposition_for` (incl. the 2D-vs-1D
disagreement re-run), `class_label_line`, `spearman_secondaries`,
`class_composition_counts`, `truth_disagreement_tables` — reads exactly as
`BUILD_RECORD_B3.md`'s FIX 2/FIX 3 sections describe them, with no logic delta.

**(b) Live re-run.** Re-executed `SYNTH_make_synth.py` in full (writes/overwrites its
own `SYNTH_*` fixtures and invokes the real scorer on them — permitted: synthetic data,
not the registered population) and diffed this review's console output against every
`^FIX [234] /` line quoted verbatim in `BUILD_RECORD_B3.md`:

```
$ diff <(grep '^FIX [234] /' BUILD_RECORD_B3.md) <(grep '^FIX [234] /' <this run's output>)
8c8
< ...truth_disagreement(C1 vs C2/C3)={'C2': {...}, 'C3': {...}}
---
> ...truth_disagreement(C1 vs C2/C3)={'C2': {'C1_true_and_cov_true': 1, ...}, 'C3': {...}}
```

The single diff line is `BUILD_RECORD_B3.md`'s own markdown-prose elision (`{...}`) of a
dict it already prints in full earlier in the same line — the actual values match
character-for-character (`class_composition_S` and `truth_disagreement` share the same
`{'C1': {'n_true': 2, ...}}`-style content, and this review's live run reproduces every
number). Every other FIX 2/FIX 3 assertion line (Findings A–D, the 2D/1D-disagreement
trigger, the g-population join gate, the WEAK-vs-Holm keying, the Spearman/class-
composition secondaries) reproduced **byte-identically**, including all FIX 4 lines
(schema mapping, the two missing-column defects, the venue mismatch, and both real-input
`--dry-run`s). `ruff check` and `mypy` were re-run by this reviewer on the file
independently and both report clean, matching `BUILD_RECORD_B3.md`'s own claim.

**Result: PASS.** No behavioral delta found in the statistics code; every change this
review can attribute is confined to the schema/loading layer.

---

## 5. One pre-existing, disclosed, still-open item (not part of the four assigned checks)

Not a rev4 finding — carried forward because it is addressed explicitly to "the launch
reviewer" in `BUILD_RECORD_B3.md` §8 ("Open item for the launch reviewer") and restated
in the reader's own module docstring (lines 31–41), and this review independently
confirmed it is still true of the files on disk today:

**Materiality (§4.2) cannot be computed in real mode with the current Phase B output.**
The reader's documented data contract requires `influence_vectors.csv` to carry
self-describing `logL_h<value>` columns (primary family only) so it can reconstruct the
full/stratum-removed log-posterior from the two registered input files alone. Confirmed
by `head -1` on the real files (permitted — header only): `influence_iiib.csv` and
`influence_joint_r1.csv` carry only `event_idx, influence_2D, influence_1D, rank` — no
`logL_h*` columns exist. This review's `--dry-run` output above shows `logL columns
present: False (h_grid n=0)` for both venues, confirming the gap without computing any
statistic. Consequence, read directly from `build_report()`: if a real run finds **zero**
separating covariates in the primary family, the arm still resolves cleanly to
DIFFUSE-IN-COVARIATES (materiality is moot); but if **any** covariate separates, the
disposition is forced to `INSTRUMENT / NO-READ` (`build_report()` lines 961–965,
973–977) rather than ever reaching `SUBSET-IDENTIFIED` — the arm cannot currently
certify its own headline claim. `REGISTRATION_DRAFT.md`'s PIN CORRECTIONS section
(2026-09-04) addresses only the covariate/influence *column-name* schema mismatch (its
own item 1) and does not mention `logL_h*` at all, so this gap is **not** resolved by
that correction. Routed here, as `BUILD_RECORD_B3.md` requested, for the §9 ratification
list or a fresh amendment before real-mode launch — Phase B (`build_influence_vector.py`)
would need to additionally emit `logL_h<value>` columns for the primary family, or the
launch block would need a third input path to `event_likelihoods.csv` for Phase C.

This does not change the GREEN verdict on the four assigned checks (column mapping,
missing-column pre-flight, §8 dry-run, statistics-code identity) — it is a distinct,
already-disclosed launch-readiness gap that predates this review and was never claimed
fixed by PIN CORRECTIONS or FIX 4.

## Files/paths of record for this review

- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/REGISTRATION_DRAFT.md`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/BUILD_RECORD_B1.md`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/BUILD_RECORD_B2.md`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/BUILD_RECORD_B3.md`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_reads.py`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/build_covariate_table.py`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/build_influence_vector.py`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_iiib.csv` (md5-of-record `90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0` sha256)
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_joint_r1.csv` (sha256 `fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a`)
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_iiib.csv`, `influence_joint_r1.csv`
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/SYNTH_make_synth.py` (re-run live for this review)
