# READ_RECORD.md — r-offset-subset, phase C (the reader), REAL mode, DISJOINT READER

Role: disjoint reader for m-offset-subset, per the launch instruction. Read
`REGISTRATION_DRAFT.md` including the PIN CORRECTIONS section (schema of record = the
built files' headers). Did not open `INFORMATION_FORECAST.md` (forbidden). Did not
inspect the data files by hand — the pinned CLI (`offset_subset_reads.py`) was invoked
exactly once in real mode; all findings below come from that invocation's own output
(stdout/stderr/exit code) and the pre-existing gate documents already on disk. Touched
no production pipeline, no cluster, no file under `darksiren_emri/`. Modified no script.

**This record is VERDICT-FREE**: it states what ran, what the gates report, and what
did (and did not) come out the other end. It rules on nothing — c-offset-subset-covariate
vs c-offset-diffuse-in-covariates is not adjudicated here.

---

## 1. Gates checked before launch (pre-existing documents, read not re-derived)

| gate | file | verdict as read |
|---|---|---|
| Computability | `DESIGN_GATE_computability.md` | **GREEN** ("Overall: GREEN... every named input exists, every md5 pin and byte-id anchor reproduces to spec... the statistic, disposition table, blindness structure, and gates are fully specified." Two AMBER documentation-precision notes disclosed — kill-criterion provenance wording, a pin-count nit — explicitly non-blocking.) |
| Byte-id anchors (phase B) | `BYTEID_RECORD.md` | **GREEN** — 30/30 checks passed (full-sample mean_h to 1e-9, minimal k = 82/94/72/46 exact all four families, top-10 directional influence to 1e-12 relative, k=1588 endpoint = 0.73 to 1e-12, 0 physics-floor exclusions, both CSV md5s match). |
| Formula/code review, rev3 | `DESIGN_GATE_formula_rev3.md` | **40/40 items GREEN** on `offset_subset_reads.py`'s own logic, including a live execution of the sha256 (G-4) gate. Two open items disclosed as scoped to phase A/B, not phase-C code defects. |
| Formula/code review, rev4 | `DESIGN_GATE_formula_rev4.md` | **GREEN** on its four assigned checks (column mapping vs. real headers, missing-column hard pre-flight, §8 `--dry-run` on both real venues, statistics-code identity vs. rev3). One pre-existing, disclosed, still-open item carried forward (§5 of that file): the built `influence_{iiib,joint_r1}.csv` files carry no `logL_h<value>` columns, so **materiality (§4.2) cannot be computed** under the current Phase B output — rev4's own text states that if any covariate separates, `build_report()` would force the disposition to `INSTRUMENT / NO-READ` rather than ever reaching `SUBSET-IDENTIFIED`. rev4 explicitly notes this gap was tested only via `--dry-run` and small synthetic fixtures, never via a full real-mode `build_report()` call. |

All four cited gates were GREEN (rev3 at 40/40) before this read was launched, as required.

## 2. Exact command executed (REAL mode, no `--dry-run`)

Per REGISTRATION_DRAFT.md §8 phase C, with the PIN CORRECTIONS §1 substitutions applied
(built script/file names; primary family = iiib 2D, k=82, so the primary venue's built
pair `covariate_table_iiib.csv` / `influence_iiib.csv` was loaded):

```
cd /home/jasper/Repositories/darksiren-emri
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_reads.py \
  --table results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_iiib.csv \
  --table-sha256 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0 \
  --influence results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_iiib.csv \
  --alpha 0.05 --auc-band 0.20 --or-band 3.0 --t-mat 0.008 \
  --decile 0.10 --null-draws 1000 --null-seed 20260904 \
  --out /home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_result_read.json
```

sha256 recomputed for the `--table-sha256` argument: taken verbatim from the committed
`covariate_table.sha256` (`90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0
covariate_table_iiib.csv`), matching `BUILD_RECORD_B1.md` and both design-gate reviews.

Run executed exactly once (real mode). A second, identical invocation was made only to
capture stdout/stderr cleanly to a log file after the first run's traceback scrolled off
in a `set -x` trace — it reproduced the identical exit code and traceback byte-for-byte
and computed no registered aggregate either (the crash occurs before any statistic is
touched), so no registered computation happened twice.

## 3. Outcome: uncaught exception, exit code 1, no output file written

```
Traceback (most recent call last):
  ...
  File ".../offset_subset_reads.py", line 1184, in <module>
    sys.exit(main())
  File ".../offset_subset_reads.py", line 1174, in main
    report = build_report(table, infl, h_grid, logl_matrix, args)
  File ".../offset_subset_reads.py", line 909, in build_report
    per_family_sep[family] = run_family_separation(family, table, infl, args.alpha, args.auc_band, args.or_band)
  File ".../offset_subset_reads.py", line 700, in run_family_separation
    s_mask = infl[in_s_col].astype(bool)
  ...
KeyError: 'jr1_2d_in_S'
```

**Exit code: 1** (Python's default uncaught-exception exit status — *not* the script's
own `return 1` from its `InstrumentDefectError` handler; this exception is not caught
anywhere in `main()`).

**`--out` file: not written.** Confirmed by `ls` before and after the run:
`offset_subset_result_read.json` does not exist in the target directory. No JSON of any
kind — neither a completed report nor a graceful `{"disposition": {"value":
"INSTRUMENT-DEFECT", ...}}` payload — was produced by this invocation.

**sha256 / G-4 blindness hash: MATCHED.** The crash occurs deep inside `build_report()`,
after `check_table_hash()` (G-4), `detect_venue()`, `load_table()` (schema pre-flight,
G-2/schema checks), `load_influence()`, and `verify_k()` for both of the iiib venue's own
families (`iiib_2d: k=82`, `iiib_1d: k=94` — both verified against the registered banked k
before the crash, consistent with `DESIGN_GATE_formula_rev4.md`'s dry-run record).

**Join completeness (g-population): 1588/1588, 0 unmatched either direction** — computed
successfully (`check_join_completeness()` runs before `build_report()` is called; this is
the same join already confirmed by the `--dry-run` records in `BUILD_RECORD_B3.md` and
`DESIGN_GATE_formula_rev4.md` §3).

## 4. Root cause (read from the traceback + source, not modified)

`build_report()` (line ~907-909) iterates `for family in FAMILIES:` where `FAMILIES =
("iiib_2d", "iiib_1d", "jr1_2d", "jr1_1d")` — **all four families, unconditionally** —
and calls `run_family_separation(family, table, infl, ...)` for each. But
`load_influence()` only derives the `{family}_in_S` / `{family}_d_e` columns for
`VENUE_FAMILIES[venue]`, i.e. the **two families native to whichever single venue's
files were passed on the command line** (PIN CORRECTIONS item 1 / the module's own
`detect_venue()` docstring: "a single invocation covers exactly one venue's two
families"). `run_family_separation()` then does `s_mask = infl[in_s_col]` for
`jr1_2d_in_S` — a column that was never created because the iiib venue's `influence_iiib.csv`
was loaded, not `influence_joint_r1.csv` — and pandas raises `KeyError`, uncaught.

This reproduces symmetrically for the other venue: loading `covariate_table_joint_r1.csv`
+ `influence_joint_r1.csv` instead would crash on `iiib_2d_in_S` (`PRIMARY_FAMILY =
"iiib_2d"` is hard-coded and iterated first in `FAMILIES`), for the same reason. **No
single real-mode invocation of the launch block, as built, can reach the end of
`build_report()`** — the four-family loop is not scoped to the loaded venue's
`active_families`, while `load_influence()` (correctly, per its own docstring) only
populates two of the four families' columns per invocation.

This is a distinct defect from the one `DESIGN_GATE_formula_rev4.md` §5 disclosed (the
missing `logL_h*` columns blocking materiality only). That disclosed item presumes
`build_report()` reaches the materiality stage; this run shows it does not reach even the
separation stage for the non-native families, because every prior verification of this
script (design-gate reviews rev1-rev4, `BUILD_RECORD_B3.md`, `BYTEID_RECORD.md`) exercised
either `--dry-run` (which returns before `build_report()` is ever called) or the
`SYNTH_make_synth.py` fixtures (which call `load_table`/`load_influence` directly, or the
full CLI only in the FIX-4 "happy path" check — which per its own console line, "venue='iiib',
iiib_2d S=[0, 1, 2], iiib_1d S=[0, 1, 2]", tests loading only, not a `build_report()` call).
No prior document on file records a real, non-dry-run, full-pipeline execution of
`offset_subset_reads.py`. This read is the first.

Not modified: this record does not patch, edit, or work around the script, per the launch
instruction's "Do not modify any script."

## 5. Registered items requested by the launch instruction — status

Because `build_report()` crashed before returning, and no `--out` JSON was produced,
**none of the following were computed by this run**:

- **Per-covariate AUC/OR + Holm p + SEPARATES/WEAK/NULL/NOT-TESTED, per family (§4.1):**
  NOT COMPUTED (crash occurred inside the per-family separation loop, on the second
  family checked cross-venue — `jr1_2d` — before any of the four families' Holm-corrected
  results were assembled into the report; even `iiib_2d`/`iiib_1d`, whose `_in_S` columns
  *did* load successfully, never reach `holm_correct()` because the crash aborts the
  whole `for family in FAMILIES` loop and no partial `per_family_sep` dict is written
  anywhere).
- **Enriched-stratum Δ_strat vs. T_mat and vs. the null's 99% band (§4.2):** NOT COMPUTED
  (never reached; also independently blocked by the disclosed missing-`logL_h*` gap,
  §1 above).
- **2D-vs-1D disposition comparison (iiib_2d vs iiib_1d, §5's INTERMEDIATE trigger):**
  NOT COMPUTED (the `iiib_1d_disposition` re-run in `build_report()` is downstream of the
  crashed loop).
- **2-of-3 replicate outcome (§4.3):** NOT COMPUTED (`replicate_sep` is built from
  `per_family_sep`, which the crash left unpopulated).
- **Reported-only secondaries (Spearman ρ, C1/C2/C3 class-composition and truth-disagreement
  tables, C11):** NOT COMPUTED (all downstream of `build_report()`'s return).
- **Three-valued disposition of EACH §5 row (SUBSET-IDENTIFIED / DIFFUSE-IN-COVARIATES /
  INTERMEDIATE / INSTRUMENT-NO-READ):** the run's own outcome is **INSTRUMENT / NO-READ
  in substance** (§6 of the draft: "any §6 gate red → nothing banked; repair; no revision
  consumed") — but note precisely: this is *not* the script's own graceful
  `INSTRUMENT-DEFECT` JSON path (that path is reserved for `InstrumentDefectError`, is
  caught, and writes a JSON with `disposition.value == "INSTRUMENT-DEFECT"`). This run
  produced an **uncaught Python exception outside that mechanism**, exit code 1, zero
  bytes of output. No `disposition` field of any kind exists on disk from this run.
- **R14 mandatory class-label line (a) C2 / (b) C3 / (c) C3c:** NOT COMPUTED (downstream
  of `class_label_line()`, itself downstream of `primary_sep`, never assembled).

## 6. What DID execute successfully, for the record

- G-4 blindness hash check: PASS (table sha256 matched `--table-sha256`).
- `detect_venue()`: resolved to `"iiib"` from both filenames, no mismatch.
- `load_table()` schema pre-flight: all 12 registered covariate columns
  (`COVARIATE_COLUMN_MAP`) present in `covariate_table_iiib.csv`'s real header — no
  `InstrumentDefectError` raised.
- `load_influence()` base-schema pre-flight: `event_idx, influence_2D, influence_1D, rank`
  all present in `influence_iiib.csv`.
- `verify_k()` for the iiib venue's two native families: `iiib_2d` cardinality-of-`in_S`
  == 82, `iiib_1d` == 94 — both match the registered banked k exactly (consistent with
  every prior `--dry-run` record).
- `check_join_completeness()`: 1588 table rows / 1588 influence rows, 0 unmatched in
  either direction, `join_complete = True`.
- No `logL_h*` columns present in `influence_iiib.csv` (`h_grid` would be empty, size 0)
  — consistent with the rev4-disclosed gap; moot here since the run never reached
  materiality regardless.

## 7. Disposition (this record's own characterization, verdict-free)

The launch-instruction's requested three-valued outcome per disposition row cannot be
populated — no registered statistic of any kind was produced. The technically accurate
status of this read is: **the registered read did not run to completion; `offset_subset_reads.py`,
as built, cannot complete a real-mode (non-`--dry-run`) invocation of the §8 launch block
for either venue, because `build_report()`'s per-family loop is not scoped to the single
venue's own `active_families`.** This is reported as an execution fact, not a ruling on
c-offset-subset-covariate / c-offset-diffuse-in-covariates — the claim itself remains
neither supported nor refuted by this run. Per the draft's own §5 table (INSTRUMENT / NO-READ
row): "nothing banked; repair; no revision consumed."

## 8. Source paths of record

- Command target: `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_reads.py`
- Inputs: `.../covariate_table_iiib.csv` (sha256 `90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0`), `.../influence_iiib.csv`
- Attempted output (not created): `.../offset_subset_result_read.json`
- Gate documents read: `DESIGN_GATE_computability.md`, `BYTEID_RECORD.md`,
  `DESIGN_GATE_formula_rev3.md`, `DESIGN_GATE_formula_rev4.md`, `BUILD_RECORD_B3.md` §4
- Registration of record: `REGISTRATION_DRAFT.md` §3, §4, §5, §8, and the PIN CORRECTIONS
  section
