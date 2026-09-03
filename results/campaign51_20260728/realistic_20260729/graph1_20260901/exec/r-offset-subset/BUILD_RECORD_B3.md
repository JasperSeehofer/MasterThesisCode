# BUILD_RECORD_B3 — offset_subset_reads.py (phase C, "the reader")

Builder: B3 (task label "SCORER"; the script implemented is phase C — "the reader" — of
`REGISTRATION_DRAFT.md` §3, the one script named in the launch order). Sonnet/medium,
mechanical implementation stage. **No real-mode run performed.** Every number in this
record is from a synthetic ≤10-row table constructed by this builder; nothing here is a
registered aggregate over the registered population (§10 of the draft governs that
prohibition and does not apply to synthetic exercise data).

## 1. What was built

`offset_subset_reads.py` — CLI matches the r-offset-subset launch block
(`REGISTRATION_DRAFT.md` §8) exactly: `--table --table-sha256 --influence --alpha
--auc-band --or-band --t-mat --decile --null-draws --null-seed --out [--dry-run]`, plus
four optional `--k-<family>` sanity-check flags (defaulted to the registered banked k —
82/94/72/46, §2) that assert `in_S` cardinality without ever re-deriving S.

Implements, per the task list:
1. **sha256 refusal (G-4).** `check_table_hash()` recomputes the table's sha256 and calls
   `SystemExit` (rc=1) on mismatch, before any covariate is touched.
2. **S = top-k influence events.** Read directly from the `in_S` columns phase B is
   contracted to write (banked constant, never re-derived here); the `--k-*` flags assert
   cardinality (`verify_k()`), raising `INSTRUMENT-DEFECT` on mismatch.
3. **Per-covariate separation.** `separation_for_covariate()` — Mann-Whitney AUC for
   continuous (`C3c, C4-C7, C10, C11`), Haldane-corrected Fisher-exact OR for binary
   (`C1, C2, C3, C8, C10b`); `holm_correct()` does Holm step-down over the tested members
   of the m=11 (or m=10 when C10b is NOT-TESTED, n<10 gate) family; C8 is restricted to
   the `in_catalog` (C1) stratum per §4.1; C11 is computed and reported but excluded from
   the Holm family and the disposition logic (`REPORTED-ONLY`).
4. **Materiality.** `materiality_for_covariate()` implements the leave-out
   re-marginalisation (§4.2) under the frozen T0 convention (gradient-trapezoid weights,
   replicated from `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py:_moments`)
   for the PRIMARY family only (iiib 2D), plus the 1000-draw random-same-size null (seed
   20260904) and its central 99% interval. **Documented data-contract addition** (module
   docstring, not present verbatim in the draft's column list for `influence_vectors.csv`):
   exact stratum-level re-marginalisation needs the per-event log-likelihood at every
   H_GRID_41 node, not just the scalar single-event `d_e` — the draft's launch block does
   not pass phase C a path to the raw `event_likelihoods.csv`, so this script requires
   phase B to additionally carry self-describing `logL_h<value>` columns (primary family
   only) in `influence_vectors.csv`. Flagged for the author/phase-B owner at launch review
   (§9 open questions); if absent, materiality returns `None` per covariate and the
   disposition carries `INSTRUMENT / NO-READ` with an explanatory `instrument_note`.
5. **2-of-3 replicate rule.** `disposition_for()` requires same-direction SEPARATES in
   ≥2 of {iiib_1d, jr1_2d, jr1_1d} (`replicate_direction()`) before a materially-separating
   primary covariate reaches `SUBSET-IDENTIFIED`; otherwise `INTERMEDIATE`.
6. **Disposition rows + R14 line.** `disposition_for()` returns exactly the four §5 values
   (`SUBSET-IDENTIFIED`, `DIFFUSE-IN-COVARIATES`, `INTERMEDIATE`, `INSTRUMENT / NO-READ`);
   `class_label_line()` always emits the mandatory R14 line for (a) C2, (b) C3, (c) C3c
   with the four canonical readings from §5, regardless of the overall disposition.
7. **JSON output.** `build_report()` writes every intermediate (per-family separation
   table, materiality table, R14 line, disposition) to `--out`.
8. **`--dry-run`.** Loads both inputs, checks the table hash and `in_S` cardinalities,
   prints per-family k and row counts, exits 0 without writing `--out` or touching
   statistics.

Typed (`from __future__ import annotations`, `npt.NDArray[np.float64]`, dataclasses),
`ruff check` and `mypy` both clean (verified below).

## 2. Synthetic input (8 rows, constructed by this builder)

Generator: a throwaway script (not committed) built
`SYNTH_covariate_table_blind.csv` and `SYNTH_influence_vectors.csv` in this directory —
both are committed here for hand-checking. `event_idx` 0-2 are the constructed
high-`z_gw`/high-`log10_M`/high-`log10_snr` "influence" cluster (S under `iiib_2d`, k=3);
3-7 are the bulk. `iiib_1d`/`jr1_2d` reuse the same k=3 membership (same direction);
`jr1_1d` uses k=2 ({0,1}) to exercise a differing replicate cardinality. `C10b` is all
`False` (n=0 < 10) to exercise the NOT-TESTED gate. `C8` is defined only for the three
`C1=True` rows (0, 3, 6). `C6` has two `NaN` rows to exercise `n_nan` reporting.

`covariate_table_blind.csv` sha256 (recomputed by the reader and matched against
`--table-sha256`, as G-4 requires):

```
cf89eb37415b3312bbbf84c385a3d1711c2ea0e7c9d79a7c402ab1f4b224e794
```

`influence_vectors.csv` carries, for the primary family only, self-describing
`logL_h0.600000 / logL_h0.665000 / logL_h0.730000` columns (a 3-node toy H-grid, not the
production 41-node grid — sufficient to exercise the T0-moments/marginalisation code
path). Both CSVs are committed alongside this record (`SYNTH_covariate_table_blind.csv`,
`SYNTH_influence_vectors.csv`).

## 3. Exercise 1 — full CLI, dry-run then real, matching launch-block flags

```
uv run python offset_subset_reads.py \
  --table SYNTH_covariate_table_blind.csv --table-sha256 cf89eb37415b3312bbbf84c385a3d1711c2ea0e7c9d79a7c402ab1f4b224e794 \
  --influence SYNTH_influence_vectors.csv \
  --k-iiib-2d 3 --k-iiib-1d 3 --k-jr1-2d 3 --k-jr1-1d 2 \
  --decile 0.10 --null-draws 200 --null-seed 20260904 \
  --out SYNTH_out.json [--dry-run]
```

**Dry-run output** (rc=0):
```
table: .../SYNTH_covariate_table_blind.csv (8 rows), sha256 OK
influence: .../SYNTH_influence_vectors.csv (8 rows)
logL columns present: True (h_grid n=3)
  family iiib_2d: k=3
  family iiib_1d: k=3
  family jr1_2d: k=3
  family jr1_1d: k=2
dry-run OK
```

**Real run output** (rc=0): `disposition = DIFFUSE-IN-COVARIATES`. Full JSON is
`SYNTH_out.json` in this directory (committed) — hand-checkable summary:

| covariate (primary iiib_2d) | kind | effect | p_raw | p_holm (m=10) | verdict |
|---|---|---|---|---|---|
| C1 | binary OR | 0.84 | 1.0 | 1.0 | NULL |
| C2 | binary OR | 5.0 | 0.464 | 1.0 | NULL |
| C3 | binary OR | 21.0 | 0.143 | 0.667 | NULL |
| C3c | AUC | 0.933 | 0.071 | 0.429 | NULL |
| C4 | AUC | 1.0 | 0.0357 | 0.357 | NULL |
| C5 | AUC | 1.0 | 0.0357 | 0.357 | NULL |
| C6 | AUC | 1.0 (n=2 vs 4) | 0.133 | 0.667 | NULL |
| C7 | AUC | 1.0 | 0.0357 | 0.357 | NULL |
| C8 (in_cat only, n=1 vs 2) | binary OR | 0.067 | 0.333 | 1.0 | NULL |
| C10 | AUC | 1.0 | 0.0357 | 0.357 | NULL |
| C10b | — | n=0 | — | — | NOT-TESTED (n<10 gate) |
| C11 (reported-only) | AUC | 1.0 | 0.0357 | n/a | REPORTED-ONLY |

Hand-check of Holm at m=10: raw p-values sorted ascending are five ties at 0.0357
(C4/C5/C7/C10, all rank-1..4) each multiplied by (10-i); the smallest, `10*0.0357=0.357`,
already exceeds α=0.05, so **no covariate can reach significance at n_S=3, n_B=5 after
Holm — mechanically correct and expected**: this is the smallest sample size at which any
separation could show, and it is well below the power threshold quoted in §5 of the
draft (n_S=82 vs n_B=1506 in the real run). Disposition correctly falls to
`DIFFUSE-IN-COVARIATES` with `named_covariates: []`. The R14 line reports
`"class is not the axis"` since none of C2/C3/C3c separate at this n — the mandatory line
is emitted even though the overall disposition is DIFFUSE, exactly as §5 requires.

`materiality` is `{}` (nothing SEPARATES in this run, so §4.2 is never invoked by the
pipeline) — see Exercise 2 for a direct, hand-checked exercise of the materiality function
itself.

## 4. Exercise 2 — direct exercise of `materiality_for_covariate()` (forced SEPARATES)

Because n=8 cannot pass Holm (Exercise 1), the materiality code path inside
`build_report()` is never entered on this synthetic table — by design, since a real
n_S=3/n_B=5 read should never claim SEPARATES. To hand-check `materiality_for_covariate()`
itself (item 4 of the task), it was called directly (not through the CLI) with C4's
separation result forced to `SEPARATES`:

```python
sep = separation_for_covariate("C4", table, s_index, b_index, 0.20, 3.0)
sep.verdict = "SEPARATES"  # forced, for direct exercise only
mat = materiality_for_covariate("C4", sep, table, event_order, logl, h_grid, s_index,
                                 decile=0.10, t_mat=0.008, null_draws=500, null_seed=20260904)
```

Output:
```
stratum_rule     = top decile (1/8)
n_stratum        = 1
delta_strat      = 0.0009016647735987648
delta_s_oracle   = 0.0010638169278917031
captured_fraction= 0.8475751324860987
null_percentile  = 75.2
null_ci99        = (-0.006637244954602162, 0.0009317806339549373)
material         = False
map_rail_full    = False
map_rail_stratum = False
```

Hand-check: `n_total=8`, `decile=0.10` -> `n_tail = max(1, round(0.8)) = 1`, so the
stratum is the single highest-`C4` event (event 0) — matches `n_stratum=1`. `delta_strat`
is positive (toward truth by construction: events {0,1,2} were built peaked at h=0.60,
the bulk at h=0.73, so removing any of them should raise mean_h) and is smaller than
`delta_s_oracle` (removing only 1 of the 3 offset-carrying events captures 85% of the full
S-oracle shift) — consistent, `captured_fraction<1` as expected for a strict subset.
`delta_strat=0.0009 < t_mat=0.008`, so `material=False` — correctly gates on the
registered threshold even though the null-percentile (75.2%, inside a wide symmetric
99% CI given only 500 draws over 8 events) would not itself have ruled it out. Both MAP
flags `False` (no rail) is consistent with a 3-node toy grid whose maximum is comfortably
interior.

## 5. Exercise 3 — gate failures

- **sha256 mismatch:** `--table-sha256 deadbeef...` (65 hex chars) -> rc=1,
  `"G-4 BLINDNESS-HASH-MISMATCH ... Refusing to run (INSTRUMENT / NO-READ)."` before any
  covariate or influence file is touched.
- **k mismatch:** `--k-iiib-2d 99` against the true `in_S` cardinality (3) -> rc=1,
  `"INSTRUMENT-DEFECT: family iiib_2d in_S cardinality 3 != registered k=99."`

## 6. Quality gate

```
$ uv run ruff check offset_subset_reads.py
All checks passed!
$ uv run mypy offset_subset_reads.py
Success: no issues found in 1 source file
```

## 7. Scope discipline

No production CRB, no `event_likelihoods.csv`, no galaxy catalogue, no cluster, and no
`darksiren_emri/` file was opened or run. The script was never invoked with
`--table`/`--influence` pointing at any path under `graph1_20260901/retrieved/` or
`seed61000/`. `covariate_table_blind.csv` and `influence_vectors.csv` (the real,
registered files phase A/B will produce) do not exist yet and were not created by this
builder. All committed artifacts under this record are prefixed `SYNTH_` to make that
unambiguous to a reviewer.

## 8. Open item for the launch reviewer (§9)

The materiality data-contract addition in §1 item 4 above (per-event `logL_h*` columns on
`influence_vectors.csv`, primary family only) is not explicit in `REGISTRATION_DRAFT.md`
§3's one-line description of phase B's output ("event_idx, d_e, rank, in_S per family").
It is mathematically required for exact stratum-level re-marginalisation and was designed
to require no new CLI flag and no third input file — but the phase-B builder must be told
this contract before that script is written, or G-4-clean joins will still fail
materiality with `INSTRUMENT / NO-READ` (`instrument_note` in the JSON explains why). This
belongs in the §9 ratification list or as a fresh amendment before real-mode launch.

**Post-hoc note (schema divergence observed, not investigated further):** at the time this
record was written, `BUILD_RECORD_B2.md` and `influence_iiib.csv` / `influence_joint_r1.csv`
already existed in this directory (a concurrent builder's phase-B output). A header check
only (`head`, no row beyond that touched, no statistic computed, real production data NOT
opened by this builder) shows B2's actual schema is `event_idx, influence_2D, influence_1D,
rank` in two venue-split files -- no `in_S` column, no per-family split, no `logL_h*`
columns. This differs from both (a) the draft's one-line phase-B description ("event_idx,
d_e, rank, in_S per family", one file) and (b) the data contract this builder's reader
requires for section 4.2 materiality (item 4 in section 1, above). None of this was used to
build, test, or alter `offset_subset_reads.py` -- the schema mismatch is flagged here,
unresolved, for the launch reviewer to reconcile (either B2's schema or this reader's
expectations must change before phase C can run in real mode; `--table`/`--influence` were
never pointed at B2's files by this builder).
