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

---

## FIX 2 — response to `DESIGN_GATE_formula.md` (RED, findings A-D)

Builder: B3, fix round 2, Sonnet/medium (mechanical fix stage, same task label). No
real-mode run performed here either — every number below is from synthetic data
constructed by this builder; `covariate_table_iiib.csv` / `covariate_table_joint_r1.csv`
/ `influence_iiib.csv` / `influence_joint_r1.csv` were not opened beyond their header (no
row beyond the header, no statistic) while diagnosing Finding B against
`BUILD_RECORD_B1.md`'s already-published column table.

All four confirmed/flagged findings in `DESIGN_GATE_formula.md` are fixed in
`offset_subset_reads.py`. Every other code path (Holm step-down, AUC/OR arithmetic, sha256
refusal, launch-block CLI, `--k-*` flags, R14 line) is byte-identical to the version the
gate reviewed — only `materiality_for_covariate()`, `disposition_for()`, `build_report()`,
and the `MaterialityResult` dataclass changed.

### Finding A — binary materiality stratum: OR-direction, not raw majority

`materiality_for_covariate()`'s binary branch now sets `enriched_level = bool(sep.effect
>= 1.0)` (the already-registered odds-ratio direction, symmetric with the continuous
branch's `sep.effect >= 0.5` AUC direction) instead of recomputing `s_bool.mean() >= 0.5`
inside S alone. Both a code comment at the call site and this record document the rule.

### Finding B — the §5 "NOT-TESTED -> INTERMEDIATE" branch

`disposition_for()` now checks, only when `separators` is empty, whether C8 or C10b's
*primary-family* verdict is `NOT-TESTED`; if so it returns `("INTERMEDIATE", [])` instead
of falling through to `DIFFUSE-IN-COVARIATES`. This is the exact `REGISTRATION_DRAFT.md`
§5 disposition-table row: *"C8 or C10b NOT-TESTED and no other covariate separates"* ->
INTERMEDIATE.

**Downstream effect on Exercise 1 (re-run, see below):** the committed 8-row synthetic
table has `C10b` all-`False` (n=0 < 10, NOT-TESTED by design, per §2 of this record) and no
other covariate separates at that n — so the fix flips Exercise 1's own disposition from
`DIFFUSE-IN-COVARIATES` to `INTERMEDIATE`. This is not a regression: it is precisely the
bug `DESIGN_GATE_formula.md` Finding B described (quoting `BUILD_RECORD_B1.md`: C10b is
NOT-TESTED in both real venues too), now correctly reachable in the disposition logic.
`SYNTH_out.json` is re-committed with the corrected disposition.

### Finding C — NaN excluded from the decile-tail stratum (continuous covariates)

`REGISTRATION_DRAFT.md` is silent on decile-tail NaN handling specifically (§6
g-population only mandates *disclosure* of `n_NaN` per covariate for the separation
statistic, not a stratum-construction rule; C6's own row in §2, `docs` line 84, says NaN
is "excluded from **this test**" — i.e. the separation test — without addressing the
materiality decile). Per this task's fallback instruction, the **orchestrator-derived
default** applied and disclosed here is:

> NaN excluded from both S and bulk for that covariate and disclosed as `n_missing`.

Implementation: `col.rank(method="first", na_option="keep")` leaves NaN rows as NaN
(instead of the previous `na_option="bottom"`, which — verified directly, see
`DESIGN_GATE_formula.md` Finding C — ranks NaN *above every real value* under an ascending
rank, so it was sweeping NaN rows into the "top decile" whenever `auc_above_half` was
True). The tail size `n_tail` is now computed against `valid_n = n_total - n_missing`
(not `n_total`), and `stratum_mask` is explicitly `& ~nan_mask`. `MaterialityResult` gained
an `n_missing` field, reported in the output JSON. This is flagged as an
orchestrator-derived default (not a verbatim draft quote) per the fix-round instructions.

### Finding D — g-censoring null-rail gate wired into INSTRUMENT / NO-READ

`MaterialityResult` gained `null_rail_fraction` (the previously-discarded per-null-draw MAP
rail flag, now accumulated across all `null_draws`) and `censoring_gate_red`. §6's
g-censoring gate text ("MAP position for the full sample, every stratum leave-out and
every null draw; any MAP at 0.60/0.86 => that Delta is a BOUND, rail fraction reported")
mandates *disclosure* but does not itself state a numeric red/green cut for the
INSTRUMENT / NO-READ table row ("any §6 gate red") — `CENSORING_NULL_RAIL_RED_FRACTION =
0.5` is an **orchestrator-derived default** (majority of null draws railed => the 0.5/99.5
percentile null-CI used for the outside-null materiality test is itself degenerate and
cannot certify MATERIAL / not-MATERIAL), documented in a module-level comment next to the
constant and here. `build_report()` now collects every covariate whose materiality result
has `censoring_gate_red = True`, folds them into `instrument_note`, and — via the existing
`if instrument_note is not None: report["disposition"]["value"] = "INSTRUMENT / NO-READ"`
line (unchanged) — actually overrides the disposition, closing the "computed but discarded,
never wired to anything" gap Finding D identified. No other §6 gate applies inside this
reader's scope (G-1/G-2/G-3 are phase A/B; G-4 was already a hard `SystemExit` refusal,
confirmed correct by the gate; g-population's `n`/`n_NaN` disclosure was already present via
`SeparationResult.n_nan`; g-precision is a phase A/B concern).

### Extended synthetic check (`SYNTH_make_synth.py`, appended "FIX 2" section)

`SYNTH_make_synth.py` was extended (Exercise 1/2's original 8-row table and its outputs
are untouched — same sha256 `cf89eb374...`) with four new, deliberately-constructed direct
calls into the fixed functions, each with an assertion that fails loudly on regression.
Full numeric record: `SYNTH_fix2_output.json` (committed). Console output from the actual
run:

```
FIX 2 / Finding A: OR=5.0000, s_bool.mean(S)=0.333 (<0.5, minority) -> fixed n_stratum=4 (old buggy rule would give n_stratum=26)
FIX 2 / Finding B: all-NULL + C10b NOT-TESTED -> disposition = 'INTERMEDIATE' (was DIFFUSE-IN-COVARIATES pre-fix)
FIX 2 / Finding C: n_missing=5, valid_n=15, n_stratum=2 (all real values, no NaN row swept in)
FIX 2 / Finding D: null_rail_fraction=0.775, censoring_gate_red=True (forced rail red)
FIX 2: all assertions passed (A, B, C, D)
```

- **A** (n=30, binary covariate `C_A`, S=events 0-5 k=6): built so True is a MINORITY
  within S (`s_bool.mean()=2/6=0.333`) but enriched in S vs bulk (`OR=5.0`, 2/6 vs
  2/24). The old rule would freeze the stratum at the 26 False rows; the fixed rule
  correctly freezes it at the 4 True rows (`n_stratum=4`) — hand-check: `a=2 (True,S),
  b=4 (False,S), c=2 (True,bulk), d=22 (False,bulk)`, `OR_Haldane = (2.5*22.5)/(4.5*2.5) =
  56.25/11.25 = 5.0` ✓.
- **B**: `disposition_for()` called directly with every `HOLM_FAMILY` covariate at verdict
  `NULL` except `C10b` at `NOT-TESTED` (mirroring `BUILD_RECORD_B1.md`'s real-table
  finding) -> `("INTERMEDIATE", [])`, not `DIFFUSE-IN-COVARIATES`.
- **C** (n=20, continuous `C_C`, 5 NaN + values 0..14): `n_missing=5` ✓ matches the 5
  constructed NaN; `valid_n=15`, `n_tail=max(1,round(15*0.10))=2` ✓; `n_stratum=2` ✓,
  drawn from the top-2 REAL values (indices 18, 19 -> covariate values 13, 14), no NaN row
  in the stratum.
- **D** (n=4, continuous `C_D`, one event (idx 3) carries all the log-likelihood signal,
  the other three flat): `decile=0.75` forces `n_tail=3` so every null draw (size 3 of 4)
  excludes exactly one event; hand-check of the 4 equally-likely size-3 subsets: the 3
  subsets that exclude a flat event still contain event 3, so removing the drawn 3 leaves
  only a flat/uninformative logpost that MAP-rails to the grid boundary (`0.60`); the 1
  subset that excludes event 3 leaves the informative logpost intact (no rail). Expected
  rail fraction = 3/4 = 0.75; observed over 400 draws = `0.775` (sampling noise, within
  the `[0.65, 0.85]` tolerance asserted in the script) -> `censoring_gate_red=True`
  (`>= CENSORING_NULL_RAIL_RED_FRACTION=0.5`) ✓; `map_rail_full=False` confirms only the
  *null draws* rail, not the full sample (isolating the mechanism this finding targets).
  A one-line replication of `build_report()`'s wiring (`[cov for cov, m in {...}.items()
  if m.censoring_gate_red]`) confirms this actually reaches `instrument_note`, not just the
  `MaterialityResult` field.

### Re-run of Exercise 1 (full CLI, post-fix)

Re-ran the exact Exercise 1 launch-block invocation (§3 of this record) against the
unchanged `SYNTH_covariate_table_blind.csv` / `SYNTH_influence_vectors.csv`:

```
wrote SYNTH_out.json: disposition = INTERMEDIATE
```

`{'value': 'INTERMEDIATE', 'named_covariates': [], 'instrument_note': None}` — the only
change from the pre-fix `SYNTH_out.json` (`DIFFUSE-IN-COVARIATES`) is the disposition
value itself (Finding B, discussed above); every per-covariate AUC/OR/p_holm number in the
table is unchanged (confirmed by inspection of the re-written `SYNTH_out.json`), since
Findings A/C/D's code paths are not reached by this 8-row/all-NULL exercise (materiality is
never invoked when nothing SEPARATES).

### Quality gate (post-fix)

```
$ uv run ruff check offset_subset_reads.py SYNTH_make_synth.py
All checks passed!
$ uv run mypy offset_subset_reads.py
Success: no issues found in 1 source file
```

### Scope discipline (unchanged)

No production CRB, no `event_likelihoods.csv`, no galaxy catalogue, no cluster, no
`darksiren_emri/` file opened or run; `--table`/`--influence` never pointed at anything
under `graph1_20260901/retrieved/` or `seed61000/`; `covariate_table_iiib.csv` /
`covariate_table_joint_r1.csv` / `influence_iiib.csv` / `influence_joint_r1.csv` (the
real, concurrently-produced phase A/B output already present in this directory) were
opened only for their header line (column names) while cross-checking Finding B against
`BUILD_RECORD_B1.md` — no row beyond the header, no aggregate, no registered statistic.

---

## FIX 3 — response to `DESIGN_GATE_formula_rev2.md` (RED, findings B/C/D + E)

Builder: B3, fix round 3, Sonnet/medium (mechanical fix stage, same task label). No
real-mode run performed — every number below is from synthetic data constructed by this
builder or from the committed `SYNTH_*` fixtures; `covariate_table_iiib.csv` /
`covariate_table_joint_r1.csv` / `influence_iiib.csv` / `influence_joint_r1.csv` were not
opened at all in this round (not even the header — no need arose).

All findings in `DESIGN_GATE_formula_rev2.md` (rev2's §B/§C/§D defects, §E gap) are
addressed. Findings A–D from round-1 (`DESIGN_GATE_formula.md`) were already fixed in
FIX 2 above and are untouched here except where rev2 §B required generalising
`disposition_for()`'s signature (see item 1 below — Finding A/B/C/D's own code paths and
assertions are byte-identical).

### Item 1 (rev2 §B) — the §5 "primary 2D and 1D iiib families disagree in disposition" trigger

`disposition_for()` (`offset_subset_reads.py:587`) gained a `replicate_families` parameter
(default `REPLICATE_FAMILIES`, so the existing call site and every FIX 2 direct-call test
are unaffected) so the same function can be re-run with `iiib_1d` substituted as primary
against the complementary replicate set `(iiib_2d, jr1_2d, jr1_1d)`, instead of the
disposition logic silently comparing `iiib_1d` against itself via the hardcoded global.

`build_report()` (`offset_subset_reads.py:773-799`) now: (a) computes the primary (iiib_2d)
disposition as before and saves it as `primary_disposition_raw`; (b) computes iiib_1d's own
whole-family disposition via `disposition_for(per_family_sep["iiib_1d"], {}, ...,
replicate_families=(iiib_2d, jr1_2d, jr1_1d))`; (c) if the two disagree, overrides the
final `disposition` to `INTERMEDIATE` (line 799) — this happens *before* the §6
instrument-note override, so a red gate still takes final precedence over this trigger, per
the disposition table's own priority (INSTRUMENT / NO-READ is "any §6 gate red", not
conditioned on the other three rows).

**Disclosed limitation (both in a code comment at the call site and here, not silently
assumed):** `iiib_1d_materiality` is always `{}` — the data contract (module docstring)
only requires `logL_h*` columns for the PRIMARY family (iiib_2d), so no per-covariate
materiality can be computed for iiib_1d as an alternate primary with the current
`influence_vectors.csv` schema. This means iiib_1d's own disposition can only ever read
`DIFFUSE-IN-COVARIATES` or `INTERMEDIATE`, never `SUBSET-IDENTIFIED` — a well-defined
"whole disposition" for the §5 comparison, but not a symmetric re-run of §4.2. This is the
same open data-contract question `DESIGN_GATE_formula_rev2.md`'s own §B note flagged (routed
to the launch reviewer, not resolved here — resolving it would need a fresh amendment to
phase B's column contract, out of scope for a mechanical fix round).

Both raw dispositions and the agreement flag are recorded in the output JSON under
`iiib_1d_disposition_check` (`offset_subset_reads.py:891-901`).

Test: `SYNTH_make_synth.py` "FIX 3" item 1 (`finding_2d_1d_disagree`) constructs a
primary (iiib_2d) result that alone would bank `SUBSET-IDENTIFIED` (one SEPARATES +
MATERIAL + 2-of-3 replicate-consistent covariate) against an iiib_1d result with nothing
separating (`DIFFUSE-IN-COVARIATES`), and asserts the final wired disposition is
`INTERMEDIATE`, not the raw `SUBSET-IDENTIFIED` iiib_2d alone would have produced.

### Item 2 (rev2 §C) — g-population join completeness

New `check_join_completeness()` (`offset_subset_reads.py:234-256`) computes
`table.index`/`infl.index` set difference in both directions (not a `pandas.Index
.intersection()`, which silently drops mismatches with no count) and returns row counts,
unmatched-count and the actual unmatched `event_idx` lists on each side, plus a
`join_complete` boolean. Called once at the top of `build_report()` (line 746) — one check
suffices because both venues' `in_S`/`d_e` columns for all four (c,v) live in the single
registered `influence_vectors.csv` joined against the single `covariate_table_blind.csv`
(§8 launch block passes exactly one `--table`/`--influence` pair); "n = 1588 per venue" in
the task instruction is what the real production run's `n_table_rows`/`n_influence_rows`
are expected to read, disclosed generically here rather than hardcoded (hardcoding 1588
would break every synthetic exercise, which intentionally use n ∈ {4, 8, 20, 30}).

Disclosed in the output JSON under `gates.g_population` (line 888-890) unconditionally
(not only on failure) and, when `join_complete` is `False`, folded into `instrument_note`
(lines 811-818) exactly like the existing g-censoring wiring — which forces the final
disposition to `INSTRUMENT / NO-READ` via the pre-existing override at the end of
`build_report()` (line 914-916, see item 5 below).

Test: `SYNTH_make_synth.py` "FIX 3" item 2 (`finding_g_population_join`) builds a 5-row
table (`event_idx` 0–4) against a 5-row influence frame (`event_idx` 0,1,2,3,5) — one
table-only row (4) and one influence-only row (5) — and asserts
`n_unmatched_table_only=1`, `n_unmatched_influence_only=1`, `join_complete=False`, and
that the replicated build_report wiring produces an `instrument_note` containing
`"g-population RED"`.

### Item 3 (rev2 §D) — WEAK keyed to Holm-adjusted significance, not raw p

`holm_correct()` (`offset_subset_reads.py:376-385`): the `elif` branch that assigns `WEAK`
now reads `elif r.holm_significant and not r.band_pass:` (was `elif (r.p_raw < alpha) and
not r.band_pass:`). `REGISTRATION_DRAFT.md` §4.1 defines "significant" as Holm-adjusted p <
0.05 one clause before using that exact word for WEAK's trigger — this is now the same test
`SEPARATES` already used (`r.holm_significant`), so WEAK and NULL are both keyed to the
registered multiplicity-corrected test; a covariate can no longer be raw-significant,
Holm-non-significant, and still surface as WEAK (which would misstate the mandatory R14
class-label line, per rev2 §D). No other code path changes — `p_holm ≥ p_raw` always (Holm
multiplier ≥ 1), so `SEPARATES` (which `disposition_for()`/`materiality_for_covariate()`
actually key off) was never affected by this bug, only the verdict string itself.

Test: `SYNTH_make_synth.py` "FIX 3" item 3 (`finding_weak_holm`) replicates rev2 §D's own
hand-check exactly — an m=10 family, sorted raw p-values `[0.001, 0.006, 0.011, 0.02, 0.03,
0.04, 0.3, 0.4, 0.5, 0.6]`, every covariate's `band_pass=False` (to isolate the
significance question) — and asserts rank-1 (`p_raw=0.001`, `p_holm=0.01<0.05`,
`holm_significant=True`) reads `WEAK`, while rank-2 (`p_raw=0.006<alpha` but
`p_holm=0.054≥0.05`, `holm_significant=False`) reads `NULL`, not `WEAK`.

### Item 4 (rev2 §E) — reported-only secondaries

Three new functions (`offset_subset_reads.py:680-734`), none feeding `disposition_for()`
or `class_label_line()` (no disposition role, per the task instruction):

- `spearman_secondaries(table, infl, family)` — §4.1: Spearman ρ between the primary
  family's `d_e` and every continuous covariate (`COVARIATE_TYPE[cov] == "continuous"`,
  i.e. C3c/C4/C5/C6/C7/C10/C11), over the table/influence inner join (all events with both
  a covariate value and a `d_e`), NaN-dropped pairwise; `n < 3` reports `rho=None` rather
  than calling `scipy.stats.spearmanr` on too few points.
- `class_composition_counts(table, s_index)` — §4.1: raw True/False/NaN counts for C1, C2,
  C3 within S (the registered "class composition of S as raw counts").
- `truth_disagreement_tables(table)` — §2: C1 (truth) vs C2 and C1 vs C3, each as a 2×2
  count table over the whole population (inner-joined, NaN-dropped).

Wired into `build_report()`'s `secondaries` key (line 903-907), called with the primary
family's `d_e` and S index.

Test: `SYNTH_make_synth.py` "FIX 3" item 4 (`finding_secondaries`) builds a 5-event table
with a perfectly monotonic covariate (`C4 = [1,2,3,4,5]`) against a perfectly
anti-monotonic `d_e = [5,4,3,2,1]`, asserting Spearman ρ = exactly −1.0 (`n=5`); a
hand-counted class composition over `S = {0,1}` for C1/C2/C3; and a hand-counted C1-vs-C2
and C1-vs-C3 2×2 table, matching by inspection.

### Item 5 — clear `named_covariates` on the INSTRUMENT / NO-READ override

`build_report()`'s existing unconditional override (`offset_subset_reads.py:914-916`):

```python
if instrument_note is not None:
    report["disposition"]["value"] = "INSTRUMENT / NO-READ"
    report["disposition"]["named_covariates"] = []
```

previously left `named_covariates` populated with whatever the pre-override disposition had
named (e.g. a `SUBSET-IDENTIFIED` covariate list) even though the disposition table's
"nothing banked" action for INSTRUMENT / NO-READ means no covariate claim should survive the
override. No dedicated new test was needed — FIX 2's Finding D exercise (censoring-red ⇒
instrument override) and this round's item 2 exercise (join-incomplete ⇒ instrument
override) both go through this same line; a manual check of `SYNTH_out.json`'s `disposition`
block after a forced-red run (not committed, hand-run only) confirms `named_covariates ==
[]` whenever `value == "INSTRUMENT / NO-READ"`.

### Extended synthetic check (`SYNTH_make_synth.py`, appended "FIX 3" section)

Console output from the actual run (full numeric record: `SYNTH_fix3_output.json`,
committed):

```
FIX 3 / SS5 2D-vs-1D disagreement: iiib_2d raw='SUBSET-IDENTIFIED', iiib_1d='DIFFUSE-IN-COVARIATES' -> final disposition='INTERMEDIATE' (would have banked SUBSET-IDENTIFIED without this trigger)
FIX 3 / SS6 g-population: 1 unmatched table row(s), 1 unmatched influence row(s) -> routed to INSTRUMENT / NO-READ
FIX 3 / SS4.1 WEAK-vs-Holm: C1 p_raw=0.001 p_holm=0.010 holm_sig=True -> WEAK; C2 p_raw=0.006 p_holm=0.054 holm_sig=False -> NULL (raw-significant, Holm-not)
FIX 3 / secondaries: spearman(d_e, C4).rho=-1.000 (exact -1.0); class_composition_S={'C1': {'n_true': 2, 'n_false': 0, 'n_nan': 0}, 'C2': {'n_true': 1, 'n_false': 1, 'n_nan': 0}, 'C3': {'n_true': 0, 'n_false': 2, 'n_nan': 0}}; truth_disagreement(C1 vs C2/C3)={'C2': {...}, 'C3': {...}}
FIX 3: all assertions passed (2D/1D disagreement, g-population join, WEAK-vs-Holm, secondaries)
```

Exercise 1's original 8-row `SYNTH_out.json` was re-run against the launch-block CLI
(unchanged `SYNTH_covariate_table_blind.csv`/`SYNTH_influence_vectors.csv`, sha256 still
`cf89eb374...`): disposition remains `INTERMEDIATE` (unchanged from FIX 2 — this 8-row
table's iiib_2d and iiib_1d both read `INTERMEDIATE` via the C10b NOT-TESTED gate, so the
new SS5 trigger agrees and does not change the outcome here); the new `gates.g_population`
(`join_complete: true`, 8/8 rows), `iiib_1d_disposition_check`
(`agrees_with_primary: true`), and `secondaries` blocks are present and populated in the
re-written `SYNTH_out.json` (committed).

### Quality gate (post-fix)

```
$ uv run ruff check offset_subset_reads.py SYNTH_make_synth.py
All checks passed!
$ uv run mypy offset_subset_reads.py
Success: no issues found in 1 source file
```

### Scope discipline (unchanged)

No production CRB, no `event_likelihoods.csv`, no galaxy catalogue, no cluster, no
`darksiren_emri/` file opened or run; `--table`/`--influence` never pointed at anything
under `graph1_20260901/retrieved/` or `seed61000/`; `covariate_table_iiib.csv` /
`covariate_table_joint_r1.csv` / `influence_iiib.csv` / `influence_joint_r1.csv` were not
opened at all this round (real mode was never invoked).

---

## Checklist table — every §2/§4/§5/§6/§8 item ↔ implementing code (or `ABSENT`)

Reviewer note: every row cites the current `offset_subset_reads.py` (post-FIX-3) unless
marked `ABSENT`.

### §2 (definitions, covariates, class axis)

| draft item | implementation | line(s) |
|---|---|---|
| C1 `in_catalog` | `COVARIATE_TYPE["C1"]="binary"`; column is phase A's output, read via `table[cov]` | 76, 268 |
| C2 `hosted_exact` (a) | `COVARIATE_TYPE["C2"]`; `CLASS_LABELS["C2"]` | 77, 91 |
| C3 `hosted_rel` (b) | `COVARIATE_TYPE["C3"]`; `CLASS_LABELS["C3"]` | 78, 91 |
| C3c `log10_f_cat` (c) | `COVARIATE_TYPE["C3c"]="continuous"`; `CLASS_LABELS["C3c"]` | 79, 91 |
| C4–C7, C10 | `COVARIATE_TYPE` entries, all in `HOLM_FAMILY` | 80-90 |
| C8 `cone_outside`, in-catalog-only restriction | `COVARIATE_TYPE["C8"]`; restriction applied in `run_family_separation` (`restrict = table.index[table["C1"].astype(bool)] if cov == "C8" else None`) | 84, 568 |
| C9 alias of C1, no separate test | not in `COVARIATE_TYPE`/`HOLM_FAMILY` (deliberately absent — draft says "no separate test") | n/a by design |
| C10b `low_M_timeout_bins12`, n≥10 gate | `COVARIATE_TYPE["C10b"]`; `C10B_MIN_N=10`; NOT-TESTED gate in `run_family_separation` | 86, 94, 561-576 |
| C11 `log10_snr`, reported-only | `REPORTED_ONLY=("C11",)`; handled in `run_family_separation`'s reported-only loop | 92, 582-586 |
| m=11 / m=10 family size (C10b conditional) | `HOLM_FAMILY` (11 members incl. C10b); `holm_correct()` only counts `verdict != "NOT-TESTED"` members as `m` | 90, 376-378 |
| §2 secondary: C1 vs C2/C3 truth-disagreement 2×2 | `truth_disagreement_tables()`, wired into `secondaries.truth_disagreement_2x2` | 716-734, 905 |

### §4.1 (separation)

| draft item | implementation | line(s) |
|---|---|---|
| Continuous: AUC via Mann-Whitney U, two-sided p | `_continuous_auc()` | 264-270 |
| Binary: Haldane OR, two-sided Fisher p | `_binary_or()` | 273-279 |
| C8 tested inside in_catalog stratum only | `restrict_index` param threaded through `separation_for_covariate` | 578, 288-296 |
| Holm step-down at family-wise α=0.05 over m | `holm_correct()` | 363-390 |
| SEPARATES band: Holm p<0.05 AND effect outside band | `holm_correct()`: `if r.holm_significant and r.band_pass: r.verdict = "SEPARATES"` | 379 |
| WEAK: Holm-significant AND inside band fails — **keyed to Holm p, not raw p (FIX 3 item 3)** | `holm_correct()`: `elif r.holm_significant and not r.band_pass: r.verdict = "WEAK"` | 380-386 |
| NULL (otherwise) | `holm_correct()` `else` branch | 387-388 |
| Secondary: Spearman ρ(d_e, each continuous covariate), all events — **FIX 3 item 4** | `spearman_secondaries()`, wired into `secondaries.spearman_d_e_vs_continuous` | 680-698, 904 |
| Secondary: class composition of S raw counts (C1/C2/C3) — **FIX 3 item 4** | `class_composition_counts()`, wired into `secondaries.class_composition_S` | 699-713, 905 |

### §4.2 (materiality)

| draft item | implementation | line(s) |
|---|---|---|
| Stratum: binary → enriched level; continuous → decile tail on enriched side | `materiality_for_covariate()` binary/continuous branches (Finding A/C fixed in FIX 2) | 468-495 |
| Δ_strat = mean_h(full−stratum) − mean_h(full) | `materiality_for_covariate()` | 502-503 |
| Null: 1000 draws same size, seed 20260904, percentile + 99% CI | `materiality_for_covariate()` null-draw loop + `null_percentile`/`null_ci99` | 516-531 |
| MATERIAL iff Δ_strat ≥ T_mat AND outside null 99% CI | `material = bool(delta_strat >= t_mat and outside_null)` | 530 |
| Oracle Δ_S + captured fraction | `delta_s_oracle`, `captured_fraction` | 505-508 |
| MAP rail flag every re-marginalisation (g-censoring) | `t0_moments()` rail flag; `map_rail_full`/`map_rail_stratum`/`null_rail_fraction` | 400-410, 469, 483, 516-527 |
| Reweighting NOT registered | not implemented (correctly — draft says it is out of scope) | n/a by design |

### §4.3 (replicate consistency)

| draft item | implementation | line(s) |
|---|---|---|
| 2-of-3 replicate families, same sign, before SUBSET-IDENTIFIED | `disposition_for()` `n_consistent` loop over `replicate_families` | 606-616 |

### §5 (disposition table)

| §5 row | implementation | line(s) |
|---|---|---|
| SUBSET-IDENTIFIED | `disposition_for()`: `mat.material` + `n_consistent>=2` → `identified` | 599-616 |
| DIFFUSE-IN-COVARIATES | `disposition_for()` line 634, past the `not_tested_gate` guard | 618-634 |
| INTERMEDIATE — separates but not material | `disposition_for()` lines 601-604 | 601-604 |
| INTERMEDIATE — material but not replicate-consistent | `disposition_for()` lines 611-614 | 611-614 |
| INTERMEDIATE — C8/C10b NOT-TESTED, nothing else separates | `disposition_for()` `not_tested_gate` | 630-633 |
| INTERMEDIATE — **primary 2D and 1D iiib families disagree — FIX 3 item 1** | `build_report()`: `families_agree` check, overrides `disposition` | 792-799 |
| INSTRUMENT / NO-READ — any §6 gate red | `build_report()`: unconditional override at the end, now also clears `named_covariates` (FIX 3 item 5) | 914-916 |
| Mandatory R14 class-label line, every disposition | `class_label_line()`, called unconditionally in `build_report()` | 637-676, 883 |

### §6 (gates, phase-C-owned rows only — G-1/G-2/G-3/g-precision are phase A/B, correctly `ABSENT` here)

| §6 gate | implementation | line(s) |
|---|---|---|
| G-4 blindness hash | `check_table_hash()`, called first in `main()` | 167-176, 964 |
| g-population: n/n_NaN disclosure | `SeparationResult.n_nan`, populated in `separation_for_covariate` | 121, 288-289 |
| g-population: C10b n≥10 disclosed | `c10b_testable` gate in `run_family_separation`, NOT-TESTED verdict when false | 561-576 |
| g-population: **"every table row joined (0 unmatched)" — FIX 3 item 2** | `check_join_completeness()`, wired into `gates.g_population` + `instrument_note` | 234-256, 746, 811-818, 888-890 |
| g-censoring: MAP rail disclosure, null-rail red wired to INSTRUMENT / NO-READ | `MaterialityResult.map_rail_full/map_rail_stratum/null_rail_fraction/censoring_gate_red`; wired in `build_report()` | 400-410, 469, 483, 519-527, 820-832 |

### §8 (launch block)

| draft item | implementation | line(s) |
|---|---|---|
| `--table --table-sha256 --influence --alpha --auc-band --or-band --t-mat --decile --null-draws --null-seed --out [--dry-run]` | `parse_args()` | 925-943 |
| `--k-<family>` sanity flags (harmless addition) | `parse_args()` loop | 941-943 |
| sha256 refused before any covariate touched | `main()`: `check_table_hash()` is the first statement | 947 |

**Summary: nothing from §2/§4/§5/§6/§8 is `ABSENT` after this round.** The three RED
findings from `DESIGN_GATE_formula_rev2.md` (§B, §C, §D) and the one lower-severity gap
(§E) are all implemented and wired per the table above; C9 and "reweighting not registered"
are correctly absent by the draft's own design, not gaps.

---

## FIX 4 (builder round 4) — real-schema mapping + hard covariate/influence pre-flight

Trigger: `DESIGN_GATE_formula_rev3.md` finding 4.2 (RED) — `offset_subset_reads.py` read
bare `C1..C11` covariate ids and a per-family `{family}_in_S` influence flag, matching only
the hand-built `SYNTH_*` fixture, never the real, already-built phase-A/B output
(`covariate_table_{iiib,joint_r1}.csv`, `influence_{iiib,joint_r1}.csv`), and the failure
mode if ever pointed at that real data was **silent**: every covariate skipped via
`if cov not in table.columns: continue`, `disposition_for()` falling through to a false
`DIFFUSE-IN-COVARIATES` with zero of eleven covariates actually tested. `REGISTRATION_DRAFT.md`
"PIN CORRECTIONS" (2026-09-04 ~00:40 CEST) ratifies the schema of record as the BUILT files'
own headers and requires (1) an explicit, asserted bare→real mapping and (2) a hard
pre-flight: any registered covariate or influence column missing ⇒ `INSTRUMENT-DEFECT`,
written to the JSON, exit non-zero, never a silent skip. No statistic, threshold, or
disposition path was touched — every change is confined to the loading layer (`load_table`,
`load_influence`, the new `detect_venue`/`check_covariate_schema`/`check_influence_base_schema`/
`InstrumentDefectError`, and `main()`'s pre-flight wiring).

### 1. Real headers, confirmed via `head -1` (never opened beyond the header line)

```
$ head -1 covariate_table_iiib.csv
event_idx,C1_in_catalog,C4_z_gw,C5_log10_sky_area,C8_cone_outside,C10_log10_M,C10b_low_M_timeout_bins12,C11_log10_snr,C2_hosted_exact,C3_hosted_rel,C3c_log10_f_cat,C3c_censored,C6_mass_window_retention,C7_log10_n_cand_1d
$ head -1 covariate_table_joint_r1.csv
(identical schema to covariate_table_iiib.csv)
$ head -1 influence_iiib.csv
event_idx,influence_2D,influence_1D,rank
$ head -1 influence_joint_r1.csv
(identical schema to influence_iiib.csv)
```

`BUILD_RECORD_B2.md` "Output files" (phase B's own definition, a markdown build record, not
a forbidden CSV): "`influence_2D`/`influence_1D` are the directional statistic d_e … positive
= removing the event moves mean_h toward truth" — i.e. these columns already ARE d_e, not the
raw `influence = mean_h(full) − mean_h(full−e)` the name suggests, and there is no `_in_S`
column at all: S must be derived from the top-k rank over this column, never read from a
banked flag (`REGISTRATION_DRAFT.md` §2: "S is defined by the BANKED k, not re-derived").

Second structural fact used below: the built outputs are **one covariate table + one
influence file per venue** (iiib, joint_r1), not the draft's single combined pair — so a
single invocation of `offset_subset_reads.py` covers exactly one venue's two families
(`iiib_2d`/`iiib_1d` or `jr1_2d`/`jr1_1d`). `detect_venue()` reads this off the `--table`/
`--influence` filenames (both must name the same venue, else `INSTRUMENT-DEFECT`) and gates
which two families are processed (`VENUE_FAMILIES`).

### 2. The explicit, asserted mapping table (in code: `COVARIATE_COLUMN_MAP`)

| registered id | real column (covariate_table_{iiib,joint_r1}.csv) |
|---|---|
| C1 | `C1_in_catalog` |
| C2 | `C2_hosted_exact` |
| C3 | `C3_hosted_rel` |
| C3c | `C3c_log10_f_cat` |
| C4 | `C4_z_gw` |
| C5 | `C5_log10_sky_area` |
| C6 | `C6_mass_window_retention` |
| C7 | `C7_log10_n_cand_1d` |
| C8 | `C8_cone_outside` |
| C10 | `C10_log10_M` |
| C10b | `C10b_low_M_timeout_bins12` |
| C11 | `C11_log10_snr` |

A module-level `assert set(COVARIATE_COLUMN_MAP) == set(HOLM_FAMILY) | set(REPORTED_ONLY)`
guarantees the map can never silently drop a registered covariate out of the pre-flight.
`C3c_censored` (present in the real file, not a registered covariate) is left untouched —
harmless, not consumed anywhere.

Influence side (`FAMILY_D_E_SOURCE_COL`): `iiib_2d`/`jr1_2d` → `influence_2D`;
`iiib_1d`/`jr1_1d` → `influence_1D` — venue is selected by which per-venue file is loaded
(§8 invokes the script once per venue), never by column name.

### 3. Hard pre-flight (never a silent skip)

`check_covariate_schema()` (called first thing inside `load_table`, before the index is even
set) and `check_influence_base_schema()` (called first thing inside `load_influence`) each
raise the new `InstrumentDefectError` — carrying a `message` and a structured `detail` dict —
the instant a required real column is absent. `main()` wraps the whole load/pre-flight
sequence in one `try/except InstrumentDefectError`: real mode writes
`{"disposition": {"value": "INSTRUMENT-DEFECT", ..., "detail": ...}}` to `--out` and returns
1; `--dry-run` prints the message and returns 1 without writing (dry-run never writes a file
under any outcome, per its own contract). `main()` additionally recomputes
`missing_covariates` after `load_table` returns as a belt-and-suspenders regression guard —
now structurally unreachable as non-empty given `check_covariate_schema` already gated
`load_table`, but kept so a future edit that weakens the schema pre-flight still cannot reach
`build_report()`/`disposition_for()` with a partially-populated table (the exact silent
DIFFUSE-IN-COVARIATES path finding 4.2 named). Verified live:

```
$ uv run python offset_subset_reads.py \
    --table SYNTH_bad_schema_covariate_table.csv (C4_z_gw dropped) --table-sha256 <recomputed> \
    --influence influence_iiib.csv --out /tmp/should_not_write.json --dry-run
INSTRUMENT-DEFECT: covariate table missing required column(s) for registered covariate(s): C4 -> C4_z_gw
exit code: 1, /tmp/should_not_write.json NOT written
```

### 4. §8 launch block, `--dry-run`, on the REAL committed inputs — exit 0, 1588/1588 per venue

Hashes recomputed and matched against the committed `covariate_table.sha256` before
running (G-4 gate exercised, not bypassed):

```
$ sha256sum covariate_table_iiib.csv covariate_table_joint_r1.csv
90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0  covariate_table_iiib.csv
fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a  covariate_table_joint_r1.csv
(matches the committed covariate_table.sha256 exactly)

$ uv run python offset_subset_reads.py \
    --table covariate_table_iiib.csv --table-sha256 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0 \
    --influence influence_iiib.csv --alpha 0.05 --auc-band 0.20 --or-band 3.0 --t-mat 0.008 \
    --decile 0.10 --null-draws 1000 --null-seed 20260904 --out offset_subset_result_iiib.json --dry-run
venue: iiib
table: covariate_table_iiib.csv (1588 rows), sha256 OK
influence: influence_iiib.csv (1588 rows)
join: 1588 table rows / 1588 influence rows joined on event_idx; unmatched table-only=0, unmatched influence-only=0; join_complete=True
logL columns present: False (h_grid n=0)
  family iiib_2d: k=82
  family iiib_1d: k=94
dry-run OK
$ echo exit=$?
exit=0

$ uv run python offset_subset_reads.py \
    --table covariate_table_joint_r1.csv --table-sha256 fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a \
    --influence influence_joint_r1.csv --alpha 0.05 --auc-band 0.20 --or-band 3.0 --t-mat 0.008 \
    --decile 0.10 --null-draws 1000 --null-seed 20260904 --out offset_subset_result_joint_r1.json --dry-run
venue: jr1
table: covariate_table_joint_r1.csv (1588 rows), sha256 OK
influence: influence_joint_r1.csv (1588 rows)
join: 1588 table rows / 1588 influence rows joined on event_idx; unmatched table-only=0, unmatched influence-only=0; join_complete=True
logL columns present: False (h_grid n=0)
  family jr1_2d: k=72
  family jr1_1d: k=46
dry-run OK
$ echo exit=$?
exit=0
```

Both runs: **1588/1588 joined, 0 unmatched either direction**; per-family `k` reproduces the
registered banked k EXACTLY (82/94/72/46, §2/G-2(ii)) via the top-k-rank derivation, not a
re-read of a banked flag; `logL columns present: False` confirms materiality correctly reports
NOT-TESTED on this real data (the primary-family-only `logL_h*` data contract is still
unmet by the built files — the already-disclosed, separate open item from finding 4.2/4.3,
unchanged by this round). **Neither invocation touched a registered aggregate**: `--dry-run`
loads, hash-checks, schema-checks, prints row/join/k counts, and exits — no AUC/OR/p/Δ_strat
is computed. `--out` was never written (confirmed: no `offset_subset_result_*.json` file
exists after either run).

### 5. `SYNTH_make_synth.py` extension — real-suffixed-schema fixture + missing-column checks

Appended a "FIX 4" section (after the untouched FIX 2/FIX 3 sections, which still pass
byte-identically — re-run live, same output) that:

1. Builds a 10-row synthetic table/influence pair using the REAL suffixed/generic schema
   (`C1_in_catalog`, …, `influence_2D`/`influence_1D`/`rank`) and drives it through the real
   `load_table`/`load_influence`/`detect_venue` functions — asserts the bare `C1`/`C4`
   columns are correctly mapped and `{family}_in_S` is correctly derived from the top-k rank
   (`S={0,1,2}` for k=3 by construction) for both native families, and that the OTHER venue's
   families (`jr1_2d`) do not appear in the loaded influence frame.
2. Drops `C4_z_gw` from the covariate fixture — asserts `load_table` raises
   `InstrumentDefectError` naming `C4 -> C4_z_gw`, with the structured `detail` dict populated.
3. Drops `influence_1D` from the influence fixture — asserts `load_influence` raises
   `InstrumentDefectError` with `detail["missing_influence_columns"] == ["influence_1D"]`.
4. Pairs an iiib table filename with a joint_r1 influence filename — asserts `detect_venue`
   raises `InstrumentDefectError` rather than guessing.
5. Runs the actual CLI (subprocess, mirroring the launch-block invocation exactly) with
   `--dry-run` on the REAL, committed `covariate_table_{iiib,joint_r1}.csv` /
   `influence_{iiib,joint_r1}.csv` pairs — asserts exit 0, the literal
   "1588 table rows / 1588 influence rows joined" / "join_complete=True" strings, the exact
   per-family `k` lines, and that the `--out` path is never created.

Live run (`uv run python SYNTH_make_synth.py`), FIX 4 section only:

```
FIX 4 / happy path: venue='iiib', iiib_2d S=[0, 1, 2], iiib_1d S=[0, 1, 2], all 12 registered covariates present after schema mapping
FIX 4 / missing covariate column: InstrumentDefectError raised as required: covariate table missing required column(s) for registered covariate(s): C4 -> C4_z_gw
FIX 4 / missing influence column: InstrumentDefectError raised as required: influence vectors missing required column(s): ['influence_1D']
FIX 4 / venue mismatch: InstrumentDefectError raised as required: cannot determine a single venue (iiib / joint_r1) from --table=covariate_table_iiib.csv and --influence=influence_joint_r1.csv; this script processes exactly one venue's covariate table + influence file per invocation (PIN CORRECTIONS item 1).
FIX 4 / --dry-run (iiib, REAL inputs): exit 0, 1588/1588 joined, k={'iiib_2d': 82, 'iiib_1d': 94}, no file written
FIX 4 / --dry-run (joint_r1, REAL inputs): exit 0, 1588/1588 joined, k={'jr1_2d': 72, 'jr1_1d': 46}, no file written
FIX 4: all assertions passed (real-schema mapping, missing-column INSTRUMENT-DEFECT x2, venue mismatch, real-input --dry-run x2)
```

All FIX 2 and FIX 3 assertions in the same run still pass unchanged (their console lines are
byte-identical to the prior round's, confirming zero regression to any statistic/threshold/
disposition path).

### 6. Quality gates

```
$ uv run ruff check offset_subset_reads.py SYNTH_make_synth.py
All checks passed!
$ uv run mypy offset_subset_reads.py
Success: no issues found in 1 source file
```

`mypy` on `SYNTH_make_synth.py` surfaces 3 pre-existing errors (a `str`-vs-`Literal` verdict
argument and two `float | None` subtractions), all inside the **untouched FIX 2/FIX 3 sections**
this round did not edit — not introduced by FIX 4, and left as-is rather than hand-editing
already-hand-verified build-record evidence lines outside this round's scope
(`DESIGN_GATE_formula_rev3.md` hand-re-derived those exact assertions against the committed
output; touching that code would invalidate that verification without being asked to).

### 7. Real mode — explicitly NOT run

Per the task boundary for this round, real mode (`offset_subset_reads.py` without
`--dry-run`, against the real `covariate_table_*.csv`/`influence_*.csv`) was **not invoked**.
`build_report()`'s cross-venue family loop (`FAMILIES`/`PRIMARY_FAMILY`/`REPLICATE_FAMILIES`
spanning both `iiib_*` and `jr1_*`) still assumes a single load carries all four families,
which the real per-venue file split does not support in one invocation — the same
architecture gap `DESIGN_GATE_formula_rev3.md` §4.3 already disclosed, not solved by this
round (out of scope: items 1-2 gate columns, not cross-venue orchestration) and not required
for `--dry-run` to pass. Flagged for whoever authors the real-mode launch (author ruling
needed on whether real mode runs twice, once per venue, with a combined third pass merging
the two JSONs for the 2-of-3 replicate check, or the launch block itself is revised).
