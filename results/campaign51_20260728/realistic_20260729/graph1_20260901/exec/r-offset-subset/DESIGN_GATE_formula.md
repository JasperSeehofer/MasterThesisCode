# DESIGN_GATE_formula.md — r-offset-subset, FORMULA-MATCH review of B3 (`offset_subset_reads.py`)

Reviewer: fresh formula-match gate (no prior context on this arm). Scope per assignment: hand
arithmetic on the committed synthetic check (`BUILD_RECORD_B3.md`); disposition-row -> code-branch
mapping incl. R14 emission; sha256 refusal path; launch-block CLI = argparse; B1 column
*definitions* only (no formula check on data). `INFORMATION_FORECAST.md` was not opened. No
registered aggregate was computed over the registered population; all checks below are either
static code reading, hand arithmetic on the committed `SYNTH_*` files, or header-only inspection
of the real `covariate_table_iiib.csv` / `influence_iiib.csv` (column names only, no row beyond
the header, no statistic).

## Verdict: **RED**

Two confirmed formula-match defects in `offset_subset_reads.py` would make the real-mode read
wrong or produce an unregistered disposition; a third is a real but conditional correctness bug;
a fourth is a missing §6 gate. None of these were caught by the committed synthetic exercises
because the exercises never drive the code paths where the bugs live. Everything the exercises
*did* cover (Holm arithmetic, AUC/OR arithmetic, sha256 refusal, CLI-to-launch-block match) hand-
checks out clean — see §1-§4 below.

---

## 1. Hand arithmetic on the committed synthetic check (Exercise 1, `SYNTH_out.json`)

Recomputed by hand from `SYNTH_covariate_table_blind.csv` / `SYNTH_influence_vectors.csv`
(S = {0,1,2}, B = {3..7}, n_S=3, n_B=5):

- **C1** (binary): a=1,b=2,c=2,d=3 -> OR_Haldane = (1.5*3.5)/(2.5*2.5) = **0.84** ✓ matches table.
- **C2**: a=2,b=1,c=1,d=4 -> OR = (2.5*4.5)/(1.5*1.5) = **5.0** ✓.
- **C3**: a=3,b=0,c=1,d=4 -> OR = (3.5*4.5)/(0.5*1.5) = **21.0** ✓.
- **C3c** (AUC, S={-1,-0.5,-2} vs B={-6,-7,-1.5,-8,-9}): U=5+5+4=14 -> AUC=14/15=**0.9333** ✓
  (only B's -1.5 beats S's -2.0).
- **C4/C5/C7/C10** (complete separation, n_S=3,n_B=5): AUC=1.0, exact two-sided Mann-Whitney
  p = 2/C(8,3) = **0.0357** ✓ for all four.
- **C6** (n=2 vs 4 after dropping the two NaNs, complete separation): AUC=1.0,
  p = 2/C(6,2) = **0.1333** ✓, `n_nan=2` ✓ (events 2 and 6).
- **C8** (restricted to C1==True, i.e. {0,3,6}; S∩cat={0}, B∩cat={3,6}): a=0,b=1,c=2,d=0 ->
  OR=(0.5*0.5)/(1.5*2.5)=**0.0667** ✓, n=1 vs 2 ✓.
- **C10b**: all False -> n=0<10 -> **NOT-TESTED** ✓ (correctly excluded from Holm's `m`, so
  `m=10`, matching the table's header).
- **Holm step-down at m=10**: sorted raw p = [0.0357×4, 0.0714(C3c), 0.1333(C6), 0.1429(C3),
  0.3333(C8), 0.464(C2), 1.0(C1)]. Running-max walk: i=0..3 give candidate multipliers
  10,9,8,7 × 0.0357, max = **0.357** (< the true 0.0357*10, confirms `(m-i)*p_raw` with 0-indexed
  `i` is right); i=4 (C3c) 6×0.0714=0.4286 -> running max 0.4286 (table shows 0.429, rounding) ✓;
  i=5 (C6) 5×0.1333=0.6667 -> running max 0.6667 (table 0.667) ✓; i=6 (C3) 4×0.1429=0.5714 <
  running max, so **carried forward at 0.667** (table shows C3 p_holm=0.667, *not* 0.571 — this
  is correct Holm monotonicity, not a bug: `running_max = max(running_max, adj)` is exactly the
  textbook enforced-monotone step-down) ✓; remaining entries clamp at 1.0 ✓.
- **Minimum attainable Holm p at this n is 0.357 > α=0.05** — so *nothing* can reach SEPARATES at
  n_S=3, confirming `DIFFUSE-IN-COVARIATES` / `named_covariates: []` is the mechanically correct
  disposition for this exercise, and confirming the record's own stated reasoning.

All of Exercise 1 hand-checks clean. Exercise 2's direct call into `materiality_for_covariate`
(continuous covariate C4, forced SEPARATES) also hand-checks: `n_tail = max(1, round(8*0.10)) = 1`
-> stratum = the single top-C4 event (event 0) ✓; `delta_strat=0.0009 < t_mat=0.008` ->
`material=False` ✓, independent of the null draw (correct AND-gate short-circuit-equivalent
behavior even though the code doesn't literally short-circuit).

**Coverage gap in the exercises themselves (relevant to what follows):** Exercise 1 never reaches
the materiality function at all (nothing separates at n=8), and Exercise 2 only exercises the
**continuous** branch of `materiality_for_covariate` (C4). The **binary** branch of that function,
the NOT-TESTED disposition trigger, the null-draw rail check, and the NaN/top-decile interaction
are exercised nowhere in `BUILD_RECORD_B3.md`. All four of the findings below live in that
uncovered surface.

## 2. Confirmed defects

### Finding A (RED) — binary materiality stratum uses the wrong enrichment direction

`materiality_for_covariate`, binary branch:

```python
s_bool = table[covariate].reindex(s_index).astype(bool)
enriched_level = bool(s_bool.mean() >= 0.5) if s_bool.size else True
stratum_mask = (col.astype(bool) == enriched_level).to_numpy()
```

This derives "the level enriched in S" from the **raw majority level inside S alone**
(`s_bool.mean() >= 0.5`), never consulting `sep.effect` — the odds ratio that is the actual,
already-computed, already-registered enrichment direction (relative to bulk B) that caused this
covariate to reach SEPARATES in the first place. The continuous branch, by contrast, correctly
uses `sep.effect >= 0.5` (the AUC) to pick top-vs-bottom decile — so the two branches use
inconsistent rules for "which side is enriched," and the binary one is wrong.

This matters concretely, not hypothetically: every registered binary covariate has population
prevalence well under 50% (C1 `in_catalog` = 76/1588 ≈ 4.8%; C10b ≈ 0.3-0.4%; C8 restricted to a
76-row stratum). §5's own worked power example for C1 says an OR of 3 corresponds to "≈ 11
in_catalog members of S vs 3.9 expected" at k=82 — i.e. **11/82 ≈ 13%** of S, which is `< 0.5`.
Under the current code, `s_bool.mean() >= 0.5` would evaluate `False` here, so
`enriched_level = False` — the code would freeze the stratum as **"not in_catalog,"** the exact
opposite of the direction (`in_catalog` enriched, OR=3 > 1) that made C1 SEPARATE. `Δ_strat` would
then be computed by leaving out the *majority, non-enriched* class rather than the registered
"stratum enriched in S," silently reporting a materiality number for a stratum the draft never
defined. This is a defect that makes the registered read wrong — RED.

Fix (for the launch reviewer): use `sep.effect >= 1.0` (the OR direction), symmetric with the
continuous branch's `sep.effect >= 0.5` (AUC direction), instead of recomputing a raw majority.

### Finding B (RED) — the §5 "NOT-TESTED -> INTERMEDIATE" disposition row has no code branch

`REGISTRATION_DRAFT.md` §5 lists as an INTERMEDIATE trigger: *"C8 or C10b NOT-TESTED and no other
covariate separates."* `disposition_for()` implements no such check:

```python
separators = [cov for cov, r in primary.items() if cov in HOLM_FAMILY and r.verdict == "SEPARATES"]
...
if not separators:
    return "DIFFUSE-IN-COVARIATES", []
```

If `separators` is empty, the function unconditionally returns `DIFFUSE-IN-COVARIATES` — it never
inspects whether C8 or C10b's verdict is `NOT-TESTED` versus `NULL`. This is not a corner case:
`BUILD_RECORD_B1.md` already reports, on the real production table, **C10b is NOT-TESTED in both
venues** (`n C10b=True = 5 (NOT-TESTED, n<10)`, iiib and joint_r1 alike) — a fact independently
confirmed by this reviewer from the same build record. So in the actual registered run, if every
other covariate comes back NULL, the code as written will emit `DIFFUSE-IN-COVARIATES` — with its
strong claim-writeback `c-offset-diffuse-in-covariates SUPPORTED` and `q-offset-subset
SETTLED-BOUNDED` — in exactly the situation §5 reserves for the weaker, revision-eligible
`INTERMEDIATE` disposition. A disposition-table row from §5 has no corresponding code branch and
the missing branch is essentially guaranteed to be reachable — RED.

### Finding C (conditional, real) — NaN rows can populate the "top decile" materiality stratum

```python
ranked = col.rank(method="first", na_option="bottom")
if auc_above_half:
    stratum_mask = (ranked > (n_total - n_tail)).to_numpy()
```

Verified directly (`pandas.Series.rank(na_option="bottom")` assigns **NaN the largest rank
values**, not the smallest — confirmed by a throwaway interpreter check, not by touching any
registered file). So when `auc_above_half` is True (AUC ≥ 0.5, "top decile" per §4.2), any NaN
rows in that covariate outrank every real value and are the first to be swept into the stratum. A
registered continuous covariate with NaN count ≥ the decile size (n_tail ≈ 159 at n=1588) would
get a "top decile" stratum consisting wholly or partly of *undefined-covariate* rows rather than
real high-value rows — not the stratum §4.2 defines. `C6` (`mass_window_retention`) is exactly
such a covariate: `BUILD_RECORD_B1.md` reports 606/1588 NaN in iiib (n_1D==0 events), far above
the ~159-row decile. This only bites if C6 (or another NaN-bearing continuous covariate) actually
SEPARATES with AUC ≥ 0.5 — unverified either way by this review (no registered aggregate was
computed) — so it is flagged as a real, mechanically-demonstrated bug of conditional impact rather
than an unconditional one.

### Finding D (gate omission) — §6 g-censoring's null-draw rail check is not implemented

§6 g-censoring requires: *"MAP position for the full sample, every stratum leave-out **and every
null draw**; any MAP at 0.60/0.86 ⇒ that Δ is a BOUND, rail fraction reported."*
`materiality_for_covariate` computes `map_rail_full` and `map_rail_stratum` but, inside the
null-draw loop, discards the MAP entirely (`mean_h_draw, _, _ = t0_moments(...)`) — there is no
`null_rail_fraction` field anywhere in `MaterialityResult` or the JSON output. If a meaningful
fraction of the 1000 null draws rail, that fact is invisible to the reader and to the disposition
logic, contrary to the registered gate. This is a gate the code silently never runs, not a
computed-wrong number — flagged for the launch reviewer, one severity notch under A/B.

### Not a defect, noted for the record — a dead disposition branch and a display-rounding non-issue

`disposition_for`'s final fallback `return "INTERMEDIATE", []` is unreachable (every covariate in
a non-empty `separators` is always appended to `identified` or `intermediate`, so if `separators`
is non-empty at least one of those lists is non-empty, and the `not separators` branch already
covers the empty case) — harmless dead code, not a correctness issue. The Holm p-values in
`BUILD_RECORD_B3.md`'s table (e.g. C3c 0.429 vs my hand value 0.4286) are display rounding, not a
computation error — reproduced exactly above.

## 3. Disposition-row -> code-branch map (§5 vs `disposition_for` / `build_report`)

| §5 row | trigger | code branch | status |
|---|---|---|---|
| SUBSET-IDENTIFIED | SEPARATES + MATERIAL + replicate-consistent | `identified.append(cov)` path, `if identified: return "SUBSET-IDENTIFIED"` | present, correct |
| DIFFUSE-IN-COVARIATES | no covariate SEPARATES in primary | `if not separators: return "DIFFUSE-IN-COVARIATES"` | present, but **over-broad** — fires even when the cause is a NOT-TESTED covariate (Finding B) |
| INTERMEDIATE — SEPARATES but not MATERIAL | | `if mat is None or not mat.material: intermediate.append(cov)` | present, correct |
| INTERMEDIATE — MATERIAL but not replicate-consistent | | `n_consistent < 2 -> intermediate.append(cov)` | present, correct |
| INTERMEDIATE — C8/C10b NOT-TESTED, nothing else separates | | **none** | **missing (Finding B)** |
| INTERMEDIATE — primary 2D vs iiib 1D family-level disagreement | | **none** (only the narrower per-covariate §4.3 2-of-3 rule exists) | **missing** |
| INSTRUMENT / NO-READ | any §6 gate red | `instrument_note` set only for the logL-columns-absent case; the g-censoring null-rail gate (Finding D) has no red condition to trigger this at all | **partial** — only one of several §6 gates is wired to this disposition |

**R14 mandatory class-label line:** `class_label_line()` is called unconditionally in
`build_report()` before the disposition/instrument branch, so it is present in every JSON output
regardless of disposition (including the DIFFUSE case exercised in `SYNTH_out.json`, where it
correctly emits `"class is not the axis"`) — this part is confirmed correct and matches §5's
"every disposition" mandate.

## 4. sha256 refusal path (G-4)

`check_table_hash()` recomputes the table's sha256 and calls `SystemExit` before `load_table` or
`load_influence` are reached, so a mismatch touches neither the covariate table's values nor the
influence file — matches G-4's intent ("refuses to run"). `BUILD_RECORD_B3.md` §5 documents an
actual mismatch run (`deadbeef...` -> rc=1, correct message). Confirmed correct by inspection; not
re-run here (no need to touch any file to verify a `raise`/`SystemExit` control-flow path already
demonstrated in the build record).

## 5. Launch-block CLI vs argparse

Every flag in `REGISTRATION_DRAFT.md` §8's phase-C invocation
(`--table --table-sha256 --influence --alpha --auc-band --or-band --t-mat --decile --null-draws
--null-seed --out [--dry-run]`) is present in `parse_args()` with matching defaults
(`0.05 / 0.20 / 3.0 / 0.008 / 0.10 / 1000 / 20260904`) — exact match, confirmed by direct
side-by-side reading of `REGISTRATION_DRAFT.md:223-225` against `offset_subset_reads.py:689-705`.
The four `--k-<family>` flags are additive-only (defaulted to the registered banked k, never
required by the launch block), consistent with "optional ... sanity-check flags" in the module
docstring.

## 6. B1 column definitions vs draft C1-C11 (definitions only, no data check)

`BUILD_RECORD_B1.md`'s column table maps each of C1-C11 to a definition that matches
`REGISTRATION_DRAFT.md` §2 **conceptually** (source column/formula, e.g. C4 = `dist_to_redshift`
on CRB `luminosity_distance`, C5 = `log10(pi * cone_radius^2)` reusing `cone_loss_reads.py`, C10b
= `M < 169568.12917853205`, etc.) — no formula divergence found in the definitions themselves.

**However**, the *names and file structure B1 actually emitted do not match what
`offset_subset_reads.py` requires*, and this reviewer independently confirmed it by reading the
real header (not a registered aggregate — no row beyond the header was read):

```
$ head -1 covariate_table_iiib.csv
event_idx,C1_in_catalog,C4_z_gw,C5_log10_sky_area,C8_cone_outside,C10_log10_M,
C10b_low_M_timeout_bins12,C11_log10_snr,C2_hosted_exact,C3_hosted_rel,C3c_log10_f_cat,
C3c_censored,C6_mass_window_retention,C7_log10_n_cand_1d
```

`offset_subset_reads.py` indexes covariates by their bare id (`table[covariate]` with
`covariate="C1"`, etc., via `COVARIATE_TYPE`/`HOLM_FAMILY`); B1 wrote `C1_in_catalog`, not `C1`.
Additionally, B1 wrote **two** per-venue files (`covariate_table_iiib.csv`,
`covariate_table_joint_r1.csv`) where the draft's §3 phase A and §8 launch block specify **one**
`covariate_table_blind.csv`. Neither file is even named `covariate_table_blind.csv`. Pointed at
today's real files, `offset_subset_reads.py` cannot run as specified (a `--table` value that
literally doesn't exist, and if renamed/pointed manually, silent `cov not in table.columns`
skips per `run_family_separation`'s `continue`, since none of the bare `C1..C11` names would
match). This corroborates, independently, what `BUILD_RECORD_B3.md` §8 already self-disclosed
about `BUILD_RECORD_B2.md`'s influence-file schema (`event_idx, influence_2D, influence_1D, rank`
— no `in_S`, no per-family split, no `logL_h*`, also not matching what the reader requires). This
is a launch blocker, not a new B3 formula bug (B3 already flagged its half); it is reported here
because it also falls under this review's explicit "confirm B1's column definitions match" charge
and because, combined with Finding B, it means **no exercise so far has actually run this reader
against real-shaped data at all** — every check that "passed" ran against hand-built ≤10-row
synthetic data, never against a table/influence pair with the registered schema.

## 7. What would flip this to GREEN

1. Finding A: derive `enriched_level` (binary) from `sep.effect >= 1.0`, matching the continuous
   branch's use of `sep.effect >= 0.5`.
2. Finding B: add the NOT-TESTED-covariate branch to `disposition_for` (C8 or C10b NOT-TESTED and
   no other covariate SEPARATES -> INTERMEDIATE, not DIFFUSE-IN-COVARIATES).
3. Finding C: exclude NaN rows from `ranked`/`stratum_mask` construction (e.g. rank only the
   non-NaN subset, or explicitly mask NaN out of `stratum_mask` regardless of `na_option`).
4. Finding D: track and report a null-draw rail fraction, and wire a rail-heavy null population
   into an INSTRUMENT/NO-READ (or at minimum a disclosed-bound) condition per §6 g-censoring.
5. §6: reconcile B1's column names/file split and B2's influence-vector schema against what this
   reader actually requires (a single `covariate_table_blind.csv` with bare `C1..C11` names; one
   `influence_vectors.csv` with `{family}_in_S`, `{family}_d_e`, and primary-family `logL_h*`
   columns) — or amend the draft's §3 phase A/B one-line schema description and re-issue the
   launch block accordingly.
