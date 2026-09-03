# READ_RECORD_rev5.md — r-offset-subset, phase C (the reader), REAL mode, DISJOINT READER (round 5)

Role: disjoint reader for **m-offset-subset**, executing exactly `REGISTRATION_DRAFT.md`'s
**PIN CORRECTION 3 (round 5, 2026-09-04)** §8 launch block — the corrected single-invocation,
both-venue CLI that replaces the round-4 per-venue block which crashed (`KeyError` on the other
venue's undefined `*_in_S` column; see the pre-existing `READ_RECORD.md`, round-4 crash record,
left untouched by this file). Read `REGISTRATION_DRAFT.md` including the PIN CORRECTIONS section.
Did **not** open `INFORMATION_FORECAST.md` (forbidden). Did **not** inspect the data files by
hand — the pinned CLI (`offset_subset_reads.py`) was invoked exactly once in real mode; every
finding below comes from that invocation's own JSON output plus the pre-existing gate documents
already on disk (computability GREEN, byte-id 30/30, formula rev3 40/40, rev4 GREEN, rev5 GREEN).
Touched no production pipeline, no cluster, no file under `darksiren_emri/`. Modified no script.

**This record is VERDICT-FREE**: it states what ran, what the gates report, and what came out the
other end, including the disposition VALUE the registered code computed. It does not itself
ADJUDICATE c-offset-subset-covariate vs c-offset-diffuse-in-covariates, nor rule on R14 — those are
author decisions per REGISTRATION_DRAFT.md §5's "every row returns as a fresh RULE."

---

## 1. Gates checked before launch (pre-existing documents, read not re-derived)

| gate | file | verdict as read |
|---|---|---|
| Computability | `DESIGN_GATE_computability.md` | **GREEN** (two non-blocking AMBER documentation notes disclosed: a pin-count nit in G-1's "four md5s" wording vs. five listed; the DIFFUSE-IN-COVARIATES kill-criterion's "verbatim" provenance tag) |
| Byte-id anchors (phase B) | `BYTEID_RECORD.md` | **GREEN — 30/30 checks passed** (full-sample mean_h to 1e-9 both primary families; minimal k = 82/94/72/46 exact all four families; top-10 directional influence to 1e-12 relative; k=1588 endpoint = 0.73 to 1e-12; 0 physics-floor exclusions; both CSV md5s match) |
| Formula/code review, rev3 | `DESIGN_GATE_formula_rev3.md` | **40/40 items GREEN** on `offset_subset_reads.py`'s statistics logic. Overall document verdict was RED-for-launch at the time, for two reasons *outside* the 40/40 statistics check: a since-addressed blindness-leak disclosure and the phase-A/phase-C schema mismatch that PIN CORRECTION 3 exists to fix |
| Formula/code review, rev4 | `DESIGN_GATE_formula_rev4.md` | **GREEN** on its four assigned checks (column mapping vs. real headers, missing-column hard pre-flight, §8 `--dry-run` on both real venues, statistics-code identity vs. rev3). Disclosed one still-open gap at the time: the built `influence_*.csv` files carry no `logL_h<value>` columns, so materiality could not be computed under the then-current data contract |
| Formula/code review, rev5 | `DESIGN_GATE_formula_rev5.md` | **GREEN**, all five checks: (1) the corrected CLI == the parser, argument-for-argument; (2) an independently re-run `--dry-run` on the real inputs, both sha256 and both md5 OK, 1588/1588 joined both venues, k=82/94/72/46, no `--out` file written; (3) materiality's third data path (`load_primary_logl_matrix`) is a literal import of `build_influence_vector._load_matrix`, and `t0_moments()` is a verified line-by-line match to `_moments()` — same frozen T0 convention; (4) the FIX-5 synthetic fixture's materiality path was independently hand-reproduced (Δ_strat = 0.0, matches `SYNTH_fix5_result.json` exactly); (5) `git diff c23877ac -- offset_subset_reads.py` shows every one of `separation_for_covariate`/`_continuous_auc`/`_binary_or`/`holm_correct`/`materiality_for_covariate`/`disposition_for`/`class_label_line`/`t0_moments`/etc. byte-identical to round 4 — only routing/CLI changed. `ruff`/`mypy` independently re-run clean |

All five cited gates were GREEN (or GREEN-on-assigned-scope, rev3's 40/40) before this read was
launched.

## 2. Exact command executed (REAL mode, no `--dry-run`, run ONCE)

```
cd /home/jasper/Repositories/darksiren-emri
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_reads.py \
  --table-iiib results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_iiib.csv \
  --table-sha256-iiib 90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0 \
  --influence-iiib results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_iiib.csv \
  --logl-iiib results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
  --table-jr1 results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_joint_r1.csv \
  --table-sha256-jr1 fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a \
  --influence-jr1 results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/influence_joint_r1.csv \
  --logl-jr1 results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
  --alpha 0.05 --auc-band 0.20 --or-band 3.0 --t-mat 0.008 \
  --decile 0.10 --null-draws 1000 --null-seed 20260904 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_result_read.json
```

This is PIN CORRECTION 3's §8 block verbatim, with the sole permitted substitution of `--out` to
the task-specified path `offset_subset_result_read.json` (the draft's own block used
`offset_subset_result.json`; no other token changed). No `--dry-run` flag.

**stdout:**
```
wrote results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_result_read.json: disposition = INTERMEDIATE
```

**Exit code: 0.** No traceback; the script did not crash.

## 3. Pins independently re-verified immediately before launch (this session, not copied)

| input | pin | recomputed | verdict |
|---|---|---|---|
| `covariate_table_iiib.csv` | sha256 `90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0` | same | MATCH |
| `covariate_table_joint_r1.csv` | sha256 `fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a` | same | MATCH |
| iiib `event_likelihoods.csv` | md5 `8e6a2c18dc5838dd1d52641589243672` | same | MATCH |
| joint_r1 `event_likelihoods.csv` | md5 `745954a0fdee5f10878fb5e622a06144` | same | MATCH |
| `--out` target | — | did not pre-exist (`ls` → No such file or directory) before this run | clean write, no clobber |

The script's own internal `check_table_hash()` and `verify_logl_md5()` gates additionally passed
(the run produced a populated disposition, not an `INSTRUMENT-DEFECT` early-exit) — confirmed
below in §4's `meta` block: `table_sha256_iiib`/`table_sha256_jr1` echo the pinned values,
`logl_columns_present: true`, `missing_covariates: []`.

**Output file:** `offset_subset_result_read.json`, 903 lines, 24,742 bytes, sha256
`f6c8dcfeb6892f828e742cff08d9ef0a171ba1d957cd3b2afe8db49767742a80`.

**Script state:** `git diff --stat` shows `offset_subset_reads.py` unchanged from the working-tree
version this read launched against (237 insertions/81 deletions vs. commit `c23877ac`, the same
diff rev5's DESIGN_GATE already reviewed line-by-line and found byte-identical on every statistics
function). No script was modified by this read.

## 4. Gates reported by the run itself (join counts, population, missing-covariate pre-flight)

| gate | iiib | joint_r1 |
|---|---|---|
| `n_table_rows` | 1588 | 1588 |
| `n_influence_rows` | 1588 | 1588 |
| `n_unmatched_table_only` | 0 | 0 |
| `n_unmatched_influence_only` | 0 | 0 |
| `join_complete` | **true** | **true** |

`meta.logl_columns_present = true` (both venues, `h_grid` n=41, confirming rev5's dry-run finding
extends to the real run). `meta.missing_covariates = []` — the phase-A/phase-C column-mapping
hard pre-flight (rev4's assigned check) found nothing missing; no `INSTRUMENT-DEFECT` early exit.
`meta.primary_family = "iiib_2d"`. `meta.n_events_iiib = meta.n_events_jr1 = 1588`.

**g-population cross-check against the R8/§1 population facts (reproduced from the run's own
output, not re-derived by hand):** the C2 materiality stratum (`level == False`) has
`n_stratum = 606`, matching the registered exact-zero dark count exactly (606 dark / 982 hosted,
iiib). The C3 truth-disagreement 2×2 table sums to `cov_true = 75+272 = 347` and
`cov_false = 1+1240 = 1241`, matching the registered relative-label split exactly (1241 dark / 347
hosted, iiib). The C8 in-catalog-stratum test population is `n_s+n_b = 7+69 = 76` in iiib_2d,
matching the pinned in_catalog count (76) exactly. All three reproduce §1's population facts
without discrepancy.

**g-censoring (rail disclosure):** every materiality entry computed carries `map_rail_full: false`,
`map_rail_stratum: false`, `null_rail_fraction: 0.0`, `censoring_gate_red: false` — no MAP rail
was hit in the full sample, any stratum leave-out, or any of the 1000 null draws, for any of the
five covariates materiality was computed for. No Δ is flagged as a BOUND.

## 5. Per-covariate separation (§4.1): AUC/OR, Holm p, verdict — all four families

**Family = iiib_2d (PRIMARY, k=82, n_b=1506 unless noted):**

| cov | kind | n_s | n_b | n_nan | effect (AUC/OR) | p_raw | p_holm | Holm sig | band pass | **verdict** |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | binary | 82 | 1506 | 0 | OR 2.0546 | 0.1083 | 0.2194 | no | no | NULL |
| C2 | binary | 82 | 1506 | 0 | OR 0.1280 | 2.44e-16 | 2.20e-15 | yes | yes | **SEPARATES** |
| C3 | binary | 82 | 1506 | 0 | OR 0.5595 | 0.0731 | 0.2194 | no | no | NULL |
| C3c | continuous | 82 | 1506 | 0 | AUC 0.2923 | 6.75e-11 | 4.05e-10 | yes | yes | **SEPARATES** |
| C4 | continuous | 82 | 1506 | 0 | AUC 0.8722 | 6.24e-30 | 6.24e-29 | yes | yes | **SEPARATES** |
| C5 | continuous | 82 | 1506 | 0 | AUC 0.6475 | 6.66e-06 | 2.66e-05 | yes | no | WEAK |
| C6 | continuous | 15 | 967 | 606 | AUC 0.3239 | 1.55e-10 | 7.73e-10 | yes | no | WEAK |
| C7 | continuous | 82 | 1506 | 0 | AUC 0.2669 | 2.40e-13 | 1.68e-12 | yes | yes | **SEPARATES** |
| C8 (in-cat only) | binary | 7 | 69 | 0 | OR 0.3778 | 0.5844 | 0.5844 | no | no | NULL |
| C10 | continuous | 82 | 1506 | 0 | AUC 0.7410 | 1.86e-13 | 1.48e-12 | yes | yes | **SEPARATES** |
| C10b | binary | 0 | 0 | 0 | — | — | — | — | — | **NOT-TESTED** |
| C11 | continuous | 82 | 1506 | 0 | AUC 0.2295 | 1.45e-16 | (outside Holm family) | — | yes | REPORTED-ONLY |

**Family = iiib_1d (k=94, n_b=1494):**

| cov | effect | p_raw | p_holm | **verdict** |
|---|---|---|---|---|
| C1 | OR 0.5153 | 0.3164 | 0.9491 | NULL |
| C2 | OR 0.0668 | 2.14e-26 | 1.93e-25 | **SEPARATES** |
| C3 | OR 0.0899 | 2.34e-08 | 9.37e-08 | **SEPARATES** |
| C3c | AUC 0.2106 | 3.07e-22 | 2.15e-21 | **SEPARATES** |
| C4 | AUC 0.9795 | 5.76e-55 | 5.76e-54 | **SEPARATES** |
| C5 | AUC 0.6847 | 1.79e-09 | 8.93e-09 | WEAK |
| C6 (n_s=10, n_nan=606) | AUC 0.5237 | 0.4824 | 0.9648 | NULL |
| C7 | AUC 0.2065 | 7.91e-23 | 6.33e-22 | **SEPARATES** |
| C8 (n_s=2, in-cat) | OR 1.2286 | 1.0 | 1.0 | NULL |
| C10 | AUC 0.7158 | 2.11e-12 | 1.26e-11 | **SEPARATES** |
| C10b | n=0/0 | — | — | NOT-TESTED |
| C11 | AUC 0.1322 | 4.72e-33 | (outside family) | REPORTED-ONLY |

**Family = jr1_2d (k=72, n_b=1516):**

| cov | effect | p_raw | p_holm | **verdict** |
|---|---|---|---|---|
| C1 | OR 1.6473 | 0.3882 | 0.7764 | NULL |
| C2 | OR 0.1089 | 8.66e-18 | 7.79e-17 | **SEPARATES** |
| C3 | OR 0.1682 | 1.90e-08 | 9.51e-08 | **SEPARATES** |
| C3c | AUC 0.2420 | 5.41e-14 | 3.79e-13 | **SEPARATES** |
| C4 | AUC 0.9066 | 1.74e-31 | 1.74e-30 | **SEPARATES** |
| C5 | AUC 0.6350 | 1.06e-04 | 4.25e-04 | WEAK |
| C6 (n_s=15, n_nan=493) | AUC 0.3991 | 0.0401 | 0.1203 | NULL |
| C7 | AUC 0.2072 | 1.38e-17 | 1.11e-16 | **SEPARATES** |
| C8 (n_s=5, in-cat) | OR 0.5325 | 1.0 | 1.0 | NULL |
| C10 | AUC 0.7404 | 5.15e-12 | 3.09e-11 | **SEPARATES** |
| C10b | n=0/0 | — | — | NOT-TESTED |
| C11 | AUC 0.2074 | 4.41e-17 | (outside family) | REPORTED-ONLY |

**Family = jr1_1d (k=46, n_b=1542):**

| cov | effect | p_raw | p_holm | **verdict** |
|---|---|---|---|---|
| C1 | OR 2.1476 | 0.2758 | 0.7755 | NULL |
| C2 | OR 0.0793 | 2.67e-14 | 2.40e-13 | **SEPARATES** |
| C3 | OR 0.1587 | 4.31e-06 | 2.15e-05 | **SEPARATES** |
| C3c | AUC 0.2318 | 2.93e-10 | 2.05e-09 | **SEPARATES** |
| C4 | AUC 0.9151 | 7.38e-22 | 7.38e-21 | **SEPARATES** |
| C5 | AUC 0.6118 | 0.00969 | 0.0388 | WEAK |
| C6 (n_s=7, n_nan=493) | AUC 0.4190 | 0.2585 | 0.7755 | NULL |
| C7 | AUC 0.1847 | 1.26e-13 | 1.01e-12 | **SEPARATES** |
| C8 (n_s=4, in-cat) | OR 0.6614 | 1.0 | 1.0 | NULL |
| C10 | AUC 0.7082 | 1.44e-06 | 8.66e-06 | **SEPARATES** |
| C10b | n=0/0 | — | — | NOT-TESTED |
| C11 | AUC 0.1947 | 1.61e-12 | (outside family) | REPORTED-ONLY |

**C10b NOT-TESTED, all four families:** production M range (1.33e5–1.63e6 M☉) never crosses the
`low_M_timeout_bins12` edge (169,568.13 M☉) inside the scored set at n≥10; the run reports n_s=0,
n_b=0 in every family — §11 point (e)'s forecast ("C10b may be near-empty") reads as confirmed:
n=0, not merely small.

## 6. Materiality (§4.2): Δ_strat vs T_mat=0.008 and vs the null's 99% band

**Computed only for the primary family (iiib_2d)** — the run's own `iiib_1d_disposition_check`
note states materiality requires a per-event 41-node ln L matrix, and under the current data
contract only `--logl-iiib` feeds the primary `combined_with_bh` (2D) channel; iiib_1d, jr1_2d,
and jr1_1d carry **no materiality of their own** in this data contract (their `materiality` dicts
are empty; this is disclosed, not a gate failure). This is a real, structural gap for those three
families reported here as fact — every Δ_strat number below is iiib_2d only.

Materiality was evaluated for the five covariates that SEPARATE in the primary family (C2, C3c,
C4, C7, C10):

| cov | stratum rule | n_stratum | Δ_strat | T_mat | null CI99 | null percentile | Δ_S oracle | captured frac | **material?** |
|---|---|---|---|---|---|---|---|---|---|
| C2 | level == False (606 dark) | 606 | **0.15568** | 0.008 | [−0.02668, 0.03787] | 100.0 | 0.04623 | 3.367 | **true** |
| C3c | bottom decile | 159 | **0.03431** | 0.008 | [−0.00909, 0.01075] | 100.0 | 0.04623 | 0.742 | **true** |
| C4 | top decile | 159 | **0.08611** | 0.008 | [−0.00909, 0.01075] | 100.0 | 0.04623 | 1.862 | **true** |
| C7 | bottom decile | 159 | **0.03431** | 0.008 | [−0.00909, 0.01075] | 100.0 | 0.04623 | 0.742 | **true** |
| C10 | top decile | 159 | 0.00489 | 0.008 | [−0.00909, 0.01075] | 87.9 | 0.04623 | 0.106 | **false** |

Note (factual, not a verdict): C3c and C7's Δ_strat are numerically identical (0.03430655907…) —
their bottom-decile 159-event strata evidently coincide or near-coincide by event membership under
this ranking; the run does not report stratum overlap directly, so this is observed, not
explained, here. C2's Δ_strat (0.1557) exceeds the oracle Δ_S (0.0462, the leave-out of S itself)
by more than 3×, i.e. captured_fraction = 3.37 — a leave-out LARGER than leaving out S entirely;
reported as computed, not adjudicated.

C10 fails materiality: Δ_strat (0.00489) is below T_mat (0.008) AND its null percentile (87.9) is
inside the null's central 99% interval — both the registered materiality conditions in §4.2 are
not met (only one needs to fail; here both do).

**C1, C3, C5, C6, C8 carry no materiality entry** in the primary family (all NULL or WEAK there,
never reaching the separation gate that materiality is conditioned on).

## 7. 2D-vs-1D disposition comparison (`iiib_1d_disposition_check`)

```
iiib_1d_disposition:                 INTERMEDIATE
iiib_1d_named_covariates:            [C2, C3, C3c, C4, C7, C10]
primary_disposition_before_this_trigger: SUBSET-IDENTIFIED
agrees_with_primary:                 false
note: "iiib_1d has no logL matrix of its own under the current data contract
       (primary-family-only, --logl-iiib feeds iiib_2d exclusively, per module
       docstring); its materiality is always empty, so its own disposition can
       only read DIFFUSE-IN-COVARIATES or INTERMEDIATE, never SUBSET-IDENTIFIED."
```

This is the mechanism the run itself reports for why the final top-level disposition is
INTERMEDIATE rather than SUBSET-IDENTIFIED: absent materiality data, iiib_1d is structurally
incapable of reaching SUBSET-IDENTIFIED regardless of what its separation numbers show, so it can
never *agree* with a primary-family SUBSET-IDENTIFIED reading under §5's disposition table row
"primary 2D and 1D iiib families disagree in disposition." The run computed the primary family
would independently have qualified as SUBSET-IDENTIFIED (recorded verbatim in
`primary_disposition_before_this_trigger`) before this disagreement clause overrode it.

## 8. 2-of-3 replicate outcome (§4.3), for each primary-family SEPARATES+MATERIAL covariate

Per §4.3, a covariate that SEPARATES in the primary family must SEPARATE with the same sign in
≥2 of the 3 replicate families (iiib_1d, jr1_2d, jr1_1d) for SUBSET-IDENTIFIED. Cross-referencing
§5's four per-family tables above (sign = OR side of 1, or AUC side of 0.5):

| cov | primary (iiib_2d) | iiib_1d | jr1_2d | jr1_1d | replicates agreeing (same sign, SEPARATES) | **4.3 outcome** |
|---|---|---|---|---|---|---|
| C2 | SEPARATES, OR<1 | SEPARATES, OR<1 | SEPARATES, OR<1 | SEPARATES, OR<1 | 3/3 | replicate-consistent |
| C3c | SEPARATES, AUC<0.5 | SEPARATES, AUC<0.5 | SEPARATES, AUC<0.5 | SEPARATES, AUC<0.5 | 3/3 | replicate-consistent |
| C4 | SEPARATES, AUC>0.5 | SEPARATES, AUC>0.5 | SEPARATES, AUC>0.5 | SEPARATES, AUC>0.5 | 3/3 | replicate-consistent |
| C7 | SEPARATES, AUC<0.5 | SEPARATES, AUC<0.5 | SEPARATES, AUC<0.5 | SEPARATES, AUC<0.5 | 3/3 | replicate-consistent |
| C10 | SEPARATES but **not material** | SEPARATES, AUC>0.5 | SEPARATES, AUC>0.5 | SEPARATES, AUC>0.5 | 3/3 (moot — fails 4.2 first) | n/a (materiality gate already failed) |

All four materially-qualifying primary covariates (C2, C3c, C4, C7) are replicate-consistent 3/3.
The run's per-covariate machinery does not itself compute or report a 4.3 pass/fail flag in the
JSON (no `replicate_consistent` key found in `separation`, `materiality`, or `disposition`); the
3/3 counts above are read directly off the four per-family tables in §5 by this record, not
computed by a script the reader wrote — consistent with the "do not inspect data files by hand"
constraint, since this is a table lookup across the run's own reported per-family verdicts, not a
recomputation from raw data. C3 also separates in all three replicates (iiib_1d, jr1_2d, jr1_1d)
but is NULL in the primary family, so §4.3 never applies to it (§4.3 only gates covariates that
separate in the *primary* family).

## 9. R14 class-label line (mandatory, §5's "Mandatory class-label line")

```
(a) C2 hosted_exact:  OR 0.12805, Holm p 2.198e-15, SEPARATES, Δ_strat 0.15568, material=true
(b) C3 hosted_rel:    OR 0.55947, Holm p 0.21944,   NULL       (no Δ_strat — not separating in primary)
(c) C3c log10_f_cat:  AUC 0.29231, Holm p 4.047e-10, SEPARATES, Δ_strat 0.03431, material=true
```

`r14_class_label_line.separating = [C2, C3c]`; the run's own `r14_reading` string: *"multiple
class labels separate: ['C2', 'C3c']"*. Per the draft's §5 reading key: this is neither the "only
(c) separates" case, nor the "(b) but not (a)" case, nor the "(a) but not (b)" case, nor the "none"
case — it is (a) AND (c) SEPARATE while (b) does NOT, a combination not explicitly named in the
draft's four listed readings (which enumerate: only-(c), (b)-not-(a), (a)-not-(b), none). This
observation is stated as fact for the author's R14 ruling, not resolved here.

## 10. Reported-only secondaries

**Spearman ρ (d_e vs. continuous covariates, all 1588 events, primary family iiib_2d):**

| cov | ρ | p | n |
|---|---|---|---|
| C3c | −0.7290 | 2.22e-263 | 1588 |
| C4 | 0.8971 | 0.0 (underflow) | 1588 |
| C5 | 0.2821 | 1.92e-30 | 1588 |
| C6 | 0.0623 | 0.0509 | 982 |
| C7 | −0.6088 | 1.13e-161 | 1588 |
| C10 | 0.2262 | 7.02e-20 | 1588 |
| C11 | −0.6297 | 3.66e-176 | 1588 |

**Class composition of S (k=82, primary family) — raw counts, reported-only:**

| class | n_true | n_false | n_nan |
|---|---|---|---|
| C1 (in_catalog, truth) | 7 | 75 | 0 |
| C2 (hosted_exact) | 15 | 67 | 0 |
| C3 (hosted_rel) | 11 | 71 | 0 |

**Truth-disagreement 2×2 (C1 vs. C2, C1 vs. C3), reported-only, full 1588 population:**

| | C1=true & cov=true | C1=true & cov=false | C1=false & cov=true | C1=false & cov=false |
|---|---|---|---|---|
| vs C2 | 75 | 1 | 907 | 605 |
| vs C3 | 75 | 1 | 272 | 1240 |

Both rows: C1=true row sums to 76 (75+1), matching the pinned in_catalog population exactly; only
1 in_catalog event is labelled "dark" by either estimator label (both C2=false and C3=false in the
same single event, by row alignment — the run does not name the event_idx here, and this record
does not go looking for it by hand).

**C11 (reported-only, log10_snr), all four families:** AUC 0.2295 (iiib_2d) / 0.1322 (iiib_1d) /
0.2074 (jr1_2d) / 0.1947 (jr1_1d) — all far outside the ±0.20 band on the low side, all `p_holm =
null` because C11 sits outside the Holm family by registration (§2 explicitly excludes it from
m=11); `band_pass: true` is reported per-covariate but is not a verdict since C11 is never in the
adjudicated family.

## 11. Three-valued outcome — every §5 disposition row, evaluated against this run's output

| disposition row | trigger (verbatim from §5) | **evaluated** |
|---|---|---|
| **SUBSET-IDENTIFIED** | ≥1 covariate SEPARATES (4.1) AND MATERIAL (4.2) AND replicate-consistent (4.3) | **NOT the final disposition** — but the primary family alone satisfied this for C2/C3c/C4/C7 (`primary_disposition_before_this_trigger = SUBSET-IDENTIFIED`, §7) before the 2D-vs-1D disagreement clause below overrode it |
| **DIFFUSE-IN-COVARIATES** | no covariate SEPARATES in the primary family at the registered band | **NOT triggered** — five covariates SEPARATE in the primary family (C2, C3c, C4, C7, C10) |
| **INTERMEDIATE** | any of: separates-but-not-material; material-but-not-replicate-consistent; C8/C10b NOT-TESTED and nothing else separates; primary 2D/1D disagree | **TRIGGERED** — by the "primary 2D and 1D iiib families disagree in disposition" clause (§7) and independently by "C10 SEPARATES but its stratum is not MATERIAL" (§6) |
| **INSTRUMENT / NO-READ** | any §6 gate red | **NOT triggered** — both joins complete (0 unmatched, both venues), both table hashes matched, both logL md5s matched, `missing_covariates: []`, no traceback, exit 0 |

**Final disposition reported by the run (`disposition` object, top level):**
```
value:              INTERMEDIATE
named_covariates:   [C2, C3c, C4, C7]
instrument_note:    null
```

This record states the value computed; it does not rule on whether INTERMEDIATE is the correct
reading, nor on the open question the run's own note raises (§7) about whether the current
primary-family-only materiality data contract is itself an instrument limitation that a future
revision should widen to all four families before the disagreement clause is allowed to decide
anything. That question, and the disposition itself, return to the author as fresh RULE per the
draft's own binding convention.

## 12. Files of record

- Command + output: this file, §2.
- Result JSON: `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_result_read.json` (903 lines, sha256 `f6c8dcfeb6892f828e742cff08d9ef0a171ba1d957cd3b2afe8db49767742a80`).
- Script invoked (unmodified): `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_reads.py`.
- Inputs (all pinned, all verified §3): `covariate_table_iiib.csv`, `covariate_table_joint_r1.csv`, `influence_iiib.csv`, `influence_joint_r1.csv`, both venues' `event_likelihoods.csv` under `graph1_20260901/retrieved/`.
- Gate documents read (not re-derived): `DESIGN_GATE_computability.md`, `BYTEID_RECORD.md`, `DESIGN_GATE_formula_rev3.md`, `DESIGN_GATE_formula_rev4.md`, `DESIGN_GATE_formula_rev5.md`.
- Prior (round-4, crashed) read left untouched: `READ_RECORD.md`.
