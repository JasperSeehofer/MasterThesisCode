# DESIGN_GATE_rev3_computability.md — r-completion-residual, REGISTRATION_DRAFT.md (post-REVISION 1)

Reviewer: fresh computability + formula-match pass (sonnet). Did **not** open any earlier
`DESIGN_GATE_*.md` for this arm (design, provenance, stats, rev1, rev2) or `INFORMATION_FORECAST.md`.
Did not run real mode, did not run `--dry-run` myself, and computed no statistic over the registered
population — the only data reads below are file existence, CSV/JSON headers, ≤5-row spot checks, and
independent hand arithmetic on numbers BUILD_RECORD.md already discloses. Sources read: the full
`REGISTRATION_DRAFT.md` (all sections incl. REVISION 1), `completion_residual_reads.py` (full),
`BUILD_RECORD.md` (FIX 3 section + surrounding context), `MUSTFIX_REVISION1_20260903.md`,
`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` (kill-criterion line only), and light header/structure probes
of the production/replicate/harness CSVs and one harness checkpoint JSON.

## Verdict: **RED**

Driven by finding F1 (a registered NO-READ-gating invariant has no code path at all — the read can
bank a disposition under exactly the condition §5 says must STOP it). Everything else the draft
registers is correctly wired, byte-id and closure gates genuinely discriminate defects (not rubber
stamps), and the hand arithmetic on FIX 3's synthetic checks matches the code to the precision
BUILD_RECORD.md quotes. This is a repair-and-resubmit RED, not a redesign RED.

---

## 1. Inputs exist (headers/keys, ≤5 rows)

| input | check | result |
|---|---|---|
| production CSV (`…/run_20260902_graph1_headrebaseline_iiib/…/event_likelihoods.csv`) | file exists, header row | GREEN — 18.4 MB, header has `event_idx,h,…,D_tilde_phi,…,B_num,…,den_log_term,num_log_term_no_bh,…` (all 5 columns §2.1 needs present) |
| production CRB (`seed61000/prepared_cramer_rao_bounds.csv`) | file exists, header row | GREEN — has `host_galaxy_index`, `in_catalog` |
| replicate CSV (`…joint_r1/…/event_likelihoods.csv`) | file exists, header row | GREEN — identical header to production CSV |
| harness root (`tree2_20260830/b8_cal_harness_work_s4_postflip`) | `universe_seed*_S.json` count | GREEN — exactly 67 files |
| one harness checkpoint (`universe_seed901000_S.json`) | keys present | GREEN — `universe.{seed,n_draw_requested}`, `resolved_flags` (13 keys, matches the draft's "13 tokens"), `score_at_truth.no_bh.{dark,catalogue_hosted,all}.mean` all present; `score_at_truth.no_bh.dark.mean = -0.03684332680707096`, byte-identical to `BUILDER_DARK_MEANS[0]` in `byteid_check.py` and to the first value BUILD_RECORD.md quotes |
| that universe's own CSV/CRB (`seed901000_S/simulations/{diagnostics/event_likelihoods.csv,prepared_cramer_rao_bounds.csv}`) | exist, header, row count | GREEN — 41 h-nodes × 176 events = 7216 rows; CRB 200 rows, `host_galaxy_index` present, split 188/12 matches the checkpoint's own `n_catalogue_hosted: 12` |

No missing input, no header/key mismatch. §2.1's "operational source" columns (`D_tilde_phi`,
`alpha_G_phi`) and the stencil nodes 0.725/0.735 are present in the production and the sampled
harness CSV's `h` grid.

## 2. Every named statistic has a code path matching the draft's formula and thresholds

| statistic | draft formula (§2.1/§2.4) | code | match |
|---|---|---|---|
| s_M,e | Δln B_num/Δh − Δln β̄_Ḡ^φ/Δh | `compute_event_terms` lines 318-320 | exact |
| s_T | Δln β̄_Ḡ^φ/Δh − Δden_log_term/Δh | lines 321-324 | exact |
| s_C,e | Δ[num_log_term_no_bh − ln B_num]/Δh | lines 325-327 | algebraically identical (term-grouped differently, same value) |
| s_e (full) | Δnum_log_term_no_bh/Δh − Δden_log_term/Δh | lines 328-334 | exact |
| g-closure | max\|s_M+s_T+s_C−s_e\| ≤ 1e-9·(\|s_e\|+1) | `check_gclosure` | exact, incl. the relative-to-\|s_e\| tolerance form |
| class closure | S_all = π_G·S_G + π_Ḡ·S_dark | `check_class_closure` | exact; correctly localises to an index/class defect (verified degenerate-empty-class red in FIX 3's synthetic check) |
| T_prod | mean_e s_M,e over dark, production | `t_prod = dark_terms["s_M"].mean()` | exact |
| SE_prod | SD_e(s_M,e\|dark,prod)/√N_Ḡ | `dark_terms["s_M"].std(ddof=1)/√len` | exact (production-sourced per-event SD, not the harness-borrowed proxy — REVISION 1 item 1 confirmed landed) |
| Z_prod | T_prod/SE_prod | `t_prod/se_prod` | exact |
| T_harn | mean over 67 universes of per-universe S_M,harn,U (own recompute from each universe's raw CSV, not the checkpoint) | `compute_harness_matched_channel_scores` | exact — confirmed it re-reads each universe's own `event_likelihoods.csv`/CRB and applies the same `compute_event_terms`, not the checkpoint's cached full-score mean (REVISION 1 item 2 + 1b landed correctly) |
| SE_harn | SD_U(S_M,harn,U)/√67 | `std(values, ddof=1)/√n` | exact, gated to n=67 via `g_harness_universes` (§4 below) |
| Z_harn | T_harn/SE_harn | exact | exact |
| ρ | T_harn/T_prod, only when \|Z_prod\|>3 | `rho = t_harn/t_prod if (abs(z_prod)>3 and t_prod) else None` | exact — the `and t_prod` clause is dead-code-safe (t_prod==0 ⇒ z_prod==0 ⇒ already excluded by the \|Z_prod\|>3 test), not a second undocumented branch |
| δh_M | N_Ḡ·T_prod/I_1D, I_1D=1/0.017526² | `compute_delta_h_m` | exact — **hand-verified**, see §4 below; correctly flagged `reported_only: True, verdict_bearing: False` and excluded from the disposition chain |
| S_all (class closure, real-mode) | same identity | `check_class_closure(terms, production_crb)` re-run in `compute_registered_statistics`, written to `"class_closure"` | exact |

Every statistic named in §2.4's table has a code path with the registered formula. No formula
mismatch found in the arithmetic itself.

**Gap (not a formula mismatch, a missing disclosure — F3 below):** the g-censoring rule (§5) says
any h-space quote — explicitly naming δh_M — "MUST carry the S3 rail fraction … a quote without the
disclosure is void." `compute_delta_h_m`'s returned dict (`N_Gbar, sigma_h_1D, I_1D, delta_h_M,
reported_only, verdict_bearing`) carries no rail-fraction field at all. The formula is right; the
mandatory disclosure the draft attaches to that formula's output is absent from the code path.

## 3. Disposition rows map 1:1 to code branches

The `elif` chain in `compute_registered_statistics` (lines 836-852) is a 6-way match against §4's
6 rows, in the same order, including REVISION 1b's added INTERMEDIATE (d):

1. ILLEGITIMATE: `|Z_harn|>3 and rho is not None and rho>=0.5` ↔ table row 1 — exact
2. FLOOR-CONSISTENT: `|Z_harn|<=3 and |Z_prod|<=3` ↔ row 2 — exact
3. INTERMEDIATE (a): `|Z_harn|<=3 and |Z_prod|>3` ↔ row 3 — exact
4. INTERMEDIATE (b): `|Z_harn|>3 and 0.2<rho<0.5` ↔ row 4 — exact
5. INTERMEDIATE (c): `|Z_harn|>3 and rho<=0.2` ↔ row 5 — exact
6. INTERMEDIATE (d): `|Z_harn|>3 and rho is None` ↔ row 6 (rev. 1b) — exact
7. else: `AssertionError` (defensive, "unreachable by construction") — the three ρ-branches
   (`>=0.5`, `(0.2,0.5)`, `<=0.2`) are an exhaustive, non-overlapping partition of the reals given
   `rho is not None`, so this is genuinely unreachable, not a silently-swallowed case.

One pre-existing ambiguity that is the **draft's**, not the code's: ρ<0 (T_harn, T_prod opposite
sign) falls into `rho<=0.2` → INTERMEDIATE (c) "minor-illegitimate," even though §3's own band
derivation ("un-owned remainder ≥ 4× the owned part") implicitly assumes same-sign ρ∈[0,1]. The code
applies the registered numeric threshold literally and exhaustively; if a negative ρ is physically
reachable this is worth an author note on the band's own scope, not a code defect.

NO-READ (§4's 6th row) is implemented as "refuse to bank a disposition, return the gate table
instead" exactly as specified (`compute_registered_statistics` lines 783-790), gated on the same
`collect_gate_report` dry-run and real mode both call (REVISION 2 item 4, confirmed landed).

## 4. FIX 3 synthetic check — hand arithmetic (BUILD_RECORD.md, "FIX 3 … Synthetic-table check")

**δh_M**, inputs `t_prod=-0.114` (arbitrary), `n_dark=1512`:

    I_1D = 1/0.017526² : 17526² = 307,160,676 ⇒ 0.017526² = 3.07160676e-4
           1/3.07160676e-4 = 3255.625079…  (matches code's 3255.6250787779877,
           and rounds to the draft's quoted "3256")
    δh_M = 1512 × (-0.114) / 3255.6250787779877
         = -172.368 / 3255.6250787779877
         = -0.0529446714…   (matches code's -0.052944671400768 to the last printed digit)

Hand arithmetic matches BUILD_RECORD.md's quoted output exactly.

**Class closure**, 5 fabricated events (3 dark, 2 catalogue-hosted; π_G=2/5=0.4, π_Ḡ=3/5=0.6),
BUILD_RECORD.md quotes `S_G=-0.50252, S_dark=1.61300, S_all=0.76679, reconstructed_S_all=0.76679`:

    0.4×(-0.50252) + 0.6×1.61300 = -0.201008 + 0.9678 = 0.766792 ≈ 0.76679 (displayed precision)

Reconstruction identity confirmed by hand from the disclosed S_G/S_dark/π values (the raw 5-row
table itself was not committed — scratch-only per BUILD_RECORD — so this is a consistency check on
the quoted aggregates, not a re-run of the fabricated rows; that is the strongest independent check
available without re-fabricating data myself, which would exceed the ≤5-row/synthetic-only bound
only if it touched the registered population, and this does not).

**Per-universe harness gate**, 2 fabricated "universes": BUILD_RECORD.md reports
`all_universes_gclosure_gznorm_green = False` driven only by `count_matches_expected` (2≠67) when
both are clean, and correctly flips to `False` on `gznorm_green` after one event/h-node is corrupted
in universe 901001. This is a boolean-logic check, not arithmetic; it demonstrates the gate is not a
tautological pass (a real defect is shown to flip it), consistent with `check_harness_universe_gates`'s
actual code (per-universe `universe_green = gclosure_green and gznorm_green`, aggregate requires both
that AND the population count).

All FIX 3 synthetic numbers reproduce by hand to the precision BUILD_RECORD.md quotes.

## 5. Gates: g-population, g-znorm, g-closure, g-byte-id, g-censoring/rail disclosure

- **g-population** — production: `check_production_population` verifies the JOIN gate (`event_idx`
  present = full range minus exactly `{1203,1356}`), `in_catalogue==76`, `dark==1512`; run on
  *both* production and replicate CSVs against the same CRB, matching §2.2's "same script" replicate
  clause. GREEN path confirmed structurally (headers/keys checked in §1); not re-derived as a full
  aggregate here per the reviewer constraint.
- **g-znorm** — `check_gznorm` (exact-equality `nunique==1` per h-node on `den_log_term`) is run on
  production, replicate, **and every harness universe** (`check_harness_universe_gates`), matching
  §5's explicit "in both venues and in every harness universe" scope and REVISION 1 item 3's
  exact-equality tolerance (no epsilon smuggled in).
- **g-closure** — the per-event identity and the class-closure identity are both run on production,
  replicate, **and every harness universe** (REVISION 2 item 2, confirmed landed — `g_closure_replicate`
  and `g_harness_universes` both feed `gates_green`, not just the production-only pre-rev.-2 scope).
- **g-byte-id** — `byte_id_count_green` (67/67 checkpoints matched to `--population`) feeds
  `gates_green`; the true byte-for-bit anchor comparison (against `BUILDER_DARK_MEANS` /
  `BYTEID_RECORD.md`) lives in the separate, non-imported `byteid_check.py`, matching §7's own
  framing ("Launch waits on the byte-id gate GREEN … stamped GREEN in BYTEID_RECORD.md" — a
  launch-time precondition, not something `completion_residual_reads.py` re-derives on every run).
  Both scripts' "reproduction" is a re-read-and-re-aggregate of the checkpoint's own stored
  `score_at_truth.no_bh.dark.mean`, not an independent re-derivation of that mean from the raw
  per-event CSV — this is explicitly disclosed in `reproduce_harness_byte_id`'s docstring, and the
  draft's own §2.3 wording ("same convention, same machine") is plausibly read as a
  reproducibility/integrity check rather than an independent-derivation check. Flagged as F2 below
  for the author to confirm intent; not scored as a formula mismatch given the disclosure.
- **g-censoring/rail disclosure** — the "grid-interior, no rail exposure" half needs no code (it is
  a property of using the fixed 0.725/0.735 stencil, not a per-run check). The "any h-space quote…
  MUST carry the S3 rail fraction" half has **no code path at all**: no field in the script computes
  or stores the S3 rail fractions (10/67, 14/67, MAP 0.665) anywhere, and `delta_h_M`'s output carries
  no such field (F3 below).
- **Real mode refuses to bank on red** — confirmed: `compute_registered_statistics` calls
  `collect_gate_report` first and returns `{"disposition": "NO-READ", ...}` without computing
  `T_prod`/`T_harn`/ρ/δh_M when `gates["gates_green"]` is `False` (lines 783-790).

### F1 [RED] — the harness↔production resolved-flags equality assertion has no code path

§5 Invariants: "harness commit `7e9e1e27` with 1112 dirty paths … **NEVER audited against the
production commit** (conditional-on: the harness's resolved 13 flags equal production's CoR-P CLI;
**asserted from the checkpoint resolved_flags block by the script**, else NO-READ)."

This is registered as a script-performed, NO-READ-gating assertion — the substitute for a real
code-identity check between the two commits. In `completion_residual_reads.py`:

- `reproduce_harness_byte_id` computes `resolved_flags_internally_consistent` (line 179) — but this
  only checks the 67 harness checkpoints agree **with each other**, never against any
  production-side reference (there is no `--production-resolved-flags` CLI argument, no file read of
  a production CoR-P CLI record, no comparison of any kind against production).
- Even that weaker internal-consistency boolean is **never read again**: it does not appear in
  `collect_gate_report`'s `gates_green` computation (only `byte_id_count_green`, a checkpoint-count
  check, feeds `byte_id_green`) and does not appear in the `triggers` list. Confirmed by direct grep
  — `resolved_flags_internally_consistent` occurs only where it is computed (lines 179-180); it is
  dead output.

Consequence: the one invariant the draft explicitly flags as unauditable-any-other-way ("NEVER
audited against the production commit") and explicitly promises a runtime substitute for, has zero
runtime enforcement. A harness run under CLI flags that do NOT match production's CoR-P configuration
would sail through every gate in this script GREEN. This is not a "wrong number" defect (T_prod/T_harn
are computed correctly *given* matching conventions) — it is exactly the "unregistered" failure mode
the task brief calls out: a named NO-READ trigger that cannot fire.

**This is the sole finding driving the RED verdict.** Fix: either (a) add a CLI input carrying
production's CoR-P CLI resolved-flags (or a path to a file recording them) and compare it to the
harness `resolved_flags` block, red on mismatch, folded into `gates_green`/`NO_READ.triggers`; or (b)
if no such reference currently exists anywhere in the repo (plausible — production doesn't run
through `b8_cal_harness.py`'s resolved-flags machinery), the draft's §5 invariant line needs to be
rewritten to say what actually is checked (internal-consistency-only, wire it into `gates_green`) and
what is **not** checked (cross-venue equality), rather than promising an assertion the code cannot
make. Either fix is small; this does not require a redesign.

### F2 [note, not scored] — byte-id "reproduction" is re-read+re-aggregate, not independent re-derivation

Documented above under g-byte-id. Worth a one-line author confirmation of intent in the next
revision (is "same convention, same machine" meant as reproducibility-of-the-stored-value, or as
independent re-derivation from raw per-event data — the latter is technically available, since
`check_harness_universe_gates` already computes `s_e` per dark event per universe and simply never
compares its mean to the checkpoint's `score_at_truth.no_bh.dark.mean`). Not scored RED: it is
disclosed in-code, does not corrupt any statistic in the disposition chain (T_harn is independently
the matched-channel score, computed from raw per-universe data by a different function entirely), and
a plausible reading of the draft's own wording supports the as-built interpretation.

### F3 [AMBER] — δh_M's mandatory rail-fraction disclosure is not attached anywhere in code

Documented above under §2 and g-censoring. `compute_delta_h_m` returns no rail-fraction fields; no
function anywhere in the script computes the S3 rail fractions (10/67 no_bh, 14/67 with_bh, MAP
0.665). Per the draft's own rule ("a quote without the disclosure is void"), the JSON's `delta_h_M`
value is not citable as-is. Does not affect the disposition (δh_M is `verdict_bearing: False` and
excluded from the `elif` chain, confirmed). Fix: add the four disclosed numbers as literal constants
(they are static per-campaign facts already quoted in the draft, not something requiring a new
compute path) into `compute_delta_h_m`'s or the top-level output's record.

### F4 [AMBER, minor] — g-precision cross-check is dead code

`check_gprecision` (lines 507-534) is defined and matches the draft's §2.1 formula (1e-3 relative
tolerance against a full-precision `selection_tables_h_*.json` where one exists) but is **never
called** anywhere — not in `collect_gate_report`, not in `compute_registered_statistics`, not in
`run_dry_run`. This is consistent with the draft (§4's NO-READ trigger list does not include
g-precision — it is disclosure-only, "else disclose and use the column definition"), so it does not
threaten `gates_green`. But the disclosure itself never happens: nothing in the output record states
whether the 7-s.f.-derived `β̄_Ḡ^φ` and a full-precision source agree, which §5 registers as an
obligation ("the two must agree to 1e-3 relative, **else disclose**"). Cheap fix: call it once per
venue inside `collect_gate_report` and fold its `nodes` dict into the report (informational, no gate
change needed).

### F5 [AMBER, cosmetic] — `T0_MEAN_H_TOLERANCE` constant is a vestigial mislabel

The module constant `T0_MEAN_H_TOLERANCE = 1.0e-9` (line 62) is written verbatim into the JSON
report's `"tolerance"` field (line 624), but the actual pass/fail logic
(`round(mean_h, 6) == T0_MEAN_H_TARGET_IIIB_1D`, line 616) does not use this constant at all — it
applies a 6-dp rounding comparison, i.e. an effective ~5e-7 tolerance. This is the **correct**
choice per REVISION 1 item 6 / MUSTFIX item 6 (which explicitly offered "state the tolerance as 1e-6
on the 6-dp display" as one of two acceptable resolutions, and the withdrawn-literal-1e-9 language in
REGISTRATION_DRAFT.md §5 is about a *different*, never-implemented full-precision-anchor-at-1e-12
path that the draft also mentions but does not mandate this script perform — MUSTFIX's "OR" framing
makes the as-built choice legitimate). The only defect is that the emitted `"tolerance"` field lies
about which tolerance was actually applied — a downstream reader of the JSON alone (not the source)
would be misled. Fix: either delete the unused constant or set it to `5e-7` / rename the reported
field to make clear it is a display-rounding bound, not a literal abs-diff threshold.

## 6. Launch block: CLI, zero fresh choices

§7's CLI (`--production-csv --production-crb --replicate-csv --harness-root --population 200
--h-lo 0.725 --h-hi 0.735 --h-true 0.73 --crb-md5 … --catalogue-md5 … --out … [--dry-run]`) matches
`build_parser()` (lines 887-901) flag-for-flag: 11 named flags + `--dry-run`, no extra, none missing,
required/optional status identical (only `--replicate-csv` and `--dry-run` are optional in both).
Every value in the launch block (population 200, h-lo/hi/true, both md5s) is a constant already fixed
earlier in the draft (§2.2/§2.3/§5 Invariants) — no value is chosen at launch time. Confirms REVISION
1's own footer claim ("matches the built `completion_residual_reads.py` argparse exactly").

The optional cell-R block (§7, `b8_cal_harness.py`) is a pre-existing script outside this node's
build scope (`b-completion-scorer` built only `completion_residual_reads.py` per BUILD_RECORD.md) —
not re-verified here; flagged for whoever reviews `b8_cal_harness.py`'s own interface if that has not
already happened elsewhere in the graph.

## 7. Kill criterion, max_revisions, blindness line

- **Kill criterion**: `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` row for `q-completion-residual`,
  read directly: `"registered arm fails to discriminate at its registered band after revision 2 ->
  park bounded-undetermined with the measured bound"` — **verbatim match** to REGISTRATION_DRAFT.md
  §8b's quote, character-for-character.
- **max_revisions**: header states "max_revisions 2 (ORCHESTRATOR-DERIVED, charter §1.7/§1.13)";
  §8b's revision-counter language ("This draft is revision 1; a NO-READ or an INTERMEDIATE (b) …
  consumes revision 2; a third failure parks…") is internally consistent with that cap.
- **Blindness line**: §8b's line matches MUSTFIX_REVISION1_20260903.md's required "Both" text
  verbatim through "…has not opened that record," plus one added sentence ("The revising author did
  not open it.") which is consistent with, not contradicting, the required line.

---

## Summary for the author

One repair item before this can go GREEN: **F1** — wire the harness-vs-production resolved-flags
equality (or its documented absence) into `gates_green`/`NO_READ`, since the draft names it as a
NO-READ trigger that currently cannot fire. Two disclosure gaps worth folding into the same repair
pass (**F3** rail-fraction on δh_M, **F4** g-precision cross-check never called) — neither is
verdict-bearing, both are obligations the draft's own §5 states. One cosmetic mislabel (**F5**). One
open design question for the author's own confirmation, not a defect (**F2**, byte-id's re-read vs.
re-derive semantics). Every registered statistic's formula, every disposition branch, the kill
criterion, the launch CLI, and the FIX 3 hand arithmetic all check out clean.
