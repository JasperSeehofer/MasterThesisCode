# DESIGN_GATE_rev4_computability.md — r-completion-residual, REGISTRATION_DRAFT.md (post-FIX 4)

FRESH reviewer. Read: REGISTRATION_DRAFT.md (all sections, incl. REVISION 1), `completion_residual_reads.py`
(full, 1135 lines), `BUILD_RECORD.md` FIX 4 (§602-750), `BYTEID_RECORD.md`, `byteid_check.py` (header/imports
only), `DESIGN_GATE_rev3_computability.md` **F1–F4 only** (lines 160-228; no other DESIGN_GATE_*.md opened,
`DESIGN_GATE_stats.md`/`INFORMATION_FORECAST.md` not opened). Cross-checked `darksiren_emri/validation/
correspondence_1d.py` for `REGISTERED_RESOLVED_FLAGS`/`_RESOLVED_FLAG_ATTRS`/`combine_log_likelihood`, and
`results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` for the T0 gradient-trapezoid convention (both
read-only, not edited, not run in real mode). Did NOT open `DESIGN_GATE_design.md`, `_provenance.md`, `_rev1`,
`_rev2` bodies, `DESIGN_GATE_stats.md`, or `INFORMATION_FORECAST.md`.

**Discipline:** no real mode run, no aggregate computed over a registered population. All hand-verification
below used either (a) file existence/header/single-row structural reads, (b) an md5 checksum (provenance, not
a scientific aggregate), or (c) synthetic ≤5-checkpoint fabricated tables passed directly to the script's own
functions (`check_resolved_flags`, `compute_rail_fraction_disclosure`) via `uv run python`. No per-event score,
no `T_prod`/`T_harn`/`Z`/`ρ`/disposition was computed on real data by this review.

## Verdict: **GREEN**

F1 (rev. 3's sole RED) is closed and independently confirmed. F3/F4 (rev. 3 AMBER) are closed. Two new
AMBER-level notes are raised below (§5, §6) — neither corrupts the registered read on the actual launch
inputs; both are documented explicitly per the review brief's "list any remaining draft-named gate with no
code path" instruction, not scored RED, because in each case the statistic computations are already
protected by population-filtering applied everywhere they run, so the described construction physically
cannot silently change `T_prod`/`T_harn`/`Z`/`ρ`/the disposition. F5 (`T0_MEAN_H_TOLERANCE` mislabel,
rev. 3, not in this fix round's scope) remains open, cosmetic, not blocking.

## 1. Inputs exist (≤5 rows / headers only)

| input | check | result |
|---|---|---|
| `--production-csv` (iiib) | file exists, header read | EXISTS; header carries `event_idx,h,...,B_num,...,den_log_term,num_log_term_no_bh,...,combined_no_bh` — all columns `compute_event_terms`/`t0_mean_h` require are present |
| `--production-crb` (`seed61000/prepared_cramer_rao_bounds.csv`) | file exists, `host_galaxy_index` column present, md5 | EXISTS; column present at position 125; `md5sum` = `9a1f2a14384a9281c97ca3be312ddaab` — **matches** the draft's `--crb-md5` and `BUILD_RECORD.md`'s launch block exactly |
| `--replicate-csv` (joint_r1) | file exists, header read | EXISTS; identical column set to production |
| `--harness-root` (`b8_cal_harness_work_s4_postflip`) | checkpoint count | `ls | grep -c universe_seed.*_S.json` = **67**, matching `N_HARNESS_UNIVERSES` |
| one harness checkpoint (`universe_seed901000_S.json`), structural read | top-level + nested keys | `universe.{seed,n_draw_requested}`, `resolved_flags` (13 keys, matching `RESOLVED_FLAG_NAMES`), `score_at_truth.no_bh.dark.{n,mean,sem}`, `posterior.{no_bh,with_bh}.map_h`, `grid.h_bounds` — every field the script reads (`c["universe"]["n_draw_requested"]`, `c["resolved_flags"]`, `c["score_at_truth"]["no_bh"]["dark"]["mean"]`, `c["posterior"][channel]["map_h"]`, `c["grid"]["h_bounds"]`) is present under the exact accessed path |
| one harness universe's per-universe CSVs (`seed901000_S/simulations/{diagnostics/event_likelihoods.csv,prepared_cramer_rao_bounds.csv}`) | line counts | 7217 lines (7216 rows = 176×41, matches draft) / 201 lines (200 rows, matches `--population 200`) |

No missing field, no missing file, on any input the launch block names.

## 2. Every named statistic has a code path matching the draft's formula/thresholds

| statistic | draft formula (§2.1/§2.4) | code | verdict |
|---|---|---|---|
| `s_M,e` | `Δln B_num/Δh − Δln β̄_Ḡ^φ/Δh`, `β̄_Ḡ^φ = D_tilde_phi − alpha_G_phi` | `compute_event_terms`: `s_M = (ln_b_num[1]-ln_b_num[0])/dh - (ln_beta_gbar_phi[1]-ln_beta_gbar_phi[0])/dh`, `beta_gbar_phi = d_tilde_phi - alpha_g_phi` | match |
| `s_T`, `s_C,e`, `s_e` | §2.1 table | same function, same secant-on-Δh pattern on `den_log_term`/`num_log_term_no_bh` | match |
| g-closure | `max_e |s_M+s_T+s_C−s_e| ≤ 1e-9·(|s_e|+1)` | `check_gclosure`: `tol = 1e-9*(|s_e|+1)`, per-row `<=` | match, `GCLOSURE_TOLERANCE=1.0e-9` |
| class closure `S_all=π_G S_G+π_Ḡ S_dark` | §2.1 | `check_class_closure`: `pi_g*s_g+pi_gbar*s_dark` vs `s_all`, same `1e-9` tol scaled | match |
| `T_prod`, `SE_prod` | `S_M,prod` (dark mean); `SE_prod=SD_e(s_M,e|dark,prod)/√N_Ḡ` | `t_prod = dark_terms["s_M"].mean()`; `se_prod = dark_terms["s_M"].std(ddof=1)/√len(dark_terms)` | match (sample SD, ddof=1, consistent with `SE_harn`'s convention) |
| `Z_prod` | `T_prod/SE_prod` | `z_prod = t_prod/se_prod` | match |
| `T_harn` (per-universe matched-channel) | mean over universes of `S_M,harn,U`, each computed from that universe's OWN diagnostics CSV, dark class only | `compute_harness_matched_channel_scores`: per matched seed, reads `seed{seed}_S/simulations/{diagnostics/event_likelihoods.csv,prepared_cramer_rao_bounds.csv}`, calls `compute_event_terms` (same function as production), masks dark, means `s_M`; `T_harn = mean` of the 67 (or fewer) `S_M_universe` values | match; explicitly does NOT reuse the checkpoint's full-score `dark.mean` (rev. 1 item 2/1b honored — confirmed by reading the function body, not just its docstring) |
| `SE_harn` | `SD_U(S_M,harn,U)/√67` | `se_harn = std(values, ddof=1)/√n` | match |
| `Z_harn` | `T_harn/SE_harn` | `z_harn = t_harn/se_harn` | match |
| `ρ` | `T_harn/T_prod`, evaluated only when `|Z_prod|>3` | `rho = t_harn/t_prod if (abs(z_prod)>Z_BAND and t_prod) else None` | match (the extra `and t_prod` guard is a no-op given `t_prod==0 ⇒ z_prod==0 ⇒ |z_prod|>3` is already False — dead defensive code, not a formula deviation) |
| `δh_M` | `N_Ḡ·T_prod/I_1D`, `I_1D=1/0.017526²=3256` | `compute_delta_h_m`: `i_1d = 1/SIGMA_H_1D_REBASELINE_IIIB**2` (`=0.017526`), `delta_h_m = n_dark*t_prod/i_1d`; `0.017526**2` inverted ≈ 3255.9 → rounds to the draft's quoted 3256 | match; `reported_only=True`, `verdict_bearing=False`, excluded from the disposition `elif` chain — confirmed by reading the chain (§4 below), not just the flag |
| `S_all` (registered gate quantity, not a §2.4 disposition input) | §2.1 identity | `check_class_closure` returns `S_all` alongside `S_G`/`S_dark`/`pi_G`/`pi_Gbar` | match |
| g-znorm | `max_e |den_log_term(e,h)−den_log_term(0,h)| = 0` exactly, per h-node, both venues + every harness universe | `check_gznorm`: `groupby("h")["den_log_term"].nunique()==1` — equivalent formulation (uniformity, not reference-to-row-0), functionally identical; called on production, replicate, and every harness universe CSV (via `check_harness_universe_gates`) | match |
| g-byte-id | 67/67 checkpoint `score_at_truth.no_bh.dark.mean` reproduced; T0 anchor `mean_h=0.666987` (6dp) / `0.6669869414473403` (full precision, 1e-12 tol) | `reproduce_harness_byte_id` (count+values); `t0_mean_h` (gradient-trapezoid `combine_log_likelihood(...,"physics_floor")`, `w=np.gradient(h_grid)`) — **independently cross-checked** against `tier0_bootstrap_jackknife.py`'s own docstring formula (`_moments`: `mean_h=(post_n*h_grid*weights).sum()`, same `_physics_floor_apply` zero-handling) and against `BYTEID_RECORD.md`'s independently-re-derived `0.6669869414473403` (exact digit match) | match |
| resolved-flags equality (FIX 4/F1) | harness's 13 resolved flags == production's CoR-P CLI | `check_resolved_flags` vs `REGISTERED_RESOLVED_FLAGS` (`correspondence_1d.py`); `RESOLVED_FLAG_NAMES` (script) asserted `== REGISTERED_RESOLVED_FLAGS.keys()` at import time; cross-checked against `_RESOLVED_FLAG_ATTRS` in `correspondence_1d.py:3117-3129` (13 tokens, underscore-stripped) — **identical set** | match, confirmed independently at the source-file level, not just by the assertion's own self-report |
| rail-fraction disclosure (FIX 4/F3) | S3 rail fraction, both channels, disclosed with `delta_h_M` | `compute_rail_fraction_disclosure`, merged into `delta_h_m["rail_fraction_disclosure"]` | match (see §6 for a narrative-vs-numbers inconsistency found in `BUILD_RECORD.md`'s own synthetic-check writeup — code confirmed correct independently) |
| g-precision (FIX 4/F4) | CSV `β̄_Ḡ^φ` vs full-precision `selection_tables_h_*.json`, 1e-3 tol, disclosure-only | `check_gprecision`, called once per venue in `collect_gate_report`, stored as `g_precision_production`/`_replicate`, confirmed NOT in `gates_green`/`triggers` | match |

## 3. Disposition rows ↔ code branches (1:1)

`compute_registered_statistics`'s `elif` chain (lines 1039-1055), read in full and traced against every cell
of `(sign(|Z_harn|≤3), sign(|Z_prod|≤3), ρ range)`:

- `|Z_harn|≤3` splits into exactly two branches on `|Z_prod|` — FLOOR-CONSISTENT / INTERMEDIATE (a). No `ρ`
  dependency in this half, matching the draft (draft's rows 2/3 don't gate on `ρ`).
- `|Z_harn|>3` splits on whether `ρ is None` (⇔ `|Z_prod|≤3`, since `ρ` is only assigned when `|Z_prod|>3`) →
  INTERMEDIATE (d); else on `ρ`'s three-way partition (`≥0.5` / `(0.2,0.5)` / `≤0.2`) → ILLEGITIMATE /
  INTERMEDIATE (b) / INTERMEDIATE (c). This partition of the real line is exhaustive (no gap, no overlap) —
  traced by hand, not merely trusted from the `pragma: no cover` comment.
- All six named dispositions (ILLEGITIMATE, FLOOR-CONSISTENT, INTERMEDIATE a/b/c/d) have exactly one code
  branch each; the trailing `else: raise AssertionError` is reachable only if `Z_harn` or `Z_prod` is NaN,
  which requires `se_harn`/`se_prod` NaN, which requires `n_universes_available≤1`/`n_dark≤1` — both
  precluded upstream by `gates_green` (which requires `harness_universes_green`, itself requiring
  `count_matches_expected` = exactly 67 matched universes, and `g_population_production`'s
  `dark_matches_expected` = exactly 1512) before any disposition is computed. Confirmed unreachable given the
  gate ordering (gates are checked and must be green before `compute_registered_statistics` proceeds past the
  `if not gates["gates_green"]: return {...NO_READ...}` early return) — not merely asserted by the comment.

**NO-READ trigger list**, traced against §4/§5's named triggers (`g-population`, `g-znorm`, `g-closure` both
venues + every harness universe, `g-byte-id`, `resolved-flags-mismatch`, plus rev. 1's `g-znorm` addition):
all twelve entries in `triggers` (lines 887-911) — `g-population (production/replicate)`,
`g-znorm (production/replicate)`, `g-closure (production/replicate, per-event identity)`,
`g-closure (class closure, production/replicate)`, `g-closure/g-znorm (harness universe)`, `g-byte-id`,
`t0-mean-h anchor`, `resolved-flags-mismatch` — feed `gates_green`, and `compute_registered_statistics`
refuses to bank a disposition on any red (`if not gates["gates_green"]: return {"NO_READ": True, ...
"disposition": "NO-READ", ...}`, confirmed by reading the branch, not just the docstring's claim of it).
`g-precision` and the rail-fraction disclosure are confirmed absent from both `gates_green` and `triggers` —
correctly, since the draft's own §4 NO-READ row does not name them (they are disclosure-only per §2.1/§5's
own wording, "the two must agree ... else disclose", "a quote ... is void" — a citation-validity rule, not a
STOP-the-read rule). This matches `DESIGN_GATE_rev3_computability.md`'s F4 finding's own conclusion, now
re-verified against the current code rather than trusted from the prior gate's prose.

## 4. FIX 4 synthetic checks — verified by hand, independently re-run (not just re-read)

Both synthetic checks BUILD_RECORD.md FIX 4 describes were reconstructed from scratch (5 fabricated
checkpoints, `uv run python`, calling the real functions from `completion_residual_reads.py` directly — not
trusting the printed numbers) rather than merely re-read:

**`check_resolved_flags` (F1).** Built 5 checkpoints: 4 with `REGISTERED_RESOLVED_FLAGS` verbatim, 1
(`seed=4`) with `theta_zwindow` flipped `"off"→"on"`. Independent re-run:
`n_checkpoints_matched_population=5`, `n_checkpoints_mismatched=1`, `resolved_flags_equality_green=False`,
`differing_keys=['theta_zwindow']`, seed-4 `diffs={'theta_zwindow': {'harness': 'on', 'production_registered':
'off'}}` — **exact match** to BUILD_RECORD.md's claimed result. Confirms the gate discriminates an injected
flag flip, not a tautological pass.

**`compute_rail_fraction_disclosure` (F3) — DISCREPANCY FOUND.** BUILD_RECORD.md's narrative: "4 with `no_bh`
MAP at the upper rail (0.86) and `with_bh` MAP at the upper rail, 1 (`seed=5`) with both channels' MAP at the
*lower* rail (0.6). Result: `no_bh` rail_fraction = 1/5 = 0.2, `with_bh` rail_fraction = 5/5 = 1.0." Built
this exact construction (4 checkpoints, both channels at 0.86; 1 checkpoint, both channels at 0.6) and ran
`compute_rail_fraction_disclosure` directly: the actual result is `no_bh: rail_fraction=1.0 (5/5)`,
`with_bh: rail_fraction=1.0 (5/5)` — **not** the `0.2`/`1.0` split BUILD_RECORD.md quotes. This is arithmetically
forced by the code's own (correct) semantics: `if m == lo or m == hi: rail_hits += 1` counts BOTH bounds as
"at the rail," so a channel with 4 values at the upper bound and 1 at the lower bound is 5/5 at-rail, not 1/5.
**Diagnosis:** the code is right (both bounds are genuinely checked, per its own docstring and per the
draft's "at either rail" framing) and my independent construction of the description reproduces a sound,
hand-countable 1.0/1.0 result via the real function — the defect is in BUILD_RECORD.md's own narrative
arithmetic/description of what it ran, not in `compute_rail_fraction_disclosure` itself. **Not scored RED**:
`rail_fraction_disclosure` carries `disposition_role: None` and is excluded from the `elif` disposition chain
and from `gates_green`/`triggers` (confirmed in §3) — this is a documentation-accuracy note on the build
record's own synthetic-check writeup, not a computability defect in the registered read. Flagged for the
author/next builder to reconcile BUILD_RECORD.md's prose with whatever 5-checkpoint construction it actually
ran (the live re-run on the real 67 checkpoints, `no_bh` 10/67=14.9%, `with_bh` 14/67=20.9%, is unaffected —
that number does not depend on the synthetic check).

`check_gprecision` (F4) carries no synthetic check in BUILD_RECORD.md (only a "live re-run on the real
inputs" claim) — nothing to hand-verify against a fabricated table; its live-run numbers are read-only
provenance (`relative_diff≈0.0055`, disclosed `within_tolerance:false`, non-gating) and were not
independently recomputed here (would require reading the real `selection_tables_h_0_72.json`/`_0_73.json`
full-precision sources, which is a real-input read of provenance data, not a registered-population aggregate
— but out of scope for this gate's ≤5-row/synthetic discipline since it isn't itself a synthetic check to
re-run).

## 5. New note: harness "0 mixed rows" / seed-range sub-clause of g-population has no dedicated code path

Draft §5: `**g-population** — harness: 0 mixed rows (--population 200, seeds 901000–901066, resolved-flag
token per checkpoint); production: ...`. The production half is fully enforced (`check_production_population`,
JOIN gate, in-cat/dark counts, all folded into `gates_green`). The harness half's "0 mixed rows" and "seeds
901000–901066" sub-clauses have **no dedicated check**: `reproduce_harness_byte_id`/`check_resolved_flags`/
`compute_rail_fraction_disclosure`/`compute_harness_matched_channel_scores`/`check_harness_universe_gates` all
filter checkpoints by `n_draw_requested == population` wherever they are used, and `byte_id_count_green`
requires exactly 67 MATCHED checkpoints — but nothing compares `n_checkpoint_files_globbed` (informational
only) against `n_checkpoints_matched_population` to catch extraneous non-200 files sitting alongside the 67,
and nothing gates on the seed set actually being `{901000..901066}` (only the count, 67, is gated; `seed_min`/
`seed_max` are informational). **Not scored RED**, and structurally different from rev. 3's F1: every function
that consumes harness checkpoints re-applies the population filter at its own call site, so a contamination
scenario (extra non-200 files present) would be silently excluded from every statistic rather than silently
corrupting one — unlike F1, where a resolved-flags mismatch would have silently changed what T_harn actually
measured. Confirmed on the actual registered inputs: `ls harness-root | grep -c universe_seed.*_S.json` = 67,
matching `n_checkpoints_matched_population` exactly — no contamination present at launch. Listed per the
review brief's instruction to name any remaining draft-named gate with no code path.

## 6. Minor: NO-READ trigger label granularity for the "JOIN gate"

Draft's own §4 NO-READ row lists "JOIN gate red" as a separate item from "g-population red." In code,
`join_gate_green` is one of three sub-conditions folded into `population_production_green` (alongside
`in_catalogue_matches_expected`/`dark_matches_expected`); a JOIN-gate-only failure fires the single trigger
label `"g-population (production)"`, not a distinctly-named `"JOIN gate"` trigger. Functionally the read
still correctly refuses to bank (any of the three sub-conditions failing red-flags the same aggregate), so
this is a labeling/granularity note, not a defect that makes the read wrong — the disposition-table brief's
requirement ("real mode refuses to bank on any red") holds regardless of which sub-check fired.

## 7. Launch block: CLI, zero fresh choices

`build_parser()` (lines 1090-1104) reproduced via `--help` (no data touched): 11 flags —
`--production-csv/--production-crb/--replicate-csv/--harness-root/--population/--h-lo/--h-hi/--h-true/
--crb-md5/--catalogue-md5/--out/--dry-run` — **exact 1:1 match**, no more no fewer, against REGISTRATION_DRAFT.md
§7's launch block. Every value in the draft's CLI is a literal (`--population 200 --h-lo 0.725 --h-hi 0.735
--h-true 0.73 --crb-md5 9a1f2a1... --catalogue-md5 c52c13b...`) — no flag is left for the launcher to choose.
`py_compile`/`ruff check`/`mypy` all clean on the current file (re-run by this reviewer, static only — no
data read). `--production-crb` md5 independently re-hashed and matches the draft's pinned value exactly (§1).

## 8. Kill criterion, max_revisions, blindness line

Kill criterion quoted in REGISTRATION_DRAFT.md §8b cross-checked **verbatim** against
`graph1_20260901/RESEARCH_GRAPH_1_PROPOSAL_20260901.md`'s `q-completion-residual` row: "registered arm fails
to discriminate at its registered band after revision 2 -> park bounded-undetermined with the measured
bound" — exact string match, word for word. `max_revisions 2` is stated in the draft header and is consistent
with the kill criterion's "after revision 2" and §8b's "this draft is revision 1 ... a third failure parks
the question." Blindness line present verbatim in §8b ("primary statistic point estimates exist in a gate
record dated 2026-09-03 ... the registered read is executed by an agent that has not opened that record. The
revising author did not open it.") — this reviewer did not open `DESIGN_GATE_stats.md` either, preserving the
same blindness for this gate pass.

## Summary for the author

**GREEN.** Rev. 3's sole RED (F1, the harness↔production resolved-flags equality assertion) is closed with a
real code path (`check_resolved_flags` against `correspondence_1d.REGISTERED_RESOLVED_FLAGS`, independently
cross-checked against that file's `_RESOLVED_FLAG_ATTRS`), wired into `gates_green`/`NO_READ`, and confirmed
by an independently-reconstructed synthetic flag-flip test (this review's own re-run, not a re-read of
BUILD_RECORD's numbers). F3 and F4 are closed as disclosure-only fields, correctly non-gating, matching the
draft. Two items are flagged for the record, neither blocking launch: (1) BUILD_RECORD.md's own FIX-4
rail-fraction synthetic-check narrative doesn't reproduce its claimed 0.2/1.0 split under hand verification
against the actual code — the code itself is confirmed correct (both rail bounds genuinely checked) via an
independent reconstruction, so this is a build-record documentation note, not a code or gate defect;
(2) the harness "0 mixed rows"/seed-range sub-clause of g-population has no dedicated enforcement beyond the
population filter every consuming function already applies — non-corrupting by construction, confirmed
non-contaminated on the actual registered inputs (67/67), listed per the review brief's instruction rather
than scored. Every statistic named in §2.4 (`T_prod, SE_prod, Z_prod, T_harn, SE_harn, Z_harn, ρ, δh_M`) plus
`S_all` has a code path matching the draft's formula and thresholds exactly; the six-way disposition
partition is exhaustive and 1:1 with the draft's table; the launch CLI has zero fresh choices; the kill
criterion is verbatim-verified against the charter; F5 (`T0_MEAN_H_TOLERANCE` cosmetic mislabel, rev. 3,
out of scope for FIX 4) remains open but blocks nothing.
