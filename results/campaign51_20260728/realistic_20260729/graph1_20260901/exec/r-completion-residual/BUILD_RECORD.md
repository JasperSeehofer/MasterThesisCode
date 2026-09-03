# BUILD_RECORD — b-completion-scorer (r-completion-residual)

Date: 2026-09-03. Node: `b-completion-scorer` (Research Graph 1, Branch G, wave 3), build only —
no fresh scientific choices (REGISTRATION_DRAFT.md §7 "Launch block (zero fresh choices)").
Builder ran **only `--dry-run`**; standing rule 2 (memory `agent-verifier-output-is-evidence-not-
authority`) reserves the real-mode invocation for a different agent, gated on this dry-run's
gates being green.

## What was built

`completion_residual_reads.py` implementing REGISTRATION_DRAFT.md §2.1–§2.4 with the §5 gates:

- `compute_event_terms` — the g-closure identity (§2.1): per-event `s_M` (matched-channel score,
  `Δ ln B_num/Δh − Δ ln β̄_Ḡ^φ/Δh`), `s_T` (global composition tilt), `s_C` (catalogue-leg
  increment), `s_e` (full score), and the closure residual, from the exact CSV columns
  (`B_num`, `D_tilde_phi`, `alpha_G_phi`, `den_log_term`, `num_log_term_no_bh`) — no
  reconstruction of `β`, per the draft's explicit "why β is never reconstructed" note.
- `check_gclosure` — §2.1 gate, tolerance `1e-9·(|s_e|+1)`.
- `check_production_population` — §2.2/§5 g-population + JOIN gate (event_idx set =
  `{0..len(crb)-1}` minus the 2 unscored gaps; in-catalogue count == 76; dark count == 1512).
- `check_gznorm` — §5 spot check: `den_log_term` identical across all event rows, per h.
- `reproduce_harness_byte_id` — §2.3/§5 g-byte-id instrument gate: globs
  `universe_seed*_S.json` under `--harness-root`, matches the `--population` tag, and reads back
  `score_at_truth.no_bh.dark.mean` from all 67 checkpoints plus the `resolved_flags` internal-
  consistency check (13-key block identical across all 67).
- `t0_mean_h` — reproduces the T0 gradient-trapezoid `mean_h` convention
  (`prod2d_closure_20260818/tier0_bootstrap_jackknife.py` docstring: `combine_log_likelihood`
  physics-floor zero handling, `w = np.gradient(h_grid)`, `mean_h = Σ post_n·h·w`) — imported
  `combine_log_likelihood` verbatim from `darksiren_emri.validation.correspondence_1d` (no
  physics re-implemented).
- `check_gprecision` — optional selection-table cross-check (§2.1), disclosed skip when no
  full-precision `selection_tables_h_{lo,hi}.json` exists for the stencil nodes (none does; the
  only such file on disk is for `h=0.73`, not the 0.725/0.735 stencil).
- `run_dry_run` — loads every input, runs every gate above, prints row counts / anchors, exits 0
  **without** computing `T_prod`/`T_harn`/`Z`/`ρ`/the disposition.
- `compute_registered_statistics` — the real-mode §2.4 statistics (`T_prod`, `SE_prod`, `Z_prod`,
  `T_harn`, `SE_harn`, `Z_harn`, `ρ`, the §4 disposition) and the full per-event-term JSON record.
  **Present in the file for completeness (the launch block names it as the real-mode CLI) but NOT
  invoked in this session** — the builder ran only `--dry-run` below.

CLI flags match REGISTRATION_DRAFT.md §7's launch block exactly: `--production-csv
--production-crb --replicate-csv --harness-root --population --h-lo --h-hi --h-true --crb-md5
--catalogue-md5 --out --dry-run`.

## Files touched

- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_reads.py` (new)
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/BUILD_RECORD.md` (this file)

No edits under `darksiren_emri/`. `--out` (`completion_residual_result.json`) was **not** written —
confirmed by directory listing after the dry-run.

## Quality gates run by the builder

- `uv run ruff check completion_residual_reads.py` → **All checks passed!**
- `uv run mypy completion_residual_reads.py --ignore-missing-imports` → **Success: no issues found
  in 1 source file** (extra, not required by the task, run for confidence).

## Dry-run invocation (exact, from repo root — runbook 42 §5 gotcha)

```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_reads.py \
  --production-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
  --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
  --replicate-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
  --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip \
  --population 200 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
  --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_result.json \
  --dry-run
```

Exit code: **0**. `stderr` line: `DRY-RUN gates all green: True`.

## Dry-run output (verbatim, `stdout` JSON — pretty-printed by the script itself)

```json
{
 "mode": "dry-run",
 "production_crb_md5": {
  "expected": "9a1f2a14384a9281c97ca3be312ddaab",
  "actual": "9a1f2a14384a9281c97ca3be312ddaab",
  "match": true
 },
 "catalogue_md5_of_record": {
  "expected": "c52c13b5cab61f6b3f04bbe202550969",
  "note": "not independently re-hashed here -- this script consumes only the CSVs the catalogue already fed into (event_likelihoods.csv, prepared_cramer_rao_bounds.csv); recorded for provenance per CLAUDE.md dataset-pinning convention"
 },
 "g_population_production": {
  "n_h_nodes": 41,
  "rows_per_h_node": [1588],
  "rows_per_h_uniform": true,
  "n_rows_total": 65108,
  "n_crb_rows": 1590,
  "missing_event_idx": [1203, 1356],
  "join_gate_green": true,
  "n_in_catalogue_scored": 76,
  "n_dark_scored": 1512,
  "in_catalogue_matches_expected": true,
  "dark_matches_expected": true
 },
 "g_znorm_production": {
  "all_h_nodes_uniform": true,
  "n_h_nodes_checked": 41,
  "max_nunique": 1
 },
 "g_population_replicate": {
  "n_h_nodes": 41,
  "rows_per_h_node": [1588],
  "rows_per_h_uniform": true,
  "n_rows_total": 65108,
  "n_crb_rows": 1590,
  "missing_event_idx": [1203, 1356],
  "join_gate_green": true,
  "n_in_catalogue_scored": 76,
  "n_dark_scored": 1512,
  "in_catalogue_matches_expected": true,
  "dark_matches_expected": true
 },
 "g_znorm_replicate": {
  "all_h_nodes_uniform": true,
  "n_h_nodes_checked": 41,
  "max_nunique": 1
 },
 "g_closure_production": {
  "n_events": 1588,
  "max_closure_residual": 2.5579538487363607e-13,
  "n_violations": 0,
  "gclosure_green": true
 },
 "g_byte_id_harness": {
  "n_checkpoint_files_globbed": 67,
  "n_checkpoints_matched_population": 67,
  "n_checkpoints_expected": 67,
  "byte_id_count_green": true,
  "n_dark_means_present": 67,
  "resolved_flags_internally_consistent": true,
  "n_distinct_resolved_flags_blocks": 1,
  "seed_min": 901000,
  "seed_max": 901066,
  "dark_full_score_means": [ /* 67 values, elided here — full list in the run's stdout capture */
   -0.03684332680707096, -0.011932528209638497, -0.02660525358654448, -0.02358195028098366,
   -0.0010693493945890227, 0.0784279920769636, -0.025728798001607652, -0.010183244139164901,
   -0.03388657085016816, 0.004105865269774426, 0.023731752097597005, 0.10119215868463913,
   0.025260387464604628, -0.018667692883903495, 0.07172738260808453, -0.0006139003054327378,
   0.07782379383070917, -0.001205113505551609, -0.08481331079745157, 0.04729880792701685,
   0.06244292902260193, 0.025356251574629764, -0.014819984047714898, 0.03487727091787526,
   0.02151834928135097, -0.03694230961310493, 0.014608079067532753, -0.040341161273799254,
   0.05250871736831085, -0.03808404972565446, -0.0623556277167366, 0.05683335890701403,
   0.001307888828097373, -0.004558712355360873, 0.09676913243112817, 0.047837721097609726,
   -0.04404359759562281, -0.05770008304042757, 0.06154580360250548, 0.08222901708541754,
   -0.004234763043548639, 0.10123823266708055, 0.00978716322728008, -0.03869145615725185,
   -0.012077805524152456, 0.04258774244463247, 0.07815949972569997, -0.05969435014684141,
   -0.056620162272346675, 0.07952884950719988, -0.0008984806036467763, -0.0012834016970765664,
   -0.01998435105463944, -0.017618410017291634, 0.003944296565434913, 0.0971613656186274,
   -0.13676645707421928, -0.10032894427449991, 0.0437279004610132, -0.03540519673679583,
   0.047814526004112616, 0.030134165176089096, -0.014216777511634094, 0.10035303029215972,
   0.01825779943064465, -0.008854623691606767, -0.008982195966789818
  ],
  "mean_of_dark_full_score_means": 0.008215870005381617,
  "sem_of_dark_full_score_means": 0.006314188695650197
 },
 "t0_mean_h": {
  "computed": 0.6669869414473403,
  "target_display_precision": 0.666987,
  "computed_rounded_to_6dp": 0.666987,
  "abs_diff": 5.855265972076751e-08,
  "reproduces_to_tolerance": true,
  "reproduction_basis": "round(computed, 6) == displayed anchor (source carries no finer precision)",
  "tolerance": 1e-09,
  "n_h_grid": 41
 },
 "anchors": {
  "production_rows": 65108,
  "production_n_h_nodes": 41,
  "crb_rows": 1590,
  "n_dark_scored": 1512,
  "n_in_catalogue_scored": 76,
  "harness_checkpoints_matched": 67,
  "h_stencil": [0.725, 0.735],
  "h_true": 0.73
 }
}
```

`stderr`: `DRY-RUN gates all green: True`

## Note on the T0 byte-id gate's precision

`READOUT_RECORD.md`'s table quotes `mean_h` at 6 decimal places (`0.666987`); no full-precision
value is stashed anywhere reachable (`repro_summary.json` and the `c0prime_eval` posterior JSONs
do not carry it). A literal `1e-9` absolute-difference check against a 6-dp display value is
unsatisfiable by construction (display rounding alone can be up to 5e-7). This script's own
full-precision `mean_h` is `0.6669869414473403`; `round(0.6669869414473403, 6) == 0.666987`
exactly, so the dry-run's `reproduces_to_tolerance` is **true** on the basis "rounds to the
displayed anchor" — the finest reproduction possible against the source of record. This is
disclosed in the script's own `t0_mean_h` report block (`reproduction_basis`), not hidden.

## What the byte-id verifier must reproduce (unchanged from the launch instruction)

1. All 67 S3 harness checkpoints' `score_at_truth.no_bh.dark.mean` bit-for-bit
   (`universe_seed90100{0..66}_S.json` under
   `tree2_20260830/b8_cal_harness_work_s4_postflip/`) — this dry-run confirms 67/67 checkpoints
   present, matched to `--population 200`, one internally-consistent `resolved_flags` block, and
   reads back all 67 `dark.mean` values verbatim (listed above). A genuine byte-for-bit
   reproduction of a value already computed and stored by the harness is a read-back identity,
   not a recomputation from raw per-event scores (the checkpoint does not store per-event scores,
   only the aggregate — noted in `reproduce_harness_byte_id`'s docstring); the verifier's task is
   to confirm this script's real-mode `T_harn`/`SE_harn` (mean/SEM of exactly these 67 values)
   match to machine precision.
2. The T0 re-baseline `mean_h = 0.666987` (iiib, 1D) to the precision the source record actually
   carries — see the note above. Computed here: `0.6669869414473403` (rounds to `0.666987`).

## Gates green at dry-run (byte-id gate GREEN — launch un-blocked per §7)

g-population (production + replicate): GREEN. g-znorm (production + replicate): GREEN. g-closure:
GREEN (max residual 2.56e-13, tolerance ~1e-9 per event). g-byte-id (harness, 67/67 +
`resolved_flags` internal consistency): GREEN. T0 mean_h reproduction: GREEN (to display
precision). g-precision (selection-table cross-check): not evaluated — no
`selection_tables_h_0_725.json` / `_0_735.json` exists on disk (only `h=0.73`'s); disclosed as a
skip in the script's `check_gprecision` function, not run in this dry-run (dry-run report does not
call it — see "known gap" below).

## Known gap / handoff note for the verifier

`check_gprecision` (§2.1's optional selection-table cross-check) is implemented but **not called**
from `run_dry_run` — the registration draft scopes it as "where a full-precision
`selection_tables_h_*.json` exists" for the *stencil* nodes (0.725/0.735), and no such file exists
anywhere under this repo (only `h=0.73`, checked via `find`). Wiring it into the dry-run report
would only ever print `"any_full_precision_source_found": false`; the function is left available
for the verifier or a future run if such files appear. This is a disclosed no-op, not a missed
gate — the identity of §2.1 does not depend on β reconstruction, and g-closure (which does not
need selection tables) is GREEN.

## FIX 2 (2026-09-03, fix round 2) — DESIGN_GATE_rev1_computability.md RED-1 + RED-2

Both RED items from the fresh computability re-gate are fixed in
`completion_residual_reads.py`. Every other code path is byte-identical to the pre-fix version
(diffed manually against git; only the two named blocks changed). Dry-run only — real mode was not
run.

### Diff summary

**Pre-check (RED-1's own demand): does the per-universe data exist for all 67 S universes?**
Verified with `ls`/`find` before writing any code, per the task instruction:

```
$ ls .../b8_cal_harness_work_s4_postflip | grep -E '^seed9010[0-9]{2}_S$' | sort | wc -l
67
$ find .../b8_cal_harness_work_s4_postflip -path '*_S/simulations/diagnostics/event_likelihoods.csv' | wc -l
67
$ find .../b8_cal_harness_work_s4_postflip -path '*_S/simulations/prepared_cramer_rao_bounds.csv' | wc -l
67
$ find .../b8_cal_harness_work_s4_postflip -maxdepth 1 -name 'universe_seed*_S.json' | wc -l
67
```
All four counts are 67/67 — full population, so Fix 1 proceeds as specified (no STOP needed).

**Fix 1 (RED-1 — S_M,harn must be matched-channel, computed per harness universe, not the harness
full score):**

- New function `compute_harness_matched_channel_scores(harness_root, population, cell, h_lo, h_hi)`.
  For each of the 67 matched checkpoints it opens THAT universe's own
  `seed{seed}_S/simulations/diagnostics/event_likelihoods.csv` and sibling
  `simulations/prepared_cramer_rao_bounds.csv`, calls the SAME `compute_event_terms` used for
  production (identical stencil/columns — `B_num`, `D_tilde_phi`, `alpha_G_phi`, `den_log_term`,
  `num_log_term_no_bh`), masks the dark class via `host_galaxy_index == -1`, and takes the
  per-universe mean of `s_M` over dark events. `T_harn` = mean of these 67 per-universe values;
  `SE_harn` = their between-universe SD/√67 — the registered statistic's OWN SE (§2.3/§2.4), not
  the harness full-score checkpoint SE.
- `compute_registered_statistics` now calls this function for `T_harn`/`SE_harn`/`Z_harn`. The old
  full-score-checkpoint aggregate (`reproduce_harness_byte_id`'s `dark_full_score_means`) is kept
  ONLY as two new INFORMATIONAL fields in the output record —
  `T_full_harn_informational` / `SE_full_harn_informational` — and continues to serve the g-byte-id
  instrument gate (unchanged: 67/67 bit-for-bit against the checkpoints), exactly as revision 1b
  specifies ("the checkpoint's `score_at_truth.no_bh.dark.mean` ... is used ONLY for the byte-id
  gate, never for Z_harn").
- `reproduce_harness_byte_id` (the byte-id gate function) and `run_dry_run` (dry-run gate suite)
  are untouched — the byte-id gate is still computed from the checkpoint full-score means, as
  designed; only the registered `Z_harn` no longer conflates the two.

**Fix 2 (RED-2 — disposition table gap; the script's own "unclassified" fallback):**

- The disposition `elif` chain in `compute_registered_statistics` gained one explicit branch,
  matching REGISTRATION_DRAFT.md §4 revision 1b verbatim:
  `elif abs(z_harn) > Z_BAND and rho is None: disposition = "INTERMEDIATE (d)
  HARNESS-ONLY-SIGNAL"` — the `|Z_harn| > 3 AND |Z_prod| ≤ 3` (ρ undefined) combination.
- The old unregistered fallback string
  (`"INTERMEDIATE (unclassified -- rho undefined, |Z_prod| <= 3)"`) is REMOVED entirely — it does
  not appear anywhere in the file any more (`grep -c unclassified` = 0).
- The final `else` is now a defensive `raise AssertionError(...)` rather than a silent label: the
  six named rows (ILLEGITIMATE, FLOOR-CONSISTENT, (a), (b), (c), (d)) are exhaustive over the
  `(Z_harn, Z_prod, rho)` state space by construction (verified by case analysis in the synthetic
  sweep below), so this branch should be unreachable; if it is ever hit the script now fails loudly
  instead of banking an unregistered outcome.

### Byte-id re-run (`byteid_check.py`, independent, does not import the fixed script)

`byteid_check.py` is untouched by this fix (it re-derives everything from raw files, never imports
`completion_residual_reads.py`) and was re-run to confirm the byte-id anchors are still exactly
where they were before the fix:

```
verdict: GREEN
n_pairs: 68  (67 harness dark-mean pairs + 1 T0 anchor pair)
max_abs_dev: 5.855265972076751e-08   (the T0 6-dp display-rounding gap, disclosed, unchanged)
harness_byte_id.count_green: true
harness_byte_id.bit_for_bit_exact_vs_build_record: true   (67/67 exact, 0.0 max deviation)
harness_byte_id.mean_matches_build_record: true
harness_byte_id.sem_matches_build_record: true
t0_mean_h.rounds_to_display_anchor: true
```

Byte-id is still GREEN at 67/67 exact after the fix.

### Dry-run output (verbatim, launch-block CLI, `--dry-run`)

```
DRY-RUN gates all green: True
```

Full JSON (elided to the gate/anchor tail; the leading per-checkpoint `dark_full_score_means` list
is identical to the FIX-1 build's and to `byteid_check.py`'s independent read — omitted here for
length, unchanged from the section above):

```json
{
 "production_crb_md5": {"expected": "9a1f2a14384a9281c97ca3be312ddaab", "actual": "9a1f2a14384a9281c97ca3be312ddaab", "match": true},
 "g_population_production": {"n_h_nodes": 41, "rows_per_h_uniform": true, "n_rows_total": 65108, "n_crb_rows": 1590, "missing_event_idx": [1203, 1356], "join_gate_green": true, "n_in_catalogue_scored": 76, "n_dark_scored": 1512, "in_catalogue_matches_expected": true, "dark_matches_expected": true},
 "g_znorm_production": {"all_h_nodes_uniform": true, "n_h_nodes_checked": 41},
 "g_population_replicate": {"n_h_nodes": 41, "rows_per_h_uniform": true, "join_gate_green": true, "in_catalogue_matches_expected": true, "dark_matches_expected": true},
 "g_znorm_replicate": {"all_h_nodes_uniform": true},
 "g_closure_production": {"n_events": 1588, "max_closure_residual": <~1e-13>, "n_violations": 0, "gclosure_green": true},
 "g_byte_id_harness": {"n_checkpoint_files_globbed": 67, "n_checkpoints_matched_population": 67, "n_checkpoints_expected": 67, "byte_id_count_green": true, "resolved_flags_internally_consistent": true},
 "t0_mean_h": {"computed": 0.6669869414473403, "target_display_precision": 0.666987, "computed_rounded_to_6dp": 0.666987, "abs_diff": 5.855265972076751e-08, "reproduces_to_tolerance": true, "tolerance": 1e-09, "n_h_grid": 41},
 "anchors": {"production_rows": 65108, "production_n_h_nodes": 41, "crb_rows": 1590, "n_dark_scored": 1512, "n_in_catalogue_scored": 76, "harness_checkpoints_matched": 67, "h_stencil": [0.725, 0.735], "h_true": 0.73}
}
```

Note: dry-run does NOT exercise the fixed `compute_harness_matched_channel_scores` or the new
disposition branch — per the launch block, `--dry-run` runs gates/closure/byte-id only and never
computes `T_harn`/`Z_harn`/the disposition (real mode does, and real mode was not run by this
builder, per standing rule 2). The two fixes are verified below by a **synthetic-table check**
instead, per the task's ≤5-row / synthetic-only builder constraint.

### (cone) Synthetic-table check — fabricated data, ≤5 rows/universe, no registered-population aggregate

A standalone script (`/tmp/.../synth_test/run_synth.py`, not committed — scratch verification
only) fabricated 3 synthetic "harness universes" (3, 2, and 4 dark rows respectively — all ≤ 5) in
a temp directory, with `B_num`/`num_log_term_no_bh` constructed so each event's `s_M` equals a
chosen target value exactly (constant `D_tilde_phi`, `alpha_G_phi`, `den_log_term` across h,
so `s_T` and the `den_log_term` contribution are algebraically zero and `s_M` is exact by
construction). It then called the real, unmodified `compute_harness_matched_channel_scores` and
`compute_event_terms` from the fixed script against this fabricated data (no import of the
registered population; no aggregate computed over any registered dataset):

```
n_universes_matched: 3 (expected 3)
T_harn computed:  0.012777777777782431  vs analytically expected 0.012777777777777777  (match)
SE_harn computed: 0.004339027597727321  vs analytically expected 0.004339027597725920  (match)
per-universe S_M: seed 901000 -> 0.013333... (n_dark=3); seed 901001 -> 0.005000... (n_dark=2);
                  seed 901002 -> 0.020000... (n_dark=4)
compute_event_terms (2-event synthetic table): max closure residual = 0.0 (exact, as designed)
```

Confirms Fix 1: `T_harn`/`SE_harn` are computed from the between-universe mean/SD of the
per-universe MATCHED-CHANNEL score, matching hand-computed values to float precision, not from the
harness full-score checkpoint aggregate.

A second synthetic sweep exercised the disposition selector (the exact logic now in
`compute_registered_statistics`) over six `(Z_harn, Z_prod, rho)` triples chosen to hit every named
row, including the new one:

```
Z_harn=5.0 Z_prod=1.0 rho=None -> INTERMEDIATE (d) HARNESS-ONLY-SIGNAL   [Fix 2's new row]
Z_harn=1.0 Z_prod=1.0 rho=None -> FLOOR-CONSISTENT
Z_harn=1.0 Z_prod=5.0 rho=0.9  -> INTERMEDIATE (a) harness-clean, production-displaced
Z_harn=5.0 Z_prod=5.0 rho=0.9  -> ILLEGITIMATE
Z_harn=5.0 Z_prod=5.0 rho=0.3  -> INTERMEDIATE (b) partial
Z_harn=5.0 Z_prod=5.0 rho=0.1  -> INTERMEDIATE (c) minor-illegitimate
```

Confirms Fix 2: the `|Z_harn| > 3 AND |Z_prod| ≤ 3` combination now returns the registered
`INTERMEDIATE (d) HARNESS-ONLY-SIGNAL` label (not `"unclassified"`), and the other five rows are
unaffected.

### Quality gates

`ruff check` — clean. `ruff format --check` — clean (one reformat applied and verified idempotent).
`mypy` — `Success: no issues found in 1 source file`.

### Status

Both RED items from `DESIGN_GATE_rev1_computability.md` are fixed; byte-id anchors remain GREEN
(67/67 exact + T0 display-rounding disclosed); the AMBER items (A1 CRB-md5-not-gated, A2 stale
`T0_MEAN_H_TOLERANCE` constant, A3 dead `check_gprecision`) were **not** in scope for this fix round
(task named only items 2 and 3 / RED-1 and RED-2) and are left as-is, unchanged from the prior
build. Real mode was not run by this builder; the next verifier should re-run
`compute_registered_statistics` on the real inputs as a DIFFERENT agent (standing rule 2) and
independently re-derive `T_harn`/`SE_harn`/`Z_harn`/the disposition rather than trusting this
record's printed numbers, per `agent-verifier-output-is-evidence-not-authority`.

## FIX 3 (2026-09-03, fix round 3) — DESIGN_GATE_rev2_computability.md checks 2 + 4 (RED items 1-3, AMBER items 4-5)

All four items named in the fix-round-3 task are implemented in `completion_residual_reads.py`.
Dry-run only was executed by this builder (standing rule 2 — real mode is run by a different
agent); all new code paths are otherwise exercised only via the ≤5-row synthetic-table check
below, per the builder's zero-real-mode / no-registered-aggregate constraint.

### Diff summary

**1. `delta_h_M` in the output dict (REGISTRATION_DRAFT.md §2.4, RED finding 3).** New function
`compute_delta_h_m(t_prod, n_dark)`: `I_1D = 1/SIGMA_H_1D_REBASELINE_IIIB**2`
(`SIGMA_H_1D_REBASELINE_IIIB = 0.017526`, new module constant), `delta_h_M = n_dark * t_prod /
I_1D`. Returns `{N_Gbar, sigma_h_1D, I_1D, delta_h_M, reported_only: True, verdict_bearing:
False}`. Called once in `compute_registered_statistics` (real mode, after the gate check passes)
and written into the top-level output dict as `"delta_h_M"`. It plays no role in the disposition
`elif` chain — confirmed by construction (the chain is unchanged except for the pre-check
described in item 4 below) and by the synthetic check below.

**2. g-closure + g-znorm on every harness universe + the replicate venue, feeding `gates_green`
(RED finding 1 + half of RED finding 2).**

- New function `check_harness_universe_gates(harness_root, population, cell, h_lo, h_hi)`: for
  every harness checkpoint matching `population` (same seed-matching logic as
  `compute_harness_matched_channel_scores`), reads that universe's own
  `event_likelihoods.csv`, calls `compute_event_terms` + `check_gclosure` + `check_gznorm` on it,
  and reports a per-universe `universe_green` plus an aggregate
  `all_universes_gclosure_gznorm_green` (also requires the matched count to equal
  `N_HARNESS_UNIVERSES`, closing part of DESIGN_GATE_rev2 finding 1's g-population gap as a
  side effect — an under-count now also fails this gate, not just the checkpoint-scoped
  g-population check).
- `g_znorm_replicate` (previously computed but dead per DESIGN_GATE_rev2 finding 1) and a new
  `g_closure_replicate` (per-event identity) are now both computed AND read into `gates_green`.
- New shared function `collect_gate_report(...)` runs the complete gate suite (production +
  replicate population/znorm/closure/class-closure, the harness-universe sweep, g-byte-id, t0
  mean_h) and folds every boolean into one `gates_green` plus a `NO_READ` record naming every
  gate that fired (`{"no_read": bool, "triggers": [...]}`) — a harness-universe or replicate-venue
  closure/znorm red now appears by name in `NO_READ["triggers"]`, exactly as §4's NO-READ row
  prescribes ("g-closure red ... g-znorm red"). `run_dry_run` is now a thin wrapper around
  `collect_gate_report` (byte-identical gate semantics to the pre-fix dry-run for every field that
  already existed; the new fields are additive).

**3. Class-closure identity (RED finding 2, second half).** New function
`check_class_closure(terms, crb, tol)`: `S_G`/`S_dark` are the class means of the full score
`s_e` over the catalogue-hosted / dark classes (from the CRB `host_galaxy_index` column, same
class split `check_production_population` already uses), `S_all` is the population mean of
`s_e`, `pi_G`/`pi_Gbar` the class-count fractions; checks `|S_all - (pi_G*S_G +
pi_Gbar*S_dark)| <= GCLOSURE_TOLERANCE*(|S_all|+1)`. This is an algebraic identity given correct
class assignment (a weighted mean of the parts always reconstructs the whole), so — mirroring
§2.1's own framing of the per-event identity — a red here localises a class-assignment/indexing
defect, not a physics read; it goes red only in the degenerate case (an empty class), confirmed
in the synthetic check below. Called on production and (when supplied) the replicate venue inside
`collect_gate_report`, feeding `gates_green`, and again standalone inside
`compute_registered_statistics` so its residual sits in the real-mode output record next to the
other per-event terms (`"class_closure"` key).

**4. Real mode self-gates; refuses to bank a disposition if any gate is red (AMBER findings 4).**
`compute_registered_statistics` gained three new required parameters (`replicate_csv_path`,
`crb_md5_expected`, `catalogue_md5_expected`, `h_true` — `main()`'s call site updated to pass
`args.replicate_csv`, `args.crb_md5`, `args.catalogue_md5`, `args.h_true`). It now calls
`collect_gate_report(...)` — the SAME function `run_dry_run` calls — as its first action. If
`gates["gates_green"]` is `False`, it returns immediately with `{"mode": "real", "NO_READ": True,
"no_read_triggers": [...], "disposition": "NO-READ", "gates": gates}` and computes nothing else
(no `T_prod`/`T_harn`/disposition are banked — the gate table is written to `--out` instead, per
§4's NO-READ row: "nothing banked"). When gates are green, the previously-existing statistic
computation proceeds unchanged, and the returned dict's `"gates"` field is now the FULL gate
table from `collect_gate_report` (previously only `g_closure` + `g_byte_id_harness`), with
`"NO_READ": False` and the new `"delta_h_M"` / `"class_closure"` keys added alongside the
existing statistic fields. AMBER finding 5 (`SE_harn`'s `/√n_available` vs the registered `/√67`)
was **not** in this task's scope (the task named only checks 2 and 4's RED items plus every
AMBER, and finding 5 is an AMBER under check 2 — re-read: the task says "RED items in checks 2
and 4, and every AMBER" — finding 5 IS in scope). **Correction:** finding 5 is addressed by item
2's `count_matches_expected` addition to `check_harness_universe_gates` — a harness population
short of 67 now fails `gates_green` before `SE_harn` is trusted, which is the cross-check finding
5 says is missing (`n_universes_available == N_HARNESS_UNIVERSES` is now enforced, just via the
harness-universe gate rather than a standalone assertion in `compute_registered_statistics`
itself). `SE_harn` continues to divide by `n_available` (unchanged, and still correct per rev. 1
item 2 — the registered `/√67` and `/√n_available` coincide exactly when the population gate
this fix adds is green).

Everything else is unchanged: the six-way disposition `elif` chain, `compute_event_terms`,
`compute_harness_matched_channel_scores`, `reproduce_harness_byte_id`, `t0_mean_h`, the CLI flags
(only the internal call to `compute_registered_statistics` gained arguments — no new CLI flag),
and every previously-existing gate function's own logic (`check_production_population`,
`check_gznorm`, `check_gclosure`, `check_gprecision` — bodies untouched, only *called more
often/in more places*).

### Byte-id re-run (`byteid_check.py`, unmodified, independent — does not import this script)

```
verdict: GREEN
n_pairs: 68  (67 harness dark-mean pairs + 1 T0 anchor pair)
max_abs_dev: 5.855265972076751e-08   (the T0 6-dp display-rounding gap, disclosed, unchanged)
```

Confirms the byte-id anchors are unaffected by this fix round (no change to
`reproduce_harness_byte_id`'s own body; `byteid_check.py` re-derives everything from raw files and
never imports `completion_residual_reads.py`).

### Dry-run output (exact §7 launch block, `--dry-run`, from repo root)

```
$ uv run python results/.../completion_residual_reads.py \
    --production-csv .../event_likelihoods.csv --production-crb .../prepared_cramer_rao_bounds.csv \
    --replicate-csv .../joint_r1/.../event_likelihoods.csv \
    --harness-root .../b8_cal_harness_work_s4_postflip \
    --population 200 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
    --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
    --out .../completion_residual_result.json --dry-run

DRY-RUN gates all green: True
```

New/changed fields in the dry-run JSON, verbatim (unchanged fields from the FIX 2 record —
`g_population_production`, `g_znorm_production`, `g_closure_production`, `g_byte_id_harness`,
`t0_mean_h`, `anchors` — omitted here for length; re-confirmed present and numerically identical
to the FIX 2 build):

```json
"g_closure_replicate": {"n_events": 1588, "max_closure_residual": 9.836575998178887e-14, "n_violations": 0, "gclosure_green": true},
"g_class_closure_production": {"n_total": 1588, "n_dark": 1512, "n_catalogue": 76, "pi_G": 0.04785894206549118, "pi_Gbar": 0.9521410579345088, "S_G": 1.207935464057304, "S_dark": -0.11420262006209346, "S_all": -0.050926490091643725, "reconstructed_S_all": -0.05092649009164371, "class_closure_residual": 1.3877787807814457e-17, "class_closure_green": true},
"g_class_closure_replicate": {"n_total": 1588, "n_dark": 1512, "n_catalogue": 76, "pi_G": 0.04785894206549118, "pi_Gbar": 0.9521410579345088, "S_G": 1.0502771840482537, "S_dark": -0.1064993120995181, "S_all": -0.05113721278765999, "reconstructed_S_all": -0.05113721278766, "class_closure_residual": 1.3877787807814457e-17, "class_closure_green": true},
"g_harness_universes": {"n_universes_checked": 67, "n_universes_expected": 67, "count_matches_expected": true, "all_universes_gclosure_gznorm_green": true, "per_universe": ["...67 entries, all universe_green: true, gclosure_green: true, gznorm_green: true..."]},
"gates_green": true,
"NO_READ": {"no_read": false, "triggers": []}
```

Note: `g_class_closure_production`/`g_class_closure_replicate`'s residuals (~1e-17) are float
roundoff on an algebraic tautology, as expected (item 3 above) — not a physics result. Dry-run
still does not compute `T_prod`/`T_harn`/the disposition/`delta_h_M` (real mode does; not run by
this builder, per standing rule 2).

### (cone) Synthetic-table check — fabricated data, ≤5 rows, no registered-population aggregate

A standalone script (`/tmp/.../synth_fix3/run_synth3.py`, not committed — scratch verification
only) imported the real, unmodified new functions against fabricated data:

**Class closure (5 fabricated events: 3 dark, 2 catalogue-hosted, constant `D_tilde_phi` /
`alpha_G_phi` / `den_log_term` so `s_e` is exact by construction):**

```
gclosure: max_closure_residual = 8.88e-16, gclosure_green = True
class_closure: S_G=-0.50252, S_dark=1.61300, S_all=0.76679, reconstructed_S_all=0.76679,
               residual=1.11e-16, class_closure_green = True
hand-computed S_dark/S_G/S_all match the code to 1e-9 (exact algebraic identity, confirmed by
  independent hand arithmetic on the same 5 rows, not by re-reading the code's own output)
```

Degenerate case (crb reassigned so the "catalogue-hosted" class is empty among the events passed
to `check_class_closure`): `class_closure_green = False`, `note = "empty class or empty
population -- cannot form pi_G/pi_Gbar"` — confirms the gate can actually go red (not a
tautologically-always-green rubber stamp).

**Per-universe harness gate (2 fabricated "universes" — a deliberately small, non-67 synthetic
sweep, not the registered population):**

```
2 clean universes -> both per_universe entries gclosure_green=True, gznorm_green=True;
  all_universes_gclosure_gznorm_green = False, driven ONLY by count_matches_expected (2 != 67),
  confirming the aggregate correctly demands the full registered count, not just per-universe cleanliness
After corrupting universe 901001's den_log_term on one event/h-node (breaking the "global per h"
  invariant for that universe only):
  per-universe gznorm_green: [(901000, True), (901001, False)]
  all_universes_gclosure_gznorm_green = False (correctly flips on the injected defect)
```

Confirms `check_harness_universe_gates` actually discriminates a real per-universe defect, not
just a synthetic-population-size mismatch.

**delta_h_M formula:**

```
inputs: t_prod = -0.114 (arbitrary), n_dark = 1512
code:   I_1D = 3255.6250787779877, delta_h_M = -0.052944671400768
hand:   I_1D = 1/0.017526**2 = 3255.6250787779877 (rounds to the draft's "3256")
        delta_h_M = 1512 * -0.114 / 3255.6250787779877 = -0.052944671400768
match: exact (both computed independently to float precision)
```

### Quality gates

`ruff check` — clean (no new warnings). `ruff format` — one reformat applied (the new code as
first written did not match the project's line-wrap convention exactly; `ruff format --check` is
now clean and idempotent). `mypy` — `Success: no issues found in 1 source file`.

### Status

All four fix-round-3 items are implemented: `delta_h_M` is a real output field (REPORTED-ONLY,
non-verdict-bearing, confirmed by inspection of the unchanged disposition chain); g-closure and
g-znorm now run on every harness universe AND the replicate venue, with every boolean folded into
`gates_green` and a named `NO_READ` trigger list; the class-closure identity has a code path with
its residual in the output; and real mode (`compute_registered_statistics`) now runs the complete
gate suite itself before computing anything, refusing to bank a disposition (writing `NO_READ` +
the gate table instead) when any gate is red. Dry-run confirms `gates_green: True` /
`NO_READ.no_read: False` on the real registered inputs (nothing here claims what real mode would
report on those inputs — real mode was not run by this builder, per standing rule 2; the next
verifier should independently re-run `compute_registered_statistics` and re-derive every field,
per `agent-verifier-output-is-evidence-not-authority`). `byteid_check.py` (unmodified, independent)
re-confirms GREEN at 67/67 exact.
