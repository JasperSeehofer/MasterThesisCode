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
