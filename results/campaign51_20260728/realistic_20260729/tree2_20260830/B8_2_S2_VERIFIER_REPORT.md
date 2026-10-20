# B8.2 S2 -- independent verifier report

`launched under rows #255/#268 -- tree 2 node B8.2.S2 (verifier)`

Reviews `results/campaign51_20260728/realistic_20260729/tree2_20260830/B8_2_S2_RECORD.md`
against the design of record
(`results/campaign51_20260728/realistic_20260729/fanout1_20260829/B8_2_HARNESS_DESIGN_20260829.md`
sections 4, 8 S2 row) and `b8_cal_harness.py`. Class: independent verifier, sonnet, high effort,
clean context -- did not write S2. No git operation performed (the orchestrator commits); no
ssh; foreground/background-local only (taskset-equivalent affinity pin -> workers=2, event-cap
<=20, per the launch stamp's resource ceiling -- a local job on this machine was reported to be
using ~14 cores until ~20:00; all live runs below stayed at workers=2). Append-only. Branch
`fix/p32d-classg-venue-repair`.

**Method.** Read the design (§4, §8 S2 row), the S2 record, and the full 1089-line
`b8_cal_harness.py` source line by line. Then went beyond static review: (a) ran the harness's
own `score_only()` against an independently hand-built fixture (not the builder's fixture) and
cross-checked every statistic against a from-scratch re-implementation; (b) **launched three real
`evaluate()`-backed universes** (two cell S, one cell T, N=20/event-cap=20/workers=2, matching
the launch stamp's resource ceiling) -- something the S2 build itself never completed -- and
independently re-derived the posterior/PIT/SD/score-at-truth statistics straight from the raw
`event_likelihoods.csv`, bit-matching the checkpoint; (c) fed the harness's own verbatim-copied
scorer primitives the REAL banked `results/venue_transfer_20260811/Tc_h0p730_results_seeds*_25.json`
(400 seeds) and `T0_h0p730_results_seeds*_25.json` (200 seeds) raw `ln_post` vectors and
confirmed the design §1.2(a)/§8 acceptance-(ii) numbers exactly; (d) tested resumability
with a planted fake checkpoint; (e) re-ran ruff/mypy myself.

**Bottom line.** The driver and scorer are correct on every quantitative check this report could
run, including several the S2 build itself disclosed it could not complete (a live checkpoint,
and the acceptance-(ii) reproduction of the banked T-c(0.730)/T-0 cells). One design-listed S1
gap (the grid-split bit-identity live test) remains untested by both S1 and S2 and could not be
completed here either within the practical time budget -- draw_realization's own cost proved
to dominate even a 2-h-value smoke attempt (killed at ~9 min, still drawing). No correctness
defect was found anywhere in `b8_cal_harness.py`; the findings below are two should-fix items
(one efficiency, one UX) plus the carried-forward untested item.

---

## Verdict table

| # | Item (design §4/§8 S2 acceptance + record's own claims) | Verifier finding | Verdict |
|---|---|---|---|
| Checkpoint schema | Every field the design §8 S2 row / record §4 lists (`schema_version`, `stamp`, `universe`, `grid`, `elapsed_s`, `resolved_flags` (13 keys), `posterior.{no_bh,with_bh}` w/ ln_post/h_grid/map_h/sd/pit/hpd50-95/n_events_scored, `score_at_truth`, `z_true_hist`, `n_pred_by_bin`, `candidate_census`, `grid_split_check`) | Verified present, by key, on a REAL checkpoint produced live this session (`universe_seed900700_S.json`) -- not just by reading the code. All top-level and nested field sets matched exactly; `resolved_flags` had exactly 13 keys as claimed. | **PASS** |
| Statistics per design §1.2(a)/§4 definitions | MAP/SD/PIT/HPD(50/68/90/95), score-at-truth secant, F=SD/floor, PIT-KS, coverage binomial bands, score-zero N-weighted pooling, absolute-count audit Z | Re-derived independently in a from-scratch script (no import of any b8_cal_harness function) against a hand-built 6-universe fixture: median SD, HPD hit counts, mean(MAP)-h_true, and the KS statistic (hand-computed via the D+/D- construction) matched `score_only()`'s output exactly. Separately, the pooled score-zero Z for `catalogue_hosted` (9.43) was hand-verified via the closed-form N-weighted mean/SEM formula the record itself states, and the count-audit Z's matched a hand computation of `(n_real-n_pred)/sqrt(n_pred)`. Then repeated end-to-end on the REAL live checkpoints (below). | **PASS** |
| Live end-to-end run (closes S2's own gap: record admits "a full checkpoint was NOT obtained") | Launch two cell-S and one cell-T universe at N=20/event-cap=20/workers=2 (the resource ceiling this launch was given) | **Universe seed900700 (cell S) completed in 221.6s wall** (incl. one-time 76.6s context build) and wrote a real checkpoint; **seed900701 (cell S) completed in 713.6s** (n_catalogue_hosted=2); **seed900750 (cell T) completed in 210.8s** (gw_scatter=False threaded correctly). All three checkpoints have `resolved_flags` matching the 13 registered production values (the assertion did not raise, i.e. `assert_resolved_production_flags` PASSED on live production code -- this is the design §3 item 1 engagement assertion, exercised for real for the first time in this stage's history). Re-derived `ln_post`, `map_h`, `sd`, `pit`, and `score_at_truth.all` for seed900700's no_bh channel directly from the raw diagnostics CSV via an independent script (no b8_cal_harness import except the reused `combine_log_likelihood`): **bit-identical to the checkpoint** (`sd=0.0035354084717960726`, `pit=0.5023282784144625`, `score mean=-0.09803831458742707` matched to all printed digits). | **PASS** (closes the record's self-disclosed gap; this is the first completed end-to-end universe this harness has ever produced) |
| Candidate-count log parser (`parse_candidate_counts`) | Design §2.4/CANDIDATE_COUNT_METHOD_SOURCE: log-line order == first-h-block CSV order, generalizes across the two-call split and workers>1 | On the real seed900700 run (workers=2, 2-call split, 3 h-nodes): grepped the raw combined `harness.log` by hand and confirmed the "possible hosts found N/M..." lines repeat in THREE identical 8-line blocks (one per h-node, spanning BOTH calls), in event_idx order (0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19) exactly matching the CSV's zero/non-zero `L_cat_no_bh` pattern for every entry. `log_parse_reason` was empty (no ALIGNMENT FAIL) and the parsed `n_cand_no_bh`/`n_cand_with_bh` arrays matched the hand-traced values exactly. This empirically confirms the ordering assumption holds under real multiprocessing (workers=2), which the design/record never demonstrated. | **PASS** (empirically confirmed under real concurrency, not just asserted) |
| Count-audit formula fidelity (`alpha_g_phi_per_bin`, `beta_gbar_phi_per_bin`) | Design §1.2(b): must restrict, not re-derive, `precompute_global_catalog_selection`/`precompute_phi_selection_integrals`'s own sums | Read both production functions' bodies (`bayesian_statistics.py:2745-3010`, `:2085-2140`) line by line. `alpha_g_phi_per_bin`'s eligibility mask, `w_g = R_eff_per_mbh(M_g)/(1+z_g)`, and `S_bar_phi` table interpolation are IDENTICAL to `precompute_global_catalog_selection(with_bh_mass=False, phi_survival_table=...)`'s own `phi_survival_table is not None` branch, term for term. `beta_gbar_phi_per_bin`'s integrand `(1-f_bar)*s_phi*p_pop` is identical to `precompute_phi_selection_integrals`'s `beta_Gbar_phi` integrand. The record's self-disclosed §2 fix (comparing the raw per-bin sum against `compute_catalogue_class_weight_p_g`'s `"sigma_phi"` key, not its `"alpha_G_phi"` key) is independently confirmed correct: `path_a_mixture_objects` (`bayesian_statistics.py:2449-2506`) shows `alpha_G_phi = beta_G_phi * r_Malm != sigma_phi` -- the record's disclosed ~14.5x comparand bug and its fix are both verified against the actual production formula, not just trusted from the record's prose. The rescale-by-a-single-global-ratio approximation (assumes Malmquist ratio r_Malm is z-bin-independent) is a genuine, disclosed simplification -- reasonable for a diagnostic instrument, does not touch the mixture law itself, correctly flagged as a named approximation per the bounded-scope rule. | **PASS** (with the pre-existing named approximation correctly disclosed, not a new finding) |
| Acceptance (ii): scorer reproduces banked T-c(0.730) N=400 and T-0 from raw vectors | Record's own claim: validated only against a **hand-built fixture**, not the actual banked data (explicitly disclosed, not run) | Loaded all 16 real chunk files for `Tc_h0p730` (400 seeds) and all 8 for `T0_h0p730` (200 seeds) from `results/venue_transfer_20260811/`, and fed their real `ln_post_1d`/`h_grid` vectors through the harness's own imported `trapz_norm`/`my_pit`/`my_post_sd`/`my_hpd_contains`/`my_ks_uniform`. Result: **coverage hpd50/68/90 = 0.000/0.000/0.000 (0/400 each), PIT-KS D = 1.000000, post_sd median = 0.004376** -- bit-exact to the registered `VENUE_TRANSFER_READOUT.md` numbers. T-0: **all 200/200 seeds' argmax lands exactly on h_true=0.73** -- matches "all 200 seeds argmax exactly on truth" exactly. This is a full, live, from-real-data close of the acceptance item the S2 build left as fixture-only. | **PASS** (independently closed; the record under-claimed here, it did not over-claim) |
| Resumability / re-invocability | `--max-wall-s` stops cleanly; re-invoking skips existing checkpoints | Planted a fake checkpoint JSON at a fresh seed/work-root, re-ran the driver: it still paid the one-time ~80s context-build cost, printed `"seed 900800 cell S: checkpoint already exists, skipping"`, and scored 0 universes -- exactly the documented behaviour. | **PASS** |
| A22 stamp | `git_stamp()` records real commit/branch/dirty-paths at run start | Compared the live checkpoint's `stamp` block to `git rev-parse HEAD`/`--abbrev-ref HEAD` run independently at the same moment: commit `5c13b82b...`, branch `fix/p32d-classg-venue-repair`, and the dirty-paths list matched the actual `git status --porcelain` output line for line. | **PASS** |
| Score-only aggregator never writes a verdict | Design rule 2 | Grepped the full source for `verdict`/`"PASS"`/`"FAIL"` literals: the only matches are in docstrings/print statements explicitly disclaiming a verdict (lines 32, 955-993); `score_only()`'s return dict is never serialized to disk by `main()` -- only printed via `print_score_only_report()`. Ran `--score-only` against both real live work-roots (cell S, n_U=1 and n_U=2; cell T, n_U=1): confirmed no file is written, only stdout. | **PASS** |
| h_bounds split bit-identity ([P3-HGRID], S1 verifier must_fix 1) | Design §3 item 3 / §8 S1 acceptance (iv), which S1 never tested and S2 claims to "close" via `verify_grid_split_bit_identity()` | Code review: the function is correctly constructed (explicit `h_bounds=(min,max)` on both the whole-grid and the 2-way split call, `mid = len(h_values)//2` identical to `run_one_universe`'s own operational split, diffs every non-identifier CSV column). **However, S2's own record never ran it live** (both its live attempts used `--no-verify-split-once` or were killed before reaching it). This verifier attempted a live run too (2-h-value smoke, `--seed-block 900760`, default `--verify-split-once`): `draw_realization` alone had not returned after **9 minutes** (killed per this repo's "never leave a backgrounded process unattended past a turn" convention) -- independently corroborating the record's own §5 finding that `draw_realization`'s cost is large and seed-dependent, dominated by the ~20.8M-row kernel-smearing step, not by `n_events` or `h_values` count. Indirect supporting evidence exists (the live seed900700 run showed IDENTICAL candidate-ball sizes across all 3 h-nodes spanning a real 2-call split with the same pinned h_bounds, consistent with -- but not identical to -- the whole-vs-split property `verify_grid_split_bit_identity` checks). | **NOT LIVE-TESTED** (gap carried forward from S1; code construction is correct on inspection; must_fix below) |
| Ruff / mypy | Clean | Re-ran independently: `ruff check` all checks passed; `ruff format --check` already formatted; `mypy` success, no issues. | **PASS, independently reproduced** |
| No unit tests for the S2 driver itself | Not explicitly required by design §8's S2 acceptance list (which specifies live/fixture tests instead), but noted | No `test_b8_cal_harness.py` exists. The design's S2 acceptance items are all live/fixture-style, not a pytest requirement, so this is not a FAIL against a stated criterion -- but the count-audit and score-zero pooling formulas are exactly the kind of logic that regresses silently without a pinned test. | **should_fix** (recommendation, not a design-acceptance failure) |
| Cell T output not flagged as coverage-invalid | Design §2.3: "No coverage claim is made from cell T (its PIT is degenerate by construction)" | Ran `--score-only --cell T` on the real cell-T checkpoint: it prints PIT-KS D and coverage exactly as it does for cell S, with no annotation that these are not meaningful for cell T. Confirmed by reading `print_score_only_report()`/`score_only()`: neither branches on `cell`. | **should_fix** (UX/documentation gap, not a numeric defect) |

---

## Must-fix / should-fix list

1. **must_fix (carried forward from S1, still open):** live-test `verify_grid_split_bit_identity()` before S3/S5 rely on the chunked-grid plan. Given the empirically-confirmed cost (draw_realization alone can exceed 9 minutes per seed, independent of `n_events`/`h_values` count -- confirmed independently by this report, corroborating S2 record §5), this needs either a much longer background budget or the cached-host-weights optimization S2's own record §5/§7 already proposes but did not implement.
2. **should_fix (efficiency):** `verify_grid_split_bit_identity()` re-runs BOTH the whole-grid call and the 2-way split from scratch, even though `run_one_universe`'s own operational split (`calls = [h_values[:mid], h_values[mid:]]`) is constructed identically to the check's own split arm. This means the "first universe of an invocation" pays for 5 total `evaluate()` calls (2 operational + 1 verify-whole + 2 verify-split) instead of the 3 it would need if the check reused the operational split's own already-computed `diag_csv` as its "split" comparand and only added the one extra whole-grid call. In a script whose central cost concern is `evaluate()`-call count, this is an avoidable ~1.7x multiplier on the one universe that pays it.
3. **should_fix (UX):** annotate `print_score_only_report()`'s cell-T output (or `score_only()`'s returned dict) with a reminder that PIT-KS/coverage are not valid coverage claims for cell T (design §2.3) -- someone running `--score-only --cell T` in isolation, without the design note open, would otherwise read a real KS/coverage number with no warning it is a degenerate-by-construction quantity.
4. **should_fix (test coverage):** no `test_b8_cal_harness.py` exists for the driver/scorer's own logic (`score_only`'s pooling, `alpha_g_phi_per_bin`/`beta_gbar_phi_per_bin`, `parse_candidate_counts`). Not a design-acceptance failure (S2's own acceptance list specifies live/fixture tests, which this report performed), but a regression-safety gap for future edits.

## What this report additionally banked (new evidence, not in the design's own acceptance list)

- Three real, live, `evaluate()`-backed checkpoints now exist under
  `results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_verifier_work/`
  (`universe_seed900700_S.json`, `universe_seed900701_S.json`, `universe_seed900750_T.json`) --
  the first completed end-to-end runs this harness has ever produced, useful as a regression
  fixture for S3.
- Independent reproduction script for the banked T-c(0.730)/T-0 cells:
  `/tmp/.../scratchpad/b8_s2_verify/reproduce_tc730.py` (scratch only, not committed) --
  reproducible from this report's method if S4 wants a permanent fixture test built from it.

## Quality gate (independently reproduced)

- `uv run ruff check b8_cal_harness.py` -- all checks passed.
- `uv run ruff format --check b8_cal_harness.py` -- already formatted.
- `uv run mypy b8_cal_harness.py` -- Success, no issues.

## A22 verifier stamp

`launched under rows #255/#268 -- tree 2 node B8.2.S2 (verifier)`, 2026-08-30. Branch
`fix/p32d-classg-venue-repair`, HEAD `5c13b82b6960c3a850a539dc0af5c043e18bb2ca` at the time this
report's live runs executed (tree dirty per the launch note's own stamp convention -- see the
live checkpoints' own `stamp.dirty_paths` for the exact list). No git operation performed by
this node. Did not touch `hier_s0_zwin_run` or `fanout1_20260829/hier_s0_driver.py`. All live
runs stayed within the launch stamp's resource ceiling (event-cap<=20, workers<=2); no
background process was left running past this report (the one grid-split live attempt was
killed after ~9 minutes per this repo's own stated convention for unattended local processes).
