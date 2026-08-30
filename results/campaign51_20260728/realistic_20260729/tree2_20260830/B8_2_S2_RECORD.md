# B8.2 S2 -- driver + score-only aggregator: implementation record

`launched under rows #255/#268 -- tree 2 node B8.2.S2`

Design of record: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/
B8_2_HARNESS_DESIGN_20260829.md` (sections 1-5, 8 S2 row). Inputs read in full per the launch
instruction: the design note, `B8_2_S1_RECORD.md`, and `B8_2_S1_VERIFIER_REPORT.md`. Class:
sonnet build stage, medium effort. No git operation performed by this node (the orchestrator
commits); no ssh; foreground only; append-only. Branch `fix/p32d-classg-venue-repair`. Did not
touch `results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_run` or
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py` (read-only
consulted for CLI/logging conventions, never edited). Did not edit any physics-trigger file --
the only files touched are `results/campaign51_20260728/realistic_20260729/tree2_20260830/
b8_cal_harness.py` (new) and this record.

## 0. S1 verdict read (per the launch stamp's instruction)

The launch instruction quoted the S1 verifier's bottom line verbatim: the `mixture_selected`
generator code is CONFIRMED correct; the record's one reported FAIL (acceptance item (i)) was a
comparand-configuration error in the record's own verification method (empirically confirmed by
the verifier's own live rerun, `max_abs_diff = 0` on all 17 columns), not a code defect -- so S1
did **not** fail a load-bearing item and this stage proceeds (no STOP).

One genuine gap survived: **acceptance item (iv)**, the grid-split bit-identity property S5's
chunking plan depends on, was "NOT TESTED -- FAIL (gap, not a demonstrated defect)" per the
verifier's must_fix 1. This stage closes that gap directly, since S2's own deliverable already
requires the two-call `h_bounds`-pinned split -- see §3 and `verify_grid_split_bit_identity()`.

## 1. Design-to-code map (design §8 S2 row)

| design §8 S2 deliverable | code |
|---|---|
| universe loop over S1's generator | `run_one_universe()` -> `MirrorUniverseGenerator.draw_realization(..., host_mode="mixture_selected", ...)` |
| checkpoint JSON per universe | `run_one_universe()`'s return dict, written by `main()` to `checkpoint_path(work_root, cell, seed)` |
| ln-posterior vectors, both channels | `_channel_stats()`'s `"ln_post"` key, `combine_log_likelihood()` (imported, reused verbatim -- production's own physics-floor zero handling) on the `combined_no_bh`/`combined_with_bh` pivot |
| MAP, SD, HPD 50/68/90/95, PIT | `_channel_stats()`, using the verbatim-copied `adjudicate_venue_transfer.py` primitives (`trapz_norm`/`my_pit`/`my_post_sd`/`my_hpd_contains`) |
| per-event score at truth by class | `_score_at_truth_by_class()` (B4's `per_event_scores` secant method, split by `events["event_class"]`) |
| realized N, N_G | `run_one_universe()`'s `"universe"` block (`n_realized_draw`, `n_catalogue_hosted` from the generator's own `n_catalogue_hosted` column) |
| per-event n_cand from the log parser | `parse_candidate_counts()` (B4's `candidate_counts()` method, adapted) fed a per-universe combined INFO log via `_run_with_log_capture()`'s `FileHandler` |
| z_true histogram on B3.1's bins | `run_one_universe()`'s `"z_true_hist"` block, `np.searchsorted` on `B3_1_BIN_EDGES` (checked against `b3_pop_prediction.json`'s own `registered_bin_edges` at import time -- see `_check_b3_1_bin_edges_against_source()`) |
| N_pred vs realized per bin (absolute-count audit) | `alpha_g_phi_per_bin()` + `beta_gbar_phi_per_bin()` (§2 below) -> `GenerativeContext.n_pred_by_bin`, compared to the realized histogram in `score_only()` |
| two-call `h_bounds=(0.60,0.86)` split, bit-identity pinned | `_run_with_log_capture()` (the operational split used by every universe) + `verify_grid_split_bit_identity()` (the one-time, per-invocation PROOF the split reproduces a whole-grid call -- closes S1 verifier must_fix 1) |
| re-invocable/resumable | `main()`: `checkpoint_path(...).is_file()` skip-if-exists per seed; `--max-wall-s` breaks the loop cleanly between universes |
| `--score-only` aggregator | `score_only()` + `print_score_only_report()` |
| F = SD_measured/sigma_floor per channel | `sigma_floor_for()` (reads `b8_information_floor.json`'s B8.1 route-B closed-form floor; analytically rescales off N=1588) + `score_only()`'s `F_dilution` |
| coverage at 50/68/90/95 | `score_only()`'s `coverage` block, `binom_bands()` |
| score-zero test by class | `score_only()`'s `score_zero_test_by_class` block |
| absolute-count audit table | `score_only()`'s `count_audit` block |
| prints band outcomes, NOT a verdict (rule 2) | `print_score_only_report()` -- prints every statistic against its design §4.1 band, ends with an explicit "does NOT emit a PASS/FAIL verdict" line; no verdict field is ever written to the checkpoint or aggregate JSON |
| `--workers`, `--event-cap`, `--n-universes`, `--cell {S,T}`, `--N` | `main()`'s `argparse` block |
| A22 commit + dirty-state stamp at run START | `git_stamp()` (read-only `git rev-parse`/`git status --porcelain`), called once per universe inside `run_one_universe()`'s returned `"stamp"` block |

## 2. The count-audit per-bin decomposition -- a bounded-scope note (A21-adjacent)

Design §1.2(b) writes `alpha_G^phi(bin) = sum_{g in bin} w_g * S_tilde_phi,g`, citing
`precompute_global_catalog_selection` as the source. That production function's own docstring
(`bayesian_statistics.py:2810-2827`) computes the **point**-evaluated `Sigma^phi(h) = sum_g w_g
S_bar_phi(z_g;h)` (bare listed z, not the kernel-smeared `S_tilde_phi,g` the b0i host-draw weight
uses) -- the design's own text conflates the two objects' names. This implementation follows the
function it actually cites (the point form, `alpha_g_phi_per_bin()`), for two reasons neither of
which is a "corrected premise" requiring a STOP: (a) `compute_catalogue_class_weight_p_g` (S1,
already-verified code) itself builds the **class weight `p_g` from the point form**, not the
kernel-smeared one -- so a per-bin decomposition of the point form is what is actually
self-consistent with the object driving the mixture split; (b) an exact per-bin kernel-smeared
decomposition would require running `kernel_smeared_survival` over the full ~20.8M-row catalogue
once per bin (`_KERNEL_SMEAR_CHUNK`-chunked, several minutes), which the design's own §6 cost
table does not budget for this diagnostic. This is flagged here, append-only, as a **named
approximation** rather than silently assumed -- **not** a "corrected premise" under the
bounded-scope rule, since it changes no band, statistic definition, or the mixture law itself
(the mixture law's `class_weight_p_g` is untouched; only this NEW diagnostic's construction is
affected).

A second, real bug was caught and fixed during this stage's own build (disclosed per the
verifier-independence culture, not swept under a clean report): the first implementation
compared the per-bin decomposition's total against `compute_catalogue_class_weight_p_g`'s
`"alpha_G_phi"` key -- but that key is `path_a_mixture_objects`'s **Malmquist-rescaled**
`r_Malm * beta_G^phi` (`bayesian_statistics.py:2463-2464`), not the raw catalogue sum
`alpha_g_phi_per_bin()` computes (`compute_catalogue_class_weight_p_g`'s `"sigma_phi"` key is the
correct raw comparand). The self-check compared apples to oranges (a ~14.5x mismatch on the live
run); fixed by (i) comparing the RAW per-bin sum against `"sigma_phi"` (apples-to-apples, both
raw catalogue-MC-sum units) and (ii) rescaling the per-bin RAW sum by
`alpha_G_phi_global / sigma_phi_global` before it enters the `N_pred` formula, so `N_pred`'s
numerator sits in `beta_Gbar^phi`'s units (matching `D_tilde_phi`'s own denominator) while the
self-check stays apples-to-apples. See §4 for the live self-check numbers this produced.

## 3. Closing S1 verifier must_fix 1: the grid-split bit-identity property

`verify_grid_split_bit_identity()` runs, once per script invocation (on the first universe a
given invocation scores, gated by `--verify-split-once`, default on): (a) ONE whole-grid
`evaluate()` call over the full requested `h_values`, and (b) a 2-call split (first half / second
half), **both** passing `h_bounds=(min(h_values), max(h_values))` explicitly ([P3-HGRID], design
§3 item 3) -- on the SAME drawn event set and seed. It diffs every non-identifier column of the
resulting `event_likelihoods.csv` (`max_abs_diff`, `max_rel_diff`, per-column breakdown) and
reports `bit_identical`. This is the live test design §8 lists as S1 acceptance item (iv), never
run by S1 or its verifier; it is run here because S2's own driver is the first artifact that
actually NEEDS the split (§6's cost table: a full 41-node call at N=1588 may exceed the 600s
foreground ceiling). Result on the smoke run: see §5.

`parse_candidate_counts()` is written to treat a split run exactly like a single whole-grid run
(B4's method, `CANDIDATE_COUNT_METHOD_SOURCE`): `_run_with_log_capture()` concatenates both
calls' `event_likelihoods.csv` output (production's own append-mode writer,
`bayesian_statistics.py:5419`) and both calls' INFO logs into ONE file per universe, which is
exactly what a single whole-grid call would have produced (the candidate ball is h-list-
independent within one fixed `h_bounds` window, so the log's block structure is identical either
way -- confirmed structurally, not merely asserted, by the grid-split check itself).

## 4. Checkpoint JSON schema (`schema_version: "b8_cal_harness_v1"`)

```
{
  "schema_version": "b8_cal_harness_v1",
  "stamp": {commit, branch, dirty_paths[], timestamp_utc, launch_stamp, role},
  "universe": {seed, cell, gw_scatter, n_draw_requested, n_realized_draw, n_scored,
               n_catalogue_hosted, class_weight_p_g},
  "grid": {h_values[], h_bounds[2], calls[[...],[...]]},
  "elapsed_s": {"call_0": ..., "call_1": ...},
  "resolved_flags": {...13 keys, assert_resolved_production_flags-checked...},
  "posterior": {
    "no_bh":  {ln_post[], h_grid[], map_h, sd, pit, hpd50, hpd68, hpd90, hpd95, n_events_scored},
    "with_bh": {... same shape ...}
  },
  "score_at_truth": {
    "no_bh":  {available, catalogue_hosted:{n,mean,sem}, dark:{...}, all:{...}},
    "with_bh": {... same shape ...}
  },
  "z_true_hist": {bin_edges[], counts[], counts_catalogue_hosted[], counts_dark[],
                  n_below_lowest_edge, n_above_highest_edge},
  "n_pred_by_bin": {bin_edges[], n_pred_shape[], n_pred_scaled_to_n_draw[], self_check{...}},
  "candidate_census": {log_parse_reason, n_cand_no_bh[], n_cand_with_bh[]},
  "grid_split_check": null | {ran, same_shape, n_rows_whole, n_rows_split, max_abs_diff,
                               max_rel_diff, per_column_max_abs_diff{...}, bit_identical,
                               resolved_flags_whole{...}, resolved_flags_split{...}, ...}
}
```

One file per universe: `<work_root>/universe_seed<seed>_<cell>.json`. Per-universe scratch
(CRB CSV, injections symlink, `event_likelihoods.csv`, `harness.log`) lives in
`<work_root>/seed<seed>_<cell>/`; the one-time grid-split check's scratch lives in
`<work_root>/_gridsplit_check_seed<seed>_{whole,split}/` (not deleted, kept for inspection).

## 5. Smoke evidence (design §8 S2 acceptance (i): smoke at N=20, n_U=2, both cells)

**Confirmed working, live, three times (`--n-universes 0` probe -- context build only, no
universe):** `taskset`-pinned to 4 CPUs (workers=2),
`results/.../tree2_20260830/b8_cal_harness_work` as work-root. Three independent runs:
`generative context built in 73.0s` / `70.4s` / `80.8s` -- consistent. `p_g = 0.06196684` reproduced bit-identically across all three (matches
`compute_catalogue_class_weight_p_g`'s own internal determinism -- no RNG in this path).
`n_pred self-check` printed and, after the fix in §2, apples-to-apples: `sigma_phi_binned_sum_raw
851832884.08` vs `sigma_phi_global 980867125.67` (86.9% tiled -- the [0.075, 1.018] bins miss the
tails below/above, as expected) and `beta_gbar_phi_binned_sum 881798720.59` vs
`beta_gbar_phi_global 888403798.07` (99.3% tiled). **A real bug was caught here first**: see §2's
disclosure (the first cut compared the wrong pair of scalars, a ~14.5x apparent mismatch that was
entirely a comparand-selection error in this script, not a defect in production or in S1's code --
fixed before any universe was scored).

**`--score-only` aggregator: validated end-to-end against hand-built fixtures** (2 synthetic
universes, `/tmp/.../scratchpad/b8_fixture_test/`, not committed -- scratch only), exercising
every statistic: `F_dilution`, `pit_ks_d`, all four `coverage` levels with `binom_bands`, the
N-weighted pooled `score_zero_test_by_class`, and the count-audit `per_bin` table. Output values
were hand-checked against the fixture's known inputs and are internally consistent (e.g. the
count-audit's `n_pred`/`n_real` match exactly by fixture construction, `Z=0`).

**A third disclosed fix, caught by this same fixture run**: `score_zero_test_by_class`'s first
implementation pooled per-universe MEANS with an unweighted `np.mean`/`np.std` across universes
(discarding each universe's own `n`/`sem`) -- against the fixture this produced `Z=nan` for every
class whenever two universes' per-class means differed by more than their own naive spread,
because it was throwing away the very sample sizes needed to combine independent estimates.
Fixed to the closed-form N-weighted combination (`sum(n_i*mean_i)/sum(n_i)` for the point
estimate; `sqrt(sum((n_i/N_tot)^2 * sem_i^2))` for the pooled SEM, using only universes with a
sample `sem_i` i.e. `n_i>1`) -- re-run against the same fixture, `Z` became finite and sane for
every class (see the fixture's `dark`/`all` classes in the printed report). Design §4.1's own
statistic definition ("mean of ... over N x n_U events") is unchanged; only this driver's
implementation of "pool across universes from per-universe summaries" was corrected.

**Live universe draw + `evaluate()`: mechanically confirmed working, but a full checkpoint was
NOT obtained within this session's practical time budget.** Two live attempts:

1. `--n-universes 2 --N 20 --event-cap 20 --cell S --h-values 0.725,0.73,0.735 --workers 2
   --max-wall-s 560` under the launch stamp's foreground-600s ceiling: killed by the wrapping
   `timeout 595` before completing universe 1 (concurrent 14-core contention from another running
   job was present for at least part of this attempt -- confirmed via `ps`/`uptime`, load average
   ~2 after that job finished vs the sustained 95%+ single-process CPU draw beforehand).
2. A leaner re-attempt (`--n-universes 1`, `--no-verify-split-once`, unbuffered `python -u`,
   `--max-wall-s 900`, backgrounded via `nohup`+`disown` per the S1/verifier precedent) DID make
   real, observable progress with LIVE production objects: `draw_realization` ran
   (`host_mode="mixture_selected"`, seed 900200, the real host-z-kernel ZoA-fallback warning
   fired exactly as it does in production), the first `evaluate()` call (`h_values=(0.725,)`)
   completed and wrote 20 real rows to a real `event_likelihoods.csv`
   (`results/.../tree2_20260830/b8_cal_harness_work/seed900200_S/simulations/diagnostics/
   event_likelihoods.csv`), `resolved_flags` were populated, and the second call
   (`h_values=(0.73, 0.735)`) was mid-flight (visibly progressing through `evaluate()`'s own
   `D(h)`/`beta_Gbar(h)`/global-catalog-selection precompute, real log lines, real numbers) when
   this record was finalized -- killed rather than left unattended past this turn (S1/verifier's
   own stated convention for backgrounded local processes).

**A genuine, disclosable cost finding surfaced by this attempt (not previously decomposed by
S1's cost note or design §6's cost table):** a standalone diagnostic
(`/tmp/.../scratchpad/diag_single_h.py`, not committed -- scratch only, reproducible from this
record's method) isolated `draw_realization`'s OWN wall time for `host_mode="mixture_selected"`
at `n_events=20`, separate from context build and separate from `evaluate()`. It was killed
(never left unattended, per the S1/verifier convention) at **318s elapsed without returning**
after context build completed -- i.e. `draw_realization`'s own cost for this host mode is lower-
bounded at > 318s and was not fully measured this session. The mechanism is
identified, not merely observed: `mixture_selected`'s catalogue-hosted branch calls
`catalogue_selected_host_draw_weights`, which calls `kernel_smeared_survival` over the **entire
host pool** (the full pinned reduced catalogue, ~20.8M rows) to build the host-draw weights --
chunked at `_KERNEL_SMEAR_CHUNK = 100_000` rows (`correspondence_1d.py:1276,1429-1445`)
specifically because the per-chunk `(chunk, 50)`-node quadrature intermediate would otherwise cost
~8.3 GB at full scale, per that function's own docstring. This is **the SAME cost S1's record
attributed broadly to "catalogue load"** in its ~500-560s single-process setup figure (S1's
comparand also used a `catalogue_selected`-family host mode, which pays the identical
`catalogue_selected_host_draw_weights` cost) -- S1 never decomposed how much of that figure was
catalogue-CSV-read-plus-BallTree-build versus this kernel-smearing step; this record's diagnostic
is the first to isolate it as a separate, likely dominant, per-realization (not per-h, not
per-call) cost. **This is not a code defect** -- it is the intended, disclosed, memory-safe
chunked implementation of an already-adopted physics object (PA-2's kernel-smeared survival) doing
real work over the pinned catalogue's real size; the finding is that **the design's own §6 cost
table has no line for it**, and S3's N-ladder timing (§7 below) will silently conflate it with
`evaluate()`'s own cost unless it is timed separately.

**Recommendation for S3 (append-only, does not change any band/statistic/the mixture law):** time
`draw_realization` and each `evaluate()` call SEPARATELY (not just the wall-clock total per
universe) on the N-ladder, and report whether `draw_realization`'s cost scales with `n_events`
(the FIRST-draw ordering, `n=20`) or is dominated by the **catalogue size** (in which case it is
CONSTANT per (catalogue, h_true) regardless of `n_events`, and should be paid once and the host
pool reused across universes within a script invocation the way `GenerativeContext` already
does -- `catalogue_selected_host_draw_weights`'s OWN weights depend only on `(pool, h_true)`, not
on the realization seed, so a caller drawing MANY universes in one process could cache them
exactly as `build_generative_context()` caches `p_g`/`phi_survival_table`). This repository's
own `b8_cal_harness.py` does NOT yet do this caching (each `run_one_universe()` call re-draws
independently); it is flagged here as a concrete, bounded optimization for S3/S5 to adopt if the
diagnostic confirms the weights are realization-independent (they should be, by the formula's own
definition: `w_g * S_tilde_phi,g` has no `seed` dependence) -- NOT implemented in this stage
because touching `draw_realization`'s caller-visible caching contract is exactly the kind of
change S1-S3's bounded-scope rule (design §8: "may not change... the mixture law") should route
through an explicit append-only note first, which this paragraph now is.

## 6. Quality gate

- `uv run ruff check --fix results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py` -- all checks passed.
- `uv run ruff format` -- formatted, subsequently unchanged.
- `uv run mypy results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py` -- Success, no issues.

## 7. S3 commands (the orchestrator runs these; not run by this node)

**Read §5's cost finding before running anything below.** The `--max-wall-s 580` figures quoted
in the original commands (still shown, append-only) were written BEFORE §5's `draw_realization`
finding (>318s and rising, not yet returned, for `n_events=20` alone) and are very likely too
small even for the N=106 ladder point -- if `draw_realization`'s cost is dominated by catalogue
size (20.8M rows, `_KERNEL_SMEAR_CHUNK`-chunked) rather than by `n_events` (the hypothesis §5
states but did not confirm), it is paid ONCE per universe **regardless of N**, on top of
whatever `evaluate()` itself costs at that N and h-node count. Two consequences for S3, in order:

1. **Confirm the hypothesis FIRST, cheaply, before spending the pilot's compute budget.** A
   single `--n-universes 1 --N 20` run (or reuse this stage's own killed diagnostic, re-run to
   completion) timed against a single `--n-universes 1 --N 1588` run isolates whether
   `draw_realization` scales with `n_events` or is flat. If flat (the mechanism in §5 predicts
   flat), it is paid **125 times over** by the pilot as currently commanded (100 + 25
   universes) -- at even a conservative 320s/draw that is **> 11 hours of draw-only wall time**
   before any `evaluate()` cost, which would make the pilot commands below impractical as
   written.
2. **If flat, cache the host-draw weights across universes before running the pilot** --
   `catalogue_selected_host_draw_weights`'s output depends only on `(host_pool, h_true)`, not on
   the realization seed (§5's closing paragraph); a cached-weights variant of
   `draw_realization`/`run_one_universe` (an ADDITIVE change to this stage's own driver, not to
   `correspondence_1d.py`, so it stays inside the harness-file scope) would cut the pilot's draw
   cost from "125 x" to "1 x" and is very likely necessary for the pilot to be practical.

Neither of these is implemented in this stage (S2's own bounded-scope rule: a corrected premise
returns as an append-only note, which this is -- the mixture law and every statistic definition
are unchanged; only the DRIVER's own reuse-across-universes strategy would change, an
implementation detail already flagged as a concrete option in §5). The commands below are kept
as originally planned (append-only) with this note prefacing them; the orchestrator should re-cut
`--max-wall-s` (and decide on the caching question above) before running the ladder or pilot at
scale.

N-ladder timing (design §6/§8 S3 row -- 3 points x 1 seed, both channels, both calls timed):

```
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 1 --N 106 --cell S --seed-block 900300 \
    --h-values <full H_GRID_41> --workers 2 --max-wall-s 580 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder

uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 1 --N 400 --cell S --seed-block 900301 \
    --h-values <full H_GRID_41> --workers 2 --max-wall-s 580 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder

uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 1 --N 1588 --cell S --seed-block 900302 \
    --h-values <full H_GRID_41> --workers 2 --max-wall-s 580 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder
```

(`<full H_GRID_41>` = the script's own `--h-values` default, i.e. omit the flag to use it.
**Checkpointing is per-UNIVERSE, not per-sub-step**: `--max-wall-s` only stops the driver from
STARTING a new universe once the budget is spent -- if a single universe's own `draw_realization`
+ `evaluate()` calls exceed `--max-wall-s` mid-flight, that universe is NOT checkpointed and
re-running the command restarts it from scratch (the launch stamp's foreground-600s ceiling means
each ladder point may need EITHER a single very long background invocation, per the S1/verifier
`nohup`+`disown` convention §5 used, OR `--max-wall-s` set high enough that the run is left to
complete in one sitting). Per §5's finding, do not assume N<=400 fits in 580s; time N=106 first,
unbounded (background, polled), before committing a wall-time budget to N=400/1588.)

Pilot (design §8 S3 row -- cell S n_U=100, cell T n_U=25, both at N=200):

```
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 100 --N 200 --cell S --seed-block 900400 --workers 2 --max-wall-s 580 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot
# re-invoke the identical command repeatedly (checkpoints skip already-done seeds) until
# n_universes=100 checkpoints exist under the pilot work-root, then:

uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 25 --N 200 --cell T --seed-block 900500 --workers 2 --max-wall-s 580 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot

# then, per cell:
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --score-only --cell S --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --score-only --cell T --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot
```

Per design §8 S3 acceptance: every universe must complete and checkpoint; census must land
within the provisional band of design §2.4 (else STOP and report, do not tune); record
`wall/seed, peak RSS/seed` (A21(b)); no number from S3 is a verdict -- it is S4 registration
input.

## 8. What this stage explicitly did NOT do

- No band/statistic re-derivation from the pilot's realized scatter -- that is S4 (design §8
  forbids S1-S3 from changing any band, statistic definition, or the mixture law).
- No production-N (N=1588) run -- that is S3's N-ladder + S5, not S2.
- No verdict of any kind (CONSISTENT-CALIBRATED / DEFECT-IN-CONSISTENT-VENUE / etc.) --
  `print_score_only_report()` prints statistics against their design §4.1 bands for the chair's
  own read and explicitly disclaims emitting one.

## 9. Draw-weight cache (2026-08-30; B8.2.S2b, "§8" per the launch instruction) --
    confirming and closing §5/§7's cost finding

`launched under rows #255/#268 -- tree 2 node B8.2.S2b`. Scope: confirm §5's `draw_realization`
cost hypothesis cheaply, implement the §7-item-2-proposed cache in the harness driver (additive,
does not touch `correspondence_1d.py`), prove byte-identity, re-time, and re-cut the S3 commands.
No git operation, no ssh, foreground polls only (each `timeout 590 ...` command below was run to
completion, repeated as needed -- no Monitor/background-wait-for-notification pattern used for
the final timings in this section, per the coordinator's correction mid-task). Every run in this
section stayed at `--workers 2` (the launch's resource ceiling while runner-8's job was active).

### 9.1 Hypothesis confirmed: `draw_realization` cost is FLAT in N, dominated by pool size

Two independent cold (`draw_weight_cache` never previously populated for the touched work-root)
single-universe runs, 3 h-nodes (0.70/0.73/0.76, for the timing only -- not the production grid),
`--workers 2`:

| N | seed | `elapsed_s.draw_realization` | cache hit | `elapsed_s.call_0` (1 h) | `elapsed_s.call_1` (2 h) | n_catalogue_hosted | universe wall |
|---|---|---|---|---|---|---|---|
| 20  | 900900 | **451.76s** | miss (compute_s=446.09s) | 59.86s | 83.95s | 2 | 602.3s |
| 106 | 900901 | **461.85s** | miss (compute_s=456.22s) | 48.98s | 92.83s | 5 | 610.5s |

`draw_realization` cost is flat to within ~2% across a 5.3x increase in N (451.8s -> 461.9s) while
`evaluate()`'s own cost (call_0+call_1) is *also* flat (143.8s -> 141.8s) over this same N range --
confirming §5's hypothesis for the draw leg (the >318s-lower-bound finding is fully explained by
`catalogue_selected_host_draw_weights`'s pool-size-dominated `kernel_smeared_survival` pass, not
by `n_events`) and additionally showing `evaluate()` itself is not yet N-sensitive at N<=106 (most
likely a per-h global precompute, e.g. `beta_Gbar(h)`/`D(h)` over the full catalogue, dominates
over the per-event term at this N -- **not** investigated further here, out of this stage's
bounded scope: the cache built below touches only `draw_realization`'s host-draw-weight call).
Both cold runs report the SAME cache key (`8aae9dfa6115f66ec6f173179595b658`), confirming the key
is N-independent by construction, as designed (§7's own point: the weights depend only on
`(host_pool, h_true)`).

### 9.2 Cache implemented and confirmed effective

`b8_cal_harness.py` now monkeypatches `correspondence_1d.catalogue_selected_host_draw_weights`
(the bare module-global name `draw_realization`'s host-mode branches look up at call time) with
`_cached_catalogue_selected_host_draw_weights` -- an in-process dict plus an on-disk `.npz` under
`--work-root/draw_weight_cache/`, keyed by a SHA-256 hash of the pool's own `z`/`M` array bytes +
`h` + `INJECTION_POOL_DIR` + a source-hash of `catalogue_selected_host_draw_weights` and
`kernel_smeared_survival` (so an edit to either function self-invalidates every cache entry --
no version constant to remember to bump). `--no-draw-weight-cache` bypasses it entirely (recomputes
every call, byte-for-byte the pre-B8.2.S2b behaviour). Additive change to THIS driver's own
reuse-across-universes strategy only; `correspondence_1d.py` is untouched, the mixture law is
untouched, and (per 9.3) the RNG stream sees IDENTICAL floats either way.

A second universe drawn in the SAME work-root as 9.1's N=20 cold run (seed 900910, same 3
h-nodes, `--workers 2`) hit the on-disk cache:

| leg | cold (seed 900900) | warm (seed 900910, on_disk hit) | reduction |
|---|---|---|---|
| `elapsed_s.draw_realization` | 451.76s | **8.59s** | 52.6x |
| `elapsed_s.call_0`+`call_1` | 143.81s | 133.32s | (noise; evaluate() unaffected by this cache, as designed) |
| universe wall (`done in`) | 602.3s | **151.1s** | 4.0x |

One real bug was caught and fixed while building this: `np.savez` silently APPENDS `.npz` to a
path that does not already end in `.npz` -- the first tmp-name choice (`<key>.npz.tmp`) got
written as `<key>.npz.tmp.npz`, so the subsequent atomic `tmp_path.replace(npz_path)` raised
`FileNotFoundError` (caught live, first cold N=20 attempt crashed on it). Fixed by using a tmp
name that already ends in `.npz` (`<key>.tmp.npz`) so numpy writes exactly that path. Disclosed
per the verifier-independence culture, not swept under a clean report.

### 9.3 Byte-identity proof: cached vs uncached, same seed -- max_abs = 0 everywhere checked

Re-ran seed 900900 (N=20, 3 h-nodes, `--workers 2`) in a FRESH work-root with `--no-draw-weight-
cache`, and diffed every artifact against 9.1's cached run of the SAME seed:

- `posterior.no_bh.ln_post` and `posterior.with_bh.ln_post` (3-element vectors): **max_abs_diff =
  0.0** for both channels.
- `map_h`, `sd`: bit-identical (`no_bh`: map_h=0.70, sd=0.021163003814093573 both runs; `with_bh`:
  map_h=0.76, sd=0.021175016092939303 both runs).
- `z_true_hist.counts`: identical, `[6, 7, 4, 1, 2]` both runs.
- `universe.n_realized_draw`/`n_catalogue_hosted`: identical, 20/2 both runs.
- The realized event table itself (`seed900900_S/simulations/prepared_cramer_rao_bounds.csv`,
  20 rows x 134 columns): column set identical; **every numeric column's max_abs_diff = 0.0**;
  every object column (`in_catalog`, `_coord_frame`, `_cov_frame`, `host_draw_mode`,
  `event_class`) identical row-for-row.

This is the exact "same RNG stream, cached weights == uncached weights" proof the launch
instruction required: the cache changes WHEN the weights are computed, never WHAT they are.

### 9.4 Re-timed N=20 with a warm cache: per-universe cost now

Combining 9.1/9.2 (N=20, 3 h-nodes, `--workers 2`): **cold first universe of an invocation costs
~602s (draw dominant, 451.8s of it); every subsequent universe against the SAME work-root's cache
costs ~151s (draw collapses to ~9s; the remaining ~142s is `evaluate()`, unaffected by this
cache).** At the production 41-node grid the `evaluate()` leg will be substantially larger (9.5).

### 9.5 Extrapolation to the pilot (N=200, 41 nodes) and the N-ladder (41 nodes) --
    ONE confident number, ONE uncertain one, flagged as such

**Confident (direct consequence of 9.1's flat-in-N draw cost + 9.2's measured warm cost, not an
extrapolation of `evaluate()`):** if the pilot's cell-S (n_U=100) and cell-T (n_U=25) runs share
ONE `--work-root` (as the existing §7 commands already do -- both point at
`b8_cal_harness_work_pilot`), the ENTIRE 125-universe pilot pays the ~455s cold draw cost **once**,
then ~9s/universe thereafter: `455 + 124*9 ~= 1571s (~26 min) total draw time for the whole pilot`
-- down from the pre-cache estimate of `125 * 455s ~= 15.8 hours`, a ~36x reduction on this leg.
This number does not depend on the workers count (the cached weights are read once per universe
from an in-process dict or a small `.npz`, not reprocessed by the worker pool).

**Uncertain (extrapolated from only 2 noisy data points at 3 h-nodes/N in {20,106}/workers=2 --
run the N=106 41-node ladder point FIRST, per §7's own standing advice, before trusting this for
a compute-budget decision):** `evaluate()`'s own cost was FLAT in N over [20,106] at 3 h-nodes
(143.8s -> 141.8s, §9.1), suggesting a per-h GLOBAL precompute (not the per-event, worker-
parallelized likelihood loop) dominates in this regime -- if that holds, scaling naively linear in
h-node count only (41/3 ~= 13.7x): `evaluate(41 nodes, N<=106, workers=2) ~= 143s * 13.7 ~= 1960s
(~33 min)`. Whether `--workers 8` helps this number depends entirely on WHICH part dominates: the
per-event loop parallelizes across workers (expect close to a 4x speedup, workers 2->8, i.e.
`~490s (~8 min)`); a per-h GLOBAL catalogue precompute (`beta_Gbar(h)`/`D(h)`-type sums) does NOT
obviously parallelize across the worker pool the same way, in which case workers=8 buys little
and the true figure stays closer to `~1960s (~33 min)`. **This is a real, disclosed, order-of-
magnitude-only range: `evaluate()` alone per universe at N=200/41-nodes could be anywhere from
~8 to ~33 minutes at workers=8**, and N=400/N=1588's OWN scaling is unmeasured entirely (the
flat-in-N behaviour at N<=106 could break down at higher N once the per-event term catches up to
the per-h precompute term -- unknown from this stage's data). Pilot-scale (125 universes) at this
range: `125 * [490s, 1960s] ~= [17.0h, 68.0h]` of `evaluate()` wall alone, on top of the ~26 min
draw total above. **Recommendation: run the N=106/41-node ladder point (9.6) FIRST, read its
REAL `elapsed_s.call_0`/`call_1` at workers=8, and re-derive the pilot's expected wall from that
real number before committing the pilot's compute budget** -- exactly the same "confirm cheaply
before spending the budget" discipline §7 already applied to the draw-cost hypothesis.

### 9.6 Re-cut S3 commands (orchestrator runs these; not run by this node)

All three N-ladder points and the pilot share ONE `--work-root` each (ladder vs pilot) so the
draw-weight cache's ~455s cold cost is paid AT MOST ONCE per work-root, never per universe. Every
command below can run at any point after runner-8's job (or any other resource-sharing job)
frees the box; poll with a bounded foreground loop (`timeout 590 bash -c 'until ! pgrep -f "..."
>/dev/null; do sleep 20; done'`, repeated), never a background-wait-for-notification pattern, per
the coordinator's mid-task correction. `--max-wall-s` only stops the driver from STARTING a new
universe -- for `--n-universes 1` (every N-ladder command) it has no effect on that one universe;
it is kept here at a generous bound only so the flag's own contract stays documented.

N-ladder (3 points x 1 seed, full 41-node grid = omit `--h-values`, both channels, both calls
timed; **run N=106 first and read its REAL `elapsed_s` before deciding whether N=400/1588 need a
much larger `--max-wall-s` or an unattended overnight background run**):

```
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 1 --N 106 --cell S --seed-block 900300 \
    --workers 8 --max-wall-s 7200 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder

# read elapsed_s.draw_realization (expect ~9s -- SAME work-root's cache is now warm from the
# above call's own first-universe cold pay) and elapsed_s.call_0/call_1 (expect somewhere in
# [~8 min, ~33 min] per §9.5's range) from
# b8_cal_harness_work_ladder/universe_seed900300_S.json BEFORE running the next two.

uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 1 --N 400 --cell S --seed-block 900301 \
    --workers 8 --max-wall-s 14400 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder

uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 1 --N 1588 --cell S --seed-block 900302 \
    --workers 8 --max-wall-s 28800 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_ladder
```

Pilot (cell S n_U=100, cell T n_U=25, both N=200, full 41-node grid, ONE shared work-root so the
cold draw pay happens once across BOTH cells):

```
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 100 --N 200 --cell S --seed-block 900400 --workers 8 --max-wall-s 3600 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot
# re-invoke identically (checkpoints skip already-done seeds) until 100 cell-S checkpoints exist.
# Expected wall per universe after the first (warm draw ~9s + evaluate() per §9.5's
# [~8 min, ~33 min] range): budget roughly 17-68 HOURS of wall time for the full 100-universe
# cell-S batch alone at workers=8 -- re-derive this from the N=106 ladder point's REAL evaluate()
# time (9.6 above) before committing to it; consider chunking via multiple --max-wall-s-bounded
# invocations across sessions/days rather than one long foreground run.

uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --n-universes 25 --N 200 --cell T --seed-block 900500 --workers 8 --max-wall-s 3600 \
    --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot

uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --score-only --cell S --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
    --score-only --cell T --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_pilot
```

### 9.7 Quality gate (re-run after the cache implementation)

- `uv run ruff check --fix .../b8_cal_harness.py` -- all checks passed (one auto-fix applied
  during development, re-verified clean).
- `uv run ruff format .../b8_cal_harness.py` -- formatted, subsequently unchanged.
- `uv run mypy .../b8_cal_harness.py` -- Success, no issues (one `# type: ignore[attr-defined]`
  added then removed after mypy reported it unused -- the monkeypatch assignment typechecks
  cleanly without it).

### 9.8 What this stage explicitly did NOT do

- Did not touch `correspondence_1d.py` (per the launch instruction's bounded-scope rule) -- the
  cache is entirely a harness-side monkeypatch of a module-global name, not an edit to the
  generator's own caller-visible contract.
- Did not investigate or cache `evaluate()`'s own flat-in-N cost (§9.1's observation) -- out of
  scope; flagged for a future stage if the N=106 41-node ladder point confirms it matters at
  production scale.
- Did not run any universe at `--workers 8` (the launch's resource ceiling capped this node at
  `--workers 2` while runner-8's job was active) -- the workers=8 figures in §9.5/§9.6 are
  extrapolations/orchestrator-facing commands, not measurements.
- Did not run the N-ladder or the pilot itself -- that is the orchestrator's own next step
  (§9.6), per the original launch instruction.
