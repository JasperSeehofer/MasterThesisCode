# S0-B dark-class count forensics: 606 (2026-08-29) → 449 (S0-B, job 6779532/6779535)

**Role:** read-only forensic pass. No pipeline runs, no cluster access. All numbers below are
recomputed directly from the two CSVs named.

## 1. File identification and set arithmetic

- **Then** (source of `b3_pop_prediction.json`'s `n_dark=606`): script
  `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b3_1_pop_measure.py`
  reads `VENUES["iiib"]` =
  `results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/event_likelihoods.csv`
  (14.7 MB, 65108 rows = 1588 events × 41 h-nodes; `run_metadata_21.json` alongside it:
  `git_commit=d04d9dc9`, timestamp 2026-08-27T19:40, `--evaluate --h_value 0.73`, job 6724169).
- **Now**:
  `.../graph1_20260901/retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv`
  (1588 rows, single h = 0.73 — this is the S0-B **truth** node; task 0, SLURM job **6779535**,
  under the array job id **6779532** named in the prompt — `provenance_6779535_0.json`
  confirms `slurm_array_job_id=6779532`, `slurm_job_id=6779535`, `slurm_array_task_id=0`; the
  task 4/`job 6779532` proper writes `node_s_minus`, not `node_truth` — no discrepancy, this is
  normal SLURM `--array` numbering).
- Dark class, `L_cat_no_bh == 0`, restricted to h=0.73 in the old CSV (`then`) vs the only
  h-node in the new CSV (`now`):

| | n |
|---|---|
| then (dark at h=0.73) | 606 |
| then (dark at **every** of the 41 h-nodes, i.e. the memo's literal criterion) | 606 (identical set) |
| now (dark) | 449 |
| **then ∩ now** | **449** |
| **then \ now** (moved dark→matched) | **157** |
| **now \ then** (moved matched→dark) | **0** |

All 157 moved events moved in one direction only: dark → matched. `now` is a strict subset
of `then`.

## 2. What distinguishes the 157 moved events (mechanical fingerprint)

For all 157, comparing the *identical event_idx* row at h=0.73 in both CSVs:

| column | then | now | changed? |
|---|---|---|---|
| `w_G`, `w_G_legacy`, `w_tilde_G`, `alpha_G_phi`, `r_Malm`, `D_tilde_phi`, `g_frac` | — | — | **bit-identical**, max abs diff = 0.0 across all 1588 events, moved or not |
| `L_cat_no_bh` | exactly `0.0` (all 157) | `1.0e-110` .. `2.3e-08` (all 157, all > 0) | **changed** |
| `L_cat_with_bh` | 0.0 | tiny positive, tracking `L_cat_no_bh` | changed (same pattern) |
| `combined_no_bh`, `combined_with_bh`, `B_num`, `B_num_wbh`, `L_comp` | — | — | changed by small amounts (see below) |

The class-defining window/candidate-selection quantities (`w_G`, `r_Malm`, `D_tilde_phi`,
`g_frac`) are **byte-identical** for these 157 events between the two runs — the set of
candidate hosts and the geometric window did not change. What changed is that the catalogue-leg
likelihood *integral value* itself moved from an exact hard `0.0` to a representable but
extremely small positive float for exactly these 157 events, spanning ~100 orders of magnitude
(1e-110 to 2e-8) — characteristic of a deep exponential-tail evaluation that used to underflow
(or was clipped) to `0.0` and now does not, rather than a real widening of the selection window
(a real window widening would also move `w_G`/`r_Malm`/`g_frac`, which it does not).

For reference, among the 606 "then"-dark events that stayed dark (449), `L_cat_no_bh` is
`0.0` in both CSVs. Among the 982 "then"-matched events (already `L_cat_no_bh > 0`),
`L_cat_no_bh` itself also shifted between runs (max abs diff 0.013, `combined_no_bh` max abs
diff 2.1e-3) — i.e. some numerical change to the no-BH catalogue-leg computation touches the
whole population, not just the class boundary; the 157 are simply the subset where that shift
crossed the hard-zero/underflow floor.

## 3. Class-definition consistency (item 4)

**Same criterion in both sources.** `b3_1_pop_measure.py`'s own registered comment: "'dark' =
C-C ... `L_cat_no_bh == 0` at *every* h node." Checked directly: for the old CSV, grouping by
`event_idx` and requiring `L_cat_no_bh == 0` at all 41 h-nodes gives exactly the same 606-event
set as requiring it only at h=0.73 — the predicate is h-invariant in the old data, so the
single-h-node "now" CSV (S0-B only evaluates h=0.73) applies the identical criterion, not a
weaker one. No alternate criterion (`n_candidates==0`, `g_frac`, an in-catalogue flag) is used
by either source.

## 4. Ranked cause candidates — config diff table

The S0-B run's CLI, as echoed in `s0a_full_output.json` / the task-4 `.out` log and
cross-checked against `READOUT_RECORD.md`'s full deviation table, matches the 2026-08-27 run's
`run_metadata_21.json:cli_args` on every field examined:

| flag | 2026-08-27 (`then`) | S0-B (`now`) | match? |
|---|---|---|---|
| `catalogue_leg_1d_mass_aware` | (flag did not exist yet at commit `d04d9dc9`) | `off` | consistent with row #287's certified `off`; **(a) A18 flip ruled out** — the run resolves `off`, not `auto` |
| `theta_phi_divisor` | (n/a) | `off` | match |
| `theta_zwindow` / `z_window_k` | (n/a) | `off` / `1.0` | match |
| `sky_cone_k` | `1.5` | `1.5` | match |
| `catalogue_numerator_survival_2d` | `off` | `off` | match — **(c) Option A′ ruled out by config** |
| `sigma4d_mass_kernel` | `point` | `point` | match |
| `eddington_m` | `on` | `on` | match |
| `host_z_kernel` | `volume_deconv` | (not printed; disclosed as unverified beyond driver echo) | assumed match |
| `normalization_mode` | `absolute_marginal` | `absolute_marginal` (implied by `catalogue_leg_1d_mass_aware` guard passing) | match |
| `h_value` | `0.73` | `0.73` | match |
| `seed` (random/CLI) | `777021` | S0-A seed `900101` (a different, but for CoR-P/iiib the seed is documented as structurally inert — `build_iiib_venue` loads the real catalogue, doesn't realize a mock one) | disclosed, not expected to matter |

**No CLI-level difference found.** Both runs resolve to the same production instrument
configuration for every flag checked.

### Code-diff pass (candidates a/b/c/e ruled out mechanically)

`git log --oneline d04d9dc9..081b1f28` for the physics-trigger files touches
`bayesian_statistics.py` (9 commits), `galaxy_catalogue/handler.py` (2, both the same
`mass_filter_geometry`/`theta_zwindow` commits), `cosmological_model.py` (1: `a26959b4`).
`physical_relations.py`, `constants.py`, `simulation_detection_probability.py`: **no commits**
in range.

- **(a) A18 mass-aware flip** (`5e7fda16`) — guarded (`getattr(..., "off") == "on"`), confirmed
  `off` in the S0-B run. Ruled out.
- **(b) WBHZERO symmetric mass window** (`cf4f8a2a`, 2026-08-25) — predates the `d04d9dc9`
  (2026-08-27) baseline; already in effect in the "then" CSV. Not a source of a *difference*.
- **(c) Option A′ 2D de-double-weight** (`d4765539`) — diff contains zero lines touching
  `no_bh`/`without_bh` symbols; confined to the with-BH/2D leg as documented. Ruled out.
- **(e) h-grid admissibility decoupling** (`a26959b4`) — diff only widens the `evaluate()`
  entry-guard ceiling (`h > _h_admissible_max`); the window call site
  (`get_redshift_outer_bounds`) is explicitly, by design, byte-untouched, and h=0.73 is
  in-bounds under either ceiling. Ruled out.
- `d40fe5c8`/`1f003da6` (θ-hook, sites 2.1/2.2/2.3) — both guarded by
  `if theta_b != 0.0 or theta_s != 1.0:`; the `truth` node runs at `theta=(0.0, 1.0)`
  (identity), so this branch does not execute in either run. Ruled out.
- `0b308828` (`mass_filter_geometry`/`mass_filter_k`) and `62f7d61e`/`6c6f2a63`
  (`catalogue_leg_1d_mass_aware`/`theta_phi_divisor` instrumentation) — all three are
  documented, and confirmed by reading the diff, to fall back to the pre-flag code path
  bit-for-bit at their default/`off` values, which is what both runs resolve to.

**No unconditional (non-flag-gated) change to the no-BH 1D catalogue-leg computation exists
in the committed diff between `d04d9dc9` and `081b1f28`.** This rules out every named candidate
(a)/(b)/(c)/(e) and every other physics-trigger-file commit in range as the *mechanical* cause
of the exact-zero → tiny-positive shift, at the level of this repo's git history.

### The open item: the run was not executed on a clean tree

Every S0-B provenance JSON records a **dirty working tree at run time**:
`tree_dirty_file_count: 606 (tasks 0,1,2,3) / 607 (task 4)`. `081b1f28` is the *pinned commit*;
the actual code that ran on the cluster differed from that commit by 606–607 uncommitted files,
in a way this local, read-only pass cannot audit (the dirty diff lived on the remote cluster
checkout at run time and was never captured to a patch or provenance artifact — already flagged
as a bare, undiagnosed fact in `READOUT_RECORD.md`, not resolved there either). Given that (i)
every named candidate is ruled out at the *committed* code level and (ii) the CLI/config is
identical, **the dirty-tree gap is the most likely repository for the actual cause** — but it
cannot be confirmed or refuted without either the cluster's dirty diff at run time or a
git-bisected re-run, both out of scope for this pass (no pipeline runs, no cluster).

**What would discriminate:** (i) `git diff` (or `git stash show -p`) on the cluster checkout at
the time of the S0-B submission, if still recoverable; (ii) a clean-tree re-run of the same
node/seed/config at commit `081b1f28` and a byte-diff against this CSV — if it reproduces 449,
the dirty files are exonerated and the cause is somewhere in the ~2000-line composite diff
across the 9 `bayesian_statistics.py` commits not exhaustively hand-traced here (deep call sites
inside `single_host_likelihood`/`single_host_likelihood_batch`); if it reproduces 606, the dirty
tree is the cause.

## 5. Three-valued existence contract

| file | status |
|---|---|
| `results/.../fanout1_20260829/b3_pop_prediction.json` | PRESENT |
| `results/.../fanout1_20260829/b3_1_pop_measure.py` | PRESENT |
| `results/.../headreadout_20260827/iiib/event_likelihoods.csv` | PRESENT (65108 rows, 41 h-nodes) |
| `results/.../headreadout_20260827/iiib/run_metadata_21.json` | PRESENT (full `cli_args`, `git_commit=d04d9dc9`) |
| `results/.../graph1_20260901/retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/.../event_likelihoods.csv` | PRESENT (1588 rows, h=0.73 only) |
| `.../s0b_run_20260902/provenance_6779532_4.json`, `_6779535_0.json` (and 3 siblings) | PRESENT |
| `.../s0b_run_20260902/run_metadata.json` (full CLI-args JSON for S0-B) | **ABSENT** — no such file exists; the per-task `provenance_*.json` + `s0a_full_output.json` + `.out` logs are the only run-config record (already noted in `READOUT_RECORD.md`) |
| dirty-tree diff for the S0-B cluster checkout at run time | **ABSENT / UNRECOVERABLE** from this local, read-only pass — only the bare `tree_dirty_file_count` is recorded, not the diff content |
| `PREREGISTRATION_HIER_HTHETA_20260826.md` (lines ~2026-2032, ~2096) | PRESENT, not independently re-quoted here — the `b3_pop_prediction.json`/`WAVE2_REGISTRATION_CHECK_20260829.md` provenance chain was traced directly to the source CSV instead |
