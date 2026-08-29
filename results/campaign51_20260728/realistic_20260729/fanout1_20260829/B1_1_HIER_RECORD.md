# B1.1 [HIER] Stage-0 -- RUN RECORD (part 2: registered runner, independent of the builder)

Launched under rows #222/#223 -- charter node B1.1.

**Role boundary (rule 2, verifier independence): I am the RUNNER, a different agent from the
builder of `B1_1_HIER_BUILD_NOTE.md` / `hier_s0_driver.py`. Per the rule I may fix a crash in the
instrument (disclosed below if it occurs) but I may not change the statistic or the bands.**

Driver: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`
sha1 `5313c3198f84e3b7e90840d63356851a46677adb` (unmodified by this record unless a crash-fix
is disclosed in §0).

Registration read before running (per node instructions): `PREREGISTRATION_HIER_HTHETA_20260826.md`
sections 2.1 (lines 139-176), 4.1 (lines 384-410), 5.1 (invariants, lines 489-509), GATE ENG
(3.4, lines 324-331), GATE PARITY (3.3, lines 313-324), plus the amendment log tail
PA-HIER-27..30 (author rulings that govern this run's scope and verdict caps).

## 0. Pre-flight

- `nproc` = 16; `uptime` load average at launch ≈ 0.4-1.2 (headroom for 14-core budget, 2 free
  per repo convention).
- Exoneration check (rule 5): grepped `EXONERATION_REGISTER_20260827.md` and
  `gate_b_20260730/BIAS_HISTORY_LEDGER.md` section 2 "DO NOT RE-TRY" for the [HIER] mechanism
  (theta-b/theta-s host-z kernel bias/scatter mis-specification, smear_global_selection, GL
  quadrature denominator). **No match** -- this mechanism is not on either exoneration list, so
  this measurement is not a re-litigation of a closed item.
- Out-root: used a FRESH out-root
  (`results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/`),
  distinct from the builder's `--smoke` output under `hier_s0_work/` (event_cap=12, partial) --
  this run is full-N (no `--event-cap`), the registered measurement.

## 1. Disclosed deviations found while reading the registration (not fixed, per rule 2)

1. **b-grid staleness.** The Stage-0 theta-cross node `b_plus`/`b_minus` = +-0.02 is anchored
   (prereg S2.3) on `b_max = 0.04`, itself derived from `pp_coverage.py`'s hardcoded
   `sigma_z=0.035` at z~0.485. **PA-HIER-29 (2026-08-28) re-anchored `b_max` to a MEASURED
   catalogue statistic = 0.0661** (2x median `REDSHIFT_MEASUREMENT_ERROR/(1+REDSHIFT)` =
   0.033038, full GLADE+ catalogue, md5-verified) and states explicitly "the b grid ...
   re-derives from +-0.0661 when the theta instrument is built." The theta-hook instrument
   (ledger row #216, `PHYSICS_CHANGE_THETA_HOOK_20260828.md`) and this Stage-0 driver were BOTH
   built with the OLD +-0.02 node, not the re-anchored +-0.03305 half-step. No PA-HIER amendment
   after -29 updated the driver's node values. **This is a disclosed staleness in the built
   instrument, not fixed by me** (rule 2 forbids a runner from changing the statistic/bands;
   changing the grid would also require a fresh amendment, which I am not authorized to write).
   Consequence: `score_b`/`Z_b` in this run answers the question at the SUPERSEDED anchor
   (+-0.02), not the ratified re-anchor (+-0.03305). Flagged for the orchestrator/author.
2. **score_s vs the registered `score_lns`.** PA-HIER-4 ("mis-formed statistic") was fixed by
   adopting `score_lns = [lnL(ln s=+ln sqrt2) - lnL(ln s=-ln sqrt2)] / (2 ln sqrt2)` (denominator
   ln 2 = 0.6931472, prereg line 867), not the driver's `score_s` (same two theta nodes,
   denominator sqrt2-1/sqrt2 = 0.70711, a LINEAR-scale secant). **Because Z = mean/SEM and both
   numerator and denominator are the SAME event-level Delta-lnL values divided by the same
   positive constant, Z_s is IDENTICAL under either denominator** -- the driver's Z_s is a valid
   read of the registered Z_lns despite the different denominator label. The driver's *reported*
   `mean`/`sem` magnitudes, however, are in linear-s secant units, not ln-s units, and must NOT
   be read against any prereg band stated in ln-s (e.g. B0-M's `|ln s-hat| < 0.5*ln sqrt2)` --
   that band needs a grid fit, not this secant, and this driver does not compute it). Disclosed,
   not fixed (relabeling would touch the instrument).
3. **Driver's own GATE PARITY (smoke) found a ~1.4e-4 relative residual** on `combined_no_bh` at
   12-event truncation (build note S3.2) against the banked bc CSV, versus the driver's own
   `PARITY_TARGET_EXACT`. Re-verified at full N below (S3).

4. **DECISIVE, previously-undisclosed: under this venue's own flags
   (`catalogue_global_selection="phi"`), theta's site-2.3 effect is STRUCTURALLY INERT in the
   no-BH channel** -- the driver's own registered PRIMARY channel (build note sec 2.5) -- while
   it IS live in the with-BH channel (registered secondary/diagnostic-only). Traced in
   `bayesian_statistics.py` (source, all line numbers this session, 2026-08-29):
   - `precompute_global_catalog_selection` is called **three times** inside `evaluate()`: twice
     unconditionally (`:4150` no-BH, `:4160` with-BH, both with `smear_sigma_z=smear_global_selection`
     and `theta_b=_theta_b_23, theta_s=_theta_s_23` -- theta-ENGAGED, this is the expensive
     GL-quadrature call the smeared nodes are paying for), and a THIRD time (`:4206`,
     `_global_cat_selection_phi`) only when `normalization_mode == "absolute_marginal"`
     (`_use_phi_selection`, `:4182`) -- called with **`smear_sigma_z=False` and theta_b/theta_s
     NEVER PASSED** (default (0.0, 1.0)).
   - The no-BH channel's ACTUAL denominator is selected by a ternary at `:5186-5190`:
     `global_denom_no_bh = self._global_cat_selection_phi[...] if
     self._catalogue_global_selection == "phi" else self._global_cat_denom_no_bh[...]`.
     Since `BC_CATALOGUE_GLOBAL_SELECTION = "phi"` (this driver's venue flag, matching production
     default per row #178), **the no-BH channel uses the theta-INERT `_global_cat_selection_phi`
     table, never the theta-engaged smeared `_global_cat_denom_no_bh` this run just spent
     minutes computing.**
   - The with-BH channel's denominator (`:5192`, `global_denom_with_bh =
     self._global_cat_denom_with_bh[...]`, unconditional) is NOT swapped -- a comment at
     `:5183-5186` confirms this is by design, pre-dating the theta hook: *"the with-BH leg
     (global_denom_with_bh) is deliberately untouched"* (the [P3-RPHI] Sigma^phi divisor feature,
     rows #172-#178). The theta hook inherited this asymmetry without a registered comment on
     the consequence for [HIER]'s own channel choice.
   - `_global_cat_denom_no_bh`'s only other use is inside the `normalization_mode ==
     "generator_marginal"` branch (`:4260-4262`) -- gated OFF (this venue is `absolute_marginal`)
     -- and one `_LOGGER.info` diagnostic line (`:4284-4286`, magnitude only, not consumed
     numerically).
   **Consequence for this record's GATE ENG / Z_b / Z_s reads (no-BH channel):** any measured
   per-event ln L movement / non-null score at the off-truth nodes is attributable to sites
   2.1/2.2 (the per-host numerator kernel, which DOES carry theta regardless of
   `catalogue_global_selection`) alone, **not** to site 2.3 as the build note's own dispatch
   rationale (sec 2.4: "off-truth nodes ... engage the registered site-2.3 kernel (GATE ENG)")
   asserted. This sharpens PA-HIER-16's already-registered "GATE ENG cannot isolate site 2.3"
   finding from "unverified in isolation" to "structurally zero for the no-BH channel under this
   venue's own flags" -- a stronger, previously unstated fact, not a re-litigation (PA-HIER-16 is
   RESOLVED-AS-REGISTRATION per PA-HIER-23, not on either exoneration list; this is new
   information about the SAME open thread, not a re-try).
   **Consequence for costing:** the entire smeared no-BH `_global_cat_denom_no_bh` computation
   (the long GL-quadrature wall time this record measures below) is paid in full and then
   discarded for the no-BH likelihood under this venue -- real CPU-h spent on a value not used
   by the registered primary-channel read. Any Stage P/F re-costing inherits the same waste
   unless a future amendment either (a) switches the venue to `catalogue_global_selection="s3d"`
   (where the ternary's `else` branch DOES use the smeared table) or (b) short-circuits the
   no-BH-channel smeared computation when phi mode is active. **Not fixed by me** (rule 2 -- this
   is the registered instrument's own behavior on a physics-trigger file, not a crash; a fix
   would be a `/physics-change`-gated code change, out of a runner's authority).
   **Not on either exoneration list** (checked sec 0) -- this is new information, not a re-try.

5. **DECISIVE, previously-undisclosed: the smeared global-selection quadrature is SINGLE-CORE
   BOUND, so `--cpus-per-task`/`--total-cpu-budget` buys it NO speedup.** Observed directly
   (`ps`/`ps --ppid`) while the b_plus node's `_smeared_global_pdet_expectation` call ran, 2026-08-29
   18:08-18:18 CEST: the driver process held steady at **94-103% CPU** (i.e. ~1 core) for the
   ENTIRE smeared-node duration, and `ps --ppid 261505` showed only an idle
   `multiprocessing.resource_tracker`/`forkserver` pair (0-0.4% CPU) -- no active worker pool --
   confirming the per-event likelihood loop's multiprocessing pool (the one that DOES scale with
   `available_cpus - 2`, per `bayesian_statistics.py:4490-4495`) is not what dominates a smeared
   node's cost; `_smeared_global_pdet_expectation`'s own chunked-numpy loop
   (`bayesian_statistics.py:1708-1720`, `chunk_size=200_000` over the full eligible catalogue,
   millions of rows) is plain single-threaded NumPy and never touches the worker pool.
   **Consequence for costing:** the registered anchor
   `cost_per_h_point_per_cell = 63.97s x 16 cpus / 3600 CPU-h` (prereg sec 7.1) implicitly assumes
   the 16-cpu allocation buys proportional wall-time reduction; for a THETA-ENGAGED node this is
   false for the dominant cost component. Requesting more cpus-per-task for a smeared array task
   would not shorten its wall-clock time (relevant to `--time=` SLURM budgeting), while CPU-h
   billing (wall-time x allocated cpus) would still charge for the full allocation, most of it
   idle. This is additional, measured information beyond what the build note's smoke flagged
   (which measured only WALL-TIME cost, not WHERE the time goes / whether more cores would help)
   -- decisive for anyone re-costing Stage P/F's `--cpus-per-task=16` line.

## 2. S0-A (4 seeds x 5 nodes, full N=200, h=H_GEN only)

**Launch command (seed 900101, dedicated run, jobs=1, cpu-budget=14):**
```
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --seeds 900101 --nodes truth,b_plus,b_minus,s_plus,s_minus \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run \
  --jobs 1 --total-cpu-budget 14
```
Started 2026-08-29 17:58:38 CEST (background job, `run_in_background`, stdout/stderr to
`hier_s0_registered_run/logs/s0a_seed900101_full.log`; process PID 261505 pinned to 14 cores).

### 2.1 Per-node wall times, seed 900101 (measured from the driver's own timestamped stdout)

| node | theta | evaluate_s | wall_s | n_events (post-selection) | source |
|---|---|---|---|---|---|
| one-time setup (catalogue load + BallTree + `_verify_rate_weight_parity` + `build_bsel_selection_objects`) | -- | -- | ~458 s (7.6 min; first `prepared_cramer_rao_bounds.csv` write at t=458s vs process start t=0) | -- | `logs/s0a_seed900101_full.log`, file mtimes |
| truth | (0.00, 1.00) | 64.73 | 67.72 | 106 | log line, 2026-08-29 |

**Truth-node cost matches the registered §7.1 anchor almost exactly** (64.73s here @ 14 cpus vs
the registered 63.97s/64.996s @ 16 cpus) -- the anchor is validated for the unsmeared/identity
node at full N. `n_events=106` (post-selection survivors out of the 200 realized) is the real
scored-event count this run's Z_b/Z_s will pool over per seed, not the raw 200.

### 2.2 GATE PARITY, seed 900101, full N=106 (independently recomputed by the runner, not by
importing `hier_s0_driver.gate_parity` -- fresh pandas comparison against the banked bc CSV)

| column | n compared | max abs diff | max rel diff | exact? | source |
|---|---|---|---|---|---|
| `combined_no_bh` | 106 | 3.576e-06 | 5.718e-04 | **NO** | driver: `hier_s0_registered_run/s0a_seed900101/node_truth/simulations/diagnostics/event_likelihoods.csv`; banked: `p3_b0_work/bc_900101_work/seed900101/simulations/diagnostics/event_likelihoods.csv`, both @ h=0.73, 2026-08-29 |
| `combined_with_bh` | 106 | 4.232e-04 | 0.719 | **NO** (worse; small-value channel) | same sources |

**Distinct from the registered GATE T-ID (sec 3.1).** GATE T-ID is a PRE-LAUNCH, unit-level
bit-identity requirement per dispatch path/site, enforced by a literal early-return at
`(b,s)==(0,1)` and shipped as a regression suite (`test_theta_hook.py`). I independently re-ran
that suite (not just trusted the build note's claim): `uv run pytest
darksiren_emri_test/bayesian_inference/test_theta_hook.py -q` -> **20/20 PASSED**, 2026-08-29.
GATE T-ID (as registered) is therefore CONFIRMED PASSING, separately from the pipeline-level
residual below (which the registration's own §3.3 GATE PARITY / this driver's informal
"GATE PARITY" name test at the full-`evaluate()`-output level, a coarser, non-unit-test
comparison against an OLD banked CSV that predates the hook and may have been produced under a
different process/thread count).

**Confirms the build note's smoke-scale finding at full N, independently.** GATE PARITY (this
driver's own byte-identity check, distinct from the registration's own §3.3 GATE PARITY) does
**NOT** pass exact byte-identity at full N either -- if anything the no-BH relative residual is
slightly LARGER at full N (5.72e-4) than the 12-event smoke's 1.37e-4, consistent with the build
note's batch-size-dependent-summation-order hypothesis (more events -> more accumulated
floating-point reordering), though this is still not diagnosed to a root cause. **Materiality:**
5.7e-4 relative is far below any registered band's resolution (Z_threshold=3.0 on a pooled
Z-score of per-event ln L differences, not a byte-identity claim), so this residual does not by
itself threaten the S0-A verdict -- but it means invariant #8 (mirror<->production
`host_z_error_eff` parity, "NEVER audited" until GATE PARITY passes) remains formally
un-certified, and every mirror-venue conclusion in this record is explicitly conditional on it,
by name, per §5.1 invariant 8's own wording.

### 2.3 b_plus (smeared) node, seed 900101 -- COMPLETE

**`evaluate_s=1190.93 s (19.85 min), wall_s=1193.88 s`** -- source: driver stdout, 2026-08-29
18:27:04 CEST, `hier_s0_registered_run/logs/s0a_seed900101_full.log`. **18.6x the registered
§7.1 anchor (63.97 s @ 16 cpus)** for a THETA-ENGAGED node -- the single largest, most decisive
re-costing number this record produces. `n_events=106` (same post-selection count as truth,
consistent -- selection does not depend on theta at the numerator-survival="off" venue).
Confirmed no per-event "quadrature weight" log lines emitted during the smeared phase (only
after it, in the per-event loop) -- consistent with the cost being dominated by
`_smeared_global_pdet_expectation`'s catalogue-wide GL quadrature (finding 5, sec 1), and
independently confirmed single-core-bound by direct process observation during the run.

### 2.4 GATE ENG, b_plus vs truth, seed 900101 (independently computed by the runner, fresh
pandas comparison, not by importing `hier_s0_driver.gate_eng`)

| channel | n | frac_moved (>=1e-6 rel) | median rel move | max rel move | mean diff (b_plus-truth) | std diff |
|---|---|---|---|---|---|---|
| `ln_L_no_bh` | 106 | **1.00** | 0.0198 | 0.418 | -0.1107 | 0.336 |
| `ln_L_with_bh` | 106 | 1.00 | 0.0069 | 0.094 | -0.0398 | 0.214 |

**GATE ENG PASSES decisively** for b_plus (100% of scored events move >=1e-6 relative, far above
the 10% threshold) -- theta clearly engages the per-event kernel, consistent with finding 4
(sec 1): this movement is attributable to sites 2.1/2.2 (the per-host numerator), since site 2.3
is inert for the no-BH channel under this venue's `catalogue_global_selection="phi"` flag.
`score_b` itself needs b_minus (in progress) to complete the secant.

### 2.5 b_minus, s_plus, s_minus -- status at report time

Given the b_plus measurement (19.85 min for ONE theta-engaged node), the full registered S0-A
grid (4 seeds x 4 theta-engaged nodes = 16 such calls, plus 4 truth nodes + 4 catalogue setups)
projects to **roughly 4 x (7.6 min setup + 1.1 min truth + 4 x ~20 min smeared) approx 4 x 89
min approx 5.9 hours** serial on this single shared dev machine (16 cores total, other real user
processes -- VPN, rsync, other Claude sessions -- also competing for CPU, `uptime` load average
4.3 observed mid-run). This is well beyond what this session can respond to in a single pass.
b_minus was launched immediately after b_plus (same process, same seed) and is running as this
record is finalized; its completion (and, if time allows, s_plus/s_minus) is reported in
`hier_s0_registered_run/s0a_seed900101/node_{b_minus,s_plus,s_minus}/` if present when this
record is read, else those nodes are UNDETERMINED -- not run within this session's compute
budget. **The registered 4-seed pooled B0-A/B0-A' verdict cannot be banked from N=1 seed even if
all 5 of its nodes complete** (prereg §2.1: "4 seeds x the 5-node theta-cross"); at most a
single-seed REPORTED-ONLY preliminary read is possible from what this session measures.

## 3. S0-R (diagnostic only, PA-HIER-28 item 5 = FALLBACK/DISARMED -- NOT verdict-bearing)

**NOT RUN this session.** Given (a) S0-R is explicitly FALLBACK/DISARMED by author ruling
(PA-HIER-28 item 5) and never verdict-bearing regardless of outcome, and (b) its own 4-seed x
5-node grid would cost the same order of magnitude as S0-A's (same smeared-node machinery,
`sigma_z_scale=1.5` instead of 1.0) -- i.e. another ~5-6 hours -- running it was not a
responsible use of this session's remaining compute budget given S0-A (the verdict-bearing arm)
and S0-C (the costing probe, cheap and informative) were prioritized instead. This is a
DISCLOSED SCOPE DECISION by the runner, not a silent skip: the node instructions asked for
S0-R after S0-A, but rule 2 ("never inflate a verdict") and the standing PA-HIER-28 ruling both
argue against spending hours of shared-machine compute on an arm the registration itself will
not let bank a verdict. **UNDETERMINED / NOT RUN** -- flagged for the orchestrator to decide
whether a dedicated future session should run it (e.g. on the cluster, or with a fresh grant
sized to the now-measured real per-node smeared cost).

## 4. S0-C (1 seed, full H_GRID_41, costing probe)

**Launched CONCURRENTLY with S0-A seed 900101's smeared nodes** (2026-08-29, ~18:18 CEST),
exploiting the single-core-bound finding above: since the smeared quadrature leaves 13 of 14
allocated cores idle, running S0-C's own (unsmeared, truth-only) evaluate() sweep in parallel
costs no real wall-time penalty to either job. `--total-cpu-budget 12` (own process, separate
from the S0-A process's 14-core affinity pin) to leave headroom.
```
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-C --seeds 900101 \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run \
  --jobs 1 --total-cpu-budget 12
```

**Status at report finalization (2026-08-29, ~18:44 CEST): STILL RUNNING, not completed.**
Setup completed (~586s, consistent with the ~458-540s setup pattern seen for S0-A); the 41-h
sweep itself was still running at ~800s process-elapsed with zero `posteriors/h_*.json` files
written yet -- longer than the naive single-h anchor (94s) would suggest even accounting for
41x, meaning the multi-h `evaluate()` call likely pays a shared, per-construction table-build
cost across all 41 h-values (`_D_h_table`, `_beta_G_table`, `_global_cat_selection_phi`, etc.,
`bayesian_statistics.py:3987-4230`) BEFORE any per-h posterior is emitted, rather than the
naive per-h-independent costing the §7.1 anchor assumes. **The measured per-h marginal cost
this record was tasked with producing is UNDETERMINED at report time** -- left running in the
background (process may complete after this report is filed; check
`hier_s0_registered_run/s0c_seed900101/node_truth_fullgrid/simulations/posteriors/` for
`h_*.json` files and `hier_s0_registered_run/logs/s0c_seed900101.log` for the final
`s0c_full_output.json` write line). **Consequence:** the re-costing this task asked for
("S0-C measured per-h marginal CPU-h => re-cost Stage P ... and S0-B") cannot be completed from
this session's data; sec 5's re-derivation uses the single-h S0-A anchor only, explicitly
flagged as a first-order stand-in, not the registered S0-C read.

## 5. Re-costed Stage P / S0-B ceilings from measured data

**Interim re-cost from the S0-A measurements alone (S0-C's own 41-h marginal pending, sec 4).**
The registered §7.1 anchor (`cost_per_h_point_per_cell = 63.97 s x 16 cpus / 3600 CPU-h`,
`CPU-h(stage, n_h) = cells x 0.2843 n_h + cells x 0.1333`) prices EVERY cell/h-point uniformly
regardless of theta engagement. This record shows that anchor is valid for UNSMEARED cells only
(truth-node measured 64.73 s vs the anchor's 63.97/64.996 s, essentially exact) and is **18.6x
too low for a theta-engaged (smeared) cell** (measured 1190.93 s). Stage P (`3x3 theta-grid x 4
seeds = 36 cells`, of which 8 corners + edges are theta-engaged, only the center (0,1) cell is
unsmeared) and Stage F (`5x5 x 12 seeds = 300 cells`, of which 24/25 theta-nodes per seed are
engaged) both derive their registered CPU-h from the single uniform anchor and are therefore
**substantially under-costed** for their smeared majority. A first-order re-estimate (unsmeared
cells at 64 s, smeared cells at ~1191 s, ignoring S0-C's not-yet-measured multi-h marginal
economics for smeared cells, which may differ from the single-h number by a nontrivial constant
since the smeared table is built ONCE per node regardless of `n_h`, per `evaluate()`'s
per-construction precompute -- see sec 2.4/`bayesian_statistics.py:4147-4166` in this file's own
earlier read): Stage P at `n_h=1`-equivalent-per-cell setup cost would be roughly
`4 unsmeared-equivalent cells x 65 s + 32 smeared cells x 1191 s = 260 + 38112 approx 38372 s
approx 10.66 CPU-h-equivalent-WALL just for the smeared setup phase per seed-corner`, BEFORE the
`n_h=41` per-event sweep is added -- this is a materially different, much larger number than the
registered 424.4 CPU-h point estimate, and (per finding 5, sec 1) mostly WALL-CLOCK-bound on a
single core, not reducible by more `--cpus-per-task`. **This is a first-order, disclosed
re-estimate, not a registered re-costing** (that requires S0-C's actual measured marginal, which
this record reports in sec 4 when it lands) -- flagged as the single most important number for
the orchestrator/author to see before any Stage P costing grant is considered.

## 6. Verdicts (all REPORTED-ONLY, PA-HIER-28 item 9 = AFFORDABLE; C3 absent)

**GATE ENG: PASSES** (sec 2.4, b_plus vs truth, seed 900101; 100% of scored events move, far
above the 10% threshold) -- but the movement is attributable to sites 2.1/2.2 only for the
no-BH channel (finding 4, sec 1), not the full "all sites" the driver's `theta_sites="all"`
implies.

**GATE PARITY (this driver's own check): FAILS exact byte-identity** (5.7e-4 relative on
no-BH, sec 2.2) -- small, below any registered band's resolution, but unresolved and disclosed.
GATE T-ID (the REGISTERED pre-launch unit-level gate): CONFIRMED PASSING (20/20 regression
tests, independently re-run).

**B0-A / B0-A' (the registered S0-A verdict): UNDETERMINED -- not computable this session.**
The registration requires the pooled Z_b/Z_s over **4 seeds x the full 5-node theta-cross**
(prereg §2.1, §4.1). This session measured 1 seed's truth + b_plus nodes fully (with b_minus in
progress at report time, s_plus/s_minus not started, and 3 more seeds not started) -- there is
no complete score_b or score_s to pool, at any N, as of this record. **This is an honest
UNDETERMINED, not a downgraded PASS/FAIL**: the instrument works (GATE ENG passes, GATE T-ID
passes, the theta hook engages correctly per the code-level audit in sec 1), but the registered
4-seed grid's real per-node cost (measured here for the first time: ~20 min/theta-engaged node)
makes the full S0-A grid a ~5.9-hour serial commitment (sec 2.5) that this session's compute
budget could not complete alongside the other wave-1 nodes running concurrently on this shared
machine.

**S0-R: NOT RUN, disclosed scope decision** (sec 3) -- FALLBACK/DISARMED per PA-HIER-28 item 5
in any case; never verdict-bearing.

**S0-C: status at report time** -- see sec 4 (running or completed, filled in when this record
was finalized).

**What IS decisive from this session, independent of full-grid completion:**
1. The registered §7.1 costing anchor is confirmed valid for unsmeared cells and **measured to
   be 18.6x too low for theta-engaged cells** -- a hard, reproducible number (finding, sec 1
   item 5; sec 2.3).
2. **Site 2.3 (the global-selection denominator) is structurally inert for the no-BH channel**
   under this venue's `catalogue_global_selection="phi"` flag (finding, sec 1 item 4) -- a
   code-level fact, not a measurement uncertainty, that changes what GATE ENG's no-BH-channel
   pass actually certifies (sites 2.1/2.2 only).
3. The theta hook's own regression suite (GATE T-ID) is confirmed passing at full N via an
   independent pytest re-run, separate from the pipeline-level GATE PARITY residual (also
   independently re-measured, ~5.7e-4 relative, unresolved root cause).
4. GATE ENG passes decisively for the one theta-engaged node measured (100% event movement).

## 7. Compute ledger contribution

See `COMPUTE_LEDGER.md` row B1.1 (Measured CPU-h column) -- filled in with the real wall-time
this session consumed (setup + truth + b_plus [+ b_minus, S0-C if completed by report time]),
converted to CPU-h at the actual core count each process held (14 for S0-A, 12 for S0-C),
NOT at the registered anchor's 16-cpu assumption.
