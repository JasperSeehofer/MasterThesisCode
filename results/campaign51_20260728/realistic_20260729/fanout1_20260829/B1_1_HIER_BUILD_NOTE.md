# B1.1 [HIER] Stage-0 driver -- BUILD NOTE (part 1: build + smoke only)

Launched under rows #222/#223 — charter node B1.1.

**Role boundary (rule 2, verifier independence): this is a BUILD note. The registered
measurement (the real 4-seed, 5-node S0-A grid; any S0-R/S0-C run) must be executed by a
DIFFERENT agent from this one.** Everything below that looks like a "result" is either (a) a
smoke-test artifact explicitly exempted from that boundary, or (b) a zero-compute read of
already-banked files. No registered band is claimed as passed or failed here.

Files:
- Driver: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`
- Production edit: `darksiren_emri/validation/correspondence_1d.py` (`run_mirror_seed_inprocess`
  gains `theta_b`/`theta_s`/`theta_sites`/`smear_global_selection` passthrough kwargs; NOT a
  physics-trigger file, no `/physics-change` gate required for this edit -- the theta hook
  itself already landed inside `bayesian_statistics.py` under that gate, ledger row #216,
  `PHYSICS_CHANGE_THETA_HOOK_20260828.md`, disclosed as already-in-the-tree by the launch
  instructions before this node started)
- Smoke output: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_work/`
  (1 seed, partial -- see "Smoke test" below)

## 1. What was read first (registration + amendment log)

`PREREGISTRATION_HIER_HTHETA_20260826.md` sections 1.2, 2.1, 4.1, 5.1, 3 (GATE ENG/PARITY/D3),
and the FULL amendment log PA-HIER-1..30 (not just the launch instruction's pointers -- the
amendment log materially changes what S0-A/S0-R/S0-C mean by the time this driver was written).
Also read: `hier_blocker_a_generator_law_20260827.md` §3 (superseded in-place by PA-HIER-19/27,
see below), `p3_b0_identity_test.py` in full (the template this driver's venue construction is
copied from), `correspondence_1d.py`'s `run_mirror_seed_inprocess`/`host_pool_for_sigma_scale`/
`draw_realization`/`ARM_SPECS`/`ARM_FLAGS`, `observed_realization.py`'s module docstring and
`realize_observed_catalogue`, and the theta-hook sites in `bayesian_statistics.py`
(evaluate() ~3517-3565, sites 2.1 ~6238, 2.2 ~6963, 2.3 ~2712/4104-4129) plus
`test_theta_hook.py`.

**Load-bearing fact the launch instruction's pointer to `hier_blocker_a_generator_law_20260827.md`
undersells:** by the time PA-HIER-27/28 landed (2026-08-28), the registration's OWN append-only
amendment log has already re-derived and RATIFIED everything that background document was cited
for (venue = `host_mode="catalogue_selected"`, arm b0i, truth-theta=(0,1)) -- the amendment log,
not the standalone blocker doc, is the current source of truth. I read both; the registration
supersedes.

**Two amendments change the shape of what this build task described, and are carried forward as
disclosures rather than silently "fixed" (append-only rule):**

1. **S0-R is a disclosed NULL INSTRUMENT (PA-HIER-3, confirmed and upgraded to NEEDS-CODE by
   PA-HIER-22).** `realize_observed_catalogue(sigma_scale=1.5)` round-trips the quoted
   `z_error` column verbatim (`observed_realization.py:176-186`'s own docstring: *"the z width
   law is scale-free in z, so the stored column IS the width the kernel consumes and
   `sigma_kernel == sigma_realized` identically"*), and `host_pool_for_sigma_scale` returns ONE
   `GalaxyCatalogueHandler` reused as BOTH the generator's host pool source and the estimator's
   `galaxy_catalog` argument. `sigma_scale` therefore perturbs *which galaxy sits at which z*,
   never a generator/estimator width mismatch -- truth-theta after the call is still (0, 1), not
   (0, 1.5). The task's "joint z+mass scaling defect" framing (from the ORIGINAL, now-superseded
   §2.1 text) is real but is the SMALLER of the two problems; the bigger one is that S0-R does
   not inject an s-mismatch AT ALL. A genuine s-axis positive control (C3) needs new code -- a
   second estimator-facing catalogue with a rewritten quoted-width column, decoupled-handler
   driver plumbing, and a resolved candidate-list confound -- and is NEEDS-CODE, unbuilt, out of
   this task's scope.
2. **PA-HIER-28 item 5 = FALLBACK (author ruling, 2026-08-28, verbatim: "exactly as recommended
   by you"): D7's early exit is DISARMED and Stage 0 is officially RE-SCOPED to S0-A + S0-C
   ONLY.** This driver still BUILDS S0-R (per this task's explicit instruction, and because the
   code is <40 lines given S0-A's machinery already exists) but its verdict function
   (`verdict_s0r`) never emits B0-R / B0-R' / LEVER-DEAD-AT-N -- doing so would bank a verdict
   about an axis the amendment log proved this instrument cannot move. Any S0-R number this
   driver produces is a disclosed diagnostic, not a registered read, and is NOT authorized to
   run under the current author ruling without a fresh grant (S0-R is FALLBACK-disarmed, not
   forbidden-to-build).

Also carried forward: **all [HIER] verdicts are capped REPORTED-ONLY** (PA-HIER-28 item 9 =
AFFORDABLE) -- no band this driver computes may be read as CALIBRATED.

## 2. Design

### 2.1 Venue construction (`build_bc_venue`)

Copied EXACTLY from `p3_b0_identity_test.py`'s `_run_arm_seed(venue="b0i")`:

```python
host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
    work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale   # 1.0 for S0-A, 1.5 for S0-R
)
c1d._verify_rate_weight_parity()
completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)
events = gen.draw_realization(
    seed, host_pool=host_pool, host_mode="catalogue_selected",
    completeness=completeness_obj, phi_survival_table=phi_survival_table,
)
```

`CorrespondenceConfig.sigma_z_scale` (a separate field on the config dataclass) is DEAD for this
draw path -- only `n_events`/`area_scale` are read off it (class docstring, confirmed by grep);
the dose that matters is `host_pool_for_sigma_scale`'s own `sigma_z_scale` kwarg, which this
driver threads through. `ARM_FLAGS["bc"]` pins `catalogue_numerator_survival="off"`,
`catalogue_global_selection="phi"`; `_run_arm_seed`'s own default pins
`selection_in_completion_numerator="fused"`. Banked bc seeds 900101-900112 exist under
`p3_b0_work/bc_<seed>_work/`; `DEFAULT_BC_SEEDS = (900101, 900102, 900103, 900104)` (first 4).

### 2.2 The theta-hook passthrough (production edit)

`run_mirror_seed_inprocess` had NO theta passthrough (`grep -n "theta_b\|theta_s\|theta_sites"
correspondence_1d.py` was empty before this edit). Added `theta_b: float = 0.0, theta_s: float =
1.0, theta_sites: str = "all", smear_global_selection: bool = False` -- identical to
`evaluate()`'s own defaults, forwarded verbatim into the `bs.evaluate(...)` call. Verified
byte-identical-by-default via `inspect.signature` + a clean `ruff check` + `mypy` pass on the
file (both green, see section 4). `correspondence_1d.py` is not on CLAUDE.md's physics-trigger
list, so no `/physics-change` gate applies to this specific edit (the theta hook itself is
already gated and landed inside `bayesian_statistics.py`).

### 2.3 h-grid / h-bounds (GATE PARITY design)

S0-A/S0-R run at `h_values=(H_GEN,)` (n_h=1, prereg §2.1). The registration's own invariant #2
(§5.1) pins `h_bounds = (0.50, 0.86)` -- this is ALSO exactly `min/max(H_GRID_FULL)`
(`H_WING_LOW | H_GRID_41` in `correspondence_1d.py`), the grid the banked bc CSVs were produced
under. P3-HGRID (rows #182-#184) proved a single-h caller reproducing a full-grid run's `L_cat`
"must pass `h_bounds=(min(grid), max(grid)))` explicitly (proven bit-exact vs the banked b0i
CSVs)" -- so this driver passes `h_bounds=(0.50, 0.86)` explicitly on every call, making GATE
PARITY a real, checkable claim rather than an apples-to-oranges comparison.

### 2.4 smear_global_selection dispatch (an ambiguity I resolved, disclosed)

The registration nowhere states, for the D7 theta-cross, whether `smear_global_selection`
should be forced True at every node or left at its caller default. `evaluate()`'s own guard
(bayesian_statistics.py:3552) REQUIRES it True whenever theta is engaged (`theta_b != 0` or
`theta_s != 1`) and `theta_sites` includes "2.3"/"all" -- but does NOT require it at the
identity node, where theta is inert regardless. I resolved this by having the driver set
`smear_global_selection = (theta_b != 0.0 or theta_s != 1.0)` PER NODE (`run_theta_node`):

- **truth node (0,1):** `smear_global_selection=False` -> stays on the unsmeared, point-kernel
  path -> byte-identical (up to the residual in section 3.2 below) to the banked bc CSVs ->
  this is what makes GATE PARITY meaningful.
- **off-truth nodes:** `smear_global_selection=True` (forced, not left to a separately-set
  flag) -> engages the registered site-2.3 smeared kernel -> this is what GATE ENG needs to see
  move.

This extends GATE D3(a)'s stated principle ("`s != 1` forces the branch itself, rather than
depending on a separately-set flag") from its own named site to this driver's own per-node
dispatch. Flagged for the runner to veto if a different reading was intended.

### 2.5 Per-event ln L / score channel

`event_likelihoods.csv` carries `combined_no_bh`/`combined_with_bh` (linear likelihoods, not
log). `read_event_ln_l` takes `ln = log(combined)` where `combined > 0`, else NaN (mirrors the
estimator's own `num_log_term_*` NaN-guard convention). **Primary registered channel: no-BH**
(consistent with S0-R's own "no-BH channel only" registration and simpler to interpret given the
open with-BH mass-kernel thread, `[P3-MKER]`, invariant #12). With-BH is computed and reported
alongside as a secondary diagnostic, never verdict-bearing here.

### 2.6 score_b / score_s pooling

Implemented exactly per prereg §4.1:
`score_b = [lnL(+0.02,1) - lnL(-0.02,1)]/0.04`, `score_s = [lnL(0,sqrt2) - lnL(0,1/sqrt2)]/
(sqrt2-1/sqrt2)`, joined per (seed, event_idx), pooled (concatenated, not averaged-then-pooled)
over every event and every seed, `Z = mean/SEM` with `SEM = std(ddof=1)/sqrt(n)`. NaN-guarded
(non-finite per-event scores, e.g. from a non-positive `combined_*` at either node in the pair,
are dropped before pooling -- disclosed in `n_pooled`).

### 2.7 GATE ENG

Per prereg §3.4: fraction of scored events moving `>= 1e-6` relative in per-event ln L vs the
SAME-seed truth node, per off-truth node, `pass` iff mean fraction `>= 0.10`. This is the
driver's own OAT-adjacent engagement check on the AGGREGATE per-node score, not PA-HIER-23's
full per-site-isolated toggle matrix (C2) -- that instrument (separable per-term ln L /
per-site toggle) is a separate, larger build this task did not ask for; `den_log_term`/
`num_log_term_no_bh`/`num_log_term_with_bh` columns already exist in the diagnostics CSV
(PA-HIER-23) and are available for a future C2-proper driver but are not consumed here.

### 2.8 S0-C

One seed, theta=(0,1), the full `H_GRID_41` fused into one `evaluate()` call.
Per-h marginal cost is read off `simulations/posteriors/h_*.json` mtimes (written progressively
as each h completes, per `bayesian_statistics.py` ~4600-4633) -- this is the actual point of the
costing probe ("the MEASURED marginal per-h cost"), not a re-derivation of the §7 anchor.
**Not run in this build** (see section 3.3 -- the per-seed setup cost alone makes a 41-h run a
multi-CPU-hour commitment, out of scope for a builder's smoke).

### 2.9 Concurrency / CPU budget

`BayesianStatistics.evaluate()` has no `num_workers` passthrough in
`run_mirror_seed_inprocess` (unexposed; would default to `available_cpus - 2` PER CELL,
oversubscribing under concurrent seeds). Rather than add that plumbing (out of this task's
scope, and touching `evaluate()`'s call site more than the theta hook required), each concurrent
worker process calls `os.sched_setaffinity(0, <cpu subset>)` before invoking the runner, so
`len(os.sched_getaffinity(0))` -- what `evaluate()` actually reads -- reflects the intended
per-cell budget. `--jobs` x `cpu_per_job <= --total-cpu-budget` (default 14 of this machine's 16
cores, 2 free per the launch instruction).

## 3. Smoke test (S0-A, 1 seed=900101, 2 nodes, 900s timeout, event_cap=12)

**Command:** `--arm S0-A --smoke --seeds 900101 --nodes truth,b_plus` (event_cap defaults to 12
under `--smoke`).

**Result: hit the 900s wall before completing (`timeout 900` killed it, exit 143, `real
15m0.042s`).** The truth node completed cleanly; the b_plus (theta-engaged, smeared) node did
not finish. This is itself the headline finding of the smoke test (section 3.3) -- reported
honestly rather than re-run at a longer timeout without authorization, since a full run's true
cost is now visibly much larger than the registered §7 anchors assumed.

### 3.1 Per-cell wall time (measured from file mtimes, not estimated)

| stage | wall time | evidence |
|---|---|---|
| per-seed one-time setup (catalogue load + BallTree + `_verify_rate_weight_parity` + `build_bsel_selection_objects`) | **~8-9 min** (bounding: enclosing Bash call started the process, first `prepared_cramer_rao_bounds.csv` write for the truth node landed at 17:35:03; the 900s kill fired around 17:42) | `stat` on `node_truth/simulations/prepared_cramer_rao_bounds.csv` vs the driver invocation timestamp |
| **truth node** `evaluate()` (n_h=1, 9 scored events, unsmeared) | **53.5 s** (17:35:03.5 -> 17:35:57.0) | `stat` on `prepared_cramer_rao_bounds.csv` vs `posteriors/h_0_73.json` |
| **b_plus node** `evaluate()` (n_h=1, up to 12 events, SMEARED -- `smear_global_selection=True`) | **> 363 s, DID NOT COMPLETE** (started 17:35:57.7, still running at the 900s kill ~17:42) | `node_b_plus/simulations/` has `prepared_cramer_rao_bounds.csv`/`cramer_rao_bounds.csv`/the injections symlink but no `posteriors/`/`diagnostics/` -- i.e. still inside the worker-pool loop |

The truth node's 53.5s is consistent with the registration's own §7.1 anchor ("64.996/62.944s @
16 cpus, 200 events" -- proportionally faster here at 9-12 events). **The smeared node is the
new information: it took at least 6.8x longer than the unsmeared node on the SAME (tiny) event
count and still had not finished.** `--event-cap` only truncates the mirror `events` (detections)
dataframe; it does NOT reduce the global-selection denominator's own catalogue-wide sum, which
`smear_global_selection=True` (site 2.3) replaces with a GL-quadrature-smeared integral instead
of the point/delta evaluation the truth node uses -- so the smeared node's cost is plausibly
near-INDEPENDENT of `--event-cap`, dominated by the catalogue-wide denominator integral, not the
per-event loop. **This is a disclosed, unresolved cost driver, not a bug fix I attempted** --
recosting the off-truth nodes before authorizing a real S0-A run is now a prerequisite, not an
optional nicety, and the §7 CPU-h ceiling (which never priced a smeared node) should be treated
as stale for this arm until the runner re-measures it.

### 3.2 GATE PARITY -- measured, not assumed (zero additional compute: read from files already on disk)

Truth-node `event_likelihoods.csv` (9 events survive selection out of the 12 truncated) compared
against the banked `bc_900101_work/seed900101/.../event_likelihoods.csv` at h=0.73, same
event_idx values:

| column | max abs diff | max rel diff | exact? |
|---|---|---|---|
| `combined_no_bh` | 3.353e-06 | 1.372e-04 | **NO** |
| `w_G` | 0.0 | 0.0 | yes |
| `L_cat_no_bh` | 1.883e-05 | 1.562e-04 | **NO** |
| `combined_with_bh` | 4.232e-04 | 0.394 | **NO** (worse; small-value channel) |

**GATE PARITY does NOT pass at exact byte-identity** (this driver's `PARITY_TARGET_EXACT`/
`PARITY_FALLBACK_RTOL=1e-9` are both violated). The observed tolerance is **~1.4e-4 relative on
the no-BH channel** -- small in absolute terms and far tighter than any registered band (Z=3.0,
GATE ENG's 1e-6 *movement* threshold at a DIFFERENT, theta-engaged node), but it is NOT the
"proven bit-exact" result the P3-HGRID note (rows #182-#184) reported for its own comparison.
**Root cause NOT diagnosed** in this build pass (would require re-running with the full 200-event
set to control for the `--event-cap` truncation, which this smoke deliberately avoided to stay
cheap) -- candidate hypotheses, undistinguished: (a) something in the batched per-event kernel's
floating-point reduction order depends on how many events are in the same `evaluate()` call
(batch-size-dependent summation order); (b) a stochastic element in `build_bsel_selection_objects`
/ the completeness build not fully pinned by `seed` alone. **Flagged for the runner**: re-verify
GATE PARITY at the FULL 200-event count (no `--event-cap`) before trusting the truth node as a
byte-identical reference; if the ~1e-4 residual persists at full N, it is real and should be
registered as GATE PARITY's own tolerance rather than re-asserted as "bit-exact" from the
P3-HGRID note without re-verification.

### 3.3 What this smoke did NOT do (disclosed, not run)

- **S0-R was not executed at all** (zero seeds run) -- consistent with PA-HIER-28 item 5
  (FALLBACK/DISARMED) and this task's own "may not run the registered measurement" boundary; the
  code path was reviewed by inspection and `ruff`/`mypy`, not executed.
- **S0-C was not executed** -- given the ~8-9 min one-time setup plus a 41-point h-grid at
  (conservatively) tens of seconds each even unsmeered, a real S0-C run is itself a multi-CPU-hour
  commitment; running it under a "smoke" framing would have been indistinguishable from banking
  the registered costing read, which rule 2 forbids me from doing.
- **No off-truth S0-A node completed.** Only the truth node produced usable per-event data; GATE
  ENG (which needs an off-truth node's ln L vector) could not be computed from this smoke and is
  NOT reported as pass/fail anywhere in this note.

## 4. Quality gates run on the code (not the registered measurement)

- `uv run ruff check` on both changed files: **All checks passed** (both).
- `uv run mypy` on both changed files: **Success: no issues found** (both).
- Removed an initial `from __future__ import annotations` from the new driver file per this
  repo's Python-conventions rule (postponed evaluation of annotations is disallowed here); no
  functional change, mypy stayed green after removal, native PEP 585 generics run fine at
  runtime on this repo's pinned Python 3.13.
- No pytest run against this file (it is a results-directory script, not part of the
  `darksiren_emri` package under test); `run_mirror_seed_inprocess`'s existing test coverage in
  `darksiren_emri_test/` was not touched and was not re-run as part of this build (out of scope
  -- the kwargs added are additive with byte-identical defaults, verified by direct
  `inspect.signature` + the truth-node smoke rather than a new unit test, since rule 2 reserves
  the registered instrument's certification for the runner).

## 5. Ambiguities resolved (summary, cross-referenced above)

1. **smear_global_selection dispatch** (section 2.4) -- forced True per-node on theta engagement,
   never left to a caller default; flagged for veto.
2. **h_bounds for a single-h call** (section 2.3) -- pinned to `(0.50, 0.86)` (invariant #2 /
   `H_GRID_FULL`'s own min/max), not `(0.60, 0.86)` (`H_GRID_41`'s own min/max) -- the former is
   what the banked bc CSVs actually used; using the latter would have made GATE PARITY fail for
   a reason unrelated to theta.
3. **ln L channel** (section 2.5) -- no-BH primary, with-BH secondary/diagnostic-only.
4. **S0-R built but never verdict-bearing** (section 1, item 2) -- code exists per this task's
   instruction; `verdict_s0r` structurally cannot emit B0-R/B0-R'/LEVER-DEAD-AT-N.
5. **GATE ENG scope** (section 2.7) -- the aggregate per-node engagement check this task asked
   for, not PA-HIER-23's full per-site toggle matrix (C2, a separate, unbuilt instrument).

## 6. Open items for the runner (a different agent, per rule 2)

- Re-verify GATE PARITY at the full 200-event count before relying on the truth node as a
  byte-identical reference (section 3.2).
- Re-cost the off-truth (smeared) nodes before trusting the §7 CPU-h ceiling for S0-A's real
  4-seed x 5-node grid (section 3.1) -- the smeared node's cost looks like it may be dominated by
  a catalogue-wide term independent of event count, which changes the scaling law the anchor
  assumed.
- S0-C is unrun; its own setup cost alone is comparable to the smoke's entire 900s budget.
- S0-R is built, disclosed NULL, and not authorized to bank a verdict (PA-HIER-28 item 5).
