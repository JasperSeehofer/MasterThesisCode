# B1.2 driver extension -- BUILD NOTE (P1 equivalence gate / P0 completion / KW-Q1)

**Launched under rows #222/#223 -- charter nodes B1.1/B4.2.**

**Role boundary (rule 2, verifier independence): this is a BUILD note. I extended
`hier_s0_driver.py` and wrote `kwq1_score.py`, and SMOKE-TESTED both (small event caps,
mostly the fast unsmeared diagnostic path to stay inside the 600s-per-command budget). The
registered measurements (the real P1 comparison, S0-A's remaining nodes at full N, S0-C, and
KW-Q1's 4-seed/3-node/2-h primary read) must be RUN by a different agent from this one, and
per `CLAIM_IMPOSTOR_DRAG_20260829.md` sec 1.3's own rule-2 clause, the KW-Q1 runner must also be
a different agent from B1.1's driver author.** No registered band is claimed as passed or
failed anywhere in this note.

Files changed/added:
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`
  (extended in place; 567 insertions / 42 deletions; every pre-existing CLI option and its
  default is byte-identical in behaviour -- see sec 1).
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_score.py` (new).

`uv run ruff check` and `uv run mypy --ignore-missing-imports` both pass clean on both files
(mypy needs `--ignore-missing-imports` only because these live under `results/`, outside the
package's own mypy config scope -- no type errors, just untyped-import warnings that flag
suppresses).

---

## 1. New options, defaults, and the byte-identity argument

All new CLI flags are additive; no existing flag's name, default, or resolved behaviour
changed. The byte-identity argument for the *combination* of new defaults is: `--theta-sites
all --smear auto --config b0i --h-nodes <unset> --score-h <unset>` resolves, at every call site,
to EXACTLY the old hardcoded values (`theta_sites="all"`, `smear_global_selection=theta_engaged`,
the bc/b0i venue flags, `h_values=(H_GEN,)`, `read_event_ln_l(diag_csv, H_GEN)`), and
`_node_dir_suffix("all", "auto", "b0i") == ""` so node output paths are UNCHANGED (verified,
sec 5).

| flag | default | resolves to (at default) | new behaviour when non-default |
|---|---|---|---|
| `--theta-sites {all,2.1,2.2,2.3}` | `all` | `theta_sites="all"` forwarded verbatim (evaluate()'s own validated set -- there is no "2.1,2.2" combo form; the estimator only accepts these 4 literal strings, confirmed by reading its guard, `bayesian_statistics.py`'s `theta_sites not in ("all","2.1","2.2","2.3")` check) | isolates which site(s) receive theta |
| `--smear {auto,on,off}` | `auto` | `smear = theta_engaged` (identical to the pre-P1 unconditional dispatch, since pre-P1 `theta_sites` was always `"all"`) | `auto` at `theta_sites in ("2.1","2.2")` now resolves to `smear=False` (P1's whole point -- those sites never read the smeared table); `on`/`off` force the flag |
| `--config {b0i,ft}` | `b0i` | the original hardcoded bc/b0i venue+flags, unchanged | `ft` = KW-Q1's B-SEL/phi/fused venue (sec 2) |
| `--h-nodes <csv floats>` | unset | `h_values=(H_GEN,)` | fuses an arbitrary h-list into one `evaluate()` call per node (KW-Q1: `0.725,0.735`) |
| `--score-h <float>` | unset | resolves to `H_GEN` (present in the default h_values) | needed only when `--h-nodes` excludes `H_GEN` (KW-Q1) -- selects which evaluated h this driver's OWN internal `compute_scores`/`gate_eng`/`gate_parity` read back |
| `--score-only` | off | n/a (a mode switch) | reads on-disk `event_likelihoods.csv` files under `--out-root`/`--seeds`/`--nodes`/`--theta-sites`/`--smear`/`--config` and computes the pooled prereg §4.1 statistic -- **zero `evaluate()` calls** |

Node output directories gain a suffix (`_node_dir_suffix`, new helper): empty at the defaults
(unchanged paths `node_<name>`); otherwise `_<config-if-not-b0i>_sites<theta_sites-if-not-all>_<smearon|nosmear-if-not-auto>`,
e.g. `node_b_plus_sites2.2_nosmear` (P1) or `node_truth_ft` (KW-Q1 at its own defaults, since
KW-Q1 always passes `--config ft`). This satisfies the launch instruction's "output dir must
encode the variant" -- confirmed empirically never to collide with the pre-existing
`node_truth`/`node_b_plus`/... paths (sec 5).

CLI-level validation (fires before any compute, not just at evaluate()'s own guard): `--smear
off` combined with `--theta-sites all` or `2.3`, when any requested node other than `truth` is
present (every registered `THETA_NODES` entry except identity IS theta-engaged by construction),
is refused immediately with a message naming the fix (`raise SystemExit` in `main()`). A second,
library-level copy of the same check lives in `run_theta_node` itself (for callers that invoke it
directly, e.g. a future `kwq1_score.py`-style script that re-runs a single node without going
through `main()`).

## 2. FT config kwargs (KW-Q1, B4.2) -- copied EXACTLY, with source lines

Copied from `p3_twin_test.py`'s `_run_bsel_seed(seed, "phi", ..., completion_cell="fused")`
(reachable via `--stage fusedarm --survival phi --completion-cell fused --tag ft`):

| kwarg | value | source |
|---|---|---|
| `sigma_z_scale, area_scale` | `(1.0, 1.0)` | `c1d.ARM_SPECS["bsel"]`, `correspondence_1d.py:423` |
| `host_mode` | `"population_selected"` | `c1d.ARM_HOST_MODE["bsel"]`, `correspondence_1d.py:469`; `p3_twin_test.py:198` |
| `completeness_obj, phi_survival_table` | `c1d.build_bsel_selection_objects()` (no `h_true` kwarg -- default is `c1d.H_TRUE`, identical to `H_GEN`) | `p3_twin_test.py:194` |
| `selection_in_completion_numerator` | `"fused"` | `p3_twin_test.py`'s `--completion-cell` CLI arg, forwarded as `completion_cell` |
| `completion_event_measure` | `"ratio"` | `c1d.ARM_EVENT_MEASURE["bsel"]`, `correspondence_1d.py:543` |
| `catalogue_numerator_survival` | `"phi"` | `p3_twin_test.py`'s `--survival` CLI arg |
| `catalogue_global_selection` | **not passed** -- left at `run_mirror_seed_inprocess`'s own `"auto"` default, which resolves to `"phi"` under `normalization_mode="absolute_marginal"` (production default) | `p3_twin_test.py:216-224` never sets this kwarg; `correspondence_1d.py:2762` default |
| `c1d._verify_rate_weight_parity()` | **not called** | absent from `_run_bsel_seed` (grepped, zero hits in `p3_twin_test.py`) -- disclosed, not added, since this driver copies the arm EXACTLY including that omission |

Implemented as `build_ft_venue()` (new, `hier_s0_driver.py`) alongside the pre-existing
`build_bc_venue()`; `_build_venue(config, ...)` dispatches between them. `run_theta_node` gained
a `config` parameter selecting which fixed venue flags (`BC_*` vs `FT_*` module constants) feed
`run_mirror_seed_inprocess`.

One deliberate deviation, disclosed: this driver passes `h_bounds=H_BOUNDS=(0.50, 0.86)`
(prereg §5.1 invariant #2) for `config="ft"` too, even though `p3_twin_test.py`'s
`_run_bsel_seed` predates that kwarg and never passes it. This keeps KW-Q1 inside [HIER]'s own
registered invariant rather than silently inheriting an older, unpinned default; flagged for
the runner to veto if a different reading was intended.

**Known limitation, disclosed:** the driver's own `gate_parity()` function (used inside
`run_arm`/`--score-only`) always compares the truth node against the **bc-venue** banked CSV
(`BC_WORK_ROOT`), regardless of `--config`. For `--config ft` this comparison is
venue-mismatched and MEANINGLESS (confirmed in the KW-Q1 smoke, sec 4: max_rel_diff ≈ 0.34,
which is simply "the FT venue's numbers differ from the BC venue's numbers", not a parity
failure of anything). KW-Q1's own GATE PARITY (T-ID, "the s=1 node vs a fresh same-commit FT
re-evaluation") is a SEPARATE, correct comparison -- see sec 3's dedicated command, not this
driver-level key. `run_arm`'s payload still emits `gate_parity` for `config="ft"` runs; readers
must ignore it for that config (not suppressed, since suppressing it silently would be a
larger, undisclosed behaviour change to an existing function).

## 3. `--score-only` (P0 completion)

New functions `gather_node_results_from_disk` (reads `event_likelihoods.csv` off disk, zero
`evaluate()` calls) and `score_only_payload` (reuses `compute_scores`/`gate_eng`/`gate_parity`/
`verdict_s0a`/`verdict_s0r` VERBATIM -- the statistic is identical whether the `ln_l` frame came
from a fresh evaluation or a disk read) plus `write_score_markdown`. Writes
`<arm>_score_output.json` and `<arm>_score.md` under `--out-root`.

**Confirmed** (sec 5): a "remaining nodes" invocation (e.g. `--nodes b_minus,s_plus` after
`--nodes truth,b_plus` already exist on disk under the same `--out-root`/seed) does NOT touch
the already-written node directories (mtimes checked, unchanged) -- each invocation only ever
creates/overwrites the node directories it was asked to run, because `run_theta_node` always
calls `evaluate()` fresh for every requested node (there is no caching/skip-if-exists logic to
go stale). `--score-only`, run afterward with all 5 node names requested, correctly unions
whatever is present and computed the full pooled B0-A verdict from disk in 1.7s (sec 4).

## 4. Smoke test evidence (seed 900101, `--event-cap 12` throughout, foreground, all ≤600s)

**Design note on the ≤600s budget:** B1.1's own run record measured a smeared (theta-engaged,
`theta_sites=all`/`smear=auto`, the pre-existing default) node at **1190.93s**, independent of
event cap (dominated by the catalogue-wide GL-quadrature denominator, not the per-event loop) --
this alone exceeds the 600s foreground cap. I therefore smoke-tested the NEW code paths
(validation, node-dir suffixing, `--config`, `--h-nodes`, `--score-only`) primarily via the
**fast, unsmeared P1 diagnostic path** (`--theta-sites 2.2 --smear off`), which is exactly what
P1 predicts should be legal and cheap, plus one default-path regression check (truth node only,
unsmeared regardless). I did not re-run a full smeared (`all`/`auto`) off-truth node in this
smoke pass -- its cost is already directly measured in `B1_1_HIER_RECORD.md` sec 2.3
(1190.93s) and re-deriving it here would either blow the 600s budget or require `--smoke`'s
1-seed restriction to be lifted, which rule 2 reserves for the runner. **P1's central claim
(bit-identity between `all`/`auto`-smeared and `2.2`/`off`-unsmeared `combined_no_bh`) was
therefore NOT independently verified numerically in this build pass** -- flagged as the runner's
first task (sec 6 gives the exact command).

| # | command (abbreviated) | wall (real) | evaluate_s | result |
|---|---|---|---|---|
| 1 | `--arm S0-A --smoke --nodes truth` (all defaults) | 505s | 46.01s | regression baseline: `node_truth` (no suffix), GATE PARITY residual unchanged (3.7e-5 rel, matches B1.1) |
| 2 | same + `--theta-sites all --smear off --nodes b_plus` | instant | -- | **refused at parse time**, exit 1, message names the fix (validates sec 1's CLI guard) |
| 3 | `--theta-sites 2.2 --smear off --nodes truth,b_plus` | 534s | b_plus: **43.77s** (vs 1190.93s smeared -- 27x) | `node_dir_suffix="_sites2.2_nosmear"`; GATE PARITY (bc-venue, meaningful here since config=b0i) unchanged |
| 4 | same out-root, `--nodes b_minus,s_plus` (remaining-nodes test) | 533s | -- | new node dirs created; `node_truth_sites2.2_nosmear`/`node_b_plus_sites2.2_nosmear` mtimes UNCHANGED (verified via `stat`) |
| 5 | same out-root, `--nodes s_minus` | 500s | -- | 5th node added; all 5 now present |
| 6 | `--score-only --nodes truth,b_plus,b_minus,s_plus,s_minus --theta-sites 2.2 --smear off` | **1.7s** | n/a (no evaluation) | full pooled verdict computed from disk: band=B0-A, Z_b=-0.908, Z_s=0.439, GATE ENG 100% pass on all 4 nodes (confirms sites 2.1/2.2 alone move the no-BH channel, consistent with B1.1's finding 4), GATE PARITY reused from disk |
| 7 | `--config ft --theta-sites 2.2 --smear off --nodes s_minus,truth,s_plus --h-nodes 0.725,0.735 --score-h 0.725` | 309s | -- | `node_dir_suffix="_ft_sites2.2_nosmear"`; FT venue setup notably cheaper than b0i's (~260s vs ~450s) |
| 8 | `kwq1_score.py` against run 7's output | **1.4s** | n/a | GATE I max rel 6.3e-8 (≪ 2e-6 tol -- assembly identity holds); GATE ENG 100% (9/9 active rows); falsifier q1 share 92.6% (not withdrawn); R=0.0415 -> printed band KERNEL-WIDTH-INERT (12-event/1-seed smoke, **not a registered read**) |
| 9 (partial) | `--seeds 900101,900102 --nodes truth --event-cap 12 --jobs 2 --total-cpu-budget 14` | timed out at 590s (killed) | -- | **structurally confirmed**: both seed-900101 and seed-900102 worker processes reached `prepared_cramer_rao_bounds.csv`/`cramer_rao_bounds.csv` concurrently (both files present, both dated inside the run) before the kill -- i.e. the `mp.Pool(processes=2)` dispatch and per-worker CPU-affinity pinning (`cpu_per_job=7`) both fired as designed; neither seed reached `evaluate()`'s posteriors/diagnostics output within 590s (2 concurrent full venue setups on this shared 16-core dev machine is evidently slower than 1x + headroom). `ps` showed no orphaned processes after the `timeout` kill. **`--jobs>1` is confirmed to dispatch correctly; a run that actually COMPLETES within one foreground command needs either a smaller node/seed combination than attempted here, more CPU headroom, or to run on the cluster/in the background (out of a builder's `--smoke` scope) -- left for the runner.**

Command 1 also used to sanity-check that `run_arm`'s payload now carries `theta_sites`/`smear`/
`config`/`h_values`/`score_h`/`node_dir_suffix` keys at their defaults (`"all"`/`"auto"`/
`"b0i"`/`[0.73]`/`null`/`""`) -- present and correct in every JSON produced.

`run_seed_s0c` (S0-C) was **not** modified and **not** re-smoked in this pass, per the launch
instruction ("S0-C's 41-node grid stays as is") -- its call site and body are byte-for-byte
unchanged (`git diff` shows zero lines touched inside `run_seed_s0c`), so no regression is
possible there.

## 5. Ambiguities / design choices resolved (flagged for veto)

1. **Node-dir suffix scope**: the suffix is computed ONCE per invocation from
   `(theta_sites, smear, config)` and applied to EVERY node directory that invocation writes --
   not per-node conditional on that node's own theta engagement. Consequence: a truth node run
   under `--theta-sites 2.2 --smear off` lands in `node_truth_sites2.2_nosmear`, a DIFFERENT
   path from the default `node_truth`, even though truth's own theta is never engaged and its
   evaluate() call is identical either way (theta_sites/smear are inert at theta=(0,1)). This
   avoids ambiguity about which invocation's truth node a given directory belongs to, at the
   cost of (harmlessly) re-evaluating an identical truth node under a new path if the same seed
   is later run at a different `--theta-sites`/`--smear` combination.
2. **`--score-h` semantics**: only meaningful when `--h-nodes` excludes `H_GEN`; documented as
   inert otherwise. Chosen over silently erroring, so `--score-only` and `run_arm` can both use
   the same `_resolve_score_h` helper without special-casing KW-Q1's 2-h grid.
3. **`gate_parity`'s venue mismatch under `config="ft"`** (sec 2) -- left as a disclosed
   limitation rather than special-cased/suppressed, since suppressing an existing function's
   output for a new config is itself an undisclosed behaviour change.
4. **KW-Q1's frozen q1 membership edge** (`kwq1_score.py`): uses `b4_imp_stage1_forecast.json`'s
   `covariates.ft.z_true.edges[0]` (≈0.35750, the ALL-EVENTS quartile, matching the claim card's
   literal "z_true < 0.358"), NOT the `z_true_active_only` edge (≈0.33814) also present in the
   same forecast JSON under a different key. Both exist; the claim card's sec 1.2 table
   ("z_true (edges 0.358/0.459/0.584)") and sec 1.3's literal "0.358" pin the former. Disclosed
   in `kwq1_score.py`'s own docstring so a future reader does not "fix" this by swapping edge
   sets.
5. **KW-Q1 GATE ENG** operationalized as: fraction of ACTIVE rows (defined as `L_cat_no_bh > 0`
   at `h_lo` on the truth/s=1 node) where `L_cat_no_bh` differs between the two EXTREME s-nodes
   (`s_minus` vs `s_plus`) -- the claim card names the >=99% threshold but not which node pair;
   the extremes give the largest expected separation and are the natural choice. Flagged for
   veto.
6. **`--jobs` parallelizes SEEDS, not NODES** (pre-existing behaviour, unchanged by this build).
   `SYNTHESIS_DOCKET_1_20260829.md` sec 2 B1 P0's "≈2h wall if run 5 nodes in parallel (smeared
   phase is single-core-bound, so parallelism across nodes is free)" describes a DIFFERENT
   parallelism axis (nodes within one seed) that this driver does not implement -- achieving it
   today requires the orchestrator to launch several separate driver invocations (one per node
   subset) as separate foreground/background jobs, not a single `--jobs>1` invocation. Adding
   node-level parallelism inside `run_arm_seed_s0a`/`s0r` was judged out of this build's scope
   (P1/P0/KW-Q1 as specified); flagged as a possible B1.3 follow-up if the 2h-wall estimate is to
   be realized via a single driver invocation.

## 6. Exact commands for the orchestrator

All commands below assume `cd /home/jasper/Repositories/darksiren-emri` and `nproc=16` (adjust
`--total-cpu-budget` if the shared machine's load differs; check `uptime` first).

### P1 equivalence gate (the real comparison, full N=200, one seed)

```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --seeds 900101 --nodes b_plus \
  --theta-sites 2.2 --smear off \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run \
  --jobs 1 --total-cpu-budget 14
```
Writes `hier_s0_registered_run/s0a_seed900101/node_b_plus_sites2.2_nosmear/.../event_likelihoods.csv`
alongside the EXISTING `node_b_plus/` (banked by B1.1's run record, `all`/`auto`, full N=106
post-selection). Expected wall: ~458s setup (re-paid, no cross-invocation cache) + ~65s unsmeared
eval (per B1.1's own truth-node anchor, since 2.2-only is per-host, not catalogue-wide) ≈ **~9
min**, vs the smeared node's already-measured 1190.93s (~20 min) for the SAME node under `all`/
`auto`. Then diff:
```bash
uv run python -c "
import pandas as pd
a = pd.read_csv('results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_seed900101/node_b_plus/simulations/diagnostics/event_likelihoods.csv')
b = pd.read_csv('results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run/s0a_seed900101/node_b_plus_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv')
a73 = a[a.h==0.73].sort_values('event_idx').reset_index(drop=True)
b73 = b[b.h==0.73].sort_values('event_idx').reset_index(drop=True)
import numpy as np
d = (a73['combined_no_bh'] - b73['combined_no_bh']).abs()
print('max abs diff:', d.max(), 'max rel diff:', (d / a73['combined_no_bh'].abs().clip(lower=1e-300)).max())
"
```
Bit-identity (max abs diff == 0.0) is P1's predicted outcome (chair's source read,
`SYNTHESIS_DOCKET_1_20260829.md` sec 2 B1 P1); anything else is decisive new information.

### P0 completion: remaining S0-A nodes

Seed 900101 (nodes b_minus, s_plus, s_minus -- truth and b_plus already banked by B1.1):
```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --seeds 900101 --nodes b_minus,s_plus,s_minus \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run \
  --jobs 1 --total-cpu-budget 14
```
Expected: ~458s setup + 3 x 1190.93s smeared ≈ **~67 min** single process (registered default,
`all`/`auto` -- do NOT substitute the P1 unsmeared path here unless P1 above confirms
bit-identity first, per the docket's own gating).

Seeds 900102-900104 (all 5 nodes each, not yet started):
```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --seeds 900102,900103,900104 --nodes truth,b_plus,b_minus,s_plus,s_minus \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run \
  --jobs 3 --total-cpu-budget 14
```
`--jobs 3` runs the 3 seeds concurrently (`cpu_per_job = 14 // 3 = 4`); per-seed cost ≈ 458s
setup + 65s truth + 4x1190.93s smeared ≈ 5287s (~88 min) -- run concurrently across 3 seeds,
wall ≈ **~88-100 min** (some slowdown expected from CPU contention at 4 cpus/job and 3 concurrent
full venue setups, per this build's own jobs=2 partial-timeout finding, sec 4 row 9); **this
single command will NOT complete inside a 600s foreground window and must be run
`run_in_background` or on a machine/session that can wait ~1.5-2h.**

Then, once all 4 seeds x 5 nodes exist:
```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --score-only --seeds 900101,900102,900103,900104 \
  --nodes truth,b_plus,b_minus,s_plus,s_minus \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run
```
Zero-compute (<5s), writes `s0a_score_output.json`/`s0a_score.md` with the pooled B0-A/B0-A'
verdict per prereg §4.1.

### S0-C (costing probe, unchanged code, still unrun to completion per B1.1's run record)

```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-C --seeds 900101 \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run \
  --jobs 1 --total-cpu-budget 12
```
B1.1's run record left this running past ~800s with zero `posteriors/h_*.json` written yet
(registered ceiling 15 CPU-h) -- **must run `run_in_background` or accept a multi-hour foreground
wait**, not a single ≤600s command.

### KW-Q1 (B4.2), primary read (4 seeds, `all`/`auto`, the registered form)

```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --config ft --seeds 900101,900102,900103,900104 \
  --nodes s_minus,truth,s_plus --h-nodes 0.725,0.735 --score-h 0.725 \
  --out-root <a fresh out-root, e.g. .../fanout1_20260829/kwq1_registered_run> \
  --jobs 4 --total-cpu-budget 14
```
Cost per the claim card: 8.4 CPU-h if P1 passes and this is re-run unsmeared (`--theta-sites
2.2 --smear off`, +GATE-ENG-on-catalogue-leg-only caveat -- see `CLAIM_IMPOSTOR_DRAG_20260829.md`
sec 1.3(c)); ≈13.7 CPU-h at the smeared default shown above (`all`/`auto`) if P1 has not yet
been confirmed. Then:
```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_score.py \
  --out-root <the same out-root> --seeds 900101,900102,900103,900104
```
(zero-compute, prints the band -- does NOT write a verdict/band into its JSON, per the launch
instruction; the runner adjudicates).

**GATE PARITY for KW-Q1 (T-ID)** -- a fresh same-commit FT re-evaluation of the s=1 (truth) node,
SEPARATE out-root, one seed:
```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --config ft --seeds 900101 --nodes truth --h-nodes 0.725,0.735 --score-h 0.725 \
  --out-root <a THIRD, separate out-root> --jobs 1 --total-cpu-budget 14
```
then diff its `combined_no_bh`/`L_cat_no_bh` columns (both h rows) against the primary run's own
`node_truth_ft/.../event_likelihoods.csv` (same pandas pattern as the P1 diff above) -- bit-identity
is the target; the 2026-08-23 banked FT CSVs under `p3_work/ft_*_work/` predate Sigma^phi and are
explicitly NOT the comparand (claim card sec 1.3).

## 7. Expected wall times at the two measured cost regimes

| regime | per-cell cost | source |
|---|---|---|
| unsmeared (truth node, or any node at `theta_sites in ("2.1","2.2")` with `--smear off`) | **~65s** (64.73-67.72s measured, B1.1 record sec 2.1/2.3, this build's own runs 1/3 corroborate at 43.77-46.01s for a 9-12-event smoke) | `B1_1_HIER_RECORD.md` sec 2.1; this note sec 4 |
| smeared (`theta_sites in ("all","2.3")` with an engaged node) | **~1191s** (1190.93s measured, independent of event cap -- catalogue-wide GL quadrature, single-core-bound) | `B1_1_HIER_RECORD.md` sec 2.3/1 item 5 |
| one-time per-seed venue setup, `config="b0i"` | ~450-586s | `B1_1_HIER_RECORD.md` sec 2.1/4; this note run 1 (503s incl. 46s eval) |
| one-time per-seed venue setup, `config="ft"` | ~260-306s (this build's own measurement, run 7 -- notably cheaper than b0i, not independently explained; population_selected host draw likely skips some catalogue-selected-specific per-event kernel setup) | this note sec 4 |

**Standing stamp: launched under rows #222/#223 -- charter nodes B1.1/B4.2.**
