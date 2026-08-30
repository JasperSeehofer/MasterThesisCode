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

## 8. Crash fix (runner-disclosed, 2026-08-29)

**Role boundary unchanged:** this section documents a BUILD-side bug fix only. No statistic,
band, or threshold was touched. Fixed by the driver's author (rule 2 permits SMOKE-testing
only; the registered P0/KW-Q1 measurements are still for a different agent to run).

### Symptom (as disclosed by the runner)

Registered P0 command (`hier_s0_registered_run/logs/runner_wave2pre_20260829.log`, stage "P0"):

```
--arm S0-A --seeds 900101,900102,900103,900104 --nodes truth,b_plus,b_minus,s_plus,s_minus \
--theta-sites 2.2 --smear off --out-root .../hier_s0_registered_run --jobs 2 --total-cpu-budget 14
```

crashed after ~18 minutes with an uncaught, un-informative pandas error:

```
File "hier_s0_driver.py", line 970 (pre-fix), in run_arm
  scores = compute_scores(all_nodes)
File "hier_s0_driver.py", line 647 (pre-fix), in compute_scores
  bp = pd.concat([...for r in all_nodes["b_plus"]], ...)
ValueError: No objects to concatenate
```

### Root cause

Post-mortem on the registered run's own on-disk state (read-only inspection, no files under
`hier_s0_registered_run/` touched): for **all four seeds**, `node_truth_sites2.2_nosmear/`
contained only the early-written `simulations/prepared_cramer_rao_bounds.csv` /
`simulations/cramer_rao_bounds.csv` / `selection_tables_h_0_73.json` artifacts and **no**
`simulations/diagnostics/event_likelihoods.csv` -- i.e. `BayesianStatistics.evaluate()` never
returned for the very FIRST node of any seed, let alone reached `b_plus`/`b_minus`/`s_plus`/
`s_minus`. Zero `"[S0-A ...]"` per-node completion lines appear anywhere in the P0 log window.
Every one of the 4 per-seed workers' `except Exception` (pre-fix `_run_one_seed_worker`,
line ~877) therefore fired on the first node, discarding that seed's entire `results` list
(including the already-on-disk-but-unread truth CSV attempt) and returning `{"seed":...,
"error":...}` -- a structured failure that WAS captured, but never printed or written to disk,
because `compute_scores` crashed the process (an uncaught top-level `ValueError`) before
`main()` ever reached `out_json.write_text(...)`.

The mechanism (confirmed both by re-deriving it from `_run_one_seed_worker`'s pre-fix affinity
block, hier_s0_driver.py:842-847, and by live observation of the orchestrator's concurrently-
running `--arm S0-C --jobs 1 --total-cpu-budget 12` process, which spawned ~10 heavy
(700-750 MB RSS, CPU-bound) `forkserver` children -- `BayesianStatistics.evaluate`'s own
internal multiprocessing pool, confirming it really does auto-size a large worker pool off the
process's OWN (possibly already-narrowed) CPU affinity mask):

```python
# pre-fix _run_one_seed_worker (EVERY concurrent worker ran this independently):
all_cpus = sorted(os.sched_getaffinity(0))
budget = max(1, min(cpu_budget, len(all_cpus)))
os.sched_setaffinity(0, set(all_cpus[:budget]))
```

Under `--jobs 2 --total-cpu-budget 14` (`cpu_per_job=7`), **both** outer `Pool` workers computed
`all_cpus[:7]` independently and pinned to the exact SAME leading 7 cores instead of disjoint
halves. Each outer worker's own `evaluate()` call then auto-sizes ITS internal pool off that
(now-narrowed) affinity (`available_cpus - 2` = 5 inner workers), so 2 outer x 5 inner = 10
heavy processes fought over 7 real cores -- confirmed by the live sibling job above to run into
several-hundred-MB-per-worker memory pressure on top of the CPU oversubscription, consistent
with the internal pool dying (OOM or a broken-pool exception) partway through the first
`evaluate()` call, every time, for every seed.

Isolated mechanism-level reproduction (fast, no `evaluate()` cost -- the full smoke command
below could not complete within the ≤600s foreground budget even once, since `config="b0i"`
venue construction alone measures ~450-586s per seed at a generous CPU budget and this
environment's smoke budget, `--total-cpu-budget 4 --jobs 2`, is much tighter):

```
OLD (pre-fix) pin, 2 Pool workers, cpu_per_job=2: both report affinity [0, 1]   <- BUG (overlap)
NEW (post-fix) pin, 2 Pool workers, cpu_per_job=2: workers report [0, 1] and [2, 3]  <- disjoint
jobs==1 direct-call path (no Pool): affinity [0, 1] -- BYTE-IDENTICAL to pre-fix
```

### Fix (`hier_s0_driver.py`)

1. **New `_pin_worker_affinity(cpu_budget)`** (replaces the inline block in
   `_run_one_seed_worker`): keys off `multiprocessing.current_process()._identity` (the 1-based
   slot the `Pool` machinery assigns each worker ONCE at spawn, stable for its lifetime) to pick
   a CPU slice DISJOINT from every other concurrent worker's slice. `--jobs 1` (no `Pool`,
   direct call): `_identity` is empty, slot=0 -- byte-identical to the pre-fix single leading
   slice. `--jobs N>1`: passed as `ctx.Pool(..., initializer=_pin_worker_affinity,
   initargs=(cpu_per_job,))` in `run_arm`, so it runs ONCE per worker at Pool startup (before any
   task, while the OS affinity mask is still the full inherited set) rather than once per task
   (which would have re-read an already-narrowed mask and corrupted the disjoint slice).
   `_run_one_seed_worker` now calls it directly ONLY when `not mp.current_process()._identity`
   (the `--jobs 1` case) so the Pool-worker case is never re-pinned.
2. **`compute_scores(all_nodes, seeds=None)`** (new optional `seeds` kwarg, both call sites
   updated): raises a clear `ValueError` listing the exact missing `(seed, node)` pairs and
   `n_present_by_node` BEFORE attempting any `pd.concat`, instead of letting an empty node list
   surface as pandas' opaque "No objects to concatenate". Defensive (both callers already gate
   on "produced", not merely "requested" -- see next item) but fires correctly if a future
   caller skips that gate.
3. **`run_arm`**: (a) prints every worker's error + traceback immediately after `pool.map`
   returns, so a swallowed per-seed exception is visible in the runner's log stream even if a
   later step raises; (b) replaced the `need_all_four = all(n in nodes for n in (...))`
   ("was requested") gate with a `requested_all_four and produced_all_four` gate
   (`produced_all_four` checks `len(all_nodes.get(n, [])) > 0`, matching the pattern
   `score_only_payload` already used correctly) so a run where every seed's worker for a
   required node failed degrades to a diagnostic `payload["note"]` (listing the missing
   `(seed, node)` pairs and `n_seeds_error`) instead of an uncaught crash that discards
   `payload["errors"]`'s real tracebacks before they are ever written to disk.

No node-dir suffix convention, CLI flag, default, or scoring formula changed. `--score-only`'s
own `score_only_payload`/`gather_node_results_from_disk` already gated on `produced`, not merely
`requested`, and needed no logic change beyond passing `seeds=seeds` into `compute_scores` for a
consistent error message.

### Verification

- `uv run ruff check hier_s0_driver.py` -- all checks passed.
- `uv run mypy hier_s0_driver.py` -- no issues found.
- Isolated multiprocessing test (`ctx.Pool(processes=2, initializer=_pin_worker_affinity,
  initargs=(2,))`, 2 concurrent 1s-sleep tasks): workers report DISJOINT affinity sets `[0, 1]`
  and `[2, 3]` (pre-fix logic, reproduced verbatim for contrast in the same test harness, gives
  `[0, 1]` / `[0, 1]` -- fully overlapping). `--jobs 1` direct-call path unchanged (`[0, 1]`).
- `compute_scores({n: [] for n in NODE_ORDER}, seeds=(900101,900102,900103,900104))` now raises
  `"compute_scores: cannot pool score_b/score_s -- missing (seed, node) pairs: [...]. "
  "n_present_by_node={'b_plus': 0, 'b_minus': 0, 's_plus': 0, 's_minus': 0}. ..."` instead of
  pandas' "No objects to concatenate".
- `run_arm(..., jobs=2, ...)` with `_run_one_seed_worker` monkeypatched to return the EXACT
  registered-crash failure mode (every one of 4 seeds returns `{"seed":..., "error":...}`, zero
  successes) now prints all 4 `WORKER ERROR` lines + tracebacks, returns a payload with
  `n_seeds_ok=0`, `n_seeds_error=4`, `"scores"` key ABSENT, and a `note` listing all 16 missing
  `(seed, node)` pairs -- **no crash**, exit code still correctly 1 (`main()`'s
  `return 0 if not result.get("errors") else 1`).
- The full end-to-end smoke command below could not be driven to completion within repeated
  ≤600s foreground polls in this environment (venue construction alone exceeds the smoke
  CPU-budget's realistic wall time; confirmed both pre-fix and post-fix runs timed out at the
  SAME pre-node-loop stage, i.e. before either could exercise the actual bug/fix) -- the fix is
  therefore verified at the mechanism level above (root-cause reproduction + contrast, plus a
  full `run_arm`-level dry run of the exact failure payload) rather than by an end-to-end smoke
  banking real `event_likelihoods.csv` files. A different agent re-running the registered P0
  command below, with a realistic (non-smoke) `--total-cpu-budget`, is the first real end-to-end
  test of the fix and should be watched for the `WORKER ERROR` print path firing (it should not,
  if the affinity fix holds).

### Commands for the next agent

P0 (now with disjoint per-worker CPU pinning; unchanged flags otherwise):

```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --seeds 900101,900102,900103,900104 \
  --nodes truth,b_plus,b_minus,s_plus,s_minus \
  --theta-sites 2.2 --smear off \
  --out-root results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_registered_run \
  --jobs 2 --total-cpu-budget 14
```

KW-Q1 (B4.2), primary read (unchanged by this fix; repeated here for convenience, see sec 6):

```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
  --arm S0-A --config ft --seeds 900101,900102,900103,900104 \
  --nodes s_minus,truth,s_plus --h-nodes 0.725,0.735 --score-h 0.725 \
  --out-root <a fresh out-root, e.g. .../fanout1_20260829/kwq1_registered_run> \
  --jobs 4 --total-cpu-budget 14
```
then:
```bash
uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/kwq1_score.py \
  --out-root <the same out-root> --seeds 900101,900102,900103,900104
```

**Standing stamp: launched under rows #222/#223 -- charter nodes B1.1/B4.2.**

## 8.1 Correction (runner-disclosed, 2026-08-29)

**Record-only entry -- no code changed by this entry.** The §8 fix (disjoint per-worker CPU
affinity pinning) was necessary but NOT sufficient. The orchestrator's registered P0 re-run
with the §8 fix in place, still under `--jobs 2`, failed on **every** seed with:

```
AssertionError: daemonic processes are not allowed to have children
```

raised inside `run_arm_seed_s0a` -> `run_theta_node` -> `run_mirror_seed_inprocess` ->
`BayesianStatistics.evaluate()` (log:
`hier_s0_registered_run/logs/runner2_wave2pre_20260829.log`).

**True structural cause:** `multiprocessing.Pool` worker processes are created as **daemonic**
(`Process.daemon = True` is the `Pool` default), and daemonic processes are forbidden by the
`multiprocessing` module from starting their own child processes. `BayesianStatistics.evaluate()`
spawns its own internal worker pool (the `available_cpus - 2` auto-sizing referenced in §7/§8
above, confirmed live via the forkserver children observed under the orchestrator's `--jobs 1`
S0-C job). When `run_arm`'s outer `ctx.Pool(processes=jobs, ...)` runs `_run_one_seed_worker` in
one of ITS (daemonic) workers, and that worker's call into `evaluate()` tries to start its own
internal pool, Python's `multiprocessing` layer raises `AssertionError` immediately -- this fires
regardless of CPU affinity, regardless of how many cores are free, and regardless of the §8 fix.
The §8 disjoint-affinity fix was a real defect (confirmed by the isolated mechanism-level test in
§8) and remains correct as far as it goes, but it addressed a SECONDARY resource-contention
symptom, not the primary blocker: **outer `--jobs N>1` is structurally incompatible with
`evaluate()`'s own internal multiprocessing pool, full stop, independent of affinity.**

**Run form of record:** `--jobs 1` is the form of record for both P0 and KW-Q1 until a
non-daemonic worker mechanism is implemented (see below). `--jobs 1` never goes through
`ctx.Pool` at all (`run_arm`'s `if jobs == 1: raw_results = [_run_one_seed_worker(a) for a in
task_args]` branch calls the worker function directly in the (non-daemonic) main/orchestrator
process), so `evaluate()`'s internal pool starts without issue -- this is exactly the form
the orchestrator's runner-3 is now using for all seeds, sequentially.

**Proper future fix (NOT implemented tonight, follow-up only):** replace the daemonic
`multiprocessing.Pool` in `run_arm`'s `jobs > 1` branch with a non-daemonic worker mechanism,
e.g. (a) one `subprocess.Popen` per seed (each a fresh, non-daemonic OS process, free to spawn
its own children) instead of an in-process `Pool` worker; (b) a `NoDaemonProcess`/
`NoDaemonContext` pool (the well-known multiprocessing recipe that subclasses `Process` to force
`daemon = False`, then builds a `Pool` on that context) so `evaluate()`'s internal pool is
allowed to start; or (c) `concurrent.futures.ProcessPoolExecutor`, which is built on the same
daemonic-worker `Process` under the hood and would need the same `NoDaemonProcess`-style
workaround to lift the restriction -- it does not sidestep the constraint on its own. Any of
these needs its own smoke-test pass (this driver's rule 2 boundary applies to that follow-up
build too) before `--jobs N>1` can be reinstated as a run form of record.

**Standing note:** §8's disjoint-affinity fix (`_pin_worker_affinity`, the `compute_scores`
clear-error guard, and `run_arm`'s produced-vs-requested gate) all remain in the file and are
still correct/beneficial in their own right (in particular the `compute_scores`/`run_arm`
error-surfacing fix is what let this AssertionError be diagnosed cleanly instead of being
swallowed the way the original crash was) -- none of it is reverted by this correction. Only the
"`--jobs N>1` is now safe" implication of §8 is withdrawn.

## §9 driver flags for T1.2 (2026-08-30)

**Record-only entry for what this build task added.** T1.1 (commit `6c6f2a63`) threaded
`theta_phi_divisor: str = "off"` and `sky_cone_k: float = 1.5` into
`correspondence_1d.py`'s `run_mirror_seed_inprocess` but left no driver-level CLI surface for
them (its own docstring, ~2912-2927, said as much: "a caller ... wanting to arm the divisor
from the command line must add its own `--theta_phi_divisor`/`--sky_cone_k` arguments" --
confirmed missing by `T1_1_DIVISOR_IMPLEMENTATION_RECORD.md`'s "driver gap" section after the
orchestrator's registered command failed with `unrecognized arguments: --theta_phi_divisor on`).

**What was added, following the exact `--theta-sites`/`--smear` pattern:**

- `--theta-phi-divisor {off,on}` (default `off`) and `--sky-cone-k FLOAT` (default `1.5`),
  forwarded verbatim to **every** `run_mirror_seed_inprocess` call site: `run_theta_node`
  (S0-A/S0-R's per-node call, both `config="b0i"` and `config="ft"` branches) and
  `run_seed_s0c` (S0-C's single truth-node call) -- unconditional forwarding is correct per
  GATE T-ID (the divisor is theta-consistent, a no-op exactly at theta=(0,1), so it disturbs
  neither S0-C nor the truth node of S0-A/S0-R).
- `_node_dir_suffix` extended with `theta_phi_divisor`/`sky_cone_k` parameters: `on` appends
  `_divisor`; a non-default `sky_cone_k` (any value != 1.5) appends `_conek<value>` (`:g`
  formatted). Byte-identical at the defaults -- empty suffix, unchanged paths.
- Threaded through the full call chain so both live runs and `--score-only` resolve the same
  suffix: `run_arm_seed_s0a`/`run_arm_seed_s0r`/`run_seed_s0c` -> `_run_one_seed_worker`'s args
  tuple (2 new trailing fields, byte-identical at `"off"`/`1.5`) -> `run_arm`'s `task_args` and
  its `node_dir_suffix`/`theta_phi_divisor`/`sky_cone_k` payload fields -> `main()`'s new
  argparse flags, both the live-run `run_arm(...)` call and the `--score-only` branch's
  `gather_node_results_from_disk(...)` call and result payload.
- `write_selection_table_json`'s `theta_phi_divisor` field needs no driver-side wiring: the
  driver never calls that function directly -- it is called internally by
  `BayesianStatistics.evaluate()` (`bayesian_statistics.py` ~4733), which already threads its
  own `self._theta_phi_divisor` through once `evaluate()` receives the kwarg forwarded above.
  Confirmed by reading the call site: no driver-side change was needed or made.

**Smoke evidence (`--event-cap 12`, `--jobs 1`, foreground, builder-only per rule 2):**

```
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py --arm S0-A \
  --seeds 900101 --nodes truth,b_plus --theta-sites 2.2 --smear off --theta-phi-divisor on \
  --event-cap 12 --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_smoke_divisor \
  --jobs 1 --total-cpu-budget 4
```

Ran clean: `n_seeds_ok=1`, `n_seeds_error=0`, `errors=[]`. `node_dir_suffix` resolved to
`_sites2.2_nosmear_divisor`; both node output dirs on disk carry it verbatim:
`s0a_seed900101/node_truth_sites2.2_nosmear_divisor/` and
`s0a_seed900101/node_b_plus_sites2.2_nosmear_divisor/` (truth node: `evaluate_s=41.34`,
`wall_s=44.12`, `n_events=9`).

**Engagement check** (b_plus, divisor on vs. the banked divisor-off b_plus comparand at the same
seed/theta_sites/smear, `fanout1_20260829/hier_s0_registered_run/s0a_seed900101/
node_b_plus_sites2.2_nosmear/`), joined on `event_idx` at h=0.73: `combined_no_bh` (the no-BH
catalogue leg the site-2.3phi divisor transforms) differs for all 9/9 events, relative
differences 0.4%-4.2% (max 0.0416, event_idx 1) -- comfortably above the driver's own GATE ENG
threshold (>=10% of events moved by >=1e-6 relative) on 100% of events. `combined_with_bh` is
UNCHANGED (0.0 relative diff on all 9 events), exactly as expected for a "no-BH divisor"
instrument that must not touch the with-BH channel. `ruff check` on the driver: all checks
passed.

**The registered re-certification command (T1.2, 4 seeds, 5 nodes):**

```
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py --arm S0-A \
  --seeds 900101,900102,900103,900104 --nodes truth,b_plus,b_minus,s_plus,s_minus \
  --theta-sites 2.2 --smear off --theta-phi-divisor on --jobs 1 \
  --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_recert_run
```

(`--sky-cone-k` left at its byte-identical default 1.5 throughout, matching
`T1_1_DIVISOR_IMPLEMENTATION_RECORD.md`'s F1 specification exactly -- a `--sky-cone-k`
passthrough only matters for a future F2 enlarged-ball arm, out of scope here. `--jobs 1` per
§8.1's standing daemonic-pool constraint.)

**The matching score-only command** (zero-compute pooled read once the above has banked its
`event_likelihoods.csv` files):

```
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py --arm S0-A \
  --seeds 900101,900102,900103,900104 --nodes truth,b_plus,b_minus,s_plus,s_minus \
  --theta-sites 2.2 --smear off --theta-phi-divisor on \
  --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_recert_run \
  --score-only
```

**Standing stamp: launched under rows #222/#223 -- charter nodes B1.1/B4.2; this entry is T1.2's
own build record (row #255 tree 2 node T1.2).**

---

## §10 scorer denominator fix (2026-08-30; verifier MUST_FIX)

Row #255 standing grant, tree 2 node T1.3-zwin. BUILDER fix in response to the independent
verifier's item 3 MUST_FIX (`tree2_20260830/T1_3_ZWINDOW_VERIFIER_REPORT.md`): `_es_null_det_
closed_form` computed `Es_null_det_i` using `score_s_raw`'s secant denominator (`sqrt2 -
1/sqrt2 = 0.70711`) instead of the registered `score_lns` secant denominator (`ln(2) =
0.69315`) that PA-HIER-32(d) (`PREREGISTRATION_HIER_HTHETA_20260826.md`, near the end of the
block) specifies verbatim: "Es_null_det_i = the closed-form expectation of score_lns_i under
host i's OWN generator kernel", and `score_s = score_lns - Es_null_det`. The two forms differ
by the constant factor `(sqrt2-1/sqrt2)/ln(2) = 1.02014...`; the raw-denominator (buggy) value
is SMALLER in magnitude than the registered ln(2)-denominator value by that same factor
(verified both analytically -- same weighted-average numerator divided by a larger denominator
-- and numerically on a synthetic single-host fixture, see the new test below).

**Fix.** Single use site: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/
hier_s0_driver.py`, inside `_es_null_det_closed_form` (around line 587), `denom_s = _SQRT2 -
1.0 / _SQRT2` renamed to `denom_lns = math.log(2.0)`, and its one consumer (the `secs = (...)
/ denom_s` line) updated to divide by `denom_lns`. Confirmed via `grep -n "denom_s\b"` that
this was the ONLY reference to that name in the file before the fix -- the window-selection
logic (`window_minus`, built from `_SQRT2` directly for the window WIDTH, not the secant
denominator) and `score_s_raw`'s own independent `denom_s_raw` (inside `compute_scores`,
untouched) are unaffected. The function's docstring, which previously stated (self-
contradicting its own registered target) "i.e. exactly what `score_s_raw` computes for a
single host's likelihood", is corrected to state the ln(2)/`score_lns` identity instead.

**Regression tests added** (`darksiren_emri_test/bayesian_inference/test_theta_zwindow.py`,
36 tests total in the two zwindow test files after this addendum, all passing):
- `test_es_null_det_closed_form_uses_the_ln2_secant_denominator` -- independently re-derives
  the RAW-denominator form using the driver's own `_es_null_det_kernel` helper (same kernel/
  window machinery, only the denominator swapped) and pins `Es_null_det(ln2 form) =
  Es_null_det(raw form) * (sqrt2-1/sqrt2)/ln(2)` to `rel=1e-9` on a synthetic single-host
  fixture -- exactly the MUST_FIX's own reproduction, in both directions.
- `test_score_s_equals_score_lns_minus_es_null_det_by_construction` -- checks the pooled-
  statistic identity `mean(score_s) == mean(score_lns) - mean(Es_null_det)` on a 3-event
  synthetic fixture with a planted `es_null_det` column (distinct from the pre-existing
  `test_compute_scores_score_s_corrected_subtracts_es_null_det`, which hand-derives the
  per-event arithmetic rather than checking this pooled identity).

**Quality gates.** `uv run ruff check` / `uv run ruff format --check` clean on the driver and
the test file. `uv run mypy --ignore-missing-imports` clean on both except 8 pre-existing
`run_mirror_seed_inprocess` kwarg-splat errors at lines 484/500 -- reproduced identically
against the unmodified `HEAD` copy of the driver, confirming they predate this fix and are
unrelated to `_es_null_det_closed_form` (out of this MUST_FIX's scope; not touched).

**Recert re-run** (`--score-only`, seeds 900101-900104, nodes truth/b_plus/b_minus/s_plus/
s_minus, `--theta-sites 2.2 --smear off --theta-phi-divisor on`, out-root `tree2_20260830/
hier_s0_recert_run`): `score_b` (mean=-0.28878240960372603, sem=0.42705252094828333,
Z=-0.6762222336551854) and `score_s_raw` (mean=-0.07195958393659582,
sem=0.012051274307800423, Z=-5.971118248467597) are bit-identical to the banked
`s0a_score_output.json` values (Z_b -0.676, raw Z_s -5.971) -- unaffected by this fix, as
expected. `score_lns` now also reports explicitly (mean=-0.07340881013441411, Z=
-5.971118248467598 -- same Z as score_s_raw, since both are the same numerator over a
different but per-event-constant denominator). The corrected `score_s`/Z_s remain reported
unavailable (`score_s_available: False`, NaN, n_pooled=0) for THIS SPECIFIC recert_run: it
predates PA-HIER-32(d) and carries no `es_null_det.csv` cache in any of its four seed
directories (confirmed: no such file exists under `hier_s0_recert_run`), and per the
implementation's own design `score_s` is never silently substituted when the cache is
absent. Producing a real corrected `score_s`/Z_s/`Es_null_det` requires a full P1-class arm
(the only path that calls `compute_es_null_det_table`, needing the realized events' host
indices plus the real GLADE handler) -- out of this node's foreground/600s budget (the
verifier's own item 2 already found a bare 12-event smoke of the full driver exceeds 300s
without completing one node, since the T1.1 divisor precompute is a fixed per-seed cost).
This matches the verifier's own closing disclosure: the denominator defect "is invisible on
this particular cross-check only because this banked run has no cache to exercise it -- it
WILL be exercised the moment the registered P1 arm runs (which always computes+caches
Es_null_det fresh per seed)". As a code-level sanity check (NOT a production number): on the
synthetic single-host fixture used by the pre-existing `test_compute_es_null_det_closed_form_
matches_delta_limit` test (flat completeness, z_g=0.10, sigma_g=0.01, h=0.73, n_grid=4001),
the fixed function returns `Es_null_det=0.03453083233123442` (ln(2) denominator) vs.
`0.03384912959343959` under the pre-fix raw denominator (ratio 1.0201394465967897, matching
`(sqrt2-1/sqrt2)/ln(2)` exactly) -- illustrative only, not a registered measurement.

Ledger: `docs/gates/PHYSICS-GATE-LEDGER.md`, "verified (revised)" row appended for T1.3-zwin.
No git operations, no other files touched.

## §11 Richardson half-step nodes (T1.4)

Row #255 standing grant, tree 2 node T1.4, registered by `tree2_20260830/
T1_3_ES_NULL_DET_VALIDITY_20260830.md` section 5's falsifier (PA-HIER-33's registered-before-
any-run decisive check): two new theta nodes, `s_plus_half` (b=0, s=2^(+1/4) ≈ 1.189207) and
`s_minus_half` (b=0, s=2^(-1/4) ≈ 0.840896), at `ln s = +/-ln(sqrt2)/2` -- the half-step pair
that, combined with the existing P1 nodes `s_plus`/`s_minus` (`ln s = +/-ln(sqrt2)`), forms a
Delta^2-free Richardson-extrapolated secant.

**Code changes** (`fanout1_20260829/hier_s0_driver.py` only):

1. `THETA_NODES_HALF_STEP: dict[str, tuple[float, float]] = {"s_plus_half": (0.0, 2.0**0.25),
   "s_minus_half": (0.0, 2.0**-0.25)}` added as a module-level constant next to `THETA_NODES`/
   `NODE_ORDER`, NOT merged into either unconditionally.
2. New CLI flag `--s-half-step` (`store_true`, dest `s_half_step`). In `main()`, immediately
   after `args = ap.parse_args()` and BEFORE the existing `for n in nodes: if n not in
   THETA_NODES: raise SystemExit(...)` validation, `if args.s_half_step: THETA_NODES.update
   (THETA_NODES_HALF_STEP)` -- a one-time mutation of the module-level dict every downstream
   lookup (`run_theta_node`'s `THETA_NODES[node]` call, `gather_node_results_from_disk`'s same
   lookup, `compute_scores`'s new `has_s_half` check) reads from unchanged. Byte-identical
   when the flag is omitted: `THETA_NODES` is never touched, so `--nodes s_plus_half` without
   `--s-half-step` still hits the pre-existing "unknown node" `SystemExit` verbatim.
   `NODE_ORDER` (the default 5-node cross for `--nodes`-omitted invocations, and the `gate_eng`
   report loop) is deliberately left untouched -- the two new nodes are reachable only via an
   explicit `--nodes ...,s_plus_half,s_minus_half`, so every default-node-list invocation
   (registered P1's own 3-node-type `--nodes truth,s_plus,s_minus` included) is unaffected.
3. `compute_scores`: a third axis-readiness flag `has_s_half = not _axis_missing(("s_plus_half",
   "s_minus_half"))`, computed alongside `has_b`/`has_s` but deliberately NOT folded into the
   "at least one axis ready" gate or the broken-pair `ValueError` loop (a missing half-step
   node degrades to `n_pooled=0`/NaN, never raises -- the falsifier doc's "printing only, no
   verdict" instruction). When `has_s and has_s_half`, per channel: `S_half = [lnL(s=2^{1/4}) -
   lnL(s=2^{-1/4})] / (ln(2)/2)` per event (an inner join of `s_plus_half`/`s_minus_half` on
   `(seed, event_idx)`, mirroring the existing `s_plus`/`s_minus` join), then `score_lns_R =
   (4*S_half - score_lns) / 3` (`score_lns` = the existing `S_full` secant, unchanged), joined
   to `score_lns`'s own event set (so an event on disk for one pair but not the other is simply
   excluded, not an error). Reported per channel: `score_lns_R` (`mean`/`sem`/`Z`/`n_pooled`/
   `per_seed`, the last a `{seed: {"mean", "n"}}` dict), `score_lns_R_available` (bool), and
   `score_lns_R_minus_score_lns` (`mean`/`sem`/`n_pooled` of the paired shift `score_lns_R -
   score_lns` -- PA-HIER-33's decisive falsifier quantity, predicted `-Es_null^{(P1,nb)} =
   -0.0013 +/- 0.0008`). None of `verdict_s0a`/`verdict_s0r` read any of these three keys (both
   read only `score_b`/`score_s`), so the new statistic is confirmed printing-only by
   construction, not just by convention. `write_score_markdown` prints all three alongside the
   existing `score_b`/`score_s`/`score_s_raw`/`score_lns` lines whenever `scores` is present.

**Correctness check (synthetic, not from any evaluate() run).** A 50-event synthetic
`ln L(ln s) = 0.01*idx - ln_s^2 + 0.05*ln_s^3` fixture (quadratic + a cubic term driving the
secant's O(delta^2) bias) confirms `score_lns_R` recovers `0.0` to floating-point precision
(`3.19e-17`) where the analytic Richardson extrapolation predicts exactly `0` (no O(delta^2)
term survives), while `score_lns` on the same fixture carries the full analytically-predicted
bias (`0.0060057`, matching `a3*delta^2/6` in closed form); the paired shift
(`score_lns_R - score_lns` = `-0.0060057`) recovers the full secant bias with the opposite
sign, exactly as PA-HIER-33 section 5's own Gaussian check states. Dropping `s_minus_half`
from the synthetic fixture and re-running `compute_scores` confirms graceful degradation:
`score_lns_R_available=False`, `n_pooled=0`, `mean=nan` -- no exception.

**Smoke evidence** (seed 900101, `--event-cap 12`, foreground, `timeout 590`, builder-only per
rule 2):

    uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
      --arm S0-A --seeds 900101 --nodes s_plus_half --s-half-step \
      --theta-sites 2.2 --smear off --theta-phi-divisor on --theta-zwindow on --z-window-k 4.0 \
      --event-cap 12 --jobs 1 --total-cpu-budget 4 \
      --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_smoke_half

Timed out at 590s (killed cleanly, `pgrep -af hier_s0_driver` confirmed no orphaned process
afterward) -- as anticipated (the P1 arm's own per-s-cell cost is 705-844s evaluate-only, per
`logs/runner7_tree2_20260830.log`; `--event-cap 12` shrinks the per-event loop but not the
fixed per-seed venue/divisor-precompute cost, matching §10's own recert-run disclosure).
**Structural checks confirmed before the kill:** `node_s_plus_half_sites2.2_nosmear_divisor_
zwin_zk4/` was created under `s0a_seed900101/` with EXACTLY the same `_node_dir_suffix` P1's
own `node_s_plus_sites2.2_nosmear_divisor_zwin_zk4` uses (`_node_dir_suffix` is node-name-
agnostic, keyed only by `theta_sites`/`smear`/`theta_phi_divisor`/`theta_zwindow`/`z_window_k`
-- confirming the new node reuses every existing suffix/config path unchanged); `THETA_NODES[
"s_plus_half"] = (0.0, 1.189207115002721)` resolved correctly (`2**0.25` to full float
precision) and was forwarded through to a real venue build (`prepared_cramer_rao_bounds.csv`,
`cramer_rao_bounds.csv`, and the `injections` symlink all present and populated); `es_null_det.
csv` was cached for the seed as a side effect, exactly as for any other node. No
`event_likelihoods.csv` was produced (evaluate() never completed within the budget), so no
scoring was possible from this smoke's own output.

`--score-only` on the smoke dir (`--nodes s_plus,s_minus,s_plus_half,s_minus_half --s-half-step`,
same theta/config flags, same `--out-root`) exits 0 with `n_present_by_node` all zero (no CSVs
exist yet) and the pre-existing "on-disk node set is INCOMPLETE for pooling ... not an error"
note -- clean graceful-degradation behaviour at the CLI level, consistent with the synthetic
`compute_scores`-level check above (which is the one that actually exercises the new
`score_lns_R` fields, since this smoke's single half-node never reached `event_likelihoods.
csv`). `hier_s0_zwin_run/` (P1's own out-root, read-only per this node's scope) was inspected
only to confirm its `s0a_seed900101/node_s_plus_sites2.2_nosmear_divisor_zwin_zk4` directory
name matches the smoke's own suffix exactly -- NOT written to.

`uv run ruff check` clean; `uv run ruff format` applied (one pre-existing-style line-length
wrap, no semantic change) then reconfirmed clean. `uv run mypy --ignore-missing-imports`:
the same 10 pre-existing `run_mirror_seed_inprocess` kwarg-splat errors as §10 (lines 501/517
after this addendum's line shift; unrelated to this change, reproduced identically before it).

**The T1.4 run command** (for the orchestrator/runner; NOT executed by this node -- 8 cells: 4
seeds x 2 node types `{s_plus_half, s_minus_half}`; reuses P1's own out-root so the truth/
s_plus/s_minus nodes already banked there are not re-run):

    uv run python3 results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
      --arm S0-A \
      --nodes s_plus_half,s_minus_half \
      --s-half-step \
      --theta-sites 2.2 \
      --smear off \
      --theta-phi-divisor on \
      --theta-zwindow on \
      --z-window-k 4.0 \
      --sky-cone-k 1.5 \
      --jobs 1 \
      --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_run

**The score-only command** (after the run above completes; combines the new half-step pair
with P1's already-banked `truth`/`s_plus`/`s_minus` in the SAME out-root, so
`score_lns_R`/`score_lns_R_minus_score_lns` are populated):

    uv run python3 results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
      --arm S0-A \
      --nodes truth,s_plus,s_minus,s_plus_half,s_minus_half \
      --s-half-step \
      --theta-sites 2.2 \
      --smear off \
      --theta-phi-divisor on \
      --theta-zwindow on \
      --z-window-k 4.0 \
      --sky-cone-k 1.5 \
      --score-only \
      --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_run

Every flag must match the run command above verbatim (same reasoning as §3/the P1 score-only
note: `gather_node_results_from_disk` reconstructs the node-directory suffix from them). No
git operations, no other files touched.
