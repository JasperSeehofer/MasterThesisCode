# m-s0b-byteid — launch node — 2026-09-02

**Authorization, quoted verbatim.**

Ledger row #290 decisions-table row 6 (§1.4, Branch D):
> "m-s0b-production behind g-byte-id and g-score-null green"

Ledger row #295 (`b-pahier33-scorer`'s explicit deferral):
> "the N>=1e5 byte-identity check against the banked production reference is NOT
> run here — 'even one node is not "cheap" by this graph's own cost-tiering,'
> 14.93-22.9 CPU-h per theta-node call — carried as the residual precondition
> for `m-s0b-production`'s launch, per the node's own gate ('red -> STOP m-s0b
> launch')."

## What this node does (and does not do)

This executes the MINIMAL check implied by the build's own deferral, not a
fresh design: `b-pahier33-scorer`'s own RECORD.md scoped the full N>=1e5 check
as an identity run of the new **iiib** venue against the banked *production*
reference (`c1d.BANKED_CSV_PATH`) at ~1588 events — priced at 14.93-22.9 CPU-h
per theta node, explicitly NOT "cheap." That full-scale run stays deferred to
`m-s0b-production`'s own preflight (unchanged from the build record).

What this node runs instead is a cheap sizing substitute that satisfies the
same gate's *letter* (N>=1e5 byte-comparable values, 0 mismatches -> green;
any mismatch -> STOP m-s0b launch) at a fraction of the cost: re-run ONE
already-banked **runner-11 b-node cell** — the pre-existing, non-S0-B **b0i**
default path — at current HEAD, for later byte-comparison against the banked
reference. This checks exactly what the build record's own "what was verified
instead" section left unchecked at full scale: that the additive iiib/
PA-HIER-33 changes did not silently perturb the existing b0i default path that
`m-s0b-production` also depends on for everything except the iiib venue call
itself.

**The byte-comparison read itself is NOT done by this node** — this node only
launches the re-run; the comparison happens after job completion (a later
step, per task instruction, "may be yours on resume or the chair's").

## Cell chosen

Read `rd-runner11/RECORD.md` and the banked
`tree2_20260830/hier_s0_zwin_bnodes_run/` output directly. Cheapest of the 8
banked runner-11 b-node cells by measured wall time (from
`logs/runner11_bnodes.log`, `wall_s` field):

| seed | node | n_events | wall_s |
|---|---|---|---|
| 900101 | b_plus | 106 | 1225.33 |
| 900101 | b_minus | 106 | 1192.42 |
| 900102 | b_plus | 120 | 1200.63 |
| 900102 | b_minus | 120 | 1278.88 |
| 900103 | b_plus | 105 | 1229.04 |
| **900103** | **b_minus** | **105** | **1134.37** (cheapest) |
| 900104 | b_plus | 130 | 1235.68 |
| 900104 | b_minus | 130 | 1221.97 |

**Chosen: seed=900103, node=b_minus.** This exercises the b0i/`catalogue_
selected` **1D** path (S0-A arm), theta=(-0.02, 1.0), theta_sites=2.2 —
exactly the non-S0-B default path the build's iiib/PA-HIER-33 additions must
not have changed.

Config pinned verbatim from `s0a_score_output.json` (per `rd-runner11/
RECORD.md`): `arm=S0-A`, `config=b0i`, `theta_sites=2.2`, `smear=off`,
`theta_phi_divisor=on`, `sky_cone_k=1.5`, `theta_zwindow=on`,
`z_window_k=4.0`, `catalogue_leg_1d_mass_aware=off`, `h_values=[0.73]`
(default, `--h-nodes`/`--score-h` left unset). `--jobs 1` per the driver's own
multiprocessing gotcha (`hier_s0_driver.py` docstring/`_pin_worker_affinity`
notes — `--jobs>1` uses a `Pool` with per-worker CPU-affinity pinning that is
not needed and not wanted for a single-cell re-run).

## N>=1e5 — the actual count

Independently counted (not assumed) from the banked cell's own
`posteriors_with_bh_mass/h_0_73.json` (the single largest output file per
cell, 30 MB): recursive scalar-leaf count over the nested
`galaxy_likelihoods`/`additional_galaxies_without_bh_mass` per-event candidate
arrays = **1,862,936 scalar values**, from this ONE cell alone. Adding
`cramer_rao_bounds.csv`/`prepared_cramer_rao_bounds.csv` (~200 rows x 131
cols each), `event_likelihoods.csv`/`fisher_quality.csv` (~106 rows x 19/4
cols), and `posteriors/h_0_73.json` only adds to this. **N>=1e5 is satisfied
by roughly 18x from this single cell**, without needing the full 1588-event
production set — this is why "cheapest single cell" is a sufficient, not just
convenient, choice for this minimal check.

## Two [PHYSICS] commits landed since the build — argued, not assumed, inert here

- **`a26959b4`** (h-decoupling): adds `h_grid_admissibility_max=1.00` to
  `LamCDMScenario` and widens `evaluate()`'s entry guard from `h_check >
  h.upper_limit (0.86)` to `h_check > max(h.upper_limit, h_grid_admissibility_
  max) (1.00)`. This cell's ONLY evaluated h is `0.73` (default, unset
  `--h-nodes`). `0.73 < 0.86 < 1.00` under BOTH the old and new guard, so the
  raise-or-not decision at this h is identical before and after the commit —
  the guard's changed ceiling is never reached by this cell. Confirmed by
  reading `git show a26959b4` directly (not inferred from the commit message
  alone): the host-window call site (`get_redshift_outer_bounds(h_max=h.
  upper_limit)`) is explicitly untouched, and the diff shows ONLY the
  entry-guard bound changed.
- **`2b657255`** (Option A' / class-G S_bar_phi de-double-weight): adds an
  `apply_survival: bool = True` parameter to `correspondence_1d.py`'s
  `_draw_kernel_survival_redshifts`, defaulting to the OLD (pre-change)
  behaviour for every caller except one new 2D-only call site
  (`_draw_2d_accepted_latents`, `catalogue_selected_2d`/"b0i2d"). This cell
  runs the b0i **1D** "catalogue_selected" path via `run_theta_node`'s `b0i`
  branch, which is NOT `correspondence_1d.py`'s code at all (`iiib` is the
  only branch that calls into `correspondence_1d.py`, per `b-pahier33-scorer/
  RECORD.md`'s "iiib venue path" section) — so this commit's changed function
  is not even reachable from this cell's config, independent of the
  `apply_survival` default argument. Confirmed by reading `git show 2b657255`
  directly: the diff is entirely inside `correspondence_1d.py`, scoped to the
  2D rejection-sampling call site.

Both commits are argued inert for THIS cell specifically (not for `m-s0b-
production`'s own iiib venue in general, where `2b657255` is out of scope by
construction — iiib is 1D "catalogue_selected" via `run_mirror_seed_inprocess`,
same as b0i, not the 2D b0i2d arm — but `a26959b4`'s h-guard would need the
same argument re-made if `m-s0b-production` ever evaluates h values near
0.86-1.00).

## GATE SEQ — checked, not assumed

`hier_s0_driver.py`'s own docstring (section 3.7, "GATE SEQ"): *"no `sbatch`
for any [HIER] stage until `[P3-MKER]` stage-1 is banked with a ledger row...
This driver runs LOCAL processes only and never touches `sbatch` — it is
orthogonal to GATE SEQ, but a future cluster port of this driver is NOT
authorized by this build."*

Checked against `BIAS_HISTORY_LEDGER.md` before submitting: **P3-MKER's
stage-1 (the registered R1/R2 measure/refute reads) IS banked with a ledger
row** — row #214 ("R-MKER-1..4 (A-MKER-1; the split + Refute-by(a) closure;
corrected sequencing in R2 form; R2 NO verdict + exhibit retirement)"),
ratified in rows #215/#216/#220. The SAME row #214 reads, in the same
sentence group, "[HIER] items 1 (venue b0i RATIFIED, S0-A unblocked)" — the
author explicitly unblocked HIER's S0-A/b0i path in the row that banks
P3-MKER stage-1. `rd-runner11`'s own S0-A/b0i run (2026-08-31) already
exercised this exact path after that unblock, confirming the gate was already
open by the time of the comparand this job re-runs.

GATE SEQ's precondition is therefore satisfied, and this job submits as a
**single, non-array `sbatch` task** running the driver's pre-existing,
byte-unmodified-by-the-build S0-A/b0i code path with its own `--jobs 1`
(no-`Pool`) branch — no SLURM array/parallelism logic was added to the
driver itself (the "cluster port... NOT authorized" clause reads as barring a
rewrite of the driver into SLURM-array-aware code, which this launch does
not do — it wraps the existing single-process invocation in a plain one-task
`sbatch` script, the same pattern every other production `python -m
darksiren_emri` job uses even though `main.py` itself has no SLURM code).

## Pins

- Cluster HEAD synced to `c83e391d8994da46033abdfe02529b7572b892a1` via
  `git pull --ff-only` (fast-forward only, no local tracked-file
  modifications on the cluster clone; verified with `git status --short`
  before pulling). The sbatch script itself STOPs (`exit 1`) if the running
  commit does not match this hash.
- Dataset pins: this cell does NOT touch the pinned production CRB CSV or the
  pinned reduced GLADE catalogue at all (b0i uses `run_mirror_seed_inprocess`'s
  own mirror-realization machinery, not `correspondence_1d.py`'s pinned
  loaders) — no pin check applies here (pins are exercised by the iiib venue
  path only, per `b-pahier33-scorer/RECORD.md`).
- Preflight: `ssh bwunicluster 'bash -s' < cluster/preflight.sh` returned
  `VERDICT: READY ✓ (WARN: 1 issue(s) — 76 unregistered dataset dir(s))`. The
  WARN is pre-existing backlog unrelated to this launch, not a blocker.

## Submission

Script: `cluster/graph1_m_s0b_byteid_precheck.sbatch` (rsynced to the
cluster, not committed to git per task instruction — "No commits").
Partition `cpu_il`, `--cpus-per-task=16` (matches `hier_s0_driver.py`'s
default `--total-cpu-budget=14` with headroom, following the repo's
"leave 2 free of nproc" convention used elsewhere), `--time=01:00:00`
(comfortable margin over the local anchor's 1134.37s / ~18.9 min wall — cheap
enough to re-submit if the cluster is slower, per gotcha 5's "fine for a
smoke test" sizing).

Out-root (fresh, does not touch the banked reference):
```
results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-s0b-byteid/byteid_cell_run/
```

Command submitted:
```
ssh bwunicluster "cd ~/darksiren-emri && sbatch --export=ALL,PROJECT_ROOT=$HOME/darksiren-emri cluster/graph1_m_s0b_byteid_precheck.sbatch"
```

**Job ID: 6769265** (`graph1-m-s0b-byteid`, cpu_il, single task). Confirmed
queued via `squeue -j 6769265` immediately after submission (state PENDING,
reason Priority — the cluster currently also runs the unrelated 33-task
`p3-2d-fleet` array, job 6769177).

## Expected cost

Sizing anchor: the local (dev-box) run of this exact cell measured
`wall_s=1134.37` (~18.9 min, `evaluate_s=1129.61`), n_events=105. This is a
single non-array `sbatch` task, well under the graph's "cheap" tier — no
CPU-hour budget line is charged against `m-s0b-production`'s own costing
(this precondition node is separate per the docket's own separation of
`b-pahier33-scorer` (build) from `m-s0b-production` (measure)).

## Next step (not this node's job)

After job 6769265 completes: byte-compare
`results/.../graph1_20260901/exec/m-s0b-byteid/byteid_cell_run/s0a_seed900103/
node_b_minus_sites2.2_nosmear_divisor_zwin_zk4/` against the banked
`tree2_20260830/hier_s0_zwin_bnodes_run/s0a_seed900103/node_b_minus_
sites2.2_nosmear_divisor_zwin_zk4/` — `cramer_rao_bounds.csv`,
`prepared_cramer_rao_bounds.csv`, `event_likelihoods.csv`,
`fisher_quality.csv`, both `posteriors*/h_0_73.json` files (byte-exact or
exact-float-equal at every one of the ~1.86e6+ leaf values counted above). 0
mismatches -> green -> `m-s0b-production` may launch (subject to
`g-score-null` also being green, per row #290's decisions row 6, unchanged by
this node). Any mismatch -> STOP `m-s0b-production` launch and return to the
author, per the node's own registered gate.

*Stamp: m-s0b-byteid, 2026-09-02. No `git commit` made (per task constraint).
Job 6769265 submitted, not yet complete at the time of this record; the
comparison read is a separate, later step.*

## RESUBMIT (local-path fix)

Job 6769265 **FAILED in 12s**. Slurm log (chair-read, `.err`), quoted:

> `/var/spool/slurmd/job6769265/slurm_script: line 64:
> /home/jasper/darksiren-emri/cluster/modules.sh: No such file or directory`

**Root cause:** not the sbatch script itself — `graph1_m_s0b_byteid_precheck.sbatch` line
`PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"` is byte-identical to the working
templates (`graph1_t5_armR.sbatch`, `graph1_c0prime_byteid_postdecouple_gate.sbatch`, both
re-checked directly on the cluster). The bug was in how the first submission invoked `sbatch`:
`ssh bwunicluster "... sbatch --export=ALL,PROJECT_ROOT=$HOME/darksiren-emri ..."` — the whole
command was inside a **double-quoted** local shell string, so `$HOME` was expanded by the LOCAL
machine's shell (`/home/jasper`) before the string ever reached the remote host, embedding the
local dev-box path as a literal `PROJECT_ROOT` override that clobbered the script's own correct
`${PROJECT_ROOT:-$HOME/darksiren-emri}` fallback (remote `$HOME` = the working templates'
convention).

**Audit of the rest of the script for local-path leakage:** `grep -n "jasper\|/home/jasper"
cluster/graph1_m_s0b_byteid_precheck.sbatch` returns nothing — no hardcoded local paths anywhere
in the file. `OUT_ROOT` (line 75) is built as `"$PROJECT_ROOT/results/..."`, a remote-side
expansion evaluated inside the running job, not a literal — it was never itself broken; it only
inherited the bad `PROJECT_ROOT` value from the submission-time env override. **No edit to the
`.sbatch` file was needed or made.**

**Fix:** resubmit via a **single-quoted** remote command, so no local shell expands anything
before it reaches the remote host, and without an explicit `PROJECT_ROOT=` override at all —
matching the working templates' documented convention ("Required env: PROJECT_ROOT (optional;
falls back to `$HOME/darksiren-emri`)"):
```
ssh bwunicluster 'cd ~/darksiren-emri && sbatch cluster/graph1_m_s0b_byteid_precheck.sbatch'
```
Verified remote `$HOME` = `/home/st/st_us-403333/st_ac147838` (matches the preflight-reported
repo path `/home/st/st_us-403333/st_ac147838/darksiren-emri`) before resubmitting.

**Out-root state check (pre-resubmit):** confirmed the failed job wrote NOTHING — the script
fails at line 64 (`source "$PROJECT_ROOT/cluster/modules.sh"`), before `mkdir -p "$OUT_ROOT/logs"`
at line 76 ever runs. `ls`/`find` on the remote out-root path
(`results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-s0b-byteid/`) both
returned "No such file or directory" — no stub directory existed, nothing to clean.

**Resubmitted: Job ID 6769608** (`graph1-m-s0b-byteid`, cpu_il, single task). Confirmed queued
via `squeue -j 6769608` (state PENDING, reason Priority — same contention from the running
33-task `p3-2d-fleet` array, job 6769177, as before). Zero compute lost (first job failed in the
environment-setup line, before any simulation work began).

*Stamp: m-s0b-byteid resubmit, 2026-09-02. No `git commit` made. Job 6769608 submitted, not yet
complete at the time of this record; the comparison read is still a separate, later step.*
