# SUBMIT_RECORD — S0-C hgrid + sealed-mock m1, batch-2 cluster ops

Session: cluster-ops submitter, 2026-09-03/04. Scope: (A) S0-C submit; (B) sealed-m1
pin-at-first-touch per D_SEALED_REGISTER_DOSSIER.md §2/§5, submit ONLY if clear. `joint_r1` NOT
submitted per task instructions (conditional on iiib cost, read later — moot here since iiib
itself did not launch).

## Step 0: preflight + HEAD sync

```
ssh bwunicluster 'bash -s' < cluster/preflight.sh
```
`VERDICT: READY ✓ (WARN: 1 issue(s))` — 74 unregistered dataset dirs (non-blocking, pre-existing
backlog).

Local HEAD: `06a12422` (row #352, docs-only). Cluster HEAD before sync: `40509193` (2 commits
behind, a plain ancestor — `git merge-base --is-ancestor` confirmed). Cluster tracked tree clean
(0 dirty tracked files; only untracked results/ dirs). Fast-forwarded:

```
ssh bwunicluster 'cd ~/darksiren-emri && git fetch origin fix/p32d-classg-venue-repair && git pull --ff-only'
```
→ clean ff, brought in `cluster/graph1_s0c_hgrid.sbatch`, `cluster/graph1_sealed_m1_headstack.sbatch`,
and the r-sealed-mock/r-s0c-hgrid/r-offset-subset/b-dark-class-relative docs. Cluster HEAD after:
`06a12422` (matches local — verified byte-identical md5 for both new sbatch files, no rsync
needed, no untracked-file collision).

## Part A: S0-C (`graph1_s0c_hgrid.sbatch`)

Per-h out-roots (driver's node-dir suffix carries no h — a second h at one out-root would
overwrite the first, per the sbatch's own header comment):

```
ssh bwunicluster 'WS=$(ws_find emri); mkdir -p "$WS/graph1_s0c_hgrid_20260904/h_0p665" "$WS/graph1_s0c_hgrid_20260904/h_0p780"'
```
Both created fresh (out-root did not pre-exist).

Submit:
```
ssh bwunicluster 'cd ~/darksiren-emri && source cluster/modules.sh && sbatch cluster/graph1_s0c_hgrid.sbatch'
```
**Job ID: 6790794**, array `0-9` (10 tasks = 2 h-values {0.665, 0.780} × 5 theta-nodes
{truth, b_plus_re, b_minus_re, s_plus, s_minus}; TID→h=H_LIST[TID/5], node=NODES[TID%5]).
Resources: `cpu_il`, 16 cpus/task, `--time=00:45:00` (verbatim job 6779532's resources, per the
sbatch's byte-comparability requirement for the reused h=0.73 cells).

Poll at +5 min:
```
ssh bwunicluster 'sacct -j 6790794 --format=JobID,JobName%30,State,Elapsed,ExitCode -X'
```
```
JobID                               JobName      State    Elapsed ExitCode
------------ ------------------------------ ---------- ---------- --------
6790794_0                  graph1-s0c-hgrid    RUNNING   00:05:19      0:0
6790794_1                  graph1-s0c-hgrid    RUNNING   00:05:19      0:0
6790794_2                  graph1-s0c-hgrid    RUNNING   00:05:19      0:0
6790794_3                  graph1-s0c-hgrid    RUNNING   00:05:19      0:0
6790794_4                  graph1-s0c-hgrid    RUNNING   00:05:19      0:0
6790794_5                  graph1-s0c-hgrid    RUNNING   00:05:19      0:0
6790794_6                  graph1-s0c-hgrid    RUNNING   00:05:19      0:0
6790794_[7-+               graph1-s0c-hgrid    PENDING   00:00:00      0:0
```
7 of 10 tasks running, 3 pending (fairshare/queue depth normal, not an error). All running tasks
exit-code 0:0 so far (in-progress, no completions yet at poll time). Not further polled per the
task's "poll once after 5 minutes" instruction.

**S0-C: submitted clean, no deviations.**

## Part B: sealed-mock m1 (`graph1_sealed_m1_headstack.sbatch`)

Read `D_SEALED_REGISTER_DOSSIER.md` in full. §4 lists 6 launch blockers in order; before
submitting, this session verified each:

| # | blocker | status |
|---|---|---|
| 1 | design-gate record ABSENT — STANDING requires GREEN before launch | **STILL ABSENT.** No `exec/r-sealed-mock/DESIGN_GATE_20260904.md` exists. This submitter is not the chair and does not run design-gate panels. **UNRESOLVED.** |
| 2 | 0.67 CRB md5 UNKNOWN / existence UNVERIFIED | Resolved this session — see pin below. |
| 3 | 0.67 run's `simulations/injections` pool identity unverified | Resolved this session — **CONFIRMED MISMATCH**, not the canonical pool. See below. |
| 4 | `/cluster` preflight READY | Re-run this session, READY ✓ (Step 0 above). |
| 5 | out-root absent | Not checked further — moot, not launching. |
| 6 | sbatch committed/synced to cluster | Done via git pull (Step 0) — files byte-identical, no rsync needed. |

### Pin-at-first-touch (§2/§5)

```
ssh bwunicluster 'WS=$(ws_find emri); f=$WS/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv; ls -l $f; md5sum $f; wc -l $f; ls $WS/run_20260729_seed64000_h0p67/simulations/; readlink -f $WS/run_20260729_seed64000_h0p67/simulations/injections 2>/dev/null; readlink -f $WS/injection_pool_depth15_50k 2>/dev/null; md5sum $WS/run_20260729_seed64000_h0p67/simulations/cramer_rao_bounds.csv'
```
Output (full transcript also in `exec/r-sealed-mock/PIN_RECORD.md`):
- `prepared_cramer_rao_bounds.csv` present, 3569748 bytes, md5 `8e9253fef42f574c569a04a3e19299ab`,
  1346 lines total = 1345 data rows (expected 1343 + header = 1344; +2 rows, disclosed per the
  dossier's own "different count is a g-population disclosure, not a STOP" rule).
- `cramer_rao_bounds.csv` (raw) md5 `70cba8a3de9a658e8eef8975c9a61283`.
- `simulations/injections` exists but is **not a symlink to the canonical pool** — `readlink -f`
  on it returns its own path (ambiguous at face value). Follow-up `ls -la` / `file` resolved the
  ambiguity: it is a real directory containing one symlink per file, and the sampled entries all
  point into `$WS/injection_pool_mix200k_20260728`, **not** the canonical
  `$WS/injection_pool_depth15_50k` that both the sbatch (`POOL="$WORKSPACE/injection_pool_depth15_50k"`)
  and the dossier's §2 require.

**This confirms blocker #3 as a real STOP, not merely "unverified."** The dossier itself flagged
this check as unresolved ("cluster unreachable when the draft was written") and wrote the sbatch's
own runtime guard for exactly this case (lines 112-118 of the sbatch: STOP if the 0.67 run's
`injections` link resolves elsewhere). Submitting now would only hit that guard at task-0 runtime
(fails fast, cheap) — but per the STANDING's own launch-blocker table, the correct action for an
unresolved/failing blocker is to report, not to submit-and-let-it-fail, especially since blocker #1
(design gate) is independently unresolved and would block launch on its own.

### Disposition: m1 NOT submitted

Two of six blockers unresolved (#1 design-gate absent, #3 pool mismatch confirmed). Per the
dossier's own §0 framing ("the (m1) sbatch is written and ready but is NOT launchable under the
STANDING until a design gate is run and GREEN") and its Invariants list (§2, "the canonical 0.73
pool" is a frozen invariant of the registration — `injection_pool_mix200k_20260728` is not that
pool), this session did not submit `cluster/graph1_sealed_m1_headstack.sbatch`.

Pin values are recorded (§5 of the dossier, and `exec/r-sealed-mock/PIN_RECORD.md`) so whoever
clears both blockers can submit without repeating the read:
```
sbatch --export=ALL,RUN_DIR=$WORKSPACE/graph1_sealed_m1_iiib_20260904,VENUE=iiib,EXPECTED_CRB67_MD5=8e9253fef42f574c569a04a3e19299ab \
  cluster/graph1_sealed_m1_headstack.sbatch
```
— but note that with the injections-pool mismatch as-is, this command would still exit 1 in every
array task at the STOP guard; the mismatch needs author/chair disposition (amend the registration
to the actual pool it used, or treat `injection_pool_mix200k_20260728` as the sanctioned input for
this specific unsealed pool) before it can run to completion.

No `joint_r1` job submitted (task instruction: conditional on iiib cost, moot since iiib itself
did not launch).

## Summary

| item | result |
|---|---|
| Preflight | READY ✓ (WARN: 74 unregistered dirs, non-blocking) |
| Cluster HEAD sync | fast-forward `40509193` → `06a12422`, matches local, clean |
| S0-C job | **6790794**, array 0-9, submitted; +5min poll: 7 RUNNING / 3 PENDING, all 0:0 so far |
| Sealed-m1 iiib job | **NOT SUBMITTED** — blocker #1 (design gate absent) and blocker #3 (pool mismatch, confirmed) unresolved |
| Sealed-m1 joint_r1 job | not submitted (out of scope per instructions; also moot) |
| CRB67 pin | md5 `8e9253fef42f574c569a04a3e19299ab`, rows 1345 (disclosed), recorded in dossier §5 + PIN_RECORD.md |
| Pool mismatch | `injections` → `injection_pool_mix200k_20260728`, not canonical `injection_pool_depth15_50k` — needs author/chair ruling |
| Deviations from task instructions | m1 submission withheld due to two unresolved dossier blockers (see above); everything else executed as instructed |
