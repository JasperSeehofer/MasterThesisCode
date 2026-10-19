# WING_RERUN_LAUNCH — G-EXT wing rerun (Research Graph 1, Branch I)

Launched 2026-09-02 by the G-EXT wing rerun launcher agent.

## Authorization (quoted)

- Row #304 (item 4b): GRANTED, cap raised to 25 CPU-h.
- Row #308 item 6: physics-change ratified.
- Byte-id gate GREEN: `exec/b-hprior-fix/byteid_eval/VERIFICATION_RECORD.md` — `[PHYSICS]`
  `a26959b4` evidenced; in-bound identity 8/8 files md5-identical (job 6768603 vs. comparand
  6764460), 0 mismatches, N >> 1e5 values compared.

All three preconditions for the 14-task wing rerun (RECORD.md §3, originally drafted 2026-09-01)
are met; this launch executes that plan.

## Preflight

```
ssh bwunicluster 'bash -s' < cluster/preflight.sh
...
VERDICT: READY ✓ (WARN: 1 issue(s))
  • 75 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
The WARN is the same pre-existing, unrelated backlog item noted in prior launches (gotcha 11) —
not a blocker.

## Cluster HEAD verification

Cluster checkout (`/home/st/st_us-403333/st_ac147838/darksiren-emri`):
```
$ git log --oneline -3
dcb2c470 docs: docket items 6-8 — rows #308 addendum + Option A' ratification ask (item 8)
8f3a52a4 docs: gate-ledger rows point at the landed [PHYSICS] commit a26959b4
a26959b4 [PHYSICS] decouple h grid-admissibility from the host-window bound (rows #293/#301/#304/#308-#309)
```
HEAD `dcb2c470` is at/after `a26959b4` — the h-decoupling fix is present. 0 tracked
modifications (`git status --porcelain | grep -vc '^??'` = 0). Confirmed in the checked-out
`darksiren_emri/cosmological_model.py`: `self.h_grid_admissibility_max = 1.00` (line 407).

## RUN_DIR verification

`$WS/run_20260831_a18_ma1d_iiib` ($WS = `/pfs/work9/workspace/scratch/st_ac147838-emri`) confirmed
present, not archived/cleaned. 41 banked posterior pairs found (`simulations/posteriors/h_0_*.json`
× 41, `simulations/posteriors_with_bh_mass/h_0_*.json` × 41 = 82 files), spanning h = 0.600–0.860
(H_GRID_41), none at or above h = 0.870 — i.e. exactly the 41 completed tasks and none of the
14 wing tasks, matching the disclosed failure state.

## Guard check

`bayesian_statistics.py:4664-4678` (post-decoupling): the evaluate() admissibility guard now
uses `_h_admissible_max = max(cosmological_model.h.upper_limit, h_grid_admissibility_max)` =
`max(0.86, 1.00)` = `1.00`. All 14 wing h-values (0.870–1.000) are `<= 1.00` and therefore
admissible; `h.upper_limit` itself (0.86, the host-window bound feeding
`get_redshift_outer_bounds`) is unchanged, so the 41 banked in-bound tasks are unaffected —
consistent with the byte-id GREEN evidence above.

## Submission

Scoped array submit (not the full `submit_a18.sh` wrapper, which submits the full 0-54 range
and whose own out-root-absent precondition does not apply here — RUN_DIR is expected to exist
with 41 banked tasks):

```
$ source cluster/modules.sh   # WORKSPACE=/pfs/work9/workspace/scratch/st_ac147838-emri
$ RUN_DIR="$WORKSPACE/run_20260831_a18_ma1d_iiib"
$ sbatch --parsable --array=41-54 --export=ALL,RUN_DIR="$RUN_DIR" \
    cluster/a18_ma1d_headreadout_iiib.sbatch
6768824
```

**Job ID: `6768824`** (array 41-54, i.e. `6768824_41` .. `6768824_54`), state PENDING at
submission time. Working dir: `$WS/run_20260831_a18_ma1d_iiib` (same RUN_DIR as the 41 banked
tasks — output set completes to 55 nodes on success).

### Note on a launch-time mistake, caught and corrected

An initial submission attempt piped `source cluster/modules.sh` through `| tail -5` for log
brevity; piping puts the `source` in a subshell, so `$WORKSPACE` did not persist to the parent
shell and `RUN_DIR` resolved to `/run_20260831_a18_ma1d_iiib` (workspace-root-relative, wrong).
That misconfigured array (job `6768820`) was caught immediately post-submission (before any task
began running — `sacct` showed `PENDING`) and `scancel`ed cleanly (`CANCELLED+`, exit `0:0`, no
compute consumed). The corrected submission (this record, job `6768824`) resolves `$WORKSPACE`
via an unpiped `source`, verified against the true RUN_DIR path before resubmission.

## Cost

14 tasks × ~1.7 CPU-h/task (per-task cost anchor unchanged from the original G-EXT costing,
`AMENDMENT G-EXT`: 55 × ~1.7 CPU-h ≈ 94 CPU-h) ≈ **23.8 CPU-h**, within the row #304 item 4b
raised cap of 25 CPU-h (headroom ≈1.2 CPU-h). Not recomputed as over cap — no STOP triggered.

## Expected read (not evaluated by this launch)

Per row #286 / the branch charter: wing posteriors are expected negligible-weight (tail ~5e-13
at h ≥ 0.85). This rerun's purpose is only to complete the 55-node G-EXT grid to a usable state
for future registrations — it does not itself adjudicate any flip or claim. Readout/retrieval is
the orchestrator's next step once job `6768824` completes.

No commits were made as part of this launch.
