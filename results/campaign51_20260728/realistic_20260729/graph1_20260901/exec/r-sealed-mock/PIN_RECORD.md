# PIN_RECORD — sealed-mock stage (m1), pin-at-first-touch

Taken by: batch-2 cluster-ops submitter session, 2026-09-03/04. Read-only ssh commands only.
Per D_SEALED_REGISTER_DOSSIER.md §2/§5.

## Command run

```
ssh bwunicluster 'WS=$(ws_find emri); f=$WS/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv; ls -l $f; md5sum $f; wc -l $f; ls $WS/run_20260729_seed64000_h0p67/simulations/; readlink -f $WS/run_20260729_seed64000_h0p67/simulations/injections 2>/dev/null; readlink -f $WS/injection_pool_depth15_50k 2>/dev/null; md5sum $WS/run_20260729_seed64000_h0p67/simulations/cramer_rao_bounds.csv'
```

## Raw output

```
=== ls -l ===
-rw-r--r-- 1 st_ac147838 st_us-403333 3569748 Jul 29 20:43 /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv
=== md5sum ===
8e9253fef42f574c569a04a3e19299ab  /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv
=== wc -l ===
1346 /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv
=== ls simulations dir ===
cramer_rao_bounds.csv
diagnostics
fisher_quality.csv
fisher_quality_diagnostic.pdf
injections
master_thesis_code_20260729_205621_h_0_73.log
master_thesis_code_20260729_205800_h_0_73.log
posteriors
posteriors_with_bh_mass
prepared_cramer_rao_bounds.csv
prepared_cramer_rao_bounds.meta.json
run_metadata_combine.json
=== readlink injections ===
/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed64000_h0p67/simulations/injections
=== canonical pool readlink ===
/pfs/work9/workspace/scratch/st_ac147838-emri/injection_pool_depth15_50k
=== cramer_rao_bounds.csv (raw) md5 ===
70cba8a3de9a658e8eef8975c9a61283  /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260729_seed64000_h0p67/simulations/cramer_rao_bounds.csv
```

Follow-up (to resolve the `readlink -f injections` ambiguity — is it a symlink to a pool dir, or
a real directory of per-file symlinks?):

```
ssh bwunicluster 'WS=$(ws_find emri); ls -la $WS/run_20260729_seed64000_h0p67/simulations/injections | head; file $WS/run_20260729_seed64000_h0p67/simulations/injections'
```

Result: `injections` is a **real directory** (not a top-level symlink), containing one symlink
per file, e.g. `injection_h_0p73_task_0.csv -> $WS/injection_pool_mix200k_20260728/injection_h_0p73_task_0.csv`.
All sampled entries point into `injection_pool_mix200k_20260728`, **not** the canonical
`injection_pool_depth15_50k` named in the sbatch/dossier.

## Pin values

| field | value |
|---|---|
| `prepared_cramer_rao_bounds.csv` md5 | `8e9253fef42f574c569a04a3e19299ab` |
| `prepared_cramer_rao_bounds.csv` rows | 1346 total lines = 1345 data rows (expected 1343 + header = 1344; disclosure, +2 rows, not a STOP per dossier §2) |
| `cramer_rao_bounds.csv` (raw) md5 | `70cba8a3de9a658e8eef8975c9a61283` |
| `simulations/injections` resolution | real dir, per-file symlinks → `injection_pool_mix200k_20260728` |
| canonical pool (`injection_pool_depth15_50k`) | present at `$WS/injection_pool_depth15_50k` |
| **pool match?** | **NO — MISMATCH** |
| cluster HEAD at pin time | `06a12422c07623c53acfa1ca268dda2c7017dc3d` (contains `081b1f28`; matches local HEAD post fast-forward) |

## Disposition

**m1 is NOT submitted.** Two of the dossier's own §4 launch blockers are unresolved:

1. **Blocker #1 (design-gate record ABSENT).** No `DESIGN_GATE_20260904.md` exists under
   `exec/r-sealed-mock/`; the docket-2.2 STANDING requires a green design gate before this node
   is launchable. This submitter is not the chair and does not run design-gate panels.
2. **Blocker #3 (pool mismatch), now CONFIRMED, not merely unverified.** The dossier flagged this
   as unverified ("cluster unreachable when the draft was written"); the pin-at-first-touch
   check above resolves it: the 0.67 run's `simulations/injections` points into
   `injection_pool_mix200k_20260728`, not the canonical `injection_pool_depth15_50k` the sbatch
   (line 92, `POOL="$WORKSPACE/injection_pool_depth15_50k"`) and this dossier's §2 both require.
   Running the sbatch as written would hit its own STOP guard (lines 112-118) at task runtime —
   submitting it now would only burn a few seconds of one array task before every task exits 1,
   but the correct action per the STANDING is to report and let the author/chair decide whether
   the (m1) registration itself needs to be amended (pool identity is one of the frozen
   invariants in §2 "Invariants (A10)": "the canonical 0.73 pool" — `injection_pool_mix200k_20260728`
   is not that pool).

`EXPECTED_CRB67_MD5` is available (`8e9253fef42f574c569a04a3e19299ab`) for whoever clears both
blockers and submits; this record does not submit.

---

## PIN CORRECTION (chair, 2026-09-04)

Chair decision: the registration's pool expectation (`injection_pool_depth15_50k`, 500 files) was
a factual error. The pin is corrected to the pool the 0.67 run ACTUALLY used. Re-verified this
session, batch-2 cluster-ops submitter:

```
ssh bwunicluster 'WS=$(ws_find emri);
readlink -f "$WS/run_20260729_seed64000_h0p67/simulations/injections"; file "$WS/run_20260729_seed64000_h0p67/simulations/injections";
ls "$WS/run_20260729_seed64000_h0p67/simulations/injections" | wc -l;
POOL="$WS/injection_pool_mix200k_20260728"; ls "$POOL" | wc -l; ls "$POOL" | sort | md5sum;
md5sum "$WS/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv";
wc -l "$WS/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv"'
```

| field | value |
|---|---|
| `injections` dir file count | 707 |
| `injection_pool_mix200k_20260728` file count | 707 (matches) |
| pool file-list md5 (sorted `ls`) | `a1dffdf561c51c8c778dce115c5fb371` |
| no manifest file found in pool dir | (checked `*.json`/`MANIFEST*`, none present — file-list md5 used instead) |
| `prepared_cramer_rao_bounds.csv` md5 | `8e9253fef42f574c569a04a3e19299ab` (re-confirmed, unchanged) |
| `prepared_cramer_rao_bounds.csv` rows | 1346 total lines = 1345 data rows (unchanged) |

**1345-vs-1344 disclosure (report, not judged):** local `results/.../closure_seed64000_h0p67/combined_posterior_2d.json`
records `n_events_total=1343, n_events_used=1343, n_events_excluded=0, n_events_empty=2` for the
same 0.67 run (posteriors_with_bh_mass variant) — i.e. the local evaluation used 1343 events. The
cluster CRB has 1345 data rows. 1343 + 2 (empty) = 1345, which matches the cluster row count
exactly, suggesting the 2-row gap between the CRB (1345) and the local dossier's expected-1343
figure is the same 2 "empty" events the local diagnostic already accounts for — but the local
`diagnostic_report.md` for this same run separately states "Empty events (all NaN): 0", which is
inconsistent with `n_events_empty=2` in the same run's combined-posterior JSON. Flagging both
numbers for the chair; not resolving the internal inconsistency here.

`cluster/graph1_sealed_m1_headstack.sbatch` pool-expectation lines corrected to
`injection_pool_mix200k_20260728` / 707 files (diff in `SUBMIT_RECORD_m1.md`); all other lines
byte-identical.

## GUARD CORRECTION (chair, 2026-09-04)

The identity check (`readlink -f injections` vs `readlink -f $POOL`) in
`cluster/graph1_sealed_m1_headstack.sbatch` can never pass: the 0.67 run's `injections` is a
real directory of symlinks, not a symlink. Replaced with a content guard: per-file md5 manifest
of the injections dir, list-md5'd, compared to a pinned `EXPECTED_POOL_LIST_MD5`.

Re-measured this session (cluster-ops, read-only): 707 files; manifest list-md5 =
`75f4030d5d3b0405fd948049bef5767e` — EXACT match to `exec/r-timeout-selection/POOL_MANIFEST.md5`
(707 lines, same list-md5). Content-identical to the canonical pool
`injection_pool_mix200k_20260728`. Guard block replaced in the sbatch (diff in
`SUBMIT_RECORD_m1.md`); submitted `EXPECTED_POOL_LIST_MD5=75f4030d5d3b0405fd948049bef5767e`,
`EXPECTED_CRB67_MD5=8e9253fef42f574c569a04a3e19299ab`, `VENUE=iiib`.
