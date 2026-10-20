# SUBMIT_RECORD_m1 — sealed-mock stage m1 pin correction + submit, batch-2 cluster ops

Session: cluster-ops submitter, 2026-09-04. Continuation of `SUBMIT_RECORD_s0c_m1.md`. Design
gate confirmed GREEN (`exec/r-sealed-mock/DESIGN_GATE_computability_m1.md`, read in full). Chair
decision (flagged, recorded verbatim below) resolves the remaining blocker: the registration's
pool expectation (`injection_pool_depth15_50k`, 500 files) was a factual error; the pin is
corrected to the pool the 0.67 run actually used.

## Step 1: cluster verification (read-only)

```
ssh bwunicluster 'WS=$(ws_find emri);
readlink -f "$WS/run_20260729_seed64000_h0p67/simulations/injections"; ...'
```

- `simulations/injections` is a real directory (not a top-level symlink); per-file symlinks all
  resolve into `$WS/injection_pool_mix200k_20260728`.
- `injections` dir file count: **707**. `injection_pool_mix200k_20260728` file count: **707**
  (match). No manifest file exists in the pool dir (checked `*.json`/`MANIFEST*`); used sorted
  `ls | md5sum` instead: **`a1dffdf561c51c8c778dce115c5fb371`**.
- `prepared_cramer_rao_bounds.csv` md5 = **`8e9253fef42f574c569a04a3e19299ab`** (matches the
  expected pin) · `wc -l` = 1346 total lines = **1345 data rows**.
- **1345-vs-1344 disclosure (reported, not judged):** local
  `results/campaign51_20260728/realistic_20260729/closure_seed64000_h0p67/combined_posterior_2d.json`
  for the same 0.67 run records `n_events_total=1343, n_events_used=1343, n_events_excluded=0,
  n_events_empty=2` (posteriors_with_bh_mass variant). 1343 + 2 = 1345, matching the cluster row
  count — but the same run's `posteriors/diagnostic_report.md` separately states "Empty events
  (all NaN): 0", inconsistent with the JSON's `n_events_empty=2`. Both numbers flagged for the
  chair; internal inconsistency not resolved here.

Full transcript appended to `exec/r-sealed-mock/PIN_RECORD.md` (§"PIN CORRECTION (chair,
2026-09-04)").

## Step 2: sbatch edit (pool-expectation lines only)

`cluster/graph1_sealed_m1_headstack.sbatch` diff:

```diff
-#   pool:  $WORKSPACE/injection_pool_depth15_50k (datasets.yaml canonical, h_ref 0.73, 500 files)
-#          = the pool the 0.67 simulate itself used; STOP if the file count != 500, and STOP if
+#   pool:  $WORKSPACE/injection_pool_mix200k_20260728 (PIN CORRECTION, chair 2026-09-04: the
+#          registration's expectation of injection_pool_depth15_50k was a factual error; the
+#          0.67 run's own injections link resolves to this pool -- 707 files, measured at
+#          first touch), STOP if the file count != 707, and STOP if
 #          the 0.67 run dir's own simulations/injections link resolves elsewhere.
@@
-POOL="$WORKSPACE/injection_pool_depth15_50k"
+POOL="$WORKSPACE/injection_pool_mix200k_20260728"
@@
-if [[ "$POOL_COUNT" -ne 500 ]]; then
-    echo "STOP: injection pool file count $POOL_COUNT != 500 (datasets.yaml canonical)" >&2
+if [[ "$POOL_COUNT" -ne 707 ]]; then
+    echo "STOP: injection pool file count $POOL_COUNT != 707 (PIN CORRECTION, chair 2026-09-04: injection_pool_mix200k_20260728 measured count)" >&2
```

Every other line (CRB md5 STOP, catalogue md5 STOP, injections-link mismatch STOP, H_GRID_41,
seeds, ancestor pin, CLI flags) byte-identical — confirmed by `git diff` showing only these 3
hunks.

Appended "PIN CORRECTION (chair, 2026-09-04)" notes with the measured values to
`exec/r-sealed-mock/D_SEALED_REGISTER_DOSSIER.md` §5 and to `exec/r-sealed-mock/PIN_RECORD.md`.

## Step 3: sync + out-root + submit

```
rsync -avz --backup --suffix=.bak_local20260904 cluster/graph1_sealed_m1_headstack.sbatch \
  bwunicluster:darksiren-emri/cluster/graph1_sealed_m1_headstack.sbatch
```
Verified byte-identical: local md5 `20ac6f617065f5d572036cc5017381db` = remote md5 (same).

```
ssh bwunicluster 'WS=$(ws_find emri); mkdir -p "$WS/graph1_sealed_m1_iiib_20260904"'
```
Out-root did not pre-exist; created fresh.

```
ssh bwunicluster 'cd ~/darksiren-emri && source cluster/modules.sh && WS=$(ws_find emri) && \
  sbatch --export=ALL,EXPECTED_CRB67_MD5=8e9253fef42f574c569a04a3e19299ab,VENUE=iiib,RUN_DIR=$WS/graph1_sealed_m1_iiib_20260904 \
  cluster/graph1_sealed_m1_headstack.sbatch'
```
**Job ID: 6790859**, array `0-40` (41 tasks = H_GRID_41). Confirmed via
`squeue -j 6790859 -r --noheader | wc -l` → 41, all `PD` immediately after submit.

## Step 4: polling

+~0 min (immediately after submit): all 41 tasks `PD` (pending), 0:00 elapsed.

**+5 min poll: NOT OBTAINED.** The background poll command (`sleep 300` inside the ssh session,
then `sacct` for 6790859/6790794/6790465) exited with code 255 — the ControlMaster session was
gone by the time it ran (`ssh -O check` afterward: "Control socket connect: No such file or
directory"). A fresh `ssh bwunicluster` attempt then hit interactive keyboard/publickey auth
(OTP-only per the device-transfer manifest), which this submitter is instructed never to attempt.
**No sacct data was obtained for 6790859, 6790794, or 6790465 at a +5min or later checkpoint in
this session.** The only confirmed post-submit state for 6790859 is the immediate all-PD/41-task
snapshot above. 6790794 (S0-C) last known state is from the prior session's own +5min poll
(`SUBMIT_RECORD_s0c_m1.md`): 7 RUNNING / 3 PENDING, all exit 0:0, at that time. 6790465 (GATE-ACC)
was not polled at all — the ssh session dropped before reaching that check.

## Summary

| item | result |
|---|---|
| Design gate | GREEN, confirmed by direct read of `DESIGN_GATE_computability_m1.md` |
| Pool measured | `injection_pool_mix200k_20260728`, 707 files (matches `injections` link) |
| CRB67 pin | md5 `8e9253fef42f574c569a04a3e19299ab` (confirmed), 1345 data rows |
| 1345 vs 1343 | disclosed, not resolved (see above); local diagnostic_report.md self-inconsistent |
| sbatch edit | 3 hunks, pool-expectation lines only, byte-identical elsewhere (diff above) |
| Dossier + PIN_RECORD | PIN CORRECTION notes appended, chair decision quoted |
| rsync to cluster | done, md5-verified identical |
| Job submitted | **6790859**, array 0-40 (41 tasks), all PD at submit time |
| +5min poll | **NOT OBTAINED — ControlMaster session dropped mid-wait; re-auth required and withheld per instruction** |
| 6790794 (S0-C) | last known: 7 RUNNING / 3 PENDING, exit 0:0 (from prior session's poll, not refreshed here) |
| 6790465 (GATE-ACC) | not polled — session dropped before this check |
