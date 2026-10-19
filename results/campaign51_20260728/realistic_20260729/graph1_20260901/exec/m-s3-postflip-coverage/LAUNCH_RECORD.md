# m-s3-postflip-coverage — LOCAL LAUNCH RECORD (invocation 1 of <=3 per cell)

Authorization: row #301 item 2 (d-s4-review RATIFIED, bands + stop rule frozen) + design gate GREEN
(exec/r-b82-s4/DESIGN_GATE_RECORD.md, all 6 checks; launch block transcribed zero-fresh-choices).
Launched by the chair directly (orchestrator-as-runner pattern for long local compute), 2026-09-02 10:55 CEST.

- Harness: tree2_20260830/b8_cal_harness.py @ the row #291 repaired state (commit 97b2062a).
- Invocation cwd: REPO ROOT (required — the galaxy-catalogue handler resolves
  ./darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv cwd-relative; same gotcha family
  as the row #288 archive-script CWD incident).
- Work root (absolute): results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip/
- Cell S: --N 200 --cell S --seed-block 901000 --n-universes 100 --max-wall-s 86400 ; PID 2428302 ; log cellS_inv1.log
- Cell T: --N 200 --cell T --seed-block 902000 --n-universes 25 --max-wall-s 86400 ; PID 2428303 ; log cellT_inv1.log
- Falsifier block 901100+ untouched. Aggregation at completion: --score-only --population 200 per cell + --score-only-ratio-t-s.
- Stop rule of record: frozen registration §3 (n_U_min 60/16; <=3 invocations x 86400 s;
  WALL-LIMITED-VALID / INCOMPLETE-RUN / INSTRUMENT-DEFECT).
- FAILED ATTEMPTS disclosed: attempt 1 (wrong shell cwd — nothing started); attempt 2 (an `&`
  backgrounded the whole &&-chain — cell S started from the harness dir and crashed on the
  cwd-relative catalogue path before any universe was drawn; no work-root state was created by it
  beyond empty cache dirs). Attempt 3 (this record) is the launch of record.
- Watcher: chair-armed harness Monitor (liveness from pids_inv1.txt + checkpoint growth + log errors).
