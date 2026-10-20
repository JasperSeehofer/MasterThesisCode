# Tree 2 Charter (mirrors runbook 37 section 2 format) — filed 2026-08-30

Launched under row #255 — tree 2 node 0 (this charter record).

## 1. Root goal of record and the grant

**Root goal (row #221, author verbatim, unchanged since tree 1):**

> "a scientifically correct mathematical setup for the bayesian inference in 1d and 2d which
> should be unbiased up to the level where we have to admit that the information has starved"

**Grant (row #255, author verbatim):** "all ratified from the docket" — against the Fan-out 1
Verifier Docket (artifact eeb5c7c3; report
fanout1_20260829/END_VERIFIER_REPORT_PART1_20260830.md section 4; items A1-A17, path decisions
P1-P10). The A17 line of that row is the new standing grant that opens tree 2:

> "A17 = NEW [STANDING] GRANT for tree 2, orchestrator-derived scope = that of rows #222/#223
> (instruments, counterfactual arms, registrations, path choices, production-default flips
> inside the tree, each with its gate presentation before code and its ledger rows; docket 2
> section 7 ranking as the tree; one synthesis docket per wave for information; lapses at the
> registered end-of-tree-2 verifier pass or on any author message narrowing it)."

**Orchestrator-derived scope (mirrors row #223's tree-1 wording, applied to tree 2):**
production changes inside this tree are covered by the same grant as instruments, arms,
registrations, and path choices. Physics gates still run in full: presentation before code,
ledger rows as always, the approval step cites row #255, and every gate goes to the end-tree-2
verifier. Cycle of record stays: tree -> verify -> plan next tree -> repeat (row #223 wording,
carried forward). Two items are explicitly NOT covered by row #255 and return to the author as
fresh [RULE]s when their inputs exist: A4 (mz_sel/eff ratification, after the wave-3 blind
readout) and A11 (row #167's D-tilde-phi factual fork). Both are tree-1 leftovers, not tree-2
nodes, and are tracked in T4 and the housekeeping bundle respectively below.

## 2. The tree (docket 2 section 7 ranking as the branches; depth-1 nodes are covered by row #255)

Cost anchors (corrected, this session): iiib approx 5-7 min per h-point; mirror unsmeared cell
65 s; theta-engaged smeared cell 1191 s single-core; S0-C marginal 24.4 s per h-node. Cluster is
DOWN (bwUniCluster Lustre /pfs/data6 OST 5 inactive) — every cluster figure below is a queued
estimate, not a schedule.

| branch | 1 (launch under grant) | 2 (fresh [RULE]) | 3 (fresh [RULE]) |
|---|---|---|---|
| **T1 [theta-fix]** (docket 2 sec.7 rank 1) | A3 gate presentation + implementation: theta-consistent no-BH divisor Sigma^phi(theta), site 2.3 extended to the phi-table branch (byte-identical at theta=(0,1)), plus the sky-cone-radius flag (bayesian_statistics.py:4869, hardcoded 1.5) — physics gate, local; divisor pass approx 1-3 min/node (UNMEASURED), cone flag approx 35-60 min wall (b-axis) + approx 1.5 h wall (s-axis), UNMEASURED point estimates (row #251 forensic sec.5) | S0-A re-certification: re-run the 20 S0-A cells at h=0.73 against the registered predictions (Z_b -> -0.62 +/- 0.43, Z_s -> -0.07 +/- 0.012 without the cone flag; -> -0.5 +/- 1 with it); rule = does the re-run land inside the registered bands (fresh RULE on the read); approx 11.5 CPU-h local (approx 6 if venue builds are cached) | A6 word (already ruled launch-after-fix, row #255): S0-B (C1) on iiib launches only after T1.1 and a passing T1.2; approx 7-27 CPU-h cluster; rule at this depth is the S0-A-pass gate itself, not a fresh ask |
| **T2 [B4.3]** (rank 2) | h-slope derivation s_beta = -3.2891/h (0 CPU-h, local, mechanical) + the per-candidate p_Di instrumentation hook; A10 already rules this an instrumentation-guard change (byte-identity check + gate-ledger row), not a full /physics-change gate | per-candidate instrumented run (serialises candidate z/mass/weight/is_true_host); approx 3.4 CPU-h local | enlarged-ball counterfactual (sky 3 sigma, z +/-4 sigma_g; median candidates 278 -> 1729); approx 3-6x a normal cell; shares the ball-radius flag edit with T1's s-axis item — rule: name the mechanism of the necessary cause of the 1D rail (forensic E9/E14/E17 already show the sign flip under ball enlargement on b0i) |
| **T3 [B8.2 harness]** (rank 3) | S1-S3 build: two-channel calibration harness, N-scaling measured first (S3); local | S4 registration (top-tier review required before S5, per docket 2 sec.7) | S5 execution + the acceptance/count audit -> stop/continue verdict against B8's F-measurement numbers; total across S1-S5 approx 130-475 CPU-h local, 13-46 h wall |
| **T4 [B7 falsifier (ii) + tree-1 wave-3 readout]** (rank 4) | tree-1 leftover, queued cluster: the wave-3 blind HEAD readout (C0-prime off-gate + the two 41-task blind arrays) — built, DRY_RUN, not submitted (row #252 P9); runs before any T4 node proper, per F2 ordering | class-G fleet Option A-prime, rung 1 repaired (24-33 tasks); approx 40-60 CPU-h cluster (chair recost from 208-286) | the wave-3 "off" arm (82 tasks, approx 160-290 CPU-h cluster), then A4 returns as a fresh [RULE] with numbers in hand: ratify catalogue_numerator_survival_2d="mz_sel"/center="eff" as production default, or revert to "off" pending falsifier (ii) — NOT covered by row #255 (approval-scope rule, input did not exist at grant time) |
| **T5 [mass-law-keyed window / k-scan]** (rank 5; A1 = (c), row #255) | k-scan {2, 2.5, 3.5} on iiib (H4 each); local; approx 5 CPU-h per k-node set (C3 measured 4.97) | joint_r1 at k=3; approx 11-15 CPU-h; venue per docket 2 sec.7 (cluster if the wave-2 batch runs it alongside S0-B) | adoption gate ruling, bounded by the INTERMEDIATE read (+0.0035): decides whether any window geometry is MATERIAL for production; design object is the impostor pool, not the true host (fresh RULE) |
| **T6 [CMEM >=90%-power registration]** (rank 6; A8 = BANK-AND-PARK, row #255 — available, not launched by default) | registration only if prioritized: approx 30 new mirror seeds x 2 arms, approx 15 CPU-h local (chair estimate) | read/decision node; structural class, REPORTED-ONLY cap carried verbatim (PA-HIER-28 item 9); low H0 yield | bank per the A8 disposition already ruled; no further action unless the author re-opens it |
| **Housekeeping** (rank 7, zero-compute; A14 approved as a batch, row #255) | file the B7.3 adoption row (already done, row #253) + A14's remaining items: replace the row-#<adoption> placeholder at bayesian_statistics.py:3274 with row #253; append the log-text reconciliation to the B7.3 presentation sec.13.1; append citation fixes F1/F2/F6/F8; the one-line GitHub-rejection note + the 8.6 CPU-h unbanked line in the ledger; re-word the A22 stamps; run the C0 sec.11.2 OAT-column identity check; the two P1-prime nodes (0.33 CPU-h each); the "two M1s" docstring cross-reference; the driver's duplicate-row assertion; A13's git-force-add of the 41-file registered-run slice + keep the local archive | retrieve C4's provenance extras and run archive_run_wave2.sh once SSH returns (A13, blocked on cluster) | — |

**Local-vs-cluster split:** T1.1, T1.2, T2 (all depths), T3 (S1-S4 build/design), T5.1, and
housekeeping run local. T1.3 (S0-B/C1), T3.5 execution if it needs cluster scale, T4 (all
depths), and T5.2's joint_r1 arm queue behind cluster recovery (OST 5 inactive — no ssh, no
cluster work this session). T6 stays parked/available, local if launched.

## 3. Wave plan

- **Wave 1 (local, now):** T1 gate presentation + implementation (A3) and S0-A re-certification
  (T1.1-T1.2); T2 derivation + instrumentation hook + the instrumented run (T2.1-T2.2);
  housekeeping bundle (A14 batch, plus A13's git-force-add half). One synthesis docket at the end
  of wave 1.
- **Wave 2 (cluster, once OST 5 recovers):** the tree-1 wave-3 readout first (C0-prime off-gate +
  the two 41-task blind arrays, already built and DRY_RUN per row #252 P9) — this is leftover
  tree-1 work that must land before A4 can be asked; then S0-B (T1.3) and any T5 cluster arm
  (T5.2's joint_r1). Second synthesis docket.
- **Wave 3 (adoption / verify):** A4 ratification once the wave-3 readout numbers exist; T3's S4
  registration review and S5 execution with its stop/continue verdict; T5.3's adoption-gate
  ruling; T6 if re-opened. Closes with the registered end-of-tree-2 verifier pass (see pointer
  below), which is the event that lapses the row #255 standing grant.

## 4. Verifier-pass registration pointer

To be authored at tree-2 close, mirroring
results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_END_VERIFIER_PASS_20260829.md:

results/campaign51_20260728/realistic_20260729/tree2_20260830/REGISTRATION_END_VERIFIER_PASS_TREE2.md
(not yet written — author this ahead of the wave-3 data, per amendment F5, once wave 2 nodes are
scheduled).

## 5. Standing rules carried (do not re-learn)

Verifier output is evidence, not authority. Subagents never run the registered measurement they
built. Never end a turn to wait on an untracked process the harness does not track — every wait
is a blocking foreground command, and this session's commands are capped at 600 s each. Exoneration
grep is for the mechanism, not the tag. Builder != runner for any registered measurement.
REPORTED-ONLY caps carry verbatim (PA-HIER-28 item 9, T6 above). No git commit/add this session —
the orchestrator commits. Append-only records; every number carries {value, source file:line,
date}. Do not edit physics-trigger files unless the task at hand explicitly gates it. Local only —
no ssh, no cluster work until OST 5 recovers.
