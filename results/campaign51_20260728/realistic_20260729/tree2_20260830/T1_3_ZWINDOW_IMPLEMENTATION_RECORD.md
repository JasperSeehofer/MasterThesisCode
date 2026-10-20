# T1.3-zwin implementation record (2026-08-30)

Row #255 standing grant, tree 2 node T1.3-zwin. Builder: sonnet-tier subagent, distinct from the
presenter of PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md and from the runner who will execute the P1
arm below. Branch fix/p32d-classg-venue-repair. No git operations performed by this node (the
orchestrator commits); no ssh; foreground only; no local runner was active, so editing
darksiren_emri/ and the driver was safe. Full account of what changed and why: the "Implementation
record" section appended to PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md itself, and the two
"implemented"/"verified" rows (plus one "implemented (addendum)" row for a gating fix found while
drafting the command below) appended to docs/gates/PHYSICS-GATE-LEDGER.md, all dated 2026-08-30.

## File list to commit

Modified (in the working tree; git status --short confirms exactly these plus the two new test
files, no others):

- darksiren_emri/galaxy_catalogue/handler.py
- darksiren_emri/bayesian_inference/bayesian_statistics.py
- darksiren_emri/arguments.py
- darksiren_emri/main.py
- darksiren_emri/validation/correspondence_1d.py
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
- docs/gates/PHYSICS-GATE-LEDGER.md
- results/campaign51_20260728/realistic_20260729/tree2_20260830/PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md
  (append-only "Implementation record" section)

New:

- darksiren_emri_test/test_theta_zwindow.py (22 tests: handler-level R1-R6 regression plan plus
  mass-filter inheritance)
- darksiren_emri_test/bayesian_inference/test_theta_zwindow.py (12 tests: evaluate()/CLI plumbing
  guards and defaults, plus the driver's PA-HIER-32(d) scorer arithmetic including the axis-gating
  fix below)
- results/campaign51_20260728/realistic_20260729/tree2_20260830/T1_3_ZWINDOW_IMPLEMENTATION_RECORD.md
  (this file)

No other file was touched. No production data file, no other reader's work-in-progress file
(BIAS_HISTORY_LEDGER.md, the T2_3_*/mass-aware gate doc, the concurrently-written T2_3_*
tree2_20260830 files) was written.

## A load-bearing finding while drafting the P1 command: the node list is 3 nodes, not 5

The gate doc's section 5.6 registers P1 as **theta_zwindow=on, z_window_k=4.0, sky_cone_k=1.5,
theta_phi_divisor=on, theta_sites=2.2, smear off, h=0.73, 4 seeds, nodes {truth, s_plus, s_minus}**
-- three node types, not five. This is confirmed twice more in the same document: section 6's cost
table reads "P1 (12 cells: 4 truth + 8 s-nodes at k=4, sky 1.5)" (4 + 4x2 = 12, consistent only with
three node types), and section 5.6 states explicitly "b-axis: NOT re-run under P1 (the flag at
default is byte-identical, so T1.2's b-certification stands for the form of record)". The
Implementation-prerequisites note appended to the gate doc (item 2) restates this a third time:
"Section 5.6's P1 registers 12 cells: 4 truth + 8 s-nodes (s_plus/s_minus x 4 seeds) -- not the
T1.2 recert's full 5-node theta-cross, and no b-nodes." An optional b_plus/b_minus re-run at k=4 is
registered separately as P1b (8 cells, not part of P1).

The command below therefore passes **--nodes truth,s_plus,s_minus**, not the driver's 5-node
default. Running this node list surfaced a real, load-bearing driver gap: before this node's work,
compute_scores/run_arm/score_only_payload all hard-required ALL FOUR of b_plus/b_minus/s_plus/
s_minus to be present before computing any score -- so the registered P1 arm's own node list would
have raised ValueError and never produced score_s at all. This was found and fixed as part of "THE
DRIVER SCORER" work item (see the ledger's "implemented (addendum)" row and the gate doc's
Implementation record for the exact diff): the three functions now gate PER AXIS (b, s
independently), so an axis simply never requested reports its score as unavailable
(n_pooled=0/NaN, score_b_available/score_s_available False) rather than crashing, while a
genuinely broken pair (one of the two axis nodes present, the other missing) still raises. Two
regression tests pin this exact shape (darksiren_emri_test/bayesian_inference/
test_theta_zwindow.py's test_compute_scores_handles_the_registered_p1_node_list_with_no_b_axis and
the two raise-path tests alongside it).

## The registered P1 arm command (for the orchestrator/runner; NOT executed by this node)

12 cells: 4 seeds (the default DEFAULT_BC_SEEDS = 900101, 900102, 900103, 900104) x 3 node types
{truth, s_plus, s_minus}. Run from the repository root (single logical command, shown here on
separate lines with a trailing backslash continuation for readability):

    uv run python3 results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
      --arm S0-A \
      --nodes truth,s_plus,s_minus \
      --theta-sites 2.2 \
      --smear off \
      --theta-phi-divisor on \
      --theta-zwindow on \
      --z-window-k 4.0 \
      --sky-cone-k 1.5 \
      --jobs 1 \
      --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_run

Notes on every flag:

- --arm S0-A: the CoR-M mirror-control arm (this is the P1 registration's own venue; S0-R is the
  disclosed-null instrument and is not part of P1).
- --nodes truth,s_plus,s_minus: the registered 3-node-type list (see above) -- omitting b_plus/
  b_minus keeps the arm at exactly 12 cells and skips the b-axis re-run P1 explicitly does not ask
  for.
- --seeds: omitted -- the default (DEFAULT_BC_SEEDS, the 4 registered seeds) is exactly the "4
  seeds" the registration names.
- --config: omitted -- the default "b0i" is the registered "'off' numerator (the bc driver)".
- --theta-sites 2.2, --smear off: the registered form of record (PA-HIER-32(a)/(b): "smear_global_
  selection=False with theta_sites='2.2' is the form of record for BOTH the CoR-P production arm
  and the CoR-M/S0-A mirror-control arm").
- --theta-phi-divisor on: T1.1's theta-consistent no-BH divisor, registered ON for P1 (section
  5.6's own flag list). sky_cone_k stays at its T1.1-registered default pairing.
- --theta-zwindow on --z-window-k 4.0: this node's own registered decisive configuration (section
  2.1/5.6: k=4 = integration_limit_sigma_multiplier, at which the selection interval equals W_g^theta
  and the capture term vanishes at every node).
- --sky-cone-k 1.5: the registered P1 pairing -- explicitly NOT E12's own sky_cone_k=3.0 pairing
  (that combination is the separate diagnostic fallback arm P3, registered only if P1 fails its
  band; the Implementation-prerequisites note's item 3 states this in full).
- --jobs 1: matches the T1.2 recert run's own concurrency (14 CPUs/job via --total-cpu-budget's
  default of 14, unset here so it stays at the default); the cost estimate (section 6, ~2.5 h wall
  at 14 cores) assumes this.
- --out-root: a fresh directory (hier_s0_zwin_run) distinct from T1.2's hier_s0_recert_run and from
  any other node's work directory, so this arm's per-seed work_root/es_null_det.csv caches and node
  directories (suffixed "_sites2.2_nosmear_divisor_zwin_zk4" per _node_dir_suffix) never collide
  with a prior run's outputs.

This single invocation runs all 12 cells sequentially (--jobs 1) and, per this node's scorer
implementation, ALSO computes and caches each seed's Es_null_det table
(<out-root>/s0a_seed<seed>/es_null_det.csv) as a side effect of building each seed's venue --
needed by the score-only command below with no extra evaluate() cost.

## The score-only command (after the P1 run above completes)

    uv run python3 results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
      --arm S0-A \
      --nodes truth,s_plus,s_minus \
      --theta-sites 2.2 \
      --smear off \
      --theta-phi-divisor on \
      --theta-zwindow on \
      --z-window-k 4.0 \
      --sky-cone-k 1.5 \
      --score-only \
      --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/hier_s0_zwin_run

Every flag matching the run command above (arm/nodes/theta-sites/smear/theta-phi-divisor/
theta-zwindow/z-window-k/sky-cone-k/out-root all identically) is REQUIRED --
gather_node_results_from_disk uses them to reconstruct the exact node-directory suffix
(_node_dir_suffix) the run command wrote to; a mismatch on any of them looks for CSVs that do not
exist and reports them as missing rather than silently reading the wrong (e.g. default-flag)
directory. This reads event_likelihoods.csv directly off disk (no evaluate() call, no venue
reconstruction) plus each seed's es_null_det.csv cache (also read-only, no venue reconstruction),
and writes s0a_score_output.json / s0a_score.md under --out-root, reporting score_b (from T1.2's
own certification, NOT re-derived here since b_plus/b_minus were not requested -- score_b_available
will read False), score_s_raw (the old/superseded linear secant, for continuity), score_lns (PA-
HIER-4's ln-s-centred secant, pre-Es_null_det-correction) and score_s (PA-HIER-32(d)'s corrected,
now-primary statistic -- this is the number Revision note 2's F1 falsifier reads, subject to that
same revision note's process constraint: report both the raw and c-weighted conventions side by
side and do not declare F1 CONFIRMED/REFUTED on this driver's score_s alone; see the gate doc
section 5.6/9 and Revision note 2 item 3).

## What this node did NOT do

No evaluate() call was made against the real GLADE catalogue by this node (the P1/P1b/P2/P3 arms
above are commands FOR the runner, not executed here). No cross-check of this implementation's
Es_null_det closed form against the archived b1_1_forensic_work/f4_mechanism.py/f4_out.json figures
on the real production catalogue was performed (both implement the same math independently; this
node's own unit test exercises the closed form on a synthetic single-host fixture only, per
darksiren_emri_test/bayesian_inference/test_theta_zwindow.py). No git operations. No file owned by
another concurrent reader (BIAS_HISTORY_LEDGER.md, TREE2_CHARTER_20260830.md, the T2_3_*/mass-aware
gate doc, PREREGISTRATION_HIER_HTHETA_20260826.md) was written.
