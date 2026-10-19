# Tree 2 (2026-08-30) -- index

Launched under row #255 -- tree 2 node README. All files below are stamped with their own
"launched under row #255" node tag; this index only points to them, it adds no new claims.
Ledger of record: results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md
rows #256-#263. Gate ledger of record: docs/gates/PHYSICS-GATE-LEDGER.md, 2026-08-30 rows.

## Charter

- TREE2_CHARTER_20260830.md -- opens tree 2 under row #255; docket 2 section 7 ranking as the
  tree (T1-T6 plus zero-compute housekeeping). Ledger row #256.

## T1 -- theta-consistent no-BH divisor + S0-A re-certification

- PHYSICS_CHANGE_THETA_DIVISOR_20260830.md -- T1.1 gate presentation (panel-clean after 0
  rounds); registered form, cost, T1.2 prediction. Ledger row #259.
- T1_1_DIVISOR_IMPLEMENTATION_RECORD.md -- T1.1 builder record: exact file list, quality-gate
  results, the decisive driver-gap finding (hier_s0_driver.py needs a new CLI flag before T1.2
  can engage the fix), and the corrected T1.2 command. Ledger row #260.
- T1_1_DIVISOR_VERIFIER_REPORT.md -- T1.1 independent verifier report (verdict table items 1-5
  all PASS; must_fix none). Ledger row #260.
- t1_1_gate_work/ -- gate-presentation scratch work.
- t1_1_verifier_work/ -- verifier scratch work, including t1_1_verifier_work/smoke_run/ (the
  live GLADE+ smoke cell, S0-A seed 900101 node truth, event-cap 12).

T1.2 (S0-A re-certification against the fixed divisor and against PA-HIER-32's score_s) is not
yet built; it is the next T1 node and requires the hier_s0_driver.py --theta_phi_divisor flag
identified above.

## T2 -- B4.3 mixture-weight derivation + instrumentation hook

- B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md -- T2.1 derivation (zero compute): candidate (b)
  the mixture-weight h-slope REFUTED as mechanism; candidate (c) depth skew of impostors inside
  the ball IS the mechanism, closed form derived; points at a mass-aware 1D catalogue-leg
  remedy returned to the author as a fresh [RULE] (not covered by row #255). Ledger row #261.
- T2_2_CANDIDATE_HOOK_RECORD.md -- T2.2 builder record: the opt-in candidate_dump_dir
  instrumentation hook on BayesianStatistics.evaluate() (A10, instrumentation-guard route, not
  a full physics-change gate); byte-identity and schema gates; exact file list and the 3.4
  CPU-h instrumented-run command. Ledger row #262.

## Registration amendments

- PA-HIER-32 is filed append-only in
  results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md (not
  in this directory) -- registers the debiased score_s statistic and drops the z-binned theta
  read. Cross-referenced at ledger row #263.

## A11 -- completed-weight fork derivation

- A11_COMPLETED_WEIGHT_FORK_DERIVATION_20260830.md -- zero-compute derivation: neither
  COMPLETED-SMALL nor COMPLETED-MATERIAL is the estimator under the one-density-everywhere
  consistency criterion; both are the same un-derived candidate times an un-derived global
  prior; the T2.3 mass-aware 1D leg moots the fork by identity. Panel clean (0 mustfix, 1
  round; a later documentation-only citation-line correction was applied append-only).
  REPORTED-ONLY, returned to the author. Ledger row #269.

## T5 -- mass-law-keyed window design / k-scan

- PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md -- T5.1 design proposal (zero compute): the
  production mass law is a delta law on iiib and log-normal (realized-forward) on joint_r1;
  a log-symmetric window at k = Phi^-1(1-epsilon/2) is exact-by-construction on the scattered
  venue where the production linear window is not; registers a two-arm k-scan (~26-35 CPU-h,
  cluster-bound). Panel REFUTED at round 2 with two must_fix items outstanding (a Section
  1.2/2 width-drift claim to correct against the exact-width remedy already in the code, and
  an unruled [RULE] on an out-of-scope pointer-note append). Ledger row #270.

## Not yet started (charter T3-T6, out of scope for this node)

- T3 -- B8.2 harness S1-S5
- T4 -- B7 falsifier (ii) + the tree-1 wave-3 readout (A4 pending, input does not exist yet)
- T6 -- CMEM >=90%-power registration (A8 = parked; available)

## Synthesis docket (information only)

- TREE2_SYNTHESIS_DOCKET_20260830.md -- chair synthesis over tree 2 to date: verdict table
  (section 1), the [HIER] instrument story (section 2 -- three sequential instrument defects
  found and fixed; S0-A null-consistent on both axes only under the still-unratified PA-HIER-33),
  the 1D-rail/B4.3 story (section 3 -- mass-blind-numerator/mass-aware-divisor mechanism,
  mass-aware remedy arm +0.1158 +/- 0.0136 above band, censored, production flip a fresh [RULE]),
  and the morning docket's author items (section 4 -- 4 primary [RULE] items plus 3 secondary
  [RULE] items bundled under (v), 7 [RULE] asks total, none actioned). Runner-9 (B8.2 S3, stage
  LADDER at filing) was running throughout and its work root was not touched. Information only --
  no approval, default change, STOP lift, or arm launch. Ledger row #276.

## Full verification (author-ordered, 2026-08-31)

- FULL_VERIFICATION_TREE1_20260831.md -- tree-1 items 1-19 re-adjudicated via OPUS verifiers in
  parallel (author's explicit model instruction, row #278(6)): 19 confirmed, 0 refuted, 0
  undetermined; one verdict changed vs the earlier sonnet pass (item 19, compute ledger:
  undetermined -> confirmed by independent reconstruction). Item 20 (wave-3 blind HEAD readout)
  still DEFERRED pending its own readout. Ledger row #280.
- FULL_VERIFICATION_TREE2_DECISIONS_20260831.md -- tree-2 items T2-1..T2-17 adjudicated (15
  confirmed, 2 REFUTED-DETAIL with headline standing, 0 headline verdicts refuted) plus a
  decisions audit of 8 orchestrator-derived itemization/path-choice lines (5 faithful, 3
  deviation, row #277 faithful with arithmetic slips). 2 items left undetermined (the A18 band
  pure input; the changed-verdict/in-flight-exclusions bookkeeping). Standing grant of rows
  #255/#268/#278 continues per the author's approval. Ledger row #280.

## Open items carried forward

- T1.2 needs the hier_s0_driver.py --theta_phi_divisor CLI flag (regression item R13) before it
  can run as intended; running the originally proposed command as-is would silently NOT engage
  the fix.
- T2.1's mass-aware 1D catalogue-leg remedy (Sigma^4D replacing Sigma_phi as the 1D global
  divisor) is a fresh [RULE] for the author, not pre-authorized by row #255 -- only its gate
  presentation and the T2.2 instrument cell proceed under the standing grant.
- Regression items R3, R5, R11 (byte-for-bit pins against the banked S0-A CSVs; the
  correspondence_1d harness-parity check) remain unattempted, correctly deferred to T1.2 itself.
