# ERRATUM to the (m2) sealed-mock registration — R27 (precedence, σ treatment) and R28 (pool identity)

Date: 2026-10-19. Drafted by mission M162 (T-1 governance package), executor top tier.
Author of record for every scientific decision: Jasper Seehofer.
**Status: DRAFT FOR RATIFICATION. Nothing here is frozen, drawn or launched.** Once the author
ratifies it, it is append-only. A change after ratification is a new dated erratum, never an
edit, and the rule-block hashes in `.planning/research/THREAD.md` § "T-1 package hashes" stop
matching the moment this text is edited.

Amends: `REGISTRATION_DRAFT.md` (2026-09-03) §3, §4, §6, §8, §9 and §10 for stage (m2) only, and
`D_SEALED_REGISTER_DOSSIER.md` §2 where it is cited for (m2). Stage (m1) is closed and is not
re-opened here.

Conventions: [DOC] means read from a committed file. [RULING] means an author ruling, quoted.
ORCHESTRATOR-DERIVED means arithmetic done here, with its derivation. Every block between
`RULE-BLOCK` markers is cell-bearing or gate-bearing and is hashed.

## E-0. Authority: the rulings this erratum implements

| ruling | date, card | text as recorded | what it binds here |
|---|---|---|---|
| R25 | 2026-10-19, `ask-20261018-w224256df6b` | m1 is banked **INTERMEDIATE** by the parent wording (`REGISTRATION_DRAFT.md` §6: "none or several of the above") | the "several cells fire ⇒ INTERMEDIATE" rule of E-2 step 3 (same text, same precedent) |
| R27 | 2026-10-19, `ask-20261018-w224256df6b` | draft the m2 erratum with the **rail clause winning a both-fire tie**, plus a treatment for a result tighter than the anchor, for ratification **before any sealed draw** | E-2 step 2 (rail wins), E-3 (σ treatment); "no (m2) launch until ratified" (docket R27, `MORNING_DOCKET_20260904.md:106`) |
| pool | 2026-10-19, `ask-20261018-w224256df6b` | **P2 first** (reuse the 0.73 pool plus a small unsealed pool at a public h with the 1e-3 invariance gate, ≈ 10 GPU-h); **P1** (fresh sealed pool, up to 250 GPU-h) as the fallback | E-5, and `POOL_OPTION_MEMO_m2.md` |
| R28 | 2026-10-19, `ask-20261018-w231446a3b9` | **ratified** as a registration erratum: the 707-file pool, contents identical to the 0.73 baseline pool | E-1 |
| R24 | 2026-10-19, `ask-20261018-w231446a3b9` | **no code change before T-1**; the freeze pins the `darksiren_emri/` tree; R24 disclosed as a limitation, taken up after the blind result | E-4 |
| T-1 date | 2026-10-19, `ask-20261018-w223027afac` R2 | T-1 deadline **2026-12-15** (second extension) | E-7 dated default |

## E-1. R28: the pool of record (replaces draft §8 G-1 "the 0.73 pool path + file count 500")

The registration named `$WS/injection_pool_depth15_50k` (500 files) as the p_det pool. That was a
factual error (`PIN_RECORD.md:92`). The 0.67 run and the 0.73 re-baseline anchor run both link the
pool below, verified content-identical by per-file manifest (`D_SEALED_REGISTER_DOSSIER.md` §GUARD
CORRECTION; `READ_RECORD_m1.md:28`; ledger row #387).

<!-- RULE-BLOCK:E1-pool-identity BEGIN -->
- **Pool of record for (m2), p_det:** `$WS/injection_pool_mix200k_20260728`, where
  `$WS = /pfs/work9/workspace/scratch/st_ac147838-emri2` (`cluster/datasets.yaml:24`; moved from
  `emri` on 2026-09-23, same layout).
- **File count:** 707 files `injection_h_0p73_task_*.csv` (200,100 rows).
- **Identity pin (STOP-gated):** the content-manifest list-md5
  `75f4030d5d3b0405fd948049bef5767e`, defined as
  `ls -1 *.csv | sort | xargs md5sum | md5sum` run inside the pool directory. This equals
  `md5sum exec/r-timeout-selection/POOL_MANIFEST.md5` (707 lines). It is the value passed as
  `EXPECTED_POOL_LIST_MD5` by `cluster/graph1_sealed_m1_headstack.sbatch:127-134`.
- **Secondary label, not a pin:** `a1dffdf561c51c8c778dce115c5fb371` is the md5 of the sorted
  filename list only (`PIN_RECORD.md:109`). It is the same 707 files measured another way. It
  must never be compared against the content-manifest value.
- G-1 for (m2) therefore reads: catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`; pool file
  count 707 **and** content-manifest list-md5 `75f4030d5d3b0405fd948049bef5767e`; the sealed
  CRB md5 recorded at merge (never printed next to h); under P2, the small pool's own manifest
  list-md5 recorded at generation.
- The same correction applies wherever the (m2) text says "the canonical 0.73 pool" (draft §3
  P2 row, §7 (m2) row, dossier §2 "DATA THE JOB READS"): it means this pool.
<!-- RULE-BLOCK:E1-pool-identity END -->

Pins re-read, requested, not performed: none of these pins has been re-read since the
`emri → emri2` move. A read-only re-read (POOL_OPTION_MEMO_m2.md §5) is a launch precondition of
T-1 and is requested of an M38 window or of the author. This mission ran no cluster command.

## E-2. R27(i): precedence of the (m2) cells (replaces draft §6's table as an assignment rule)

The draft §6 table stays the definition of each cell. This block adds the order in which the cells
are tested. Without an order, a posterior on which two clauses hold has no registered cell, which
is what happened on m1 (M65 `VERDICT-m1.md`, "Why both clauses fire").

<!-- RULE-BLOCK:E2-precedence BEGIN -->
**Inputs, per channel c ∈ {2D `combined_with_bh` (primary), 1D `combined_no_bh` (replicate)}:**
mean_h,c, MAP_c and σ_h,c from the frozen T0 gradient-weighted scorer on the (m2) 55-node grid
(E-6), together with the edge-node masses m_lo,c (node 0.600) and m_hi,c (node 1.000). Anchors are
channel-specific (CONVENTIONS C-21):

| anchor | 2D | 1D |
|---|---|---|
| TUNED centre (mean / MAP) | 0.6659 / 0.665 | 0.6670 / 0.665 |
| TRANSFER offset δ_c (mean − 0.73 at the 0.73 re-baseline) | −0.0641 | −0.0630 |

Cell definitions, from draft §6 with the anchors made channel-specific:
- **TUNED_c:** |mean_h,c − TUNED-mean_c| ≤ 3σ_band,c AND |MAP_c − 0.665| ≤ 3σ_band,c. The
  draft's precondition |h_inj − 0.73| ≥ 0.06 holds for every draw from P-A.
- **TRANSFER_c:** |(mean_h,c − h_inj) − δ_c| ≤ 3σ_band,c AND |(MAP_c − h_inj) − δ_c| ≤ 3σ_band,c.
- **UNBIASED_c:** |mean_h,c − h_inj| ≤ 2σ_band,c.
- **Rail_c (the rail clause):** MAP_c on an edge node (0.600 or 1.000), OR m_lo,c > 1e-3, OR
  m_hi,c > 1e-3. Side: LOW if the 0.600 condition holds, HIGH if the 1.000 condition holds,
  BOTH if both hold.

σ_band,c = σ_h,c (the measured value) unless E-3 says otherwise.

**Assignment, applied in this order. The first step that assigns a cell ends the procedure.**

1. **Gates.** Any red gate in E-6 ⇒ **NO-READ**. Nothing is banked and the seal is spent
   (revision cap 1).
2. **Rail clause wins (R27).** If Rail_2D fires, the read is **AT-RAIL-<side of 2D>**, banked and
   read as a one-sided **bound**, not NO-READ. If side = BOTH, the read is INTERMEDIATE instead.
   TUNED, TRANSFER and UNBIASED are **not assigned**, whatever their bands say. The band
   distances are still computed and reported, labelled "reported, not cell-bearing". The
   replicate must agree: if Rail_1D does not fire, or fires on the other side, the read is
   **INTERMEDIATE (channel disagreement)**. If Rail_1D fires and Rail_2D does not, the read is
   also INTERMEDIATE (channel disagreement).
3. **Band cells.** Evaluate TUNED_c, TRANSFER_c and UNBIASED_c on both channels.
   - Exactly one band cell holds in 2D, and the same single cell holds in 1D ⇒ that cell.
   - Two or more band cells hold in either channel ⇒ **INTERMEDIATE (several)**. This is the
     parent wording "none or several of the above", the same text R25 applied to m1.
   - No band cell holds in 2D ⇒ **INTERMEDIATE (none)**.
   - 2D and 1D hold different single cells ⇒ **INTERMEDIATE (channel disagreement)**.
4. **σ robustness (E-3).** E-3 may turn a cell from steps 2–3 into INTERMEDIATE (σ-sensitive).
   It never turns one band cell into another.

**Bound semantics.**
- AT-RAIL-LOW banks the statement "mean_h,2D ≤ X at the 0.600 floor", with X the measured mean.
  The posterior is then censored by the grid floor and lies below both the TRANSFER window and
  the UNBIASED window for every draw from P-A: under P-A the TRANSFER mean is 0.726–0.756 and
  the UNBIASED mean is 0.79–0.82.
- AT-RAIL-HIGH banks "mean_h,2D ≥ X at the 1.000 ceiling".
- Each side gets its own row in the disposition table (E-8).
<!-- RULE-BLOCK:E2-precedence END -->

**Why the rail is banked as a bound and is not NO-READ.** This settles open defect D-2
(CONVENTIONS.md). The draft makes the rail NO-READ (§8 g-censoring); the operative m1 dossier
made it a banked BOUND (§2). R27's "rail clause wins" is a statement about which cell is
assigned, so a winning rail clause has to book a cell. That cell is the dossier's BOUND reading,
which is the m1 precedent the ruling was made on. NO-READ stays reserved for gate failures
(pins, freeze, commitment, blindness, population, resolved flags, the P2 invariance gate). This
reading is put to the author as its own line on the ratification card (Q1b).

**Disclosure on the rail threshold (CONVENTIONS C-22).** The 1e-3 edge-node threshold is kept
exactly as registered. It is tight: on m1 it fired at a declining tail with an interior MAP. The
55-node grid moves the upper edge to 1.000, which is ≥ 9σ above any P-A truth, so HIGH is
practically reachable only through a defect. The 0.600 floor sits 0.066 below the TUNED centre.
Under TUNED with σ ≈ 0.014–0.020 that is 3.3–4.7σ, so a TUNED-shaped posterior can still trip
the LOW rail through its tail, as m1 did. That outcome is booked AT-RAIL-LOW by this rule, not
TUNED. **This is the main cost of "rail wins", and the author should see it before ratifying:**
a TUNED-like posterior with a fat lower tail reads AT-RAIL-LOW, a bound that excludes TRANSFER
and UNBIASED. It cannot read TUNED. The consequence row for AT-RAIL-LOW (E-8) therefore carries
the TUNED-band distance as a reported number, and returns to the author as a fresh RULE before
any paper consequence attaches.

**Where two band cells can both hold under P-A** (ORCHESTRATOR-DERIVED; disclosed so that
INTERMEDIATE (several) is not a surprise):
- TUNED and TRANSFER windows are separated by h_inj − 0.73 = 0.06–0.09. Each window has
  half-width 3σ, so they overlap when σ > (h_inj − 0.73)/6, i.e. σ > 0.010 (h_inj = 0.79) up to
  σ > 0.015 (h_inj = 0.82). At the m1 width σ ≈ 0.014, the band cells cannot tell TUNED from
  TRANSFER for the lower part of the prior.
- TRANSFER and UNBIASED windows are separated by |δ| ≈ 0.064 and their half-widths sum to 5σ, so
  they overlap when σ > 0.0128.
- TUNED and UNBIASED are separated by ≥ 0.124 and never overlap at σ ≤ 0.024.
- Reading of the overlaps: if the posterior falls in an overlap, the registration answers
  INTERMEDIATE (several) rather than choosing. This is the R25 precedent applied forward.

## E-3. R27(ii): treatment of σ_measured < σ_anchor (and the symmetric case)

m1 came in with σ_h = 0.0140 (2D) against the anchor's 0.0185 at larger N, 24 % narrower. The
registration expected about 9 % wider (CONVENTIONS D-3). A width discrepancy moves every 3σ band,
so it changes which cell fires. E-3 keeps the registered statistic as the primary rule and adds a
mechanical robustness check.

<!-- RULE-BLOCK:E3-sigma-treatment BEGIN -->
**Definitions, per channel c:**
- N_used = the number of distinct `event_idx` in the scored likelihood table.
- σ_anchor,2D = 0.018475 and σ_anchor,1D = 0.017526, both at N_anchor = 1588 (row #302).
- σ_exp,c = σ_anchor,c · sqrt(N_anchor / N_used). This assumes Fisher information adds over
  events, so σ ∝ N^(−1/2). It is the same scaling the draft's §5 SNR-loss estimate used.
- ρ_c = σ_h,c / σ_exp,c.

**Rule:**
1. **Primary.** σ_band,c = σ_h,c, the measured width, exactly as registered. Assign the cell by
   E-2.
2. **Width check.** If 0.8 ≤ ρ_c ≤ 1.25 in both channels, the primary cell banks as is.
3. **Width-discrepant** (ρ_c < 0.8 or ρ_c > 1.25 in either channel). Re-run E-2 steps 2–3 with
   σ_band,c = σ_exp,c in both channels. Then:
   - Same cell as the primary ⇒ the primary cell banks, annotated "σ-discrepant, cell-robust
     (ρ_2D = …, ρ_1D = …)".
   - Different cell ⇒ the read is **INTERMEDIATE (σ-sensitive)**. Both assignments and both
     ρ values are reported, and the read returns to the author as a fresh RULE.
   - The rail clause does not depend on σ, so an AT-RAIL read is never changed by this step.
4. **Reported, not cell-bearing.** Both ρ values; all band distances under σ_h and under σ_exp;
   the T-4 per-event pull calibration (KS, mean, std of the pulls; draft §6 secondary) on events
   with curvature > 0. A pull std well below 1 is the expected companion of ρ < 0.8 and is
   reported as such.
<!-- RULE-BLOCK:E3-sigma-treatment END -->

**Where 0.8 and 1.25 come from** (ORCHESTRATOR-DERIVED; flagged for ratification):
- The band is symmetric in log width. 1/1.25 = 0.8.
- It is wider than the change between N = 1588 and the draft's expected P-A N = 1230–1340: that
  scaling is a factor 1.09–1.14 and is already absorbed into σ_exp.
- It is narrower than m1's observed ρ = 0.0140/(0.018475·sqrt(1588/1343)) = 0.0140/0.02009 = 0.70.
  So the m1 case would have triggered the check.
- Applied to m1 for illustration only (m1 stays banked INTERMEDIATE by R25): at σ_exp the 2D
  TUNED distance is 0.0382/0.0201 = 1.90σ, so the band cell is unchanged. The rail would still
  fire. Under this erratum, m1 would have read AT-RAIL-LOW with no σ-sensitivity.
- The check is symmetric because R27 named the narrow case, but the same logic applies to an
  unexplained wide posterior. A one-sided check would itself be a forking path.

## E-4. R24: the code freeze pins the `darksiren_emri/` tree (amends draft §4 item 1 and §8 G-2)

<!-- RULE-BLOCK:E4-code-freeze BEGIN -->
- **Physics freeze:** the `darksiren_emri/` git tree hash is
  `71acb98b0fd8a0f9d3aad9595eb9fc084cae0b6d`. That hash is identical at the m1 registration
  freeze `081b1f28`, at both m1 run commits (8f933e7b, d9e50179) and at the work-branch tip
  387cf33f. The last commit touching `darksiren_emri/` is 2b657255 (2026-09-02).
- **H_code** is the darksiren-emri commit at launch. It is written into the launch ledger row
  before the draw. It must satisfy `git rev-parse H_code:darksiren_emri` == the tree above.
  `draw_sealed_h.py` refuses to draw otherwise.
- **G-2 for (m2), per array task, STOP on failure:**
  - (a) `git rev-parse HEAD` == H_code, by strict equality. The ancestor form used for m1 is not
    acceptable here.
  - (b) `git rev-parse HEAD:darksiren_emri` == `71acb98b0fd8a0f9d3aad9595eb9fc084cae0b6d`.
  - (c) `git status --porcelain -- darksiren_emri | wc -l` == 0, the dirty count. m1's cluster
    tree carried 665/702 porcelain lines, which a commit-level pin cannot see
    (`READ_RECORD_m1.md:29`). Check (c) closes that hole for `darksiren_emri/`.
  - (d) `GIT_COMMIT_AT_RUN.txt` and `provenance_*.json` record all three values.
- **No code change under `darksiren_emri/` before the unseal row (R24).** A needed change voids
  the registration: no draw happens, or a drawn seal is spent. It returns as a new registration.
- Scripts outside `darksiren_emri/` (the draw script, the sealed submit and sbatch, the reader)
  are not physics. They are pinned by their own blob hashes in the launch row and committed
  before the draw.
<!-- RULE-BLOCK:E4-code-freeze END -->

**R24 disclosure (limitation, registered).**
- Row #385 found that the B_num completion-numerator term owns 93 % of the high-z zero-candidate
  pull. Freezing those 144 events' term profiles removes the entire 2D offset (+0.064). The
  finding is ESTIMATOR-INTERNAL by the R12 convention, and its mapping (illegitimate vs
  floor-consistent) is unruled (docket R24).
- T-1 runs with B_num unchanged. Under TRANSFER, the −0.064 offset carried to h_inj is therefore
  the offset this term produces.
- T-1 tests whether the stack is tuned to 0.73. It does not test whether the offset is
  legitimate. A later R24 fix changes the tree and so needs its own closure evidence; it does
  not inherit T-1's stamp.

## E-5. Pool option (amends draft §3; detail in `POOL_OPTION_MEMO_m2.md`)

<!-- RULE-BLOCK:E5-pool-option BEGIN -->
- **P2 first (ruling).** The sealed run reuses the E-1 pool of record. Before the draw, a small
  **unsealed** pool is generated at the public value h_P2 = 0.67: K = 20 tasks, ≈ 10 GPU-h, using
  the same `BASE_SEED + task` seeds as tasks 0–19 of the pool of record.
- **Invariance gate:** max |Δp_det| ≤ 1e-3 over the production p_det grid (60 d_L bins × 40 M
  bins, `local_linear`, z-resolved). Δp_det is the difference between the survival grid built
  from the K small-pool tasks and the grid built from the same K task IDs of the pool of record.
  The comparison is paired by design: same seeds and same task IDs, so the Monte Carlo noise
  cancels.
- **Gate P2-0, pairing check (run first).** Small-pool rows are matched to rows of the paired
  pool-of-record tasks by exact equality of z, M and the remaining drawn parameters (every column
  except `h_inj`, `luminosity_distance` and `SNR`). Exact equality is possible because the
  injection loop draws z and sets `d_L = dist(z, h)` afterwards (`main.py:1220`).
  - P2-0 passes if ≥ 95 % of small-pool rows find a match. Per-task wall-clock timeouts can drop
    rows on either side, so row-for-row equality is not required.
  - The invariance gate is then computed on matched rows only.
  - P2-0 fails if < 95 % match. The 1e-3 gate is then not meaningful (an unpaired 20-task grid
    has MC noise far above 1e-3), and P2 is **FAILED (unpaired)**.
- **Outcomes:** P2 PASS ⇒ reuse is registered and the draw may follow. P2 FAIL, from the gate or
  from P2-0 ⇒ **P1**: a fresh sealed pool at h_inj, cap 250 GPU-h, generated inside the sealed
  RUN_DIR and covered by the E-6 procedural blindness rule. P1 needs no naming shim and no code
  change (see the memo, §3).
- The gate is computed and recorded **before the draw**, because P2 is unsealed. Its number,
  its pass/fail and the small pool's list-md5 go into the launch row.
<!-- RULE-BLOCK:E5-pool-option END -->

## E-6. Grid, read rule and gates for (m2) (amends draft §4 item 3, §8)

<!-- RULE-BLOCK:E6-read-rule-gates BEGIN -->
**Grid:** the 55-node grid. That is `H_GRID_41` (0.600–0.860) plus the G-EXT wing 0.870–1.000 at
0.010 spacing, as the literal array in `cluster/a18_ma1d_headreadout_iiib.sbatch:131`
(admissibility decoupling `a26959b4`, ratified row #308). Edge nodes: 0.600 and 1.000. One h per
array task, task index = grid index, `EVAL_SEED 777000 + TID`. The T0 scorer weights are
`np.gradient(grid)` on the non-uniform grid.

**Read whitelist, before unseal.** The pre-unseal reader opens only:
- (i) `simulations/posteriors*/h_*.json`. Keys: `h` (the evaluation node, not the truth) and the
  per-event likelihood lists (`posterior_combination.py:42-80`).
- (ii) `simulations/diagnostics/event_likelihoods.csv`, restricted at load time to the columns
  `event_idx, h, combined_no_bh, combined_with_bh`. The full column set carries no truth, no d_L
  and no z (`bayesian_statistics.py:5439-5464`). Reading the m1 statistic required this file
  (`READ_RECORD_m1.md:19`), so it is whitelisted explicitly rather than by omission.
- The reader is a committed script whose blob hash is in the launch row. It prints moments,
  edge masses and N_used only, and never prints a CLI argument.

**Forbidden to every agent until the unseal row:**
- `RUN_DIR/logs`, `run_metadata_*.json`, `provenance_*.json`, the CRB files, the simulate
  `.out`/`.err`, and `~/.sealed/`. The simulate echoes `H_VALUE (truth)` (`simulate.sbatch:91`),
  `main.py:404-406` puts h in the log name, `cli_args.h_value` sits in the run metadata, and the
  CRB holds `luminosity_distance`.
- **Monitoring is by `squeue`/`sacct` only.** Job names and RUN_DIR carry no h (RUN_DIR is
  named by the commitment prefix).

**Gates, in two layers:**
- **Pre-unseal gates.** These need no sealed file and are scored by the reader:
  - g-censoring numbers: MAP position and edge masses. These feed E-2, not NO-READ.
  - g-znorm / g-precision: full-precision columns.
  - resolved-flag equality: the 13 tokens vs `REGISTERED_RESOLVED_FLAGS`, read by the ops script
    from `run_metadata` with `cli_args.h_value` stripped. Only the boolean pass/fail and the
    token count are printed.
  - P2 invariance gate (E-5).
- **Ceremony gates.** These need sealed-side files and are scored during the unseal ceremony,
  after the unseal and **before** the cell is computed:
  - G-1 pins (E-1).
  - G-2 freeze (E-4), from `provenance_*.json`.
  - G-3 commitment: sha256 recomputed equals the launch-row commitment.
  - G-4 blindness audit: a grep of every agent-written record between draw and unseal for the
    sealed value at 3 decimals returns nothing.
  - g-population: CRB rows vs N_used vs the simulate yield, against the draft §5 expectation of
    15–23 % SNR loss.
- Any red gate in either layer ⇒ NO-READ.
<!-- RULE-BLOCK:E6-read-rule-gates END -->

## E-7. Dated default (kill criterion 3 as extended)

<!-- RULE-BLOCK:E7-dated-default BEGIN -->
- **Deadline:** 2026-12-15, end of day Europe/Berlin (ruling `ask-20261018-w223027afac` R2;
  second extension).
- **A blind-mock verdict exists** iff an unseal ledger row exists with a banked cell, including
  INTERMEDIATE and AT-RAIL. A NO-READ, an unlaunched run, or a run that has not been unsealed is
  **no verdict**.
- **If no verdict exists by the deadline, kill criterion 3 fires as written.** The thread ships
  as a **methods** paper with the closure reported as **unblinded**, not as an H₀ measurement.
- **No third extension without a new author ruling.** The default fires by calendar; nobody
  re-dates it silently.
<!-- RULE-BLOCK:E7-dated-default END -->

## E-8. (m2) disposition table (replaces draft §9 rows 3–6; every row returns as a fresh RULE)

<!-- RULE-BLOCK:E8-disposition BEGIN -->
| (m2) cell | consequence proposed to the author (not executed by any agent) |
|---|---|
| TRANSFER | the −0.064 offset is h-independent: the anti-tuning check **passes**. c-no-tuning-to-0.73 → VERIFIED, and the anti-tuning stamp attaches per STANDING A-S. The mapping to kill criterion 1 is the card's Q4 |
| TUNED | an undetected tuning dependency: every iiib/joint_r1 verdict is re-scoped and paper claims HALT. Kill criterion 1 is proposed to fire |
| UNBIASED-AT-h_inj | the 0.73 offset is 0.73-specific. That is itself anomalous ⇒ a new stage-0 claim. Kill criterion 1 is proposed **not** to fire |
| AT-RAIL-LOW | a bound below both the TRANSFER and UNBIASED windows: the stack does not recover the sealed truth. The TUNED-band distance is reported. Proposed reading: kill criterion 1 fires ("closes anywhere other than the sealed h_inj"); the author rules |
| AT-RAIL-HIGH | a bound above every reference: a defect signature. Proposed reading: kill criterion 1 fires pending a defect hunt; the author rules |
| INTERMEDIATE (several / none / channel disagreement / σ-sensitive) | banked; the sub-reason is recorded. The seal is spent (cap 1): a second sealed draw is a new registration. For kill criterion 3 this counts as a verdict (E-7) |
| NO-READ | nothing banked; the seal is spent. For kill criterion 3 it is no verdict |
<!-- RULE-BLOCK:E8-disposition END -->

**Flag: kill criterion 1 against TRANSFER (not a change to the criterion).**
- Kill criterion 1 reads "if T-1 closes anywhere other than the sealed h_inj, the adoption
  reverts".
- The draft's TRANSFER row reads "T-1 PASSED", but TRANSFER means the posterior closes at
  h_inj − 0.064, which is literally "elsewhere".
- The two texts disagree on the most likely non-failing outcome. M162 does not change either
  text; the mapping is put to the author as a [RULE] (card Q4).
- Recommendation, flagged: TRANSFER does **not** fire kill criterion 1. The criterion targets
  tuning to 0.73. The h-independent offset is a separately tracked bias (R24, row #385) that
  already bars a closure-at-truth claim for the realistic stack at any h.

## E-9. Freeze slots (filled only by the launch ledger row, never by this erratum)

H_code = ______ · tree = `71acb98b0fd8a0f9d3aad9595eb9fc084cae0b6d` · prior = P-A U[0.79, 0.82] ·
pool option = P2 (gate value ______, small-pool list-md5 ______) or P1 · commitment = ______ ·
params sha256 = (THREAD.md) · draw UTC = ______ · reader blob = ______ · sealed sbatch blob = ______.

## E-10. Parameters read by `scripts/sealed_mock/draw_sealed_h.py`

The draw script reads the block below and refuses to draw unless:
- the block's sha256 equals the value recorded in THREAD.md at ratification;
- the prior equals the script's own registered constant;
- HEAD == H_code, the tree matches and the dirty count is 0.

<!-- T1-SEAL-PARAMS BEGIN -->
```
prior_lo = 0.79
prior_hi = 0.82
code_tree = 71acb98b0fd8a0f9d3aad9595eb9fc084cae0b6d
commitment_format = sha256(h_inj_6dp|salt_hex)
revision_cap = 1
```
<!-- T1-SEAL-PARAMS END -->

*Stamp: mission M162, 2026-10-19. No cluster command, no draw, no simulation, no edit under
`darksiren_emri/`.*
