# FULL VERIFICATION — TREE 1 ADJUDICATION (opus pass vs the 2026-08-30 sonnet pass)

Stamp: author-ordered full verification of both trees (row #278, author verbatim: "please also
do the full verification of both trees and your decisions via opus subagents and in parallel if
possible"). Tree-1 adjudicator: one top-tier agent, foreground only, no ssh, no git commit/add,
no code edits, nothing touched under b8_cal_harness_work_ladder (runner-9 in flight, its readout
out of scope, disclosed). Repo HEAD at adjudication: 7ab27ae3, branch fix/p32d-classg-venue-repair.
Date: 2026-08-31. Falsification brief A20 carried: the job was to refute, from source, never
from a record restating a number.

Inputs: verdicts_all.json (this directory), DEDUP_CONFLICTS.md (this directory),
REGISTRATION_END_VERIFIER_PASS_20260829.md (20 items),
END_VERIFIER_REPORT_PART1_20260830.md (the earlier sonnet pass: 18 confirmed, 0 refuted,
1 undetermined = item 19, 1 deferred = item 20), and the opus work directory
full_verification_20260831/work/ (scripts + saved outputs for T1-1..T1-19).

## 0. Disclosure: the opus pass's own aggregation failure, and how this adjudication handled it

verdicts_all.json holds ONE record (T1-1) although work/ carries completed re-derivation
evidence for all 42 items of both trees (DEDUP_CONFLICTS.md section 0: every work artifact
predates the file's last write at 10:47:45 by 4+ minutes; the mechanism is a lost-update race
on a shared JSON array). Consequence for tree 1: 18 of 19 opus verdict texts are lost.
This adjudication therefore ruled each tree-1 item from (a) the surviving T1-1 record, (b) the
saved opus numeric outputs (items 3, 4, 5, 7, 8, 12, 13, 14, 15, 17, 18, 19), and (c) the
adjudicator RE-EXECUTING the opus re-derivation scripts in foreground for the six items whose
output went only to the lost transcripts (items 2, 6, 9, 10, 11, 16). Every decisive number
that differed between passes, and every refutation-shaped signal in the opus outputs, was
additionally re-derived by the adjudicator directly from source (git objects, raw CSV/JSON/logs,
file mtimes) before ruling. The aggregation loss is reported as a governance item (G-1 below);
no verdict below rests on a lost text.

## 1. Verdict table

Columns: item | subject | sonnet verdict (2026-08-30) | opus pass evidence | ADJUDICATION.
"(re-exec)" marks items where the adjudicator ran the opus script itself; "(adj-src)" marks a
number the adjudicator re-derived from source independently of both passes.

| # | subject | sonnet | opus | ADJUDICATION |
|---|---|---|---|---|
| 1 | B1.1 wave-1 record (site 2.3 inertness, 18.6x cost anchor, GATE PARITY, must-fix cites) | confirmed | confirmed (surviving record; 1190.93/64.73 s verbatim from the raw log, ratio 18.6170; ternary at ff230621:5187-5191 theta-identity and unsmeared proven by ast on kw-defaults; PARITY 3.576e-06/5.718020e-04 and 4.232e-04/0.719 by keyed join) | CONFIRMED. New third-layer citation defect verified (adj-src): ledger row #258's F1 correction says the ln-guard sits at hier_s0_driver.py:452 at ecd33336; git show ecd33336 puts it at :444 (HEAD worktree :790). One-line append needed. REPORTED-ONLY carried. |
| 2 | B1.1 Stage-0 S0-A: B0-A-prime INSTRUMENT-DEFECT, STOP | confirmed | (re-exec) Z_b -3.676430700, Z_s -7.078606543 from the 20 raw CSVs; ENG mean fraction moved 0.98858; new robustness: leave-one-seed-out Z_b -2.48..-3.80 all negative, seed-level Z_b -2.27 / Z_s -5.53 | CONFIRMED, bit-level agreement with both passes; the STOP is robust to any single seed. REPORTED-ONLY carried. |
| 3 | B2.1 [CMEM] A1: R2c NOT-DISTINGUISHED, parked | confirmed | T = -0.12311421153794763 bit-identical; census 380/2336 = 0.16267 both arms; replication p_equal 0.0358, six fresh permutation seeds 0.0287-0.0358, none under 0.01; NEW: corr(d_bc, d_bt) = 0.99942 across the 10 seeds, dependence-respecting shared-permutation p = 0.1271, seed-level t = -1.58 (n = 10) | CONFIRMED (adj-src: strata mean equals T exactly; correlation and seed-level t re-computed by adjudicator). The park stands and is STRENGTHENED: the registered 20-strata permutation null treats bc/bt as independent, but they are near-duplicates — the effective unit is the seed (n about 10). Routed as a mandatory input to author items A7/A8 (any pooled or follow-up power calculation must use the seed-level null). REPORTED-ONLY / structural carried. |
| 4 | B3 closure PREMISE-REFUTED (provenance, zero compute) | confirmed | md5 9a1f2a14 re-hashed; 1514 dark / 76 in-catalogue; five-bin 605/491, underflow 1/2; sigma vs historical 7.1637 / 5.9522 | CONFIRMED, matches sonnet to every stated digit. No cap. |
| 5 | B4.1 [IMP] NOT EXONERATED; necessary cause of the 1D rail | confirmed | Delta_FT +0.12274490948527605 sem 0.0077368, 12/12 positive; fc +0.15181, ratio 0.80853; production pure-dark-only mean_h 0.7133924929 / MAP 0.70 / sigma 0.02771 bit-identical; gate I 5.5e-07 | CONFIRMED. [LOCAL] carried; row #167 and A11/A13 unchanged. |
| 6 | B4.2 KW-Q1 KERNEL-WIDTH-INERT | confirmed | (re-exec) pooled R = 0.08481225026498802 (sonnet third implementation 0.08481225026529439 — agreement to 3e-13, float-summation order); S triplet -1.0456670/-1.0205308/-0.9591134 exact; per-seed max R 0.156 < 0.2; GATE I 7.613e-08; T-ID parity max abs 0.0 over 348 rows; q1 share 0.92247 | CONFIRMED. The ENG figure reconciles: sonnet's 486/486 counts events, the re-exec's 972/972 counts rows (486 events x 2 h-nodes) — same fact. REPORTED-ONLY carried; the E21 divisor-gap narrowing (sonnet must-fix F8) was appended in row #258 and stands. |
| 7 | B5.1 [WIN] flag; zero-compute count refuted-in-direction | confirmed | pass fractions 0.9576806 / 0.6950869; true-host retention 0.9566563 -> 0.7890314; byte-identity 100000 pairs x 2 orderings, 0 mismatches; production-vs-replica per-event count diff 0 | CONFIRMED. No cap. The F2 verbatim-label correction was appended in row #258. |
| 8 | B5.2 C3 INTERMEDIATE, adoption NOT granted; retention transfer FALSIFIED | confirmed | Delta mean_h,pred = 0.00352244773070654 (equals sonnet's own re-derivation 0.0035224477; the record's 0.0035225271 differs 2.3e-05 rel, INTERMEDIATE either way); R6 max rel 2.407e-14 PASS; R2 982/951; 66/76 at all four h-nodes from the raw err logs, recomputed 86.842 percent, outside the mirror band 0.762-0.816; 621 collapse events 100 percent dark | CONFIRMED. No cap. The dedup-pass "conflict" with item 7 stays resolved (two different statistics). |
| 9 | B6.1 [ALIGN] s scales raw sigma_z before the PV fold; bit-identical at SIGMA_V_PEC_KM_S = 0 | confirmed | (re-exec) SIGMA_V_PEC_KM_S = 0.0 both modules; at sigma_pv = 200, s = 1.4142 the hooked production call matches the pre-fold closed form at relmax 0.0 and differs from the post-fold form (0.042432346 vs 0.042438691) at all three sites; hooked output matches the raw-z reading (relmax 0.0), differs from the z-tilde literal at 3.09e-03; at sigma_pv = 0 the theta hook equals direct substitution exactly for all nine (b, s) combinations | CONFIRMED, non-vacuous at production-output level. The raw-z vs z-tilde judgment call remains author item A9/A12. |
| 10 | B7.1 [2D-TWIN] proposal: centering numerically inert; cost 74.7-101.4 CPU-h | confirmed | (re-exec) sigma_cond p50 = 8.795887e-08 over all 1590 CRB rows (claim 8.8e-08); centering ratio p50 8.596e-14 (proposal band 8.6e-14..8.6e-16 reproduced), worst case 8.28e-11 — inert at any physical scale; cost band 74.7-101.3 vs record 74.7-101.4 (rounding); G27 418.1-567.5 vs 418.0-567.6 | CONFIRMED. Cap "supported" carried. Sonnet finding F3 (no panel report artifact on disk) is unchanged — not testable by re-derivation. |
| 11 | B7.2 C4 IMMATERIAL-PREDICTED +0.0025057; PROVISIONAL | confirmed | (re-exec) R1 0 violations / 6352 (2424 both-zero); R2 982/982 = 1.0; R6 exact 0.0 all four nodes; Delta ell-prime(0.665) = 7.429354969044, Delta ell-double-prime = -30.311364, Delta mean_h,pred = 0.002505684644, IMMATERIAL-PREDICTED; I_HEAD 2964.63 | CONFIRMED. STATUS CHANGE ON THE PROVISIONAL (adj-src): the C4 provenance extras the sonnet pass confirmed absent (run_metadata x4, logs/, GIT_COMMIT_AT_RUN.txt = ff230621, with-BH posteriors at all four h-nodes) were retrieved to wave2_20260829/c4/ at 02:18:25 on 2026-08-30 — 39 minutes AFTER the sonnet report was committed (9369a2ae, 01:39:28). The absence claim was true at writing and is now overtaken. The provenance leg of the PROVISIONAL is dischargeable on inspection; PROVISIONAL is carried solely on the unrun falsifier (ii) (row #220). |
| 12 | B7.3 adoption d4765539 confined to declared sites; suite 1896/15/27 | confirmed | suite claim REPRODUCED the hard way: git-archive snapshot of d4765539 out-of-tree gives 1880 passed / 31 skipped / 27 deselected, and the 16-test delta is exactly the 16 data-availability skips that only fire out-of-tree — in-tree 1896/15/27; HEAD today 1995/15/30; 12/12 pin tests pass; zero stray value-bearing changes; worker/kernel signature defaults unchanged (evaluate flipped only); five archived-script pins present; ruff/format/py_compile clean; 4 mypy errors in scripts/quick_validation_15.py are pre-existing and outside the gate scope | CONFIRMED. PROVISIONAL + supported carried; A4 ratification still pending wave 3. |
| 13 | B8.1 [CAL] floor sigma_h = 0.001747; Route A unstable | confirmed | independent implementation: 0.0017470584 (1D photo) / 0.0005604198 (spec) / 2D at all mass-scatter doses 0.0017375-0.0017470; algebra-vs-matrix rel 1e-15; disclosed slip re-derived (floor x0.799); width/floor 10.5726, bias/floor 38.2376; Route-A instability reproduced (n_z 400: 0.000371, n_eff 5.2; top-10 events all z < 0.075, worst idx 889 at 0.025 pts per sigma) | CONFIRMED. [INFO-STARVATION] stays un-resurrected per the register. |
| 14 | B8.2 harness cost correction, order-100 CPU-h, 20-80x | confirmed | REFUTES-IN-PART THE SONNET PASS'S OWN CORRECTION: the section-8 table rows are Cell S 100-380 and Cell T 25-95, so S+T = 125-475 (not the sonnet/F6 125-471; 380+95 = 475) and all-mandatory-rows = 159/160-516 (not 160-513). The F6 append (design note section 10) and row #258's F6 sentence therefore replaced the note's ORIGINALLY CORRECT upper bound 475 with an erroneous 471. Opus also quantifies: the 20x low end rests entirely on the undocumented 1.0 CPU-h/universe judgment (the note's own anchor gives a 5.2x floor); direction robust regardless | CONFIRMED on the decisive claim (order-100 CPU-h local, 20-80x the docket anchor — 20.6-78.3x with the corrected sum) — the adjudicator re-summed the source table (adj-src) and sides with opus. NEW MUST-FIX F6-prime: append a one-line correction to design-note section 10 and the row #258 F6 sentence: cell S+T = 125-475 CPU-h, all rows = 159-516 CPU-h, wall = 8.9-33.9 h at 14 cores. Note: tree-2 commit 05982a1b (S2c cache, warm evaluate 178 s -> 4-7 s) has since re-costed the pilot; that supersession is tree-2 scope and does not alter this item. |
| 15 | C0 baseline gate PASS bit-identical; costing 9-13x over | confirmed | max_abs = 0.0 on all 14 columns, 22232 value pairs, all cell STRINGS identical; both posterior JSONs md5-identical (563ef45b, 2b4fb3e0); OAT identity 3.1086e-15 both channels; 1.7244 CPU-h vs 15-23 = 8.70-13.34x; anchor line verbatim says 3355 events vs venue 1588 | CONFIRMED. The elapsed-string primitive is item 19's business (see below). |
| 16 | B1.2 PA-HIER-31 / F-A divergence; registration staleness | confirmed | (re-exec, exit 0, ALL CHECKS PASS) 9-event: L_cat_no_bh bit-identical; combined_no_bh max_rel 7.447115e-03; alpha_G_phi -12.017913 percent; D_tilde_phi -0.744711 percent; w_G 0.06196684 -> 0.05492879; full-N 106-event extension identical per-event-constant; F-B all 17 numeric columns bit-identical over the 9 shared events | CONFIRMED. The apparent cross-pass conflicts DISSOLVE (adj-src): 7.447e-03 vs 7.503e-03 and 12.02 vs 13.66 percent are the same value pairs under opposite denominator conventions (9.470921/9.400390 = 1.0075030; 5.8688310/5.1635200 = 1.1365896) — verified by direct arithmetic. Cosmetic: the P1 full-N appended entry mixes the two conventions within one line (quotes 7.503e-03 for D_tilde_phi and 13.66 percent for alpha where its own table convention gives 7.447e-03 / 12.02 percent); append-if-touched, no substance change. |
| 17 | Path choices + tree state; byte-identity stamped | confirmed | deviation table re-read (rows 3 and 9 DEVIATE, row 1 AGREE-with-three-deviations — matches sonnet's "rows 1, 3, 9"); wave-2 bands 179-357 / 224-447 re-derived exactly; dirty-vs-clean S0-A CSVs md5-identical (both 106-event and 9-event) — the byte-identity stamp is now TRIPLE-stamped; B6.1/B5.1 default-identity re-proven on 200000 pairs each | CONFIRMED. The opus probe's "sha1 mismatch" is RESOLVED (adj-src): 5313c319 IS the driver blob at dd63fe0c (the stale L2 pin), 9f831b9f IS the blob at ff230621 (the wave-2 re-pin ordered by the registration check; matches SYNTHESIS_DOCKET_1:232), 06f30030 is today's tree-2 HEAD blob — three distinct, correctly-documented states; no integrity issue. |
| 18 | Governance incidents; commit hygiene; F4 archival gap | confirmed with one new finding (the .gitignore-defeats-COMMIT_PLAN_3 finding, [DO] A13/A15) | runner chain reproduced from the raw logs (concat ValueError; daemonic assertion structural — reproduced in-process by opus; jobs field 1, label cosmetic); commit hygiene: A13 WAS EXECUTED at ecd33336 (2026-08-30 02:38) — 62 paths, 7.41 MB, zero filter violations, largest 2.7 MB — closing the 41-file archival gap in git; C4 duplicate with-BH posteriors (nested + toplevel) byte-identical, 261 MB wasted; residual: s0a_full_output.json untracked and NOT ignored | CONFIRMED. The sonnet F4 finding is now DISCHARGED-IN-GIT (cluster copy still owed once SSH returns). The "genuinely not retrieved" C4 sub-claim is overtaken as in item 11. New minor residuals for the A14-class housekeeping list: the untracked s0a_full_output.json; the 261 MB duplicate. |
| 19 | Compute ledger + F4 (cluster cost primitives) | UNDETERMINED (registered criterion unrunnable: no local sacct copy, SSH down) | mtime reconstruction from source artifacts: for all 8 C3/C4 tasks, (out-file mtime minus provenance-stamp mtime) sits a UNIFORM +17..19 s below the recorded sacct Elapsed (slurm prologue + epilogue; spread 2 s); C0 (no logs retrieved) reconstructs to about 389 s vs recorded 388 s via the c3 calibration; totals 1.7244 + 4.9733 + 6.8000 = 13.4978 CPU-h; every locally-sourced figure (KW-Q1 6.1517, P0 11.5097, S0-C 10.4170, T1.2 50.885) re-read from its own JSON | VERDICT CHANGED: CONFIRMED (by independent reconstruction). The adjudicator re-ran the span computation itself from the provenance stamps and out-file mtimes (adj-src) and reproduces all eight offsets (+17..+19 s, mean 18.0, spread 2 s) and the C0 gap (38 s, inside the 35-38 s calibration band). A mistyped or invented Elapsed string would break this uniformity at the seconds level; all nine primitives pass. Cap: this is corroboration through an independent local source, not the cited sacct source itself — the sacct dump re-pull (sonnet F7) stays on the part-2 list as belt-and-braces, and the ledger arithmetic (13.26-26.45x vs the 13-task band; 8.79-19.63x vs launched-arms-only) stands as sonnet computed it. The opus probe's apparent KW-Q1 discrepancy (17.66 CPU-h) is RESOLVED: it summed the P0 S0-A run (11.51, banked separately) together with the KW-Q1 main + parity runs (5.514 + 0.638 = 6.152, exactly the banked figure). |
| 20 | Wave-3 blind HEAD readout | DEFERRED (SSH down, wave 3 built not submitted) | not attempted (cluster down; Lustre /pfs/data6 OST 5 incident, commit 12e2436d) | STILL DEFERRED. Nothing in this pass reads or pre-judges it; the part-2 checklist in the sonnet report section 5 remains the binding plan, now plus: re-pull sacct (item 19 belt-and-braces), run the wave-2 archive script, register the wave-3 datasets. |

## 2. Every difference from the earlier pass, explained

1. Item 19 is the ONLY verdict change: UNDETERMINED -> CONFIRMED. Ground: a genuinely
   independent, source-level test (file-mtime spans vs the recorded Elapsed strings) that did
   not exist in the sonnet pass, re-executed by the adjudicator. The registered literal
   criterion (re-open sacct) remains unrunnable until SSH returns; the confirmation is
   explicitly by-reconstruction and the sacct dump stays owed in part 2.
2. Item 14: verdict unchanged, but the sonnet pass's own corrective sums (125-471 / 160-513)
   are refuted by the source table (125-475 / 159-516); the F6 append and row #258 propagate
   the error. A correction-of-the-correction (F6-prime) is required. This is the second
   instance of a defect inside a correction; see next.
3. Item 1: verdict unchanged; a third-layer citation defect found and verified (row #258 F1
   says :452 at ecd33336, actual :444). Pattern note for the record: two of the five A14
   housekeeping fixes themselves carry errors — corrections should cite a commit and be
   re-checked against it before appending.
4. Item 3: verdict unchanged; NEW material caveat — the bc/bt strata are correlated at 0.9994,
   so the registered 20-strata permutation p (0.029-0.036) overstates the evidence; the
   dependence-respecting p is 0.127. Binds A7/A8 inputs, not the A1 park itself.
5. Items 11/18: the C4 provenance extras landed locally at 02:18:25 on 2026-08-30, after the
   sonnet commit at 01:39:28 — the sonnet absence claims were true at writing and are now
   overtaken. Item 11's PROVISIONAL narrows to the falsifier-(ii) leg only. A13 (the 41-file
   force-add) was executed at ecd33336 and verified here.
6. Item 6: agreement between the sonnet ("bit-identical") and re-executed opus R differs at
   3e-13 — float-summation order across independent implementations, not a conflict. ENG
   486 vs 972 is events vs rows (x2 h-nodes).
7. Items 16/17: all apparent numeric conflicts between the passes dissolve as
   denominator-direction conventions; the driver sha1 chain (5313c319 at dd63fe0c ->
   9f831b9f at ff230621 -> 06f30030 at HEAD) is verified and self-consistent.
8. New minor governance observations: COMPUTE_LEDGER.md had one in-place cell edit at
   ff230621 (an empty B2.1 cost cell filled with the measured 0.017 CPU-h; +110/-1) — a
   trivial deviation from strict append-only in a living tally table, noted for completeness;
   s0a_full_output.json is untracked and not ignored; the C4 retrieval duplicated 261 MB of
   with-BH posteriors byte-identically.

## 3. Governance items from THIS pass

- G-1 (process, material): the opus fan-out lost 41 of 42 verdict records to a lost-update
  race on verdicts_all.json (DEDUP_CONFLICTS.md section 0; file write 10:47:45 postdates all
  work evidence). Tree-1 rulings above were reconstructed from saved outputs plus adjudicator
  re-execution; nothing rests on lost text. Fix for the next pass: one verdict file PER
  verifier, merged by the collector, never a shared read-modify-write array.
- G-2: F6-prime correction needed (item 14) — row #258 and design-note section 10.
- G-3: F1-prime correction needed (item 1) — row #258, :452 -> :444.
- G-4: A7/A8 must carry the seed-level dependence null (item 3).

## 4. Caps carried (verbatim)

REPORTED-ONLY: items 1, 2 (PA-HIER-28 item 9), 3 (structural, single-h), 6 (instrument-defect
disclosure carried; A14 falsifier not withdrawn). supported: items 10, 12 (B7 calibration
status). PROVISIONAL: items 11, 12 (attribution until falsifier (ii); item 11's provenance leg
dischargeable per section 2.5). [LOCAL]: item 5 (B4.1 forecast inputs). No cap upgraded or
dropped by this pass; item 19's new verdict carries its own by-reconstruction cap.

## 5. Summary (about 150 words)

The opus pass re-executed every decisive tree-1 computation from source and the adjudicator
re-ran or re-derived every disputed number. Outcome: 19 of 19 verifiable items CONFIRMED,
0 refuted, 0 undetermined, item 20 still deferred on the cluster outage. One verdict changed:
item 19 (cluster cost primitives) rises from UNDETERMINED to CONFIRMED because file-mtime
spans reproduce all nine recorded sacct Elapsed values to a uniform 17-19 s slurm overhead —
an independent local test the first pass lacked. Two of the five A14 corrections themselves
carry defects now verified from source: the F6 cost re-sum used 471/513 where the table gives
475/516, and the F1 line cite says :452 where ecd33336 has :444. The CMEM permutation
evidence weakens under a dependence-respecting null (p 0.127 vs 0.036), binding A7/A8. The
pass's own aggregation race lost 41 verdict texts — reconstructed here without loss of rigor.

Counts (tree 1, items 1-19): confirmed 19, refuted 0, undetermined 0; item 20 DEFERRED.
Changed vs the sonnet pass: item 19 only (UNDETERMINED -> CONFIRMED by reconstruction).

Append-only; nothing above any other file's divider was edited. Adjudicator: top-tier, tree 1,
full verification 2026-08-31.
