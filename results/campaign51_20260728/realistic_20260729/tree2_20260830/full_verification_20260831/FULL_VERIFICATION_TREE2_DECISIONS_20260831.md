# FULL VERIFICATION — TREE 2 + DECISIONS AUDIT — 2026-08-31

Adjudicator: top tier (tree 2 + decisions), launched under row #278 item (6). Falsification
brief A20: every decisive number below was re-derived FROM SOURCE (raw CSV/JSON/code/git) by
this adjudicator or by a verifier whose work-directory artifact this adjudicator re-read and,
where conflicted or lost, re-executed. Foreground only; no ssh, no git writes, no code edits;
b8_cal_harness_work_ladder and runner-9 untouched (their readout is out of scope, disclosed).
Adjudicator scripts: work/ADJ_t28_conflict.py, work/ADJ_spure.py, work/ADJ_spure2.py, plus
reruns of the eight verifier scripts whose stdout was lost (see PROCESS NOTE).

PROCESS NOTE (breach, disclosed): verdicts_all.json holds 1 of 42 verdict records (T1-1 only);
a lost-update race between parallel verifier writers clobbered the other 41 (DEDUP_CONFLICTS.md
section 0; confirmed — the file's single record and the 42-item work/ footprint were both read
directly). The tree-2 and decisions verdicts below were therefore reconstructed by this
adjudicator from the surviving per-item numeric artifacts in work/ plus reruns of the scripts
whose outputs were not persisted (T2-2, T2-3, T2-6, T2-9, T2-11, D-1, D-3, D-4 — all read-only
re-derivation scripts, rerun verbatim). Every decisive number quoted below was seen by this
adjudicator in a from-source artifact or recomputed directly; nothing below restates a record
without that check.

---

## Section 1 — Tree-2 verdict table

Verdicts: CONFIRMED = the record's decisive numbers reproduce from source to the stated digits.
REFUTED-DETAIL = headline stands, a named detail in the row/record is wrong (listed in sec. 3).
Caps carried verbatim throughout: every [HIER] statement REPORTED-ONLY (PA-HIER-28 item 9);
T2.3 and A11 instrument-only; the A18 band REPORTED-ONLY pending T2.2b; B7.3 PROVISIONAL
untouched; nothing here lifts a STOP or flips a default.

| item | node (ledger row) | verdict | decisive re-derivation |
|---|---|---|---|
| T2-1 | T1.1 divisor build (rows #259-#260) | CONFIRMED | rho((0,1)) = 1.0 bit-exact (hex 000000000000f03f); module vs independent Sigma^phi(theta) rel diff 2.7e-9 to 7.3e-9; banked-CSV pin: L_cat ratio equals rho to 4.1e-14, all other columns 0.0 except num_log_term_no_bh (the divisor site, as designed) |
| T2-2 | T1.2 S0-A re-cert (row #266) | CONFIRMED | score_b -0.28878 +/- 0.42705 (Z -0.6762), score_lns -0.073409 +/- 0.012294 (Z -5.9711), n=461; per-seed score_b -1.714/-1.283/+1.112/+0.659 vs forecast -1.71/-1.26/+1.17/+0.69; dark n=5 exactly zero; all reproduce the driver from raw CSVs |
| T2-3 | T1.3-zwin build (gate) | CONFIRMED | capture model: consistent-convention k=4 removes 99.93-99.99 percent of the s-capture term (fixed-window 98.1-99.4); T1.2 raw comparands reproduced |
| T2-4 | T1.3 P1 + Es-null validity (rows #273-#274) | CONFIRMED | three nulls from raw CSVs: driver-form Z -3.3228, c-weighted Z -1.7861 (c_i mean 0.6161, median 0.6592), Bartlett-null Z +0.2058; Bartlett Es_null +0.00131 +/- 0.00078 (bootstrap); mis-scale ratios 35.35x / 20.6x; flatness 22.47x; Es closed form to machine precision |
| T2-5 | T1.4 Richardson (row #275) | CONFIRMED | score_lns_R Z +0.4703 (n=461); paired shift +0.0024346 +/- 0.0014037 per-event / 0.0017241 clustered; exclusions 34.51 sigma (unweighted null), 19.54 (c-weighted), 2.66 / 2.17 (Bartlett, not excluded); with-BH Z +2.145; per-seed values match the row to the digit |
| T2-6 | T2.1 B4.3 derivation (row #261) | CONFIRMED | Z(0.73) = 1.099921 from two independent sources (FT tables; production CSV identities); d ln Z/dh = -0.18895; s_beta = -3.28915; -3/h common-factor decomposition and the re-booked split (global +0.013/event wrong sign ~6 percent) all reproduce; fleet tilt -300 nats raw, -273 by the /Z convention (both stated) |
| T2-7 | T2.2 instrument + readout (rows #264-#265) | CONFIRMED | Phi_low = 0.72986 +/- 0.01394 (16.49 SE from 0.5; 11.47 SE past 0.57); GATE R max rel 5.1e-13 to 6.6e-13; BI max_abs_diff = 0.0 on 18 columns, both KW-Q1 h-nodes, all 4 seeds; engagement 157/191 = 82.2 percent; h-stability delta -0.00043 (0.02 sigma); true-host rows 0/606,571 |
| T2-8 | T2.3 arm (a) (row #267) | REFUTED-DETAIL (headline stands) | headline CONFIRMED: per-seed Delta mean_h +0.09695/+0.12936/+0.08951/+0.14738, mean +0.11580 +/- 0.013624, 4/4 positive; dark class exactly 0.0 every seed every node; GATE T-ID max_abs_diff = 0.0. TWO details refuted, sec. 3 items 1-2 |
| T2-9 | A11 fork derivation (row #269) | CONFIRMED | Z_b1 = 1.183382 (slope -0.49048), Z_b2 = 1.146747 (-0.35367), both farther from 1 than coded 1.099921 at every H_GRID_41 node with R defined; banked fork numbers reproduce EXACT on the registered twin basis (-0.002810 +/- 0.000467; +0.034358 +/- 0.004342); b2/b1 data-independent prior, fleet ln-span 7.19 nats at N=179.33 |
| T2-10 | T5.1 mass-law window (rows #270-#272) | CONFIRMED | injection mass zero-scatter (max rel dev 3.5e-7, CSV round-trip scale); log window 0.9973 at k=3 CV-independent; linear k=1.5 LN retention 0.784-0.832, k for 0.9973 = 11.56 at CV 0.86; floor-clip 16.8 percent; post-repair 94.4 (log k=3) vs 95.4 (linear k=1.5); pre-repair 78.7 vs recorded 78.9 (different arm subset, immaterial); Revision-note-4 verbatim adjudication present (adjudicator direct read) |
| T2-11 | B8.2 S1 (docket row) | CONFIRMED | off-arm reproduction max_abs_diff = 0 and float64 bit-identical on ALL 17 columns under the b0i off pin; new test defs 88-70 = 18 (the verifier's correction of the record's 34 confirmed) |
| T2-12 | B8.2 S2 scorer (docket row) | CONFIRMED | Tc(0.730) coverage 0/400 all levels, PIT-KS D = 0.9999999999995; scratch-vs-harness and banked-pin diffs 0.0; T0 control 200/200 all levels |
| T2-13 | B8.2 S2b cache (docket row) | CONFIRMED | draw 451.76 s cold vs 8.59 s warm (52.61x); same-seed artifacts bitwise identical across JSON/CSV/npz |
| T2-14 | B8.2 S2c cache (row #277) | CONFIRMED | fresh-seed byte-identity cache-on vs cache-off (ln_post max_abs 0.0, CSV raw bytes equal) on 4 new universes; warm evaluate 4.20-6.59 s inside the record's 4.1-7.0 band; monkeypatch boundary claim confirmed in code |
| T2-15 | A14 housekeeping (row #258) | REFUTED-DETAIL (substance stands) | F2/GitHub-reflog/append-only/8.6-CPU-h arithmetic all confirmed; F1's own new citation and F6's exact figures wrong — sec. 3 items 3-4 |
| T2-16 | A5 re-grade (row #257) | CONFIRMED | +11/-0 pure append; G7 row-16 cell untouched (append-only convention); retirement note present at line 97; one-byte row-#138 text drift vs its original (trivial); N-check numbers (r_M1 span 0.53-1.39, ratio 0.653, dark measured -0.4668 vs predicted -0.5334) match banked |
| T2-17 | T2.2b runsheet correction (appended to docket sec. 5) | CONFIRMED | theta-driver CONFIG_CHOICES = (b0i, ft) only, no iiib path, no production markers; injection pool cluster-only (7 seed61000 symlinks all broken locally); local mix200k pool fingerprint-matches the cluster copy (707 files, 200100 rows, strata and sky-bands equal); CRB and catalogue md5 pins MATCH |

The [HIER] certified-under-PA-HIER-33 chain (T1.1 -> T1.2 -> T1.3-zwin -> Es-null -> T1.4) is
CONFIRMED end to end from raw data; the 1D-rail chain (T2.1 -> T2.2 -> T2.3 -> A11) is
CONFIRMED end to end with the two row-#267 detail corrections (which STRENGTHEN the censoring
caveat and the R13 regression read). The B8.2 chain S1-S2c is CONFIRMED; S3/pilot readout
excluded (in flight, out of scope).

---

## Section 2 — Decisions audit (orchestrator itemizations and path choices)

| decision | verdict | wording at issue |
|---|---|---|
| Row #255 itemization (A1-A17, P1-P10) | FAITHFUL | A-item union maps one-to-one onto the docket/verifier items; A4 and A11 correctly excluded under the approval-scope rule; re-verified that both excluded items' inputs did not exist at ruling time (no wave-3 outputs on disk; no ledger row #168-#254 answers the row #167 fork) |
| Row #268 itemization (A18, A4, A11, night extension) | DEVIATION (4 counts) | (a) A18 cost: "approx 30-60 CPU-h at the corrected anchor" has NO source on disk — the registration (on disk 10.8 h earlier) says 41 x 1.7 = 69.7 CPU-h; the only other occurrence cites row #268 itself (self-referential). (b) A18 wording drops the registered rule's second condition (Z-CONFIRMED iff map_h AND mean_h in band), the arm-(b)/T2.2b hard sequencing STOP, the band's REPORTED-ONLY cap, and the flip's fresh-[RULE] label — all present in the registration it cites. (c) A4 comparand: "against the banked readout" vs the registered "separate off arm at the same commit, BOTH venues, after the C0-prime off-gate, pending falsifier (ii)". (d) A11 "as recommended in spirit" is an admitted non-match; also the A18 label collides with adopted research-cycle amendment A18 (row #153), undisambiguated |
| Row #278 itemization (items 1-6) | DEVIATION (4 counts) | (i) item 3 pre-authorizes the flip on "1D MAP 0.60 -> [0.64, 0.72]" alone — the registered rule requires MAP AND mean_h, and the band itself is REPORTED-ONLY pending T2.2b (the -117 in-catalogue input UNSUBSTANTIATED, the band's own inline cap); neither condition nor cap is carried. (ii) item 1's consequence "the [HIER] instrument is CERTIFIED on both axes at the T1.3 configuration" overstates: the b-axis was measured only at the T1.2 configuration (divisor on, z-window OFF); P1/T1.4 ran no b-nodes under zwin-on. (iii) item 5's "(already in flight under the lifted veto)" mislabels the live process — the running command is the ladder N=1588 costing point (n-universes 1, cell S, work-root b8_cal_harness_work_ladder, started 04:53), not pilot option (a); the veto-lift itself exists only in commit 05982a1b's message, no ledger row. (iv) item 2 answers docket 4(iii) with Arm S only — the design's decisive joint_r1 Arm R, the 78.9-percent re-attribution ratification, and the Appendix-B scope word are not addressed |
| Row #277 (S2c + pilot re-cut) | FAITHFUL, arithmetic slips | option-table sums: (a) hi 68690 s written where components give 67690 (wall 19.1 h vs 18.80 — conservative); (b) hi 24036 vs 25060 (understates 0.28 h); (c) lo off 100 s. Headline recommendation and the byte-identity/warm-cost claims fully confirmed (T2-14) |
| Path decisions P1/P3/P10 (wave-2, re-affirmed in tree 2) | FAITHFUL | P1 combined_no_bh max rel 7.447115e-3 reproduced to 4e-8 of the claim; P3 CRB md5 9a1f2a14384a9281c97ca3be312ddaab matches at worktree and HEAD blob, 1590/1514/76 split exact; P10 batch arithmetic (16 tasks 224-447; 13 tasks 179-357; add-on 119.4-172.7) reproduces |
| T1.2 -> T1.3 path decision (row #266) and Es-null-before-rerun path (row #273) | FAITHFUL | both recorded as orchestrator path decisions with their alternatives; both cite their decision-table anchors correctly (checked in the gate docs) |
| Row #271 orchestrator adjudication (T5.1 pointer note) | FAITHFUL | the adjudication IS recorded verbatim in Revision note 4 (adjudicator direct read — refuting the D-6 fragment's contrary sub-finding, which was a grep artifact); correctly flagged for this verifier pass, correctly NOT settled as precedent |
| Row #276 docket self-description | DEVIATION (1 count, cosmetic) | "7 [RULE] asks total" — the docket's own tags give 6 (4 primary + 2 secondary); the third bundled item is [INFO], no ask, by the docket's own text |
| Governance-incident disclosure (D-6) | FAITHFUL | all three incident lists verified present and consistent; every checked event is disclosed somewhere on the record (some only in ledger prose rather than an incidents list: runner-9 in-flight, the PA-HIER-32(d)-vs-T1.2 tension, the R13 seed-900103 read, the es_null_det class-split anomaly) |

---

## Section 3 — Refuted / undetermined items, in plain language

1. ROW #267 GATE ENG FRACTIONS (refuted detail). The row says combined_no_bh changed on
   92.97/96.15/85.47/96.40 percent of active events and that seed 900103 MISSES the R13
   90-percent regression bar (PASS 3/4). Re-derived directly from the on/off CSVs at h=0.73
   over the matched class: 96.88/100.0/96.58/99.10 percent — ALL FOUR seeds pass the bar.
   The recorded numbers do not reproduce under the row's own stated definition. Consequence:
   the R13 "miss, disclosed not adjudicated" item DISSOLVES; no follow-up owed on it.
2. ROW #267 / DOCKET CEILING RAIL (refuted detail, caveat strengthens). "Matched-class MAP
   rails at 0.86 in 3 of 4 seeds" — re-derived under the row #146 combine: it rails at 0.86 in
   4 OF 4 seeds (per-seed matched MAP on: 0.86/0.86/0.86/0.86; off: 0.60/0.63/0.61/0.60). The
   two-opposite-rails censoring caveat on +0.1158 is therefore STRONGER than recorded, and the
   grid-extension question (docket item 4(ii)) more clearly warranted.
3. ROW #258 F1 (refuted detail — a citation fix that itself mis-cites). F1 states the
   ln-transform guard sits "now at :452 at current tree-2 HEAD (ecd33336)". Re-derived across
   six commits: at ecd33336 the guard is at line 444; 452 is its position at the LATER commit
   6c6f2a63. The guard's code is byte-identical everywhere (that part of F1 holds), and the
   :425-at-ff230621 claim holds. Same defect family as the tree-1 verifier's independent
   finding on the A14 append.
4. ROW #258 F6 (refuted detail, direction conservative). The "corrected" bracket 125-471 CPU-h
   / 8.9-33.7 h wall does not follow from the table it cites: the cell S+T production rows sum
   to 125-475 CPU-h / 8.93-33.93 h; the every-row sum is 159-516 not 160-513. The 20.6x-77.7x
   headline factor is confirmed (re-derived 20.60x-78.30x from the raw brackets).
5. ROW #258 RUNNER-1 WINDOW (undetermined, informational). The record books the runner-1
   crashed attempt as 20:28:21-20:46:40 (18.3 min); the log's own stamps span 20:19:53-21:38:48
   (78.9 min, 9 stamps). The 8.6 CPU-h arithmetic follows from the claimed windows exactly;
   whether those windows bound the crashed attempt alone could not be established from the log.
   The cost line is informational (no ledger total depends on it); left undetermined.
6. THE A18 BAND'S PURE INPUT (undetermined — NEW, decisive for a pre-authorization). The
   registered band [0.64, 0.72] uses pure = +158 nats per unit h. Re-derived from the iiib CSV:
   +157.92 IS exactly the standalone pure-arm score (secant of B_num/D_tilde_phi — reproduced
   by this adjudicator to the digit), BUT the identity-consistent complement (Sigma s_full
   minus Sigma s_imp = -297.77 - (-291.16 - 129.72)) is +123.11, which is also exactly the O2
   JSON's score_pure_mean x 1588. Only +123.11 makes the band's three components sum to the
   measured full-fleet score at rho=1 (-297.77; with +158 they sum to -264). If +123.11 is the
   correct decomposition, the same section 6.3 arithmetic gives edges [0.6226, 0.6787] — the
   registered band shifts by about -0.027. Which decomposition binds is a derivation question
   (the pure-arm slope includes the composition tilt d ln(beta_Gbar/D_tilde)/dh; the complement
   does not) — squarely inside T2.2b's remit. Until derived, the band's pure input is
   UNDETERMINED between two exactly-reproduced candidates 35 nats apart. (Also: the band's high
   edge 0.72 is 0.01 wider than outward-to-grid rounding of the raw edge 0.7054 gives.)
7. VERDICTS_ALL.JSON LOST-UPDATE (process breach, confirmed). 41 of 42 verdict records absent
   despite all 42 items' computation completing before the file's last write (mtime evidence).
   This report reconstructs the tree-2 and decisions halves from the surviving artifacts and
   reruns; the tree-1 half (T1-2 through T1-19) needs the same recovery by its adjudicator.
8. D-6 SUB-FINDING ON REVISION NOTE 4 (refuted). The fragment recorded the verbatim
   adjudication as absent from Revision note 4; direct read shows it present, verbatim, with
   stamp. Row #271 is accurate on this point.

---

## Section 4 — What returns to the author

(B8.2 pilot readout and wave 3 excluded as in-flight, per the brief.)

1. [RULE — restate before it can bind] The row #278 item 3 A18 flip pre-authorization should be
   re-stated to the registered rule: flip iff map_h AND mean_h land in the band, the band
   itself being REPORTED-ONLY pending T2.2b — and now ALSO pending the pure-input adjudication
   (section 3 item 6: +157.92 pure-arm vs +123.11 identity-complement; the band may be
   [0.62, 0.68] rather than [0.64, 0.72]). Recommendation: fold the decomposition question into
   T2.2b's derivation scope (it is the same object — which in-catalogue/pure split the full
   likelihood actually implies), and ask the grid-extension question (docket 4(ii)) with the
   corrected 4/4 ceiling-rail count from section 3 item 2.
2. [RULE — small] Row #278 item 1's certification wording: the b-axis is certified at the T1.2
   configuration (divisor-only); no b-nodes have run under z-window-on. Either accept the
   transfer explicitly (the z-window transform does move with b, so it is a real assumption) or
   authorize a cheap 2-cell b-node pair under the T1.3 configuration before S0-B unblinds.
3. [RULE — restate] The A4 pre-authorization (row #278 item 4) should carry the registered
   comparand and scope: the separate off arm at the same commit, BOTH venues, after the
   C0-prime off-gate, pending falsifier (ii) — not "the banked readout".
4. [RULE — one word each, carried from docket 4(iii), not answered by row #278 item 2] (a) the
   joint_r1 Arm R launch scope (the design's own decisive arm); (b) ratify the 78.9-percent
   re-attribution; (c) the Appendix-B scope word.
5. [DO — housekeeping appends, zero compute] (a) row #267 correction note: GATE ENG fractions
   96.9/100/96.6/99.1 (R13 PASS 4/4) and ceiling rail 4/4; (b) row #258 F1 correction: guard at
   444 at ecd33336 (452 belongs to 6c6f2a63); (c) row #258 F6 bracket 125-475 / 8.93-33.93;
   (d) a one-line ledger row recording the B8.2 sizing-veto lift now living only in commit
   05982a1b's message; (e) row #276's ask-count is 6, not 7.
6. [DO — process] Re-aggregate verdicts_all.json (append-merge per item) before any downstream
   record treats it as the 42-item verification file; the tree-1 adjudicator should recover
   T1-2..T1-19 the same way this report recovered tree 2 and decisions.
7. [INFO] Everything else audited is clean: rows #255 (itemization), #256-#257, #259-#266,
   #269-#276 verdicts and caps reproduce from source; path decisions P1/P3/P10 faithful; all
   governance incidents disclosed somewhere on the record; dataset pins (CRB, catalogue) MATCH;
   T2.2b correctly moved to the cluster queue (local pool absent, fingerprints match cluster).

---

## Section 5 — Summary (root-goal register)

Tree 2's two scientific chains survive full falsification. The [HIER] instrument story is real:
all three defects (theta-blind divisor, theta-blind z-window, mis-scaled null offset) reproduce
from raw data, the fixes land where registered, and under the now-ratified PA-HIER-33 the
photo-z error-model instrument reads null on both axes — so S0-B, when the cluster returns,
will finally measure venue physics, not instrument artefact; the one caveat returned is that
the b-axis certification transfers from the divisor-only configuration untested under the
z-window. The 1D-rail story also survives: the depth skew is confirmed model-consistent
physics, the mass-blind/mass-aware composition defect (Z = 1.0999, -273 nats per unit h) is
exact, and the mass-aware leg moves the mirror 1D posterior by +0.116 with the dark class
untouched — but the production flip's pre-authorized band rests on a pure-leg input this pass
found ambiguous by 35 nats (band possibly [0.62, 0.68], not [0.64, 0.72]) and on a MAP-only
restatement of a MAP-and-mean rule, so the flip must wait for T2.2b and a restated condition.
Two recorded details were corrected in the arm's favor of caution (ceiling rail 4/4, R13 pass
4/4). One process breach: the shared verdict file lost 41 of 42 records; recovered here for
tree 2 and decisions.

Stamp: TREE-2 + DECISIONS ADJUDICATOR, 2026-08-31. No git, no ssh, no code edits; runner-9 and
b8_cal_harness_work_ladder untouched; append-only outputs under full_verification_20260831/.
