# TREE 2 SYNTHESIS DOCKET — 2026-08-30 (night)

Launched under rows #255/#268 — tree 2 docket. INFORMATION ONLY (row #222 form; charter section 3:
one synthesis docket per wave). Chair: top tier. Nothing in this docket asks for approval, changes a
default, lifts a STOP, or launches an arm; sections 4-5 stage the MORNING docket, at which the row
#268 night grant lapses ("until tomorrow", author verbatim, BIAS_HISTORY_LEDGER.md row #268,
2026-08-30). Branch fix/p32d-classg-venue-repair, HEAD f91d8a37. Cluster DOWN (Lustre OST 5); every
cluster item below is a queued estimate. Runner-9 (B8.2 S3) RUNNING at stage LADDER (b8_s3_runner9_stage.txt,
read 22:13; log START S3 ladder N=106 at 22:00:51, b8_s3_runner9_20260830.log:2) — its work root was
not touched. Every number carries {value, source, date}; all sources dated 2026-08-30 unless noted.
Chair re-derivations performed where a record's number is decisive: the T1.4 sigma arithmetic
(0.048435/0.0014037 = 34.5; 0.027435/0.0014037 = 19.5; 0.003735/0.0014037 = 2.66; /0.001724 = 2.17),
the T2.3 4-seed mean/SEM from the per-seed vector (+0.09695 +0.12936 +0.08951 +0.14738 -> +0.11580,
SD 0.0272 -> SEM 0.0136), the P1 Z ratios (0.042371/0.012752 = 3.3228; 0.023052/0.012906 = 1.7861),
and the section-6 compute sum — all reproduce the records to the printed digits. No record-vs-verifier
disagreement on a decisive number remains unresolved (the one substantive disagreement, B8.2 S1
acceptance (i), was settled empirically by the verifier's own live rerun; section 8 item 6).

---

## 1. Verdict table

| node | verdict + caps | decisive number {value, source} | reader/verifier state |
|---|---|---|---|
| T1.1 theta-consistent no-BH divisor (gate + build) | BUILT, byte-identical at default; panel-clean 0 rounds; [PHYSICS] commit 6c6f2a63 | rho((0,1)) == 1.0 exact; kernel cross-check 5.9e-8 rel {T1_1_DIVISOR_VERIFIER_REPORT.md items 2-3} | independent verifier: items 1-5 all PASS, must_fix none {T1_1_DIVISOR_VERIFIER_REPORT.md:12-24}; driver-gap finding flagged pre-run (row #260) |
| T1.2 S0-A re-certification | b-axis CERTIFIED (mechanism (i) CONFIRMED); s-axis B0-A-prime STOP stands, as pre-registered for divisor-only; REPORTED-ONLY cap (PA-HIER-28 item 9) | Z_b = -0.6762 (score_b -0.28878, within 0.0208 of the -0.268 prediction) {T1_2_RECERT_READOUT_RECORD.md:33,55-60}; Z_s = -5.9711 vs predicted Z ~ -6 {:34,62-68} | independent reader reproduced the driver to the last digit; per-seed forecast (-1.71/-1.26/+1.17/+0.69) matched almost to the digit {:38-45} |
| T1.3-zwin theta-consistent z-window (gate + build) | BUILT, byte-identical at default; [PHYSICS] commit 7e1ed96f; 3 gate revision notes (label -> T1.3-zwin; c-weighted convention downgraded to PROPOSED) | capture model: k=4 removes 99.9-100 percent of the s-capture term {PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md:296-316} | verifier: 1 MUST_FIX (scorer denominator, section 8 item 2) — applied before P1; ledger-row citation drift noted {T1_3_ZWINDOW_VERIFIER_REPORT.md:28-31} |
| T1.3 P1 arm (k=4 decisive run) | literal verdict B0-A-prime persists on the driver statistic; raw null RESTORED; convention gap is the decisive open item; REPORTED-ONLY | raw score_s Z = +0.3075 (was -5.971); driver corrected (unweighted) Z = -3.3228 FAIL; c-weighted Z = -1.7861 PASS, inside registered band [-0.031,+0.005] {T1_3_ZWINDOW_P1_READOUT_RECORD.md:34-36,88-94} | independent reader; Es_null_det closed form confirmed to 8.3e-17 {:17-21}; F1 not callable on one convention alone per the gate doc's own constraint {:66-107} |
| Es_null_det validity derivation + PA-HIER-33 | PROPOSED (row #274), NOT adopted — author [RULE]; P1 verdict of record unchanged until ruled | single-host null mis-scaled ~35x (unweighted) / ~20x (c-weighted); true many-candidate null +0.0013 +/- 0.0008 (Bartlett, 3 banked nodes); mixture 22x flatter in ln s {T1_3_ES_NULL_DET_VALIDITY_20260830.md:192-200,174} | top-tier derivation node; three-null P1 table at :251-256; falsifier registered before run |
| T1.4 Richardson half-step falsifier | EXECUTED on fresh data: PA-HIER-32(d) null REFUTED ~34.5 sigma; c-weighted REFUTED ~19.5 sigma; Bartlett null NOT excluded (2.66 per-event / 2.17 clustered); STOP stands until PA-HIER-33 ruled | paired shift +0.002435 +/- 0.001404 (0.001724 clustered); score_lns_R Z = +0.470 {T1_4_RICHARDSON_READOUT_RECORD.md:36,58-60} | independent reader re-derived from raw CSVs, reproduces driver to the digit; secant null Z +0.47 = s-axis consistent with zero UNDER PA-HIER-33 |
| T2.1 B4.3 derivation | candidate (b) mixture-weight h-slope REFUTED as mechanism (bookkeeping artefact of the -3/h common factor); candidate (c) depth skew IS the mechanism; Z(h) composition defect identified; zero compute | Z(0.73) = 1.099921, d ln Z/dh = -0.18895 per unit h per event (~ -273 nats/unit h on 1588 events) {B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md:146,359-367} | panel must_fix (1514 vs validated 1512) reconciled append-only, bands unchanged {:750-781} |
| T2.2 candidate-dump instrument + readout | DEPTH-SKEW-CONFIRMED, fully gate-clean after BI closure | Phi_low = 0.7299 +/- 0.0139 (16.5 SE from 0.5; 11.5 SE past the 0.57 threshold; predicted band [0.60,0.70] modestly undershot) {T2_2_CANDIDATE_DUMP_READOUT_RECORD.md:148-156}; BI bit-identical at both KW-Q1 h-nodes, all 4 seeds {:294-321}; Phi_low h-stable to 0.02 sigma {:345-359} | independent reader; GATE R 1e-13 rel; u_W magnitude overshoot disclosed (sign/threshold confirmed) {:165-181} |
| T2.3 mass-aware 1D leg (gate + build) | BUILT, byte-identical at default; [PHYSICS] commit 62f7d61e; 4 gate revision notes incl. the -117 in-catalogue input relabelled UNSUBSTANTIATED and the 17.1 sequencing rule (arm (c) BLOCKED on arm (b)) | Z = 1 identically under on (R2, r_Malm = 0.850 fixture); coded Z = 1.0999 control {PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md:1115-1126} | verifier: items 1-4 PASS; 1 must_fix, documentation-only (stale verified-row ledger citations — disposition still OPEN, section 8 item 8) {T2_3_MA1D_VERIFIER_REPORT.md:20,66-70} |
| T2.3 arm (a) mirror counterfactual | MASS-AWARE-MATERIAL, ABOVE the registered band — instrument-only, REPORTED; production flip = fresh author [RULE] | Delta mean_h = +0.1158 +/- 0.0136, 4/4 seeds positive; band was [+0.03,+0.10], point +0.05 (2.3x surprise) {T2_3_MA1D_ARM_A_READOUT_RECORD.md:30-32}; GATE T-ID bit-identical vs KW-Q1 {:38-51}; dark class exactly 0.0 {:117-124}; matched class rails at 0.86 in 3/4 seeds {:128-142} | independent reader; cost 66.56 CPU-h = 8-16x anchor {:156-159} |
| A11 completed-weight fork | SETTLED BY DERIVATION (row #268 "as recommended in spirit"): NEITHER branch is the estimator; D-tilde-phi does NOT complete; fork mooted by the mass-aware leg (Z=1 by identity); REPORTED-ONLY, returned to the author | Z_b1 = 1.183382 (slope -0.490), Z_b2 = 1.146747 (-0.354), both worse than coded 1.0999; b2 = b1 x a data-independent prior e^7.2 across the grid {A11_COMPLETED_WEIGHT_FORK_DERIVATION_20260830.md:184-190,243-254} | panel clean (refuted false, 1 cosmetic must_fix applied at 3 occurrences, panel said 2) {:420-463}; both banked fork numbers reproduced EXACT {:314-317} |
| T5.1 mass-law-keyed window design | DESIGN + REGISTRATION FILED, zero compute; panel REFUTED at round 2 -> corrected -> independent re-check PASS 6/6 (rows #270-#272) | production injects EMRI mass with ZERO scatter (host M = catalogue BH_MASS); delta law on iiib, log-normal on joint_r1; log window at k = Phi^-1(1-eps/2) exact-by-construction (0.9973 at k=3, CV-independent); linear k=1.5 retains 0.78-0.83, needs k~11.6 {PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md:15-48,170-186}; mirror 78.9 percent = floor-clip artefact (16.8 pts), post-repair 94.4 vs 95.4 {:206-235} | re-check note PASS on all 6 items {:758-842}; two-arm k-scan registered (~26-35 CPU-h, cluster) {:309-363} |
| B8.2 S1 generator | BUILT (mixture_selected, gw_scatter, resolved-flags STOP-gate); acceptance (i) = PASS on the verifier's corrected comparand | off-arm reproduction max_abs_diff = 0 on all 17 columns once the b0i "off" pin is used {B8_2_S1_VERIFIER_REPORT.md:36,160-199} | verifier PASS w/ re-diagnosis; 2 record-correction must_fix + grid-split gap carried to S2 {:210-230} |
| B8.2 S2 driver/scorer + S2b cache | BUILT + VERIFIED; scorer reproduces the banked venue-transfer cells bit-exact; draw-weight cache proven byte-identical | Tc(0.730) coverage 0/400 at all levels, PIT-KS D = 1.000000, reproduced from real banked vectors {B8_2_S2_VERIFIER_REPORT.md:48}; cache: draw 451.76 s -> 8.59 s warm (52.6x), same-seed artifacts max_abs 0 {B8_2_S2_RECORD.md:357-360,390-394,403-418} | verifier: all quantitative items PASS incl. first-ever live end-to-end universes; 1 carried must_fix (grid-split live test) + 3 should_fix {B8_2_S2_VERIFIER_REPORT.md:43-64} |
| B8.2 S3 ladder | IN FLIGHT (runner-9): N=106 point of the 3-point N-ladder, full 41-node grid, workers 8 | stage LADDER {b8_s3_runner9_stage.txt, 22:13}; started 22:00:51 {b8_s3_runner9_20260830.log:1-2}; expected evaluate() per universe [~8, ~33] min band, disclosed order-of-magnitude only {B8_2_S2_RECORD.md:441-460} | read the checkpoint before N=400/1588 (the record's own instruction, :474-486); S4 registration needs top-tier review before S5 (charter T3) |

Caps carried everywhere: every [HIER] statement REPORTED-ONLY (PA-HIER-28 item 9); T2.3 and A11
instrument-only; nothing above lifts the s-axis STOP, flips a production default, or launches S0-B.

---

## 2. The [HIER] instrument story, in plain language

The [HIER] program asks whether the production analysis would notice a mis-calibrated photo-z error
model (a bias b and a width factor s on every catalogue redshift). Before it can ask that on real
production data (S0-B), the measuring instrument itself must read zero on a control venue built so
that the true answer IS zero (S0-A: generator kernel == estimator kernel at theta = (0,1)). At the
start of tree 2 it did not: Z_b = -3.68, Z_s = -7.08. Tree 2 found and fixed THREE instrument
defects, in sequence:

1. THE THETA-BLIND DIVISOR (T1.1). When the instrument tilts every catalogue redshift by theta, the
   numerator of the catalogue likelihood follows, but its normaliser Sigma_phi did not — so the score
   at truth was non-zero by construction. Fix: an exact per-node ratio rho(theta) on the divisor
   (byte-identical at the identity). Result: the b-axis fell from Z -3.68 to -0.68, landing within
   0.021 of the pre-registered prediction, with all four per-seed values matching the forecast almost
   to the digit {T1_2_RECERT_READOUT_RECORD.md:33,38-45}. Mechanism (i) CONFIRMED.

2. THE THETA-BLIND Z-WINDOW (T1.3-zwin). The candidate ball selected galaxies with a fixed listed-z
   +/- 1 sigma window while the kernel it selects for moves with b and scales with s — selection and
   kernel were different objects away from the identity. Fix: transform the selection window with
   theta and widen it to k = 4, at which the selection interval IS the kernel's own support. Result:
   the raw s-secant fell from Z -5.97 to +0.31, inside its registered band
   {T1_3_ZWINDOW_P1_READOUT_RECORD.md:34,55-60} — the truncation defect the raw statistic sees is
   essentially gone.

3. THE MIS-SCALED NULL OFFSET (Es_null_det). The registered corrected statistic subtracted, per
   event, the finite-step bias of a SINGLE isolated host (+0.0463), but the actual event likelihood
   sums hundreds of candidates and is 22x flatter in ln s — its true null is +0.0013 +/- 0.0008,
   ~35x smaller {T1_3_ES_NULL_DET_VALIDITY_20260830.md:192-200}. Subtracting the oversized offset
   from a near-null raw score MANUFACTURED the surviving Z = -3.32. The fix is a registration
   amendment, PA-HIER-33 (the arm's own Bartlett-identity null), which was PROPOSED, not adopted
   (row #274), and then tested on fresh, unseen data: T1.4 added half-step nodes and formed the
   Richardson secant, which is bias-free by construction for any smooth likelihood. The fresh data
   REFUTED the old single-host null at ~34.5 sigma and the intermediate c-weighted convention at
   ~19.5 sigma; only the Bartlett-scale null survives (2.66 sigma per-event, 2.17 clustered), and the
   Richardson statistic itself is null at truth (Z = +0.470) {T1_4_RICHARDSON_READOUT_RECORD.md:58-79}.

Where that leaves the instrument: UNDER PA-HIER-33 it is null-consistent on BOTH axes (b: Z -0.68;
s: Richardson Z +0.47, corrected score Z +0.21) — but PA-HIER-33 awaits the author, so the verdict
of record tonight remains B0-A-prime INSTRUMENT-DEFECT (s) STOP under PA-HIER-32(d), exactly as the
pre-registered reading rule requires. No number was re-read under the amended null except as the
disclosed, numbered exposure table the amendment itself carries.

What S0-B will read when the cluster returns (only after PA-HIER-33 is ruled — section 5): the same
theta-cross on the PRODUCTION venue, where no truth-theta exists. With the instrument certified
null-clean, any theta-pull S0-B finds is finally interpretable as venue physics — evidence about the
real photo-z error model and its leverage on H0 — rather than as instrument artefact. It must run
with the divisor on, the z-window at k = 4, and whichever s-statistic the author ratifies; its
capture term is predicted STRONGER on production (narrower h-prior envelope, 0.6-0.86)
{PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md:443-447}.

---

## 3. The 1D-rail story (the B4.3 chain)

Why does the production 1D channel rail at the 0.60 grid floor while the 2D channel does not?

1. DEPTH SKEW CONFIRMED (T2.2). For a dark event, the catalogue-leg weight inside its candidate ball
   sits systematically BELOW the true redshift: 73.0 percent of the weight (SE 1.4 percent, ~16 SE
   from the no-skew null), stable across seeds and h-nodes {T2_2_CANDIDATE_DUMP_READOUT_RECORD.md:
   148-156,345-359}. This drag is MODEL-CONSISTENT physics of the impostor population against a
   completeness function falling with z — not itself the defect. The claimed alternative (the
   mixture-weight h-slope owning ~63 percent) was REFUTED as a bookkeeping artefact: re-booked with
   the common -3/h volume factor removed, the global term is +0.013/event, wrong sign, ~6 percent
   {B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md:173-211}.

2. THE MASS-BLIND WEIGHT Z(h) DEFECT. The 1D catalogue leg pairs a MASS-BLIND numerator and class
   weight (beta_G_phi, S_bar_phi) with a MASS-AWARE divisor (D_tilde_phi carries alpha_G_phi =
   beta_G_phi x r_Malm). The likelihood therefore integrates over the data to Z(h) = 1.0999 with
   d ln Z/dh = -0.189 per unit h PER EVENT — an un-derived h-dependent prior worth ~ -273 nats per
   unit h on the 1588-event fleet, by itself a -0.21 shift against I_1D = 1303: the floor
   {B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md:359-367}. The 2D channel has Z = 1 identically, which
   is exactly where the two channels differ. A11's derivation confirms no completion of the old
   R-family fixes this (both branches make Z worse: 1.183/1.147) — the only derived resolutions are
   mass-blind-everywhere or mass-aware-everywhere.

3. THE MASS-AWARE LEG, ARM (a). The registered remedy (per-candidate S_4D, Sigma_4D divisor,
   alpha_G_phi weight — the exact 1D image of the 2D assembly, Z = 1 by identity) was built as a
   default-off instrument and run paired on the 4-seed mirror FT fleet: Delta mean_h =
   +0.1158 +/- 0.0136, all four seeds positive, mirror 1D mean_h moving from ~0.63 to ~0.75
   {T2_3_MA1D_ARM_A_READOUT_RECORD.md:100-106}. The dark class moved EXACTLY zero (the flag reaches
   only catalogue candidates — the designed structural blindness, confirmed on production-shaped
   data), so the entire effect lives in the matched class. Verdict bucket: MASS-AWARE-MATERIAL.
   THE TWO-OPPOSITE-BIASES CAVEAT: the measured +0.1158 is squeezed between two opposite grid
   truncations — the off arm's MAP rails at the 0.60 FLOOR (4/4 seeds) and the on arm's
   matched-class MAP rails at the 0.86 CEILING (3/4 seeds) — so the paired Delta is a lower bound
   censored from both sides {:107-142}; and it is 2.3x the registered point prediction, ABOVE the
   band's upper edge (+0.10), a size surprise the registration has no bucket for (section 8 item 1).

WHAT THE PRODUCTION ARM (A18, as ruled in row #268) WILL DECIDE: the 41-node iiib posterior under
the flag. Z-CONFIRMED if MAP and mean land in [0.64, 0.72]; REFUTED if the MAP stays at the floor
(<= 0.605) with the dark-only pure arm unchanged — in which case the rail is owned by the
completion-leg residual instead and the flag is structural-consistency only; MIXED otherwise
{PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md:449-460,510-515}. Two binding riders: (i) the 17.1
SEQUENCING RULE — arm (c)/A18 is BLOCKED until arm (b) (T2.2b, local, ~4-5 CPU-h) derives the
in-catalogue S_4D/S_bar_phi transform, because the [0.64, 0.72] band's in-catalogue input (-117
nats) is currently REPORTED-ONLY/UNSUBSTANTIATED {:804-836}; (ii) truth is NOT predicted — a
~-0.14/event dark-class completion-leg residual (B4.3 section 4.4) remains and is B8's object, now
the largest unexplained item on the board (section 7). The production flip itself returns to the
author as a fresh [RULE] with the arm numbers (gate section 11; row #267 caps).

---

## 4. Author items for the MORNING docket

Each with its inputs and a one-line question; tags per the approval-scope convention.

(i) [RULE] PA-HIER-33 ratification. Inputs: the validity derivation (mis-scale ~35x; Bartlett null
+0.0013 +/- 0.0008; options (a) re-read / (b) Richardson-first / (c) decline, T1_3_ES_NULL_DET_
VALIDITY_20260830.md:291-371) AND the fresh-data T1.4 result already in hand — old null excluded
~34.5 sigma, c-weighted ~19.5 sigma, Bartlett survives (2.66/2.17 sigma), Richardson secant null at
truth (Z +0.470). Option (b)'s falsifier has effectively ALREADY RUN and favoured the amendment.
Question: ratify PA-HIER-33 (the s-statistic of record becomes the Richardson/Bartlett-null form,
closing the Revision-note-2 convention question), and accept the P1/T1.4 re-read under it — YES/NO?
(A YES makes the S0-A instrument certified on both axes and unblocks S0-B; a NO leaves B0-A-prime
in force with the located bug recorded against the statistic itself.)

(ii) [RULE, queued — inputs arrive with the A18 arm] Mass-aware 1D production flip. Inputs when it
returns: the A18 production-arm verdict against [0.64, 0.72] (after the 17.1-gated T2.2b transform),
tonight's +0.1158 +/- 0.0136 above-band mirror result, the two-opposite-rails caveat, row #169
pairing precedent. Morning sub-question that IS ripe now: given the matched-class 0.86 ceiling rail,
should the A18 array carry an extended grid above 0.86 (and the H_GRID_FULL low wing) so the arm is
not censored — YES/NO? (A grid change to the registered arm is a registration amendment; asked
before submission, not after.)

(iii) [RULE] F-ii design, with T5.1's per-venue laws. Inputs: production mass law is venue-dependent
(delta on iiib, log-normal on joint_r1); the log window at k = Phi^-1(1-eps/2) is exact-by-construction
on the scattered venue (0.9973 at k=3) while linear k=1.5 loses 16-22 percent of true hosts
one-sidedly and is not eps-keyable (k ~ 11.6); the 78.9 percent mirror retention is a pre-repair
floor-clip artefact (94.4 vs 95.4 post-repair) {PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md:15-48,
206-235}. Questions (T5.1 section 9): (1) F-ii = (a) adopt log k=3 / (b) keep linear / (c) arms
first (A1 = (c) already granted — confirm the joint_r1-first order); (2) ratify the re-attribution
of record (retire 78.9 percent as a design input, pointer notes to the three records); (3) the
Appendix-B [RULE]: does the standing grant cover launching the registered joint_r1 arm alongside the
k-scan, or does it wait? Plus [DO, tree 3]: the mirror mass_law flag (lognormal_observed, Convention
A in the mirror) as a gate item — presentation before code {:373-408}.

(iv) [RULE, queued — inputs arrive with the wave-3 readout] A4: ratify catalogue_numerator_survival_2d
= mz_sel / center = eff as production default if the wave-3 blind readout lands |Delta| <= T_mat =
0.008 against the banked comparand; else revert to off pending falsifier (ii) (row #268 A4 "as
recommended"). Nothing to rule tonight; listed so the morning docket carries it visibly.

(v) Everything else the records return:
- [RULE] A11 one-word ratification: "D_tilde_phi stays as coded; the completed-weight family is
  closed as un-derived" (A11_COMPLETED_WEIGHT_FORK_DERIVATION_20260830.md:400-416; REPORTED-ONLY
  until the word; mooted-by-identity if the mass-aware flip is later adopted).
- [RULE, small] The PA-HIER-32(d)-vs-T1.2 scope-note tension (row #266 disclosure): ratifying
  PA-HIER-33 in (i) supersedes the disputed subtrahend and closes this reconciliation item; if (i)
  is declined it needs its own word.
- [INFO] The T5.1 pointer-note scope question was orchestrator-adjudicated as within-grant
  (Revision note 4, rows #271-#272) and stands flagged for the end-of-tree-2 verifier — disclosed,
  no ask.
- [INFO] kappa(h) (smeared-vs-point phi divisor) is being logged for free by the T1.1 pass; a
  future production-smearing [RULE] input, not ripe.
- [INFO] T6 (CMEM >= 90-percent-power registration) stays BANK-AND-PARK per A8; Stage P stays MOOT
  per A15; S0-R stays FALLBACK/DISARMED per A16.

---

## 5. Cluster queue — submit order once Lustre OST 5 returns (each behind a fresh /cluster
preflight VERDICT: READY; costs at the corrected anchors)

0. LOCAL PREREQUISITE, runs before or during the outage: T2.2b (arm (b), ~4-5 CPU-h local,
   derivation section 6.4) — the 17.1 sequencing gate for item 2; and the B8.2 S3 ladder readout
   (runner-9 in flight).
1. WAVE-3 (tree-1 leftover, F2 ordering — must land before A4 can be asked): C0-prime off-gate +
   the two 41-task blind arrays, built and DRY_RUN at commit 60f9996e (row #252 P9); 82 tasks,
   159.8-290.1 CPU-h estimate {row #252}. Then A4 returns as item 4(iv).
2. A18 PRODUCTION ARM (mass-aware 1D on iiib, 41 nodes): ~30-60 CPU-h at the corrected anchor
   {row #268} (gate section 9 item 4's C4-anchored figure: 41 x 1.7 ~ 70 CPU-h — carry both,
   measured will adjudicate). SUBMIT ONLY AFTER T2.2b's derived transform exists (17.1 hard STOP,
   PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md:817-835); grid-extension question 4(ii) asked first.
3. S0-B (C1, [HIER] on iiib) — ONLY AFTER PA-HIER-33 IS RULED. Why: (a) A6 (row #255) requires a
   PASSING S0-A first, and the s-axis is certified only under PA-HIER-33 — under the current rule
   of record the STOP stands; (b) T1.4 showed at ~34.5 sigma on fresh data that the current
   statistic's null is mis-scaled — an S0-B scored under it would manufacture a spurious ~ -3-sigma
   s-read on the production venue and spend real CPU on an uninterpretable number; (c) the statistic
   of record must be fixed BEFORE unblinding a decisive read (prereg discipline). Cost: charter base
   ~7-27 CPU-h x 2.5-3.6 per-event count growth at k=4 -> ~18-97 CPU-h, against PA-HIER-31(i)'s
   60-92 band {TREE2_CHARTER_20260830.md:42; PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md:443-447}.
4. T5 K-SCAN: Arm S (iiib, log k in {2.0, 2.5, 3.5} + optional k = inf anchor) ~15-20 CPU-h (C3
   measured anchor 4.97 CPU-h per 4-task set); Arm R (joint_r1, k=3) ~11-15 CPU-h + a joint_r1
   C0-prime ingredient gate ~1-2 CPU-h {PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md:309-363}. Arm R
   launches only per the 4(iii) scope word; zero-compute census extensions (k = 2.0/3.5 pass
   fractions) banked BEFORE submission.
5. AVAILABLE, unranked: falsifier (ii) class-G fleet Option A-prime rung 1 (~40-60 CPU-h, chair
   recost, charter T4) — unblocks the B7.3 PROVISIONAL cap; the B4.3 enlarged-ball falsifier arm
   (~15 CPU-h local, B4_3 section 8.2) if the depth-skew attribution needs its independent test.
Also on recovery: A13's archive_run_wave2.sh + C4 provenance extras (housekeeping, blocked on ssh).

---

## 6. Compute ledger (F4), tree 2 — measured vs estimates

| item | measured CPU-h | anchor/estimate | ratio | source |
|---|---|---|---|---|
| T1.2 S0-A re-certification | 50.885 | 11.5 (6 cached) | 4.4x (8.5x) | T1_2_RECERT_READOUT_RECORD.md:132-149 |
| T2.3 arm (a) paired counterfactual | 66.56 | ~4-9 | 8-16x | T2_3_MA1D_ARM_A_READOUT_RECORD.md:156-159 |
| T1.3 P1 arm | 32.08 | ~35 nominal | ~0.9x (wall within 8 percent) | T1_3_ZWINDOW_P1_READOUT_RECORD.md:149 |
| T1.4 Richardson arm | 31.0 | ~20 | 1.5x | T1_4_RICHARDSON_READOUT_RECORD.md:48 |
| T2.2 dumps (1.31 + 2.21) | 3.52 | 3.4-3.9 | inside | T2_2_CANDIDATE_DUMP_READOUT_RECORD.md:260,371-373 |
| zero-compute nodes (T2.1, A11, T5.1, Es-validity, charter, docket) | 0 | 0 | — | their own stamps |
| B8.2 S1/S2/S2b builds + verifier smokes | not separately banked; order 1-3 CPU-h at <= 4 cores (multiple ~500-700 s runs) | — | — | B8_2_S1_RECORD.md resource note; B8_2_S2_RECORD.md sections 5, 9.1-9.4; B8_2_S2_VERIFIER_REPORT.md item 3 |
| B8.2 S3 ladder (runner-9) | IN FLIGHT, uncounted | evaluate() [~8, ~33] min/universe band at 41 nodes (disclosed order-of-magnitude) | — | b8_s3_runner9_stage.txt (LADDER); B8_2_S2_RECORD.md:441-460 |
| SUM of measured tree-2 items | 184.045 | — | — | chair ARITH: 50.885 + 66.56 + 32.08 + 31.0 + 3.52 (matches TREE2_DOCKET_PACKAGE_20260830.md section 3) |
| all-in with prior-day anchors (KW-Q1 5.514 + A14 unbanked 8.6) | 198.159 | — | — | package section 3; row #258 |

Cost-anchor lesson for the morning: the two big overruns share one cause each — T1.2's divisor pass
was costed assuming the row-parallel mitigation would engage (it did not fully), and T2.3's anchor
assumed a light per-seed wall for a full-41-node HEAD-basis run. The corrected per-cell anchors now
on the record (off-truth divisor cell ~702 s; ma1d arm ~8,550 s per arm per 4 seeds) supersede the
charter figures for any re-costing.

---

## 7. Next-tree candidates, ranked

1. B8 DARK-CLASS COMPLETION RESIDUAL, ~ -0.14 to -0.15/event on production — now the LARGEST
   UNEXPLAINED OBJECT on the board. The production dark-class completion leg scores ~ -0.15/event
   below the estimator's own composition tilt; the C5 "dark-only pure arm covers truth" is a
   cancellation, not a closure {B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md:416-430}. It bounds every
   impostor-leg remedy (truth NOT predicted even under the mass-aware flip) and is exactly what the
   B8.2 harness (S4 registration -> S5 execution, top-tier review before S5) exists to adjudicate.
   Candidates already named: the noiseless data law, the dark mass law in S_bar_phi, the
   analysis-depth cap.
2. THE MATCHED-CLASS 0.86 CEILING — grid extension. The mass-aware arm's matched-class MAP rails at
   the H_GRID_41 top in 3/4 seeds (and the off arm at the 0.60 floor in 4/4): both the A18 arm and
   any future paired mirror read are censored until the grid is extended (H_GRID_FULL low wing +
   an above-0.86 extension) {T2_3_MA1D_ARM_A_READOUT_RECORD.md:128-142}. Cheap, decisive, and a
   prerequisite for reading the true asymptotic size of the +0.1158.
3. THE MIRROR MASS-LAW FIX (mass_law = lognormal_observed) — tree-3 gate item, specified not built
   {PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md:373-408}: turns the mirror into a venue-faithful twin
   for the with-BH host-mass leg; the B8.2 harness is the consumer; retention-identity regression
   already designed.
4. FALSIFIER (ii) — the class-G fleet (Option A-prime rung 1, ~40-60 CPU-h): the only item that can
   lift the B7.3 attribution's PROVISIONAL cap (row #253); becomes ripe the moment A4 is ruled.
5. Also live, lower yield: the E12-successor n1/n2 candidates (S_bar_phi's own sigma_z dependence;
   the V2 mixture-weight covariance) — needed ONLY if PA-HIER-33 is declined or a certified S0-A
   still fails; the kappa(h) smeared-phi-divisor question (data already logging); seed 900102's
   consistently negative s-mean (flagged twice, E17-class strata, unadjudicated); T6 CMEM
   >= 90-percent-power registration (parked, available).

---

## 8. Governance incidents since row #255 (all disclosed in the primary records; none hidden)

1. T2.3 ARM-SIZE SURPRISE ABOVE BAND: measured +0.1158 vs registered band [+0.03, +0.10] and point
   +0.05 (2.3x); 2/4 seeds individually above the edge; the registration's bucket rule has no
   above-band category, so it was reported as-is, not reclassified {T2_3_MA1D_ARM_A_READOUT_RECORD.md:
   30-32,165-171}. Cost also overran 8-16x. Both facts travel with the flip [RULE].
2. SCORER DENOMINATOR MUST_FIX: the driver's _es_null_det_closed_form used the raw-secant
   denominator (sqrt2 - 1/sqrt2) where PA-HIER-32(d) requires ln 2 — a 1.02014x error that no test
   caught (the two tests either checked sign only or planted values); found by the T1.3-zwin
   verifier's independent re-derivation and fixed before P1 ran {T1_3_ZWINDOW_VERIFIER_REPORT.md:30}.
   A regression test pinning the closed form's own denominator was the verifier's named remedy.
3. BUILDER MONITOR-PARKING RECURRENCES: the B8.2 S2/S2b builder used background-wait patterns that
   the coordinator had to correct mid-task (the record's own words: "no Monitor/background-wait...
   pattern used for the FINAL timings... per the coordinator's correction mid-task",
   B8_2_S2_RECORD.md:343-349); S1 and the S2 verifier ran nohup+disown local processes (disclosed,
   polled, killed rather than left unattended — within the letter of the rule, at its edge). The
   2026-08-20 standing rule (never end a turn waiting on an untracked process) needed re-assertion
   for the third session running.
4. T5 FACTUAL REFUTATION CAUGHT BY THE PANEL: T5.1's width-drift claim (the seed-900001 realization
   "predates the exact-width writer"; 7.6 percent / +/-18 percent drift; k_eff = k/0.929) was
   REFUTED by the sidecar's own n_mass_width_floor = 24100 key and the file's git history (one
   creating commit, 7b30d1ff); panel verdict REFUTED at round 2 (row #270), corrected append-only
   (Revision note 3), independently re-checked PASS 6/6 (row #272). No registered band depended on
   the struck figures.
5. T5.1 SCOPE OVERREACH, SELF-RAISED: the node appended a pointer note to a banked record
   (B5_2_PULL_READ) before the author had ruled on the disposition that proposed it; the
   orchestrator adjudicated it within the rows #255/#268 grant (append-only cross-reference notes),
   recorded verbatim in Revision note 4, and flagged it — not settled as precedent — for the
   end-of-tree-2 verifier {PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md:729-753; rows #271-#272}.
6. B8.2 S1 RECORD MISDIAGNOSIS: the S1 builder reported acceptance (i) as a byte-identity FAIL and
   attributed it to code drift across five same-day [PHYSICS] commits; the independent verifier
   re-diagnosed it as the record's own comparand-configuration error (the b0i arm's deliberate
   catalogue_numerator_survival = "off" pin vs the "phi" the record's script used) and PROVED the
   correct-comparand run exact (max_abs_diff = 0, all 17 columns) {B8_2_S1_VERIFIER_REPORT.md:
   101-204}. The same record also claimed "34 new tests" where the diff contains 18. Both corrections
   are must_fix items against the record's text, not the code.
7. T2.2 EXECUTED-VS-REGISTERED DEVIATION: the instrumented run was commanded at a single h-node
   (0.73) instead of the design's 3-node grid, making the registered BI gate NOT EXECUTABLE as
   named; disclosed by the reader (UNDETERMINED, not failure) and closed by a targeted re-run at the
   comparand's own h-nodes (rows #264-#265). Verdict unaffected; the gap cost 2.21 extra CPU-h.
8. RECURRING DOCUMENTATION/CITATION DRIFT: stale line citations in gate-ledger rows (T1.1 cosmetic
   1-2 lines; T1.3-zwin up to ~63 lines; T2.3's verified row citing pre-implementation numbers —
   this last is a still-OPEN must_fix, correction not yet evidenced on disk
   {T2_3_MA1D_VERIFIER_REPORT.md:66-70}); the A11 panel's own "twice" undercount of a 3-occurrence
   mis-citation (all three corrected); the zwindow panel's own replacement line numbers off by 2
   (corrected from a fresh re-grep). Pattern: citations composed from intermediate states — the
   morning housekeeping bundle should carry the T2.3 ledger-row correction.
9. GATE PARITY NOT EXACT ON P1 (max rel ln-L diffs 3.9-44.7 percent vs the 5.718e-4 headline
   comparand delta): disclosed as consistent in kind with the RATIFIED E19 residual and its
   amplification path, not re-adjudicated {T1_3_ZWINDOW_P1_READOUT_RECORD.md:138-145}; carried.
10. NODE-LABEL COLLISION: "T1.3" was reused (charter: S0-B launch; orchestrator path decision: the
   z-window gate); resolved by relabelling the gate node T1.3-zwin (Revision note 2 item 2); the
   recommended one-line charter disambiguation note is not yet appended — morning housekeeping.

---

Stamp: TREE2 SYNTHESIS DOCKET, chair (top tier), 2026-08-30 night — launched under rows #255/#268 —
tree 2 docket. Information only; append-only; no git, no ssh, no code, zero evaluate() calls by this
node; runner-9's work root untouched. The row #268 grant lapses in the morning with a fresh docket
and the verifier pass; the registered end-of-tree-2 verifier registration
(REGISTRATION_END_VERIFIER_PASS_TREE2.md) remains to be authored ahead of the wave-3 data per
amendment F5 (charter section 4).

## Appended correction (2026-08-30 night, orchestrator) — §5 item 0 runnability

T2.2B_ARM_B_RUNSHEET.md (extraction agent, verified in code and on disk) shows §5 item 0's "LOCAL
PREREQUISITE" label is wrong: arm (b) needs (i) a production-venue path the θ-driver does not have
(CONFIG_CHOICES = b0i, ft only — it runs via `python -m darksiren_emri` directly) and (ii) the
`simulations/injections` pool, which exists only on the cluster ($WS/run_20260729_seed61000; every
local `*_iiib` symlink to it is broken while Lustre is down). The CRB CSV is local and pin-verified.
Consequence: T2.2b moves INTO the cluster queue (before the A18 arm, same submission session); item 0's
cost estimate stands. SUPERSEDES the "runs before or during the outage" clause only.
