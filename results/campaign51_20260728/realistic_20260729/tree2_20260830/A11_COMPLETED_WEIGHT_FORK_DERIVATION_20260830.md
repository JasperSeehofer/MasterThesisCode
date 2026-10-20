# A11 — the row #167 completed-weight fork: does the global divisor D-tilde-phi "complete"? DERIVATION

launched under rows #255/#268 — tree 2 node A11

Date: 2026-08-30. HEAD at derivation: 647e86d9 (branch fix/p32d-classg-venue-repair; darksiren_emri/ clean at
read time). Zero evaluate(), no estimator code touched, no git, no ssh. Every number below is either read from a
committed or banked artifact (file:line or file:key given) or is arithmetic on such numbers (marked ARITH; the two
scratchpad scripts that did the arithmetic are named in section 5 and reproduce every banked comparand to the digit
before any new number is read). Author: the tree-2 A11 derivation agent (top tier). Status: DERIVATION, REPORTED-ONLY;
returned to the author as the answer to A11 (row #255: "A11 ... needs one word, 'ratified' cannot select a value";
row #268: "A11 = decide by derivation ... recorded as a tree-2 node, REPORTED-ONLY, and returned with the
derivation"). Nothing here is banked as a measurement and nothing here changes a production default.

## 0. Question of record and the decision in one paragraph

Question (BIAS_HISTORY_LEDGER.md:2417-2436, row #167 items 1-2; PREREGISTRATION_P3_TWIN_20260822.md:483,
amendment 12; A20_REVIEW_P3_COMPLETED_20260822.md:13, MAJOR-1): for the impostor-weight-switch family — the
registered candidate cat_term_completed(e,h) = cat_term_phi(e,h) R(h), R(h) = beta_G(h)/beta_G_phi(h)
(PREREGISTRATION_P3_TWIN_20260822.md:428) — does the global divisor D_tilde_phi also "complete" (COMPLETED-MATERIAL,
+0.034357 +/- 0.004342, 12/12 positive) or not (COMPLETED-SMALL, −0.002810 +/- 0.000467, 0/12 positive)? The fork
sits inside the claim card's remedy-family range [0, +0.123] (CLAIM_IMPOSTOR_DRAG_20260829.md:271 and :334).

Decision (REPORTED-ONLY, returned to the author): NEITHER BRANCH IS THE ESTIMATOR, and the fork dissolves under the
consistency criterion the author named (one density everywhere). Writing the assembled per-event likelihood as
p_i = (W L_cat,i + B_i)/D and integrating it over the data, the divisor is consistent with the numerator exactly when D
equals the sum of the two legs' data-integrals under ONE detection model (Z(h) := integral p_i dd = 1 at every h;
section 2). The catalogue leg's data-integral is its global weight W, because the per-candidate S_bar_phi inside
L_cat cancels against Sigma_phi in aggregate. Under the coded pairing W = beta_G_phi (the selected, mass-blind
catalogue mass); under BOTH branches of the fork W = R beta_G_phi = beta_G = integral f_bar p_pop dz, the
UNSELECTED catalogue mass (the instrument's S_bar == 1 table, p3_completed_rescore.py:110-115), so the numerator
integrates to a mass that no selection-weighted divisor can match: Z_b1 = (beta_G + beta_Gbar_phi)/D_tilde_phi =
1.1834 with d ln Z/dh = −0.490 per unit h (divisor not completed), Z_b2 = (beta_G + beta_Gbar_phi)/(R alpha_G_phi
+ beta_Gbar_phi) = 1.1467 with d ln Z/dh = −0.354 (divisor completed) — both FURTHER from 1 than the coded Z =
1.0999 (−0.189) and both h-dependent, i.e. both branches multiply the 1D likelihood by an un-derived prior Z(h)^N
of exactly the B_scale class (row #131). The only divisor that would close the identity for the R numerator is
beta_G + beta_Gbar_phi — P_det == 1 on the catalogue class against S_bar_phi on the dark class, a two-detection-model
hybrid MFG A2 forbids (its measured cost, +0.1846, is reported in section 5 as the third un-derived arrangement).
Therefore: (i) D_tilde_phi does NOT complete — the coded D_tilde_phi = alpha_G_phi + beta_Gbar_phi is the only
derived object in the family (bscale memo section 1, "ONE common denominator"; row #169 Appendix B (i) "D_tilde
stays" is CONFIRMED by derivation); (ii) the numerator does not complete either (row #168: R is a free global
multiplier) — so COMPLETED-SMALL (−0.0028) is the banked number of a registered but un-derived candidate, and
COMPLETED-MATERIAL (+0.0344) is that same candidate multiplied by a second un-derived global factor, the prior
[D_tilde_phi(h)/D_tilde_phi^b2(h)]^N (per-event log-slope +0.137 per unit h at h = 0.73; +7.2 nats across the
prior range for N = 179), which is why the D_tilde lever "flips the sign": it is a prior, not a repair; (iii) under
the T2.3 mass-aware 1D leg (alpha_G_phi at site W1, S_4D per candidate, Sigma_4D as divisor) the question is MOOT
by identity: the numerator's weight and the divisor's catalogue term are the same float, Z = 1 exactly, and there
is no S_bar content left to "complete" (section 4). Bearing on the claim card: the R-family's two values are
struck from the remedy list (re-labelled "measured cost of an un-derived multiplier"); the structural bound
[0, +0.123] on any paired re-weighting stands unchanged, and it no longer contains a +/-0.037 convention lever —
every admissible member of the impostor-weight family is now fixed by Z = 1 with one detection model (the twin,
adopted, row #197; the mass-aware leg, T2.3, its own fresh [RULE]). Zero-compute check (section 5): this node
reproduces, from the banked tables and the committed scorer, −0.002810 +/- 0.000467 (to 6.7e-16 per seed),
+0.034358 +/- 0.004342 and +0.042362 +/- 0.005033 (the reviewer's two REPORTED-ONLY arms, never on disk until now,
identified as D_tilde^b2 = R alpha_G_phi + beta_Gbar_phi on the b1 and on the OFF basis respectively).

## 1. Question (1): what "the impostor weight completes" means in the code's objects (HEAD 647e86d9)

The 1D per-event likelihood under production defaults (absolute_marginal; catalogue_numerator_survival = phi;
catalogue_global_selection = phi; catalogue_leg_1d_mass_aware = off):

    p_i(h) = ( W(h) L_cat,i(h) + B_i(h) ) / D_tilde_phi(h)                                   site W1, :6685-6692
    W(h) = beta_G_phi(h)  ["off"]   |   alpha_G_phi(h)  ["on", T2.3 instrument]               :6685-6689
    L_cat,i(h) = sum_{g in ball_i} w_g S_bar_phi(z_g;h) N_g(d_i|h) / Sigma_phi(h)              numerator sum :6161-6177;
                                                                                              per-candidate factor: site N1
                                                                                              :8444 (batch, point), :8466 (batch,
                                                                                              quadrature), :7694 (scalar twin);
                                                                                              divisor: site D1 :6091-6100
    w_g = R_eff_per_mbh(M_g)/(1+z_g)                                                          :1036 (_rate_weight)
    B_i(h) = integral (1 − f_k) S_bar_phi N_GW p_pop dz     (fused 1D completion numerator)   :6342 (integrand), :6507 (use)
    beta_G_phi(h)   = integral f_bar S_bar_phi p_pop dz                                       :2131
    beta_Gbar_phi(h) = integral (1 − f_bar) S_bar_phi p_pop dz                                :2132
    Sigma_phi(h) = sum_g w_g S_bar_phi(z_g;h);  Sigma_4D(h) = sum_g w_g S_4D(d_L(z_g;h), M_g(1+z_g))   :2745 (both, with_bh_mass flag)
    n_hat_w = Sigma_phi/beta_G_phi; r_Malm = Sigma_4D/Sigma_phi; alpha_G_phi = Sigma_4D/n_hat_w = beta_G_phi r_Malm;
    D_tilde_phi = alpha_G_phi + beta_Gbar_phi; D_phi = beta_G_phi + beta_Gbar_phi (reported)  :2493-2498

The "impostor weight" of the row #167 family is the per-event catalogue-leg term for a dark event, whose ball
contains only impostors in the B-SEL venue: W L_cat,i / D_tilde_phi. In the instrument it is literally
w = alpha_G_phi/r_Malm/D_tilde_phi = beta_G_phi/D_tilde_phi times L_cat_no_bh (p3_completed_rescore.py:163;
A20_REVIEW_P3_COMPLETED_20260822.md:10-11 "IS the production no-BH catalogue weight"). "Completing" it means
multiplying the GLOBAL class weight at site W1 by R(h):

    W^completed(h) = R(h) beta_G_phi(h) = beta_G(h),   beta_G(h) := integral f_bar(z;h) p_pop(z;h) dz          (S_bar == 1)

i.e. replacing the selected catalogue mass beta_G_phi by the UNSELECTED catalogue mass beta_G at the one site that
carries the class weight. It is NOT the per-candidate factor at N1 (that is the twin, adopted as production physics,
row #197): the family keeps S_bar_phi(z_g) inside the sum and re-multiplies the class weight so that "weight times
per-candidate S_bar has the same ensemble mean as before" (PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md:60-77, section 2).

Identification of beta_G (a labelling correction, no content change). The instrument builds R from
precompute_phi_selection_integrals called on an S_bar == 1 table over the SAME z-grid (p3_completed_rescore.py:110-115,
:143), so beta_G = integral f_bar p_pop dz — NOT the legacy beta_G = D(h) − beta_Gbar(h) of the S_3D-selected
partition (:4734). Proof from the banked R(h): f_bar is exactly h-free and p_pop is proportional to h^(−3), so
d ln beta_G/dh = −3/h = −4.1096 per unit h; the banked secant d ln R/dh = [ln R(0.735) − ln R(0.725)]/0.01 = −0.82050
(p3_completed_rescore_output.json r_of_h "0.725" 1.521825 / "0.735" 1.509390; ARITH) equals −4.10965 −
(−3.28915) = −0.82050, the negative of sigma_G, the survival slope of the selected in-catalogue population
(B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md section 2.3, sigma_G = +0.8205). A legacy S_3D beta_G would carry the
S_3D survival slope instead. B4.3 section 5.4's phrase "R = beta_G(legacy)/beta_G_phi" is therefore loose;
A20_REVIEW_APPENDIX_A_20260822.md:16 ("R = beta_G/beta_G_phi, measured 1.386-1.729") is the accurate one. The
derivation below is unaffected either way: any global multiplier on W that is not the numerator's own data-integral
has the same status.

"D_tilde_phi completes" (the fork's second branch) means applying the same replacement inside alpha_G_phi at
:2496-2497, since alpha_G_phi = beta_G_phi r_Malm enters D_tilde_phi:

    D_tilde_phi^b2(h) = R(h) alpha_G_phi(h) + beta_Gbar_phi(h) = beta_G(h) r_Malm(h) + beta_Gbar_phi(h)

(PROPOSAL section 2 "beta_G_phi -> beta_G in alpha_G_phi's construction"; amendment 12 :483-491). Section 5 confirms
this is the construction behind the reviewer's +0.034357 to the last printed digit.

The two branches, in symbols:

    b1 (numerator only; banked, row #167 item 1):   p_i = ( beta_G L_cat,i + B_i ) / ( alpha_G_phi + beta_Gbar_phi )
    b2 (D_tilde completed too; reviewer arm):        p_i = ( beta_G L_cat,i + B_i ) / ( R alpha_G_phi + beta_Gbar_phi )

## 2. Question (2): the mixture identity under each arrangement — the algebra

### 2.1 What the identity says

MFG (2019) Eqs. (5)-(7), as the repo's own derivation states it (bscale_completion_normalization.md:23-46): p_i =
num_i(h)/D(h) with ONE common denominator D(h) = integral P_det p_pop over BOTH classes, i.e. D is the data-integral
of the numerator under the model that generated the numerator. Split the numerator by class, num_i = A_i + B_i
with A_i = W L_cat,i, and define the class data-masses

    M_G(h) := integral A_i dd (summed over the full catalogue),   M_Gbar(h) := integral B_i dd.

Then "D = beta_G + beta_Gbar" is the statement D(h) = M_G(h) + M_Gbar(h), equivalently

    Z(h) := integral p_i dd = ( M_G + M_Gbar ) / D = 1   at every h,                                         (I)

together with the requirement that M_G, M_Gbar and D are generated by ONE population density and ONE detection
model (MFG A2; B3.2's principle, PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md:101: "the estimator must use ONE
population density at every site where it integrates over hosts it has not seen"). (I) is necessary, not
sufficient: any numerator can be given SOME divisor with Z = 1; the identity is only a derivation when that divisor
is the numerator's own data-integral under the model (section 2.4 shows the counter-example).

### 2.2 The class data-masses of each arrangement

integral N_g(d) dd = 1 for every candidate (the GW likelihood is a normalised density in the observable), so for a
catalogue leg of the form W sum_g w_g s_g N_g / Sigma_s with s_g a per-galaxy factor and Sigma_s = sum_g w_g s_g over the
eligible catalogue,

    M_G = W (sum_g w_g s_g)/Sigma_s = W.                                                                       (II)

The per-candidate factor inside the sum cancels its own global sum: the catalogue leg integrates to its GLOBAL
WEIGHT, whatever is put inside the sum. (Two disclosures common to every arrangement and inert for the fork: the
ball truncation captures a fraction kappa of each candidate's N_g mass, B4.3 section 5.3; on the quadrature path
S_bar sits inside the z-integrand while Sigma_phi is evaluated at z_g, a first-order residual the T1.1 instrument
measures. Neither depends on W or on D.) The completion leg integrates to M_Gbar = beta_Gbar_phi (fused; f_k versus
f_bar disclosed, bscale memo section 4).

    arrangement                         W (site W1)          s_g (site N1)      Sigma_s (site D1)    M_G          divisor D                        Z = (M_G + beta_Gbar_phi)/D
    coded (twin, production)            beta_G_phi           S_bar_phi(z_g)     Sigma_phi            beta_G_phi   alpha_G_phi + beta_Gbar_phi      (beta_G_phi + beta_Gbar_phi)/D_tilde_phi = D_phi/D_tilde_phi
    b1 (numerator completed)            beta_G               S_bar_phi(z_g)     Sigma_phi            beta_G       alpha_G_phi + beta_Gbar_phi      (beta_G + beta_Gbar_phi)/D_tilde_phi
    b2 (numerator + D_tilde completed)  beta_G               S_bar_phi(z_g)     Sigma_phi            beta_G       R alpha_G_phi + beta_Gbar_phi    (beta_G + beta_Gbar_phi)/(R alpha_G_phi + beta_Gbar_phi)
    mass-aware (T2.3 "on")              alpha_G_phi          S_4D(z_g, M_g)     Sigma_4D             alpha_G_phi  alpha_G_phi + beta_Gbar_phi      1 identically

### 2.3 The algebra of the two branches

With beta := beta_G_phi, bbar := beta_Gbar_phi, r := r_Malm, R := beta_G/beta:

    Z_coded − 1 = (1 − r) beta / (r beta + bbar)                                                              (III)
    Z_b1 − 1    = (R − r) beta / (r beta + bbar)             > Z_coded − 1   since R > 1                     (IV)
    Z_b2 − 1    = R (1 − r) beta / (R r beta + bbar)                                                          (V)

(IV): completing the numerator but not the divisor moves the estimator AWAY from Z = 1 by exactly the amount
(R − 1) beta/D_tilde_phi that the completion added to the catalogue class's data-mass with nothing added to the
divisor. (V): completing the divisor too does not restore the identity — Z_b2 = 1 requires r = 1, i.e. a catalogue
whose realised masses are as detectable as the population average, which the tables refute (r_Malm = 0.3828). Z_b1 = 1
would require R = r, impossible (R > 1 > r). Numerically, on the FT selection tables at h = 0.725/0.730/0.735
(p3_work/ft_900101_work/seed900101/selection_tables_h_0_725.json, _0_73.json, _0_735.json; run of 2026-08-23 at
53b7831e; identical to production to 7 s.f. per B4.3 section 2.5) with R(h) from p3_completed_rescore_output.json
r_of_h (ARITH, secant slopes over 0.01 in h):

    object                                   h = 0.725        h = 0.730        h = 0.735        d ln X/dh (per unit h)
    beta_G_phi                               155871370.5      153322758.6      150827919.0      −3.28915
    beta_Gbar_phi                            893324861.1      888403798.1      883510508.8      −1.10471
    R = beta_G/beta_G_phi                    1.521825         1.515549         1.509390         −0.82050
    beta_G = R beta_G_phi                    237209000        232368100        227658100        −4.10965  (= −3/h: the unselected volume)
    alpha_G_phi = r_Malm beta_G_phi          59650990         58688310         57746960         −3.24400
    D_tilde_phi (coded)                      952975900        947092100        941257500        −1.23728
    D_tilde_phi^b2 = R alpha + beta_Gbar     984103200        977348800        970673200        −1.37410
    D_phi = beta_G_phi + beta_Gbar_phi       1049196000       1041727000       1034338000       −1.42624
    beta_G + beta_Gbar_phi (the R numerator's mass) 1130534000 1120772000      1111169000       −1.72777
    Z coded = D_phi/D_tilde_phi              1.100968         1.099921         1.098890         −0.18895
    Z b1                                     1.186319         1.183382         1.180515         −0.49048
    Z b2                                     1.148796         1.146747         1.144740         −0.35367
    Z mass-aware                             1                1                1                 0
    1D catalogue class weight, coded (beta_G_phi/D_tilde)     0.1635628  0.1618879  0.1602409   −2.05187
    1D catalogue class weight, b1 (beta_G/D_tilde)            0.2489139  0.2453490  0.2418659   −2.87237
    1D catalogue class weight, b2 (beta_G/D_tilde^b2)         0.2410407  0.2377535  0.2345363   −2.73556
    1D catalogue class weight, mass-aware (alpha/D_tilde)     0.0625944  0.0619668  0.0613509   −2.00671

Reading. The coded pairing already integrates to Z = 1.0999 with d ln Z/dh = −0.189 (B4.3 section 4.3, the Z(h)
defect). Branch b1 raises that to 1.183 with slope −0.490; branch b2 to 1.147 with slope −0.354. Both branches make
the 1D likelihood LESS normalised and give it a STEEPER un-derived tilt than the coded leg, and both push the 1D
catalogue class share (0.245, 0.238) further from the generator's mass-aware share (0.062; B4.3 section 4.3,
monitored gate (ii) predicts 0.0453 against the realised 0.0479) than the coded 0.162 already is. Under the
identity, the fork is not a choice between a right and a wrong divisor; it is a choice between two wrong
normalisations of a numerator whose data-mass (beta_G, the unselected catalogue) no selection-weighted divisor can
equal.

### 2.4 The divisor that WOULD close the identity for the R numerator, and why it is not a model

Z = 1 with W = beta_G requires D = beta_G + beta_Gbar_phi: the catalogue class enters with P_det == 1 (its whole
population mass counted as detected) while the dark class enters with S_bar_phi. That is two detection models inside
one likelihood — the class of object the bscale memo removed from the completion leg (section 3, "the difference of
two detection models' volume-response slopes ... precisely the MFG-A2 violation") and the Appendix-A review's
"hybrid denominator no model produces" (A20_REVIEW_APPENDIX_A_20260822.md:10-11), now with the hybrid named exactly:
selection-free class G against selection-weighted class Gbar. Its measured cost on the banked tables is +0.1846
(section 5, REPORTED-ONLY) — an arrangement that satisfies (I) and is nevertheless not an estimator, which is the
demonstration that (I) is necessary and not sufficient. The identity is a derivation only when W is the
data-integral of the catalogue class UNDER THE SAME MODEL as the divisor: that is W = beta_G_phi with D_phi
(mass-blind everywhere, Z = 1, inconsistent with the generator's mass-aware detection of catalogue hosts — B4.3
section 4.3's first resolution) or W = alpha_G_phi with D_tilde_phi and S_4D per candidate (mass-aware everywhere,
Z = 1, consistent with the generator — B4.3 section 5.4, T2.3). The R family belongs to neither.

## 3. Question (3): the answer — consistency forbids the completion at BOTH sites; the resulting bound

### 3.1 The one-density-everywhere criterion, applied site by site

Sites of the 1D channel that integrate over hosts the catalogue does not contain, and the density each uses at HEAD:
beta_G_phi (site W1, the class weight; :2131): S_bar_phi p_pop. beta_Gbar_phi (inside D_tilde_phi and, via
B_i, the completion numerator; :2132, :6342): S_bar_phi p_pop. n_hat_w = Sigma_phi/beta_G_phi (inside alpha_G_phi,
:2494-2496): S_bar_phi p_pop in its denominator against the catalogue's own S_bar_phi(z_g) in its numerator — the
measure conversion whose S_bar content cancels (A20_REVIEW_APPENDIX_A_20260822.md:12-18; bscale memo :47-72, "no
remaining slot for any factor on either leg"). One density, one detection model, at every unseen-host site.

Branch b1 puts, at site W1 alone, the density p_pop with P_det == 1 (beta_G): a second density at one site.
Branch b2 puts it at two sites (W1 and inside D_tilde_phi through alpha_G_phi -> R alpha_G_phi = r_Malm beta_G, "the
unselected catalogue mass times the Malmquist ratio" — an object no model produces: the Malmquist ratio is defined
between two SELECTED sums, Sigma_4D/Sigma_phi, and has no meaning against an unselected mass). The criterion the
author named therefore FORBIDS the completion at the divisor (b2 fails at two sites) and, by the same clause,
forbids it at the numerator (b1 fails at one). "Does D_tilde_phi complete?" is answered NO — and the reason it is
"no" is the same reason the numerator's R was refuted in row #168: the only derived object in the family is the
coded D_tilde_phi = alpha_G_phi + beta_Gbar_phi, and R has no slot.

### 3.2 What each banked number IS

- −0.002810 +/- 0.000467 (COMPLETED-SMALL; row #167 item 1): the venue effect of an un-derived global multiplier
  R(h) on the class weight, against the derived divisor. It stands as the banked magnitude bound of a REGISTERED
  candidate under its sub-convention (amendment 12's wording) and licenses nothing about the estimator; row #168
  already re-labelled it "the twin contaminated by the spurious R-inflation".
- +0.034357 +/- 0.004342 (COMPLETED-MATERIAL; row #167 item 2): the same candidate multiplied by a SECOND
  un-derived global factor. Since every event's p_i^b2 = p_i^b1 times D_tilde_phi(h)/D_tilde_phi^b2(h) with the
  factor event-independent, the b2 joint posterior is the b1 posterior times the prior
  [D_tilde_phi(h)/D_tilde_phi^b2(h)]^N — data-independent, monotone increasing in h: ln(D_tilde/D_tilde^b2) =
  −0.058850 at h = 0.60, −0.031447 at 0.73, −0.018745 at 0.86 (ARITH from the banked bsel_seed900101 CSV columns
  alpha_G_phi, D_tilde_phi and r_of_h; section 5), per-event log-slope +0.1368 per unit h at 0.73 (−1.23728 −
  (−1.37410) from the table), span +0.0401 nats per event over the prior range, +7.19 nats for the fleet's mean
  N = 179.3 events per seed (n_events 174-188; ARITH) — a prior ratio of about 1330 between h = 0.86 and 0.60. THAT
  is the "15x lever": it is the strength of an h-prior, and it "flips the sign and the band" because a prior of
  e^7 across the grid moves any censored posterior upward. It is not a per-event repair of anything (B4.3 section
  5.2: the fork "acts only through the global weight and cannot touch the per-event skew" — confirmed, and
  sharpened: the D_tilde lever does not even act through the weight, it acts through nothing the data see).

### 3.3 The bound on C1's remedy-family range

CLAIM_IMPOSTOR_DRAG_20260829.md:271 lists the impostor-weight switch family as "twin / shape / completed / FULL-F"
with "COMPLETED −0.0028 or +0.0344 depending on the D_tilde sub-convention (row #167, author [RULE] pending)" and
:334 bounds "every catalogue-leg re-weighting" by C1's [0, +0.123] (the FT drag itself, +0.12274 +/- 0.00774,
:110-137). What this derivation changes: (a) the "completed" member is struck from the family's admissible set
in both sub-conventions — its two values are re-labelled measured costs of un-derived multipliers (R at W1; R
inside D_tilde), not remedy candidates; (b) the +/-0.037 convention lever that the fork placed INSIDE the range is
removed — no member of the admissible family (twin, adopted, +0.0291 on the fused basis, row #173/#197; the
mass-aware leg, T2.3 arm (a) +0.1158 +/- 0.0136 on 4 FT seeds, row #267, its own fresh [RULE] pending) carries a
free convention: each is fixed by Z = 1 under one detection model; (c) the structural bound [0, +0.123] itself is
UNCHANGED — it is the drag, not a property of any member — and the mass-aware arm's +0.1158 sitting at 94 percent of
it is T2.3's finding, noted here only as the family's current upper member. No number in this section is new
evidence about the drag; the claim card's C1-C5 stand as banked.

## 4. Question (4): the T2.3 mass-aware assembly resolves the fork by identity — under it, A11 is moot

Under catalogue_leg_1d_mass_aware = "on" (PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md:239-272, sections 2.3-2.4;
implemented :6685-6692 W1, :6091-6100 D1, :8444/:8466/:7694 N1 with the factor at :7025):

    p_i^(1D,4D) = ( alpha_G_phi L_cat,i^(4D) + B_i ) / D_tilde_phi,   L_cat,i^(4D) = sum_g w_g S_4D(z_g, M_g; h) N_g / Sigma_4D

M_G = alpha_G_phi by (II); D_tilde_phi = alpha_G_phi + beta_Gbar_phi by :2497; hence Z = 1 identically at every h
(gate doc section 2.4; arm (a) plausibility read: alpha_G_phi, r_Malm, D_tilde_phi constant across events at every
node, row #267). The weight at W1 and the catalogue term of the divisor are the SAME float (alpha_G_phi, :6685 and
:2497 through path_a at :6657-6660). The fork's question — "does the S_bar content that the numerator's global
weight carries get completed in the divisor?" — has no referent: writing alpha_G_phi L^(4D) = sum_g w_g S_4D_g N_g /
n_hat_w with n_hat_w = Sigma_phi/beta_G_phi (gate doc section 2.3, exact by :2494-2496), the S_bar content sits once
in Sigma_phi and once in beta_G_phi and cancels in expectation; there is no residual global S_bar factor anywhere in
the assembly for a completion factor to act on, and any R inserted at W1 would re-open exactly the Z != 1 defect of
section 2.3. So: if the mass-aware leg is adopted (A18, the production arm before any flip, row #268), A11 is moot by
identity; if it is not adopted, A11 is answered "neither branch" by sections 2-3 under the coded leg. In both states
of the world the D_tilde_phi of record is the coded alpha_G_phi + beta_Gbar_phi, un-completed.

Relation to row #169's ratified pairing. Appendix B (i) "D_tilde stays" is confirmed by derivation. Appendix B (ii)
"beta_G_phi stays" is confirmed AGAINST R (the alternative then on the table) and is the object T2.3 proposes to
replace by alpha_G_phi for a different reason (Z = 1 with the generator's mass-aware detection) — that replacement is
the fresh [RULE] B4.3 section 4.3 and the gate doc section 11 already return to the author; this node does not
pre-empt it.

## 5. Question (5): zero-compute check on the banked tables — every fork number reproduced, each branch's construction identified

Instruments (this node, 2026-08-30, scratchpad a11_check.py and a11_check2.py; pure arithmetic on banked CSVs plus
the committed scorer compute_seed_statistics over H_GRID_41, darksiren_emri/validation/correspondence_1d.py:3206
[SUPERSEDED 2026-08-30, see section 11: correct line is :3625],
PHYSICS_FLOOR + trapezoid — the row #146 form; no evaluate()). Inputs: the banked off-cell B-SEL CSVs
results/prod2d_closure_20260818/arm_event_likelihoods/bsel_seed<seed>/seed<seed>/simulations/diagnostics/
event_likelihoods.csv and the twin-cell CSVs results/campaign51_20260728/realistic_20260729/p3_work/phi_<seed>_work/
seed<seed>/simulations/diagnostics/event_likelihoods.csv, seeds 900101-900112 (both sets untracked/unchecksummed,
the recurring row #167 item 4 housekeeping — disclosed again); R(h) at the 41 grid nodes from
p3_completed_rescore_output.json r_of_h (commit 6885fc11, 2026-08-22, GATE T-C 4.76e-7). Reconstruction per row:
beta_G_phi = alpha_G_phi/r_Malm, beta_Gbar_phi = D_tilde_phi − alpha_G_phi, the off-cell completion numerator
B = combined_no_bh D_tilde_phi − beta_G_phi L_cat_no_bh(off); arms assembled as in section 2.2; nodes outside
H_GRID_41 left at banked (the instrument's own keep rule, p3_completed_rescore.py:165-175); statistic = mean_h(arm)
− mean_h(banked, trapezoid), paired per seed, 12 seeds.

    arm (numerator, divisor)                                   Delta mean_h (12)            n_pos   banked comparand                                   status
    banked baseline bias (fleet)                               −0.108302                    —       −0.108302 (row #162 anchor)                        EXACT
    twin (S_bar_phi num, beta_G_phi weight, D_tilde coded)     +0.015524 +/- 0.003657        12/12   +0.015524 +/- 0.003657 (row #162)                  EXACT
    b1 (R beta_G_phi weight, D_tilde coded)                    −0.002810 +/- 0.000467        0/12    −0.002810 +/- 0.000467 (row #167 item 1)           EXACT; per-seed max |diff| vs the JSON 6.7e-16
    b2 (R beta_G_phi weight, D_tilde^b2 = R alpha + beta_Gbar) +0.034358 +/- 0.004342        12/12   +0.034357 +/- 0.004342 (row #167 item 2)           EXACT to the printed digits
    D_tilde lever alone, OFF basis (coded off num, D_tilde^b2) +0.042362 +/- 0.005033        12/12   +0.042362 +/- 0.005033 (row #167 item 2)           EXACT
    D_tilde lever alone, twin basis (S_bar num, D_tilde^b2)    +0.075887 +/- 0.007828        12/12   — (new, REPORTED-ONLY)                             the same prior, non-linear response
    b1alt (R beta_G_phi weight, D = beta_G + beta_Gbar_phi)    +0.184645 +/- 0.002701        12/12   — (new, REPORTED-ONLY)                             the Z = 1 two-model hybrid of section 2.4

What each banked branch ASSUMED, now read off its reproduced construction: b1 assumed R at W1 with the derived
divisor (Sigma-chain, n_hat_w, r_Malm, Sigma ratios all at coded — the registration's "first-order, Sigma-chain
invariant" disclosure, PREREGISTRATION_P3_TWIN_20260822.md:466-468); b2 assumed R at W1 AND inside alpha_G_phi
within D_tilde_phi, with Sigma_phi, Sigma_4D, n_hat_w and r_Malm still at coded — i.e. NOT "the Sigma-chain
completed" either: a completed chain would rescale n_hat_w = Sigma_phi/beta_G_phi by 1/R and alpha_G_phi =
Sigma_4D/n_hat_w by R, which is what b2 did to alpha but then Sigma_phi (site D1) was left S_bar-weighted, so b2 is
itself a partial completion with no derivation behind the choice of which objects to complete. The reviewer's
"D_tilde lever alone" was measured on the OFF basis (coded numerator, no S_bar_phi per candidate), which is why it
exceeds the b2 − b1 difference (+0.0372): the identical prior applied on the twin basis gives +0.0759 — the
response to a data-independent prior of e^7 across the grid depends only on where the censored posterior's mass
already sits. Both readings confirm section 3.2: the D_tilde lever is a prior, and its size is set by N and the
posterior shape, not by any per-event object.

Linear-response cross-check (sign and order only; the prior is far outside the linear regime): per-event score
shift of the D_tilde lever +0.1368 per unit h, N = 179, banked censored sigma_h = 0.0231 (seed 900101 SeedStats)
gives Delta mean_h = N sigma_h^2 x 0.1368 = +0.013; the measured +0.037 to +0.076 exceeds it because a prior of
e^(24 Delta h) is not a small tilt on a posterior half-railed at 0.60 (floor mass 0.446 on production, claim card
C5). Sign and order agree; the exact rescore above is the decisive reproduction.

## 6. Consequences recorded for the claim card and the ledger (no edits made by this node; wording proposed)

- CLAIM_IMPOSTOR_DRAG_20260829.md:271, row (i): "COMPLETED −0.0028 or +0.0344 depending on the D_tilde
  sub-convention (row #167, author [RULE] pending)" -> "COMPLETED: struck from the admissible family by the A11
  derivation (tree 2, 2026-08-30) — both values are measured costs of un-derived global multipliers (R at W1;
  R inside D_tilde), the second a data-independent prior [D_tilde/D_tilde^b2]^N; the sub-convention fork is
  dissolved; the family's admissible members are the twin (adopted) and the mass-aware leg (T2.3, fresh [RULE])."
- :334: the bound [0, +0.123] stands; append "no convention lever remains inside the range".
- Row #167 item 2's "the D_tilde sub-convention returns to the author as the pivotal open [RULE]" is discharged by
  this derivation to the extent a derivation can discharge a [RULE]: the ruling proposed below is REPORTED-ONLY
  until the author's word.
- B4.3 section 5.4's label "beta_G(legacy)" for the R denominator: loose; the instrument's beta_G is the S_bar == 1
  phi-grid integral (section 1). Content unchanged.

## 7. Consistency with the record; exoneration check

Rows #168-#169 (A20_REVIEW_APPENDIX_A_20260822.md:10-18; PROPOSAL Appendix B :166): "D_tilde stays" — CONFIRMED
here by the Z identity, independently of the review's two routes (hybrid-denominator, S_bar -> c S_bar
homogeneity). "beta_G_phi stays" against R — CONFIRMED by (IV); the review's measure-conversion argument
(beta_G_phi/Sigma_phi = 1/n_hat_w, S_bar cancels) is the same fact as (II) read at the aggregate level. B4.3 section
4.3 (the coded pairing is itself a mass-blind/mass-aware hybrid with Z = 1.0999) — CONSISTENT: this derivation
shows the R family worsens that hybrid in both branches (section 2.3) and that the mass-aware pairing is the
resolution (section 4). B4.3 section 5.2 ("the fork acts only through the global weight") — CONFIRMED and
sharpened (section 3.2). bscale memo section 1 ("ONE common denominator ... no remaining slot") — the same
principle, applied to the catalogue leg's class weight. Exoneration grep (mechanism vocabulary: "completed",
"R-rescale", "B_scale", "beta_G/beta_G_phi", "D_tilde stays", "hybrid denominator", "Z(h)") over
EXONERATION_REGISTER_20260827.md: no entry exonerates or forbids this object; the register's binding items
[NUMERATOR-ONLY-CLEAN] and [WPOP-TUNING] are respected (nothing here proposes an unpaired change or touches w_pop);
rows #130-#131 (B_scale retired as un-derived) are the precedent this derivation applies. No new claim is opened.

## 8. What is not claimed; blindness

- Not claimed: any change to a production default; any value for the T2.3 flip (A18 runs first, row #268); any
  statement about the drag's size (C1-C5 stand as banked).
- Not claimed: that Z = 1 alone certifies an arrangement (section 2.4 is the counter-example); that the mass-aware
  leg is correct beyond what its own gate doc claims.
- Not claimed: the two new REPORTED-ONLY numbers (+0.0759, +0.1846) as anything but diagnostics of the algebra;
  they are off-basis (banked B-SEL, Sigma_3D divisor, pre-e35ea018, commit-6885fc11 tables) like every P3 fork number.
- Blindness: the derivation is exact in the assembled objects and first-order in the two truncation disclosures
  of section 2.2 (kappa, quadrature-path S_bar), both common to all arrangements; the FT selection tables and the
  banked off-cell CSVs are from different bases (fused 2026-08-23 versus off 2026-08-22) — the selection objects are
  shared catalogue-plus-pool objects (B4.3 section 2.5) and the Z arithmetic uses only them; the OFF/twin CSVs are
  unchecksummed at the consumer.

## 9. Numbers of record (value, source, date)

- Row #167 fork: −0.002810 +/- 0.000467 (0/12); +0.034357 +/- 0.004342 (12/12); D_tilde lever alone +0.042362 +/-
  0.005033 — BIAS_HISTORY_LEDGER.md:2417-2430; PREREGISTRATION_P3_TWIN_20260822.md:476-491; A20_REVIEW_P3_COMPLETED_20260822.md:13; 2026-08-22.
- Registered form cat_term_completed = cat_term_phi R(h), R = beta_G/beta_G_phi — PREREGISTRATION_P3_TWIN_20260822.md:428; instrument p3_completed_rescore.py:110-115, :143, :163, :170; 2026-08-22.
- R(h): 1.728540 (0.60), 1.521825 (0.725), 1.515549 (0.73), 1.509390 (0.735), 1.386157 (0.86) — p3_completed_rescore_output.json r_of_h; 2026-08-22 (commit 6885fc11).
- FT selection tables at 0.725/0.73/0.735 (beta_G_phi, beta_Gbar_phi, sigma_phi, sigma_4d, r_Malm) — p3_work/ft_900101_work/seed900101/selection_tables_h_0_725.json, _0_73.json, _0_735.json; run 2026-08-23 (53b7831e); r_Malm(0.73) = 0.38277622.
- Section 2.3 table and slopes (Z coded 1.099921 / −0.18895; Z b1 1.183382 / −0.49048; Z b2 1.146747 / −0.35367; D_tilde slopes −1.23728 / −1.37410; class weights) — ARITH, this node, 2026-08-30 (a11_check.py); Z coded and its slope agree with B4.3 section 2.2 to all printed digits.
- ln(D_tilde/D_tilde^b2) = −0.058850 / −0.043357 / −0.031447 / −0.023535 / −0.018745 at h = 0.60 / 0.66 / 0.73 / 0.80 / 0.86; ln Z_b1 = 0.255641 / 0.168377 / 0.119066 at 0.60 / 0.73 / 0.86 — ARITH on bsel_seed900101 event_likelihoods.csv columns alpha_G_phi, r_Malm, D_tilde_phi and r_of_h; 2026-08-30 (a11_check2.py).
- Rescore table of section 5 — ARITH + compute_seed_statistics (correspondence_1d.py:3206 [SUPERSEDED 2026-08-30, see section 11: correct line is :3625]), this node, 2026-08-30; n_events per seed 174, 184, 174, 182, 175, 188, 180, 181, 181, 178, 175, 180 (mean 179.33); sigma_h(900101, banked) 0.0230583.
- Twin +0.015524 +/- 0.003657 (row #162, :2309-2340); fused twin +0.029068 +/- 0.005088 (row #173, :2509); T2.3 arm (a) +0.1158 +/- 0.0136 (row #267, :3157; PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md:1205-1264); C1 FT drag +0.12274 +/- 0.00774 (CLAIM_IMPOSTOR_DRAG_20260829.md:110-137); range [0, +0.123] (:271, :334).
- Generator class share: w_tilde_G(0.73) = 0.0619668, predicted in-cat share 0.0453 vs realised 76/1588 = 0.0479 — B4.3 section 2.2 / 4.3, claim card C5; 2026-08-30 / 2026-08-27.
- Code anchors at HEAD 647e86d9 (2026-08-30 16:53 +0200): :1036, :1990, :2085, :2131-2132, :2449, :2493-2498, :2509, :2745, :4734, :6091-6100, :6161-6177, :6342, :6507, :6657-6660, :6685-6695, :7025, :7694, :8444, :8466 (bayesian_statistics.py); correspondence_1d.py:3206 [SUPERSEDED 2026-08-30, see section 11: correct line is :3625].

## 10. Decision and return text

DECISION (REPORTED-ONLY; returned to the author as the answer to A11 with this derivation as its input):

    A11: D_tilde_phi does NOT complete. Neither branch of the row #167 fork is the estimator: the impostor-weight
    completion R(h) makes the 1D catalogue leg integrate to the UNSELECTED catalogue mass beta_G, which no
    selection-weighted divisor can match (Z_b1 = 1.183, Z_b2 = 1.147, both h-dependent, both worse than the
    coded 1.0999); the only divisor that would restore Z = 1 is a two-detection-model hybrid MFG A2 forbids.
    COMPLETED-SMALL (−0.0028) is the banked cost of one un-derived multiplier; COMPLETED-MATERIAL (+0.0344) is the
    same plus a data-independent prior [D_tilde/D_tilde^b2]^N of e^7.2 across the h-grid. Row #169 Appendix B (i)
    "D_tilde stays" is confirmed by derivation; the R family is struck from the remedy list; the claim card's
    bound [0, +0.123] stands and no longer contains a convention lever. Under the T2.3 mass-aware leg the question
    is moot by identity (alpha_G_phi at W1 equals alpha_G_phi in D_tilde_phi, Z = 1 exactly).

Suggested one-word form for the author's record, if the author agrees: "ratified" on the sentence "D_tilde_phi
stays as coded; the completed-weight family is closed as un-derived" — a [RULE] on evidence already in front of the
author (rows #167-#169, B4.3, this node). If the author prefers to keep the fork's own vocabulary: the branch of
record is COMPLETED-SMALL's sub-convention on the DIVISOR (D_tilde un-completed) with the explicit rider that its
numerator is not adopted either.

launched under rows #255/#268 — tree 2 node A11 — derivation complete, zero evaluate(), no code, no git.

## 11. Revision note (appended 2026-08-30, post-refutation-panel pass; SUPERSEDED markers only, original text not rewritten)

A refuter panel reviewed this document (refuted=false; all findings CONFIRMED, no formula/algebra/number
overturned) and returned one must_fix item, cosmetic, not touching any banked number or the section 10 decision:

"Correct the compute_seed_statistics citation from correspondence_1d.py:3206 to correspondence_1d.py:3625 in both
places it appears (section 5 intro and section 9 Rescore table of section 5 bullet) -- cosmetic, does not affect
any banked number or the decision."

Verified directly against the source file at HEAD 647e86d9 (this node, 2026-08-30):

- darksiren_emri/validation/correspondence_1d.py:3206 is the line "catalogue_leg_1d_mass_aware: str = "off","
  inside the keyword-argument list of run_mirror_seed_inprocess (the [HIER T2.3] mass-aware 1D catalogue-leg
  instrument forwarded to BayesianStatistics.evaluate(); comment block :3199-3203 in the source). It has nothing
  to do with compute_seed_statistics.
- def compute_seed_statistics(...) is actually defined at darksiren_emri/validation/correspondence_1d.py:3625
  (signature opens "def compute_seed_statistics(diagnostics_csv: str | Path, seed: int,
  h_grid: tuple[float, ...] = H_GRID_41, h_true: float = H_TRUE, zero_handling: ZeroHandling = "physics_floor",").
  This is the function this node actually called for the section-5 rescore table; the panel's identification is
  correct.

The panel's own finding text says the mis-citation appears "twice." Re-grepping this document for the literal
string "correspondence_1d.py:3206" before this revision found it in three places, not two: the section-5
instrument-list sentence (originally line 299), the section-9 "Rescore table of section 5" bullet (originally
line 392), and a third instance the panel's finding does not separately call out — the closing sentence of the
section-9 code-anchors bullet ("...; correspondence_1d.py:3206.", originally line 395), which is the same citation
repeated as a bibliography-style anchor list entry rather than inline prose. All three point at the same wrong
line number for the same reason (the digits 3206 vs 3625 were transposed when this node drafted the document), so
the same correction applies to all three; the panel's "twice" undercounts the document's own repetition by one
occurrence but does not change the substance of the finding. All three are marked
"[SUPERSEDED 2026-08-30, see section 11: correct line is :3625]" in place above; the original erroneous line
number is left visible per the append-only / never-rewrite constraint on this document.

No other content in sections 0-10 is touched by this revision. In particular: the section-5 rescore table's own
numbers (n_events per seed, sigma_h(900101)) were produced by calling the function at its actual location
(:3625) regardless of what line number the prose cited, so nothing about the reproduction is affected — this is a
citation-text defect only, exactly as the panel's finding states. The section-10 decision (D_tilde_phi does not
complete; COMPLETED-SMALL branch as coded; the T2.3 mass-aware leg moots the fork by identity; bound [0, +0.123]
stands) is unchanged and is not itself re-ratified by this note — it remains REPORTED-ONLY, returned to the author,
per section 10.

Disposition: must_fix item applied (as an appended, non-destructive correction) at all three occurrences found,
not only the two the panel named. Status of this node: unchanged (REPORTED-ONLY; zero evaluate(), no code, no
git, this revision is a documentation-only append).
