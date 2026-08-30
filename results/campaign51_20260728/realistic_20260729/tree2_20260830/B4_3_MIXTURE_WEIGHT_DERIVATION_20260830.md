# B4.3 — the mixture-weight h-slope and the catalogue-vs-completion split of a dark event: DERIVATION

launched under row #255 — tree 2 node T2.1

Date: 2026-08-30. HEAD at derivation: ecd33336 (branch fix/p32d-classg-venue-repair). Zero compute, no
estimator code touched; every number below is either read from a committed artifact (file:line given) or
is arithmetic on such numbers (marked ARITH, with the formula). Author: the tree-2 T2.1 derivation agent
(top tier). Status: DERIVATION OF RECORD for docket 2 section 2 B4 / claim card C6 (b)/(c); it settles
the mechanism question at the derivation level and hands the per-candidate instrumented run (T2.2) a
registered statistic with predictions. Nothing here is banked as a measurement. Builder/runner rule:
this agent may not run T2.2, the rescoring statistic of section 6.6, or the enlarged-ball arm.

Question of record (docket 2 section 7 rank 2; CLAIM_IMPOSTOR_DRAG_20260829.md C3, C6 (b)/(c), section
1.4; B4_2_KWQ1_READOUT_RECORD.md; b4_imp_stage1_forecast.json covariates): the impostor-leg remainder
(about 81 percent of the B-SEL −0.1083; kernel width INERT, R = +0.0848; localised to z_true < 0.358 at
92.25 percent; catalogue-share correlated) was split at first order into a global mixture-weight h-slope
s_beta = −3.2891 per unit h owning about 63 percent of the per-event impostor score and a per-event
catalogue-vs-completion slope owning about 37 percent. Which of these is the mechanism, is the code-derived
s_beta what the model should produce, and which remedy does the derivation point at?

Decisive result in one paragraph. (1) The code reproduces s_beta = −3.2891 per unit h exactly, because the
quantity the split script differentiated, alpha_G_phi / r_Malm, is beta_G_phi by identity (code lines
2495-2496); its value decomposes as −3/h (the comoving-volume factor common to EVERY leg of the mixture,
−4.110 per unit h) plus the survival slope of the model's selected in-catalogue population, +0.821 per unit
h. The −3/h part is cancelled term-by-term by the same factor inside the completion leg, so the 63/37
split is a bookkeeping artefact of where a common factor is booked: re-booked with the common factor
removed, the global mixture-weight-and-divisor term is +0.30 per unit h times the catalogue share, i.e.
+0.013 per event (6 percent of the −0.218 score, WRONG SIGN), and the per-event catalogue-leg slope carries
106 percent. Candidate (b) of the claim card (the mixture weight's h-slope) is REFUTED as the mechanism;
candidate (c) (depth skew of the impostors inside the ball) is the mechanism, and the derivation gives
it a closed form: for a dark event at z_true the model-consistent expected impostor score is
E[s_imp | z] = f (d ln f / dz) z_eff / (h (1 − f)), with f the estimator's own completeness at z_true and
z_eff = d_L / (d d_L / dz); it is independent of the GW distance precision and of the photo-z kernel
width at first order (which is why SNR carries nothing and KW-Q1 came back INERT), it is largest where
df/dz is largest (z 0.15-0.36, the q1 localisation), and it reproduces the sign and the order of the −0.80
q1 score from the banked c(z) table (section 3.4). (2) That negative dark-class score is model-consistent
physics, not a defect by itself: the estimator's own normalisation identity requires it to be balanced by
the catalogue class in the ratio of the mixture weights, and the two banked class scores DO satisfy the
estimator's identity at the estimator's own composition (−0.158 measured versus −0.189 required, 1.8 sigma;
section 4.2). The defect is in the composition and the normalisation of the 1D channel: its catalogue
weight is the MASS-BLIND selection integral beta_G_phi (class odds beta_G/beta_Gbar = 0.1726) while the
generator's detected population has the MASS-AWARE class odds alpha_G/beta_Gbar = 0.0661 (the code's own
monitored gate (ii) predicts an in-catalogue share 0.0620 x 0.7305 = 0.0453 against the realised 76/1588 =
0.0479 on production); the 1D catalogue leg is therefore over-weighted by 1/r_Malm = 2.61, and the 1D
likelihood integrates over the data to Z(h) = D_phi / D_tilde_phi = 1.0999 at h = 0.73 with
d ln Z / dh = −0.189 per unit h per event — an un-derived h-dependent global factor of exactly the B_scale
class (row #131), worth −273 nats per unit h on the 1588-event production fleet (section 4.3). (3) The
derivation therefore points at a remedy the claim card did not list: the mass-aware 1D catalogue leg —
per-candidate S_4D(z_g, M_g) in place of S_bar_phi(z_g) in the 1D numerator, Sigma^4D as its global
divisor, alpha_G_phi as its weight — which makes Z equal to 1 identically and the 1D class share equal
to the generator's, exactly parallel to the 2D assembly the code already uses (section 5.4); because it
modifies the pairing ratified as Appendix B (i)/(ii) in row #169 (whose justifying display wrote D_phi and
identified it with the coded D_tilde, section 4.3), the flip is returned to the author as a fresh [RULE] and
only its gate presentation and instrument cell proceed under row #255. Its predicted effect (F3, section 7): the dark-class impostor score scales by the in-ball Malmquist ratio rho (anchor
r_Malm = 0.383): mirror FT fleet Delta mean_h = +0.05 (band [+0.03, +0.10]); production 1D MAP from the
0.60 floor to about 0.67 (band [0.64, 0.72]), NOT to truth, because a dark-class completion-leg residual
of about −0.14 per event on production (section 4.4, out of B4 scope, handed to B8 [CAL]) remains. Of the
three remedies the card did list: (a) a per-event mixture weight is inconsistent with the D_tilde
derivation and is the already-dead local-ratio family; (b) the row #167 completed-weight fork acts only
through the global weight and cannot touch the per-event skew (both branches stated, A11 stays with the
author); (c) the enlarged ball is the A14 falsifier of the depth-skew attribution, predicted to make the
q1 score MORE negative by the captured-mass ratio 1/kappa (1.15-1.5), not to cure anything. The per-candidate
instrumented run (T2.2, about 3.4-3.9 CPU-h local, 4 seeds x 3 h-nodes) is designed in section 6 with a
byte-identity guard, a reconstruction gate, the registered statistic (impostor-weight share below z_true,
Phi_low, predicted 0.60-0.70 against a null of 0.50, SE about 0.02) and, because it serialises the
per-candidate S_4D, a zero-compute rescoring of the mass-aware leg on the 4 seeds BEFORE any physics
change is written.

---

## 1. The estimator as coded (every formula with the line it was read from; bayesian_statistics.py at ecd33336 unless stated)

1D per-event likelihood, phi convention (absolute_marginal, catalogue_numerator_survival = phi,
catalogue_global_selection = phi; all production defaults):

    p_i(h) = ( beta_G_phi(h) L_cat,i(h) + B_i(h) ) / D_tilde_phi(h)          [:6270-6273]

    L_cat,i(h) = sum_{g in ball_i} w_g S_bar_phi(z_g;h) N_g(d_i | h) / Sigma_phi(h)
                                                                             [:5754-5757, :5771 (division by global_denom_no_bh);
                                                                              global_denom_no_bh = Sigma_phi under "phi" :5677-5690;
                                                                              per-candidate S_bar_phi factor on the numerator :7833
                                                                              (point path) / :7838-7841 (quadrature path)]
    w_g = R_eff_per_mbh(M_g) / (1 + z_g)                                     [:1036-1048 (_rate_weight); :2955-2957 (same w_g in Sigma)]
    N_g(d|h) = integral N_GW(d | z, Omega_g; h) rho_g(z;h) dz,               [:7699 point node, :7716 quadrature nodes; :7829 point value, :7841-7846 reduce]
    rho_g(z;h) = Gauss(z; z_g, sigma_g) f_k(g)(z;h) w_pop(z;h) / Z_g(h),     [C7-core prior :7677-7681; w_pop :7700-7704;
                                                                              Z_g = quadrature of Gauss x f x w_pop :7682]
    B_i(h) = integral (1 − f_k(z;h)) N_GW(d | z; h) (dVc/dz)/(1+z) S_bar_phi(z;h) dz
                                                                             [:5919-5950 (fused 1d)]
    beta_G_phi(h)  = integral f_bar(z;h) S_bar_phi(z;h) p_pop(z;h) dz         [:2131]
    beta_Gbar_phi(h) = integral (1 − f_bar) S_bar_phi p_pop dz               [:2132]
    Sigma_phi(h) = sum_{g in catalogue, z_g < z_max(h)} w_g S_bar_phi(z_g;h)  [:2961-2968]
    Sigma_4D(h)  = sum_g w_g S_4D(d_L(z_g;h), M_g (1+z_g))                   [same function, with_bh_mass=True; :2828-2835 point mass]
    n_hat_w = Sigma_phi / beta_G_phi;  r_Malm = Sigma_4D / Sigma_phi;
    alpha_G_phi = Sigma_4D / n_hat_w = beta_G_phi r_Malm;
    D_tilde_phi = alpha_G_phi + beta_Gbar_phi;  w_tilde_G = alpha_G_phi / D_tilde_phi;
    D_phi = beta_G_phi + beta_Gbar_phi ("reported alongside")                 [:2493-2498]
    2D: p_i = ( alpha_G_phi L_cat,i^2D + B_i g_i ) / D_tilde_phi              [:6274-6276]

h-dependence of the ingredients:
- p_pop(z;h) = (dVc/dz)/(1+z) with dVc/dz = d_com^2 c / H(z), proportional to h^(−3) at fixed z
  (physical_relations.py:571-584; dark_siren_injection.py:177-194). The same h^(−3) sits in B_i (its dVc/dz
  at :5920), in beta_G_phi, beta_Gbar_phi, D_phi and D_tilde_phi. It is ABSENT from L_cat,i (a ratio of two
  catalogue sums over fixed galaxies) and from Sigma_phi, Sigma_4D, r_Malm.
- f_bar(z;h) and f_k(z;h) are exactly h-independent: M_star carries +5 log10 h which cancels the
  −5 log10 h of the distance modulus (pixel_completeness.py:219-223 and the class note at :160-163).
- S_bar_phi(z;h) and S_4D(d_L(z;h), M_z) depend on h only through d_L(z;h) = D(z)/h; they rise with h
  (smaller distances at fixed z).
- N_GW(d|z;h): a Gaussian in (phi, theta, d_L/d_hat) with the event's Fisher covariance; the sky block is
  h-independent; in the distance block d ln N / dh = (d_L(z;h) − d_hat) d_L(z;h) / (sigma_dL^2 h)
  = u (1 + eps u) / (h eps) with u = (d_L(z;h) − d_hat)/sigma_dL and eps = sigma_dL/d_hat.
- The candidate ball is h-invariant: z-window from d_hat +/- 2 sigma_dL over the full h prior
  [0.6, 0.86] (:5310-5318 sigma_multiplier=2.0; cosmological_model.py:388-389), sky cone radius
  1.5 sqrt(lambda_max) of the sky Fisher block (:5332 with self._sky_cone_k = 1.5, :3734; handler.py:574-595).

## 2. Question (1): d ln w_tilde_G / dh and d ln beta_G_phi / dh at h = 0.73 from the code, against the measured s_beta

### 2.1 What the split script differentiated

b4_imp_stage1_split.py (fanout1_20260829, lines 46-52) forms beta(h) = alpha_G_phi / r_Malm from the
diagnostics columns and takes the secant of ln beta over 0.725/0.735. By :2495-2496,
alpha_G_phi / r_Malm = (Sigma_4D beta_G_phi / Sigma_phi) / (Sigma_4D / Sigma_phi) = beta_G_phi exactly.
So the measured s_beta is d ln beta_G_phi / dh, the h-slope of the MASS-BLIND in-catalogue selection
integral — not the h-slope of w_tilde_G, and not a "mixture weight" in the sense of a class probability
(the 1D class probability is beta_G_phi / D_tilde_phi; see section 4).

### 2.2 Table verification (selection_tables_h_*.json of the FT arm; identical to production, section 2.5)

Source: results/campaign51_20260728/realistic_20260729/p3_work/ft_900101_work/seed900101/selection_tables_h_0_725.json,
_0_73.json, _0_735.json (written by write_selection_table_json, :2605-2670; run of 2026-08-23 at 53b7831e).
Secant slope := [ln X(0.735) − ln X(0.725)] / 0.01 (the split script's own stencil).

| object X | X(0.725) | X(0.73) | X(0.735) | secant d ln X/dh (per unit h) |
|---|---|---|---|---|
| beta_G_phi | 155871370.548 | 153322758.616 | 150827919.040 | −3.28915 |
| beta_Gbar_phi | 893324861.108 | 888403798.071 | 883510508.796 | −1.10471 |
| Sigma_phi | 978309834.499 | 980867125.674 | 983391311.104 | +0.51807 |
| Sigma_4D | 374392995.326 | 375452610.321 | 376507626.971 | +0.56323 |
| r_Malm | 0.38269368 | 0.38277622 | 0.38286654 | +0.04516 |
| alpha_G_phi = beta_G r_Malm (ARITH) | 59650935.6 | 58688305.9 | 57746961.0 | −3.24400 |
| D_tilde_phi (ARITH) | 952975796.7 | 947092104.0 | 941257509.8 | −1.23728 |
| D_phi (ARITH) | 1049196231.7 | 1041726556.7 | 1034338427.8 | −1.42624 |
| w_tilde_G = alpha/D_tilde (ARITH) | 0.062594 | 0.061967 | 0.061351 | −2.00671 |
| beta_G/D_tilde (the coded 1D catalogue class weight) | 0.163563 | 0.161888 | 0.160241 | −2.05187 |
| beta_G/D_phi | 0.148563 | 0.147181 | 0.145821 | −1.86292 |
| beta_Gbar/D_tilde (the coded 1D completion class weight) | 0.937406 | 0.938033 | 0.938649 | +0.13257 |
| Z = D_phi/D_tilde (ARITH) | 1.100968 | 1.099921 | 1.098890 | −0.18895 |
| n_hat_w = Sigma_phi/beta_G (ARITH) | 6.27640 | 6.39740 | 6.51995 | +3.80722 |
| −3 ln h (the volume factor) | | | | −4.10965 (−3/0.73 = −4.10959) |

Checks against the record: s_beta table −3.28915 versus the split script's −3.2891472031202764 (all 12 seeds,
both arms, spread 0.0; b4_imp_stage1_split.json fleet.s_beta_global); s_D table −1.23728 versus the
measured −1.2372867562 (same file); w_tilde_G(0.73) = 0.0619668 versus the log line
"path-A(h=0.7300): w_tilde_G=0.06196684 ... n_hat_w_phi=6.397401 ... D_tilde_phi=9.470921e+08"
(p3_work/ft_900101.log) and the claim card's "w_tilde_G(0.73) = 0.0620" (C4). Everything the code
computes reproduces the measured s_beta and s_D; the derivation's content is in what the numbers MEAN.

### 2.3 Decomposition of s_beta

d ln beta_G_phi/dh = −3/h + <d ln S_bar_phi/dh>_{f S p}, the second term being the integrand-weighted mean
survival slope of the model's selected in-catalogue population (f is h-free, section 1). Numerically
(ARITH): sigma_G := s_beta + 4.10965 = +0.8205 per unit h. Likewise sigma_Gbar := s_Gbar + 4.10965 =
+3.0049 (the dark population sits where S_bar_phi is still rising steeply with h; the catalogue population
sits at low z where it is nearly saturated). And sigma_cat := d ln Sigma_phi/dh = +0.5181 is the
w-weighted mean survival slope of the ACTUAL catalogue.

Physical reading of n_hat_w = Sigma_phi / beta_G_phi (slope +3.807 = +3/h − (sigma_G − sigma_cat)): it is
the catalogue's selected rate weight per unit of the model's selected in-catalogue volume; the h^(+3)
is the comoving volume of a fixed set of galaxies shrinking with h, which is correct physics and is
exactly what makes beta_G_phi L_cat carry the same h^(−3) as B_i and D_tilde_phi. The volume factor is
therefore a COMMON factor of all three terms of p_i and cancels identically in every per-event score
and in every ratio p_i(h)/p_i(h').

### 2.4 The 63/37 split is a bookkeeping artefact; the invariant split

The first-order identity of C3 is s_imp,i = c_i (s_beta + s_L,i − s_B,i) with c_i = beta_G L/(beta_G L + B).
Write s_beta = −3/h + sigma_G and s_B,i = −3/h + s'_B,i (B_i carries the same dVc/dz), while s_L,i has no
volume factor. Then

    s_imp,i = c_i [ sigma_G + s_L,i − s'_B,i ]                                                 (I)

and the −3/h that C3 booked into the "global" term is cancelled by the −3/h it booked into the "per-event"
term. Re-booking the fleet split of b4_imp_stage1_split.json (ft, 12 seeds; mean_c_all = 0.04153 =
−0.13659/−3.2891, ARITH):
- global term as booked: mean(c) s_beta = −0.1366 (62.7 percent);   re-booked: mean(c) sigma_G = +0.0341 (−15.7 percent, wrong sign);
- per-event term as booked: mean c (s_L − s_B) = −0.0812 (37.3 percent);   re-booked: mean c (s_L − s'_B) = −0.2519 (115.7 percent);
- sum −0.2178 in both bookkeepings (the measured exact −0.21778).

Splitting s_L,i itself: s_L,i = <d ln N_g/dh + d ln S_bar_phi(z_g)/dh>_W − sigma_cat with W_g = w_g S_bar_g N_g
the catalogue-leg weight of candidate g. Hence the invariant form

    s_imp,i = c_i [ (sigma_G − sigma_cat) + Delta_N,i + Delta_S,i ]                                (II)

    (sigma_G − sigma_cat) = +0.3024 per unit h   (global: the non-volume slope of the conversion factor beta_G/Sigma_phi, ARITH from the table)
    Delta_N,i = <d ln N_g/dh>_W − <d ln N/dh>_B   (per event: the GW-likelihood slope averaged over the ball's weight minus over the completion integrand)
    Delta_S,i = <d ln S_bar/dh>_W − <d ln S_bar/dh>_B   (per event: survival slopes; both about 0 at q1 redshifts where S_bar is saturated)

The only global mixture-weight object in the impostor score is (sigma_G − sigma_cat) = +0.30 per unit h: it
would vanish exactly if the catalogue's selected z-distribution matched the model's f S p_pop, and its
actual value pushes the score UP by mean(c) x 0.30 = +0.013 per event (6 percent of |s_imp|). Under the FT
arm's own divisor (Sigma_3D rather than Sigma_phi; the FC/FT runs predate e35ea018, claim card section 4
item 4) the global term is sigma_G − d ln Sigma_3D/dh; the HEAD-basis KW-Q1 run gives S(1) = −1.0205 on the
same frozen 191 q1 events on which the FT basis gives −0.9028 (b4_imp_stage1_events.csv, seeds 900101-900104,
z_true < 0.358; ARITH), a −0.118 difference = mean(c_q1) x (−0.71): the HEAD divisor makes the global term
about 0.7 per unit h more negative than the FT one, still a 10-15 percent object on q1 and the wrong order
of magnitude for the −0.9.

VERDICT on question (1): the code reproduces −3.2891 per unit h identically (it is the same table); the
model-consistent value of the global mixture-weight slope in the impostor score is +0.30 per unit h (FT
basis) to about −0.4 (HEAD basis), i.e. at most 15 percent of the q1 score and mostly the wrong sign; the
claim card's candidate (b) is REFUTED as a mechanism. The A14 falsifier of the "63 percent" attribution is
discharged at the derivation level (section 8.1).

### 2.5 Production tables

headreadout_20260827/iiib/event_likelihoods.csv (row #213, d04d9dc9) carries alpha_G_phi, r_Malm and
D_tilde_phi per row; at h = 0.725/0.73/0.735 they are 59650990 / 58688310 / 57746960, 0.3826937 / 0.3827762
/ 0.3828665 and 952975900 / 947092100 / 941257500 — the mirror's tables to the 7 s.f. the CSV stores
(the selection objects are catalogue-plus-pool objects shared by both venues). Every slope of section 2.2
therefore holds on production: d ln Z/dh = −0.18895, d ln(beta_Gbar/D_tilde)/dh = +0.13257,
w_tilde_G = 0.0619668.

## 3. Question (2): the expected impostor score of a q1 event as a function of its catalogue share and candidate count

### 3.1 Dense-ball (continuum) limit of the catalogue leg

For a ball dense enough that the sum over candidates samples the catalogue's intensity, the numerator sum
tends to its expectation. With listed z_g = noisy observation of a true z drawn from the model's catalogued
intensity lambda(z) = f(z) w_pop(z) w, the C7-core kernel is the Bayes posterior p(z | z_g) whose normaliser
Z_g (:7682, the quadrature of Gauss x f x w_pop) is the model's LISTED density at z_g; the identity
integral n_obs(z_g) p(z | z_g) dz_g = lambda(z) then makes

    sum_g w_g S_bar(z_g) N_g(d)  ->  n_hat_w integral f(z) S_bar(z) w_pop(z) N_GW(d | z) dz   (in the units of :5727-5733)

with f counted ONCE (no double count from the kernel prior; the per-galaxy normaliser cancels the listed
density's tilt). So in this limit beta_G L_cat,i -> A_i = integral f S_bar w_pop N dz and the mixture is
p_i = (A_i + B_i)/D_tilde with B_i = integral (1 − f) S_bar w_pop N dz: the two legs partition the same
population integral by f versus (1 − f), and the responsibility c_i = A_i/(A_i + B_i) -> f(z_true) for a
narrow GW window. This is the model's own statement that, from the GW data alone, an event at z_true is
in-catalogue with probability f(z_true).

### 3.2 The per-event slope difference in closed form

Inside the GW window the two integrands differ by the factor f/(1 − f), i.e. by the local tilt
lambda_A − lambda_B = d ln f/dz − d ln(1 − f)/dz = (d ln f/dz)/(1 − f). For a Gaussian likelihood in u
tilted by exp(lambda z) = exp(lambda sigma_z u) the weight-averaged offset is <u> = lambda sigma_z with
sigma_z = sigma_dL / (d d_L/dz) = eps d_hat / (d d_L/dz). Using d ln N/dh = u(1 + eps u)/(h eps) from
section 1,

    Delta_N = (<u>_A − <u>_B)/(h eps) + eps(<u^2>_A − <u^2>_B)/(h eps)
            = [(d ln f/dz)/(1 − f)] sigma_z/(h eps) + O(eps)
            = [(d ln f/dz)/(1 − f)] z_eff / h,      z_eff(z) := d_L / (d d_L/dz)                    (III)

eps has cancelled: the depth skew is set by the population tilt across the GW window in redshift units,
and the window's width in redshift is itself proportional to eps. z_eff (ARITH, flat LCDM Omega_m = 0.2726):
0.138 at z = 0.15, 0.21 at 0.25, 0.258 at 0.30, 0.334 at 0.40.

With c -> f(z_true) the expected impostor score of a dark event at z_true is

    E[s_imp | z_true] = f (d ln f/dz) z_eff / (h (1 − f)) = (df/dz) z_eff / (h (1 − f))              (IV)

plus the small global term c (sigma_G − sigma_cat) and the survival term c Delta_S (about 0 below
z = 0.4 where S_bar is saturated; negative at higher z where the ball's S_bar slope is smaller than the
completion integrand's — a second-order contribution to the q2 tail).

Properties of (IV), each matched to a banked fact:
- sign: df/dz < 0 everywhere (GLADE+ completeness falls with z) so E[s_imp | z] < 0 for every dark event — the drag is a MODEL-CONSISTENT expectation for the dark class (its balance is the subject of section 4);
- localisation: (IV) peaks where |df/dz| is largest and vanishes where f -> 0; the banked z-quartile shares 91.7 / 7.6 / 0.7 / 0.0 percent (b4_imp_stage1_forecast.json, covariates.ft.z_true) and the KW-Q1 92.25 percent are this;
- no eps: SNR carries nothing — eta^2 = 0.009, r = −0.005 (same file, covariates.ft.SNR);
- no sigma_g at first order: the kernel width enters neither the tilt nor c in the dense-ball limit — KW-Q1 R = +0.0848 INERT, with the residual small positive R being the second-order kernel-prior shift (the prior's tilt times sigma_g^2 lifts each kernel mean; at z > 0.2 the f-fall dominates the volume rise so the shift is downward and a wider kernel moves it further — E15 of the forensic reports +0.91 sigma_k at the lowest z_g bin flipping to −0.71 at the highest, B1_1_S0A_DEFECT_FORENSIC_20260829.md:90);
- catalogue share: s_imp is proportional to c at fixed z, giving the eta^2 = 0.384, r = −0.76 association with share_073 (same file).

### 3.3 Finite ball: the candidate-count dependence

With A_i a discrete (Poisson-like) sum, s_imp,i = (d_h A_i − A_i s_B,i)/(A_i + B_i) has a linear numerator
(unbiased for its expectation) over a fluctuating denominator. Expanding in A/B for the small-c regime,
E[s_imp] = s_imp^(cont) − Cov(d_h A − A s_B, A)/B^2 + O(c^3); the covariance is the single-galaxy variance
term integral lambda a_g^2 (d ln a_g/dh − s_B) with a_g proportional to N_g, so it is weighted toward the
window centre where (d ln N/dh − s_B) is dominated by the tilt (negative) — the correction is POSITIVE and
scales as 1/n_eff (the number of candidates inside the GW window). Consequences: (i) at fixed z, |s_imp|
grows with candidate count toward the continuum value (IV); (ii) events whose ball is dominated by one
close candidate have high c and a mild slope, which is the banked anticorrelation corr(log c, Delta) = +0.54
over the 1427 active FT events (b4_imp_stage1_events.csv, ARITH; Delta := s_imp/c); (iii) the C2 candidate-count
quartile trend (−0.043 / −0.294 / −0.491 for the three non-empty quartiles) has the predicted direction but
is z-confounded (low z means both dense and tilted), so the within-z-bin dependence is left to T2.2 (section 6.5).

### 3.4 Do the sign and size of the measured −0.80 follow? (zero-evaluate check on the banked per-event table)

b4_imp_stage1_events.csv (ft arm, 2152 events, 12 seeds; columns z_true, share_073 = c_i(0.73), s_imp, s_pure,
n_cand_no_bh; ARITH by z bin; seed SEM from the 12 per-seed means):

| z_true bin | n | mean c | mean s_imp (seed SEM) | c-weighted Delta = sum s_imp / sum c (active) | mean s_pure | median n_cand |
|---|---|---|---|---|---|---|
| [0, 0.15) | 30 | 0.654 | −1.611 (0.94) | −2.46 | +2.60 | 54 |
| [0.15, 0.25) | 142 | 0.283 | −1.210 (0.092) | −4.28 | +1.60 | 101 |
| [0.25, 0.358) | 368 | 0.072 | −0.570 (0.029) | −7.92 | +0.83 | 155 |
| [0.358, 0.459) | 533 | 0.0054 | −0.066 (0.0054) | −12.3 | +0.15 | 141 |
| [0.459, 0.584) | 541 | 0.0004 | −0.0062 | −17.7 | −0.32 | 30 |
| [0.584, ...) | 538 | 0.0000 | −0.0000 | −9.1 | −0.68 | 0 |

q1 as a whole: n = 540, mean s_imp = −0.796 (the card's −0.798 +/- 0.041), mean c = 0.160, median c = 0.066,
c-weighted Delta = −4.98; active q1 (425): mean c = 0.203, mean s_imp = −1.012, unweighted mean Delta = −11.5.

Using c(z) as a proxy for f(z) in (III)-(IV) (c is a LOWER bound on f: the sky cone captures only a fraction
kappa of the catalogue leg's mass, section 5.3, and the finite ball lowers c further), the log-slope
d ln c/dz between adjacent bins is −8.4 (z 0.1 to 0.2), −13.7 (0.2 to 0.3), −25.9 (0.3 to 0.4), so
(III) predicts Delta = −1.6, −3.9, −10.6 at z = 0.15, 0.25, 0.35 against the c-weighted −2.5 / −4.3 / −7.9
(bins centred near 0.10, 0.20, 0.30) and −12.3 (0.36-0.46) — sign and z-trend reproduced, magnitudes within
a factor 0.6-1.6 without any fit. For the expectation (IV): (dc/dz) z_eff/h = −0.86, −0.61, −0.27 against the
measured bin means −1.61, −1.21, −0.57 — a uniform factor about 2, of which 1/(1 − f) (1.5 at f about 0.35
in the second bin, 1.1 in the third) and 1/kappa (1.15-1.5) account for most. VERDICT on question (2): the
sign, the z-localisation, the catalogue-share scaling and the order of magnitude of the −0.80 per-event q1
score follow from (IV) with no free parameter; the exact curve needs the estimator's own f_bar(z) tabulated
(a zero-compute read; requested from T2.2 as a by-product, section 6.2), and the residual factor is the
registered F3 prediction of section 7.2 with its A14 falsifier in section 8.2.

## 4. What the negative dark-class score means: the normalisation identity, the composition mismatch and the Z(h) defect

### 4.1 The identity the estimator's own model must satisfy

Write the 1D mixture as p_hat = a p_G + b p_Gbar with a = beta_G/D_tilde, b = beta_Gbar/D_tilde and p_G, p_Gbar
the class-conditional densities normalised over the data (integral L_cat dd = 1 over the full catalogue since
integral N_g dd = 1 and sum_g w_g S_bar_g = Sigma_phi; integral B dd = beta_Gbar by the same token). For data
drawn from the estimator's own class-conditional laws, E_G[d ln p_G/dh] = E_Gbar[d ln p_Gbar/dh] = 0, and with
s_comp := d ln p_hat/dh − d ln(a p_G)/dh (the completion leg's contribution to a catalogue-class event's
score, the mirror image of s_imp) one gets, for ANY a(h), b(h):

    a E_G[s_comp] + b E_Gbar[s_imp] = 0                                                              (V)
    E_Gbar[score] = E_Gbar[s_imp] + d ln b/dh,   E_G[score] = E_G[s_comp] + d ln a/dh
    a E_G[score] + b E_Gbar[score] = d(a + b)/dh = Z d ln Z/dh,   Z := a + b = D_phi / D_tilde        (VI)

(V) says the dark class's negative impostor score is REQUIRED to be balanced by the catalogue class's
positive completion-leg score in the ratio of the mixture weights — a negative E_Gbar[s_imp] is not a
violation. (VI) says the total score at the ESTIMATOR's own composition vanishes only if the weights sum to
one at every h; here they sum to Z(h) = 1.0999 with d ln Z/dh = −0.189 per unit h (section 2.2).

### 4.2 The banked class scores satisfy (VI) at the estimator's composition — the estimator is internally consistent

Inputs: E_Gbar[score] = −0.1470 (ft mean s_full, b4_imp_stage1_events.csv; = s_imp −0.2178 + s_pure +0.0708);
E_G[score] = −0.1238 +/- 0.0527 (b0i, claim card C4). ARITH: a E_G + b E_Gbar = 0.1619 x (−0.1238) + 0.9380 x
(−0.1470) = −0.1579; required Z d ln Z/dh = 1.0999 x (−0.18895) = −0.2078 (or −0.189 if one uses the normalised
weights); the gap is +0.03 to +0.05 against a combined seed SEM of about 0.017 (0.938 x 0.0158 and 0.162 x
0.053 in quadrature), with the disclosed basis mismatch (b0i on the HEAD Sigma_phi basis, FT on Sigma_3D;
E_Gbar's pure part +0.0708 +/- 0.025 against its model value +0.1326, section 4.4). Reading: the two class
scores are consistent with the estimator's own normalisation identity within 2-3 sigma. The claim card's C4
composite −0.146 at the GENERATOR's composition (0.062 / 0.938) is therefore not a class-conditional-law
violation; it is the estimator's Z(h) tilt plus the composition mismatch of section 4.3.

### 4.3 The composition mismatch and the Z(h) defect (the structural finding of this derivation)

The generator detects an in-catalogue host with its OWN catalogue mass (draw_rate_weighted_hosts carries M
from the catalogue row, handler.py:1024-1080; the EMRI SNR depends on M), so the detected population's
class odds are mass-aware, alpha_G/beta_Gbar = 0.0661 (w_tilde_G = 0.0620). This is what the code's own
monitored gate (ii) predicts (:2589-2598): "predicted in-cat share = w_tilde_G (SNR-only) -> rescore(w_tilde_G)
(S_and, rho = 0.7305)"; ARITH 0.0619668 x 0.7305 = 0.0453 against the realised production share
76/1588 = 0.0479 (claim card C5). The 1D channel's catalogue weight, however, is the mass-blind beta_G with
class odds beta_G/beta_Gbar = 0.1726 (x 0.7305 = 0.126 after the p0 window): the 1D mixture treats the
catalogue class as 2.61 = 1/r_Malm times more probable than the generator makes it. Two consequences:

(i) the un-normalised 1D likelihood: integral p_hat dd = (beta_G + beta_Gbar)/D_tilde = Z(h) = 1.0999, with
d ln Z/dh = −0.189 per unit h per event. A likelihood integrating to an h-dependent Z(h) is equivalent to a
normalised likelihood times an un-derived prior Z(h)^N — the same object class as the B_scale multiplier
retired in row #131 (docs/derivations/bscale_completion_normalization.md section 6). On the 1588-event
production fleet Z(h)^N tilts the 1D log-posterior by N d ln Z/dh = −300 nats per unit h (−273 nats after
dividing by Z, (VI)); against the 1D information scale I_1D = 1/0.0277^2 = 1303 (claim card C5, the dark-only
pure arm's sigma_h) that alone is a −0.21 to −0.23 shift, i.e. the floor. The 2D channel has Z = 1
identically ((alpha_G + beta_Gbar)/D_tilde); the two channels differ precisely here — which is consistent
with the 2D channel un-railing (0.665) while the 1D rails.

Provenance of the convention, and the ratified rows it touches. fixb_pathA_phi_marginal_selection.md:67-86
defines D_phi = beta_G + beta_Gbar as "the 1D channel's full-volume normalisation re-derived in the same
convention" and then assembles the 1D with D_tilde_phi; bscale_completion_normalization.md:44-46 states
"D = D_tilde_phi = alpha_G_phi + beta_Gbar_phi (the catalogue leg deliberately carries the Malmquist-aware
alpha_G_phi = beta_G_phi r_Malm; that design choice is not under examination here)". The alpha-pairing fork of
rows #162-#169 then ratified, as Appendix B (i)/(ii) (row #169, author verbatim "Ratify B, run fused re-measure
+ b0 test"), "D_tilde stays" and "beta_G_phi stays" — AGAINST the alternative then on the table, the R-rescale
R = beta_G(legacy S_3D)/beta_G_phi, refuted as an un-derived B_scale-class multiplier
(A20_REVIEW_APPENDIX_A_20260822.md:10-18; row #168). The display that justified "D_tilde stays"
(PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md:124-133) writes the denominator as "the TOTAL SELECTED MASS,
integral f_k lambda S_bar_phi + integral (1 − f_bar) lambda S_bar_phi — both class terms S_bar-weighted. That
is D_tilde_phi = alpha_G_phi + beta_Gbar_phi as coded": the first expression is D_phi (both terms carry the
SAME S_bar_phi), the second is the coded D_tilde (the catalogue term carries S_4D through r_Malm); the two are
equal only at r_Malm = 1 and differ at the coded r_Malm = 0.383 by exactly Z(h). The identification was made
by the review's own criterion ("a hybrid-mass denominator no model produces") without noticing that the CODED
pairing — mass-blind numerator beta_G_phi, mass-aware denominator D_tilde — is itself a hybrid no model
produces. This derivation does not re-open the R-rescale (it stays refuted: R is a free multiplier) and does
not dispute the twin (row #197); it identifies a different inconsistency with two derivable resolutions:
mass-blind everywhere (beta_G_phi numerator with D_phi divisor: Z = 1, 1D class share 0.147, inconsistent with
the generator's mass-aware detection of catalogue hosts) or mass-aware everywhere (alpha_G_phi with per-galaxy
S_4D and D_tilde: Z = 1, class share 0.062, consistent with the generator and identical in form to the 2D
assembly). The examination the bscale memo declined is this section: the 1D numerator's per-galaxy survival
S_bar_phi(z_g) is NOT the detection model D_tilde uses for those same galaxies (S_4D(z_g, M_g)); MFG (2019) A2
("selection alpha must use the SAME population and detection model as every numerator", the code's own
reference at :1804-1808) is violated in the 1D channel by exactly r_Malm. Because remedy (d) of section 5.4
modifies the row #169 (i)/(ii) ratified pairing, it is NOT executed under the row #255 standing grant: its gate
presentation may be prepared, but the flip returns to the author as a fresh [RULE] with this section as its
input. Exoneration grep (mechanism vocabulary "mass-blind", "class share", "in-cat share", "integrates to",
"Z(h)", "over-normalis", "D_phi", "w_tilde_G", "alpha-pairing") over EXONERATION_REGISTER_20260827.md and the
ledger: no hit names this object; rows #162-#169 examined the alpha-pairing fork with horns (per-candidate
S_bar_phi twin vs global) and (beta_G_phi vs legacy beta_G) — neither horn is the mass-aware pairing; the
register keeps the mixture-weight calibration LIVE (C9, Gate C item 1; the claim card's section 0.2 row 15) —
this derivation supplies its mechanism. Related, not identical: row #209's "class-G S_bar_phi double-weight
REAL (13.5-16 percent)" found by the [P3-2D] identity test on the 2D side.

(ii) the composition mismatch at the generator's composition. From (VI) with the generator's realised
shares pi^g = (0.0479, 0.9521) against the estimator's normalised (0.1472, 0.8528) (ARITH):
Sigma_score/N = [a E_G + b E_Gbar]/Z + (pi_G^g − 0.1472)(E_G − E_Gbar). On production, with C5's per-class
impostor scores (dark −0.193, in-catalogue −1.707) and the pure-arm reads of section 4.4 (dark −0.014 per
event, in-catalogue +2.3), E_Gbar = −0.207, E_G = +0.59: the three pieces are −273 (Z tilt), −126
(composition: (0.0479 − 0.1472) x 0.797 x 1588) and +130 (the class-law deviations of section 4.4), total
−269 nats per unit h against the direct read Sigma s_full = −298 (headreadout_20260827/iiib/event_likelihoods.csv,
ARITH over 1588 rows, secant 0.725/0.735; the 10 percent difference is the assumption-join and C5's event
selection). Linearly, −269/1303 = −0.21: the floor. The rail is owned, in this accounting, first by the Z(h)
defect, second by the 2.6x over-weighting of the catalogue class — both the SAME object (the mass-blind
beta_G in the 1D numerator against the mass-aware D_tilde) — and only third by anything class-conditional.

### 4.4 Out-of-scope finding, recorded for B8 [CAL]: the dark-class completion leg on production

The pure arm B/D_tilde on a dark-only subset carries the model's composition tilt d ln(beta_Gbar/D_tilde)/dh
= +0.1326 per event (row #150 item 1: "the tilt is measured at −0.133/h per event (ln D_tilde/beta_Gbar)";
table, section 2.2). Mirror FT: measured dark s_pure = +0.0706 +/- 0.025 (12-seed SEM; b4_imp_stage1_events.csv)
— a matched-channel residual E_Gbar[d ln(B/beta_Gbar)/dh] = −0.062 +/- 0.025 (2.5 sigma; cf. O8's matched
+0.006 in h, row #165, near-closed). Production iiib: the dark-only pure arm sits at 0.7134 with sigma 0.0277
(C5), i.e. a total dark pure score of about −22 nats per unit h (−0.014 per event), where the model's
composition tilt alone would put it at +0.133 x 1514 = +201: the production dark-class completion leg scores
about −0.15 per event BELOW the estimator's own model. The C5 reading "dark-only pure arm covers truth" is
therefore a cancellation, not a closure. Candidates (not adjudicated here; not B4's object): the noiseless
data law (d_hat = d_L(z_true) exactly on both venues while the estimator's law is Fisher-smeared; the
second-order term is of order eps^2 times the log-slope curvature and production's low-SNR tail has eps up to
0.3), the dark mass law entering S_bar_phi versus the generator's, the analysis-depth cap. This residual
bounds what ANY impostor-leg remedy can do on production (section 7).

## 5. Question (3): the candidate remedies and what each requires

### 5.1 (a) A per-event mixture weight — INCONSISTENT with the D_tilde derivation; REFUTED at the derivation level

In the absolute_marginal derivation (:5727-5733) beta_G L_cat = A_i = sum_ball w_g S_g N_g / n_hat_w is a
per-volume intensity; the "weight" beta_G/Sigma_phi is the per-galaxy-to-p_pop units conversion n_hat_w^(−1),
and the completeness already enters twice per event through the data — in the realised catalogue density
inside the ball (and its kernel prior f_k, :7677-7681) and in the (1 − f_k) of B_i. The per-event
catalogue-vs-completion responsibility c_i = A_i/(A_i + B_i) is what the mixture computes; a per-event weight
multiplying A_i by a function of the event (f(z_hat), the ball's local density, ...) counts f a second time
and has no normaliser to pair with. The only per-event normalisation that has ever been derived is the local
ratio-of-sums (:5738-5745), which is scale-inconsistent and rails (issue #30). The D_tilde derivation
(bscale memo section 1: "ONE common denominator D(h) = integral P_det p_pop over BOTH classes") has no
per-event slot; [NUMERATOR-ONLY-CLEAN] and [WPOP-TUNING] bind (claim card section 0.2 rows 3 and 7).
Nothing to run.

### 5.2 (b) The impostor-weight family, row #167 fork (A11 — NOT covered by row #255; both branches stated, no choice made here)

Registered form (PREREGISTRATION_P3_TWIN_20260822.md:428): cat_term_completed(e,h) = cat_term_phi(e,h) R(h),
R(h) = beta_G(h)/beta_G_phi(h) — a GLOBAL, h-dependent multiplier on the catalogue leg.
- Branch b1 (numerator only; banked, row #167 item 1): Delta mean_h = −0.002810 +/- 0.000467, COMPLETED-SMALL.
- Branch b2 (D_tilde completed as well; reviewer arm, REPORTED-ONLY, row #167 item 2): +0.034357 +/- 0.004342;
  the D_tilde lever alone +0.042362 +/- 0.005033.
Bearing of this derivation: R(h) enters (II) only through the global term (its slope d ln R/dh, weighted by
mean(c) = 0.04) and through the LEVEL of c_i (R(0.73) rescales every A_i); it cannot change Delta_N,i, the
per-event depth skew that owns the score. The two branches differ by a Z-type global normalisation tilt
(b2 changes a + b and its h-slope; b1 does not), which is why the fork is a 15x lever: it is a fork about the
same object as section 4.3 (which global normalisation makes integral p_hat dd equal to 1 at every h), and
the answer to THAT question is fixed by normalisation, not by a convention choice — under the criterion
integral p_hat dd = 1 neither branch as registered satisfies it, because both keep the mass-blind
beta_G_phi pairing. This is offered to the author as the input to A11, not as a ruling: A11 stays a fresh
[RULE] (row #255 wording).

### 5.3 (c) The enlarged ball (sky 3 sigma, z +/- 4 sigma_g; median candidates 278 -> 1729) — the falsifier arm, not a remedy

What it changes: the sky cone at 1.5 sqrt(lambda_max) captures a fraction kappa of the catalogue leg's
expected mass, kappa = 1 − exp(−1.125) = 0.675 for a round error ellipse and up to about 0.87 for an
elongated one (radius set by the LARGEST eigenvalue, handler.py:574-578); kappa is h-independent, so it
lowers every c_i by kappa without touching Delta_N,i (in the continuum). The z-window at truth is +/- 4-5
sigma_GW (the +/- 2 sigma_dL over the h prior [0.6, 0.86] maps to [0.71, 1.34] z_true at eps = 0.07; ARITH)
and clips only at the grid edges; z +/- 4 sigma_g recovers kernel tails of candidates listed just outside it.
What the derivation predicts: c_i rises toward c_i/kappa (x 1.15-1.5), the added candidates carry the same
Malmquist tilt, so the dark-class impostor score becomes MORE negative by the same factor — q1 HEAD-basis
S(1) from −1.02 to about −1.2 to −1.5; the FT fleet mean shifts DOWN by about 0.02-0.05; the in-catalogue
class loses true-host share to the added impostors and its s_imp becomes more negative too. On the
mirror b0i the forensic's E9/E12/E14 show the catalogue-leg THETA-secant flipping sign under enlargement
(B1_1_S0A_DEFECT_FORENSIC_20260829.md:84, :87, :89) — that is the s-axis response, not the h-score; the
derivation makes no claim about it. What it tests: whether the impostor score is a ball-truncation object.
Cost: 3-6x a normal cell (docket 2 section 7 rank 2); 4 seeds x 3 h-nodes x 4.5 x 0.2843 CPU-h = about
15 CPU-h local (12 seeds: about 46). Registration is in section 8.2 (it IS the A14 falsifier of the
depth-skew attribution). It shares the sky-cone-radius flag with T1 (self._sky_cone_k, :3734, :5332 —
already a flag at HEAD).

### 5.4 (d) NEW — the remedy the derivation points at: the mass-aware 1D catalogue leg

Replace, in the 1D channel only, the three mass-blind objects by their mass-aware twins that the code
already computes for the 2D channel:

    L_cat,i^(1D,4D) = sum_{g in ball_i} w_g S_4D(d_L(z_g;h), M_g (1+z_g)) N_g^(3D)(d_i | h) / Sigma_4D(h)
    p_i^(1D) = ( alpha_G_phi L_cat,i^(1D,4D) + B_i ) / D_tilde_phi

i.e. (1) per-candidate S_4D(z_g, M_g; h) — the SAME point query Sigma_4D makes (:2828-2835, "point") — in place
of S_bar_phi(z_g; h) at :7833/:7838-7841 for the no-BH numerator; (2) global_denom_with_bh (Sigma_4D, already
in hand at :5694) in place of Sigma_phi at :5771; (3) alpha_G_phi in place of beta_G_phi at :6272. Then
integral p_i dd = (alpha_G + beta_Gbar)/D_tilde = 1 identically (Z = 1), the 1D class odds equal the
generator's alpha_G/beta_Gbar, MFG A2 holds in the 1D channel, and the 1D assembly is the exact no-mass-
likelihood image of the 2D assembly (alpha_G L^2D + B g)/D_tilde. beta_G L^1D reduces to sum w S_bar N/n_hat_w
and alpha_G L^(1D,4D) to sum w S_4D N/n_hat_w: the SAME conversion n_hat_w (mass-blind, as the Path-A
package wanted), with each galaxy weighted by its own detectability instead of the population-average one.
Mass uncertainty: the point evaluation matches Sigma_4D's production convention; the kernel form of
instrument J (:2828-2835) is the paired alternative if Sigma_4D is ever switched.

Not the refuted R-rescale: R = beta_G(legacy)/beta_G_phi was a free global multiplier (row #168); (d) multiplies
each galaxy by its own S_4D/S_bar_phi and the class weight by r_Malm, which is the Path-A package's own
definition of alpha_G_phi (:2496) — every object in (d) is already derived and already computed for the 2D
channel; nothing new is fitted.

What it requires: a /physics-change gate on bayesian_statistics.py (a formula change in the 1D catalogue
leg: gate presentation before code, ledger rows, end-of-tree verifier). Row #255's standing grant covers the
gate presentation and the instrument cell; the production flip itself modifies the row #169 (i)/(ii) ratified
pairing and is therefore returned to the author as a fresh [RULE] (section 4.3) — not executed under the
grant. The byte-identical fallback is the current "phi" cell
(catalogue_numerator_survival = "phi", catalogue_global_selection = "phi", the beta_G weight). Suggested
cell name: catalogue_numerator_survival = "s4d" with catalogue_global_selection = "4d" and the alpha_G weight,
all three toggled together (an unpaired subset is exactly the class of defect [NUMERATOR-ONLY-CLEAN]
forbids). Before any code: the T2.2 instrumented run serialises S_4D per candidate, so the paired effect on
the 4 FT seeds is a zero-compute rescore (section 6.6) — an A12-style read that either supports or kills the
remedy before the gate is opened.

Relation to the [P3-IMP] twin (row #197): the twin put S_bar_phi(z_g) into the 1D numerator to close the
tower identity r_phi = 1 for the POPULATION; for a catalogue galaxy of KNOWN mass the tower identity is
not the relevant statement — S_4D(z_g, M_g) is. The twin is the population-average approximation to (d);
its calibration residual (TWIN-FUSED-MATERIAL +0.0291 against the coded-leg drag +0.1518, i.e. 19 percent of
the coded drag removed) is consistent with the mass-blind survival having taken out the z-dependence of
S_bar but not the 1/r_Malm over-weight.

## 6. Question (4): the per-candidate instrumented run — design (T2.2; A10 = instrumentation guard, row #255)

### 6.1 Placement: a read-only serialiser in evaluate(), after p_Di returns — no change inside p_Di

Everything the hook needs already exists in memory after p_Di returns for detection index i:
- the candidate objects candidate_hosts / candidate_hosts_with_bh_mass (HostGalaxy: phiS, qS, z, z_error, M,
  M_error, catalog_index; handler.py:66-81), built at :5320-5335;
- the per-host results, retained per event keyed by catalog_index in
  self.posterior_data_with_bh_mass[GALAXY_LIKELIHOODS][i] (zip(catalog_index, results_with_bh_mass), :5589-5595)
  and [ADDITIONAL_GALAXIES_WITHOUT_BH_MASS][i] (:5598-5604): r[0] = the no-BH numerator N_g WITH the S_bar
  factor as used, r[1] = D_g;
- the weights: _rate_weight(host) (:1036), recomputable from host.M, host.z;
- the true-host translation _translated = galaxy_catalog.resolve_host_recovery_position(host_galaxy_index)
  (:5416-5420), already compared against the candidate lists at :5421-5424;
- the detection: self.detection.d_L, d_L_uncertainty, phi, theta, M, host_galaxy_index;
- the tables and objects for diagnostic reads: self._phi_survival_table[h] (S_bar_phi at z_g by np.interp,
  the same read as :2968), detection_probability_obj.detection_probability_with_bh_mass_interpolated
  (S_4D at (d_L(z_g;h), M_g(1+z_g)), the :2828-2835 query), completeness.f_bar / f_k.
The hook writes one row per (event, candidate) to <directory>/per_candidate_h_<label>.csv (the
write_selection_table_json pattern, :2605-2670), behind a flag --per_candidate_dump <dir> whose default
(None) is a single boolean check. It reads; it never writes into any object the likelihood consumes.

### 6.2 What it must serialise

Per candidate: event_idx, h, catalog_index, batch (with_bh / no_bh_only), z_g, z_err_g, M_g, M_err_g,
phiS_g, qS_g, w_g, N_g_used (= r[0], S_bar-inclusive), D_g (= r[1]), S_bar_phi(z_g;h), S_4D(z_g, M_g; h),
u_g = (d_L(z_g;h) − d_hat)/sigma_dL (listed z, point kernel proxy), sky_mahalanobis (from the sky Fisher
block; optional), is_true_host (catalog_index == _translated). Per event (one row per (event, h)): d_hat,
sigma_dL, z_true (from the CRB truth as C2 did), n_cand_no_bh, n_cand_with_bh, f_bar(z_true;h),
f_k(z_true, pixel(event); h), L_cat_no_bh, B_num, D_tilde_phi (copied from the diagnostics row for the
reconstruction gate). Note the KW-Q1 task asked for "per-candidate S_bar_phi and kernel value": the kernel
value the numerator actually integrates is folded into N_g_used; S_bar_phi(z_g) is serialised separately so
N_g_used / S_bar_phi(z_g) recovers the survival-free numerator on the point path (on the quadrature path
S_bar sits inside the integrand and only the product is exact — disclosed).

### 6.3 Byte-identity guard (A10) and instrument gates

- GATE BI: with the flag on, event_likelihoods.csv (all 16 columns, :6311-6340) bit-identical (max |Delta| =
  0.0) to the unhooked run at the same h, and the two posterior JSON files md5-identical — the row #246 C0
  form; a ledger row in docs/gates/PHYSICS-GATE-LEDGER.md ("instrumentation, no computed value changes").
- GATE T-ID (free): the hooked truth node (seed 900101, s = 1, FT config sites2.2_nosmear) must reproduce
  fanout1_20260829/kwq1_registered_run/s0a_seed900101/node_truth_ft_sites2.2_nosmear bit-identically on
  combined_no_bh and L_cat_no_bh (the KW-Q1 parity form).
- GATE R (reconstruction, the instrument-validity gate): per event and h, sum_g w_g N_g_used / Sigma_phi(h)
  must reproduce the diagnostics column L_cat_no_bh to <= 1e-12 relative (Sigma_phi from the run's own
  selection_tables_h_*.json); and the count of serialised candidates must equal the "possible hosts found"
  log line. If R fails the run is INSTRUMENT-DEFECT and nothing downstream is read.
- GATE ENG (A13): >= 60 percent of q1 events have n_cand_no_bh > 0 at every node (the banked fraction is
  425/540); the per-candidate N_g must differ across the three h-nodes on >= 99 percent of rows.

### 6.4 Cost and configuration

4 seeds 900101-900104 (the KW-Q1 realisations; frozen q1 membership from b4_imp_stage1_events.csv) x
h in {0.725, 0.730, 0.735} (secant slopes per candidate; the 0.73 node gives c_i at the midpoint and the
T-ID parity) in the FT/HEAD configuration of KW-Q1 (absolute_marginal, phi twin, Sigma_phi divisor, fused,
theta = (0,1), sites 2.2, no smear). Cost at the registered mirror anchor 0.2843 CPU-h per single-h cell +
0.1333 per cell overhead (PREREGISTRATION_HIER_HTHETA_20260826.md:584): 12 x 0.2843 + 4 x 0.1333 = 3.94 CPU-h;
at KW-Q1's measured 0.23 CPU-h per cell (5.514 CPU-h / 24 cells, claim card section 5): about 3.3 CPU-h.
Registered: 3.4-3.9 CPU-h, local, no cluster. Optional T2.2b (recommended, decisive for H0): the same hook on
production iiib at the 3 nodes, about 5-7 min wall per h-point (tree-2 charter cost anchor) — about 4-5
CPU-h local — which makes the section 6.6 rescore available on the production fleet.

### 6.5 The registered statistic (F3 predictions now; A15 characteristics)

Population: dark (host_galaxy_index = −1), q1 (z_true < 0.358), active (L_cat_no_bh > 0 at both secant nodes)
events of the 4 seeds; N about 190 at 4 seeds (191 in the KW-Q1 frozen set). Weight of candidate g in event i:
W_ig = w_g N_g_used (the catalogue leg's own summand, h = 0.73).

Primary: Phi_low,i = sum_{g: z_g < z_true,i} W_ig / sum_g W_ig — the impostor-weight share listed below the
true redshift — and its q1 mean; secondary: the W-weighted mean listed offset <u>_W,i and its W-weighted SD.
Null (no depth skew): Phi_low = 0.5, <u>_W = 0. Prediction from section 3 with the C7 kernel: the W-weighted
listed-z offset is the tilt of the listed density times sigma_g^2, in GW-sigma units lambda sigma_g
(sigma_g/sigma_z,GW) = about −1 sigma_GW at z 0.3 (lambda about −10 per unit z, sigma_g 0.035, sigma_g/sigma_z
about 2.7), with a W-weighted SD of 2-4 sigma_GW set by sigma_g/sigma_z; hence Phi_low(q1) in [0.60, 0.70]
and <u>_W in [−1.5, −0.5] sigma_GW. Operating characteristics at N = 190: per-event Phi_low has SD <= 0.35
(bounded variable), so SE(mean) <= 0.025; the predicted 0.60-0.70 sits 4-8 sigma from the null; the paired
within-run design has zero sampling variance under the null of no change between nodes (A15 corollary),
and the across-seed SD of the mean (4 seeds) is reported as the generalisation width. Bands: DEPTH-SKEW-
CONFIRMED if the q1 mean Phi_low >= 0.57 AND <u>_W <= −0.3 sigma_GW; DEPTH-SKEW-REFUTED if Phi_low <= 0.53 or
<u>_W >= 0 while the q1 s_imp stays <= −0.6 (then the score is carried by Delta_S or by a divisor object,
section 8.2); MIXED otherwise. Tertiary (the [HIER] b-axis object, reported only): the W-weighted mean of the
kernel-mean offset against the listed-z offset — the C7 prior shift lambda sigma_g^2 — by z_g bin.
Candidate-count read (section 3.3): within each q1 z-bin, Spearman rho between n_cand_no_bh and Delta_i =
s_imp,i/c_i, predicted negative (denser balls, more negative Delta), REPORTED-ONLY (confounded by the ball
volume).

### 6.6 The zero-compute rescore the run enables (the pre-read for remedy (d))

With S_4D(z_g, M_g; h) and N_g_used serialised at the 3 nodes, the mass-aware 1D leg of section 5.4 is a
per-event rescore on the 4 seeds: L'_i(h) = sum_g w_g [N_g_used / S_bar_phi(z_g;h)] S_4D(z_g, M_g; h) /
Sigma_4D(h) (point path exact; quadrature path to first order in the ratio's variation across the window,
disclosed), p'_i = (alpha_G L'_i + B_i)/D_tilde, combined with the row #146 corrected combine over the two
secant nodes and H_GRID_41 where available. Statistic: the paired Delta mean_h (mass-aware minus phi) on the
4 FT seeds, and the in-ball Malmquist ratio rho_i = sum W S_4D/S_bar / sum W per q1 event. Prediction: rho_q1
in [0.2, 0.5] (anchor r_Malm = 0.383, catalogue-wide); q1 s_imp scaled by about rho: −0.80 -> −0.31 (+/- 0.1);
Delta mean_h(FT) = +0.05, band MASS-AWARE-MATERIAL >= +0.03, NULL <= +0.008 (T_mat), MIXED between. Runner:
not this agent, not the hook's author.

## 7. Question (5): F3 predictions for each remedy on the production read

Inputs (all ARITH on the record): I_1D = 1303 (sigma_h = 0.0277 of the dark-only pure arm, claim card C5);
per-event impostor scores at truth on iiib: dark −0.193, in-catalogue −1.707 (C5); pure-arm score totals:
all events +158 nats per unit h (headreadout_20260827/iiib/event_likelihoods.csv, B_num/D_tilde secant,
ARITH), dark about −22 (from the 0.7134 read), in-catalogue about +2.3 per event (C5); linear response
Delta h = Sigma score / I_1D, uncensored (the 41-node grid censors at 0.60, so a linear prediction below 0.60
reads "floor"). Current: Sigma = 1514 x (−0.193) + 76 x (−1.707) + 158 = −264 -> −0.20 -> floor (measured 0.6077,
floor mass 0.446).

| remedy | production full 1D (1588) | production dark-only full mixture (1514) | mirror FT fleet (12 seeds) |
|---|---|---|---|
| (a) per-event weight | no consistent form; not registered | — | — |
| (b1) completed, numerator only | floor (mirror −0.0028 scales to about −0.004) | floor | −0.0028 (banked) |
| (b2) completed incl. D_tilde | a global tilt of order +0.12 per event: Sigma −264 + 190 = −74 -> about 0.67 (band [0.63, 0.71]); this is a Z-type re-normalisation, not a per-event repair | about 0.63 | +0.0344 (banked, REPORTED-ONLY) |
| (c) enlarged ball | Sigma more negative by (1/kappa − 1) x 422 = −60 to −210 -> floor; q1 S(1) −1.02 -> −1.2 to −1.5 | floor | −0.02 to −0.05 (down) |
| (d) mass-aware 1D leg, rho = 0.383 | dark impostor −292 -> −112; in-cat −130 -> about −117; Sigma = −112 − 117 + 158 = −71 -> −0.054 -> about 0.675, band [0.64, 0.72] over rho in [0.25, 0.5] | −112 − 22 = −134 -> −0.10 -> about 0.63, band [0.60, 0.67] | +0.05, band [+0.03, +0.10] (section 6.6) |
| (d) + a closed dark-class completion residual (section 4.4; not B4's) | Sigma about +140 -> 0.73 within sigma_h | about 0.73 | — |

Reading: no impostor-leg remedy reaches truth on production while the dark-class completion leg carries the
−0.15 per event of section 4.4; remedy (d) is predicted to lift the 1D off the floor by about +0.07 and to
close the class-share and Z(h) defects (an exact structural statement, section 4.3), which is the part of the
rail B4 owns. The MAP-at-floor of the current production 1D is predicted to persist under (a), (b1) and (c).

## 8. Question (6): A14 falsifiers

### 8.1 Of the attribution "the mixture-weight h-slope owns about 63 percent" — DISCHARGED BY DERIVATION

The statement is true as arithmetic of C3's first-order identity with s_beta carrying the −3/h volume factor
and false as a mechanism statement, because it is not invariant under the placement of a factor common to
all three terms of p_i (section 2.4). Re-booked invariantly, the global mixture-weight-and-divisor term is
mean(c)(sigma_G − sigma_cat) = +0.013 per event (FT basis). The attribution is WITHDRAWN as a mechanism and
replaced by (II). Residual falsifier of the re-booked statement ("the global term is <= 15 percent of the q1
score"): compute sigma_G − d ln(global_denom_no_bh)/dh from the T2.2 run's own selection tables
(selection_tables_h_0_725/0_735.json) — if |sigma_G − sigma_cat| > 1.0 per unit h on the run-of-record
tables, the bound fails and the global term must be re-quantified; and GATE R plus the section 6.5 primary:
if the W-weighted <d ln N_g/dh> minus the completion analogue accounts for < 80 percent of s_L,i − s'_B,i on
q1, Delta_N is not the carrier and the remainder is Delta_S or a divisor object.

### 8.2 Of the depth-skew attribution (section 3) — the enlarged-ball arm, registered here

Arm: section 5.3 geometry (sky 3.0 sigma via self._sky_cone_k = 3.0; z-window widened by +/- 4 sigma_g),
4 seeds x 2 secant nodes, HEAD/FT configuration of KW-Q1, paired against the KW-Q1 truth-node values on the
frozen 191 q1 events. Statistic: ratio Q = S_q1(enlarged)/S_q1(1.5 sigma) of the q1 mean impostor scores, and
the paired FT Delta mean_h. Prediction: Q in [1.1, 1.6] (more negative; the captured-mass ratio), Delta mean_h
in [−0.06, −0.01]. Bands: DEPTH-SKEW-CONFIRMED if Q >= 1.1; TRUNCATION-OBJECT (attribution refuted) if
Q <= 0.9 or Delta mean_h >= +0.01; MIXED otherwise. A15: the paired deterministic design has zero sampling
variance under Q = 1; the across-seed SD of Q from 4 seeds is the generalisation width (KW-Q1's per-seed R
scatter, SD 0.055 on a ratio of the same kind, is the anchor: a 0.1 band is about 2 sigma per seed and
4 sigma on the 4-seed mean). Cost about 15 CPU-h local (section 5.3). Builder/runner: neither this agent nor
the T2.2 runner.

### 8.3 Of the section 5.4 pointer (remedy (d))

Falsified before any code if the section 6.6 rescore returns Delta mean_h(FT) <= +0.008 or rho_q1 > 0.8
(the catalogue galaxies in q1 balls would then be as detectable as the population average and the class-share
argument would have no in-ball counterpart); falsified on production if, after the gate, the full 1D MAP
stays <= 0.605 (floor) with the dark-only pure arm unchanged. Independent zero-compute check available now:
the monitored gate (ii) of the T2.2 run must log predicted in-cat share = 0.0620 x 0.7305 = 0.0453 against
the mirror's realised class share (all-dark by construction on B-SEL: this check is for production T2.2b,
where the realised 76/1588 = 0.0479 already sits 6 percent from the mass-aware prediction and 62 percent
from the 1D channel's own 0.1472 x 0.7305 = 0.108).

## 9. Exoneration check (mechanism-grepped, both layers), scope, blindness

Grepped EXONERATION_REGISTER_20260827.md and BIAS_HISTORY_LEDGER.md sections 2-4 for: "mass-blind",
"mass blind", "mass-aware", "class share", "in-cat share", "w_tilde_G", "D_phi", "integrates to",
"over-normalis", "Z(h)", "Malmquist", "depth". Hits: the register's "Malmquist / magnitude-limit: 0 hits as a
mechanism" (claim card section 0.2 row 17, re-confirmed); the register's C9 / Gate C item 1 keeps the
mixture-weight calibration LIVE (claim card row 15); [DEPTH-TRUNC] binds only against a z_max cut (not
proposed); [WPOP-TUNING], [NUMERATOR-ONLY-CLEAN], [HARD-CLAMP-OBSERVED-Z] bind on remedies and are respected
by (d) (paired three-object change; no clamp; w_pop untouched). No entry exonerates the 1D class-share /
Z(h) object or the in-ball depth skew. Not in scope: the 2D channel (B7's), the mass window (B5's), the sky
cone as a bias object (B2's), the dark-class completion residual of section 4.4 (B8 [CAL]).

Blindness of this derivation: it is first-order in c and in the secant stencil; the continuum limit
assumes the listed density of the ball matches the model's smeared intensity (the venue-truth question of
the C7 kernel is the [HIER] b-axis, tested only as the tertiary read of section 6.5); the noiseless data law
is treated as a second-order caveat and not derived; the enlarged-ball kappa is bounded, not measured; the
production per-class numbers rest on C5's assumption-join.

## 10. Numbers of record (value, source, date)

- s_beta = −3.2891472031202764 per unit h — b4_imp_stage1_split.json fleet.s_beta_global (fanout1_20260829), 2026-08-29; table secant −3.28915 — selection_tables_h_0_725/0_735.json (p3_work/ft_900101_work/seed900101), 2026-08-23 run.
- s_D = −1.2372867562 (same JSON) versus table −1.23728.
- sigma_G = +0.8205, sigma_Gbar = +3.0049, sigma_cat = +0.51807, sigma_G − sigma_cat = +0.30243, volume −4.10965 — ARITH, section 2.2, 2026-08-30.
- Z(0.73) = 1.099921, d ln Z/dh = −0.18895; beta_G/D_tilde = 0.161888 (slope −2.05187); beta_G/D_phi = 0.147181 (−1.86292); beta_Gbar/D_tilde = 0.938033 (+0.13257); w_tilde_G = 0.0619668 (−2.00671); r_Malm = 0.38277622 (+0.04516); n_hat_w = 6.397401 (+3.80722) — ARITH on the tables; w_tilde_G and n_hat_w cross-checked against p3_work/ft_900101.log "path-A(h=0.7300)".
- Production tables identical to 7 s.f. — headreadout_20260827/iiib/event_likelihoods.csv columns alpha_G_phi, r_Malm, D_tilde_phi at h = 0.725/0.73/0.735 (row #213, d04d9dc9).
- Monitored gate (ii): predicted in-cat share 0.0619668 x 0.7305 = 0.0453 (P0_WINDOW_CLASS_RETENTION_RATIO :1858; rescore_class_share_joint_selection :2698); realised 76/1588 = 0.0479 (claim card C5).
- FT per-event (12 seeds): mean s_imp −0.21776 (SEM 0.0158), mean s_pure +0.07076 (per-seed mean +0.07056, SEM 0.0250), mean s_full −0.14700; mean_c_all 0.04153; q1 (z < 0.358, n 540): mean s_imp −0.7963, mean c 0.1599, c-weighted Delta −4.981; z-bin table of section 3.4 — b4_imp_stage1_events.csv, ARITH, 2026-08-30 read.
- 4-seed FT q1 (n 191): mean s_imp −0.9028, mean c 0.1655 — same file; KW-Q1 HEAD S(1) = −1.0205308 (n 191) — CLAIM_IMPOSTOR_DRAG_20260829.md:386.
- b0i full score −0.1238 +/- 0.0527; w_tilde_G(0.73) = 0.0620 — claim card C4 (:187-190).
- Production C5: full 0.6077 / MAP 0.60 / floor mass 0.446; pure all 0.8396; pure dark-only 0.7134 / sigma 0.0277; s_imp pooled −0.265 +/- 0.051, dark −0.193, in-catalogue −1.707 — claim card :207-215.
- Production direct read: Sigma s_full = −297.77, Sigma s_pure(all) = +157.92 over 1588 rows — iiib event_likelihoods.csv, ARITH secant 0.725/0.735, 2026-08-30.
- Row #150 item 1 tilt −0.133/h per event (ln D_tilde/beta_Gbar) — BIAS_HISTORY_LEDGER.md:1932-1946; row #167 fork −0.002810 +/- 0.000467 and +0.034357 +/- 0.004342 — :2417-2430; row #171 −0.004309 +/- 0.000736 — :2486.
- Forensic E9/E12/E14/E15/E17 — B1_1_S0A_DEFECT_FORENSIC_20260829.md:84-92.
- Mirror cell cost 0.2843 CPU-h + 0.1333 overhead — PREREGISTRATION_HIER_HTHETA_20260826.md:584; KW-Q1 5.514 CPU-h / 24 cells — claim card :425-427.
- z_eff(z) = d_L/(d d_L/dz): 0.138 / 0.21 / 0.258 / 0.334 at z = 0.15 / 0.25 / 0.30 / 0.40 — ARITH, flat LCDM Omega_m 0.2726.

## 11. What is not claimed

- Not claimed: that remedy (d) un-rails the production 1D to truth (section 4.4 forbids it); that the
  dark-class completion residual is a defect (recorded, candidates listed, not adjudicated).
- Not claimed: any value for A11; not claimed that the twin adoption (row #197) was wrong — it is the
  population-average approximation to (d).
- Not claimed: any T2.2 number. The predictions of sections 6.5, 6.6, 7, 8.2 are F3 registrations with bands
  and characteristics; every one of them can fail.
- Not claimed: the enlarged ball as a remedy; it is a falsifier arm.

## 12. Hand-off (tree 2)

T2.2 (instrumented run, builder != runner != this agent) per section 6, with the section 6.6 rescore as its
zero-compute second half; T2.3 the enlarged-ball falsifier per section 8.2 (or, if the rescore of 6.6 comes
back MATERIAL, the gate presentation for remedy (d) first — path choice under row #255, stated in the wave-1
docket; the production flip of (d) itself is a fresh [RULE] for the author because it modifies row #169
(i)/(ii), section 4.3); the section 4.4 residual to B8 [CAL] as a registered finding; A11 returns to the
author with section 5.2 as its input. Author-facing items produced by this node, tagged: [RULE] the mass-aware
1D pairing (d) after its gate presentation and the 6.6 pre-read; [RULE] A11 with section 5.2; [DO] T2.2 as
designed (covered by row #255); [DO] T2.3 enlarged-ball falsifier (covered).

launched under row #255 — tree 2 node T2.1 — derivation complete, zero compute, no code.
