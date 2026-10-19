# BAND RE-DERIVATION AND PURE-INPUT ADJUDICATION — T2.2b follow-on, rows #278/#280 — 2026-08-31

Derivation node (top-tier), foreground, no commits, no source edits. Every number below re-derived
by this node from the T2.2b arm data (off + on, 3 nodes h = 0.725/0.730/0.735), the banked
headreadout_20260827/iiib 41-node grid, the repo-root selection_tables_h_*.json, and the O2
artifacts under fanout1_20260829/. Statistic conventions: secant d_h over 0.725/0.735 (step 0.01);
s_full,i = d_h ln combined_no_bh,i; s_pure,i = d_h ln(B_num,i/D_tilde_phi); s_imp,i = s_full,i −
s_pure,i (the C2/C5 subtraction); class split by the T2.2b per-event dump host_galaxy_index
(validated join), dark = −1 (1512 events), in-catalogue = 76; active dark = n_cand_no_bh > 0 at
0.73 (907 events). Data: off/simulations/diagnostics/event_likelihoods.csv and the on twin (1588
events x 3 h each, full float precision, B_num > 0 on all 1588 events at all nodes); posterior
JSONs off/on/simulations/posteriors/h_*.json; per-candidate and per-event dumps under
*/candidate_dump/. The pure columns (B_num, D_tilde_phi) are bit-identical between the on and off
arms (max relative difference 0.0), so s_pure is one shared vector and delta_s,i = s_full,i(on) −
s_full,i(off) = s_imp,i(on) − s_imp,i(off) exactly.

---

## 1. PURE-INPUT ADJUDICATION (the row #280 fork): +157.92 binds; +123.11 is a storage-precision artifact

### 1.1 The fleet pure sum, re-derived

Sum over all 1588 events of d_h ln(B_num,i/D_tilde_phi), secant 0.725/0.735, from the off-arm
event_likelihoods.csv at full stored precision:

    Sigma_i s_pure,i = +157.9219 nats per unit h   (1588 of 1588 finite; no B_num = 0 events)

This reproduces the disputed standalone number +157.92 to the digit. The class decomposition of
the same sum: dark −96.864, in-catalogue +254.786.

### 1.2 The identity and the two internally consistent triples

The identity is definitional, not physical: for ANY choice of the per-event pure object,
s_full,i = s_pure,i + s_imp,i with s_imp defined as the difference, hence
Sigma s_full = Sigma s_pure + Sigma s_imp always holds for a SELF-CONSISTENT pair. Re-derived
off-arm sums at full precision:

    Sigma s_full = −297.7743 (mean −0.18752, matching the O2 JSON score_full_mean −0.187515 exactly)
    Sigma s_imp  = −455.6961 = dark −291.162 + in-catalogue −164.535
    check: −291.162 − 164.535 + 157.922 = −297.775  (sums, as it must)

The verifier's triple (−291.16, −129.72, +123.11) ALSO sums to −297.77. So the row #280 argument
"only +123.11 makes the three components sum" does not discriminate: BOTH triples sum, each with
its own in-catalogue member. The fork is therefore not "which complement closes the identity" but
"which in-catalogue/pure pair is the correct measurement of the defined objects". That is decided
in 1.3.

### 1.3 Where +123.11 comes from — reproduced and diagnosed

The O2 JSON (fanout1_20260829/b4_imp_stage1_production_o2.json, iiib block, score_pure_mean =
0.0775226; x 1588 = +123.106) was produced by b4_imp_stage1_production_o2.py, which does NOT read
B_num/D_tilde_phi directly. It reconstructs, from the banked headreadout_20260827/iiib
event_likelihoods.csv (7-significant-figure storage):

    cat  = beta_G_phi x L_cat_no_bh / D_tilde_phi
    pure = clip(combined_no_bh − cat, 0, None)          (script line 96)

This node re-ran that exact construction on the banked CSV and reproduced score_pure_mean x 1588 =
+123.1059 to the digit. Per-event comparison against the direct secant of B_num/D_tilde_phi from
the same CSV (+157.9219): the two differ by more than 1e-3 on EXACTLY 18 events, all 18
in-catalogue, whose catalogue share cat/(cat + comp) at 0.73 is >= 0.9923 (median 0.99756;
comp/cat median 0.00245 on these events). On such events combined_no_bh − cat is a subtraction of
two numbers agreeing to ~2.6+ significant digits stored at 7 significant figures: the difference
retains ~4 or fewer significant digits and its h-secant is noise at the 0.1–10 per-unit-h scale.
The summed corruption is −34.8159 nats per unit h = exactly 157.9219 − 123.1059. On the other 1570
events the two pure scores agree.

The same corruption propagates, with opposite sign, into O2's in-catalogue impostor score: O2's
s_imp in-catalogue mean is −1.70689 (x 76 = −129.72), against the full-precision −2.16494 (x 76 =
−164.54) — the two differences are the same 34.82 nats. That is why each triple closes internally:
the O2 pair (pure +123.11, in-cat −129.72) shares one error that cancels in the sum.

### 1.4 The disputed hypothesis, tested

Row #280 / FULL_VERIFICATION section 3 item 6 conjectured a derivational difference ("the pure-arm
slope includes the composition tilt d ln(beta_Gbar/D_tilde)/dh; the complement does not"). This is
REFUTED: the two candidates are the SAME defined object, d_h ln(B_num/D_tilde_phi), evaluated once
from the full-precision columns (+157.92) and once through a catastrophically cancelling
reconstruction at 7-s.f. storage (+123.11). No composition-tilt term separates them; the entire
gap sits on 18 catalogue-dominated events. (For the record, the composition objects re-derived
from the run's own selection tables: d ln sigma_phi/dh = +0.51807, d ln sigma_4d/dh = +0.56323,
d ln beta_G_phi/dh = −3.28915, d ln beta_Gbar_phi/dh = −1.10471, d ln r_Malm/dh = +0.04516;
Sigma_phi/D_tilde_phi = 1.026584/1.035662/1.044763 at the three nodes, slope +1.7552 — none of
these is the 34.82-nat discriminator.)

### 1.5 Adjudication (a derivation, offered to the author's fresh [RULE] — nothing flipped here)

- The pure input that BINDS as the measurement of the defined object is **+157.92** nats per unit
  h. +123.11 is an artifact of 7-s.f. CSV storage in the O2 reconstruction and should be retired
  as a pure input (and with it O2's in-catalogue s_imp −1.707, already superseded by the T2.2b
  measured −2.1649).
- HOWEVER, the registered section 6.3 arithmetic is thereby shown to be internally MIXED, not
  vindicated: it consumed pure = +158 (the clean number) TOGETHER WITH in-catalogue −130 -> −117
  (the corrupted number, times an unsubstantiated transform — already relabelled REPORTED-ONLY in
  the gate doc revision note). A consistent full-precision linear-response arithmetic at the same
  rho in [0.25, 0.5], using the measured on-arm in-catalogue sum (−162.98) and pure +157.92, gives
  Sigma = −291.16 rho − 162.98 + 157.92, i.e. edges 0.730 − Sigma/1303 in **[0.614, 0.670]** —
  neither the registered [0.64, 0.72] nor the alternative [0.6226, 0.6787]. All three are
  superseded by the measured band of section 2, which needs no decomposition at all.
- Verdict on the fork as posed: NOT undetermined. +157.92 is the correct pure; +123.11 is
  reproduced and explained as precision noise; and the practical consequence for the band is moot
  because section 2 replaces the decomposition arithmetic with the measured on-arm response.

---

## 2. BAND RE-DERIVATION FROM MEASUREMENT (no decomposition consumed)

### 2.1 Measured fleet response at 0.73

Per-event delta_s,i = s_full,i(on) − s_full,i(off) (identical to the s_imp difference, section 0):

    fleet Delta-ell-prime(0.73) = Sigma_i delta_s,i = +216.9030 nats per unit h
      dark +215.35 (off −291.16 -> on −75.81; effective rho = 0.2604)
      in-catalogue +1.554 (off −164.54 -> on −162.98; +0.0204 per event)
    on-arm total ell-prime(0.73) = −80.8713  (posterior-JSON triplet secant −80.87127 agrees with
      the CSV sum to 5e-11; off-arm −297.7743 likewise)

### 2.2 Curvature: which I was used and why

The LOCAL second difference of the total log-likelihood at 0.73 is POSITIVE (off arm ell'' =
+2169.9 per unit h^2 from the 3-node triplet; on arm +1264.9): both curves are convex there
(0.73 sits on the high-h flank of a posterior peaked far below), so a local-quadratic MAP read
from the 3 nodes alone is impossible — the 3-node curvature is not merely coarse, it has the wrong
sign for a peak. The prediction therefore uses the banked full 41-node grid
(headreadout_20260827/iiib/posteriors/h_*.json; grid 0.60–0.86, step 0.01 coarse / 0.005 in
0.65–0.75), which the T2.2b off arm reproduces EXACTLY (total ln L equal at all 3 shared nodes to
1e-12: −5644.41557/−5645.93157/−5647.39332), plus the measured shift
Delta(h) = ell_on − ell_off at the 3 nodes:

    Delta(0.725/0.730/0.735) = −62.3808 / −61.2850 / −60.2118
    Delta-prime(0.73) = +216.903;  Delta-double-prime = −905.03 (single second difference, noisy)

Predicted on-arm full-grid curve: ell_on(h) = ell_banked(h) + Delta(0.73) + Delta-prime (h−0.73)
[+ 1/2 Delta-double-prime (h−0.73)^2 for the quadratic variant]. Moments with the trapezoid
weights and flat prior (this convention re-derives the banked off arm to the row #213 record
exactly: MAP 0.6000, mean 0.6077, floor-node mass 0.4460).

### 2.3 Predicted post-flip production 1D (41-node arm)

    linear Delta-extension:    MAP 0.650   mean_h 0.6524   floor mass 0.0023
    quadratic Delta-extension: MAP 0.670   mean_h 0.6727   floor mass 0.0000
    linear-response cross-check (I_1D = 1303, the sigma_h = 0.0277 Gaussian-equivalent):
                               h_hat = 0.730 − 80.87/1303 = 0.6679

**Prediction: post-flip MAP ~ 0.66, honest bracket [0.65, 0.67] (grid nodes); mean_h ~ 0.66,
bracket [0.652, 0.673]. The posterior leaves the 0.60 floor decisively (floor mass <= 0.002
against 0.446 off). Predicted Delta mean_h(on − off) = +0.045 to +0.065.**

Error-bracket sources (all included in the bracket above, stated honestly):
1. Delta-extension order: lin vs quad spread is the dominant term (0.650 vs 0.670 MAP). The
   destination sits 3–16 measured-window widths below 0.725, so Delta(h) there is an
   extrapolation; Delta-double-prime rests on ONE second difference of numbers of size ~1 nat.
2. Secant vs local slope: all scores are 0.01-secants at 0.73, not local derivatives at the
   destination; the quadratic variant is exactly the first correction for this, and its size
   (0.02 in h) is the honest scale of the residual.
3. Truth-vs-MAP evaluation: the response is measured at h = 0.73 (truth), the MAP lands near
   0.66; any curvature of the per-event scores between those points is unmeasured by this arm
   (only the 41-node run itself closes this).
4. Censoring (gate doc amendment 20): the off mean 0.6077 is floor-censored while the predicted
   on mean is not, so the predicted Delta mean_h compares a censored baseline to an uncensored
   prediction and is a LOWER bound on the un-truncated effect.

### 2.4 Which band contains the prediction

- Registered section 6.3 band **[0.64, 0.72]: CONTAINS** the full bracket (MAP and mean, all
  variants).
- Alternative (identity-complement) band **[0.6226, 0.6787]: ALSO CONTAINS** the full bracket.
- The consistent-arithmetic linear-response band of section 1.5, [0.614, 0.670]: contains the
  linear variant and the cross-check; its upper edge sits ON the quadratic variant.
- The measurement therefore cannot discriminate the two registered candidates (the bracket lies in
  their intersection [0.64, 0.6787]); the discrimination is the algebra of section 1, which
  retires +123.11 anyway. The Z-CONFIRMED rule as registered (map_h AND mean_h in [0.64, 0.72])
  is PREDICTED SATISFIED by the 41-node arm.

---

## 3. THE Q-MEDIAN STRUCTURE (registered 6.2 statistic, re-derived)

On the 907 active dark events, q_i = s_imp,i(on)/s_imp,i(off):

    median q = 0.002553;  q > 1 on 7/907 = 0.77 percent (F-3 refuter bar > 10 percent: NOT TRIGGERED)
    deciles (10th..90th): 4.50e-5, 1.00e-4, 3.09e-4, 8.65e-4, 2.55e-3, 8.59e-3, 3.70e-2, 1.78e-1, 6.90e-1
    range: min −0.018, max 1.26

Concentration of the on-arm dark-class sum (−75.809 over 907 active events):

    46 events (5.07 percent of active dark; 3.04 percent of all 1512 dark) carry 90 percent of it
    top 10 percent of active dark carry 99.1 percent
    off-arm contrast: 199/907 = 21.9 percent carried 90 percent — the flip concentrates the class
    signal by a factor ~4.3 in event count

Who survives: the per-event candidate-weighted mean survival ratio rho_i = sum_g w_g N_g
(S_4D/S_bar_phi) / sum_g w_g N_g (off dump, h = 0.73) rises MONOTONICALLY across the q deciles
(mean rho_i 0.0000 in decile 1 to 0.0902 in decile 10), and the 46 carrier events have mean rho_i
0.269 (median 0.203, median q 0.574) against 0.0032 for the remaining active dark events; the
carriers are also candidate-rich cones (median 502 candidates vs 84). (Disclosed: the raw Spearman
of q against rho_i over all 907 is −0.20 — an artifact of rank noise among the ~1e-12 floor
values; the decile means above are the robust read.) For reference, the 66 recovered true
in-catalogue hosts have S_4D/S_bar_phi median 1.0391, mean 1.0336, range [0.9128, 1.0829] —
confirming the run-record banked transform.

Physical paragraph. The mass-aware leg replaces the population-average survival S_bar_phi(z_g) by
the per-galaxy point survival S_4D(d_L(z_g; h), M_g(1+z_g)) inside the 1D catalogue numerator. For
the TYPICAL impostor candidate — a low-stellar-mass galaxy whose implied BH mass sits far below
the detection band at its redshift — S_4D/S_bar_phi is of order 1e-12..1e-2, so the typical
impostor's weight, and with it its h-score, is annihilated (median per-event shrink ~390x, not the
class-mean 0.26). The class MEAN nonetheless lands exactly in the registered band because a thin
tail of S_4D-FAVOURED impostors survives: candidates massive (and near) enough that their point
survival rivals or exceeds the population average. The remedy does not uniformly shrink the
impostor field by rho — it prunes it to its mass-selected tail, which is precisely what a
mass-aware Malmquist weight is supposed to do. Consequences for the record: the class-mean band
[−0.097, −0.048] **PASSED** (measured −0.0501, effective rho 0.2604 at the rho-range lower edge)
and nothing about that registration changes; the median-q band [0.25, 0.5] is **REFUTED-IN-DETAIL**
— the "dark class scales by rho" statement is true of the mean and false of the per-event
distribution, and must be recorded as a corrected mechanism narrative (tail-carried, not uniform),
NOT as a failure of the remedy; the F-3 refuter is clean. No other registered conclusion is
touched.

---

## 4. VERDICT SUMMARY TABLE AND RECOMMENDATION

| Registered item (gate doc section) | Registered band / rule | Measured / derived (this node) | Verdict |
|---|---|---|---|
| 6.2 dark-class s_imp mean, on | [−0.097, −0.048] (rho in [0.25, 0.5]) | −0.0501 (off anchor −0.1926 reproduced exactly; rho_eff 0.2604) | PASS |
| 6.2 median q_i, active dark | [0.25, 0.5] | 0.00255 | REFUTED-IN-DETAIL (mean-vs-median mechanism; section 3) |
| 6.2 F-3 refuter: q_i > 1 share | > 10 percent refutes | 0.77 percent | NOT TRIGGERED |
| 6.2 in-catalogue, on | −1.707 -> ~−1.54, band [−1.7, −1.4] (SUPERSEDED, REPORTED-ONLY) | off −2.1649 -> on −2.1445 (Delta +0.0204/event; +1.55 nats fleet) | SUPERSEDED band n/a; measured value now BANKED (matches run record) |
| 6.2 pooled, on | [−0.166, −0.120] (SUPERSEDED — inherits −117) | −0.1504 | SUPERSEDED band n/a (numerical coincidence that it lands inside; do not resurrect) |
| 6.3 pure input | +158 consumed by the registered arithmetic | +157.92 ADJUDICATED CORRECT; +123.11 = O2 storage-precision artifact (18 in-cat events, −34.82 nats) | RESOLVED — not undetermined |
| 6.3 MAP band | [0.64, 0.72] (REPORTED-ONLY pending T2.2b) | predicted MAP 0.65–0.67, mean 0.652–0.673 | CONTAINS prediction; arithmetic behind it was mixed-convention (1.5) — band superseded by the measured band below |
| 6.3 alternative band (row #280) | [0.6226, 0.6787] | same prediction | ALSO CONTAINS; retired with +123.11 |
| 6.3 dark-only pure arm pin | unchanged 0.7134 +/− 0.0277 | pure columns bit-identical on vs off (max rel 0.0) | PIN HOLDS structurally |
| 6.3 floor departure (A14/F-2) | map_h <= 0.605 refutes attribution | predicted floor mass <= 0.002 (vs 0.446 off) | PREDICTED CLEAR |

**Recommendation for the author's fresh [RULE] (this node flips nothing):** adopt the MEASURED
band — post-flip production 1D MAP predicted 0.66, bracket [0.65, 0.67], mean_h [0.652, 0.673],
Delta mean_h +0.045 to +0.065 — as the operative comparison for the A18 41-node arm; treat the
registered [0.64, 0.72] Z-CONFIRMED rule as PREDICTED SATISFIED, and retire both the +123.11 pure
input and the [0.6226, 0.6787] alternative as artifacts of the O2 storage-precision cancellation
(row #280 fork RESOLVED). Record the median-q band [0.25, 0.5] as REFUTED-IN-DETAIL with the
corrected tail-carried mechanism; the class-mean registration PASSED unchanged.

**120-word summary.** The row #280 pure-input fork is resolved: the fleet pure score is +157.92
nats per unit h, re-derived directly from full-precision columns; the +123.11 candidate is
reproduced exactly and shown to be a storage-precision artifact — O2 reconstructed the pure term
as combined minus catalogue from 7-significant-figure CSVs, and 18 catalogue-dominated
in-catalogue events lose the subtraction, accounting for the entire 34.82-nat gap. Both disputed
triples close the identity; only the full-precision one measures the defined objects. The band no
longer needs the decomposition: the measured on-arm response (Delta-ell-prime +216.90, on-arm
slope −80.87) applied to the banked 41-node grid predicts post-flip MAP 0.66 [0.65, 0.67], mean_h
0.652–0.673, off the floor — inside both candidate bands. Median-q [0.25, 0.5]:
REFUTED-IN-DETAIL; class-mean band: PASSED.
