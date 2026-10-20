# T1.3-zwin — is PA-HIER-32(d)'s Es_null_det the null expectation of score_lns under the theta-consistent window? (derivation addendum) — 2026-08-30

**Launched under ledger rows #255 / #268 — tree 2 node T1.3-zwin (derivation addendum).** Top-tier derivation
node; zero evaluate() calls; no git; no source edits; foreground only; append-only. Read-only on
`tree2_20260830/hier_s0_zwin_run/**` (the P1 arm of record) and on every concurrently-written B8.2 file.
Every number below carries {value, source, date}; the zero-compute readbacks that produced the new numbers are
archived with their outputs in `tree2_20260830/t1_3_es_null_det_work/` (`t13_esnull.py` -> `t13_esnull_out.json`,
`t13_esnull2.py` -> `t13_esnull2_out.json`, `t13_esnull3.py` -> `t13_esnull3_out.json`; all 2026-08-30).
Every [HIER] statement carries the REPORTED-ONLY cap (PA-HIER-28 item 9). **This node does not re-adjudicate
P1: the verdict of record stays B0-A' (`hier_s0_zwin_run/s0a_score.md`, 2026-08-30) until the amendment
proposed in section 5 is registered by an author [RULE].**

Inputs of record: `PREREGISTRATION_HIER_HTHETA_20260826.md` PA-HIER-4 (lines 839-875), PA-HIER-32(d) (lines
2727-2800); `fanout1_20260829/B1_1_S0A_DEFECT_FORENSIC_20260829.md` E12/E13/E14/E17 (table rows 87-88, 92, 95;
section 3 "s-axis"; section 4 item (3)); `fanout1_20260829/b1_1_forensic_work/f4_mechanism.py` (`kern()`, `E()`),
`f4_out.json`, `f7_events.csv`; `fanout1_20260829/hier_s0_driver.py` lines 508-733 (`_es_null_det_closed_form`,
`_es_null_det_kernel`, `compute_es_null_det_table`) and 1104-1335 (`compute_scores`);
`tree2_20260830/PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md` sections 1c, 2.1-2.3, 3 (table), 5.6, Revision note 2
item 3, Implementation prerequisites item 1; `tree2_20260830/hier_s0_zwin_run/{s0a_score.md, s0a_score_output.json,
s0a_seed9001{01..04}/es_null_det.csv, s0a_seed*/node_{truth,s_plus,s_minus}_sites2.2_nosmear_divisor_zwin_zk4/
simulations/diagnostics/event_likelihoods.csv, logs/runner7_tree2_20260830.log}`;
`tree2_20260830/hier_s0_recert_run/**` (T1.2, for the cross-configuration check only);
`darksiren_emri/bayesian_inference/bayesian_statistics.py` lines 4818-4835, 6700-6745, 8101-8144 (working tree,
HEAD e35f9d4e).

Notation. `Delta = ln sqrt2 = 0.346574`; `2 Delta = ln 2 = 0.693147`; `t = ln s`; per event i,
`l_i(t) = ln L_i(b=0, ln s = t)` on the `combined_no_bh` channel (the registered primary);
`score_lns_i = [l_i(+Delta) - l_i(-Delta)] / ln 2` (PA-HIER-4, line 867); `u = (z - z_g)/sigma_g`.

---

## 0. Plain-language summary

PA-HIER-32(d) subtracts from every event's s-score a number, Es_null_det, that is the exact finite-step bias of the
score for ONE isolated galaxy: with a symmetric step of +/- ln sqrt2 in ln s, even a perfectly correct Gaussian
redshift kernel returns a positive score on average (+0.082 for an untruncated Gaussian, +0.046 once the kernel's
own +/-4 sigma window, redshift floor and volume weighting are included). Tracing that number line by line shows
it has nothing to do with the candidate z-window that T1.3 made theta-consistent: the per-host kernel window was
already theta-transformed before T1.3, and the candidate filter is not an ingredient of the closed form at all —
so Es_null_det is unchanged by T1.3 (pooled mean +0.0463, identical to the forensic's value to 0.001). The problem
is different: Es_null_det is the null expectation of the score of a SINGLE host, whereas the registered score is
taken on an event likelihood that is a sum over hundreds of candidate galaxies (the true host carries a median 0.6 %
of it) plus a completion term. Summing many overlapping kernels makes the likelihood almost flat in s — 22 times
flatter per event than a single host — and the finite-step bias of a flat function is almost zero. Estimating the
actual likelihood's finite-step bias from the three banked s-nodes themselves (via the third Bartlett identity,
which any correctly specified likelihood must satisfy) gives +0.0013 +/- 0.0008 for the T1.3 arm (and -0.0006 +/-
0.0006 for the earlier fixed-window arm): the correct null of score_lns is about zero, in every configuration, not
+0.046 and not +0.027. Subtracting +0.046 therefore manufactures a spurious Z of about -3.3 on a null venue. The
consequence for the record is a registration amendment (PA-HIER-33, section 5) that replaces the per-host
subtraction by the actual-likelihood null estimate, with a cheap decisive falsifier (two extra s-nodes at half the
step); it returns to the author as a [RULE], and only after it is registered can the P1 data be re-read — as a
disclosed re-read of already-seen data, with the pre-registration-violation risk stated. The measured raw score
(+0.004 +/- 0.013) and the with-BH channel (raw +0.031 +/- 0.017; corrected null +0.0004) are both consistent
with that ≈ 0 null; nothing in this note lifts or upholds the P1 STOP.

---

## 1. E13's Es_null_det, line by line — what it assumes and what T1.3 changes (nothing)

### 1.1 Definition as implemented

`hier_s0_driver.py:518-633` (`_es_null_det_closed_form`, `_es_null_det_kernel`; a vectorised port of
`f4_mechanism.py`'s `kern()`/`E()`, verified equal below):

    Es_null_det_i = INTEGRAL_{W_i^-} k_i^0(z) secs_i(z) dz / INTEGRAL_{W_i^-} k_i^0(z) dz
    k_i^s(z)      = N(z; z_g, s sigma_g) x dV_c/dz /(1+z) x f_pix(z) / Z_i(s),  supported on W_i^s = [max(z_g - 4 s sigma_g, 1e-6), z_g + 4 s sigma_g]
    Z_i(s)        = INTEGRAL_{W_i^s} N(z; z_g, s sigma_g) dV_c/dz/(1+z) f_pix(z) dz          (per-node self-normalisation)
    secs_i(z)     = [ln k_i^{sqrt2}(z) - ln k_i^{1/sqrt2}(z)] / ln 2                         (the ln-s secant at FIXED z)
    W_i^-         = W_i^{1/sqrt2}                                                              (the narrower node's support; outside it ln k^{1/sqrt2} = -inf)

Inputs per host: `z_g`, `sigma_g` (catalogue columns), the host's HEALPix completeness `f_pix`, `h = H_TRUE`; the
constants `_ES_NULL_DET_Z_FLOOR = 1e-6` (site 2.2's `_z_lower_floor`) and `_ES_NULL_DET_WINDOW_SIGMA = 4.0`
(`integration_limit_sigma_multiplier`, `bayesian_statistics.py:8101`). It is the expectation, over a redshift drawn
from the host's OWN s = 1 kernel, of the ln-s secant of that same kernel's log-density at the drawn redshift: i.e.
the null expectation of `score_lns` for an event whose entire likelihood is one host's kernel evaluated at a
GW-precise redshift (`sigma_zGW/sd_k` median 0.130, `f4_out.json`).

### 1.2 Why it is non-zero: the ln-s Gaussian scale family's own secant bias (no window needed)

For an untruncated Gaussian, `ln N(u; s) = -ln s - u^2 e^{-2 ln s}/2 + const`, so with `g(t) = l(t)`:

    [g(+Delta) - g(-Delta)] / (2 Delta) = -1 + (u^2/2) sinh(2 Delta)/Delta = -1 + 1.08202 u^2      (2 Delta = ln 2, sinh(ln 2) = 3/4)

whose expectation under `u ~ N(0,1)` is **+0.08202** {t13_esnull2_out.json `gauss_full_line_secant_bias`,
2026-08-30}. Exactly the same number is the KL asymmetry of the family,
`[KL(N_1 || N_{1/sqrt2}) - KL(N_1 || N_{sqrt2})]/ln 2 = (0.153426 - 0.096574)/0.693147 = +0.08202`
{`gauss_KL_asym`}, and its leading Taylor term is `(Delta^2/6) E[g'''(0)] = 0.020022 x 4 = +0.08008`
{`gauss_bias_O(D2)`} — the odd `(ln sqrt2)^2/6 . g'''` term PA-HIER-4 named as the form's "leading error" (line 872)
and did not size. The score identity `E[g'(0)] = 0` holds; the SECANT is not the derivative, and for a
log-likelihood that is not quadratic in ln s its null expectation is `O(Delta^2) != 0`. **This is intrinsic to the
step, not to any window, floor or selection object.**

### 1.3 Decomposition of the +0.0455 (why the number is smaller than +0.082)

| ingredient (in the order the closed form applies them) | value | source |
|---|---|---|
| (a) full-line Gaussian secant bias | +0.0820 | 1.2 |
| (b) restrict the expectation to `W^-` (|u| <= 4/sqrt2 = 2.83; drops the high-u^2 tail that carries the bias) | +0.0371 | `gauss_truncated_Wminus_bias`, 2026-08-30 |
| (c) per-node self-normalisation `Z_i(s)` over `W_i^s` (each node loses only its own > 4 sigma tail, 6e-5 of mass) | < 1e-4 | inspection of 1.1 (same +/-4 sigma in own units at every s) |
| (d) 1e-6 floor + `dV_c/dz/(1+z)` + `f_pix` tilt (host-dependent; at low z the floor truncates `W^0` and `W^-` asymmetrically) | per host 0.0048 - 0.0589; pooled mean **+0.0463 +/- 0.0005** (n 461) | `hier_s0_zwin_run/s0a_seed*/es_null_det.csv`; `t13_esnull_out.json` `es_unw`, 2026-08-30 |

Five representative hosts (z_g quantiles 2/25/50/75/98 %; `t13_esnull2.py`, 2026-08-30; bare = (a)+(b) with the
floor; +vol = adds `dV_c/dz/(1+z)`; cache = the driver's value with completeness; f4 = the forensic's archived value):

| z_g | sigma_g | zeta = z_g/sigma_g | bare Gauss on W^- | +volume | cache | f4 |
|---|---|---|---|---|---|---|
| 0.0446 | 0.0345 | 1.29 | +0.0255 | -0.0109 | +0.0257 | +0.0252 |
| 0.1244 | 0.0372 | 3.35 | +0.0501 | +0.0181 | +0.0431 | +0.0422 |
| 0.1746 | 0.0388 | 4.50 | +0.0382 | +0.0284 | +0.0527 | +0.0517 |
| 0.2366 | 0.0408 | 5.80 | +0.0372 | +0.0337 | +0.0561 | +0.0550 |
| 0.3235 | 0.0438 | 7.39 | +0.0372 | +0.0366 | +0.0456 | +0.0447 |

Driver cache vs the forensic's `f4_events.csv`/`f7_events.csv` `Es_null_det` column on the same 461 (seed,
event_idx) pairs: max |diff| 0.00116, correlation 1.000 {`t13_esnull_out.json` `es_cache_vs_f4`} — the difference
is the T1.3 verifier's MUST_FIX denominator swap (`sqrt2 - 1/sqrt2` -> `ln 2`, a 1.0201 factor;
`T1_3_ZWINDOW_VERIFIER_REPORT.md` item 3) plus grid effects; the two implementations agree. The P1 pooled mean
+0.0463 vs E13's +0.0455 on the forensic's own event set (`f4_out.json` `Es_null_det_unweighted`): same quantity.

### 1.4 Which of these assumptions the theta-consistent window removes: none

T1.3's registered change is to the CANDIDATE z-filter, `handler.py:668-676` (`PHYSICS_CHANGE_THETA_ZWINDOW_
20260830.md` sections 1b, 2.1): which rows in the sky cone are summed, decided from the theta-transformed kernel's
+/- k sigma support against the GW envelope, at k = 4. The objects in 1.1 are: one host, its kernel, its kernel's
own window `W_i^s`, the floor, the volume element, its pixel completeness, and the step. The per-host kernel
window `W_g^theta = [z_g^theta - 4 sigma_g^theta, z_g^theta + 4 sigma_g^theta]` was ALREADY theta-transformed at
site 2.2 before T1.3 (`bayesian_statistics.py:8116-8117, :8126, :8143-8144`; forensic E7: "window and Z_g on the
theta-kernel"); the candidate filter does not appear in 1.1 at all (a single host that is a candidate at every
node — 99.65 % of the s = 1 kernel mass lies inside `W^-`, forensic section 4 item (3) — sees no filter). Therefore:

- the truncation to `W^-` (b) is the kernel's own support, theta-consistent before and after T1.3 — unchanged;
- the 1e-6 floor (d) is site 2.2's `_z_lower_floor`, applied to the theta-kernel — unchanged; the 1e-10 floor
  named in the task lives only in site 2.3's smeared kernel (forensic section 4 item (3)) and is not in the closed
  form;
- the `S_bar_phi` factor enters only the non-registered "gen" variant (`Es_gen_det`, survival-weighted expectation;
  `f4_mechanism.py`), not the registered `Es_null_det`; after T1.1 the divisor `rho(theta)` is a per-node scalar on
  the population sum (`bayesian_statistics.py:4834`), invisible to a per-host secant at fixed z;
- the +/- ln sqrt2 step is the same PA-HIER-4 grid.

**Finding 1.** `Es_null_det_i` as defined is configuration-free and NOT ≈ 0 under T1.3 — exactly as PA-HIER-32(d)'s
scope note states ("per host and configuration-free"). The premise "the theta-consistent window removes it" is
false for the quantity as defined. What T1.3 removed is a different term entirely — the capture term of the
impostor mixture (`PHYSICS_CHANGE_THETA_ZWINDOW` section 2.3, model -0.074 to -0.105) — which was never part of
the closed form. The defect is upstream of T1.3: `Es_null_det_i` is the null expectation of a single-host score,
and the registered statistic is not a single-host score (section 2).

---

## 2. The correct null expectation of score_lns under T1.3

### 2.1 Exact form

For a correctly specified family `P_i^t` of the event data under `ln s = t` (the null: generator kernel = estimator
kernel at t = 0, the b0i venue's construction, forensic E1/E3), with `E_0` the expectation under `t = 0`,
`E_0[l_i(t)] - E_0[l_i(0)] = -KL(P_i^0 || P_i^t)`, hence

    E_0[score_lns_i] = [ KL(P_i^0 || P_i^{-Delta}) - KL(P_i^0 || P_i^{+Delta}) ] / ln 2        (exact)
                     = (Delta^2 / 6) . E_0[ d^3 l_i / dt^3 |_0 ] + O(Delta^4)                       (Taylor; E_0[l'] = 0, the even l'' term cancels)

It is zero iff the family's KL is symmetric in +/-Delta — true for a log-likelihood quadratic in ln s, false in
general. It reduces to `Es_null_det_i` when, and only when, `l_i(t)` IS the single host's log-kernel at a fixed z:
i.e. (i) the true host carries the whole catalogue leg (`pi_true = 1`) and (ii) the catalogue leg carries the whole
combined channel (`c_i = 1`). On b0i: `pi_true` median 0.006 (forensic E14, `f7_events.csv`), `c_i` mean 0.616 /
median 0.659 (P1 truth nodes, `t13_esnull_out.json` `c_stats`). The T1.3 gate doc's c-weighted convention
(section 5.6, `c_i x Es_null_det_i`) repairs (ii) to first order and leaves (i) untouched.

### 2.2 Why the mixture's term is small: the dense-comb limit

The catalogue leg is `SUM_g w_g INTEGRAL k_g^s(z) p_GW(z) dz / rho(s)`. For candidates whose listed redshifts are
dense on the scale `sigma_g` (T1.2 median `n_cand` 278, registered to grow by 2.2-3.6x under the k = 4 widening,
`PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md` 5.6; forensic E6/E9 and
`PHYSICS_CHANGE_THETA_ZWINDOW` 5.6 "count growth"), `SUM_g w_g N(z; z_g, s sigma_g)` is a smoothing of a slowly
varying density and is nearly s-INVARIANT: both its derivative and its secant in ln s vanish pointwise, and the
divisor `rho(s)` is a per-node scalar. Measured per-event curvature confirms it: the pooled Fisher information in
ln s of the combined channel is `I_hat = -<l''> = 0.0890` per event on P1 {`t13_esnull_out.json` `bartlett_nb`
`I_hat`, 2026-08-30} against `I = 2` for an isolated Gaussian host — **22x flatter**. The third derivative that
sets the secant bias is correspondingly small.

### 2.3 Zero-compute estimate from the three banked nodes: the third Bartlett identity

Any correctly specified likelihood satisfies `E_0[l''' + 3 l' l'' + l'^3] = 0` (differentiate `INTEGRAL P^t = 1`
three times). Check on the Gaussian: `E[l'''] = 4, E[l' l''] = -4, E[l'^3] = 8` -> `4 - 12 + 8 = 0`. So

    Es_null^{(cfg)} := (Delta^2/6) . E_0[l'''] = (Delta^2/6) . ( -3 E_0[l' l''] - E_0[l'^3] ),

and both moments are estimable per event from the three s-nodes already on disk:
`l'_i ≈ score_lns_i` (central secant), `l''_i ≈ [l_i(+Delta) - 2 l_i(0) + l_i(-Delta)]/Delta^2` (their own
finite-difference errors are `O(Delta^2)`, i.e. `O(Delta^4)` in the bias). Results (`t13_esnull.py`,
`t13_esnull2.py`, `t13_esnull3.py`; 2026-08-30; bootstrap SD from 4000 event resamples):

| arm / channel | n | `<l' l''>` | `<l'^3>` | `E[l''']_hat` | `I_hat` | **Es_null^{(cfg)} = 0.020022 x E[l''']_hat** | 5 %-trimmed | per seed (900101/02/03/04) |
|---|---|---|---|---|---|---|---|---|
| **P1 (T1.3, theta-consistent k = 4), no-BH** | 461 | -0.03206 | +0.03069 | +0.0655 | 0.0890 | **+0.0013 +/- 0.0008** (95 % CI [-0.0001, +0.0030]) | -0.0003 | +0.0031 / +0.0009 / +0.0017 / -0.0001 |
| P1, with-BH | 461 | -0.01679 | +0.03246 | +0.0179 | 0.1238 | **+0.0004 +/- 0.0011** | -0.0017 | -0.0007 / +0.0022 / +0.0008 / -0.0008 |
| T1.2 recert (theta-blind k = 1), no-BH | 461 | — | — | -0.0292 | 0.2449 | **-0.0006 +/- 0.0006** | — | — |
| T1.2 recert, with-BH | 461 | — | — | +0.0205 | 0.2420 | +0.0004 +/- 0.0008 | — | — |

**Finding 2.** The null expectation of `score_lns` for the ACTUAL event likelihood is ≈ 0 — `+0.0013 +/- 0.0008` on
P1, a tenth of the arm's SEM (0.0129) — and it is ≈ 0 in the theta-blind configuration too (`-0.0006`). Neither
+0.0463 (PA-HIER-32(d) as implemented) nor +0.0270 (`<c_i Es_null_det_i>`, the gate doc's c-weighted proposal;
`t13_esnull_out.json` `c_x_es`) is the null of the registered statistic; they overstate it by ≈ 35x and ≈ 20x.

### 2.4 Consistency of the estimator with E13 in the limit where E13 applies

On the true-host-dominated subset the same estimator moves toward the single-host value, as it must
{`t13_esnull2_out.json` `bartlett_pi_true>*_nb`, 2026-08-30}:

| subset | n | `I_hat` | Bartlett `Es_null` | `<c_i Es_null_det_i>` | ratio |
|---|---|---|---|---|---|
| all | 461 | 0.089 | +0.0013 | 0.0270 | 0.05 |
| `pi_true > 0.2` | 53 | 0.359 | +0.0081 | 0.0289 | 0.28 |
| `pi_true > 0.5` | 24 | 0.545 | +0.0147 | 0.0313 | 0.47 |

(`pi_true > 0.5` has `c` mean 0.79, so the full single-host limit would be `I ≈ 2 c^2 ≈ 1.25`; the subset reaches
44 % of that information and 47 % of the c-weighted single-host bias.) The single-host closed form is the
`pi_true -> 1, c -> 1` corner of the general formula; b0i lives at the opposite corner.

### 2.5 Caveats on the Bartlett estimate (disclosed, none changes the order of magnitude)

1. The moments are sample averages over the realised P1 data, i.e. computed as if H0 held. A null expectation is by
   definition an H0 object; under H1 the identity fails, but so would any null correction — that is what the
   registered band then detects. Under H1 of the size seen (mean `l'` ≈ 0.004-0.07), the induced error in
   `-3<l' l''>` is `≈ 3 x mean(l') x I_hat ≈ 0.001-0.02` in `E[l''']`, i.e. `< 0.0004` in the bias.
2. Finite-difference `l'`, `l''` carry `O(Delta^2)` relative errors, and at this step they are not small on a
   strongly curved family: on the Gaussian test family the same 3-node estimator returns 0.0663 vs the exact
   0.0820 (19 % low; `E[l''']_hat` 3.31 vs 4; checked numerically on a 400001-node grid, 2026-08-30). Taken as a
   worst-case relative error, the P1 value +0.0013 is at most ≈ +0.0017 exact-equivalent — still a tenth of the
   SEM. The Richardson falsifier of section 5 is free of this error by construction.
3. The estimate is of the POOLED null mean; it is not a per-event correction (per-event values cannot be resolved
   from three nodes, and need not be for a mean/SEM band).
4. Cancellation structure of the realised data (REPORTED-ONLY, not adjudicated): the P1 `score_lns` pooled +0.004
   is the sum of opposite-sign strata — c-quartiles +0.023 +/- 0.008 / +0.102 +/- 0.017 / +0.094 +/- 0.019 /
   **-0.234 +/- 0.029**; `z_g < 0.125`: -0.147 +/- 0.034 vs +0.055 / +0.073 / +0.003 above {`t13_esnull_out.json`
   `by_c`, `by_z_g`, 2026-08-30} — the same E17 class the forensic named ("a cancellation of two opposite-sign
   classes"). A pooled |Z| <= 3 band does not test strata; PA-HIER-33 registers the c-stratified read as a
   REPORTED-ONLY companion, not as a band (the prereg's own E16 warning: condition on `z_g`/`c`, never on `z_true`).
5. `L_cat_no_bh` in the node CSVs is the divisor-normalised catalogue term (the ratio
   `(combined . D_tilde - B_num)/L_cat_no_bh` is node-invariant to 1e-9 across truth/s+/s-, `t13_esnull2_out.json`
   `beta_over_rho_*`; the divisor multiplies the population sum at `bayesian_statistics.py:4834`), so the
   catalogue-leg secants quoted in this note already contain `rho(theta)`.

---

## 3. What the P1 data read under each null (numbers only — NOT an adjudication)

All from the P1 nodes of record (`hier_s0_zwin_run`, 4 seeds x {truth, s_plus, s_minus}, `theta_sites=2.2`, smear
off, `theta_phi_divisor=on`, `sky_cone_k=1.5`, `theta_zwindow=on`, `z_window_k=4.0`, h = 0.73; `logs/
runner7_tree2_20260830.log`, 2026-08-30), n = 461 (456 matched + 5 dark scoring exactly 0.0). The driver's own
numbers are reproduced to the last digit by `t13_esnull.py` (`score_lns` +0.0039648 +/- 0.0128936, Z +0.3075;
`score_s` -0.0423709 +/- 0.0127515, Z -3.3228; `s0a_score_output.json`).

| statistic (no-BH, registered primary channel) | null used | mean | per-event SEM | Z | seed-clustered SEM / Z (PA-HIER-5, 4 clusters) |
|---|---|---|---|---|---|
| `score_lns` (raw, PA-HIER-4) | 0 | +0.0040 | 0.0129 | +0.31 | 0.0159 / +0.25 |
| `score_s` = `score_lns - Es_null_det_i` (driver; PA-HIER-32(d) literal) | +0.0463 (per host) | **-0.0424** | 0.0128 | **-3.32** | 0.0154 / -2.75 |
| `score_lns - c_i Es_null_det_i` (gate-doc 5.6 proposal) | +0.0270 | -0.0231 | 0.0129 | -1.79 | 0.0158 / -1.46 |
| `score_lns - Es_null^{(P1)}` (section 2.3; PA-HIER-33 form) | +0.0013 +/- 0.0008 | +0.0027 | 0.0129 (+0.0008 in quadrature: 0.0129) | +0.21 | 0.0159 / +0.17 |

Per-seed raw `score_lns`: +0.0039 / -0.0350 / +0.0448 / +0.0070 (SEM 0.026 / 0.027 / 0.031 / 0.020)
{`t13_esnull2_out.json` `per_seed_nb`}. Against the gate doc's registered P1 predictions (`PHYSICS_CHANGE_THETA_
ZWINDOW_20260830.md` 5.6): c-weighted point -0.026 +/- 0.012, band of the point [-0.031, +0.005] — measured -0.023;
raw form [0.000, +0.031] — measured +0.004; the driver's unweighted form had no 5.6 prediction (it is the PA-HIER-
32(d) literal, flagged as convention-open by Revision note 2 item 3). Every convention is inside its own
registered prediction; the ONLY one outside |Z| <= 3 is the one whose null this note finds mis-scaled by ≈ 35x.
The T1.2 re-certification's STOP (raw `score_lns` -0.0734 +/- 0.0123, Z -5.97 in ln-s units; -0.07196 in the raw
linear form of record) fails the band under every null in this table (Z between -6.1 and -9.7), so the amendment
below disturbs neither T1.2's s-axis STOP nor T1.1's b-axis certification (b-nodes not re-run under P1).

**This table does not lift the STOP.** Under the registered rule as it stands (PA-HIER-32(d)), the P1 verdict is
B0-A'; the table shows what the same data read under an amended null, and is the pre-registration-violation
exposure of section 5 stated in numbers.

---

## 4. The with-BH channel as a cross-check (REPORTED-ONLY; invariant 12, forensic section 3)

| statistic (with-BH) | mean | per-event SEM | Z | clustered SEM / Z |
|---|---|---|---|---|
| `score_lns` raw | +0.0309 | 0.0167 | +1.85 | 0.0198 / +1.56 |
| driver `score_s` (unweighted `Es_null_det`) | -0.0155 | 0.0165 | -0.94 | 0.0192 / -0.80 |
| c-weighted | +0.0038 | 0.0166 | +0.23 | 0.0197 / +0.20 |
| Bartlett null `Es_null^{(P1,wb)}` = +0.0004 +/- 0.0011 -> corrected | +0.0305 | 0.0167 | +1.82 | — |

The with-BH channel's own null expectation is ≈ 0 by the same estimator (`I_hat` 0.124, `E[l''']_hat` +0.018), so the
channel's raw Z (+1.85) IS its corrected Z; every convention sits inside |Z| <= 3. The cross-check confirms the
no-BH conclusion in kind (the secant's null bias is negligible on a mixture likelihood in either channel) and adds
nothing on the P1 verdict: on the 1-D b0i venue the with-BH channel is uninterpretable (donor-row mass, forensic
section 3; the b0i2d object fixes it) and carries no band.

---

## 5. PA-HIER-33 — proposed registration amendment (returns to the author as a [RULE])

**Status: PROPOSAL. Not registered; not in force. The P1 verdict of record remains B0-A' under PA-HIER-32(d)
until this amendment is ratified.** Presented in registration form so a one-word reply is unambiguous.

**Supersedes (on ratification).** PA-HIER-32(d)'s "Correction, registered" block insofar as it defines the
subtrahend as the per-host single-host closed form `Es_null_det_i`; PA-HIER-32(d)'s "Bias-free argument"
paragraph ("E[score_s | generator kernel = estimator kernel] = 0 by construction") — shown in section 2 to hold only
in the `pi_true -> 1, c -> 1` corner; `PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md` 5.6's PROPOSED c-weighted
convention (Revision note 2 item 3), which becomes moot. Untouched: PA-HIER-4's `score_lns` form and nodes; `score_b`;
every band structure of section 4.1; T1.1/T1.2's b-axis certification; T1.2's s-axis STOP.

**Rule.** For an arm with s-nodes at `ln s = 0, +/-Delta` (`Delta = ln sqrt2`), define on each channel

    Es_null^{(arm)} = (Delta^2/6) . [ -3 <l'_i l''_i> - <l'_i^3> ],   l'_i = score_lns_i,   l''_i = [l_i(+Delta) - 2 l_i(0) + l_i(-Delta)]/Delta^2,
    score_s_i = score_lns_i - Es_null^{(arm)}      (a pooled scalar shift, the arm's own null; NOT a per-host table),
    Z_s = mean(score_s) / SEM,   SEM = max(per-event SEM, seed-clustered SEM)   (PA-HIER-5 leg (a)),
    with the bootstrap uncertainty of Es_null^{(arm)} added in quadrature to the SEM.

`Es_null^{(arm)}` is computed from the arm's own three nodes at zero compute (the third Bartlett identity is the
closed form; section 2.3), re-derived per configuration as PA-HIER-32(d)'s scope note already requires, and banked
with its bootstrap SD before the band is read. Registered values for P1: `Es_null^{(P1,nb)} = +0.0013 +/- 0.0008`,
`Es_null^{(P1,wb)} = +0.0004 +/- 0.0011` (section 2.3) — recorded here BEFORE any re-read is licensed.

**Band (A8 two-sided, unchanged in structure).** `|Z_s| <= 3.0` on `combined_no_bh` -> B0-A; `|Z_s| > 3.0` ->
B0-A' (INSTRUMENT-DEFECT, STOP) — the existing section 4.1/4.5 table, restated in this `score_s`. Two-sided by
construction (the null is a point, 0 after subtraction).

**A15 at N = 461 (the arm's own N, its own scatter).** Per-event SEM 0.0129 (P1 measured; `Es_null` bootstrap
0.0008 adds nothing at this precision); false-fail under the exact null 0.27 % two-sided; 80 % power at
`3.84 x 0.0129 = 0.0495` per unit ln s (per-event SEM) or `3.84 x 0.0159 = 0.061` (clustered SEM, the binding one
by PA-HIER-5). The alternatives this band would have missed at P1 — a residual of the c-weighted single-host size
(0.027) — sit at 2.1 SEM: disclosed as below the 80 %-power point (A8 band-derivation disclosure); the falsifier
below resolves them independently at ≈ 0.001 precision.

**A8 checks.** Branch referent: the band is satisfiable by the P1 arm's own nodes (truth, s+, s-) and by any
future s-arm; two-sidedness: yes; execution-completeness: the Richardson arm below is the only registered arm
capable of changing the count — it is run BEFORE the re-read, or withdrawn by an author [RULE].

**Falsifier (registered before any run; decisive, model-free).** Add two s-nodes at `ln s = +/-Delta/2`
(s = 2^{+/-1/4} = 1.1892 / 0.8409), same 4 seeds, same flags as P1 (8 cells; P1 s-cells cost 705-844 s evaluate
each, `logs/runner7_tree2_20260830.log` -> ≈ 1.7 h serial, ≈ 0.5 h wall at 4-way; truth nodes reused). Per event,
`S_i(Delta/2) = [l_i(+Delta/2) - l_i(-Delta/2)]/Delta` and the Richardson secant `S_R,i = [4 S_i(Delta/2) -
S_i(Delta)]/3` has NO `O(Delta^2)` term for ANY smooth `l_i` (Gaussian check: `E[S_R] = -0.0005` vs
`E[S(Delta)] = +0.0820`; the paired difference `E[S_R] - E[S(Delta)] = -0.0825` returns the full secant bias with
the opposite sign; 400001-node grid, 2026-08-30). Prediction under this amendment:
`mean(S_R) - mean(score_lns) = -Es_null^{(P1)} = -0.0013 +/- 0.0008` (to `-0.0017` allowing the finite-difference
underestimate of 2.5 item 2); under PA-HIER-32(d)'s single-host null the same paired difference would be
`-0.046`, under the c-weighted null `-0.027`. The SEM of the paired difference is set by the per-event scatter
of `l'''` (expected `~ 0.001` at N = 461 for a mixture this flat; the single-Gaussian scatter, `SD(4u^2) = 5.7`,
would give 0.005) and is measured, not assumed. Rule:
PA-HIER-33 is REFUTED if `|[mean(S_R) - mean(score_lns)] + Es_null^{(P1)}| > 3 SEM_paired`; it is CONFIRMED
otherwise; and in EITHER case `S_R` becomes the s-statistic of record for the arm (it needs no null correction at
all — a Delta^2-free secant is the cleanest instrument, and the 5-node s-grid then also yields `l'''` directly).
If the author prefers not to spend the 8 cells, the amendment stands on section 2 alone, with the falsifier
recorded as UNRUN and the band's dependence on the Bartlett identity disclosed as the residual assumption.

**Companion reads (REPORTED-ONLY, no band).** `score_s` by c-quartile and by `z_g` bin (conditioning on
pre-selection quantities only, per E16); `I_hat` and `E[l''']_hat` per channel; the with-BH channel's `score_s`.

**Re-read of P1 under PA-HIER-33 — disclosure of the pre-registration-violation risk.** The P1 numbers were seen
(section 3) before this amendment was drafted; reading them under the amended null is a re-read of already-seen
data. Mitigations, stated so the author can weigh them: (i) the amended null is derived from an identity every
correctly specified likelihood obeys, with the single-host closed form recovered in its limit (2.4), not fitted to
the outcome; (ii) it moves the pooled null by +0.0013, a tenth of the SEM — the raw `score_lns` band read is
unchanged in kind by it, and the raw form was itself registered in the gate doc's 5.6 as a reported convention
with a prediction ([0.000, +0.031]) that P1 met; (iii) the Richardson arm is a fresh, unseen measurement that can
refute the amendment on its own. The author's options, each a [RULE]: (a) ratify PA-HIER-33 and re-read P1 under
it, disclosed as a re-read; (b) ratify PA-HIER-33 and require the Richardson arm BEFORE any re-read (recommended —
it converts the re-read into a read of fresh data on a Delta^2-free statistic); (c) decline, leaving PA-HIER-32(d)
and the B0-A' verdict in force — in which case section 2's finding that the registered null is mis-scaled must be
recorded against the verdict as a known instrument defect of the STATISTIC, and the STOP's disposition "a bug in
the hook, the venue, or GATE PARITY" (section 4.5) is amended to name the statistic's null as the located bug.

**What returns to the author.**
- [RULE] PA-HIER-33 as written (the amendment itself is a registration change on a registered rule).
- [RULE] the open Revision-note-2 item 3 convention question, which PA-HIER-33 answers (neither raw-unweighted nor
  c-weighted; the arm's own Bartlett null) — ratifying PA-HIER-33 closes it.
- [DO] the 8-cell Richardson falsifier arm (option (b)); runner != this node; scored by an independent reader.
- [RULE] whether the P1 data are re-read under PA-HIER-33 (option (a)/(b)) or the B0-A' stands (option (c)).
No Stage-P/F, S0-B, C1/C3 or production change is licensed by anything in this note.

---

## 6. Answers to the four questions, in one place

1. **Assumptions behind E13's non-zero Es_null_det:** the finite +/- ln sqrt2 secant of a log-density that is not
   quadratic in ln s (the Gaussian scale family: +0.082 untruncated, the KL asymmetry), reduced by the expectation's
   restriction to the narrow node's support (+0.037) and re-shaped per host by the 1e-6 floor and the
   volume/completeness tilt (0.005-0.059, mean +0.0463). The theta-consistent window removes NONE of them: the
   candidate filter is not an object in the closed form, and the kernel window was theta-consistent before T1.3.
2. **Correct null under T1.3:** `E_0[score_lns] = [KL(P^0||P^-) - KL(P^0||P^+)]/ln 2 = (Delta^2/6) E_0[l'''] +
   O(Delta^4)` — not identically 0, but for the actual mixture likelihood ≈ 0: **+0.0013 +/- 0.0008** (no-BH),
   +0.0004 +/- 0.0011 (with-BH), from the three banked nodes via the third Bartlett identity; the mixture is 22x
   flatter in ln s than a single host, and the single-host value is recovered only as `pi_true -> 1`. The same
   estimate under the theta-blind window is -0.0006 +/- 0.0006: the registered subtrahend was the wrong object in
   both configurations, not a configuration-specific term.
3. **Consequence:** PA-HIER-32(d)'s `Es_null_det_i` subtraction overstates the null by ≈ 35x and by itself
   produces Z ≈ -3.3 on a null venue at N = 461; PA-HIER-33 (section 5) must be registered by an author [RULE]
   before any re-adjudication; the P1 verdict stays B0-A' on the record until then; the recommended path runs the
   8-cell Richardson falsifier so the eventual read is of fresh data on a Delta^2-free statistic.
4. **With-BH cross-check:** raw Z +1.85, driver-corrected -0.94, c-weighted +0.23, Bartlett null ≈ 0 (+0.0004) —
   consistent with the no-BH finding; REPORTED-ONLY under invariant 12.

---

## 7. Provenance

- New numbers: `tree2_20260830/t1_3_es_null_det_work/{t13_esnull.py, t13_esnull2.py, t13_esnull3.py}` and their
  `*_out.json` (2026-08-30); inputs are the P1 and T1.2 node CSVs (read-only), the P1 `es_null_det.csv` caches
  (read-only), `f7_events.csv` (read-only). No file under `hier_s0_zwin_run/` or any B8.2 file was written.
- Driver reproduction: `score_lns`, `score_s` (no-BH and with-BH) match `s0a_score_output.json` to all printed
  digits.
- Gaussian reference values: analytic (1.2), cross-checked numerically on a 200001-node grid.
- Worker: [FABLE-ORCH] inherit-tier derivation node, 2026-08-30; no git operations; no source edits; foreground only.

---

## ADOPTED STAMP — 2026-08-31

**PA-HIER-33 RATIFIED by the author, ledger row #278 item (1)** (verbatim ruling "I approve all
decisions and suggestions for the next steps", itemization orchestrator-derived per the
approval-scope convention). The corrected null — the arm's own likelihood at its own drawn
parameters, not the mis-scaled single-host `Es_null_det` offset — is the rule of record for the
[HIER] s-score. Consequences of record: the [HIER] instrument is CERTIFIED on both axes at the
T1.3 configuration; the P1 B0-A′ row re-adjudicates under this rule by appended note (not edit);
S0-B is unblocked (docket §5 item 3 precondition "ONLY AFTER PA-HIER-33 IS RULED" is now met),
pending the scorer-side implementation of the amended rule before any S0-B read.
