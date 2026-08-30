# /physics-change PRESENTATION GATE — [HIER] theta-consistent no-BH global-selection divisor Sigma_phi(theta) + the sky-cone-radius instrument flag — 2026-08-30

**Launched under row #255 — tree 2 node T1.1.** Approval stamp: **row #255 (standing grant, tree 2 node T1.1)**
(author verbatim "all ratified from the docket", A3 = [DO]+[RULE] "the theta-consistent no-BH divisor Sigma^phi(theta)
gate presentation (site 2.3 extended to the phi-table branch, byte-identical at theta=(0,1)) + the sky-cone-radius flag
= first node of tree 2"; A17 = the standing grant whose scope covers instruments and production-default flips inside the
tree, each with its gate presentation BEFORE code and its ledger rows).
Branch fix/p32d-classg-venue-repair, HEAD ecd33336. Presenter: top-tier subagent; **no code is written under this node**
(presentation before code, the B7.3 precedent, docs/gates/PHYSICS-GATE-LEDGER.md rows dated 2026-08-29). Builder must be
a different agent from this presenter, and the T1.2 re-certification runner a different agent from the builder
(builder != runner). Every [HIER] statement carries the REPORTED-ONLY cap (PA-HIER-28 item 9). No backtick characters
in this record. Every number below carries {value, source file:line or artefact, date}.

Companion zero-compute instruments (this node, foreground, zero evaluate() calls, catalogue md5 pin verified):
results/campaign51_20260728/realistic_20260729/tree2_20260830/t1_1_gate_work/t11_census_timing.py -> t11_census_timing_out.json
(E20 census reproduced bit-for-bit; S0-B-node census; C7-kernel smear wall-time) and t11_pool_stats.py -> t11_pool_stats_out.json
(pool extremes; zero-error rows; degenerate windows at truth).

---

## 0. Scope and one-paragraph summary

The S0-A control (mirror b0i, generator kernel == estimator kernel at theta=(0,1) by construction) returned
score_b = -1.616 +/- 0.440 (Z -3.68) and score_s = -0.0863 +/- 0.0122 (Z -7.08) on the registered no-BH channel
(N = 461; s0a_score_output.json; B1_1_HIER_STAGE0_RECORD.md section 2.1; 2026-08-29) -> B0-A' INSTRUMENT-DEFECT -> STOP.
The forensic (B1_1_S0A_DEFECT_FORENSIC_20260829.md section 0, E7, E10, E11) localised the b-axis to the instrument's own
FORM: theta enters the site-2.2 per-host numerator kernel (bayesian_statistics.py:7119-7129) but the no-BH catalogue
divisor Sigma_phi = sum_g w_g S_bar_phi(z_g; h) (:2906-2915, :3022; consumed at :5215-5219 and :5296-5298) carries no
theta in any built form; the score at truth-theta is therefore, to first order, <c_i> d/dtheta ln Sigma_phi(theta) != 0
BY CONSTRUCTION. Restoring the divisor's theta-dependence post hoc with a per-node scalar rho(theta) moves the registered
b-statistic from -1.634 +/- 0.444 (Z -3.68) to -0.268 +/- 0.431 (Z -0.62) (E11, f12_out.json corrected_combined_pool).
This presentation registers the divisor's theta-consistent form, proves it is byte-identical at theta=(0,1), derives the
score identity it restores (A12), states that the per-node scalar rho(theta) is EXACT (not first-order) with respect
to the registered form, adds the sky-cone-radius instrument flag (default 1.5, byte-identical; bayesian_statistics.py:4869
-> handler.py:662) needed for the s-axis truncation test (E9/E12), costs the change from MEASURED anchors (this node
measured the C7-kernel pool smear at 27.3-31.8 microseconds per row; 569-662 s per full-pool pass), registers the
regression plan, the A10 invariants/blindness, and the A14 falsifier: the S0-A re-certification (T1.2) must return
|Z_b| <= 3 (point prediction Z_b = -0.62, score_b = -0.27 +/- 0.43) or the mechanism-(i) attribution is REFUTED.

Site label used below: **site 2.3phi** = the phi-table leg of the global-selection precompute (the leg site 2.3 never
reached: the phi-table branch at :2906-2915 precedes and excludes the only theta-receiving branch at :2916-2934).

---

## 1. OLD formula (exact, as implemented at HEAD ecd33336, with lines)

**(1a) The phi-marginal survival table — theta-free by construction, UNCHANGED by this presentation.**
precompute_phi_marginal_survival (bayesian_statistics.py:1982-2075):

    S_bar_phi(z; h) = INTEGRAL phi(log10 M) S_4D(d_L(z; h), M (1 + z)) dlog10 M          (:2030-2061, trapezoid,
    z_grid = linspace(1e-6, z_max(h), 1500), 600 log10-M nodes; table[h] = (z_grid, s_phi)   :2062)

theta never enters: the table is a function of z alone. Zero additional table builds are needed per theta node (section 6).

**(1b) The no-BH catalogue divisor Sigma_phi — point-evaluated, theta-INERT (the defect locus, mechanism (i)).**
precompute_global_catalog_selection (:2692-3025), phi-table branch:

    :2906    if phi_survival_table is not None:
    :2914        _z_phi_grid, _s_phi_grid = phi_survival_table[h]
    :2915        p_det = np.interp(z_g, _z_phi_grid, _s_phi_grid)
    :3022    global_table[h] = float(np.sum(w_g * p_det))            # Sigma_phi(h) = sum_g w_g S_bar_phi(z_g; h)

with w_g = R_eff_per_mbh(M_g) / (1 + z_g) (:2903) on the eligibility mask (z_g < z_max(h)) & isfinite(M_g) & (M_g > 0)
(:2880). The only theta-receiving branch is the smear branch:

    :2916    elif smear_sigma_z:
    :2917-2934   p_det = _smeared_global_pdet_expectation(..., theta_b=theta_b, theta_s=theta_s)

which the phi branch precedes and excludes, and which the guard at :2799-2806 ties to smear_sigma_z=True. The call site
of the phi divisor passes neither theta nor smearing:

    :4234    _global_cat_selection_phi = precompute_global_catalog_selection(
    :4235-4243   ..., with_bh_mass=False, z_max_cap=REDSHIFT_UPPER_LIMIT, smear_sigma_z=False,
                 phi_survival_table=_phi_survival_table)                      # no theta_b / theta_s kwargs

and its site-2.3 dispatch (:4175-4177) forwards theta only to the two point-evaluated legs Sigma_3D / Sigma_4D
(:4178-4199), never to :4234. Under the registered CoR-P/CoR-M form (theta_sites="2.2", smear_global_selection=False,
A2(a)/(b) row #255) theta reaches NO global object at all.

**(1c) The consumer — the no-BH L_cat divisor (the site that must become theta-consistent).**
p_Di (:5026-...):

    :5215    global_denom_no_bh: float = (
    :5216        self._global_cat_selection_phi.get(self.h, 0.0)
    :5217        if getattr(self, "_catalogue_global_selection", "s3d") == "phi"
    :5218        else self._global_cat_denom_no_bh.get(self.h, 0.0)
    :5219    )
    :5296    L_cat_without_bh_mass = (
    :5297        cat_num_sum_no_bh / global_denom_no_bh if global_denom_no_bh > 0 else 0.0
    :5298    )

**(1d) The other consumer of Sigma_phi — path-(A) mixture objects, which this presentation leaves at theta=(0,1).**

    :5774    sigma_phi = self._global_cat_selection_phi.get(self.h, 0.0)
    :5775    path_a = path_a_mixture_objects(beta_G_phi, beta_Gbar_phi, sigma_phi, global_denom_with_bh)
    (path_a_mixture_objects :2441-2500: n_hat_w_phi = Sigma_phi / beta_G_phi; r_Malm = Sigma_4D / Sigma_phi;
     alpha_G_phi = Sigma_4D / n_hat_w_phi; D_tilde_phi = alpha_G_phi + beta_Gbar_phi; w_tilde_G = alpha_G_phi / D_tilde_phi)

and the log/JSON emitters at :4244-4270 (sigma_phi=_global_cat_selection_phi[_h_phi]).

**(1e) The site-2.2 numerator hook whose theta the divisor must follow (single_host_likelihood_batch).**

    :7117    sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    :7118    host_z_error_eff = np.sqrt(host_z_error**2 + sigma_z_pv**2)
    :7119    if theta_b != 0.0 or theta_s != 1.0:
    :7128        host_z_error_eff = np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2)
    :7129        host_z = host_z + theta_b * (1.0 + host_z)
    :7138    den_hi = host_z + integration_limit_sigma_multiplier * host_z_error_eff          # 4.0 (:7113)
    :7155    den_lo = np.maximum(host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor)   # 1e-6
    :7190    f_host_den = _completeness_at_host_nodes(completeness_model, y_den, host_pixels, h)   # C7-core f_k, ZoA rows -> 1.0
    :7208    z_prior_norm = _batched_gl_reduce(den_lo, den_hi, _GL_WEIGHTS_50, gauss_den * w_pop_den)   # Z_g, w_pop_den = dVc/dz/(1+z) * f_k
    :7209    z_prior_norm = np.where(z_prior_norm <= 0.0, 1.0, z_prior_norm)                    # the E20 guard
    :7366    _num_integrand = _num_integrand * np.interp(y_num_nodes, _z_s, _s_phi)              # "phi" numerator (endpoint-clamped interp)

**(1f) The sky-cone radius — hardcoded.**

    bayesian_statistics.py:4869        sigma_multiplier=1.5,  # type: ignore[arg-type]
    handler.py:662                     radius = float(sigma_multiplier * np.sqrt(max(lambda_max, 0.0)))
    handler.py:668-677                 redshift_filter_mask: z_min <= z_g + 1 * z_err_g  and  z_max >= z_g - 1 * z_err_g

(after B5.1, sigma_multiplier feeds ONLY the sky-cone radius; the mass window is mass_filter_k — handler.py:598-609.)

**(1g) The site-2.3 smear kernel as built (for the record — NOT reused by this presentation).**
_smeared_global_pdet_expectation (:1638-1760) smears with the BARE kernel N(z; z_g, sigma_eff) * dVc/dz/(1+z)
(:1718-1721) — it predates C7-core and carries NO f_k(z) factor, so it is not the site-2.2 numerator kernel of :7183-7208.
This pre-existing num/denom asymmetry of the smeared branch is disclosed and left untouched (it is absent from the form
of record, smear_global_selection=False, A2(a)/(b)); the harness twin kernel_smeared_survival
(correspondence_1d.py:1248-1345) DOES carry f_k (A20 review finding 1: the bare form mis-centres z_true by +0.32 sigma
and biases S_tilde by ~3 %).

---

## 2. NEW formula (the registered form)

### 2.1 Definitions, per h and per theta = (b, s), for every eligible catalogue row g

    z_g^theta      = z_g + b (1 + z_g)                                                    (HIER section 1.2; Ma-Hu-Huterer 2006 sec. 2)
    sigma_pv,g     = (1 + z_g) SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S                     (from the UNSHIFTED z_g — A9 CONFIRMED)
    sigma_g^theta  = sqrt( (s sigma_g)^2 + sigma_pv,g^2 )                                 (s on the RAW catalogue error, before the PV fold)
    W_g^theta      = [ max(z_g^theta - 4 sigma_g^theta, 1e-6),  z_g^theta + 4 sigma_g^theta ]   (the site-2.2 window, :7138/:7155)
    k_g^theta(z)   = N(z; z_g^theta, sigma_g^theta) * f_k(g)(z; h) * (dVc/dz)(z; h) / (1 + z)   (the C7-core kernel of :7183-7208,
                                                                                            per-host ZoA fallback f_k -> 1 as at :7193-7196)
    Z_g^theta      = INTEGRAL_{W_g^theta} k_g^theta(z) dz                                  (GL-50, _batched_gl_nodes/_batched_gl_reduce)
    S_tilde_g(theta; h) = INTEGRAL_{W_g^theta} k_g^theta(z) S_bar_phi(z; h) dz / Z_g^theta   (GL-50; S_bar_phi by endpoint-clamped
                                                                                            np.interp on the 1500-node table, as at :7366)
                        := 0 if W_g^theta is degenerate (upper <= lower), see 2.4

    Sigma_phi_smear(theta; h) = sum_{g eligible} w_g S_tilde_g(theta; h)                  (eligibility mask on the LISTED z_g, identical
                                                                                            to Sigma_phi_point's and Sigma_4D's, decision D2)
    rho(theta; h)             = Sigma_phi_smear(theta; h) / Sigma_phi_smear((0, 1); h)

**Registered divisor:**

    Sigma_phi_reg(theta; h) = Sigma_phi_point(h) * rho(theta; h),        Sigma_phi_point(h) = the stored table of (1b)
    Sigma_phi_reg((0, 1); h) = Sigma_phi_point(h)   BY LITERAL SKIP (the stored dict object itself; rho is not evaluated)

**Consumer:** global_denom_no_bh (:5215-5219) under catalogue_global_selection resolving to "phi" — and ONLY that consumer.
path_a_mixture_objects (:5774-5776), the log/JSON at :4244-4270, the with-BH divisor Sigma_4D and the s3d divisor keep
their theta=(0, 1) objects. Reason (registered, load-bearing): (i) feeding Sigma_phi(theta) into n_hat_w_phi / r_Malm /
alpha_G_phi / D_tilde_phi would move combined_no_bh for every C-C event by a theta-dependent constant — exactly the
"infinite Z manufactured by the global table" pathology F-A measured (WAVE2_REGISTRATION_CHECK_20260829.md section 0:
every C-C event x1.00750 at b=+0.02) — and would fail the registered C-C identity check (PA-HIER-31(d): combined_no_bh
bit-identical across the five theta-nodes for L_cat_no_bh == 0 events); (ii) r_Malm = Sigma_4D / Sigma_phi is a pure
Malmquist ratio only if both legs are on the same footing (D2 rule); Sigma_4D is theta-inert under the form of record,
so a theta-transformed Sigma_phi in r_Malm would be a mixed ratio (the with-BH-side analogue is what "all"+smeared already
does, F-A). The forensic's column audit (section 1: w_G, alpha_G_phi, r_Malm, D_tilde_phi, B_num bit-identical across
nodes) therefore continues to hold after this change.

### 2.2 The switch, its site label, and the guard pattern

New evaluate() / CLI / run_mirror_seed_inprocess parameter (the B5.1 / B7.3 flag pattern, byte-identical default):

    theta_phi_divisor: str = "off"        # {"off", "on"}; "on" = site 2.3phi armed

- "off" (default): no code path changes; every existing configuration is byte-identical (regression items R1, R2).
- "on": Sigma_phi_reg replaces Sigma_phi_point at the no-BH L_cat divisor whenever theta != (0, 1). At theta == (0, 1)
  the literal skip applies and an INFO line records "site 2.3phi armed, identity theta, divisor = point table".
- Independent of theta_sites (so it composes with "2.2" for the CoR-P/CoR-M-faithful form of record, and with
  "all"+smeared for the superseded smeared form) and valid with smear_global_selection=False: the phi leg carries its
  own per-host kernel integral, so PA-HIER-10's smear requirement (the guard at :2799-2806, which stays UNTOUCHED — the
  new pass is a separate function, not the smear branch) does not apply to it.
- Guards (raise, never a silent no-op): "on" with catalogue_global_selection resolving to "s3d" -> ValueError (no phi
  table to transform); "on" with normalization_mode != "absolute_marginal" -> ValueError (no phi objects are built,
  :4210); invalid token -> ValueError. _validate_theta as today.
- Hook counter key "site_2_3_phi" in _THETA_HOOK_COUNTERS (:1623), incremented once per (h) pass when engaged; the
  decisive engagement evidence remains the per-event ln L diagnostics (PA-HIER-23), never the counter alone.
- Implementation shape (registered): a NEW module-level function, suggested name
  precompute_phi_divisor_theta_ratio(h_values, galaxy_catalog, completeness, phi_survival_table, theta_b, theta_s,
  z_max_cap, ...) -> dict[h -> (Sigma_phi_smear(theta), Sigma_phi_smear((0,1)), rho, n_degenerate, w_share_degenerate)],
  called from evaluate() immediately after :4234-4243 when armed and engaged, reusing the site-2.2 batch primitives
  (_batched_gl_nodes :505, _batched_gl_reduce :529, _gaussian_pdf :479, _host_pixels :554,
  _completeness_at_host_nodes :571, comoving_volume_element, R_eff_per_mbh) in row chunks. The existing branches of
  precompute_global_catalog_selection are not edited. The stored attribute self._global_cat_selection_phi keeps the
  point table; a new self._global_cat_selection_phi_theta (dict) is consumed at :5215-5219 only when armed and engaged.
- Instrumentation: write_selection_table_json (:2593) gains optional fields sigma_phi_theta, sigma_phi_smear_truth,
  rho_theta, kappa_smear_over_point (= Sigma_phi_smear((0,1))/Sigma_phi_point, REPORTED diagnostic, section 5.5),
  n_degenerate_rows, w_share_degenerate, theta_b, theta_s, theta_phi_divisor — so the T1.2 gates are scored from files.

### 2.3 Per-node scalar rho(theta): EXACT, not first-order (the E11 question)

Under 2.1 the theta-dependence of the divisor enters the no-BH channel through one global scalar per (h, theta), and
every event's L_cat,i(theta) = num_i(theta) / Sigma_phi_reg(theta; h) = [num_i(theta) / Sigma_phi_point(h)] / rho(theta; h).
Hence dividing a banked node's L_cat,i(theta) by the per-node scalar rho(theta; h) — E11's operation,
(beta L_cat,i(theta)/rho + B_num,i)/D_tilde_phi, f12_robust_rho.py — is ALGEBRAICALLY IDENTICAL to re-evaluating the
node with the registered divisor: exact per event, not a linearisation. The first-order object is the c-weighted
decomposition <c_i> C_b (forensic section 3), which linearises the mixture; it is not what E11 computed. The only
differences between E11's rho and the estimator's are estimation, not form: E11's rho came from a 200k-row pool
subsample (and, alternatively, the 797 well-posed drawn hosts), while the estimator sums the full 20,834,171-row eligible
pool; the two E11 estimates differ by 8e-4 in rho(b+) and 1.0e-3 in rho(b-) (f12_out.json rho_drawn_robust vs
rho_pool200k_robust, 2026-08-30), i.e. by 0.05 in C_b and by about 0.03 in score_b — one fourteenth of the SEM. The
degenerate-row convention of 2.4 adds at most 1.0e-3 to rho(b-) at b=-0.02 (section 2.4). Consequently the T1.2
re-certification's b-axis outcome is PREDICTED to SEM/10, and its value is instrument certification (prereg section 2.1:
"a null there refutes nothing about the error-model lever — it only certifies the instrument"), not discovery.

Registered form choice — ratio-to-point rather than "smear the phi leg at every node": (i) A2(a)/(b) (row #255) ratified
the unsmeared point form as the form of record at both venues, so the (0,1) node must be byte-identical to production;
(ii) the fully smeared phi divisor would multiply the h-profile at (0,1) by kappa(h) = Sigma_phi_smear((0,1); h) /
Sigma_phi_point(h) — a production change outside A3's scope whose magnitude on the phi leg is UNMEASURED (the with-BH
analogue F-A measured D_tilde_phi -0.745 %); (iii) at fixed h the two forms have IDENTICAL theta-scores, secants and
curvatures (ln Sigma_phi_reg(theta) = ln Sigma_phi_smear(theta) + const(h)), so nothing is lost for any [HIER] read.
kappa(h) is logged for free from the normaliser pass (section 5.5) and returns as a fresh [RULE] input.

### 2.4 Degenerate transformed windows (the E20 edge case), registered rule

At b < 0 a row with z_g < -b/(1+b) gets a negative kernel centre; when additionally z_g^theta + 4 sigma_g^theta <= 1e-6
the window W_g^theta is empty (upper <= lower): the transformed model asserts the galaxy has NO support at physical
redshift. Census (t11_census_timing_out.json, 2026-08-30, full pool, md5 PASS; the b=-0.02 line reproduces the
forensic's E20 bit-for-bit):

| node | negative centre rows (frac) | w_g-share upper bound | inverted window rows (frac) | w_g-share upper bound | inverted at s = 1/sqrt2 |
|---|---|---|---|---|---|
| b = -0.02 (S0-A as-built) | 63,036 (0.3026 %) | 0.418 % | 15,618 (0.0750 %) | 0.104 % | 18,821 |
| b = -0.033 (S0-B re-derived) | 215,663 (1.035 %) | 1.513 % | 51,394 (0.2467 %) | 0.334 % | 60,265 |
| theta = (0, 1) and every b >= 0 | 0 | 0 | 0 (t11_pool_stats_out.json n_degenerate_at_truth = 0; z_error_min = 5.317e-4 > 0) | 0 | 0 |

Registered rule: S_tilde_g(theta) := 0 for degenerate rows (zero physical support => zero survival mass; the unique
value consistent with the s -> 0 delta limit for a delta placed below the floor), counted and written to the JSON
(n_degenerate_rows, w_share_degenerate). Rows with a negative centre but a non-empty window are integrated by the same
floored quadrature as site 2.2 (well-posed; their S_tilde is 0.96-0.99, f12_out.json drawn_bad_hosts_detail). The
site-2.2 numerator's own guard on the same rows (:7209, Z_g <= 0 -> 1.0; forensic E20: 3/800 drawn hosts, one with a
harness S_tilde(b-) = -310; this node's dummy-table smear at b=-0.02 shows 116 negative and 34 non-finite of 200k rows,
t11_census_timing_out.json smear200k_b_minus) is NOT changed here (section 10, item 6): changing it would alter the banked
theta-node comparands on those events and break the exactness pin R3. Consequence bound: the zeroed rows carry at most
0.104 % (b=-0.02) / 0.334 % (b=-0.033) of the weight, so rho(b-) is lowered by <= 1.0e-3 / <= 3.3e-3 relative to the
forensic's exclude-at-all-nodes convention, i.e. |delta C_b| <= 0.026 / <= 0.05 and |delta score_b| <= 0.016 / <= 0.03 —
inside the SEM (0.43 / 0.45-0.54) by a factor > 15.

### 2.5 The sky-cone-radius instrument flag (A3's second item) and the z-window companion

    sky_cone_k: float = 1.5      # evaluate() / CLI --sky_cone_k / run_mirror_seed_inprocess; finite and > 0 else ValueError
    bayesian_statistics.py:4869   sigma_multiplier=1.5  ->  sigma_multiplier=self._sky_cone_k

handler.py:662 is unchanged (it already takes sigma_multiplier). Default 1.5 is the same float literal -> the mask, the
candidate lists and every downstream value are byte-identical (regression item R9). The candidate ball is shared by the
no-BH and with-BH channels, so a non-default value moves BOTH channels' candidate sets (disclosed). Values are pure
instrument settings (counterfactual), never a production posterior.

The forensic's enlarged-ball counterfactual (E9, E12) used sky 3.0 sigma_max AND a z-window widening from +/-1 sigma_g
to +/-4 sigma_g (handler.py:668-677, hardcoded 1 x REDSHIFT_ERROR). The sky cone is the dominant truncation of the TRUE
host (E6: 14.5 % sky-cone vs 2.2 % z-window exclusion), but the s-secant response to the sky cone ALONE is UNMEASURED.
Registered companion knob, beyond A3's literal text and open to the orchestrator's path choice under row #255:

    z_window_k: float = 1.0      # handler.py:668-677: z_min <= z_g + z_window_k * z_err_g, z_max >= z_g - z_window_k * z_err_g
                                 # single read/validate site in handler.py (the mass_filter_k pattern); default 1.0 byte-identical

The E12 prediction (Z_s -> -0.5 +/- 1) is registered for the pair (sky_cone_k = 3.0, z_window_k = 4.0); the cone flag
alone carries NO band (REPORTED-ONLY). Without z_window_k the s-axis read of T1.2 cannot be matched to E12.

---

## 3. Reference (citations + derivation)

- B1_1_S0A_DEFECT_FORENSIC_20260829.md: mechanism (i) (section 0, E10, E11: C_b = -2.20/-2.25, rho values; Z_b -3.68 -> -0.62);
  mechanism (ii) (E9, E12: Z_s -> -0.5 +/- 1 with the enlarged ball; E6 the ball model); E13 (secant bias, PA-HIER-4 class);
  E20 (edge case); section 5 ("The decisive mechanism test needs code": Sigma_phi(theta) = sum_g w_g S_tilde_g(theta),
  site 2.3 extended to the phi-table branch, or the per-node rho(theta) scalar, exactly E11's operation).
- PREREGISTRATION_HIER_HTHETA_20260826.md section 1.2 (theta identity; s on the raw catalogue error before the PV fold;
  truth-theta = (0,1)); section 2.1 (S0-A certifies, never discovers); section 4.1 (bands, B0-A / B0-A'); section 5.2
  item 3 (the shared-misspecification cancellation on the mirror: what a self-consistent divisor can and cannot show);
  PA-HIER-31 (b)/(d)/(g) + REVISION NOTES 1-2 (the CoR-P/CoR-M form, the C-C identity check); A2(a)/(b) row #255.
- WAVE2_REGISTRATION_CHECK_20260829.md section 0 F-A (site 2.3 reaches the no-BH channel through Sigma_4D -> r_Malm ->
  alpha_G_phi -> D_tilde_phi; the constant per-event offset) — the reason path-A objects stay at (0,1) (2.1).
- Ma, Hu & Huterer (2006), ApJ 636, 21, arXiv:astro-ph/0506614, sec. 2: the affine photo-z systematic (bias, scatter)
  per redshift bin — the parameterisation of theta (as at :7119-7129).
- Mandel, Farr & Gair (2019), MNRAS 486, 1086, arXiv:1809.02063, Eqs. (5)-(7): the selection-normalised population
  likelihood — the divisor of the catalogue-host term is the population-integrated detection probability under the
  SAME hypothesis (theta) as the numerator.
- Gray et al. (2020), PRD 101, 122001, arXiv:1908.06050, Eq. (29): the discrete catalogue sum as the Monte-Carlo
  realisation of beta_G; Eq. (A.10): the host-z kernel in numerator and normaliser.
- Research-cycle amendment A12 (.claude/skills/research-cycle/SKILL.md:149-157): for data drawn from the estimator's own
  model, E[d ln L / d theta] at truth is zero (the score identity).
- DERIVATION_ESTIMATOR_REDESIGN.md section 3.3 / section 7 risk R4 (num/denom sigma_z symmetry); FIXB_PATHA_PACKAGE.md
  section 3.2 (Sigma_phi slot 2, the D2 same-rows rule); GATE_PACKAGE_FINAL.md section 1.2 (C7-core: f_k in the kernel
  and in Z_g); PHYSICS_CHANGE_THETA_HOOK_20260828.md (row #216; the site-2.2/2.3 hooks and the s-placement note).

**Derivation of the score identity the registered divisor restores (mirror b0i, no-BH channel).**
Generator (host_mode "catalogue_selected", PA-HIER-19; correspondence_1d.py catalogue_selected_host_draw_weights and
_draw_kernel_survival_redshifts, E1 reproduced to 8e-14): host g is drawn with probability w_g S_tilde_g / sum_g' w_g' S_tilde_g',
then z_true ~ k_g(z) S_bar_phi(z) / S_tilde_g on W_g, then the GW data d ~ p(d | z_true), with k_g, W_g, S_tilde_g the
theta=(0,1) objects of 2.1 (the harness kernel is the C7-core kernel, correspondence_1d.py:1248-1345). Hence the density
of the observed data is

    p_gen(d) = sum_g w_g INTEGRAL k_g(z) S_bar_phi(z) p(d | z) dz / Sigma_phi_smear((0,1)).

The estimator's catalogue-leg likelihood under hypothesis theta with the "phi" numerator (production default) and the
registered divisor is

    L_cat(d | theta) = sum_g w_g INTEGRAL k_g^theta(z) S_bar_phi(z) p(d | z) dz / Sigma_phi_reg(theta)
                     = [ sum_g w_g INTEGRAL k_g^theta(z) S_bar_phi(z) p(d | z) dz / Sigma_phi_smear(theta) ] * const(h),

and since INTEGRAL p(d | z) dd = 1, INTEGRAL [numerator] dd = sum_g w_g S_tilde_g(theta) = Sigma_phi_smear(theta): the
smeared sum is exactly the normaliser that makes L_cat(. | theta)/const a probability density in d for EVERY theta.
Therefore, at theta = (0,1), L_cat(. | (0,1))/const = p_gen and

    E_gen[ d/dtheta ln L_cat(d | theta) ]_(0,1) = INTEGRAL p_gen(d) d/dtheta ln L_cat dd = d/dtheta INTEGRAL L_cat/const dd = d/dtheta 1 = 0.

This is A12. With the theta-INERT point divisor the same computation gives E_gen[d/dtheta ln L_cat] = - d/dtheta ln
Sigma_phi_smear(theta) != 0 — the forensic's mechanism (i), with the measured value C_b = -2.20 per unit b (E10),
sign fixed by <d ln S_bar_phi/dz>_k = -2.40 +/- 0.04 (E3). Three disclosed departures of the registered arm from this
identity, none of them divisor properties: (V2) the estimator's mixture p_i = (beta_G_phi L_cat,i + B_num,i)/D_tilde_phi
gives d/dtheta ln p_i = c_i d/dtheta ln L_cat,i with a data-dependent c_i while the generator has no dark hosts
(E17, Cov(c, score) != 0); (V3) the candidate ball truncates the host sum (E6, E9, E14); (V4) the Fisher-quality
exclusion selects on d_L (E5); plus the driver's "off" numerator (E8: <= 0.03 per event on the b-secant, immaterial in
the GW-precise regime, sigma_zGW/sigma_k median 0.130). That is why the registered prediction is E11's -0.27 +/- 0.43
(which contains V2-V4 as they act on the banked data), not exactly zero, and why the s-axis needs (V3)'s flag (2.5).
On the production venue (S0-B, no truth-theta) the identity is not available; there the divisor fix removes a
by-construction offset of order <c> C_b ~ -1.3 per unit b (forensic section 7) that would otherwise be read as a lever.

---

## 4. Dimensional analysis

| symbol | units | check |
|---|---|---|
| z_g, b, z_g^theta | dimensionless | z + b(1+z): dimensionless + dimensionless x dimensionless |
| sigma_g (REDSHIFT_MEASUREMENT_ERROR), s, sigma_pv,g, sigma_g^theta | dimensionless | sqrt((s sigma)^2 + sigma_pv^2): both terms dimensionless; SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S is km/s over km/s |
| N(z; ., .) | 1/z = dimensionless | Gaussian density in a dimensionless variable |
| (dVc/dz)(z; h) | Mpc^3 sr^-1 (comoving_volume_element) | cancels between numerator and Z_g of S_tilde_g |
| f_k(z; h) | dimensionless in [0, 1] | cancels likewise |
| S_bar_phi(z; h) | dimensionless in [0, 1] | phi(log10 M) is a normalised density in log10 M, S_4D in [0, 1] |
| S_tilde_g(theta; h) | dimensionless in [0, 1] | ratio of two integrals with identical measure; the integrand of the numerator is the denominator's times a [0, 1] factor; degenerate rows := 0 |
| w_g = R_eff(M_g)/(1+z_g) | events per year per galaxy (emri_rate.R_eff_per_mbh) | as Sigma_phi_point (:2903) |
| Sigma_phi_smear, Sigma_phi_point, Sigma_phi_reg | events per year | Sigma_phi_reg = Sigma_phi_point x rho, rho dimensionless |
| rho(theta; h), kappa(h) | dimensionless | ratio of two like sums |
| L_cat = sum_ball w_g N_g / Sigma_phi_reg | [N_g] = units of the GW likelihood density, unchanged | the divisor's units are unchanged, so L_cat's are |

No mixed units anywhere; the change alters one dimensionless factor rho on a divisor whose units are untouched.

---

## 5. Limiting cases

**5.1 theta = (0, 1) — the identity (GATE T-ID).** By literal skip Sigma_phi_reg((0,1)) IS the stored point table: bit-
identical, no floating operation performed. Even with the skip removed (a test forces the computation at (0,1)):
rho((0,1)) = Sigma_phi_smear((0,1)) / Sigma_phi_smear((0,1)) is the quotient of two runs of the same deterministic code on
the same inputs -> exactly 1.0 -> Sigma_phi_point * 1.0 == Sigma_phi_point bit-for-bit (IEEE-754 multiplication by 1.0 is
exact). Registered as regression items R1 (skip) and R2 (forced).

**5.2 s -> 0 (delta kernel).** N(z; z_g^theta, sigma -> 0) -> delta(z - z_g^theta); with the C7 measure f_k w_pop the
normalised kernel still collapses to the delta (the measure is smooth and positive at z_g^theta > 1e-6), so
S_tilde_g(theta) -> S_bar_phi(z_g^theta; h). At b = 0 this is S_bar_phi(z_g) and Sigma_phi_smear((0, s->0)) -> Sigma_phi_point:
the point table is the delta limit of the smeared sum (regression item R4, rtol 1e-6 at s = 1e-4 on a 1000-row
subsample; mirrors test_sigma_z_to_zero_limit_recovers_point_evaluation of test_smear_global_selection.py:175 for the
phi leg). Hence rho((0, s -> 0)) -> Sigma_phi_point / Sigma_phi_smear((0,1)) = 1/kappa(h): the ratio form's s -> 0 limit
exposes the point-vs-smear factor kappa, which is a CONSTANT in theta and cancels from every theta-statistic (2.3).
A delta placed below the floor (z_g^theta < 1e-6, only at b < 0) integrates to 0 — the 2.4 rule.

**5.3 b shifts at s = 1 (sign and size).** For kernels narrow against the S_bar_phi scale, S_tilde_g(b) ~ S_bar_phi(z_g + b(1+z_g)),
so d/db ln Sigma_phi_smear ~ < (1 + z_g) d ln S_bar_phi/dz >_w. With <d ln S_bar_phi/dz>_k = -2.40 +/- 0.04 (E3, f3_out.json)
and 1 + z_g in 1.0-1.4 on the eligible pool, C_b in -2.4 to -3.4 for the low-z-weighted sum — the measured secant
C_b = -2.20 (drawn) / -2.25 (pool) (E10) sits at the low end because w_g S_tilde weights low z where the kernel-mean tilt
(E15) softens the slope. Signs: S_bar_phi falls with z => rho(b > 0) < 1 and rho(b < 0) > 1 => ln rho has negative
b-secant => the corrected ln L_cat = ln L_cat,raw - ln rho gains +|C_b| c_i per unit b => score_b moves from -1.634 by
-<c> C_b = -(0.616)(-2.2) = +1.36 to about -0.27 (E11: -0.268 +/- 0.431). Registered production values (from E10, pool):
rho(+0.02, 1) = 0.9538, rho(-0.02, 1) = 1.0435, rho(0, sqrt2) = 0.9893, rho(0, 1/sqrt2) = 1.0059, each +/- 0.002 (the
subsample-vs-pool spread of E10 plus the 2.4 rule); C_b = -2.25 +/- 0.05, C_s = -0.024 +/- 0.002 per unit (linear s).

**5.4 Small-theta linearisation and the secant.** Sigma_phi_reg(b) = Sigma_phi_point [1 + b C_b + O(b^2)]; at +/-0.02 the
measured asymmetry (1 - 0.9546 = 0.0454 vs 1.0424 - 1 = 0.0424) shows an O(b^2) term of ~7 % of the linear one, which the
registered two-sided secant symmetrises. On the s-axis the +/- ln sqrt2 secant carries the intrinsic +0.0455/event bias
(E13) that A2(d) (row #255) replaces via PA-HIER-32 — a registration item, not a divisor property.

**5.5 h at theta = (0, 1).** The h-profile of every production object is byte-identical (5.1), so the A2(a) form of record is
untouched. kappa(h) = Sigma_phi_smear((0,1); h)/Sigma_phi_point(h) is logged from the normaliser pass as a REPORTED
diagnostic (no band; it is the no-BH analogue of the F-A -0.745 % D_tilde_phi shift and is the input a future [RULE] on
smearing the production phi leg would need).

**5.6 The registered prediction of the S0-A re-certification (divisor on, as-built +/-0.02 nodes, 4 seeds, h = 0.73,
form theta_sites="2.2" / smear off / catalogue_numerator_survival "off" as the bc driver, sky_cone_k = 1.5).**
score_b = -0.27 +/- 0.43, i.e. **Z_b = -0.62** (E11 pool rho; drawn rho gives -0.30 +/- 0.43, Z -0.69) — inside the
registered band |Z_b| <= 3 (prereg section 4.1, B0-A). Exactness prediction for the implementation: the re-run's pooled
score_b lies within +/-0.10 of -0.268 (rho estimation error <= 0.03 plus the 2.4 rule <= 0.02 plus the truth node's
literal skip: 0). Per seed (f12_out.json corrected_combined_pool per_seed, pool rho): 900101 -1.71 (-1.5 sigma), 900102
-1.26 (-1.4 sigma), 900103 +1.17 (+2.0 sigma), 900104 +0.69 (+0.9 sigma) — all within their own 3 sigma. score_s =
-0.0728 +/- 0.0122 (Z -5.97): the s-axis STAYS outside the band by the truncation mechanism (ii), AS PREDICTED; it is not
a falsifier of this change (section 9). With the enlarged ball (sky_cone_k 3.0, z_window_k 4.0) the c-weighted registered
statistic is predicted at -0.005 +/- 0.011 (Z -0.5) (E12), unweighted +0.036 +/- 0.016 (= the E13 secant bias).

---

## 6. Cost (A11: measured anchors; bands where unmeasured)

Anchors {value, source, date}: S0-A cell (evaluate only, 14 cpu, 106-130 events) 60.9-64.0 s, mean 62 s
{s0a_full_output.json per_seed_summary[*].nodes[*].elapsed_s, 2026-08-29}; S0-A pass 2959.6 s wall at 14 cpu = 11.5 CPU-h
including venue builds {same file, wall_s}; S0-C marginal per-h cost 24.37 s, first-h 1704.3 s (venue + tables)
{s0c_full_output.json, 2026-08-29}; theta-engaged SMEARED cell 1191 s single-core {TREE2_CHARTER_20260830.md section 2
anchors, from B1_1_HIER_RECORD.md}; C7-kernel GL-50 pool smear, this node, single process, 200,000 random pool rows,
completeness from cache, unit survival table (interp cost identical to the real table): **27.32 microseconds/row at
(0,1) and 31.76 microseconds/row at (-0.02, 1)** -> **569 s and 662 s per full 20,834,171-row pass**
{t11_census_timing_out.json smear200k_truth / smear200k_b_minus, 2026-08-30}; pool load 49 s; md5 2.6 s.

1. **S_bar_phi table builds per theta node: ZERO additional.** The table is theta-free (1a); one build per
   BayesianStatistics as today. The new work is per-host kernel integrals against the existing table.
2. **Pool passes per off-truth node per h: 2** (the theta pass and the (0,1) normaliser pass) = 569 + 662 s ~ 20.5 min
   single-process at h = 0.73. The normaliser pass is theta-independent; caching it to disk keyed by (catalogue md5,
   h, completeness cache id, S_bar_phi table hash, n_quad, window constants) reduces this to 1 pass (~11 min) for the
   2nd-4th nodes of a seed — optional, and only with a bit-identity test (R8) since GATE TABLE-FRESH forbids sharing
   objects across nodes without proof of determinism.
3. **Is a theta-engaged no-BH cell single-core-bound like the smeared path? YES under the reference implementation:**
   the pass is a ~1e9-node numpy sweep on one core (the same shape as the 1191 s smeared cell), so the cell goes
   62 s -> ~1290 s (x21; x12 with the cached normaliser). Registered mitigation: row-chunk parallelism (the pass is
   row-independent, so chunking/parallel scheduling is a pure memory-shape transform with bit-identical results —
   the harness's own _KERNEL_SMEAR_CHUNK precedent, correspondence_1d.py:1248-1345 docstring), run BEFORE evaluate()'s
   per-event worker pool is opened (the nested-pool restriction seen in runner-2: daemonic processes cannot spawn
   children; the driver's --jobs>1 limitation is unchanged). At 14 workers: ~47 s per pass -> cell ~2.6 min (x2.5).
4. **S0-A re-certification (T1.2, 20 cells; the 4 truth cells are literal-skip):** single-process serial: 16 x 20.5 min
   + 20 x 62 s + ~27 min of venue/table builds (2959.6 s - 20 x 62 s) ~ **6.3 h wall**; with the cached normaliser per
   (seed, h): ~4.4 h; with 14-way row parallelism: ~1.3 h. Local only (cluster down).
5. **S0-B (C1, iiib, 4 off-truth nodes at h = 0.73):** the same catalogue -> +2 passes ~ +0.35 CPU-h per node on one core,
   immaterial against the 60-92 CPU-h band (PA-HIER-31(i)); the enlarged-ball arms scale the per-event cost by the
   candidate-count ratio (E9: median 278 -> 1729, ~3-6x per cell, forensic section 5).
6. **41-h grids (S0-C shape, Stage P):** +41 x 20.5 min ~ 14 h single-core per node (~1 h at 14 workers) — Stage P stays
   MOOT (A15) and must be re-costed with this term before any launch.
7. The sky-cone flag costs nothing at default; at 3.0 the cell cost is the candidate-count ratio above.

---

## 7. Regression plan (tests written BEFORE the change where they pin existing values)

R1  Literal skip (unit): with theta_phi_divisor="on" and theta=(0,1), the divisor dict consumed at :5215 IS the stored
    point table (identity of object and of every value); no pass executed (hook counter site_2_3_phi unchanged).
R2  Forced (0,1) (unit): call the new pass directly at theta=(0,1) on a 2,000-row catalogue fixture: rho == 1.0 exactly
    and Sigma_phi_point * rho == Sigma_phi_point bit-for-bit (np.array_equal on the float64 bits). (5.1)
R3  Exactness pin against the BANK (zero compute after one node): re-run seed 900101 node b_plus (theta=(+0.02,1),
    sites 2.2, smear off, divisor on) and diff against hier_s0_registered_run/s0a_seed900101/node_b_plus_sites2.2_nosmear/
    simulations/diagnostics/event_likelihoods.csv (dedupe keep="last"): for every event with L_cat_no_bh > 0,
    L_cat_no_bh(banked) / L_cat_no_bh(new) == rho(+0.02, 1; 0.73) from the new JSON, rtol 1e-12; all other numeric columns
    (B_num, D_tilde_phi, alpha_G_phi, r_Malm, w_G, L_cat_with_bh, ...) max_abs 0.0. This proves the change touches ONLY
    the no-BH divisor and reproduces E11's operation exactly. Same check on s_plus.
R4  Delta limit (unit): s = 1e-4, b = 0 on a 1,000-row subsample: sum_g w_g S_tilde_g -> Sigma_phi_point (rtol 1e-6). (5.2)
R5  Harness parity (unit, the PA-HIER-30 pattern): production S_tilde_g(theta) vs correspondence_1d.kernel_smeared_survival
    (z_g + b(1+z_g), s sigma_g, ...) at the four as-built nodes on 1,000 random well-posed pool rows: rtol 1e-12
    (same GL-50 nodes, same window, same C7 measure; report the achieved figure). Plus the total at (0,1):
    Sigma_phi_smear((0,1); 0.73) vs the harness's catalogue_selected_host_draw_weights total (rtol 1e-9).
R6  Engagement and signs (unit, synthetic monotone-decreasing table + a two-host hand computation via scipy quad,
    rtol 1e-8): rho(+b) < 1 < rho(-b); rho(0, s>1) < 1 < rho(0, s<1); C_b, C_s signs as 5.3.
R7  Guards (unit): "on" + "s3d" raises; "on" + non-absolute_marginal raises; bad token raises; theta validation as today.
R8  Chunk invariance (unit): the pass with chunk sizes {97, 1000, n} and with 1 vs 3 workers gives bit-identical sums.
R9  Degenerate rows (unit): a fixture row with z_g + 4 s sigma_g <= 1e-6 after the b-shift contributes exactly 0 and is
    counted; a negative-centre row with a non-empty window is integrated (finite, in [0,1]).
R10 Sky-cone flag: default byte-identity of get_possible_hosts_from_ball_tree results at sigma_multiplier 1.5 through
    the new parameter (the B5.1 100,000-pair pattern on 50 events x 2,000 candidates: 0 mismatches); engagement at 3.0
    (candidate count grows, true-host recovery counters P6 grow); finite/positive validation. If z_window_k is built:
    default 1.0 byte-identity + engagement at 4.0 (the E9 numbers: median candidates 278 -> 1729 on b0i seed 900101).
R11 C0-style pin (the truth node): re-run seed 900101 node_truth (divisor on, theta=(0,1)) and diff against the banked
    node_truth_sites2.2_nosmear CSV: max_abs 0.0 on all numeric columns and md5-identical selection_tables_h_0_73.json
    (up to the appended new fields). The E19-ratified comparand (A2(c)) is the current generator grid, so no bank
    regeneration is needed.
R12 Suite: full pytest -m "not gpu and not slow" baseline 1896 passed / 15 skipped / 27 deselected (PHYSICS-GATE-LEDGER
    2026-08-29 rows) + the new tests; ruff / mypy clean; existing test_theta_hook.py (15 tests) and
    test_smear_global_selection.py (6 tests) untouched and green.
R13 Plumbing: --theta_phi_divisor and --sky_cone_k (and --z_window_k if built) on arguments.py / main.py /
    run_mirror_seed_inprocess with byte-identical defaults; run_metadata captures them (to_dict); hier_s0_driver.py
    pass-through is the T1.2 builder's job (driver owned outside this gate; non-physics file).
Reference comment above the changed lines: "Eqs. (5)-(7) in Mandel, Farr & Gair (2019), arXiv:1809.02063; sec. 2 in
Ma, Hu & Huterer (2006), arXiv:astro-ph/0506614; PHYSICS_CHANGE_THETA_DIVISOR_20260830.md section 2 (row #255, T1.1)".
Commit prefix [PHYSICS]; ledger rows implemented / verified appended by the builder and by a different verifier.

---

## 8. A10 — invariants (with last-audited dates) and structural blindness

**Invariants held fixed by this change:**
1. S_bar_phi table construction (:1982-2075) — unchanged, theta-free (FIXB_PATHA 2026-08-04; end-to-end NEVER audited,
   B3.2 section 8 item 6 — carried, by name).
2. Sigma_phi_point (:2906-2915, :3022) and its eligibility mask, identical to Sigma_4D's (D2, 2026-08-04) — unchanged.
3. Site-2.2 kernel form f_k * dVc/dz/(1+z) with the +/-4 sigma window, 1e-6 floor, GL-50 (G2b/C7-core 2026-08-04;
   K1-K4 NEVER re-audited against a z-dependent population, B3.2 section 8 item 5 — carried) — the divisor kernel is
   DEFINED as this kernel (R5).
4. theta identity (HIER section 1.2) with s on the raw error and the PV term from the unshifted z_g (A9 CONFIRMED, row #255).
5. smear_global_selection = False at CoR-P and CoR-M/S0-A (A2(a)/(b), row #255, 2026-08-30).
6. catalogue_global_selection = "phi" production default (rows #171-#178, 2026-08-22/23).
7. Path-A objects (n_hat_w_phi, r_Malm, alpha_G_phi, D_tilde_phi, w_tilde_G) consume Sigma_phi_point ONLY (this
   presentation; the C-C identity check PA-HIER-31(d)).
8. Reduced catalogue md5 c52c13b5cab61f6b3f04bbe202550969 (verified 2026-08-30, t11 script; A11 dataset pin).
9. The with-BH divisor Sigma_4D theta-inert under the form of record (invariant 12 of the prereg, [P3-MKER] state).
10. Sky-cone radius 1.5 sigma_max and the +/-1 sigma_g z-window at the production default (handler.py:662, :668-677).
11. Literal-skip identity at theta = (0,1) at every site (GATE T-ID, 2026-08-28).
12. The driver's banked comparands (node CSVs of 2026-08-29) — R3/R11 pins.

**Structural blindness (what this design cannot detect by construction):**
(a) the with-BH channel's divisor and catalogue leg (out of scope; its theta-response — secants +11.31 +/- 0.78 (Z +14.5)
and +0.229 +/- 0.029 (Z +7.9), forensic section 3 — is untouched and uninterpretable on the 1-D venue);
(b) any theta-response that would live in the completeness-weight chain (alpha_G_phi, D_tilde_phi, w_tilde_G) — held at
(0,1) by design (2.1), so invisible; (c) the point-vs-smear factor kappa(h) on the h-posterior — reported, not acted on;
(d) prereg section 5.2 item 3: any misspecification SHARED by the generator's S_tilde_g and the estimator's S_tilde_g (same
kernel code, same S_bar_phi table, same completeness cache) cancels on the mirror — the re-certification certifies the
divisor's ARITHMETIC and theta-consistency, never the survival model; (e) V2 (mixture weight vs an all-catalogue
generator), V3 (ball truncation — only partially addressed by the flags) and V4 (Fisher-quality selection) remain in
the re-certification and bound how close to zero it can land; (f) the degenerate-row asymmetry (site 2.2 keeps its 1.0
guard) — confined to b < 0 rows with z_g < |b|/(1+b), weight share <= 0.10 % (b=-0.02) / <= 0.33 % (b=-0.033);
(g) the s-axis secant form (E13) until PA-HIER-32 lands; (h) theta's 2-D span (prereg section 5.2 item 2); (i) single h.

---

## 9. A14 — falsifiers, registered before any code

F1 (mechanism (i), the b-axis): the T1.2 S0-A re-certification (divisor on; as-built +/-0.02 nodes; 4 seeds; h = 0.73;
   theta_sites="2.2"; smear off; "off" numerator as the bc driver; sky_cone_k 1.5) must return **|Z_b| <= 3** on the
   registered no-BH channel. |Z_b| > 3 REFUTES the attribution of the S0-A b-axis non-null to mechanism (i): the divisor
   theta-gap is then not the cause, the fix stays in the tree as a structural-consistency change only, and the STOP
   returns as INSTRUMENT-DEFECT-UNRESOLVED for a fresh diagnosis. Sharper: |score_b(re-run) - (-0.268)| > 0.10 with
   |Z_b| <= 3 means the implementation does NOT realise E11's operation (STOP, implementation defect, even though the
   band passes) — the band and the exactness check are both required.
F2 (mechanism (ii), the s-axis): the enlarged-ball arm (sky_cone_k 3.0 AND z_window_k 4.0) must return |Z_s| <= 3 on
   the registered statistic once PA-HIER-32's secant replacement is in force (E12 point -0.5); |Z_s| > 3 refutes the
   truncation attribution of the s-axis. Divisor-only: score_s is PREDICTED to remain at -0.073 +/- 0.012 (Z ~ -6);
   a divisor-only |Z_s| <= 3 would itself be a SURPRISE that contradicts E9/E12 and must be reported as such. The cone
   flag alone carries no band (UNMEASURED; REPORTED-ONLY).
F3 (implementation level, before the re-run): the four production rho values outside the +/-0.002 windows of 5.3 ->
   STOP, the pass is not E11's operation.
F4 (instrument identity): the C-C identity check (dark class, L_cat_no_bh == 0: combined_no_bh bit-identical across all
   five nodes; the 5 pooled dark events scored exactly 0.0 on 2026-08-29) must still pass; any deviation ->
   INSTRUMENT-DEFECT (the path-A objects would have moved).
F5 (byte-identity): R11's truth-node pin fails -> INSTRUMENT-DEFECT; R3's ratio pin fails -> the change reached more
   than the divisor.
Rule-1 exoneration check (mechanism, not tag): EXONERATION_REGISTER_20260827.md and the ledger's DO-NOT-RE-TRY list
were grepped by the forensic and the stage-0 record for the theta-hook / host-z-kernel-misspecification /
smear_global_selection mechanism with no match (B1_1_HIER_STAGE0_RECORD.md section 4); this presentation adds no new
mechanism beyond the divisor theta-gap the forensic registered — no collision.

---

## 10. Decision table (approval-scope tags; the standing grant is the approval)

| # | tag | item | disposition under row #255 |
|---|---|---|---|
| 1 | [DO] | Implement Sigma_phi_reg(theta) = Sigma_phi_point x rho(theta) at the no-BH L_cat divisor only, behind theta_phi_divisor (default "off"), with the 2.4 degenerate rule, the JSON/log instrumentation, and R1-R9, R11-R12 | COVERED (A3); builder = a different agent from this presenter; presentation cited verbatim; [PHYSICS] commit; ledger rows implemented/verified |
| 2 | [DO] | sky_cone_k flag (default 1.5, byte-identical) replacing the literal at :4869, with R10 and plumbing R13 | COVERED (A3) |
| 3 | path choice | z_window_k companion knob (handler.py:668-677, default 1.0, byte-identical) — required to match E12; beyond A3's literal text | orchestrator's call under row #255 (registered here with its consequence: without it, F2 has no band) |
| 4 | fresh [RULE] later | The T1.2 re-certification READ (band comparison on data that do not yet exist) | returns to the author per the charter (T1 depth 2); nothing here pre-decides it |
| 5 | fresh [RULE] later | Whether the production phi divisor should be SMEARED (kappa(h) reported by this change) | inputs do not exist yet; not asked here |
| 6 | NOT DONE, disclosed | The site-2.2 Z_g guard on degenerate transformed windows (:7209; E20) — the mirror of the 2.4 rule on the numerator side | deferred until after T1.2 so the R3 pin stays exact against the bank; a housekeeping gate item |
| 7 | NOT DONE, disclosed | The site-2.3 smear kernel's missing f_k (1g) | absent from the form of record; not touched |

What this presentation does NOT license: any S0-B (C1) launch (A6: after A3 AND a passing T1.2); any Stage P/F costing
or launch (A15 MOOT); any change to Sigma_4D, path-A objects, the with-BH channel, or the generator (GATE GEN-FROZEN,
PA-HIER-2); any code under this node.

---

## 11. Provenance

Read in full: B1_1_S0A_DEFECT_FORENSIC_20260829.md; B1_1_HIER_STAGE0_RECORD.md; PREREGISTRATION_HIER_HTHETA_20260826.md
sections 1.2, 2.1, 4.1, 5.2, PA-HIER-21, PA-HIER-31 + REVISION NOTES 1-2 + the P1/P0 appendix; WAVE2_REGISTRATION_CHECK_20260829.md
section 0; SYNTHESIS_DOCKET_2_20260829.md section 7; END_VERIFIER_REPORT_PART1_20260830.md section 4; BIAS_HISTORY_LEDGER.md
rows #216, #254-#256; TREE2_CHARTER_20260830.md; PHYSICS_CHANGE_THETA_HOOK_20260828.md (headings); P6_THETA_CLI_PLUMBING_RECORD.md;
docs/gates/PHYSICS-GATE-LEDGER.md; .claude/skills/research-cycle/SKILL.md:125-178 (A10-A14).
Code read (no edits): bayesian_statistics.py :479-600 (batch primitives), :1620-1760 (theta helpers, site 2.3),
:1811-1813 (grid constants), :1982-2075, :2441-2500, :2593-2640, :2692-3025, :3385, :3585-3615, :3756-3785, :4165-4275,
:4538-4543, :4835-4880, :5026-5130, :5180-5225, :5290-5300, :5765-5785, :6294-6341, :7017-7020, :7030-7370;
handler.py :558-700; correspondence_1d.py :983-1060, :1161-1345, :1440-1515, catalogue_selected_host_draw_weights;
hier_s0_driver.py (grep only); arguments.py (grep only). Forensic instruments: b1_1_forensic_work/f12_robust_rho.py,
f12_out.json (E10/E11/E20 numbers quoted above), f1/f3/f4/f8 via the forensic's tables.
Instruments of this node (foreground, < 70 s each, zero evaluate() calls, no source edits):
tree2_20260830/t1_1_gate_work/t11_census_timing.py -> t11_census_timing_out.json; t11_pool_stats.py -> t11_pool_stats_out.json.
No git operations by this node; the orchestrator commits. Append-only.

---

## 12. Implementation record (row #255, tree 2 node T1.1; builder pass, 2026-08-30)

Builder is a different agent from the presenter (per section 0's requirement). No git operations
performed by this node; the orchestrator commits. Branch fix/p32d-classg-venue-repair.

### 12.1 What was built, exactly per section 2

Section 2.1's Sigma_phi_reg(theta;h) = Sigma_phi_point(h) x rho(theta;h) is realised as: a new
module-level function precompute_phi_divisor_theta_ratio(h_values, galaxy_catalog, completeness,
phi_survival_table, theta_b, theta_s, n_quad=50, chunk_size=200000) -> dict[h -> {sigma_phi_smear_theta,
sigma_phi_smear_truth, rho, n_degenerate_rows, w_share_degenerate}], built on a helper
_phi_divisor_kernel_pass that reproduces the site-2.2 C7-core kernel exactly (the +/-4 sigma window
with the 1e-6 floor, GL-50 quadrature via the SAME _batched_gl_nodes/_batched_gl_reduce/_gaussian_pdf
primitives, per-host completeness via _host_pixels/_completeness_at_host_nodes with the ZoA
all-or-nothing fallback) and mirrors sigma_pv,g from the UNSHIFTED z_g / s on the RAW catalogue error
exactly as at bayesian_statistics.py site 2.2 (:7117-7129). No z_max_cap parameter was added to the
public signature (a deliberate deviation from the presentation's "suggested name" list, section 2.2):
the eligibility z_max is read directly off phi_survival_table's own z_grid[-1], which already carries
the run's z_max_cap by construction (precompute_phi_marginal_survival built the table with the identical
cap), so decision D2's "identical eligibility mask" requirement is met without a redundant parameter.

Section 2.4's degenerate-row rule (S_tilde_g := 0, counted, never integrated) is implemented by
computing the degenerate mask (hi <= lo) on the full row set BEFORE chunking, so n_degenerate_rows
and w_share_degenerate are chunk-invariant by construction, and only the ACTIVE rows are chunked for
the GL-50 kernel evaluation. Regression R8 (chunk invariance) is satisfied by filling a per-row
contribution array across chunks and reducing with exactly one np.sum call at the end -- the reduction
itself never depends on chunk_size, only the array's contents do, and those are chunk-independent by
construction (each row's window/nodes depend only on that row).

Section 2.2's switch: theta_phi_divisor: str = "off" was added to BayesianStatistics.evaluate()'s
signature (class-level default and __init__ copy also added), independent of theta_sites, and is
built into a per-h dict self._global_cat_selection_phi_theta immediately after the existing
_global_cat_selection_phi = precompute_global_catalog_selection(...) call (bayesian_statistics.py,
inside the "if _use_phi_selection:" block). The literal skip at theta=(0,1) is realised by simply
never populating that dict when theta is the identity (or when the flag is "off") -- the consumer at
the former :5215-5219 site (global_denom_no_bh) does getattr(self, "_global_cat_selection_phi_theta",
{}).get(self.h, self._global_cat_selection_phi.get(self.h, 0.0)), which falls through to the EXACT
stored point-table float value (no floating operation) whenever the new dict has no entry for the
current h. The getattr wrapper (rather than a bare self._global_cat_selection_phi_theta.get(...)) was
required because several pre-existing tests (test_catalogue_global_selection.py) construct
BayesianStatistics via object.__new__ and call p_Di directly without ever running evaluate(), so the
plain instance attribute does not exist on those objects; this matches the codebase's own existing
convention at the same line (getattr(self, "_catalogue_global_selection", "s3d")).

Guards implemented exactly as registered: theta_phi_divisor not in ("off","on") raises; "on" with
self._catalogue_global_selection != "phi" raises ("no phi table to transform" -- covers BOTH the
explicit catalogue_global_selection="s3d" route and the implicit auto-resolves-to-s3d route, since
both leave self._catalogue_global_selection == "s3d"); "on" with normalization_mode != "absolute_marginal"
also raises, though by construction this second condition is currently unreachable in isolation (the
existing catalogue_global_selection="phi" guard already forces normalization_mode="absolute_marginal"
before this point is ever reached with self._catalogue_global_selection == "phi") -- kept as
registered defense-in-depth per the presentation's explicit two-clause guard, disclosed here as
redundant-but-harmless rather than silently dropped.

Section 2.2's instrumentation: write_selection_table_json gained nine new optional keyword arguments
(sigma_phi_theta, sigma_phi_smear_truth, rho_theta, kappa_smear_over_point, n_degenerate_rows,
w_share_degenerate, theta_b, theta_s, theta_phi_divisor), each None by default and omitted from the
JSON payload when None -- so the JSON is byte-identical (same key set) whenever the divisor is off or
at the theta=(0,1) literal skip. kappa_smear_over_point is computed at the call site as
sigma_phi_smear_truth / Sigma_phi_point(h) (section 5.5's REPORTED diagnostic, never a divisor).

Section 2.5's sky_cone_k: added as an evaluate() kwarg (class-level default 1.5, __init__ copy,
guarded finite-and->0), consumed at exactly one site: the sigma_multiplier=1.5 literal at the
get_possible_hosts_from_ball_tree call (former line 4869) is now sigma_multiplier=self._sky_cone_k.
handler.py was NOT touched -- confirmed unnecessary, per the presentation's own text ("handler.py:662
is unchanged"), since get_possible_hosts_from_ball_tree already accepts sigma_multiplier as a
parameter (introduced by the mass_filter_k commit, 0b308828). The z_window_k companion knob (decision
table item 3, "path choice") was NOT implemented -- the orchestrator's task text for this node
threads only "the cone flag" (singular) through evaluate()/arguments.py/main.py/correspondence_1d.py,
and the presentation itself flags z_window_k as a path choice open to the orchestrator, not covered
by A3's literal text. Its absence is disclosed: without it, F2 (the s-axis falsifier at sky_cone_k=3.0
AND z_window_k=4.0) has no way to be matched to the E12 prediction from this node's code alone.

The hook counter key "site_2_3_phi" was added to _THETA_HOOK_COUNTERS and is incremented once per
evaluate() call when the divisor is armed AND engaged (mirrors the existing site_2_1/2_2/2_3 pattern).

### 12.2 Plumbing (mass_filter_geometry pattern, commit 0b308828)

theta_phi_divisor and sky_cone_k were threaded exactly like mass_filter_geometry/mass_filter_k:
arguments.py gained two properties and two argparse entries (--theta_phi_divisor {off,on} default
off; --sky_cone_k type float default 1.5); main.py's module-level evaluate() function gained both
parameters and forwards them into BayesianStatistics.evaluate(); validation/correspondence_1d.py's
run_mirror_seed_inprocess gained both parameters (defaults "off"/1.5, byte-identical) and forwards
them into its bs.evaluate(...) call, with a docstring Note appended. run_metadata capture is automatic
(Arguments.to_dict() serialises the full parsed namespace; no separate stamp-path edit needed, same as
every prior instrument flag).

### 12.3 THE DRIVER GAP -- decisive finding for the T1.2 command

results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py was read (grep only,
per this node's authorization) and confirmed to have NO --theta_phi_divisor or --sky_cone_k CLI
argument, and NO passthrough of either name at any of its run_mirror_seed_inprocess call sites (three
call sites grepped, all confirmed absent). This means:

**--theta-sites 2.2 does NOT engage the new divisor.** The registered design (section 2.2) makes
theta_phi_divisor an INDEPENDENT flag from theta_sites by construction -- it composes with theta_sites
2.2 (both governed by the same self._theta_b/self._theta_s once engaged) but is never implied by it.
No new theta_sites value/label was needed or added (theta_sites stays exactly {"all","2.1","2.2","2.3"});
what is missing is a wholly separate driver CLI surface for theta_phi_divisor/sky_cone_k.

Consequence: the orchestrator's literal proposed T1.2 command (hier_s0_driver.py --arm S0-A, 4 seeds,
5 nodes, --theta-sites 2.2 --smear off, --jobs 1, a new out-root) will run with theta_phi_divisor
defaulting to "off" throughout -- i.e. it will reproduce the S0-A INSTRUMENT-DEFECT result byte-for-byte
(Z_b approx -3.68, the ORIGINAL defect), not the registered prediction (Z_b = -0.62). This would be
misread as a falsification of mechanism (i) when in fact the fix was never armed. Before T1.2 can run
as intended, hier_s0_driver.py needs a --theta_phi_divisor {off,on} argument (default off, byte-identical)
threaded to its run_mirror_seed_inprocess call sites, matching the existing --theta-sites/--smear
pattern; this is R13's explicit statement that "driver.py pass-through is the T1.2 builder's job
(driver owned outside this gate; non-physics file)" -- so it is NOT done by this node, but is flagged
here, loudly, as a blocking prerequisite rather than left to be discovered by a failed re-certification.
sky_cone_k does not need a driver flag for F1 (it stays at its default 1.5 throughout the S0-A
re-certification per section 9's F1 spec); it would only be needed for F2's enlarged-ball arm.

### 12.4 Tests and quality gate

New file darksiren_emri_test/bayesian_inference/test_theta_phi_divisor.py, 19 tests, covering
regression items R1, R2, R4, R6, R7, R8, R9 and R13 in full; R12 by construction (this file plus the
full suite run). Regression items R3, R5 and R11 (bit-for-bit pins against the banked S0-A CSVs and
the correspondence_1d harness-parity check) require a full evaluate() run against the real GLADE
catalogue -- an integration-level cost, not a fast unit test -- and are DISCLOSED as deferred to the
T1.2 re-certification's own verification pass rather than attempted by this node (builder != T1.2
runner is the charter's own separation of duties; R3/R11 ARE literally what T1.2 measures).

One regression was found and fixed during this pass (not a design defect, a plumbing gap the guard-
pattern review caught): the getattr fallback at the global_denom_no_bh consumer site, described in
12.1, was needed to keep test_catalogue_global_selection.py's five p_Di-level tests (which build
BayesianStatistics via object.__new__, bypassing evaluate() and __init__ entirely) passing.

Verified: ruff check --fix and ruff format clean on every touched source file; mypy clean on the whole
darksiren_emri/ package (70 source files); the new test file plus test_theta_hook.py,
test_smear_global_selection.py, test_catalogue_global_selection.py and test_mass_filter_geometry.py all
green (85 tests); full pytest -m "not gpu and not slow" -q -p no:cacheprovider (run in two halves for
the 600 s foreground limit, darksiren_emri_test/validation split from the rest) 1915 passed / 15
skipped / 27 deselected (baseline 1896 + 19 net-new, zero regressions), reproduced on a second run.

### 12.5 What this record does NOT license

No git operations. No S0-A re-certification run (T1.2, a different agent's job). No hier_s0_driver.py
edit (T1.2 builder's job per R13, flagged loudly in 12.3). No z_window_k implementation (path choice,
not authorized by this node's task text). No change to Sigma_4D, path-A objects, the with-BH channel,
or the generator (GATE GEN-FROZEN, unchanged). Cluster inactive this node; every check above ran local
and foreground.

---

## Revision note 1 (2026-08-30; panel must_fix; append-only)

**Trigger.** The refuter panel on this document returned two must_fix items (severity major; verdict:
not refuted — neither item changes a registered number or a plan item). This note supersedes the
affected equations of section 3 IN THIS NOTE ONLY; the section-3 text above is left as originally
written and is marked SUPERSEDED below rather than edited in place (append-only discipline).

**Panel finding.** Section 3's score-identity derivation writes the estimator's per-host numerator
integrand with the UNnormalised kernel k_g^theta(z) and asserts

    INTEGRAL_d [numerator] dd = sum_g w_g S_tilde_g(theta) = Sigma_phi_smear(theta)                (as at section 3, lines 308-311)

directly from INTEGRAL k_g^theta(z) S_bar_phi(z) p(d|z) dz. But section 2.1's own definition of
S_tilde_g is normalised:

    S_tilde_g(theta; h) = INTEGRAL_{W_g^theta} k_g^theta(z) S_bar_phi(z) dz / Z_g^theta            (section 2.1, as written)
    => INTEGRAL_{W_g^theta} k_g^theta(z) S_bar_phi(z) dz = Z_g^theta * S_tilde_g(theta; h)

so the section-3 step as literally written equates INTEGRAL k_g^theta S_bar_phi dz to S_tilde_g(theta)
with no Z_g^theta factor — a step that is dimensionally inconsistent with the section-4 units table
(Z_g^theta carries the units of (dVc/dz)/(1+z), i.e. the same units as k_g^theta itself, since
S_bar_phi and dz are both dimensionless-and-measure-neutral per the section-4 table; the LHS therefore
carries those units while the asserted RHS, S_tilde_g, is dimensionless per the section-4 table's own
"S_tilde_g(theta; h) | dimensionless in [0, 1]" row). The panel additionally confirmed against the
production code that the numerator kernel actually consumed at the site-2.2/2.3 numerator is the
NORMALISED kernel n_g^theta(z) = k_g^theta(z) / Z_g^theta, not the bare k_g^theta(z) section 3 uses:

    darksiren_emri/bayesian_inference/bayesian_statistics.py:8000-8008 (single_host_likelihood_batch,
    the function section 1(e) cites):

        def _z_prior_pdf_at(z_nodes, w_pop):
            """Per-host z-prior pdf at (n, k) nodes; mirrors galaxy_redshift_prior_pdf."""
            base = _gaussian_pdf(z_nodes, host_z[:, None], host_z_error_eff[:, None])
            if _use_volume_deconv:
                return base * w_pop / z_prior_norm[:, None]          # <-- the 1/Z_g^theta normalisation
            return base

    :8011-8013 (prior_num, the numerator kernel actually multiplied by S_bar_phi at the section-1(e)
    :7366-equivalent interp step and integrated):

        prior_num = (
            None if _use_generator_point else _z_prior_pdf_at(y_num_nodes, w_pop_num)
        )

    The scalar mirror (single_host_likelihood, not the _batch function section 1(e) names, but the
    same construction) is at :7291-7295 (galaxy_redshift_prior_pdf, division at :7294:
    "base * _w_pop_eff(z) / _z_prior_norm") with :7272-7290 building _z_prior_norm (the docstring at
    :8003 states the mirror relationship explicitly). Both confirmed present at HEAD ecd33336 (line
    numbers verified directly against the file at the time of this note, superseding the approximate
    "~7277-7288" location cited when the panel finding was first raised — that range is the SCALAR
    function's _z_prior_norm construction, not the batch function's _z_prior_pdf_at/prior_num pair
    that section 1(e) actually documents).

**This satisfies must_fix item 2** (section 1(e)'s code excerpt, lines 101-114 of this document, jumps
from :7208 (z_prior_norm built) to :7366 (the S_bar_phi interp) without showing the intervening
normalisation step; :8000-8008 and :8011-8013 above are that missing step, cited here rather than
inserted into section 1(e) in place).

**Corrected derivation (must_fix item 1) — restoring the 1/Z_g^theta normalisation.**

Per-host numerator (theta-consistent, normalised kernel, matching the code cited above):

    numerator_g(theta) = (1/Z_g^theta) INTEGRAL_{W_g^theta} k_g^theta(z) S_bar_phi(z) p(d | z) dz

Integrating over the data d (INTEGRAL p(d|z) dd = 1, unchanged from the original derivation) and using
section 2.1's own S_tilde_g definition to eliminate the inner integral:

    INTEGRAL_d numerator_g(theta) dd = (1/Z_g^theta) INTEGRAL_{W_g^theta} k_g^theta(z) S_bar_phi(z) dz
                                      = (1/Z_g^theta) * [ Z_g^theta * S_tilde_g(theta; h) ]
                                      = S_tilde_g(theta; h)                                          -- the Z_g^theta cancels EXACTLY

    sum_g w_g * INTEGRAL_d numerator_g(theta) dd = sum_g w_g S_tilde_g(theta; h) = Sigma_phi_smear(theta; h)     -- unchanged conclusion

so

    L_cat(d | theta) = [ sum_g w_g numerator_g(theta) ] / Sigma_phi_reg(theta)
                     = { [ sum_g w_g numerator_g(theta) ] / Sigma_phi_smear(theta) } * const(h),

and INTEGRAL_d L_cat(. | theta) dd / const(h) = Sigma_phi_smear(theta) / Sigma_phi_smear(theta) = 1 for
EVERY theta, identically to the original (uncorrected) section-3 conclusion. Section 3's remaining
steps (the p_gen(d) construction, the theta=(0,1) identification L_cat(.|(0,1))/const = p_gen, and the
score-identity line

    E_gen[ d/dtheta ln L_cat(d | theta) ]_(0,1) = d/dtheta INTEGRAL L_cat/const dd = d/dtheta 1 = 0    (A12)

are UNCHANGED and still hold, because the corrected chain reproduces exactly the same
INTEGRAL_d L_cat/const dd = 1 identity for every theta that the uncorrected chain asserted — the
Z_g^theta factor the panel identified as missing cancels exactly against the same factor implicit in
Z_g^theta's role inside S_tilde_g, rather than surviving into the final normalisation. A12 holds.

**Units check against the section-4 table.** k_g^theta(z) carries the units of the section-4 row
"(dVc/dz)(z; h) | Mpc^3 sr^-1 ... cancels between numerator and Z_g of S_tilde_g" (the Gaussian N(.),
f_k and the (1+z)^-1 factor are all dimensionless per that table). Z_g^theta = INTEGRAL k_g^theta dz
(dz dimensionless, per the table's z-row) therefore carries the SAME units, Mpc^3 sr^-1. The corrected
numerator_g(theta) = (1/Z_g^theta) INTEGRAL k_g^theta S_bar_phi p(d|z) dz has units
[Mpc^3 sr^-1]^-1 x [Mpc^3 sr^-1] x [dimensionless S_bar_phi] x [p(d|z)] = [p(d|z)], i.e. the units of
the GW-likelihood density carried in d — matching the section-4 table's L_cat row ("[N_g] = units of
the GW likelihood density, unchanged") exactly, with no residual Mpc^3 sr^-1 factor. After the d
-integration (INTEGRAL p(d|z) dd = 1, dimensionless), INTEGRAL_d numerator_g dd carries units
[Mpc^3 sr^-1]^-1 x [Mpc^3 sr^-1] = dimensionless, matching the table's "S_tilde_g(theta; h) |
dimensionless in [0, 1]" row exactly (this is precisely the term the ORIGINAL section-3 step got wrong:
it left the extra Mpc^3 sr^-1 factor of Z_g^theta on the identification, which is what "dimensionally
inconsistent with the section-4 units table" refers to). sum_g w_g S_tilde_g(theta), with w_g in
"events per year per galaxy" (section-4 table), then carries units "events per year", matching the
table's Sigma_phi_smear/Sigma_phi_point/Sigma_phi_reg row exactly. No section-4 table row changes; the
table is confirmed consistent with the corrected chain and was already written in a form compatible
with it (the table's own S_tilde_g and Sigma_phi_smear unit entries anticipate exactly this
normalisation — only the prose derivation of section 3 had dropped the 1/Z_g^theta factor before
integrating over d).

**Scope — what is, and is not, affected.**

- SUPERSEDED (marked here, not edited in place): section 3, lines 308-311 (the two "L_cat(d | theta) ="
  lines and the "INTEGRAL [numerator] dd = ..." line immediately below them) — replaced by the
  corrected chain above. Section 3's citations list (lines 271-296), the p_gen(d) construction
  (line 303), the theta=(0,1) identification and the final A12 line (line 315) are UNCHANGED and are
  NOT superseded — the corrected chain reproduces them without modification, as shown above.
- Section 2.1's definitions (k_g^theta, Z_g^theta, S_tilde_g, Sigma_phi_smear, rho) are UNCHANGED and
  UNAFFECTED: S_tilde_g was already defined there with the /Z_g^theta normalisation the corrected
  section-3 chain uses. rho(theta; h) = Sigma_phi_smear(theta;h)/Sigma_phi_smear((0,1);h) is therefore
  unaffected — it was never computed from the flawed section-3 identity line, only from S_tilde_g and
  Sigma_phi_smear as defined in 2.1, which are and always were normalised correctly.
  E_gen[d/dtheta ln L_cat]_(0,1) = 0 (A12) holds exactly under the corrected chain, unchanged from the
  document's registered conclusion.
- Falsifiers F1-F5 (section 9) are UNAFFECTED: all five are empirical bands/pins on measured quantities
  (Z_b, Z_s, the four production rho values, the C-C identity check, the R3/R11 byte-identity pins) —
  none of them is derived from, or depends on, the section-3 algebra corrected above.
- Regression items R1-R13 (section 7, referenced throughout sections 5/9/12) are UNAFFECTED: they pin
  code behaviour (byte-identity at theta=(0,1), the s->0 delta limit, chunk invariance, degenerate-row
  handling, guard raises, CLI defaults) against the implementation as built (section 2.1/2.2), which
  already used the normalised kernel; none of them encodes or checks the section-3 prose derivation
  directly.
- The numeric predictions of sections 5.6, 6 and 9 (score_b = -0.27 +/- 0.43 / Z_b = -0.62; score_s
  staying at -0.073 +/- 0.012; the E12 enlarged-ball prediction; the cost anchors of section 6) are
  UNAFFECTED: every one of them is read from measured rho/C_b/timing values (E10, E11, t11_*_out.json)
  computed by running the code as implemented (section 2.1's normalised S_tilde_g), never from the
  section-3 prose identity, which is used only to justify WHY A12 should hold at truth, not to compute
  any registered number.
- The decision table (section 10), the A10 invariants (section 8), the sky-cone-radius instrument flag
  (section 2.5) and the section-12 implementation record are UNAFFECTED.

**Net effect.** Corrects a derivation-presentation bug (a dropped, exactly-cancelling normalisation
factor in one prose step of section 3) that the panel correctly flagged as dimensionally inconsistent
with the section-4 table as literally written. No registered number, falsifier, regression item, or
plan item changes. A12 (the score identity) holds under the corrected chain exactly as it was claimed
to hold under the original, uncorrected chain.

Verified against darksiren_emri/bayesian_inference/bayesian_statistics.py at HEAD ecd33336 (git rev-parse
confirmed 2026-08-30, same commit as this document's header). Presenter for this note: top-tier
subagent, per the branch's standing grant (row #255) covering production/documentation changes within
the tree (author-verbatim, row #223). No code written; no git operations; foreground only.
