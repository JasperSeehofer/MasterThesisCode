# /physics-change PRESENTATION GATE — [HIER] theta-consistent candidate z-window (the s-axis remainder of the S0-A instrument defect) + the z_window_k half-width knob — 2026-08-30

**Launched under row #255 — tree 2 node T1.3.** Approval stamp: **row #255 (standing grant, tree 2 node T1.3)**
(A17 standing grant: instruments and their gates BEFORE code, production-default flips inside the tree; this node is the
orchestrator's path decision of record at row #266: "T1.3 = the z-window/cone companion knob as its own gate (the
presenter's decision-table item 3), re-run s-nodes only after it; S0-B stays unlaunched"). Resolves decision-table
item 3 of PHYSICS_CHANGE_THETA_DIVISOR_20260830.md (the z_window_k companion knob, "beyond A3's literal text", "without
it F2 has no band").
Branch fix/p32d-classg-venue-repair, HEAD 5e1e66aa; working tree carries the uncommitted T1.1 divisor build
(bayesian_statistics.py, arguments.py, main.py, correspondence_1d.py: 4 files, +396/-24 at write time) — every
bayesian_statistics.py line number below is the WORKING-TREE line at 2026-08-30, re-grepped this node.
Presenter: top-tier subagent; **no code is written under this node** (presentation before code). Builder must be a
different agent from this presenter; the re-run runner a different agent from the builder. Every [HIER] statement
carries the REPORTED-ONLY cap (PA-HIER-28 item 9). No backtick characters in this record. Every number carries
{value, source, date}. NO ssh, NO git, foreground only; the T1_2_* files, the HIER prereg and the ledger were not
touched (concurrent writers); row #266 is quoted from the ledger as read.

Companion zero-compute instrument (this node, foreground, 2 x 40 s, zero evaluate() calls, no source edits):
results/campaign51_20260728/realistic_20260729/tree2_20260830/t1_3_gate_work/t13_residual_and_capture.py -> t13_out.json.
Inputs: the T1.2 re-certification CSVs (hier_s0_recert_run/s0a_seed9001{01..04}/node_*_sites2.2_nosmear_divisor/
simulations/diagnostics/event_likelihoods.csv, h = 0.73, dedupe keep last) joined 1:1 on (seed, event_idx) to the
forensic's per-event table fanout1_20260829/b1_1_forensic_work/f7_events.csv (true-host z_g, sigma_g, the E6-exact
ball bounds z_min_ball/z_max_ball, z_GW, n_cand, class flags, c_nb, Es_null_det, mu_k, sd_k). The join reproduces
s0a_score.md to the last digit (score_b -0.28878 +/- 0.42705, score_s -0.07196 +/- 0.01205; t13_out.json
pooled_recert, 2026-08-30) — the recert artefacts are read correctly.

---

## 0. Scope and one-paragraph summary

After T1.1 (theta-consistent no-BH divisor) the S0-A re-certification restored the b-axis (Z_b -0.68, row #266) and
left the s-axis at score_s = -0.0720 +/- 0.0121 (Z_s -5.97, raw linear form; s0a_score.md 2026-08-30), exactly the
divisor-only prediction (-0.0728 +/- 0.0122, T1.1 section 5.6). The forensic attributed the s-axis to the candidate
BALL (E9: the enlarged ball moves the catalogue-leg s-secant -0.052 -> +0.019; E12: both corrections -> Z_s -0.5).
This node identifies the MECHANISM inside the ball and registers its fix. The candidate z-window
(handler.py:668-676) accepts a galaxy iff its LISTED redshift, widened by +/- 1 x its QUOTED sigma_g, overlaps the
GW envelope [z_min, z_max] (physical_relations.py:563-566: d_L -/+ 3 sigma_dL at h_min/h_max). Under theta = (b, s)
the estimator's kernel for the same galaxy is N(z; z_g + b(1+z_g), s sigma_g) (site 2.2, bayesian_statistics.py
:8113-8117), so the selection window and the kernel it selects for are DIFFERENT objects whenever theta != (0,1):
the window neither moves with b nor scales with s. The score identity (A12) then acquires a normalisation-defect term
d/dtheta ln C(theta), C = the kernel-mass fraction the fixed window captures, which is negative in s (a wider
kernel loses more of its mass outside a fixed window) — the sign and, in a capture model built on the drawn hosts'
own kernels and E6-exact ball bounds, the SIZE of the whole T1.2 residual (model -0.074 to -0.105 vs measured
-0.100 +/- 0.012 in PA-HIER-32's debiased statistic; t13_out.json, 2026-08-30). The residual's structure confirms
the mechanism at zero compute: it lives ENTIRELY in the quartile of events whose window is narrowest in sigma_g
units (half-width < 1.08 sigma_g: -0.254 +/- 0.037, Z -6.9; the other three quartiles: -0.042 +/- 0.019,
+0.011 +/- 0.010, -0.006 +/- 0.012), in the low-candidate-count half (n_cand <= 281: -0.13; >= 1261: +0.005 +/- 0.010),
and NOT in the sky-excluded class (-0.019 +/- 0.029, n = 64) — the sky cone is not where the s-defect is.
Registered form: a flag theta_zwindow in {"off","on"} (default "off", byte-identical) that builds the galaxy-side
term of the z-filter from the THETA-TRANSFORMED kernel (centre z_g + b(1+z_g), width sqrt((s sigma_g)^2 +
sigma_pv,g^2)), plus the half-width knob z_window_k (default 1.0, byte-identical). The derivation shows the
transform alone (k = 1) removes only the +/- k sigma part of the s-dependence — 55 % of the residual in both model
variants — because the GW envelope's half-width, a theta-independent GW-side object, still cuts the kernel at a
number of kernel-sigmas that scales as 1/s; the selection becomes literally the same object as the kernel only when
k equals the kernel's own integration half-width, k = 4 = integration_limit_sigma_multiplier (:8101), at which
point the selection interval IS the site-2.2 window W_g^theta and the capture term vanishes at every node (model:
99.9 % removed). The registered decisive arm is therefore theta_zwindow = "on", z_window_k = 4.0, sky_cone_k = 1.5
(the sky cone carries no theta, section 2.5); the k = 1 arm is the mechanism-isolating control. Predictions,
cost (2.5 h wall for the decisive arm), regression plan, invariants, falsifiers and the decision table follow.

---

## 1. OLD formula (exact, as implemented in the working tree at 2026-08-30, with lines)

**(1a) The GW-side redshift envelope — theta-free, UNCHANGED by this presentation.**
physical_relations.py:546-567, get_redshift_outer_bounds(distance, distance_error, h_min, h_max, ..., sigma_multiplier):

    :563    z_min = dist_to_redshift(distance - 3 * distance_error, h_min)
    :564-565    if distance - 3 * distance_error < 0: z_min = 0.0
    :566    z_max = dist_to_redshift(distance + 3 * distance_error, h_max)

DISCLOSED (not changed here, code-health item): the sigma_multiplier parameter is DEAD — the body hardcodes 3; the
call site passes sigma_multiplier=2.0 (bayesian_statistics.py:5631), which is inert. The forensic's "+/- 3 sigma_d"
(E6) is the true form. Call site (bayesian_statistics.py, p_D):

    :5624-5632    z_min, z_max = get_redshift_outer_bounds(distance=self.detection.d_L, distance_error=self.detection.d_L_uncertainty,
                       h_min=self.cosmological_model.h.lower_limit, h_max=self.cosmological_model.h.upper_limit, ..., sigma_multiplier=2.0)
    :5634         z_max = min(z_max, redshift_upper_limit)                      # GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT = 1.55 (constants.py:102)

h bounds: mirror venue H_BOUNDS = (0.50, 0.86) (hier_s0_driver.py:79-84, :408, :715 -> correspondence_1d.py:2954-2959);
production cosmological_model.py:388-389 = (0.6, 0.86). Measured envelope on the S0-A events: z_min/z_GW median 0.656,
z_max/z_GW median 1.241, half-width median 1.45 sigma_g of the true host {f5_out.json z_window_rel; f7_events.csv,
2026-08-29}; this node: hw_sig quartile edges 1.08 / 1.45 / 1.95 sigma_g, range 0.40-9.30 {t13_out.json, 2026-08-30}.

**(1b) The candidate ball — the sky cone (theta-free, UNCHANGED) and the z-filter (the defect locus).**
handler.py get_possible_hosts_from_ball_tree (:558-770):

    :662    radius = float(sigma_multiplier * np.sqrt(max(lambda_max, 0.0)))        # lambda_max of J Sigma_sky J^T; sigma_multiplier = self._sky_cone_k (1.5)
    :664    indices = self.catalog_ball_tree.query_radius(query_point, r=radius)[0]
    :668-676    redshift_filter_mask = (z_min <= candidate_hosts[REDSHIFT] + candidate_hosts[REDSHIFT_ERROR])
                                     & (z_max >= candidate_hosts[REDSHIFT] - candidate_hosts[REDSHIFT_ERROR])
    :677    candidate_hosts_without_bh_mass = candidate_hosts[redshift_filter_mask]

i.e. z_min <= z_g + 1 x sigma_g and z_max >= z_g - 1 x sigma_g, with z_g the listed redshift and sigma_g the quoted
REDSHIFT_MEASUREMENT_ERROR column (which already carries the parse-time PV fold, handler.py:461-479). No theta enters:
the ball is built before p_Di, from Detection and catalogue columns only (bayesian_statistics.py:5636-5650, the single
call site; :5646 sigma_multiplier=self._sky_cone_k). The mass filter (:688-753) then acts on the z-filtered set with the
GW-side z_min/z_max in (M_z -/+ k sigma)/(1 + z_max/z_min) — theta-free in form.

**(1c) The kernel the window is supposed to select for — site 2.2 (single_host_likelihood_batch), theta-transformed.**

    :8101    integration_limit_sigma_multiplier = 4.0
    :8107    sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S            # SIGMA_V_PEC_KM_S = 0.0 (constants.py:95)
    :8108    host_z_error_eff = np.sqrt(host_z_error**2 + sigma_z_pv**2)
    :8109    if theta_b != 0.0 or theta_s != 1.0:
    :8115        _theta_hook_count("site_2_2")
    :8116        host_z_error_eff = np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2)
    :8117        host_z = host_z + theta_b * (1.0 + host_z)
    :8126    den_hi = host_z + integration_limit_sigma_multiplier * host_z_error_eff
    :8143-8144    den_lo = np.maximum(host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor)   # 1e-6

(scalar twin single_host_likelihood: :7321, :7407-7408, :7418, :7426-7427.) So the estimator integrates each selected
candidate's kernel over W_g^theta = [z_g^theta - 4 sigma_g^theta, z_g^theta + 4 sigma_g^theta] while having selected it by
[z_g - 1 sigma_g, z_g + 1 sigma_g] overlapping the envelope — two different objects even at theta = (0,1) (k = 1 vs 4),
and objects that move apart as theta leaves (0,1). The first is the production approximation V3 (forensic E6/E14);
the second is the theta-inconsistency this node registers.

**(1d) The divisor after T1.1 — for the record, UNCHANGED by this presentation (section 2.4).**
Sigma_phi_reg(theta; h) = Sigma_phi_point(h) x rho(theta; h), rho from precompute_phi_divisor_theta_ratio
(bayesian_statistics.py:3206-...), summing S_tilde_g(theta) over ALL eligible rows (listed z_g < z_max(h), the D2 mask)
with each S_tilde_g integrated over W_g^theta — no per-event window enters it.

---

## 2. NEW formula (the registered form)

### 2.1 Definitions, per event i and per candidate row g in the sky cone, under theta = (b, s)

    z_g^theta      = z_g + b (1 + z_g)                                               (HIER section 1.2; site 2.2 :8117)
    sigma_pv,g     = (1 + z_g) SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S               (from the UNSHIFTED z_g, as at :8107; = 0.0 today)
    sigma_g^theta  = sqrt( (s sigma_g)^2 + sigma_pv,g^2 )                            (s on the RAW quoted error, as at :8116)
    k              = z_window_k                                                       (new knob, default 1.0)

**Registered z-filter (galaxy side transformed, GW side untouched):**

    accept g  iff  z_min <= z_g^theta + k sigma_g^theta   and   z_max >= z_g^theta - k sigma_g^theta

with z_min, z_max exactly the objects of (1a) (3 sigma_dL envelope at h_min/h_max, clamped at 1.55). Sky cone (:662-664),
mass filter (:688-753) unchanged in form; the mass filter's INPUT set inherits the new z-filter (disclosed: both
channels' candidate sets move whenever the z-filter does, as with sky_cone_k).

Reading: the selection interval [z_g^theta - k sigma_g^theta, z_g^theta + k sigma_g^theta] is the +/- k sigma support of the
SAME kernel N(z; z_g^theta, sigma_g^theta) that site 2.2 integrates for g; "accept iff that support meets the GW
envelope". At k = 4 = integration_limit_sigma_multiplier the interval is W_g^theta itself (up to the 1e-6 floor,
which only truncates below and cannot create overlap with a z_min > 0): selection and kernel are then literally one
object at every theta.

### 2.2 The switch, the knob, guards, counter, plumbing (the B5.1 / T1.1 flag pattern)

    theta_zwindow: str = "off"       # {"off","on"}; evaluate() / CLI --theta_zwindow / run_mirror_seed_inprocess / hier_s0_driver.py
    z_window_k: float = 1.0          # evaluate() / CLI --z_window_k / ...; finite and > 0 else ValueError; single read/validate site in handler.py

- "off" (default): handler.py:668-676 executes exactly as today with k = z_window_k and the bare sigma_g; at k = 1.0
  the mask, the candidate lists (indices AND order) and every downstream value are byte-identical (R1).
- "on" and theta == (0, 1): LITERAL SKIP — the "off" path with k = z_window_k; no floating operation on theta (R2).
  (Even without the skip: z_g + 0.0 x (1+z_g) == z_g and sqrt((1.0 sigma)^2 + 0.0) == sigma bit-for-bit in IEEE-754
  round-to-nearest absent over/underflow; the skip is registered so that no floating-point argument is load-bearing.)
- "on" and theta != (0, 1): the 2.1 mask. theta reaches the handler as two new keyword arguments theta_b, theta_s
  (defaults 0.0, 1.0) of get_possible_hosts_from_ball_tree, passed from BayesianStatistics's stored self._theta_b /
  self._theta_s at the single call site :5636-5650 ONLY when the flag is "on" — the handler is a non-physics file and
  keeps no theta state; the flag and k are read/validated at that one site (the mass_filter_k precedent).
- Independent of theta_sites and of theta_phi_divisor (composes with the form of record theta_sites="2.2", smear off,
  divisor "on"; valid with either). Guards (raise, never a silent no-op): invalid token; z_window_k not finite or <= 0;
  "on" with theta_b/theta_s failing _validate_theta (:1640-1643).
- Hook counter key "site_2_2_window" in _THETA_HOOK_COUNTERS (:1625-1633), incremented once per event when engaged;
  the decisive engagement evidence is the candidate-count change and the per-event ln L diagnostics (PA-HIER-23),
  never the counter alone.
- Instrumentation: the existing per-event "Found N possible hosts" INFO line (handler.py:764-766) and the P6
  host-recovery counters already carry the engagement evidence; run_metadata captures the two new arguments via
  Arguments.to_dict(). The T2.2 candidate_dump_dir hook (opt-in) serialises the candidate sets for R4.
- Driver: hier_s0_driver.py needs --theta_zwindow {off,on} and --z_window_k (defaults byte-identical) threaded to its
  three run_mirror_seed_inprocess call sites, and a node-dir suffix "_zwin<k:g>" appended when the flag is "on"
  (the T1.1 driver-gap lesson, T1_1 record 12.3: an arm launched without the driver flag runs at default and SILENTLY
  reproduces the defect). The builder must show the suffix in the smoke cell's directory name before the arm runs.

### 2.3 Why the transform alone is not enough — the capture term and the role of k (derivation summary; full form in section 3)

With the fixed window, the kernel-mass fraction the window captures for row g is (bare-Gaussian form, sigma_pv = 0)

    c_g(theta) = P_{z ~ N(z_g^theta, s sigma_g)} ( f_lo z - k sigma_g <= z_g <= f_hi z + k sigma_g ),   f_lo = z_min/z, f_hi = z_max/z,

i.e. the source must fall in [ (z_g - k sigma_g)/f_hi , (z_g + k sigma_g)/f_lo ]; in units of the kernel width the two
edges sit at -[ (1 - 1/f_hi) zeta + k/f_hi ]/s and +[ (1/f_lo - 1) zeta + k/f_lo ]/s with zeta = z_g/sigma_g: BOTH the
envelope term (proportional to zeta) and the +/- k sigma_g term scale as 1/s. Under the 2.1 transform the k-term becomes
k s sigma_g / (s sigma_g) = k (theta-invariant) while the envelope term still scales as zeta/s: the GW envelope is a
fixed z-interval, and the fraction of a kernel of width s sigma_g inside a fixed interval necessarily depends on s.
On the S0-A hosts (zeta median 4.7, range 1.0-31; f_lo 0.656, f_hi 1.241 medians) the envelope term dominates the
lower edge (0.194 zeta vs 0.806 k at k = 1), so the transform removes only the k-part: 55 % of the s-response in
both model variants (section 3, table). The residual dependence dies only when the interval is wide in kernel-sigma
units at EVERY node, i.e. when k s sigma_g^theta reaches the kernel's own support: at k = 4 the capture is 1.0000 /
1.0000 / 0.9999 at s = 1/sqrt2 / 1 / sqrt2 (bare) and 1.0000 at all three (tilted) — the s-response is gone
(section 3). This is why the registered decisive arm is k = 4, not k = 1, and why z_window_k is not an optional
cosmetic knob but the parameter that makes "selection = kernel" true.

### 2.4 Must the divisor Sigma_phi(theta) ALSO be evaluated over the theta-transformed candidate window? — NO (registered)

The divisor is the population-integrated detection probability under hypothesis theta (Mandel, Farr & Gair 2019
Eqs. 5-7; Gray et al. 2020 Eq. 29): Sigma_phi(theta; h) = sum over ALL eligible catalogue rows of w_g S_tilde_g(theta; h).
It is data-INDEPENDENT by definition — it is the integral of the numerator over all possible data. In that integral the
per-event candidate window does not truncate the population sum; it becomes the capture factor c_g(theta) of 2.3
(section 3, line "integral over d"). A divisor restricted to a per-event window would be a different object per event
and would not be the normaliser of anything. T1.1's form therefore stands unchanged: eligibility = the D2 mask on the
listed z_g (identical to Sigma_phi_point's and Sigma_4D's), each S_tilde_g integrated over ITS OWN W_g^theta (T1.1
section 2.1). Consistency between T1.1 and T1.3 is at the level of principle, and exact at k = 4: T1.1 made the
divisor's per-host integration support follow the theta-kernel; T1.3 makes the numerator's candidate selection follow
it; at k = 4 both supports are the same interval W_g^theta. The one remaining asymmetry — the divisor's eligibility on
the LISTED z_g rather than z_g^theta near z_max(h) = 1.55 — concerns rows within |b|(1+z) <= 0.05 of the catalogue cap,
where S_bar_phi is at the LISA EMRI horizon floor; it is T1.1's own D2 decision, unchanged, and bounded by T1.1 8(f)-class
weight shares (not re-measured here; REPORTED-ONLY).

### 2.5 Does the sky cone need theta? — NO (registered); sky_cone_k stays a separate, theta-free instrument knob

The cone radius (handler.py:662) is sky_cone_k x sqrt(lambda_max(J Sigma_sky J^T)) with Sigma_sky the GW Fisher sky block
(phi_sigma, theta_sigma, cov_theta_phi from Detection) and J = diag(|sin theta|, 1): it contains no redshift, no
sigma_z and no catalogue column. theta = (b, s) is a model of the catalogue's photo-z ERROR (the host-z kernel,
HIER section 1.2) and says nothing about a galaxy's sky position (exact in the catalogue) or about the GW sky
posterior; a "theta-consistent cone" would have nothing to transform. Consequence for the score: the sky capture is
a theta-INDEPENDENT multiplicative factor on c_g (the ball uses the marginal 2x2 sky block and the marginal d_L, so
sky- and z-capture factorise to the order the ball itself assumes; the sky/d_L Fisher correlation is a second-order
disclosure) and cancels from d/dtheta ln C(theta). It survives only in the V3-intrinsic term (WHICH hosts the mixture
contains: E6 14.5 % true-host sky exclusion; E14 impostor share). Empirical confirmation at zero compute: the
sky-excluded class carries NO s-residual (-0.019 +/- 0.029, n = 64) while the recovered class carries it
(-0.079 +/- 0.013, n = 387) and the z-excluded class carries the largest (-0.293 +/- 0.086, n = 8) {t13_out.json
by_class, 2026-08-30}. The E12 configuration (sky 3.0 AND z +/- 4 sigma_g) enlarged both; its s-null did not need the
sky part. sky_cone_k = 1.5 in the decisive arm; sky 3.0 is the diagnostic fallback arm P3 (section 5.6) if P1 fails.

### 2.6 The E20 edge case (b < 0, negative kernel centre), registered rule

At b < 0 rows with z_g < |b|/(1+b) get z_g^theta < 0. The 2.1 mask handles them without a special case: a row is
accepted iff z_g^theta + k sigma_g^theta >= z_min with z_min >= 0 — a row whose transformed support lies entirely below
the physical floor is never a candidate (consistent with T1.1 section 2.4: zero physical support => zero survival
mass => not a host); a row with a negative centre whose support reaches the envelope is accepted and integrated by
site 2.2's floored quadrature exactly as today. The site-2.2 Z_g <= 0 -> 1.0 guard (T1.1 decision-table item 6)
stays untouched.

---

## 3. Reference (citations) and derivation

**Citations.**
- B1_1_S0A_DEFECT_FORENSIC_20260829.md: E6 (the ball model, reproduced exactly: 91/106, 105/120, 87/105, 104/130;
  sky-cone exclusion 14.5 %, z-window 2.2 %; half-width 1.45 sigma_g median); E9 (enlarged ball: catalogue-leg
  s-secant -0.052 +/- 0.022 -> +0.019 +/- 0.016; b +1.94 -> +1.17 +/- 0.69); E12 (both corrections, c-weighted:
  s -0.005 +/- 0.011, Z -0.5; b -0.78 +/- 0.47); E13 (secant bias +0.0455 unweighted / +0.0265 c-weighted); E14/E15
  (impostor share; kernel tilt +0.91 ... -0.71 sigma by z_g bin); E20 (edge case); section 3 s-axis account ("the
  surviving candidates are, by selection, closer than 1 sigma to the data, so the likelihood prefers narrower
  kernels"); section 5 (the s-axis test needs the ball as a flag; prediction Z_s -> -0.5 +/- 1; cost 3-6x per cell).
- PREREGISTRATION_HIER_HTHETA_20260826.md: section 1.2 (theta identity; s on the raw error; SIGMA_V_PEC_KM_S = 0.0
  load-bearing); 2.1 (S0-A certifies, never discovers); 4.1 (bands, B0-A / B0-A'); 5.1 invariants 2 (h_bounds
  (0.50, 0.86)), 7; 5.2 items 3, 5, 6; PA-HIER-31 (b)/(d)/(g); PA-HIER-32 (d) (the debiased score_s of record and
  its scope note: Es_null_det_i must be RE-DERIVED for any new configuration).
- PHYSICS_CHANGE_THETA_DIVISOR_20260830.md (T1.1): sections 2.1 (W_g^theta, S_tilde_g), 2.5 (the cone flag and the
  z_window_k registration with its E12 pairing), 3 + Revision note 1 (the score identity with the normalised kernel),
  5.6 / 9 F2 (the s-axis prediction with the divisor only, and the enlarged-ball band), decision table item 3.
- BIAS_HISTORY_LEDGER.md row #266 (T1.2 readout: b CERTIFIED, s STOP stands; the raw-vs-PA-HIER-32 statistic
  tension disclosed; T1.3 path decision of record); row #183 ([P3-HGRID]: the candidate-ball z-window's dependence on
  the h-bounds PINNED as a real effect on the with-BH channel — the same envelope object as 2.3's zeta-term).
- Ma, Hu & Huterer (2006) ApJ 636, 21, arXiv:astro-ph/0506614, sec. 2 (the affine photo-z systematic).
- Mandel, Farr & Gair (2019) MNRAS 486, 1086, arXiv:1809.02063, Eqs. (5)-(7) (selection-normalised likelihood: the
  normaliser integrates over all data; a data-dependent truncation of the numerator sum enters the identity as a
  capture probability, not as a truncation of the normaliser).
- Gray et al. (2020) PRD 101, 122001, arXiv:1908.06050, Eq. (29), Eq. (A.10).
- Research-cycle amendment A12 (.claude/skills/research-cycle/SKILL.md:149-157): the score identity.

**Derivation (mirror b0i, no-BH channel; the T1.1 section-3 identity with a data-dependent candidate set).**
Let B_theta(d) be the candidate set for data d under the (possibly theta-dependent) window, n_g^theta the normalised
site-2.2 kernel, N_g(d; theta) = integral n_g^theta(z) S_bar_phi(z) p(d | z, Omega_g) dz, and

    L_cat(d | theta) = sum_{g in B_theta(d)} w_g N_g(d; theta) / Sigma_phi_reg(theta).

Integrating over the data (integral p(d|z) dd = 1) and exchanging the sum and the integral,

    integral L_cat(d | theta) dd = [ sum_g w_g S_tilde_g(theta) c_bar_g(theta) ] / Sigma_phi_reg(theta) =: C(theta) x const(h),
    c_bar_g(theta) = integral n_g^theta(z) S_bar_phi(z) P( g in B_theta(d) | d ~ p(. | z, Omega_g) ) dz / S_tilde_g(theta),

so C(theta) is the w S_tilde-weighted mean capture fraction. Writing L_cat = q(d|theta) C(theta) const with q a
normalised density, the generator expectation of the score at truth splits into

    E_gen[ d/dtheta ln L_cat ]_(0,1) = d/dtheta ln C(theta)  +  C x Cov_q( d/dtheta ln q , 1/r(d) ),   r(d) = in-ball share of the full mixture at d,

the first term the NORMALISATION DEFECT of a theta-inconsistent selection (zero iff C is theta-flat), the second the
V3-intrinsic mismatch (events whose true-host-type mass sits outside the ball; zero iff r == 1). With the T1.1 divisor
the divisor's own theta-dependence is already consistent, so the s-axis residual is these two terms plus V2/V4 (T1.1
section 3). The registered change acts on the first term: under the 2.1 window with k = 4 the capture is unity at
every node (C flat => term 1 = 0); it also shrinks the second term by moving hosts from outside to inside the ball
(r -> 1 for the z-side; E6: 2.2 % of true hosts z-excluded, the z_out class's -0.293 +/- 0.086).

**The capture model (zero compute, this node) and its calibration.** For each of the 456 matched S0-A events the
true host is a draw from w_g S_tilde_g (the generator law, PA-HIER-19) and its event's ball bounds are E6-exact;
the mean of c_g(theta) over these hosts, with their own (f_lo, f_hi), estimates C(theta). Two kernel variants:
BARE = N(z_g^theta, s sigma_g); TILTED = N(mu_k + b(1+z_g), s sd_k) with (mu_k, sd_k) the actual C7-core kernel moments
of each host (f7_events.csv; the E15 tilt to first order in theta). The registered-statistic shift is <c_i> x the
lns-secant of ln C, <c_i> = 0.616 (f7 c_nb over the 456 matched events). All numbers {t13_out.json capture_model /
capture_model_tilted, 2026-08-30}:

| window | k | C(s = 1/sqrt2, 1, sqrt2) bare | registered s-term bare | registered s-term tilted | fraction of the fixed-k1 term removed (bare / tilted) |
|---|---|---|---|---|---|
| fixed (today) | 1 | 0.9862 / 0.9482 / 0.8762 | **-0.105** | **-0.074** | — |
| fixed | 2 | 0.9994 / 0.9918 / 0.9591 | -0.037 | -0.017 | 65 % / 77 % |
| fixed (E12's z-widening) | 4 | 1.0000 / 1.0000 / 0.9977 | -0.002 | -0.0004 | 98 % / 99 % |
| theta-consistent (2.1) | 1 | 0.9707 / 0.9482 / 0.9200 | -0.048 | -0.034 | **55 % / 55 %** |
| theta-consistent | 2 | 0.9958 / 0.9918 / 0.9861 | -0.009 | -0.004 | 92 % / 95 % |
| theta-consistent | 3 | 0.9996 / 0.9992 / 0.9985 | -0.001 | -0.0002 | 99.0 % / 99.7 % |
| **theta-consistent** | **4** | 1.0000 / 1.0000 / 0.9999 | **-0.0001** | **0.0000** | **99.9 % / 100 %** |

Calibration against measurement: (i) the measured T1.2 residual in PA-HIER-32's debiased statistic (c-weighted
convention, section 5.6) is -0.0997 +/- 0.0123 (Z -8.08) {t13_out.json pooled_recert ss_deb_c}: the bare model
reproduces it (-0.105), the tilted model leaves -0.026 unexplained; (ii) E9/E12's measured c-weighted s-move from the
banked ball to the enlarged ball is +0.067 (from -0.072 divisor-only to -0.005) {forensic E11, E12}: the tilted model
predicts +0.074, the bare +0.103 — the tilted variant is the calibrated one on the enlargement, and its -0.026
leftover equals E12's own debiased residual (-0.005 - 0.0262 = -0.031 +/- 0.011), i.e. the non-capture terms
(V2/V4/sky-intrinsic), which no window fix removes; (iii) the residual by window-half-width quartile — model (tilted /
bare) vs measured: q1 (hw < 1.08 sigma_g) -0.143 / -0.232 vs **-0.290 +/- 0.038**; q2 -0.089 / -0.132 vs -0.074 +/- 0.019;
q3 -0.051 / -0.075 vs -0.013 +/- 0.010; q4 -0.027 / -0.038 vs -0.026 +/- 0.013 {t13_out.json
*_model_vs_measured_by_hw_sig_quartile}: the right ordering and the right concentration, within a factor 2 per
quartile. The capture term is the s-axis mechanism, and k = 4 with the transform removes it entirely in both variants.

**Structure of the T1.2 residual by other covariates (all raw linear form, combined_no_bh, {t13_out.json, 2026-08-30}).**
n_cand quartiles (<= 43 / 47-281 / 282-1237 / >= 1261): -0.135 +/- 0.038, -0.127 +/- 0.022, -0.034 +/- 0.015,
+0.005 +/- 0.010 (correlation with ln n_cand +0.21). zeta = z_g/sigma_g quartiles (< 3.53 / 3.54-4.73 / 4.73-5.98 /
> 5.99): -0.228 +/- 0.035, -0.039 +/- 0.021, +0.015 +/- 0.016, -0.039 +/- 0.014. Catalogue-share quartiles (c < 0.435 /
0.44-0.65 / 0.65-0.84 / > 0.837): +0.014 +/- 0.008, +0.046 +/- 0.016, -0.012 +/- 0.017, **-0.338 +/- 0.031 (Z -11.0)**.
Per seed (raw): -0.091 +/- 0.026, -0.104 +/- 0.027, -0.039 +/- 0.020, -0.054 +/- 0.022. Dark class (n = 5): exactly 0.0.
Bank (divisor off) vs recert (divisor on) per-event s-secant correlation 0.99978 — the divisor moved every event by
the per-node scalar only, as T1.1 R3 requires. The c > 0.84 class concentrating the residual is the V2 covariance's
signature (E17) AND the low-z / narrow-window class — the two are the same events on b0i; the c-stratified read of the
P1 CSVs (section 9, F1 next-candidate) separates them at zero compute.

**The b-axis under the fixed window (disclosed model limit).** The same model gives the fixed-k1 window a POSITIVE
b-term (+2.33 bare / +1.28 tilted registered) from the envelope's asymmetry (the lower edge at 0.806(z_g - k sigma_g)
is the near one), which is 2.6-5x larger than E9/E12's measured b-move under ball enlargement (catalogue-leg
+1.94 -> +1.17; registered -0.27 -> -0.78 +/- 0.47) {t13_out.json E9_b_check}. The b-capture is set by the kernel's
CENTRE against the near edge, where the C7 tilt (first order only in the tilted variant) and the S_bar_phi slope act
at order one; the s-capture is set by the WIDTH, where they are second order. The model is therefore registered for
the s-axis only; the b-axis prediction of 5.6 is E12-calibrated, not model-derived.

---

## 4. Dimensional analysis

| symbol | units | check |
|---|---|---|
| z_g, z_min, z_max, z_g^theta, b | dimensionless | z + b(1+z): dimensionless throughout |
| sigma_g (REDSHIFT_MEASUREMENT_ERROR), s, sigma_pv,g, sigma_g^theta, k | dimensionless | sqrt((s sigma)^2 + ((1+z) v/c)^2): km/s over km/s; k sigma: pure number x pure number |
| d_L, sigma_dL | Mpc | enter only through dist_to_redshift (unchanged) |
| the mask | boolean | compares dimensionless quantities of identical kind (a redshift against a redshift) |
| lambda_max, radius | rad^2, rad (chord on the unit sphere) | untouched |
| L_cat, N_g, Sigma_phi | unchanged | the change alters WHICH rows are summed, never the units of any summand |

No numerical value is produced by the change; a candidate SET is. No mixed units anywhere.

---

## 5. Limiting cases (and the registered predictions, 5.6)

**5.1 theta = (0, 1) — the identity (GATE T-ID).** Literal skip: the "off" mask with k = z_window_k. At k = 1.0 the
candidate lists are byte-identical to today's (R1/R2). Forced computation gives the same bits (2.2).

**5.2 s -> 0 (delta kernel).** sigma_g^theta -> sigma_pv,g = 0: accept g iff z_min <= z_g^theta <= z_max — the delta host
is a candidate iff its (shifted) redshift lies inside the GW envelope, exactly the delta limit of "kernel support meets
the envelope" and consistent with site 2.2's own s -> 0 limit (the kernel -> delta(z - z_g^theta), T1.1 section 5.2).

**5.3 s -> infinity (window growth).** sigma_g^theta -> infinity: every row in the sky cone passes the z-filter — the
candidate set becomes the whole cone (then the mass filter). Consistent: an infinitely wide kernel has support
everywhere; the estimator must sum over everything the cone admits. Cost grows with the count; the with-BH channel's
set grows identically (disclosed). Monotonicity: the accepted set at s_2 > s_1 contains the set at s_1 (at fixed b, k).

**5.4 b shift at s = 1.** The interval translates rigidly by b(1+z_g) with the kernel centre: rows at the envelope
edges swap in/out; at b = -0.02 on b0i the shift is 0.02(1+z) ~ 0.6 sigma_g for the median host (sigma_g 0.038,
z 0.18) — a visible change of the candidate set (R3 engagement). E20 rows: 2.6.

**5.5 k -> 4 = integration_limit_sigma_multiplier.** The selection interval equals W_g^theta (:8126, :8143-8144); the
capture is unity at every node (section 3 table); the z-side of V3 collapses (the z_out class, n = 8, is recovered).
k -> infinity: the whole cone, as 5.3.

**5.6 Registered predictions (PA-HIER-32's debiased statistic is PRIMARY; the raw prereg-4.1 linear form is reported
alongside because the driver's compute_scores() still emits it — row #266's disclosed tension; the CONVENTION for the
combined channel — Es_null_det_i raw vs c_i x Es_null_det_i — must be fixed by the runner BEFORE unblinding P1; this node
registers the c-weighted form as primary because d ln combined_no_bh = c_i d ln L_cat to first order (forensic section 3),
so the deterministic secant expectation of the combined channel is c_i x Es_null_det_i; T1.2 in that form: score_s
-0.0997 +/- 0.0123, Z -8.08; raw convention -0.1188 +/- 0.0121, Z -9.86; Es_null_det means +0.0454 / +0.0262 c-weighted,
reproducing E13's +0.0455 / +0.0265 {t13_out.json, 2026-08-30}). Per PA-HIER-32's scope note, Es_null_det_i is per host
and configuration-free, but c_i changes with the candidate set, so the c-weighted convention needs the ARM's own truth
node (included in P1).**

**P1 — the decisive arm (registered): theta_zwindow = "on", z_window_k = 4.0, sky_cone_k = 1.5, theta_phi_divisor = "on",
theta_sites = "2.2", smear off, "off" numerator (the bc driver), h = 0.73, 4 seeds, nodes {truth, s_plus, s_minus}.**
- Debiased c-weighted score_s: capture term removed (tilted model shift +0.074 from -0.0997 -> -0.026; bare +0.105 ->
  +0.005); the non-capture residual measured by E12 (-0.031 +/- 0.011 debiased) bounds the low side. Registered
  point: **score_s = -0.026 +/- 0.012 (Z_s ~ -2.1)**, band of the point **[-0.031, +0.005] (Z_s in [-2.6, +0.4])**;
  registered BAND: **|Z_s| <= 3.0** (prereg 4.1 B0-A, restated by PA-HIER-32). Raw linear form (the E12 quote):
  score_s in [0.000, +0.031], Z_s in [0, +2.5]; E12's measured raw point -0.005 +/- 0.011 (Z_s -> -0.5 +/- 1.0,
  sky 3.0) is the closest measured analogue and is quoted as the reference. The debiasing term itself is 2.1 SEM,
  so the two conventions differ by more than the margin to the band edge — hence the pre-registration of the
  convention above.
- Structure predictions (zero-compute reads of the P1 CSVs joined to f7): the hw_sig-q1 class moves from -0.290 to
  within +/- 3 SEM of the other quartiles' mean; the n_cand distribution's median grows from 278 by a factor in
  [2.2, 3.6] (linear-extent ratio 2.22; z^2-weighted 2.95 median, IQR 2.53-3.56; n_cand-weighted 2.64
  {t13_out.json count_growth_k4_sky15}); the z_out class (n = 8) is recovered (P6 counters: z_in 447/456 -> 456/456
  or within 2); the dark class stays exactly 0.0 (F4).
- Exactness prediction for the implementation: at the P1 truth node, every event's L_cat_no_bh differs from T1.2's
  truth node ONLY through the added candidates (the T1.2 candidates' contributions are bit-identical — testable via
  the T2.2 candidate dump: the intersection rows carry identical per-candidate values, R4).
- b-axis: NOT re-run under P1 (the flag at default is byte-identical, so T1.2's b-certification stands for the form
  of record). If the b-nodes are re-run at k = 4 (optional arm P1b, 8 cells): E12-calibrated point score_b -> -0.78 +/- 0.47
  (Z_b -1.7; T1.2 -0.29 +/- 0.43, i.e. within 1.1 SEM), band |Z_b| <= 3; the section-3 model bound reaches
  -1.6 (Z_b -3.7) and is disclosed as the pessimistic edge; a b-band failure there would be a NEW finding (the fixed
  window's positive b-term masking a negative residual), not a refutation of this node's s-attribution (F5).

**P2 — the mechanism-isolating control (optional, path choice): as P1 with z_window_k = 1.0 (the transform alone).**
Debiased c-weighted score_s: -0.042 (bare) to -0.059 (tilted) +/- 0.012, Z_s in [-4.8, -3.4] — PARTIAL, outside the
band, removing 55 % of the fixed-window capture term (+0.041 to +0.057 shift; 3.3-4.7 SEM, detectable at A15 power).
Raw linear: -0.016 to -0.033 (Z -1.3 to -2.7). This arm tests the theta-inconsistency of the +/- 1 sigma_g term in
isolation; it is not expected to pass B0-A and is REPORTED-ONLY.

**P3 — the E12 reproduction (diagnostic fallback, path choice): as P1 with sky_cone_k = 3.0.** Registered only as the
arm to run if P1 FAILS its band: it isolates the sky-intrinsic V3 term (section 2.5 predicts no s-move from it:
|delta score_s| <= 0.02).

---

## 6. Cost (A11: measured anchors; bands where derived)

Anchors {value, source, date}: T1.2 cells at 14 cpu, jobs 1: off-truth 688.1-708.6 s (16 cells, mean 698 s), truth
61.2-63.0 s (4 cells) {s0a_full_output.json per_seed_summary, 2026-08-30}; wall 13084.8 s / 20 cells = **654 s per
cell**, of which venue builds ~1670 s per pass (~417 s per seed) {13084.8 - 16 x 698 - 4 x 62}; the divisor pass
therefore ~636 s per off-truth cell (698 - 62) and evaluate() ~62 s for 106-130 events at 278 median candidates
{row #266: 701.65 s per off-truth cell vs 169.5 s in the no-divisor pass}. E9's enlarged ball (sky 3.0 AND z 4 sigma_g):
median candidates 278 -> 1729 (6.2x) {forensic E9}. This node: k = 4 at sky 1.5 grows the z-extent by 2.22x (linear)
and the count by 2.5-3.6x (z^2-weighted IQR, median 2.95) {t13_out.json count_growth_k4_sky15, 2026-08-30}.

1. **P1 (12 cells: 4 truth + 8 s-nodes at k = 4, sky 1.5):** evaluate 62 s -> 160-225 s per cell (x2.5-3.6);
   s-node cell = 636 + 160..225 = **800-860 s**; truth cell 160-225 s (no divisor pass at the identity). Total
   8 x 830 + 4 x 190 + 1670 (venue) = **~9,000 s ~ 2.5 h wall at 14 cores (~35 CPU-h nominal)**, local.
   The divisor pass does not depend on the window, so P1's rho tables are bit-identical to T1.2's; caching them
   (keyed by seed, h, theta, catalogue md5, table hash) would cut P1 to ~1.2 h but needs a bit-identity proof
   (T1.1 R8 pattern) — optional, not assumed.
2. **P2 (8 s-cells at k = 1):** candidate counts within ~x1.5 of today's (the set changes membership at the edges);
   8 x ~720 s + 1670 = **~7,400 s ~ 2.0 h wall**. P1 + P2: ~4.5 h.
3. **P1b (8 b-cells at k = 4):** 8 x 830 = **~1.8 h** (venue builds shared if run in the same pass as P1).
4. **P3 (8 s-cells at k = 4, sky 3.0):** counts ~6x (E9): evaluate ~370 s, cell ~1,000 s; 8 cells **~2.3 h**.
5. **S0-B (C1, production venue, 1588 events, iiib):** the production h_bounds (0.6, 0.86) make the envelope NARROWER
   than the mirror's (0.50, 0.86), so the fixed-window capture term is LARGER there (the zeta/s term grows as the
   envelope shrinks); mechanism (ii) is inherited by S0-B in a stronger form, as (i) was (forensic section 7). Any S0-B
   launch must carry the flag at k = 4 and re-cost the per-event term by the count growth (~x2.5-3.6 of the
   band in PA-HIER-31(i)). Not licensed here.
6. **Default (flag "off", k = 1.0): zero cost, byte-identical.**

---

## 7. Regression plan (tests written BEFORE the change where they pin existing values; the B5.1 / T1.1 pattern)

R1  Byte-identity at "off" (unit, handler): get_possible_hosts_from_ball_tree on a 2,000-row fixture catalogue for 50
    synthetic detections, before vs after the change, default arguments: identical index lists (values AND order) for
    both returned channels (the B5.1 100,000-pair pattern: 0 mismatches). Plus a production pin: the candidate_dump
    of one T1.2 truth event (T2.2 hook) re-run at "off" — identical rows.
R2  Literal skip (unit): "on" at theta = (0,1), k = 1.0: identical lists to R1; hook counter site_2_2_window unchanged;
    forced-path variant (the skip removed by monkeypatch): still identical bits.
R3  Engagement and monotonicity (unit): "on" at (0, sqrt2), k = 1: the accepted set is a superset of the (0,1) set;
    at (0, 1/sqrt2) a subset; at (+/-0.02, 1) edge rows swap (count the symmetric difference > 0); s -> 1e-6 gives the
    5.2 set; s = 1e3 gives the whole cone; all sets nested as 5.3 states.
R4  k-consistency at the identity (unit + production): at theta = (0,1) "on" and "off" give identical sets for every
    k in {1, 2, 4}; the k = 4 set contains the k = 1 set; on the real catalogue (one T1.2 event via the T2.2 dump) the
    k = 4 / k = 1 count ratio lies in [1.5, 6].
R5  Guards (unit): bad token raises; z_window_k in {0, -1, inf, nan} raises; theta validation as today; the handler
    rejects theta kwargs when the flag is "off" (they must not be passed).
R6  Mass-filter inheritance (unit): the with-BH set at k = 4 equals the mass filter applied to the k = 4 no-BH set
    (the mask order is unchanged).
R7  Plumbing: --theta_zwindow / --z_window_k on arguments.py / main.py / run_mirror_seed_inprocess with byte-identical
    defaults; run_metadata captures them; hier_s0_driver.py gains both flags, threads them to its three call sites and
    emits the "_zwin<k>" node-dir suffix (a smoke cell must show the suffix and a candidate-count change before P1).
R8  C0-style pin (the identity node): the P1 truth node at k = 1.0 (a 62 s cell) diffs against T1.2's truth CSV at
    max_abs 0.0 on every numeric column; at k = 4 the intersection candidates' per-candidate values (T2.2 dump) are
    bit-identical and only added rows differ.
R9  Suite: pytest -m "not gpu and not slow" baseline 1915 passed / 15 skipped / 27 deselected {T1_1 verifier report,
    2026-08-30} + the new tests; ruff / mypy clean; test_theta_phi_divisor.py (19), test_theta_hook.py (15),
    test_mass_filter_sigma.py / test_mass_filter_geometry.py untouched and green.
Reference comment above the changed mask: "Sec. 2 in Ma, Hu & Huterer (2006), arXiv:astro-ph/0506614; Eqs. (5)-(7)
in Mandel, Farr & Gair (2019), arXiv:1809.02063; PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md section 2 (row #255, T1.3)".
Commit prefix [PHYSICS]; ledger rows implemented / verified appended by the builder and by a different verifier.

---

## 8. A10 — invariants (with last-audited dates) and structural blindness

**Invariants held fixed by this change:**
1. GW-side envelope: 3 sigma_dL at (h_min, h_max) (physical_relations.py:563-566; dead sigma_multiplier disclosed), the
   1.55 clamp (:5634); mirror H_BOUNDS (0.50, 0.86) (prereg 5.1 invariant 2, 2026-08-25); production (0.6, 0.86).
2. Sky cone 1.5 sigma_max (handler.py:662; sky_cone_k default, T1.1 2.5, 2026-08-30) — no theta (2.5).
3. Mass filter "symmetric", "linear", k = 1.5 (rows #198-#202, #220-#223) — form unchanged; input set inherits.
4. Site-2.2 kernel: 4 sigma window, 1e-6 floor, GL-50, C7-core (T1.1 invariant 3; K1-K4 never re-audited, carried).
5. theta identity (HIER 1.2): s on the raw error; PV from the unshifted z_g; SIGMA_V_PEC_KM_S = 0.0 (constants.py:95,
   2026-08-26) — the same placement in the window as at site 2.2 (2.1).
6. The T1.1 divisor form (site 2.3phi; D2 eligibility on listed z_g) — unchanged (2.4).
7. Path-A objects, Sigma_4D, the with-BH channel's divisor — untouched; the C-C identity (dark class exactly 0.0).
8. Generator frozen (GATE GEN-FROZEN, PA-HIER-2); banked comparands = the T1.2 CSVs (R8).
9. Reduced catalogue md5 c52c13b5cab61f6b3f04bbe202550969 (verified by T1.1 2026-08-30; NOT re-verified this node —
   this node read only banked CSVs; the builder's smoke cell must re-verify, A11 dataset pin).
10. Literal-skip identity at theta = (0,1) at every site (GATE T-ID).

**Structural blindness (what this design cannot detect by construction):**
(a) the non-capture terms — V2 (mixture-weight covariance, the c > 0.84 class), V4 (Fisher-quality selection on
d_L), the sky-intrinsic V3 term — remain in P1 and set its floor (E12's -0.031 +/- 0.011 debiased); a P1 landing at
Z_s ~ -2 to -2.6 is the PREDICTED outcome, not a marginal pass; (b) the capture model's approximations (Gaussian
kernel with the E15 tilt only to first order; S_bar_phi weighting inside c_bar_g ignored; the 1e-6 floor ignored;
the b-axis registered as model-unreliable, section 3); (c) the E13 debiasing convention (raw vs c-weighted, 2.1 SEM
apart) — pre-registered above, but its correctness for the combined channel is an inference, not a measurement; a
harness null-venue check of the combined-channel secant expectation is the clean test (not asked here); (d) the
production h-profile effect of k = 4 at theta = (0,1) — the P1 truth cells give a single-h read only (a production
default flip needs the h-profile; decision-table item 6); (e) shared misspecification between generator and estimator
(prereg 5.2 item 3) — the certification is of theta-consistency, never of the survival model; (f) theta's 2-D span
(5.2 item 2); single h; mirror venue N ~ 461 (5.2 item 6); (g) the with-BH channel (uninterpretable on b0i, T1.1 8(a));
(h) the sky/d_L Fisher correlation neglected by the ball's marginal blocks (2.5).

---

## 9. A14 — falsifiers, registered before any code

F1 (the s-axis attribution; P1): P1 must return **|Z_s| <= 3** in PA-HIER-32's debiased statistic (c-weighted
   convention, fixed before unblinding; the raw convention reported alongside). |Z_s| > 3 REFUTES the attribution of the
   s-axis residual to the candidate-window truncation: at k = 4 with the transform the capture term is removed BY
   CONSTRUCTION (section 3 table), so a surviving residual is not a selection-window effect. The change then stays in
   the tree as a structural-consistency instrument and the STOP returns as INSTRUMENT-DEFECT (s) UNRESOLVED with the
   next candidates named here: (n1) the S_bar_phi table's own sigma_z dependence — the survival S_bar_phi(z; h) is a
   POINT function of z built from S_4D with no photo-z smearing (T1.1 1a), while the generator's draw weights and the
   theta-divisor use the kernel-smeared S_tilde_g; the T1.1 kappa(h) = Sigma_phi_smear/Sigma_phi_point diagnostic
   (logged for free by the divisor pass) is the zero-compute first read, and a smeared-table arm the compute test;
   (n2) the V2 mixture-weight covariance — read at zero compute by c-stratifying the P1 CSVs: if the c > 0.84 class
   still carries a Z < -3 residual while the hw_sig-q1 class has moved to null, the residual is the mixture weight's,
   not the window's. Sharper (implementation-level, before the band): the hw_sig-q1 class must move by >= +0.15
   (from -0.290) and the n_cand median must grow by x2.2-3.6; either failing => the flag is not doing what 2.1 says
   (STOP, implementation defect, regardless of the pooled band).
F2 (the transform's own contribution; P2, if run): P2 must remove >= 25 % of the T1.2 debiased residual (score_s
   >= -0.075 c-weighted); < 25 % (or a move of the wrong sign) refutes the theta-inconsistency of the +/- 1 sigma_g term
   as a contributor — the truncation would then be purely the envelope term, which only k cures. The registered
   expectation is 55 % (both variants); > 85 % would itself be a SURPRISE (the envelope term smaller than derived)
   and must be reported as such.
F3 (identity and inheritance): R1/R2/R8 pins fail -> INSTRUMENT-DEFECT; R4's count ratio outside [1.5, 6] -> the
   window is not the 2.1 object.
F4 (C-C identity): the 5 pooled dark events score exactly 0.0 on every node; any deviation -> INSTRUMENT-DEFECT
   (a global object moved).
F5 (b-axis, only if P1b runs): |Z_b| > 3 at k = 4 -> a NEW finding on the b-axis (the fixed window's positive
   capture term unmasking a negative residual), filed as a fresh diagnosis; it does not touch F1.
Rule-1 exoneration check (mechanism, not tag): EXONERATION_REGISTER_20260827.md grepped for "candidate window",
"z-window", "ball truncation", "sky cone", "redshift_filter", "get_possible_hosts", "sigma_multiplier": no entry
exonerates the candidate-ball truncation as a mechanism (line 145 lists "z-window unification / host-z clip" only as
search aliases of a different, volume-truncation entry; line 591 concerns the mass filter emptying a ball). The
ledger's DO-NOT-RE-TRY list (BIAS_HISTORY_LEDGER.md section 2) has no window/ball item; row #183 PINNED the
candidate-ball z-window's h-bound dependence as a real with-BH effect — the same envelope object as 2.3, consistent
with, not against, this node. No collision.

---

## 10. Decision table (approval-scope tags; the standing grant is the approval)

| # | tag | item | disposition under row #255 |
|---|---|---|---|
| 1 | [DO] | theta_zwindow flag (default "off") + z_window_k knob (default 1.0), the 2.1 mask, guards, counter, R1-R6, R9 | COVERED (A17 standing grant: instrument, gate before code; resolves T1.1 decision-table item 3 per row #266's path decision); builder != presenter; [PHYSICS] commit; ledger rows implemented / verified |
| 2 | [DO] | Plumbing R7 incl. hier_s0_driver.py flags + "_zwin<k>" suffix (non-physics files) | COVERED; the smoke cell must show the suffix and a count change BEFORE P1 |
| 3 | [DO] | P1 (12 cells, ~2.5 h wall, local): theta_zwindow on, k = 4.0, sky 1.5, divisor on, sites 2.2, smear off, h = 0.73, 4 seeds, nodes {truth, s_plus, s_minus}; runner != builder; scored in PA-HIER-32's statistic with the convention fixed before unblinding | COVERED (instrument certification, tree 2 T1 depth 3) |
| 4 | path choice | P2 (8 cells, ~2.0 h) and/or P1b (8 b-cells, ~1.8 h); P3 only if P1 fails | orchestrator's call under row #255; predictions registered for each |
| 5 | fresh [RULE] later | The P1 READ (band comparison on data that do not yet exist) | returns to the author per the charter; nothing here pre-decides it |
| 6 | fresh [RULE] later | Production default z_window_k 1.0 -> 4.0 (a production candidate-set change; h-profile effect UNMEASURED; count x2.5-3.6 per event) | inputs do not exist yet (needs an h-profile read, not the single-h P1 truth cells); not asked here |
| 7 | NOT DONE, disclosed | get_redshift_outer_bounds's dead sigma_multiplier (physical_relations.py:555, :563/:566; call-site 2.0 inert) | code-health item; changing it would move every candidate set (a physics change of its own); left byte-identical |
| 8 | NOT DONE, disclosed | The site-2.2 Z_g <= 0 guard on degenerate windows (T1.1 item 6) | unchanged; the 2.6 rule needs no special case |
| 9 | NOT DONE, disclosed | The E13 debiasing convention for the combined channel (raw vs c-weighted) | pre-registered as c-weighted primary; the reconciliation row #266 asks for remains the author's/orchestrator's |

What this presentation does NOT license: any S0-B (C1) launch (row #266: "S0-B stays unlaunched"); any Stage P/F
costing or launch (A15 MOOT); any change to the GW-side envelope, the sky cone's form, Sigma_4D, path-A objects, the
with-BH channel, the divisor, or the generator; any code under this node.

---

## 11. Provenance

Read in full: PHYSICS_CHANGE_THETA_DIVISOR_20260830.md (sections 0-12 + Revision note 1); T1_1_DIVISOR_VERIFIER_REPORT.md;
B1_1_S0A_DEFECT_FORENSIC_20260829.md sections 0-3, 5-8 (E6, E9, E12, E13, E14, E15, E20 quoted); PREREGISTRATION_HIER_
HTHETA_20260826.md sections 1.2, 2.1, 4.1, 5.1-5.2, PA-HIER-31 (outline), PA-HIER-32 (full); TREE2_CHARTER_20260830.md
(T1 rows); tree2 README.md; BIAS_HISTORY_LEDGER.md rows #183, #251, #266 (grep); EXONERATION_REGISTER_20260827.md
(grep); docs/gates/PHYSICS-GATE-LEDGER.md (header + 2026-08-30 rows); T2_2_CANDIDATE_HOOK_RECORD.md section 1.
Code read (no edits): physical_relations.py:546-567; handler.py:440-770 (the reduce PV fold :461-479, ball tree
:540-556, get_possible_hosts_from_ball_tree :558-770); bayesian_statistics.py :1625-1643 (counters, _validate_theta),
:3206-3300 (divisor eligibility), :3659/:3753/:3945/:3967/:4228 (flag defaults), :5620-5650 (call site), :7321-7427
(scalar site 2.1), :8101-8144 (site 2.2), :7975-8030 (batch docstring); correspondence_1d.py :2734-2998 (grep);
hier_s0_driver.py :79-84, :141-147, :282-336, :345-414, :494-599 (grep); arguments.py / main.py (grep);
cosmological_model.py:388-397; constants.py:25, :36, :95, :102.
Instruments of this node (foreground, 2 x ~40 s, zero evaluate() calls, no source edits):
tree2_20260830/t1_3_gate_work/t13_residual_and_capture.py -> t13_out.json (pooled_recert, per_seed, by_class,
by_*_quartile, corr, capture_model, capture_model_tilted, prediction, prediction_tilted, *_by_hw_sig_quartile,
E9_b_check, count_growth_k4_sky15). Inputs: the T1.2 recert CSVs (read only) and f7_events.csv (read only).
No git operations by this node; the orchestrator commits. Append-only. Ledger row appended to
docs/gates/PHYSICS-GATE-LEDGER.md ("presented", pre-commit, this date).

---

## Revision note 1 (2026-08-30; panel must_fix; append-only)

**Trigger.** A refuter panel on this document (refuted=false; two items, both must_fix, both severity
non-major — neither changes a registered number, band, prediction, or plan item) flagged (1) an
internal contradiction in section 2.2 about which file holds the single read/validate/raise site for
the new `z_window_k` knob, and (2) a stale line-citation in section 1(c) (`:8107-8109`, "a 2-line
reference-comment offset"). Both are addressed below by SUPERSEDED markers on the affected passages;
no text above this note is edited in place (append-only discipline, the B5.1/T1.1 pattern).

**Item 1 — the read/validate-site contradiction. RESOLVED, panel item confirmed correct.**

Section 2.2, as originally written, states two things that cannot both be true of a single site:

    "z_window_k: float = 1.0 ... single read/validate site in handler.py"                    (flag-declaration line)
    "... the flag and k are read/validated at that one site" [meaning the bayesian_statistics.py
    call site, :5636-5650, the sentence immediately preceding it] "(the mass_filter_k precedent)."

The panel is correct that these disagree, and correct that the `mass_filter_k` precedent, checked
against the working tree, settles it in favour of **handler.py**, not the call site:

    darksiren_emri/galaxy_catalogue/handler.py:679-694 (inside get_possible_hosts_from_ball_tree,
    re-verified fresh this note):

        # Single read/validate site for mass_filter_geometry/mass_filter_k
        # (charter node B5.1, ...)
        if mass_filter_geometry not in ("linear", "log"):
            raise ValueError(...)
        _mass_filter_k = float(mass_filter_k)
        if not np.isfinite(_mass_filter_k):
            raise ValueError(f"mass_filter_k must be finite, got {mass_filter_k!r}")

    darksiren_emri/bayesian_inference/bayesian_statistics.py:4225, :5649 (re-verified fresh this note):

        self._mass_filter_k = float(mass_filter_k)                       # :4225, a CAST, not a guard — no raise
        ...
        mass_filter_k=self._mass_filter_k,                                # :5649, plain pass-through at the call site

i.e. the actual precedent is: `bayesian_statistics.py` stores the value (a `float()` cast, not a
validating guard) and passes it through positionally at the single call site; the one `raise` site —
the thing "single read/validate site" means — lives entirely inside `handler.py`. There is no guard on
`mass_filter_k`/`mass_filter_geometry` anywhere in `bayesian_statistics.py`.

**Corrected registered plumbing (supersedes section 2.2's second bullet-paragraph, "'on' and theta !=
(0, 1): ... keeps no theta state; the flag and k are read/validated at that one site (the
mass_filter_k precedent)." — that sentence's file attribution is corrected here; nothing else in
section 2.2 changes):**

- `z_window_k`'s guard (`not np.isfinite(...)` or `<= 0` -> `ValueError`) is added inside
  `get_possible_hosts_from_ball_tree` in **handler.py**, immediately alongside the existing
  `mass_filter_k` guard (:692-694) — same function, same pattern, same exception type. This is the
  single raise site for `z_window_k`; `bayesian_statistics.py` only stores `self._z_window_k =
  float(z_window_k)` (a cast, mirroring `self._mass_filter_k` at :4225) and passes it through
  positionally at the one call site (:5636-5650), exactly as `mass_filter_k` does today.
- `theta_zwindow`'s token guard (`{"off","on"}` or raise) and the `theta_b`/`theta_s` validity guard
  (`_validate_theta`, :1640-1643) stay where every other θ-hook site puts them: on the
  `BayesianStatistics` side, at or immediately before the call site (:5636-5650) — this part of the
  original sentence was correct and is unchanged. `handler.py` remains a non-physics file that holds
  no θ state, per section 2.2's own (unaffected) framing; it now additionally holds exactly one
  numeric guard for `z_window_k`, matching what it already does for `mass_filter_k`.
- **Net: exactly one raise site per guard, in the file the precedent actually uses.** R5's guard test
  ("bad token raises; z_window_k in {0, -1, inf, nan} raises; ... the handler rejects theta kwargs when
  the flag is 'off'") targets **handler.py**'s `get_possible_hosts_from_ball_tree` for the `z_window_k`
  half of that check (a unit test constructing the handler directly, the same fixture pattern R1 uses)
  and the `bayesian_statistics.py` call site / `_validate_theta` for the token and θ halves. No
  duplicate guard is registered at either site.

Scope: this corrects only the file-attribution clause of section 2.2's plumbing paragraph. The flag
semantics (2.1), the switch's off/skip/on behaviour, the counter, the driver threading, and every
downstream section (3-10) are unaffected — none of them depended on which file raises the ValueError.

**Item 2 — the section 1(c) line citation. Panel's diagnosis CONFIRMED; panel's own suggested
replacement (`:8105-8107`) is ITSELF off by 2 lines — corrected here from a fresh re-grep, with
evidence, rather than applied as proposed.**

Re-grepped `darksiren_emri/bayesian_inference/bayesian_statistics.py` fresh this note (`cat -n`, working
tree, 2026-08-30; `grep -n "sigma_z_pv = (1.0 + host_z)"` returns exactly two hits, :7393 [scalar] and
:8103 [batch] — confirming which occurrence section 1(c) means). The batch site's true lines:

    :8099    integration_limit_sigma_multiplier = 4.0
    :8103    sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
    :8104    host_z_error_eff = np.sqrt(host_z_error**2 + sigma_z_pv**2)
    :8105    if theta_b != 0.0 or theta_s != 1.0:
    :8113        _theta_hook_count("site_2_2")
    :8114        host_z_error_eff = np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2)
    :8115        host_z = host_z + theta_b * (1.0 + host_z)
    :8124    den_hi = host_z + integration_limit_sigma_multiplier * host_z_error_eff
    :8141-8143    den_lo = np.maximum(
                      host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor
                  )

so the panel's diagnosis (the `:8107-8109` triple is stale) is CONFIRMED, but its proposed fix
(`:8105-8107`) reproduces only the `if` line correctly (:8105) and mislabels the two comment lines that
follow it (:8106-8107 are `# [HIER] θ-hook site 2.2 ...` / `# arXiv:astro-ph/0506614 ...`, not
`sigma_z_pv`/`host_z_error_eff`) — the panel's own correction has a residual 2-line error in the
opposite direction for that sub-span, evidenced by the `cat -n` excerpt above. The true offset for this
quoted triple is 4 lines (`:8107`->`:8103`, `:8108`->`:8104`, `:8109`->`:8105`), not the panel's implied
2; the offset for the surrounding lines of the same excerpt (`:8101`->`:8099`, `:8115-8117`->
`:8113-8115`, `:8126`->`:8124`, `:8143-8144`->`:8141-8143`) IS 2, consistent with two comment lines
(:8101-8102, "# Residual peculiar-velocity dispersion...") having been inserted between the
`integration_limit_sigma_multiplier` line and the `sigma_z_pv` line since this document's citations were
last written — the panel spotted the right symptom (staleness) at the right sub-span but read off the
wrong replacement numbers for it.

**Corrected citation (supersedes the entire quoted code block of section 1(c), the nine `:XXXX` lines
between "integration_limit_sigma_multiplier = 4.0" and "den_lo = ..."; the prose above and below the
block, and the scalar-twin parenthetical that follows it, are handled next):**

    :8099    integration_limit_sigma_multiplier = 4.0
    :8103    sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S            # SIGMA_V_PEC_KM_S = 0.0 (constants.py:95)
    :8104    host_z_error_eff = np.sqrt(host_z_error**2 + sigma_z_pv**2)
    :8105    if theta_b != 0.0 or theta_s != 1.0:
    :8113        _theta_hook_count("site_2_2")
    :8114        host_z_error_eff = np.sqrt((theta_s * host_z_error) ** 2 + sigma_z_pv**2)
    :8115        host_z = host_z + theta_b * (1.0 + host_z)
    :8124    den_hi = host_z + integration_limit_sigma_multiplier * host_z_error_eff
    :8141-8143    den_lo = np.maximum(host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor)   # 1e-6

The immediately following parenthetical, "(scalar twin single_host_likelihood: :7321, :7407-7408,
:7418, :7426-7427.)", was checked against the same fresh re-grep and found stale by a larger, non-
uniform margin (the scalar function `single_host_likelihood` carries its own, differently-sized comment
block between the `if` line and the θ-hook body) — corrected here on the same evidentiary basis:

    :7319         integration_limit_sigma_multiplier = 4.0
    :7403-7404    _validate_theta(theta_b, theta_s); _theta_hook_count("site_2_1")
    :7406         host_z = host_z + theta_b * (1.0 + host_z)
    :7424-7426    denominator_integration_lower_redshift_limit = max(host_z - integration_limit_sigma_multiplier * host_z_error_eff, _z_lower_floor)

**Everything else in section 1(c) — the prose readings ("the estimator integrates each selected
candidate's kernel over W_g^theta = ...", "two different objects even at theta = (0,1) (k = 1 vs 4)",
the attribution to V3/E6/E14) and every citation OUTSIDE this one code excerpt (section 1(a)'s
`physical_relations.py:546-567`/`:563-566`, `constants.py:102`, `bayesian_statistics.py:5624-5632`,
`:5634`; section 1(b)'s `handler.py:558-770`, `:662`, `:664`, `:668-676`/`:677`, `:461-479`, `:688-753`;
section 1(d)'s `bayesian_statistics.py:3206-...`; section 2.2's `:1625-1633`, `:1640-1643`, `:5636-
5650`; section 8's line citations) — were independently re-grepped fresh this note (`cat -n`, working
tree, 2026-08-30) and reproduce byte-for-byte, confirming the refuter panel's own finding that these
were already exact. No other line number in the document is corrected by this note.

**Scope — what is, and is not, affected.**

- SUPERSEDED (marked here, not edited in place): section 2.2's second bullet-paragraph's file
  attribution for the `z_window_k` guard (handler.py, not the call site); section 1(c)'s quoted code
  block's nine line numbers and the scalar-twin parenthetical's four line numbers.
- UNAFFECTED: sections 0, 2.1, 2.3-2.6, 3-10 in full, including the 5-item form's substance (2.1's
  mask, 2.3's capture-fraction derivation, 2.4's divisor-independence argument, 2.5's sky-cone
  argument, 2.6's edge case, the section-3 derivation and capture-model table, section 4's dimensional
  analysis, section 5's limiting cases and registered predictions P1/P2/P3, section 6's cost, section
  7's regression plan R1-R9 other than R5's guard-site target, section 8's invariants, section 9's
  falsifiers, section 10's decision table). No registered number, band, prediction, falsifier, or
  decision-table disposition changes. R5's text is read with the item-1 correction applied (guard-site
  = handler.py for `z_window_k`); no other regression item is affected.
- The T1.2 outputs, the t13 instrument, and its outputs are untouched (this note wrote no new
  computation; only two verification re-greps: `mass_filter_k`'s actual guard location, and the working-
  tree line numbers of sites 2.1/2.2).

**Net effect.** Corrects (i) a file-attribution error in the plumbing paragraph that could have led a
builder to place `z_window_k`'s guard in the wrong file or duplicate it, per the panel's own concern,
and (ii) a stale line-citation whose panel-suggested replacement was itself imprecise, replaced here
with numbers re-verified directly against the working tree. Neither correction touches this node's
mechanism, derivation, predictions, cost, regression plan, invariants, falsifiers, or decision table.

Verified against `darksiren_emri/bayesian_inference/bayesian_statistics.py` and
`darksiren_emri/galaxy_catalogue/handler.py` in the working tree at HEAD `7b1bd9dc` (git rev-parse
confirmed 2026-08-30; `bayesian_statistics.py` carries uncommitted diffs per this document's header,
`handler.py` does not). Presenter for this note: top-tier subagent, per the branch's standing grant
(row #255) covering production/documentation changes within the tree (author-verbatim, row #223). No
code written; no git operations; foreground only.

---

## Revision note 2 (2026-08-30; panel must_fix; append-only)

**Trigger.** A second refuter panel on this document (refuted=false; three items, all must_fix,
none changing any registered number, band, prediction, falsifier, or plan item) flagged (1) an
undisclosed tension between this document's own capture-model finding and a standing in-code
comment sitting inside the exact span this document directs a reader to open; (2) a node-label
collision with TREE2_CHARTER_20260830.md's own table; (3) F1's falsifier band being wired to a
convention (c-weighted primary) that row #266 flagged as an open author/orchestrator
reconciliation item, not a settled one. All three are addressed below by SUPERSEDED markers /
disclosure additions; no text above this note (including Revision note 1) is edited in place
(append-only discipline, the B5.1/T1.1 pattern).

**Item 1 — the :7391-7392 "second-order" comment. CONFIRMED tension; RECONCILED (narrowed
reading), not superseded as a measurement, because it is not one.**

Re-verified fresh this note (cat -n, working tree, 2026-08-30): the comment reads, verbatim,
at bayesian_statistics.py:7391-7392 (inside the site-2.1 scalar function, the exact :7319-7427
span section 1(c) and its Revision-note-1 correction both direct the reader to open; it does
**not** recur at the batch site 2.2, :8099-8144 — confirmed by a whole-file grep, one hit):

    # window and catalogue pruning (handler.py) intentionally keep the bare
    # catalogue z_error — a ±1σ, second-order candidate-list effect.

git log -S "second-order candidate-list effect" traces this line to commit 8568d9fc
("[PHYSICS] marginalize residual host peculiar velocity into the host-z kernel (issue #16)",
2026-07-03) — **seven weeks before theta existed as a construct** (theta_b/theta_s are
introduced by the HIER program starting PREREGISTRATION_HIER_HTHETA_20260826.md, 2026-08-26,
and built at commit d40fe5c8, 2026-08-28, per project memory). Reading that commit's own diff
and message: "bare catalogue z_error" there means the ball-tree window's z_error **without the
newly-added residual-peculiar-velocity fold** sigma_z_pv — a term that is IDENTICALLY ZERO
today (SIGMA_V_PEC_KM_S = 0.0, constants.py:95, re-confirmed this note) — not "without the
theta_b/theta_s transform," a concept that commit could not reference. A repo-wide grep for
"second-order candidate-list effect" and "second-order.*candidate" outside this one file
(this note, plus the search already run for Revision note 1) finds no other hit that discusses
the ball-tree window; the one unrelated match (SIZING_ANALYSIS.md:162) concerns KDE boundary
weighting. **So the comment is an unverified, scope-narrow design-time assertion about a
different (and currently null) term — not a measurement this document's capture-model finding
supersedes, and not, read in its original context, literally false.**

The panel is nonetheless correct that a plain reading of the comment, taken at face value in
2026-08-30 where theta now exists, directly conflicts with this document's own section-3 finding:
the same "bare catalogue z_error" choice this comment calls "second-order" is, under theta != (0,1),
the DOMINANT term — the theta-consistent transform at k=4 removes 99.9%/100% of the model s-term
(section 3 table, bare/tilted), and the bare model alone reproduces 105% of the measured T1.2
debiased residual (-0.105 model vs -0.0997 +/- 0.0123 measured, section 3 "Calibration against
measurement" (i)) — i.e. first-order, not second-order, once theta departs from the identity the
2026-07-03 commit was reasoning about. Leaving the comment unqualified in the exact span this
document's own Provenance section (11) and section 1(c) tell the reader (and the builder) to open
is a genuine, if narrow, internal-consistency defect: a careless reader could conclude the code
already dismisses the mechanism this node registers.

**Disclosure (added here; changes no registered number):** the comment's claim and this
document's finding are about two different terms (the zeroed PV fold vs. the theta-s/b window
transform) and are therefore not in logical contradiction as originally written — but the
comment's wording no longer accurately scopes itself now that theta exists, and should not be
read as covering the theta-consistency question. **Builder instruction (documentation-only, not a
registered number; filed as regression-plan follow-up R7c):** when implementing this node's
registered change, amend the comment at bayesian_statistics.py:7391-7392 to read, in substance,
"the ball-tree window keeps the bare catalogue z_error w.r.t. the (currently zero) peculiar-
velocity fold — second-order at theta=(0,1); see PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md section 3
for the theta-s/b window-vs-kernel term, which is NOT second-order under theta != (0,1)." This is a
comment-text change riding alongside the registered code change at the same site, not a new
formula, and carries no new dimensional analysis or limiting case.

**Item 2 — the "T1.3" label collision with TREE2_CHARTER_20260830.md. CONFIRMED; node relabeled,
effective this note.**

Re-verified fresh this note: TREE2_CHARTER_20260830.md's own branch table (section 2, T1 row,
depth-3 column) and its local-vs-cluster split both use "T1.3" for the S0-B/C1 production-venue
launch, gated on "T1.1 and a passing T1.2" — quoted verbatim: "A6 word (already ruled
launch-after-fix, row #255): S0-B (C1) on iiib launches only after T1.1 and a passing T1.2" and
"T1.3 (S0-B/C1), T3.5 execution if it needs cluster scale, T4 (all depths), and T5.2's joint_r1
arm queue behind cluster recovery." T1.2 in fact FAILED its s-axis band (ledger row #266), so that
node is correctly unlaunched — consistent with, and not contradicted by, this document's own
"S0-B stays unlaunched" quote. This document's header, approval stamp, and decision table reused
the SAME label "T1.3" for an unrelated node (the z-window gate); that reuse originates from the
orchestrator's own row #266 path-naming ("T1.3 = the z-window/cone companion knob as its own
gate"), not from an invention by this presenter, and was disclosed consistently throughout the
document — but the charter itself was never amended to reflect it, so a future reader or agent
grepping "tree 2 T1.3" cannot tell which of the two unrelated nodes is meant.

**Resolution (relabel, per the panel's own "alternatively" option; no edit to
TREE2_CHARTER_20260830.md or BIAS_HISTORY_LEDGER.md by this note — both are outside this
presentation's file scope, the ledger explicitly so as a file two other readers are concurrently
writing).** Effective from this note forward, this node's label is **"T1.3-zwin"**, not "T1.3".
The body above this note (including Revision note 1) is left untouched under append-only
discipline and keeps its original "T1.3" text as the historical record of what was written before
the collision was caught; every reference this note or any later record makes to this node uses
"T1.3-zwin". The gate-ledger row this note appends (below) uses the corrected label. Any future
code comment, commit message, or driver node-dir suffix for this node's registered change should
cite "T1.3-zwin", not "T1.3" — a correction the builder should apply at the reference-comment site
(section 7's "row #255, T1.3" citation string becomes "row #255, T1.3-zwin"). **Recommended, not
performed here:** the orchestrator append a one-line disambiguating note to
TREE2_CHARTER_20260830.md recording that "T1.3" in that document's own table refers only to the
S0-B/C1 launch, once the file is not concurrently being written.

**Item 3 — F1's convention (c-weighted primary) is a PROPOSAL, not a ruled convention. CONFIRMED;
F1's operative status corrected.**

Re-verified fresh this note against BIAS_HISTORY_LEDGER.md row #266 (read-only, quoted
verbatim): "the two registered texts disagree on T1.2's scope... disclosed, not adjudicated, a
fresh reconciliation item for the author/orchestrator." Section 5.6 and decision-table item 9
already disclose the raw-vs-c-weighted disagreement and note the final reconciliation "remains the
author's/orchestrator's" — but section 9's falsifier F1 and section 5.6's registered point/band
both name PA-HIER-32's debiased (c-weighted) statistic as **PRIMARY** for the pass/fail read that
will decide STOP vs. CONFIRMED on the whole s-axis attribution. Per CLAUDE.md's approval-scope
rule, a convention choice that a decisive falsifier's verdict will be read against is exactly a
"branch call, verdict or band comparison" that "returns to the author as a fresh [RULE]" once it
is about to decide an outcome — and per the binding default, this presentation's own
derivation-based preference (section 3, "d ln combined_no_bh = c_i d ln L_cat to first order")
cannot itself make that ruling stick as the convention the eventual verdict is read against,
however well-derived. No author or orchestrator [RULE] adopting the c-weighted convention has been
issued as of this note (re-checked against the same ledger read).

**Resolution (process constraint; no registered number, band, or prediction changes — section
5.6's two convention-numbers stand exactly as written, both already computed there).** Section
5.6's "PA-HIER-32's debiased statistic is PRIMARY" clause is downgraded from a settled convention
to this presentation's **proposed** convention, pending an explicit author/orchestrator [RULE].
Until that [RULE] is issued: (i) whoever reads P1's outcome MUST report both conventions side by
side, exactly as section 5.6 already computes them (c-weighted score_s = -0.026 +/- 0.012 vs. raw
score_s in [0.000, +0.031]); (ii) the reader MUST NOT declare F1 CONFIRMED or REFUTED, or the
s-axis STOP lifted or upheld, on the strength of one convention alone — making that declaration
IS the fresh [RULE] this document cannot pre-empt, and it must be flagged to the author/
orchestrator the way row #266 flagged the T1.2 scope disagreement, not silently resolved by
citing this section. This changes who may say a P1 run "passed" or "failed" F1 and on what
authority; it changes no arithmetic, band, or falsifier threshold in sections 5.6 or 9.

**Scope — what is, and is not, affected.**

- SUPERSEDED / narrowed (marked here, not edited in place): section 5.6's "PA-HIER-32's debiased
  statistic is PRIMARY" clause is downgraded to PROPOSED pending [RULE] (item 3); the operative
  reading of the bayesian_statistics.py:7391-7392 comment is narrowed to the PV-fold question only,
  with a builder instruction to update its wording (item 1); this node's label is corrected from
  "T1.3" to "T1.3-zwin" for every reference from this note onward (item 2).
- UNAFFECTED: every registered number, band, point prediction, cost figure, regression item, A10
  invariant, and falsifier threshold in sections 1-10 and in Revision note 1; the 5-item
  physics-change form's substance; the decision table's dispositions (items 1-9 stand, with item 9
  now cross-referenced by this note's item 3 rather than left as a bare disclosure).
- The t13 instrument and its outputs, the T1.2 CSVs, TREE2_CHARTER_20260830.md, and
  BIAS_HISTORY_LEDGER.md are untouched by this note (read-only verification greps against the
  latter two; no write to either).

**Net effect.** Reconciles (i) a real but narrow tension between a pre-theta code comment and this
node's own finding, without claiming a measurement supersedes an assertion that was never about
the same term; (ii) a genuine node-label collision, resolved by relabeling this node
"T1.3-zwin" rather than editing a charter file two other readers are not touching but this
presenter also declines to edit unprompted; (iii) a decisive-falsifier convention that had drifted
from "disclosed as unreconciled" (section 5.6/item 9, as originally written) to "silently
operative as primary" (F1's wording) — restored to its correct approval-scope status: proposed,
not ruled. None of the three touches this node's mechanism, derivation, predictions, cost,
regression plan, invariants, or falsifier thresholds.

Verified against darksiren_emri/bayesian_inference/bayesian_statistics.py (working tree, HEAD
7b1bd9dc, uncommitted diffs unchanged since Revision note 1), TREE2_CHARTER_20260830.md, and
BIAS_HISTORY_LEDGER.md row #266 (all read-only) in this repository at 2026-08-30. Presenter for
this note: top-tier subagent, per the branch's standing grant (row #255) covering production/
documentation changes within the tree (author-verbatim, row #223). No code written; no git
operations; foreground only; neither concurrently-written file (BIAS_HISTORY_LEDGER.md, the
T1_2_*/T2_2_* files, the HIER prereg) was written to by this note.

---

## Revision note 3 (2026-08-30; panel must_fix, documentation only; append-only)

**Trigger.** A third refuter panel on this document (refuted=false; four items, all must_fix,
none changing any registered number, band, prediction, falsifier, or plan item) flagged (1) an
unlabelled statistic collision between section 0's summary and section 3's calibration table;
(2) a missing item in section 8(b)'s capture-model-approximations list; (3) a missing inline
pointer from section 3 item (ii) to the 5.6 footnote's config-dependence caveat, even though
item (ii)'s number sets the registered band's floor; (4) a header claim ("No backtick
characters in this record") that Revision notes 1-2 (both appended UNDER this same header)
falsify. All four are addressed below by SUPERSEDED-by-note pointers / disclosure additions;
no text above this note (including Revision notes 1-2) is edited in place (append-only
discipline, the B5.1/T1.1 pattern).

**Item 1 — section 0's hw_sig-quartile numbers are the RAW statistic, not section 3's debiased
one. CONFIRMED; label added.**

Section 0 (lines 45-47) reads: "...it lives ENTIRELY in the quartile of events whose window is
narrowest in sigma_g units (half-width < 1.08 sigma_g: -0.254 +/- 0.037, Z -6.9; the other three
quartiles: -0.042 +/- 0.019, +0.011 +/- 0.010, -0.006 +/- 0.012)..." with no statistic-convention
label attached. Section 3's "Structure of the T1.2 residual by other covariates (all raw linear
form, combined_no_bh...)" paragraph (line 318) independently states its own numbers are the raw
linear form, but section 0's quartile figures are never tied to that same label — a reader
comparing section 0's numbers to section 3's "Calibration against measurement (iii)" quartile
table (lines 312-315: q1 **-0.290 +/- 0.038**, q2 -0.074 +/- 0.019, q3 -0.013 +/- 0.010, q4
-0.026 +/- 0.013, the debiased c-weighted ss_deb_c convention per line 306's "PA-HIER-32's
debiased statistic") could take section 0's four numbers as the same statistic under a different
grouping, when they are in fact a different statistic (RAW ss_lin) computed on the same
quartiling. Re-checked against this document's own citations: section 0's numbers match section
3's "Structure...(all raw linear form...)" n_cand/zeta/catalogue-share quartile block's
convention, not its "Calibration...(iii)" debiased block's convention — both are read from
t13_out.json, but from different keys (by_hw_sig_quartile, raw ss_lin vs.
*_model_vs_measured_by_hw_sig_quartile's "measured" column, debiased ss_deb_c).

**Disclosure (added here; changes no registered number).** Section 0's quartile figures
-0.254 +/- 0.037 (q1, hw < 1.08 sigma_g) / -0.042 +/- 0.019 (q2) / +0.011 +/- 0.010 (q3) /
-0.006 +/- 0.012 (q4) are the **RAW ss_lin statistic** (t13_out.json by_hw_sig_quartile key,
2026-08-30) — the same raw-linear convention as section 3's "Structure of the T1.2 residual by
other covariates" paragraph, computed pooled across all four seeds without the PA-HIER-32(d)
Es_null_det correction. They are **distinct from** section 3's "Calibration against measurement
(iii)" debiased ss_deb_c quartile numbers (-0.290 +/- 0.038 / -0.074 +/- 0.019 / -0.013 +/- 0.010
/ -0.026 +/- 0.013, t13_out.json *_model_vs_measured_by_hw_sig_quartile "measured" column,
2026-08-30), which are what section 3's model-vs-measured comparison and this document's
falsifier F1 (section 9, "the hw_sig-q1 class must move by >= +0.15 from -0.290") are actually
read against. The two conventions differ by design (Es_null_det_i's per-host correction, PA-HIER-
32(d)), not by measurement error; neither number is wrong, but section 0 as originally written
does not say which one it is quoting, and a reader who assumed section 0's -0.254 was the same
number as section 3's -0.290 would be reading two different statistics as one. This does not
change section 0's summary claim (the mechanism concentrates in the narrowest-window quartile) —
both conventions show the same ordering and the same concentration — only the labelling of which
statistic the summary's numbers are.

**Item 2 — section 8(b)'s capture-model-approximations list omits the envelope-linearity
assumption. CONFIRMED; item added.**

Section 8(b) (lines 504-506) lists the capture model's approximations as: "Gaussian kernel with
the E15 tilt only to first order; S_bar_phi weighting inside c_bar_g ignored; the 1e-6 floor
ignored; the b-axis registered as model-unreliable, section 3." Section 2.3's own derivation
(lines 178-193) defines the capture fraction with f_lo = z_min/z, f_hi = z_max/z HELD FIXED at
each event's own observed envelope ratios while treating the source redshift z_s as the varying
quantity the kernel probability integrates over ("c_g(theta) = P_{z ~ N(z_g^theta, s sigma_g)}
(...)"). The model's calibration section (lines 288-290) confirms the practice: "its event's ball
bounds are E6-exact" — i.e. f_lo, f_hi are read once per event from f7_events.csv's
z_min_ball/z_max_ball/z_GW columns and held constant across the s-sweep, not recomputed as a
function of the drawn z_s. This is a real, load-bearing approximation of the capture model (it
linearizes the GW-side envelope at the event's OBSERVED z_GW rather than tracking how [z_min,
z_max] would move if the true host redshift itself varied within the kernel's support) and
section 8(b)'s list, as originally written, does not disclose it alongside the other four.

**Disclosure (added here; changes no registered number, appended to section 8(b)'s list as a
fifth item).** (b) [continued] — the capture model additionally holds the envelope-linearity
assumption: f_lo = z_min/z, f_hi = z_max/z (section 2.3) are evaluated once per event at that
event's OBSERVED z_min_ball, z_max_ball and z_GW (f7_events.csv, E6-exact ball bounds) and held
FIXED while z_s varies inside the capture-fraction integral of section 2.3 — the model does not
re-evaluate the GW-side envelope at the varying source redshift the kernel integrates over. This
is consistent with the GW envelope itself being a fixed, theta-independent object at the level of
a single realized detection (section 2.4's own point: the envelope is data-side, not theta-side),
but it means the capture-fraction model is a per-event LOCAL linearization, not a re-derivation of
the envelope under a counterfactual host redshift; it is bounded by the same calibration evidence
already reported (section 3's three-way check against the measured T1.2 residual, the E9/E12
enlargement, and the hw_sig-quartile structure) and does not change the registered s-term shifts
of the section-3 table.

**Item 3 — section 3 item (ii) needs an inline pointer to the 5.6 footnote's c_i caveat, since
item (ii)'s number sets the registered band's floor. CONFIRMED; pointer added.**

Section 3's "Calibration against measurement" item (ii) (lines 308-311) reads: "...the tilted
variant is the calibrated one on the enlargement, and its -0.026 leftover equals E12's own
debiased residual (-0.005 - 0.0262 = -0.031 +/- 0.011), i.e. the non-capture terms (V2/V4/sky-
intrinsic), which no window fix removes" — with no pointer to where the -0.005 c-weighted input
to that subtraction comes from or what convention-dependence it carries. Section 5.6's own
footnote (lines 382-384) discloses: "Per PA-HIER-32's scope note, Es_null_det_i is per host and
configuration-free, but c_i changes with the candidate set, so the c-weighted convention needs
the ARM's own truth node (included in P1)." E12's -0.005 c-weighted point (cited at item (ii) and
at section 5.6's own P1 prose, line 392) was computed under the BANKED/T1.2 candidate set's c_i
(the enlarged-ball E12 measurement predates this node's own arm and has no truth node of its
own at the enlarged configuration), i.e. it debiases E12's own-arm raw score using a c_i drawn
from a DIFFERENT configuration (T1.2/baseline) than the one the -0.005 itself was measured under
— exactly the approximation section 5.6's footnote names, applied here one step earlier than
section 5.6's own P1 discussion states it. Item (ii)'s resulting -0.031 +/- 0.011 debiased
residual is not a free-standing number: it is quoted, unchanged, as the P1 floor in section 8(a)
("the non-capture terms... remain in P1 and set its floor (E12's -0.031 +/- 0.011 debiased)")
and as F1's calibration point for what a "PREDICTED, not marginal" P1 outcome looks like — so an
approximation entering it propagates directly into the registered band's interpretation.

**Disclosure (added here; changes no registered number).** Section 3 item (ii)'s -0.031 +/- 0.011
figure (E12's own debiased residual, which section 8(a) uses to set P1's predicted floor) is
subject to the same config-dependence caveat section 5.6's footnote states for the c-weighted
convention generally: c_i is config-dependent, and using T1.2's/baseline c_i to debias E12's own
(enlarged-ball) arm's raw score — because no truth node exists at E12's own configuration — is an
approximation, not an exact debiasing. This does not change the -0.031 +/- 0.011 number or the
section 8(a)/F1 floor it sets; it discloses that the floor itself inherits an un-quantified,
config-dependence-sized uncertainty beyond its stated +/- 0.011 SEM, on top of (not instead of)
the "reported, not adjudicated" convention status Revision note 2 item 3 already assigned to the
raw-vs-c-weighted choice generally.

**Item 4 — the header's "No backtick characters in this record" claim is false as of Revision
notes 1-2. CONFIRMED; withdrawn.**

The document header (line 14) states: "No backtick characters in this record." Revision notes 1
and 2 (both appended under this header, both dated 2026-08-30, both part of "this record") quote
code and file paths extensively in backtick-delimited spans (e.g. `mass_filter_k`, `z_window_k`,
`handler.py`, `single_host_likelihood`). Re-counted fresh this note (python3, character-exact,
2026-08-30) rather than taken on the panel's own say-so — the same evidentiary discipline
Revision note 1 item 2 applied to the panel's own suggested line-number fix: **182 backtick
characters** total in the document, ALL 182 within Revision note 1 (0 in Revision note 2, which
uses quotation marks instead of backticks throughout); 0 backticks in sections 0-11 above
Revision note 1. This is the actual count as re-verified against the working file; it differs
from the figure supplied to this presenter's launch instruction (46), which this note does not
reproduce uncorrected, on the same basis Revision note 1 corrected the panel's own imprecise
replacement line numbers rather than applying them as given.

**Disclosure (added here; withdraws the header claim; edits no earlier text).** The header's "No
backtick characters in this record" claim (line 14) is WITHDRAWN as of this note. It was accurate
for sections 0-11 as originally written (0-14, 2026-08-30, before any Revision note existed) and
remains accurate for Revision note 2, but is false for the document as a whole once Revision note
1 is included (182 backtick characters, re-counted fresh this note). Per append-only discipline,
line 14 is not edited; this disclosure is the correction of record. No registered number, band,
prediction, or falsifier is affected by a formatting-character count.

**Scope — what is, and is not, affected.**

- SUPERSEDED / disclosed (marked here, not edited in place): section 0's hw_sig-quartile numbers
  (lines 45-47) labelled RAW ss_lin, distinguished from section 3's debiased ss_deb_c numbers
  (lines 312-315) (item 1); section 8(b)'s capture-model-approximations list (lines 504-506)
  gains a fifth, disclosed item (the envelope-linearity assumption) (item 2); section 3 item
  (ii) (lines 308-311) gains an inline pointer to section 5.6's c_i config-dependence footnote
  (lines 382-384) (item 3); the header's line 14 "no backtick characters" claim is withdrawn
  (item 4).
- UNAFFECTED: every registered number, band, point prediction, cost figure, regression item, A10
  invariant, and falsifier threshold in sections 1-10 and in Revision notes 1-2; the 5-item
  physics-change form's substance; the decision table's dispositions (items 1-9 stand); the
  "T1.3-zwin" relabelling and the F1-convention downgrade-to-proposed from Revision note 2 are
  unchanged by this note.
- The t13 instrument and its outputs, the T1.2 CSVs, TREE2_CHARTER_20260830.md, and
  BIAS_HISTORY_LEDGER.md are untouched by this note (no write to any of them).

**Net effect.** Adds a statistic-convention label the summary needed to avoid a raw-vs-debiased
number collision (i); discloses a real, previously-unlisted capture-model approximation (ii);
threads an inline pointer from a floor-setting number to the caveat that already governs it (iii);
and withdraws a header claim the document's own later revision notes had already falsified,
correcting the withdrawal's own cited count (182, not 46) by fresh re-verification rather than
by repeating the figure handed down (iv). None of the four touches this node's mechanism,
derivation, predictions, cost, regression plan, invariants, or falsifier thresholds.

Verified against t13_out.json (by_hw_sig_quartile, *_model_vs_measured_by_hw_sig_quartile keys,
read-only), f7_events.csv (z_min_ball/z_max_ball/z_GW columns, read-only), and this document's own
text (character-exact backtick count, python3) at 2026-08-30. Presenter for this note: top-tier
subagent, per the branch's standing grant (row #255) covering production/documentation changes
within the tree (author-verbatim, row #223). No code written; no git operations; foreground only;
no concurrently-written file (BIAS_HISTORY_LEDGER.md, the T1_2_*/T2_2_* files, the HIER prereg,
TREE2_CHARTER_20260830.md) was written to by this note.

## Implementation prerequisites for T1.3-zwin (2026-08-30; moved here from docs/gates/PHYSICS-GATE-LEDGER.md by the orchestrator — the ledger stays tabular)


Scoped to whoever builds PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md's registered theta_zwindow/
z_window_k flag (row #255, tree 2 node T1.3-zwin; decision-table items 1-2 COVERED by the row
above). The flag and knob alone do not make P1's registered predictions (that gate doc's section
5.6) the numbers P1 will actually read off the driver. Three items beyond the flag itself:

1. **compute_scores() must implement PA-HIER-32(d)'s corrected score_s.** The T1.2 independent
   reader (PREREGISTRATION_HIER_HTHETA_20260826.md, Stage-0-recert record, "Disclosure on score_s
   form") found, verbatim: "The driver's compute_scores() (hier_s0_driver.py, unedited,
   grep-confirmed no Es_null_det term) computes the OLD/superseded raw linear secant, not this
   PA-HIER-32(d)'s corrected score_s = score_lns - Es_null_det" (hier_s0_driver.py:394-449, commit
   dd63fe0c per that document's own citation). PA-HIER-32(d)'s registered form
   (PREREGISTRATION_HIER_HTHETA_20260826.md:2757-2759), quoted:

       score_s_i = score_lns_i - Es_null_det_i
       Z_s = mean(score_s) / SEM(score_s), pooled per the existing (A8) two-sided convention

   with score_lns the now-superseded symmetric secant (same file, line 867):
   score_lns = [ lnL(b=0, ln s=+ln sqrt2) - lnL(b=0, ln s=-ln sqrt2) ] / (2 ln sqrt2). The
   T1.3-zwin flag changes the CANDIDATE SET compute_scores() sums over; it does not touch the
   scoring formula applied to that set. Both must change together, or P1's section 5.6 numbers
   (registered against the debiased score_s) will not be the numbers the unmodified driver
   actually emits.

2. **The s-node-only re-run form: nodes s_plus, s_minus, plus a fresh truth cell -- NOT a reuse of
   hier_s0_recert_run's existing truth/b nodes.** The gate doc's own P1 (section 5.6) registers 12
   cells: 4 truth + 8 s-nodes (s_plus/s_minus x 4 seeds) -- not the T1.2 recert's full 5-node
   theta-cross, and no b-nodes. Whether the recert's existing truth/b node CSVs can be spliced in
   byte-identically at s=1 instead of re-run: NO, at the registered z_window_k=4 -- per that
   document's own R2/R8/5.5: the theta=(0,1) LITERAL SKIP still applies k=z_window_k to the bare
   window (section 2.2: "'on' and theta==(0,1): LITERAL SKIP -- the 'off' path with k =
   z_window_k"), and R8 pins byte-identity against hier_s0_recert_run's truth CSV only "at
   k = 1.0"; at k = 4 only the intersection candidates' per-candidate values are bit-identical,
   added rows differ, so the truth node's aggregate output is NOT byte-identical to the recert
   run's. This is consistent with, not contradicted by, P1's own node list already including a
   fresh truth cell rather than citing the recert CSV -- the builder must run all 12 P1 cells
   fresh (hier_s0_recert_run/s0a_seed9001{01..04}/node_*_sites2.2_nosmear_divisor is NOT a valid
   substitute for any P1 node at k=4). b-nodes are not part of P1 at all (deferred to the optional
   P1b arm, section 5.6/6 item 3); T1.2's own b-axis certification (score_b, flag OFF throughout)
   stands unchanged as a claim about the untransformed window -- it is not, and cannot be treated
   as, a P1 node reused at k=4.

3. **The enlarged-ball parameters E12 named, as the registered P1 arm -- NOT E12's own pairing.**
   Section 5.6's P1 registers theta_zwindow="on", z_window_k=4.0, sky_cone_k=1.5 -- this is NOT
   B1_1_S0A_DEFECT_FORENSIC_20260829.md's own E12 measurement, which paired the z-widening with
   sky_cone_k=3.0; that combination is registered separately as the diagnostic fallback arm P3
   (section 5.6/9), to be run only if P1 fails its band. The builder must configure
   hier_s0_driver.py's three run_mirror_seed_inprocess call sites (the R7 plumbing item, section
   2.2) with z_window_k=4.0 and sky_cone_k=1.5 for P1, and must not default to E12's own
   sky_cone_k=3.0 pairing -- the two configurations test different mechanisms (section 2.1's
   theta-consistency of the z-filter vs. section 2.5's theta-free sky-intrinsic V3 term) and are
   registered as non-interchangeable by that document's own section 2.5 disclosure.

This note registers no new number, band, prediction, or falsifier, and authorizes no code beyond
what the row above and PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md's own decision table already
cover; it is a cross-read disclosure, not a new gate. Presenter: top-tier subagent, per the
branch's standing grant (row #255) covering production/documentation changes within the tree
(author-verbatim, row #223). No code, no git, foreground only; PREREGISTRATION_HIER_
HTHETA_20260826.md, hier_s0_driver.py and B1_1_S0A_DEFECT_FORENSIC_20260829.md read-only, not
written.
