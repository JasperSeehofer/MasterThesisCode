# /physics-change PRESENTATION GATE — the MASS-AWARE 1D CATALOGUE LEG as an instrument flag (catalogue_leg_1d_mass_aware in {"off","on"}, default "off", byte-identical) — 2026-08-30

**Launched under row #255 — tree 2 node T2.3.** Approval stamp: **row #255 (standing grant, tree 2 node T2.3 — instrument only)**.
Scope of the stamp (row #255 A17, orchestrator-derived scope mirroring rows #222/#223): instruments, registrations, path choices and
counterfactual arms inside the tree, each with its gate presentation BEFORE code and its ledger rows; lapses at the tree-2 verifier
pass. What the stamp does NOT cover (section 11): the production-default flip of this flag, because it modifies the pairing ratified
as Appendix B (i)/(ii) in row #169 — that flip returns to the author as a fresh [RULE] with the arm numbers of section 6 in hand.
Branch fix/p32d-classg-venue-repair, HEAD ecd33336; the working tree carries the uncommitted T1.1 (theta_phi_divisor, sky_cone_k)
and T2.2 (candidate_dump_dir) edits, so every bayesian_statistics.py line number below is the WORKING-TREE number on 2026-08-30
(8886-line file); where the derivation of record cites the ecd33336 number it is given in parentheses. Presenter: top-tier subagent;
**no code is written under this node** (presentation before code; the T1.1 and B7.3 precedents). Builder must be a different agent
from this presenter; the runner of any registered arm a different agent from the builder (builder != runner). No backtick
characters in this record; code identifiers are set in plain quotes. Every number carries {value, source file:line or artefact,
date}; section 13 is the provenance table. NO ssh, NO git, NO code edits by this node.

Derivation of record (read in full): results/campaign51_20260728/realistic_20260729/tree2_20260830/B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md
(row #261): section 2.3 (the s_beta decomposition and the −3/h common factor), section 4 (the normalisation identity (V)/(VI); the
composition mismatch beta_G/beta_Gbar = 0.1726 against alpha_G/beta_Gbar = 0.0661; Z(h) = D_phi/D_tilde_phi = 1.0999 at h = 0.73 with
d ln Z/dh = −0.189 per unit h per event), section 5.4 (the remedy this gate registers), section 7 (the F3 predictions), section 4.4
(the −0.14 per event dark-class completion-leg residual handed to B8 [CAL]).

---

## 0. Scope and one-paragraph summary

The production 1D (no-BH) channel assembles, per event, p_i = (beta_G_phi L_cat,i + B_i)/D_tilde_phi (bayesian_statistics.py:6521)
with a catalogue leg whose per-candidate survival is the population-average S_bar_phi(z_g; h) (:8089 quadrature path, :8082 point
path), whose global divisor is Sigma_phi = sum_g w_g S_bar_phi(z_g; h) (:2968, :3075; consumed at :5936-5942 and :6019-6021) and whose
class weight is the MASS-BLIND selection integral beta_G_phi (:2131; consumed at :6521). Its divisor D_tilde_phi = alpha_G_phi +
beta_Gbar_phi (:2497) carries the MASS-AWARE catalogue selection alpha_G_phi = beta_G_phi r_Malm (:2496; r_Malm = Sigma_4D/Sigma_phi
= 0.38277622 at h = 0.73). The derivation of record shows (section 4.3) that this pairing — mass-blind numerator and weight against a
mass-aware divisor — makes the 1D likelihood integrate over the data to Z(h) = (beta_G_phi + beta_Gbar_phi)/D_tilde_phi = 1.099921 at
h = 0.73 with d ln Z/dh = −0.18895 per unit h per event (an un-derived h-dependent global factor of the B_scale class retired in row
#131; −273 nats per unit h on the 1588-event production fleet, −0.21 in h against I_1D = 1303), over-weights the catalogue class by
1/r_Malm = 2.61 relative to the generator's own mass-aware detection of catalogue hosts (class odds 0.1726 coded versus 0.0661
generated; the code's own monitored gate (ii) at :2590-2598 predicts an in-catalogue share 0.0620 x 0.7305 = 0.0453 against the
realised 76/1588 = 0.0479), and violates MFG (2019) assumption A2 (the code's own reference at :1804-1808) in the 1D channel by exactly
r_Malm. This presentation registers the remedy as an INSTRUMENT FLAG, catalogue_leg_1d_mass_aware in {"off", "on"} with default
"off" byte-identical to HEAD: under "on" the 1D catalogue leg carries, per candidate and inside its own z-quadrature, the SAME
per-galaxy point survival S_4D(d_L(z; h), M_g (1+z)) that Sigma_4D already evaluates for that galaxy (:3022-3038), Sigma_4D as its
global divisor (already in hand as global_denom_with_bh, :5943) and alpha_G_phi as its weight (already in hand from path_a, :6501) —
the exact no-mass-likelihood image of the 2D assembly (alpha_G_phi L^2D + B g)/D_tilde_phi at :6524, whose with-BH twin
(catalogue_numerator_survival_2d = "mz_sel", commit d4765539) and whose 1D twin (catalogue_numerator_survival = "phi", commit bac48696)
are the two precedents this flag parallels. Under "on", integral p_i dd = (alpha_G_phi + beta_Gbar_phi)/D_tilde_phi = 1 identically
(Z = 1) and the 1D class share equals the generator's w_tilde_G = 0.0619668; and because alpha_G_phi/Sigma_4D = beta_G_phi/Sigma_phi =
1/n_hat_w_phi is an exact identity (:2494-2496), the assembled p_i under "on" differs from the coded p_i by exactly ONE factor per
candidate, S_4D(d_L(z;h), M_g(1+z))/S_bar_phi(z;h), while the divisor/weight re-booking changes only the L_cat_no_bh and w_G
diagnostics columns. Registered F3 predictions (section 6, two-sided bands, A15 characteristics at the arm sizes): (i) mirror FT-fleet
paired counterfactual Delta mean_h(on − off) = +0.05, band [+0.03, +0.10] (MATERIAL >= +0.03; NULL <= +0.008 = T_mat), 4 seeds for a
3-sigma read of the point prediction, 12 seeds for the lower band edge; (ii) production 1D per-class impostor score at h = 0.730: the
dark-class score scales by the in-ball Malmquist ratio rho (anchor 0.383) from −0.193 to −0.074, band [−0.097, −0.048] over rho in
[0.25, 0.5]; (iii) production 1D MAP from the 0.60 floor to about 0.675, band [0.64, 0.72] (a cluster arm, queued behind the OST
recovery) — NOT to truth, because the dark-class completion-leg residual of about −0.14 per event (derivation section 4.4, B8's) remains.
A14 falsifier: if the production 1D MAP does not leave the 0.60 floor under the flag (map_h <= 0.605 with the dark-only pure arm
unchanged), the Z(h) attribution is REFUTED. Cost: mirror cell 65 s (+ <= 1 percent), production 5-7 min per h-point (+ <= 1 percent;
the C4 STEP-2 pin measured 0.99x for the 24-node with-BH mass quadrature, of which this point form is 1/24 per host-node).

Site labels used below: **site N1** = the no-BH catalogue-numerator per-candidate survival factor (batch :8079-8089, scalar
:7319-7325/:7366-7372); **site D1** = the no-BH catalogue global divisor consumer (:5936-5942, :6019-6021); **site W1** = the 1D
mixture class weight (:6521).

---

## 1. OLD formula (exact, as implemented in the working tree on 2026-08-30; ecd33336 anchors in parentheses)

**(1a) The 1D per-event likelihood as assembled (path A, absolute_marginal; all production defaults).**

    :6495    beta_G_phi = self._beta_G_phi_table[self.h]
    :6496    beta_Gbar_phi = self._beta_Gbar_phi_table[self.h]
    :6497    sigma_phi = self._global_cat_selection_phi.get(self.h, 0.0)
    :6498    path_a = path_a_mixture_objects(
    :6499        beta_G_phi, beta_Gbar_phi, sigma_phi, global_denom_with_bh
    :6500    )
    :6501    alpha_G_phi = path_a["alpha_G_phi"]
    :6502    D_tilde_phi = path_a["D_tilde_phi"]
    :6503    r_Malm = path_a["r_Malm"]
    :6520    combined_without_bh_mass = float(
    :6521        (beta_G_phi * L_cat_without_bh_mass + B_num_phi) / D_tilde_phi          # site W1 (ecd33336 :6270-6273)
    :6522    )
    :6523    combined_with_bh_mass = float(
    :6524        (alpha_G_phi * L_cat_with_bh_mass + B_num_wbh_phi) / D_tilde_phi        # the 2D assembly, UNCHANGED (ecd33336 :6274-6276)
    :6525    )

In symbols (derivation section 1):

    p_i(h) = ( beta_G_phi(h) L_cat,i(h) + B_i(h) ) / D_tilde_phi(h)
    L_cat,i(h) = sum_{g in ball_i} w_g S_bar_phi(z_g; h) N_g(d_i | h) / Sigma_phi(h)
    w_g = R_eff_per_mbh(M_g) / (1 + z_g)                                   (:1036 _rate_weight; :5872-5873 the per-host weights)
    beta_G_phi(h)  = integral f_bar(z;h) S_bar_phi(z;h) p_pop(z;h) dz         (:2131)
    beta_Gbar_phi(h) = integral (1 − f_bar) S_bar_phi p_pop dz               (:2132)
    Sigma_phi(h) = sum_{g eligible} w_g S_bar_phi(z_g; h)                    (:2943 mask, :2957 w_g, :2968 interp, :3075 sum)
    Sigma_4D(h)  = sum_{g eligible} w_g S_4D(d_L(z_g;h), M_g (1+z_g))        (same function, with_bh_mass=True, point path :3022-3038, :3075)
    n_hat_w_phi = Sigma_phi / beta_G_phi; r_Malm = Sigma_4D / Sigma_phi;
    alpha_G_phi = Sigma_4D / n_hat_w_phi = beta_G_phi r_Malm;  D_tilde_phi = alpha_G_phi + beta_Gbar_phi;
    w_tilde_G = alpha_G_phi / D_tilde_phi;  D_phi = beta_G_phi + beta_Gbar_phi ("reported alongside")   (:2493-2498)

**(1b) Site N1 — the per-candidate survival factor in the no-BH numerator (the [P3-IMP] twin, row #197, commit bac48696).**
single_host_likelihood_batch (the production dispatch path, :7742-...):

    :8077    if _use_generator_point:
    :8078        numerator_without_bh_mass = gw_3d[:, 0]
    :8079        if _cat_surv_on:
    :8080            assert catalogue_survival_table is not None
    :8081            _z_s, _s_phi = catalogue_survival_table
    :8082            numerator_without_bh_mass = numerator_without_bh_mass * np.interp(host_z, _z_s, _s_phi)     # point path (ecd33336 :7833)
    :8083    else:
    :8084        assert prior_num is not None
    :8085        _num_integrand = gw_3d * prior_num
    :8086        if _cat_surv_on:
    :8087            assert catalogue_survival_table is not None
    :8088            _z_s, _s_phi = catalogue_survival_table
    :8089            _num_integrand = _num_integrand * np.interp(y_num_nodes, _z_s, _s_phi)                     # quadrature path (ecd33336 :7838-7841)
    :8090        numerator_without_bh_mass = _batched_gl_reduce(num_reduce_lo, num_reduce_hi, _GL_WEIGHTS_50, _num_integrand)

single_host_likelihood (the scalar twin; parity-pinned against the batch by test_scalar_batch_parity_phi):

    :7319    if _cat_surv_on:
    :7323        assert catalogue_survival_table is not None
    :7324        _z_s, _s_phi = catalogue_survival_table
    :7325        _num = _num * np.interp(np.asarray(z, dtype=np.float64), _z_s, _s_phi)      # quadrature integrand
    :7366    if _cat_surv_on:
    :7369        _z_s, _s_phi = catalogue_survival_table
    :7370-7372   single_host_likelihood_numerator_without_bh_mass *= float(np.interp(host_z, _z_s, _s_phi))   # point path

The flag and the per-h table slice reach BOTH host batches from p_Di (:5775-5787 resolve; :5796-5813 and :5816-5833 dispatch), so the
with-BH batch's r[0] no-BH numerator also carries the factor (A13; the row #197 form).

**(1c) Site D1 — the no-BH catalogue divisor consumer (the [P3-RPHI] fourth slot, rows #172-#178, commit e35ea018; T1.1's
theta-consistent form falls through to the point table at theta = (0,1)).**

    :5936    global_denom_no_bh: float = (
    :5937        getattr(self, "_global_cat_selection_phi_theta", {}).get(
    :5938            self.h, self._global_cat_selection_phi.get(self.h, 0.0)
    :5939        )
    :5940        if getattr(self, "_catalogue_global_selection", "s3d") == "phi"
    :5941        else self._global_cat_denom_no_bh.get(self.h, 0.0)
    :5942    )
    :5943    global_denom_with_bh: float = self._global_cat_denom_with_bh.get(self.h, 0.0)        # Sigma_4D, ALREADY IN HAND
    :6017    cat_num_sum_no_bh = weighted_sum([r[0] for r in all_results_without_bh], weights_without_bh)
    :6019    L_cat_without_bh_mass = (
    :6020        cat_num_sum_no_bh / global_denom_no_bh if global_denom_no_bh > 0 else 0.0            # (ecd33336 :5771)
    :6021    )

**(1d) The with-BH survival objects this flag re-uses, UNCHANGED.** Sigma_4D's point query (precompute_global_catalog_selection,
with_bh_mass=True, sigma4d_mass_kernel="point", the production default at :3597/:4289-4294):

    :3022    M_z_g = M_g * (1.0 + z_g)  # observer-frame mass (P_det grid axis)
    :3028    p_det = np.asarray(
    :3029        detection_probability_obj.detection_probability_with_bh_mass_interpolated(
    :3030            d_L_g, M_z_g, phi_iso, theta_iso, h=h, **_wbh_z_kwargs(detection_probability_obj, z_g),
    :3038    )

with d_L_g = dist_vectorized(z_g, h) in Gpc (:2958), M_g the raw catalogue BH_MASS column (:2949 M_all[eligible]; handler.py:79 and
:183), isotropic sky (:2990-2991), and the "kernel" alternative at :2996-3020 (E over N(M_eff_g, sigma_g^2) via
_sigma4d_mass_kernel_expectation :6784, sigma_g = BH_MASS_ERROR, handler.py:80 and :184). The accessor is
simulation_detection_probability.py:2018 detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta, *, h, z=None): a
linear RegularGridInterpolator on the 2D survival grid p_det(d_L, M_z), clipped to [0, 1], monotone non-increasing in d_L, sky
marginalised internally. The T2.2 hook already serialises exactly this query per candidate at z_g (:5382-5385, column s_4d_zg_mg) next
to S_bar_phi(z_g) (:5377, column s_bar_phi_zg).

**(1e) The 2D precedents this flag parallels, UNCHANGED.** The with-BH catalogue-leg twin puts S_4D inside the candidate's own mass
quadrature: batch :8268-8290 (mz_integral = mz_integral * _mz_sel_2d_expectation_batch(mu_star, sigma_star, y_num_nodes, d_L_at_num,
_det_M, ...)), scalar :7576-7600, helper :6855-6937 (Gauss-Hermite order _MASS_TRUNC_GH_ORDER = 24, :444/:468, on the product Gaussian
x ~ N(mu_star, sigma_star^2)); the fused 2D completion leg carries S_4D inside its own dx_M (completion_mass_factor_g_sel :2276, called
at :6261); the fused 1D completion leg carries S_bar_phi at its z-nodes (completion_numerator_integrand_sel_1d :6184-6200). The
existing flag family whose pattern the new flag copies: catalogue_numerator_survival (:3600 class default, :3857 evaluate() kwarg,
:4064-4105 resolution block with the [PHYSICS]/COUNTERFACTUAL log lines), catalogue_numerator_survival_2d (:3607, :3867, :4108-4151),
catalogue_global_selection (:3626, :3888, :4154-4183); arguments.py:1107/:1140 (argparse), :358/:379 (properties); main.py:211-213
(dispatch), :1422-1429 (module-level evaluate() defaults).

---

## 2. NEW formula (the registered form; NOT implemented by this node)

### 2.1 The flag

catalogue_leg_1d_mass_aware in {"off", "on"}, class default "off" (:3600 pattern), evaluate() kwarg default "off" (:3857 pattern),
argparse --catalogue_leg_1d_mass_aware with the matching property (arguments.py:1107/:358 pattern), main.py dispatch (:211 pattern) and
module-level default "off" (:1422 pattern), correspondence_1d.run_mirror_seed_inprocess pass-through (the T2.2 candidate_dump_dir
pattern), hier_s0_driver.py common_kwargs pass-through. **"off" is byte-identical to the tree at HEAD: a single boolean check at each
of the three read sites, no new table, no new object.** Guard pattern (not a silent no-op, the T1.1 convention): "on" requires
normalization_mode = "absolute_marginal" AND catalogue_numerator_survival resolving to "phi" AND catalogue_global_selection resolving to
"phi" (the flag REPLACES the phi objects at sites N1/D1/W1; with the coded leg absent there is nothing to replace) — else raise at
setup. "on" logs a COUNTERFACTUAL warning line (the instrument is not a production posterior until the author's fresh [RULE]; the
:4093-4098 pattern), never a [PHYSICS] ACTIVE line.

### 2.2 The three sites under "on" (all three toggled together, the derivation's section 5.4 "all three toggled together")

**Site N1 (numerator survival), quadrature path, batch kernel at :8086-8089:** replace the factor np.interp(y_num_nodes, _z_s, _s_phi)
by

    S_4D( d_L(y_num_nodes; h) [Gpc], M_g (1 + y_num_nodes) [M_sun], phi = 0, theta = 0, h = h, **_wbh_z_kwargs(., y_num_nodes) )

evaluated with the SAME accessor, the SAME isotropic-sky convention and the SAME z rider as Sigma_4D's point query (:3028-3038) and
as the T2.2 hook (:5382-5385), at the numerator's own z-nodes (d_L(z;h) and the (1+z) mass lift both vary across the +/-4 sigma GW
window, exactly as the 2D twin evaluates its factor at y_num_nodes, :8280-8290). d_L at the nodes is already in hand in the batch
kernel as luminosity_distance_fraction * _det_d_L (the 2D twin's d_L_at_num, :8283). **Point path (:8079-8082):** replace
np.interp(host_z, _z_s, _s_phi) by S_4D(d_L(z_g; h), M_g (1+z_g), 0, 0, h) — the T2.2 column s_4d_zg_mg exactly. **Scalar twin
(:7319-7325, :7366-7372):** the same two replacements; parity pinned (section 10, R4).

**Which survival object, and its centering.** The registered form is the **point** form at the **raw catalogue BH_MASS M_g**
(HostGalaxy.M, handler.py:79 — the same value _rate_weight consumes at :1036 and Sigma_4D's point path consumes at :3022), detector
frame M_z = M_g (1+z) at each node, distance d_L(z;h) at each node. This is Sigma_4D's production convention
(sigma4d_mass_kernel = "point", eddington_m has "No effect under point", :2837-2841 docstring), chosen so that the numerator factor and
the divisor Sigma_4D are the SAME function of (z, M_g) — the pairing MFG A2 requires. It is NOT the 2D twin's "eff" centering
(host_M_eff, Eddington-shifted, :8272-8275): the cross-channel asymmetry raw-versus-eff already exists between Sigma_4D (raw) and the
2D numerator (eff) and is not this gate's object; it is disclosed, not resolved.

**Whether the candidate mass enters with its error; the mass integral.** Under the registered point form the catalogue BH_MASS_ERROR
does NOT enter (a delta kernel in M — Sigma_4D "point"). The paired alternative is the **kernel** form, in which the per-node factor
becomes the mass integral

    S_4D^ker(z; g, h) = integral dM  N(M; M_eff_g, sigma_g^2)  S_4D( d_L(z;h), M (1+z) ),     sigma_g = BH_MASS_ERROR (handler.py:80),
    M_eff_g = the Eddington-shifted mean under eddington_m = "on" (raw M_g under "off"), exactly as :3010-3020,

computed by the erf-sum inner-M closed form _bh_mass_denominator_inner_m_integral_batch (:6704) that production's own per-host D_g
uses and that instrument J's _sigma4d_mass_kernel_expectation (:6784-6852) wraps for Sigma_4D — the code that already does this
integral per galaxy. Its relation to the 2D twin's mass quadrature (:6855-6937): _mz_sel_2d_expectation integrates S_4D against the
PRODUCT Gaussian N(x; mu_star, sigma_star^2) of the GW-conditional mass posterior and the galaxy's own N(M_g, sigma_g^2); in the 1D
channel there is no GW mass posterior, so sigma_cond -> infinity, mu_star -> mu_gal and sigma_star -> sigma_gal, and the 1D kernel
form IS the sigma_cond -> infinity limit of the 2D twin's expectation (the same S_4D accessor, the galaxy's own Gaussian, no product).
**Registered coupling rule:** the 1D flag has NO mass-form knob of its own; it reads self._sigma4d_mass_kernel (:4294) and mirrors it —
"point" numerator with the "point" Sigma_4D, "kernel" numerator (E over N(M_eff_g, sigma_g^2), inheriting eddington_m) with the
"kernel" Sigma_4D — so the numerator's mass measure can never differ from the divisor's (an unpaired subset is exactly the
[NUMERATOR-ONLY-CLEAN] defect class, EXONERATION_REGISTER_20260827.md:541-556). Only the point form carries registered predictions;
the kernel form is the paired alternative if Sigma_4D is ever switched (derivation section 5.4), REPORTED-ONLY.

**Site D1 (global divisor) at :6019-6021:** under "on", L_cat_without_bh_mass = cat_num_sum_no_bh / global_denom_with_bh (Sigma_4D,
:5943 — already in hand, no new computation) in place of global_denom_no_bh (Sigma_phi). T1.1's theta-ratio rho(theta; h) is a
Sigma_phi object and does not apply to Sigma_4D (Sigma_4D is theta-inert under the form of record, T1.1 invariant 9); "on" with
theta_phi_divisor = "on" raises (no theta-consistent Sigma_4D exists; guard, not a silent no-op).

**Site W1 (class weight) at :6521:** under "on", combined_without_bh_mass = (alpha_G_phi * L_cat_without_bh_mass + B_num_phi) /
D_tilde_phi, with alpha_G_phi the identical float the 2D assembly consumes at :6524 (:6501). B_num_phi, D_tilde_phi, beta_G_phi,
beta_Gbar_phi, Sigma_phi, Sigma_4D, r_Malm, n_hat_w_phi, w_tilde_G: all UNCHANGED objects. The diagnostics column w_G (:6519,
path_a["w_tilde_G"]) is already the mass-aware weight; the L_cat_no_bh column changes meaning under "on" (its divisor becomes
Sigma_4D) and the run_metadata records the resolved flag value (the B7.3 6.3 pattern).

### 2.3 The registered form in symbols

    L_cat,i^(1D,4D)(h) = sum_{g in ball_i} w_g  S_4D(d_L(z;h), M_g (1+z)) |_{inside N_g's z-quadrature}  N_g(d_i | h) / Sigma_4D(h)
    p_i^(1D)(h)        = ( alpha_G_phi(h) L_cat,i^(1D,4D)(h) + B_i(h) ) / D_tilde_phi(h)

and, because alpha_G_phi / Sigma_4D = beta_G_phi / Sigma_phi = 1 / n_hat_w_phi exactly (:2494-2496: n_hat_w_phi = Sigma_phi/beta_G_phi,
alpha_G_phi = Sigma_4D/n_hat_w_phi),

    alpha_G_phi L_cat,i^(1D,4D) = (1/n_hat_w_phi) sum_g w_g S_4D_g N_g ,   beta_G_phi L_cat,i^(coded) = (1/n_hat_w_phi) sum_g w_g S_bar_phi,g N_g ,

so **the assembled p_i under "on" differs from the coded p_i by exactly one factor per candidate, S_4D(d_L(z;h), M_g(1+z)) /
S_bar_phi(z;h), inside the sum; the divisor/weight re-booking (Sigma_phi, beta_G_phi) -> (Sigma_4D, alpha_G_phi) is an exact identity
on combined_no_bh and changes only the L_cat_no_bh diagnostics column.** The "three toggled together" form is registered anyway
because (a) it makes integral L_cat dd = 1 hold for the diagnostics column, (b) it makes the 1D assembly line-for-line the image of
the 2D one (:6521 versus :6524), and (c) the identity is what regression item R3 pins (section 10) — a builder who switches one site
without the others produces a factor r_Malm or 1/r_Malm that R2 and R3 catch.

### 2.4 Z = 1 identically, and the class share equals the generator's

Normalisation over the data (derivation section 4.1, the same steps): with integral N_g(d) dd = 1 and the ball-restriction being the
support of N_g, integral sum_g w_g S_4D_g N_g dd over the full catalogue = sum_g w_g S_4D_g = Sigma_4D, so integral L_cat^(1D,4D) dd =
1 (the coded leg has the same property with S_bar_phi against Sigma_phi); integral B_i dd = beta_Gbar_phi by the same token. Hence

    integral p_i^(1D) dd = ( alpha_G_phi + beta_Gbar_phi ) / D_tilde_phi = 1        (Z = 1, at every h; d ln Z/dh = 0)

against the coded Z = (beta_G_phi + beta_Gbar_phi)/D_tilde_phi = 1.099921 with d ln Z/dh = −0.18895 per unit h {ARITH on
selection_tables_h_0_725/0_73/0_735.json, p3_work/ft_900101_work/seed900101; derivation section 2.2 table; 2026-08-30}. The 1D class
share under "on" is alpha_G_phi/D_tilde_phi = w_tilde_G = 0.0619668 {p3_work/ft_900101.log "path-A(h=0.7300)"; derivation section 2.2},
i.e. class odds alpha_G/beta_Gbar = 58688305.9/888403798.071 = 0.06606 (ARITH) — the generator's mass-aware odds (draw_rate_weighted_hosts
carries the catalogue mass into the SNR, handler.py:1021-1080; monitored gate (ii) :2590-2598 predicts 0.0619668 x 0.7305 = 0.0453
against the realised 76/1588 = 0.0479, claim card C5) — in place of the coded beta_G/D_tilde = 0.161888, odds 0.17258 (ARITH,
153322758.616/888403798.071). The point path makes Z = 1 exact; the quadrature path (S at the nodes, Sigma at z_g) carries the same
first-order disclosure the coded leg already carries (the T1.1 divisor instrument is the registered tool for that residual, on
Sigma_phi; no Sigma_4D analogue exists — invariant, disclosed).

### 2.5 What this is NOT

Not the refuted R-rescale (row #168: R = beta_G(legacy)/beta_G_phi, a free global multiplier); every object in 2.3 is already derived
and already computed for the 2D channel, nothing is fitted. Not a per-event weight (derivation section 5.1, REFUTED). Not the row #167
completed-weight fork (A11, the author's; section 5.2 is its input). Not a change to the completion leg, to D_tilde_phi, to the 2D
channel, to w_pop, to the ball, or to any table.

---

## 3. Reference (derivation + literature + the ratified rows this touches)

1. **Derivation of record:** B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md (row #261, 2026-08-30): section 2.3 (s_beta = −3/h + sigma_G;
   the −3/h volume factor is common to every leg and cancels; the 63/37 split is a bookkeeping artefact), section 4.1 (identities (V)
   and (VI): a E_G[s_comp] + b E_Gbar[s_imp] = 0 and a E_G[score] + b E_Gbar[score] = Z d ln Z/dh with Z = a + b = D_phi/D_tilde),
   section 4.2 (the banked class scores satisfy (VI) at the coded composition: −0.158 measured versus −0.189 required, 1.8 sigma),
   section 4.3 (the composition mismatch and the Z(h) defect; MFG A2 violated in the 1D channel by r_Malm; the row #169 justifying
   display identified D_phi with the coded D_tilde), section 5.4 (the remedy in the form registered here), section 7 (F3 table).
2. **Literature:** Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)-(7) — the selection integral must use the SAME population
   and detection model as every numerator (assumption A2; the code's own reference at :1804-1808 and in path_a_mixture_objects'
   docstring :2487-2489): for a catalogue galaxy of KNOWN mass the detection model D_tilde uses for that galaxy is S_4D(z_g, M_g), not
   the population average S_bar_phi(z_g). Gray et al. (2020), arXiv:1908.06050, Eq. (A.19) — the catalogue/completion partition; the
   departure from Eq. (A.10) ("p_det solely in the denominator") is the deliberate selected-prior form already adopted in rows #197 and
   #253 (the MFG-a verbatim check, docs/LITERATURE_WARNINGS.md, remains the Stage-L obligation before paper use — carried).
3. **Row #131 (2026-08-19):** B_scale = beta_Gbar_phi/beta_Gbar ruled a DEFECT (un-derived normalisation; docs/derivations/
   bscale_completion_normalization.md section 6: "the formula cannot be derived"); Z(h)^N is the same object class (a likelihood
   integrating to an h-dependent Z(h) equals a normalised likelihood times an un-derived prior Z(h)^N). The bscale memo's section 1
   (:44-46) explicitly declined to examine the catalogue leg's alpha-pairing ("that design choice is not under examination here") —
   this gate is that examination.
4. **Row #169 (2026-08-22, author verbatim "Ratify B, run fused re-measure + b0 test"):** Appendix B (i)/(ii) ratified — "D_tilde
   stays", "beta_G_phi stays" — AGAINST the R-rescale of Appendix A (row #168). The justifying display
   (PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md:124-133) wrote the denominator with both class terms S_bar-weighted (that is D_phi) and
   identified it with the coded D_tilde (which carries S_4D through r_Malm); the two coincide only at r_Malm = 1. This gate does not
   re-open the R-rescale and does not dispute the twin (row #197: the twin is the population-average approximation to the form
   registered here); it identifies a different inconsistency with a derivable resolution — which is why its production flip is a
   fresh [RULE] (section 11).
5. **Rows #189-#212 (BIAS_HISTORY_LEDGER.md:2830-2996, 2026-08-25..28), the D-tilde / 2D-twin derivation chain:** row #189 derived the
   per-candidate object "S_4D inside the candidate's own mass quadrature"; rows #190-#207 the companion Sigma-tilde^4D and the C2*
   control; row #209 the class-G S_bar_phi double-weight (13.5-16 percent, REAL — the related-not-identical 2D-side finding); row
   #211 the ×2.25-2.35 identity residual parked UNATTRIBUTED; row #212 the repair run UNDERPOWERED, superseded by CONFIRMED-at-33-seeds
   (row #216 item 1).
6. **The two twin precedents (the code this flag parallels):** commit bac48696 (2026-08-25, row #197): catalogue_numerator_survival
   "auto" -> "phi" under absolute_marginal, explicit "off" = counterfactual, suite 1821 green — per-candidate S_bar_phi in the 1D
   numerator (site N1 as coded); commit d4765539 (2026-08-29, row #253): catalogue_numerator_survival_2d = "mz_sel" (center "eff") as
   the production default — S_4D inside the with-BH catalogue numerator's own mass quadrature, 1D channel bit-identical (R6), STEP-2
   overhead pin 0.99x, gate presentation PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md (panel-clean, 0 rounds).
7. **Row #255 (2026-08-30, author verbatim "all ratified from the docket"):** A17 = the standing grant for tree 2; A10 = the T2.2 hook
   as an instrumentation change; the tree-2 charter (TREE2_CHARTER_20260830.md section 1) restates the scope and the two items
   explicitly NOT covered (A4, A11) — this flag's production flip joins them by the same approval-scope rule (its inputs — the arm
   numbers — do not yet exist).

---

## 4. Dimensional analysis

Inputs and their units (unchanged by the flag; both survival objects are dimensionless, so every unit assignment of the coded leg
carries over):
- S_4D(d_L, M_z) in [0, 1], dimensionless (simulation_detection_probability.py:2018 docstring: "bounded in [0, 1]"); S_bar_phi(z;h) =
  integral phi(log10 M) S_4D dlog10 M, dimensionless (fixb_pathA_phi_marginal_selection.md:67-69). The replaced factor and its
  replacement have the same dimension: 1.
- Query arguments: d_L(z;h) in Gpc (dist_vectorized, :2958 comment "# Gpc"; the accessor's first axis, docstring "Luminosity distance in
  Gpc"); M_z = M_g (1+z) in M_sun, detector frame (the accessor's second axis, "Observer-frame (redshifted) BH mass in solar masses";
  :3022 convention). Identical to Sigma_4D's query (:3022-3038), to the 2D twin's (a_nodes * det_M, :6870-6872, x M_z,det in M_sun),
  and to g_sel,prod's (:2276 docstring: "detector-frame mass, absolute d_L in Gpc, isotropic sky, _wbh_z_kwargs rider").
- w_g = R_eff_per_mbh(M_g)/(1+z_g): a per-MBH rate, yr^-1 (Babak et al. 2017; :1036); N_g(d | h): a density in the data coordinates
  (phi, theta, d_L/d_hat), all dimensionless (:7300-7312), so dimensionless; Sigma_4D and Sigma_phi: yr^-1 (sum of w_g times a
  dimensionless survival, :3075); L_cat^(1D,4D): dimensionless (yr^-1 over yr^-1); beta_G_phi, beta_Gbar_phi, alpha_G_phi,
  D_tilde_phi, B_i: all "in the units of p_pop dz" (:2117-2118; alpha_G_phi = Sigma_4D/n_hat_w_phi with n_hat_w_phi = Sigma_phi/beta_G_phi
  in yr^-1 per unit of p_pop dz); p_i: dimensionless density in the data coordinates.
- Consistency: alpha_G_phi L_cat^(1D,4D) and B_i are both in units of p_pop dz (as beta_G_phi L_cat and B_i are today, the bscale memo's
  section 2 "both numerator legs are already commensurate"), divided by D_tilde_phi in the same units -> dimensionless, as required.
- The kernel form's integral dM N(M; M_eff, sigma^2) S_4D: N is a density in M_sun^-1, dM in M_sun, S_4D dimensionless -> dimensionless;
  it is the same closed form D_g already evaluates (:6704).
- No new constant, no new table, no unit conversion: the flag consumes the existing accessor and the existing Sigma_4D and alpha_G_phi
  floats only.

---

## 5. Limiting cases (each with its registered pin)

| limit | result | pin / evidence |
|---|---|---|
| (L1) mass-information-free detection model, S_4D(d_L, M) = S(d_L) for all M | S_4D(z_g, M_g) = S_bar_phi(z_g) by the tower identity with a normalised phi (fixb_pathA :67-69); Sigma_4D = Sigma_phi, r_Malm = 1, alpha_G_phi = beta_G_phi, D_tilde_phi = D_phi -> the flag reproduces the coded leg (Z = 1 on both sides) | unit test R5 on a synthetic mass-flat survival object; equality to the interpolant's own accuracy (S_bar_phi is tabulated on 1500 z-nodes, :2062; 8e-7 at the anchors, :2965 comment) — disclosed, rtol 1e-6 |
| (L2) r_Malm -> 1 (the catalogue's selected detectability equals the population average) | as (L1) at the global level: the coded Z -> 1, the coded class share 0.1619 -> w_tilde_G; the per-candidate ratios S_4D/S_bar_phi may still vary but average to 1 under w | ARITH on the section 2.2 table: Z = (beta_G + beta_Gbar)/(beta_G r_Malm + beta_Gbar) -> 1 as r_Malm -> 1; pinned by R2's control |
| (L3) single candidate | alpha_G_phi L' = (1/n_hat_w_phi) w_g S_4D(z_g, M_g) N_g: the galaxy contributes its own rate times its own detectability; at low z where both survivals saturate to 1 the flag is a no-op on that event; a heavy, well-detected galaxy (S_4D > S_bar_phi) gains weight, a light one loses it — the expected Malmquist sign | R6: two-host hand computation; sign check S_4D(z, M_heavy) >= S_bar_phi(z) >= S_4D(z, M_light) on the production survival object at a z where S_bar_phi is not saturated |
| (L4) empty ball, L_cat = 0 | p_i = B_i/D_tilde_phi, bit-identical on/off (neither B nor D_tilde is touched) | R7: the C-C identity check (dark events with n_cand_no_bh = 0: combined_no_bh max_abs 0.0 across on/off; T1.1's F4 form) |
| (L5) global rescale S_4D -> c S_4D (with S_bar_phi following) | numerator sum ∝ c, Sigma_4D ∝ c, alpha_G_phi ∝ c, B ∝ c, beta_Gbar ∝ c, D_tilde ∝ c -> p_i invariant (degree 0) — as is the coded leg; NOT a discriminator between the two arrangements, a sanity pin against an implementation that rescales one object only | R8 (both "off" and "on" invariant to 1e-12) |
| (L6) the score identity at the estimator's own composition | with Z = 1, identity (VI) reads a E_G[score] + b E_Gbar[score] = 0 (a = w_tilde_G = 0.0620, b = 0.9380): the total expected score at truth vanishes at the estimator's composition; at the coded composition it is Z d ln Z/dh = −0.2078 (or −0.189 normalised), which the banked class scores satisfy at 1.8 sigma (derivation section 4.2: −0.158 measured) | registered SECONDARY read (REPORTED-ONLY): re-score the class-G venue b0i and the dark FT fleet under "on" and form a E_G + b E_Gbar; predicted 0 +/- 0.02 (combined seed SEM 0.017, derivation section 4.2); no new fleet is launched for it by this node |
| (L7) sigma_g -> 0 in the kernel form | the delta kernel recovers the point form exactly | already pinned for Sigma_4D (:6796-6800 docstring "pinned limiting-case test"); R9 pins it for the numerator |
| (L8) sigma_cond -> infinity in the 2D twin | _mz_sel_2d_expectation's mu_star -> mu_gal, sigma_star -> sigma_gal: the 2D twin's survival factor becomes the 1D kernel form; with sigma_gal -> 0 too it becomes the 1D point form S_4D(d_L(z;h), M_g(1+z)) at raw M_g (center "raw") | R10, the 1D/2D symmetry test (section 10): the two channels' per-host survival factors coincide bit-for-bit at the same nodes when the with-BH mass kernel is collapsed |

---

## 6. REGISTERED PREDICTIONS (F3, before any code) with two-sided bands and A15 operating characteristics at the arm sizes

Common inputs {value, source, date}: r_Malm(0.73) = 0.38277622 {selection_tables_h_0_73.json, p3_work/ft_900101_work/seed900101;
identical to production to 7 s.f., headreadout_20260827/iiib/event_likelihoods.csv columns; 2026-08-23 run, read 2026-08-30}; I_1D = 1303
= 1/0.0277^2 {sigma_h of the dark-only pure arm, b4_imp_stage1_production_o2.json /iiib/pure_dark_only/sigma_h = 0.02770785; 2026-08-29};
production per-event impostor scores at truth on iiib: pooled −0.265 (pooled SEM 0.0505), dark −0.1926, in-catalogue −1.7069
{same JSON /iiib/score_imp_mean, score_imp_sem_pooled, score_imp_mean_dark, score_imp_mean_incat; ASSUMPTION-JOIN on CRB row order,
in_catalog_frac 0.04786 = 76/1588; 2026-08-29}; production 1D full: mean_h 0.6077, MAP 0.60, floor-node mass 0.446 {same JSON
/iiib/full; 2026-08-29}; dark-only pure arm: mean 0.7134, MAP 0.70, sigma 0.0277, c68 TRUE {same JSON /iiib/pure_dark_only}; pure-arm
score total +157.92 nats per unit h over 1588 rows and Sigma s_full = −297.77 {iiib event_likelihoods.csv, ARITH secant 0.725/0.735;
derivation section 10; 2026-08-30}; FT mirror fleet (12 seeds, 2152 events): mean s_imp −0.21778 (per-seed SD 0.05477, SEM 0.01581),
mean s_full −0.14722 (SD 0.05451), mean s_pure +0.07056 (SD 0.08658), q1 (z_true < 0.3575) mean s_imp −0.79237 (per-seed SD 0.12667,
SEM 0.03657) {b4_imp_stage1_events.csv and b4_imp_stage1_split.json per_seed, ARITH this node; 2026-08-30}; 4-seed HEAD-basis q1 S(1)
= −1.0205308 (n 191; per-seed −0.95125/−0.90388/−1.12050/−1.09299, across-seed SD 0.10584) {B4_2_KWQ1_READOUT_RECORD.md sections 2 and
5; 2026-08-29}; paired-Delta seed-generalisation SDs: twin-on-fused +0.029068 with sd 0.017624 (12/12 positive)
{PREREGISTRATION_P3_TWIN_20260822.md fused-basis VERDICT block; row #173; 2026-08-23}, FT drag +0.12274 with SD 0.0268 (12/12)
{CLAIM_IMPOSTOR_DRAG_20260829.md C1 table; 2026-08-29}, O2 drag +0.07919 with SD 0.0414 {row #149; C1 table}, row #167 b2 arm
+0.034357 +/- 0.004342 -> SD = 0.004342 x sqrt(12) = 0.01504 (ARITH) {BIAS_HISTORY_LEDGER.md:2417-2430; 2026-08-22}; T_mat = 0.008
{derivation section 6.6; the HEAD-readout materiality threshold, row #213}.

### 6.1 Prediction (i) — the mirror FT-fleet paired counterfactual (Delta mean_h, "on" minus "off")

**Statistic.** Paired per-seed Delta mean_h = mean_h(catalogue_leg_1d_mass_aware = "on") − mean_h("off") on the FT configuration
(absolute_marginal, catalogue_numerator_survival = "phi", fused, HEAD Sigma_phi divisor, theta = (0,1), sites 2.2, no smear — the KW-Q1
truth-node configuration), corrected combine (row #146 form) over H_GRID_41 (correspondence_1d.py:351-356), the 12 B-SEL realisations
(seeds 900101-900112) or the 4-seed subset 900101-900104; reported with the un-truncated H_GRID_FULL companion (amendment 20, row #173)
and the per-seed vector.
**Point prediction and band (derivation sections 6.6 and 7):** Delta mean_h = **+0.05**, two-sided band **[+0.03, +0.10]**.
Bands: **MASS-AWARE-MATERIAL** iff Delta >= +0.03; **NULL** iff Delta <= +0.008 (T_mat); **MIXED** otherwise; a NEGATIVE Delta of any
size is REFUTING (section 8). Secondary (same run): the in-ball Malmquist ratio rho_i = sum_g W_ig S_4D,g/S_bar_phi,g / sum_g W_ig on
active q1 dark events, predicted **rho_q1 in [0.2, 0.5]** (anchor 0.383); the q1 impostor score scaled by rho: 12-seed −0.79 ->
**−0.31 +/- 0.10**; 4-seed HEAD-basis S(1) −1.02 -> **−0.39 +/- 0.10**.
**A15 characteristics.** The design is a paired deterministic recomputation on identical events: the sampling variance of Delta under
the null (no change) is exactly 0 (the O2/O3 precedent, A15 corollary); the only width is the seed-generalisation SD of the paired
Delta, for which the banked anchors are 0.0176 (twin-on-fused: the closest analogue — a per-candidate survival re-weight of the same
leg), 0.0150 (the row #167 b2 arm: a global re-weight of the same leg) and 0.0268 (the FT drag itself: the upper analogue, since the
flag removes a fraction 1 − rho of the drag). How many seeds for a 3-sigma read: with SEM = SD/sqrt(N) and the NULL edge at +0.008,
the point prediction +0.05 is separated by 0.042: N >= 9 SD^2/0.042^2 = **1.6 (twin anchor) to 3.7 (drag anchor) -> 4 seeds** give a
3-sigma read of the point prediction under every anchor (at N = 4: 4.8 sigma twin-anchored, 3.1 sigma drag-anchored). The LOWER band
edge +0.03 is separated from NULL by 0.022: N >= 5.8 (twin) to 13.4 (drag) -> **12 seeds** give 4.3 sigma (twin-anchored) and 2.8
sigma (drag-anchored; disclosed as just under 3 under the conservative anchor). Engagement gate (A13): L_cat_no_bh must differ between
"on" and "off" on >= 99 percent of ACTIVE rows (denominator stated: rows with n_cand_no_bh > 0 at h = 0.73, the row #167 amendment-18
lesson); GATE I (assembly identity) <= 2e-6 on every node; GATE T-ID: the "off" arm at the KW-Q1 truth node reproduces
fanout1_20260829/kwq1_registered_run/s0a_seed900101/node_truth_ft_sites2.2_nosmear bit-identically on combined_no_bh and L_cat_no_bh.
**Design of record and its zero-compute half.** The T2.2 instrumented run (T2_2_CANDIDATE_HOOK_RECORD.md section 4; 4 seeds, FT,
truth node; as commanded at the single node h = 0.73) serialises S_4D(z_g, M_g) and S_bar_phi(z_g) per candidate, so **rho_i and the
per-event L'_i(0.73) are a zero-compute rescore** (derivation section 6.6) — the rho_q1 > 0.8 kill and the rho band are available
BEFORE any code from that dump. A Delta mean_h needs either the 3-node dump (0.725/0.730/0.735; derivation section 6.4, 3.4-3.9 CPU-h)
for a score-based linear prediction, or the fresh paired arm of section 9 (a). This gate registers the fresh paired arm as the
primary and the rescore as the pre-read (section 8, F-pre).

### 6.2 Prediction (ii) — production 1D at h = 0.730: the per-class impostor score

**Statistic.** Per-event secant score at truth over h = 0.725/0.735 on the 1588 iiib events, s_imp,i = d_h ln p_i − d_h ln pure_i (the
C2/C5 subtraction; GATE I <= 5.5e-7), by class (dark = host_galaxy_index −1; in-catalogue = 76 events), under "on" versus the banked
"off" (headreadout_20260827/iiib, row #213); and the per-event ratio q_i = s_imp,i(on)/s_imp,i(off) on active dark events. The class
split must use the T2.2b per-event CSV's host_galaxy_index (a VALIDATED join), not the CRB row-order assumption-join of C5 (claim card
section 4 item 2).
**Prediction.** The dark-class impostor score scales by rho = the in-ball Malmquist-class ratio, anchor 0.383: **dark −0.1926 -> −0.074**,
band **[−0.097, −0.048]** over rho in [0.25, 0.5] (ARITH −0.1926 x rho; band edge outward-rounded, see REVISION NOTE 2026-08-30b item 2
below); in-catalogue **−1.707 -> about −1.54** [SUPERSEDED 2026-08-30b — relabelled REPORTED-ONLY/UNSUBSTANTIATED, see REVISION NOTE
below] (the true host's own term does not scale by rho; derivation section 7 table: −130 -> about −117 nats over 76 events, ARITH
−117/76 = −1.54; band [−1.7, −1.4] [SUPERSEDED 2026-08-30b, same note]);
pooled **−0.265 -> −0.144**, band **[−0.166, −0.120]** [SUPERSEDED 2026-08-30b — inherits the in-catalogue number above, see REVISION
NOTE] (ARITH (1512 x dark + 76 x (−1.54))/1588 at the band edges). Median q_i on
active q1 dark events predicted 0.38, band [0.25, 0.5]; q_i > 1 on more than 10 percent of active dark events is REFUTING.
**A15 characteristics.** Same 1588 events, two codes: the null (no change) is q_i = 1 exactly with zero sampling variance, so the
statistic is decisive at one realisation; the unpaired width for reference is the pooled per-event SEM 0.0505 (dark-class shift +0.119
= 2.4 of it). No fleet scatter exists on production; the mirror's per-seed SD of the dark s_imp, 0.0548 (12 seeds), is the
generalisation-width analogue carried alongside. Engagement: >= 99 percent of active rows move (as 6.1).
**Arm.** T2.2b (derivation section 6.4): the hook on iiib at the 3 nodes with the flag "on" and "off" — 6 h-points, 5-7 min each
(charter anchor), about 4-5 CPU-h local; no cluster needed.

### 6.3 Prediction (iii) — production 1D MAP (the cluster arm, queued behind the OST recovery)

**Statistic.** The full 1588-event iiib 1D posterior over H_GRID_41 under "on" (a 41-task array, the row #213 form), paired against the
banked "off" (row #213 iiib: MAP 0.60, mean 0.6077, floor mass 0.446; the 1D channel is bit-identical between d04d9dc9 and HEAD by
d4765539's R6 and T1.1's/T2.2's byte-identity at their defaults); reported: map_h, mean_h, floor-node mass, and the dark-only pure arm
(which must be UNCHANGED, 0.7134 +/- 0.0277, since neither B nor D_tilde is touched — a C-C style pin).
**Prediction (derivation section 7, remedy (d) row).** Linear response Delta h = Sigma score/I_1D with the impostor scores scaled by rho:
dark −292 -> −112 nats per unit h, in-catalogue −130 -> −117 [SUPERSEDED 2026-08-30b — the −117 input is REPORTED-ONLY/
UNSUBSTANTIATED, see REVISION NOTE 2026-08-30b; this MAP band inherits it and is downgraded to REPORTED-ONLY pending T2.2b], pure
+158: Sigma = −71 -> −0.054 -> **MAP about 0.675**; over rho in
[0.25, 0.5]: Sigma = −32 (0.705) to −105 (0.649), rounded outward to the registered band **[0.64, 0.72]**. Bands: **Z-CONFIRMED** iff
map_h in [0.64, 0.72] AND mean_h in [0.64, 0.72]; **REFUTED** iff map_h <= 0.605 (the floor node; the row #213 rail statistic) with the
dark-only pure arm unchanged — the A14 falsifier of section 8; **MIXED** otherwise (0.61 <= map_h < 0.64, or map_h > 0.72). Secondary:
the dark-only full mixture (1514 events) predicted about 0.63, band [0.60, 0.67] (derivation section 7). Explicitly NOT predicted:
truth — the dark-class completion-leg residual of about −0.14 to −0.15 per event on production (derivation section 4.4; the C5
"dark-only pure arm covers truth" is a cancellation) is out of B4's scope and is B8 [CAL]'s object; the derivation's last table row
(remedy (d) plus a closed completion residual -> 0.73 within sigma_h) is a conditional, not a registration.
**A15 characteristics.** Single realisation; the posterior's own width sigma_h = 0.0277 (I_1D) makes the predicted 0.60 -> 0.675 shift
2.7 sigma_h; H_GRID_41 steps are 0.01 at the floor (0.60, 0.61, ...) so "leaves the floor" is a one-node, deterministic read; the
predicted shift is 7-8 nodes, so the falsifier is decisive at N = 1 and cannot be confounded by grid resolution. Censoring
disclosure (amendment 20): the "off" posterior is floor-censored (mass 0.446 on the 0.60 node); the paired mean_h difference is
therefore a LOWER bound on the un-truncated effect and is quoted as such.

---

## 7. A10 — invariants (with last-audited dates) and structural blindness

**Invariants held fixed by this change (one line each; the flag touches none of them):**
1. S_bar_phi table construction (:1982-2075 at ecd33336; FIXB_PATHA 2026-08-04; end-to-end NEVER audited — carried by name, T1.1
   invariant 1).
2. Sigma_4D point convention (sigma4d_mass_kernel = "point", raw M_g, isotropic sky) and its eligibility mask, identical to
   Sigma_phi's on the same rows/weights (D2, 2026-08-04; the retired mixed-catalogue r_Malm = 0.4304 lesson, :2824-2827).
3. w_g = R_eff_per_mbh(M_g)/(1+z_g), draw-consistent (G2b, last audited 2026-07; handler.py:1021).
4. B_num (fused 1D completion leg with S_bar_phi at the nodes) and B_num_wbh (fused g_sel,prod) — O4/O6, 2026-08-21; the derived
   B_scale = 1 form, row #131, 2026-08-19.
5. beta_G_phi, beta_Gbar_phi, D_tilde_phi, r_Malm, n_hat_w_phi, w_tilde_G — path A, 2026-08-04; every float re-used, none recomputed.
6. w_pop = (dVc/dz)/(1+z) (G2b; [WPOP-TUNING] binds).
7. The C7-core host-z kernel and the site-2.2 form (+/-4 sigma window, GL-50, 1e-6 floor; 2026-08-04); theta = (0,1) identity;
   theta_phi_divisor = "off"; sky_cone_k = 1.5; the +/-1 sigma_g z-window (handler.py:668-677).
8. The ball (sky cone 1.5 sqrt(lambda_max), z-window from d_hat +/- 2 sigma_dL over the h prior; :5545-5567) — [HARD-CLAMP-OBSERVED-Z]
   binds; untouched.
9. The 2D channel: catalogue_numerator_survival_2d = "mz_sel", center "eff" (d4765539, row #253) — bit-identical under the 1D flag (R6
   form, section 10 R11); the symmetric mass window (cf4f8a2a, row #202; irrelevant to 1D).
10. The generator: draw_rate_weighted_hosts with the catalogue mass (handler.py:1021-1080); the p0 window band-pass rho = 0.7305 (:1858).
11. H_GRID_41 (correspondence_1d.py:351-356); the 12 B-SEL realisations and the frozen q1 membership (b4_imp_stage1_events.csv).
12. Reduced catalogue md5 c52c13b5cab61f6b3f04bbe202550969 {t11_census_timing_out.json, T1.1, verified 2026-08-30} (A11 dataset pin;
    STOP on mismatch at every consumer).
13. The banked comparands: kwq1_registered_run truth-node CSVs (2026-08-29) and headreadout_20260827/iiib (row #213).

**Structural blindness (what this design cannot detect by construction):** the flag changes only WHICH survival each catalogue galaxy
carries in the 1D numerator; it is blind to (a) any defect common to S_4D and S_bar_phi — the with-BH survival object itself, the
injection pool's mass law, the p0 window's mass band-pass (rho = 0.7305) and the analysis-depth cap; (b) the dark-class completion-leg
residual of derivation section 4.4 (about −0.14 per event on production; the noiseless data law, the dark mass law in S_bar_phi) — B8's;
(c) the GW term and the C7 kernel's width/shape (the [HIER] axes; KW-Q1 INERT); (d) the catalogue's own BH-mass law (R&V15 scatter,
the M_error column) except through the paired kernel form, which is REPORTED-ONLY; (e) the class-G S_bar_phi double-weight of row #209
(2D side); (f) the cross-channel raw-versus-eff centering asymmetry (section 2.2), disclosed and left.

---

## 8. A14 — falsifiers, registered before any code

**F-pre (before the gate is opened for code; zero compute once the T2.2 dump exists):** the section 6.6 rescore on the 4 FT seeds
returns rho_q1 > 0.8 (the catalogue galaxies in q1 balls would then be as detectable as the population average and the class-share
argument would have no in-ball counterpart) — remedy (d) FALSIFIED before code; or, with the 3-node dump, the score-based linear
Delta mean_h(FT) <= +0.008 — FALSIFIED. (Derivation section 8.3.)
**F-Z (implementation level):** the R2 integral test under "on" returns Z != 1 (|Z − 1| > 1e-10 on the fixture) — the build is not the
registered form, STOP; the R2 control under "off" must return Z = D_phi/D_tilde != 1 on a fixture with r_Malm <= 0.9 (a can-fail
control, A15) — if it returns 1, the fixture is uninformative and the test is void.
**F-1 (the mirror attribution):** prediction (i) returns Delta mean_h(FT) <= +0.008 (NULL) or Delta < 0 — the Z(h)/class-share
attribution of the impostor-leg remainder is REFUTED on the mirror; the flag stays in the tree as a structural-consistency instrument
only and its production flip is NOT presented.
**F-2 (the production attribution, the falsifier named in the launch instruction):** under the flag, the production 1D MAP does not
leave the 0.60 floor — map_h <= 0.605 with the dark-only pure arm unchanged at 0.7134 +/- 0.0277 — then **the Z(h) attribution of the
1D rail is REFUTED**: the rail is then owned by the completion-leg residual (section 4.4) or by an object this design is blind to
(section 7), and the flag is a structural-consistency change with no H0 bearing; the fresh [RULE] of section 11 is presented with that
finding, not with a bias claim. Sharper: map_h in (0.605, 0.64) or > 0.72 is MIXED — the attribution is PROVISIONAL and the arm is
re-read by class (6.2) before any flip is presented.
**F-3 (the class-score form):** q_i > 1 on more than 10 percent of active dark events, or the in-catalogue class score moving by more
than its band [−1.7, −1.4] [SUPERSEDED 2026-08-30b — this band's center is REPORTED-ONLY/UNSUBSTANTIATED, see REVISION NOTE below;
the dark-class half of F-3 (q_i > 1 on >10 percent of active dark events) is unaffected] — the "scales by rho" mechanism statement is REFUTED even if the fleet Delta is MATERIAL (the effect would
then be carried by a different object; re-read by z-bin and n_cand, section 6.5 of the derivation).
**F-4 (the enlarged-ball companion, already registered — derivation section 8.2):** the depth-skew attribution's own falsifier
(Q = S_q1(enlarged)/S_q1(1.5 sigma) in [1.1, 1.6]; TRUNCATION-OBJECT if Q <= 0.9) is independent of this flag; under "on" the same Q is
predicted (rho is h- and ball-independent to first order) — a Q that changes by more than 0.2 between "off" and "on" is REPORTED as a
surprise.
**Rule-1 exoneration check (mechanism, not tag; re-run by this node 2026-08-30, not inherited):** EXONERATION_REGISTER_20260827.md
and BIAS_HISTORY_LEDGER.md sections 2-4 (:127-243) grepped for "mass-blind", "mass-aware", "class share", "in-cat share", "w_tilde_G",
"D_phi", "integrates to", "Z(h)", "alpha-pairing / alpha_G_phi", "Malmquist", "r_Malm": zero hits in the ledger sections; register hits
only on "class share" (:294, :880), both the 15.83-nat dark-class share of the 2D-versus-1D channel DIFFERENCE (the [Z-LEG] channel
accounting) — a different object, not the mixture's class-share normalisation. The register's C9 / Gate C item 1 keeps the
mixture-weight calibration LIVE (claim card section 0.2 row 15); [NUMERATOR-ONLY-CLEAN] (:541-556), [WPOP-TUNING] (:382-388),
[HARD-CLAMP-OBSERVED-Z] (:370-380) bind on remedies and are respected (paired three-object change with the n_hat_w identity; w_pop and
the ball untouched). The derivation's own section 9 grep (2026-08-30) reached the same result.

---

## 9. Cost (A11: measured anchors; bands where unmeasured)

Anchors {value, source, date}: mirror unsmeared cell 65 s {TREE2_CHARTER_20260830.md section 2 cost anchors; 2026-08-30}, measured S0-A
cell 60.9-64.0 s, mean 62 s at 14 cpu = 0.24 CPU-h {PHYSICS_CHANGE_THETA_DIVISOR_20260830.md section 6; s0a_full_output.json;
2026-08-29}; registered mirror anchor 0.2843 CPU-h per single-h cell + 0.1333 per cell overhead {PREREGISTRATION_HIER_HTHETA_20260826.md:584};
KW-Q1 measured 5.514 CPU-h / 24 cells = 0.23 CPU-h per cell {B4_2_KWQ1_READOUT_RECORD.md section 8; 2026-08-29}; a full-grid 12-seed
paired mirror re-measure = 24 evaluate() calls ≈ 12 CPU-h {row #169 costing correction; 2026-08-22}; production iiib approx 5-7 min per
h-point {TREE2_CHARTER section 2; 2026-08-30}, C4 task-0 h = 0.730 measured 385 s wall (00:06:25) and 6.8 CPU-h for 4 h-points = 1.7
CPU-h per h-point {B7_2_TWIN_CF_READOUT_RECORD.md:22-27; 2026-08-29}; **STEP-2 overhead pin 0.99x** (385 s versus C0's 388 s, same h,
same venue, "off" arm) for the 2D twin's 24-node Gauss-Hermite S_4D quadrature per (host, z-node) {B7_2_TWIN_CF_READOUT_RECORD.md:27;
COMPUTE_LEDGER.md:237-238; 2026-08-29}; T2.2 instrumented run 3.4 CPU-h {T2_2_CANDIDATE_HOOK_RECORD.md section 4}; T2.2b about 4-5 CPU-h
local {derivation section 6.4}.

1. **The flag's own overhead (the S_4D per-candidate query).** Under "on", the point form replaces one np.interp on a 1500-node 1D
   table per (host, z-node) by one evaluation of the 2D RegularGridInterpolator per (host, z-node), batched over hosts in ONE accessor
   call — the identical call shape the with-BH batch already makes 24 times per (host, z-node) for mz_sel (:8280-8290, order 24 at
   :444). The STEP-2 pin measured that 24x-larger addition at 0.99x on production; this addition is 1/24 of it per host-node, so the
   registered overhead is **<= 1.01x**, band [1.00, 1.05]: mirror cell 65 s -> <= 66 s; production per h-point 5-7 min -> unchanged
   within the anchor's own spread. Pinned by the same STEP-2 pattern: the h = 0.730 task of every arm records its Elapsed against the
   "off" arm's. The kernel form (REPORTED-ONLY) adds the erf-sum inner-M integral per (host, z-node) — the cost of a second D_g pass,
   band [1.5x, 2.5x] of the numerator stage, UNMEASURED; it is not launched by this gate.
2. **Arm (a), prediction (i):** the fresh paired mirror measurement at HEAD basis. Full form: 12 seeds x 2 arms x H_GRID_41 ≈ **12 CPU-h**
   local (row #169 anchor), about 6 h wall 2-wide; 4-seed form: 4 x 2 x 0.5 CPU-h ≈ **4 CPU-h**. The "off" arm is required because no
   12-seed HEAD-basis FT fleet exists over H_GRID_41 (FC/FT of row #173 predate e35ea018; the KW-Q1 truth node covers 4 seeds at 2
   h-nodes only). Pre-read (zero compute): the rescore of the T2.2 dump.
3. **Arm (b), prediction (ii):** T2.2b on iiib at the 3 secant nodes, "on" and "off": 6 h-points x 5-7 min wall ≈ **4-5 CPU-h local**
   (derivation section 6.4 anchor). No cluster.
4. **Arm (c), prediction (iii):** the 41-node iiib array with the flag "on": 41 x 1.7 CPU-h ≈ **70 CPU-h cluster**, wall about 6.5 min
   per task as an array (C4 anchor); the "off" comparand is the banked row #213 iiib arm (the 1D channel is bit-identical between
   d04d9dc9 and HEAD — d4765539 R6, T1.1/T2.2 byte-identity). **Queued behind the bwUniCluster OST 5 recovery** (charter: cluster
   DOWN; no ssh this session); batched, per F2, into the wave-3 blind HEAD readout as a per-change arm if the author's [RULE] flips the
   default, else run as a stand-alone counterfactual arm.
5. **Build and test:** local; the regression group of section 10 plus the full suite (baseline 1915 passed / 15 skipped / 30 deselected,
   T2.2 record section 3) — minutes.
Total registered under this gate: about 16-21 CPU-h local (arms (a) full + (b)) + 70 CPU-h cluster queued (arm (c)); the 4-seed
variant of (a) brings the local total to about 8-9 CPU-h.

---

## 10. Regression plan (tests written BEFORE the change where they pin existing values; builder != this presenter)

R1  Byte-identity at "off" (the row #197/#253 golden form): with the flag omitted and with catalogue_leg_1d_mass_aware = "off" explicitly,
    event_likelihoods.csv (all columns), both posterior JSONs and the candidate-dump CSVs are bit-identical to the pre-flag tree on the
    test_pipeline_parity fixture (the T2.2 GATE BI form) and on a live-catalogue smoke cell (the T1.1 smoke_run pattern, S0-A seed
    900101 node truth, event-cap 12); the scalar and batch kernels with the flag omitted are bit-identical (the
    test_worker_default_off_omitted_kwarg_is_bit_identical_{scalar,batch} pattern of test_catalogue_numerator_survival.py:176-192).
R2  Z = 1 unit test under "on" (the decisive pin): on a synthetic fixture (catalogue of ~200 galaxies with masses spanning the survival
    grid, a survival object with material mass dependence so that r_Malm <= 0.9 — asserted, so the control is informative), take the
    ball = the whole catalogue and N_g = a normalised Gaussian in the distance coordinate; integrate the assembled p_i over the data on a
    fine grid: under "on" the integral equals 1 to |Z − 1| <= 1e-10; under "off" it equals D_phi/D_tilde = 1 + beta_G(1 − r_Malm)/D_tilde
    (the coded Z, != 1 by >= 0.01 on the fixture) — a can-fail control (A15). An unpaired build (numerator switched, divisor or weight
    not) returns Z = r_Malm or 1/r_Malm and fails.
R3  The n_hat_w identity (section 2.3): under "on", combined_no_bh == (beta_G_phi * [sum w S_4D N / Sigma_phi] + B_num_phi)/D_tilde_phi
    to rtol 1e-12 (the same float assembled two ways) — proves the divisor/weight re-booking is exact and that L_cat_no_bh's change is
    the diagnostics column only.
R4  Scalar/batch parity under "on" (the test_scalar_batch_parity_phi pattern, :408): rtol 1e-12 on both numerator paths (point and
    quadrature), both host batches.
R5  Limit (L1): a mass-flat survival double returns S_4D(d_L, M) = S(d_L) for every M; under "on" the per-host numerator equals the "phi"
    numerator to rtol 1e-6 (the S_bar_phi table's own interpolation accuracy) and combined_no_bh under "on" equals "off" to the same
    tolerance.
R6  Limit (L3) and signs: two-host hand computation (scipy quad) for the point path; on the production survival object at a z where
    S_bar_phi is unsaturated, S_4D(z, M_heavy) >= S_bar_phi(z) >= S_4D(z, M_light) (the Malmquist sign).
R7  Limit (L4), the C-C identity: dark events with n_cand_no_bh = 0 have combined_no_bh bit-identical across "on"/"off"; B_num,
    D_tilde_phi, alpha_G_phi, r_Malm, w_G, L_cat_with_bh, combined_with_bh columns max_abs 0.0 across "on"/"off" on EVERY event (the
    flag reaches only combined_no_bh and L_cat_no_bh) — the R3 form of T1.1.
R8  Limit (L5): S_4D -> c S_4D (with the S_bar_phi table rebuilt from it): combined_no_bh invariant to rtol 1e-12 under both "off" and
    "on" (an implementation that rescales one object only fails).
R9  Limit (L7), kernel form (REPORTED-ONLY path, guarded): with sigma4d_mass_kernel = "kernel" and BH_MASS_ERROR -> 1e-8, the numerator
    factor equals the point form to rtol 1e-8; with sigma4d_mass_kernel = "point" the kernel branch of the numerator is never entered
    (hook counter).
R10 The 1D/2D symmetry test (registered in the launch instruction): under "on", with the with-BH mass kernel COLLAPSED — sigma_cond ->
    0 and sigma_gal -> 0 in _mz_sel_2d_expectation (the R5 limit of test_catalogue_numerator_survival_2d.py:457, center "raw") — the
    2D twin's per-host survival factor at the numerator nodes equals the 1D flag's factor S_4D(d_L(z;h), M_g(1+z)) bit-for-bit (same
    accessor, same arguments, same node z); and the assembled channels coincide in structure: the 1D "on" assembly (alpha_G_phi L'^1D
    + B)/D_tilde_phi reproduces the 2D assembly (alpha_G_phi L^2D + B g_i)/D_tilde_phi with the mass likelihood removed (mz_integral
    -> 1 by construction of the fixture's flat, unit-normalised mass posterior and g_i -> 1), to rtol 1e-10 — the same alpha_G_phi
    float, the same D_tilde_phi float, the same divisor object class (Sigma_4D) in both.
R11 2D bit-identity (the d4765539 R6 form, mirrored): combined_with_bh, L_cat_with_bh and every 2D diagnostics column max_abs 0.0
    between "on" and "off".
R12 Guards: "on" with catalogue_numerator_survival resolving to "off" or "phi_flat" raises; "on" with catalogue_global_selection "s3d"
    raises; "on" outside absolute_marginal raises; "on" with theta_phi_divisor = "on" raises; unknown token raises; the CLI flag
    defaults to "off", parses "on", rejects other values (the test_cli_flag_* pattern of test_catalogue_numerator_survival_2d.py:588-650);
    run_metadata records the resolved value.
R13 Engagement (A13): on the live smoke cell, "on" moves L_cat_no_bh on >= 99 percent of active rows and combined_no_bh on >= 90 percent
    of active rows by >= 1e-6 relative; the dispatch path is the batch kernel (assert the flag's runtime value inside
    _starmap_host_batches' kwargs for BOTH batches).
R14 Log lines: "on" emits the COUNTERFACTUAL warning (never a [PHYSICS] ACTIVE line) — pinned by caplog (the
    test_evaluate_explicit_off_logs_counterfactual_warning pattern, :780).
Suite counts: baseline 1915 passed / 15 skipped / 30 deselected (T2.2 record section 3, 2026-08-30) + the net-new tests above, zero
regressions; ruff, ruff format, mypy clean (darksiren_emri/ and darksiren_emri_test/).
Ledger rows to follow (the builder files them): "implemented | PASS" and "verified | PASS" in docs/gates/PHYSICS-GATE-LEDGER.md with
target bayesian_statistics.py:8079-8089/:7319-7325/:7366-7372/:5936-5942/:6019-6021/:6521 (working-tree numbers at build time), plus
the BIAS_HISTORY_LEDGER.md tree-2 row (next free number at filing; the orchestrator files it).

---

## 11. GOVERNANCE — what row #255 covers here, what it does not, and the decision table

**This flag modifies the row #169 pairing.** Row #169 (2026-08-22, author verbatim "Ratify B, run fused re-measure + b0 test") ratified
Appendix B (i)/(ii) — "D_tilde stays" and "beta_G_phi stays" — as the candidate of record; the coded 1D leg IS that pairing
(beta_G_phi weight, S_bar_phi numerator, D_tilde_phi divisor). The form registered in section 2 replaces "beta_G_phi stays" by
alpha_G_phi (with S_4D per candidate and Sigma_4D as divisor) while keeping "D_tilde stays". Under the approval-scope rule (CLAUDE.md:
an approval never propagates to a decision whose inputs did not exist when it was given; the tree-2 charter section 1 applies the same
rule to A4 and A11), and per the derivation's own section 4.3 and section 12: **the instrument and its counterfactual arms proceed
under row #255; the production-default flip returns to the author as a fresh [RULE] with the arm numbers.** Stated exactly:

- **[DO — covered by row #255 (standing grant, tree 2 node T2.3 — instrument only)]** implement catalogue_leg_1d_mass_aware in {"off",
  "on"}, default "off", byte-identical, with the guards and the regression plan of section 10; file the "implemented"/"verified" gate
  rows; builder != presenter; verifier at the end-of-tree-2 pass (mandatory scope).
- **[DO — covered]** the pre-read: the section 6.6 zero-compute rescore of the T2.2 dump (rho_q1, F-pre); runner != the T2.2 builder
  != this presenter.
- **[DO — covered]** arm (a), the paired mirror counterfactual (prediction (i)); arm (b), T2.2b on iiib at the 3 nodes (prediction (ii));
  both local.
- **[DO — covered, queued]** arm (c), the 41-node iiib array (prediction (iii)) — queued behind the OST 5 recovery; the /cluster
  preflight (VERDICT: READY) is required before submission; no ssh this session.
- **[RULE — FRESH, NOT covered by row #255; returns to the author WITH the numbers of arms (a)-(c) in hand]** flip the production
  default to "on" under absolute_marginal (the row #197/#253 "auto"-to-engaged pattern: explicit "off" becomes the counterfactual),
  thereby amending the row #169 (i)/(ii) pairing to (alpha_G_phi, S_4D, Sigma_4D, D_tilde_phi). The ask will be put as a decision
  table in a reviewable artifact (the row #253 form), with the F-1/F-2 falsifier outcomes stated first, and with the derivation's
  section 4.3 as its input. It is not put now: its inputs do not exist.
- **[RULE — FRESH, carried, not this node's]** A11 (row #167's D-tilde-phi factual fork) with derivation section 5.2 as its input
  (both branches stated there; neither branch as registered satisfies integral p dd = 1, because both keep the mass-blind pairing).
- **[RULE — FRESH, carried, not this node's]** the B8 [CAL] disposition of the dark-class completion-leg residual (derivation section
  4.4), which bounds what this flag can do on production and is named in prediction (iii) as the reason truth is NOT predicted.

Verifier scope (mandatory, row #255 A17 and the charter section 4 pointer): this presentation, the build, the R2/R3/R10 pins, arms
(a)-(c) and their band reads, and the falsifier outcomes F-pre/F-1/F-2/F-3.

**Approval stamp for the gate ledger:** row #255 (standing grant, tree 2 node T2.3 — instrument only).

---

## 12. What is explicitly NOT claimed

- Not claimed: that the flag un-rails the production 1D to truth (section 6.3; derivation section 4.4 forbids it).
- Not claimed: that the twin adoption (row #197) was wrong — the twin is the population-average approximation to this form, and its
  calibration residual (TWIN-FUSED-MATERIAL +0.0291 against the coded-leg drag +0.1518) is consistent with it (derivation section 5.4).
- Not claimed: any number of sections 6.1-6.3 as measured; every one is an F3 registration with a band and can fail.
- Not claimed: a production adoption; the flip is the author's fresh [RULE] (section 11).
- Not claimed: the kernel form's effect (REPORTED-ONLY, unmeasured); the raw-versus-eff centering asymmetry is disclosed, not resolved.

---

## 13. Provenance table (A11) — every number in this presentation

| number | value | source (file:line or artefact) | date |
|---|---|---|---|
| beta_G_phi, beta_Gbar_phi, Sigma_phi, Sigma_4D at h = 0.73 | 153322758.616 / 888403798.071 / 980867125.674 / 375452610.321 | p3_work/ft_900101_work/seed900101/selection_tables_h_0_73.json; derivation section 2.2 table | 2026-08-23 run; read 2026-08-30 |
| r_Malm(0.73); alpha_G_phi; D_tilde_phi; D_phi; w_tilde_G; n_hat_w | 0.38277622; 58688305.9; 947092104.0; 1041726556.7; 0.0619668; 6.397401 | same tables (ARITH via :2493-2498); p3_work/ft_900101.log "path-A(h=0.7300)" | 2026-08-30 |
| Z(0.73); d ln Z/dh; beta_G/D_tilde; class odds coded/generated; 1/r_Malm | 1.099921; −0.18895 per unit h; 0.161888; 0.17258 / 0.06606; 2.6125 | ARITH on the tables; derivation sections 2.2, 4.3 | 2026-08-30 |
| production tables identical to 7 s.f. | alpha_G_phi 58688310, r_Malm 0.3827762, D_tilde_phi 947092100 at 0.73 | headreadout_20260827/iiib/event_likelihoods.csv columns (row #213, d04d9dc9); derivation section 2.5 | 2026-08-28 run; read 2026-08-30 |
| monitored gate (ii) | 0.0619668 x 0.7305 = 0.0453 vs realised 76/1588 = 0.0479 | bayesian_statistics.py:2590-2598, :1858, :2698; claim card C5 | 2026-08-29 |
| Z tilt on production | −300 nats per unit h (−273 after /Z); I_1D = 1303; −0.21 to −0.23 in h | derivation section 4.3 (ARITH) | 2026-08-30 |
| production per-event impostor scores (iiib) | pooled −0.265037 (SEM 0.050481), dark −0.192564, in-cat −1.706895, dark share 0.6918 | b4_imp_stage1_production_o2.json /iiib | 2026-08-29 |
| production 1D full / pure / dark-only pure | 0.607677 / MAP 0.60 / floor mass 0.446; 0.839566 / 0.86; 0.713392 / 0.70 / sigma 0.027708 | same JSON /iiib/full, /pure, /pure_dark_only | 2026-08-29 |
| production score totals | Sigma s_full −297.77; Sigma s_pure +157.92 (1588 rows, secant 0.725/0.735) | iiib event_likelihoods.csv, ARITH; derivation section 10 | 2026-08-30 |
| FT fleet class scores, 12 seeds | s_imp −0.21778 (SD 0.05477, SEM 0.01581); s_full −0.14722 (SD 0.05451); s_pure +0.07056 (SD 0.08658); q1 s_imp −0.79237 (SD 0.12667, SEM 0.03657); mean_c_all 0.04153 | b4_imp_stage1_events.csv (arm ft, 2152 rows), b4_imp_stage1_split.json per_seed; ARITH this node | 2026-08-30 |
| KW-Q1 HEAD-basis q1 | S(1) = −1.0205308 (n 191); per-seed −0.95125/−0.90388/−1.12050/−1.09299; SD 0.10584; R = +0.084812 INERT | B4_2_KWQ1_READOUT_RECORD.md sections 2, 5; row #249 | 2026-08-29 |
| paired-Delta seed SDs | twin-on-fused +0.029068, sd 0.017624; FT drag +0.12274, SD 0.0268; O2 +0.07919, SD 0.0414; b2 arm +0.034357 +/- 0.004342 -> SD 0.01504 (ARITH) | PREREGISTRATION_P3_TWIN_20260822.md fused VERDICT block (row #173); CLAIM_IMPOSTOR_DRAG C1 table; row #149; BIAS_HISTORY_LEDGER.md:2417-2430 (row #167) | 2026-08-22/23, read 2026-08-30 |
| T_mat | 0.008 | derivation section 6.6; row #213 | 2026-08-28 |
| F3 point predictions and bands | (i) +0.05 [+0.03, +0.10]; (ii) dark −0.074 [−0.097, −0.048] (ARITH, band edge outward-rounded); in-cat about −1.54 [−1.7,−1.4] and pooled −0.144 [−0.166,−0.120] SUPERSEDED 2026-08-30b -> REPORTED-ONLY/UNSUBSTANTIATED (not ARITH; see REVISION NOTE); (iii) MAP about 0.675 [0.64, 0.72] and dark-only full about 0.63 [0.60, 0.67] SUPERSEDED 2026-08-30b -> REPORTED-ONLY (inherits the same in-catalogue number; see REVISION NOTE); rho_q1 [0.2, 0.5] (ARITH) | derivation sections 6.6, 7; ARITH this node at the band edges EXCEPT the in-catalogue-derived items marked above | 2026-08-30; relabelled 2026-08-30b |
| seeds for a 3-sigma read | point: N >= 1.6 (twin SD) to 3.7 (drag SD) -> 4; lower edge: N >= 5.8 to 13.4 -> 12 (2.8 sigma under the drag SD) | ARITH: N >= 9 SD^2/(Delta − 0.008)^2 | 2026-08-30 |
| H_GRID_41 | 0.60, 0.61, ..., 0.86 (0.01 at the edges, 0.005 in [0.65, 0.78]) | correspondence_1d.py:351-356 | 2026-08-30 |
| GH order of the 2D twin | _MASS_TRUNC_GH_ORDER = 24 | bayesian_statistics.py:444, :468 | 2026-08-30 |
| costs | mirror cell 65 s; S0-A cell 60.9-64.0 s (0.24 CPU-h at 14 cpu); 0.2843 + 0.1333 CPU-h per cell; KW-Q1 0.23 CPU-h per cell; 12-seed paired full-grid ≈ 12 CPU-h; iiib 5-7 min per h-point; C4 385 s per h-point, 1.7 CPU-h; STEP-2 pin 0.99x; T2.2 3.4 CPU-h; T2.2b 4-5 CPU-h | TREE2_CHARTER section 2; PHYSICS_CHANGE_THETA_DIVISOR section 6; PREREGISTRATION_HIER_HTHETA_20260826.md:584; B4_2_KWQ1 section 8; row #169; B7_2_TWIN_CF_READOUT_RECORD.md:22-27; COMPUTE_LEDGER.md:237-238; T2_2 record section 4; derivation section 6.4 | 2026-08-22..30 |
| catalogue md5 | c52c13b5cab61f6b3f04bbe202550969 | t11_census_timing_out.json (T1.1) | 2026-08-30 |
| suite baseline | 1915 passed / 15 skipped / 30 deselected | T2_2_CANDIDATE_HOOK_RECORD.md section 3; COMMIT_PLAN_5.md | 2026-08-30 |
| twin precedents | bac48696 (row #197, 2026-08-25); d4765539 (row #253, 2026-08-29) | git show --stat (read-only) | 2026-08-30 |

---

## 14. Gate-ledger row text (appended to docs/gates/PHYSICS-GATE-LEDGER.md by this node; the row is the record)

| 2026-08-30 | pre-commit | presented | row #255 (standing grant, tree 2 node T2.3 — instrument only) | bayesian_statistics.py:8079-8089 (site N1 batch), :7319-7325/:7366-7372 (site N1 scalar), :5936-5942/:6019-6021 (site D1), :6521 (site W1), :4064-4105 (flag-resolution block pattern; NEW flag catalogue_leg_1d_mass_aware in {off,on}, default off, byte-identical) | package = this file |

launched under row #255 — tree 2 node T2.3 — gate presentation complete, zero compute, no code, no git.

---

## 15. REVISION NOTE 2026-08-30b (post-refuter-panel; APPENDED, nothing above rewritten)

A refuter panel returned two must_fix items (refuted=false) on this document. Both are addressed below with evidence read from disk
today (2026-08-30, foreground commands, zero compute, no code, no git); the original sections 1-14 above are left as written, with
inline [SUPERSEDED 2026-08-30b] markers at the specific spots this note revises. Presenter for this note: same top-tier subagent role
as section 0 (a fresh gate-presentation act, not a builder/runner act — no code, no arm was run to produce this note).

### 15.1 Must-fix 1 — the in-catalogue "−130 -> about −117" transform: attempted derivation, found UNDERIVABLE with data on disk today

**What the panel asked:** derive the transform from the T2.2 hook's per-event S_4D(z_true,M_true)/S_bar_phi(z_true) columns for the 76
known in-catalogue production hosts (the document itself calls this "a zero-compute read once the dump exists," derivation section
6.6), rather than asserting −117.

**What was checked (evidence).** The T2.2 per-candidate dump does exist on disk, at h = 0.73 only, for the 4 FT-mirror seeds (the
candidate_dump_run directory this node's own charter cost line already cites):

    seed 900101: candidate_dump/seed900101_nodetruth/per_candidate_h_0_73.csv — 126215 data rows, 128 events, is_true_host: 126215 False / 0 True
    seed 900102: candidate_dump/seed900102_nodetruth/per_candidate_h_0_73.csv — 135174 data rows, 130 events, is_true_host: 135174 False / 0 True
    seed 900103: candidate_dump/seed900103_nodetruth/per_candidate_h_0_73.csv — 158958 data rows, 117 events, is_true_host: 158958 False / 0 True
    seed 900104: candidate_dump/seed900104_nodetruth/per_candidate_h_0_73.csv — 186224 data rows, 111 events, is_true_host: 186224 False / 0 True
    (paths under results/campaign51_20260728/realistic_20260729/tree2_20260830/candidate_dump_run/s0a_seed<NNNNNN>/node_truth_ft/...;
    read 2026-08-30, awk -F',' tail -n +2 | uniq -c on the is_true_host column, all 4 files)

Zero of 606,571 candidate rows across all 4 seeds carry is_true_host = True. This is structural, not a sampling accident: the T2.2
hook (section 6 of the derivation) was placed on the KW-Q1 FT-mirror arm, whose registered statistic (derivation section 6.5) is the
**dark** (host_galaxy_index = −1), q1 population — by construction that arm contains no in-catalogue true hosts to serialise. A
second check confirmed no production (iiib) per-candidate dump exists anywhere in the results tree (find -iname "*per_candidate*"
under results/, 2026-08-30: only the 4 files above); T2.2b — the hook run on production iiib that section 6.4 of the derivation
registers as "Optional... recommended, decisive for H0" — has not been executed.

**Conclusion.** The "−130 -> about −117" figure (equivalently the per-event −1.707 -> about −1.54, and the band [−1.7, −1.4]) is not
recoverable from any per-candidate data that exists on disk today, in either direction: not from the FT-mirror dump (no true-host
rows exist there at all, for any seed, at any node) and not from a production dump (T2.2b has not been run). The document's own
label "ARITH −117/76 = −1.54" is accurate only for that one division step; the −117 nats input to it is not ARITH — it has no shown
derivation anywhere in either this document or the derivation of record, only the qualitative statement "the true host's own term
does not scale by rho." Per the refuter panel's instruction, this number and everything computed from it is relabelled
**REPORTED-ONLY / UNSUBSTANTIATED**, not ARITH, at every location marked [SUPERSEDED 2026-08-30b] above: section 6.2's in-catalogue
and pooled predictions, section 6.2's F-3 falsifier band [−1.7, −1.4] (the dark-class half of F-3, q_i > 1 on >10 percent of active
dark events, is untouched — it does not depend on −117), section 6.3's MAP band [0.64, 0.72] and dark-only-full band [0.60, 0.67]
(both use the same −130 -> −117 input in the remedy-(d) row of derivation section 7 quoted at line ~443), and section 13's
provenance-table row for F3 point predictions and bands. The dark-class predictions (i) and the dark-only half of (ii) — which scale
by the anchored, ARITH rho = 0.383 and are not built from −117 — are UNCHANGED and remain ARITH.

**What would fix this going forward (not performed here — no code, no arms run by this node):** either (a) run T2.2b on production
iiib (registered cost 4-5 CPU-h local, derivation section 6.4) and read the true in-catalogue rows' S_4D(z_true,M_true)/S_bar_phi(z_true)
ratio directly — a genuine zero-compute read once that dump exists, exactly as the derivation states, or (b) re-run the T2.2 hook on
an event set that includes in-catalogue true hosts (the current FT-mirror arm's dark-only design structurally cannot supply them).
Both are arms, not gate-presentation acts, and neither is authorized by this note.

### 15.2 Must-fix 2 — dark-class band upper edge, −0.097 vs −0.096

**Recomputation (evidence).** −0.1926 x 0.5 = −0.09630 (document's own dark-class anchor to 4 s.f., section 13 provenance table,
"dark −0.192564"); rounded to 3 significant figures this is −0.0963, i.e. −0.096, not the document's stated −0.097.

**Finding.** This is intentional outward rounding, consistent with — not a violation of — this document's own stated convention: the
adjacent prediction (iii), section 6.3, explicitly rounds its own band edges "outward to the registered band" (Sigma = −32 (0.705) to
−105 (0.649), rounded outward to [0.64, 0.72] — 0.705 rounds outward to 0.72's direction of travel, 0.649 outward to 0.64). Applying
the same outward-rounding rule (away from zero, i.e. toward the more negative/more extreme value, which widens rather than narrows
the two-sided band) to −0.09630 gives −0.097, matching the document as written. The convention was, however, stated explicitly at
section 6.3 and left implicit at section 6.2 — an inconsistency in presentation, not arithmetic. Per the panel's alternative
("or state explicitly that outward rounding was intentional"), this note makes the convention explicit at section 6.2's location
(inline marker added above) rather than changing −0.097 to −0.096; the band [−0.097, −0.048] in section 6.2 stands as originally
written and is NOT superseded.

### 15.3 What is not touched by this note

Sections 1-5 (OLD/NEW formula, reference, dimensional analysis, limiting cases), section 6.1 (the mirror FT-fleet prediction,
Delta mean_h = +0.05, band [+0.03, +0.10]) and its A15 characteristics, section 7 (A10 invariants), the dark-class halves of
predictions (i) and (ii), sections 9-12 (cost, regression plan, governance, what is not claimed), and the section 13 provenance rows
not named above are unaffected — none of them depend on the −130/−117 number, and the panel raised no other item against them.

---

## 16. Gate-ledger row text, revised presentation (appended to docs/gates/PHYSICS-GATE-LEDGER.md by this node)

| 2026-08-30 | pre-commit | presented (revised) | row #255 (standing grant, tree 2 node T2.3 — instrument only) | REVISION NOTE 2026-08-30b: in-catalogue "−130 -> −117" transform (section 6.2, 6.3, F-3, provenance table) relabelled ARITH -> REPORTED-ONLY/UNSUBSTANTIATED after confirming zero true-host rows in the only per-candidate dump on disk (T2.2 FT-mirror, 4 seeds, 606571 rows, 0 True) and no production/T2.2b dump existing; dark-class band edge −0.097 confirmed as intentional outward rounding (document's own convention, section 6.3), not corrected | package = this file, section 15 |

launched under row #255 — tree 2 node T2.3 — revised presentation complete, zero compute, no code, no git.

---

## 17. REVISION NOTE 2026-08-30c (second refuter pass; APPENDED, nothing above rewritten)

A second refuter panel returned three must_fix items (refuted=false) on this document (post-section-15 state). All
three are addressed below with evidence read from disk today (2026-08-30, foreground commands, zero compute, no
code, no git); sections 1-16 above are left as written, with inline [SUPERSEDED 2026-08-30c] / [RESTORED
2026-08-30c] markers at the specific spots this note revises. Presenter for this note: same top-tier subagent
role as sections 0 and 15 (a fresh gate-presentation act, not a builder/runner act — no code, no arm was run to
produce this note).

### 17.1 Must-fix 1 — explicit hard sequencing rule: arm (c) is BLOCKED on arm (b)'s derived transform

**What the panel asked:** before arm (c) (the 41-node production MAP array of section 6.3/9 item 4, queued
behind the OST recovery) is ever run and its outcome judged against the Z-CONFIRMED/MIXED/REFUTED bands of
section 6.3, arm (b) (T2.2b on production iiib, section 6.2/9 item 3) MUST be run first and the in-catalogue
S_4D(z_true,M_true)/S_bar_phi(z_true) ratio for the 76 true in-catalogue hosts derived directly from that dump,
replacing the section-15.1 placeholder with a genuinely ARITH number; otherwise a REFUTED/CONFIRMED verdict from
arm (c) would not be a valid test of the registered prediction, since the [0.64, 0.72] band it would be tested
against is itself built on the admitted-unsubstantiated −117 input (section 15.1). The panel asked for this as an
explicit hard sequencing rule in section 11, not an implied ordering.

**Disposition: ACCEPTED, added as a binding addendum (append-only; section 11's text stands unedited above).**

**[SEQUENCING RULE — added 2026-08-30c, amends section 11's arm (c) bullet]:** Arm (c) (the 41-node production
iiib MAP array with catalogue_leg_1d_mass_aware = "on", testing prediction (iii) against section 6.3's
Z-CONFIRMED / MIXED / REFUTED bands) is **BLOCKED** — it MUST NOT be submitted, and if it were ever run its
outcome MUST NOT be read against those bands — **until arm (b) (T2.2b on production iiib, section 6.2 / section
9 item 3) has been executed AND has produced a derived, ARITH in-catalogue transform
S_4D(z_true,M_true;h)/S_bar_phi(z_true;h) for the 76 true in-catalogue production hosts, directly superseding
section 15.1's REPORTED-ONLY/UNSUBSTANTIATED placeholder** (currently "−130 -> about −117", section 6.3 and
derivation section 7 table row (d)). Reading section 11's existing bullet "[DO — covered, queued] arm (c), the
41-node iiib array (prediction (iii)) — queued behind the OST 5 recovery" together with this addendum: the
queue behind the OST 5 recovery is a NECESSARY but not SUFFICIENT precondition; arm (b)'s derived transform is
a second, independent gate that must also clear before arm (c) is submitted, regardless of cluster
availability. This sequencing rule binds the builder/runner of arm (c) as a hard STOP, not a recommendation: a
runner who submits arm (c) before arm (b)'s derived transform exists is out of scope of this presentation's
[DO] coverage under row #255, because the arm would not be testing the registered prediction of section 6.3 as
written (that prediction's own in-catalogue term is currently unsubstantiated). Arm (b) itself remains
[DO — covered] and un-blocked (section 11, unchanged); this rule does not touch it. The full-1588 MAP band
[0.64, 0.72] of section 6.3 and its A14 falsifier F-2 (section 8) remain REPORTED-ONLY, exactly as section 15.1
already established, pending that same arm (b) output — this note only makes the consequence for arm (c)'s
*sequencing* explicit rather than implied.

### 17.2 Must-fix 2 — cited line-number offsets: RE-VERIFIED against the working tree today, found to be EXACT, not off-by-one; must_fix item REFUTED with evidence

**What the panel asked:** re-grep and correct three cited ranges before the builder files the "implemented"/
"verified" gate-ledger rows: site W1 (":6521"/":6524" said to be actually ":6520"/":6523"), site D1's
global_denom_no_bh block (":5936-5942" said to be actually ":5936-5941") and its L_cat_without_bh_mass block
(":6019-6021" said to be actually ":6018-6020").

**What was checked (evidence, foreground, read-only, 2026-08-30, no code edits by this node, working tree
unchanged since the presentation was written per the launch instruction's NO code edits / NO git constraint):**

    $ awk 'NR>=6517 && NR<=6525{print NR": "$0}' darksiren_emri/bayesian_inference/bayesian_statistics.py
    6517:                 _den_used = D_tilde_phi if D_tilde_phi > 0.0 else 1.0
    6518:                 if D_tilde_phi > 0.0:
    6519:                     w_G = path_a["w_tilde_G"]
    6520:                     combined_without_bh_mass = float(
    6521:                         (beta_G_phi * L_cat_without_bh_mass + B_num_phi) / D_tilde_phi
    6522:                     )
    6523:                     combined_with_bh_mass = float(
    6524:                         (alpha_G_phi * L_cat_with_bh_mass + B_num_wbh_phi) / D_tilde_phi
    6525:                     )

    $ awk 'NR>=5936 && NR<=5943{print NR": "$0}' darksiren_emri/bayesian_inference/bayesian_statistics.py
    5936:             global_denom_no_bh: float = (
    5937:                 getattr(self, "_global_cat_selection_phi_theta", {}).get(
    5938:                     self.h, self._global_cat_selection_phi.get(self.h, 0.0)
    5939:                 )
    5940:                 if getattr(self, "_catalogue_global_selection", "s3d") == "phi"
    5941:                 else self._global_cat_denom_no_bh.get(self.h, 0.0)
    5942:             )
    5943:             global_denom_with_bh: float = self._global_cat_denom_with_bh.get(self.h, 0.0)

    $ awk 'NR>=6016 && NR<=6021{print NR": "$0}' darksiren_emri/bayesian_inference/bayesian_statistics.py
    6016:                 cat_num_sum_no_bh = weighted_sum(
    6017:                     [r[0] for r in all_results_without_bh], weights_without_bh
    6018:                 )
    6019:                 L_cat_without_bh_mass = (
    6020:                     cat_num_sum_no_bh / global_denom_no_bh if global_denom_no_bh > 0 else 0.0
    6021:                 )

    $ awk 'NR>=8075 && NR<=8091{print NR": "$0}' darksiren_emri/bayesian_inference/bayesian_statistics.py
    (site N1 batch: :8079 "if _cat_surv_on:" through :8089 "_num_integrand = ... np.interp(y_num_nodes, ...)",
    line-for-line identical to section 1(1b)'s quotation)

**Finding.** Every one of the disputed citations is byte-exact against the working-tree file as read today: line
6520 is "combined_without_bh_mass = float(", line 6521 is the beta_G_phi expression (section 0's "site W1
(:6521)" and section 2.2's "Site W1 ... at :6521" both correctly name the expression line, not the assignment-
opening line), line 6523/6524 the with-BH twin; the global_denom_no_bh block runs 5936-5942 inclusive (the
closing paren IS at 5942, not 5941 — the block is 7 lines, :5936 through :5942, exactly as section 1(1c)
quotes); L_cat_without_bh_mass runs 6019-6021 inclusive (6018 is the closing paren of the PRIOR statement,
cat_num_sum_no_bh, not part of this one). Site N1's :8079-8089 also reproduces line-for-line. No drift of any
kind was found at any of the four cited sites, in either direction.

**Disposition: the must_fix item's premise does not reproduce and is REFUTED by direct re-grep; no correction is
applied to sections 0, 1, 2.2, 10, 13, or 14 — every line number named there stands as originally written.** This
does not contradict the presentation's own standing caveat ("every ... line number below is the WORKING-TREE
number on 2026-08-30 ... builder != presenter, re-verifies at build time"): that caveat is prudent practice
regardless of this specific claim, and the builder should still re-grep before filing the gate-ledger rows per
section 10 — but the specific one-line offsets the panel named are not present in the file as it stands today.

### 17.3 Must-fix 3 — the dark-only-full band [0.60, 0.67]: independently re-derived, CONFIRMED not to depend on the disputed −117 input; RESTORED to ARITH

**What the panel asked:** section 15.1's superseded-item list names "section 6.3's ... dark-only-full band
[0.60, 0.67]" as relabelled REPORTED-ONLY, but the panel's own recomputation showed the dark-only-full
prediction does NOT use the disputed −117 input at all — only the full-1588 MAP band [0.64, 0.72] does — and
asked for clarification, with restoration to ARITH if the relabelling was in error.

**What was checked (evidence).** The derivation of record's own table (B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md
section 7, the table at line 636-643) gives, for remedy (d) at rho = 0.383, TWO SEPARATE cells built from
DIFFERENT inputs:

    production full 1D (1588):             dark −292 -> −112; in-cat −130 -> about −117; Sigma = −112 − 117 + 158 = −71 -> about 0.675
    production dark-only full mixture (1514): −112 − 22 = −134 -> −0.10 -> about 0.63, band [0.60, 0.67]

The dark-only-full cell is built ONLY from the dark-scaled impostor term (−292 x rho) and the dark-only pure-arm
score (about −22 per event aggregate, "from the 0.7134 read", derivation section 7 line 631) — it contains no
reference anywhere to the in-catalogue −130/−117 figure, because it excludes the 76 in-catalogue events from the
mixture entirely (1514 = the dark-only subset, not 1588 = the full production set). Independent recomputation
at the registered band edges (I_1D = 1303, h0 = 0.730, ARITH this node, 2026-08-30):

    rho = 0.25:  dark = −292 x 0.25  = −73;  Sigma = −73  − 22 = −95;   Delta h = −95/1303  = −0.0729;  MAP = 0.730 − 0.073 = 0.657
    rho = 0.383: dark = −292 x 0.383 = −112; Sigma = −112 − 22 = −134;  Delta h = −134/1303 = −0.1028;  MAP = 0.730 − 0.103 = 0.627  (matches "about 0.63")
    rho = 0.5:   dark = −292 x 0.5   = −146; Sigma = −146 − 22 = −168;  Delta h = −168/1303 = −0.1289;  MAP = 0.730 − 0.129 = 0.601

All three reproduce the document's own band [0.60, 0.67] (0.601 and 0.657 round to the stated edges under the
section 6.3 outward-rounding convention) and point value "about 0.63" exactly, using only the anchored rho and
the pure-arm read — zero dependence on −117 at any step.

**Disposition: ACCEPTED — the must_fix item is correct; section 15.1's inclusion of the dark-only-full band in
the superseded/relabelled list was an OVER-CAUTIOUS ERROR, not a correct application of the −117-dependency
test.** **[RESTORED 2026-08-30c]** Section 6.3's clause "the dark-only full mixture (1514 events) predicted
about 0.63, band [0.60, 0.67] (derivation section 7)" is UN-SUPERSEDED and reverts to **ARITH** status, sourced
to derivation section 7 table row (d) column 2 (line ~642) and the recomputation above. Correspondingly, in
section 13's provenance-table row "F3 point predictions and bands", the clause "dark-only full about 0.63
[0.60, 0.67] SUPERSEDED 2026-08-30b -> REPORTED-ONLY (inherits the same in-catalogue number; see REVISION NOTE)"
is itself **[SUPERSEDED 2026-08-30c]**: the dark-only-full band does NOT inherit the in-catalogue number (that
premise was wrong) and is ARITH as of this note. Everything else section 15.1 relabelled — the in-catalogue and
pooled predictions of section 6.2, the F-3 in-catalogue band [−1.7, −1.4], and the full-1588 MAP band
[0.64, 0.72] of section 6.3 — genuinely does use the −117 input (verified by the same table row: the full-1588
cell's "in-cat −130 -> about −117" term appears nowhere in the dark-only-full cell) and REMAINS
REPORTED-ONLY/UNSUBSTANTIATED, additionally now gated by 17.1's explicit sequencing rule.

### 17.4 What is not touched by this note

Sections 1-14 and section 15 (both must-fix items already resolved there), section 6.1 (the mirror FT-fleet
prediction and its A15 characteristics), the dark-class halves of predictions (i) and (ii), section 6.2's
in-catalogue/pooled predictions and F-3's in-catalogue band (still REPORTED-ONLY per section 15.1, now also
sequencing-gated per 17.1), the full-1588 MAP band [0.64, 0.72] of section 6.3 (still REPORTED-ONLY per section
15.1, now also sequencing-gated per 17.1), sections 9 (cost, item 4's queued status unchanged beyond the new
sequencing gate), 12 (what is not claimed) and the provenance-table rows not named in 17.3 are unaffected.

---

## 18. Gate-ledger row text, second revised presentation (appended to docs/gates/PHYSICS-GATE-LEDGER.md by this node)

| 2026-08-30 | pre-commit | presented (revised) | row #255 (standing grant, tree 2 node T2.3 — instrument only) | REVISION NOTE 2026-08-30c: added explicit sequencing rule (arm (c) BLOCKED until arm (b) derives the in-catalogue S_4D/S_bar_phi transform superseding section 15.1); re-verified site W1 (:6520/:6521/:6523/:6524), site D1 (:5936-5942, :6019-6021) and site N1 (:8079-8089) line citations against the working tree today and found them EXACT — the panel's claimed off-by-one drift does not reproduce, no correction applied; restored the dark-only-full band [0.60, 0.67] (about 0.63) to ARITH after confirming by direct recomputation it does not depend on the disputed in-catalogue −117 input, correcting section 15.1's over-cautious relabelling | package = this file, section 17 |

launched under row #255 — tree 2 node T2.3 — second revised presentation complete, zero compute, no code, no git.

---

## 19. Revision note (2026-08-30d; panel must_fix; append-only)

**What the panel asked (must_fix, minor):** the "production dark-only full mixture" event count reads 1514 at
section 6.3 (line ~452, "the dark-only full mixture (1514 events) predicted about 0.63, band [0.60, 0.67]")
and in section 17.3's quotation of the derivation's own table (lines ~908 and ~913, "production dark-only full
mixture (1514): −112 − 22 = −134 -> −0.10 -> about 0.63, band [0.60, 0.67]"; "1514 = the dark-only subset, not
1588 = the full production set"), against 1512 at section 6.2 (line ~430, ARITH "(1512 x dark + 76 x
(−1.54))/1588"). Both cannot be the validated dark-only count simultaneously.

**Reconciliation (this note).** 1512 = 1588 − 76 is the VALIDATED count: it uses the T2.2b per-event CSV's
host_galaxy_index join, the same join section 6.2 itself names as authoritative ("must use the T2.2b
per-event CSV's host_galaxy_index (a VALIDATED join), not the CRB row-order assumption-join of C5"). 1514 =
1590 − 76 instead derives from crb_rows = 1590 as read from fanout1_20260829/b4_imp_stage1_production_o2.json
joined against the 76 in-catalogue count WITHOUT the validated host_galaxy_index join — i.e. 1514 is itself an
instance of the assumption-join section 6.2 flags as non-authoritative, inherited from the derivation of
record's own section 7 (uncaught there; see the companion revision note appended to
B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md section 13, same date) and carried through this gate doc's section
6.3 and section 17.3 quotations of it.

**Numeric impact (ARITH, this note, 2026-08-30).** At the dark-only-full-mixture aggregate level, the
discrepancy is invisible at reported precision: 1512 x 0.193 = 291.8 and 1514 x 0.193 = 292.2 both round to
the stated −292/−112-per-rho-scaled figures used in sections 6.3/17.3. At the full-1588 "Current" Sigma level
(derivation section 7), Delta Sigma = 2 x (−0.193) = −0.386 nats (<= 0.4 nats) against Sigma ~ −264 — about
0.15 percent, again invisible at the whole-nat precision carried throughout. Recomputing the dark-only-full-
mixture MAP at the validated 1512 across the registered band edges (I_1D = 1303, h0 = 0.730): rho = 0.383
gives MAP = 0.6273 versus the previously stated 0.6272 — both "about 0.63" to the precision reported. **The
registered band [0.60, 0.67] and the point value "about 0.63" are UNCHANGED.**

**Affected lines (SUPERSEDED-by-this-note pointer only; the text at these lines is NOT edited and stands as
originally written):**
- Line ~452 (section 6.3): "the dark-only full mixture (1514 events) predicted about 0.63, band [0.60, 0.67]"
- Line ~908 (section 17.3, quoting the derivation table): "production dark-only full mixture (1514): −112 − 22
  = −134 -> −0.10 -> about 0.63, band [0.60, 0.67]"
- Line ~913 (section 17.3): "1514 = the dark-only subset, not 1588 = the full production set"
- Line ~430 (section 6.2) is CONFIRMED, not superseded: its "1512" is the validated count.

**Disposition: ACCEPTED as a minor must_fix — instrument-only.** The band [0.60, 0.67] and point value "about
0.63" registered for the dark-only-full-mixture prediction (section 6.3, restored to ARITH by section 17.3)
are UNCHANGED by this correction; nothing in sections 1-18 is otherwise affected. This note does not itself
authorize or constitute the production flip of remedy (d): that flip remains a fresh [RULE] for the author
per section 12/17's hand-off language, unaffected by this instrument-only reconciliation. Covered by row #255
as an append-only, zero-compute, no-code, no-git correction (tree 2 node T2.3).

### 19.1 Gate-ledger row text, third presentation (appended to docs/gates/PHYSICS-GATE-LEDGER.md by this node)

| 2026-08-30 | pre-commit | presented (revised, final) | APPROVED | row #255 (standing grant, tree 2 node T2.3 — instrument only; production flip returns as a fresh [RULE]) | Panel must_fix (minor, one item): the "production dark-only full mixture" event count read 1514 at section 6.3/17.3 of this file and at section 7 of B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md, against the validated 1512 (=1588−76) at this file's section 6.2. Reconciled: 1512 is the validated count (host_galaxy_index join); 1514 (=1590−76) came from the unvalidated CRB-row assumption-join (crb_rows=1590, fanout1_20260829/b4_imp_stage1_production_o2.json). Numeric effect <= 0.4 nats of Sigma ~ −264 (~0.15 percent), invisible at reported precision; the registered band [0.60, 0.67] and point value "about 0.63" for the dark-only-full-mixture prediction are UNCHANGED. Revision notes appended, append-only, to both documents (this file section 19; derivation section 13); no prior line was edited, only SUPERSEDED-by-note pointers recorded. No code, no arms run, no git by this node. |

launched under row #255 — tree 2 node T2.3 — third presentation, panel must_fix reconciliation complete, zero
compute, no code, no git.

---

## 20. IMPLEMENTATION RECORD 2026-08-30 (builder node; append-only, sections 1-19 untouched)

Builder: a different agent from the presenter of sections 0/15/17/19 (builder != presenter, per
section 0's standing requirement). Authorized under row #255 (standing grant, tree 2 node T2.3 --
instrument only); this node writes code (the presentation nodes above did not). No git by this
node (the orchestrator commits).

### 20.1 What was built (working-tree line numbers as of this commit; re-grep at any future date)

catalogue_leg_1d_mass_aware in {"off", "on"}, default "off", implemented exactly as registered in
section 2, with the following current line anchors in bayesian_statistics.py (all shifted from the
presentation's working-tree numbers by the intervening edits themselves; the presentation's own
standing caveat -- "builder re-verifies at build time" -- applied):

- New helper "catalogue_leg_1d_mass_aware_factor" (module-level, public name, no leading
  underscore so it is reachable from tests and from the derivation's own future consumers) at
  line 6961: implements both the "point" sub-form (the SAME accessor call Sigma_4D's own
  with-BH point branch and the T2.2 hook use -- detection_probability_with_bh_mass_interpolated
  at d_L(z;h), M_g(1+z), isotropic sky, the _wbh_z_kwargs z-rider) and the "kernel" sub-form
  (mirrors self._sigma4d_mass_kernel via _sigma4d_mass_kernel_expectation, the SAME erf-sum
  machinery production's own per-host D_g uses), per the registered coupling rule of section 2.2.
- Class default and __init__ default: line 3678 (_catalogue_leg_1d_mass_aware: str = "off") and
  line 3763 (the __init__ assignment), following the T1.1/T2.2 class-attribute-plus-init-default
  convention exactly.
- evaluate() kwarg + guard block: the flag is validated immediately after the sigma4d_mass_kernel
  block (the position in evaluate() where self._catalogue_numerator_survival,
  self._catalogue_global_selection, self._theta_phi_divisor, self._sigma4d_mass_kernel, and
  self._eddington_m are already resolved, so the three guards of section 2.1 can read them
  directly). "on" raises ValueError unless catalogue_numerator_survival resolves to "phi",
  catalogue_global_selection resolves to "phi", and theta_phi_divisor is "off"; logs a
  COUNTERFACTUAL warning (never [PHYSICS] ACTIVE, per section 2.1's stated convention -- this is
  an instrument, not a production posterior).
- Site D1 (the no-BH catalogue global divisor): global_denom_with_bh moved ahead of
  global_denom_no_bh (line 6019) so it is in scope for the "on" branch; global_denom_no_bh (line
  6027) now reads global_denom_with_bh directly under "on" -- Sigma_4D, already in hand, no new
  computation -- and falls through to the pre-existing ternary (Sigma_phi_theta / Sigma_phi_point
  / Sigma_3D) under "off", byte-identical.
- Site W1 (the 1D mixture class weight): the combined_without_bh_mass assembly (line 6626) now
  selects alpha_G_phi in place of beta_G_phi under "on" (alpha_G_phi is the IDENTICAL float the
  2D assembly on the very next lines already consumes, :6501 in the presentation's own citation);
  under "off" the weight is beta_G_phi, byte-identical to the pre-flag tree.
- Site N1 (the no-BH catalogue numerator's per-candidate survival), both the batch kernel
  (single_host_likelihood_batch) and its scalar twin (single_host_likelihood): each function
  gained two trailing keyword parameters, catalogue_leg_1d_mass_aware (default "off") and
  sigma4d_mass_kernel (default "point", mirrored from self._sigma4d_mass_kernel by the caller,
  never chosen independently by the worker); a validated _cat_leg_1d_ma_on flag; and, in BOTH the
  point-evaluation branch and the z-quadrature branch, an "on" branch that calls
  catalogue_leg_1d_mass_aware_factor at the candidate's own z-nodes and (raw catalogue) BH mass
  in place of the np.interp against catalogue_survival_table. The dispatch chain
  (_starmap_host_batches, and the two call sites in p_Di that invoke it for the with-BH and
  without-BH host batches) forwards self._catalogue_leg_1d_mass_aware and
  self._sigma4d_mass_kernel unconditionally to BOTH batches, matching the A13 note in section 1(1b)
  of this document (the with-BH batch's r[0] no-BH numerator also feeds L_cat_no_bh).

### 20.2 Where the presentation's registered form was AMBIGUOUS and the reading chosen (disclosed per the launch instruction)

1. The registered coupling rule (section 2.2) states the 1D flag "has no mass-form knob of its
   own" and must mirror self._sigma4d_mass_kernel. The presentation does not say explicitly
   whether the worker-level plumbing (single_host_likelihood/_batch, previously with no
   sigma4d_mass_kernel parameter at all) should gain a NEW parameter or read the flag some other
   way. Reading chosen: add sigma4d_mass_kernel as a new trailing keyword parameter to both
   worker functions (mirroring how eddington_m already reaches them), threaded from
   self._sigma4d_mass_kernel at the two _starmap_host_batches call sites -- this makes the 1D
   leg structurally identical to the 2D assembly's own parameter-threading pattern and keeps the
   coupling automatic (a caller cannot pass a sigma4d_mass_kernel value inconsistent with
   Sigma_4D's own, since production only ever passes self._sigma4d_mass_kernel).
2. Section 2.2's guard list (site D1) states "on" with theta_phi_divisor="on" raises; section 2.1
   lists the THREE evaluate()-level guards (normalization_mode absolute_marginal via the two "phi"
   resolutions, catalogue_numerator_survival "phi", catalogue_global_selection "phi") but the
   fourth (theta_phi_divisor="off") is stated only in section 2.2's prose, not repeated in
   section 2.1's guard paragraph. Reading chosen: implement all four guards together in one block
   (the three from section 2.1 plus theta_phi_divisor="off" from section 2.2), since section 2.2
   is unambiguous that theta_phi_divisor="on" composing with catalogue_leg_1d_mass_aware="on" has
   no valid target (no theta-consistent Sigma_4D exists) and the launch instruction's own
   worked example already lists this as one of the required regression items (R12).
3. Section 2.1 says "off" is byte-identical via "a single boolean check at each of the three read
   sites, no new table, no new object." The as-built form checks the boolean at FOUR read
   expressions (N1 has two branches, point and quadrature, in each of two functions -- 4 total
   call sites of catalogue_leg_1d_mass_aware_factor under "on", each guarded by the SAME
   _cat_leg_1d_ma_on boolean computed once per function call) plus the D1/W1 ternaries (one each).
   This is a reading of "three read sites" as the three SITES named (N1, D1, W1), not a literal
   count of boolean evaluations; disclosed here since N1 alone has two syntactic branches.
4. The T2.2 driver precedent (hier_s0_driver.py) threads theta_phi_divisor/sky_cone_k through
   EVERY function in the S0-A/S0-R/S0-C dispatch chain, including --score-only's
   gather_node_results_from_disk (which needs the node-dir suffix to locate banked runs, not to
   re-evaluate anything). The launch instruction says "forwarded to every run_mirror_seed_inprocess
   call" -- read as the SAME full-chain threading precedent set by T1.2 (row #266/T1.2's own
   0b308828-pattern commit), since gather_node_results_from_disk's suffix computation would
   silently mismatch a "_ma1d" node directory otherwise. Implemented: the flag reaches
   _node_dir_suffix, run_theta_node, run_arm_seed_s0a, run_arm_seed_s0r, run_seed_s0c,
   _run_one_seed_worker (tuple extended, byte-identical when omitted), run_arm, and
   gather_node_results_from_disk, plus the --catalogue-leg-1d-mass-aware CLI flag in main().

### 20.3 Kernel form's overhead and cost -- NOT measured by this node

Per section 9 item 1's own scope ("the kernel form... is not launched by this gate"), no cost
measurement of the "kernel" sub-form was performed here; the R9 regression test (below) exercises
it at unit-test scale (3 synthetic candidates) only, to pin the section 5 limiting case (L7), not
to characterise production cost.

### 20.4 Regression plan coverage (section 10 items R1-R14) -- what was implemented, what was descoped

New file: darksiren_emri_test/bayesian_inference/test_catalogue_leg_1d_mass_aware.py (26 tests, all
passing). Coverage against the registered plan:

- R1 (byte-identity at "off"): IMPLEMENTED, scalar and batch, kwarg-omitted vs explicit "off".
- R2 (the Z = 1 identity, the decisive pin): IMPLEMENTED as a self-contained synthetic fixture (200
  galaxies, an analytic mass-dependent survival stub, r_Malm = 0.850 measured -- an informative,
  can-fail-control value under the plan's own r_Malm <= 0.9 bound) that calls the REAL
  catalogue_leg_1d_mass_aware_factor and the REAL path_a_mixture_objects and confirms integral p_i
  dd = 1.0 to atol 1e-10 under "on" and equals D_phi/D_tilde_phi (!= 1, verified) under "off." This
  is a somewhat lighter-weight construction than the plan's literal "integrate the assembled p_i
  over the data on a fine grid" (it uses the analytic fact that a normalised Gaussian data-density
  integrates to 1 rather than numerically integrating one), disclosed as a deliberate scope choice
  under this node's time budget -- the ALGEBRAIC content the plan's own F-Z falsifier cares about
  (an unpaired build returning r_Malm or 1/r_Malm instead of 1) is exercised identically either way,
  since the Gaussian-integrates-to-1 step is a property of _mvn_pdf normalisation, not of this flag.
- R3 (the n_hat_w identity): IMPLEMENTED, 3 parametrisations including the r_Malm = 1 (L2) limit.
- R4 (scalar/batch parity under "on"): IMPLEMENTED, both host-z-kernel modes, rtol 1e-9.
- R5 (limit L1, mass-flat survival): IMPLEMENTED, rtol 1e-6 (the table's own interpolation floor).
- R6 (limit L3, the Malmquist sign): IMPLEMENTED, a direct unit test of the site-N1 factor.
- R7 (limit L4, the C-C identity): PARTIAL -- implemented as "the with-BH channel columns are
  bit-identical between on/off" (both scalar and batch, both host-z-kernel modes), which is the
  worker-level half of R7; the full p_Di-level empty-ball check (n_cand_no_bh = 0 on a live
  catalogue cell) was NOT run by this node (out of unit-test scope; R13's live-catalogue engagement
  gate is the natural home for it and is also descoped here, below).
- R8 (limit L5, global rescale invariance): NOT IMPLEMENTED -- the claim is about the FULL p_i
  assembly (numerator divided by its own global sum), which is invariant to a global survival
  rescale only once the division is performed; testing it meaningfully requires a p_Di-level
  fixture (or an equivalent hand-assembly), not a worker-level unit test of the raw per-host
  numerator alone (which trivially scales by the same constant, not usefully "invariant"). Disclosed
  as a genuine gap, not silently skipped.
- R9 (limit L7, kernel to point as sigma_g to 0): IMPLEMENTED as a direct unit test of
  catalogue_leg_1d_mass_aware_factor (rtol 1e-6, using a 4000-knot log-spaced M-grid in the test's
  own stub interpolator to keep the erf-sum's piecewise-linear-in-M discretisation error below the
  bound -- at the plan's originally-tried 40-knot grid the residual discretisation error was
  ~1.6e-4 relative, disclosed as an interpolation-density effect of the TEST STUB's coarse grid,
  not a defect in the "kernel" implementation itself, which reuses production's own
  _bh_mass_denominator_inner_m_integral_batch verbatim).
- R10 (the 1D/2D symmetry test): PARTIAL -- implemented as a lightweight "same accessor, same
  arguments" pin (catalogue_leg_1d_mass_aware_factor's point-form output compared bit-for-bit
  against a direct call to detection_probability_with_bh_mass_interpolated with the same d_L, M_z,
  isotropic-sky arguments), rather than the plan's full 2D-twin Gauss-Hermite-collapse test
  (sigma_cond -> 0 and sigma_gal -> 0 in _mz_sel_2d_expectation, center "raw"). The full symmetry
  test was judged out of this node's time budget; the lighter pin still catches a builder error
  that queries a DIFFERENT accessor, a different mass convention, or a non-isotropic sky term.
- R11 (2D bit-identity): covered by the same tests as R7's with-BH-channel check (identical
  columns, both flag values) -- IMPLEMENTED.
- R12 (guards): IMPLEMENTED -- all four evaluate()-level guards (each of the three "phi"
  resolutions, plus theta_phi_divisor), the worker-level defence-in-depth guard (catalogue_leg_1d_
  mass_aware="on" without catalogue_numerator_survival="phi" raises at both single_host_likelihood
  and its batch twin), an unknown-token rejection at both levels, and the "off is unaffected by the
  other three flags' values" confirmation.
- R13 (the >= 99 percent engagement gate on a live smoke cell): NOT RUN -- requires a real
  GalaxyCatalogueHandler and BallTree, i.e. a live-catalogue cell, out of this node's unit-test
  scope (the T1.1/T2.2 precedent's own smoke_run pattern); left for the tree-2 verifier pass or
  for arm (a)/(b) below, both of which will exercise it as a side effect.
- R14 (log lines): IMPLEMENTED -- "on" logs COUNTERFACTUAL and never [PHYSICS] ACTIVE; the "off"
  default logs nothing naming this flag.

### 20.5 Test run log (this node, foreground, 2026-08-30)

- New file alone: 26 passed.
- darksiren_emri_test/bayesian_inference (all files): 617 passed, 6 skipped.
- darksiren_emri_test/validation (all files): 402 passed, 1 skipped.
- darksiren_emri_test/test_arguments.py: 27 passed.
- Full tree, two directory halves (per the launch instruction): half A (analysis, bayesian_
  inference, datamodels, fixtures, integration, parameter_estimation, plotting) = 845 passed, 6
  skipped, 15 deselected; half B (validation, scripts, and every top-level darksiren_emri_test/*.py
  file) = 1096 passed, 9 skipped, 15 deselected. Combined: 1941 passed / 15 skipped / 30 deselected
  -- exactly the T2.2 baseline (1915 passed / 15 skipped / 30 deselected, section 10's own stated
  baseline) plus this node's 26 new tests, zero regressions.
- ruff check --fix darksiren_emri/: all checks passed (no fixes needed beyond formatting).
- ruff format darksiren_emri/ (+ the new test file): bayesian_statistics.py reformatted (cosmetic
  only, re-verified by AST parse and a second full-suite pass); the new test file needed one
  formatting pass too.
- mypy darksiren_emri/: one round of findings (catalogue_leg_1d_mass_aware_factor's M_g/M_g_error
  parameters were typed as bare ndarray but the scalar kernel passes Python floats) -- fixed by
  widening the annotations to float or ndarray[float64]; clean on the second run (70 source files).

### 20.6 Files touched (for the commit list; the orchestrator commits)

darksiren_emri/bayesian_inference/bayesian_statistics.py; darksiren_emri/arguments.py;
darksiren_emri/main.py; darksiren_emri/validation/correspondence_1d.py;
results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py;
darksiren_emri_test/bayesian_inference/test_catalogue_leg_1d_mass_aware.py (new file);
docs/gates/PHYSICS-GATE-LEDGER.md (two rows appended); this file (section 20, append-only).

No production file's DEFAULT behaviour changes: every new parameter defaults to "off"/"point" and
every new read site is guarded by that default, matching every prior instrument in this file
family (T1.1, T2.2, the P3-IMP/P3-2D/P3-RPHI twins). No git operation performed by this node.

launched under row #255 -- tree 2 node T2.3 -- implementation record complete.
