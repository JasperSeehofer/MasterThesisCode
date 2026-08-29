# PHYSICS-CHANGE GATE PRESENTATION — production-default flip of the with-BH catalogue-leg twin: catalogue_numerator_survival_2d "off" -> "mz_sel", catalogue_numerator_survival_2d_center "unset" -> "eff"

**Launched under rows #222/#223 — charter node B7.3.** [FABLE-B7.3 2026-08-29]

**Date:** 2026-08-29 · **Step:** PRESENTED (authored BEFORE code; this document writes no code) ·
**Trigger file:** darksiren_emri/bayesian_inference/bayesian_statistics.py (plus the non-trigger
plumbing files arguments.py, main.py, validation/correspondence_1d.py) · **Branch/HEAD at
authoring:** fix/p32d-classg-venue-repair @ ff230621 (git status clean under darksiren_emri/ and
darksiren_emri_test/ at authoring; every code line number below is against that tree, 2026-08-29) ·
**Form:** the 5-item package of .claude/rules/physics-validation.md ("Protocol — before writing
any code") plus the A5 item 6 validity conditions, mirroring the row-#195 1D twin gate
(docs/derivations/PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md -> [PHYSICS] bac48696, row #197)
and the row-#202 mass-filter default flip ([PHYSICS] cf4f8a2a: "default flip at 5 declaration
sites, read-site logic untouched, explicit old value = the counterfactual").

**Reviewable artifact this gate is built from:** fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md
(sections 1-9 the proposal, 13 the falsifier-(i) note and the registered arm, 15 the RESULT
RECORD). Measure-first production read: fanout1_20260829/B7_2_TWIN_CF_READOUT_RECORD.md §6
(ledger row #248). Falsifier (i): fanout1_20260829/B7_2_FALSIFIER_I_RECORD.md (row #236).

Every number carries {value, source file:line, date} (A11); §11 is the provenance table. No
backtick characters are used in this document; code identifiers are set in plain quotes.

---

## 0. Approval stamp, F2 statement, and the one design decision this gate makes

### 0.1 Approval stamp

**APPROVED column = "row #223 (standing grant, charter node B7.3)".** Row #223 (author ruling,
verbatim, BIAS_HISTORY_LEDGER.md:3020, 2026-08-29): "everything that is part of the tree can be
decided including production changes. It will be checked afterwards, we want to maximize the
scientific insights we can gather in this tree and then verify, plan the next tree and repeat."
Binding form quoted from the same row: "the 5-item presentation is still AUTHORED BEFORE CODE and
the three ledger rows are still filed — the gate's 'wait for author approval' step is
pre-authorized by this row for tree-scoped changes (cite 'row #223' in the APPROVED column), and
every such gate is in the end-of-fan-out verifier's mandatory scope." Row #222 (the [STANDING]
grant itself, BIAS_HISTORY_LEDGER.md:3018) had left production-default flips as an open question;
row #223 resolves it. The orchestrator's path decision under row #248 (BIAS_HISTORY_LEDGER.md,
"ORCHESTRATOR PATH DECISION (rows #222/#223 judgement, 2026-08-29)") opened this gate: "B7 -> 7.3
adoption gate OPENED: the physics-change presentation for the production default
catalogue_numerator_survival_2d = mz_sel with center = eff is authored before code;
implementation serialized behind the local runner; adoption batched into the wave-3 blind HEAD
readout (F2); it is the ONLY adoption candidate of wave 2."

Approval-scope convention (CLAUDE.md "Approval scope"): this gate is a **[RULE, tree-scoped]**
adoption pre-authorized by row #223; it returns in full to the end-of-fan-out verifier (§10) and,
through the verifier, to the author.

### 0.2 F2 statement (serialized adoption, one blind readout)

Per amendment F2 (charter, ratified row #222 with the F5 substitution) and row #223's last clause
("Adoption stays serialized (F2): batched, one blind HEAD readout, per-change arms"): the
production-default flip presented here is batched into the ONE wave-3 blind HEAD readout. Its H0
effect is read there, once, on both venues (iiib and joint_r1) at H_GRID_41, with a per-change arm
for THIS flag (HEAD default = adopted, versus HEAD with explicit "--catalogue_numerator_survival_2d
off" = the pre-adoption estimator). **No per-change attribution is taken from the readout's
composed delta** — the HEAD readout's registered structural blindness (MEASUREMENT_HEAD_READOUT_20260827.md
§4.2, "NOT LICENSED — registered structural blindness") is respected precisely because the
per-change arm is the only object that licenses attribution for this change. B7.2 (row #248) is the
measure-first production read that informs this gate; it is NOT the adoption's H0 verdict.

Sequencing of record: B6 [ALIGN] landed (1f003da6) and B5.1 landed (0b308828) before the wave-2
commit ff230621; B7.2 ran at ff230621 (jobs 6739000/6739001); this gate is authored after B7.2 read
out; the flip commit is authored after this presentation, cites row #223, is serialized behind the
live local runner (hier_s0_*/kwq1_* directories under fanout1_20260829/, §6.1 item 10), and rides
into the wave-3 batch.

### 0.3 Design decision made by this gate: a LITERAL default flip, not an "auto" token (G-3 resolved)

The proposal's §1.3/§8 G-3 wording proposed an "auto" resolution token (mirroring the 1D twin's
"auto" -> "phi" under absolute_marginal, bac48696). This gate presents the **literal** flip instead
— "off" -> "mz_sel" and "unset" -> "eff" at the declaration sites — exactly as the orchestrator's
row-#248 path decision phrases it and as the row-#202 mass-filter adoption (cf4f8a2a) did. Reasons,
each checkable:

1. The 1D twin needed "auto" because its object (the S-bar-phi table) exists only under
   normalization_mode="absolute_marginal" (bayesian_statistics.py:3672-3677). The 2D twin's object
   is the production with-BH survival accessor S_4D
   (simulation_detection_probability.py:2018, detection_probability_with_bh_mass_interpolated),
   which exists in every normalization mode: the existing test matrix exercises "mz_sel" under
   _MODE_CASES = ["generator_marginal", "volume_deconv", "absolute_marginal"]
   (test_catalogue_numerator_survival_2d.py:77, :364-366, :445-447, :468-470). There is no
   mode-dependent availability for an "auto" token to resolve on.
2. G-1 (composition guard) is ALREADY realized with zero new logic under a literal default: the
   kernel guards at bayesian_statistics.py:6376-6382 (scalar) and :7316-7322 (batch) RAISE when
   "mz_sel" is composed with a host_mass_kernel resolving to "trunc_lognormal" (resolve_host_mass_kernel,
   :240-261: under "auto" that is exactly normalization_mode="mass_trunc") or with
   catalogue_mass_overlap != "production". This is the proposal's RECOMMENDED G-1 behaviour
   ("must RAISE (require an explicit off)"). Consequence, disclosed: after the flip, a run in
   normalization_mode="mass_trunc", with host_mass_kernel="trunc_lognormal", or with
   catalogue_mass_overlap in {"neutralized", "inflated"} requires an explicit
   "--catalogue_numerator_survival_2d off". No sbatch under cluster/ and no driver under
   results/campaign51_20260728/realistic_20260729/*.py invokes any of those compositions (grep for
   "mass_trunc", "trunc_lognormal", "catalogue_mass_overlap neutralized|inflated": zero hits,
   2026-08-29). §6.1 item (a-iv) adds the same guard at the evaluate() layer so the refusal is
   early and explicit rather than a worker exception (the arguments.py:522-530 defense-in-depth
   pattern).
3. run_metadata then records the literal resolved value with no resolution step to audit — the
   PA-2D-6 lesson (PREREGISTRATION_P3_2D_20260825.md:238, "a STALE FLAG PIN found and fixed before
   the fleet"). At HEAD the CLI already stamps both flags:
   wave2_20260829/c0/run_metadata_21.json cli_args carries
   catalogue_numerator_survival_2d="off", catalogue_numerator_survival_2d_center="unset"
   (read 2026-08-29; main.py:359-369 _write_run_metadata writes arguments.to_dict(), arguments.py:123-133).
4. The C4 arm's CLI ("--catalogue_numerator_survival_2d mz_sel --catalogue_numerator_survival_2d_center eff",
   cluster/wave2_c4_twin_mz_sel.sbatch:143-144) becomes exactly the default. The wave-3 per-change
   arm is therefore the CoR-P CLI with the two flags OMITTED (adopted side) versus the same CLI with
   "--catalogue_numerator_survival_2d off" (pre-adoption side): a clean per-change arm with no token
   translation in between.

"unset" stays a legal CLI/evaluate() value for the center (choices unchanged), so the existing
refusal "mz_sel + unset" (arguments.py:522-530; bayesian_statistics.py:3706-3715, :6323-6330,
:7306-7313) remains as defense in depth — now reachable only by passing "unset" explicitly.

---

## 1. Old formula, new formula, byte-level consequences of the default flip (items 1 + 2)

### 1.1 Symbols (HEAD ff230621)

| symbol | meaning | code site |
|---|---|---|
| x | detector-frame mass fraction M_z / M_z,det,i (dimensionless coordinate of the with-BH mass marginal) | bayesian_statistics.py, the "Eq. (14.22)" comment block immediately above :6823 (scalar) and :7517 (batch) |
| mu_cond(z), sigma2_cond | GW conditional mean/variance of x given (phi, theta, d_L(z;h)/d_L,det) | mu_cond computed from _mu_obs_4d and _proj just above the mass-marginal branches (scalar) / :7455-7457 (batch) |
| mu_gal,g(z) = M_eff,g (1+z)/M_z,det ; sigma_gal,g(z) = sigma_M,g (1+z)/M_z,det | candidate g's mass prior in x, centred at the Eddington-shifted effective mass | mu_gal_frac / sigma_gal_frac lines immediately above :6823 and :7517; _host_M_eff assignment :6715-6720; eddington_shifted_host_mass :602 |
| mz_g(z;h) | analytic Gaussian-product overlap N(mu_cond; mu_gal, sigma2_cond + sigma2_gal) ("Eq. (14.31)") | mz_integral lines immediately above :6823 (scalar) and :7517 (batch) |
| S_4D(d_L, M_z) | production with-BH survival (pooled-2D grid, isotropic sky), clipped to [0, 1] | simulation_detection_probability.py:2018 (def), clip sites :758/:1008/:1050; _wbh_z_kwargs bayesian_statistics.py:1140 |
| k-bar_g(z) | volume-deconvolved host-z kernel (production host_z_kernel=volume_deconv), fixed_quad with _HOST_QUAD_N = 50 | bayesian_statistics.py:409 |
| g_sel,prod(z;h) | the FUSED 2D completion mass density (S_4D inside the completion mass quadrature; rows #117-#118) | completion_mass_factor_g_sel :2268 |
| Sigma^4D(h) | global with-BH catalogue selection sum, per-row POINT query S_4D(d_L(z_g;h), M_g(1+z_g)) | precompute_global_catalog_selection :2692 |
| alpha_G_phi = Sigma^4D / n-hat_w^phi ; r_Malm = Sigma^4D / Sigma^phi ; D-tilde_phi | Path-A with-BH class weight, Malmquist ratio, normaliser | :2487-2488 |
| E_GH[.] | Gauss-Hermite expectation of order _MASS_TRUNC_GH_ORDER = 24 (nodes/weights _MT_GH_NODES/_MT_GH_WEIGHTS) | :444, :468 |

### 1.2 OLD formula — the production default at HEAD (catalogue_numerator_survival_2d="off")

Per event i, per candidate g, the with-BH catalogue numerator carries NO survival factor inside the
candidate mass quadrature:

    N_g^wbh(h)      = INTEGRAL dz  k-bar_g(z) * gw_3D(z;h) * mz_g(z;h)
    mz_g(z;h)       = INTEGRAL dx  N(x; mu_cond(z), sigma2_cond) * N(x; mu_gal,g(z), sigma2_gal,g(z))
                    = N(mu_cond; mu_gal,g, sigma2_cond + sigma2_gal,g)            [analytic, Eq. 14.31]
    L_cat,wbh,i(h)  = weighted_ratio_of_sums(N_g^wbh, D_g, w_g) over the candidate set   [:5290-5296]
    combined_wbh,i  = (alpha_G_phi * L_cat,wbh,i + B_num,wbh,i) / D-tilde_phi             [:5772-5774]

Code path at the default: the scalar kernel's "else" (production Gaussian-product) branch computes
mz_integral and, because _cat_surv_2d_on is False (:6322), skips the block at :6823-6846; the
delta-kernel branch skips :6872-6896; the batch kernel (production dispatch through
_starmap_host_batches, :7627) skips :7517-7540. The Gray et al. (2020) Eq. (A.10) convention
"p_det is applied solely in the denominator" is the coded arrangement (:6743-6745 scalar comment,
:6851, :5304). Meanwhile the completion term B_num,wbh carries S_4D INSIDE its mass quadrature via
g_sel,prod (:2268; rows #117-#118) and the 1D catalogue leg carries S-bar-phi per candidate inside
its z-quadrature (row #197).

### 1.3 NEW formula — the production default after the flip (catalogue_numerator_survival_2d="mz_sel", center="eff")

The survival enters the innermost quadrature — per candidate, per z-node of the 50-node host-z
quadrature, per Gauss-Hermite node of the mass quadrature:

    mz_sel,g(z;h)   = INTEGRAL dx  N(x; mu_cond(z), sigma2_cond) * N(x; mu_gal,g(z), sigma2_gal,g(z)) * S_4D(d_L(z;h), x * M_z,det,i)
                    = mz_g(z;h) * E_{x ~ N(mu*_g(z), sigma*2_g(z))} [ S_4D(d_L(z;h), x * M_z,det,i) ]
    mu*_g           = (mu_cond * sigma2_gal,g + mu_gal,g * sigma2_cond) / (sigma2_cond + sigma2_gal,g)
    sigma*2_g       = sigma2_cond * sigma2_gal,g / (sigma2_cond + sigma2_gal,g)
    N_g^wbh,sel(h)  = INTEGRAL dz  k-bar_g(z) * gw_3D(z;h) * mz_sel,g(z;h)

with E[.] evaluated by Gauss-Hermite of order 24 (the same nodes the mass_trunc kernel uses) in
_mz_sel_2d_expectation (:6104-6186; docstring carries the completing-the-square identity) and its
host-batched twin _mz_sel_2d_expectation_batch (:6189). Placement in code: scalar quadrature branch
:6823-6846 (mz_integral = mz_integral * E[S_4D]); scalar delta branch :6872-6896 (same factor at
z = z_g); batch :7517-7540 (flows through BOTH the generator-point and quadrature reduces via
mz_integral). L_cat,wbh,i and combined_wbh,i are then formed by the UNCHANGED code at :5290-5296 and
:5772-5774.

**Centering definition ("eff").** The center parameter selects ONLY the mean of the galaxy Gaussian
fed to the survival expectation (the "_mu_gal_surv"/"mu_gal_surv" lines inside :6823-6846, :6872-6896,
:7517-7540); the overlap prefactor mz_g is ALWAYS computed with mu_gal = M_eff,g(1+z)/M_z,det and
sigma_gal is unchanged either way. "eff" builds mu* from the SAME mu_gal,g the prefactor uses, so
the code computes exactly the single integral above (completing the square is an identity only when
both factors share mu_gal). "raw" (mu* built from M_g while the prefactor stays at M_eff,g) is the
value of no integral of the form INTEGRAL N_cond p_gal S dx and stays available as an explicit
instrument only. Ruling of record for "eff": A20_REVIEW_P3_2D_DESIGN_20260825.md:17-19 (F2, "the
latent model wants _host_M_eff, and kernel identity is what makes W-tilde_2 <= 1 eventwise"), folded
as PA-2D-1; the CONFIRMED-at-33-seeds fleet ran with center "eff" (p3_2d_fleet.py:27;
p3_2d_companion.py:46). Numerically the two centers differ by (sigma2_cond/sigma2_sum) *
(mu_gal,eff - mu_gal,raw) <= ~1e-14 in x at the production operating point (sigma_cond p50 = 8.8e-8,
bayesian_statistics.py:2314-2317 as cited by the proposal §2.2, row #118/MAJOR-1, 2026-08-17): the
centering is a DEFINITIONAL choice, inert to double precision in production (proposal §2.2 item 2).

**Mechanically the flip changes NO arithmetic.** Every numeric path already exists at HEAD,
is tested (52 collected cases across test_catalogue_numerator_survival_2d.py and
test_survival_2d_homogeneity_falsifier.py, pytest --collect-only 2026-08-29) and was exercised on
production (C4, row #248). The change is which of two existing, tested branches the DEFAULT selects.

### 1.4 Byte-level consequences of the default flip

**Declaration sites that flip (the whole diff on the numeric side):**

| # | site | current | new |
|---|---|---|---|
| 1 | bayesian_statistics.py:3275 class attribute _catalogue_numerator_survival_2d | "off" | "mz_sel" |
| 2 | bayesian_statistics.py:3278 class attribute _catalogue_numerator_survival_2d_center | "unset" | "eff" |
| 3 | bayesian_statistics.py:3359-3360 __init__ instance defaults | "off" / "unset" | "mz_sel" / "eff" |
| 4 | bayesian_statistics.py:3497 evaluate() signature default | "off" | "mz_sel" |
| 5 | bayesian_statistics.py:3502 evaluate() signature default (center) | "unset" | "eff" |
| 6 | arguments.py:1062-1078 argparse "--catalogue_numerator_survival_2d" default (at :1065) | "off" | "mz_sel" |
| 7 | arguments.py:1079-1093 argparse "--catalogue_numerator_survival_2d_center" default (at :1082) | "unset" | "eff" |
| 8 | main.py:1418-1419 module-level evaluate() defaults | "off" / "unset" | "mz_sel" / "eff" |
| 9 | validation/correspondence_1d.py:2756-2757 run_mirror_seed_inprocess defaults | "off" / "unset" | "mz_sel" / "eff" |

Sites 1-3 and 4-5 are the six bayesian_statistics.py declaration sites; sites 6-9 are the
plumbing. Threading is unchanged: main.py:211-212 (CLI -> evaluate), main.py:1471-1472
(module evaluate -> BayesianStatistics.evaluate), correspondence_1d.py:2943-2944 (harness ->
evaluate), bayesian_statistics.py:5070-5071 and :5090-5091 (evaluate -> _starmap_host_batches for
the with-BH and without-BH host batches), :7710-7711 (_starmap -> single_host_likelihood_batch).

**Sites that do NOT flip (kernel read sites and worker signatures):** single_host_likelihood
signature defaults :6274 ("off") and :6280 ("unset"); single_host_likelihood_batch :7000/:7005;
_starmap_host_batches :7641-7642. The kernels are workers that ALWAYS receive the resolved value from
the dispatch (:5070-5071/:5090-5091); their own defaults are a test/instrument convenience and stay
"off"/"unset", exactly as the 1D twin's kernel default stayed "off" (:6265) when evaluate() flipped
to "auto" (bac48696). This keeps every kernel-level golden (test_kernel_parity.py goldens,
test_kernel_batch_equivalence.py, test_theta_hook.py pins, test_eddington_m_instrument.py,
test_c7_*.py, test_catalogue_mass_overlap.py, test_generator_marginal_mode.py's kernel calls)
bit-unchanged by construction.

**Which code paths change value at the default (2D channel only).** Measured on production (C4
arm at ff230621 vs the banked baseline d04d9dc9, all 4 H4 nodes, 1588 events each; recomputed
column-by-column 2026-08-29 from wave2_20260829/c4/simulations/diagnostics/event_likelihoods.csv vs
headreadout_20260827/iiib/event_likelihoods.csv):

| column | max_abs (4 nodes) | max_rel | status |
|---|---:|---:|---|
| L_cat_with_bh | 4.901328e-03 | 9.581687e-01 | CHANGES (the twin's object) |
| combined_with_bh | 3.522871e-04 | 1.694709e-01 | CHANGES (through :5772-5774) |
| num_log_term_with_bh (post-d40fe5c8 column, absent from the 16-column baseline header) | — | — | CHANGES (ln of the with-BH numerator term) |
| posteriors_with_bh_mass/h_0_*.json | — | — | CHANGES (per-event dict of combined_with_bh) |
| w_G, w_G_legacy, w_tilde_G, alpha_G_phi, r_Malm, D_tilde_phi | 0.0 | 0.0 | UNCHANGED (Sigma-chain untouched; Sigma^4D cancels algebraically in the catalogue term, A20 review F3 max rel dev 6.942e-8, 2026-08-25) |
| B_num, B_num_wbh, g_frac, L_comp | 0.0 | 0.0 | UNCHANGED (completion legs untouched) |
| L_cat_no_bh, combined_no_bh | 0.0 | 0.0 | UNCHANGED — R6 PASS, max_abs exactly 0.0 at every H4 node (b7_2_readout.json:gates.R6; row #248) |
| posteriors/h_0_*.json | — | — | UNCHANGED (1D channel) |

Per-event direction at h=0.730 (982 active events, 606 empty candidate sets): the ratio
L_cat_with_bh^T / L_cat_with_bh^B has min 0.066179, median 0.763745, max 0.999859, with 0 events
> 1 and 0 events == 1 (recomputed 2026-08-29 from the same two CSVs) — the eventwise bound E[S] <= 1
of §4 holds strictly on production wherever a candidate set is non-empty (R1 PASS, 0/6352
violations, row #248).

**The 1D channel is bit-identical by construction AND by measurement.** The without-BH numerator
never reads the flag: _cat_surv_2d_on is consumed only inside the evaluate_with_bh_mass branches
(:6823, :6872, :7517); the existing tests test_1d_channel_unaffected_scalar/_batch
(test_catalogue_numerator_survival_2d.py:468-490, 6 parametrised cases) pin it at the kernel; R6
pins it at production scale (max_abs 0.0).

**How the OLD behaviour stays reachable.** Explicit "off" is retained as the COUNTERFACTUAL:
"--catalogue_numerator_survival_2d off" on the CLI (choices unchanged, arguments.py:1064),
catalogue_numerator_survival_2d="off" on evaluate()/main.evaluate()/run_mirror_seed_inprocess. At
"off" the center value is never read (the kernels' _cat_surv_2d_on is False; evaluate() validates the
center only when "mz_sel", :3706-3715), so explicit "off" with the new center default "eff" is
byte-identical to today's "off"+"unset" (the existing kernel tests
test_default_off_omitted_kwarg_is_bit_identical_scalar/_batch, :157-212, and
test_off_matches_the_pre_flag_golden_across_modes, :215-233, keep pinning that at the kernel; §6.1
(b) adds the evaluate()-level pin). The evaluate() log block (:3716-3723) is re-pointed per G-2:
"[PHYSICS] catalogue_numerator_survival_2d=\"mz_sel\" (center=\"eff\") ACTIVE (row #<adoption>)" at
INFO for the resolved production value, and a "COUNTERFACTUAL: catalogue_numerator_survival_2d=\"off\"
— the pre-adoption WITH-BH catalogue numerator (no per-candidate survival factor). Not a production
posterior." WARNING for explicit "off" — exactly the :3679-3697 pattern of the 1D twin.

---

## 2. Reference / derivation (item 3)

1. **Proposal of record:** fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md §1.4 (derivation),
   §1.5 (the structural-asymmetry argument: degrees in S at HEAD — beta_G_phi 1, Sigma^phi 1 (ratio 0);
   N_g^wbh (coded) 0; B_num,wbh (fused g_sel,prod) 1; D-tilde_phi 1 — so the coded with-BH mixture is
   NOT homogeneous under S -> c*S while the twin is homogeneous of degree 0), §2 (centering ruling).
2. **Literature:** Mandel, Farr & Gair (2019) arXiv:1809.02063 Eqs. (5)-(7): the per-event
   likelihood of a latent-thresholded detection model carries the selection at the HYPOTHESIS (the
   candidate's own (z, M) posterior), not only in the population normaliser. Departure from
   Gray et al. (2020) arXiv:1908.06050 Eq. (A.10) ("p_det solely in the denominator") is deliberate
   and mirrors the 1D adoption's §3 (docs/derivations/PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md);
   the MFG-a verbatim check (docs/LITERATURE_WARNINGS.md) remains the Stage-L obligation before any
   paper-facing quotation — carried, not blocking (proposal §11 item 6).
3. **Derivation chain of record (rows #189-#212, BIAS_HISTORY_LEDGER.md:2830-2996; 2026-08-25..28):**
   stage 0 CLAIM_P3_2D_20260825.md §1 (row #189) derived the per-candidate object as exactly §1.3
   ("survival inside the candidate's own (Eddington-shifted) mass posterior quadrature, NOT point-S_4D
   and NOT S-bar-phi(z)"); PREREGISTRATION_P3_2D_20260825.md (+PA-2D-1..10) and
   PREREGISTRATION_P3_2D_REPAIR_20260827.md (v2, PA-2DR-1..15); the selection fusion gate
   docs/derivations/GATE_PRESENTATION_SELECTION_FUSION_20260817.md §1 (rows #117-#118) is the 2D
   completion analogue; the generator is latent-thresholded (A20/O4 review; O6 MECHANISM-CONFIRMED,
   row #158).
4. **Row #216 item 1 (author, "all approved", 2026-08-28):** "[P3-2D] repair CONFIRMED-at-33-seeds is
   the verdict of record (supersedes the row #212 24-seed UNDERPOWERED disposition ... capped
   supported)" — P3_2D_REPAIR_READOUT_20260828.md §7.
5. **Row #195 (author, verbatim, 2026-08-25): "all approved, the new finding is huge, lets see what the
   verification agent returns with."** -> row #197 "THE TWIN IS PRODUCTION PHYSICS" (bac48696): the 1D
   precedent whose §4 "S-bar -> c S-bar homogeneity" argument this gate's §1.5 argument generalises.
6. **This wave's C4 record (row #248; B7_2_TWIN_CF_READOUT_RECORD.md §6; proposal §15):** R1/R2/R6
   PASS; Delta-mean_h,pred = +0.0025057 (IMMATERIAL-PREDICTED); falsifier (i) PASS (row #236,
   B7_2_FALSIFIER_I_RECORD.md).

---

## 3. Dimensional analysis (item 4)

- S_4D in [0, 1] is dimensionless (simulation_detection_probability.py clips its survival outputs
  with np.clip(., 0.0, 1.0) at :758/:1008/:1050); E_GH[S_4D] is a weighted mean of dimensionless
  values (weights _MT_GH_WEIGHTS / sqrt(pi), :468; :6184-6185) -> dimensionless. Hence
  mz_sel = mz * E[S] keeps mz's measure: a density in the dimensionless x = M_z/M_z,det — the SAME
  measure as g_sel,prod (:2268), so the catalogue/completion addability of the fused 2D mixture
  (fusion gate (i)) is preserved.
- Query arguments: d_L(z;h) in Gpc (dist_vectorized, physical_relations.py:226) — the accessor's
  grid axis; M_z = x * M_z,det in M_sun, detector frame (a_nodes * det_M, :6178) — the accessor's
  second axis; z rider through _wbh_z_kwargs (:1140; inert while pdet_wbh_z_resolved=False, every run
  of record). Identical to the Sigma^4D / S-bar-phi / g_sel,prod query convention (:2043-2050).
- Degree bookkeeping in S: the with-BH mixture becomes homogeneous of degree 0 under S -> c*S
  (proposal §1.5); measured at unit-test scale: twin relative deviation 2.60e-16 (c=0.4) / 1.30e-16
  (c=0.15) versus coded 1.500 / 5.667 (B7_2_FALSIFIER_I_RECORD.md §2, 2026-08-29).
- No new table, no new constant, no unit conversion: the flip consumes the existing
  detection_probability_with_bh_mass_interpolated accessor only (:6170-6180).

---

## 4. Limiting cases (item 5)

| limit | result | status / evidence |
|---|---|---|
| S_4D == 1 (survival to 1) | E[S] = 1 -> mz_sel = mz -> the OLD code exactly (§1.2) | structural: the block at :6823-6846 multiplies by E[S]; the explicit "off" path skips it. Kernel byte-identity pins test_default_off_omitted_kwarg_is_bit_identical_{scalar,batch} (:157-212) |
| S_4D -> c (constant) | with-BH combined posterior INVARIANT under the twin (degree 0); NOT invariant under the coded arrangement | falsifier (i) PASS: twin 2.60e-16/1.30e-16 (gate <= 1e-10); coded 1.500/5.667; A15 double-applied-survival probe flagged at 0.600 (B7_2_FALSIFIER_I_RECORD.md §2; test_survival_2d_homogeneity_falsifier.py, 4 tests) |
| single candidate, sigma_z -> 0 (delta kernel) | gw_3D(z_g) * mz(z_g) * E[S](z_g) — the selected-prior single-host form | code path :6872-6896 (delta branch), same product-Gaussian factor evaluated at z = z_g |
| sigma_cond -> 0 (sharp GW mass; the production operating point, sigma_cond p50 = 8.8e-8) | mu* -> mu_cond, sigma* -> 0, E[S] -> S_4D(d_L, mu_cond * M_z,det) — the stage-0 §1 registered limit; both centers coincide | test_mz_sel_sharp_gw_mass_limit_matches_point_s4d (test_catalogue_numerator_survival_2d.py:384) passes at HEAD |
| sigma_gal -> 0 (mass-certain host) | mu* -> mu_gal; the Eddington shift M_eff - M_g proportional to sigma2_rel (:602) vanishes; E[S] -> S_4D(d_L(z;h), M_g(1+z)) — Sigma^4D's own per-row point query (:2692) | derivation (proposal §2.2 item 1); regression item R5 of §6.1 pins it |
| mass-information-free (sigma_cond -> infinity AND the host mass prior -> the population phi_x) | mz_sel -> INTEGRAL phi_x(x;z) S_4D(d_L, x M_z,det) dx = S-bar-phi(z;h) by the tower identity — the 1D twin's per-candidate factor; the same limit takes g_sel,prod -> S-bar-phi, so the 1D twin (row #197) is the mass-blind limit of BOTH fused 2D legs | derivation (proposal §4) |
| E[S] <= 1 eventwise | L_cat,wbh^twin <= L_cat,wbh^coded for every (event, h), equality only on empty candidate sets | test_mz_sel_moves_with_bh_numerator_by_a_survival_factor_in_0_1 (:364-382); production R1 PASS 0/6352 violations, 2424 empty-set equalities; ratio census at h=0.730 min 0.066179 / max 0.999859, 0 events > 1 (§1.4) |

**A5 item 6 — source-equation validity conditions.** MFG (2019) Eqs. (5)-(7) require a
latent-thresholded detection model (proven for the generator, row #158) and a survival evaluated at
the hypothesis; the pooled-2D S(d_L | M_z) grid is that object only under the isotropic-sky
decision (residual bounded 1.000202, fusion gate (ii-e)) and pdet_wbh_z_resolved=False (every run of
record); the FIX-3 joint grid would ride along through _wbh_z_kwargs unchanged.

---

## 5. The known unknown: the 2.25-2.35x residual (row #211) and what this adoption IS and IS NOT

**Stated plainly: this adoption is a STRUCTURAL-CONSISTENCY change.** The with-BH catalogue leg
receives the same survival treatment that the completion leg already carries (g_sel,prod, rows
#117-#118) and that the 1D catalogue leg already carries (S-bar-phi per candidate, row #197): the
three fused legs become the same construction ("the numerator's own integrand times S", proposal §2.2
item 3), and the with-BH mixture becomes S-degree-matched (§1.5; falsifier (i) PASS). It is justified
by (i) the derivation and the confirmed mechanism class, (ii) the CONFIRMED-at-33-seeds venue model
(row #216 item 1, capped "supported"), (iii) an IMMATERIAL production H0 delta (Delta-mean_h,pred =
+0.0025057 <= T_mat/2 = 0.004; row #248), and (iv) passing gates R1/R2/R6 and falsifier (i). **It is
NOT justified by a bias reduction, and none is claimed**: the correctness-over-bias-removal ruling
(2026-08-05; memory "author-values-correctness-over-bias-removal") is the operative principle — a
structural omission with a derivation, a confirmed mechanism class and an exact regression invariant
is corrected on its merits, and its H0 leverage is MEASURED (B7.2; wave-3), never presumed.

**The C2-star 2D identity has NOT closed.** Ladder of record (all {value, source, date}):

| stage | X = RHS2/LHS2 (bt / twin) | X (bc / coded) | source |
|---|---|---|---|
| sigma freeze, PA-2D-9 | 2.898 +/- 0.113 | 3.494 +/- 0.138 | PREREGISTRATION_P3_2D_20260825.md:309-311; p32d_residual_accounting_20260827.md §0 (2026-08-26/27) |
| after rung 1 (S-bar-phi double-application, reweight x1.1585) | 2.502 +/- 0.101 — "the x2.5" of rows #209-#211 | — | p32d_residual_accounting_20260827.md §1 (2026-08-27) |
| after rungs 2+3 (venue mass floor x1.1944; dead-row convention x1.0680), MEASURED end-to-end at 33 seeds | 2.253 +/- 0.082 (RHS2 0.01451300 +/- 0.00045293 / P1 0.00644266 +/- 0.00012212) | 2.700 +/- 0.101 (RHS2,coded 0.01507225 +/- 0.00046202 / P4 0.00558246 +/- 0.00012014) | P3_2D_REPAIR_READOUT_20260828.md §7 (2026-08-28), ratified row #216 item 1 |
| conditional on rung 1 (unimplemented) | 1.961 +/- 0.090 (registered v2.9 conditional prediction LHS2(bt) = 0.00740040 +/- 0.00024951) | 2.348 +/- 0.113 | PREREGISTRATION_P3_2D_REPAIR_20260827.md v2.9; p32d_residual_accounting_20260827.md §5 |

Exonerated for the residual (rows #207-#211): C2-star correct (blind re-derivation); the
completion-side mass axis, two constructions (X = 0.047 +/- 0.014 and X_alt = 0.9997 +/- 0.0003);
machinery at machine precision; the two-rung venue model CONFIRMED (R = 1 excluded 6.82 sigma at
33 seeds). NOT exonerated: the class-G draw-law contraction vs Sigma-tilde^4D and the identity's
acceptance-measure step (STUCK_P3_2D_SYMPTOM_CARD_20260826.md rungs 3/4). The residual is
common-mode across arms (X_bt/X_bc = 0.834 at 33 seeds; G4 arm-coherence 0.866484 in
[0.8613, 0.8675], P3_2D_REPAIR_READOUT_20260828.md §7) — the F8 coherence-clause signature of a
venue/identity-frame mechanism rather than of the twin law.

**Consequence, unchanged from the proposal §5:** the 1D adoption rested on a four-rung ladder
(derivation -> mechanism -> leverage -> CALIBRATION, row #186). This 2D adoption rests on
derivation, the structural-asymmetry argument, the confirmed venue model and the exercised
instrument — **without the calibration rung**. The epistemic status of the twin's calibration is
therefore "supported", capped, until either the identity closes on a repaired venue or the residual is
attributed to the venue side (falsifier (ii), §8). The row-#211 PARK is not reopened by this gate.
The attribution of the C4 IMMATERIAL read to the twin's S_4D-homogeneity property is PROVISIONAL
until falsifier (ii) returns (proposal §14 item 1; B7_2_TWIN_CF_READOUT_RECORD.md §6.6 item 1).

---

## 6. Regression plan for the implementer (builder != runner, standing rule 2)

### 6.1 (a) The flip, and every call site / test that must keep the OLD value explicitly

**(a-i) Flip the nine declaration sites of §1.4** (bayesian_statistics.py:3275, :3278, :3359-3360,
:3497, :3502; arguments.py:1065, :1082; main.py:1418-1419; correspondence_1d.py:2756-2757). Update
the adjacent comments/docstrings/help strings (arguments.py:358-376 properties, :1066-1077 and
:1083-1092 help; main.py:1414-1417 comment; correspondence_1d.py:2752-2755 comment;
bayesian_statistics.py:3271-3278, :3357-3360, :3491-3502 comments) from "off (default) is
byte-identical to the pre-flag path ... Never a production posterior" to the adopted wording
("mz_sel/eff (default) = the production with-BH catalogue numerator (row #<adoption>); explicit
off = the pre-adoption COUNTERFACTUAL") — the cf4f8a2a verified row found 6 stale comment blocks on
that adoption; do the sweep before the commit.

**(a-ii) Do NOT flip** single_host_likelihood :6274/:6280, single_host_likelihood_batch :7000/:7005,
_starmap_host_batches :7641-7642 (worker signatures; §1.4).

**(a-iii) Log block G-2** (bayesian_statistics.py:3716-3723): INFO "[PHYSICS] ... ACTIVE (row
#<adoption>)" when the resolved value is "mz_sel"; WARNING "COUNTERFACTUAL: ... off ..." when "off";
the "raw" center keeps a COUNTERFACTUAL warning (instrument only, §1.3). Pattern: :3679-3697.

**(a-iv) Early composition guard at the evaluate() layer** (defense in depth, mirrors
arguments.py:522-530 and the proposal's G-1 RAISE recommendation): in the evaluate() validation
block (:3701-3725) raise ValueError when catalogue_numerator_survival_2d == "mz_sel" and
(catalogue_mass_overlap != "production" or resolve_host_mass_kernel(host_mass_kernel,
normalization_mode, host_z_kernel) == "trunc_lognormal"), with the message naming the explicit
"off" escape. The kernel guards (:6376-6382, :7316-7322) stay as the second layer. Disclosed
consequence: mass_trunc / trunc_lognormal / neutralized / inflated runs now need explicit "off"
(§0.3 item 2). If the implementer judges the early guard out of the minimal diff, the kernel guards
alone still realize G-1 — say which was done in the implemented row.

**(a-v) Call sites that must keep "off" (or their registered value) EXPLICITLY** — these preserve
banked mirror/production byte-identity. Two classes:

Class A — ALREADY explicit at HEAD (verify unchanged; no edit needed):

| # | file:line | value passed | why it must stay explicit |
|---|---|---|---|
| A1 | results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py:169-170, :438-439, :453 (ARM_FLAGS_2D[arm]; CENTER "eff" at :27) | "off" for the coded arm (bc), "mz_sel" for the twin arm (bt) | the 33-seed repair fleet basis (row #216 item 1); regeneration must not follow the production default |
| A2 | results/campaign51_20260728/realistic_20260729/ca_rhs_scorer.py:1274-1275 -> :1293-1294 (threaded from ARRANGEMENT_FLAGS_2D at :1548-1549, :1925-1926, :1997-1998; center from the "--wbh_center" CLI :2471) | "off" (coded arrangement) / "mz_sel" (twin arrangement) | score2d banked basis (row #197 AMEND-1 pattern already pins the 1D cell "off" at :1282-1285) |
| A3 | results/campaign51_20260728/realistic_20260729/p3_wbhzero_measure.py:268-269 | "off" / "unset" | banked [P3-WBHZERO] production/mirror reads (rows #198-#202); its comment "also the flag's own default" becomes STALE — amend the comment on next use, never re-run banked |
| A4 | results/campaign51_20260728/realistic_20260729/p3_2d_companion.py:159-160 | "mz_sel" / CENTER ("eff") | computes the twin's own object by design |
| A5 | results/campaign51_20260728/realistic_20260729/gate_b_20260730/wbhzero_gate_b_scripts/wbhzero_probe.py:52-53 | FLAG / "eff" (explicit per-arm) | Gate-B probe basis |
| A6 | results/campaign51_20260728/realistic_20260729/p3_2d_forensic_20260826/rhs_inflation_confirmation.py:165-166, :173-174; rhs_inflation_alt_construction.py:188-189, :196-197 | "mz_sel" / "eff" | forensic re-scores of the twin arm (rows #207-#210) |
| A7 | cluster/wave2_c0_baseline.sbatch:146; cluster/wave2_c1_s0b_TEMPLATE.sbatch:162; cluster/wave2_c3_win_k3.sbatch:135 | "--catalogue_numerator_survival_2d off" | CoR-P arms of wave 2 (C0 gate, C1 S0-B registered form PA-HIER-31, C3 counterfactual) — the registered configuration of record is the pre-adoption estimator |
| A8 | cluster/wave2_c4_twin_mz_sel.sbatch:143-144 | "mz_sel" / "eff" | the C4 arm; after the flip these two lines equal the default (harmless; keep for A22 traceability) |

Class B — RELY ON THE DEFAULT at HEAD (must gain an explicit "catalogue_numerator_survival_2d=\"off\",
catalogue_numerator_survival_2d_center=\"unset\"" pin BEFORE any post-flip re-run whose byte-identity
with a banked artefact matters; the row-#197 AMEND-2 "fix on next use, never re-run banked" precedent
applies to the banked instruments):

| # | file:line | banked purpose | disposition |
|---|---|---|---|
| B1 | results/campaign51_20260728/realistic_20260729/p3_twin_test.py:211 | banked [P3-IMP] twin instrument (rows #159-#173) | AMEND on next use (pin "off"/"unset"); banked outputs stand; not re-run |
| B2 | results/campaign51_20260728/realistic_20260729/p3_b0_identity_test.py:945 | banked b0 identity instrument (rows #180-#186) | same as B1 |
| B3 | results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py:374, :386, :590 (DIAG_VALUE_COLUMNS = ("combined_no_bh", "combined_with_bh") at :171 — it READS the with-BH column) | LIVE local runner ([HIER] S0-A/S0-C, kwq1); registered CoR-P form is "off" (cf. A7, wave2_c1_s0b_TEMPLATE.sbatch:162) | NOT touched by this workflow (owned by another agent; a local runner is writing hier_s0_*/kwq1_*). The flip commit is SERIALIZED BEHIND the runner (row #248 path decision). Before any post-flip run the driver's owner must pin "off"/"unset" at the three sites; the end verifier checks this (§10 item 9) |
| B4 | darksiren_emri/validation/selfgen_control.py:1447, :1455 | C-SG control arms (rows #145-#157); registered reads are 1D-channel (csg_channel_scores on combined_no_bh) | 1D reads stay bit-identical (R6); the with-BH columns of a regenerated CSV would change — pin "off" if full-CSV byte-identity of a banked C-SG arm is ever regenerated; no edit in this gate |
| B5 | darksiren_emri/validation/correspondence_1d.py:3299 (G-1 calibration), :3372 (G-2), :3871 (arm runner) | harness-internal callers; registered reads are 1D (compute_seed_statistics :3144 = Sigma log combined_no_bh) | same as B4; the harness default deliberately mirrors production (PRODUCTION_FLAGS, :328-337); an ARM_*-registry pin for the 2D cell (the ARM_SELECTION_CELL :506 pattern) is an OPTIONAL follow-up, outside this gate |
| B6 | cluster/evaluate.sbatch, evaluate_closure_h065.sbatch, evaluate_closure_h065_finegrid.sbatch, evaluate_closure_h_true_finegrid.sbatch, evaluate_densecore.sbatch, evaluate_production_h0p73_dense.sbatch, evaluate_production_h0p73_superdense.sbatch (no 2D flag) | production evaluate scripts | post-flip they produce the ADOPTED estimator — intended. Any re-run meant to reproduce a banked pre-adoption artefact (e.g. the d04d9dc9 HEAD readout) must add "--catalogue_numerator_survival_2d off". The wave-3 blind readout's per-change baseline arm = CoR-P CLI + explicit off |

**(a-vi) Tests that pin the OLD default and must be RE-PINNED to an explicit old value (not
rewritten):**

| test | current reliance | required edit |
|---|---|---|
| test_catalogue_numerator_survival_2d.py::test_evaluate_mz_sel_with_unset_center_raises (:347-359) | passes only catalogue_numerator_survival_2d="mz_sel" and relies on evaluate()'s center default "unset" to trigger the refusal | add catalogue_numerator_survival_2d_center="unset" explicitly; assertion unchanged |
| test_catalogue_numerator_survival_2d.py::test_cli_validate_refuses_mz_sel_with_unset_center (:538-541) | passes only "--catalogue_numerator_survival_2d mz_sel", relies on the CLI center default "unset" | add "--catalogue_numerator_survival_2d_center unset"; assertion unchanged |

**Test that IS the default pin and is therefore REWRITTEN (not re-pinned):**
test_catalogue_numerator_survival_2d.py::test_cli_flag_defaults_to_off_and_unset (:504-509) ->
becomes the new default pin of §6.2 (rename to ..._defaults_to_mz_sel_and_eff) plus a sibling
test that explicit "--catalogue_numerator_survival_2d off --catalogue_numerator_survival_2d_center
unset" parses, validates and stamps "off"/"unset" in to_dict().

**Tests ALREADY explicit (no change; verify they stay green):**
test_catalogue_numerator_survival.py:486-487 and :581-582 (instance._catalogue_numerator_survival_2d
= "off", ..._center = "unset" — the p_Di harness); every kernel-level test in
test_catalogue_numerator_survival_2d.py (they pass mode/center explicitly or rely on the KERNEL
defaults, which do not change: :157-233 byte-identity/golden, :236-330 validation, :364-500
engagement/limits/parity/1D-unaffected/centering); all 4 tests in
test_survival_2d_homogeneity_falsifier.py (explicit modes); the kernel goldens
(test_kernel_parity.py golden/kernel_parity_pins.json, test_kernel_batch_equivalence.py,
test_theta_hook.py pins). Tests that build instances with object.__new__(BayesianStatistics) and
dispatch through MagicMock pools (test_absolute_marginal_mode.py, test_fixb_pathA_regression_pins.py,
test_partition_norm_restructure.py, test_catalogue_global_selection.py, test_b_num_analysis_depth_cap.py,
test_generator_marginal_mode.py, test_catalog_only_diagnostic.py, ...) now thread "mz_sel"/"eff"
from the class attribute into a CANNED pool — no numeric change; none asserts the flag positionally
(grep for the flag name and for "unset" across darksiren_emri_test/: only the three flag-test files
and the two explicit pins above, 2026-08-29).

### 6.2 (b) New pin tests

1. **evaluate()-level default pin (zero compute, MagicMock pool):** build the instance through
   BayesianStatistics() (so __init__ runs) and through evaluate() with defaults, capture the
   _starmap_host_batches call args (the test_catalogue_numerator_survival.py:527/:622 mock-pool
   pattern) and assert positions 13-14 of both host-batch calls carry ("mz_sel", "eff") at the
   default and ("off", "unset") under explicit off; assert the class attributes, __init__ values,
   evaluate() signature defaults, argparse defaults, main.evaluate defaults and
   run_mirror_seed_inprocess defaults ALL equal ("mz_sel", "eff") — one six-site consistency test
   (the cf4f8a2a "end-to-end default trace"). Add the inspect.signature test for
   run_mirror_seed_inprocess in the test_correspondence_1d.py:172-195 pattern.
2. **Kernel-level default-vs-explicit bit-identity on the stub fixture** (_HOSTS/_BASE_KW of
   test_catalogue_numerator_survival_2d.py:43-75, the same three synthetic hosts): the rows produced
   by the worker when it receives the resolved production pair ("mz_sel", "eff") are bit-identical
   to _scalar_rows("mz_sel", center="eff") / _batch_rows(...), and the rows produced under explicit
   ("off", "unset") are bit-identical to the pre-flip golden (test_off_matches_the_pre_flag_golden_across_modes
   already pins this; keep it and reference it from the new test).
3. **Production-scale pin (cluster; A22-stamped; rides the wave-3 per-change arm — cannot run
   locally: it needs the production CRB set md5 9a1f2a14384a9281c97ca3be312ddaab, the reduced
   catalogue md5 c52c13b5cab61f6b3f04bbe202550969 and the injection pool, cluster/datasets.yaml:246):**
   the HEAD default at h=0.730 on iiib reproduces the C4 arm's L_cat_with_bh, combined_with_bh and
   num_log_term_with_bh columns to <= 1e-12 relative (PROD-A0 form, row #201; observed floor <= 8.5e-15;
   C0 achieved exact 0.0, row #246), and the same commit with explicit "--catalogue_numerator_survival_2d
   off" reproduces the banked d04d9dc9 columns bit-identically (the C0 gate re-run). A small golden
   (the first 8 active events at h=0.730 from wave2_20260829/c4/simulations/diagnostics/event_likelihoods.csv,
   L_cat_with_bh and combined_with_bh) is banked under darksiren_emri_test/bayesian_inference/golden/
   for the cluster-side comparison script; it is NOT a local pytest (disclosed).
4. **Log-line tests (caplog):** default -> one INFO line containing "[PHYSICS] catalogue_numerator_survival_2d=\"mz_sel\"";
   explicit off -> one WARNING containing "COUNTERFACTUAL: catalogue_numerator_survival_2d=\"off\"".
5. **Early-guard test** for (a-iv) if implemented: evaluate() with default flags and
   normalization_mode="mass_trunc" raises naming the explicit "off" escape; with explicit "off" it
   proceeds.
6. **Regression items from the proposal §8 not yet closed:** R5 sigma_gal -> 0 limit equals the
   Sigma^4D point query at M_g(1+z) (§4 row 5) — NEW; R3 (homogeneity) is CLOSED by
   test_survival_2d_homogeneity_falsifier.py (row #236); R4 eventwise inequality is covered by
   test_mz_sel_moves_with_bh_numerator_by_a_survival_factor_in_0_1 (:364-382).

### 6.3 (c) run_metadata records the resolved value

Already true at HEAD (main.py:359-369 writes arguments.to_dict(), arguments.py:123-133; verified on
wave2_20260829/c0/run_metadata_21.json: cli_args.catalogue_numerator_survival_2d="off",
..._center="unset", git_commit ff2306213e9e65abbd474f66348bc05a6f3e6547). After the flip the stamp
reads "mz_sel"/"eff" with no resolution step (§0.3 item 3). The A22 stamp set for the wave-3 arms
must include both keys (proposal §8 R8). The mirror harness's per-seed meta JSON (the p3_2d_fleet.py:453
pattern) already stamps both keys.

### 6.4 (d) Suite counts

Baseline at HEAD ff230621: **1889 passed / 15 skipped / 27 deselected** (uv run pytest -m "not gpu
and not slow", 186.09 s, coverage 73.23%, run 2026-08-29 by this node; matches row #243's count).
Flag-test files: 52 collected (48 in test_catalogue_numerator_survival_2d.py + 4 in
test_survival_2d_homogeneity_falsifier.py; --collect-only 2026-08-29). Expected after
implementation: 1889 + the new tests of §6.2 (about 6-9; the rewritten default pin replaces one)
with ZERO numeric-expectation drift outside the flag's own tests; ruff, ruff-format and mypy clean
on all touched files. The implemented ledger row states the exact count.

---

## 7. A10 invariants and blindness

**Invariants held FIXED by the flip (last derivation-audit date in parentheses):**
normalization_mode=absolute_marginal · host_z_kernel=volume_deconv ·
selection_in_completion_numerator=fused (rows #117-#118, 2026-08-17) · catalogue_numerator_survival=phi
(row #197, 2026-08-25) · catalogue_global_selection=phi (rows #172-#178, 2026-08-23) ·
mass_filter_sigma=symmetric (row #202, cf4f8a2a, 2026-08-25) · mass_filter_geometry=linear, mass_filter_k=1.5
(0b308828, byte-identical) · theta=(0,1), theta_sites="all" (d40fe5c8/1f003da6/fb9d8aff, identity) ·
completion_b_scale=derived (2026-08-20) · eddington_m=on (G2d) · sigma4d_mass_kernel=point (Instrument J;
never derivation-audited as a design choice — disclosed) · catalogue_mass_overlap=production ·
pdet_wbh_z_resolved=False · the S_4D survival table object (NEVER independently re-derived — the
six-instrument common mode) · H_GRID_41 · CRB and catalogue md5 pins · EVAL_SEED=777000.

**Implementation-level invariants (checkable by diff):** no arithmetic line changes; the diff to
bayesian_statistics.py is confined to the declaration sites :3275/:3278/:3359-3360/:3497/:3502, the
validation/log block :3701-3725 (+ the optional early guard) and comments; the kernel bodies
:6231-6957 and :6958-7626 and _starmap_host_batches :7627-7723 are untouched; the without-BH
channel never reads the flag.

**Blindness sentence.** By construction the wave-3 readout of this adoption cannot detect (a) a
defect SHARED by the twin and the coded arrangement inside S-bar-phi / D-tilde-phi / beta_G_phi /
Sigma^phi — the Sigma-chain is identical in both arms (§1.4 table: max_abs 0.0 on alpha_G_phi,
r_Malm, D_tilde_phi) and a uniform rescaling of the S_4D table is exactly what the twin makes
invisible (§1.5 degree 0); (b) a mismatch between the point-mass per-row convention of the Sigma^4D
divisor (:2692) and the kernel-integrated numerator (Instrument J's axis, out of scope); (c) anything
in the 1D channel (bit-identical); (d) which SIDE — venue or estimator — carries the 2D identity
residual (§5: x1.96-2.35): no production arm has a truth anchor, so the readout can only bound the
H0 leverage, never adjudicate calibration; (e) the centering choice (inert to ~1e-14 at the
production sigma_cond).

---

## 8. A14 falsifier of the adoption

**Registered (this gate):** in the wave-3 blind HEAD readout, per-change arm for this flag (HEAD
default versus HEAD + explicit "--catalogue_numerator_survival_2d off"), both venues (iiib, joint_r1),
H_GRID_41, A22-stamped:

- **Primary (two-sided, band = the ratified T_mat):** the 2D (with-BH) posterior mean_h moves by
  |Delta-mean_h| < T_mat = 0.008 on BOTH venues (T_mat provenance: max(node spacing, sigma_h/3) at
  the row-#132 sigma_h, MEASUREMENT_HEAD_READOUT_20260827.md:268-285, ratified row #213 §10 item 4).
  **|Delta-mean_h| >= 0.008 on either venue FALSIFIES the IMMATERIAL-PREDICTED classification of row
  #248**; the adoption then returns to this gate as a MATERIAL change and, per proposal §6.1(iii)
  and the correctness-over-bias-removal ruling, a MATERIAL-DOWN outcome opens a mandatory stage-0 on
  the sign (it does not by itself refute the derivation), while a MATERIAL-UP outcome is reported as
  such with the same stage-0 obligation. Registered point prediction (REPORTED-ONLY, iiib):
  Delta-mean_h ≈ +0.0025 (first-order stencil, row #248; secondary 4-node cross-check +0.000192);
  predicted direction upward, toward truth (HEAD 2D offset -0.066653 iiib / -0.066987 joint_r1,
  MEASUREMENT_HEAD_READOUT_20260827.md §C.1). No joint_r1 number is predicted.
- **Instrument-defect falsifiers at full grid:** R1 (ln L_cat,wbh^adopted <= ln L_cat,wbh^off for every
  (event, h) on both venues, equality only on empty candidate sets) and R6 (L_cat_no_bh,
  combined_no_bh bit-identical between the two arms at every node, <= 1e-12 operational floor).
  Any violation => INSTRUMENT-DEFECT => the adoption is REVERTED pending diagnosis (the flag stays;
  the default returns to "off").
- **Falsifier (ii) stays registered and UNRUN:** the class-G fleet re-run with rung 1 repaired in
  the Option A-prime form (24-33 tasks, ~208-286 CPU-h; proposal §6.1(ii); tests the v2.9
  conditional prediction LHS2(bt) = 0.00740040 +/- 0.00024951 and the G4 band [0.8613, 0.8675]).
  Outcome map unchanged: inside both => attribution stays provisional-but-supported; LHS2 outside =>
  calibration status drops to derivation-only; G4 outside => adoption RETURNS to this gate as
  REFUTED-AS-CALIBRATED. Cited per the task/record convention as "row #220" (the row whose "registered
  falsifier stays unrun, attribution provisional" clause is the pattern being applied; row #220's own
  text is the WGEOM/CMEM ratification — the registration of record for falsifier (ii) is proposal
  §6.1(ii) + §14 item 1 and B7_2_TWIN_CF_READOUT_RECORD.md §6.6 item 1). It returns separately; it is
  not a precondition of the flip.

---

## 9. Ledger rows to file, and the [PHYSICS] commit message

### 9.1 docs/gates/PHYSICS-GATE-LEDGER.md

**Row 1 — presented (FILED with this document, 2026-08-29; commit ref ff230621; APPROVED column =
"row #223 (standing grant, charter node B7.3)"):** see the ledger's last row.

**Row 2 — implemented (to be filed by the implementer, "pre-commit", verdict PASS/FAIL):** target =
the nine declaration sites of §1.4 (+ :3701-3725 log block, + the early guard if added); note = the
literal flip; explicit "off" = the COUNTERFACTUAL; tests re-pinned per §6.1 (a-vi), new pins per
§6.2 items 1-2, 4-6; exact suite count; ruff/format/mypy clean; which of (a-iv) was done.

**Row 3 — verified (independent A20 clean-context verifier, "pre-commit" then updated to the short
SHA):** sign/units (all-dimensionless survival factor, §3); limits (§4 rows 1, 3, 7 by test; §4 row
5 by the new R5 test); diff scope exactly the declaration sites + log block (+ guard) + comments +
tests; kernel bodies untouched (git diff on :6231-7723 empty); with-BH leg the ONLY changed leg;
the six-site default trace; every Class-A pin of §6.1 (a-v) still explicit; Class-B item B3
(hier_s0_driver.py) pinned by its owner or the local runner confirmed finished before the commit;
COMMIT-READY.

### 9.2 BIAS_HISTORY_LEDGER.md (next free row — #249 at authoring; the orchestrator files it)

"## Row #249 — 2026-08-29 — Fan-out 1 wave 2, charter node B7.3 [2D-TWIN]: **/physics-change gate
PRESENTED (before code) for the production-default flip catalogue_numerator_survival_2d \"off\" ->
\"mz_sel\", catalogue_numerator_survival_2d_center \"unset\" -> \"eff\"** — APPROVED column \"row #223
(standing grant, charter node B7.3)\"; literal flip at nine declaration sites (six in
bayesian_statistics.py + arguments.py/main.py/correspondence_1d.py), kernel read sites and worker
signature defaults untouched, explicit \"off\" = the COUNTERFACTUAL; G-3 resolved to the literal
default (no \"auto\" token: the S_4D accessor exists in every normalization mode; G-1 RAISE already
realized by the kernel guards :6376-6382/:7316-7322, early evaluate()-layer guard specified). Stated
plainly as a STRUCTURAL-CONSISTENCY change (three fused legs same construction; with-BH mixture
S-degree-matched), justified by CONFIRMED-supported (row #216 item 1) + IMMATERIAL-PREDICTED
production delta (row #248: Delta-mean_h,pred +0.0025057 <= 0.004; R1/R2/R6 PASS; falsifier (i) PASS
row #236) — NOT by bias reduction; the C2-star 2D identity NOT closed (x2.253 +/- 0.082 bt / x2.700 +/-
0.101 bc at 33 seeds; conditional x1.961 +/- 0.090), calibration status supported-capped, row #211
PARK not reopened; attribution PROVISIONAL until falsifier (ii) (unrun, ~208-286 CPU-h). A14 falsifier
of the adoption registered: wave-3 per-change arm, |Delta-mean_h(2D)| >= T_mat = 0.008 on either venue
falsifies IMMATERIAL-PREDICTED; R1/R6 at full grid are INSTRUMENT-DEFECT falsifiers. F2: batched into
the one wave-3 blind HEAD readout, no per-change attribution from the composed delta. Serialized
behind the live local runner (hier_s0_driver.py:374/:386/:590 relies on the default and must be pinned
by its owner). Regression plan + the complete keep-\"off\" call-site list (8 explicit, 6 default-reliant)
+ the two tests to re-pin: PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §6. Suite baseline 1889 passed
/ 15 skipped / 27 deselected at ff230621. Gate-ledger 'presented' row filed. Implementation and the
'implemented'/'verified' rows follow in the flip commit; the whole gate is in the end-of-fan-out
verifier's mandatory scope (§10). Launched under rows #222/#223 — charter node B7.3."

### 9.3 The [PHYSICS] commit (subject line; body per the repo's commit convention)

    [PHYSICS] adopt the with-BH catalogue-leg twin in production (row #223 standing grant, charter node B7.3; rows #189-#212/#216 evidence chain; B7.2 C4 read IMMATERIAL-PREDICTED +0.0025, R1/R2/R6 PASS, row #248) — catalogue_numerator_survival_2d default 'off'->'mz_sel', center 'unset'->'eff' at nine declaration sites, kernel read-site logic untouched, explicit 'off' = the counterfactual; STRUCTURAL-CONSISTENCY change, no bias claim; suite <N> green; independently verified COMMIT-READY; batched into the wave-3 blind HEAD readout (F2)

Body: cite this file, the proposal §1-§9/§13/§15, B7_2_TWIN_CF_READOUT_RECORD.md §6,
B7_2_FALSIFIER_I_RECORD.md, the three gate-ledger rows, and the BIAS_HISTORY_LEDGER rows #248/#249;
end with the Co-Authored-By and Claude-Session trailers of the repo convention. The commit must
include the gate-ledger 'implemented' and 'verified' rows and the test edits of §6.1 (a-vi)/§6.2.

---

## 10. What the end-of-fan-out verifier must check (mandatory scope, row #223)

1. **Authorization form:** APPROVED column cites "row #223 (standing grant, charter node B7.3)";
   this presentation predates every code line of the flip (git log order: this file's commit before
   the [PHYSICS] commit); three ledger rows exist for the same target.
2. **Diff scope:** the [PHYSICS] commit's diff to bayesian_statistics.py touches ONLY the
   declaration sites (:3275, :3278, :3359-3360, :3497, :3502), the validation/log block (:3701-3725,
   plus the early guard if added) and comments; git diff over :6231-7723 (both kernels + _starmap) is
   EMPTY; worker signature defaults :6274/:6280/:7000/:7005/:7641-7642 still read "off"/"unset".
3. **Six-site default trace:** class attribute, __init__, evaluate() signature, argparse defaults,
   main.evaluate defaults, run_mirror_seed_inprocess defaults all ("mz_sel", "eff"); CLI to_dict()
   stamps them; run_metadata of any post-flip run carries them.
4. **Counterfactual reachability:** explicit "off" (+ any center) is byte-identical to the pre-flip
   default at the kernel (existing tests :157-233 green) and at the evaluate() layer (§6.2 item 1);
   the "mz_sel"+"unset" refusal still fires when "unset" is passed explicitly (re-pinned tests of
   §6.1 (a-vi)).
5. **Numbers re-derived, not trusted:** re-run b7_2_readout.py on the retrieved C4 CSV (R1 0/6352,
   R2 982/982, R6 max_abs 0.0, Delta-mean_h,pred +0.0025057); re-run the falsifier (i) tests;
   recompute the §1.4 per-column table (all Sigma-chain/completion/1D columns max_abs 0.0; only
   L_cat_with_bh and combined_with_bh move; ratio census 0 events > 1).
6. **The residual disclosure is intact:** §5's ladder numbers match P3_2D_REPAIR_READOUT_20260828.md
   §7 and p32d_residual_accounting_20260827.md §1/§5; no text anywhere in the flip commit claims a
   bias reduction or a closed identity; the row-#211 PARK is unchanged.
7. **Exoneration re-check (standing rule 5):** the MECHANISM grep of proposal §10 (items 6 and 17 of
   the DO-NOT-RE-TRY list delimited, not covering; [PDET-IO] delimited) still holds against the
   EXONERATION_REGISTER at commit time; no new entry was added between ff230621 and the flip commit
   that covers "survival inside the with-BH catalogue numerator's mass quadrature, paired with the
   fused completion leg".
8. **Call-site pins:** every Class-A site of §6.1 (a-v) still passes its explicit value; the
   p3_wbhzero_measure.py:268 comment "also the flag's own default" has been marked STALE or amended
   on next use; the wave-3 sbatch for the per-change baseline arm carries "--catalogue_numerator_survival_2d
   off" and the adopted arm omits both flags.
9. **Serialization behind the local runner:** the flip commit's timestamp postdates the completion
   of the hier_s0_*/kwq1_* local runs that were in flight on 2026-08-29, OR hier_s0_driver.py:374/:386/:590
   were pinned to "off"/"unset" by the driver's owner before any post-flip run; any hier_s0 output
   produced after the flip without the pin is flagged as running the ADOPTED estimator, not CoR-P.
10. **F2 respected:** the wave-3 blind HEAD readout includes a per-change arm for this flag on both
    venues at H_GRID_41; the adoption's H0 statement is taken from that arm only; the composed HEAD
    delta is not attributed to this change; §8's falsifier is evaluated with T_mat = 0.008 exactly as
    registered here (no band tightening after the data).
11. **Suite:** the implemented row's count = 1889 + new tests, 0 failures, ruff/format/mypy clean;
    zero numeric-expectation drift outside the flag's own tests.
12. **Row #223 boundary:** the flip is inside the charter tree (B7 branch, depth 3); nothing in the
    commit changes Sigma^4D's per-row convention (Instrument J), the eligibility window (B5's axis),
    or the 1D channel — the proposal's "Not proposed" list (§11) holds.

---

## 11. Provenance table (A11)

| quantity | value | source (file:line) | date |
|---|---|---|---|
| HEAD at authoring | ff2306213e9e65abbd474f66348bc05a6f3e6547 | git rev-parse HEAD; git status clean under darksiren_emri/ | 2026-08-29 |
| class-attribute defaults | "off" / "unset" | bayesian_statistics.py:3275, :3278 | HEAD |
| __init__ defaults | "off" / "unset" | bayesian_statistics.py:3359-3360 | HEAD |
| evaluate() signature defaults | "off" / "unset" | bayesian_statistics.py:3497, :3502 (def evaluate :3380) | HEAD |
| evaluate() 2D validation/log block | :3701-3725 | bayesian_statistics.py | HEAD |
| 1D twin resolution + log pattern | :3658-3699 ("auto" -> "phi" under absolute_marginal) | bayesian_statistics.py | HEAD (bac48696 origin) |
| dispatch threading | :5070-5071 (with-BH batch), :5090-5091 (without-BH batch) | bayesian_statistics.py | HEAD |
| L_cat,wbh formation; combined_wbh | :5290-5296; :5772-5774 | bayesian_statistics.py | HEAD |
| scalar kernel def / flag / guard / use sites | :6231 / :6322 / :6376-6382 / :6823-6846, :6872-6896 | bayesian_statistics.py | HEAD |
| batch kernel def / flag / guard / use site | :6958 / :7305 / :7316-7322 / :7517-7540 | bayesian_statistics.py | HEAD |
| worker signature defaults (unchanged) | :6274, :6280, :7000, :7005, :7641-7642; passthrough :7710-7711 | bayesian_statistics.py | HEAD |
| _mz_sel_2d_expectation / batch twin | :6104-6186 / :6189 | bayesian_statistics.py | HEAD |
| _HOST_QUAD_N / _MASS_TRUNC_GH_ORDER / GH nodes | 50 (:409) / 24 (:444) / :468 | bayesian_statistics.py | HEAD |
| eddington_shifted_host_mass; completion_mass_factor_g_sel; precompute_global_catalog_selection; alpha_G_phi/r_Malm | :602; :2268; :2692; :2487-2488 | bayesian_statistics.py | HEAD |
| resolve_host_mass_kernel ("auto" -> trunc_lognormal iff mass_trunc) | :240-261 | bayesian_statistics.py | HEAD |
| Gray (2020) A.10 convention comments | :6743-6745, :6851, :5304 | bayesian_statistics.py | HEAD |
| S_4D accessor def; clip to [0,1] | :2018; :758/:1008/:1050 | simulation_detection_probability.py | HEAD |
| dist_vectorized | :226 | physical_relations.py | HEAD |
| CLI properties / validate refusal / argparse | :358-376 / :522-530 / :1062-1078 (default :1065), :1079-1093 (default :1082) | arguments.py | HEAD |
| to_dict (run_metadata source) | :123-133; main.py:359-369 | arguments.py; main.py | HEAD |
| main.py sites | :211-212, :1418-1419, :1471-1472 | main.py | HEAD |
| correspondence_1d.py sites | :2756-2757 (defaults), :2943-2944 (passthrough), :1549 (mirror guard comment), :328-337 (PRODUCTION_FLAGS), :3144 (compute_seed_statistics 1D), :3299/:3372/:3871 (internal callers), :506 (ARM_SELECTION_CELL) | validation/correspondence_1d.py | HEAD |
| selfgen_control callers | :1447, :1455 | validation/selfgen_control.py | HEAD |
| flag-test file structure | _HOSTS :43, _BASE_KW :70-74, _CENTER_CASES :76, _MODE_CASES :77; default-pin tests :504-509, :538-541; evaluate-level refusal :347-359; byte-identity :157-233; 1D-unaffected :468-490; sharp-GW limit :384 | test_catalogue_numerator_survival_2d.py | HEAD |
| explicit pins in the 1D test harness | :486-487, :581-582 | test_catalogue_numerator_survival.py | HEAD |
| signature-default test pattern | :172-195 | darksiren_emri_test/validation/test_correspondence_1d.py | HEAD |
| collected flag tests | 52 (48 + 4) | pytest --collect-only | 2026-08-29 |
| full fast suite | 1889 passed / 15 skipped / 27 deselected, 186.09 s, coverage 73.23% | uv run pytest -m "not gpu and not slow" | 2026-08-29 |
| falsifier (i) | twin 2.60e-16 / 1.30e-16; coded 1.500 / 5.667; probe 0.600 | B7_2_FALSIFIER_I_RECORD.md §2; row #236 | 2026-08-29 |
| C4 gates | R1 0/6352 (2424 empty-set equalities); R2 982/982 = 1.0; R6 max_abs 0.0 all nodes | b7_2_readout.json:gates; row #248 | 2026-08-29 |
| C4 stencil | Delta-ell(0.660/0.665/0.670) = -3.030674 / -2.993148 / -2.956381; Delta-ell' = +7.429355; Delta-ell'' = -30.311364; I_HEAD = 2965; Delta-mean_h,pred = +0.0025057 | b7_2_readout.json:stencil | 2026-08-29 |
| C4 secondary | 4-node Delta-mean +0.000192, Delta-MAP 0.0; sign census h=0.730: 0 positive / 872 negative / 110 ≈ 0 of 982 | b7_2_readout.json; B7_2_TWIN_CF_READOUT_RECORD.md §6.4 | 2026-08-29 |
| C4 per-column deltas vs baseline (this node's recompute) | L_cat_with_bh max_abs 4.901328e-03 / max_rel 0.958; combined_with_bh 3.522871e-04 / 0.169; all other shared columns 0.0; ratio census min 0.066179 / median 0.763745 / max 0.999859, 0 events > 1 | wave2_20260829/c4/simulations/diagnostics/event_likelihoods.csv vs headreadout_20260827/iiib/event_likelihoods.csv | 2026-08-29 |
| C4 run | jobs 6739000 (task 0, h=0.730, 00:06:25) + 6739001 (tasks 1-3: 00:06:38/00:06:17/00:06:10); overhead ≈ 0.99x; 6.8 CPU-h | B7_2_TWIN_CF_READOUT_RECORD.md §1/§6; row #248 | 2026-08-29 |
| C0 baseline gate | PASS bit-identical, max_abs 0.000 on 14 shared numeric columns at h=0.73; 1.7 CPU-h | row #246; REGISTRATION_C0_BASELINE_GATE_20260829.md §13 | 2026-08-29 |
| run_metadata stamp check | cli_args catalogue_numerator_survival_2d="off", _center="unset"; git_commit ff230621 | wave2_20260829/c0/run_metadata_21.json | 2026-08-29 |
| T_mat / T_mat/2 | 0.008 / 0.004 | MEASUREMENT_HEAD_READOUT_20260827.md:268-285; ratified row #213 §10 item 4 | 2026-08-28 |
| HEAD 2D (iiib / joint_r1): mean_h, offset, sigma_h, MAP | 0.663347, -0.066653, 0.018366, 0.665 / 0.663013, -0.066987, 0.018637, 0.660 | MEASUREMENT_HEAD_READOUT_20260827.md §C.1 | 2026-08-28 |
| sigma_cond p50 (production) | 8.8e-8 | bayesian_statistics.py:2314-2317 as cited in proposal §2.2 (row #118/MAJOR-1) | 2026-08-17 |
| centering ruling | F2, "eff" for the Gaussian branch | A20_REVIEW_P3_2D_DESIGN_20260825.md:17-19 | 2026-08-25 |
| fleet center used | "eff" | p3_2d_fleet.py:27; p3_2d_companion.py:46 | 2026-08-25 |
| residual ladder | 2.898 +/- 0.113 / 3.494 +/- 0.138; 2.502 +/- 0.101; 2.253 +/- 0.082 / 2.700 +/- 0.101; 1.961 +/- 0.090 / 2.348 +/- 0.113 | PREREGISTRATION_P3_2D_20260825.md:309-311; p32d_residual_accounting_20260827.md §0/§1/§5; P3_2D_REPAIR_READOUT_20260828.md §7 | 2026-08-26..28 |
| repair fleet P1/P4, G4, R=1 exclusion | 0.00644266 +/- 0.00012212 / 0.00558246 +/- 0.00012014; 0.866484 in [0.8613, 0.8675]; 6.82 sigma | P3_2D_REPAIR_READOUT_20260828.md §7; row #216 | 2026-08-28 |
| falsifier (ii) cost | ~8.67 CPU-h/task x 24-33 tasks ≈ 208-286 CPU-h | proposal §6.1(ii) | 2026-08-29 |
| 1D twin precedent | bac48696: evaluate() "off" -> "auto" (:3398 at the time), kernel default stayed "off"; correspondence_1d.py:2700; AMEND-1 ca_rhs_scorer.py; suite 1821 | git show bac48696 --stat; gate-ledger rows 2026-08-25 | 2026-08-25 |
| mass-filter precedent | cf4f8a2a: default flip at 5 declaration sites, read-site logic untouched, explicit "asymmetric" = counterfactual; suite 1827 | gate-ledger rows 2026-08-25; git log | 2026-08-25 |
| authorization | row #222 (BIAS_HISTORY_LEDGER.md:3018), row #223 (:3020), row #248 path decision | BIAS_HISTORY_LEDGER.md | 2026-08-29 |
| workspace expiry | 2026-09-23 | COMPUTE_LEDGER.md; row #245 | 2026-08-29 |

---

## 12. Exoneration check (standing rule 5) — no new mechanism

The mechanism of this gate is identical to the proposal's §10 grep ("survival / p_det factor
inside the with-BH (2D) catalogue numerator, per candidate, inside the mass quadrature; paired with
the fused completion leg"). Conclusion carried unchanged: not exonerated; DO-NOT-RE-TRY items 6
("adding p_det inside the numerator ALONE — refuted (#66); only the joint pair works (#67)") and 17
("numerator-only normalization cleans") are adjacent and delimited — "mz_sel" is the JOINT
arrangement (fused completion + twin catalogue; A22 stamp selection_in_completion_numerator="fused"
in every arm, PA-2D-1 F7) and the denominator chain is untouched by design. No register entry was
added between a794404c (proposal authoring) and ff230621 that covers this mechanism (the fan-out
records added rows #224-#248, none an exoneration). The verifier re-runs the grep at commit time
(§10 item 7).

---

*Builder/runner independence (standing rule 2): this gate's author built no instrument and ran no
registered measurement; the only executions were the fast test suite for the baseline count, a
pytest --collect-only, and a zero-compute per-column recompute of the already-banked C4/baseline
CSVs (§1.4). Nothing here addresses the author directly; every item returns to the orchestrator
and, per row #223, to the end-of-fan-out verifier.*

**Stamp:** launched under rows #222/#223 — charter node B7.3; PRESENTED 2026-08-29, before code.
