---
status: gaps_found
reviewer: gpd-check-proof
manuscript_path: paper_a/main.tex
manuscript_sha256: af60a4304bfca7e325a5934f291509e93439a954df61afe0c76f78094d4dc7e1
round: 1
claim_ids:
  - CLM-006
  - CLM-007
  - CLM-008
  - CLM-013
  - CLM-015
  - CLM-019
  - CLM-020
proof_artifact_paths:
  - paper_a/sections/appendix_volume_deconv.tex
  - paper_a/sections/appendix_sky_marginal.tex
  - paper_a/sections/appendix_eddington_m.tex
  - paper_a/sections/appendix_gray_mapping.tex
  - paper_a/sections/appendix_beta_g.tex
  - paper_a/sections/estimators.tex
  - paper_a/sections/abstract.tex
  - paper_a/sections/framework.tex
  - paper_a/sections/introduction.tex
  - paper_a/sections/conclusions.tex
  - paper_a/sections/codes.tex
  - paper_a/sections/coverage.tex
  - paper_a/sections/realdata.tex
  - paper_a/sections/pitfall.tex
  - paper_a/sections/postmortem.tex
  - paper_a/main.tex
missing_parameter_symbols: []
missing_hypothesis_ids:
  - "CLM-015-H-mass-domain: restriction of the Z_M integral / R_eff support to physical masses (implementation clamps the moment quadrature at M >= max(M_g - 5 sigma_M, 1e3 Msun), bayesian_statistics.py:99, re-verified this run; the proof artifact writes Z_M as an unrestricted integral and never states the clamp)"
  - "CLM-020-H-completeness-h-dependence: whether the smooth completeness f inherits the h-dependence of Gray's magnitude threshold z(M, m_th, H0); manuscript writes f(z, Omega) in framework.tex:214 / appendix_gray_mapping.tex but f(z, Omega, h) in appendix_beta_g.tex:98 (re-verified this run), and the rewriting hypothesis does not pin this down"
  - "CLM-013-H-validation-configuration: the 'silent under spectroscopic-quality validation' property of the full defect superposition requires the unstated hypothesis that the validation configuration suppresses the completion channel (complete catalogue f -> 1 or equivalent); only Defect 1 (bare-kernel Eddington bias) is sigma_z^2-suppressed, while Defect 2 (peak-density over-weight 2/(sigma_phi sigma_theta)) and Defect 3 (the -17.2 per cent discrete-sum tilt, measured from p_det point-evaluated at catalogue redshifts, i.e. already the sigma_z = 0 limit of the denominator) are zeroth order in sigma_z; introduction.tex attributes second-order sigma_z suppression to all three defects"
coverage_gaps:
  - "CLM-008 conclusion clause 'valid to relative error <~ 1e-5': not implied by the appendix's own error budget, which contains a 'sub-percent, h-smooth drift' from the u-dependence of the conditional sky mean (rho_theta-u term) with no bound on |rho_theta-u|; the source derivation (docs/derivations/G2a §4 item (c), VERDICT) states the bound as '<~ 1e-5 (+ sub-percent h-smooth correlation drift)' and the manuscript dropped the qualifier (appendix_sky_marginal.tex:95-99, re-verified this run)"
  - "CLM-015 domain obligation: the described quadrature 'over +/- 5 sigma_M' omits the implemented low-mass domain clamp lo = max(M_g - 5 sigma_M, 1e3) (bayesian_statistics.py:99), active for essentially every analysed galaxy at median sigma_rel = 0.98 (5-sigma lower edge ~ -3.9 M_g; ~15 per cent of the Gaussian mass sits at unphysical M < 0); the 'exact first moment of eq. (eq:eddm:prior)' is exact only on the clamped domain, and unlike the z-channel's disclosed z >= 0 clamp this restriction is nowhere stated or bounded"
  - "CLM-019 conclusion clause 'the de-railing it produces on GLADE+ cannot be manufactured by the fix itself': asserted as a consequence of the h^-3 cancellation alone; the cancellation excludes only an explicit h-dependent normalization tilt, not an h-neutral but wrong-shaped kernel moving the peak -- the latter is excluded by the Section 5 coverage calibration, which the 'so' in estimators.tex does not invoke"
  - "CLM-006 conclusion clause 'two prior-consistent estimator repairs': the blanket label holds for local_ratio only under requirement (i) of the manuscript's own two-part definition of prior consistency (framework.tex sec. 2.6); requirement (ii) -- population measure counted exactly once in every redshift integral -- is violated by local_ratio's retained bare kernel, as estimators.tex sec. 4.5 itself states ('retains the bare redshift kernel and with it the -K sigma_z^2 law') and Table 1 quantifies (0-3 per cent coverage for that kernel); the abstract's phrasing 'numerator--denominator prior consistency' is correctly scoped, the registry text and section title are not"
  - "CLM-006 supporting lemma 'every per-galaxy constant --- number-density calibration, rate normalization, completeness weighting --- cancels row by row' (estimators.tex sec. 4.1): false for genuinely per-galaxy factors in a ratio of sums (two-galaxy counterexample: (w1 c1 N1 + w2 c2 N2)/(w1 c1 D1 + w2 c2 D2) depends on c_g when c1 != c2); only ball-common constants cancel, which is exactly how appendix_beta_g.tex states it ('ball-common factors cancel') and what appendix_volume_deconv.tex concedes for Z_g ('cancels in a single galaxy's ratio but deliberately not in the ratio of sums'); the estimators.tex list including per-galaxy completeness weighting contradicts both"
  - "CLM-013 quantifier 'every independently implemented dark-siren pipeline' (introduction.tex:100-104): contradicted as a literal universal by the manuscript's own pinned-version audit (codes.tex: the rail is 'structurally excluded' from gwcosmo in both generations and 'absent' from CHIMERA, icarogw, and DarkSirensStat); the claim survives only in the ex-ante reading over future/unaudited implementations, which is the scoped version conclusions.tex and codes.tex sec. 8.3 actually state"
  - "CLM-013 conclusion clause 'silent rail ... while calibrating perfectly on spectroscopic-quality validation inputs': no in-paper demonstration that the full pre-fix estimator (peak-density sky factor + global denominator) calibrates on spectroscopic-quality inputs; the sigma_z -> 0 silence is measured only for the kernel defect in the clean single-host harness without completion machinery (bias -0.0016 at sigma_z = 0.005, floor-dominated), and the introduction's mechanism sentence extends sigma_z^2 suppression to all three defects when it holds only for Defect 1 (see missing hypothesis CLM-013-H-validation-configuration)"
scope_status: narrower_than_claim
quantifier_status: narrowed
counterexample_status: narrowed_claim
---

# Proof Redteam

Round-1 adversarial proof audit (re-run) of the SEVEN theorem-bearing claims of `paper_a/main.tex` (sha256 re-verified on disk this run: `af60a430...4dc7e1`, match with `CLAIMS.json`). This artifact REPLACES the prior five-claim audit at this path and binds the runtime's authoritative theorem-bearing set: CLM-006, CLM-007, CLM-008, CLM-013, CLM-015, CLM-019, CLM-020. The prior run's findings for CLM-007/008/015/019/020 were not taken on trust: the Eddington-in-z and sky-marginal numerics were independently recomputed again this session (Probe 1), the CLM-015 implementation clamp and the CLM-020 notation split were re-inspected at source, and the CLM-015 gate numbers were re-read from `.planning/gate/G7row9_eddington_m_impact.json`. All prior findings stand. Two new claims were audited fresh: CLM-006 (estimator-construction claim) and CLM-013 (field-level significance claim); each contributes new gaps. The core mathematics of the five carried claims remains sound and every recomputed number reproduced exactly; the artifact fails closed on the carried three gaps plus four new CLM-006/CLM-013 items.

## Proof Inventory

- exact claim / theorem text: "Two prior-consistent estimator repairs are constructed: local_ratio (self-normalized local ratio of sums, numerator and selection denominator over the same localization-ball galaxy set with the same weights and kernel) and volume_deconv (comoving-volume deconvolution of the host-redshift kernel used identically in numerator and denominator); volume_deconv is adopted as the library default for every headline result." [CLM-006 — Construction of the two prior-consistent estimator repairs; quoted from CLAIMS.json]
- claim / theorem target: `estimators.tex` eqs. `eq:est:lcat`, `eq:est:pgz` (and `eq:est:Ng`/`eq:est:Dg`, `eq:est:pgM`); the prior-consistency definition of `framework.tex` sec. 2.6; the default declaration of `estimators.tex` sec. 4.5.
- named parameters: `G_i` (localization-ball galaxy set), `G_i^sel` (selection denominator set), `w_g` (per-galaxy rate weight), `p_g(z)` (host-redshift kernel), `w_pop(z)`, `Z_g`, `h`.
- hypotheses: (H1) "prior consistency" as defined in framework sec. 2.6 — a two-part invariant: (i) the identical kernel `p_g(z)` multiplies the measurement factor in `N_g` and the selection factor in `D_g`; (ii) every redshift integral carries the population measure exactly once, never zero times, never twice; (H2) `G_i^sel = G_i` with the same `w_g` and the same `p_g(z)` in both sums of eq:est:lcat; (H3) the deconvolved kernel eq:est:pgz is used identically in eq:est:Ng and eq:est:Dg (and its mass analogue eq:est:pgM in both sides of the mass channel); (H4) `volume_deconv` = deconvolved z- and M-kernels inside the local ratio of sums, and is the library default.
- quantifier / domain obligations: same-set/same-weights/same-kernel for all galaxies g and all trial h; kernel identity across numerator and denominator integration windows; "every headline result" (universal over the paper's quoted adopted-estimator numbers).
- conclusion clauses: (C1) local_ratio is constructed with the stated structural properties; (C2) local_ratio is a *prior-consistent* repair; (C3) volume_deconv is constructed with the kernel used identically on both sides; (C4) volume_deconv is adopted as the library default for every headline result.

- exact claim / theorem text: "Eddington-in-z law: omitting the population measure from the host-redshift kernel shifts each inferred host redshift by delta_z = sigma_z^2 q(z_g) + O(sigma_z^4) with q = d ln w_pop/dz, and biases the Hubble constant low by Delta_h ~= -K(zbar) sigma_z^2; verified on the single-host synthetic suite with K_meas ~= 17--20, constant to +-8 per cent across a factor 11 in sigma_z^2, consistent with zbar_eff ~= 0.25--0.27 inside the detected population; at GLADE-like z_g=0.05 the expansion parameter exceeds unity and exact quadrature deconvolution replaces the leading-order shift." [CLM-007 — Eddington-in-z law]
- claim / theorem target: `appendix_volume_deconv.tex` eqs. `eq:app:eddz`, `eq:app:dhlaw`; `estimators.tex` eq. `eq:est:eddh`.
- named parameters: `sigma_z`, `z_g`, `zbar`, `q(z)`, `K(zbar)`, `h`.
- hypotheses: (H1) flat LambdaCDM with Omega_L = 1 - Omega_m; (H2) redshift-independent comoving source-frame rate density R(z); (H3) Gaussian photo-z likelihood N(z_g; z, sigma_z^2); (H4) first-order expansion of ln w_pop valid, sigma_z q(z_g) << 1, for the leading-order form; (H5) GW data fix d_L, posterior peaks where model distance at assumed host redshift matches measured distance.
- quantifier / domain obligations: z >= 0 clamp of the quadrature window; sign clause ("low") requires q(zbar) > 0; zbar inside the detected population; breakdown regime sigma_z q >~ 1 handled by exact quadrature.
- conclusion clauses: (C1) delta_z = sigma_z^2 q(z_g) + O(sigma_z^4); (C2) Delta_h ~= -K(zbar) sigma_z^2, H0 biased low; (C3) K_meas ~= 17--20, constant to +-8 per cent over a factor 11 in sigma_z^2; (C4) consistent with zbar_eff ~= 0.25--0.27; (C5) leading order invalid at GLADE-like z_g = 0.05, exact quadrature used instead.

- exact claim / theorem text: "The exact narrow-beam marginal of the coordinate-density Fisher Gaussian over the isotropic sky prior is pbar_GW(u) = (sin theta_hat / 4pi) N(u; 1, Sigma_uu) with the marginal (not conditional) distance-fraction variance, valid to relative error <~1e-5 at the median 0.2 deg^2 localization; the replaced peak-density evaluation over-weights the completion term by 2/(sigma_phi sigma_theta) ~ 1.6e3 (2 deg error) to 1.8e5 (median localization), and a residual missing sin(theta) Jacobian (median 1.15x, mean pi/2) was found by this derivation and corrected." [CLM-008 — Completion-term sky marginal]
- claim / theorem target: `appendix_sky_marginal.tex` eq. `eq:app:sky:marginal`; `pitfall.tex` eq. `eq:pitfall:skymarg`.
- named parameters: `theta_hat`, `sigma_phi`, `sigma_theta`, `Sigma_uu`, `u`.
- hypotheses: (H1) trivariate Gaussian GW likelihood in bare coordinates (phi, theta, u); (H2) density with respect to the coordinate measure dphi dtheta du, not solid angle; (H3) narrow beam, sigma_phi, sigma_theta << 1 rad, away from ecliptic poles; (H4) isotropic sky prior 1/(4pi); (H5) completeness f approximately constant across the GW sky support.
- quantifier / domain obligations: extension of the compact chart to R^2; pole exclusion; validity across the +-4 sigma u-window.
- conclusion clauses: (C1) exact narrow-beam marginal = (sin theta_hat / 4pi) N(u; 1, Sigma_uu); (C2) Sigma_uu is the marginal, not conditional, variance; (C3) relative error <~ 1e-5 at median localization; (C4) peak-density over-weight 2/(sigma_phi sigma_theta) ~ 1.6e3--1.8e5; (C5) residual sin(theta) Jacobian, median 1.15, mean pi/2, found and corrected.

- exact claim / theorem text: "The silent rail under realistic photometric errors -- while calibrating perfectly on spectroscopic-quality validation inputs -- is a field-level hazard for every independently implemented dark-siren pipeline in the LISA / next-generation-detector era; prior-consistent normalization and known-truth coverage tests at photometric width are prerequisites for catalogue dark-siren cosmology." [CLM-013 — Field-level hazard and prerequisites; quoted from CLAIMS.json]
- claim / theorem target: `abstract.tex` (final sentence), `introduction.tex:100-104` ("is a field-level hazard for every independently implemented dark-siren pipeline, not a code anecdote"), `conclusions.tex` (second paragraph), `codes.tex` sec. 8.3.
- named parameters: `sigma_z` (photometric width), the `sigma_z/z ~ 0.7` regime, the era scope (LISA / next-generation detectors).
- hypotheses: (H1) the rail exists and was demonstrated (CLM-001/002/003 chain); (H2) each defect is an individually plausible reading of the published equations; (H3) the defects are invisible to the field's standard validation practice ("silent"); (H4) the pre-fix estimator calibrates perfectly on spectroscopic-quality validation inputs.
- quantifier / domain obligations: universal quantifier over "every independently implemented dark-siren pipeline"; necessity modality of "prerequisites"; the regime designation sigma_z/z ~ 0.7 must be occupied by the cited evidence.
- conclusion clauses: (C1) the rail is silent under realistic photometric errors while the estimator calibrates perfectly on spectroscopic-quality validation inputs; (C2) field-level hazard for every independently implemented pipeline in the LISA/next-gen era; (C3) prior-consistent normalization is a prerequisite for catalogue dark-siren cosmology; (C4) known-truth coverage tests at photometric width are a prerequisite.

- exact claim / theorem text: "Eddington-in-M: the bare host-mass kernel omits the per-MBH rate prior R_eff(M); under a log-linear expansion the prior-consistent kernel is exactly a shifted Gaussian M_g_eff = M_g(1 + alpha_g sigma_rel^2), but at GLADE+ scatter (median sigma_rel ~ 0.98) the local slope can predict the wrong sign near the rate roll-off, so the implementation moment-matches the exact first moment by per-galaxy quadrature; on the real-data configuration the fix leaves the 1D posterior essentially unchanged, moves the 2D mean from 0.790 to 0.770 toward the injected 0.73, and suppresses grid-edge mass from 0.216 to 0.023." [CLM-015 — Eddington-in-M rate-weighted host-mass prior]
- claim / theorem target: `appendix_eddington_m.tex` eqs. `eq:eddm:prior`, `eq:eddm:tilt`.
- named parameters: `M_g`, `sigma_M`, `sigma_rel`, `alpha_g`.
- hypotheses: (H1) Gaussian mass-measurement likelihood N(M; M_g, sigma_M^2); (H2) log-linear expansion of ln R_eff about M_g (for the exact-shifted-Gaussian form only); (H3) Gaussian times exponential tilt is exactly a shifted Gaussian of unchanged width.
- quantifier / domain obligations: domain of the Z_M integral (written unrestricted over M); +-5 sigma_M quadrature window; sigma_rel clipped at 2 (stated); support/definition of R_eff(M) at small and non-positive M (NOT stated — see gap).
- conclusion clauses: (C1) exact shifted Gaussian under log-linear expansion; (C2) local slope gets the sign wrong near the roll-off at GLADE scatter; (C3) implementation moment-matches the exact first moment by per-galaxy quadrature; (C4) real-data numbers: 1D unchanged (mean shift -5e-5), 2D mean 0.790 -> 0.770, edge mass 0.216 -> 0.023.

- exact claim / theorem text: "The volume deconvolution is exactly h-neutral: writing w_pop(z; h) = h^-3 wtilde(z) with wtilde an h-independent shape function, the h^-3 prefactor cancels exactly against the per-galaxy normalization Z_g, so the deconvolution injects no h-dependence into L_cat and the de-railing it produces on GLADE+ cannot be manufactured by the fix itself." [CLM-019 — h^-3 neutrality of the volume deconvolution]
- claim / theorem target: `estimators.tex` eq. `eq:est:pgz` and sec. 4.4/4.5; `appendix_volume_deconv.tex` property (ii).
- named parameters: `h`, `z`.
- hypotheses: (H1) flat LambdaCDM, E(z) = H(z)/H0 independent of h (Omega_m fixed), d_L(z, h) = A(z)/h factorizes; (H2) w_pop(z; h) = h^-3 wtilde(z) with wtilde proportional to I^2(z)/[E(z)(1+z)].
- quantifier / domain obligations: for all trial h on the grid; for all galaxies g (any z_g, sigma_z, including the clamped low-z window, which must itself be h-independent).
- conclusion clauses: (C1) h^-3 prefactor cancels exactly against Z_g; (C2) deconvolution injects no h-dependence into L_cat; (C3) "the de-railing it produces on GLADE+ cannot be manufactured by the fix itself".

- exact claim / theorem text: "Gray mapping: the assembled per-event likelihood p_i(h) = [beta_G L_cat + B_num]/D(h) is an exact algebraic rearrangement of the Gray et al. (2020) hypothesis mixture (their eq. 9 with A14/A15/A17/A18/A19 identifications, up to the smooth-completeness rewriting of the magnitude threshold), remaining finite in the complete-catalogue limit f -> 1; the term-by-term audit found no algebraic discrepancy, and every remaining difference is a deliberate documented deviation, the most consequential being the localization-ball selection denominator (nearest published analogue: their eq. A20)." [CLM-020 — Exact rearrangement of Gray et al. (2020)]
- claim / theorem target: `appendix_gray_mapping.tex` (incl. eq. `eq:gray:a10`, Table `tab:gray:mapping`); `framework.tex` eqs. `eq:mixture`, `eq:assembled`.
- named parameters: `h`, `f` (completeness), `beta_G`, `beta_Gbar`, `D(h)`.
- hypotheses: (H1) G and Gbar exhaustive and exclusive; (H2) hard apparent-magnitude threshold of Gray A14/A18 rewritable as the smooth per-pixel incompleteness weight 1 - f(z, Omega); (H3) hypothesis weights w_G = beta_G/D and completion likelihood B_num/beta_Gbar as identified from Gray A14--A19.
- quantifier / domain obligations: for all h; f in [0,1]; the f -> 1 limit; the identification must hold for every term of the mixture.
- conclusion clauses: (C1) exact algebraic rearrangement given the identifications; (C2) finite at f -> 1; (C3) no algebraic discrepancy found; (C4) every remaining difference is a deliberate documented deviation, most consequential the localization-ball denominator (nearest analogue A20).

## Coverage Ledger

### Named-Parameter Coverage

| Claim | Parameter | Status | Notes |
|---|---|---|---|
| CLM-006 | G_i (localization ball) | covered | Defined in framework eq:lcat / estimators eq:est:lcat; G_i^sel = G_i stated explicitly for local_ratio |
| CLM-006 | G_i^sel | covered | The denominator-set choice is named as one of the two normalization choices at the heart of the paper (framework sec. 2.3) |
| CLM-006 | w_g | covered | w_g = R_eff(M_g)/(1+z_g) defined; instantiates Gray's p(s\|z)p(s\|M); scaling invariance verified in code (weighted_ratio_of_sums docstring + implementation) |
| CLM-006 | p_g(z) | covered | General kernel form eq:kernel with Pi switch; bare vs deconvolved cases both defined |
| CLM-006 | w_pop(z) | covered | eq:est:wpop; same measure as completion/selection integrals (measure-symmetry property) |
| CLM-006 | Z_g | covered | Per-galaxy unit prior mass; the deliberate non-cancellation in the ratio of sums is stated in appendix_volume_deconv property (i) |
| CLM-006 | h | covered | h-neutrality of the deconvolution (CLM-019); grid domain stated |
| CLM-007 | sigma_z | covered | Enters eq:app:eddz/eq:app:dhlaw and tab:app:eddz; scaling tested over factor 11 in sigma_z^2 |
| CLM-007 | z_g | covered | Expansion point of ln w_pop; z_g = 0.05 breakdown case computed explicitly |
| CLM-007 | zbar | covered | Representative detected-host redshift; convexity caveat stated |
| CLM-007 | q(z) | covered | q = 2/(IE) - E'/E - 1/(1+z); independently recomputed this run, q(0.05) = 38.11 matches quoted 38.1 |
| CLM-007 | K(zbar) | covered | Independently recomputed this run at zbar = 0.20/0.25/0.26/0.30/0.357: 32.7/20.1/18.4/13.4/8.9, exact match; K(0.05) = 569 matches |
| CLM-007 | h | covered | Enters through Delta_h ~= -h (dlnD/dz) sigma_z^2 q; suite injects h = 0.72 (disclosed, differs from campaign 0.73) |
| CLM-008 | theta_hat | covered | sin(theta_hat) via E[sin theta] = sin(mu) exp(-s^2/2); pole divergence disclosed and gated |
| CLM-008 | sigma_phi | covered | Over-weight 2/(sigma_phi sigma_theta); 1.64e3 at 2 deg and 1.84e5 at median re-verified this run |
| CLM-008 | sigma_theta | covered | Curvature term (1/2) sigma_theta^2 ~ 5.4e-6 re-verified from Delta-Omega = 6.1e-5 sr (manuscript quotes 5.6e-6 with its sin theta_hat convention) |
| CLM-008 | Sigma_uu | covered | Marginal-vs-conditional distinction derived from the exact Gaussian factorization; correct |
| CLM-008 | u | covered | u = d_L/d_L_hat; u-dependence of mu_theta\|u is the unbounded drift term (see gap) |
| CLM-013 | sigma_z | partial | Load-bearing only for Defect 1 (sigma_z^2 law, verified); Defects 2 and 3 are zeroth order in sigma_z, yet the claim's "under realistic photometric errors ... while calibrating perfectly on spectroscopic-quality inputs" framing attributes sigma_z-dependence to the whole superposition (introduction.tex mechanism sentence) |
| CLM-013 | sigma_z/z ~ 0.7 regime | partial | The paper's own coverage tests live at absolute sigma_z = 0.035 with median detected z ~ 0.3 (sigma_z/z ~ 0.1--0.2); the 0.7 ratio is occupied only by the low-z photometric subset; cross-ref reader finding REF-R002 |
| CLM-013 | era scope (LISA/next-gen) | covered | Framing scope only; conclusions and codes.tex tie it to the multiplication of independent implementations |
| CLM-015 | M_g | covered | Expansion point; regression grid M_g in {1e5, 3e5, 8e5} Msun |
| CLM-015 | sigma_M | covered | Width retained in moment-matched Gaussian; sigma_M -> 0 limit stated |
| CLM-015 | sigma_rel | covered | Median 0.98 matches G7row9 JSON (0.9847, re-read this run); clip at 2 stated and present in code (bayesian_statistics.py:98) |
| CLM-015 | alpha_g | covered | Central finite difference, eps = 1 per cent; median -0.130 matches JSON (re-read this run) |
| CLM-019 | h | covered | h^-3 factorization exact given H1; verified algebraically from eq:dl |
| CLM-019 | z | covered | wtilde(z) proportional to I^2/[E(1+z)] verified; clamped window max(z_g - 4 sigma_z, 0) is h-independent as required |
| CLM-020 | h | covered | Rearrangement holds identically in h |
| CLM-020 | f | partial | f in [0,1] per pixel; manuscript internally inconsistent on whether f carries h (f(z,Omega) in framework.tex:214 vs f(z,Omega,h) in appendix_beta_g.tex:98, re-verified this run) — see missing hypothesis CLM-020-H |
| CLM-020 | beta_G | covered | beta_G = D - beta_Gbar by construction (eq:betag); A14-numerator identification traced |
| CLM-020 | beta_Gbar | covered | A18 identification with smooth-f rewriting; cancels in the assembled ratio |
| CLM-020 | D(h) | covered | A14-denominator identification; the G2c source additionally documents an in-code docstring miscitation (citation-level, not algebraic) |

### Hypothesis Coverage

| Claim | Hypothesis | Status | Notes |
|---|---|---|---|
| CLM-006 | H1 two-part prior-consistency definition | covered (definition), partial (application) | Definition stated precisely in framework sec. 2.6; its application to local_ratio covers requirement (i) only — requirement (ii) is violated by the retained bare kernel, which framework sec. 2.6 itself names as the bare-Gaussian defect (coverage_gaps item 4) |
| CLM-006 | H2 same set/weights/kernel in eq:est:lcat | covered | Stated explicitly (G_i^sel = G_i, same w_g, same p_g); code path confirms local_ratio/volume_deconv use the local ratio (bayesian_statistics.py:853-854, 1620-1629) |
| CLM-006 | H3 deconvolved kernel identical on both sides | covered | eq:est:pgz "used identically in the numerator and denominator"; the one window asymmetry is disclosed and bounded (6.3e-5 Gaussian >4-sigma tail, appendix_volume_deconv) |
| CLM-006 | H4 volume_deconv = z+M deconvolution inside local ratio; library default | covered | estimators sec. 4.5; verified in code: `_normalization_mode: str = "volume_deconv"` (bayesian_statistics.py:809, 843, 1857); Eddington-M shift applied only in volume_deconv mode (line 2062), matching the appendix statement |
| CLM-007 | H1 flat LambdaCDM | covered | Used in E(z), E'(z), I(z); Omega_m = 0.3 for the suite comparison |
| CLM-007 | H2 z-independent R | covered | Yields w_pop = (1/(1+z)) dV_c/dz; cross-checked in G2b sec. 1.1 |
| CLM-007 | H3 Gaussian photo-z likelihood | covered | Symmetric-argument reading N(z_g; z, sigma_z^2) made explicit |
| CLM-007 | H4 expansion validity sigma_z q << 1 | covered | Stated; breakdown quantified (sigma_z q = 0.19/0.57/1.33/1.91 at z_g = 0.05, re-verified this run) and handled by exact quadrature; the CLAIMS.json compression "exceeds unity" drops the manuscript's own condition sigma_z >~ 0.035 (registry wording, not a manuscript defect) |
| CLM-007 | H5 GW data fix d_L | covered | Single-event peak-matching propagation; heuristic status acknowledged via the population-averaging caveat and the 30--50 per cent amplitude accuracy statement |
| CLM-008 | H1 trivariate Gaussian | covered | Fisher/CRB provenance stated (eq:app:sky:gauss) |
| CLM-008 | H2 coordinate measure | covered | Load-bearing convention stated twice (framework and appendix) |
| CLM-008 | H3 narrow beam, away from poles | covered | Error budget items (a)-(b); pole exclusion via Fisher condition-number gate |
| CLM-008 | H4 isotropic 1/(4pi) prior | covered | Gray A.2c provenance |
| CLM-008 | H5 f constant across sky support | covered | Declared as construct (iv) in appendix_gray_mapping.tex; excellent at 0.2 deg^2 vs pixel scale |
| CLM-013 | H1 rail demonstrated | covered | CLM-001/002/003 chain (realdata.tex, tab:derail); one implementation, full-sample archived run |
| CLM-013 | H2 defects are plausible readings | covered | External evidence per defect: bare kernel is gwcosmo's declared choice; global-vs-local normalization documented as a live design point (Gray 2023) and an acknowledged icarogw approximation family; peak-density-vs-marginal argued via the coordinate-density/steradian ambiguity (appendix_sky_marginal) |
| CLM-013 | H3 silence under standard validation | partial | Rigorous only for Defect 1 (second order in sigma_z); Defects 2 and 3 are zeroth order in sigma_z, and their invisibility to the published validation record follows from configuration properties (complete-catalogue spectroscopic mocks; code-vs-code common-mode blindness), a hypothesis the manuscript uses but does not state (missing hypothesis CLM-013-H-validation-configuration) |
| CLM-013 | H4 pre-fix estimator calibrates on spectroscopic-quality inputs | gap | Not demonstrated in-paper for the full pre-fix estimator; only the kernel defect's sigma_z -> 0 limit is measured (clean single-host harness, no completion machinery, bias -0.0016 at sigma_z = 0.005, floor-dominated); an incomplete-catalogue spectroscopic validation would still expose Defect 2's sigma_z-independent 1.6e3--1.8e5 completion inflation (coverage_gaps item 7) |
| CLM-015 | H1 Gaussian mass likelihood | partial | At median sigma_rel = 0.98 this Gaussian places ~15 per cent of its mass at unphysical M < 0 and the artifact never says what R_eff or the integral does there; the implementation resolves it with an undisclosed domain clamp (missing hypothesis CLM-015-H) |
| CLM-015 | H2 log-linear expansion (tilt form only) | covered | Properly scoped; its failure at the roll-off is the appendix's own point |
| CLM-015 | H3 tilt identity | covered | N(M; mu, sigma^2) e^{tM} proportional to N(M; mu + t sigma^2, sigma^2) is exact; shift = M_g alpha_g sigma_rel^2 checks out |
| CLM-019 | H1 E(z) h-independent, d_L = A(z)/h | covered | Requires Omega_m fixed, which the paper's fiducial setup satisfies and the claim states |
| CLM-019 | H2 w_pop = h^-3 wtilde | covered | dV_c/dz per steradian = (c/100)^3 h^-3 I^2/E verified algebraically; cancellation against Z_g exact for any window because numerator and Z_g carry the same h |
| CLM-020 | H1 exhaustive/exclusive G, Gbar | covered | Standard mixture premise, stated in framework.tex |
| CLM-020 | H2 threshold -> smooth 1 - f | partial | The rewriting is asserted as exact after pre-integrating the Schechter integrals, but the h-dependence Gray's threshold z(M, m_th, H0) imparts to f is not pinned down, and the manuscript's own notation disagrees across sections (missing hypothesis CLM-020-H) |
| CLM-020 | H3 A14--A19 identifications | covered | Substitution p_i = (beta_G/D) L_cat + (beta_Gbar/D)(B_num/beta_Gbar) = (beta_G L_cat + B_num)/D re-verified symbolically this run; exact |

### Quantifier / Domain Coverage

| Claim | Obligation | Status | Notes |
|---|---|---|---|
| CLM-006 | same set/weights/kernel for all g, all h | covered | Structural, per-event; holds by construction of eq:est:lcat |
| CLM-006 | kernel identity across numerator/denominator windows | covered | The window impropriety is disclosed and bounded (6.3e-5); acceptable as stated |
| CLM-006 | "every headline result" uses volume_deconv | covered (with note) | Code default verified (bayesian_statistics.py:809); the diagnostic rows of tab:derail and the coverage-table bare rows deliberately use other configurations — they are comparisons, not headline results; the pending full-scale confirmation and the adopted numbers use volume_deconv |
| CLM-007 | z >= 0 clamp of quadrature window | covered | Disclosed with a Phi(-z_g/sigma_z) bound; probe used the same clamp and reproduced the exact-shift table |
| CLM-007 | sign clause needs q(zbar) > 0 | covered (scope note) | q > 0 for z <~ 1.4; all detected populations in the paper are safely inside; blanket "biases low" is regime-limited but fine in context |
| CLM-007 | zbar inside detected population | covered | zbar_eff 0.25--0.27 vs median 0.30--0.36 with the convexity direction correct |
| CLM-008 | chart extension to R^2 | covered | Error O(exp(-theta_hat^2/2 sigma_theta^2)), negligible away from poles |
| CLM-008 | u-window validity of sin(mu_theta\|u) ~ sin(theta_hat) | gap | The drift item is budgeted "sub-percent" with no bound on \|rho_theta-u\| over the sample, yet the headline bound is <~ 1e-5 (coverage_gaps item 1) |
| CLM-013 | universal over "every independently implemented pipeline" | gap | The manuscript's own audit (codes.tex) establishes the rail is structurally excluded from gwcosmo and absent from CHIMERA/icarogw/DarkSirensStat — the literal universal fails on the very implementations the paper inspected; the defensible reading (ex-ante hazard for unaudited/future implementations, protected only by structural exclusion or known-truth validation) is stated in conclusions.tex and codes.tex sec. 8.3 but not in introduction.tex or the registry text (coverage_gaps item 6) |
| CLM-013 | "prerequisites" necessity modality | partial | Sufficiency of the repairs is demonstrated; necessity is argued: for C3 the paper's own alternative (empirically calibrated redshift error models, codes.tex) is declared "equivalent in intent" — saving the clause as a class statement; the gwcosmo declared-posterior defence, which the paper itself formulates as an if-and-only-if on catalogue widths, conditions which kernel realizes the invariant |
| CLM-013 | evidence occupies sigma_z/z ~ 0.7 | partial | Coverage tests at sigma_z/z ~ 0.1--0.2; real-data sample mostly near-spectroscopic; cross-ref REF-R002 (reader stage) — recorded here as a domain-coverage note, severity owned by the reader finding |
| CLM-015 | domain of Z_M / R_eff support at M <= 0 | gap | eq:eddm:prior written with unrestricted integral; implementation clamps at max(M_g - 5 sigma_M, 1e3 Msun) (bayesian_statistics.py:99, re-verified this run), active for essentially all galaxies at sigma_rel ~ 1; not stated in the artifact (coverage_gaps item 2) |
| CLM-015 | +-5 sigma_M window, sigma_rel clip at 2 | covered | Both stated and match the implementation (lines 98-100) |
| CLM-019 | for all h on the grid, all galaxies | covered | Cancellation is per-galaxy, per-h, window-independent; exact |
| CLM-020 | for all h; f in [0,1]; f -> 1 limit | covered | Limit traced: beta_Gbar -> 0, B_num -> 0, p_i -> w_G L_cat with w_G -> 1; no division by beta_Gbar in the assembled form |

### Conclusion-Clause Coverage

| Claim | Clause | Status | Notes |
|---|---|---|---|
| CLM-006 | C1 local_ratio structural construction | supported (with lemma note) | eq:est:lcat has exactly the claimed properties; but the supporting cancellation lemma "every per-galaxy constant ... cancels row by row" is false for genuinely per-galaxy factors in a ratio of sums (two-galaxy counterexample; contradicted by the manuscript's own Z_g remark and appendix_beta_g's precise "ball-common factors cancel") — coverage_gaps item 5 |
| CLM-006 | C2 local_ratio is "prior-consistent" | NOT fully supported | Satisfies requirement (i) of framework sec. 2.6 only; requirement (ii) is violated by the retained bare kernel, per the manuscript's own sec. 4.5 and the 0--3 per cent coverage of that kernel in Table 1; the abstract's "numerator--denominator prior consistency" is the correctly scoped label — blocking for passed (coverage_gaps item 4) |
| CLM-006 | C3 volume_deconv identical-kernel construction | supported | eq:est:pgz in both eq:est:Ng and eq:est:Dg; measure-symmetry property (i) of appendix_volume_deconv extends it to D(h), beta_Gbar; disclosed bounded asymmetry only |
| CLM-006 | C4 library default for every headline result | supported | Verified in code (default "volume_deconv", bayesian_statistics.py:809/843/1857) and consistent with sec. 4.5, the coverage harness default, and the pending full-scale run |
| CLM-007 | C1 delta_z law | supported | Standard Laplace/complete-the-square argument; next correction genuinely O(sigma_z^4) |
| CLM-007 | C2 Delta_h ~= -K sigma_z^2, low | supported | Peak-matching propagation; sign correct in the stated regime |
| CLM-007 | C3 K_meas 17--20, +-8 per cent | supported | 0.0044/0.015^2 = 19.6, 0.021/0.035^2 = 17.1, 0.044/0.05^2 = 17.6 — verified; conditional on the disclosed sigma_z-independent floor subtraction (-0.002) |
| CLM-007 | C4 zbar_eff 0.25--0.27 | supported | K table independently reproduced to all quoted digits (this run) |
| CLM-007 | C5 exact quadrature at z_g = 0.05 | supported | Probe re-run this session: exact shifts +0.0009/+0.0079/+0.0324/+0.0534 vs quoted +0.00094/+0.0079/+0.0325/+0.0535; leading-order +0.0467/+0.0953 exact match |
| CLM-008 | C1 marginal formula | supported | Exact factorization + E[sin theta] identity verified; manuscript displays both exact and approximate forms |
| CLM-008 | C2 marginal variance | supported | Marginal-vs-conditional distinction correct and load-bearing |
| CLM-008 | C3 relative error <~ 1e-5 | NOT fully supported | The same paragraph budgets a "sub-percent, h-smooth drift" (rho_theta-u term) and then concludes <~ 1e-5; the source derivation keeps the qualifier, the manuscript dropped it — blocking for passed |
| CLM-008 | C4 over-weight 1.6e3--1.8e5 | supported | 2/(0.0349 rad)^2 = 1.64e3; 2/(3.3e-3)^2 = 1.84e5 — re-verified this run |
| CLM-008 | C5 sin theta residual, median 1.15, mean pi/2 | supported | median 1/sin theta = 1/sqrt(0.75) = 1.155; E[1/sin theta] = pi/2 exactly — re-verified analytically this run |
| CLM-013 | C1 silent rail while calibrating perfectly on spectroscopic inputs | NOT fully supported | sigma_z^2-silence proved for Defect 1 only; Defects 2/3 are zeroth order in sigma_z; no in-paper spectroscopic-quality calibration of the full pre-fix estimator exists; the introduction's "whose effects are second order in the redshift kernel width" over-generalizes — blocking for passed (coverage_gaps item 7, missing hypothesis CLM-013-H) |
| CLM-013 | C2 hazard for every independent pipeline | NOT fully supported (narrowed) | Literal universal contradicted by the paper's own four-pipeline audit; survives as the ex-ante hazard statement of conclusions.tex/codes.tex sec. 8.3 — blocking for passed (coverage_gaps item 6) |
| CLM-013 | C3 prior-consistent normalization prerequisite | supported (as class claim) | Sufficiency demonstrated (de-rail matrix, ablation, coverage); the paper's own "equivalent in intent" framing of calibrated z-error models keeps the clause true as a class requirement; the posterior-vs-likelihood iff condition should accompany it |
| CLM-013 | C4 known-truth coverage tests at photometric width prerequisite | supported (with regime note) | The demonstrated silent class is undetectable by code-vs-code or real-data tests (argued from the record); note the paper's own coverage test occupies sigma_z/z ~ 0.1--0.2, not the 0.7 named in the same breath (REF-R002) |
| CLM-015 | C1 exact shifted Gaussian under tilt | supported | Exact identity |
| CLM-015 | C2 wrong sign at roll-off | supported | Mechanism and example documented in G2d sec. 4; regression tests cover both slope-sign regimes |
| CLM-015 | C3 "exact first moment by per-galaxy quadrature" | NOT fully supported | Exact only on the undisclosed clamped domain M in [max(M_g - 5 sigma_M, 1e3), M_g + 5 sigma_M]; the artifact describes the quadrature as "over +-5 sigma_M" and writes Z_M unrestricted — blocking for passed |
| CLM-015 | C4 real-data numbers | supported | All five numbers re-verified this run against .planning/gate/G7row9_eddington_m_impact.json (2D mean 0.78967 -> 0.76969; edge 0.21586 -> 0.02291; 1D mean shift -5.17e-5; MAP unchanged on the 7-point grid) |
| CLM-019 | C1 h^-3 cancellation exact | supported | Exact algebra given H1/H2 |
| CLM-019 | C2 no h-dependence injected into L_cat | supported | With the manuscript's precise reading: no explicit h-dependence; only the (h-independent) z-shape reweights already-h-dependent integrals |
| CLM-019 | C3 "cannot be manufactured by the fix itself" | NOT fully supported | h-neutrality excludes only a mechanical normalization tilt; an h-neutral but mis-shaped kernel could still relocate the posterior peak. That the relocated peak is calibrated is established by the Section 5 coverage tests, which the "so ..." in estimators.tex sec. 4.5 does not invoke — blocking for passed |
| CLM-020 | C1 exact rearrangement | supported | Re-verified symbolically this run; identical to Gray eq. 9 under the five identifications |
| CLM-020 | C2 finite at f -> 1 | supported | No division by beta_Gbar in the assembled form; beta_G -> D, w_G -> 1 |
| CLM-020 | C3 no algebraic discrepancy | supported | Term-by-term table consistent with the G2c source audit |
| CLM-020 | C4 every difference deliberate and documented | supported with note | D1--D9 traceable in the manuscript (w_g-outside-the-integral sub-deviation explicitly stated; finite windows visible in framework definitions); the h-dependence of f under the threshold rewriting remains the one under-specified identification detail (missing hypothesis CLM-020-H) |

## Adversarial Probe

**Probe 1 — independent recomputation (re-executed this run, not reused).**

Probe type: counterexample attempt on CLM-007/CLM-008 numerics via independent recomputation. Re-derived w_pop proportional to I^2/[E(1+z)] and recomputed q(z), K(zbar), the expansion parameters, and the exact posterior-mean shift by clamped quadrature, flat LambdaCDM Omega_m = 0.3, in the project venv this session. Results: q(0.05) = 38.11; K(0.05) = 569; K(0.20/0.25/0.26/0.30/0.357) = 32.7/20.1/18.4/13.4/8.9; sigma_z q(0.05) = 0.19/0.57/1.33/1.91; exact shifts at z_g = 0.05: +0.0009/+0.0079/+0.0324/+0.0534 vs manuscript +0.00094/+0.0079/+0.0325/+0.0535 (quadrature tolerance); leading order +0.0467/+0.0953 exact. Sky factors: 2/(2 deg)^2 = 1.64e3; 2/(3.3e-3)^2 = 1.84e5; curvature 5.4e-6; median 1/sin theta = 1.155, mean = pi/2 exactly.

Result: no counterexample; every reused number independently confirmed.

**Probe 2 — dropped-parameter/domain test on CLM-015 (re-verified at source this run).**

Probe type: dropped-parameter / domain test. At median sigma_rel = 0.98 the Gaussian mass kernel places ~15 per cent of its mass at unphysical M < 0 (5-sigma lower edge ~ -3.9 M_g). `eddington_shifted_host_mass` (bayesian_statistics.py:74-108) clamps the moment quadrature at `lo = max(host_M - 5*sigma, 1e3)` (line 99) with `sigma = min(host_M_error, 2*host_M)` (line 98). The clamp is active for essentially every analysed galaxy and appears nowhere in the proof artifact, which describes the quadrature as "over +-5 sigma_M" and writes Z_M unrestricted.

Result: claim narrowed — the "exact first moment" is exact on an undisclosed truncated domain; the z-channel's analogous z >= 0 clamp received a disclosed paragraph, the M-channel's did not.

**Probe 3 — scope-narrowing challenge on CLM-019 and CLM-008 (carried, re-checked against the manuscript text this run).**

Probe type: scope-narrowing challenge. (i) CLM-019: an h-neutral kernel replacement can still relocate the posterior peak via its z-shape (that is its purpose); h-neutrality alone cannot establish that the observed de-railing is not an artifact of the fix. The inference in estimators.tex sec. 4.5 ("exactly h-neutral ... so the de-railing ... cannot be manufactured by the fix itself") survives only in the narrower normalization-tilt reading; the full defense needs the Section 5 coverage calibration as an additional premise. The appendix property (ii) wording ("injects no spurious h-dependence") is already correctly scoped. (ii) CLM-008: the claimed <~ 1e-5 bound conflicts with the same paragraph's "sub-percent, h-smooth drift" item (appendix_sky_marginal.tex:95-99) unless |rho_theta-u| is bounded, which it is not.

Result: both clauses narrowed; no counterexample to the underlying algebra.

**Probe 4 — NEW: quantifier attack on CLM-013 (literal-universal check plus sigma_z-order check).**

Probe type: quantifier attack / narrower-case challenge. (i) Tested the universal "every independently implemented dark-siren pipeline" against the manuscript's own evidence base: codes.tex concludes the rail mechanism is "structurally excluded" from gwcosmo (both generations) and "absent" from CHIMERA, icarogw, and DarkSirensStat — four independently implemented pipelines for which the hazard, as a rail mechanism, is refuted by the paper itself. The universal fails literally; the claim's defensible content is the ex-ante form (any unaudited or future independent implementation is at risk unless protected by structural exclusion or known-truth photometric-width validation), which conclusions.tex and codes.tex sec. 8.3 state and introduction.tex/the registry text do not. (ii) Tested the "silent ... while calibrating perfectly on spectroscopic-quality validation inputs" clause by checking each defect's order in sigma_z: Defect 1 is O(sigma_z^2) — silent at spectroscopic quality (verified by the coverage suite's sigma_z scan); Defect 2's over-weight 2/(sigma_phi sigma_theta) contains no sigma_z at all; Defect 3's -17.2 per cent tilt is measured with p_det point-evaluated at catalogue redshifts, i.e. already the sigma_z = 0 configuration of the denominator. A spectroscopic-quality validation on an *incomplete* catalogue would therefore still expose Defect 2 (inflated completion term dominating) — the pre-fix estimator's "perfect calibration" at spectroscopic quality requires the additional, unstated hypothesis that the validation configuration suppresses the completion channel (as the historical record's complete-catalogue mocks and code-vs-code comparisons in fact did).

Result: quantifier narrowed and one conclusion clause unsupported as stated; the underlying field-level significance survives in the conclusions' scoped form.

**Probe 5 — NEW: definition-consistency attack on CLM-006.**

Probe type: definition-consistency attack plus counterexample attempt on the supporting lemma. Applied the manuscript's own two-part prior-consistency definition (framework sec. 2.6) to each constructed repair. volume_deconv: satisfies (i) and (ii) — pass. local_ratio: satisfies (i) (identical bare kernel in N_g and D_g) but violates (ii) (population measure counted zero times in the in-catalogue channel — the definition's named "bare-Gaussian defect"); the manuscript itself concedes this (estimators sec. 4.5: local_ratio "retains the bare redshift kernel and with it the -K sigma_z^2 law"; code comment bayesian_statistics.py:1904-1905: "global"/"local_ratio" use the BARE photo-z Gaussian). Additionally attacked the supporting cancellation lemma: in a ratio of sums, a genuinely per-galaxy factor c_g does not cancel (two-galaxy counterexample above); estimators.tex's list ("number-density calibration, rate normalization, completeness weighting") includes per-galaxy completeness weighting, while appendix_gray_mapping's version of the same sentence quietly drops it and appendix_beta_g states the correct form ("ball-common factors cancel").

Result: C2 narrowed (label holds under requirement (i) only); supporting lemma imprecise with an internal contradiction across three sections; the constructions themselves and the default adoption are confirmed, including at code level.

## Verdict

Scope status: `narrower_than_claim` — the proofs establish narrower results than five conclusion clauses assert: CLM-008 error bound, CLM-015 exact-moment domain, CLM-019 "cannot be manufactured" inference (carried); CLM-006 "prior-consistent" label for local_ratio and CLM-013 "silent while calibrating perfectly" clause (new). The core constructions and theorems (Eddington-in-z law, sky-marginal factorization, tilt identity, h^-3 cancellation, Gray rearrangement, both estimator constructions, the volume_deconv default) are correct as stated.

Quantifier status: `narrowed` — CLM-013's universal "every independently implemented dark-siren pipeline" is contradicted as a literal universal by the manuscript's own four-pipeline audit and survives only in the ex-ante form used by conclusions.tex; carried narrowings: CLM-015 mass-domain restriction, CLM-008 unbounded u-window drift.

Counterexample status: `narrowed_claim` — no counterexample found to any mathematical content (Probe 1 reproduced every recomputed number; Probe 5's two-galaxy counterexample defeats only the supporting cancellation lemma as phrased, not the construction); Probes 2-5 succeeded in narrowing five conclusion clauses and one universal quantifier.

Blocking gaps: the seven `coverage_gaps` frontmatter items and the three `missing_hypothesis_ids` (CLM-015 mass-domain clamp; CLM-020 completeness h-dependence; CLM-013 validation-configuration hypothesis).

Per-claim: CLM-007 and CLM-020 survive with notes; CLM-006, CLM-008, CLM-013, CLM-015, and CLM-019 each carry at least one unsupported, over-broad, or under-specified clause. Under fail-closed rules the artifact-level status is `gaps_found`. All gaps are wording/disclosure/scoping repairs — none is a mathematical error, and none threatens the paper's central de-rail results; the CLM-013 repairs matter most because they sit in the abstract/introduction where referees will read them first.

## Required Follow-Up

1. **CLM-008 (appendix_sky_marginal.tex, error-budget paragraph):** restore the source derivation's qualifier — state the bound as "valid to relative error <~ 1e-5 up to a sub-percent, h-smooth drift from the u-dependence of the conditional sky mean", or bound |rho_theta-u| over the event sample and keep the 1e-5 figure with that bound cited.
2. **CLM-015 (appendix_eddington_m.tex, moment-matching paragraph):** disclose the quadrature domain clamp M >= max(M_g - 5 sigma_M, 1e3 Msun) exactly as the z-channel disclosed its z >= 0 clamp, state that Z_M is taken over the physical support of R_eff, and add one sentence bounding the truncation's effect on the first moment.
3. **CLM-019 (estimators.tex sec. 4.5, third reason):** narrow the inference — e.g., "the deconvolution is exactly h-neutral, so it cannot inject a spurious h-tilt through its own normalization; that the relocated peak is calibrated rather than manufactured is established independently by the coverage tests of Section 5."
4. **CLM-020 (framework.tex / appendix_gray_mapping.tex / appendix_beta_g.tex):** harmonize the completeness notation (f(z, Omega) vs f(z, Omega, h)) and state whether the implemented number-count completeness carries h-dependence, and how that relates to Gray's h-dependent threshold z(M, m_th, H0) that the smooth-f rewriting replaces.
5. **CLM-006 (estimators.tex secs. 4.1 and 4.5; section title):** scope the label — describe local_ratio as restoring *numerator--denominator* prior consistency (requirement (i) of Section 2.6), with the full measure-once invariant (requirement (ii)) restored only by volume_deconv; the abstract already uses the correct phrasing. Separately, replace "every per-galaxy constant ... cancels row by row" with the precise statement (ball-common constants cancel; per-galaxy factors enter numerator and denominator with identical weights), matching appendix_beta_g's wording, and drop "completeness weighting" from the cancellation list or reclassify it.
6. **CLM-013 (introduction.tex:100-104 and the mechanism sentence at introduction.tex:81-87):** (a) replace the universal with the conclusions' scoped version — the rail is a hazard of independent implementation, multiplied by the LISA/next-generation era, checkable via the consistency invariant, with gwcosmo as the audited positive example; (b) correct the mechanism sentence: only the kernel defect is second order in sigma_z; the sky-factor and global-tilt defects are zeroth order in sigma_z and were invisible to past validations for configuration reasons (complete-catalogue mocks, code-vs-code common-mode blindness) — which is exactly what codes.tex documents; (c) qualify "calibrating perfectly on spectroscopic-quality validation inputs" to the kernel defect, or state the completion-suppressed-validation condition explicitly. Coordinate with reader finding REF-R002 on the sigma_z/z regime bookkeeping.
7. **CLAIMS.json registry (non-manuscript):** CLM-007's compressed clause "the expansion parameter exceeds unity" should carry the manuscript's own condition (sigma_z >~ 0.035); CLM-006's registry text inherits the unscoped "prior-consistent" label (follow-up 5); CLM-013's registry text inherits the introduction's universal (follow-up 6).
