# G5b — Paper-A checkpoint, part 2: photo-z host marginalization in CHIMERA, icarogw, and DarkSirensStat

**Date:** 2026-07-02 · **Author:** commission subagent (G5b) · **Companion:** G5a (gwcosmo)
**Repos inspected (shallow clones, `/tmp/g5b/`):**

| code | remote | commit / tag inspected |
|---|---|---|
| CHIMERA v1 | `github.com/CosmoStatGW/CHIMERA` | tag `v1.0.0` = `cc5f1b3` (paper version, Borghi et al. 2024) |
| CHIMERA v2 | same | HEAD `fd35f89` = tag `v2.2` (2026-06-09) |
| icarogw | `github.com/simone-mastrogiovanni/icarogw` | HEAD `f8277b7` (2025-09-10) |
| DarkSirensStat | `github.com/CosmoStatGW/DarkSirensStat` | HEAD (shallow, 2026-07-02) |

**Papers:** Borghi et al. 2024, ApJ 964:191, arXiv:2312.05302 (full text read via arXiv mirror);
Mastrogiovanni et al. 2023, A&A 682:A167, arXiv:2305.17973 (full text read);
Finke et al. 2021, JCAP 08:026, arXiv:2101.12660 (code + selected text).

---

## 0. The pitfall under test (definitions)

This note checks three failure modes identified in our EMRI pipeline
(`master_thesis_code/bayesian_inference/bayesian_statistics.py`, branch `physics/derail-completion-4pi`,
commits `cb16142` + `6d4c4e1`), against the three published dark-siren codes:

**P1 — bare photo-z numerator.** Writing the in-catalogue host term with the *bare* redshift
likelihood,
$$L_{\rm cat}(h) \propto \sum_g w_g \int \mathcal{N}\!\left(z;\, \tilde z_g, \tilde\sigma_{z,g}^2\right)\, p_{\rm GW}(z \mid h)\, dz,$$
instead of the *posterior* obtained by multiplying the redshift likelihood by a population
(volume) prior and renormalizing per galaxy,
$$p(z \mid d_g^{\rm EM}) = \frac{\mathcal{N}(z;\tilde z_g,\tilde\sigma_{z,g}^2)\; \frac{dV_c}{dz}}{\int \mathcal{N}(z';\tilde z_g,\tilde\sigma_{z,g}^2)\; \frac{dV_c}{dz'}\, dz'}.$$
The bare form carries an Eddington-type bias of relative size $\mathcal{O}(\sigma_z^2\, d\ln(dV_c/dz)/dz)$
per galaxy that grows with $\sigma_z$ and, in our pipeline, produced the $h\to0.86$ rail
(cured by `normalization_mode="volume_deconv"`, `bayesian_statistics.py:804-827`).

**P2 — completion-term sky prefactor.** The out-of-catalogue ("completion") term must carry
a sky measure consistent with the in-catalogue term: for an isotropic completion this is the
$p(\Omega)=1/(4\pi)$ prior (our fix `cb16142`; `bayesian_statistics.py:1679`), or, better,
a per-line-of-sight completeness $P_{\rm comp}(z,\hat\Omega)$.

**P3 — in-catalogue normalization structure.** Whether the catalogue term is normalized
globally (one denominator for the whole sky/catalogue), locally (ratio-of-sums over the
event region, Gray et al. 2020 arXiv:1908.06050 Eqs. A.9–A.10), or per pixel — and whether
the numerator and the selection denominator $\xi(\lambda)$ (or $D(h)$) use the *same* galaxy
distribution model.

---

## 1. CHIMERA (Borghi et al. 2024, arXiv:2312.05302)

### 1.1 Photo-z host marginalization — volume-weighted, NOT bare (P1: correctly handled)

**Paper.** Section 2.2: the catalogue term is (paper Eq. 11)
$$p_{\rm cat}(z,\hat\Omega \mid \boldsymbol\lambda_c) = \frac{\sum_g w_g\, p(z \mid d_g^{\rm EM}, \boldsymbol\lambda_c)\, \delta(\hat\Omega - \hat\Omega_g)}{\sum_g w_g},$$
and each galaxy kernel is explicitly the *posterior* under a uniform-in-comoving-volume prior
(paper Eq. 12):
$$p(z \mid d_g^{\rm EM}, \boldsymbol\lambda_c) = \frac{\mathcal{N}(z;\, \tilde z_g, \tilde\sigma_{z,g}^2)\; \frac{dV_c}{dz}}{\int \mathcal{N}(z;\, \tilde z_g, \tilde\sigma_{z,g}^2)\; \frac{dV_c}{dz}\, dz}.$$
Verbatim from the paper (Sec. 2.2): *"To get $p(z|d_g^{\rm EM},\boldsymbol\lambda_c)$ we need to
multiply it by a prior on the redshift distribution, which in the absence of other information is
naturally chosen as uniform in comoving volume (Gair et al. 2023)."* This is mathematically
identical to our `volume_deconv` per-galaxy renormalization.

**Code, v1.0.0** (`CHIMERA/EM.py:261-288`, `sum_Gaussians_UCV`, UCV = uniform comoving volume):
```python
gauss = Gaussian(z_grid, mu, sigma)*fLCDM.dV_dz(z_grid, lambda_cosmo)
norm  = jnp.trapz(gauss, z_grid, axis=0)
return jnp.sum(weights * gauss/norm, axis=1) / jnp.sum(weights)
```
Gaussian $\times\, dV_c/dz$, renormalized **per galaxy** — exactly Eq. 12. The bare-Gaussian
variant `sum_Gaussians` (`EM.py:304-327`) exists but is not what `Galaxies.precompute_event`
calls (`EM.py:124-126` calls `sum_Gaussians_UCV`).

**Code, v2.2** (`CHIMERA/catalog/catalog.py:293-302`, `_sum_gaussians_ucv`): byte-equivalent
logic (`gauss *= dVcdz_at_z(...); norm = trapz(...); sum(weights*gauss/norm)/sum(weights)`),
still the default kernel (`precompute_p_cat(..., sumgauss="dVdz")`, `catalog.py:137,207-210`).
v2 adds `_sum_gaussians_pbkg` (`catalog.py:305-313`), which weights each Gaussian by the full
background distribution $p_{\rm bkg}(z)$ instead of the plain volume element — a strictly more
general population-prior weighting.

Note the weighting uses a *fixed fiducial* cosmology (`{"H0": 70, "Om0": 0.3}` in v1;
the reference `cosmo` object in v2). This is safe for $H_0$ because $dV_c/dz \propto H_0^{-3}$
cancels in the per-galaxy normalization; the paper says so explicitly: *"this last approximation
is not a problem for $H_0$ inference since both the numerator and denominator ... have a $H_0^3$
dependence, [but] the impact of a possible bias on $\Omega_m$ should be carefully assessed when
future analyses using events at higher $z$ will be carried out."*

**Numerator/denominator consistency.** Numerator (v1 `Likelihood.py:254`, v2
`catalog/catalog.py:118-121`):
$$p_{\rm gal}(z,\hat\Omega) = f_R\, p_{\rm cat}(z,\hat\Omega) + \bigl(1 - P_{\rm comp}(z)\bigr)\, p_{\rm bkg}(z),$$
(paper Eqs. 13–16, with $f_R \equiv \frac{1}{V_c}\int P_{\rm comp}\, dV_c$; v2
`completeness.py:213-218` computes $f_R = \int P_{\rm comp}\, p_{\rm bkg}\, dz$ per MCMC step).
Denominator: v1 `Bias.py:56` takes `p_bkg` as an argument, defaulting to `model_cosmo.dV_dz`,
with the docs directing the user to pass the sky-averaged interpolant
$\;\_p\_bkg\_fcn(z,\lambda) = f_R(\lambda)\, p_{\rm cat,int}(z) + (1-P_{\rm comp}(z))\, dV_c/dz$
built by `EM.py:224-230`. v2 hard-wires the consistency: the injection weights use
`pop_lambdas.gal_cat.p_bkg(...)` (`population/pop_wrapper.py:103-112`), i.e. the *same*
completeness object that builds the numerator's completion term
(`completeness.py:192-207` for `p_bkg`, normalized over the analysis range). So the selection
term $\xi(\lambda)$ integrates the sky-averaged version of the very $p_{\rm gal}$ used per event —
structurally the same "global $D(h)$ vs. per-event numerator" split as ours, without a
bare-Gaussian anywhere.

### 1.2 Out-of-catalogue completion sky handling (P2: correctly handled, isotropic-per-pixel)

The paper defines homogeneous completion as *"uniform in comoving volume and in sky position"*
(Sec. 2.2, Eq. 15: $p_{\rm miss}^{\rm HOM} \propto (1-P_{\rm comp})\, dV_c/dz\, /\, [(1-f_R)V_c]$).
In code, the completion term $(1-P_{\rm comp})\,dV_c/dz$ is added identically in **every** pixel
of the event's localization (v1 `Likelihood.py:254`, v2 `catalog.py:120`), and the event
likelihood is the pixel sum with a uniform sky measure:
```python
like_events[e] = np.nansum(like_pix, axis=0) / hp.pixelfunc.nside2npix(self.nside[e])   # v1 Likelihood.py:266
```
i.e. $\mathcal{L}_e = \sum_{\rm pix} \mathcal{L}_{\rm pix}\, \Delta\Omega_{\rm pix}/(4\pi)$ since
$1/N_{\rm pix}^{\rm tot} = \Delta\Omega_{\rm pix}/(4\pi)$. The completion mass is therefore
distributed with the isotropic $1/(4\pi)$ prior over the sky — precisely the structure of our
commit `cb16142` (sky-marginalized completion numerator $B_{\rm num}$ over an isotropic prior,
`bayesian_statistics.py:1650-1688`). In v2 the pixel sum drops the constant $1/N_{\rm pix}$
(`likelihood.py:337-340`, `jnp.sum(..., axis=-1)`), which is $\lambda$-independent per event and
thus posterior-irrelevant. The mask-based variant (`mask_completeness`,
v2 `completeness.py:235-533`) makes $P_{\rm comp}(z,\hat\Omega)$ sky-dependent per mask,
following Finke et al. (2021).

### 1.3 In-catalogue normalization structure (P3: local per-pixel self-normalization; one observation)

$p_{\rm cat}$ is *self-normalized* — a ratio of sums $\sum_g w_g \hat p_g(z) / \sum_g w_g$ —
which is the same estimator family as our `local_ratio`/Gray A.9–A.10 fix (numerator and its
own normalization built from the same local galaxy set), not a global catalogue-wide constant.
The $\lambda$-dependent amplitude enters only through $f_R(\lambda)$ and the shared rate
normalization `_get_p_z_norm` (v1 `Likelihood.py:66-73`), which divides numerator and
(normalized-mode) denominator alike.

**Observation (code vs. paper Eq. 11).** In both v1 (`EM.py:124-126`) and v2
(`catalog.py:207-214`), the per-galaxy weights are normalized **per pixel**
($\sum_{g\in\rm pix} w_g$ in the denominator), whereas paper Eq. 11 normalizes by the sum over
*all* galaxies. With per-pixel normalization, $\int p_{\rm cat}\, dz = 1$ in every non-empty
pixel, so the *relative galaxy count between pixels does not weight the catalogue term*; sky
weighting comes only from $p_{\rm gw}(z,\hat\Omega)$. `N_gal` per pixel is computed
(`EM.py:132`) but not used in `compute()`. Whether this equal-pixel-weight choice is intended
(a flat sky prior conditioned on "in catalogue") or a deviation from Eq. 11 (which would weight
pixels by galaxy density, as gwcosmo's LOS $dN/dz$ does) is not resolvable from the paper text;
it does not touch the P1 photo-z question but belongs to the same P3 normalization family.
Flagged for awareness, not as a confirmed bug.

### 1.4 Published validation regime in $\sigma_z$ — and the commission's belief

**The commission's belief that CHIMERA "defers photo-z to future work" is FALSE.** The
photometric case is one of the paper's two headline regimes. Paper Sec. 3.1 (Eq. 22) defines
exactly two uncertainty regimes applied to the 1.6M-galaxy MICEv2 parent sample:
$$\sigma_z^{\rm spec} = 0.001\,(1+z), \qquad \sigma_z^{\rm phot} = 0.05\,(1+z),$$
with the photometric value motivated by DES/Euclid (*"assuming an uncertainty
$\sigma_z/(1+z) = 0.05$ ... easily accessible with current ongoing surveys like DES"*).
Headline results (Abstract + Sec. 5): 100 best BBHs, complete catalogue —
$O4$-like: $\sigma_{H_0}/H_0 \approx 7\%$ (spec) vs. $\approx 18\%$ (phot, "three times greater");
$O5$-like: $\approx 1\%$ (spec) vs. degraded *"up to a factor of $\sim 9$"* (phot), *"leaving a
significant correlation between $H_0$ and the mass scales that must be carefully modeled to
avoid bias."* What the paper *does* defer to future work: sky-varying $\sigma_z$ per survey
mode (*"with Euclid it would change between the photometric or the spectroscopic survey mode
... requiring a more detailed assessment in a future study"*) and the $\Omega_m$-bias of the
fixed-fiducial $p_{\rm cat}$ interpolant (Sec. 1.1 quote above). Possible origin of the
commission's belief: those two deferral sentences, or the abstract's advocacy of spectroscopic
catalogues — neither amounts to deferring photo-z *treatment*; the treatment (Eq. 12 = UCV
kernel) is implemented and exercised at $\sigma_z/(1+z)=0.05$.

**CONCLUSION (CHIMERA): pitfall correctly handled.** Volume-weighted per-galaxy kernels
(P1 ✓), isotropic-per-pixel completion consistent with a $1/(4\pi)$ sky prior (P2 ✓),
local self-normalized catalogue term with a matching sky-averaged selection denominator
(P3 ✓, with the per-pixel-vs-global Eq. 11 observation), validated up to
$\sigma_z/(1+z) = 0.05$ — the photometric regime — where it yields broadened-but-unbiased
$H_0$ (no rail), provided the mass model is correct.

---

## 2. icarogw 2.0 (Mastrogiovanni et al. 2023, arXiv:2305.17973)

### 2.1 Photo-z host marginalization (P1: correctly handled; bare Gaussian exists only as opt-in)

The single entry point for the galaxy redshift kernel is
`icarogw/catalog.py:996`, `EM_likelihood_prior_differential_volume`, whose docstring reads
*"Calculates the EM likelihood in redshift times a uniform in comoving volume prior"*
(preceded by an `# LVK Reviewed` tag, `catalog.py:995`). Three `ptype` options:

- `'uniform'` (`catalog.py:1032`): box likelihood of half-width $N_\sigma \sigma_z$ times the
  volume prior, per-galaxy normalized —
  $\; p(z) = \frac{4\pi\, \frac{dV_c}{dz\,d\Omega}\, \mathbb{1}[|z-z_{\rm obs}|\le N_\sigma\sigma_z]}{V_c(z_{\rm max}) - V_c(z_{\rm min})}$.
- `'gaussian'` (`catalog.py:1039-1050`):
  `prior_eval = cosmology.dVc_by_dzdOmega_at_z(z) * user_normal(z, zobs, sigmaz)` normalized by
  `trapz(dVc_by_dzdOmega * user_normal)` over $\pm 5\sigma_z$ — i.e.
  $\mathcal{N}(z;z_{\rm obs},\sigma_z^2)\,\frac{dV_c}{dz\,d\Omega}$, renormalized per galaxy:
  identical in form to CHIMERA Eq. 12 and to our `volume_deconv`.
- `'gaussian_nocom'` (`catalog.py:1053-1071`): the **bare Gaussian without the comoving-volume
  prior** — the P1 pitfall — exists, but only as an explicit opt-in; it is nobody's default.

Defaults: the legacy `galaxy_catalog.calc_dN_by_dzdOmega_interpolant` uses `ptype='uniform'`
(`catalog.py:1330`); the current pixelated-file pipeline `calculate_interpolant_files` uses
`ptype='gaussian'` (`catalog.py:470`). Both defaults are volume-weighted.

### 2.2 Out-of-catalogue completion sky handling (P2: per-line-of-sight, magnitude-threshold based)

The per-pixel in-catalogue interpolant is an *effective number density per steradian*
(`catalog.py:526-531`): each galaxy contributes its Schechter-luminosity weight
(`absM_rate.evaluate(sch_fun, Mv)`) times the volume-weighted z-kernel, divided by the pixel
solid angle `dOmega_sterad`. At evaluation time
(`catalog.py:1456-1513`, `effective_galaxy_number_interpolant`):
$$\frac{dN_{\rm gal}^{\rm eff}}{dz\, d\Omega}(z,\hat\Omega) = \underbrace{\texttt{gcpart}(z,\hat\Omega)}_{\text{catalogue interpolant}} + \underbrace{\phi_{\rm miss}\!\bigl(M_{\rm thr}(z,\hat\Omega)\bigr)\, \frac{dV_c}{dz\, d\Omega}}_{\texttt{bgpart},\ \text{completion}},$$
where $M_{\rm thr}(z,\hat\Omega)$ comes from a per-pixel apparent-magnitude-threshold map
(`calc_Mthr`, `catalog.py:688-720`) and $\phi_{\rm miss}$ is the Schechter number density of
galaxies fainter than threshold (`background_effective_galaxy_density`). The completion is thus
resolved **per line of sight** — strictly finer than an isotropic $1/(4\pi)$ prefactor — and the
sky integral is performed by Monte Carlo over the PE posterior samples' `sky_indices`
(`rates.py:317`), so the catalogue and completion terms share the same per-steradian measure by
construction. Both terms are per steradian; no $4\pi$ mismatch is possible.

### 2.3 In-catalogue normalization structure (P3: global, with an acknowledged inconsistency)

Numerator weights per PE sample (`rates.py:317-323`):
`log_weights = rate.log_evaluate(z) + log(dNgaleff) − log1p(z) − log|ddL/dz| − log(prior)`.
The normalization is *global*: the hierarchical likelihood divides by
$N_{\rm exp} \propto \xi(\lambda)$ evaluated on injections. But the injection weights
(`rates.py:333-353`, `log_rate_injections`) use
```python
# We assume the galaxy catalog empty to apply completeness correction     (rates.py:351)
dNgaleff = self.catalog.sch_fun.background_effective_galaxy_density(-inf, z) * dVc_by_dzdOmega_at_z(z)
```
i.e. the **total** (complete, sky-isotropic) Schechter density instead of the anisotropic
catalogue+completion sum used in the numerator, and the docstring says so:
*"FIX-ME this method should be made consistent with the galaxy catalog below"* (`rates.py:336`).
This is exact only insofar as the catalogue term sky- and realization-averages to the total
density; it is the same *class* of numerator/denominator inconsistency our commission chased
(global-vs-local normalization), here acknowledged in-code as an approximation. It is a
smooth $\mathcal{O}(\delta n/\bar n)$ effect on the amplitude of $\xi(\lambda)$, not a
$\sigma_z$-dependent rail mechanism, because both numerator kernel and denominator carry the
same volume prior in $z$.

### 2.4 Published validation regime in $\sigma_z$

The icarogw 2.0 paper (Sec. 7.2) validates the catalogue mode on **real data**: the GWTC-3
42-BBH $H_0$ analysis against `gwcosmo` (Abbott et al. 2021, arXiv:2111.03604) using GLADE+
K-band, Schechter parameters $(M_{\rm min}, M_{\rm max}, \alpha, \phi^*) = (-27.85, -19.84,
-1.09, 0.03\,{\rm Mpc}^{-3})$ at $H_0 = 67.7$, plus GW190814 against gwcosmo's newer LOS
version (Gray et al. 2023) at 3 deg² pixels — *"excellent agreement."* The per-galaxy
$\sigma_z$ values are inherited from GLADE+ (Dálya et al. 2022): $\sim 1.5\times10^{-4}$
(spectroscopic) up to a few $\times 10^{-2}$ (photometric), at host redshifts $z \lesssim 0.1$
where the constraint is catalogue-driven; the LOS construction details are deferred to
Mastrogiovanni et al. 2023, PRD 108, 042002 (arXiv:2305.10488). There is no published icarogw
validation at $\sigma_z/(1+z) \sim 0.05$ with a *dominant* photo-z population comparable to
CHIMERA's forecast regime; the GWTC-3 result is a cross-code consistency check, not a
calibration (P–P) test.

**CONCLUSION (icarogw): pitfall correctly handled** in both default kernels (P1 ✓ — the bare
Gaussian survives only as opt-in `'gaussian_nocom'`); completion is per-LOS and
measure-consistent (P2 ✓); normalization is global with an *in-code acknowledged*
numerator/denominator approximation (P3: approximation, `rates.py:336,351` — not the photo-z
pitfall, but the same structural family our audit flagged).

---

## 3. DarkSirensStat (Finke et al. 2021, arXiv:2101.12660) — brief

**P1 — handled, with a heavier-tailed likelihood model.** Galaxy redshift pdfs are modeled as
bounded Keelin (metalog) distributions (`keelin.py:47-113`) built from
$\{z-3\sigma, z-\sigma, z, z+\sigma, z+3\sigma\}$ quantiles. Before use, every pdf is
multiplied by the comoving-volume prior and refitted: `galCat.py:169-224`,
`include_vol_prior`, with the Jacobian
```python
jac = fiducialcosmo.comoving_distance(zGrid).value**2 / fiducialcosmo.H(zGrid).value   # galCat.py:180
```
i.e. $\propto d_C^2(z)/H(z) \propto dV_c/dz$ at fixed fiducial $(H_0,\Omega_m)=(70,0.3)$,
invoked from `GLADE.py:277-281` (comment: *"Estimate galaxy posteriors with contant-in-comoving
prior"*). Same $H_0^{-3}$-cancellation argument as CHIMERA. This is the earliest of the three
implementations of the volume-deconvolved photo-z posterior (2021) and is cited by Borghi et
al. 2024 as the source of the completion formalism.

**P2 — per-LOS completion.** The homogeneous part is evaluated at the GW posterior samples'
sky positions: `galCat.py:428-457` `eval_hom` returns $1 - \mathrm{conf}(P_{\rm comp}(\theta,\phi,z))$
per sample, and `GWgal.py:255` computes
$\mathcal{L}_{\rm hom} = (H_0/70)^3\, \langle\, \mathrm{jac}\cdot\psi(z)\cdot \mathrm{eval\_hom}\,\rangle_{\rm samples}$ —
completion follows the GW sky posterior weighted by local incompleteness, with the
mask-completeness of Finke et al. Sec. 2.5 ("mix" interpolation between multiplicative and
homogeneous completion, `galCat.py:461-468`).

**P3 — density-ratio normalization; homogeneous $\beta$.** Catalogue weights are divided by the
per-galaxy completeness (multiplicative completion, `galCat.py:338`), damped by the confidence
interpolator (`:340`), and normalized by the reference comoving density
`_comovingDensityGoal` (`:348`) — i.e. the catalogue term is a local density *ratio* to
$\bar n$, unit-consistent with the homogeneous term. The selection denominator is
catalogue-free and homogeneous: `betaHom.py:43-46`,
$\beta(H_0) = (H_0/70)^3 / [\Xi(z)(1+z)]^3$ (Finke et al. Eq. 2.81) — again the "global,
catalogue-agnostic denominator" structure, chosen deliberately and discussed in the paper.

**Validation regime.** `GLADE.py:258-259`: $\sigma_z = 1.5\times10^{-4}$ for spectroscopic
entries (`flag2==3`) and $\sigma_z = 1.5\times10^{-2}$ for photometric ones — applied to the
real GLADE 2.4 catalogue in the O2+O3 dark-siren analyses of Finke et al. 2021 ($H_0$ and
$\Xi_0$). So DarkSirensStat has been run in a mixed spec/photo regime at the
$\sigma_z \sim 10^{-2}$ level, with the volume prior always on.

**CONCLUSION (DarkSirensStat): pitfall correctly handled** (P1 ✓ volume-prior-convolved Keelin
posteriors; P2 ✓ per-LOS completion; P3 — density-ratio catalogue term with a deliberate
homogeneous denominator).

---

## 4. Synthesis — comparison table and implications for Paper A

| question | CHIMERA v1/v2 | icarogw 2.0 | DarkSirensStat |
|---|---|---|---|
| P1 photo-z kernel | $\mathcal{N}\cdot dV_c/dz$, per-galaxy renorm (Eq. 12; `EM.py:285-288`, `catalog.py:293-302`) | same, `'gaussian'`/`'uniform'` defaults (`catalog.py:1032,1039`); bare Gaussian opt-in only | Keelin $\times\, d_C^2/H$, refit (`galCat.py:169-224`) |
| bare-z-pdf numerator? | **no** | **no** (unless `'gaussian_nocom'` forced) | **no** |
| P2 completion sky | isotropic per pixel $\equiv 1/(4\pi)$ measure (`Likelihood.py:254,266`) | per-LOS $M_{\rm thr}(z,\hat\Omega)$ map (`catalog.py:1511`) | per-LOS mask completeness (`galCat.py:428-457`) |
| P3 num/denom consistency | consistent by construction in v2 (`pop_wrapper.py:103-112`) | **acknowledged FIX-ME**: empty-catalogue denominator (`rates.py:336,351`) | deliberate homogeneous $\beta$ (`betaHom.py:43-46`) |
| validated $\sigma_z$ | $0.001(1+z)$ and $0.05(1+z)$, mocks (Eq. 22) | GLADE+ per-galaxy ($10^{-4}$–few$\times10^{-2}$), GWTC-3 vs gwcosmo | GLADE $1.5\times10^{-4}$ / $1.5\times10^{-2}$, O2–O3 |
| photo-z "deferred to future work"? | **No** — headline regime; only sky-varying $\sigma_z$ and $\Omega_m$-interpolant bias deferred | No (kernel is core, LVK-reviewed) | No |

**Implications for our de-rail claim.** (i) All three published codes implement precisely the
volume-weighted photo-z posterior that our `volume_deconv` mode introduces
(`bayesian_statistics.py:804-827`); none marginalizes hosts with a bare Gaussian z-pdf. Our
pre-fix `global` mode was therefore *outside* published practice, and the fix brings us *into*
it — this independently corroborates the "curable normalization artifact" verdict. (ii)
CHIMERA's per-pixel completion with the uniform $1/N_{\rm pix} \equiv \Delta\Omega/(4\pi)$
measure is structurally the same isotropic-sky completion as commit `cb16142`. (iii) CHIMERA's
published photometric result ($\sigma_z/(1+z)=0.05$, complete catalogue) shows the correctly
weighted kernel yields a *broadened, unbiased* $H_0$ (factor 3–9 in $\sigma_{H_0}$, no rail),
with the residual danger being $H_0$–mass-scale degeneracy — consistent with our finding that
information starvation manifests as widening, not railing, once the normalization is fixed.
(iv) The one acknowledged inconsistency in the field (icarogw's empty-catalogue injection
denominator) is of the global-normalization family, supporting the commission's follow-up item
of checking our global selection denominator $D(h)$ against the real GLADE+ $\beta_G$ sum.

---

## VERDICT

**CONFIRMED** — the implementations of CHIMERA (v1.0.0 = paper version, and v2.2 HEAD),
icarogw (HEAD `f8277b7`), and DarkSirensStat match the volume-weighted photo-z host
marginalization derived in the respective papers (CHIMERA Eqs. 11–12; icarogw
"EM likelihood × UCV prior", LVK-reviewed; Finke et al. 2021 comoving-prior posteriors), and
none of the three exhibits the bare-z-pdf pitfall in its default or published configuration.
The commission's belief that CHIMERA defers photometric redshifts to future work is
**refuted**: Borghi et al. 2024 treats $\sigma_z/(1+z) = 0.05$ photometric catalogues as one of
its two headline regimes (Eq. 22), with the UCV kernel active (verified in code at
`EM.py:285-288` and reproduced in v2.2 at `catalog/catalog.py:293-302`).

Caveats recorded (not verdict-changing):
1. **icarogw `rates.py:336,351`** — in-code acknowledged numerator/denominator inconsistency
   (injection weights assume an empty catalogue); approximation, not the photo-z pitfall.
2. **CHIMERA `EM.py:124-126` / `catalog.py:207-214`** — per-pixel (not global) normalization of
   $p_{\rm cat}$ drops relative galaxy counts between pixels, in apparent tension with the
   global normalization of paper Eq. 11; UNCERTAIN whether intended — resolvable only by
   asking the authors or reproducing a two-pixel toy comparison against Eq. 11.
3. **icarogw `catalog.py:1053-1071`** — `'gaussian_nocom'` provides the bare-Gaussian kernel as
   an explicit opt-in; any analysis selecting it would reproduce the pitfall.
