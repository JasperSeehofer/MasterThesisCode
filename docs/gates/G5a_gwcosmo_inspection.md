# G5a — gwcosmo inspection: how the LVK dark-siren code treats the two normalization pitfalls

**Date:** 2026-07-02.
**Inspected source:** `https://git.ligo.org/lscsoft/gwcosmo.git`, full history cloned to `/tmp/gwcosmo_inspect`; current `master` HEAD `5a4f8c2d56dc` (2026-06-04, "Merge branch 'master'", post-v3.0.0), and the O3-era tag `v1.0.0` (the code behind Gray et al. 2020/2022 and the GWTC-3 cosmology paper).
**Papers:** Gray et al. 2020 (arXiv:1908.06050, PRD 101, 122001), Gray et al. 2022 (arXiv:2111.04629, MNRAS 512), Gray et al. 2023 (arXiv:2308.02281, JCAP 12 (2023) 023), Turski et al. 2023 (arXiv:2302.12037, MNRAS).
**Context (this project's pitfalls, from `.planning/HANDOFF-DERAIL-CLUSTER-CONFIRM-20260702.md`):**
(P1) *host photo-z marginalization*: bare-Gaussian host-z numerator vs. volume-weighted ($dV_c/dz$-type) integrand, and whether the same prior is counted once across numerator and selection denominator (the mechanism behind the H₀ rail, cured by `normalization_mode ∈ {local_ratio, volume_deconv}`);
(P2) *completion-term sky handling*: the missing isotropic $1/4\pi$ factor in the out-of-catalogue term (commit `cb16142`), which alone flips the rail 0.86 → 0.60.

Two gwcosmo generations are relevant and are treated separately throughout: **v1.x** (per-event pixelated likelihood, Gray et al. 2020 App. A + Gray et al. 2022) and **v2/v3** (precomputed line-of-sight (LOS) redshift prior + injection-based selection, Gray et al. 2023; the current `master`).

---

## 1. Architecture summary

**v1.x** (`gwcosmo/gwcosmo.py` at tag `v1.0.0`): per event, per healpix pixel $i$,

$$
\mathcal{L}_i(H_0) \;=\; \frac{p(x|G,H_0)}{p(D|G,H_0)}\,p(G|D,H_0)
\;+\; \frac{p(x|\bar G,H_0)}{p(D|\bar G,H_0)}\,p(\bar G|D,H_0)
\;+\; \frac{p(x|O,H_0)}{p(D|O,H_0)}\,p(O),
$$

summed over pixels (`gwcosmo.py:1035-1040`, method `likelihood`), where $G$/$\bar G$/$O$ denote host-in-catalogue / host-beyond-threshold / pixel-with-no-catalogue-support. This is Gray et al. 2020 Eq. (9) with the appendix expansions (A.9, A.10, A.14, A.19 ff.), extended to pixels in Gray et al. 2022.

**v2/v3** (current `master`): a per-pixel LOS redshift prior

$$
p(z|\Omega_i, s, \Lambda) \;\propto\; \frac{p(G|\Omega_i)}{N_{\rm gal}(\Omega_i)}\sum_{k\in i} p(z|\hat z_k)\, p(s|M(z,m_k,\Lambda))
\;+\; \int_{M(m_{\rm th}(\Omega_i),z)}^{M_{\max}} p(z|\Lambda)\,p(M|\Lambda)\,p(s|M)\, dM
$$

(Gray et al. 2023 Eqs. 2.8–2.22; implemented in `gwcosmo/prior/LOS_redshift_prior.py:234-292`, `create_redshift_prior`) is precomputed per pixel and stored; the event likelihood is a skymap-weighted sum of per-pixel numerators divided by an injection-based selection integral that reuses the **same** LOS prior (`gwcosmo/likelihood/dark_siren_likelihood.py`).

---

## 2. Q1 — Host photo-z treatment in the numerator, and the "$dV_c$ counted once" question

### 2.1 The in-catalogue z-integrand is the *bare* catalogue z-pdf — no volume prior

Current `master`, `gwcosmo/prior/LOS_redshift_prior.py:257-270` (in-catalogue part of the LOS prior):

```python
pz_G = np.zeros(len(z_array))
for i in range(len(self.zs)):
    ...
    trunk = truncnorm.pdf(z_array3, low_z_lim, high_z_lim, self.zs[i], self.sigmazs[i])
    interpolate_trunk = interp1d(z_array3, trunk, bounds_error=False, fill_value=0)
    kill_weights = M_mdl(self.ms[i],dl_array,Kcorr=Kcorr) < Mmax
    pz_G += interpolate_trunk(z_array) * self.luminosity_weights(M_mdl(self.ms[i],dl_array,Kcorr=Kcorr)) * kill_weights
```

Each galaxy contributes a truncated Gaussian $p(z|\hat z_k) = \mathcal{N}_{[0,z_{\max}]}(z;\hat z_k,\hat\sigma_k)$ times a luminosity/host weight $p(s|M)$ — **no $dV_c/dz$ or population $p(z)$ factor** multiplies the z-integrand of catalogued galaxies. The population prior enters only the completion term: `uninformative_host_galaxy_prior` (`LOS_redshift_prior.py:189-209`) returns `self.zprior(z) * self.luminosity_prior(M,H0) * self.luminosity_weights(M)`, where `zprior = cosmology.p_z` is "Uniform in comoving volume distribution of galaxies", $p(z)\propto \frac{(\int_0^z dz'/h)^2}{h(z)}$ i.e. $\propto dV_c/dz$ up to constants (`gwcosmo/utilities/cosmology.py:328-368`).

The same is true in **v1.0.0**: `gal_nsmear` (`gwcosmo.py:1680-1738`) draws MC samples from `truncnorm.rvs(a, 5, loc=z, scale=sigmaz, ...)` with **uniform weights** `count = nsmear` per galaxy; `pxD_GH0` (`gwcosmo.py:254-303`) then weights those samples by

```python
tempnum[k] = np.sum(numinner*tempsky*Lweights*zweights*normsamp)   # numerator, p(x|G,H0)
tempden[k] = np.sum(deninner*Lweights*zweights*normsamp)           # denominator, p(D|G,H0)
```

where `zweights = self.zrates(sampz)` is the rate-evolution factor $p(s|z)\propto R(z)/(1+z)$ only — again **no** $dV_c/dz$.

This matches the papers exactly. Gray et al. 2023 Eq. (2.8)–(2.9): $p(z,m|G,\Omega_i,I) = \frac{1}{N_{\rm gal}(\Omega_i)}\sum_k p(z|\hat z_k)\,\delta(m-\hat m_k)$ with $p(z|\hat z_k)=\mathcal{G}(z-\hat z_k;\hat\sigma_k)$, and — decisive for interpretation — **footnote 10** states that "the galaxy measurements provided in the catalogue used for the analyses later in this paper are posteriors": gwcosmo *defines* the catalogue z-pdf as the full posterior on the galaxy's true redshift, so by construction no further prior may be applied. Gray et al. 2020 makes the same move: Eq. (A.9) is the delta-function-z ratio of sums and Eq. (A.10) is the same expression with $\delta(z-z_i)\to p(z_i)(z)$ ("in the case the galaxies in the catalogs are provided along with their redshift uncertainties $p(z_i)$, these can be implemented in the above equations as", 1908.06050 App. A.2a). Turski et al. 2023 (§3) operationalizes the posterior interpretation correctly: they build empirical error models $p(z|z_{\rm photo})$ by calibrating against overlapping spectroscopic samples ($\Delta z = z_{\rm photo}-z_{\rm spec}$ residuals), which *is* the data-driven application of the true redshift prior — the honest resolution of the posterior-vs-likelihood ambiguity.

**Relation to this project's P1:** gwcosmo makes the *same modelling choice* as this project's railed production numerator — a bare (trunc-)Gaussian in $z$ with no $dV_c/(1+z)$ weight. The D2 calibration result here (bare-Gaussian numerator coverage ≈ 0%, MAP bias $\propto -\sigma_z^2$, `results/commission_20260701/scratch/d2/NOTE_calibration_findings.md`) therefore applies to gwcosmo's structure *if and only if* the catalogue $\hat\sigma_k$ are likelihood widths rather than genuine posteriors. gwcosmo is protected by (i) the explicit posterior declaration (assumption, not enforcement), (ii) the σ_z regime of its published results (§5 below), and (iii) — crucially for the rail mechanism specifically — the consistency property established next.

### 2.2 Is the same prior applied in the selection denominator? Yes — literally the same object

**v2/v3:** `gwcosmo/likelihood/dark_siren_likelihood.py:620-652` (`log_likelihood_denominator_single_event`):

```python
z_prior = torch_evaluatable_interpolator(self.z_array, self.zprior_full_sky * self.ps_z_array, ...)
z_prior_norm = torch.trapezoid(z_vals, self.z_array)
self.injections.update_VT(self.cosmo, self.mass_priors, z_prior, z_prior_norm)
...
log_den = torch.log(self.injections.gw_only_selection_effect())
```

and `gwcosmo/injections.py:142-179` (`update_VT`) weights every found injection by that prior: `log_numer = torch.log(z_prior_new) + m_prior.log_joint_prob(...)`; `self.VT_fraction = self.VT_sens / z_prior_norm_new`. Here `zprior_full_sky` is the **catalogue-informed** sky-averaged LOS prior (`get_zprior_full_sky`, dataset `combined_pixels`), i.e. the identical $p(z|\Omega)$ used in the numerator, not a smooth $dV_c/dz$. The normalization is counted once, explicitly: `log_combined_event_likelihood` (`dark_siren_likelihood.py:655-671`) does `den = den_single * self.n_events` and `num = -self.n_events * zprior_norm_log`, carrying the prior's norm into the numerator so that the ratio uses one normalized prior. Paper-side: "the model used for $p(z|\Lambda,I)$ is the LOS redshift prior of Eq. 2.22" inside the injection-reweighted $P_{\rm det}$ (2308.02281 §2.2, around Eqs. 2.23–2.26).

**v1.0.0:** the in-catalogue selection term $p(D|G,H_0)$ (`tempden`, quoted above) is computed as a sum over the **same truncnorm samples with the same weights** as the numerator — the ratio $p(x|G,H_0)/p(D|G,H_0)$ is a local ratio of sums over identical galaxy z-pdfs (Gray et al. 2020 Eqs. A.9/A.10). The volume prior $p(z)\propto dV_c/dz$ appears once each in $p(x|\bar G,H_0)$, $p(D|\bar G,H_0)$, and $p(G|D,H_0)$ (`px_zH0_times_pz_times_ps_z_times_pM_times_ps_M`, `gwcosmo.py:381-411`, used in `pGB_DH0`/`px_BH0`/`pD_BH0`, `gwcosmo.py:415-533`) — always pairwise, in matched numerator/denominator ratios.

**Answer to Q1:** numerator z-integrand = bare catalogue z-pdf (declared posterior); population $dV_c/dz$ prior applied only to completion/out-of-catalogue terms; and the selection denominator uses the *identical* (catalogue-informed) prior with a single, explicitly-carried normalization. The "$dV_c$ counted once" consistency **holds by construction** in both generations. The H₀-dependent normalization mismatch that produced this project's rail cannot arise in gwcosmo's structure; the *shape*-level bare-pdf question (Eddington-in-z) is handled by assumption, not by validation (see §5).

---

## 3. Q2 — Completion/out-of-catalogue sky handling and the $1/4\pi$ bookkeeping

**v2/v3: full-sky pixelated line-of-sight marginal.** Each healpix pixel carries its own magnitude threshold $m_{\rm th}(\Omega_i)$ and its own completion integral (`LOS_redshift_prior.py:277-290`: `pz_Gbar1` integrates the Schechter function from $M(m_{\rm th}(\Omega_i),z)$ to $M_{\max}$ against `zprior`). The event numerator weights each pixel's LOS prior by the normalized GW skymap probability: `low_res_skyprob = low_res_skyprob/np.sum(low_res_skyprob)` then `zprior_times_pxOmega[i,:] += np.sum(zpriors * low_res_skyprob[hi_res_pixel_indices][:,None], axis=0)` (`dark_siren_likelihood.py:252-317`). The dark-host sky position is therefore marginalized as a **pixelated sum along lines of sight**, not as an isotropic $1/4\pi$ approximation.

**Where the $1/4\pi$ lives.** In the derivation, the uniform sky prior is $p(\Omega_j|I) = 1/N_{\rm pix}$ per equal-area pixel (the discretized $\Delta\Omega/4\pi$), and Gray et al. 2023 states it explicitly: "The $p(\Omega_j|I)$ terms are recognised as being identical between same-sized pixels, so come out the front and **cancel in numerator and denominator**" (2308.02281 §2.1, after Eq. 2.5). In code, the sky-averaged prior used in the denominator is built as an explicit equal-area average, i.e. the discrete form of $\bar p(z) = \int \frac{d\Omega}{4\pi}\, p(z|\Omega)$: `bin/gwcosmo_compute_redshift_prior:40-58` accumulates `combined_pixels += p_of_z` over all $N_{\rm pix}$ pixels and then divides `combined_pixels /= (denom * npix)`.

**v1.0.0** keeps all sky measures as *fractions of the sky probability*: `self.pixel_area_hi_res = 1./hp.nside2npix(self.hi_res_nside)` (`gwcosmo.py:869`) is the pixel's prior mass $\Delta\Omega/4\pi$; the GW skymap enters as normalized per-pixel probability `px_Omega = hi_res_skyprob[idx]` ($\sum_i = 1$); and in the galaxy term `tempsky = self.skymap.skyprob(ra,dec) * self.skymap.npix` (`gwcosmo.py:279`) is the dimensionless ratio $\frac{p(\Omega_{\rm gal}|x)}{1/(4\pi)}$ of the sky posterior density to the isotropic prior density. Because *every* sky quantity is expressed as a probability (or probability ratio) rather than a per-steradian density, a dangling $4\pi$ mismatch between the galaxy term and the completion term is structurally impossible: no term is ever expressed in steradian units. The empty-pixel term makes the pairing visible (`gwcosmo.py:987-990`): `pxO = temp_pxO * px_Omega` (GW probability in pixel) against `pDO = temp_pDO * pixel_area_hi_res` and `pO = pixel_area_hi_res` (prior fraction of sky) — likelihood weighted by data-probability, selection weighted by prior-probability, ratio dimensionless.

**Answer to Q2:** full-sky pixelated marginal (v2/v3) resp. pixel-fraction bookkeeping (v1). The isotropic $1/4\pi$ appears only as the uniform per-pixel prior $1/N_{\rm pix}$, which is either explicitly cancelled (paper) or explicitly applied via `/npix` (code). The class of error this project fixed in `cb16142` — an out-of-catalogue term carrying different implicit solid-angle units than the in-catalogue term — **cannot arise** in gwcosmo's structure, because both terms live inside the same per-pixel prior with a common $1/N_{\rm pix}$.

One cosmetic asymmetry was found and checked: per-pixel datasets are stored *without* the empty-catalogue normalization `denom`, while `combined_pixels` is divided by `denom * npix` (`bin/gwcosmo_compute_redshift_prior:46-60`). Since the LOS file is computed once at a fixed reference cosmology, `denom` is a $\Lambda$-independent constant and shifts $\ln\mathcal{L}$ by $N_{\rm ev}\ln(\rm denom)$ uniformly — no effect on any posterior. Noted as bookkeeping style, not a bug.

---

## 4. Q3 — In-catalogue normalization: local ratio-of-sums or global denominator?

**v1.0.0 is literally the local ratio-of-sums** of Gray et al. 2020 Eqs. (A.9)/(A.10): within each pixel, `full_pixel` (`gwcosmo.py:880-926`) computes `likelihood = (pxG/pDG)*pG + (pxB/pDB)*pB`, where `pxG` and `pDG` are sums over the *same* galaxies of the *same* smeared samples (`pxD_GH0_multi`, `gwcosmo.py:306-379`; both divided by the same `nGal`). The denominator of the in-catalogue term is local to the pixel's galaxy set — precisely the estimator this project adopted as fix #2 (`normalization_mode="local_ratio"`, commit `6d4c4e1`, following Gray A.9/A.10).

**v2/v3 moves to a global selection denominator, but keeps it consistent.** The in-catalogue sum inside each pixel's LOS prior is normalized by a **local effective galaxy count**: `pz_G = pz_G/self.galaxy_norm` (`LOS_redshift_prior.py:271-272`), where `galaxy_norm` is $\sum_k \left[\Phi_k(z_{\rm cut}) - \Phi_k(\text{Mmax-excluded region})\right]$, the truncnorm CDF mass each galaxy retains below the cuts (`gwcosmo/maps/create_norm_map.py:78-111`, `calc_norm`) — i.e. $N_{\rm gal}(\Omega_i)$ of Gray et al. 2023 Eq. 2.8 corrected for pdf mass leaking past $z_{\rm cut}$. The single global denominator $P_{\rm det}(\Lambda)$ then reuses the identical sky-averaged LOS prior (§2.2), so the global variant is the mathematically equivalent form $\mathcal{L}\propto \frac{\sum_i P_i \int p(x|z)\,p(z|\Omega_i)\,\psi(z)\,dz}{\int p(D|z,\Lambda)\,\bar p(z)\,\psi(z)\,dz}$ with one shared normalization — not the inconsistent global-denominator variant that railed in this project (numerator prior absent from the denominator).

Notably, Gray et al. 2023 §2.1.4 documents that they themselves hit a pathology of *local* normalization: with per-(fine-)pixel $1/N_{\rm gal}(\Omega_i)$, small-number statistics in the high-resolution limit down-weight galaxies that happen to share a pixel (their Fig. 1: "two of your galaxies would be down-weighted relative to the other two, simply due to small number statistics"), and empty pixels retain non-zero $p(G|\Omega_i)$. Their mitigation: compute $N_{\rm gal}(\Omega_i)$ and $m_{\rm th}$ on a coarse map ($n_{\rm map}$) and divide by the number of sub-pixels, making the estimator "robust in the limit of infinitely high resolutions" (2308.02281 §2.1.4 and footnote 12; implemented in `LOS_redshift_prior.py:171-182` via the low-resolution `galaxy_norm` map). This is independent confirmation that the field regards the in-catalogue normalization as a live, non-trivial design point — the same territory as this project's `global`/`local_ratio` distinction, encountered from the opposite direction.

**Answer to Q3:** v1 = local ratio-of-sums (their own A.9/A.10); v2/v3 = local in-catalogue normalization (coarse-map effective counts) inside a globally-normalized likelihood whose selection denominator carries the same catalogue-informed prior. Neither generation implements the inconsistent global-denominator variant.

---

## 5. Q4 — In what σ_z regime has gwcosmo actually been validated?

The GLADE+ preparation script shipped with gwcosmo (`scripts_galaxy_catalogs/GLADE+/create_glade+.py:60-95`) assigns:
$\sigma_z^{\rm spec} = 1.5\times10^{-4}$ (absolute), $\sigma_z^{\rm WISE} = 0.04\,(1+z)$, $\sigma_z^{\rm photo} = 1.5\times10^{-2}$ (absolute), $\sigma_z^{\rm HyperLEDA} = 0.36\,z$, each convolved with the peculiar-velocity term. So redshifts with $\sigma_z/z$ up to $\sim0.36$ (HyperLEDA) — and formally $\mathcal{O}(1)$ for photo-z galaxies at $z \lesssim 0.015$ — *pass through* the production machinery. Passing through is not validation. The published validation record:

1. **Gray et al. 2020 (1908.06050):** the MDC that established the method used catalogues where "the galaxies are generated with **no redshift uncertainties** or peculiar velocities", and the text states the analyses "ignore these crucial redshift uncertainties altogether... left aside for possible future study" (§ MDC description). Known-truth recovery validated **only at $\sigma_z = 0$**.
2. **Gray et al. 2022 (2111.04629):** introduces the pixelated method; Gaussian catalogue σ_z marginalized (§2: "The redshift uncertainty of each galaxy is assumed to be Gaussian... and is marginalised over"); applied to real O3 events. No known-truth test with nonzero σ_z.
3. **Gray et al. 2023 (2308.02281):** LOS method; validation = cross-code agreement with icarogw and with previous gwcosmo on real GWTC-3 data, plus injection-set selection checks (§2.2, §4). The z-grid is *engineered* around GLADE+'s σ_z structure (their Fig. 4; `create_zarray` in `gwcosmo/utilities/zprior_utilities.py`), but no mock catalogue with known cosmology and large σ_z is analysed. Their conclusions flag exactly this frontier: non-Gaussian redshift uncertainties "may be of high importance in the regime where the galaxy catalogue information becomes a more dominant contributor to the measurement of $H_0$."
4. **Turski et al. 2023 (2302.12037)** — the closest approach to the large-σ_z regime, on **real data** (46 GWTC-3 events, 2MPZ/WISC photo-z catalogues, σ_z ≈ 0.015–0.04): comparing Gaussian vs. modified-Lorentzian vs. no-uncertainty models shifts $H_0$ at a level "of the same order of magnitude as some of the principal sources of systematic errors" in the LVK result but small against the statistical error. Their σ-boost experiment reaches this project's regime: Gaussian σ inflated ×2 and ×5 ("amounting to between 0.1 up to 0.25" absolute, i.e. $\sigma_z/z \gtrsim 0.5\text{–}1$ at WISC redshifts) — and the observed behaviour is that "the catalogue is uninformative and the posterior of $H_0$... is pushed towards the empty catalogue case". Because this is real data, there is no truth to test bias against; it demonstrates information wash-out, not calibration.

**Answer to Q4:** gwcosmo's known-truth validations are at $\sigma_z = 0$ (Gray 2020) or code-vs-code on real data (Gray 2022/2023); the only excursion toward $\sigma_z/z \sim 0.7$ is Turski et al.'s ×5-boost on real data, which shows collapse to the completion-dominated posterior and cannot detect a bias. **No published gwcosmo test validates unbiased $H_0$ recovery in the $\sigma_z/z \sim 0.7$, in-catalogue-information-dominated regime** — the regime of this project's EMRI/GLADE+ analysis. (The 2025 Blinded MDC, arXiv:2504/2506-era, tests the mass-spectrum method's robustness, not photo-z; recent independent bias studies, e.g. arXiv:2503.18887, address host-weighting/incompleteness, not the z-pdf prior question.)

---

## 6. Conclusion — is this project's pitfall (a) present, (b) absent because done correctly, or (c) not applicable in gwcosmo?

Split by pitfall, because the answer differs:

- **P1a, the rail mechanism (H₀-dependent normalization inconsistency between numerator z-marginalization and selection denominator):** **(b) absent — structurally excluded.** In both generations the selection denominator is built from the *same* prior object as the numerator (v1: same smeared samples and weights in `tempnum`/`tempden`, `gwcosmo.py:300-303`; v2/v3: `zprior_full_sky` fed into `update_VT` with the norm explicitly carried to the numerator, `dark_siren_likelihood.py:632-671`), and the papers state the cancellations explicitly. The "$dV_c$ counted once" property is a design invariant of gwcosmo, not an accident.
- **P1b, the bare photo-z pdf itself (Eddington-in-z shape bias when catalogue σ_z are likelihood widths):** **(a)-adjacent — the same modelling choice is present**, guarded by an *assumption* (Gray 2023 footnote 10: catalogue redshifts are posteriors) rather than by code, and **unvalidated in the regime where it matters** (§5). In gwcosmo's published regime (completion-dominated posteriors; spec-z-dominated K-band GLADE+; σ_z effects subdominant to population systematics per Turski et al.) the consequence is negligible; in a catalogue-dominated, $\sigma_z/z\sim0.7$ analysis it would not be. Turski et al.'s spec-z-calibrated $p(z|z_{\rm photo})$ is the field's correct-prior escape route, equivalent in intent to this project's `volume_deconv`.
- **P2, the $1/4\pi$ completion factor:** **(b)/(c) — cannot arise.** All sky measures are probabilities per pixel (fractions of $4\pi$) on both sides of every ratio; the uniform $1/N_{\rm pix}$ prior cancels explicitly. gwcosmo never mixes per-steradian densities with per-sky probabilities, which is the unit mismatch this project's `cb16142` repaired.
- **P3 (in-catalogue normalization):** **(b) absent**, with the instructive addendum that Gray et al. 2023 §2.1.4 independently documents and engineers around a *different* failure mode of the same normalization (small-number statistics of local $1/N_{\rm gal}$), confirming this design point is a recognized hazard in the field.

**Implication for Paper A:** the paper is **not** a single-pipeline case study, but its field-relevant claim must be stated precisely. The *specific* rail artifact (inconsistent normalization) is absent from gwcosmo and should be framed as a hazard for *independently implemented* dark-siren pipelines (of which LISA/ET-era pipelines will be many) — with gwcosmo as the positive example of the consistency invariant. The *bare-z-pdf/Eddington* component, by contrast, is a live, present-by-assumption issue in the flagship code: gwcosmo applies the identical bare-Gaussian choice, its posterior-interpretation defence is untestable from the catalogue alone, and no known-truth validation exists at $\sigma_z/z\sim0.7$ — exactly the regime LISA dark sirens and next-generation photo-z catalogues (LSST, Euclid) will occupy, and one gwcosmo's own authors flag as important future territory (Gray et al. 2023 §6; Turski et al. 2023 §6). That is a legitimate, evidenced, field-level warning.

---

## VERDICT

**CONFIRMED** — the gwcosmo implementation matches its published derivations at every point inspected:

1. In-catalogue z-marginalization: bare truncated-Gaussian z-pdf, no volume weight — matches Gray et al. 2023 Eqs. 2.8–2.9 + footnote 10 and Gray et al. 2020 Eq. A.10 (`LOS_redshift_prior.py:257-270`; v1 `gwcosmo.py:1650-1676`, `254-303`).
2. Selection denominator reuses the same catalogue-informed prior with the normalization counted once — matches 2308.02281 Eqs. 2.23–2.26 (`dark_siren_likelihood.py:632-671`, `injections.py:163-179`).
3. Completion term: per-pixel $m_{\rm th}$-limited Schechter integral against $dV_c/dz$, uniform pixel prior $1/N_{\rm pix}$ cancelling/averaged explicitly — matches 2308.02281 Eqs. 2.14–2.22 (`LOS_redshift_prior.py:277-290`; `gwcosmo_compute_redshift_prior:40-58`; v1 `gwcosmo.py:869`, `987-990`).
4. In-catalogue normalization: v1 local ratio-of-sums per A.9/A.10; v2/v3 coarse-map effective counts per §2.1.4 (`create_norm_map.py:78-111`).

Non-deviations worth recording: (i) the per-pixel vs. combined `denom` asymmetry in `bin/gwcosmo_compute_redshift_prior:46-60` is a $\Lambda$-independent constant offset (harmless); (ii) the posterior interpretation of catalogue σ_z (Gray 2023 footnote 10) is an assumption of the derivation faithfully implemented — the derivation-vs-*reality* question it defers is precisely the subject of this project's D2 finding and remains open in the field for $\sigma_z/z\sim0.7$.
