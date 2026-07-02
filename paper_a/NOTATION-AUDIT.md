# NOTATION AUDIT — Paper A (MNRAS)

- **Date:** 2026-07-02
- **Base:** branch `paper/paper-a-draft`, draft commit `efe4b54` (audited working tree includes the concurrent bibliographer pass that resolved `MISSING:` citation keys; line numbers below refer to the post-audit working tree)
- **Scope:** all 16 files in `paper_a/sections/*.tex`; `main.tex` is GENERATED and was inspected read-only
- **Contract (owned by `sections/framework.tex`):** `h = H0/(100 km s^-1 Mpc^-1)`; `z` CMB-frame redshift; `sigma_z` photometric redshift uncertainty; `d_L` luminosity distance; `M` host MBH mass; `G`/`bar G` in-/out-of-catalogue hypotheses (Gray et al. 2020); `p_det` detection probability; `beta_G(h)` catalogue selection normalization; `dV_c/dz` comoving volume element.

Severity scale: **CRITICAL** = same symbol used for two different quantities, or notation-contract violation; **MAJOR** = load-bearing symbol used but never defined, or one quantity under two symbols; **MINOR** = formatting/typography.

---

## 1. Symbol table

First-definition location for every math symbol used in the paper. "Restated" = an equivalent re-definition elsewhere (checked consistent unless flagged in §2).

### Cosmology and global parameters

| Symbol | Meaning | Defined at | Notes |
|---|---|---|---|
| $h$ | $H_0/(100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$ | framework.tex:20–21 | contract ✓; also abstract.tex:13 |
| $H_0$ | Hubble constant | framework.tex:21 | only via $h$; never written "H0" |
| $z$ | CMB-frame redshift | framework.tex:21–22 | contract ✓ |
| $\sigma_z$ | per-galaxy photometric redshift uncertainty | framework.tex:22–23 | contract ✓; indexed variant $\sigma_{z,g}$ — see M5 |
| $d_L(z,h)$ | luminosity distance | framework.tex:26 (eq:dl) | contract ✓ |
| $E(z)$ | $H(z)/H_0$ | framework.tex:29 (eq:dl) | restated estimators.tex:64, appendix_volume_deconv.tex:60 (consistent) |
| $\Omega_\mathrm{m},\ \Omega_\Lambda$ | density parameters | framework.tex:29, 32 | |
| $E'(z)$ | $\mathrm{d}E/\mathrm{d}z$ | appendix_volume_deconv.tex:85 | |
| $I(z)$ | $\int_0^z \mathrm{d}z'/E(z')$ | appendix_volume_deconv.tex:61 | |
| $\mathcal{D}(z)$ | dimensionless distance $(1+z)I(z)$ | estimators.tex:64 | post-fix also appendix_volume_deconv.tex:93 (was $f(z)$ — see C5, fixed) |
| $A(z)$ | tabulated amplitude, $d_L = A(z)/h$ | coverage.tex:12 | harness-local; $= (c/100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})\,\mathcal{D}(z)$ |
| $h'$ | $h$ matching $d_L$ at wrong $\Omega_\mathrm{m}$ | budget.tex:43 | inline definition |
| $h_\mathrm{true}$ | injected truth (synthetic suite) | coverage.tex:12 | also appendix_volume_deconv.tex:117 |
| $h_\mathrm{ref}$ | shape-normalization point 0.72 | appendix_beta_g.tex:49 | |

### Events, data, and measurement model

| Symbol | Meaning | Defined at | Notes |
|---|---|---|---|
| $i$ | event index | framework.tex:40 | |
| $x_i$ | per-event GW data | framework.tex:40 | duplicate symbol $x_\mathrm{GW}$ — see M3 |
| $D_i$ | detection indicator | framework.tex:40–41 | |
| $\rho_\mathrm{thr}$ | SNR detection threshold ($=20$) | framework.tex:42 | reused appendix_beta_g.tex:41 ✓ |
| $\pi(h)$ | prior on $h$ | framework.tex:44 | |
| $N_\mathrm{det}$ | number of detected events | framework.tex:45 | bare $N$ used for event counts in pitfall.tex:76, postmortem.tex — minor drift |
| $p_i(h)$ | per-event likelihood $p(x_i \mid D_i, h)$ | framework.tex:50 (eq:posterior) | |
| $(\phi,\theta)$ | ecliptic azimuth / polar angle | framework.tex:59–60 | appendix_sky_marginal.tex:10–11 ✓ |
| $u$ | distance fraction $d_L/\hat d_L$ | framework.tex:61 | ✓ everywhere |
| $\hat d_L$ | maximum-likelihood distance | framework.tex:61–62 | |
| $\hat\mu_i$ | ML point $(\hat\phi_i, \hat\theta_i, 1)$ | framework.tex:62–63 | unindexed $\mu$ in appendix_sky_marginal.tex:16 (generic event) |
| $\Sigma_i$ | projected 3×3 covariance | framework.tex:63 | unindexed $\Sigma$ in pitfall.tex:46, appendix_sky_marginal.tex:19 |
| $\Sigma_{i,uu}$ | marginal distance-fraction variance | framework.tex:227 | unindexed $\Sigma_{uu}$: appendix_sky_marginal.tex:55, pitfall.tex:52 (post-fix; was $\Sigma_{22}$ — see M2, fixed) |
| $\sigma_{d_L}$ | per-event distance uncertainty | first use framework.tex:150 | never explicitly defined — see M7 |
| $\sigma_\phi, \sigma_\theta$ | sky-localization widths | pitfall.tex:52 | appendix_sky_marginal.tex:59 ✓ |
| $\Omega$ | sky position $(\phi,\theta)$ | framework.tex:73 | $\Omega_g$ galaxy sky position (estimators.tex:22) |
| $\mathrm{d}\Omega$ | solid-angle measure $\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\phi$ | framework.tex:82 | |
| $\omega$ | sky pair $(\phi,\theta)$ (split of $x$) | appendix_sky_marginal.tex:45 | appendix-local |
| $\mu_{\omega\mid u}, \Sigma_{\omega\mid u}$ | sky-conditional moments | appendix_sky_marginal.tex:50 | |
| $\mathcal{N}_3, \mathcal{N}_2, \mathcal{N}_1$ | 3-/2-/1-D normal densities | framework.tex:67; appendix_sky_marginal.tex:49–50 | unsubscripted $\mathcal{N}$ also used for 1-D — minor |
| $\mathcal{N}(x;\mu,\sigma^2)$ | normal density, **variance** third argument | estimators.tex:30 | now uniform paper-wide after fixes (F3) |
| $\bar p_\mathrm{GW}(u)$ | isotropic narrow-beam sky marginal | pitfall.tex:48 (eq:pitfall:skymarg); appendix_sky_marginal.tex:63, 70 | consistent |
| $p_\mathrm{GW}$ | GW distance likelihood (harness) | coverage.tex:21 | |
| $\Delta\Omega$ | sky-localization area | appendix_sky_marginal.tex:79 | |
| $\beta$ | ecliptic latitude (in $\lvert\sin\beta\rvert$ bands) | realdata.tex:9 | **never defined** — see M6 |
| $\kappa$ | Fisher condition number | budget.tex:20 | implicit inline definition |

### Catalogue, hypotheses, and likelihood structure

| Symbol | Meaning | Defined at | Notes |
|---|---|---|---|
| $G,\ \bar G$ | in-/out-of-catalogue hypotheses | framework.tex:90–91 | contract ✓ |
| $g$ | galaxy index | framework.tex:111 | collides with shape function $g(z)$ — see C4 |
| $L_\mathrm{cat}(h)$ | in-catalogue likelihood | framework.tex:113 (eq:lcat) | restated estimators.tex:8 (eq:est:lcat), consistent |
| $\mathcal{G}_i,\ \mathcal{G}_i^{\mathrm{sel}}$ | numerator / denominator galaxy sets | framework.tex:115–116 | vs $\mathcal{B}$ in estimators — see M4 |
| $\mathcal{B}$ | 3-D localization ball | estimators.tex:22 | see M4 |
| $w_g$ | rate weight $R_\mathrm{eff}(M_g)/(1+z_g)$ | framework.tex:118 | restated estimators.tex:22, appendix_beta_g.tex:15 ✓ |
| $M_g,\ z_g$ | catalogued MBH mass / redshift | framework.tex:122 | |
| $R_\mathrm{eff}(M)$ | per-MBH EMRI rate (Babak et al. 2017) | framework.tex:126 | |
| $N_g(h)$ | per-host numerator integral | framework.tex:139 (eq:ng) | restated estimators.tex:15 (eq:est:Ng) ✓ |
| $D_g(h)$ | per-host selection integral | framework.tex:143 (eq:dg) | restated estimators.tex:18 (eq:est:Dg) ✓ |
| $p_\mathrm{det}$ | detection probability | framework.tex:152, 241–243 | contract ✓ (formatting unified, F1) |
| $p_g(z)$ | host-redshift kernel | framework.tex:157–162 (eq:kernel) | volume instantiation estimators.tex:37 (eq:est:pgz), appendix_volume_deconv.tex:30 (eq:app:pgz) |
| $\sigma_{z,g}$ | per-galaxy scatter (indexed $\sigma_z$) | framework.tex:158 | see M5 |
| $\Pi(z)$ | population prior placeholder in kernel | framework.tex:170 | $\Pi\equiv 1$ bare; $\Pi = w_\mathrm{pop}$ deconvolved |
| $Z_g$ | per-galaxy kernel normalization | framework.tex:164 (eq:kernel) | restated estimators.tex:39, coverage.tex:28, appendix_volume_deconv.tex:32 ✓ |
| $M_z,\ \hat M_z$ | redshifted MBH mass $M(1+z)$, its ML value | framework.tex:183–184 | contract ✓ ($M$ source-frame) |
| $p_g(M)$ | host-mass kernel | framework.tex:186 | instantiated estimators.tex:74 (eq:est:pgM), appendix_eddington_m.tex:20 (eq:eddm:prior) ✓ |
| $p(\Omega) = 1/4\pi$ | isotropic sky prior | framework.tex:198 | |
| $w_\mathrm{pop}(z)$ | population redshift measure | framework.tex:203 (eq:wpop) | **conflicting definitions** — see C1 |
| $\mathrm{d}V_\mathrm{c}/(\mathrm{d}z\,\mathrm{d}\Omega)$ | comoving volume element per solid angle | framework.tex:201 | typography unified to $V_\mathrm{c}$ (F2) |
| $f(z,\Omega),\ f_k(z)$ | rate-weighted completeness (pixel $k$) | framework.tex:209–210 | collision with $f(z)=(1+z)I(z)$ resolved (C5, fixed); arity drift $f(z,\Omega,h)$ appendix_beta_g.tex:97 — minor |
| $k(\Omega_i)$ | pixel containing event line of sight | framework.tex:225 | |
| $B_\mathrm{num}(h)$ | completion-term numerator | framework.tex:215 (eq:bnum) | ✓ pitfall, appendix_sky |
| $z_-,\ z_+$ | redshift image of $\pm4\sigma_{d_L}$ support | framework.tex:225 | ✓ appendix_sky_marginal.tex:35 |
| $D(h)$ | full-volume selection integral | framework.tex:250 (eq:Dh) | |
| $z_\mathrm{min},\ z_\mathrm{max}(h)$ | analysis redshift limits | framework.tex:250 | $z_\mathrm{min}$ never specified (minor); $z_\mathrm{max}$ glossed differently in appendix_beta_g.tex:19 (minor) |
| $\langle\cdot\rangle_\Omega$ | isotropic sky average | framework.tex:262 | |
| $\beta_{\bar G}(h)$ | out-of-catalogue selection normalization | framework.tex:255 (eq:betabar) | |
| $\beta_G(h)$ | catalogue selection normalization | framework.tex:269 (eq:betag) | contract ✓ |
| $w_G(h)$ | hypothesis weight $\beta_G/D$ | framework.tex:274 | budget.tex:26 writes $w_G(z)$ — minor argument drift |

### Pitfall / estimator / calibration constructs

| Symbol | Meaning | Defined at | Notes |
|---|---|---|---|
| $\delta z_\mathrm{Edd}$ | Eddington redshift shift $\sigma_z^2 s(z_g)$ | pitfall.tex:16 (eq:pitfall:eddz) | ✓ appendix_volume_deconv.tex:92 |
| $s(z)$ | $\mathrm{d}\ln w_\mathrm{pop}/\mathrm{d}z$ | pitfall.tex:18 | collides with Gray's host indicator $s$ — see C3 |
| $\Delta h$ | bias in $h$ | pitfall.tex:24 (eq:pitfall:hbias) | |
| $C(\bar z)$ | Eddington bias coefficient | pitfall.tex:25 | collides with bridge constant $C$ — see C2 |
| $\bar z$ | representative detected-host redshift | pitfall.tex:29–31 | ✓ estimators.tex:64 |
| $\bar z_\mathrm{eff}$ | effective redshift matching $C_\mathrm{meas}$ | appendix_volume_deconv.tex:129 | |
| $C_\mathrm{meas}$ | measured Eddington coefficient | estimators.tex:64 | ✓ appendix_volume_deconv.tex:122 |
| $\Delta h_\mathrm{sub}$ | floor-subtracted bias | appendix_volume_deconv.tex:143 | table-local |
| $\Sigma_\mathrm{global}(h)$ | discrete global selection sum | pitfall.tex:64 (eq:pitfall:sigmaglobal) | duplicate labelled definition appendix_beta_g.tex:10–17 (eq:app:sigmaglob, adds explicit $z_g < z_\mathrm{max}(h)$ cutoff); symbol unified (M1, fixed) |
| $C$ | bridge constant $\Sigma_\mathrm{global}\approx C\beta_G$ | pitfall.tex:70; appendix_beta_g.tex:28 | see C2 |
| $n_\mathrm{gal},\ \bar n_\mathrm{gal}$ | (mean) comoving galaxy number density | pitfall.tex:70; appendix_beta_g.tex:24 | |
| $\langle R_\mathrm{eff}\rangle$ | averaged rate (measure unspecified) | appendix_beta_g.tex:28 | minor undefined average |
| $x_\mathrm{GW}$ | event GW data ($= x_i$) | estimators.tex:22; appendix_sky_marginal.tex:14 | see M3 |
| $g(z)$ | $h$-independent shape of $w_\mathrm{pop}$ | estimators.tex:43; appendix_volume_deconv.tex:62 | see C4 |
| $\sigma_M$ | host-mass measurement scatter | estimators.tex:71 | ✓ appendix_eddington_m.tex:4 |
| $Z_M$ | mass-kernel normalization | estimators.tex:76 (eq:est:pgM) | ✓ appendix_eddington_m.tex:22 |
| $\alpha_g$ | $\mathrm{d}\ln R_\mathrm{eff}/\mathrm{d}\ln M\vert_{M_g}$ | estimators.tex:79 | ✓ appendix_eddington_m.tex:26 |
| $\sigma_\mathrm{rel}$ | $\sigma_M/M_g$ | estimators.tex:79 | ✓ appendix_eddington_m.tex:34 |
| $M_g^\mathrm{eff}$ | Eddington-shifted mass | estimators.tex:79 | ✓ appendix_eddington_m.tex:32 (eq:eddm:tilt) |
| $\Delta\ln M$ | applied mass shift | appendix_eddington_m.tex:83 | |
| $\mathrm{num}_\mathrm{bare},\ \mathrm{num}_\mathrm{vol}$ | kernel-specific numerators | coverage.tex:23, 25 (eq:cov:bare/vol) | harness-local |
| $\mathcal{R}(z)$ | comoving source-frame rate density | appendix_volume_deconv.tex:18 | |
| $z_t$ | true host redshift | appendix_volume_deconv.tex:93 | |
| $N_\mathrm{gal}$ | total catalogue size (Gray eq. A10) | appendix_gray_mapping.tex:79 | |
| $p(s\mid z),\ p(s\mid M)$ | Gray's host-probability weights | framework.tex:124; appendix_gray_mapping.tex:88 | quoted notation; see C3 |
| $\Omega_\mathrm{rest}$ | Gray's sky remainder (eq. A20) | appendix_gray_mapping.tex:120 | quoted notation |

---

## 2. Findings

### CRITICAL (same symbol, two meanings / contract violation)

| ID | Finding | Locations | Status |
|---|---|---|---|
| **C1** | **$w_\mathrm{pop}(z)$ has two conflicting definitions.** Framework (notation owner) defines it *per steradian*: $w_\mathrm{pop} \equiv \frac{1}{1+z}\frac{\mathrm{d}V_\mathrm{c}}{\mathrm{d}z\,\mathrm{d}\Omega}$ (framework.tex:203–206, eq:wpop; consistently used in eq:bnum and appendix_sky_marginal.tex:38). Three other sections define the *same symbol* without the $\mathrm{d}\Omega$: $w_\mathrm{pop} \equiv \frac{1}{1+z}\frac{\mathrm{d}V_\mathrm{c}}{\mathrm{d}z}$ (pitfall.tex:8 eq:pitfall:wpop; estimators.tex:32 eq:est:wpop; appendix_volume_deconv.tex:25 eq:app:wpop). The two differ by the solid-angle measure (a factor $4\pi$ after isotropic averaging). In the per-galaxy kernels the constant cancels against $Z_g$, but in $B_\mathrm{num}$, $D(h)$, $\beta_{\bar G}$ it is load-bearing; this is also a contract touch-point ("dV_c/dz comoving volume element"). **Recommended repair:** keep the framework per-steradian definition once, and state in pitfall/estimators/voldeconv that the full-sky element differs by a constant that cancels in every ratio where it appears. | framework.tex:203 vs pitfall.tex:8, estimators.tex:32, appendix_volume_deconv.tex:25 | **FLAGGED — not auto-fixable** (requires a physics-content decision per usage; Physics Change Protocol if any integral is touched) |
| **C2** | **$C$ used for two different quantities**, both in pitfall.tex: the Eddington bias coefficient $C(\bar z)$ (pitfall.tex:25 eq:pitfall:hbias; estimators.tex:61; appendix_volume_deconv.tex:102) and the $h$-independent bridge constant in $\Sigma_\mathrm{global}(h)\approx C\,\beta_G(h)$ (pitfall.tex:70; appendix_beta_g.tex:26–28, where $C = \bar n_\mathrm{gal}\langle R_\mathrm{eff}\rangle$). Recommend renaming the bridge constant (e.g. $K$ or $C_\mathrm{cat}$). | pitfall.tex:25 vs pitfall.tex:70; appendix_beta_g.tex:28 | FLAGGED — rename requires editorial choice |
| **C3** | **$s$ used for two different quantities**, both in estimators.tex: the log-slope $s(z)\equiv\mathrm{d}\ln w_\mathrm{pop}/\mathrm{d}z$ (pitfall.tex:18; estimators.tex:52; appendix_volume_deconv.tex:80) and Gray et al.'s host indicator in $p(s\mid z)\,p(s\mid M)$ (framework.tex:124; estimators.tex:22; appendix_gray_mapping.tex:88). The indicator is quoted notation from the cited paper, so the slope is the safer rename (or add a disambiguating footnote at framework.tex:124). | estimators.tex:22 vs :52 | FLAGGED — rename requires editorial choice |
| **C4** | **$g$ used for two different quantities**: the galaxy index (framework.tex:111 ff., ubiquitous) and the shape function $g(z)$ in $w_\mathrm{pop}(z;h)=h^{-3}g(z)$ (estimators.tex:43; appendix_volume_deconv.tex:62–63). Recommend $\tilde g(z)$ or $w_0(z)$ for the shape function. | estimators.tex:43; appendix_volume_deconv.tex:62 | FLAGGED — rename requires editorial choice |
| **C5** | **$f$ used for two different quantities**: catalogue completeness $f(z,\Omega)$/$f_k(z)$ (framework.tex:209) and the dimensionless distance $f(z)\equiv(1+z)I(z)$ in appendix_volume_deconv (old lines 93–104), which was moreover the same quantity as estimators' $\mathcal{D}(z)$ (also a (b)-type duplicate). | appendix_volume_deconv.tex:93, 99, 103 | **FIXED** — renamed to $\mathcal{D}(z)$ (fix F6) |

### MAJOR (quantity under two symbols / undefined load-bearing symbol)

| ID | Finding | Locations | Status |
|---|---|---|---|
| **M1** | Same quantity, two symbols: the discrete global selection sum was $\Sigma_\mathrm{global}(h)$ in pitfall.tex:64 (eq:pitfall:sigmaglobal) but $\Sigma_\mathrm{glob}(h)$ in appendix_beta_g.tex (eq:app:sigmaglob and 3 further uses). | appendix_beta_g.tex:11, 26, 48, 90 | **FIXED** — unified to $\Sigma_\mathrm{global}$ (fix F5). Residual note: the object is still *defined twice* in two labelled equations (eq:pitfall:sigmaglobal, eq:app:sigmaglob — the appendix version adds the explicit $z_g<z_\mathrm{max}(h)$ cutoff); eq:app:sigmaglob is never cross-referenced. Consider making the appendix reference the pitfall equation. |
| **M2** | Same quantity, three symbols: the marginal fractional-distance variance was $\Sigma_{i,uu}$ (framework.tex:227), $\Sigma_{22}$ (pitfall.tex old:50–54), and $\Sigma_{uu}$ (appendix_sky_marginal.tex:55). | pitfall.tex:48, 52 | **FIXED** — pitfall unified to $\Sigma_{uu}$ (fix F4). The remaining indexed/unindexed pair ($\Sigma_{i,uu}$ vs $\Sigma_{uu}$) follows the paper's event-index-dropping convention and is acceptable. |
| **M3** | Same quantity, two symbols: per-event GW data is $x_i$ (framework.tex:40, eq:gwlike) but $x_\mathrm{GW}$ in estimators.tex:16, 22 (eq:est:Ng) and appendix_sky_marginal.tex:14 (eq:app:sky:gauss) — estimators.tex:22 even cross-references Section 2, where the symbol is $x_i$. Recommend unifying on $x_i$ (or defining $x_\mathrm{GW}\equiv x_i$ once). | estimators.tex:16, 22; appendix_sky_marginal.tex:14 | FLAGGED — symbol choice is editorial (index-free form may be deliberate) |
| **M4** | Same quantity, two symbols: the localization-region galaxy set is $\mathcal{G}_i$ (numerator) / $\mathcal{G}_i^{\mathrm{sel}}$ (denominator choice) in framework.tex:115–116, 130, but $g\in\mathcal{B}$ (ball) in estimators.tex:9–10, 22 (eq:est:lcat). The split is arguably deliberate ($\mathcal{G}_i^{\mathrm{sel}}$ is the free choice; $\mathcal{B}$ its local instantiation), but the correspondence $\mathcal{G}_i^{\mathrm{sel}}=\mathcal{G}_i=\{g\in\mathcal{B}\}$ is never stated. One linking sentence in estimators §4.1 would close it. | framework.tex:115–116 vs estimators.tex:22 | FLAGGED |
| **M5** | Same quantity, two symbols: per-galaxy photometric scatter is $\sigma_{z,g}$ in framework.tex:158–169 (eq:kernel) but $\sigma_z$ (declared "per-galaxy" at framework.tex:22–23) everywhere else, including the restatements of the same kernel (estimators.tex:37, appendix_volume_deconv.tex:30, coverage.tex:23–28). Recommend a "galaxy index dropped where unambiguous" clause at framework.tex:23, or using $\sigma_{z,g}$ in the estimator equations. | framework.tex:158–169 vs estimators.tex:30 ff. | FLAGGED |
| **M6** | Symbol used but never defined: $\beta$ = ecliptic latitude in "six equal-$\lvert\sin\beta\rvert$ sky bands" (realdata.tex:9). Load-bearing for reproducing the $p_\mathrm{det}$ estimate, and visually collides with the $\beta_G$/$\beta_{\bar G}$ family. Needs "ecliptic latitude $\beta$" in prose (not an auto-fixable formatting change). | realdata.tex:9 | FLAGGED |
| **M7** | Symbol used but never explicitly defined: $\sigma_{d_L}$ (event distance uncertainty), first used framework.tex:150 ("$\pm4\sigma_{d_L}$ distance image") and only pinned implicitly through $\Sigma_{uu}=\sigma_{d_L}^2/\hat d_L^{\,2}$ (pitfall.tex:52; appendix_sky_marginal.tex:55). One clause at first use suffices. | framework.tex:150 | FLAGGED |

### MINOR (formatting / typography) — fixed

See §4 for the complete edit list.

1. `\rm` vs `\mathrm` in math (77 sites; e.g. $p_{\rm det}$ vs $p_\mathrm{det}$, $w_\mathrm{pop}$, $L_\mathrm{cat}$, $B_\mathrm{num}$, $\Omega_\mathrm{m}$, deg/sr/rad units) — unified to `\mathrm`.
2. $V_c$ (italic subscript) vs framework's $V_\mathrm{c}$ (9 sites) — unified to $V_\mathrm{c}$.
3. Gaussian third argument: std vs variance. The paper's declared convention is $\mathcal{N}(x;\mu,\sigma^2)$ (estimators.tex:30), matching the variance slot $\Sigma_{uu}$ in eq:bnum/eq:pitfall:skymarg; framework eq:kernel (3 sites), appendix_volume_deconv (4 sites) and coverage (3 sites) carried an un-squared $\sigma$ in the same slot — unified to $\sigma^2$ (no numerical content changed; the intended distribution is the same Gaussian). |
4. $2^\circ$ (appendix_sky_marginal) vs $2\degr$ (pitfall) — unified to MNRAS `\degr`.
5. Spelling: dominant convention is Oxford (-ize with "analysed"); outliers "localisation" (framework) and "marginalises" (appendix_eddington_m) — unified to "localization"/"marginalizes".

### MINOR — flagged, not fixed

1. Gaussian argument order: likelihood form $\mathcal{N}(z_g; z, \sigma_z^2)$ (framework, estimators, voldeconv) vs $\mathcal{N}(z; z_g, \sigma_z^2)$ (pitfall.tex:6; coverage.tex:23–28). Numerically identical by symmetry; harmonization would touch prose that distinguishes "likelihood of the catalogued value" from "density for $z$", so it is left editorial.
2. Unsubscripted $\mathcal{N}$ for 1-D Gaussians (framework eq:kernel, eq:bnum) vs dimension-subscripted $\mathcal{N}_1$ (appendix_sky_marginal).
3. Differential placement: $\int\mathrm{d}z\,f(z)$ (framework, estimators) vs $\int f(z)\,\mathrm{d}z$ (coverage, appendix_volume_deconv $Z_g$, appendix_eddington_m $Z_M$) — both styles occur even for the same quantity ($Z_g$). All differentials do use `\mathrm{d}` (contract-conform); only placement varies.
4. Equation-reference style: "equation~\eqref{}" (framework) vs "Eq.~\eqref{}" (pitfall, appendices); appendix_volume_deconv.tex:145 uses `Eq.~\ref{eq:app:dhlaw}` (the only parenthesis-free `\ref` to an equation, inside a caption where `\eqref` would double the parentheses — left as is).
5. Gray equation citing: body uses "eq.~A10 / eqs~A9–A10" (no dots); GENERATED main.tex appendix title says "Eqs. A.9/A.10" (dotted). Fix belongs in the builder config, not the sections.
6. $w_G(h)$ (framework.tex:274) vs "$w_G(z)$ split" (budget.tex:26) — same symbol with a different argument; the budget usage means the weight as a function of population depth.
7. $z_\mathrm{max}(h)$ glossed as "redshift horizon of the analysis" (framework.tex:264) vs "redshift reach of the population model" (appendix_beta_g.tex:19–20).
8. Completeness arity drift: $f(z,\Omega)$ (framework.tex:209) vs $f(z,\Omega,h)$ (appendix_beta_g.tex:97).
9. Undefined-but-peripheral: $z_\mathrm{min}$ (framework.tex:250, never specified), $\langle R_\mathrm{eff}\rangle$ (appendix_beta_g.tex:28, averaging measure unspecified).
10. Symbol-family overloads that remain distinguishable and are accepted: $D_i$/$D(h)$/$D_g(h)$/$\mathcal{D}(z)$; $N$/$N_\mathrm{det}$/$N_\mathrm{gal}$/$N_g$/$\mathcal{N}$; unindexed $\mu,\Sigma$ for generic events in the appendices.
11. eq:app:sigmaglob is labelled but never referenced (harmless; related to the duplicate definition noted in M1).
12. Noted in passing (numbers, not notation — for CONSISTENCY-AUDIT): coverage.tex:15 states grid spacing $0.004$ while the fig:pp caption (coverage.tex:67) states "$h$-grid step $0.001$"; realdata.tex:73 still carries the `$\text{[PENDING]}$` cluster-confirmation placeholder.

### Contract compliance (d)

| Contract item | Status |
|---|---|
| $h \equiv H_0/(100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$ | ✓ defined framework.tex:20–21; used consistently; no bare "H0" anywhere |
| $z$ CMB-frame | ✓ framework.tex:21–22 (budget.tex:29 correctly discusses the heliocentric→CMB rebuild as a systematic) |
| $\sigma_z$ photometric redshift uncertainty | ✓ framework.tex:22–23; indexed variant drift → M5 |
| $d_L$ luminosity distance | ✓ framework.tex:26 |
| $M$ host MBH mass | ✓ framework.tex:184 (source-frame; $M_z=M(1+z)$ detector-frame) |
| $G/\bar G$ hypotheses | ✓ framework.tex:90–91 |
| $p_\mathrm{det}$ | ✓ after F1 (was split $p_{\rm det}$/$p_\mathrm{det}$ across 7 files) |
| $\beta_G(h)$ | ✓ framework.tex:269 |
| $\mathrm{d}V_\mathrm{c}/\mathrm{d}z$ | **partially violated** via the $w_\mathrm{pop}$ per-steradian vs full-sky split → C1 |
| GLADE+ typography | ✓ "GLADE+" throughout; the two bare "GLADE" mentions (codes.tex:33 = DarkSirensStat's actual catalogue; pitfall.tex:39 "GLADE-like") are semantically correct and untouched |

---

## 3. Equation hygiene

- **`\left`/`\right` pairing:** balanced in every file (appendix_eddington_m 4/4, pitfall 2/2, estimators 1/1, appendix_volume_deconv 1/1, appendix_sky_marginal 1/1 — the apparent extra `\left` there is `\leftrightarrow` at line 106).
- **Differentials:** every differential in every integral uses `\mathrm{d}` (`\mathrm{d}z`, `\mathrm{d}z'`, `\mathrm{d}M`, `\mathrm{d}\Omega`, `\mathrm{d}t^2`); no bare italic-$d$ differentials found. Placement (leading vs trailing) varies — minor flag §2.
- **Cross-references:** all `\ref`/`\eqref` targets resolve against section labels plus the GENERATED main.tex labels; no dangling references. No equation is referenced without carrying a label; all `equation`/`align` environments are labelled (align rows each carry a label). One labelled equation is never referenced (eq:app:sigmaglob).
- **Tables/figures:** every `table`/`figure` label (tab:pp-clean, tab:derail, tab:partialfix, tab:budget, tab:gray:mapping, tab:app:eddz, fig:betag, fig:pp, fig:derail, fig:ablation) is referenced at least once.

---

## 4. Fixes applied

All fixes are mechanical harmonizations (formatting/typography or within-paper symbol unification with a uniquely determined target); no physics content, numbers, or prose meaning changed. 108 replacements, uncommitted in the working tree.

| # | Fix | Files (count) |
|---|---|---|
| F1 | `{\rm X}` → `\mathrm{X}` in math mode ($p_\mathrm{det}$, $w_\mathrm{pop}$, $L_\mathrm{cat}$, $B_\mathrm{num}$, $x_\mathrm{GW}$, $R_\mathrm{eff}$, $\Omega_\mathrm{m}$, $h_\mathrm{true}$, $C_\mathrm{meas}$, $\sigma_\mathrm{rel}$, $M_g^\mathrm{eff}$, $\mathcal{G}_i^{\mathrm{sel}}$, $\Omega_\mathrm{pix}$, $\bar p_\mathrm{GW}$, $\delta z_\mathrm{Edd}$, $z_\mathrm{med}$, $\bar z_\mathrm{eff}$, $\Delta h_\mathrm{sub}$, units deg/sr/rad) | estimators (30), appendix_volume_deconv (22), appendix_sky_marginal (15), realdata (4), budget (4), framework (1), codes (1) — **77 total** |
| F2 | $V_c$ → $V_\mathrm{c}$ (framework/pitfall convention) | appendix_volume_deconv (3), coverage (2), estimators (1), codes (1), appendix_sky_marginal (1), appendix_beta_g (1) — **9 total** |
| F3 | Gaussian third argument unified to **variance** per the declared convention $\mathcal{N}(x;\mu,\sigma^2)$ (estimators.tex:30): framework eq:kernel $\sigma_{z,g}\to\sigma_{z,g}^2$ (3); appendix_volume_deconv $\sigma_z\to\sigma_z^2$ in eq:app:pgz and the bare-kernel reference (4); coverage eq:cov:bare/vol and $Z_g$ (3) | framework (3), appendix_volume_deconv (4), coverage (3) — **10 total** |
| F4 | pitfall.tex $\Sigma_{22}\to\Sigma_{uu}$ (matches appendix_sky_marginal and framework's $\Sigma_{i,uu}$) | pitfall (2) |
| F5 | appendix_beta_g $\Sigma_\mathrm{glob}\to\Sigma_\mathrm{global}$ (matches pitfall eq:pitfall:sigmaglobal) | appendix_beta_g (4) |
| F6 | appendix_volume_deconv $f(z)\equiv(1+z)I(z)\to\mathcal{D}(z)$ (definition sentence + twice in eq:app:dhlaw) — removes the collision with completeness $f$ and matches estimators.tex:64 | appendix_volume_deconv (3) |
| F7 | $2^\circ \to 2\degr$ (MNRAS style, matches pitfall.tex:52) | appendix_sky_marginal (1) |
| F8 | Spelling unified to the paper's dominant Oxford convention: "localisation"→"localization" (framework.tex:126), "marginalises"→"marginalizes" (appendix_eddington_m.tex:3) | framework (1), appendix_eddington_m (1) |

**Not fixed (require author/physics decision):** C1 ($w_\mathrm{pop}$ measure split — Physics Change Protocol territory), C2–C4 (renames of $C$, $s$, $g(z)$), M3–M7 (symbol unifications/definitions needing prose), and all flagged minors above.
