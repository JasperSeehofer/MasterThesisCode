# NOTATION AUDIT — Paper A (MNRAS manuscript, `paper_a/`)

Date: 2026-07-02
Scope: all files in `paper_a/sections/*.tex` + `main.tex`.
Contract checked: `h = H0/100`, `z` CMB-frame, `beta_G`, `p_det`, `dV_c/dz`,
plus `\mathrm{d}` differentials and `\left`/`\right` pairing.

Severity legend: **CRITICAL** = same symbol denotes inequivalent quantities in
displayed equations / contract broken in a way that changes meaning;
**MAJOR** = will confuse a careful reader or referee, needs a real edit;
**MINOR** = cosmetic or style inconsistency.

---

## 1. Symbol table

Location key: intro = introduction.tex, fw = framework.tex, pit = pitfall.tex,
est = estimators.tex, cov = coverage.tex, rd = realdata.tex, pm = postmortem.tex,
codes = codes.tex, bud = budget.tex, conc = conclusions.tex,
A = appendix_sky_marginal, B = appendix_volume_deconv, C = appendix_gray_mapping,
D = appendix_eddington_m, E = appendix_beta_g. Line numbers are approximate.

### Cosmology and global parameters

| Symbol | Meaning | Defined | Also used | Notes |
|---|---|---|---|---|
| $h \equiv H_0/(100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$ | dimensionless Hubble constant | abstract:13, fw:21 | everywhere | CONTRACT OK |
| $H_0$ | Hubble constant | via $h$ definition | abstract, intro, codes, est, B | OK |
| $H(z)$ | Hubble rate | never (standard) | est:64, B:61 | acceptable |
| $z$ | CMB-frame redshift | fw:21–23 ("All redshifts $z$ are CMB-frame") | everywhere | CONTRACT: see Finding F10 (budget admits Sec. 6 baselines are heliocentric) |
| $\sigma_z$ | per-galaxy photometric redshift scatter | fw:22 | pit, est, cov, B, codes | fw eq. (kernel) instead writes $\sigma_{z,g}$ — F15 |
| $\sigma_{z,g}$ | same as $\sigma_z$ | fw eq:kernel (l.162) | fw only | F15: two symbols for one quantity |
| $d_L(z,h)$ | luminosity distance | fw eq:dl (l.26) | everywhere | bud:43 extends silently to $d_L(z; h, \Omega_m)$ |
| $\hat d_L$ | maximum-likelihood distance | fw:61 | pit, A | OK |
| $E(z)$ | $\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}$ / $H(z)/H_0$ | fw eq:dl; redefined est:64, B:61 | pit, B | equivalent definitions, duplicated |
| $E'(z)$ | $\mathrm{d}E/\mathrm{d}z$ | B eq:app:eddz (l.85) | B | OK |
| $\Omega_\mathrm{m}$, $\Omega_\Lambda$ | density parameters | fw eq:dl, fw:32 | pit, cov, B, bud | fiducial value 0.2726 appears only in bud; synthetic suites use 0.30 — F22 |
| $I(z)$ | $\int_0^z \mathrm{d}z'/E(z')$ | B:61 | B | see F4 (distance-shape aliases) |
| $\mathcal{D}(z)$ | $(1+z)\int_0^z \mathrm{d}z'/E(z')$ | est:64 | est only | F4: same as B's $f(z)$ and $\propto$ cov's $A(z)$ |
| $f(z) \equiv (1+z)I(z)$ | dimensionless distance | B:94 | B eq:app:dhlaw | **F3: collides with completeness $f$** |
| $A(z)$ | $d_L = A(z)/h$ amplitude | cov:12 | cov | F4 alias |
| $c$ | speed of light | never (standard) | fw eq:dl, est:64 | acceptable |
| $w_0$, $w_a$ | dark-energy parameters | never (standard) | bud:24 | acceptable |

### Event data and GW likelihood

| Symbol | Meaning | Defined | Also used | Notes |
|---|---|---|---|---|
| $i$ | event index | fw:40 | everywhere | OK |
| $x_i$ | event data | fw:40 | fw | F8: est/A call the same thing $x_{\rm GW}$ |
| $x_{\rm GW}$ | event data | est:22 (points back to Sec. 2) | est eq:est:Ng, A:14 | F8 |
| $D_i$ | detection indicator | fw:40–41 | fw, C | letter D heavily overloaded — F5 note |
| $\rho_\mathrm{thr}$ | SNR detection threshold (=20) | fw:42 | E:41 | OK |
| $(\phi,\theta)$ | ecliptic azimuth/polar sky coords | fw:60, A:10 | pit, A | OK |
| $\Omega = (\phi,\theta)$ | sky position | fw:73 | fw, est ($\Omega_g$), A ($\mathrm{d}\Omega$) | F23: A instead uses $\omega=(\phi,\theta)$, keeping $\Omega$ for solid angle |
| $\omega = (\phi,\theta)$ | sky part of coordinates | A:45 | A | F23 |
| $u \equiv d_L/\hat d_L$ | distance fraction | fw:61, A:11 | pit, A | OK |
| $u(z,h)$ | $d_L(z,h)/\hat d_L$ | fw eq:gwlike | fw, A | OK |
| $\hat\mu_i = (\hat\phi_i,\hat\theta_i,1)$ | ML point | fw:62 | fw | A:16 calls it $\mu$ (index dropped) — F23 |
| $\Sigma_i$ / $\Sigma$ | projected 3×3 covariance | fw:63 / A:19, pit:46 | fw, pit, A | Σ also = catalogue sums Σ_global — F6 note |
| $\Sigma_{i,uu}$ | marginal distance-fraction variance | fw:228 | fw eq:bnum, C:154 | **F5: same quantity is $\Sigma_{22}$ (pit) and $\Sigma_{uu}$ (A)** |
| $\Sigma_{22}$ | ditto | pit:54 | pit eq:pitfall:skymarg | F5: 0-based index leak; 1-based reading of (φ,θ,u) gives the θθ element |
| $\Sigma_{uu}$ | ditto | A:55 | A | F5 |
| $(\Sigma^{-1})_{uu}$ | conditional-variance inverse element | A:56 | A | OK |
| $\mu_{\omega\mid u}$, $\Sigma_{\omega\mid u}$ | sky conditional moments | A eq:app:sky:factor | A | OK |
| $\sigma_\phi,\sigma_\theta$ | sky-localization widths | pit:54 | A:80 | OK |
| $\sigma_{d_L}$ | GW distance uncertainty | never formally | fw:149 ($\pm4\sigma_{d_L}$), pit:54, A:55 | F22 |
| $\mathcal{N}_3, \mathcal{N}_2, \mathcal{N}_1, \mathcal{N}$ | multivariate/scalar normal | est:30 declares $\mathcal{N}(x;\mu,\sigma^2)$ | everywhere | **F2: third argument is σ² in est/pit/D but σ in cov/B; σ_{z,g} ambiguous in fw** |
| $p_{\rm GW}$ | GW distance likelihood (harness) | cov:21 | cov | distinct from $\bar p_{\rm GW}$; OK |
| $\bar p_{\rm GW}(u)$ | isotropic sky marginal of GW likelihood | pit eq:pitfall:skymarg, A eq:app:sky:marginal | pit, A | duplicated definition (F17), consistent |
| $\hat\theta$ / $\hat\theta_i$ | estimated polar angle | pit:54 / fw eq:bnum | A | index dropped in pit/A; OK |
| $\rho$ | (SNR, implicit in $\rho_{\rm thr}$) | — | — | only used as $\rho_{\rm thr}$; OK |
| $\kappa$ | Fisher condition number | never | bud:20 | F22 |
| $(M, e_0, p_0)$ | EMRI mass, eccentricity, semi-latus rectum | never | bud:30 | F22 |
| $\beta$ | ecliptic latitude (sky bands) | never | rd:9 ($|\sin\beta|$) | **F13: undefined + collides with $\beta_G$ family** |

### Hypotheses, likelihood terms, selection

| Symbol | Meaning | Defined | Also used | Notes |
|---|---|---|---|---|
| $G$, $\bar G$ | in-/out-of-catalogue hypotheses | fw:90–91 | everywhere | OK |
| $p_i(h) \equiv p(x_i\mid D_i,h)$ | per-event likelihood | fw eq:posterior | fw eq:assembled, cov | OK |
| $\pi(h)$ | prior on $h$ | fw:44 | fw | OK |
| $N_\mathrm{det}$ | number of detected events | fw eq:posterior | fw | F14: pit/pm/E use bare $N$ for the same |
| $N$ | event count | — | pit:76, pm, E:85 | F14 |
| $L_\mathrm{cat}(h)$ | in-catalogue likelihood | fw eq:lcat; redefined est eq:est:lcat | pit, E | duplicated definition; denominator set differs in notation — F9 |
| $\mathcal{G}_i$, $\mathcal{G}_i^{\rm sel}$ | numerator/denominator galaxy sets | fw eq:lcat (l.115–130) | fw only | **F9: est uses $\mathcal{B}$ instead** |
| $\mathcal{B}$ | 3-D localization ball (sum domain) | est:22 | est eq:est:lcat | F9 |
| $g$ | galaxy index | fw:112 | everywhere | **F12: also $g(z)$ shape function** |
| $g(z)$ | $w_{\rm pop}$ shape, $w_{\rm pop}=h^{-3}g(z)$ | est:43, B:63 | est, B | F12 |
| $w_g = R_\mathrm{eff}(M_g)/(1+z_g)$ | per-galaxy rate weight | fw eq:lcat, est:22, E eq:app:sigmaglob | C, D | consistent; visually close to $w_G$ — F24 |
| $w_G(h) = \beta_G/D$ | in-catalogue hypothesis weight | fw:276 | fw, rd:10, E:92 | F24 |
| $M_g$, $z_g$ | catalogued MBH mass, redshift | fw:123 | everywhere | OK |
| $R_\mathrm{eff}(M)$ | per-MBH EMRI rate | fw:125 | est, C, D, E | OK |
| $p(s\mid z)$, $p(s\mid M)$ | Gray host-probability weights | fw:124 (cited) | est, C eq:gray:a10 | F25: $s$ also Eddington slope $s(z)$ |
| $N_g(h)$, $D_g(h)$ | per-host numerator / selection integrals | fw eq:ng, eq:dg; redefined est eq:est:Ng, eq:est:Dg | C | duplicated with different integrand notation (F17) |
| $p_\mathrm{det}$ | detection probability | fw:243 | everywhere | CONTRACT OK (lowercase, denominators only) |
| $p_g(z)$ | host-redshift kernel | fw eq:kernel (general, $\Pi$); est eq:est:pgz; B eq:app:pgz | cov | consistent as special cases; B adds truncated window |
| $\Pi(z)$ | population prior placeholder | fw eq:kernel | fw | OK |
| $Z_g$ | per-galaxy kernel normalization | fw eq:kernel, est eq:est:pgz, cov:28, B eq:app:pgz | — | consistent; integration limits differ (documented in B) |
| $w_\mathrm{pop}(z)$ | population redshift measure | fw eq:wpop; pit eq:pitfall:wpop; est eq:est:wpop; B eq:app:wpop | cov, pm | **F1 (CRITICAL): fw defines per steradian, others total — 4π mismatch** |
| $f(z,\Omega)$, $f_k(z)$ | rate-weighted completeness (pixel $k$) | fw:210–212 | pit, cov:75, A, C, E | F3: collides with $f(z)=(1+z)I(z)$ in B |
| $k(\Omega_i)$ | pixel containing event LOS | fw:227 | fw | OK |
| $B_\mathrm{num}(h)$ | completion-term numerator | fw eq:bnum; A eq:app:sky:bnum | pit, C | duplicated (F17), consistent given fw's per-sr $w_{\rm pop}$ |
| $[z_-,z_+]$ | redshift image of $\pm4\sigma_{d_L}$ support | fw:226 | A | OK |
| $D(h)$ | full-volume selection integral | fw eq:Dh; cov eq:cov:singlehost | est, rd, E | letter D overloaded ($D_i$, $D_g$, $D(h)$, $\mathcal{D}(z)$) — note under F5/F4 |
| $\beta_{\bar G}(h)$ | out-of-catalogue selection integral | fw eq:betabar | A, C, E | OK |
| $\beta_G(h) = D - \beta_{\bar G}$ | catalogue selection normalization | fw eq:betag | everywhere | CONTRACT OK |
| $\langle\cdot\rangle_\Omega$ | isotropic sky average | fw:264 | fw | OK |
| $z_\mathrm{min}$, $z_\mathrm{max}(h)$ | analysis redshift limits | $z_{\max}$: fw:265, E:19; $z_{\min}$: never | fw eq:Dh, eq:betabar | F22: $z_{\min}$ undefined |
| $\Sigma_\mathrm{global}(h)$ | discrete global selection sum | pit eq:pitfall:sigmaglobal | pm (Table) | **F6: E calls it $\Sigma_{\rm glob}$ and adds $z_g<z_{\max}(h)$ restriction** |
| $\Sigma_\mathrm{glob}(h)$ | ditto | E eq:app:sigmaglob | E | F6 |
| $C$ | $h$-independent bridge constant | pit:70; E eq:app:bridge ($=\bar n_{\rm gal}\langle R_{\rm eff}\rangle$) | pit, E | **F7: collides with Eddington coefficient $C(\bar z)$ in the same section** |
| $\bar n_\mathrm{gal}$ / $n_\mathrm{gal}(h)$ | comoving galaxy number density | E:24 / pit:72 | — | barred and unbarred variants — fold into F6 |
| $V_c$ | comoving volume | pit:72, E:64 | — | subscript style varies ($V_{\mathrm{c}}$ vs $V_c$) — F20 |

### Eddington-in-z / Eddington-in-M machinery

| Symbol | Meaning | Defined | Also used | Notes |
|---|---|---|---|---|
| $s(z) \equiv \mathrm{d}\ln w_{\rm pop}/\mathrm{d}z$ | log-slope of population measure | pit eq:pitfall:eddz; est eq:est:eddz; B eq:app:eddz (explicit form) | — | triplicated (F17), consistent; F25 collision with host indicator $s$ |
| $\delta z_\mathrm{Edd} = \sigma_z^2 s(z_g)$ | Eddington redshift shift | pit eq:pitfall:eddz | B:93 | OK |
| $\langle z\rangle$ | recentred kernel mean | est eq:est:eddz, B eq:app:eddz | — | OK |
| $\Delta h$ | Hubble bias | pit eq:pitfall:hbias | est, cov, B | OK |
| $C(\bar z)$ | Eddington bias coefficient | pit eq:pitfall:hbias; est eq:est:eddh; B eq:app:dhlaw | — | triplicated with **three different inner derivatives**: $\mathrm{d}\ln d_L/\mathrm{d}z$ (pit), $\mathrm{d}\ln\mathcal{D}/\mathrm{d}z$ (est), $\mathrm{d}\ln f/\mathrm{d}z$ (B) — F4; collides with bridge $C$ — F7 |
| $C_\mathrm{meas}$ | measured coefficient | est:64, B:123 | tab:app:eddz | OK |
| $\bar z$, $\bar z_{\rm eff}$ | representative detected-host redshift | pit:31, B:130 | est | OK |
| $z_t$, $\hat z$ | true / assumed host redshift | B:94–96 | B | OK |
| $d_L^\star$ | GW-fixed distance | B:94 | B | local, OK |
| $\Delta h_{\rm sub}$ | floor-subtracted bias | B tab caption | B | implicit but clear |
| $\Phi$ | standard normal CDF | never | B:45 | F22 |
| $\mathcal{R}(z)$ | comoving source-frame rate density | B:18 | B | OK, distinct from $R_{\rm eff}$ |
| $M_z = M(1+z)$ | redshifted MBH mass | fw:184, D:2 | D | OK |
| $\hat M_z$ | ML redshifted mass | fw:184 (implicit) | D:68 | OK |
| $M$ | source-frame MBH mass | fw:184 | D | also bare $M$ in bud:30 tuple |
| $p_g(M)$ | host-mass kernel | est eq:est:pgM; D eq:eddm:prior | fw:186 | est version has dummy-variable capture — F16 |
| $Z_M$ | mass-kernel normalization | est eq:est:pgM, D eq:eddm:prior | — | F16 |
| $\sigma_M$ | catalogue mass scatter | est:71, D:4 | — | OK |
| $\alpha_g$ | $\mathrm{d}\ln R_{\rm eff}/\mathrm{d}\ln M|_{M_g}$ | est:79, D:26 | — | consistent |
| $\sigma_\mathrm{rel} = \sigma_M/M_g$ | relative mass scatter | est:79, D:34 | — | consistent |
| $M_g^\mathrm{eff}$ | shifted/moment-matched mean | est:79, D eq:eddm:tilt | — | consistent |
| $\Delta\ln M$ | mass shift | D:83 | — | OK |
| $\mathrm{d}n_\mathrm{cat}/\mathrm{d}M$ | catalogue mass function | D:98 | D | local, OK |

### Coverage-harness locals (coverage.tex)

| Symbol | Meaning | Defined | Notes |
|---|---|---|---|
| $h_\mathrm{true}$ | injected truth | cov tab:pp-clean, cov:57 | OK |
| $\mathrm{num}(h)$, $\mathrm{num}_{\rm bare}$, $\mathrm{num}_{\rm vol}$ | numerator variants | cov eq:cov:singlehost/bare/vol | local, OK |
| $\sigma_{d_L}/d_L$ | fractional distance error | cov tab caption | OK |

---

## 2. Findings

### CRITICAL

**F1 — `w_pop` is defined with two inequivalent measures (4π mismatch), and the framework definition deviates from the notation contract.**
- framework.tex eq:wpop (l.203–208): $w_\mathrm{pop}(z) \equiv \frac{1}{1+z}\,\frac{\mathrm{d}V_\mathrm{c}}{\mathrm{d}z\,\mathrm{d}\Omega}$ — **per unit solid angle**.
- pitfall.tex eq:pitfall:wpop (l.8), estimators.tex eq:est:wpop (l.32), appendix_volume_deconv.tex eq:app:wpop (l.25): $w_{\rm pop}(z) \equiv \frac{1}{1+z}\,\frac{\mathrm{d}V_c}{\mathrm{d}z}$ — **total (full-sky)**.
- These differ by 4π under the same symbol. The notation contract specifies `dV_c/dz`. The framework version is the one wired into the units argument of eq:assembled (Mpc³ sr⁻²) and into eq:bnum/eq:app:sky:bnum, so the fix is not a blind find-and-replace: either (a) rename the framework object (e.g. $w_\mathrm{pop}^{\,\Omega}$ or fold the $1/4\pi$ explicitly) and keep $w_{\rm pop} = (1+z)^{-1}\mathrm{d}V_c/\mathrm{d}z$ everywhere else, or (b) adopt the per-steradian object globally and say so once. In a paper whose headline defect is a mis-handled $1/(4\pi)$, an internal 4π ambiguity in the central symbol is the first thing a referee will probe.

### MAJOR

**F2 — Gaussian third-argument convention is inconsistent (variance vs standard deviation).**
- Declared convention (estimators.tex l.30): "$\mathcal{N}(x;\mu,\sigma^2)$ denotes a normal density" — variance.
- Variance used: pitfall.tex l.6 ($\mathcal{N}(z;z_g,\sigma_z^2)$), estimators eq:est:pgz/pgM, appendix_eddington_m eq:eddm:prior/tilt, all covariance-matrix uses ($\Sigma_i$, $\Sigma_{uu}$).
- Standard deviation used: coverage.tex eq:cov:bare, eq:cov:vol, l.28 ($\mathcal{N}(z; z_g, \sigma_z)$ — four instances); appendix_volume_deconv l.13, eq:app:pgz ($\mathcal{N}(z_g;z,\sigma_z)$ — three instances); framework eq:kernel writes $\sigma_{z,g}$ (ambiguous).
- Fix: normalize to $\sigma^2$ everywhere (coverage: 4 edits; volume_deconv: 3 edits; framework eq:kernel: write $\sigma_{z,g}^2$).
- Secondary: argument order flips between $\mathcal{N}(z_g; z, \cdot)$ (likelihood reading — fw, est, B) and $\mathcal{N}(z; z_g, \cdot)$ (density reading — pit, cov). Symmetric in the Gaussian so not wrong, but pick one and state it.

**F3 — Symbol `f` denotes two different quantities.**
- $f(z,\Omega)$, $f_k(z)$ = rate-weighted catalogue completeness (framework l.210, used in pitfall, coverage l.75, appendices A/C/E).
- $f(z) \equiv (1+z)I(z)$ = dimensionless distance shape (appendix_volume_deconv l.94, inside eq:app:dhlaw via $\mathrm{d}\ln f/\mathrm{d}z$).
- Fix: rename the appendix-B distance function to $\mathcal{D}(z)$ (already defined in estimators.tex l.64 as exactly $(1+z)\int_0^z\mathrm{d}z'/E(z')$), which also resolves half of F4.

**F4 — The distance-shape function appears under four notations.**
Same (or trivially proportional) quantity: $\mathrm{d}\ln d_L/\mathrm{d}z$ (pitfall eq:pitfall:hbias), $\mathcal{D}(z)$ (estimators eq:est:eddh), $f(z)=(1+z)I(z)$ (appendix_volume_deconv eq:app:dhlaw), $A(z)$ with $d_L = A(z)/h$ (coverage l.12, dimensionful). The three $C(\bar z)$ definitions (pit/est/B) are numerically identical but a reader must prove it. Fix: define $\mathcal{D}(z)$ once in framework, use it in all three $C(\bar z)$ equations; keep $A(z) = (c/100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})\,\mathcal{D}(z)$ with the relation stated in coverage.

**F5 — Marginal distance-fraction variance appears under three symbols, one with a 0-based index leak.**
- $\Sigma_{i,uu}$ (framework eq:bnum, l.228), $\Sigma_{22}$ (pitfall eq:pitfall:skymarg, l.50/54), $\Sigma_{uu}$ (appendix_sky_marginal l.49–75).
- $\Sigma_{22}$ is code-style 0-based indexing of $(\phi,\theta,u)$: in the 1-based convention of the manuscript's matrices the $u$-element is $\Sigma_{33}$; a reader will parse $\Sigma_{22}$ as the θθ element. Fix: use $\Sigma_{uu}$ (with event index where needed: $\Sigma_{i,uu}$) in all three places.

**F6 — Global catalogue sum: `\Sigma_global` vs `\Sigma_glob`, with silently differing definitions.**
- pitfall eq:pitfall:sigmaglobal: $\Sigma_{\mathrm{global}}(h) = \sum_g w_g\, p_{\mathrm{det}}(d_L(z_g,h))$ (unrestricted sum).
- appendix_beta_g eq:app:sigmaglob: $\Sigma_{\mathrm{glob}}(h) = \sum_{g:\,z_g<z_{\max}(h)} w_g\, p_{\mathrm{det}}(...)$ (restricted).
- Same object, two names, and the restriction $z_g < z_{\max}(h)$ appears only in the appendix. Unify the symbol and carry the restriction (or state it is implied by $p_{\rm det}=0$ beyond the horizon) in both places. Note also the overload of $\Sigma$ with the covariance matrices $\Sigma_i$; consider $S_{\rm glob}(h)$.

**F7 — Letter `C` denotes two different quantities within pitfall.tex.**
- $C(\bar z)$ = Eddington bias coefficient (eq:pitfall:hbias, l.27; also est eq:est:eddh, B eq:app:dhlaw, $C_{\rm meas}$).
- $C$ = $h$-independent bridge constant in $\Sigma_{\rm global}\approx C\,\beta_G$ (pitfall l.70–72; appendix_beta_g eq:app:bridge, $C=\bar n_{\rm gal}\langle R_{\rm eff}\rangle$; also fig:betag caption).
- Both appear in the same section. Fix: rename the bridge constant (e.g. $K$ or $C_{\rm cat}$) in pitfall l.70, fig:betag caption, and appendix_beta_g.

**F9 — In-catalogue sum domain: $\mathcal{G}_i / \mathcal{G}_i^{\rm sel}$ (framework) vs $\mathcal{B}$ (estimators).**
framework eq:lcat sums over galaxy sets $\mathcal{G}_i$ (numerator) and $\mathcal{G}_i^{\rm sel}$ (denominator); estimators eq:est:lcat sums over "$g \in \mathcal{B}$" where $\mathcal{B}$ is the localization *ball* (a region, used as an index set). Since the local-vs-global denominator set is the central estimator distinction of the paper, define once: $\mathcal{G}_i = \{g : (\Omega_g, z_g) \in \mathcal{B}_i\}$ and write both equations over $\mathcal{G}_i$ (and $\mathcal{G}^{\rm sel} \in \{\mathcal{G}_i, \mathcal{G}_{\rm all}\}$).

**F10 — Contract "z is CMB-frame" is contradicted for the headline real-data baselines.**
framework l.21–23 asserts "All redshifts $z$ are CMB-frame redshifts", but budget.tex l.29/46 states the committed baselines of Section 6 (realdata) used *heliocentric* redshifts (catalogue rebuilt later; net effect +0.15 per cent on $H_0$). realdata.tex itself never mentions this. Fix: qualify the framework sentence ("CMB-frame throughout, except the archived Section 6 baselines, which predate the rebuild — see Section 8/Table 3") or add the caveat where the seed600 data are introduced (realdata §6.1).

**F11 — Same missing reference requested under multiple citation keys (will produce duplicate bib entries).**
- Eddington 1913 under 4 keys: `MISSING:Eddington1913-correcting-statistics` (pitfall), `MISSING:Eddington1913-statistical-bias` (estimators), `MISSING:Eddington1913-bias` (coverage, appendix_volume_deconv), `MISSING:Eddington1913-noise-bias-correction` (appendix_eddington_m).
- Turski et al. 2023 under 3 keys: `-photoz-uncertainties` (introduction), `-photometric-redshift-dark-sirens` (postmortem), `-photoz` (codes).
- Gray et al. 2022 pixelated under 3 keys: `MISSING:Gray2022-pixelated-completeness` (intro, fw, C), `MISSING:Gray2022-pixelated` (codes), and `MISSING:Gray2023-pixelated` (appendix_sky l.111 — wrong year, and its inline comment gives the *wrong arXiv number* 2308.02281, which is the LOS z-prior paper, not the pixelated paper 2111.04629).
- Gray et al. 2023 LOS under 2 keys: `MISSING:Gray2023-gwcosmo-los-zprior` (intro, est, C) vs `MISSING:Gray2023-los-zprior` (codes).
Fix: one canonical key per work; correct the year/arXiv in appendix_sky_marginal.tex l.111.

### MINOR

**F8 — Event data: $x_i$ (framework) vs $x_{\rm GW}$ (estimators eq:est:Ng, appendix_sky l.14).** Same quantity, two symbols; estimators even cross-references "Section 2" where the symbol is $x_i$. Suggest $x_i$ throughout.

**F12 — $g$ is both the galaxy index (subscripts everywhere) and the shape function $g(z)$ in $w_{\rm pop}(z;h)=h^{-3}g(z)$** (estimators l.43; appendix_volume_deconv l.63, in the same paragraph as $p_g$, $Z_g$). Suggest $\tilde w(z)$ or $\varphi(z)$.

**F13 — $\beta$ (ecliptic latitude) undefined** in realdata l.9 ("six equal-$|\sin\beta|$ sky bands") and visually adjacent to $\beta_G$, $\beta_{\bar G}$. Add "with $\beta$ the ecliptic latitude".

**F14 — Bare $N$ for the event count** (pitfall l.76 "$N\approx500$", postmortem "N-amplified", appendix_beta_g l.85 "$N \simeq 500$") although framework defines $N_{\rm det}$; $N$ also risks collision with $N_g(h)$ and $\mathcal{N}$. Use $N_{\rm det}$.

**F15 — $\sigma_{z,g}$ vs $\sigma_z$** for the per-galaxy photometric scatter: framework eq:kernel uses the per-galaxy subscript, every other section uses plain $\sigma_z$ (declared "(per-galaxy)" in fw l.22). Pick one; if $\sigma_z$ is kept, drop the ",g" in eq:kernel.

**F16 — Dummy-variable capture in estimators eq:est:pgM:** $Z_M = \int \mathrm{d}M\;\mathcal{N}(M;M_g,\sigma_M^2)\,R_{\rm eff}(M)$ reuses $M$, which is free on the left-hand side $p_g(M)$. The appendix version (eq:eddm:prior) correctly uses $M'$. Change the estimators integrand to $M'$.

**F17 — Systematic duplication of displayed definitions is the vector for F1/F2/F4/F5:** $w_{\rm pop}$ is defined 4 times (eq:wpop, eq:pitfall:wpop, eq:est:wpop, eq:app:wpop), $s(z)$ 3 times, the Eddington shift 3 times (eq:pitfall:eddz, eq:est:eddz, eq:app:eddz), the $\Delta h$ law 3 times (eq:pitfall:hbias, eq:est:eddh, eq:app:dhlaw), $p_g(z)$ 3 times, $L_{\rm cat}$, $N_g/D_g$, $B_{\rm num}$, and the sky marginal twice each. Every duplicate drifted somewhere. Suggest: define once, cross-reference elsewhere ("cf. eq. (7)"), or add a build-time consistency pass after edits.

**F18 — Equation-reference style is mixed:** "equation~\eqref" (framework, estimators, coverage, sky, eddington_m), "Eq.~\eqref" (pitfall, postmortem, volume_deconv, beta_g), "eq.~\eqref" (gray_mapping, which mixes both forms internally), and one "Eq.~\ref" (volume_deconv table caption — renders without parentheses). MNRAS style is lowercase "equation (N)"; normalize.

**F19 — Degree notation inconsistent:** `$2\degr$` (pitfall l.54) vs `$2^\circ$` (appendix_sky l.132) for the same 2° sky error; `{\rm deg}^2` (appendix_sky) vs `\mathrm{deg}^2` (pitfall).

**F20 — `{\rm ...}` vs `\mathrm{...}` in math mode:** estimators (19), appendix_volume_deconv (19), appendix_sky (12), realdata (3), budget (3), codes (1), framework (1) use deprecated `{\rm }`; all other files use `\mathrm{}` exclusively. Also $\mathrm{d}V_\mathrm{c}$ (fw, pit) vs $\mathrm{d}V_c$ (est, B, E) — upright vs italic subscript c. Cosmetic (renders identically) but normalize before submission.

**F21 — "per cent" style:** `per~cent` (pitfall ×9, postmortem ×1) vs `per cent` (all other files). MNRAS accepts "per cent"; unify (non-breaking space is fine, but be consistent). The `\%` uses in coverage are inside a table header — acceptable.

**F22 — Symbols used but never defined (each needs a half-sentence gloss):** $z_{\rm min}$ (framework eq:Dh/eq:betabar lower limit — value/meaning never given, unlike $z_{\max}(h)$); $\kappa$ (budget l.20, Fisher condition number); $M, e_0, p_0$ (budget l.30 timeout-binning tuple — EMRI mass, initial eccentricity, initial semi-latus rectum); $\sigma_{d_L}$ (framework l.149, pitfall l.54 — GW distance uncertainty, deducible but never introduced); $\Phi$ (appendix_volume_deconv l.45 — standard normal CDF). Also: the production fiducial $\Omega_{\rm m}=0.2726$ first appears in budget.tex although framework §2.1 introduces the ΛCDM background — state it there.

**F23 — Sky-vector notation drift between framework and appendix A:** framework sets $\Omega=(\phi,\theta)$ and $\hat\mu_i$; appendix_sky uses $\omega=(\phi,\theta)$ (reserving $\Omega$ for solid angle) and drops the hat/index on $\mu$, $\Sigma$. Internally consistent but the mapping is never stated; one sentence ("we write $\omega$ for the sky pair, reserving $\Omega$ for solid angle; event index suppressed") would fix it — or adopt $\omega$ in framework too, since framework itself also uses $\mathrm{d}\Omega$ as solid-angle measure in the same paragraph where $\Omega$ is a coordinate pair.

**F24 — $w_g$ vs $w_G$:** per-galaxy rate weight vs in-catalogue hypothesis weight differ only in subscript case and co-occur (framework §2.5, realdata l.10, appendix_beta_g). Consider $P_G(h)$ or $w_{\rm cat}(h)$ for the hypothesis weight.

**F25 — $s$ is both Gray's host indicator ($p(s\mid z)\,p(s\mid M)$: framework l.124, estimators l.22, gray_mapping eq:gray:a10) and the Eddington slope $s(z)$ (pitfall/estimators/volume_deconv).** Contexts are distant, but both are load-bearing; a footnote at the first $s(z)$ definition or renaming the slope to $s_{\rm pop}(z)$ would remove the ambiguity.

---

## 3. Contract compliance summary

| Contract item | Status | Notes |
|---|---|---|
| $h = H_0/100\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$ | **PASS** | defined in abstract and framework §2.1; used consistently; no bare `H` in the manuscript |
| $z$ CMB-frame | **PASS with exception** | asserted globally in framework; contradicted for the Section 6 baselines by budget.tex (F10) — needs an explicit caveat |
| $\beta_G$ | **PASS** | eq:betag; used consistently ($\beta_G$, $\beta_{\bar G}$, $w_G=\beta_G/D$); watch undefined latitude $\beta$ (F13) |
| $p_{\rm det}$ | **PASS** | always lowercase $p_{\rm det}$; denominators-only rule stated and respected in every equation |
| $\mathrm{d}V_c/\mathrm{d}z$ | **FAIL in framework** | framework eq:wpop uses $\mathrm{d}V_c/(\mathrm{d}z\,\mathrm{d}\Omega)$; all other sections use the contract form (F1) |

## 4. Equation formatting summary

- **`\mathrm{d}` differentials: PASS.** Every integral and derivative in all 16 files uses upright `\mathrm{d}`; no italic-d differentials found (checked all `\int`, `\frac{d...}`, `d\ln` sites).
- **`\left`/`\right` pairing: PASS.** All pairs balanced per file (appendix_eddington_m 4/4, pitfall 2/2, estimators 1/1, volume_deconv 1/1, sky_marginal 1/1 — the apparent extra `\left` there is the substring of `\leftrightarrow` on l.106, a false positive).
- Residual formatting nits: one `Eq.~\ref` instead of `\eqref` (F18); `{\rm}` vs `\mathrm{}` split (F20); `\degr` vs `^\circ` (F19); `per~cent` vs `per cent` (F21).

## 5. Suggested priority order

1. F1 (w_pop 4π) — touches framework/pitfall/estimators/coverage/appendices; decide the convention first.
2. F10 (CMB-frame caveat) + F13 (define latitude β) — realdata/framework, quick.
3. F2 (Gaussian σ vs σ²) — 8 mechanical edits.
4. F3+F4 (rename appendix-B f(z) → 𝒟(z); unify C(z̄) definitions).
5. F5 (Σ_22 → Σ_uu), F6 (Σ_glob), F7 (bridge constant C → K).
6. F9 (𝒢 vs ℬ), F11 (citation keys) — before the bibliographer pass.
7. Minor sweep: F8, F12–F25.
