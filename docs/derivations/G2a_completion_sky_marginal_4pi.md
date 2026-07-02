# G2a — Isotropic sky-marginalization of the completion term: first-principles derivation and code audit

**Target code:** `master_thesis_code/bayesian_inference/bayesian_statistics.py`,
completion-numerator block, lines 1640–1702 (working tree of branch
`physics/derail-completion-4pi`), introduced by commit `cb16142`
("[PHYSICS] sky-marginalise completion term B_num over isotropic 1/(4π) prior").

**Claim under audit** (commit message and inline comment at
`bayesian_statistics.py:1645–1652`): the isotropic sky-marginal of the 3D GW
measurement Gaussian equals
$\frac{1}{4\pi}\,\mathcal{N}\!\left(u;\,1,\,\Sigma_{22}\right)$
with $u = d_L/\hat d_L$ the luminosity-distance fraction and
$\Sigma_{22}$ the $(d_L\text{-frac},d_L\text{-frac})$ element of the
covariance $\Sigma = (\Sigma^{-1})^{-1}$ stored as `cov_inv`.

---

## 1. Setup: the 3D GW measurement Gaussian as implemented

The pipeline approximates the per-event GW likelihood by a trivariate Gaussian
in the coordinates $x = (\phi, \theta, u)$, where $\phi = \phi_S$ (ecliptic
azimuth), $\theta = q_S$ (ecliptic polar angle, $\theta \in [0,\pi]$; see
`datamodels/detection.py:83,119–122`), and $u = d_L/\hat d_L$:

$$
p(x_{\rm GW} \mid \phi, \theta, d_L)
\;\simeq\;
\mathcal{N}_3\!\big( (\phi,\theta,u);\ \mu,\ \Sigma \big)
= \frac{1}{(2\pi)^{3/2}\sqrt{\det\Sigma}}
\exp\!\left[-\tfrac12 (x-\mu)^{\mathsf T}\,\Sigma^{-1}\,(x-\mu)\right],
\tag{1}
$$

with mean $\mu = (\hat\phi, \hat\theta, 1)$
(`bayesian_statistics.py:1052`: `_means_3d[slot] = [det.phi, det.theta, 1]`)
and covariance $\Sigma$ assembled from the Cramér–Rao bound (CRB) of the
14-parameter EMRI Fisher matrix (`bayesian_statistics.py:982–1000`):

$$
\Sigma =
\begin{pmatrix}
\sigma_\phi^2 & C_{\theta\phi} & C_{d_L\phi}/\hat d_L\\[2pt]
C_{\theta\phi} & \sigma_\theta^2 & C_{d_L\theta}/\hat d_L\\[2pt]
C_{d_L\phi}/\hat d_L & C_{d_L\theta}/\hat d_L & \sigma_{d_L}^2/\hat d_L^{\,2}
\end{pmatrix},
\qquad
\Sigma^{-1} \equiv \texttt{cov\_inv\_3d}.
\tag{2}
$$

Two conventions matter for everything below:

1. **Coordinate measure.** The Fisher matrix is computed with respect to the
   bare coordinates $(\phi_S, q_S)$ (`delta_phiS_delta_phiS`,
   `delta_qS_delta_qS` in the CRB CSV; `detection.py:120,122`). Hence Eq. (1)
   is a probability **density with respect to the Lebesgue measure**
   $d\phi\, d\theta\, du$, *not* with respect to the solid-angle measure
   $d\Omega = \sin\theta\, d\theta\, d\phi$. The project itself acknowledges
   the distinction: the sky-localization *area* is computed as
   $\Delta\Omega = 2\pi\,|\sin\theta|\sqrt{\sigma_\phi^2\sigma_\theta^2 -
   C_{\theta\phi}^2}$ (`detection.py:16–37`), which carries the
   $|\sin\theta|$ Jacobian explicitly.

2. **In-catalogue usage.** The same Gaussian (1) is evaluated *pointwise* at
   candidate-host coordinates $(\phi_g, \theta_g, u(z))$ in the in-catalogue
   numerator (`bayesian_statistics.py:1896–1901`), with no $\sin\theta$
   factor. So the pipeline-wide convention is "coordinate-density evaluated at
   points"; any sky *integral* must then be carried out in the same
   convention, i.e. with the explicit Jacobian $\sin\theta$ when an isotropic
   (per-steradian) prior is intended.

## 2. The completion numerator in the Gray et al. formalism

The out-of-catalogue ("completion") branch of the dark-siren likelihood is,
in Gray et al. (2020), arXiv:1908.06050, Appendix A.2c ["Likelihood when host
is not in catalog", Eq. (A.14); published as Phys. Rev. D 101, 122001, main-text
numbering Eq. (32) as cited in the code], the numerator of

$$
p(x_{\rm GW}\mid \bar G, D_{\rm GW}, H_0)
\;\propto\;
\int dz\, d\Omega\; p(x_{\rm GW}\mid z, \Omega, H_0)\;
\big[1 - f(z,\Omega)\big]\; p(z)\, p(\Omega),
\tag{3}
$$

where the appendix of that paper states explicitly that "$p(z)$ is the prior
distribution of galaxies in the universe, taken to be uniform in comoving
volume-time, [and] $p(\Omega)$ is the prior on galaxy sky location, assumed
**uniform over the celestial sphere**", i.e.

$$
p(\Omega) = \frac{1}{4\pi}, \qquad \int_{S^2} p(\Omega)\, d\Omega = 1,
\qquad
p(z) \propto \frac{1}{1+z}\frac{dV_c}{dz\,d\Omega}.
\tag{4}
$$

The factor $[1-f]$ is the smooth-completeness form of the apparent-magnitude
Heaviside $\Theta(m - m_{\rm th})$ after marginalizing the Schechter absolute
magnitudes [Gray et al. 2020, Eqs. (A.12)–(A.14)]. The pixelated successor
formalism [Gray et al. 2023, arXiv:2308.02281, Eq. (2.3)] writes the same sky
marginalization as an explicit sum over $N_{\rm pix}$ **equal-area** HEALPix
pixels, $p(\Omega_j \mid I) = 1/N_{\rm pix}$, which "come out the front and
cancel in numerator and denominator" — equal-area pixelization implements the
$\sin\theta$ measure exactly by construction. The code's selection denominator
$D(h)$ follows exactly that convention
(`precompute_completion_denominator`, `bayesian_statistics.py:284–338`):
$D(h) = \int \frac{1}{N_{\rm pix}}\sum_k p_{\rm det}(d_L(z,h),\Omega_k)\,
\frac{dV_c}{dz\,d\Omega}\frac{dz}{1+z}$, in units of ${\rm Mpc}^3\,{\rm sr}^{-1}$.

So the object the code must compute for the completion numerator is

$$
B_{\rm num}(h) = \int_{z_-}^{z_+} dz\;\big[1 - f_k(z)\big]\;
\underbrace{\left[\int_{S^2} \frac{d\Omega}{4\pi}\,
\mathcal{N}_3\big((\phi,\theta,u(z,h));\,\mu,\Sigma\big)\right]}_{\displaystyle
\equiv\; \bar p_{\rm GW}(u)}
\;\frac{dV_c}{dz\,d\Omega}\,\frac{1}{1+z},
\tag{5}
$$

with $u(z,h) = d_L(z,h)/\hat d_L$ and $f_k$ the per-pixel completeness at the
event pixel (Change 5.3; Gray–Messenger–Veitch 2022, arXiv:2111.04629, Eq. (5)).

## 3. Exact factorization of the sky integral

Split $x = (\omega, u)$ with $\omega = (\phi,\theta)$ and block-decompose
$\Sigma$. Every multivariate Gaussian factorizes exactly into marginal ×
conditional [Bishop 2006, PRML, Eqs. (2.81)–(2.82); already used in this file
for the BH-mass branch, `bayesian_statistics.py:1083–1089`]:

$$
\mathcal{N}_3\big((\omega,u);\mu,\Sigma\big)
= \mathcal{N}_1\!\big(u;\ \mu_u,\ \Sigma_{uu}\big)\;
\mathcal{N}_2\!\big(\omega;\ \mu_{\omega|u},\ \Sigma_{\omega|u}\big),
\tag{6}
$$

with

$$
\mu_u = 1,\qquad
\Sigma_{uu} = \Sigma_{22} = \frac{\sigma_{d_L}^2}{\hat d_L^{\,2}},\qquad
\mu_{\omega|u} = \mu_\omega + \Sigma_{\omega u}\,\Sigma_{uu}^{-1}(u-1),\qquad
\Sigma_{\omega|u} = \Sigma_{\omega\omega} -
\Sigma_{\omega u}\Sigma_{uu}^{-1}\Sigma_{u\omega}.
\tag{7}
$$

Two consequences, both load-bearing for the audit:

- **The $u$-factor carries the *marginal* variance $\Sigma_{22}$** (element of
  the covariance), *not* the conditional variance
  $1/(\Sigma^{-1})_{22} = \Sigma_{22} - \Sigma_{u\omega}\Sigma_{\omega\omega}^{-1}\Sigma_{\omega u} \le \Sigma_{22}$.
  The code computes `_comp_cov_3d = np.linalg.inv(_comp_cov_inv_3d)` and takes
  `_comp_cov_3d[2, 2]` (`bayesian_statistics.py:1653–1654`) — this is the
  **correct marginal** variance. (The *old*, pre-`cb16142` peak evaluation
  $\mathcal{N}_3((\hat\phi,\hat\theta,u))$ instead had the conditional
  precision $(\Sigma^{-1})_{22}$ in its exponent — a too-narrow $u$-Gaussian
  on top of the $\sim\!\!1/(2\pi\sigma_\phi\sigma_\theta)$ peak-density
  prefactor.)
- $\mathcal{N}_2$ integrates to unity **under $d\phi\,d\theta$**, so the
  sky integral in Eq. (5) becomes, exactly,

$$
\bar p_{\rm GW}(u)
= \frac{1}{4\pi}\,\mathcal{N}_1(u;1,\Sigma_{22})
\int_0^{2\pi}\!\!d\phi\int_0^{\pi}\!\!d\theta\;
\sin\theta\;\mathcal{N}_2\big(\omega;\mu_{\omega|u},\Sigma_{\omega|u}\big)
= \frac{1}{4\pi}\,\mathcal{N}_1(u;1,\Sigma_{22})\;
\mathbb{E}_{\mathcal{N}_2}\!\big[\sin\theta\big].
\tag{8}
$$

## 4. The compact-support / flat-sky approximation and its error budget

Eq. (8) is exact except for the domain of the $\theta$ (and wrapped $\phi$)
integral. Evaluating $\mathbb{E}[\sin\theta]$ with the Gaussian extended to
$\theta \in \mathbb{R}$ gives the closed form (from
$\mathbb{E}[e^{i\theta}] = e^{i m - s^2/2}$ for $\theta \sim \mathcal{N}(m,s^2)$):

$$
\mathbb{E}_{\mathcal{N}_2}[\sin\theta]
= \sin\!\big(\mu_{\theta|u}\big)\,
\exp\!\left[-\tfrac12\big(\Sigma_{\omega|u}\big)_{\theta\theta}\right].
\tag{9}
$$

Hence the **exact narrow-beam isotropic sky-marginal** is

$$
\boxed{\;
\bar p_{\rm GW}(u)
= \frac{\sin\!\big(\mu_{\theta|u}\big)\,
e^{-(\Sigma_{\omega|u})_{\theta\theta}/2}}{4\pi}\;
\mathcal{N}_1\!\big(u;\,1,\,\Sigma_{22}\big)
\;\approx\;
\frac{\sin\hat\theta}{4\pi}\;
\mathcal{N}_1\!\big(u;\,1,\,\Sigma_{22}\big).
\;}
\tag{10}
$$

The code implements $\bar p_{\rm GW}^{\rm code}(u) = \frac{1}{4\pi}
\mathcal{N}_1(u;1,\Sigma_{22})$ (`bayesian_statistics.py:1677–1679`), i.e.
Eq. (10) **with $\sin\hat\theta \to 1$**. The individual approximation errors,
quantified at the median EMRI sky error $\Delta\Omega \sim 0.2\,{\rm deg}^2
= 6.1\times10^{-5}\,{\rm sr}$ — using the project's own definition
$\Delta\Omega = 2\pi\sin\theta\,\sigma_\phi\sigma_\theta$ (`detection.py:25`),
so $\sigma_\theta \sim \sigma_\phi \sim
\sqrt{\Delta\Omega/(2\pi\sin\hat\theta)} \approx 3.3\times10^{-3}\,{\rm rad}
\approx 0.19^\circ$ at $\sin\hat\theta \approx 0.87$ — are:

| # | Approximation | Relative error | Verdict |
|---|---|---|---|
| (a) | Extending $\theta\in[0,\pi]$, wrapped $\phi\in[0,2\pi)$ to $\mathbb{R}$ (compact support on the chart) | $\mathcal{O}\!\big(e^{-\hat\theta^2/2\sigma_\theta^2}\big)$; at $\hat\theta = 60^\circ$, $\hat\theta/\sigma_\theta \approx 320$, error $\sim e^{-5\times10^4}$ | negligible |
| (b) | Gaussian curvature of $\sin\theta$ (flat-sky): the factor $e^{-(\Sigma_{\omega\vert u})_{\theta\theta}/2}$ in Eq. (9) | $\tfrac12\sigma_\theta^2 \approx 5.6\times10^{-6}$ | negligible |
| (c) | $u$-dependence of the conditional mean, $\mu_{\theta\vert u} = \hat\theta + (\Sigma_{\theta u}/\Sigma_{uu})(u-1)$, over the $\pm4\sigma$ integration window | $\lesssim 4\,\lvert\rho_{\theta u}\rvert\,\sigma_\theta\,\lvert\cot\hat\theta\rvert \sim 10^{-2}\lvert\rho_{\theta u}\rvert\cot\hat\theta$, i.e. sub-percent and $h$-smooth | negligible |
| (d) | **Dropping $\sin\hat\theta$ entirely** ($\sin\hat\theta \to 1$) | $1-\sin\hat\theta$; median 13% ($\sin\hat\theta = \sqrt{3}/2$ for isotropic events), mean weight $\mathbb{E}[\sin\hat\theta]=\pi/4 \approx 0.785$, unbounded only as $\hat\theta \to \{0,\pi\}$ | **the one real deviation — see §5** |

So the *functional form* — pulling the $d_L$ marginal out of the sky integral
with mean 1 and marginal variance $\Sigma_{22}$, prefactor $\propto 1/4\pi$ —
is justified to $\mathcal{O}(10^{-5})$ (narrow sky error vs. the $4\pi$ prior
scale: $\Delta\Omega/4\pi \sim 5\times10^{-6}$), *provided* the $\sin\hat\theta$
factor is kept.

## 5. Measure check: where the $\sin\theta$ went

**Is the code integrating $d\Omega$ with the correct $\sin\theta$ measure?
No — it approximates.** The chain is:

- The isotropic prior is $p(\Omega)\,d\Omega = \frac{\sin\theta}{4\pi}\,
  d\theta\, d\phi$ (Eq. 4).
- The GW Gaussian is a **coordinate** density w.r.t. $d\phi\,d\theta$ (§1,
  convention 1), which integrates to 1 *without* $\sin\theta$.
- Therefore the exact isotropic marginal picks up the Jacobian evaluated
  under the beam, $\approx \sin\hat\theta$ (Eq. 10). The implementation at
  `bayesian_statistics.py:1677–1679` sets this factor to 1.

Two readings, and why neither rescues exactness:

1. *"The sky Gaussian is a per-steradian density on the sphere."* Then
   $\int \frac{d\Omega}{4\pi} q(\Omega|u) = \frac{1}{4\pi}$ exactly and the code
   line is exact — but then the **in-catalogue** numerator, which evaluates the
   *same* Gaussian object at galaxy coordinates with the coordinate
   normalization $(2\pi)^{-3/2}(\det\Sigma)^{-1/2}$
   (`bayesian_statistics.py:1896–1901`), is off by the same $\sin\theta$
   relative to a per-steradian density. The inconsistency between the two
   branches is invariant under the choice of reading.
2. *"The Gaussian is the likelihood function; only ratios matter."* Correct —
   x-space normalization constants common to both numerator branches cancel in
   the normalized $H_0$ posterior. But the $\sin\hat\theta$ is **not** common:
   it multiplies $B_{\rm num}$ only, not $\beta_G L_{\rm cat}$. In the mixture
   $p_i(h) = \big[\beta_G(h)\,L_{\rm cat}(h) + B_{\rm num}(h)\big]/D(h)$
   (`bayesian_statistics.py:1715–1722`), a per-event constant multiplying one
   term changes the catalogue-vs-completion *weight* and therefore the shape
   of $p_i(h)$ — it does not cancel.

**Net effect of the deviation:** $B_{\rm num}$ is over-weighted relative to the
in-catalogue term by $1/\sin\hat\theta \in [1,\infty)$: median $\approx 1.15$,
mean $\pi/2 \approx 1.57$ for isotropically distributed events; large only for
near-polar events (which are also where the Fisher sky block degenerates and
events are excluded by the condition-number gate,
`bayesian_statistics.py:1033–1049`). It is $h$-independent per event and
$\mathcal{O}(1)$ — three orders of magnitude below the
$2/(\sigma_\phi\sigma_\theta) \sim 1.6\times10^3$–$1.8\times10^5$ peak-density
over-count that commit `cb16142` removed. It is a cousin of the commission
audit's existing LOW finding #7 ("Sky Gaussian omits sinθ metric that the host
BallTree search applies", `results/commission_20260701/audit/physics.md:105`).
A one-line cure preserving all conventions:
`p_gw *= np.sin(self.detection.theta)` inside the completion branch
(equivalently multiply `1/(4π)` by $\sin\hat\theta$).

## 6. Dimensional analysis of the full completion integrand

Integrand at `bayesian_statistics.py:1698`:
$\big(1-f_k(z)\big)\cdot \bar p_{\rm GW}^{\rm code}(u)\cdot
\frac{dV_c}{dz\,d\Omega}\cdot\frac{1}{1+z}$.

| Factor | Units |
|---|---|
| $1-f_k(z)$ | dimensionless (clipped to $[0,1]$, line 1690–1697) |
| $\frac{1}{4\pi}$ | ${\rm sr}^{-1}$ |
| $\mathcal{N}_1(u;1,\Sigma_{22})$ | per unit $u$; $u$ dimensionless $\Rightarrow$ dimensionless |
| $\frac{dV_c}{dz\,d\Omega}$ | ${\rm Mpc}^3\,{\rm sr}^{-1}$ (per steradian; `physical_relations.py:387–406`) |
| $\frac{1}{1+z}$, $dz$ | dimensionless |
| $B_{\rm num}$ | ${\rm Mpc}^3\,{\rm sr}^{-2}$ |

Cross-check against the in-catalogue term: $\beta_G$ has the units of $D(h)$,
${\rm Mpc}^3\,{\rm sr}^{-1}$; $L_{\rm cat} = \sum w_g N_g / \sum w_g D_g$ with
$N_g = \mathcal{N}_3(\cdot)\times p(z)$ of units
${\rm rad}^{-2}$ (coordinate sky density × dimensionless-$u$ density ×
per-$z$ density integrated $dz$) and $D_g$ dimensionless, so
$\beta_G L_{\rm cat} \sim {\rm Mpc}^3\,{\rm sr}^{-1}\,{\rm rad}^{-2}
\equiv {\rm Mpc}^3\,{\rm sr}^{-2}$ under the flat-sky identification
${\rm rad}^2 \leftrightarrow {\rm sr}$ — **the same units as $B_{\rm num}$**.
The two numerator terms are dimensionally commensurable, and
$p_i = (\beta_G L_{\rm cat} + B_{\rm num})/D(h) \sim {\rm sr}^{-1}$ carries a
per-event constant unit that cancels in the normalized posterior. (The
${\rm rad}^2$-vs-${\rm sr}$ identification is precisely where the
$\sin\hat\theta$ of §5 hides: dimensional analysis alone cannot see an
$\mathcal{O}(1)$ dimensionless Jacobian.)

## 7. Limiting cases

1. **$f \to 1$ (complete catalogue).** $(1-f_k) \to 0 \Rightarrow
   B_{\rm num} \to 0$ pointwise in the integrand; also
   $\beta_{\bar G} \to 0$, $w_G = \beta_G/D \to 1$, and
   $p_i \to \beta_G L_{\rm cat}/D$ — the pure in-catalogue Gray limit
   [arXiv:1908.06050, Eq. (A.14) with the Heaviside support vanishing].
   Verified by `test_completion_vanishes_for_complete_catalogue`
   (`master_thesis_code_test/test_completion_sky_marginal.py`). ✓
2. **$f \to 0$ (empty catalogue).** $B_{\rm num} \to \int
   \frac{1}{4\pi}\mathcal{N}_1(u)\,\frac{dV_c}{dz\,d\Omega}\frac{dz}{1+z}$:
   the catalogue-free dark-siren likelihood — GW distance shell × comoving
   volume prior — matching Gray et al.'s observation that highly incomplete
   catalogues drive the posterior toward the volume-prior/selection-dominated
   ("population-only") limit [arXiv:1908.06050 §V; arXiv:2308.02281 footnote 7]. ✓
3. **Sky error $\to 0$.** Eq. (10) is manifestly independent of
   $\Sigma_{\omega\omega}$: a perfectly localized event still has only prior
   weight $\sin\hat\theta\,d\theta d\phi/4\pi$ for the dark host to lie in its
   beam. The new expression stays finite and constant while the old peak
   density diverged as $\big(2\pi\sqrt{\det\Sigma_{\omega|u}}\big)^{-1}
   \sim 1/(2\pi\sigma_\phi\sigma_\theta)$. Verified by
   `test_sky_marginal_is_finite_as_localisation_sharpens`. ✓
4. **$\sigma_{d_L} \to 0$.** $\mathcal{N}_1(u) \to \delta(u-1)$ and
   $B_{\rm num} \to \frac{1-f_k(z^*)}{4\pi}\,
   \frac{dV_c}{dz\,d\Omega}\frac{1}{1+z^*}\,
   \frac{\hat d_L}{|dd_L/dz|}\Big|_{z^*: d_L(z^*,h)=\hat d_L}$ — finite, with
   the correct $dz/du$ Jacobian emerging automatically because the quadrature
   is in $z$. ✓
5. **Old/new magnitude ratio** (regression bookkeeping): for a factorized
   $\Sigma$, $\text{old}/\text{new} = 4\pi/(2\pi\sigma_\phi\sigma_\theta) =
   2/(\sigma_\phi\sigma_\theta)$ — $\approx 1.6\times10^3$ at
   $\sigma_{\rm sky}=2^\circ$ (commit message, and asserted to 5% in
   `test_completion_sky_marginal_reduces_magnitude`), and
   $\approx 1.8\times10^5$ at the median $0.2\,{\rm deg}^2$ localization.
   The de-rail mechanism claim (completion term previously dominating by
   $\sim\!10^3$–$10^5$) is arithmetically consistent. ✓

## 8. Comparison with Gray et al.

- **Gray et al. 2020 (arXiv:1908.06050), Eq. (A.14)/(A.19) [published
  Eqs. (32)/(33)]:** structure (3)–(4) — GW likelihood × isotropic
  $p(\Omega)=1/4\pi$ × uniform-comoving-volume $p(z)$ × incompleteness
  weight — is exactly what Eq. (5) encodes; the code's $1/(1+z)$ matches
  "comoving volume-*time*" (detector-frame rate,
  Mandel–Farr–Gair 2019, arXiv:1809.02063). Their numerical implementation
  integrates the posterior sky map over the sphere, which carries the
  $\sin\theta$ measure exactly.
- **Gray et al. 2023 (arXiv:2308.02281), Eq. (2.3):** the sky marginal is a
  sum over **equal-area** pixels with $p(\Omega_j|I) = 1/N_{\rm pix}$; equal
  area ⇒ the measure is exact by construction. The code's $D(h)$
  (`bayesian_statistics.py:285–317`) reproduces this pixel-sum convention;
  $B_{\rm num}$'s $\frac{1}{4\pi}\int d\Omega \to \frac{1}{4\pi}$ pull-out is
  its continuum analogue but drops the $\sin\hat\theta$ Jacobian of the
  *coordinate*-space Gaussian (§5). D(h) and $B_{\rm num}$ are therefore
  consistent with each other **up to that same factor**, which is common to
  neither.

## VERDICT

**DEVIATION FOUND** (sub-leading; the headline de-rail fix is confirmed).

- **Confirmed:** the isotropic $1/(4\pi)$ sky-marginalization replacing the
  peak sky density is the correct reading of Gray et al. 2020 Eq. (A.14)
  [published Eq. (32)]; the $d_L$-fraction Gaussian correctly uses the
  **marginal** variance $\Sigma_{22}$ of the covariance (not the conditional
  precision) with mean $\mu_u = 1$
  (`bayesian_statistics.py:1653–1655, 1677–1679`); the factorization
  pull-out is valid to relative error $\lesssim 10^{-5}$ (+ sub-percent
  $h$-smooth correlation drift) at the median $0.2\,{\rm deg}^2$ EMRI sky
  error; dimensions and all four limiting cases check out; the claimed
  $\sim\!1640\times$ magnitude reduction at $\sigma_{\rm sky}=2^\circ$ is
  reproduced by $2/(\sigma_\phi\sigma_\theta)$.
- **Deviation:** `bayesian_statistics.py:1677–1679` — the exact
  narrow-beam isotropic marginal is
  $\frac{\sin\hat\theta}{4\pi}\mathcal{N}_1(u;1,\Sigma_{22})$
  (Eq. 10); the implemented $\frac{1}{4\pi}\mathcal{N}_1$ omits the
  $\sin\hat\theta$ Jacobian of the solid-angle measure
  ($d\Omega = \sin\theta\,d\theta\,d\phi$) that the coordinate-space
  ($\phi_S, q_S$) Fisher Gaussian requires. This over-weights
  $B_{\rm num}$ relative to $\beta_G L_{\rm cat}$ by
  $1/\sin\hat\theta$ per event (median $\approx 1.15$, mean $\pi/2$,
  divergent only toward the excluded-degenerate poles); it is
  $h$-independent per event but shape-affecting through the
  catalogue/completion mixture. Severity: LOW (same family as commission
  physics-audit finding #7), $\mathcal{O}(10^3)$ smaller than the bug fixed
  by `cb16142`. Suggested cure: multiply the completion `p_gw` by
  $\sin(\hat\theta)$ (`self.detection.theta`), via the
  `/physics-change` protocol.
