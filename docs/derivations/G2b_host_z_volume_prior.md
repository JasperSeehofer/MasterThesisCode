# G2b — The volume-consistent host-redshift prior (`volume_deconv` kernel)

**Scope.** This note derives the per-galaxy host-redshift posterior implemented as the
`volume_deconv` kernel in `single_host_likelihood`
(`master_thesis_code/bayesian_inference/bayesian_statistics.py:1844-1914`, introduced in
commit `6d4c4e1`, "[PHYSICS] de-rail in-catalogue H0 normalization via normalization_mode"),
shows that omitting the prior produces an Eddington-type bias with the
$\sigma_z^2$ law measured empirically by the commission's coverage test
(`results/commission_20260701/scratch/d2/NOTE_calibration_findings.md`, Results 1–2), and
verifies the per-galaxy renormalization $Z_g$ and the "$dV_c$ counted once" measure
symmetry between numerator and denominator. All numerical checks below were re-computed
independently for this note (flat $\Lambda$CDM, $\Omega_m = 0.3$, $h_{\rm true} = 0.72$,
matching the commission synthetic).

---

## 1. The correct per-galaxy host-redshift posterior

### 1.1 Ingredients

**Photo-z likelihood.** The catalogue reports, for galaxy $g$, an observed redshift
$z_g$ with Gaussian error $\sigma_z$, i.e. a *likelihood* for the true redshift $z$:

$$
p(z_g \mid z) \;=\; \mathcal{N}(z_g;\, z,\, \sigma_z)
\;=\; \frac{1}{\sqrt{2\pi}\,\sigma_z}\,
\exp\!\left[-\frac{(z_g - z)^2}{2\sigma_z^2}\right].
$$

Because the Gaussian is symmetric in $(z_g - z)$, the same expression read as a function
of $z$ at fixed $z_g$ is numerically $\mathcal{N}(z;\,z_g,\,\sigma_z)$ — this is exactly
what `scipy.stats.norm(loc=host_z, scale=host_z_error).pdf(z)` evaluates at
`bayesian_statistics.py:1842`. But a likelihood is *not* a posterior: turning it into a
probability density for the true $z$ requires a prior.

**Population prior.** EMRI hosts are not uniformly distributed in $z$; they are drawn
from the astrophysical event *rate*. The number of EMRI events per unit **detector**
time per unit true redshift is

$$
\frac{dN}{dt_{\rm det}\,dz}
\;=\; \mathcal{R}(z)\,\frac{dV_c}{dz}\,\frac{dt_{\rm src}}{dt_{\rm det}}
\;=\; \mathcal{R}(z)\,\frac{dV_c}{dz}\,\frac{1}{1+z},
\tag{1.1}
$$

where $\mathcal{R}(z)$ is the comoving, **source-frame** rate density
(events per comoving volume per source-frame time), $dV_c/dz$ is the comoving volume
element, and $dt_{\rm src}/dt_{\rm det} = 1/(1+z)$ is cosmological time dilation: a
clock at redshift $z$ ticks $(1+z)$ times slower as seen by the detector, so a
source-frame *rate* is suppressed by exactly one factor of $1/(1+z)$ when counted in
detector time. References: Babak et al. (2017), arXiv:1703.09722, Sec. III.4 (the LISA
EMRI detected-rate integral carries $dV/dz$ and the time-dilation factor); Mandel, Farr
& Gair (2019), arXiv:1809.02063, Sec. 2 (detector-frame population rate density);
Hogg (1999), arXiv:astro-ph/9905116, Eq. (28) for $dV_c/dz$; Gray et al. (2020),
arXiv:1908.06050, Appendix A.2.3 (Eqs. 31–33 use this volume element as the
completion-term prior).

**Cross-check against the project's own rate module.** `master_thesis_code/emri_rate.py`
implements precisely this bookkeeping. `R_EMRI(z, M)`
(`emri_rate.py:260-293`) is documented and implemented as the *intrinsic, comoving,
source-frame* rate density of the Babak et al. (2017) M1 model, with the docstring
stating explicitly that "$dV_c/dz$ and the time dilation $1/(1+z)$ are NOT included
here and must be supplied by the caller exactly once each." The caller is
`p_pop_unnormalized` (`emri_rate.py:296-331`), whose docstring equation is

$$
p_{\rm pop}(z, M) \;\propto\; R_{\rm EMRI}(z, M)\,\frac{1}{1+z}\,\frac{dV_c}{dz},
\tag{1.2}
$$

and whose body (`emri_rate.py:328-330`) is literally
`R_EMRI(z, M) / (1.0 + z) * dVc_dz`. Under the module's $p_0 = 1$ surrogate
(`p0_cusp_retention`, `emri_rate.py:198-228`), $R_{\rm EMRI}$ is $z$-independent, so
after marginalizing the mass the $z$-shape of the population prior is exactly

$$
w_{\rm pop}(z) \;\equiv\; \frac{1}{1+z}\,\frac{dV_c}{dz}.
\tag{1.3}
$$

The same weight appears verbatim in the selection denominator
$D(h) = \int \overline{p_{\rm det}}(z)\, \frac{dV_c}{dz}\frac{dz}{1+z}$
(`bayesian_statistics.py:336`, `return ... p_det * dVc / (1.0 + z)`), in
$\beta_{\bar G}$ (`:479`), and in the completion numerator $B_{\rm num}$ (`:1698`).
So Eq. (1.3) is the unique population weight consistent with the rest of the pipeline
and with the event generator.

### 1.2 The posterior

By Bayes' theorem, the density of the *true* host redshift given the catalogue
measurement, for a galaxy known to be a potential EMRI host (i.e. drawn from the event
population), is

$$
\boxed{\;
p_g(z) \;\equiv\; p(z \mid z_g)
\;=\; \frac{\mathcal{N}(z_g;\, z,\, \sigma_z)\; w_{\rm pop}(z)}{Z_g},
\qquad
Z_g \;=\; \int dz'\; \mathcal{N}(z_g;\, z',\, \sigma_z)\; w_{\rm pop}(z')\;
}
\tag{1.4}
$$

with $w_{\rm pop}$ from Eq. (1.3). This is exactly what the code builds:
the unnormalized integrand `_z_prior_unnorm`
(`bayesian_statistics.py:1857-1860`) is
`comoving_volume_element(z, h) / (1 + z) * norm(host_z, host_z_error).pdf(z)`;
$Z_g$ (`_z_prior_norm`) is its `fixed_quad` integral over
$[z_g - 4\sigma_z,\, z_g + 4\sigma_z]$ (`:1862-1869`); and
`galaxy_redshift_prior_pdf` (`:1873-1878`) returns
$\mathcal{N}\cdot w_{\rm pop}/Z_g$ when `normalization_mode == "volume_deconv"` and the
bare Gaussian otherwise.

**Dimensional analysis.** $[\mathcal{N}] = z^{-1}$ (dimensionless-per-unit-redshift);
`comoving_volume_element` returns $dV_c/dz/d\Omega$ in ${\rm Mpc^3\,sr^{-1}}$
(`physical_relations.py:387-446`, Hogg 1999 Eq. 28), so
$[w_{\rm pop}] = {\rm Mpc^3\,sr^{-1}}$ per unit $z$·… precisely,
$[\mathcal{N} w_{\rm pop}] = {\rm Mpc^3\,sr^{-1}}\,z^{-1}$ and
$[Z_g] = {\rm Mpc^3\,sr^{-1}}$; the ratio $p_g(z)$ therefore has units $z^{-1}$ and unit
integral over the normalization window: a proper probability density in $z$. The
omitted constant $4\pi$ (per-steradian vs full-sky volume element) cancels between
numerator and $Z_g$, as does any constant amplitude of $\mathcal{R}$.

**Exact $h$-independence of the prior (important non-obvious property).** Writing
$d_L = (c/100h)(1+z)I(z)$ with $I(z) = \int_0^z dz'/E(z')$ and $H(z) = 100h\,E(z)$
(and $E(z)$ independent of $h$), one gets
$dV_c/dz = (c/100h)^3 I^2(z)/E(z)$, hence

$$
w_{\rm pop}(z; h) \;=\; h^{-3}\, g(z),
\qquad
g(z) \;=\; \left(\frac{c}{100}\right)^{3} \frac{I^2(z)}{E(z)\,(1+z)},
\tag{1.5}
$$

so the $h^{-3}$ prefactor cancels exactly between the numerator of Eq. (1.4) and $Z_g$:
$p_g(z)$ is the same function of $z$ for every trial $h$. Passing `h=h` into
`comoving_volume_element` at `:1858`/`:1876` is therefore harmless — the deconvolved
prior injects **no** spurious $h$-dependence into $L_{\rm cat}$; only its $z$-*shape*
acts. (This matches the commission note's observation that "every overall $1/h^3$
cancels; only the shape $w_{\rm pop}(z)\propto I^2/[E(1+z)]$ matters.")

**Limiting case $\sigma_z \to 0$.**
$\mathcal{N}(z_g; z, \sigma_z) \to \delta(z - z_g)$, so
$Z_g \to w_{\rm pop}(z_g)$ and
$p_g(z) \to \delta(z - z_g)\, w_{\rm pop}(z_g)/w_{\rm pop}(z_g) = \delta(z - z_g)$:
the prior becomes irrelevant and `volume_deconv` reduces continuously to the
spectroscopic (bare-Gaussian) limit. This is the correct behaviour — a prior can only
matter where the likelihood has finite width — and it matches the empirical bias
$\to 0$ as $\sigma_z \to 0$ (Sec. 2.3).

---

## 2. Eddington bias of the bare Gaussian and the $\sigma_z^2$ law

### 2.1 Leading-order posterior-mean shift

The bare-Gaussian kernel (modes `global` / `local_ratio`) uses
$\tilde p_g(z) = \mathcal{N}(z; z_g, \sigma_z)$ in place of Eq. (1.4). Expand
$\ln w_{\rm pop}$ about $z_g$ inside Eq. (1.4):

$$
p_g(z) \;\propto\; \exp\!\left[-\frac{(z-z_g)^2}{2\sigma_z^2}
+ (z - z_g)\, s(z_g) + \mathcal{O}\big((z-z_g)^2 w''\big)\right],
\qquad
s(z) \;\equiv\; \frac{d \ln w_{\rm pop}}{dz}.
$$

Completing the square, the true posterior is (to leading order) a Gaussian of the same
width recentred at

$$
\langle z \rangle \;=\; z_g + \sigma_z^2\, s(z_g) + \mathcal{O}(\sigma_z^4),
\qquad
\delta z_{\rm Edd} \;=\; \sigma_z^2\, \frac{d \ln w_{\rm pop}}{dz}.
\tag{2.1}
$$

This is the classical Eddington (1913)/Malmquist correction transcribed to redshift:
because $w_{\rm pop}$ rises steeply with $z$ (more volume, more hosts at higher $z$),
a galaxy observed at $z_g$ is more probably a higher-$z$ galaxy scattered *down* than a
lower-$z$ galaxy scattered *up*. The bare Gaussian omits this $+\sigma_z^2 s$ shift, so
it systematically **under-estimates** every host's redshift by $\delta z_{\rm Edd}$.

From Eq. (1.5),

$$
s(z) \;=\; \frac{2}{I(z)\,E(z)} \;-\; \frac{E'(z)}{E(z)} \;-\; \frac{1}{1+z},
\qquad
E'(z) = \frac{3\,\Omega_m (1+z)^2}{2E(z)} .
\tag{2.2}
$$

At low $z$, $I \approx z$ and $s \approx 2/z$: the shift is largest exactly where the
GLADE-like hosts sit.

### 2.2 Propagation to the $H_0$ bias

The GW measurement fixes $d_L^\star = (c/100h_{\rm true}) f(z_t)$ with
$f(z) \equiv (1+z) I(z)$ and $z_t$ the true host redshift. The estimator peaks at the
$h$ for which the model distance at its *assumed* host redshift $\hat z$ matches:
$d_L(\hat z, h) = d_L^\star \Rightarrow h = h_{\rm true}\, f(\hat z)/f(z_t)$. The
bare-Gaussian analysis effectively uses $\hat z \simeq z_g$ while the population truth,
conditioned on $z_g$, sits at $z_t \simeq z_g + \sigma_z^2 s$; hence

$$
\boxed{\;
\Delta h \;\simeq\; -\,h_{\rm true}\,
\left.\frac{d\ln f}{dz}\right|_{\bar z}\,
\sigma_z^2\, s(\bar z)
\;\equiv\; -\,C(\bar z)\,\sigma_z^2,
\qquad
\frac{d\ln f}{dz} = \frac{1}{1+z} + \frac{1}{I(z) E(z)},
\;}
\tag{2.3}
$$

evaluated at a representative detected-host redshift $\bar z$ (the coefficient must
properly be averaged over the detected population; $C(z)$ is convex, so the average
exceeds $C(z_{\rm med})$). The sign is negative — $H_0$ biased **low** — matching the
observed direction, and the leading order is strictly $\propto \sigma_z^2$, vanishing
as $\sigma_z \to 0$. This is the "$\approx \sigma_z^2\, d\ln(dV_c/dz)/dz$" law quoted
by the commission (NOTE_calibration_findings.md, Result 2), made dimensionally complete
by the $h\, d\ln f/dz$ propagation factor.

### 2.3 Quantitative check against the commission's coverage test

The commission measured (flat-prior MAP bias in $h$, $h_{\rm true}=0.72$,
$\Omega_m = 0.3$, single-host clean test):

| $\sigma_z$ | measured $\Delta h$ | floor-subtracted$^{(\dagger)}$ | implied $C_{\rm meas} = -\Delta h/\sigma_z^2$ |
|---|---|---|---|
| 0.005 | $-0.0016$ | $+0.0004$ | (floor-dominated) |
| 0.015 | $-0.0064$ | $-0.0044$ | $19.6$ |
| 0.035 | $-0.023$  | $-0.021$  | $17.1$ |
| 0.050 | $-0.046$  | $-0.044$  | $17.6$ |

$^{(\dagger)}$ The VOLUME estimator retains a $\sigma_z$-independent residual
$\approx -0.002$ at every $\sigma_z$ (same note, Result 2); subtracting this common
floor isolates the prior-omission term.

**Scaling.** The floor-subtracted biases scale as
$0.0044 : 0.021 : 0.044 = 1 : 4.8 : 10.0$ versus
$\sigma_z^2 = 1 : 5.44 : 11.1$ — the $\sigma_z^2$ law holds to $\sim 10\%$, and
$C_{\rm meas}$ is constant to $\pm 8\%$ across a factor 11 in $\sigma_z^2$. The
$\sigma_z = 0.005$ point sits inside the $\pm 0.002$ floor, as the law predicts
($C\,\sigma_z^2 \approx 4\times 10^{-4}$ there).

**Amplitude.** With Eq. (2.2)–(2.3) at $\Omega_m = 0.3$ (this note's independent
numerical evaluation):

| $\bar z$ | $s(\bar z)$ | $d\ln f/dz$ | $C(\bar z) = h\, s\, d\ln f/dz$ |
|---|---|---|---|
| 0.20 | 8.14 | 5.58 | 32.7 |
| 0.25 | 6.15 | 4.55 | 20.1 |
| 0.26 | 5.84 | 4.39 | 18.4 |
| 0.30 | 4.82 | 3.85 | 13.4 |
| 0.357$^{(\ddagger)}$ | 3.77 | 3.28 | 8.9 |

$^{(\ddagger)}$ $z(d_L = D_{50} = 1.85\,$Gpc$, h = 0.72) = 0.357$; the note quotes a
median detected $z \approx 0.3$.

The measured $C_{\rm meas} \approx 17\text{–}20$ corresponds to an effective
$\bar z_{\rm eff} \approx 0.25\text{–}0.27$ — squarely inside the synthetic's detected
distribution and, as required by the convexity of $C(z)$ (it grows like $z^{-2}$ toward
low $z$), somewhat *below* the median $0.30\text{–}0.36$: the population-averaged
coefficient $\langle C \rangle$ exceeds $C(z_{\rm med}) = 9\text{–}13$. Taking the
median value literally, the leading-order formula reproduces the measured amplitude
within $\sim 30\text{–}50\%$; allowing the population average over a realistic detected
$z$-spread (roughly $0.15\text{–}0.45$), it reproduces it within its own accuracy.
**Conclusion: the empirical numbers are quantitatively consistent with Eq. (2.3).**

**The $z \sim 0.05$ host requested by the task.** For a representative *low-redshift
catalogue* host at $z_g = 0.05$ (GLADE-like; the pipeline's photo-z model
$\sigma_z = 0.013(1+z)^3 \approx 0.015$ there, `datamodels/galaxy.py`), Eq. (2.2) gives
$s(0.05) = 38.1$ and $d\ln f/dz = 20.7$, i.e. $C(0.05) = 569$. Two things follow.
(i) The commission's measured amplitude ($C \approx 18$) is manifestly **not** produced
by $z \sim 0.05$ hosts — it is produced by the synthetic's $\bar z \approx 0.26$
population, confirming the identification above. (ii) At $z_g = 0.05$ the
*leading-order* law breaks down for large $\sigma_z$: the expansion parameter is
$\sigma_z\, s = 0.19,\ 0.57,\ 1.33,\ 1.91$ at
$\sigma_z = 0.005, 0.015, 0.035, 0.050$, so perturbation theory is valid only for
$\sigma_z \lesssim 0.015$. Exact numerical evaluation of the posterior-mean shift
of Eq. (1.4) at $z_g = 0.05$ (this note) versus the leading order
$\sigma_z^2 s$:

| $\sigma_z$ | exact $\delta z$ | leading order $\sigma_z^2 s$ |
|---|---|---|
| 0.005 | $+0.00094$ | $+0.00095$ |
| 0.015 | $+0.0079$ | $+0.0086$ |
| 0.035 | $+0.0325$ | $+0.0467$ |
| 0.050 | $+0.0535$ | $+0.0953$ |

The exact shift saturates below the quadratic law but remains *comparable to $z_g$
itself* — a $\gtrsim 60\%$ fractional redshift error per host. This is precisely why
the implementation performs the **exact deconvolution** (Eq. 1.4, evaluated by
quadrature) rather than applying the leading-order correction $z_g \to z_g +
\sigma_z^2 s$: for the real catalogue's low-$z$ hosts the bias is non-perturbative, and
only the full product $\mathcal{N}\cdot w_{\rm pop}/Z_g$ is correct there.

---

## 3. Propriety of $Z_g$ and the "$dV_c$ counted once" measure symmetry

### 3.1 What the code computes

With `normalization_mode == "volume_deconv"`, `galaxy_redshift_prior_pdf`
(`bayesian_statistics.py:1873-1878`) returns $p_g(z)$ of Eq. (1.4) and is used in:

- **In-catalogue numerator** $N_g$ (`:1896-1901`, no-mass branch; `:2044`, mass
  branch): $N_g = \int p(x_{\rm GW} \mid d_L(z,h), \Omega_g)\; p_g(z)\, dz$ over the GW
  window $[z(d_L - 4\sigma_{d_L}),\, z(d_L + 4\sigma_{d_L})]$. Per Gray et al. (2020)
  Eq. (A.10), the numerator carries GW likelihood × host-z prior only ($p_{\rm det}$
  excluded — the Mandel–Farr–Gair numerator mistake is avoided; comment at
  `:1890-1895`).
- **Per-galaxy selection denominator** $D_g$ (`:1903-1914`):
  $D_g = \int p_{\rm det}(d_L(z,h), \Omega_g)\; p_g(z)\, dz$ over
  $[z_g - 4\sigma_z,\, z_g + 4\sigma_z]$ (Gray Eq. A.19 shared-$p_{\rm det}$ symmetry).
- **Mass branch MC denominator** (`:2080-2089`): importance sampling with proposal
  $q(z,M) = \mathcal{N}(z)\mathcal{N}(M)$ and weights `integrand / sampling_pdf`, so the
  $w_{\rm pop}/Z_g$ factor is carried correctly by the weights — consistent with the
  quadrature branch.

$Z_g$ (`:1862-1869`) is computed by 50-point Gauss–Legendre `fixed_quad` over the
*same* window $[z_g - 4\sigma_z, z_g + 4\sigma_z]$ used for $D_g$ — identical interval,
hence identical quadrature nodes: $p_g$ integrates to 1 on that window under the same
discretization that consumes it. The prior is therefore **proper** (unit mass,
dimension $z^{-1}$, non-negative since $dV_c/dz = d_{\rm com}^2 c/H \ge 0$ and
$\mathcal{N} > 0$).

### 3.2 The "$dV_c$ counted once" rule

Collecting every $z$-integral of the per-event likelihood
$p_i(h) = (\beta_G L_{\rm cat} + B_{\rm num})/D(h)$ in `volume_deconv` mode
($L_{\rm cat} = \sum_g w_g N_g / \sum_g w_g D_g$, the Gray A.9/A.10 local
ratio-of-sums, `:1602-1616`):

| term | integrand | measure factor | file:line |
|---|---|---|---|
| $N_g$ | $p_{\rm GW} \cdot \mathcal{N}\, w_{\rm pop}/Z_g$ | $w_{\rm pop}$ **once** | `:1896-1901` |
| $D_g$ | $p_{\rm det} \cdot \mathcal{N}\, w_{\rm pop}/Z_g$ | $w_{\rm pop}$ **once** | `:1903-1914` |
| $B_{\rm num}$ | $(1-f)\, p_{\rm GW}\cdot dV_c/dz/(1+z)$ | $w_{\rm pop}$ **once** | `:1698` |
| $D(h)$ | $\overline{p_{\rm det}}\cdot dV_c/dz/(1+z)$ | $w_{\rm pop}$ **once** | `:336` |
| $\beta_{\bar G}$ | $(1-f)\,\overline{p_{\rm det}}\cdot dV_c/dz/(1+z)$ | $w_{\rm pop}$ **once** | `:479` |

Every $z$-integral in the estimator now carries the population measure
$w_{\rm pop}(z)\,dz$ exactly once — never zero times (the pre-fix bare-Gaussian
numerator, the commission's bug #1) and never twice (no term multiplies
$p_g \cdot w_{\rm pop}$). Numerator $N_g$ and denominator $D_g$ use the *same function
object* `galaxy_redshift_prior_pdf`, so measure symmetry within the ratio is exact by
construction. In the single-galaxy ratio $N_g/D_g$, $Z_g$ cancels identically; in the
ratio-of-sums it does not cancel, and that is *correct*: $Z_g$ is precisely the factor
that makes each galaxy contribute one unit of prior mass, so galaxies are weighted only
by the external candidate weights $w_g$ and by how well they fit the data — not by how
much comoving volume happens to lie under their photo-z error bar. (The commission's
`B_naive` variant — $w_{\rm pop}$ multiplied in but *not* renormalized — performed
almost identically on bias in their Result 3, confirming $Z_g$ is a propriety/weighting
refinement, not the bias mechanism itself.)

The catalogue terms use the *renormalized per-galaxy* density while
$B_{\rm num}, D(h), \beta_{\bar G}$ use the *unnormalized* population measure; this is
not an asymmetry: the latter are population-level integrals whose overall constant
(total rate × $4\pi$ × mass integral, $z$-independent under $p_0 = 1$, per the comment
at `:330-335`) cancels in the ratios $B_{\rm num}/D$ and $\beta_G/D$, while the former
are conditional densities per (given) galaxy, which must individually normalize. Both
conventions realize the same measure $w_{\rm pop}\,dz$.

### 3.3 Remaining asymmetries (flags)

None of the following breaks the derivation; they are listed in decreasing order of
concern.

1. **Unclamped negative-$z$ window for low-$z$ hosts**
   (`bayesian_statistics.py:1834-1839`). The $Z_g$/$D_g$ window
   $[z_g - 4\sigma_z,\, z_g + 4\sigma_z]$ is *not* clamped to $z \ge 0$, unlike
   $B_{\rm num}$ (`:1638`, `z_lower = max(z_lower, 1e-6)`) and $D(h)$
   (`z_min = 1e-6`). For GLADE hosts with $z_g < 4\sigma_z$ (e.g. $z_g = 0.03$,
   $\sigma_z \approx 0.014$) the window dips below zero, where — verified numerically
   for this note — `comoving_volume_element(z<0)` returns *positive* values
   ($d_{\rm com} < 0$ is squared at `physical_relations.py:442`), so $Z_g$ silently
   accrues unphysical prior mass at $z < 0$. Magnitude: bounded by the Gaussian tail
   $\Phi(-z_g/\sigma_z)$ times an $\mathcal{O}(1)$ volume ratio — $\lesssim$ a percent
   of $Z_g$ for $z_g/\sigma_z \gtrsim 2$, negligible for the commission configurations,
   but a one-line `max(..., 0.0)` clamp would restore exact symmetry with
   $B_{\rm num}$/$D(h)$.
2. **Prior evaluated outside its normalization window in $N_g$.** $Z_g$ normalizes over
   $z_g \pm 4\sigma_z$, but the numerator's GW window can extend beyond it, where
   `galaxy_redshift_prior_pdf` is still evaluated (not zeroed). The prior is thus very
   slightly improper over the union domain; excess mass is bounded by the Gaussian
   $>4\sigma$ tail ($6.3\times 10^{-5}$) times a moderate $w_{\rm pop}$ growth factor —
   negligible.
3. **Mass channel not deconvolved** (mass branch, `:1991-2089`). The host-mass prior
   remains the bare $\mathcal{N}(M; M_g, \sigma_M)$; the population mass weight
   $dn/d\log_{10}M \propto M^{-0.3}$ × $R_{\rm eff}(M)$ (`emri_rate.py:68-93, 231-257`)
   is omitted *symmetrically* from $N_g$ and $D_g$. The analogous Eddington-in-$M$ term
   $\propto \sigma_M^2\, d\ln[\,dn/dM\,]/dM$ is unaddressed; the log-mass slope is
   shallow, so this is second-order relative to the $z$-channel fix, but it is the same
   class of omission and worth a follow-up estimate.
4. **Silent $Z_g \le 0$ fallback** (`:1870-1871`): `_z_prior_norm` reset to 1.0 reverts
   that galaxy to the *unnormalized* $\mathcal{N}\, w_{\rm pop}$ kernel without a log
   message. Reachable only via NaN/pathological quadrature (the integrand is
   non-negative), but a warning would be safer.
5. **Numerical, not physical:** $Z_g$ and $D_g$ share quadrature nodes; $N_g$ uses a
   different interval (GW window) with its own 50 Gauss–Legendre nodes — a standard
   discretization choice, no measure implication. Likewise the comment at `:1851`
   describes the bug as a "missing $dd_L/dz$-Jacobian Jensen bias"; the precise
   statement (this note) is a missing *population prior*, of which the volume-Jacobian
   growth is the dominant part — the implemented fix is the correct one either way.

---

## 4. VERDICT

**CONFIRMED.** The `volume_deconv` kernel
(`master_thesis_code/bayesian_inference/bayesian_statistics.py:1853-1878`) implements
exactly the Bayes-correct per-galaxy host-redshift posterior
$p_g(z) = \mathcal{N}(z_g; z, \sigma_z)\, w_{\rm pop}(z)/Z_g$ with
$w_{\rm pop} = (dV_c/dz)/(1+z)$, the unique weight consistent with the project's own
rate model (`emri_rate.py:296-331`, Babak et al. 2017 arXiv:1703.09722) and with every
selection integral in the pipeline ($D(h)$ at `:336`, $\beta_{\bar G}$ at `:479`,
$B_{\rm num}$ at `:1698`) — the "$dV_c$ counted once" rule holds term-by-term, the
prior is proper, exactly $h$-independent, dimensionally a density in $z$, and reduces
to the bare-Gaussian kernel as $\sigma_z \to 0$. The bare-Gaussian bias law
$\Delta h \simeq -h\,(d\ln f/dz)\, s(\bar z)\,\sigma_z^2$ derived here reproduces the
commission's empirical coverage-test biases ($-0.0016/-0.0064/-0.023/-0.046$ at
$\sigma_z = 0.005/0.015/0.035/0.050$): the $\sigma_z^2$ scaling matches to $\sim$10%
after subtracting the $-0.002$ estimator floor, and the amplitude coefficient
$C_{\rm meas} \approx 17\text{–}20$ matches $C(\bar z)$ at
$\bar z_{\rm eff} \approx 0.26$, inside the synthetic's detected population
(median $z \approx 0.30\text{–}0.36$) as convexity requires; for a real $z \sim 0.05$
host the effect is non-perturbative ($\sigma_z\, s > 1$), justifying the exact
deconvolution over a leading-order correction. Minor deviations flagged, none
invalidating the implementation: unclamped $z<0$ window in $Z_g/D_g$
(`bayesian_statistics.py:1834-1839`, contrast `:1638`), prior evaluated slightly
outside its normalization window in $N_g$, mass channel left undeconvolved
(symmetric omission), and the silent $Z_g \le 0$ fallback (`:1870-1871`).
