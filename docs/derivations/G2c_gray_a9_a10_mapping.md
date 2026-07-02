# G2c — Symbol-by-symbol mapping between the implemented dark-siren likelihood and Gray et al. (2020), arXiv:1908.06050

**Commission task G2c** (2026-07-02, branch `physics/derail-completion-4pi`).
Code under audit: `master_thesis_code/bayesian_inference/bayesian_statistics.py`
(`p_Di` at lines 1420–1753, `single_host_likelihood` at 1793–2104, and the
precompute helpers `precompute_completion_denominator` (223–362),
`precompute_missing_completion_denominator` (365–490),
`precompute_global_catalog_selection` (493–656)).
Reference: R. Gray et al., *Cosmological Inference using Gravitational Wave
Standard Sirens: A Mock Data Analysis*, Phys. Rev. D **101**, 122001 (2020),
arXiv:1908.06050. All equations below were transcribed from the arXiv v4 LaTeX
source (`method_main.tex`, `method_appendix.tex`, `mdc.tex`), not from memory.

---

## 0. Equation-numbering convention (decoded)

The arXiv v4 source contains **15 numbered main-text equations** and **28
appendix equations** (`\appendix*`, numbered (A1)–(A28)). The repository's code
comments use a **continuous numbering**: appendix equation (A$n$) is cited as
"Eq. $(15+n)$" while the appendix-native "A.$n$" form is used interchangeably.
This note verifies the correspondence explicitly:

| Repo citation | arXiv v4 equation | LaTeX label | Content |
|---|---|---|---|
| "Eq. 9" | main-text (9) | `Eq:sum G` | $G/\bar G$ marginalisation of the per-event likelihood |
| "Eqs. 24–25" / "A.9 / A.10" | (A9), (A10) | — / `Eq:p(x|G,D,H0)` | in-catalogue ratio of sums (without / with galaxy-$z$ uncertainty) |
| "Eq. 26" | (A11) | `Eq:G_DH0_start` | $p(G\mid D_{\rm GW},s,H_0)$, first expansion |
| "Eq. 29" | (A14) | `Eq:G_DH0_end` | $p(G\mid D_{\rm GW},s,H_0)$ final ratio of selection integrals |
| "Eq. 30" | (A15) | — | $p(\bar G\mid D_{\rm GW},s,H_0) = 1 - p(G\mid D_{\rm GW},s,H_0)$ |
| "Eq. 31" | (A16) | `Eq:px_H0GbarD` | out-of-catalogue likelihood, first expansion |
| "Eq. 32" | (A17) | — | out-of-catalogue **numerator** integral |
| "Eq. 33" | (A18) | — | out-of-catalogue **selection denominator** integral |
| "A.19" | (A19) | `Eq:p(x|barG,D,H0)` | out-of-catalogue likelihood, final ratio (A17)/(A18) |

The decoding is self-consistent: every semantic claim in the code comments
matches the content of the equation under this convention (with the two
citation-level exceptions flagged in §6).

For completeness, the four appendix equations most heavily used are transcribed
verbatim (Gray et al. 2020, arXiv:1908.06050):

**(A9)** — in-catalogue likelihood, delta-function galaxy redshifts:

$$
p(x_{\rm GW}\mid G, D_{\rm GW}, s, H_0)
 = \frac{\sum_{i=1}^{N} p(x_{\rm GW}\mid z_i,\Omega_i,s,H_0)\,p(s\mid z_i)\,p(s\mid M(z_i,m_i,H_0))}
        {\sum_{i=1}^{N} p(D_{\rm GW}\mid z_i,\Omega_i,s,H_0)\,p(s\mid z_i)\,p(s\mid M(z_i,m_i,H_0))}.
$$

**(A10)** — the same with per-galaxy redshift uncertainty $p(z_i)$:

$$
p(x_{\rm GW}\mid G, D_{\rm GW}, s, H_0)
 = \frac{\sum_{i=1}^{N_{\rm gal}} \int p(x_{\rm GW}\mid z_i,\Omega_i,s,H_0)\,p(s\mid z_i)\,p(s\mid M(z_i,m_i,H_0))\,p(z_i)\,dz_i}
        {\sum_{i=1}^{N_{\rm gal}} \int p(D_{\rm GW}\mid z_i,\Omega_i,s,H_0)\,p(s\mid z_i)\,p(s\mid M(z_i,m_i,H_0))\,p(z_i)\,dz_i}.
$$

Note that **both** sums in (A9)/(A10) run over the *entire* catalogue
($i = 1 \dots N_{\rm gal}$); no restriction to a localisation region appears in
the published equation. This matters for §5.

**(A14)** ("Eq. 29") — probability the host is in the catalogue:

$$
p(G\mid D_{\rm GW},s,H_0)
 = \frac{\displaystyle\int_0^{z(M,m_{\rm th},H_0)}\!\!dz \int d\Omega \int dM\;
        p(D_{\rm GW}\mid z,\Omega,s,H_0)\,p(s\mid z)\,p(z)\,p(\Omega)\,p(s\mid M,H_0)\,p(M\mid H_0)}
        {\displaystyle\iiint p(D_{\rm GW}\mid z,\Omega,s,H_0)\,p(s\mid z)\,p(z)\,p(\Omega)\,p(s\mid M,H_0)\,p(M\mid H_0)\;dz\,d\Omega\,dM}.
$$

**(A17)/(A18)/(A19)** ("Eqs. 32/33/A.19") — the out-of-catalogue channel:
(A17) is the numerator $\propto \int_{z(M,m_{\rm th},H_0)}^{\infty} dz \int d\Omega \int dM\;
p(x_{\rm GW}\mid z,\Omega,s,H_0)\,p(s\mid z)\,p(z)\,p(\Omega)\,p(s\mid M,H_0)\,p(M\mid H_0)$,
(A18) the identical integral with $p(x_{\rm GW}\mid\cdot)$ replaced by
$p(D_{\rm GW}\mid\cdot)$, and (A19) their ratio
$p(x_{\rm GW}\mid\bar G, D_{\rm GW}, s, H_0) = \text{(A17)}/\text{(A18)}$.

Main-text **(9)** assembles the two channels:

$$
p(x_{\rm GW}\mid D_{\rm GW},H_0)
 = p(x_{\rm GW}\mid G,D_{\rm GW},H_0)\,p(G\mid D_{\rm GW},H_0)
 + p(x_{\rm GW}\mid \bar G,D_{\rm GW},H_0)\,p(\bar G\mid D_{\rm GW},H_0).
$$

---

## 1. The implemented per-event likelihood

For each detected event $i$ and each Hubble value $h$, `p_Di`
(`bayesian_statistics.py:1420–1753`) returns (non-`catalog_only` path, lines
1564–1730):

$$
p_i(h) \;=\; \frac{\beta_G(h)\, L_{\rm cat}(h) \;+\; B_{\rm num}(h)}{D(h)},
\qquad\text{(line 1721)}
$$

with the identities $w_G \equiv \beta_G/D$, $1 - w_G = \beta_{\bar G}/D$
(since $\beta_G \equiv D - \beta_{\bar G}$, line 910) and the diagnostic-only
$L_{\rm comp} \equiv B_{\rm num}/\beta_{\bar G}$ (line 1730). Algebraically,

$$
p_i = w_G\,L_{\rm cat} + (1-w_G)\,\underbrace{\frac{B_{\rm num}}{\beta_{\bar G}}}_{L_{\rm comp}}
$$

which is **exactly** the structure of Gray main-text Eq. (9) with the
identifications $L_{\rm cat} \leftrightarrow p(x_{\rm GW}\mid G,D_{\rm GW},H_0)$,
$w_G \leftrightarrow p(G\mid D_{\rm GW},H_0)$ [Eq. (A14)],
$1-w_G \leftrightarrow p(\bar G\mid D_{\rm GW},H_0)$ [Eq. (A15)], and
$L_{\rm comp} \leftrightarrow p(x_{\rm GW}\mid\bar G,D_{\rm GW},H_0)$
[Eq. (A19) = (A17)/(A18)]. The single-ratio form
$(\beta_G L_{\rm cat} + B_{\rm num})/D$ never divides by $\beta_{\bar G}$,
which is numerically safer as $f \to 1$ ($\beta_{\bar G} \to 0$) but is an
*identical* rearrangement, not an approximation.

Dimensional check: $B_{\rm num}$ carries
$[\mathrm{Mpc^3\,sr^{-1}}]\times[\mathrm{sr^{-1}}]$ (volume integral times the
$1/4\pi$-marginalised GW pdf per unit $d_L$-fraction), and
$\beta_G L_{\rm cat}$ carries $[\mathrm{Mpc^3\,sr^{-1}}]\times[\mathrm{sr^{-1}}]$
(the 3-D GW Gaussian in $(\phi,\theta,d_L/d_L^{\rm det})$ is per steradian per
unit $d_L$-fraction, and $D_g$ is dimensionless). The two channels are therefore
commensurate, and the overall factor left after dividing by
$D\,[\mathrm{Mpc^3\,sr^{-1}}]$ is common to all $h$, hence irrelevant for the
posterior shape.

---

## 2. Symbol-by-symbol mapping table

Notation: $g$ indexes candidate host galaxies; $N_g, D_g$ are the per-host
numerator/denominator returned by `single_host_likelihood`
(`r[0], r[1]` without BH mass; `r[2], r[3]` with BH mass);
$\mathcal N(x;\mu,\sigma)$ is a normal pdf; $f$ is catalogue completeness;
$V_c$ comoving volume; $p_{\rm det}$ the injection-calibrated detection
probability.

| Code quantity (file:line) | Definition as implemented | Gray et al. (2020) symbol and equation | Correspondence / caveat |
|---|---|---|---|
| `N_g` = `single_host_likelihood_numerator_without_bh_mass` (1883–1924) | $N_g = \int_{z^{\rm det}_-}^{z^{\rm det}_+} \mathcal N_3\!\big((\phi_g,\theta_g,\tfrac{d_L(z,h)}{d_L^{\rm det}});\,\boldsymbol\mu_3,\Sigma_3\big)\,p_g(z)\,dz$ over the event's $\pm4\sigma_{d_L}$ window | The $i$-th **numerator integrand** of (A10): $\int p(x_{\rm GW}\mid z_i,\Omega_i,s,H_0)\,p(z_i)\,dz_i$ | GW likelihood $p(x_{\rm GW}\mid z,\Omega)$ realised as the Fisher-matrix 3-D Gaussian with mean $(\phi_{\rm det},\theta_{\rm det},1)$ (line 1052); galaxy sky position enters through the Gaussian's $(\phi,\theta)$ arguments (smoothed sky delta of Gray §II C footnote 3). **No $p_{\rm det}$ in the numerator** (comment 1890–1895), matching (A10) and avoiding the Mandel–Farr–Gair (2019, arXiv:1809.02063) double-counting. |
| `D_g` = `single_host_likelihood_denominator_without_bh_mass` (1903–1933) | $D_g = \int_{z_g - 4\sigma_z}^{z_g + 4\sigma_z} p_{\rm det}\big(d_L(z,h),\phi_g,\theta_g\big)\,p_g(z)\,dz$ | The $i$-th **denominator integrand** of (A10): $\int p(D_{\rm GW}\mid z_i,\Omega_i,s,H_0)\,p(z_i)\,dz_i$ | $p(D_{\rm GW}\mid z,\Omega,s,H_0)$ realised by `SimulationDetectionProbability` — the Monte-Carlo estimator of Gray (A23) built from the EMRI injection campaign instead of BNS SNR draws. |
| `p_g(z)` = `galaxy_redshift_prior_pdf` (1873–1878) | mode-dependent: bare $\mathcal N(z; z_g, \sigma_{z,g})$, or volume-deconvolved $\mathcal N(z;z_g,\sigma_{z,g})\,\frac{dV_c}{dz}\frac{1}{1+z}/Z_g$ | $p(z_i)$ in (A10) ("modeled with a Gaussian or a more complicated distribution", footnote 3) | The **bare Gaussian** is the literal (A10) choice. The **volume-deconvolved** form maps to *no equation in Gray 2020* — see §5.3 and §4. |
| `w_g` = `_rate_weight(host)` (149–172), applied at 1524–1530 | $w_g = R_{\rm eff}(M_g)/(1+z_g)$, per-MBH EMRI rate (Babak et al. 2017, arXiv:1703.09722) over source-to-detector time dilation | $p(s\mid z_i)\,p(s\mid M(z_i,m_i,H_0))$ in (A9)/(A10); generic forms (A3) (luminosity weighting) and (A4) (rate evolution) | Gray's placeholder host weights instantiated with an astrophysical EMRI-rate prior: $p(s\mid M)\to R_{\rm eff}(M_{\rm BH})$ (MBH mass, not galaxy luminosity) and $p(s\mid z)\to 1/(1+z)$ (detector-frame rate; Gray's constant-rate MDC limit plus time dilation). Identical to the generative host draw `draw_rate_weighted_hosts`, closing the simulation–inference loop. Deviation D1, §5. |
| `weighted_ratio_of_sums` (74–122) | $L_{\rm cat} = \dfrac{\sum_g w_g N_g}{\sum_g w_g D_g}$ over the **local candidate ball** | (A9)/(A10) ratio-of-sums | Exact discrete form of (A10) *if* the sums run over the full catalogue; here both run over the BallTree candidate set — see the per-mode discussion, §4. |
| `weighted_sum` (125–146) + `_global_cat_denom_*` | $L_{\rm cat} = \dfrac{\sum_{g\in\rm local} w_g N_g}{\Sigma_{\rm global}(h)}$ | (A10) with the denominator sum over all $N_{\rm gal}$ | "global" mode; see §4.1. |
| `Sigma_global(h)` = `precompute_global_catalog_selection` (493–656) | $\Sigma_{\rm global}(h) = \sum_{g:\,z_g<z_{\max}(h)} w_g\,p_{\rm det}\big(d_L(z_g,h),\Omega_g\big)$ | Denominator of (A10), $\sum_i p(D_{\rm GW}\mid z_i,\Omega_i)\,p(s\mid z_i)p(s\mid M_i)$, in the narrow-$p(z_i)$ limit $\int p_{\rm det}\,p(z_i)dz_i \to p_{\rm det}(z_g)$ (docstring 524) | Also the discrete Monte-Carlo realisation of $\beta_G(h)$ up to the constant $\bar n_{\rm gal}$ (docstring 508–510); the catalogue rows *are* the sample of the in-catalogue density. Sky-resolved for the 3-D channel (galaxy ecliptic latitudes, 618–635); isotropic for the statistics-starved 4-D channel (603–617, user-approved flag). |
| `D(h)` = `precompute_completion_denominator` (223–362) | $D(h) = \int_{z_{\min}}^{z_{\max}(h)} \big\langle p_{\rm det}(d_L(z,h),\Omega)\big\rangle_{\Omega}\,\frac{dV_c}{dz\,d\Omega}\,\frac{dz}{1+z}$, sky average $= \frac{1}{N_{\rm pix}}\sum_k p_{\rm det}(\cdot,\Omega_k)$; $[\mathrm{Mpc^3\,sr^{-1}}]$ | **Denominator of (A14)** ("Eq. 29"): $\iiint p(D_{\rm GW}\mid z,\Omega,s,H_0)\,p(s\mid z)p(z)p(\Omega)p(s\mid M)p(M)\,dz\,d\Omega\,dM$ — i.e. the full-volume GW selection normalisation $p(D_{\rm GW}\mid s,H_0)$ expanded | $p(z)\,dz \to \frac{dV_c}{dz}dz$ (uniform-in-comoving-volume prior, Gray Appendix 1); $p(s\mid z) \to 1/(1+z)$; $p(\Omega) \to 1/N_{\rm pix}$ over equal-area HEALPix pixels; the mass integral $\int p(s\mid M)p(M\mid H_0)\,dM$ is $z$-independent under the `p0=1` surrogate and cancels (Option A, comment 261–268). The docstring's "Eqs. 33 / A.19" citation is imprecise — see §6 (C2). |
| `beta_Gbar(h)` = `precompute_missing_completion_denominator` (365–490) | $\beta_{\bar G}(h) = \int \big\langle (1-f_k(z))\,p_{\rm det}\big\rangle_{\Omega}\,\frac{dV_c}{dz\,d\Omega}\,\frac{dz}{1+z}$ | **(A18)** ("Eq. 33"): out-of-catalogue selection integral $\int_{z(M,m_{\rm th})}^{\infty} p(D_{\rm GW}\mid z,\Omega)\cdots$ | Gray's hard threshold $\int_{z(M,m_{\rm th},H_0)}^\infty dz \int dM\,\Theta[m-m_{\rm th}]\,p(M)\cdots$ is replaced by the smooth incompleteness weight $(1-f(z,\Omega))$: performing Gray's $M,m$ integrals first yields exactly $\int_0^\infty (1-f(z,\Omega))\,p(D_{\rm GW}\mid z,\Omega)\,p(s\mid z)p(z)\,dz$ where $f$ is the fraction of the (rate-weighted) galaxy population above threshold — the completeness. Same smooth-$f$ identification used per pixel (Gray–Messenger–Veitch 2022, arXiv:2111.04629, Eq. 5). |
| `beta_G(h)` (line 910) | $\beta_G = D - \beta_{\bar G} = \int \langle f\,p_{\rm det}\rangle\,\frac{dV_c}{1+z}\,dz$ | **Numerator of (A14)**: $\int_0^{z(M,m_{\rm th})}dz\cdots = \int f\,p(D_{\rm GW}\mid z,\Omega)\,p(s\mid z)p(z)p(\Omega)\,dz\,d\Omega$ | complementarity (A15) built in by construction. |
| `w_G` = $\beta_G/D$ (1720) | selection-weighted catalogue membership probability, event-independent | **(A14)** ("Eq. 29") $p(G\mid D_{\rm GW},s,H_0)$; (A15) gives $1-w_G$ | Replaces the earlier scalar approximation $f(z_{\rm det})$; exact within the Option-A constant-density assumption. |
| `B_num` (1618–1702) | $B_{\rm num} = \int_{z_-}^{z_+} (1-f_{k(\Omega_e)}(z))\,\frac{1}{4\pi}\,\mathcal N\!\big(\tfrac{d_L(z,h)}{d_L^{\rm det}};\,1,\sigma_{\rm marg}\big)\,\frac{dV_c}{dz\,d\Omega}\,\frac{dz}{1+z}$, with $\sigma_{\rm marg}^2 = (\Sigma_3)_{22}$ and $z_\pm = z(d_L^{\rm det}\pm4\sigma_{d_L})$ | **(A17)** ("Eq. 32"): out-of-catalogue numerator $\int_{z(M,m_{\rm th})}^{\infty} p(x_{\rm GW}\mid z,\Omega,s,H_0)\,p(s\mid z)p(z)p(\Omega)p(s\mid M)p(M)\,dz\,d\Omega\,dM$ | $\int d\Omega\,p(x_{\rm GW}\mid z,\Omega)\,p(\Omega)$ with $p(\Omega)=1/4\pi$ evaluated **exactly** for the Gaussian: the isotropic sky marginal of the 3-D GW Gaussian is the 1-D Gaussian in $d_L$-fraction times $1/4\pi$ (4π de-rail fix, commit cb16142; comments 1645–1679). Hard threshold $\to$ smooth $(1-f)$ as for $\beta_{\bar G}$, but $f$ taken at the **event pixel** $k(\Omega_e)$ (GMV 2022 Eq. 5; Change 5.3, 1656–1662). GW selection $p_{\rm det}$ correctly **absent** (denominator-only; 1622–1625). Finite $\pm4\sigma$ window is numerically inert since $p(x_{\rm GW}\mid z)\approx0$ outside. Mixed sky treatment flagged in §5 (D5). |
| `L_comp` (1730) | $B_{\rm num}/\beta_{\bar G}$, diagnostic only | **(A19)** = (A17)/(A18): $p(x_{\rm GW}\mid\bar G,D_{\rm GW},s,H_0)$ | Never used in $p_i$; the single ratio is the algebraically identical safe form. |
| `L_cat` per mode (1584–1616) | see §4 | (A9)/(A10) or the catalogue-patch case (A20), mode-dependent | §4 states exactly which equation each mode implements. |
| with-BH-mass channel `r[2], r[3]` (1990–2089) | 4-D Gaussian $(\phi,\theta,d_L\text{-frac},M_z\text{-frac})$; analytic conditional-Gaussian $M_z$ marginalisation (Bishop 2006 Eqs. 2.81–2.82); denominator $\iint p_{\rm det}(d_L,M_z,\Omega_g)\,p_g(z)\,\mathcal N(M;M_g,\sigma_{M,g})\,dz\,dM$ by MC importance sampling | **No Gray 2020 equation** — Gray's masses enter only through $p(D_{\rm GW}\mid H_0)$ population priors (A25–A28), never as a per-galaxy mass likelihood channel | Internal extension; derivation in `derivations/dark_siren_likelihood.md` Eqs. (14.21)–(14.33). Structurally it is (A10) with the host observable vector enlarged by $M_z = M_g(1+z)$; the $(1+z)$ lift is the coordinate transform of (A25), $M_z = (1+z)M$. |
| `catalog_only` (1541–1563) | $p_i = L_{\rm cat}$ (local ratio), $f\equiv1$, no completion | (A9)/(A10) in the complete-catalogue limit, i.e. Gray's MDA1 ($p(G\mid D)=1$, Eq. 9 collapses to the first term) | Validation mode; byte-identical to the pre-restructure behaviour. |

---

## 3. Discrete sum ↔ continuous integral correspondence

Gray's in-catalogue channel is a **discrete sum over catalogue rows** (A9/A10);
the selection normalisations (A14), (A17)–(A18) are **continuous population
integrals**. The implementation must make the two commensurate because
$\beta_G\,L_{\rm cat}$ mixes them. The bridge (Option A, documented at
223–268 and 500–525) is:

1. The catalogue is treated as a Poisson/Monte-Carlo realisation of the
   in-catalogue number density
   $n_G(z,\Omega) = \bar n_{\rm gal}\, f(z,\Omega)$, with $\bar n_{\rm gal}$
   **constant in comoving coordinates** (Gray Appendix 1:
   $p(z) \propto dV_c/dz$; "uniform in comoving volume-time").
2. Hence for any smooth per-galaxy statistic $Q(z,\Omega,M)$,
$$
\sum_{g\in\text{cat}} w_g\, Q(z_g,\Omega_g)
 \;\approx\; \bar n_{\rm gal} \int dz\,d\Omega\; f(z,\Omega)\,\bar w(z)\,
   Q(z,\Omega)\,\frac{dV_c}{dz\,d\Omega},
$$
   where $\bar w(z) = \langle R_{\rm eff}(M)\rangle_{p(M)}/(1+z)$ and the
   mass-integrated rate $\int dM\,R_{\rm EMRI}(z,M)$ is $z$-independent under
   the `p0=1` surrogate (comment 330–335), so it factors out as a constant.
3. Applying this to the (A10) denominator gives
   $\Sigma_{\rm global}(h) \approx C\,\beta_G(h)$ with
   $C = \bar n_{\rm gal}\,\langle R_{\rm eff}\rangle$ **independent of $h$ up to
   second-order density/rate-evolution effects**. Therefore, in "global" mode,
$$
\frac{\beta_G\,L_{\rm cat}}{D}
 = \frac{\beta_G}{D}\cdot\frac{\sum_{\rm local} w_g N_g}{\Sigma_{\rm global}}
 \;\longleftrightarrow\;
 p(G\mid D_{\rm GW},H_0)\; p(x_{\rm GW}\mid G,D_{\rm GW},H_0)
$$
   with the unknowable constants $\bar n_{\rm gal}$, `C_NORM` cancelling —
   the discrete sum is the numerator/denominator MC pair of (A10) and the
   continuous pair is (A14). Conversely, in the local modes the ratio
   $L_{\rm cat}$ is self-normalised over the same galaxy set, so **all**
   per-galaxy constants cancel row-by-row and no density calibration is needed
   at all; the price is that the denominator is no longer Gray's global
   selection sum (§4.2).

The completion channel needs no bridge: $B_{\rm num}$, $\beta_{\bar G}$, $D$
are all continuous integrals of the same population prior
$\frac{dV_c}{dz\,d\Omega}\frac{1}{1+z}$, mutually consistent by construction
and consistent with the event generator (`emri_rate.p_pop_unnormalized`,
`dark_siren_injection._draw_dark_redshifts` — generator/inference
bit-consistency asserted at 388–392 and 1686–1689).

---

## 4. The three `normalization_mode`s vs. the published equations

Dispatch: `evaluate()` argument (804, validated 816–827) → threaded to `p_Di`
(1584: global vs. local branch) and to `single_host_likelihood`
(1461/1481 → 1853: kernel choice).

### 4.1 `"global"` (legacy; warns at 818–826)

$$
L_{\rm cat} = \frac{\sum_{g\in\text{local ball}} w_g N_g}{\Sigma_{\rm global}(h)}.
$$

**This is the faithful discrete transcription of (A10).** In (A10) both sums
run over the full catalogue; restricting the *numerator* sum to the candidate
ball is numerically inert because $p(x_{\rm GW}\mid z_g,\Omega_g)\approx0$
outside the localisation region, while the *denominator* is kept global exactly
as published. Combined with $\beta_G/D$ it reproduces Eq. (9)'s first term
$p(x\mid G,D,H_0)\,p(G\mid D,H_0)$ with all constants cancelled (§3).
Empirical status: the 2026-07-01 commission found this mode mis-calibrated for
photometric-redshift catalogues (~0 % P–P coverage; posterior rails to the grid
edge — `.planning/INDEPENDENT-VERIFICATION-REPORT-20260701.md` §7); the failure
is attributed to the $h$-dependence mismatch between the discrete
$\Sigma_{\rm global}(h)$ (photo-$z$-smeared catalogue rows) and the continuous
$\beta_G(h)$, i.e. a breakdown of the §3 constancy of $C$, not to an algebra
error.

### 4.2 `"local_ratio"` (de-rail fix #2)

$$
L_{\rm cat} = \frac{\sum_{g\in\text{local ball}} w_g N_g}{\sum_{g\in\text{local ball}} w_g D_g}.
$$

The code comment (1578–1581) calls this "the Gray A.9/A.10 **literal** local
self-normalized ratio-of-sums". Strictly, this is **not** the literal (A10):
(A10)'s selection sum runs over all $N_{\rm gal}$, not over the event's
candidate ball. The closest published analogue is the **catalogue-patch case**,
Gray Eq. (A20) ("the first term is equivalent to the regular galaxy catalog
case, but with limits on the integral over $\Omega$"), with the patch
identified event-by-event with the localisation ball — a 3-D (sky *and*
distance) ball rather than Gray's RA/dec patch, and without (A20)'s explicit
$\Omega_{\rm rest}$ term (which is instead absorbed, approximately, by the
completion channel built from full-sky $D$, $\beta_{\bar G}$). It is therefore
a **deliberate deviation** from (A10), empirically justified by P–P
calibration, not a literal transcription; the in-code label overstates it
(deviation C1, §6).

### 4.3 `"volume_deconv"` (default; de-rail fix #1 + #2)

Same local ratio as §4.2, but with the host-redshift kernel replaced inside
both $N_g$ and $D_g$ (1844–1878):

$$
p_g(z) = \frac{\mathcal N(z; z_g, \sigma_{z,g})\; \dfrac{dV_c}{dz}\dfrac{1}{1+z}}
              {\displaystyle\int \mathcal N(z'; z_g, \sigma_{z,g})\,\frac{dV_c}{dz'}\frac{dz'}{1+z'}}.
$$

This reinterprets the catalogue Gaussian as a redshift *likelihood* and forms
the posterior against the population prior
$\frac{dV_c}{dz}\frac{1}{1+z}$ — the same prior that $D(h)$, $\beta_{\bar G}$
and $B_{\rm num}$ carry, removing the $dd_L/dz$-Jacobian Jensen bias for broad
photo-$z$ (commission report bug #1). **No equation in Gray 2020 prescribes
this**: (A10) treats $p(z_i)$ as given ("a Gaussian or a more complicated
distribution", footnote 3) and the MDCs had zero redshift error, so the paper
is silent on the likelihood-vs-posterior convention for $p(z_i)$. The
prior-times-likelihood construction is standard in later gwcosmo work
(e.g. Gray et al. 2023, arXiv:2308.02281, LOS redshift prior construction) but
must be cited as such, not as 1908.06050. Note the per-galaxy normalisation
$Z_g$ does **not** cancel between galaxies in the ratio of sums; it acts as a
deliberate per-galaxy reweighting toward volume-consistent redshift support.

---

## 5. Deliberate deviations from Gray et al. (2020), each with its justification

**D1 — EMRI rate weighting $w_g = R_{\rm eff}(M_g)/(1+z_g)$** (149–172,
1510–1530, 599–601). Replaces Gray's luminosity weighting
$p(s\mid M)\propto L$ (A3) and rate-evolution factor $p(s\mid z)$ (A4) with the
per-MBH EMRI rate of Babak et al. (2017, arXiv:1703.09722) and the
$1/(1+z)$ detector-frame time dilation (Mandel–Farr–Gair 2019, arXiv:1809.02063,
detector-frame rate density). Identical weight in the generative host draw ⇒
self-consistent by construction; enters numerator and denominator identically,
so overall normalisation cancels (74–122). *Sub-deviation:* $w$ is evaluated at
the catalogue $z_g$ and pulled outside the $\int dz_i$, whereas (A10) has
$p(s\mid z_i)$ inside the integral; second-order for narrow $\sigma_z$ but not
for GLADE+ photo-$z$ widths — undocumented approximation.

**D2 — smooth completeness $f(z)$ / per-pixel $f_k(z,\Omega)$ instead of the
hard $m_{\rm th}$ threshold** (365–490, 1690–1697). Gray's
$\Theta[m_{\rm th}-m]$ + Schechter-$M$ integrals are pre-integrated into the
completeness fraction; per-pixel resolution follows Gray–Messenger–Veitch 2022
(arXiv:2111.04629, Eq. 5) and Gray et al. 2023 (arXiv:2308.02281, Eq. 2.3).
The $f$ used is the number-count completeness of the actual catalogue build
(same frozen $m_{\rm th}$ map as the injection campaign; C1 consistency,
878–884), i.e. semantically Gray's main-text Eq. (12) (`Eq:completeness`)
rather than a Schechter-derived fraction.

**D3 — detection-horizon survival $p_{\rm det}$ from the EMRI injection
campaign** (`SimulationDetectionProbability`; band survival
`survival_per_band`, equal-$|\sin\beta|$ ecliptic bands). Implements Gray (A23)
(MC estimator of $p(D_{\rm GW}\mid z,\Omega,H_0)$) with LISA-specific sky
dependence through ecliptic latitude only (orbit-averaged response, Cutler
1998, arXiv:gr-qc/9703068), where Gray approximated $p_{\rm det}$ sky-uniform.
One shared $p_{\rm det}$ object across $D$, $\beta_{\bar G}$,
$\Sigma_{\rm global}$ and $D_g$ (624–631, 1907–1913) so the convention cancels
in every ratio.

**D4 — single-ratio assembly $(\beta_G L_{\rm cat} + B_{\rm num})/D$**
(1715–1727) instead of the two-term convex form of Eq. (9). Algebraically
identical (§1); avoids $\beta_{\bar G}\to0$ division. No physics content.

**D5 — mixed sky treatment inside $B_{\rm num}$** (1645–1697): the GW
likelihood is sky-marginalised against the isotropic prior $1/4\pi$ (exact
Gaussian marginal — the 4π de-rail fix, commit cb16142), while the
incompleteness is delta-collapsed to the event pixel $f_{k(\Omega_e)}$
(GMV 2022 Eq. 5 rationale: $p_{\rm GW}$ is sky-sharp relative to pixel scale).
Strictly (A17) requires $\int d\Omega\,(1-f(z,\Omega))\,p(x_{\rm GW}\mid z,\Omega)\,p(\Omega)$
— a *single* sky integral weighting both factors; the implementation
approximates it as $(1-f_{k(\Omega_e)}(z))\cdot\frac{1}{4\pi}\mathcal N(\cdot)$,
which is exact only when $f$ is constant across the GW sky support. Internally
consistent to first order but maps to no single published equation.

**D6 — with-BH-mass channel** (1990–2089): no Gray analogue (see mapping
table); internal derivation `derivations/dark_siren_likelihood.md`
Eqs. (14.21)–(14.33); Bishop (2006) PRML Eqs. (2.81)–(2.82) for the
conditional Gaussian. The 4-D $\Sigma_{\rm global}$ companion stays isotropic
(statistics-starved sky×$M_z$ survival; user-approved flag, 603–608).

**D7 — local selection denominator (modes `local_ratio`/`volume_deconv`)**:
deviation from (A10)'s global sum; nearest published form is the patch case
(A20); justification is empirical P–P calibration
(`.planning/INDEPENDENT-VERIFICATION-REPORT-20260701.md` §7). See §4.2.

**D8 — volume-deconvolved photo-$z$ kernel (default mode)**: beyond
Gray 2020; see §4.3.

**D9 — finite integration windows**: numerator/completion $z$-windows are the
event's $\pm4\sigma_{d_L}$ image, host windows $z_g\pm4\sigma_z$, and all
selection integrals stop at $z_{\max}(h) = z(d_L^{\rm max,grid},h)$ rather than
$\infty$ — justified because $p_{\rm det}=0$ beyond the injection horizon
(zero-fill accessor) and the Gaussians are negligible outside $4\sigma$.
Numerical, not physical.

## 5b. Quantities that map to NO published equation

1. The **volume-deconvolved per-galaxy $z$-prior** with its normalisation
   $Z_g$ (1853–1878) — D8.
2. The **local selection denominator** $\sum_{\rm local} w_g D_g$ as the
   normalisation of $p(x\mid G,D,H_0)$ (1602–1616) — D7 (patch case A20 is an
   analogue, not an equation-level match).
3. The **with-BH-mass channel** in its entirety (4-D Gaussian, $M_z$
   conditional marginal, mass-resolved $p_{\rm det}$) — D6.
4. The **product form $(1-f_{k(\Omega_e)})\times\frac{1}{4\pi}\mathcal N$**
   inside $B_{\rm num}$ — D5 (each factor is published; their combination as a
   factorised sky treatment is not).
5. The **per-MBH rate weight** $R_{\rm eff}(M)/(1+z)$ as the concrete
   $p(s\mid z)p(s\mid M)$ — D1 (published as a rate, Babak 2017; its use as a
   Gray host weight is a project choice).

---

## 6. Citation-level inaccuracies found (documentation, not mathematics)

**C1** — `bayesian_statistics.py:1578–1582` (and `evaluate()` comment 810):
"`local_ratio` → Gray A.9/A.10 **literal**". Inaccurate: (A10)'s denominator
sum is catalogue-global; the local restriction is a deviation (§4.2). The
mathematics implemented is as documented; only the "literal" attribution is
wrong.

**C2** — `precompute_completion_denominator` docstring, lines 232–241: cites
"Gray Eqs. 33 / A.19" for $D(h)$. $D(h)$ carries **no** $(1-f)$ weight (as the
docstring itself stresses at 248–259), so it is the **denominator of (A14)**
("Eq. 29" in repo numbering, $p(D_{\rm GW}\mid s,H_0)$ expanded), not the
$\bar G$-restricted (A18)/(A19). $\beta_{\bar G}$ (365–490) is the true
Eq.-(A18) object and is cited correctly. Legacy citation from the era when
$D(h)$ normalised the completion term alone.

**C3** — `evaluate()` comment at 878 ("Gray et al. 2020 … Eq. 9 … per-HEALPix
completeness"): under arXiv v4 numbering, Eq. (9) is the $G/\bar G$
marginalisation; the completeness definition is main-text Eq. (12)
(`Eq:completeness`). Same slip in the `precompute_missing_completion_denominator`
docstring line 399 ("Gray Eq. 9").

**C4** — stale comment at 1121–1122: "Partition-norm precompute tables …
**not yet read by p_Di**" — they are read at 1565–1569.

**C5** — `single_host_likelihood` signature default
`normalization_mode: str = "global"` (1803) differs from the class default
`"volume_deconv"` (772, 804). Harmless today (p_Di always passes it
explicitly, 1461/1481), but a direct caller relying on the default would get
the railed legacy kernel.

**C6** — minor: `precompute_global_catalog_selection` uses the narrow-PDF
approximation $D_g \approx p_{\rm det}(z_g)$ (docstring 524) while the local
$D_g$ integrates over the photo-$z$ PDF; in "global" mode the numerator
($z$-integrated) and denominator (delta-approximated) therefore treat the
photo-$z$ smearing asymmetrically. This asymmetry is part of why "global"
mis-calibrates for photo-$z$ catalogues (§4.1) and is absent in the local
modes.

---

## VERDICT

**CONFIRMED** — with documented deliberate deviations and five citation-level
notes (no mathematical error found).

- The per-event assembly $p_i = (\beta_G L_{\rm cat} + B_{\rm num})/D$
  (`bayesian_statistics.py:1721`) is an exact algebraic rearrangement of Gray
  main-text Eq. (9) with $w_G = \beta_G/D \leftrightarrow$ (A14),
  $1-w_G \leftrightarrow$ (A15), and $L_{\rm comp} = B_{\rm num}/\beta_{\bar G}
  \leftrightarrow$ (A19) = (A17)/(A18). Verified by direct comparison with the
  arXiv v4 LaTeX source.
- The repo's equation-numbering convention is internally consistent
  (continuous numbering: A9→24, A10→25, A11→26, A14→29, A17→32, A18→33), so
  "Eqs. 9, 26–33" in code comments resolve unambiguously to the equations they
  describe.
- $p_{\rm det}$ appears **only** in denominators ((A10) denominator, $D$,
  $\beta_{\bar G}$, $\Sigma_{\rm global}$), never multiplying a GW-likelihood
  numerator — the Mandel–Farr–Gair (2019) criterion is respected at
  1890–1895, 2006–2010 and 1622–1625.
- Deviations D1–D9 are all deliberate, individually referenced, and (for the
  de-rail fixes D7/D8) empirically calibrated; items in §5b map to no published
  equation and must be cited as project derivations in the thesis/paper, not
  as Gray (2020).
- If a stricter standard is wanted ("default mode = a published equation"),
  the verdict would instead be: the default `volume_deconv` mode implements
  Gray Eq. (9) + (A14)/(A15)/(A17)/(A18) for the mixture and completion
  channels, but its in-catalogue term implements a patch-restricted (A20)-like
  variant of (A10) with a beyond-Gray photo-$z$ kernel — exactly as the
  commission's de-rail study intended, and now stated precisely.
