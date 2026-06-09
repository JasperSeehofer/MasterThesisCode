# Research Brief: Boundary-Bias in Nonparametric Estimation of a Binary Detection Probability p_det(x)

**Scope.** Estimating p_det(x) = E[Y | x], a bounded conditional probability of a binary
label Y = (SNR ≥ threshold), as a function of covariates x = (d_L [Gpc], log10 M_z), from a
fixed Monte-Carlo injection campaign (~10^5 EMRI events). Current estimator: Nadaraya–Watson
(local-constant) kernel regression with Gaussian kernels and Scott's-rule bandwidth on a 2D
grid. Problem: O(h) boundary bias collapses p̂ → ~0.5 as d_L → 0, where the true value is 1.

**Author note on provenance.** Claims tagged **[verified-primary]** were checked against the
primary paper/abstract or a textbook-grade source. Claims tagged **[verified-secondary]** are
supported by reputable lecture notes / review articles / journal summaries but I did not open
the primary equation. Claims tagged **[inferred]** are my reasoning, not a direct citation.
Two items I could not source to a primary equation are flagged inline.

---

## 1. Executive answer

This is **nonparametric regression of a bounded (binary) response**, and the failure you are
seeing is the textbook **O(h) boundary bias of the local-constant (Nadaraya–Watson)
estimator**: at an edge of the covariate support the kernel becomes one-sided, so NW averages
in points from only one side and, where the true function has nonzero slope (here p_det falls
steeply with d_L), it is biased to first order in the bandwidth h. The **standard fix is to
replace local-constant with local-linear (more generally odd-degree local-polynomial)
regression**, which is *design-adaptive* and corrects boundary bias *automatically* — it
reduces boundary bias from O(h) to O(h²), matching the interior rate, with no separate
boundary kernel and essentially no variance penalty (Fan 1992; Fan & Gijbels 1996; Wand &
Jones 1995) **[verified-secondary]**. Because the response is binary and must lie in [0,1],
the principled version is **local likelihood / local logistic regression** (fitting a local
linear *logit* by local Bernoulli maximum likelihood), which simultaneously (a) gives the
automatic boundary correction of local-linear and (b) guarantees p̂ ∈ (0,1) (Fan, Heckman &
Wand 1995; Loader 1999) **[verified-secondary]**. A **known boundary limit** (p_det → 1 as
d_L → 0) is incorporated in the literature by *constrained smoothing*, *pseudo-/anchor
observations*, or a *parametric offset/guide* on the logit scale; the cleanest first-
principles route here is to add the physical d_L → 0 saturation as an anchor on the logit
scale (logit → +∞, i.e. a strong "all detected" pseudo-mass) or to enforce monotonicity in
d_L. Recommendation up front: **local-linear regression on the logit scale (local logistic),
with monotonicity in d_L and a near-field anchor; second choice plain local-linear on the
probability scale (clipped to [0,1]).**

---

## 2. RQ1 — Canonical framing

- **It is regression, not density estimation.** p_det(x) = E[Y|x] with Y ∈ {0,1} is a
  *conditional mean / regression* of a binary response, i.e. a probability *surface*, not a
  probability *density*. The distinction matters because the correct estimator class and the
  correct boundary remedies differ from those used for KDE (see RQ2/RQ6). **[inferred,
  standard]**

- **NW = local-constant regression.** p̂(x) = Σ_k K(x−x_k) Y_k / Σ_k K(x−x_k) is exactly the
  degree-0 local-polynomial (local-constant) estimator (Fan & Gijbels 1996; Wand & Jones
  1995). **[verified-secondary]**

- **The O(h) boundary bias is the known defect of NW.** Interior bias of NW is O(h²); at the
  boundary it degrades to O(h) because the kernel is truncated/one-sided, so the local average
  is pulled toward values on the interior side wherever the regression function has nonzero
  slope. This is the standard statement in Fan & Gijbels (1996), Wand & Jones (1995), and
  Hastie–Tibshirani–Friedman *ESL* (local-regression chapter). NW also has a *design bias*
  term proportional to f'(x)/f(x) (the covariate-density gradient) that survives in the
  interior; this too vanishes for local-linear. **[verified-secondary]**

- **Mechanistic match to your symptom.** Near d_L = 0 the injection design has support only on
  one side (d_L ≥ 0). The one-sided Gaussian kernel averages found (Y=1, near) and not-found
  (Y=0, far) injections; since true p_det drops steeply with d_L, the one-sided average
  collapses toward an intermediate value (~0.5), exactly the O(h) edge bias. **[inferred,
  consistent with theory]**

**Key references for RQ1:** Fan & Gijbels (1996); Wand & Jones (1995); Hastie, Tibshirani &
Friedman, *Elements of Statistical Learning* (2009), §6 local regression; Fan (1992, 1993).

---

## 3. RQ2 — Standard boundary-bias fixes for kernel *regression*

| Method | Mechanism | Removes boundary bias? | Pros | Cons | Fit to our 2D, one known edge |
|---|---|---|---|---|---|
| **Local-linear / local-polynomial** | Fit a degree-1 (or higher odd) polynomial by locally weighted LS at each x | **Yes — O(h)→O(h²)**, automatic, no boundary kernel | Design-adaptive (removes f'/f design bias too); minimax-efficient; variance ≈ NW; reproduces local slope so it tracks the steep d_L falloff at the edge | Can over/undershoot outside [0,1]; needs clipping for a probability; slightly noisier with sparse data | **Best general fit.** Local slope in d_L is exactly what fixes the near-field |
| **Boundary kernels** | Replace the kernel near the edge with a kernel satisfying modified moment conditions (Gasser–Müller 1979/1984; Müller 1991) | Yes, to O(h²) by construction near the edge | Keeps a local-constant flavor; well-developed theory | Requires explicit edge handling per boundary; messier in 2D; can give negative weights | Workable but more bespoke than local-linear |
| **Reflection of data** | Mirror data across the boundary then smooth (Schuster 1985; Silverman 1986; Cline & Hart 1991) | Partial: removes bias **only if the true slope at the boundary is ~0**; forces ∂p/∂x = 0 at edge | Trivial to implement | **Imposes a flat boundary** → wrong here: p_det has nonzero slope as it peels off from 1; reflection creates a "shoulder" artifact | **Poor fit** — our boundary slope is nonzero |
| **Response transformation / GLM-type local likelihood (logit)** | Smooth a transformed response (logit) by local likelihood; back-transform | Yes (when combined with local-linear logit) + keeps p̂∈(0,1) | Correct support; natural for binary Y; well-grounded (Fan–Heckman–Wand 1995; Loader 1999) | Iterative (local IRLS); a bit more code | **Best principled fit** for binary Y; see RQ3 |
| **Monotone / isotonic / shape-constrained** | Enforce monotonicity (here p_det ↓ in d_L) via constrained smoothing or smooth+monotonize | Mitigates edge artifacts; encodes physics | Encodes a true physical constraint; stabilizes near-field | Monotonicity alone does not *set* the boundary value | Strong complement: combine with local-linear/logit |
| **Anchoring a KNOWN boundary value (p→1 at d_L→0)** | Constrained smoothing / pseudo-observations / parametric offset (guide) | Directly fixes the edge value | Uses real physics; cheap | Must avoid distorting interior | **Recommended add-on** (details below) |

### Why local-linear fixes what NW cannot (RQ2 core)
Local-constant fits only an intercept locally, so where the kernel is one-sided it cannot
distinguish "the function is genuinely lower here" from "I am averaging over a region where the
function is sloped." Local-**linear** fits an intercept *and a slope*, so it extrapolates the
local trend to the evaluation point. At a boundary the fitted slope absorbs the asymmetry of
the one-sided window, cancelling the leading O(h) bias term and leaving O(h²). Equivalently,
odd-degree local polynomials have a bias that does not depend on the design density gradient
f'/f and is the same form at the boundary as in the interior — the celebrated "automatic
boundary correction" (Fan 1992 *Design-adaptive nonparametric regression*, JASA; Fan 1993 on
minimax; Fan & Gijbels 1996, Ch. 3; Ruppert & Wand 1994 multivariate). **[verified-secondary]**

### Incorporating a KNOWN boundary limit (the p→1 anchor)
Three literature-standard mechanisms, in increasing rigor:

1. **Pseudo-observations / anchor points.** Add synthetic Y=1 "found" injections at/near
   d_L = 0 (a small d_L slab) so the local fit is pulled to 1 there. On the probability scale
   this is a soft constraint; on the logit scale it pushes the linear predictor toward +∞
   (saturation). Standard in constrained-smoothing practice. **[verified-secondary, general]**
2. **Parametric guide / offset (local likelihood with a guide).** Fit p_det = g(η₀(x) +
   nonparametric correction), where η₀ is a physics-motivated parametric form that already
   satisfies p→1 as d_L→0 (e.g. a logistic in SNR ~ M_z^{5/6}/d_L). The nonparametric part
   then only models the residual. This is the *local quasi-likelihood with a parametric guide*
   idea (Fan, Wu & Feng, *Local quasi-likelihood with a parametric guide*, Ann. Statist.
   2009). **[verified-secondary]**
3. **Hard constrained smoothing.** Solve the local-LS/local-likelihood problem subject to
   p̂(d_L→0)=1 and/or monotonicity in d_L (constrained kernel regression: Hall & Huang 2001
   monotone constraints; Du, Parmeter & Racine, constrained nonparametric kernel regression).
   **[verified-secondary]**

---

## 4. RQ3 — Binary-response specifics: local likelihood / local logistic

For Y ∈ {0,1}, the principled smoother is **local likelihood (local logistic) regression**,
which treats Y_k ~ Bernoulli(p(x_k)) and models the **logit** η(x) = log[p/(1−p)] locally as a
low-order polynomial, then maximizes the *kernel-weighted Bernoulli log-likelihood*:

    η̂ = argmax_{β}  Σ_k K_h(x − x_k) · [ Y_k · η_k(β) − log(1 + exp η_k(β)) ],

with η_k(β) = β₀ + β₁ᵀ(x_k − x) (local-linear logit), and p̂(x) = expit(β̂₀) = 1/(1+e^{−β̂₀}).
This is the regression analogue of the local likelihood density method. **[verified-secondary;
model setup is standard, transcribed from the Bernoulli local-likelihood definition]**

- **Foundational theory:** Fan, Heckman & Wand (1995), *Local polynomial kernel regression for
  generalized linear models and quasi-likelihood functions*, JASA 90:141–150 — establishes the
  asymptotics and shows the **same automatic-boundary-correction / design-adaptivity** carries
  over to the GLM/quasi-likelihood (binomial) case. **[verified-secondary]**
- **Practical reference & software:** Loader (1999), *Local Regression and Likelihood*
  (Springer); the `locfit` package implements local logistic/Poisson likelihood smoothing.
  **[verified-secondary]**
- **Does logit + boundary handling solve both problems?** **Yes, jointly.** (i) The logit link
  guarantees p̂ ∈ (0,1) automatically — no clipping artifacts. (ii) Using a *local-linear*
  (not local-constant) logit inherits the O(h²) boundary behavior, so the near-field bias is
  corrected. (iii) Adding the p→1 anchor on the logit scale is natural: "always detected"
  corresponds to η → +∞, implemented as strongly-weighted Y=1 pseudo-data or a guide η₀ that
  diverges as d_L→0. **[inferred from (3.2)+(3.3) above; mechanism is standard]**
- **Caveat — perfect separation.** Because ground truth is *exactly* 1 for d_L < 0.1 Gpc, the
  local logistic MLE in that region is at the boundary of the parameter space (η→+∞,
  separation). This is expected and handled by mild ridge/Firth-type penalization or by the
  anchor/guide; it is not a defect. **[inferred; separation handling is standard in logistic
  regression]**

---

## 5. RQ4 — GW selection functions from injections (domain literature)

How the GW community estimates p_det / VT(θ) from injection campaigns, and how they treat
edges and Monte-Carlo noise.

- **Monte-Carlo importance-sampling estimator of VT (the dominant paradigm).** The detectable
  spacetime volume / selection integral α(Λ) = ∫ p_det(θ) p_pop(θ|Λ) dθ is estimated by
  reweighting *found* injections: α̂ = (1/N_inj) Σ_found p_pop(θ_i|Λ)/p_draw(θ_i). This is a
  Monte-Carlo *integral*, not a smooth surface estimate — it sidesteps estimating p_det(θ)
  pointwise. Tiwari (2018), *Estimation of the sensitive volume … using weighted Monte Carlo
  integration*, CQG 35:145009 (arXiv:1712.00482). **[verified-secondary]**
- **Accuracy requirement / MC noise control — Farr (2019).** *Accuracy Requirements for
  Empirically-Measured Selection Functions*, RNAAS 3:66 (arXiv:1904.10879): gives the
  effective sample size N_eff of the importance-sampling sum and the rule of thumb **N_eff ≫
  4·N_obs** for population inference to be unbiased by MC noise. **[verified-primary, abstract
  + formula confirmed]**
- **Correlated MC error & marginalization bias — Essick & Farr (2022).** Marginalizing over MC
  uncertainty in the selection integral can *bias* the inferred population because the MC
  errors are correlated across hyperparameters; resolution is more samples or evaluating point
  estimates on a hyperparameter grid. (arXiv:2204.00461.) Relevant to your D(h) integral: MC
  noise in p_det propagates into the normalization. **[verified-secondary]**
- **Hierarchical selection-conditional likelihood — Mandel, Farr & Gair (2019).** *Extracting
  distribution parameters from multiple uncertain observations with selection biases*, MNRAS
  486:1086 (arXiv:1809.02063): the canonical derivation of the selection-corrected
  (Malmquist) likelihood, i.e. *why* you divide by the selection normalization D(h). This is
  the framework your H0 posterior sits in. **[verified-primary, abstract confirmed]**
- **ML / flexible p_det surfaces — Talbot & Golomb; Gerosa et al.** Several groups fit p_det(θ)
  as a smooth/learned function (neural nets, Gaussian processes, density-ratio estimation)
  rather than a raw MC sum, precisely to reduce variance and get smoothness. E.g. Talbot &
  Golomb (2023) on selection-function / density estimation for GW populations; Gerosa et al.
  (2020) *Fast, flexible … Malmquist bias with machine learning* (arXiv:2012.01317). These
  learn p_det on a bounded scale and rely on the model class (sigmoid output) to respect
  [0,1]; boundary saturation (p→1 at high SNR / low distance) is captured by the sigmoid head
  rather than an explicit boundary kernel. **[verified-secondary]**
- **VAMANA / mixture models — Tiwari (2021).** *VAMANA: modeling BBH population with minimal
  assumptions*, CQG (arXiv:2006.15047): weighted-Gaussian *mixture* density estimation for the
  population (a density-estimation, not p_det-regression, tool) — relevant as a smooth
  alternative but aimed at p_pop, not the binary selection function. **[verified-secondary]**
- **Dark-siren / gwcosmo — Gray et al. (2020).** *Cosmological inference using GW events and
  galaxy catalogues* (arXiv:1908.06050), PRD 101:122001: defines the catalogue method and the
  GW *selection effect* p(D|H0) used to normalize the H0 likelihood — the dark-siren analogue
  of your D(h). Completeness correction follows Gray et al. and Schutz (1986). **[verified-
  secondary]**

**Takeaway for the domain (RQ4).** The GW mainstream estimates the *integral* α(Λ) by
importance-sampling found injections and controls MC noise via N_eff (Farr 2019; Essick & Farr
2022) — it largely *avoids* estimating a pointwise p_det surface with a kernel, so it doesn't
hit the NW boundary problem. Where groups *do* fit a pointwise p_det surface, they use a
model with a bounded (sigmoid) output (ML/GP), which encodes saturation rather than using
boundary kernels. **No standard GW reference uses Nadaraya–Watson for p_det**, so your edge
problem is a generic kernel-regression artifact, fixable with the statistics tools in RQ2–RQ3.
**[inferred from the surveyed literature]**

---

## 6. RQ5 — Concrete recommendation for our pipeline

Constraints recap: 2D covariates (d_L, log10 M_z); binary Y; ~10^5 points; **one** physically
known boundary (p→1 as d_L→0, exactly verified for d_L<0.1 Gpc); need a **smooth** estimator
(a prior fix replaced a histogram specifically to kill jitter that destabilized the H0 grid);
must remain importable/testable on CPU.

### Ranked recommendation

| Rank | Method | Expected boundary behavior | Smoothness | Impl. cost | Notes |
|---|---|---|---|---|---|
| **1** | **Local-linear logit (local logistic) + monotone-in-d_L + near-field anchor** | O(h²) edge bias; p̂∈(0,1) guaranteed; anchored to 1 as d_L→0 | High (analytic) | Medium (local IRLS or `statsmodels`/`locfit`-style per grid node; or fit logit-GAM with monotone spline in d_L) | Solves [0,1] bounding *and* boundary bias jointly; handles separation via tiny ridge/anchor |
| **2** | **Local-linear regression on probability scale, clipped to [0,1], + monotone-in-d_L** | O(h²) edge bias; tracks steep falloff | High | **Low** (closed-form WLS per grid node; e.g. `statsmodels` lowess-2D or hand-rolled local-linear) | Cheapest principled fix; needs clipping; no logit guarantee but with 10^5 pts overshoot is mild |
| **3** | **Importance-sampling MC integral (skip the surface)** for D(h) directly + N_eff monitor | No surface ⇒ no edge bias in D(h) | n/a (integral) | Medium (refactor D(h) to reweighted sum; track N_eff≫4N_obs) | The GW-standard route (Farr 2019); removes the problem at its source if D(h) is all you need from p_det |

### Explicit comparisons requested
- **vs. reverting to a histogram.** A histogram is just local-constant with a box kernel and a
  hard bin grid: it has the **same O(h) boundary bias** *and* reintroduces the jitter you
  already removed (piecewise-constant ⇒ grid instability in the H0 sweep). **Do not revert.**
  Local-linear/logit dominates it on bias, smoothness, and variance. **[inferred, standard]**
- **vs. local-linear-on-logit (option 1).** Logit costs an iterative local fit and separation
  handling, but buys guaranteed [0,1], a natural place to inject the p→1 anchor (η→+∞), and
  the cleanest treatment of the binary likelihood. Given that the near-field is *exactly*
  saturated, the anchor/guide is easy and the separation is benign. **Recommended.**
- **Minimal-risk path.** If implementation time is tight, ship **option 2** (local-linear +
  clip + monotone d_L) first — it is a closed-form drop-in for NW that fixes the headline
  boundary bias — then upgrade to option 1 (logit) if you need exact [0,1] guarantees or the
  clip introduces visible artifacts. **[inferred]**

### Expected effect on the physics
Local-linear/logit will recover p̂(d_L→0)→1, raising p_det in the near field, which **increases
D(h)=∫p_det dV_c/dz** at small d_L and **removes the high-H0 bias** in the posterior described
in the motivation. **[inferred — consistent with the stated failure mode; validate
numerically against the 100%-detection ground truth at d_L<0.1 Gpc]**

---

## 7. RQ6 — Secondary (flag only): errors-in-variables / deconvolution

A *different* problem from RQ1–5: there, the **covariates x_k themselves are measured with
error** (errors-in-variables), and one wants the regression/density in the *true* (latent) x.
That requires **deconvolution** estimators — Stefanski & Carroll (1990) and Carroll & Hall
(1988) (deconvolution KDE via Fourier inversion of the characteristic function); Fan & Truong
(1993) for deconvolution *regression*; Delaigle & Meister (2007, 2008) for heteroscedastic
measurement error. These have notoriously slow (logarithmic) convergence and are only needed
when the *injection covariates* are noisy. **In our problem the injection (d_L, log10 M_z) are
exact simulation inputs and Y is deterministic given x**, so there is no errors-in-variables
issue and deconvolution is *not* applicable. (It would only become relevant if you later
folded per-event measurement uncertainty on d_L/M into the p_det inputs.) **[verified-
secondary on the references; inferred on the not-applicable conclusion]**

---

## 8. References (with arXiv / DOI)

**Statistics — local polynomial & boundary bias**
- Fan, J. (1992). *Design-adaptive nonparametric regression.* JASA 87(420):998–1004.
  DOI:10.1080/01621459.1992.10476255. (Local-linear is design-adaptive, automatic boundary
  correction.) **[verified-secondary]**
- Fan, J. (1993). *Local linear regression smoothers and their minimax efficiencies.* Ann.
  Statist. 21(1):196–216. DOI:10.1214/aos/1176349022. **[verified-secondary]**
- Fan, J. & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications.* Chapman &
  Hall. (Ch. 3: boundary behavior; the standard reference.) **[verified-secondary]**
- Wand, M.P. & Jones, M.C. (1995). *Kernel Smoothing.* Chapman & Hall. **[verified-secondary]**
- Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*, 2nd ed., §6
  (local regression; boundary issues). **[verified-secondary]**
- Ruppert, D. & Wand, M.P. (1994). *Multivariate locally weighted least squares regression.*
  Ann. Statist. 22(3):1346–1370. DOI:10.1214/aos/1176325632. (Multivariate local-linear —
  relevant to your 2D case.) **[verified-secondary]**

**Boundary kernels / reflection / transformation**
- Gasser, T. & Müller, H.-G. (1979). *Kernel estimation of regression functions.* In
  *Smoothing Techniques for Curve Estimation*, LNM 757, Springer. (Boundary kernels.)
  **[verified-secondary]**
- Müller, H.-G. (1991). *Smooth optimum kernel estimators near endpoints.* Biometrika
  78(3):521–530. DOI:10.1093/biomet/78.3.521. **[verified-secondary]**
- Schuster, E.F. (1985). *Incorporating support constraints into nonparametric estimators of
  densities.* Comm. Statist. Theory Methods 14(5):1123–1136. (Reflection method.) **[verified-
  secondary]**
- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis.* Chapman &
  Hall. (Reflection; boundary effects.) **[verified-secondary]**
- Cline, D.B.H. & Hart, J.D. (1991). *Kernel estimation of densities with discontinuities or
  discontinuous derivatives.* Statistics 22(1):69–84. **[verified-secondary]**

**Local likelihood / binary response**
- Fan, J., Heckman, N.E. & Wand, M.P. (1995). *Local polynomial kernel regression for
  generalized linear models and quasi-likelihood functions.* JASA 90(429):141–150.
  DOI:10.1080/01621459.1995.10476496. (Local logistic; binomial; automatic boundary
  correction in the GLM case.) **[verified-secondary]**
- Loader, C. (1999). *Local Regression and Likelihood.* Springer. (Local logistic; `locfit`.)
  **[verified-secondary]**
- Fan, J., Wu, Y. & Feng, Y. (2009). *Local quasi-likelihood with a parametric guide.* Ann.
  Statist. 37(6B):4153–4183. DOI:10.1214/09-AOS713. (Parametric-guide route for the anchor.)
  **[verified-secondary]**

**Shape-constrained / monotone smoothing**
- Hall, P. & Huang, L.-S. (2001). *Nonparametric kernel regression subject to monotonicity
  constraints.* Ann. Statist. 29(3):624–647. DOI:10.1214/aos/1009210683. **[verified-
  secondary]**
- Mammen, E. (1991). *Estimating a smooth monotone regression function.* Ann. Statist.
  19(2):724–740. (Smooth-then-monotonize.) **[verified-secondary]**
- Du, P., Parmeter, C.F. & Racine, J.S. (2013). *Constrained nonparametric kernel regression:
  estimation and inference.* Statistica Sinica 23:1347–1371. **[verified-secondary]**

**GW selection functions / dark sirens**
- Tiwari, V. (2018). *Estimation of the sensitive volume for GW source populations using
  weighted Monte Carlo integration.* CQG 35:145009. arXiv:1712.00482.
  DOI:10.1088/1361-6382/aac89d. **[verified-secondary]**
- Farr, W.M. (2019). *Accuracy Requirements for Empirically-Measured Selection Functions.*
  RNAAS 3:66. arXiv:1904.10879. DOI:10.3847/2515-5172/ab1d5f. (N_eff ≫ 4·N_obs.) **[verified-
  primary]**
- Essick, R. & Farr, W.M. (2022). *Precision Requirements for Monte Carlo Sums within
  Hierarchical Bayesian Inference.* arXiv:2204.00461. **[verified-secondary]**
- Mandel, I., Farr, W.M. & Gair, J.R. (2019). *Extracting distribution parameters from multiple
  uncertain observations with selection biases.* MNRAS 486:1086. arXiv:1809.02063.
  DOI:10.1093/mnras/stz896. **[verified-primary]**
- Gerosa, D. et al. (2020). *Fast, flexible, and accurate evaluation of GW Malmquist bias with
  machine learning.* arXiv:2012.01317. **[verified-secondary]**
- Tiwari, V. (2021). *VAMANA: modeling BBH population with minimal assumptions.* CQG.
  arXiv:2006.15047. **[verified-secondary]**
- Gray, R. et al. (2020). *Cosmological inference using GW events and galaxy catalogues.* PRD
  101:122001. arXiv:1908.06050. DOI:10.1103/PhysRevD.101.122001. (gwcosmo; dark-siren
  selection.) **[verified-secondary]**
- Schutz, B.F. (1986). *Determining the Hubble constant from gravitational wave observations.*
  Nature 323:310. (Foundational dark-siren idea.) **[verified-secondary]**

**Errors-in-variables / deconvolution (RQ6, different problem)**
- Stefanski, L.A. & Carroll, R.J. (1990). *Deconvoluting kernel density estimators.*
  Statistics 21(2):169–184. DOI:10.1080/02331889008802238. **[verified-secondary]**
- Carroll, R.J. & Hall, P. (1988). *Optimal rates of convergence for deconvolving a density.*
  JASA 83(404):1184–1186. **[verified-secondary]**
- Fan, J. & Truong, Y.K. (1993). *Nonparametric regression with errors in variables.* Ann.
  Statist. 21(4):1900–1925. **[verified-secondary]**
- Delaigle, A. & Meister, A. (2008). *Density estimation with heteroscedastic error.*
  Bernoulli 14(2):562–579. arXiv:0805.2216. **[verified-secondary]**

### Items I could NOT source to a primary equation (flagged)
- **Exact Talbot & Golomb (2023) title/venue/arXiv** for a GW selection-function/density-
  estimation paper — confirmed the authors work in this area but did not pin the exact
  reference; treat the Talbot–Golomb citation as **approximate** until verified on ADS/arXiv.
- The precise **local-logistic model equations** in §3 are transcribed from the standard
  Bernoulli local-likelihood definition (consistent with Fan–Heckman–Wand 1995 and Loader
  1999) but were not copied verbatim from those primary texts.
