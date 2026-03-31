# Computational and Analytical Methods: Enhanced Sampling for P_det Grid Estimation

**Project:** EMRI Enhanced Sampling for Injection Campaign
**Physics Domain:** Gravitational wave detection statistics, Monte Carlo sampling, selection effects
**Researched:** 2026-03-31

### Scope Boundary

This file covers sampling and variance-reduction METHODS for estimating detection probability P_det(z, M, h) from injection campaigns. It does NOT cover software tools (see COMPUTATIONAL.md) or the underlying GW physics. It answers: "How do we sample injections efficiently and build unbiased P_det grids?"

---

## Context and Problem Statement

We estimate P_det = P(SNR >= threshold | z, M, h) on a 2D grid in (z, M) for 7 values of h. Each grid cell's P_det is the fraction of injections in that cell that are detected. The current approach uses roughly uniform sampling in (log M, z) with uniform sky angles. Detection fraction is likely <10% at moderate-to-high z, making this a rare-event estimation problem in much of the parameter space.

**Core tension:** Uniform sampling wastes most compute on cells where P_det is near 0 or near 1 (well-determined). The interesting region is the detection boundary where P_det transitions from ~1 to ~0, and this is where we need the most samples.

---

## Recommended Methods

### Primary Method: Stratified Sampling with Importance Weighting

Use stratified sampling across (z, M) bins with importance-weight correction. This is the recommended approach because it is simple to implement, guaranteed unbiased with correct weights, and directly compatible with histogram-based P_det estimation.

### Secondary Method: Adaptive Boundary Refinement

After an initial uniform pass, concentrate additional samples near the P_det ~ 0.5 boundary. This is a two-stage approach that uses the first pass to identify where the detection boundary lies.

### Tertiary (defer unless needed): Cross-Entropy Adaptive Importance Sampling

For extreme rare-event cells (P_det < 0.01), use the cross-entropy method to iteratively optimize a proposal distribution. This is significantly more complex and only warranted if the simple stratified approach has unacceptable variance in low-P_det cells.

---

## Method Details

### Method 1: Importance-Weighted Histogram Estimator

**What:** Sample injections from a non-uniform proposal distribution q(theta) instead of the prior p(theta), then correct each sample's contribution to the histogram bin with the importance weight w = p(theta) / q(theta).

**Mathematical basis:**

The true P_det in bin B is:

```
P_det(B) = E_p[I(SNR >= thr) | theta in B]
         = integral_B p(theta) I(SNR(theta) >= thr) d theta  /  integral_B p(theta) d theta
```

where p(theta) is the physical prior over (z, M, sky angles, etc.) and I is the indicator function.

When sampling from proposal q(theta) instead of p(theta), the unbiased estimator is:

```
P_det_hat(B) = sum_{i in B} w_i * I(SNR_i >= thr)  /  sum_{i in B} w_i
```

where w_i = p(theta_i) / q(theta_i) is the importance weight for sample i. This is the self-normalized importance sampling estimator (also called the ratio estimator or Haji-Akbari estimator). It is:
- Biased for finite N (bias ~ O(1/N)), but consistent
- More stable than the unnormalized estimator when the normalization of p is approximate
- The standard estimator used in GW selection-effect studies (Tiwari 2018, arXiv:1712.00482)

**Key insight for histogram binning:** The denominator sum_{i in B} w_i acts as the effective number of prior-distributed samples in bin B. If you sample more densely near the detection boundary, the weights in easy bins (P_det ~ 0 or 1) will be large but few, and the weights in boundary bins will be small but numerous -- exactly as desired.

**Weight computation for current setup:**

The current sampler draws (log M, z) roughly uniformly and sky angles uniformly. If we modify the (z, M) proposal:

```
w_i = p(z_i, M_i) / q(z_i, M_i)
```

Sky angles remain uniformly sampled (unchanged), so they cancel in the weight ratio. Only the (z, M) marginal needs reweighting.

If the prior p(z, M) is the astrophysical rate density from Model1CrossCheck and the proposal q(z, M) is designed to oversample the boundary region:

```python
# Per-injection weight
w_i = p_prior(z_i, M_i) / q_proposal(z_i, M_i)

# Per-bin P_det (histogram estimator)
numerator = sum(w_i for i in bin if SNR_i >= threshold)
denominator = sum(w_i for i in bin)
P_det_bin = numerator / denominator if denominator > 0 else 0.0
```

**Variance of the estimator:**

```
Var[P_det_hat(B)] ~ (1/N_eff(B)) * P_det(B) * (1 - P_det(B))
```

where N_eff(B) = (sum w_i)^2 / sum w_i^2 is the effective sample size (Kish's formula). The variance is minimized when N_eff is maximized, which happens when all weights are equal (uniform sampling within the bin) or when the proposal concentrates samples where the variance P*(1-P) is largest (near P_det ~ 0.5).

**Convergence:** Standard Monte Carlo: O(1/sqrt(N_eff)). No improvement in convergence order, but N_eff can be much larger in the boundary region compared to uniform sampling.

**Known failure modes:**
- If q(theta) has thin tails relative to p(theta), some samples get enormous weights, inflating variance. Always ensure q has support wherever p does (q(theta) > 0 whenever p(theta) > 0).
- If the proposal is too concentrated, bins far from the concentration get very few samples with very large weights, potentially worse than uniform sampling in those bins.
- Self-normalized estimator is biased at O(1/N). For N ~ 1000+ per bin, this bias is negligible compared to statistical error.

**References:**
- Tiwari (2018), "Estimation of the Sensitive Volume for Gravitational-wave Source Populations Using Weighted Monte Carlo Integration," CQG. arXiv:1712.00482
- Gerosa & Fishbach (2024), "Quick recipes for gravitational-wave selection effects," CQG 41(12). arXiv:2404.16930

---

### Method 2: Stratified Sampling Across (z, M) Bins

**What:** Divide the (z, M) space into strata (which can coincide with or be finer than the P_det grid bins). Allocate a fixed number of samples per stratum rather than letting random sampling determine the per-bin count.

**Mathematical basis:**

Stratified sampling reduces variance compared to simple random sampling by eliminating between-stratum variance. For estimating P_det per bin, if we stratify by the bins themselves, we guarantee N_k samples per bin k rather than a Poisson-distributed count.

The estimator per bin is simply:

```
P_det_hat(k) = (1/N_k) * sum_{i in stratum k} I(SNR_i >= thr)
```

No importance weights needed if the within-bin distribution matches the prior. If the within-bin distribution is uniform (rather than following the prior within the bin), apply importance weights as in Method 1.

**Optimal allocation (Neyman allocation):**

Allocate samples proportional to the within-stratum standard deviation:

```
N_k ~ sigma_k = sqrt(P_det(k) * (1 - P_det(k)))
```

This concentrates samples in bins where P_det ~ 0.5 (maximum uncertainty) and reduces samples in bins where P_det ~ 0 or 1. Since P_det is unknown a priori, use a pilot run (or the initial uniform pass) to estimate it.

**Practical allocation for P_det grids:**

After a pilot run of N_pilot uniform injections:
1. Estimate P_det(k) in each bin from the pilot
2. Compute sigma_k = sqrt(P_det_hat(k) * (1 - P_det_hat(k)))
3. Allocate remaining budget N_remaining proportional to sigma_k
4. Within each bin, sample (z, M) from the prior restricted to that bin
5. Combine pilot + allocated samples (both are valid for the same estimator)

**Variance reduction factor:** For a 2D grid where most cells are P_det ~ 0, stratified sampling with Neyman allocation can reduce total variance by a factor of 5-20x compared to uniform sampling, because it stops wasting samples on dead bins.

**Convergence:** Same O(1/sqrt(N)) per bin, but N is now allocated optimally.

**Known failure modes:**
- If the grid is too coarse, within-bin variation in P_det is large and the bin-averaged P_det is a poor representation. Ensure grid resolution is fine enough that P_det does not vary dramatically within a single bin.
- Pilot run must be large enough to identify the boundary region. With 10000 total injections, using 2000-3000 as a pilot is reasonable.

**References:**
- Cochran (1977), "Sampling Techniques," 3rd ed., Wiley. (Standard reference for stratified sampling)
- Owen (2013), "Monte Carlo theory, methods and examples," Chapter 8. (Stratified sampling variance reduction)

---

### Method 3: Adaptive Boundary Refinement (Two-Stage)

**What:** Run an initial uniform injection campaign, identify the P_det boundary contour in (z, M) space, then run a second targeted campaign concentrating samples near the boundary.

**Algorithm:**

```
Stage 1: Uniform pilot
  - Run N_pilot injections uniformly in (z, M)
  - Compute P_det histogram on coarse grid
  - Identify boundary bins: 0.05 < P_det < 0.95

Stage 2: Boundary-targeted sampling
  - Define proposal q(z,M) with higher density in/near boundary bins
  - Simple approach: uniform within boundary bins, reduced density elsewhere
  - Run N_targeted injections from q
  - Combine both stages with importance weights:
    P_det(B) = sum_{all i in B} w_i * I_i / sum_{all i in B} w_i
    where w_i = 1 for pilot samples (drawn from prior)
    and w_i = p(z_i,M_i) / q(z_i,M_i) for targeted samples
```

**Multiple importance sampling (MIS) combination:**

When combining samples from the pilot (distribution p) and targeted (distribution q) stages, use the balance heuristic (Veach & Guibas 1995):

```
P_det_hat(B) = [sum_{i from p, in B} f(theta_i)/(N_p * p_i + N_q * q_i)
              + sum_{j from q, in B} f(theta_j)/(N_p * p_j + N_q * q_j)]
```

where f(theta) = I(SNR >= thr) * p(theta). In practice, for histogram-based estimation, the simpler approach of weighting each sample by w_i = p_i/q_i and using the self-normalized estimator works well and is much easier to implement.

**Practical simplification:** If the pilot uses the same prior as the physics model, pilot samples have w_i = 1. Targeted samples have w_i = p(z_i,M_i) / q(z_i,M_i). Pool all samples and use the weighted histogram estimator from Method 1.

**Convergence:** The boundary refinement can reduce the error in boundary bins by a factor of sqrt(N_total/N_boundary_uniform) compared to uniform, where N_boundary_uniform is the number of uniform samples that would have fallen in boundary bins.

**Known failure modes:**
- If the boundary is not well-identified from the pilot (too few pilot samples), the targeted stage may miss important regions.
- Sharp boundaries in (z, M) that are sensitive to sky angles can be missed if the boundary is identified only from (z, M) marginals.

---

### Method 4: Proposal Distribution Design for EMRI Injections

**What:** Design q(z, M) to oversample the detection-boundary region where the P_det gradient is steepest.

**Recommended proposal for EMRI P_det grids:**

The detection boundary is primarily set by the SNR threshold. SNR scales roughly as:

```
SNR ~ M^(5/6) / d_L(z) * sqrt(T_obs)
```

where d_L(z) is the luminosity distance. The boundary contour SNR = threshold defines a curve z_max(M) in the (z, M) plane. We want to oversample near this curve.

**Concrete proposal construction:**

```python
def proposal_density(z, M, z_max_estimate):
    """
    Proposal that oversamples the detection boundary.

    z_max_estimate: function M -> z giving approximate boundary from pilot.
    """
    # Base: uniform in (log M, z) as currently used
    base = 1.0 / (z_max * (log_M_max - log_M_min))

    # Enhancement near boundary: Gaussian bump centered on z_max(M)
    sigma_z = 0.1  # width of boundary region (tune from pilot)
    boundary_enhancement = exp(-0.5 * ((z - z_max_estimate(M)) / sigma_z)**2)

    # Mixture: alpha * base + (1-alpha) * boundary_enhanced
    alpha = 0.3  # fraction kept uniform (defensive sampling)
    q = alpha * base + (1 - alpha) * boundary_enhancement * normalization
    return q
```

**Critical: defensive mixture.** Always include a uniform component (alpha ~ 0.2-0.4) to ensure coverage of the full parameter space. This prevents the proposal from having zero density anywhere the prior is nonzero, which would make importance weights undefined.

**References:**
- Chua & Vallisneri (2023), "Rapid determination of LISA sensitivity to EMRIs with machine learning," MNRAS 522(4). arXiv:2212.06166 -- demonstrates that the SNR function over EMRI parameter space is smooth enough to interpolate, supporting the idea that the detection boundary is a well-defined contour.

---

### Method 5: Variance Diagnostics and Effective Sample Size Monitoring

**What:** Monitor the quality of importance-weighted estimates in real time to detect pathological weight distributions.

**Diagnostics to compute per bin:**

| Diagnostic | Formula | Healthy Range | Action if Unhealthy |
|---|---|---|---|
| Effective sample size | N_eff = (sum w)^2 / sum w^2 | > 50 per bin | Allocate more samples |
| Maximum weight fraction | max(w) / sum(w) | < 0.1 | Proposal too different from prior |
| Coefficient of variation of weights | std(w) / mean(w) | < 2 | Proposal mismatched |
| P_det standard error | sqrt(P*(1-P)/N_eff) | < 0.05 (5%) | Need more samples in this bin |

**Implementation:**

```python
def weighted_pdet_with_diagnostics(weights, detected, min_neff=50):
    """
    Compute importance-weighted P_det for a single bin.

    Parameters
    ----------
    weights : array, importance weights w_i = p(theta_i)/q(theta_i)
    detected : array of bool, whether SNR >= threshold
    min_neff : minimum effective sample size for reliable estimate

    Returns
    -------
    pdet : float, estimated detection probability
    se : float, standard error estimate
    neff : float, effective sample size
    reliable : bool, whether estimate meets quality threshold
    """
    w = weights
    n_eff = np.sum(w)**2 / np.sum(w**2)

    pdet = np.sum(w * detected) / np.sum(w)
    se = np.sqrt(pdet * (1 - pdet) / n_eff) if n_eff > 1 else np.inf
    reliable = (n_eff >= min_neff)

    return pdet, se, n_eff, reliable
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|---|---|---|---|
| Estimator type | Self-normalized IS (ratio estimator) | Unnormalized IS | Unnormalized requires exact normalization of p(theta); self-normalized is more robust |
| Proposal design | Boundary-enhanced mixture | Pure boundary sampling | Pure boundary gives zero coverage in interior; need defensive uniform component |
| Adaptive method | Two-stage (pilot + targeted) | Fully adaptive (iterative) | Two-stage is simpler, sufficient for 2D grid; iterative adds complexity for marginal gain |
| Rare-event method | Importance sampling | Cross-entropy method | CE is powerful but overkill for P_det estimation; our "rare event" is detection, not extreme tail |
| Grid construction | Histogram with IS weights | Kernel density estimation (KDE) | Histogram is simpler, directly gives per-bin estimate, no bandwidth selection needed |
| ML surrogate | Not recommended for this project | Neural network P_det interpolator (Chua & Vallisneri) | Requires large training set and validation; histogram approach is adequate for 2D grid at current scale |

---

## Integration with Current Codebase

### Current setup (from milestone context):
- `Model1CrossCheck` MCMC sampler draws (log M, z)
- Sky angles: phiS ~ Uniform[0, 2pi], qS = arccos(Uniform(-1, 1))
- z > 0.5 cutoff for high-z importance sampling
- SNR-only computation (no Fisher matrix)
- ~10000 injections per h-value, 7 h-values

### Minimal change for importance weighting:

1. Record the proposal density q(z_i, M_i) for each injection alongside the injection parameters
2. Compute the prior density p(z_i, M_i) (from the astrophysical model)
3. When building the P_det histogram, use the weighted estimator instead of simple counting

### Stratified allocation change:

1. After pilot run (~3000 injections uniform), compute preliminary P_det grid
2. Identify boundary bins (0.05 < P_det < 0.95)
3. Allocate remaining ~7000 injections with Neyman-optimal allocation across bins
4. Within each bin, sample from the prior restricted to that bin

### What NOT to change:
- Sky angle sampling (already optimal: uniform on the sphere)
- The SNR computation itself
- The h-value grid (7 values, run independently)

---

## Validation Strategy

| Check | Expected Result | Tolerance | Reference |
|---|---|---|---|
| Unweighted vs weighted P_det from uniform samples | Identical (weights all = 1 for uniform prior) | Machine precision | Self-consistency |
| P_det in high-SNR region (low z, high M) | ~1.0 | < 0.02 deviation | Physical expectation |
| P_det in low-SNR region (high z, low M) | ~0.0 | < 0.02 deviation | Physical expectation |
| Effective sample size per bin | > 50 for reliable bins | Flag if < 50 | Kish (1965) |
| sum of weights per bin | Proportional to prior mass in bin | < 10% deviation | Normalization check |
| P_det monotonicity in z (fixed M) | Decreasing with z | No inversions beyond stat. noise | Physical: farther = fainter |
| P_det monotonicity in M (fixed z) | Generally increasing with M | Allow non-monotonicity from LISA band edge | Physical: heavier = louder (mostly) |
| Comparison: 10k uniform vs 3k pilot + 7k stratified | Same P_det within 2-sigma | Error bars overlap | Method validation |

---

## Computational Cost Estimates

| Approach | Injections | Wall Time (est.) | P_det Boundary Accuracy |
|---|---|---|---|
| Current: uniform | 10000/h | ~3 hours/h (1s/injection) | ~10% per boundary bin (few samples there) |
| Stratified (Neyman) | 10000/h (3k pilot + 7k targeted) | ~3 hours/h (same total) | ~4-5% per boundary bin |
| Importance-weighted boundary enhancement | 10000/h | ~3 hours/h (same total) | ~3-5% per boundary bin |
| Doubled budget, stratified | 20000/h | ~6 hours/h | ~3% per boundary bin |

**Key point:** Stratified sampling and importance weighting do NOT increase the total number of injections. They redistribute the same budget more efficiently. The computational cost per injection is unchanged (~1 second for SNR). The improvement is purely statistical: better allocation of fixed compute.

---

## Practical Recommendation: Implementation Order

1. **First (minimal effort, high impact):** Record proposal densities for all injections. When building P_det grids, use the self-normalized importance-weighted histogram estimator. This makes ANY future change to the proposal distribution automatically handled.

2. **Second (moderate effort, moderate impact):** Implement stratified sampling with Neyman-optimal allocation. Run a pilot of ~30% of budget uniformly, then allocate the remaining 70% based on pilot P_det estimates.

3. **Third (only if needed):** Design a boundary-enhanced proposal distribution using the pilot P_det boundary contour. This gives the most variance reduction but requires estimating z_max(M) from the pilot.

4. **Defer:** Cross-entropy method, neural network surrogates, fully adaptive sequential methods. These are overkill for the current 2D grid at 10k injections per h.

---

## Sources

- [Tiwari (2018), "Weighted Monte Carlo for GW sensitive volume," arXiv:1712.00482](https://arxiv.org/abs/1712.00482) -- Weighted MC estimator for reusing injections across population models. Directly applicable weight formula.
- [Gerosa & Fishbach (2024), "Quick recipes for GW selection effects," arXiv:2404.16930](https://arxiv.org/abs/2404.16930) -- Detection probability approximations, Marcum Q-function, noise realization effects. State of the art for GW P_det.
- [Chua & Vallisneri (2023), "Rapid EMRI sensitivity with ML," arXiv:2212.06166](https://arxiv.org/abs/2212.06166) -- Neural network SNR interpolation for EMRI selection functions. Demonstrates smoothness of SNR landscape.
- [Owen (2013), "Monte Carlo theory, methods and examples"](https://artowen.su.domains/mc/) -- Chapters 8-9 on stratified sampling and importance sampling. Standard MC reference.
- [Rubinstein & Kroese (2004), "The Cross-Entropy Method," Springer](https://link.springer.com/book/10.1007/978-1-4757-4321-0) -- Adaptive IS via cross-entropy for rare events. Deferred method.
- [Veach & Guibas (1995), "Optimally Combining Sampling Techniques for MC Rendering," SIGGRAPH](https://dl.acm.org/doi/10.1145/218380.218498) -- Multiple importance sampling / balance heuristic for combining samples from different proposals.
- [Kish (1965), "Survey Sampling," Wiley](https://www.wiley.com/en-us/Survey+Sampling-p-9780471109495) -- Effective sample size formula N_eff = (sum w)^2 / sum w^2.
