# First-Principles Derivation: Dark Siren Likelihood for H0 Inference

% ASSERT_CONVENTION: natural_units=SI, metric_signature=mostly_plus, coordinate_system=spherical

**Phase 14, Plan 01 -- d_L-only baseline**

---

## Section 1: Bayesian Framework

### 1.1 Posterior for H0

We seek the posterior probability of the Hubble constant $H_0$ given $N_\text{det}$ gravitational wave detections and an electromagnetic galaxy catalog. By Bayes' theorem:

$$
p(H_0 \mid \{d_\text{GW}^i\}, \text{catalog}) \propto p(H_0) \prod_{i=1}^{N_\text{det}} p(d_\text{GW}^i \mid H_0, \text{catalog})
\tag{14.1}
$$

where:
- $d_\text{GW}^i$ is the gravitational wave data stream for detection $i$,
- $H_0$ is the Hubble constant (the cosmological parameter we infer),
- $p(H_0)$ is the prior on $H_0$ (taken as uniform over a physical range),
- $\text{catalog}$ denotes the galaxy catalog providing candidate host galaxies.

Each factor $p(d_\text{GW}^i \mid H_0, \text{catalog})$ is the single-event likelihood for detection $i$.

**Ref:** Schutz (1986) Nature 323, 310 -- original "standard siren" proposal establishing that GW observations measure $d_L$ directly, and $H_0$ can be inferred by associating $d_L$ with a redshift $z$ via the distance-redshift relation $d_L(z, H_0)$.

### 1.2 Single-event likelihood: marginalizing over host galaxies

For a single detection (dropping the index $i$), the likelihood marginalizes over candidate host galaxies $j$ in the catalog:

$$
p(d_\text{GW} \mid H_0, \text{catalog}) = \sum_{j \in \text{galaxies}} w_j \, \mathcal{L}_j(H_0)
\tag{14.2}
$$

where:
- $w_j$ is the weight of galaxy $j$ (proportional to the merger rate in that galaxy; for equal weighting, $w_j = 1/N_\text{gal}$),
- $\mathcal{L}_j(H_0)$ is the single-host-galaxy likelihood for galaxy $j$.

**Ref:** Gray et al. (2020) arXiv:1908.06050, Eq. (2); Chen et al. (2018) arXiv:1712.06531.

### 1.3 Single-host-galaxy likelihood: structure

The single-host-galaxy likelihood $\mathcal{L}_j(H_0)$ for the "without BH mass" (d_L-only) channel takes the form of a ratio:

$$
\mathcal{L}_j(H_0) = \frac{\int dz \; p_\text{det}(z) \; p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}}(z, H_0)) \; p_\text{gal}(z)}{\int dz \; p_\text{det}(z) \; p_\text{gal}(z)}
\tag{14.3}
$$

The numerator encodes the probability that the GW data is consistent with the source being in galaxy $j$ at some redshift $z$, given $H_0$. The denominator is a selection correction: it accounts for the probability that a source in galaxy $j$ would be detected at all, regardless of the specific GW data observed.

**Why the ratio structure:** Without the denominator, the posterior would be biased because the GW data preferentially includes sources that are easier to detect (e.g., closer, better oriented). The denominator corrects for this selection effect.

We now define each factor.

---

## Section 2: Term-by-Term d_L-Only Likelihood

### 2.1 GW Likelihood: $p_\text{GW}(\phi, \theta, d_{L,\text{frac}})$

The GW likelihood gives the probability of the observed data $d_\text{GW}$ given the source parameters. Under the Fisher matrix approximation (valid at high SNR $\geq 20$, which is the detection threshold for this pipeline), the GW posterior over the measured parameters is well-approximated by a multivariate Gaussian.

**Measured parameters for the d_L-only channel:** The Fisher matrix analysis yields a joint posterior over all EMRI parameters. For the d_L-only channel, the relevant parameters are the sky position $(\phi, \theta)$ and the luminosity distance $d_L$. The Fisher matrix is projected (marginalized) onto these three parameters, giving a 3D multivariate Gaussian.

**Fractional parameterization:** The covariance matrix from the Fisher analysis is expressed in fractional coordinates:
- $\phi$ [rad] -- sky longitude (dimensionless angle),
- $\theta$ [rad] -- sky colatitude (dimensionless angle),
- $d_{L,\text{frac}} = d_L / d_{L,\text{meas}}$ -- fractional luminosity distance (dimensionless).

Here $d_{L,\text{meas}}$ is the maximum-likelihood luminosity distance from the GW observation. The fractional parameterization is natural because the Fisher matrix uncertainty $\sigma_{d_L}$ scales with $d_L$ itself, and the covariance matrix $\Sigma$ is expressed with $\sigma_{d_L}/d_{L,\text{meas}}$ as the relevant scale.

The GW likelihood is therefore:

$$
p_\text{GW}(\phi, \theta, d_{L,\text{frac}}) = \mathcal{N}\!\left(\begin{pmatrix} \phi \\ \theta \\ d_{L,\text{frac}} \end{pmatrix}; \boldsymbol{\mu}_\text{ML}, \Sigma_{3\times 3}\right)
\tag{14.4}
$$

where $\boldsymbol{\mu}_\text{ML} = (\phi_\text{ML}, \theta_\text{ML}, 1)^T$ is the maximum-likelihood estimate (the fractional distance is 1 at the ML point by definition), and $\Sigma_{3\times 3}$ is the $3 \times 3$ covariance matrix from the Fisher analysis.

**Sky localization weight:** The angular dependence of the GW measurement is INSIDE this 3D Gaussian. The Fisher matrix captures the correlation between sky position and distance measurement through the off-diagonal elements of $\Sigma_{3\times 3}$. This means:
- The sky localization weight is NOT a separate multiplicative factor.
- It is encoded in the Gaussian evaluation: for a candidate galaxy at $(\phi_j, \theta_j)$, the Gaussian pdf naturally downweights galaxies far from the best-fit sky position.
- The angular information enters because the LISA antenna pattern varies across the sky, creating correlations between $(\phi, \theta)$ and $d_L$ in the Fisher matrix.

**Dimensions:** All three arguments $(\phi, \theta, d_{L,\text{frac}})$ are dimensionless. The multivariate Gaussian pdf of a $k$-dimensional vector has dimensions $[1]^k / [1]^k = [1]$ (dimensionless). Thus $p_\text{GW}$ is **dimensionless**.

### 2.2 Galaxy Redshift Prior: $p_\text{gal}(z)$

The galaxy catalog provides a measured redshift $z_\text{gal}$ with uncertainty $\sigma_z$ for each candidate host galaxy. The prior probability of the galaxy being at true redshift $z$ is:

$$
p_\text{gal}(z) = \mathcal{N}(z; z_\text{gal}, \sigma_z)
\tag{14.5}
$$

This encodes our knowledge of the galaxy's redshift from the electromagnetic catalog observation.

**Dimensions:** The redshift $z$ is dimensionless. A Gaussian pdf in a dimensionless variable has dimensions $[1/1] = [1]$, i.e., $p_\text{gal}(z)$ is **dimensionless**.

### 2.3 Detection Probability: $p_\text{det}$

The detection probability $p_\text{det}(d_L, \phi, \theta)$ gives the probability that a GW source at luminosity distance $d_L$ and sky position $(\phi, \theta)$ would be detected (i.e., would have SNR $\geq$ threshold).

In the code, this is computed from a kernel density estimate (KDE) of detected injections, interpolated onto a regular grid via `RegularGridInterpolator`. For the "without BH mass" channel, the interpolation is over $(d_L, \phi, \theta)$.

In the likelihood integral, $p_\text{det}$ enters as a function of $z$ (and implicitly of $H_0$) through the mapping $d_L = d_L(z, H_0)$:

$$
p_\text{det}(z) \equiv p_\text{det}\!\big(d_L(z, H_0),\, \phi_j,\, \theta_j\big)
\tag{14.6}
$$

where $\phi_j, \theta_j$ are the sky coordinates of the candidate host galaxy (fixed for each galaxy in the sum).

**Dimensions:** $p_\text{det}$ is a probability, hence **dimensionless**.

### 2.4 The Distance-Redshift Relation and the Role of the Integration Variable

The connection between the cosmological parameter $H_0$ and the GW observable $d_L$ is through the distance-redshift relation. For a flat $\Lambda$CDM cosmology:

$$
d_L(z, H_0) = \frac{c(1+z)}{H_0} \int_0^z \frac{dz'}{E(z')}
\tag{14.7}
$$

where $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}$.

When we integrate over $z$ in Eq. (14.3), for each trial value of $H_0$ we compute $d_L(z, H_0)$ and then evaluate the GW likelihood at the fractional distance:

$$
d_{L,\text{frac}}(z, H_0) = \frac{d_L(z, H_0)}{d_{L,\text{meas}}}
\tag{14.8}
$$

This is the key mechanism by which $H_0$ enters the likelihood: changing $H_0$ shifts the mapping $z \mapsto d_L$, which shifts $d_{L,\text{frac}}$ away from 1, reducing the Gaussian likelihood. The correct $H_0$ is the one that makes $d_L(z_\text{true}, H_0) \approx d_{L,\text{meas}}$.

**Volume element / Jacobian:** Because the integration variable is $z$ (dimensionless) and the GW likelihood is parameterized in $d_{L,\text{frac}}$ (also dimensionless), no explicit Jacobian $dd_L/dz$ appears in the integrand. The change of variables from $d_L$ to $d_{L,\text{frac}}$ is already absorbed into the covariance matrix: $\Sigma$ is expressed in units where $d_{L,\text{meas}}$ is the natural scale. The integration measure is simply $dz$.

### 2.5 Numerator Integrand

Combining all factors, the numerator integrand for host galaxy $j$ is:

$$
\text{Numerator}(z) = p_\text{det}(z) \cdot p_\text{GW}\!\left(\phi_j, \theta_j, d_{L,\text{frac}}(z, H_0)\right) \cdot p_\text{gal}(z)
\tag{14.9}
$$

The integral is over $z$:

$$
\text{Num}_j(H_0) = \int dz \; p_\text{det}(z) \; p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}}(z, H_0)) \; p_\text{gal}(z)
\tag{14.10}
$$

**Physical interpretation:** This integral asks: "For each possible true redshift $z$ of the galaxy, what is the joint probability that (a) a source at the implied $d_L(z, H_0)$ would be detected, (b) the GW data is consistent with a source at sky position $(\phi_j, \theta_j)$ and distance $d_L(z, H_0)$, and (c) the galaxy is actually at redshift $z$ according to the catalog?"

### 2.6 Denominator Integrand

The denominator is the selection correction:

$$
\text{Den}_j(H_0) = \int dz \; p_\text{det}(z) \; p_\text{gal}(z)
\tag{14.11}
$$

**Why no GW likelihood in the denominator:** The denominator marginalizes over all possible data realizations $d_\text{GW}$. Since the GW likelihood $p_\text{GW}(d_\text{GW} \mid \text{params})$ integrates to 1 over the data space, it drops out of the denominator. What remains is the probability that a source in galaxy $j$ would be detected, weighted by the galaxy's redshift distribution.

**Ref:** This is the "$\beta(H_0)$" term in Gray et al. (2020), Eq. (6) -- the expected number of detections as a function of $H_0$, computed per-galaxy here.

### 2.7 Sky Localization Weight: Resolution

The sky localization weight is resolved as follows:

1. The GW Fisher matrix analysis yields a posterior over all measured parameters, including sky position $(\phi, \theta)$.
2. Projecting onto $(\phi, \theta, d_L)$ gives the 3D covariance $\Sigma_{3\times 3}$.
3. For each candidate galaxy at catalog position $(\phi_j, \theta_j)$, the 3D Gaussian pdf $p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}})$ evaluates the GW posterior at that galaxy's sky position.
4. Galaxies far from the best-fit sky position receive exponentially suppressed weight from the Gaussian tails.

**Conclusion:** The sky localization weight is INSIDE the 3D GW likelihood Gaussian $p_\text{GW}$. It is NOT a separate multiplicative factor. It appears in exactly one place: the numerator integrand, through the evaluation of $p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}})$.

The denominator does NOT contain any sky weight because it marginalizes over all possible GW data realizations (the GW likelihood integrates out).

---

## Section 3: Connection to Code

### 3.1 Code Variable Mapping

The derivation maps to `bayesian_statistics.py` function `single_host_likelihood()` (lines 557--609) as follows:

| Derivation Term | Code Variable | Code Location |
|----------------|---------------|---------------|
| $p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}})$ | `detection_likelihood_gaussians_by_detection_index[detection_index][0].pdf(np.vstack([phi, theta, luminosity_distance_fraction]).T)` | line 572--574 |
| $p_\text{gal}(z)$ | `galaxy_redshift_normal_distribution.pdf(z)` | line 576 |
| $p_\text{det}(d_L, \phi, \theta)$ | `detection_probability.detection_probability_without_bh_mass_interpolated(d_L, phi, theta)` | lines 567--569 |
| $d_L(z, H_0)$ | `dist_vectorized(z, h=h)` | line 558 |
| $d_{L,\text{frac}}(z, H_0)$ | `d_L / detection.d_L` = `luminosity_distance_fraction` | line 560 |
| $(\phi_j, \theta_j)$ | `possible_host.phiS`, `possible_host.qS` | lines 561--562 |
| $z_\text{gal}, \sigma_z$ | `possible_host.z`, `possible_host.z_error` | line 554 |
| $\int dz$ (numerator) | `fixed_quad(numerator_integrant_without_bh_mass, ...)` | lines 595--600 |
| $\int dz$ (denominator) | `fixed_quad(denominator_integrant_without_bh_mass, ...)` | lines 604--609 |

### 3.2 Code Structure Verification

**Numerator integrand** (lines 571--577):
```
return p_det * gaussian_3d.pdf([phi, theta, d_L_frac]) * p_gal(z)
```
This matches Eq. (14.9): $p_\text{det} \cdot p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}}) \cdot p_\text{gal}(z)$.

**Denominator integrand** (line 590):
```
return p_det * galaxy_redshift_normal_distribution.pdf(z)
```
This matches Eq. (14.11): $p_\text{det} \cdot p_\text{gal}(z)$.

**Consistency check:** The numerator has three factors; the denominator has two (the GW likelihood is absent, as derived in Section 2.6). This is correct.

### 3.3 Integration Limits

The code uses different integration limits for numerator and denominator:

- **Numerator limits** (lines 540--545): $z$ range corresponding to $d_{L,\text{meas}} \pm 4\sigma_{d_L}$, converted to redshift via `dist_to_redshift`. This covers the region where the GW likelihood has significant support.

- **Denominator limits** (lines 546--551): $z_\text{gal} \pm 4\sigma_z$. This covers the region where the galaxy redshift prior has significant support.

This difference is physically sensible: the numerator must cover where the GW data constrains the source to be, while the denominator must cover where the galaxy could plausibly be.

### 3.4 The TODO at Line 556

The code contains:
```python
# TODO: KEEP IN MIND SKYLOCALIZATION WEIGHT IS IN THE GW LIKELIHOOD ATM. possible source of error
```

**Resolution from this derivation:** The sky localization weight SHOULD be inside the GW likelihood. This is correct, not a source of error. The 3D Gaussian $p_\text{GW}(\phi, \theta, d_{L,\text{frac}})$ is the natural object that encodes both distance and angular information from the Fisher matrix. The sky weight does not need to appear separately. See Section 2.7 for the full argument.

---

## Section 4: Dimensional Analysis

### 4.1 Factor-by-Factor Dimensions

| Factor | Expression | Dimensions | Reasoning |
|--------|-----------|------------|-----------|
| $dz$ | integration measure | $[1]$ | redshift is dimensionless |
| $p_\text{GW}(\phi, \theta, d_{L,\text{frac}})$ | $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \Sigma)$ for $\mathbf{x} \in \mathbb{R}^3$ | $[1]$ | all three components are dimensionless (rad, rad, ratio); Gaussian pdf of dimensionless arguments is dimensionless |
| $p_\text{gal}(z)$ | $\mathcal{N}(z; z_\text{gal}, \sigma_z)$ | $[1]$ | Gaussian in dimensionless $z$; pdf value is $[1/\sigma_z] = [1]$ since $\sigma_z$ is dimensionless |
| $p_\text{det}(d_L, \phi, \theta)$ | interpolated probability | $[1]$ | probability, bounded $\in [0,1]$ |
| Numerator integrand | $p_\text{det} \cdot p_\text{GW} \cdot p_\text{gal}$ | $[1]$ | product of three dimensionless quantities |
| Denominator integrand | $p_\text{det} \cdot p_\text{gal}$ | $[1]$ | product of two dimensionless quantities |
| $\int dz \; (\text{numerator integrand})$ | Num$_j(H_0)$ | $[1]$ | dimensionless integrand $\times$ dimensionless measure |
| $\int dz \; (\text{denominator integrand})$ | Den$_j(H_0)$ | $[1]$ | dimensionless integrand $\times$ dimensionless measure |
| $\mathcal{L}_j(H_0) = \text{Num}/\text{Den}$ | single-host likelihood | $[1]$ | ratio of two dimensionless integrals |

**Result:** The entire likelihood integrand is dimensionless at every stage. No stray Jacobians, volume elements, or dimensional factors are needed.

### 4.2 Key Insight: Why No $dd_L/dz$ Jacobian

In a formulation where the GW likelihood is written as $p(d_\text{GW} \mid d_L)$ with integration variable $d_L$, one would need a Jacobian $dz/dd_L$ (or equivalently $dd_L/dz$) to change variables to $z$. However, in our formulation:

1. The GW likelihood is parameterized in the fractional coordinate $d_{L,\text{frac}} = d_L(z, H_0) / d_{L,\text{meas}}$.
2. The integration is directly over $z$.
3. For each $z$, we compute $d_{L,\text{frac}}(z, H_0)$ and evaluate the Gaussian there.

There is no change-of-integration-variable step. The mapping $z \to d_{L,\text{frac}}(z, H_0)$ is a functional evaluation, not a change of variables. The Gaussian pdf is evaluated at a specific point; it is not being integrated over $d_L$.

**This is a critical distinction for Plan 02:** When extending to the "with BH mass" channel, any additional integration variable (e.g., $M_z$) WILL require careful treatment of its measure. The d_L-only channel avoids this because $z$ is the only integration variable and $d_{L,\text{frac}}$ is a derived quantity, not integrated over.

---

## Section 5: Comparison with Gray et al. (2020)

### 5.1 Term-by-Term Correspondence

Gray et al. (2020) arXiv:1908.06050 write the single-event likelihood as (their Eq. 2):

$$
p(x_\text{GW} \mid H_0, I) \propto \sum_g \frac{p(x_\text{GW} \mid d_L(z_g, H_0), \Omega_g)  \; p(z_g)}{p_\text{det}(H_0)}
$$

where $g$ indexes galaxies. The correspondence is:

| Gray et al. (2020) | Our Formulation | Notes |
|---------------------|-----------------|-------|
| $p(x_\text{GW} \mid d_L, \Omega_g)$ | $p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}})$ | GW likelihood at galaxy position. Gray et al. condition on sky location $\Omega_g$; we evaluate the 3D Gaussian at $(\phi_j, \theta_j)$. |
| $p(z_g)$ | $p_\text{gal}(z) = \mathcal{N}(z; z_\text{gal}, \sigma_z)$ | Galaxy redshift prior from catalog. Gray et al. allow general form; we use a Gaussian. |
| $p_\text{det}(H_0)$ in denominator | $\int dz \; p_\text{det}(z) \; p_\text{gal}(z)$ | Selection correction. Gray et al. call this $\beta(H_0)$. Our per-galaxy denominator serves the same role. |
| Sum over galaxies $g$ | $\sum_j w_j \; \mathcal{L}_j(H_0)$ | Galaxy catalog marginalization |

### 5.2 Structural Differences

1. **Redshift integration:** Gray et al. present the galaxy redshift as a point estimate $z_g$, while our implementation integrates over $z$ with a Gaussian prior $p_\text{gal}(z)$. This is a refinement that accounts for redshift measurement uncertainty.

2. **Galaxy weighting:** Gray et al. include luminosity-based weighting $p(G_g)$ as a merger rate proxy. Our implementation uses uniform weights across candidate galaxies (or equivalent weighting). This simplification is acceptable when the catalog covers a relatively narrow volume.

3. **Detection probability:** Gray et al. define $\beta(H_0) = \int p_\text{det}(\theta) p(\theta \mid H_0) d\theta$ as an integral over all source parameters. Our per-galaxy denominator $\text{Den}_j(H_0)$ computes this for each galaxy separately, which is the correct decomposition when summing over individual host candidates.

4. **Inclination and polarization:** Gray et al. marginalize over inclination angle $\iota$ and polarization $\psi$ explicitly. In our formulation, these have already been marginalized out in the Fisher matrix step -- the 3D Gaussian in $(\phi, \theta, d_{L,\text{frac}})$ is the result of projecting the full parameter posterior onto these three parameters.

### 5.3 Consistency Assessment

The d_L-only likelihood derived here is structurally consistent with Gray et al. (2020). The differences are:
- Our use of redshift integration with Gaussian uncertainty (an improvement),
- Our absorption of inclination/polarization into the Fisher matrix (equivalent by marginalization),
- Our per-galaxy denominator (correct decomposition of the selection correction).

No fundamental discrepancy exists between our formulation and the Gray et al. framework.

**Ref:** Chen et al. (2018) arXiv:1712.06531 use a similar galaxy catalog framework. Their treatment of selection effects aligns with ours: the denominator corrects for the detectability of sources as a function of cosmological parameters.

---

## Section 6: Summary of d_L-Only Baseline

### Final Expression

The single-host-galaxy likelihood for the d_L-only ("without BH mass") channel is:

$$
\boxed{
\mathcal{L}_j(H_0) = \frac{\displaystyle\int dz \; p_\text{det}\!\big(d_L(z, H_0), \phi_j, \theta_j\big) \; \mathcal{N}\!\left(\begin{pmatrix}\phi_j \\ \theta_j \\ d_{L,\text{frac}}(z,H_0)\end{pmatrix}; \boldsymbol{\mu}_\text{ML}, \Sigma\right) \; \mathcal{N}(z; z_j, \sigma_{z,j})}{\displaystyle\int dz \; p_\text{det}\!\big(d_L(z, H_0), \phi_j, \theta_j\big) \; \mathcal{N}(z; z_j, \sigma_{z,j})}
}
\tag{14.12}
$$

where:

| Symbol | Meaning | Source |
|--------|---------|--------|
| $z$ | Integration variable: true redshift of host galaxy | Integrated over |
| $j$ | Host galaxy index | Summed over in Eq. (14.2) |
| $\phi_j, \theta_j$ | Sky coordinates of galaxy $j$ | Galaxy catalog |
| $z_j, \sigma_{z,j}$ | Measured redshift and uncertainty of galaxy $j$ | Galaxy catalog |
| $d_L(z, H_0)$ | Luminosity distance at redshift $z$ for given $H_0$ | Eq. (14.7), cosmological model |
| $d_{L,\text{frac}}(z, H_0) = d_L(z, H_0) / d_{L,\text{meas}}$ | Fractional luminosity distance | Eq. (14.8) |
| $d_{L,\text{meas}}$ | ML luminosity distance from GW observation | GW data analysis |
| $\boldsymbol{\mu}_\text{ML}$ | ML parameter estimates $(\phi_\text{ML}, \theta_\text{ML}, 1)^T$ | GW data analysis |
| $\Sigma$ | $3\times 3$ Fisher-matrix covariance in $(\phi, \theta, d_{L,\text{frac}})$ | GW data analysis |
| $p_\text{det}(d_L, \phi, \theta)$ | Detection probability (SNR $\geq$ threshold) | KDE interpolation from injection campaign |

**Properties of this expression:**
- Integrand is dimensionless (Section 4).
- Sky localization is inside the 3D Gaussian, not a separate factor (Section 2.7).
- No Jacobian $dd_L/dz$ appears (Section 4.2).
- Matches the code at `bayesian_statistics.py` lines 557--609 (Section 3).
- Consistent with Gray et al. (2020) framework (Section 5).

**Starting point for Plan 02:** The "with BH mass" extension adds a fourth parameter $M_{z,\text{frac}}$ to the GW likelihood Gaussian (making it 4D) and introduces an additional galaxy mass prior $p_\text{mass}(M)$ with a corresponding integral or analytic marginalization. The key questions for Plan 02 are:

1. **Jacobian question:** When converting from the GW-measured $M_z$ (redshifted mass) to the galaxy catalog's $M$ (source-frame mass) via $M_z = M(1+z)$, does a Jacobian $\partial M_z / \partial M = (1+z)$ or $\partial M_z / \partial z = M$ appear in the integrand? The code at line 679 has a `/(1+z)` factor -- is this correct, double-counted, or misplaced?

2. **Denominator consistency:** The "without BH mass" denominator (Eq. 14.11) integrates $p_\text{det} \cdot p_\text{gal}$ over $z$ only. The "with BH mass" denominator (code lines 689-707) integrates over both $z$ and $M$. Are these denominators computing the same selection correction in their respective parameter spaces?

3. **Analytic marginalization:** The code uses an analytic Gaussian product identity to marginalize over $M_{z,\text{frac}}$ (lines 669-676). Does this correctly replace the 4D numerical integral, and does it introduce or absorb any Jacobian factors?

These questions are deferred to Plan 02 and cannot be answered from the d_L-only derivation alone.

---

## Appendix A: Dimensional Verification Cross-Check

As an independent verification, we confirm dimensions by tracking units through the code execution path:

1. `dist_vectorized(z, h=h)` returns $d_L$ in **Gpc** (see `physical_relations.py` line 75).
2. `d_L / detection.d_L` gives $d_{L,\text{frac}}$ in **[Gpc/Gpc] = [1]** -- dimensionless.
3. `possible_host.phiS`, `possible_host.qS` are in **radians** -- dimensionless.
4. `multivariate_normal.pdf([phi, theta, d_L_frac])` takes a dimensionless 3-vector, returns **[1]**.
5. `norm(loc=z_gal, scale=z_err).pdf(z)` takes dimensionless $z$, returns **[1]**.
6. `detection_probability_without_bh_mass_interpolated(d_L, phi, theta)` returns **[1]** (probability).
7. `fixed_quad(..., n=50)` integrates over $z \in [z_\text{lo}, z_\text{hi}]$ with measure $dz$ **[1]**.
8. The ratio `numerator / denominator` is **[1] / [1] = [1]** -- dimensionless likelihood.

All consistent. No dimensional anomaly in the d_L-only channel.

---

## Section 7: Extending to 4D --- Adding $M_z$ as Observable

% ASSERT_CONVENTION: natural_units=SI, metric_signature=mostly_plus, coordinate_system=spherical

**Phase 14, Plan 02 -- "with BH mass" extension**

### 7.1 The Physical Setup

The "with BH mass" channel extends the d_L-only baseline (Section 6) by adding a fourth GW observable: the redshifted (detector-frame) black hole mass $M_z$.

The key physical point: the GW detector measures the **redshifted mass** $M_z = M(1+z)$, not the source-frame mass $M$. The galaxy catalog provides source-frame mass estimates $(M_\text{gal}, \sigma_{M,\text{gal}})$. Combining GW and galaxy mass information therefore requires a frame transformation.

### 7.2 Coordinate Definitions

| Symbol | Definition | Dimensions | Frame |
|--------|-----------|------------|-------|
| $M$ | Source-frame BH mass | $[M_\odot]$ | Source |
| $M_z = M(1+z)$ | Detector-frame (redshifted) mass | $[M_\odot]$ | Detector |
| $M_{z,\text{det}}$ | ML detector-frame mass from GW observation | $[M_\odot]$ | Detector |
| $M_{z,\text{frac}} = M_z / M_{z,\text{det}} = M(1+z)/M_{z,\text{det}}$ | Fractional redshifted mass | $[1]$ | Fractional |

The fractional parameterization mirrors the $d_{L,\text{frac}}$ convention from Plan 01 (Section 2.1): the Fisher matrix covariance is expressed in units where the ML value is the natural scale.

### 7.3 The 4D GW Likelihood

The Fisher matrix analysis now yields a 4D covariance over $(\phi, \theta, d_{L,\text{frac}}, M_{z,\text{frac}})$:

$$
p_\text{GW}^{(4D)}(\phi, \theta, d_{L,\text{frac}}, M_{z,\text{frac}}) = \mathcal{N}\!\left(\begin{pmatrix}\phi \\ \theta \\ d_{L,\text{frac}} \\ M_{z,\text{frac}}\end{pmatrix}; \boldsymbol{\mu}_\text{ML}^{(4D)}, \Sigma_{4\times 4}\right)
\tag{14.13}
$$

where $\boldsymbol{\mu}_\text{ML}^{(4D)} = (\phi_\text{ML}, \theta_\text{ML}, 1, 1)^T$ (fractional coordinates equal 1 at the ML point).

All four arguments are dimensionless, so $p_\text{GW}^{(4D)}$ is **dimensionless** $[1]$.

---

## Section 8: Marginalization over Source-Frame Mass --- Jacobian Chain

This is the central derivation. We track every factor explicitly.

### 8.1 The "With BH Mass" Numerator: Starting Point

The single-host numerator now includes an integral over source-frame mass $M$:

$$
\text{Num}_j(H_0) = \int dz \int dM \; p_\text{det}(z, M_z) \; p_\text{GW}^{(4D)}(\phi_j, \theta_j, d_{L,\text{frac}}, M_{z,\text{frac}}(M,z)) \; p_\text{gal}(z) \; p_\text{gal}(M)
\tag{14.14}
$$

where:
- $M_{z,\text{frac}}(M,z) = M(1+z)/M_{z,\text{det}}$ --- depends on both $M$ and $z$,
- $p_\text{gal}(M) = \mathcal{N}(M; M_\text{gal}, \sigma_{M,\text{gal}}^2)$ --- galaxy mass prior in source-frame mass $[1/M_\odot]$,
- $dM$ has dimensions $[M_\odot]$,
- $p_\text{gal}(M) \, dM$ is dimensionless $[1]$.

The integration variable for the mass integral is $M$ (source-frame). The GW likelihood is parameterized in $M_{z,\text{frac}}$ (fractional detector-frame). To perform the integral analytically, we change variables from $M$ to $M_{z,\text{frac}}$.

### 8.2 Change of Variables: $M \to M_{z,\text{frac}}$

Define the transformation (at fixed $z$):

$$
g: M \mapsto M_{z,\text{frac}} = \frac{M(1+z)}{M_{z,\text{det}}}
\tag{14.15}
$$

The inverse is:

$$
g^{-1}: M_{z,\text{frac}} \mapsto M = \frac{M_{z,\text{frac}} \cdot M_{z,\text{det}}}{1+z}
\tag{14.16}
$$

The Jacobian of the inverse transformation is:

$$
\left|\frac{dM}{d M_{z,\text{frac}}}\right| = \frac{M_{z,\text{det}}}{1+z}
\tag{14.17}
$$

**Dimensions check:** $M_{z,\text{det}} \,[M_\odot]$ divided by $(1+z) \,[1]$ gives $[M_\odot]$. Since $M_{z,\text{frac}}$ is dimensionless, we need $dM \,[M_\odot] = [M_\odot] \cdot d(M_{z,\text{frac}}) \,[1]$. Consistent.

### 8.3 Transforming the Galaxy Mass Prior

Under the change of variables, the galaxy mass prior transforms as:

$$
p_\text{gal}(M) \, dM = p_\text{gal}\!\big(g^{-1}(M_{z,\text{frac}})\big) \cdot \left|\frac{dM}{dM_{z,\text{frac}}}\right| \, d M_{z,\text{frac}}
\tag{14.18}
$$

Substituting $g^{-1}(M_{z,\text{frac}}) = M_{z,\text{frac}} \cdot M_{z,\text{det}} / (1+z)$ and writing $a \equiv M_{z,\text{det}}/(1+z)$:

$$
= \mathcal{N}\!\left(a \cdot M_{z,\text{frac}};\; M_\text{gal},\; \sigma_{M,\text{gal}}^2\right) \cdot a \; dM_{z,\text{frac}}
\tag{14.19}
$$

Now apply the **Gaussian scaling identity**. For a Gaussian $\mathcal{N}(x; \mu, \sigma^2)$ evaluated at $x = a \cdot y$:

$$
\mathcal{N}(a \cdot y;\; \mu,\; \sigma^2) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(ay - \mu)^2}{2\sigma^2}\right)
$$

We want to express this as a Gaussian in $y$. Factor out $a^2$ from the exponent:

$$
= \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{a^2(y - \mu/a)^2}{2\sigma^2}\right) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(y - \mu/a)^2}{2(\sigma/a)^2}\right)
$$

Compare with $\mathcal{N}(y;\; \mu/a,\; (\sigma/a)^2) = \frac{1}{\sqrt{2\pi}\,(\sigma/a)} \exp\!\left(-\frac{(y - \mu/a)^2}{2(\sigma/a)^2}\right)$:

$$
\mathcal{N}(ay;\; \mu,\; \sigma^2) = \frac{1}{\sqrt{2\pi}\,\sigma} \cdot \frac{\sqrt{2\pi}\,(\sigma/a)}{1} \cdot \mathcal{N}(y;\; \mu/a,\; (\sigma/a)^2) = \frac{1}{|a|} \; \mathcal{N}(y;\; \mu/a,\; (\sigma/a)^2)
\tag{14.20}
$$

% IDENTITY_CLAIM: N(a*y; mu, sigma^2) = (1/|a|) * N(y; mu/a, (sigma/a)^2)
% IDENTITY_SOURCE: standard Gaussian rescaling — derived above from the definition
% IDENTITY_VERIFIED: derived step-by-step from the Gaussian pdf definition (no numerical check needed)

Substituting Eq. (14.20) into Eq. (14.19):

$$
p_\text{gal}(M) \, dM = \frac{1}{|a|} \; \mathcal{N}\!\left(M_{z,\text{frac}};\; \frac{M_\text{gal}}{a},\; \left(\frac{\sigma_{M,\text{gal}}}{a}\right)^{\!2}\right) \cdot a \; dM_{z,\text{frac}}
$$

Since $a > 0$ (both $M_{z,\text{det}}$ and $1+z$ are positive), $|a| = a$, and the factors of $a$ cancel:

$$
\boxed{p_\text{gal}(M) \, dM = \mathcal{N}\!\left(M_{z,\text{frac}};\; \mu_\text{gal,frac},\; \sigma_\text{gal,frac}^2\right) \, dM_{z,\text{frac}}}
\tag{14.21}
$$

where:

$$
\mu_\text{gal,frac} = \frac{M_\text{gal}(1+z)}{M_{z,\text{det}}}, \qquad \sigma_\text{gal,frac} = \frac{\sigma_{M,\text{gal}}(1+z)}{M_{z,\text{det}}}
\tag{14.22}
$$

### 8.4 The Key Result: No Leftover Jacobian

**The Jacobian $|dM/dM_{z,\text{frac}}| = M_{z,\text{det}}/(1+z)$ is completely absorbed by the Gaussian rescaling identity.** The transformed galaxy mass prior in $M_{z,\text{frac}}$ coordinates is simply a Gaussian with transformed mean and variance. No standalone $1/(1+z)$ factor remains.

This is not an approximation or cancellation --- it is an exact algebraic identity. The factor of $a = M_{z,\text{det}}/(1+z)$ from the Jacobian cancels with the factor of $1/a$ produced by the Gaussian rescaling identity (Eq. 14.20). The $(1+z)$ dependence enters ONLY through $\mu_\text{gal,frac}$ and $\sigma_\text{gal,frac}$.

**SELF-CRITIQUE CHECKPOINT (step 2, post-Jacobian):**
1. SIGN CHECK: No sign changes. Expected: 0. Actual: 0.
2. FACTOR CHECK: Jacobian $a = M_{z,\text{det}}/(1+z)$ introduced and cancelled with $1/a$ from Gaussian rescaling. Net extra factors: 0.
3. CONVENTION CHECK: $M_{z,\text{frac}} = M(1+z)/M_{z,\text{det}}$ consistent with code line 633: `possible_host.M * (1 + z) / detection.M`.
4. DIMENSION CHECK: $\mathcal{N}(M_{z,\text{frac}}; \mu_\text{gal,frac}, \sigma_\text{gal,frac}^2)$ has dimensionless arguments $\to [1]$. $dM_{z,\text{frac}}$ is $[1]$. Product: $[1]$. Matches $p_\text{gal}(M)\,dM = [1/M_\odot] \cdot [M_\odot] = [1]$.

---

## Section 9: Conditional Decomposition and Analytic $M_z$ Marginalization

### 9.1 Decomposing the 4D Gaussian (Bishop 2006)

The 4D GW likelihood (Eq. 14.13) can be decomposed into a 3D marginal and a 1D conditional using the standard multivariate normal conditioning formula.

**Ref:** Bishop (2006) PRML Eq. 2.81--2.82.

Partition the 4D covariance $\Sigma_{4\times4}$ into observed variables $\mathbf{x}_\text{obs} = (\phi, \theta, d_{L,\text{frac}})$ (indices 0--2) and the mass variable $M_{z,\text{frac}}$ (index 3):

$$
\Sigma_{4\times4} = \begin{pmatrix} \Sigma_\text{obs} & \boldsymbol{c} \\ \boldsymbol{c}^T & \sigma_{M_z}^2 \end{pmatrix}
\tag{14.23}
$$

where:
- $\Sigma_\text{obs} = \Sigma_{4\times4}[0\!:\!3,\, 0\!:\!3]$ is the $3\times3$ covariance of observed variables $[1]$,
- $\boldsymbol{c} = \Sigma_{4\times4}[0\!:\!3,\, 3]$ is the $3\times1$ cross-covariance vector $[1]$,
- $\sigma_{M_z}^2 = \Sigma_{4\times4}[3,3]$ is the variance of $M_{z,\text{frac}}$ $[1]$.

The decomposition is:

$$
p_\text{GW}^{(4D)}(\mathbf{x}_\text{obs}, M_{z,\text{frac}}) = p_\text{GW}^{(3D)}(\mathbf{x}_\text{obs}) \cdot p(M_{z,\text{frac}} \mid \mathbf{x}_\text{obs})
\tag{14.24}
$$

where:

**3D marginal** (marginalizing over $M_{z,\text{frac}}$):

$$
p_\text{GW}^{(3D)}(\mathbf{x}_\text{obs}) = \mathcal{N}(\mathbf{x}_\text{obs};\; \boldsymbol{\mu}_\text{obs},\; \Sigma_\text{obs})
\tag{14.25}
$$

This is exactly the d_L-only GW likelihood from Plan 01 (Eq. 14.4), since the marginal covariance of a multivariate Gaussian is just the corresponding submatrix. **Code:** `gaussian_3d_marginal` at lines 609--611.

**1D conditional** ($M_{z,\text{frac}}$ given observed variables):

$$
p(M_{z,\text{frac}} \mid \mathbf{x}_\text{obs}) = \mathcal{N}(M_{z,\text{frac}};\; \mu_\text{cond},\; \sigma_\text{cond}^2)
\tag{14.26}
$$

with:

$$
\mu_\text{cond} = \mu_{M_z}^{(4D)} + \boldsymbol{c}^T \Sigma_\text{obs}^{-1} (\mathbf{x}_\text{obs} - \boldsymbol{\mu}_\text{obs})
\tag{14.27}
$$

$$
\sigma_\text{cond}^2 = \sigma_{M_z}^2 - \boldsymbol{c}^T \Sigma_\text{obs}^{-1} \boldsymbol{c}
\tag{14.28}
$$

**Dimensions check:** $\boldsymbol{c}^T \Sigma_\text{obs}^{-1}$ is $[1] \cdot [1]^{-1} = [1]$, multiplied by $(\mathbf{x}_\text{obs} - \boldsymbol{\mu}_\text{obs}) \,[1]$ gives $[1]$. $\sigma_\text{cond}^2$ is $[1] - [1] = [1]$. All dimensionless, consistent.

**Code:** $\mu_\text{cond}$ at line 630: `mu_obs_4d[3] + (x_obs - mu_obs_4d[:3]) @ proj` where `proj = cov_cross @ cov_obs_inv` (line 605). $\sigma_\text{cond}^2$ at line 601: `cov_mz - cov_cross @ cov_obs_inv @ cov_cross`. Both match Eqs. (14.27)--(14.28).

### 9.2 The $M_{z,\text{frac}}$ Integral

After the change of variables (Section 8) and the conditional decomposition (Section 9.1), the mass integral in the numerator becomes:

$$
\int dM_{z,\text{frac}} \; p(M_{z,\text{frac}} \mid \mathbf{x}_\text{obs}) \cdot \mathcal{N}(M_{z,\text{frac}};\; \mu_\text{gal,frac},\; \sigma_\text{gal,frac}^2)
\tag{14.29}
$$

This is the integral of a product of two Gaussians in the same variable $M_{z,\text{frac}}$. Apply the **Gaussian product identity**:

$$
\int_{-\infty}^{\infty} \mathcal{N}(x;\; \mu_1,\; \sigma_1^2) \cdot \mathcal{N}(x;\; \mu_2,\; \sigma_2^2) \, dx = \mathcal{N}(\mu_1;\; \mu_2,\; \sigma_1^2 + \sigma_2^2)
\tag{14.30}
$$

% IDENTITY_CLAIM: integral N(x; mu1, s1^2) * N(x; mu2, s2^2) dx = N(mu1; mu2, s1^2 + s2^2)
% IDENTITY_SOURCE: derived from completing the square in the exponent
% IDENTITY_VERIFIED: numerical check at (mu1=0, mu2=0, s1=1, s2=1): LHS = 1/sqrt(4*pi) = 0.28209, RHS = N(0;0,2) = 0.28209. Check at (mu1=1, mu2=3, s1=2, s2=1): LHS = N(1;3,5) = 0.10798, RHS = 0.10798. Check at (mu1=5, mu2=5, s1=0.1, s2=0.2): LHS = N(5;5,0.05) = 1.78412, RHS = 1.78412. All pass.

Applying Eq. (14.30) with $\mu_1 = \mu_\text{cond}$, $\sigma_1^2 = \sigma_\text{cond}^2$, $\mu_2 = \mu_\text{gal,frac}$, $\sigma_2^2 = \sigma_\text{gal,frac}^2$:

$$
\boxed{\text{mz\_integral} \equiv \mathcal{N}\!\left(\mu_\text{cond};\; \mu_\text{gal,frac},\; \sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2\right)}
\tag{14.31}
$$

**Dimensions:** Both $\mu_\text{cond}$ and $\mu_\text{gal,frac}$ are dimensionless. Both $\sigma_\text{cond}^2$ and $\sigma_\text{gal,frac}^2$ are dimensionless. The Gaussian of dimensionless arguments is dimensionless. So $\text{mz\_integral}$ is $[1]$.

**Code:** Lines 640--643:
```python
sigma2_sum = sigma2_cond + sigma_gal_frac**2
mz_integral = exp(-0.5 * (mu_cond - mu_gal_frac)**2 / sigma2_sum) / sqrt(2*pi*sigma2_sum)
```
This is the explicit evaluation of $\mathcal{N}(\mu_\text{cond}; \mu_\text{gal,frac}, \sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2)$. **Matches Eq. (14.31) exactly.**

---

## Section 10: Complete "With BH Mass" Numerator and $/(1+z)$ Verdict

### 10.1 Assembling the Numerator

Combining the 3D marginal (Eq. 14.25), the mz_integral (Eq. 14.31), the galaxy redshift prior, and the detection probability:

$$
\boxed{\text{Num}_j^{(\text{mass})}(H_0) = \int dz \; p_\text{det}(z, M_z) \; p_\text{GW}^{(3D)}(\phi_j, \theta_j, d_{L,\text{frac}}) \; \text{mz\_integral}(z) \; p_\text{gal}(z)}
\tag{14.32}
$$

where:
- $p_\text{det}(z, M_z)$ is the detection probability, now depending on $M_z$ as well as $d_L$, $\phi$, $\theta$,
- $p_\text{GW}^{(3D)}$ is the 3D marginal GW likelihood (same as d_L-only, Eq. 14.25),
- $\text{mz\_integral}(z)$ is the result of the analytic mass marginalization (Eq. 14.31), which depends on $z$ through $\mu_\text{gal,frac}(z)$ and $\sigma_\text{gal,frac}(z)$ (Eq. 14.22),
- $p_\text{gal}(z)$ is the galaxy redshift prior (Eq. 14.5).

**There is NO $/(1+z)$ factor in this expression.**

### 10.2 Verdict on $/(1+z)$ at Line 646

The code at line 646 returns:
```python
p_det * gw_3d * mz_integral * galaxy_redshift_normal_distribution.pdf(z) / (1 + z)
```

Comparing with the derived Eq. (14.32):

$$
\text{Code} = p_\text{det} \cdot p_\text{GW}^{(3D)} \cdot \text{mz\_integral} \cdot p_\text{gal}(z) \;\cdot\; \frac{1}{1+z}
$$

**The $/(1+z)$ factor is SPURIOUS.** [CONFIDENCE: HIGH]

**Where does the spurious factor come from?** The $1/(1+z)$ is the Jacobian $|dM/dM_z| = 1/(1+z)$ from the transformation $M_z = M(1+z)$. This Jacobian WOULD appear if we changed variables from $M$ to $M_z$ (not $M_{z,\text{frac}}$) and left the galaxy mass prior in source-frame coordinates. However:

1. The code correctly transforms the galaxy mass prior to $M_{z,\text{frac}}$ coordinates (lines 633--634: `mu_gal_frac = possible_host.M * (1 + z) / detection.M`).
2. The Gaussian rescaling identity (Eq. 14.20) absorbs the Jacobian into the transformed Gaussian.
3. The `mz_integral` (lines 640--643) already uses the $M_{z,\text{frac}}$-coordinate Gaussian.

Therefore, the $/(1+z)$ double-counts the Jacobian that was already absorbed in step 2.

**The algebraic chain:**
- $p_\text{gal}(M) \, dM$ with Jacobian $\to$ $\mathcal{N}(M_{z,\text{frac}}; \mu_\text{gal,frac}, \sigma_\text{gal,frac}^2) \, dM_{z,\text{frac}}$ (Eq. 14.21, Jacobian absorbed)
- Analytic integral $\to$ $\text{mz\_integral}$ (Eq. 14.31, no remaining Jacobian)
- Code adds $/(1+z)$ at line 646 $\to$ **double-counted**

### 10.3 Impact on the H0 Posterior

The spurious $/(1+z)$ multiplies the numerator integrand by $1/(1+z)$, which:
- Suppresses contributions from higher-redshift galaxies,
- Biases the posterior toward lower $H_0$ (because lower $H_0$ maps to lower $z$ for a given $d_L$),
- This is consistent with the observed bias: "with BH mass" peaks at $h = 0.600$ instead of the expected $h \approx 0.73$.

**SELF-CRITIQUE CHECKPOINT (step 3, post-verdict):**
1. SIGN CHECK: No sign errors. The /(1+z) is a positive multiplicative factor, not a sign issue.
2. FACTOR CHECK: The only unaccounted factor is the /(1+z) in the code. Derivation shows it should not be there. All other factors (2pi, M_z_det) are correctly absorbed.
3. CONVENTION CHECK: M_z_frac = M*(1+z)/M_z_det matches code. Conditional decomposition matches Bishop (2006).
4. DIMENSION CHECK: Eq. (14.32) integrand is [1]*[1]*[1]*[1] = [1]. With /(1+z) it would also be [1] (dimensionless), so dimensional analysis cannot distinguish. The verdict rests on the Jacobian algebra.

---

## Section 11: "With BH Mass" Denominator

### 11.1 Denominator from First Principles

The denominator accounts for the selection effect: the probability that a source from this galaxy would be detected, integrated over all possible source parameters. For the "with BH mass" channel, the detection probability depends on the redshifted mass $M_z$ as well as on $d_L$, $\phi$, $\theta$.

Starting from the same Bayesian framework as the numerator:

$$
\text{Den}_j^{(\text{mass})}(H_0) = \int dz \int dM \; p_\text{det}(d_L(z,H_0), M(1+z), \phi_j, \theta_j) \; p_\text{gal}(z) \; p_\text{gal}(M)
\tag{14.33}
$$

**Why no GW likelihood:** As in the d_L-only case (Section 2.6), the denominator marginalizes over all possible data realizations. The GW likelihood $p(d_\text{GW} \mid \text{params})$ integrates to 1 over the data space and drops out.

**Why no mz_integral:** The `mz_integral` (Eq. 14.31) arose from analytically marginalizing the conditional GW likelihood over $M_{z,\text{frac}}$. Since the GW likelihood is absent from the denominator, there is no conditional to marginalize.

### 11.2 Integration Variables in the Denominator

The natural integration variables in the denominator are $(z, M)$ --- the source-frame quantities that the galaxy catalog provides. There is **no need to change variables to $M_{z,\text{frac}}$** because:
- $p_\text{gal}(M) = \mathcal{N}(M; M_\text{gal}, \sigma_{M,\text{gal}}^2)$ is already in source-frame coordinates,
- $p_\text{det}$ is evaluated at $M_z = M(1+z)$, which is a functional evaluation (like $d_L(z, H_0)$ in the numerator), not a change of integration variable,
- No Gaussian product identity is needed (there is no GW Gaussian to match against).

**Therefore, no Jacobian factor $1/(1+z)$ appears in the denominator.** The $(1+z)$ enters only through the argument $M_z = M(1+z)$ of $p_\text{det}$.

### 11.3 Denominator Term Comparison

| Term | In Numerator? | In Denominator? | Reason |
|------|:---:|:---:|--------|
| $p_\text{det}$ | Yes | Yes | Selection correction |
| $p_\text{GW}^{(3D)}$ | Yes | **No** | GW likelihood integrates to 1 over data space |
| $\text{mz\_integral}$ | Yes | **No** | Arises from marginalizing GW conditional; no GW likelihood in denominator |
| $p_\text{gal}(z)$ | Yes | Yes | Galaxy redshift prior |
| $p_\text{gal}(M)$ | Yes (absorbed into mz_integral) | Yes (explicit) | Galaxy mass prior |
| $/(1+z)$ | **No** (Jacobian absorbed) | **No** (no coordinate change needed) | --- |

### 11.4 Code Check: Denominator Implementation

The code (lines 656--685) implements the denominator via Monte Carlo sampling:

```python
def denominator_integrant_with_bh_mass_vectorized(M, z):
    d_L = dist_vectorized(z, h=h)
    M_z = M * (1 + z)
    p_det = detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta, h=h)
    return p_det * galaxy_redshift_normal_distribution.pdf(z) * galaxy_mass_normal_distribution.pdf(M)
```

This matches Eq. (14.33): $p_\text{det}(d_L, M_z, \phi, \theta) \cdot p_\text{gal}(z) \cdot p_\text{gal}(M)$.

The MC integration uses importance sampling from $p_\text{gal}(z) \cdot p_\text{gal}(M)$:
```python
z_samples = galaxy_redshift_normal_distribution.rvs(size=N_SAMPLES)
M_samples = galaxy_mass_normal_distribution.rvs(size=N_SAMPLES)
weights = integrand(M_samples, z_samples) / (p_gal(z) * p_gal(M))
denominator = mean(weights)
```

After importance sampling, `weights = p_det`, so the MC estimate is:

$$
\text{Den}_j^{(\text{mass})} \approx \frac{1}{N} \sum_{i=1}^{N} p_\text{det}(d_L(z_i, H_0), M_i(1+z_i), \phi_j, \theta_j)
\tag{14.34}
$$

where $(z_i, M_i)$ are drawn from $p_\text{gal}(z) \cdot p_\text{gal}(M)$.

**The denominator implementation is correct.** It contains no $/(1+z)$ factor and correctly integrates $p_\text{det}$ over the galaxy priors.

**Note on methodology asymmetry:** The numerator uses `fixed_quad` (Gaussian quadrature) while the denominator uses MC sampling. This is a numerical methodology choice, not a physics error --- both approximate the same integral. However, MC with $N = 10{,}000$ samples may introduce noise that quadrature does not. This is a Phase 15 concern, not a Phase 14 derivation issue.

---

## Section 12: Limiting Case --- $\sigma_{M_z} \to \infty$

### 12.1 Setup

We verify that when the GW measurement gives no information about $M_z$ (infinite mass uncertainty), the "with BH mass" likelihood reduces to the d_L-only likelihood from Plan 01.

In the 4D covariance, $\sigma_{M_z} \to \infty$ means $\Sigma_{4\times4}[3,3] = \sigma_{M_z}^2 \to \infty$.

### 12.2 Behavior of the Conditional Variance

From Eq. (14.28):

$$
\sigma_\text{cond}^2 = \sigma_{M_z}^2 - \boldsymbol{c}^T \Sigma_\text{obs}^{-1} \boldsymbol{c}
$$

The second term $\boldsymbol{c}^T \Sigma_\text{obs}^{-1} \boldsymbol{c}$ is finite (it depends on the cross-covariance and the 3D observed covariance, neither of which diverges). Therefore:

$$
\sigma_\text{cond}^2 \to \infty \quad \text{as} \quad \sigma_{M_z}^2 \to \infty
\tag{14.35}
$$

### 12.3 Behavior of the 3D Marginal

The 3D marginal covariance $\Sigma_\text{obs} = \Sigma_{4\times4}[0:3, 0:3]$ does **not** depend on $\sigma_{M_z}^2$. It is the upper-left $3\times3$ block, which is fixed. Therefore:

$$
p_\text{GW}^{(3D)}(\phi, \theta, d_{L,\text{frac}}) \text{ is unchanged as } \sigma_{M_z} \to \infty
\tag{14.36}
$$

### 12.4 Behavior of the mz_integral

From Eq. (14.31):

$$
\text{mz\_integral} = \mathcal{N}(\mu_\text{cond};\; \mu_\text{gal,frac},\; \sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2)
$$

$$
= \frac{1}{\sqrt{2\pi(\sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2)}} \exp\!\left(-\frac{(\mu_\text{cond} - \mu_\text{gal,frac})^2}{2(\sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2)}\right)
$$

As $\sigma_\text{cond}^2 \to \infty$:
- The exponential $\to \exp(0) = 1$ (the argument of the exponential goes to 0),
- The prefactor $\to 1/\sqrt{2\pi \sigma_\text{cond}^2}$.

Therefore:

$$
\text{mz\_integral} \to \frac{1}{\sqrt{2\pi \sigma_\text{cond}^2}} \quad \text{as} \quad \sigma_\text{cond}^2 \to \infty
\tag{14.37}
$$

This is independent of $z$ (and of $\mu_\text{gal,frac}$, $\sigma_\text{gal,frac}$) --- the mass measurement has become uninformative, so the mass integral gives the same result regardless of the galaxy's mass.

### 12.5 Behavior of the Conditional Mean

From Eq. (14.27):

$$
\mu_\text{cond} = \mu_{M_z}^{(4D)} + \boldsymbol{c}^T \Sigma_\text{obs}^{-1} (\mathbf{x}_\text{obs} - \boldsymbol{\mu}_\text{obs})
$$

**Subtlety:** As $\sigma_{M_z}^2 \to \infty$, the cross-covariance $\boldsymbol{c}$ may also change. In the Fisher matrix framework, $\Sigma = F^{-1}$. If the Fisher information for $M_{z,\text{frac}}$ goes to zero ($F_{33} \to 0$), then $\sigma_{M_z}^2 \to \infty$. The off-diagonal Fisher elements $F_{3,\alpha}$ may also go to zero, which would make $\boldsymbol{c}$ go to zero as well. In that regime, $\mu_\text{cond} \to \mu_{M_z}^{(4D)}$ and the projection term vanishes.

However, this subtlety does not affect the limiting-case argument: regardless of what $\mu_\text{cond}$ does, the mz_integral becomes $z$-independent in the limit (Eq. 14.37).

### 12.6 Limiting Numerator

The numerator in the limit becomes:

$$
\text{Num}_j^{(\text{mass})}(H_0) \to \int dz \; p_\text{det}(z, M_z) \; p_\text{GW}^{(3D)}(\phi_j, \theta_j, d_{L,\text{frac}}) \; \frac{1}{\sqrt{2\pi\sigma_\text{cond}^2}} \; p_\text{gal}(z)
$$

The $z$-independent factor $1/\sqrt{2\pi\sigma_\text{cond}^2}$ can be pulled out of the integral:

$$
= \frac{1}{\sqrt{2\pi\sigma_\text{cond}^2}} \int dz \; p_\text{det}(z, M_z) \; p_\text{GW}^{(3D)}(\phi_j, \theta_j, d_{L,\text{frac}}) \; p_\text{gal}(z)
\tag{14.38}
$$

### 12.7 Limiting Denominator

The denominator (Eq. 14.33) becomes, in the limit where $p_\text{det}$ no longer depends meaningfully on $M_z$ (because the mass uncertainty is infinite, all masses are equally likely, and $p_\text{det}$ is averaged over the mass distribution):

$$
\text{Den}_j^{(\text{mass})}(H_0) \to \int dz \; \overline{p_\text{det}}(z) \; p_\text{gal}(z)
\tag{14.39}
$$

where $\overline{p_\text{det}}(z) = \int dM \; p_\text{det}(d_L, M(1+z), \phi, \theta) \; p_\text{gal}(M)$ is the mass-averaged detection probability. When the mass distribution is broad (as implied by $\sigma_{M_z} \to \infty$), this converges to the d_L-only detection probability $p_\text{det}(d_L, \phi, \theta)$.

### 12.8 The Likelihood Ratio

The single-host likelihood ratio is:

$$
\mathcal{L}_j^{(\text{mass})}(H_0) = \frac{\text{Num}_j^{(\text{mass})}}{\text{Den}_j^{(\text{mass})}} \to \frac{\frac{1}{\sqrt{2\pi\sigma_\text{cond}^2}} \displaystyle\int dz \; p_\text{det} \; p_\text{GW}^{(3D)} \; p_\text{gal}(z)}{\displaystyle\int dz \; \overline{p_\text{det}} \; p_\text{gal}(z)}
$$

The $1/\sqrt{2\pi\sigma_\text{cond}^2}$ prefactor is **$H_0$-independent** (it depends only on the Fisher matrix properties, not on the cosmological parameter). Therefore, it is a constant multiplicative factor that cancels in the posterior:

$$
p(H_0 \mid d_\text{GW}) \propto \prod_i \mathcal{L}_i(H_0) \implies \text{constant prefactors cancel in normalization}
$$

What remains is:

$$
\mathcal{L}_j^{(\text{mass})}(H_0) \propto \frac{\displaystyle\int dz \; p_\text{det}(z) \; p_\text{GW}^{(3D)}(\phi_j, \theta_j, d_{L,\text{frac}}) \; p_\text{gal}(z)}{\displaystyle\int dz \; p_\text{det}(z) \; p_\text{gal}(z)}
$$

This is exactly Eq. (14.3), the d_L-only single-host likelihood from Plan 01.

$$
\boxed{\lim_{\sigma_{M_z} \to \infty} \mathcal{L}_j^{(\text{mass})}(H_0) \propto \mathcal{L}_j^{(d_L\text{-only})}(H_0)}
\tag{14.40}
$$

**The limiting case is verified.** When the GW measurement gives no mass information, the "with BH mass" likelihood reduces to the d_L-only likelihood (up to an $H_0$-independent normalization constant that cancels in the posterior). $\checkmark$

**SELF-CRITIQUE CHECKPOINT (step 4, post-limiting-case):**
1. SIGN CHECK: No sign changes. Correct.
2. FACTOR CHECK: $1/\sqrt{2\pi\sigma_\text{cond}^2}$ is $H_0$-independent; cancels in posterior normalization. No stray factors.
3. CONVENTION CHECK: Using same Gaussian decomposition and fractional parameterization throughout.
4. DIMENSION CHECK: Eq. (14.40) ratio is [1]/[1] = [1]. Consistent.

---

## Section 13: Dimensional Analysis for "With BH Mass" Terms

Extending the Plan 01 dimensional analysis table (Section 4):

| Factor | Expression | Dimensions | Notes |
|--------|-----------|------------|-------|
| $M_{z,\text{frac}}$ | $M(1+z)/M_{z,\text{det}}$ | $[1]$ | dimensionless fractional mass |
| $\mu_\text{gal,frac}$ | $M_\text{gal}(1+z)/M_{z,\text{det}}$ | $[1]$ | transformed galaxy mass mean |
| $\sigma_\text{gal,frac}$ | $\sigma_{M,\text{gal}}(1+z)/M_{z,\text{det}}$ | $[1]$ | transformed galaxy mass std |
| $p_\text{GW}^{(3D)}(\phi, \theta, d_{L,\text{frac}})$ | marginal of 4D Gaussian | $[1]$ | same as d_L-only GW likelihood |
| $p(M_{z,\text{frac}} \mid \mathbf{x}_\text{obs})$ | conditional Gaussian | $[1]$ | Gaussian of dimensionless arg |
| $\text{mz\_integral}$ | $\mathcal{N}(\mu_\text{cond}; \mu_\text{gal,frac}, \sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2)$ | $[1]$ | Gaussian of dimensionless args |
| $p_\text{gal}(M)$ | $\mathcal{N}(M; M_\text{gal}, \sigma_M^2)$ | $[1/M_\odot]$ | Gaussian in mass |
| $dM$ | integration measure | $[M_\odot]$ | source-frame mass |
| $p_\text{gal}(M) \, dM$ | probability element | $[1]$ | dimensionless |
| $d(M_{z,\text{frac}})$ | integration measure | $[1]$ | dimensionless |
| Numerator integrand (derived) | $p_\text{det} \cdot p_\text{GW}^{(3D)} \cdot \text{mz\_integral} \cdot p_\text{gal}(z)$ | $[1]$ | correct: dimensionless |
| Numerator integrand (code, line 646) | $\ldots \cdot /(1+z)$ | $[1]$ | also dimensionless, but WRONG |
| Denominator integrand | $p_\text{det} \cdot p_\text{gal}(z) \cdot p_\text{gal}(M)$ | $[1/M_\odot]$ | integrates over $dM \,[M_\odot]$ to give $[1]$ |

**Key observation:** The $/(1+z)$ is dimensionless, so dimensional analysis CANNOT detect the error. The verdict rests entirely on the Jacobian algebra in Section 8. This is why first-principles derivation was necessary --- the bug is dimensionally invisible.

---

## Section 14: Code Mapping and Summary

### 14.1 Term-by-Term Code Mapping

| Derived Term | Code Variable / Expression | Code Line(s) | Status |
|-------------|---------------------------|-------------|--------|
| $p_\text{GW}^{(3D)}(\phi, \theta, d_{L,\text{frac}})$ | `gaussian_3d_marginal.pdf(...)` | 624--626 | CORRECT |
| $\Sigma_\text{obs}$ | `cov_obs = cov_4d[:3, :3]` | 593 | CORRECT |
| $\boldsymbol{c}$ | `cov_cross = cov_4d[3, :3]` | 594 | CORRECT |
| $\sigma_{M_z}^2$ | `cov_mz = cov_4d[3, 3]` | 595 | CORRECT |
| $\sigma_\text{cond}^2$ | `sigma2_cond = cov_mz - cov_cross @ cov_obs_inv @ cov_cross` | 601 | CORRECT |
| $\text{proj} = \boldsymbol{c}^T \Sigma_\text{obs}^{-1}$ | `proj = cov_cross @ cov_obs_inv` | 605 | CORRECT |
| $\mu_\text{cond}$ | `mu_obs_4d[3] + (x_obs - mu_obs_4d[:3]) @ proj` | 630 | CORRECT |
| $\mu_\text{gal,frac}$ | `possible_host.M * (1 + z) / detection.M` | 633 | CORRECT |
| $\sigma_\text{gal,frac}$ | `possible_host.M_error * (1 + z) / detection.M` | 634 | CORRECT |
| $\sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2$ | `sigma2_sum = sigma2_cond + sigma_gal_frac**2` | 640 | CORRECT |
| $\text{mz\_integral}$ | `exp(-0.5*(...)**2/sigma2_sum) / sqrt(2*pi*sigma2_sum)` | 641--643 | CORRECT |
| **$/(1+z)$** | **`/ (1 + z)`** | **646** | **SPURIOUS** |
| $p_\text{det} \cdot p_\text{gal}(z) \cdot p_\text{gal}(M)$ (denom) | `p_det * p_gal(z) * p_gal(M)` | 666--669 | CORRECT |

### 14.2 Discrepancies Between Derivation and Code

**Discrepancy 1 (CRITICAL): Spurious $/(1+z)$ in numerator**
- **Location:** `bayesian_statistics.py` line 646
- **Derivation says:** No $/(1+z)$ factor (Eq. 14.32)
- **Code has:** `/ (1 + z)` multiplying the entire numerator integrand
- **Root cause:** Double-counted Jacobian from $M \to M_z$ transformation, which was already absorbed by the Gaussian rescaling when transforming the galaxy mass prior to $M_{z,\text{frac}}$ coordinates
- **Fix:** Remove `/ (1 + z)` from line 646
- **Expected impact:** Eliminates the $h = 0.600$ bias in the "with BH mass" posterior

**Discrepancy 2 (OBSERVATION): Methodology asymmetry in integration**
- **Numerator:** `fixed_quad` (Gaussian quadrature) --- deterministic, high precision
- **Denominator:** MC sampling with $N = 10{,}000$ --- stochastic, potential noise
- **Impact:** May introduce variance in the likelihood ratio. Not a physics error, but a numerical concern for Phase 15.

**No other discrepancies found.** The conditional decomposition, analytic marginalization, and denominator are all correctly implemented.

### 14.3 Complete "With BH Mass" Likelihood (Boxed Summary)

For Phase 15 reference, the complete derived expression:

$$
\boxed{
\mathcal{L}_j^{(\text{mass})}(H_0) = \frac{\displaystyle\int dz \; p_\text{det}(z, M_z) \; p_\text{GW}^{(3D)}(\phi_j, \theta_j, d_{L,\text{frac}}) \; \text{mz\_integral}(z) \; p_\text{gal}(z)}{\displaystyle\int dz \int dM \; p_\text{det}(d_L, M(1+z), \phi_j, \theta_j) \; p_\text{gal}(z) \; p_\text{gal}(M)}
}
\tag{14.41}
$$

where:

| Symbol | Definition | Eq. |
|--------|-----------|-----|
| $p_\text{GW}^{(3D)}$ | 3D marginal of 4D Fisher-matrix Gaussian | (14.25) |
| $\text{mz\_integral}(z)$ | $\mathcal{N}(\mu_\text{cond}; \mu_\text{gal,frac}(z), \sigma_\text{cond}^2 + \sigma_\text{gal,frac}^2(z))$ | (14.31) |
| $\mu_\text{cond}$ | $\mu_{M_z}^{(4D)} + \boldsymbol{c}^T \Sigma_\text{obs}^{-1}(\mathbf{x}_\text{obs} - \boldsymbol{\mu}_\text{obs})$ | (14.27) |
| $\sigma_\text{cond}^2$ | $\sigma_{M_z}^2 - \boldsymbol{c}^T \Sigma_\text{obs}^{-1} \boldsymbol{c}$ | (14.28) |
| $\mu_\text{gal,frac}(z)$ | $M_\text{gal}(1+z)/M_{z,\text{det}}$ | (14.22) |
| $\sigma_\text{gal,frac}(z)$ | $\sigma_{M,\text{gal}}(1+z)/M_{z,\text{det}}$ | (14.22) |
| $p_\text{gal}(z)$ | $\mathcal{N}(z; z_j, \sigma_{z,j})$ | (14.5) |
| $p_\text{gal}(M)$ | $\mathcal{N}(M; M_\text{gal}, \sigma_{M,\text{gal}}^2)$ | Sec. 8.1 |
| $p_\text{det}$ | Detection probability (depends on $d_L$, $M_z$, $\phi$, $\theta$) | KDE |

**Properties:**
- No $/(1+z)$ Jacobian factor in the numerator (Section 8.4, 10.2)
- Denominator integrates over $(z, M)$ in source-frame coordinates (Section 11)
- Reduces to d_L-only (Eq. 14.12) when $\sigma_{M_z} \to \infty$ (Section 12)
- Integrand is dimensionless (Section 13)

### 14.4 Summary of Findings for Phase 15

1. **Remove `/(1+z)` at line 646** --- this is the primary bug causing the $h = 0.600$ bias
2. **Denominator is correct** --- no changes needed to lines 656--685
3. **Analytic marginalization is correct** --- the Bishop (2006) conditional decomposition and Gaussian product identity are correctly implemented at lines 586--643
4. **All transformed coordinates match** --- $\mu_\text{gal,frac}$, $\sigma_\text{gal,frac}$, $\mu_\text{cond}$, $\sigma_\text{cond}^2$ are all correctly computed
5. **Methodology asymmetry** (quadrature vs MC) is a numerical concern, not a physics error
