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

**Starting point for Plan 02:** The "with BH mass" extension adds a fourth parameter $M_{z,\text{frac}}$ to the GW likelihood Gaussian (making it 4D) and introduces an additional galaxy mass prior $p_\text{mass}(M)$ with a corresponding integral or analytic marginalization. The key question for Plan 02 is how the $M_z \to M$ Jacobian $(1+z)$ enters and whether the denominator is consistent.
