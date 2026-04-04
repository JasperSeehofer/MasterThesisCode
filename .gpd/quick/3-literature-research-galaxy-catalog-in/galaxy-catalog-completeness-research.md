# Galaxy Catalog Completeness Correction for Dark Siren H0 Inference

## Literature Research Report

**Context:** Our bias investigation (scripts/bias_investigation/FINDINGS.md) identified GLADE
catalog incompleteness at z > 0.08 as the root cause of the H0 posterior bias (MAP = 0.66 vs
true h = 0.73, with 534 detections). This document surveys the standard methods for correcting
this bias.

**Primary references:**
- Gray et al. (2020), arXiv:1908.06050 — "Cosmological Inference using Gravitational Wave
  Standard Sirens: A Mock Data Analysis" [ref-gray2020]
- Gray et al. (2022), arXiv:2111.04629 — "A Pixelated Approach to Galaxy Catalogue
  Incompleteness" [ref-gray2022]
- Finke et al. (2021), arXiv:2101.12660 — "Cosmology with LIGO/Virgo dark sirens:
  Hubble parameter and modified gravitational wave propagation" [ref-finke2021]
- Dalya et al. (2022), arXiv:2110.06184 — "GLADE+: An Extended Galaxy Catalogue for
  Multimessenger Searches with Advanced Gravitational-Wave Detectors" [ref-dalya2022]
- Laghi et al. (2021), arXiv:2102.01708 — "Gravitational wave cosmology with extreme
  mass-ratio inspirals" [ref-laghi2021]

---

## 1. The Standard Dark Siren Likelihood Formula

### 1.1 Starting Point: Our Current Implementation

Our current code in `bayesian_statistics.py:p_Di()` (lines 352-460) computes the per-event
likelihood as:

```
p(x_GW | D_GW, H0) = sum_j numerator_j / sum_j denominator_j
```

where the sum runs over all galaxies j in the GLADE catalog within the GW error volume, and:

- **numerator_j** = integral over z of: p_det(d_L(z,H0), phi_j, theta_j) *
  p_GW(phi_j, theta_j, d_L(z,H0)/d_L_det) * p_gal(z | z_j, sigma_z_j) dz

- **denominator_j** = integral over z of: p_det(d_L(z,H0), phi_j, theta_j) *
  p_gal(z | z_j, sigma_z_j) dz

This corresponds to the **complete catalog** case: we implicitly assume that the true host
galaxy is always in the catalog. As demonstrated by our bias investigation, this assumption
fails for events at z > 0.08 where GLADE is incomplete.

### 1.2 The Completeness-Corrected Likelihood (Gray et al. 2020)

The standard framework was established by Gray et al. (2020), arXiv:1908.06050, Sec. II.3.1
and Appendix A.2. The key insight is to marginalize over whether the host galaxy is in the
catalog (G) or not (G-bar):

**Equation (9) from Gray et al. (2020):**

```
p(x_GW | D_GW, H0) = p(x_GW | G, D_GW, H0) * p(G | D_GW, H0)
                    + p(x_GW | G-bar, D_GW, H0) * p(G-bar | D_GW, H0)
```

This decomposes the likelihood into two terms:

1. **Catalog term** (host IS in catalog): weighted by the probability that the host is
   cataloged, p(G | D_GW, H0)
2. **Completion term** (host is NOT in catalog): weighted by the complementary probability,
   p(G-bar | D_GW, H0) = 1 - p(G | D_GW, H0)

#### 1.2.1 The Catalog Term: p(x_GW | G, D_GW, H0)

From Gray et al. (2020) Appendix A.2, Eqs. (24)-(25), the catalog term when redshift
uncertainties are included is:

```
p(x_GW | G, D_GW, H0) =

    sum_{j=1}^{N_gal} integral[ p(x_GW | z_j, Omega_j, H0) * p(s|z_j) * p(s|M_j(H0)) * p(z_j) ] dz_j
    -----------------------------------------------------------------------------------------
    sum_{j=1}^{N_gal} integral[ p(D_GW | z_j, Omega_j, H0) * p(s|z_j) * p(s|M_j(H0)) * p(z_j) ] dz_j
```

**Term-by-term:**
- **p(x_GW | z_j, Omega_j, H0):** GW likelihood — probability of observing the GW data
  given a source at redshift z_j, sky position Omega_j, and cosmology H0. In our code, this
  is the multivariate Gaussian evaluated at (phi_j, theta_j, d_L(z,H0)/d_L_det).
- **p(D_GW | z_j, Omega_j, H0):** GW detection probability — our P_det interpolator.
- **p(s|z_j):** Rate evolution factor — probability that a GW source exists at redshift z_j.
  For z << 1, typically taken as constant. For higher z, can use p(s|z) proportional to
  (1+z)^lambda.
- **p(s|M_j(H0)):** Luminosity weighting — probability that a galaxy with absolute magnitude
  M_j hosts a GW source. Can be set to: (a) constant (equal weights), or (b) proportional to
  L(M_j) (luminosity weighting). Gray et al. (2020) Eq. (18).
- **p(z_j):** Galaxy redshift prior — the Gaussian uncertainty on the galaxy's redshift.

**This is exactly what our current code computes** (with p(s|z) = const, p(s|M) = const, and
the sum/integral structure in `single_host_likelihood` and `p_Di`). So our catalog term is
correct.

#### 1.2.2 The Completion Term: p(x_GW | G-bar, D_GW, H0)

From Gray et al. (2020) Appendix A.2.3, Eqs. (31)-(32), the completion term assumes that
for uncataloged galaxies, the prior on (z, Omega) is uniform in comoving volume:

```
p(x_GW | G-bar, D_GW, H0) =

    integral[ p(x_GW | z, Omega, H0) * p(s|z) * p(z) * p(Omega) * integral[p(s|M,H0) * p(M|H0) dM] ] dz dOmega
    ----------------------------------------------------------------------------------------------------------
    integral[ p(D_GW | z, Omega, H0) * p(s|z) * p(z) * p(Omega) * integral[p(s|M,H0) * p(M|H0) dM] ] dz dOmega
```

where:
- **p(z)** is uniform in comoving volume: p(z) proportional to d_com^2(z) * c/H(z)
  (the comoving volume element dV_c/dz per steradian)
- **p(Omega)** is uniform on the sky: p(Omega) = 1/(4*pi)
- **p(M|H0)** is the Schechter luminosity function
- The inner integral over M can be absorbed into a normalization when p(s|M) = const

In practice, for the completion term the GW likelihood p(x_GW|z, Omega, H0) is evaluated
using the same Gaussian measurement model, but integrated over a smooth galaxy density
rather than discrete galaxy positions.

#### 1.2.3 The Completeness Weights: p(G | D_GW, H0)

From Gray et al. (2020) Appendix A.2.2, Eq. (29):

```
p(G | D_GW, H0) =

    integral_0^{z(M,m_th,H0)} dz integral dOmega integral dM  p(D_GW|z,Omega,H0) p(s|z) p(z) p(Omega) p(s|M,H0) p(M|H0)
    --------------------------------------------------------------------------------------------------------------------------
    integral_0^{z_max} dz integral dOmega integral dM  p(D_GW|z,Omega,H0) p(s|z) p(z) p(Omega) p(s|M,H0) p(M|H0)
```

where z(M, m_th, H0) is the maximum redshift at which a galaxy of absolute magnitude M is
visible given the apparent magnitude threshold m_th. The numerator integrates only over the
cataloged region (m < m_th), while the denominator integrates over all possible sources.

And p(G-bar | D_GW, H0) = 1 - p(G | D_GW, H0)   [Eq. (30)].

### 1.3 Limiting Cases

These limiting cases serve as correctness checks for any implementation:

1. **f_complete = 1 everywhere (complete catalog):**
   p(G | D_GW, H0) = 1, so the completion term vanishes.
   The full likelihood reduces to the catalog-only term — **this recovers our current code**.

2. **f_complete = 0 everywhere (empty catalog):**
   p(G | D_GW, H0) = 0, so the catalog term vanishes.
   The full likelihood is entirely the completion term — a uniform-in-comoving-volume prior.
   This is the "statistical" or "spectral siren" method (no catalog information at all).

3. **Intermediate completeness:**
   Both terms contribute. In regions where the catalog is more complete, the discrete galaxy
   sum dominates and provides tighter constraints. In regions where it is incomplete, the
   smooth completion term fills in, preventing the bias we observe.

**Why our bias occurs:** Our current code implicitly sets p(G | D_GW, H0) = 1 (complete
catalog assumption). For events at z = 0.08-0.13, the actual completeness is much less than
1. The catalog term sum over galaxies at z < z_true is inflated (more galaxies at low z
than high z in the catalog), and there is no completion term to compensate. This creates
a systematic preference for H0 values that map the measured d_L to lower redshifts where
more catalog galaxies exist — exactly the bias from MAP=0.66 < true h=0.73 that we observe.

### 1.4 Simplified Form Used in Practice

Many implementations (including the original gwcosmo) use a simplified form where the
completeness fraction f(Omega, z) is precomputed. The Finke et al. (2021) formulation
(arXiv:2101.12660, Sec. 3.2) makes this particularly clear:

```
p_0(z, Omega) = f(Omega, z) * p_cat(z, Omega) + (1 - f(Omega, z)) * p_miss(z, Omega)
```

where:
- **p_cat(z, Omega):** the normalized galaxy catalog density (sum of delta functions or
  Gaussians at galaxy positions)
- **p_miss(z, Omega):** the density of missing galaxies, typically assumed uniform in
  comoving volume (after marginalizing over the Schechter function)
- **f(Omega, z):** the completeness fraction — the fraction of total galaxy weight
  (number or luminosity) captured by the catalog at a given sky position and redshift

This prior p_0(z, Omega) then enters the dark siren likelihood:

```
p(x_GW | D_GW, H0) = integral[ p(x_GW | z, Omega, H0) * p_0(z, Omega) ] dz dOmega
                      -----------------------------------------------------------------
                      integral[ p(D_GW | z, Omega, H0) * p_0(z, Omega) ] dz dOmega
```

This is mathematically equivalent to the Gray et al. formulation but computationally
simpler, because the completeness weighting is absorbed into the prior rather than
appearing as a separate marginalization over G/G-bar.

---

## 2. GLADE+ Catalog Completeness

### 2.1 GLADE+ Completeness Characterization (Dalya et al. 2022)

From Dalya et al. (2022), arXiv:2110.06184, Section 3 ("Catalogue Completeness"):

**Method 1: Integrated B-band luminosity comparison**

GLADE+ completeness is quantified by comparing the cumulative B-band luminosity of GLADE+
galaxies within different d_L shells to the expected total luminosity assuming a
homogeneous galaxy distribution with B-band luminosity density:

  j_B = (1.98 +/- 0.16) x 10^{-2} L_10 Mpc^{-3}

where L_10 = 10^{10} L_{B,sun}.

**Key completeness figures from the GLADE+ paper:**

| Distance range | Completeness (B-band luminosity) | Notes |
|---|---|---|
| d_L < 47 (+4/-2) Mpc (z < ~0.011) | ~100% | Complete in total B-band luminosity |
| d_L < 130 Mpc (z < ~0.029) | ~90% (brightest galaxies) | Contains galaxies giving 90% of B and K band luminosity |
| d_L ~ 200-300 Mpc (z ~ 0.045-0.067) | ~50-80% (estimated) | Main contributions from 2MPZ and WISExSCOSPZ |
| d_L ~ 330 Mpc (z ~ 0.074) | Drops significantly | Completeness from WISExSCOSPZ begins to dominate |
| d_L > 500 Mpc (z > 0.11) | << 50% | Highly incomplete; most fainter galaxies missing |

**Method 2: Schechter function comparison**

The completeness is also assessed by comparing the luminosity distribution of GLADE+ galaxies
in d_L shells against the expected Schechter luminosity function. At larger distances, only
the bright end of the Schechter function is sampled (faint galaxies fall below the flux limit),
allowing an estimate of what fraction of the total luminosity is missing.

### 2.2 Completeness in Our Redshift Range

Our EMRI detections span z = 0.03-0.20 (d_L ~ 130-900 Mpc). The GLADE+ completeness in
this range is:

- **z < 0.03 (d_L < 130 Mpc):** ~90% luminosity completeness. The catalog term dominates;
  completion term is a small correction.
- **z = 0.03-0.08 (d_L = 130-350 Mpc):** 50-90%, declining with distance. Both terms
  contribute significantly.
- **z = 0.08-0.13 (d_L = 350-580 Mpc):** << 50%. The completion term should dominate.
  This is exactly the range where 67% of our detections lie and where the bias is worst.
- **z > 0.13 (d_L > 580 Mpc):** Very low completeness. The analysis should be almost
  entirely driven by the completion term (uniform-in-comoving-volume assumption).

### 2.3 Estimating f(z) for GLADE+ in Practice

There are several approaches to estimate the completeness fraction:

**Approach A: B-band luminosity fraction (global)**

Use the GLADE+ data directly. For each redshift shell [z, z+dz]:
1. Sum the B-band luminosities of all GLADE+ galaxies in the shell
2. Compare to the expected total B-band luminosity:
   L_expected(z, dz) = j_B * dV_c(z, dz) where dV_c is the comoving volume of the shell
3. f(z) = L_observed(z) / L_expected(z)

This is the method used by Dalya et al. (2022) in their Figure 2.

**Approach B: Schechter function extrapolation (per line-of-sight)**

Used by Gray et al. (2022), arXiv:2111.04629:
1. For each HEALPix pixel, fit the apparent magnitude distribution of GLADE+ galaxies
2. Compare to the expected distribution from the Schechter function
3. The ratio gives a direction-dependent completeness f(z, Omega)

**Approach C: Number density comparison**

Used by Finke et al. (2021), arXiv:2101.12660, Sec. 3.2:
1. Estimate the total comoving number density of galaxies n_gal ~ 0.1-0.2 Mpc^{-3}
   (from Conselice et al. 2016 and similar surveys)
2. Count observed galaxies in each volume element
3. f(z, Omega) = n_cat(z, Omega) / n_gal (for number weighting)
4. For luminosity weighting: use the Schechter function integral above the magnitude limit

**For our implementation (Approach A is recommended):**

The simplest approach that addresses our specific bias is a redshift-dependent (but
angle-averaged) completeness f(z). This is justified for LISA EMRI because:
- LISA has much better sky localization (~1 deg^2) than LIGO (~100 deg^2), so the GW
  error box is small and angular variation of completeness within it is small
- The dominant incompleteness effect is redshift-dependent (flux limit), not directional
- The angular dependence is a second-order correction that can be added later if needed

### 2.4 Angular Dependence of GLADE+ Completeness

GLADE+ is constructed from multiple sub-catalogs with different sky coverage:
- **2MPZ:** Near-infrared selected, nearly all-sky, photometric redshifts, complete to K~13.9
- **WISExSCOSPZ:** Mid-infrared + optical, dec > -70 deg, photometric redshifts
- **HyperLEDA:** Optical, heterogeneous depth across the sky
- **SDSS-DR16Q:** Quasars only, limited to SDSS footprint (~10,000 deg^2)

This means completeness varies across the sky, particularly at d_L > 200 Mpc. The
Gray et al. (2022) pixelated approach addresses this by computing completeness in each
HEALPix pixel independently.

For LISA EMRI with ~1 deg^2 sky localization, the angular variation within one error box
is negligible. However, different events at different sky positions will have different
completeness values — this should be accounted for by using direction-dependent completeness
when computing the completion term for each event.

---

## 3. Implementation Approaches in the Literature

### 3.1 gwcosmo (Gray et al. 2020, 2022)

The official LVK tool for dark siren cosmology. Key implementation choices:

- **Completeness:** Precomputed as a function of d_L (or equivalently z, given H0) using
  the Schechter function and the catalog's apparent magnitude threshold m_th
- **Pixelation:** Gray et al. (2022) introduced HEALPix-based pixelation where completeness
  is computed per pixel. This gives ~5% improvement in H0 constraints over the uniform-
  completeness approach
- **Luminosity weighting:** Galaxies are weighted by their B-band or K-band luminosity,
  which increases the effective completeness (bright galaxies, which are more likely hosts
  of massive BH mergers, are more complete in the catalog)
- **Completion term:** Uses uniform-in-comoving-volume assumption for uncataloged galaxies,
  with the Schechter function integrated over the magnitude range below the catalog threshold

### 3.2 DarkSirensStat (Finke et al. 2021)

From arXiv:2101.12660. Key innovations:

- **"Multiplicative completion":** Rather than treating the catalog and completion terms
  separately, they use a prior p_0 that is a weighted sum:
  ```
  p_0(z, Omega) = f_R * p_cat(z, Omega) + (1 - f_R) * p_miss(z, Omega)
  ```
  where f_R is the completeness fraction and p_miss is estimated from the comoving number
  density of galaxies (n_gal ~ 0.1-0.2 Mpc^{-3}).

- **Direction-dependent completeness:** Computed per sky region using the ratio of observed
  to expected galaxy number/luminosity density

- **Luminosity weighting with lower cut:** They apply a lower luminosity cut L_min when
  computing both the catalog weights and the completeness. This has the effect of increasing
  the effective completeness by removing faint galaxies from both the "expected" and
  "observed" counts.

- **Public code:** Available at https://github.com/CosmoStatGW/DarkSirensStat

### 3.3 LISA EMRI-Specific Considerations (Laghi et al. 2021)

From arXiv:2102.01708:

- Laghi et al. explicitly state they did **not** account for catalog incompleteness in their
  EMRI dark siren analysis (Sec. 3.3): "we do not fold into our analysis the possible
  incompleteness of the galaxy catalogs"
- They assume all GW events are hosted by galaxies within the reconstructed comoving volume,
  regardless of whether the galaxy is luminous enough to be listed in the catalog
- In their conclusions (Sec. 6), they note: "Another improvement would be to account for
  incompleteness of the galaxy catalogue. [...] Incompleteness can be accounted for in the
  analysis, for example by weighting galaxies in the catalogue by the number of nearby
  galaxies that are missing, or by adding an appropriate number of missing galaxies into
  the assumed redshift distribution."

This confirms that incompleteness correction is a recognized open problem for EMRI
cosmology, and no published LISA EMRI analysis has yet implemented it.

### 3.4 LIGO vs LISA: Key Differences

| Aspect | LIGO BBH/BNS | LISA EMRI |
|---|---|---|
| Sky localization | ~10-1000 deg^2 | ~0.01-1 deg^2 |
| Distance range | d_L < 1-5 Gpc | d_L < ~1 Gpc (z < ~0.2 for our setup) |
| N galaxies in error volume | ~10^3 - 10^6 | ~10 - 10^3 |
| Catalog completeness at typical distances | << 10% (most events) | 10-90% (varies strongly with z) |
| Angular completeness variation within error box | Important (large boxes) | Negligible (small boxes) |
| Dominant completeness issue | Most hosts uncataloged | Transition from cataloged to uncataloged around z~0.08 |

**Key implication for our implementation:** Because LISA EMRI have excellent sky
localization, the angular dependence of completeness can be approximated as constant
within each event's error box. This simplifies the implementation considerably compared
to LIGO analyses where the angular integral over a large sky patch is essential.

---

## 4. Mathematical Specification for Our Codebase

### 4.1 Current Code Structure

The relevant code flow is:

```
BayesianStatistics.evaluate()           [bayesian_statistics.py:119]
  -> BayesianStatistics.p_D()           [bayesian_statistics.py:273]
    -> BayesianStatistics.p_Di()        [bayesian_statistics.py:352]
      -> (parallel) single_host_likelihood()  [bayesian_statistics.py:500]
```

In `p_Di()`, the per-event likelihood is computed as:

```python
# Current: sum numerators / sum denominators (no completeness correction)
likelihood = sum(numerator_j) / sum(denominator_j)
```

### 4.2 Required Modification: Completeness-Corrected Likelihood

The corrected per-event likelihood for event i at trial H0 value h is:

```
p(x_GW^i | D_GW, h) = f_i(h) * L_cat^i(h) + (1 - f_i(h)) * L_comp^i(h)
```

where:

**f_i(h) = effective completeness for event i at trial h:**

```
f_i(h) = integral[ P_det(d_L(z,h), Omega_det) * p(z) * w_cat(z, Omega_det) dz ] /
         integral[ P_det(d_L(z,h), Omega_det) * p(z) * w_tot(z, Omega_det) dz ]
```

Here w_cat(z, Omega) is the weight (number or luminosity) of cataloged galaxies per
comoving volume at (z, Omega), and w_tot includes all galaxies (from the Schechter
function extrapolation). In the simple redshift-only approximation:

```
f_i(h) ~ integral[ P_det(d_L(z,h), Omega_det) * f(z) * p_uniform(z) dz ] /
         integral[ P_det(d_L(z,h), Omega_det) * p_uniform(z) dz ]
```

where f(z) is the precomputed angle-averaged GLADE+ completeness and p_uniform(z) is
uniform in comoving volume.

In the simplest implementation, f(z) can be used directly without the P_det-weighted
averaging, treating it as a constant within each event's sensitive volume.

**L_cat^i(h) = catalog term (our current likelihood):**

This is exactly what `p_Di()` currently computes:

```
L_cat^i(h) = sum_j numerator_j(h) / sum_j denominator_j(h)
```

No changes needed to the catalog term.

**L_comp^i(h) = completion term (NEW — must be added):**

```
L_comp^i(h) = integral[ p_GW(x | z, Omega, h) * P_det(d_L(z,h), Omega) * p_uniform(z) dz dOmega ] /
              integral[ P_det(d_L(z,h), Omega) * p_uniform(z) dz dOmega ]
```

where p_uniform(z) is the comoving volume element:

```
p_uniform(z) proportional to d_com^2(z) * c / H(z)
```

In the completion term, the sky position integral is over the GW posterior. For LISA EMRI
with tight sky localization, this simplifies:

```
L_comp^i(h) ~ integral_z [ p_GW(x | z, Omega_det, h) * P_det(d_L(z,h), Omega_det) * dV_c/dz ] dz
              ---------------------------------------------------------------------------------
              integral_z [ P_det(d_L(z,h), Omega_det) * dV_c/dz ] dz
```

where we evaluate at the detected sky position Omega_det since the sky localization is tight.
The comoving volume element is:

```
dV_c/dz = 4*pi * d_com^2(z) * c / H(z)
```

But since we divide numerator by denominator, the 4*pi cancels.

### 4.3 Implementation Plan

#### Step 1: Precompute f(z) — The GLADE+ Completeness Function

Create a new module or function that computes the angle-averaged GLADE+ completeness
as a function of redshift:

**File:** `master_thesis_code/galaxy_catalogue/completeness.py` (new file)

```python
def glade_completeness(z: ndarray, band: str = "B") -> ndarray:
    """
    Estimate GLADE+ completeness fraction as a function of redshift.

    Compares the cumulative luminosity (B-band or K-band) of GLADE+ galaxies
    in redshift shells to the expected total from the Schechter function
    integrated over the comoving volume.

    Args:
        z: Array of redshift values.
        band: Photometric band for weighting ("B" or "K").

    Returns:
        f(z): Completeness fraction at each redshift, in [0, 1].

    References:
        Dalya et al. (2022), arXiv:2110.06184, Section 3
    """
    ...
```

**Data needed:**
- The GLADE+ reduced catalog already loaded by `GalaxyCatalogueHandler` — extract B-band
  magnitudes (if available) or galaxy counts per redshift bin
- The reference B-band luminosity density: j_B = (1.98 +/- 0.16) x 10^{-2} L_10 Mpc^{-3}
  (Dalya et al. 2022)
- The Schechter function parameters for the completeness integral

**Fallback if magnitude data is unavailable:**
Use the number density approach (Finke et al. 2021): estimate total galaxy number density
n_gal ~ 0.1 Mpc^{-3}, count GLADE+ galaxies per comoving volume shell, and compute
f(z) = n_cat(z) / n_gal.

**Output:** A callable or interpolated function f(z) -> [0, 1].

#### Step 2: Implement the Completion Term

**File:** `master_thesis_code/bayesian_inference/bayesian_statistics.py`

Add a new function `completion_term()`:

```python
def completion_term(
    detection: Detection,
    detection_index: int,
    h: float,
    completeness_fn: Callable[[ndarray], ndarray],
    z_min: float,
    z_max: float,
) -> float:
    """
    Compute the completion term for uncataloged galaxies.

    This integrates the GW likelihood * P_det over a smooth
    uniform-in-comoving-volume galaxy density, for the fraction
    of galaxies not in the catalog.

    Returns:
        L_comp = numerator_comp / denominator_comp
    """
    ...
```

The completion term numerator integrand at redshift z:

```python
def completion_numerator_integrand(z):
    d_L = dist(z, h=h)
    d_L_frac = d_L / detection.d_L
    phi_det = detection.phi
    theta_det = detection.theta

    p_det = detection_probability.interpolated(d_L, phi_det, theta_det, h=h)
    p_gw = gaussian_likelihood.pdf([phi_det, theta_det, d_L_frac])

    # Comoving volume element: d_com^2 * c/H(z)
    # Using d_com = d_L / (1+z) and H(z) from cosmology
    dVc_dz = comoving_volume_element(z, h)  # needs implementation

    return p_det * p_gw * dVc_dz
```

The completion term denominator integrand:

```python
def completion_denominator_integrand(z):
    d_L = dist(z, h=h)
    p_det = detection_probability.interpolated(d_L, phi_det, theta_det, h=h)
    dVc_dz = comoving_volume_element(z, h)
    return p_det * dVc_dz
```

#### Step 3: Modify p_Di() to Combine Catalog and Completion Terms

**File:** `bayesian_statistics.py`, function `p_Di()` (line 352)

Currently:
```python
return (
    likelihood_without_bh_mass / selection_effect_correction_without_bh_mass,
    likelihood_with_bh_mass / selection_effect_correction_with_bh_mass,
)
```

Modified:
```python
# Catalog term (existing computation)
L_cat = likelihood_without_bh_mass / selection_effect_correction_without_bh_mass

# Completeness fraction for this event
f = effective_completeness(detection, h, completeness_fn, detection_probability)

# Completion term (new)
L_comp = completion_term(detection, detection_index, h, completeness_fn, z_min, z_max)

# Combined likelihood
likelihood = f * L_cat + (1 - f) * L_comp

return (likelihood, ...)
```

#### Step 4: Add comoving_volume_element() to physical_relations.py

**File:** `master_thesis_code/physical_relations.py`

```python
def comoving_volume_element(z: float, h: float = H) -> float:
    """
    Compute dV_c/dz per steradian.

    dV_c/dz/dOmega = d_com^2(z) * c / H(z)

    where d_com = d_L / (1+z) is the comoving distance.

    Args:
        z: Redshift.
        h: Dimensionless Hubble constant.

    Returns:
        dV_c/dz per steradian in Mpc^3.
    """
    d_L_val = dist(z, h=h)
    d_com = d_L_val / (1 + z)
    # H(z) = h * 100 * sqrt(Omega_m * (1+z)^3 + Omega_Lambda)
    Hz = h * 100 * np.sqrt(OMEGA_M * (1 + z)**3 + (1 - OMEGA_M))  # km/s/Mpc
    c_km_s = SPEED_OF_LIGHT_KM_S  # from constants.py
    return d_com**2 * c_km_s / Hz  # Mpc^3 / sr
```

### 4.4 Summary of Files to Modify

| File | Change | Type |
|---|---|---|
| `galaxy_catalogue/completeness.py` | **NEW** — GLADE+ completeness estimation | New module |
| `physical_relations.py` | Add `comoving_volume_element()` | Physics change |
| `bayesian_inference/bayesian_statistics.py` | Add `completion_term()`, modify `p_Di()` | Physics change |
| `bayesian_inference/bayesian_statistics.py` | Thread completeness_fn through `evaluate()` -> `p_D()` -> `p_Di()` | Refactor |
| `constants.py` | Add Schechter function parameters, B-band luminosity density | Physics constants |

### 4.5 Physics Change Protocol Checklist

Both `physical_relations.py` and `bayesian_statistics.py` modifications are **physics changes**
(modifying computed values) and require the Physics Change Protocol per CLAUDE.md:

For each change:
1. Old formula vs new formula
2. Reference (Gray et al. 2020, specific equation numbers)
3. Dimensional analysis
4. Limiting case (f=1 recovers current code)
5. Approval before implementation

---

## 5. Practical Considerations

### 5.1 Sensitivity to Completeness Estimate Errors

The corrected posterior is robust to moderate errors in f(z) because:

1. **The completion term is diffuse:** It spreads the likelihood over a smooth prior, which
   contributes broad (uninformative) support for all H0 values. Even with an imperfect
   completeness estimate, the completion term prevents the sharp bias we currently see.

2. **The catalog term is always informative when complete:** Where f(z) ~ 1, the catalog
   galaxies dominate and provide tight constraints, regardless of the completion term.

3. **The combined likelihood interpolates smoothly:** As f(z) decreases, the analysis
   gracefully transitions from catalog-dominated (informative) to uniform-prior-dominated
   (uninformative but unbiased).

Gray et al. (2020) demonstrate (Table 2) that unbiased H0 recovery works even at 25%
catalog completeness, though with 3x larger uncertainties than the complete-catalog case.
Their MDA2 results: H0 = 70.14 (+2.18/-2.18) km/s/Mpc with 25% completeness vs
68.91 (+1.36/-1.22) with 100%.

### 5.2 Is Angle-Averaged f(z) Sufficient?

For our use case (LISA EMRI), a redshift-only completeness f(z) is a good first
approximation because:

- LISA EMRI sky localization is ~0.01-1 deg^2 — much smaller than the angular scale of
  GLADE+ completeness variation
- The dominant effect is redshift-dependent flux limiting, which is approximately isotropic
- The angular dependence matters mainly for LIGO where the sky error boxes span tens of
  degrees across regions of varying catalog depth

However, for a second-order improvement: events at sky positions covered by SDSS will
have better completeness than events in regions covered only by 2MPZ. This can be accounted
for by computing f(z, Omega_event) using the local catalog density around each event's
best-fit sky position.

### 5.3 Minimum Catalog Completeness for the Method to Work

The method works at any completeness level, but the information content degrades:

- **f > 0.5:** Catalog term dominates; H0 constraints primarily from galaxy positions
- **f = 0.2-0.5:** Both terms contribute; wider posteriors but still informative
- **f < 0.2:** Completion term dominates; approaching the statistical-siren limit
- **f = 0:** Pure statistical siren; no catalog information (very broad posteriors)

For our case: the 355 events at z = 0.08-0.13 (where the bias is worst) have f ~ 0.2-0.5.
The completion term will provide the missing information, broadening each event's
contribution but removing the systematic bias.

### 5.4 Interaction with Existing P_det Grid

The completion term uses the same P_det interpolator as the catalog term. No changes to the
P_det computation are needed. The P_det grid (from injection campaigns) already covers the
relevant d_L range — our Test 5 from the bias investigation confirmed no boundary clipping.

The denominator of the completion term (integral of P_det * dV_c/dz) is essentially the
same as the GW selection effect normalization used in other analyses. It can be precomputed
once for each h value and cached.

### 5.5 Computational Cost

The completion term adds one 1D numerical integral (over z) per event per h value. This is
much cheaper than the per-galaxy integrals already computed for the catalog term (which
involve N_gal integrals). The overhead should be negligible.

The completeness function f(z) can be precomputed once and interpolated, adding zero
computational overhead per event.

### 5.6 Verification Strategy

After implementation, verify the correction works by:

1. **Limiting case test:** Set f(z) = 1 everywhere, confirm the result matches the current
   (uncorrected) code exactly.

2. **Statistical siren test:** Set f(z) = 0, confirm the result uses only the completion
   term (uniform prior) — should give a very broad posterior centered near the true H0.

3. **Known-answer test:** Use the same 534-detection dataset with the corrected likelihood.
   The MAP should shift from 0.66 toward 0.73 (the true value).

4. **Synthetic catalog test:** Generate a perfectly complete synthetic catalog (uniform in
   comoving volume), run the corrected analysis with f(z) = 1, and verify unbiased recovery.

5. **Sensitivity test:** Vary f(z) by +/-20% and verify the posterior peak is stable
   (it should shift by much less than 20%).

---

## Summary

The root cause of our H0 bias (MAP=0.66 vs true h=0.73) is a known, well-studied problem in
dark siren cosmology: galaxy catalog incompleteness at higher redshifts creates a systematic
preference for lower H0 values. The standard solution, established by Gray et al. (2020) and
implemented in the LVK gwcosmo pipeline, is to add a "completion term" to the likelihood
that accounts for the probability that the true host galaxy is not in the catalog.

The correction requires:
1. **Estimating GLADE+ completeness f(z)** using B-band luminosity comparison or galaxy
   number density methods
2. **Adding a completion term** to `p_Di()` that integrates the GW likelihood over a smooth
   uniform-in-comoving-volume prior
3. **Weighting** the catalog and completion terms by f(z) and (1-f(z)) respectively

The implementation is well-defined, maps directly to our existing code structure, and should
resolve the bias while maintaining the informative power of the galaxy catalog where it is
complete. No published LISA EMRI analysis has yet implemented this correction, making this
a novel contribution.
