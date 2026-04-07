# Phase 25: Likelihood Correction - Research

**Researched:** 2026-04-04
**Domain:** Gravitational-wave dark siren cosmology / Bayesian inference
**Confidence:** HIGH

## Summary

Phase 25 implements the completeness-corrected dark siren likelihood from Gray et al. (2020), arXiv:1908.06050, Eq. (9). The current code in `bayesian_statistics.py:p_Di()` computes only the catalog term (sum over galaxies in GLADE+). This implicitly assumes the host galaxy is always in the catalog, which fails at z > 0.08 where GLADE+ completeness drops below 50%, producing the observed H0 posterior bias (MAP = 0.66 vs true h = 0.73).

The correction adds a completion term representing the contribution from uncataloged galaxies, weighted by the catalog incompleteness fraction (1 - f_i). The completion term replaces the discrete galaxy sum with a smooth comoving volume prior, integrated over the GW error volume. All building blocks already exist in the codebase: the completeness function f(z, h) from Phase 24, the comoving volume element dVc/dz/dOmega from `physical_relations.py`, and the detection probability P_det from `SimulationDetectionProbability`.

**Primary recommendation:** Implement Eq. (9) of Gray et al. (2020) by computing L_comp as a 1D integral over redshift in `p_Di()`, using `comoving_volume_element()` as the prior, and combining with the existing catalog term weighted by f_i from `GladeCatalogCompleteness.get_completeness_at_redshift()`.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| Gray et al. (2020), arXiv:1908.06050 | method | Defines the completeness-corrected likelihood (Eq. 9, Appendix A.2) | cite equations, follow structure | plan, execution, reference comments |
| Dalya et al. (2022), arXiv:2110.06184 | benchmark | Source of GLADE+ completeness data used in f(z) | use via GladeCatalogCompleteness | execution |
| Hogg (1999), arXiv:astro-ph/9905116 | method | Comoving volume element formula (Eq. 28) | already implemented, cite | reference comments |
| Phase 24 deliverables | prior artifact | GladeCatalogCompleteness class, comoving_volume_element() | use directly | execution, verification |
| Bias investigation (project_bias_status.md) | prior artifact | Root-caused H0 bias to catalog incompleteness | this phase is the fix | verification |

**Missing or weak anchors:** The digitized completeness data may represent number completeness rather than B-band luminosity completeness (flagged in Phase 24 verification). This affects the numerical values of f(z) but NOT the mathematical structure of the correction. The code change is valid regardless; the completeness data can be re-digitized independently.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Distance unit (d_L) | Gpc (internal), Mpc (volume element) | -- | codebase convention |
| Hubble parameter | h = H0 / (100 km/s/Mpc), dimensionless | H0 in km/s/Mpc | constants.py |
| Comoving volume element | dVc/dz/dOmega in Mpc^3/sr | per unit volume | Hogg (1999) Eq. 28 |
| Completeness | fraction in [0, 1] | percent | GladeCatalogCompleteness |
| GW likelihood coordinates | (phi, theta, d_L/d_L_det) or (phi, theta, d_L/d_L_det, M_z/M_z_det) | absolute values | bayesian_statistics.py |

**CRITICAL: All equations and results below use these conventions. The GW Gaussian is parameterized in fractional coordinates (d_L_model / d_L_detected), NOT absolute d_L. The comoving_volume_element returns Mpc^3/sr but dist() returns Gpc, so the GPC_TO_MPC factor must be applied when converting between the two.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| p(x_GW\|D_GW,H0) = f_i * L_cat^i + (1-f_i) * L_comp^i | Completeness-corrected per-event likelihood | Gray et al. (2020) Eq. 9 | The master equation to implement |
| L_cat^i = sum_j(num_j) / sum_j(denom_j) | Catalog term | Gray et al. (2020) Eqs. 24-25, Appendix A.2.1 | Already implemented in p_Di() |
| L_comp^i = integral[p_GW * P_det * dVc/dz dz] / integral[P_det * dVc/dz dz] | Completion term | Gray et al. (2020) Eqs. 31-32, Appendix A.2.3 | NEW: to be implemented |
| dVc/dz/dOmega = d_com^2 * c / H(z) | Comoving volume element | Hogg (1999) Eq. 28 | Already in physical_relations.py |
| f(z, h) | GLADE+ completeness fraction | Dalya et al. (2022) Fig. 2, via GladeCatalogCompleteness | Phase 24 deliverable |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| 1D Gauss-Legendre quadrature | Numerical integration over z | L_comp numerator and denominator | scipy.integrate.fixed_quad |
| Multivariate Gaussian evaluation | GW likelihood at trial (phi, theta, d_L(z)/d_L_det) | L_comp numerator integrand | scipy.stats.multivariate_normal |
| Redshift-distance conversion | Maps z to d_L for GW likelihood evaluation | integrand of L_comp | physical_relations.dist_vectorized |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Angle-averaged completeness f(z) | angular variation of f within GW error box | LISA sky localization ~1 deg^2 | Second-order correction; negligible for EMRI | Pixelated completeness (Gray et al. 2022) |
| p(s\|z) = const (no source evolution) | source rate evolution | z < 0.2 (EMRI range) | O(10%) for lambda~1 | (1+z)^lambda weighting |
| p(s\|M) = const (equal galaxy weights) | host luminosity weighting | all z | Conservative; may slightly reduce constraining power | Luminosity weighting L(M) |
| Completion term independent of BH mass | BH mass marginalization | uncataloged host (no galaxy info) | Exact: when host is unknown, no mass info exists | Schechter function marginalization |

## Standard Approaches

### Approach 1: Direct Implementation of Gray et al. (2020) Eq. 9 (RECOMMENDED)

**What:** Add a completion term L_comp computed as a 1D integral over z inside `p_Di()`, combine with existing catalog term using completeness weight f_i.

**Why standard:** This is the standard framework used by gwcosmo and all major dark siren analyses since 2020. The mathematical structure is well-established.

**Track record:** Used by LVK in the O3 H0 measurement (Abbott et al. 2021, arXiv:2111.03604), Finke et al. (2021), Gray et al. (2022), and all subsequent dark siren papers.

**Key steps:**

1. **Instantiate GladeCatalogCompleteness** in `evaluate()` and pass it through to `p_Di()`.
2. **Compute f_i** for each detection: `f_i = completeness.get_completeness_at_redshift(z_det, h)` where `z_det = dist_to_redshift(detection.d_L, h=h)`.
3. **Compute L_comp numerator:** 1D integral over z of `p_GW(phi_det, theta_det, d_L(z,h)/d_L_det) * P_det(d_L(z,h), phi_det, theta_det) * dVc_dz(z, h)`.
4. **Compute L_comp denominator:** 1D integral over z of `P_det(d_L(z,h), phi_det, theta_det) * dVc_dz(z, h)`.
5. **Combine:** `likelihood = f_i * L_cat + (1 - f_i) * L_comp`.
6. **Add Gray et al. (2020) reference comment** to `comoving_volume_element()` in physical_relations.py.

**Known difficulties at each step:**

- Step 2: `detection.d_L` is in Gpc; `dist_to_redshift` expects Gpc. Correct as-is.
- Step 3: The GW Gaussian must be evaluated at the detection's own sky position (phi_det, theta_det), NOT at a galaxy position. The fractional coordinate d_L(z,h)/d_L_det uses `dist_vectorized(z, h=h)` which returns Gpc, and `detection.d_L` is in Gpc, so the ratio is dimensionless. Correct.
- Step 3: `comoving_volume_element()` returns Mpc^3/sr, which is a large number (O(10^6) at z~0.1). The absolute scale cancels in the ratio L_comp = num/denom. No normalization issue.
- Step 5: Both L_cat and L_comp are ratios (num/denom), so both are on a comparable scale. The f_i weighting is dimensionally consistent.

### Approach 2: Finke et al. (2021) Combined Prior (FALLBACK)

**What:** Instead of separate catalog and completion terms, construct a combined prior p_0(z, Omega) = f * p_cat + (1 - f) * p_miss and use it throughout.

**When to switch:** If the two-term approach (catalog + completion) produces numerical instabilities at extreme completeness values (f ~ 0 or f ~ 1).

**Tradeoffs:** Mathematically equivalent to Approach 1. Requires restructuring how galaxies are summed (prior weighting rather than post-hoc combination). More invasive code change.

### Anti-Patterns to Avoid

- **Computing f_i as a detection-probability-weighted average:** This is the "correct" form from Gray et al. (2020) Eq. 29, but it requires an expensive integral over the entire sensitive volume. The simplified form f_i = f(z_det, h) evaluated at the detected redshift is standard practice and adequate for EMRI with small d_L errors.
  - _Example:_ gwcosmo uses the simplified form for most analyses; the full integral is only needed when P_det varies rapidly within the completeness transition region.

- **Re-deriving the catalog term with completeness weighting:** The existing catalog term in `p_Di()` is correct as-is. Do NOT multiply individual galaxy contributions by f(z_j). The completeness enters only at the level of combining the two terms.

- **Adding the completion term inside `single_host_likelihood()`:** The completion term is per-detection, not per-galaxy. It should be computed ONCE per detection in `p_Di()`, after the galaxy loop completes.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Catalog term L_cat | sum_j(num_j) / sum_j(denom_j) | bayesian_statistics.py lines 430-459 | Use as-is; this is `likelihood_without_bh_mass` and `likelihood_with_bh_mass` |
| Comoving volume element | d_com^2 * c / H(z) | physical_relations.py:comoving_volume_element() | Call directly |
| Completeness f(z, h) | Interpolated from digitized Dalya et al. data | glade_completeness.py:get_completeness_at_redshift() | Call directly |
| GW 3D Gaussian | multivariate_normal(mean=[phi_det, theta_det, 1], cov=...) | bayesian_statistics.py lines 207-209 | Access from detection_likelihood_gaussians_by_detection_index |
| P_det (3D, no BH mass) | detection_probability_without_bh_mass_interpolated(d_L, phi, theta, h=h) | simulation_detection_probability.py | Call directly; d_L in Gpc |

**Key insight:** ALL building blocks exist. This phase is pure INTEGRATION WORK: wire existing components into the correct mathematical formula. No new physics functions need to be written.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| dist_to_redshift(d_L, h) | Inverts d_L(z, h) to find z | physical_relations.py | d_L in Gpc |
| dist_vectorized(z, h) | d_L in Gpc for array of z | physical_relations.py | Returns Gpc |
| get_redshift_outer_bounds() | z_min, z_max for integration limits | physical_relations.py | Already used in p_D() |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| arXiv:1908.06050 | Gray et al. | 2020 | Master reference for the likelihood formula | Eq. 9, Appendix A.2 equations 24-32 |
| arXiv:2101.12660 | Finke et al. | 2021 | Alternative formulation with combined prior | Sec. 3.2, Eq. for p_0 |
| arXiv:2110.06184 | Dalya et al. | 2022 | GLADE+ completeness characterization | Fig. 2 (already digitized) |
| arXiv:2102.01708 | Laghi et al. | 2021 | EMRI-specific dark siren analysis | Confirmation that standard method applies to EMRI |
| arXiv:astro-ph/9905116 | Hogg | 1999 | Comoving volume element | Eq. 28 (already implemented) |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| scipy.integrate.fixed_quad | scipy | 1D Gauss-Legendre quadrature for L_comp integrals | Already used for catalog term integrals with N=50 points |
| scipy.stats.multivariate_normal | scipy | GW likelihood Gaussian evaluation | Already used throughout bayesian_statistics.py |
| numpy | numpy | Array operations in integrands | Standard |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| GladeCatalogCompleteness | Evaluate f(z, h) | Once per detection in p_Di() |
| comoving_volume_element | Evaluate dVc/dz/dOmega | Inside L_comp integrands |
| dist_vectorized | Convert z to d_L (Gpc) | Inside L_comp integrands |
| dist_to_redshift | Convert detection d_L to z for f_i | Once per detection |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| L_comp per detection (2 x fixed_quad, N=50) | ~0.01s per detection | dist_vectorized called 50 times | Negligible vs galaxy loop |
| f_i per detection | ~0.001s | Single interpolation | Negligible |
| Total overhead for 534 detections | ~5-10s additional | Runs in main process, not workers | Acceptable |

**No additional packages needed.** All dependencies are already installed.

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| f=1 limit | Completeness=1 recovers catalog-only result | Set f_i=1 for all detections, compare to current code | Identical posteriors |
| f=0 limit | Empty catalog recovers volume-only result | Set f_i=0, check L_comp alone | Smooth posterior, broader than catalog-only |
| Intermediate f | Weighted combination | Vary f_i manually | Posterior peak between catalog-only and volume-only |
| Ratio sanity | L_comp is a proper likelihood ratio | Check L_comp > 0 for all detections | Always positive |
| H0 bias reduction | Main science goal | Run full evaluation with correction | MAP closer to 0.73 than current 0.66 |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| Complete catalog (f=1 everywhere) | z < 0.01 where GLADE is complete | Current code output (unchanged) | Existing posteriors |
| Empty catalog (f=0 everywhere) | Hypothetical | Uniform-in-volume prior: broader posterior peaked at true H0 | Gray et al. (2020) Sec. IV.A |
| Single detection, single galaxy, f=1 | Toy model | L_cat = p_GW * P_det / P_det = p_GW (GW likelihood only) | Analytic |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| L_comp num/denom ratio | Compare fixed_quad N=50 vs N=100 | < 1% relative difference | Self-consistency |
| L_comp vs L_cat at low z | At z < 0.01 where f ~ 1, completion term should be negligible | L_comp * (1-f) << f * L_cat | Volume term suppressed by (1-f) ~ 0 |

### Red Flags During Computation

- If L_comp is negative for any detection: sign error in integrand or integration limits are wrong.
- If L_comp is orders of magnitude larger than L_cat: normalization mismatch between the two terms. Check that both are ratios (numerator/denominator) and thus on comparable scales.
- If the corrected posterior shifts AWAY from h=0.73 (i.e., bias increases): the correction has the wrong sign or the completeness function is inverted (using f instead of 1-f or vice versa).
- If L_comp denominator is zero or near-zero: P_det is zero throughout the integration range. This means the detection cannot be explained at any redshift -- check integration limits.

## Common Pitfalls

### Pitfall 1: Unit Mismatch Between dist() and comoving_volume_element()

**What goes wrong:** `dist()` and `dist_vectorized()` return d_L in Gpc. `comoving_volume_element()` works internally in Mpc (applies GPC_TO_MPC). P_det expects d_L in Gpc. If the completion term integrand mixes units, the result is wrong by factors of 10^9.

**Why it happens:** Historical convention: the distance functions return Gpc, but volume integrals are naturally in Mpc^3.

**How to avoid:** In the L_comp integrand, use `dist_vectorized(z, h=h)` (Gpc) for P_det and for the d_L/d_L_det ratio. Use `comoving_volume_element(z, h)` (Mpc^3/sr) directly -- its absolute scale cancels in the num/denom ratio.

**Warning signs:** L_comp values that are O(10^9) different from L_cat values before the f_i weighting.

**Recovery:** Check that d_L passed to P_det is in Gpc and that the volume element is used consistently in both numerator and denominator.

### Pitfall 2: Computing f_i at the Wrong Redshift

**What goes wrong:** Evaluating f at z corresponding to d_L_true (unknown) vs z corresponding to d_L_detected (known) vs some h-dependent z.

**Why it happens:** The completeness depends on both z and h because d_L(z, h) determines which galaxies are visible.

**How to avoid:** Use `f_i = completeness.get_completeness_at_redshift(z_det, h)` where `z_det = dist_to_redshift(detection.d_L, h=h)`. This evaluates the completeness at the redshift implied by the detected distance and the trial H0 value, which is exactly what Gray et al. (2020) prescribe.

**Warning signs:** f_i that does not vary with h (it should, because z_det = z(d_L, h) depends on h).

**Recovery:** Ensure dist_to_redshift is called with the current trial h, not the fiducial H.

### Pitfall 3: Forgetting That p_Di Returns Both "Without BH Mass" and "With BH Mass" Variants

**What goes wrong:** Applying the completion correction only to one variant.

**Why it happens:** The existing code returns a tuple (likelihood_without_bh_mass, likelihood_with_bh_mass).

**How to avoid:** Apply the correction to BOTH return values. For the completion term, use the "without BH mass" Gaussian (3D) for both variants, because when the host is not in the catalog, there is no galaxy mass information available.

**Warning signs:** One posterior corrected, the other still biased.

**Recovery:** Ensure both `likelihood_without_bh_mass` and `likelihood_with_bh_mass` are combined with L_comp.

### Pitfall 4: Integration Limits for L_comp Too Narrow or Too Wide

**What goes wrong:** If integration limits for L_comp are too narrow, the integral misses probability mass. If too wide, the integrand is effectively zero and the quadrature wastes points.

**Why it happens:** The catalog term uses detection-specific limits (d_L +/- 4 sigma mapped to z). The completion term should use similar limits because the GW Gaussian falls off at 4 sigma.

**How to avoid:** Use the same integration limits as the catalog term numerator: `z_lower = dist_to_redshift(d_L - 4*sigma_dL, h)` and `z_upper = dist_to_redshift(d_L + 4*sigma_dL, h)`. These are already computed in `single_host_likelihood()` as `numerator_integration_lower/upper_redshift_limit`.

**Warning signs:** L_comp that is sensitive to small changes in integration limits (indicates limits are cutting into the peak).

**Recovery:** Widen to 5 sigma or use adaptive quadrature (scipy.integrate.quad) for verification.

### Pitfall 5: Multiprocessing Globals Not Updated

**What goes wrong:** The completion term needs access to `detection_probability` and `detection_likelihood_gaussians_by_detection_index`, which are module-level globals set in `child_process_init()`.

**Why it happens:** The completion term is computed in `p_Di()` in the main process, NOT in worker processes.

**How to avoid:** Ensure the completion term computation in `p_Di()` has access to the detection probability and GW Gaussian objects. The `evaluate()` method creates `detection_probability` locally (line 148) and `detection_likelihood_multivariate_gaussian_by_detection_index` (line 155). These are passed to workers via `child_process_init`, but `p_Di()` is called within the `with pool:` context, so these objects exist in the main process scope. They need to be passed explicitly to `p_Di()` or accessed via `self`.

**Warning signs:** NameError or accessing stale global state.

**Recovery:** Pass `detection_probability` and the Gaussian dict as arguments to `p_Di()` and use them directly for the completion term computation.

## Level of Rigor

**Required for this phase:** Physicist's proof with numerical validation.

**Justification:** The mathematical framework (Gray et al. 2020) is well-established and peer-reviewed. This phase is implementation work, not derivation. The primary risk is implementation bugs (unit errors, sign errors, wrong variable), not mathematical errors.

**What this means concretely:**

- Reference comments citing Gray et al. (2020) equation numbers on every new physics line
- Dimensional consistency checked at each step (Gpc vs Mpc, fractions vs absolutes)
- f=1 limiting case must reproduce current code output exactly
- f=0 limiting case must produce a sensible (broader, less biased) posterior
- No formal convergence proofs needed; N=50 fixed_quad is validated by comparison with N=100

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Catalog-only likelihood (no completeness) | Completeness-corrected (catalog + completion) | Gray et al. 2020 | Removes bias from incomplete catalogs |
| Global completeness f(z) | Pixelated f(z, Omega) per HEALPix pixel | Gray et al. 2022 | ~5% improvement; overkill for EMRI with good sky loc |
| Number completeness | Luminosity-weighted completeness | Gray et al. 2020, Sec. II.C | Increases effective completeness for bright hosts |

**Superseded approaches to avoid:**

- **Catalog-only (no completion term):** This is what the code currently does. It produces biased H0 posteriors when the catalog is incomplete. Must be replaced.

## Open Questions

1. **Number vs luminosity completeness in the digitized data**
   - What we know: Phase 24 verification flagged that the digitized data appears to be number completeness (~50% at d_L=122 Mpc) rather than B-band luminosity completeness (~90% at d_L=130 Mpc).
   - What's unclear: Which curve from Dalya et al. (2022) Fig. 2 was actually digitized.
   - Impact on this phase: The mathematical structure of the correction is IDENTICAL regardless. Only the numerical values of f_i change. Using number completeness is conservative (lower f, more weight on completion term).
   - Recommendation: Proceed with the existing digitized data. Note the caveat. The completeness data can be updated independently in a future phase without changing any code in bayesian_statistics.py.

2. **Luminosity weighting of galaxies (p(s|M))**
   - What we know: Gray et al. (2020) Eq. 18 includes an optional luminosity weighting where brighter galaxies are more likely hosts.
   - What's unclear: Whether this matters for EMRI (where the central BH mass, not the galaxy luminosity, is the primary parameter).
   - Impact on this phase: Setting p(s|M) = const is standard for EMRI analyses (Laghi et al. 2021). Can be added later.
   - Recommendation: Use p(s|M) = const (equal galaxy weights). Defer luminosity weighting.

3. **Source rate evolution p(s|z)**
   - What we know: For z < 0.2, the EMRI rate evolution is a small correction.
   - What's unclear: The exact EMRI rate-redshift relation.
   - Impact on this phase: Setting p(s|z) = const is conservative. The comoving volume element already captures the geometric volume growth.
   - Recommendation: Use p(s|z) = const. Note in the code that this can be upgraded.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Direct f_i * L_cat + (1-f_i) * L_comp | Numerical instability at f ~ 0 or f ~ 1 | Combined prior approach (Finke et al. 2021) | Moderate: restructure p_Di to use combined prior |
| fixed_quad N=50 for L_comp | Poor convergence of volume-weighted integral | scipy.integrate.quad (adaptive) | Low: swap one function call |
| Angle-averaged f(z) | Sky-dependent completeness matters | Pixelated f(z, Omega) | High: requires HEALPix infrastructure |

**Decision criteria:** If the f=1 limiting case does not reproduce existing results to < 0.1% relative error, there is a bug in the implementation, not a method failure. If the corrected posterior shifts in the wrong direction, check the sign of the (1-f_i) term.

## Caveats and Alternatives (Self-Critique)

1. **Assumption: angle-averaged completeness is sufficient.** For LIGO with ~100 deg^2 error boxes, this might fail. For LISA EMRI with ~1 deg^2, angular variation within one error box is negligible. However, DIFFERENT events at different sky positions will have different completeness — this is captured by f(z), since all events use the same angle-averaged curve. If GLADE+ has significant sky-dependent incompleteness (e.g., Galactic plane avoidance), this could matter.

2. **Assumption: p(s|z) = const.** If the EMRI rate increases steeply with redshift (unlikely at z < 0.2), ignoring rate evolution could slightly bias the completion term. The effect is small because the GW likelihood Gaussian is narrow compared to the range over which the rate varies.

3. **Simplification: "without BH mass" L_comp for both variants.** For the "with BH mass" case, one could marginalize L_comp over a Schechter-like mass function. This would give a slightly different completion term. The simplification is justified because when the host is not in the catalog, we have NO galaxy mass information, so using a mass prior adds little value and significant complexity.

4. **The completion term uses the detected sky position.** In the full Gray et al. formulation, the completion term is integrated over Omega. But since the GW Gaussian is sharply peaked at (phi_det, theta_det), evaluating at the detected position and absorbing the solid angle integral into a normalization constant (that cancels in the ratio) is standard practice.

5. **Would a dark siren specialist disagree?** Possibly on the choice of p(s|M) = const — luminosity weighting is preferred for BBH. For EMRI, equal weighting is standard (Laghi et al. 2021). The core f_i * L_cat + (1-f_i) * L_comp structure is universally agreed upon.

## Sources

### Primary (HIGH confidence)

- [Gray et al. (2020), arXiv:1908.06050](https://arxiv.org/abs/1908.06050) — Master reference for completeness-corrected dark siren likelihood. Eq. 9 and Appendix A.2.
- [Hogg (1999), arXiv:astro-ph/9905116](https://arxiv.org/abs/astro-ph/9905116) — Comoving volume element Eq. 28. Already implemented.
- [Dalya et al. (2022), arXiv:2110.06184](https://arxiv.org/abs/2110.06184) — GLADE+ catalog completeness characterization. Fig. 2 digitized in Phase 24.

### Secondary (MEDIUM confidence)

- [Finke et al. (2021), arXiv:2101.12660](https://arxiv.org/abs/2101.12660) — Alternative combined-prior formulation. Sec. 3.2.
- [Laghi et al. (2021), arXiv:2102.01708](https://arxiv.org/abs/2102.01708) — EMRI-specific dark siren analysis confirming standard method applies.
- [Gray et al. (2022), arXiv:2111.04629](https://arxiv.org/abs/2111.04629) — Pixelated completeness approach. Not needed for this phase but documents the next-order improvement.
- [Abbott et al. (2021), arXiv:2111.03604](https://arxiv.org/abs/2111.03604) — LVK O3 H0 measurement using this framework.

### Tertiary (LOW confidence)

- Existing codebase literature research: `.gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md` — detailed comparison of approaches.

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH — Gray et al. (2020) is the standard reference, universally adopted
- Standard approaches: HIGH — direct implementation of a peer-reviewed formula with all building blocks available
- Computational tools: HIGH — all tools already in use in the codebase; no new dependencies
- Validation strategies: HIGH — clear limiting cases (f=1, f=0) provide unambiguous correctness checks

**Research date:** 2026-04-04
**Valid until:** Indefinitely for the mathematical framework. The completeness data (f(z)) may need updating if re-digitized.
