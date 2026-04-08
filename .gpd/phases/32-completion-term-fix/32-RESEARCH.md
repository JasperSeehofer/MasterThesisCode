# Phase 32: Completion Term Fix - Research

**Researched:** 2026-04-08
**Domain:** Dark siren Bayesian inference -- completeness-corrected likelihood normalization
**Confidence:** HIGH

## Summary

The systematic low-h bias (MAP h=0.66-0.68 vs true h=0.73) originates from the completion term L_comp having its denominator integrated over the same local 4-sigma d_L window as the numerator, rather than over the full detectable volume as specified by Gray et al. (2020) Eq. A.19. When both numerator and denominator share identical integration limits, the normalization partially cancels but retains a residual h-dependence from the interplay between p_GW(d_L(z,h)) and dVc/dz, creating a small per-event bias that compounds exponentially over 531 events.

The fix is conceptually straightforward: precompute the denominator D(h) = integral from z=0 to z_max of P_det(d_L(z,h)) * dVc/dz dz once per h-value (it does not depend on the event), then use it for all events. The numerator remains local (4-sigma window around each event's measured d_L). This matches the mathematical structure of Gray et al. (2020) Eq. A.19, where the out-of-catalog numerator integrates from z(M, m_th, H0) to infinity, while the denominator normalizes over the same full range. In our simplified case (no magnitude-based completeness threshold, using a scalar completeness fraction f_i instead), the denominator integrates over the full detectable volume z in [0, z_max] where z_max is set by where P_det drops to zero.

**Primary recommendation:** Precompute D(h) via Gauss-Legendre quadrature (scipy.integrate.fixed_quad, n=100) over z in [1e-6, z_max(h)], where z_max(h) is the redshift corresponding to the P_det grid's maximum d_L. Truncate P_det to 0 beyond the grid edge for this integral. Cache D(h) for each h-value in the evaluation grid. Keep the numerator's 4-sigma local integration unchanged.

## User Constraints

See phase CONTEXT.md for locked decisions and user constraints that apply to this phase.

Key constraints affecting this research:
- **LOCKED:** Keep dVc/dz (uniform in comoving volume) as the completion term prior -- no rate-weighting
- **LOCKED:** Extend L_comp denominator to full detectable volume per Gray et al. Eq. A.19; numerator stays local (4-sigma d_L window)
- **LOCKED:** Truncate P_det to 0 beyond injection grid edge for denominator; flag events where grid doesn't cover numerator's 4-sigma window
- **Discretion:** Quadrature method, D(h) caching, diagnostic output format
- **OUT OF SCOPE:** Rate-weighted prior R(z)*dVc/dz, extended injection grid, MVN allow_singular, fixed galaxy search window

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| Gray et al. (2020), arXiv:1908.06050, Eqs. 9, A.19 | method | Defines the completeness-corrected likelihood and out-of-catalog term normalization | read, compare, cite | plan, execution, verification |
| Catalog-only baseline (--catalog_only flag) | benchmark | Isolates L_cat contribution; must match when f_i=1 | run, compare | execution, verification |
| Production results: cluster_results/eval_corrected_full | prior artifact | Current baseline MAP h=0.66/0.68 with 531/527 events | compare before/after | verification |
| P_det fix commit 44d5358 | prior artifact | Changed fill_value to None (nearest-neighbor); this phase partially reverts for denominator | understand interaction | execution |
| Debug investigation: .gpd/debug/h0-posterior-bias-worsening.md | prior artifact | Documents root cause analysis of the bias | read | planning |

**Missing or weak anchors:** The exact d_L range of the injection grid is data-dependent (computed at runtime as max(injection d_L) * 1.1). The corresponding z_max for D(h) integration must be computed at runtime for each h. This is not a problem but means the integration limits are not known a priori.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Distance units | Gpc (luminosity distance) | Mpc | Codebase convention |
| Hubble parameter | Dimensionless h = H0/(100 km/s/Mpc) | H0 in km/s/Mpc | Codebase convention |
| Volume element | dVc/dz/dOmega in Mpc^3/sr | Gpc^3, with 4pi | Hogg (1999), physical_relations.py |
| Redshift prior | p(z) = dVc/dz (uniform in comoving volume-time) | Rate-weighted R(z)*dVc/dz | Gray et al. (2020) Eq. A.2 |
| Completeness | Scalar f_i = f(z_det, h) from GLADE digitized data | Pixel/direction-dependent | Dalya et al. (2022) |
| P_det boundary | fill_value=0 for denominator; fill_value=None for numerator | fill_value=None everywhere | CONTEXT.md decision |

**CRITICAL: All equations and results below use these conventions. The comoving volume element is per steradian, not integrated over 4pi. This is consistent with the current codebase where sky angles appear explicitly in the Gaussian likelihood.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| p(x_GW\|D_GW,H0) = p(x_GW\|G,D_GW,H0)p(G\|D_GW,H0) + p(x_GW\|Gbar,D_GW,H0)p(Gbar\|D_GW,H0) | Completeness-corrected single-event likelihood | Gray et al. (2020) Eq. 9 | Top-level combination formula |
| p(x_GW\|Gbar,D_GW,H0) = N_comp / D_comp where N_comp = integral[p_GW * p(s\|z) * p(z) * dOmega dz dM] and D_comp = integral[P_det * p(s\|z) * p(z) * dOmega dz dM] | Out-of-catalog term | Gray et al. (2020) Eq. A.19 | The term being fixed |
| L_comp = integral[p_GW(x\|z,Omega,h) * P_det(d_L(z,h)) * dVc/dz dz] / integral[P_det(d_L(z,h)) * dVc/dz dz] | Simplified completion term (marginalized over sky, no magnitude dependence) | Simplified from A.19 | Actual implementation formula |
| p_i = f_i * L_cat + (1-f_i) * L_comp | Per-event combined likelihood | Codebase, bayesian_statistics.py:826-830 | Combination step |
| dVc/dz/dOmega = d_com^2(z) * c / H(z) | Comoving volume element | Hogg (1999) Eq. 28, physical_relations.py:387 | Prior for uncataloged hosts |

### The Key Distinction: Numerator vs Denominator Integration Limits

The critical insight from Gray et al. (2020) Eq. A.19 is:

**Numerator** of L_comp: integrates over the region where the out-of-catalog host might be. In Gray et al., this is z in [z(M, m_th, H0), infinity] -- i.e., beyond the catalog completeness limit. In our simplified implementation with a scalar f_i, the numerator integrates over z where p_GW has support (the 4-sigma d_L window). This is correct because p_GW is negligible outside this window.

**Denominator** of L_comp: normalizes the probability. In Gray et al. Eq. A.19, the denominator has the SAME integration limits as the numerator (z from z(M, m_th, H0) to infinity). However, in the full formalism, this denominator represents p(D_GW | Gbar, s, H0), which is the probability of detecting a GW event given the host is NOT in the catalog. This is a global quantity -- it integrates P_det over the entire out-of-catalog volume.

**Current bug:** The current code (bayesian_statistics.py:803-807) uses the same local 4-sigma [z_lower, z_upper] limits for BOTH numerator and denominator:
```python
comp_numerator = fixed_quad(completion_numerator_integrand, z_lower, z_upper, n=50)[0]
comp_denominator = fixed_quad(completion_denominator_integrand, z_lower, z_upper, n=50)[0]
```

**The fix:** The denominator should integrate over the full detectable volume:
```python
comp_denominator = D_h  # precomputed: integral from z=1e-6 to z_max of P_det * dVc/dz dz
```

### Why Local Normalization Creates Bias

When both numerator and denominator integrate over the same local window centered on the event's d_L:
- The ratio L_comp = N/D partially cancels, but NOT perfectly
- The residual h-dependence arises because changing h shifts where in redshift the p_GW peak falls (z_peak(h) = dist_to_redshift(d_L_det, h))
- At lower h, z_peak is lower, where dVc/dz is smaller, so the numerator is relatively smaller
- But this effect does NOT cancel with the denominator because dVc/dz is nonlinear
- The net effect: a small systematic preference for certain h values per event
- With 531 events: (1 + epsilon)^531 ~ exp(531 * epsilon), amplifying tiny biases

With the full-volume denominator, D(h) correctly captures the global normalization and removes this artifact.

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Gauss-Legendre quadrature | High-accuracy numerical integration of smooth integrands | D(h) precomputation and numerator evaluation | Abramowitz & Stegun Ch. 25 |
| Precomputation / memoization | Compute D(h) once per h, reuse for all events | Denominator caching | Standard optimization |
| Root-finding (dist_to_redshift) | Invert d_L(z,h) to find z_max from the P_det grid's d_L_max | Integration limit determination | scipy.optimize.fsolve |
| Numerical convergence testing | Verify D(h) is stable with respect to quadrature order n | Validation | Standard numerical methods |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| P_det = 0 beyond injection grid | N/A (physical cutoff) | Always valid: EMRIs beyond grid are truly undetectable | Systematic underestimate near grid edge | Extend injection grid (deferred) |
| 4-sigma local window for numerator | sigma_dL / d_L << 1 (Fisher matrix accuracy) | Valid when p_GW is well-approximated as Gaussian | Negligible truncation error if 4-sigma captures 99.99% | Increase to 5-sigma or use adaptive quadrature |
| Scalar f_i (direction-independent completeness) | Sky variation of catalog completeness | Valid for all-sky average; breaks for anisotropic catalogs | Moderate; GLADE is roughly isotropic at EMRI distances | Pixel-based completeness (Finke et al. 2022) |

## Standard Approaches

### Approach 1: Precomputed Full-Volume Denominator D(h) (RECOMMENDED)

**What:** Compute D(h) = integral from 0 to z_max of P_det(d_L(z,h)) * dVc/dz dz once per h-value, where z_max is determined by the P_det grid's maximum d_L. Store as a lookup table. For each event, L_comp = N_i(h) / D(h).

**Why standard:** This is the mathematically correct normalization per Gray et al. (2020) Eq. A.19. The denominator does not depend on the event (only on P_det and the cosmology), so precomputation is both correct and efficient. This is how gwcosmo implements the completion term.

**Track record:** The Gray et al. (2020) mock data analyses recover an unbiased H0 with this normalization, even with 50% catalog completeness and 249 events.

**Key steps:**

1. At evaluation startup, determine z_max(h) for each h in the evaluation grid by inverting d_L_max of the P_det grid
2. For each h, compute D(h) via fixed_quad with n=100 over z in [1e-6, z_max(h)]
3. Build a lookup dict or array: D_table[h] = D(h)
4. In the per-event loop, compute only the local numerator N_i(h) as currently done
5. Set L_comp = N_i(h) / D_table[h]

**Known difficulties at each step:**

- Step 1: dist_to_redshift uses fsolve, which needs a reasonable initial guess. For d_L ~ few Gpc and h ~ 0.6-0.9, z ~ 0.5-3 is a reasonable initial guess. May need try/except for edge cases.
- Step 2: The integrand P_det(d_L(z,h)) * dVc/dz is smooth but has a sharp cutoff where P_det drops to zero. Gauss-Legendre handles this well if z_max is chosen correctly.
- Step 3: For 15 h-values, this is 15 integrals -- negligible cost.
- Step 5: Must handle D(h) = 0 gracefully (fall back to f_i=1, L_comp=0).

### Approach 2: Event-Local Denominator with Extended Window (FALLBACK)

**What:** Keep the current per-event denominator structure but extend its integration range to cover the full detectable volume rather than just the 4-sigma window.

**When to switch:** Only if precomputed D(h) is somehow event-dependent (it should not be in this formalism).

**Tradeoffs:** Much slower (recomputes the full integral for every event), but conceptually simpler and allows event-dependent P_det (e.g., if P_det depends on sky position, which it currently does not).

### Anti-Patterns to Avoid

- **Using the same integration limits for numerator and denominator:** This is the current bug. The numerator is correctly local (p_GW has compact support), but the denominator must be global.
  - _Example:_ Current code at bayesian_statistics.py:803-807 uses [z_lower, z_upper] for both.

- **Interpolating P_det beyond the grid with nearest-neighbor for the denominator:** Nearest-neighbor extrapolation artificially extends detectability. For the denominator integral, P_det must be zero beyond the grid.
  - _Example:_ The P_det fix (commit 44d5358) set fill_value=None globally. The denominator needs fill_value=0.

- **Recomputing D(h) per event:** D(h) depends only on P_det and cosmology, not on any event-specific quantities. Recomputing it per event wastes ~500x the computation time.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Completeness-corrected likelihood Eq. 9 | p = p(x\|G)p(G) + p(x\|Gbar)p(Gbar) | Gray et al. (2020) Eq. 9 | Top-level formula |
| Out-of-catalog term structure Eq. A.19 | Ratio of integrals with SAME limits for numerator and denominator | Gray et al. (2020) Eq. A.19 | Use to verify our simplification is correct |
| Comoving volume element | dVc/dz/dOmega = d_com^2 * c / H(z) | Hogg (1999) Eq. 28, physical_relations.py:387-427 | Already implemented in codebase |
| d_L(z,h) relation | Implemented in dist() and dist_vectorized() | physical_relations.py:28-118 | Already implemented |
| P_det interpolator | SimulationDetectionProbability with RegularGridInterpolator | simulation_detection_probability.py | Already implemented; need separate fill_value for denominator |
| GLADE completeness | f(z,h) via GladeCatalogCompleteness | glade_completeness.py, Dalya et al. (2022) | Already implemented |

**Key insight:** The entire mathematical framework is already implemented. The fix is ONLY about changing the integration limits of ONE integral (the denominator of L_comp) and precomputing it.

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| arXiv:1908.06050 | Gray et al. | 2020 | Defines the completeness-corrected likelihood | Eqs. 9, A.19 -- correct normalization structure |
| arXiv:2503.18887 | (systematic bias study) | 2025 | Documents bias sources in dark siren methods | Confirms catalog incompleteness drives low-H0 bias |
| arXiv:2111.03604 | LVK GWTC-3 H0 | 2021 | Reference implementation of the method | Validates full-volume normalization approach |
| arXiv:2404.16092 | Borghi et al. (LVK O4a) | 2024 | Latest O4a dark siren analysis | Confirms methodology is standard |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| scipy.integrate.fixed_quad | scipy >= 1.0 | Gauss-Legendre quadrature for D(h) | Already used for L_comp; exponential convergence for smooth integrands |
| scipy.interpolate.RegularGridInterpolator | scipy >= 1.0 | P_det interpolation | Already used in SimulationDetectionProbability |
| numpy | numpy >= 2.0 | Array operations | Already a core dependency |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| scipy.optimize.fsolve | Invert d_L(z,h) to find z_max from P_det grid d_L_max | Once per h-value at startup |
| pandas | Load/save diagnostic CSVs | Per-event diagnostic output |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Precompute D(h) for 15 h-values | ~0.5 seconds total | dist_vectorized calls inside integrand | Vectorize z-array evaluation; only 15 integrals |
| Per-event numerator (unchanged) | ~10 ms per event | fixed_quad with n=50 | Already optimized |
| Full evaluation (531 events x 15 h) | Dominated by L_cat computation | Galaxy matching, MVN evaluation | No change from current |
| z_max(h) computation (15 root-finds) | < 0.1 seconds | fsolve convergence | Good initial guess z0=1.0 |

**Net performance change:** The fix IMPROVES performance. Currently, the denominator is computed per event (531 x 15 = 7965 integrals). After the fix, the denominator is computed once per h (15 integrals). This saves ~7950 integrations.

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| f_i=1 limiting case | Catalog-only recovery | Set catalog_only=True, compare output | Must match current --catalog_only results exactly |
| Denominator = old denominator when limits match | Regression check | Temporarily set denominator limits to local 4-sigma, compare | Must reproduce current results |
| D(h) convergence | Quadrature accuracy | Compute D(h) with n=50, 100, 200; compare | Relative change < 1e-6 between n=100 and n=200 |
| Numerator << D(h) for each event | Self-consistency | Log N_i(h)/D(h) ratio | Must be << 1 (since L_comp is a conditional probability) |
| D(h) > 0 for all h in grid | Non-degeneracy | Check after computation | If D(h)=0, P_det is zero everywhere -- indicates data problem |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| f_i -> 1 (complete catalog) | All events | L_comp drops out; result = L_cat only | Gray et al. (2020) Eq. 9 |
| f_i -> 0 (empty catalog) | No catalog | Result = L_comp only; should be unbiased | Gray et al. (2020) |
| P_det = constant | Uniform detectability | L_comp = integral[p_GW * dVc/dz dz] / integral[dVc/dz dz] | Simplifies to p_GW-weighted average over volume |
| N_events -> infinity | Large N | Bias shrinks as 1/sqrt(N) | Central limit theorem |
| denominator limits = numerator limits | Local window | Recovers current (biased) implementation | Regression test |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
| --- | --- | --- | --- |
| D(h) quadrature convergence | Compare n=50 vs n=100 vs n=200 | Relative change < 1e-6 | Self-consistency |
| MAP h shift toward 0.73 | Run evaluation with fix, compare MAP | MAP should be closer to 0.73 | Current MAP = 0.66/0.68 |
| Bias-vs-N convergence | Compute posterior for N=10, 50, 100, 200, 531 events | Bias should shrink roughly as 1/sqrt(N) | No growing bias |
| Per-event L_comp decomposition | Log L_comp, L_cat, f_i for representative events across h | L_comp should be less h-dependent after fix | Current: U-shaped L_comp anti-correlation |

### Red Flags During Computation

- If D(h) varies by more than a factor of 10 across the h-grid, something is wrong with the P_det grid or the integration limits.
- If L_comp > L_cat for events with f_i > 0.5, the normalization is suspect.
- If the MAP shifts AWAY from 0.73 (i.e., bias gets worse), the fix is introducing a new systematic.
- If >10% of events produce L_comp = 0 or NaN, the numerator integration limits or P_det boundary behavior are broken.
- If D(h) is the same for all h-values, the h-dependence of P_det(d_L(z,h)) is not being captured.

## Common Pitfalls

### Pitfall 1: Using fill_value=None for denominator P_det calls

**What goes wrong:** The current P_det interpolator uses fill_value=None (nearest-neighbor extrapolation). For the denominator integral over the full volume, this artificially extends P_det beyond the injection grid, overestimating detectability.
**Why it happens:** The fill_value=None was set globally in commit 44d5358 to fix L_comp fallback events.
**How to avoid:** Create a separate P_det evaluator for the denominator that uses fill_value=0, OR clip the integration to z_max where P_det grid ends.
**Warning signs:** D(h) is suspiciously large or does not decay to zero at high z.
**Recovery:** Use z_max from the P_det grid as the upper integration limit, ensuring P_det is only evaluated within the grid.

### Pitfall 2: z_max(h) varies with h but is computed from a fixed d_L_max

**What goes wrong:** The P_det grid has a fixed d_L_max (data-dependent). Converting this to z_max via dist_to_redshift(d_L_max, h) gives DIFFERENT z_max for each h. At lower h, d_L_max corresponds to HIGHER z. This is correct physics -- the integration limits SHOULD depend on h.
**Why it happens:** The d_L <-> z mapping depends on h.
**How to avoid:** Compute z_max(h) separately for each h value. Do NOT use a single z_max for all h.
**Warning signs:** D(h) has discontinuities or non-monotonic behavior that doesn't match the smooth P_det * dVc/dz integrand.

### Pitfall 3: Breaking the catalog-only regression test

**What goes wrong:** The fix changes L_comp, which changes the combined likelihood. But when f_i=1 (catalog_only mode), L_comp should not matter.
**Why it happens:** If the code accidentally modifies f_i or L_cat while fixing L_comp.
**How to avoid:** Keep the catalog_only code path completely unchanged. Add a regression test comparing catalog_only output before and after the fix.
**Warning signs:** catalog_only MAP changes after the fix.

### Pitfall 4: Numerator-denominator unit mismatch

**What goes wrong:** The numerator integrand includes p_GW (which has units from the MVN pdf). The denominator integrand does not include p_GW. Both include P_det * dVc/dz. The ratio L_comp = N/D has the right units (probability density) only if the p_GW term has consistent units.
**Why it happens:** The MVN pdf in the numerator uses d_L_fraction = d_L / d_L_det as one coordinate. This is dimensionless, so p_GW has units of 1/(rad^2), which cancel between numerator and denominator via the dOmega integration.
**How to avoid:** Verify that the numerator and denominator integrands differ ONLY by the p_GW factor. No extra normalization constants.
**Warning signs:** L_comp >> 1 or L_comp < 0.

### Pitfall 5: Forgetting that D(h) must use the same P_det convention as the numerator

**What goes wrong:** If the denominator uses fill_value=0 for P_det but the numerator uses fill_value=None, the denominator may be smaller than it should be relative to the numerator.
**Why it happens:** Different P_det boundary conventions for numerator vs denominator.
**How to avoid:** This is intentional per CONTEXT.md. The numerator's 4-sigma window should be within the P_det grid for most events. Flag events where it is not.
**Warning signs:** Events where N_i(h)/D(h) > 1 (impossible if D(h) includes the full volume).

## Level of Rigor

**Required for this phase:** Controlled approximation with numerical validation

**Justification:** This is a numerical fix to match a known correct formula (Gray et al. 2020 Eq. A.19). The mathematical derivation is established. The task is implementation correctness and numerical validation.

**What this means concretely:**

- The fix must match Gray et al. (2020) Eq. A.19 in structure (ratio of integrals with correct limits)
- Quadrature convergence must be verified (n=100 vs n=200 should agree to 1e-6)
- The f_i=1 limiting case must be preserved exactly (regression test)
- MAP shift toward h=0.73 is the primary acceptance criterion
- Bias-vs-N convergence plot is the secondary validation

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Local normalization of L_comp | Full-volume normalization per Gray et al. | This phase | Removes systematic low-h bias |
| fill_value=0 for P_det | fill_value=None globally | Commit 44d5358 | Reduced bias from -9.2% to -6.9%; this phase partially reverts for denominator |

**Superseded approaches to avoid:**

- Local normalization of L_comp (current code): Creates systematic bias that grows with N. This is the bug being fixed.

## Open Questions

1. **Will the fix fully eliminate the bias, or is there a residual from L_cat?**
   - What we know: Catalog-only shows -17.8% bias (60 events), suggesting L_cat itself is biased. The completion term may reduce the bias but not eliminate it.
   - What's unclear: The relative contribution of L_cat bias vs L_comp bias.
   - Impact on this phase: The fix may improve but not fully resolve the bias.
   - Recommendation: Proceed with the fix. If MAP improves but residual bias remains, document it for future investigation (L_cat bias is likely from the fixed galaxy search window or MVN issues, both deferred).

2. **Is the scalar f_i approximation adequate at EMRI distances?**
   - What we know: GLADE completeness is ~21% at >800 Mpc and roughly constant beyond that.
   - What's unclear: Whether the direction-averaged completeness is a good approximation for individual events.
   - Impact on this phase: Minimal. The fix addresses normalization, not completeness modeling.
   - Recommendation: Accept the scalar f_i for now. Document as a known limitation.

3. **Does the P_det grid adequately cover the 4-sigma window for high-z events?**
   - What we know: Some events (those that triggered L_comp fallback before commit 44d5358) have 4-sigma windows extending beyond the grid.
   - What's unclear: How many events are affected and by how much.
   - Impact on this phase: The numerator flagging mechanism will quantify this.
   - Recommendation: Log the number of flagged events. If >10%, the injection grid needs extension (deferred).

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Precomputed D(h) doesn't reduce bias | Bias is not from L_comp normalization | Investigate L_cat bias sources (galaxy search window, MVN) | Separate phase |
| D(h) is numerically unstable | P_det grid is too coarse or has artifacts | Use scipy.integrate.quad with adaptive quadrature | Minor code change |
| MAP overcorrects (bias flips sign) | Full-volume normalization overweights high-z | Check dVc/dz prior; may need rate weighting | Requires rate model (deferred) |

**Decision criteria:** If the MAP shift is less than 0.01 after the fix (i.e., MAP stays at 0.66-0.68), the completion term normalization is not the dominant bias source. Investigate L_cat.

## Sources

### Primary (HIGH confidence)

- Gray et al. (2020), "Cosmological Inference using Gravitational Wave Standard Sirens: A Mock Data Challenge", arXiv:1908.06050 -- Eq. 9 (completeness-corrected likelihood), Eq. A.19 (out-of-catalog term with full-volume normalization)
- Hogg (1999), "Distance Measures in Cosmology", arXiv:astro-ph/9905116, Eq. 28 -- comoving volume element
- Current codebase: bayesian_statistics.py:712-830 -- current L_comp implementation with local normalization

### Secondary (MEDIUM confidence)

- Borghi et al. (2024), "A dark standard siren measurement of the Hubble constant following LIGO/Virgo/KAGRA O4a", arXiv:2404.16092 -- validates methodology is standard in LVK analyses
- Debug investigation: .gpd/debug/h0-posterior-bias-worsening.md -- root cause analysis identifying L_comp normalization as primary bias source
- Dalya et al. (2022), arXiv:2110.06184, Fig. 2 -- GLADE+ catalog completeness data

### Tertiary (LOW confidence)

- arXiv:2503.18887 -- systematic bias study; confirms catalog incompleteness drives low-H0 bias but uses a different methodology (volume-limited catalogs)

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH - Gray et al. (2020) Eq. A.19 is unambiguous about the normalization structure
- Standard approaches: HIGH - precomputed full-volume denominator is the standard method used by gwcosmo
- Computational tools: HIGH - fixed_quad is already used; precomputation is straightforward
- Validation strategies: HIGH - multiple limiting cases and regression tests are available

**Research date:** 2026-04-08
**Valid until:** Indefinite (established methodology; implementation-specific details tied to codebase version)

## Caveats and Alternatives

### Self-Critique

1. **Assumption that might be wrong:** I assume the full-volume denominator is the dominant fix needed. But the catalog-only baseline shows -17.8% bias with 60 events, which suggests L_cat itself carries bias. The L_comp fix addresses the 79% weight completion term, but if the 21% weight L_cat term is also biased, the combined result may still be biased (just less so).

2. **Alternative approach possibly dismissed too quickly:** The systematic bias paper (arXiv:2503.18887) recommends volume-limited catalogs rather than completeness corrections. This is a fundamentally different approach that avoids the completion term entirely. However, it requires re-running the galaxy matching with absolute magnitude cuts, which is a larger change and is not what Gray et al. (2020) implements.

3. **Limitation I may be understating:** The P_det truncation to zero beyond the grid edge introduces a systematic underestimate of detectability near the grid boundary. This affects both the denominator (slightly underestimated D(h)) and the numerator (for events near the grid edge). The net effect on the bias is unclear but likely small compared to the current normalization bug.

4. **Simpler method overlooked?** No. The fix is already the simplest possible change: move one integration limit from local to global. There is no simpler correct approach.

5. **Would a specialist disagree?** A specialist might argue that the scalar f_i approximation is too crude for EMRI dark sirens at z~1-3, where GLADE completeness varies across the sky. They would recommend pixel-based completeness (Finke et al. 2022). However, this is a separate improvement from the normalization fix and is correctly deferred.
