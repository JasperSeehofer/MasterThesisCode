# Phase 15: Code Audit & Fix - Research

**Researched:** 2026-03-31
**Domain:** Bayesian inference / dark siren likelihood / code verification
**Confidence:** HIGH

## Summary

Phase 15 is a code audit and correction phase, not a derivation phase. The reference derivation (Phase 14, `derivations/dark_siren_likelihood.md`) is complete and verified. The task is to (1) systematically confirm each term in `bayesian_statistics.py` matches the derivation, (2) remove the one identified discrepancy (`/(1+z)` at line 646), (3) audit secondary concerns (sky weight placement, denominator consistency, quadrature-vs-MC asymmetry), and (4) run a quick validation to confirm the bias shifts in the expected direction.

The code audit is straightforward because Phase 14 already produced a complete term-by-term mapping (Section 14.1) with 12 terms checked and 1 discrepancy found. The fix is a single-line deletion. The validation requires running the evaluation pipeline at a few h-values and comparing the posterior peak against the pre-fix baseline of h=0.600.

**Primary recommendation:** Remove `/ (1 + z)` from line 646 of `bayesian_statistics.py`, add a reference comment citing Eq. (14.32), and run evaluation at h=0.678 (expected peak) plus 2-3 neighboring h-values to confirm the posterior shifts toward h=0.678.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| `derivations/dark_siren_likelihood.md` Sections 7-14 | derivation | Reference formula for every term in the code | read, compare term-by-term | plan (audit tasks), execution (term checks), verification |
| Phase 14 Verification (`14-VERIFICATION.md`) | verification | Confirms derivation is sound; lists all spot-checks | read | verification (cross-reference) |
| Phase 14 Summary (`14-02-SUMMARY.md`) | summary | Lists the 1 discrepancy and 12 correct terms | read | plan (scope the fix) |
| Commit `15b49a3` (analytic M_z marginalization) | prior artifact | Must not be reverted; confirmed correct by Phase 14 | preserve | execution (do not modify lines 586-643) |
| h=0.600 pre-fix baseline | benchmark | Posterior peak before the fix; must shift toward h=0.678 after | compare | verification (quick-validation run) |
| `bayesian_statistics.py` line 646 | code location | The `/(1+z)` to be removed | modify | execution (the fix), verification (confirm removal) |

**Missing or weak anchors:** None. All required anchors are present and well-documented from Phase 14.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Units | SI (c, G, H0 in km/s/Mpc) | Natural units | derivation Section 1 |
| Fractional parameterization | d_L_frac = d_L/d_L_det, M_z_frac = M*(1+z)/M_z_det | Absolute parameterization | derivation Section 7 |
| Covariance matrix ordering | [phi, theta, d_L_frac, M_z_frac] | Other orderings | code line 180-205 |
| Galaxy mass prior | Source-frame M, Gaussian N(M_gal, sigma_M^2) | Detector-frame | code line 583, derivation Sec. 8.1 |

**CRITICAL: All equations and results below use these conventions. The fractional parameterization is essential -- the covariance matrix entries are already divided by d_L and M at construction (lines 164-205), so the Gaussian is in fractional coordinates centered at 1.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| Eq. (14.32) | "With BH mass" numerator integrand (no /(1+z)) | derivation Sec. 10 | Reference for the numerator audit |
| Eq. (14.41) | Complete "with BH mass" likelihood (boxed) | derivation Sec. 14.3 | Master reference for the full expression |
| Eq. (14.21) | Jacobian absorption identity | derivation Sec. 8.4 | Justification for removing /(1+z) |
| Eq. (14.33) | Denominator expression | derivation Sec. 11 | Reference for denominator audit |
| Eq. (14.31) | mz_integral (analytic Gaussian product) | derivation Sec. 9 | Verify code lines 640-643 |
| Eq. (14.12) | d_L-only single-host likelihood | derivation Sec. 5 | Limiting case target |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Term-by-term code mapping | Matches each code expression to derivation | All audit tasks (AUDT-01 through AUDT-04) | Phase 14 Section 14.1 (already done) |
| Limiting case test | sigma_Mz -> inf should recover d_L-only | Post-fix validation | derivation Sec. 12 |
| MC convergence check | Verify N=10000 MC samples give stable denominator | AUDT-04 | Standard MC error: sigma/sqrt(N) |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Fixed quadrature (n=50) for numerator | Integrand smoothness | Gaussian-shaped integrands | Machine precision for smooth functions | Adaptive quadrature (quad) |
| MC sampling (N=10000) for denominator | 1/sqrt(N) | Finite variance of integrand | sigma/sqrt(10000) ~ 1% if sigma ~ 1 | dblquad (available in testing function) |

## Standard Approaches

### Approach 1: Systematic Code-to-Derivation Mapping (RECOMMENDED)

**What:** Walk through each line of the "with BH mass" numerator and denominator, confirming each code expression matches the corresponding derivation equation. Phase 14 already did this (Section 14.1), so Phase 15 confirms the mapping, implements the fix, and adds reference comments.

**Why standard:** This is the only rigorous way to verify code correctness against a mathematical specification.

**Track record:** Phase 14 successfully mapped 12 terms and found 1 discrepancy using this method.

**Key steps:**

1. Re-read the term-by-term mapping from derivation Section 14.1
2. Confirm each "CORRECT" entry by reading the corresponding code line
3. For the "SPURIOUS" entry (line 646), remove `/ (1 + z)`
4. Add reference comments above every term citing the derivation equation
5. Audit the sky weight (phi, theta) placement -- confirm it appears only inside `gaussian_3d_marginal.pdf()` (line 624) and not as a separate factor
6. Audit the denominator (lines 656-685) for consistency with the numerator in terms of Jacobians, weights, and mass terms
7. Audit the quadrature-vs-MC methodology asymmetry
8. Run quick validation

**Known difficulties at each step:**

- Step 3: The fix is a single-line change (`/ (1 + z)` removal). Must also check the testing function at line 866 which has the same bug.
- Step 7: The MC denominator uses importance sampling from the galaxy priors. The weights simplify to just `p_det` (see code lines 676-685). This is correct but noisy for N=10000.

### Approach 2: Re-derive and Compare (FALLBACK)

**What:** Independently re-derive the likelihood from scratch and compare.

**When to switch:** Only if Phase 14 derivation is found to have errors during the audit.

**Tradeoffs:** Wasteful since Phase 14 already did this with HIGH confidence verification.

### Anti-Patterns to Avoid

- **"Looks wrong, remove it" without derivation backing:** Every code change must cite a specific equation from the derivation. The /(1+z) removal is justified by Eq. (14.21) and the full chain in Sections 8-10.
- **Ad-hoc numerical fix without term-by-term justification:** Do not add fudge factors or rescaling to make the posterior peak match. The fix must follow from the derivation.
- **Declaring the fix correct without re-running evaluation:** The quick-validation run is a contract requirement. The posterior peak must shift from h=0.600 toward h=0.678.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Jacobian absorption identity | p_gal(M) dM = N(M_z_frac; mu_gal_frac, sigma_gal_frac^2) dM_z_frac | derivation Eq. (14.21) | Cite as justification for removing /(1+z) |
| Term-by-term mapping (12 terms) | Table in Section 14.1 | derivation Sec. 14.1 | Confirm each entry; do not re-derive |
| Denominator correctness | No changes needed | derivation Sec. 11, Phase 14 verification | Do not modify denominator lines 656-685 |
| Limiting case | sigma_Mz -> inf recovers d_L-only (CV < 2e-7) | derivation Sec. 12, verification Check 4 | Use as post-fix validation check |
| Sky weight placement | Inside 3D Gaussian, not separate factor | derivation Sec. 2.7 | Confirm and document; do not re-derive |

**Key insight:** Phase 14 already did the hard work of derivation and verification. Phase 15 should cite and apply, not re-derive.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| MC denominator simplification | weights = p_det (after importance sampling cancellation) | code lines 676-685 | Galaxy priors used as proposal distribution |
| "Without BH mass" channel gives h=0.678 | Unbiased baseline | Phase 14 context | Use as reference for expected peak position |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Gray et al. 2020 (arXiv:1908.06050) | Gray, Magee, et al. | 2020 | d_L-only dark siren framework | Structure of likelihood; confirms no /(1+z) in d_L-only case |
| Bishop (2006) PRML | Bishop | 2006 | Conditional decomposition of MVN | Eq. 2.81-2.82 for the conditional mean/variance formulas |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| scipy.integrate.fixed_quad | scipy | Gaussian quadrature for numerator | Already used in production code |
| scipy.stats.multivariate_normal | scipy | 3D/4D Gaussian GW likelihood | Already used in production code |
| scipy.stats.norm | scipy | Galaxy redshift/mass priors | Already used in production code |
| numpy | numpy | Array operations | Already used throughout |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| evaluation pipeline (`python -m master_thesis_code <dir> --evaluate --h_value X`) | Quick validation run | After implementing the fix |
| ruff + mypy | Code quality gates | Before committing any changes |
| pytest | Regression tests | After fix, before commit |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Single h-value evaluation | ~30-60 min on cluster (16 CPUs) | Galaxy catalog lookups + quadrature per host | Run only 3-4 h-values for quick validation |
| Local evaluation (dev machine) | May be slow or infeasible | GLADE catalog size, SimulationDetectionProbability setup | Use cluster for evaluation runs |

**Quick validation strategy:** Run evaluation at h = {0.652, 0.678, 0.704, 0.730} (4 points around expected peak). Compare posterior values to determine peak location. If peak is near h=0.678 (the "without BH mass" channel result), the fix is validated.

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Term-by-term match | Every code term matches derivation | Read Section 14.1 mapping, confirm each line | All 12 terms correct, /(1+z) removed |
| Sky weight placement | phi, theta appear only inside GW Gaussian | Search for phi, theta usage in numerator integrand | phi, theta only in `gaussian_3d_marginal.pdf()` at line 624 and `p_det` at line 619 |
| Denominator consistency | Same physics in numerator and denominator | Compare terms in lines 656-670 to Eq. (14.33) | p_det * p_gal(z) * p_gal(M), no mz_integral, no /(1+z) |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| sigma_Mz -> infinity | Large mass uncertainty | Recovers d_L-only likelihood | derivation Sec. 12, verified numerically |
| h = 0.678 posterior peak (post-fix) | "With BH mass" channel | Peak near h=0.678 (matching "without BH mass" channel) | Phase 14 context; "without BH mass" is unbiased |
| h = 0.600 posterior peak (pre-fix) | Current code | Biased peak at h=0.600 | Known baseline from bias audit |

### Red Flags During Computation

- If the posterior peak does not shift at all after removing /(1+z), the fix is either not applied correctly or the bias has a different root cause
- If the posterior peak shifts beyond h=0.73 (the true value), something else is wrong
- If evaluation crashes after the code change, a syntax error or indentation issue was introduced
- If the "without BH mass" channel results change, the fix accidentally modified the wrong code path

## Common Pitfalls

### Pitfall 1: Modifying the wrong /(1+z)

**What goes wrong:** The file has multiple `(1 + z)` expressions. Only the one at line 646 (in the return statement of `numerator_integrant_with_bh_mass`) is spurious. The `(1 + z)` factors in `mu_gal_frac` (line 633) and `sigma_gal_frac` (line 634) are CORRECT -- they perform the coordinate transformation M -> M_z_frac.

**Why it happens:** Searching for `(1 + z)` without context.

**How to avoid:** Only modify line 646 (the `/ (1 + z)` at the end of the return statement). Leave lines 633-634 untouched.

**Warning signs:** If `mu_gal_frac` or `sigma_gal_frac` no longer contain `(1 + z)`, the wrong line was changed.

**Recovery:** Revert and re-apply carefully.

### Pitfall 2: Forgetting the testing function

**What goes wrong:** `single_host_likelihood_integration_testing()` (line 841-866) has the same `/ (1 + z)` bug at line 866. If only the production function is fixed, the testing function gives different results, causing confusion.

**Why it happens:** The testing function is a copy of the production function with minor differences.

**How to avoid:** Fix both line 646 AND line 866.

**Warning signs:** Testing function and production function give different results.

### Pitfall 3: Breaking the "without BH mass" code path

**What goes wrong:** The `single_host_likelihood` function handles both channels. Accidentally modifying the "without BH mass" path (lines 536-561) invalidates the unbiased baseline.

**Why it happens:** The function is long and both paths share similar structure.

**How to avoid:** Only modify code inside the `if evaluate_with_bh_mass:` block (line 582+). Run "without BH mass" regression check.

### Pitfall 4: MC denominator noise masking the fix

**What goes wrong:** The MC denominator (N=10000 samples, line 672) introduces stochastic noise. If the noise is comparable to the effect of removing /(1+z), the validation run may not show a clear signal.

**Why it happens:** MC has O(1/sqrt(N)) relative error.

**How to avoid:** For the quick validation, focus on the overall posterior shape across multiple h-values, not single-point comparisons. The shift from h=0.600 to h=0.678 is large enough (~13%) to be clearly visible above MC noise.

**Warning signs:** Large variance between evaluation runs at the same h-value.

## Level of Rigor

**Required for this phase:** Code audit with reference comments; numerical validation via evaluation run.

**Justification:** The derivation (Phase 14) provides the mathematical rigor. This phase is about implementation correctness and empirical confirmation.

**What this means concretely:**

- Every modified line must have a reference comment citing the derivation equation
- The fix must follow the physics change protocol (old formula, new formula, reference, dimensional analysis, limiting case)
- The quick-validation run must show a qualitative shift in posterior peak direction
- Exact numerical agreement with the true H0 is NOT required (other approximations and known issues remain)

## State of the Art

Not applicable for a code audit phase. The derivation is current (Phase 14, 2026-03-31).

## Open Questions

1. **Will removing /(1+z) fully resolve the h=0.600 bias?**
   - What we know: The /(1+z) suppresses higher-z contributions, biasing H0 downward. The "without BH mass" channel (which has no /(1+z)) peaks at h=0.678.
   - What's unclear: Whether h=0.678 (not h=0.73) indicates residual bias from other sources (P_det, galaxy catalog completeness, etc.)
   - Impact on this phase: The success criterion is "shift toward h=0.678", not "shift to h=0.73". Residual bias from other sources is out of scope (Phase 11.1 handles P_det).
   - Recommendation: Accept h~0.678 as the expected post-fix result. Further bias investigation is separate work.

2. **Is the MC denominator (N=10000) introducing significant noise?**
   - What we know: Phase 14 flagged this as a numerical concern. The testing function has a dblquad alternative for comparison.
   - What's unclear: The actual variance of the MC estimate relative to the likelihood ratio.
   - Impact on this phase: AUDT-04 should check the MC convergence. If variance is large, document it but do not fix (out of scope unless trivially addressable).
   - Recommendation: Compute MC standard error from existing code (`np.std(weights) / np.sqrt(N_SAMPLES)`) and compare to the integrand magnitude. If relative error > 10%, flag for future work.

3. **Should p_det use M*(1+z) at trial z instead of fixed detection.M?**
   - What we know: Phase 14 noted this at INFO level. The numerator (line 620) passes `detection.M` (the ML mass estimate) to p_det, not `M_gal*(1+z)` at trial z.
   - What's unclear: Whether this is an approximation or an error.
   - Impact on this phase: INFO-level observation. The audit should document it but not fix it (the derivation does not address this detail beyond noting it).
   - Recommendation: Document in the audit. Do not change without further derivation work.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Removing /(1+z) doesn't shift posterior | Other dominant bias source | Investigate P_det, galaxy catalog, or cosmological model | Significant -- new derivation work needed |
| Quick validation cannot run locally | Cluster dependency for SimulationDetectionProbability | Run on cluster via evaluate.sbatch | Need cluster access; adds latency |
| MC denominator too noisy to validate | N=10000 insufficient | Increase to N=100000 or use dblquad | Minor code change but slower runtime |

**Decision criteria:** If the posterior peak does not shift by at least 0.02 in h (from 0.600 toward 0.678), the /(1+z) fix alone is insufficient and additional bias sources must be investigated.

## Specific Code Locations for the Audit

### Production function: `single_host_likelihood` (line 500)

| Line(s) | Code Expression | Derivation Eq. | Status (Phase 14) | Phase 15 Action |
| --- | --- | --- | --- | --- |
| 583 | `norm(loc=possible_host.M, scale=possible_host.M_error)` | p_gal(M) Sec. 8.1 | CORRECT | Confirm, add ref comment |
| 586-611 | Bishop conditional decomposition setup | Eqs. (14.23)-(14.28) | CORRECT | Confirm, ref comments exist |
| 613-647 | `numerator_integrant_with_bh_mass(z)` | Eq. (14.32) | 1 DISCREPANCY | Fix line 646, add ref comment |
| 619-621 | `p_det(d_L, detection.M, phi, theta, h)` | p_det | CORRECT (INFO note: uses detection.M) | Document INFO observation |
| 624-626 | `gaussian_3d_marginal.pdf(...)` | Eq. (14.25) p_GW^(3D) | CORRECT | Confirm sky weight inside |
| 629-630 | `mu_cond` computation | Eq. (14.27) | CORRECT | Confirm |
| 633 | `mu_gal_frac = M*(1+z)/M_z_det` | Eq. (14.22) | CORRECT | Confirm; DO NOT remove (1+z) here |
| 634 | `sigma_gal_frac = sigma_M*(1+z)/M_z_det` | Eq. (14.22) | CORRECT | Confirm; DO NOT remove (1+z) here |
| 640-643 | `mz_integral` (Gaussian product) | Eq. (14.31) | CORRECT | Confirm |
| **646** | **`/ (1 + z)`** | **NOT IN Eq. (14.32)** | **SPURIOUS** | **REMOVE** |
| 656-670 | `denominator_integrant_with_bh_mass_vectorized` | Eq. (14.33) | CORRECT | Confirm |
| 672-685 | MC sampling for denominator | Importance sampling | CORRECT (numerical concern) | Audit convergence |

### Testing function: `single_host_likelihood_integration_testing` (line 699)

| Line | Issue | Action |
| --- | --- | --- |
| 866 | Same spurious `/ (1 + z)` | Remove (same fix as line 646) |
| 773-789 | Commented-out dblquad numerator (no /(1+z) -- uses full 4D Gaussian) | Leave as-is (commented out) |

### TODO flags in the code

| Line | TODO text | Resolution |
| --- | --- | --- |
| 210, 215 | `allow_singular=True` should not be needed | Out of scope for Phase 15 (Fisher matrix conditioning issue) |
| 535, 718 | Sky localization weight in GW likelihood | AUDT-02 resolves this: sky weight IS correctly inside GW likelihood (derivation Sec. 2.7) |

## How to Run Quick Validation

```bash
# On cluster (recommended):
# 1. Submit evaluation at 4 h-values around expected peak
for h in 0.652 0.678 0.704 0.730; do
    python -m master_thesis_code "$RUN_DIR" --evaluate --h_value $h --log_level INFO
done

# 2. Compare posterior JSON files in simulations/posteriors_with_bh_mass/
# Look for: which h-value gives the largest sum of per-detection likelihoods

# On dev machine (if SimulationDetectionProbability data is available):
uv run python -m master_thesis_code <working_dir> --evaluate --h_value 0.678 --log_level INFO
```

**What to check in the output:**
- The posterior value at h=0.678 should be higher than at h=0.600 (post-fix)
- The peak should be near h=0.678, matching the "without BH mass" channel
- The "without BH mass" channel values should be UNCHANGED from pre-fix

## Sources

### Primary (HIGH confidence)

- Phase 14 derivation: `derivations/dark_siren_likelihood.md` Sections 7-14 -- complete first-principles derivation with numerical verification
- Phase 14 verification: `.gpd/phases/14-first-principles-derivation/14-VERIFICATION.md` -- 7/7 contract targets verified, 10/10 physics checks passed
- Phase 14 summary: `.gpd/phases/14-first-principles-derivation/14-02-SUMMARY.md` -- key decisions and equations

### Secondary (MEDIUM confidence)

- Gray et al. (2020), arXiv:1908.06050 -- d_L-only dark siren framework (structural comparison)
- Bishop (2006) PRML Eq. 2.81-2.82 -- conditional decomposition of multivariate normal

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH -- Phase 14 derivation is complete and verified with 10/10 physics checks
- Standard approaches: HIGH -- term-by-term mapping is the standard method; Phase 14 already did it
- Computational tools: HIGH -- using existing pipeline tools; no new dependencies
- Validation strategies: MEDIUM -- quick validation depends on cluster access and assumes the "without BH mass" channel is correct

**Research date:** 2026-03-31
**Valid until:** Indefinitely (code audit methodology does not expire; line numbers may shift with future commits)

## Caveats and Alternatives

1. **What assumption might be wrong?** The assumption that removing /(1+z) is the ONLY change needed. The derivation identified this as the only discrepancy, but the posterior peak at h=0.678 (not 0.73) suggests residual bias from other sources. However, those other sources (P_det, catalog completeness, WMAP cosmology) are explicitly out of scope for this phase.

2. **What alternative approach was dismissed too quickly?** None. The code audit approach is the only sensible approach given that the derivation is complete. Re-deriving would be wasteful.

3. **What limitation am I understating?** The MC denominator noise. With N=10000, the relative MC error could be significant for some host galaxies (those where p_det varies rapidly over the prior). This could affect individual galaxy likelihood ratios, though the sum over ~100s of galaxies should average out the noise.

4. **Is there a simpler method I overlooked?** No. The fix itself IS simple (one line deletion). The audit formalism ensures nothing else is wrong.

5. **Would a specialist disagree?** A specialist might argue that the p_det(detection.M) vs p_det(M*(1+z)) issue (INFO-level observation) should also be fixed in this phase. However, Phase 14 noted this at INFO level (not as a discrepancy), and fixing it would require additional derivation work to determine the correct treatment.
