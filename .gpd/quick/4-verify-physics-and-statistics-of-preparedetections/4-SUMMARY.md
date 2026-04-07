---
phase: quick-4
plan: 01
status: completed
one_liner: "PrepareDetections sigma chain correct; independent sampling non-standard but defensible (theta-d_L correlation median |rho|=0.43)"
---

# Quick Task 4: Physics Audit of PrepareDetections

## Verdict

The PrepareDetections procedure is **correct in its mathematical implementation** (sigma extraction, truncation bounds, pipeline data flow). The **independent sampling simplification** introduces statistical inconsistency for the theta-d_L pair (median |rho| = 0.43) but the impact on the H0 posterior is likely small because the full covariance is used in the likelihood evaluation. No code changes required for correctness, but correlated multivariate sampling would be more rigorous.

## Finding 1: Sigma Extraction Chain — PASS

The 1-sigma error chain is correct end-to-end:

- **Fisher matrix** Gamma_{ij} computed via 5-point stencil + noise-weighted inner product (parameter_estimation.py:349-378)
- **CRB** = np.linalg.inv(Gamma) at line 395; negative diagonals caught and rejected (lines 398-407)
- **CSV storage** (lines 412-418): lower triangle, so delta_X_delta_X = Gamma^{-1}_{XX} = variance
- **Detection.__init__** (detection.py:93-111): sigma = sqrt(delta_X_delta_X) — correct (sqrt of variance = std dev). Off-diagonal elements stored directly as covariances (no spurious sqrt)
- **convert_to_best_guess_parameters** (detection.py:121-148): truncnorm(a=(low-mu)/sigma, b=(high-mu)/sigma, loc=mu, scale=sigma) — matches scipy convention

**No missing factors of 2, no sqrt errors, no variance/std confusion anywhere in the chain.**

## Finding 2: Truncation Bounds — PASS

| Parameter | Bounds | Physical justification | Effective? |
|---|---|---|---|
| phi (phiS) | [0, 2pi] | Azimuthal angle periodicity | Never reached (sigma_phi ~ 0.05 rad) |
| theta (qS) | [0, pi] | Polar angle range; consistent with ParameterSpace | Never reached (sigma_theta ~ 0.014 rad) |
| d_L | [0, dist(1.5)] | dist(1.5) = 10.89 Gpc; conservative LISA EMRI horizon | Never reached (all events d_L < 0.6 Gpc) |
| M | [1e4, 1e6] M_sun | EMRI BH mass range | Never reached (sigma_M/M ~ 3e-8) |

Minor inconsistency: ParameterSpace defines M.upper_limit = 1e7, but truncnorm uses 1e6. Irrelevant in practice (sigma_M/M ~ 3e-8).

## Finding 3: Independent Sampling Impact — CONDITIONAL PASS

Production data (19 events) correlation coefficients:

| Pair | Median |rho| | % with |rho|>0.3 | Max |rho| |
|---|---|---|---|
| **theta-d_L** | **0.43** | **68%** | **0.62** |
| d_L-M | 0.06 | 42% | 0.55 |
| phi-theta | 0.18 | 16% | 0.45 |
| theta-M | 0.02 | 11% | 0.71 |
| phi-d_L | 0.05 | 11% | 0.53 |
| phi-M | 0.01 | 0% | 0.23 |

**The theta-d_L correlation is significant** (median |rho| = 0.43). Independent draws produce (theta, d_L) pairs inconsistent with the joint measurement distribution.

**Mitigating factors:**
1. The full covariance IS used in the likelihood evaluation (bayesian_statistics.py lines 165-222), so posterior shape is correct
2. Independent draws only affect the likelihood center (galaxy neighborhood), not its shape
3. Over many events, independent draws are unbiased in the mean
4. M is effectively unperturbed (sigma_M/M ~ 3e-8)

## Finding 4: Pipeline Consistency — PASS

- prepare_detections.py overwrites exactly M, luminosity_distance, phiS, qS
- All 105 delta_* covariance columns are byte-identical between raw and prepared CSV
- bayesian_statistics.py reads prepared CSV for Gaussian means and uses preserved covariance for the full matrix
- true_cramer_rao_bounds loaded (line 112) but never used in the evaluate path
- No double-perturbation risk: prepare runs once, evaluate reads the result

## Finding 5: Literature Context

- **Standard practice**: Cutler & Flanagan (1994) and Vallisneri (2008) draw from full multivariate Gaussian N(theta_true, Gamma^{-1}), not independent marginals
- **This pipeline's simplification**: Independent marginal draws are non-standard but defensible because the likelihood evaluation uses the full covariance
- **Dark siren context** (Gray et al. 2020): Likelihood structure more important than mock generation

## Recommendations

1. **No blocking issues** for current results — the procedure is mathematically correct
2. **Optional improvement**: Replace 4 independent truncnorm draws with a single multivariate normal draw using the full 4x4 covariance submatrix, then clip to physical bounds. This would be more rigorous and match standard GW literature practice.
3. **For the paper**: Document the independent sampling as a simplification; note that the likelihood uses the full covariance. The theta-d_L correlation (|rho| ~ 0.4) could be mentioned as a caveat.
4. **Not a bias source**: The known H0 bias (MAP 0.66/0.68 vs true 0.73) is NOT caused by this simplification — the bias is present in both "with" and "without" BH mass channels and tracks the completeness correction.

## Key Files Analyzed

- scripts/prepare_detections.py
- master_thesis_code/datamodels/detection.py:121-148
- master_thesis_code/parameter_estimation/parameter_estimation.py:349-418
- master_thesis_code/bayesian_inference/bayesian_statistics.py:104-222
- master_thesis_code/datamodels/parameter_space.py
- master_thesis_code/constants.py
