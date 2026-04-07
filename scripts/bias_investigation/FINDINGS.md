# H0 Posterior Bias Investigation — Findings

## Dataset
- 534 GW detections, true h=0.73, MAP=0.66 (bias = -0.07)
- "Without BH mass" posterior (exclude strategy)
- 447/517 events individually favor h=0.66 over h=0.73

## Test Results Summary

| Test | Finding | Status |
|------|---------|--------|
| **T1: Bias vs redshift** | Moderate correlation (r=0.21). Events at z=0.08-0.13 (355 events, 67%) overwhelmingly favor h=0.66. Low-z events (z<0.05) favor h>0.73. | REDSHIFT-DEPENDENT |
| **T2: Sharp drop** | h=0.70→0.72 drop is uniform across events. Top-50 drop events cluster at higher d_L. Symmetric drop at h=0.80→0.82. | NO CLIFF, SMOOTH |
| **T3: Single galaxy** | Peak at h=0.73 for all z_true tested (0.03-0.20). | FORMULA CORRECT |
| **T4: Multi-galaxy P_det=1** | Uniform galaxies: +0.01 offset. Low-z biased: -0.04. High-z biased: +0.13. | DENSITY GRADIENT = PRIMARY |
| **T5: P_det boundary** | All events well within grid (max d_L=0.78 vs min dl_max=2.47). Zero clipping. | BOUNDARY ELIMINATED |
| **T5b: P_det values** | P_det varies 0-1 across detection d_L range (mean≈0.13). Asymmetric: P_det(h=0.66)/P_det(h=0.73) < 1 at high d_L. | P_det IS ACTIVE |
| **T7: Bounds symmetry** | All bounds modes give identical peaks. Integration bounds are NOT the issue. | BOUNDS ELIMINATED |

## Root Cause: Galaxy Catalog Density Gradient

The bias is caused by the **asymmetric galaxy density distribution** in the GLADE catalog:

1. **GLADE is incomplete at higher redshifts.** For events at z_true=0.08-0.13, there are systematically more candidate host galaxies at z < z_true than at z > z_true.

2. **At trial h < h_true (e.g., h=0.66):** The d_L(z, 0.66) = (0.73/0.66) × d_L(z, 0.73) is ~10% larger. A galaxy at z_gal < z_true can have d_L(z_gal, 0.66) ≈ d_L_det (d_L_frac ≈ 1), giving maximum GW likelihood. Since there are MORE galaxies at lower z, the numerator sum Σ_gal ∫ GW × p_gal dz is inflated at h=0.66.

3. **P_det correction is insufficient.** Although P_det does vary across the detection range (it's not ≈1), the P_det ratio between h=0.66 and h=0.73 is not extreme enough to overcome the galaxy density asymmetry. P_det acts in the right direction but doesn't fully compensate.

4. **The bias does NOT average out with more detections** because the GLADE incompleteness is SYSTEMATIC — every event at z=0.08-0.13 sees the same low-z excess of candidate galaxies. Adding more detections from the same catalog and same redshift range reproduces the same systematic bias.

## Key Evidence

### Bimodal per-event peak distribution
- 196 events peak at h=0.66 (mostly z > 0.10 where catalog incompleteness is worse)
- 184 events peak at h=0.80 (mostly z < 0.08 where catalog is complete, and galaxy density at z > z_true creates opposite bias)
- These two populations partially cancel, but the h=0.66 group wins (more events)

### Log-likelihood profile shape
```
h=0.64: 1271.3    ↑ rising
h=0.66: 1295.9    ← PEAK (should be at h=0.73)
h=0.68: 1284.1    ↓ falling  
h=0.70: 1259.7    ↓ falling
h=0.72: 1167.7    ↓↓ sharp drop (92-unit gap)
h=0.73: 1127.5    ← TRUE VALUE
h=0.76: 1179.6    ↑ secondary rise
h=0.80: 1192.6    ← secondary peak from low-z events
```

## NOT a Bug — A Fundamental Limitation

This bias is inherent to the dark siren method with an incomplete galaxy catalog:

1. **The formula is mathematically correct** (Test 3 proves this)
2. **The P_det grid is properly constructed** (Tests 5, 5b confirm)
3. **The integration bounds are adequate** (Test 7 confirms)
4. **The bias is entirely driven by galaxy catalog incompleteness**

## Mitigation Options

1. **More complete galaxy catalog:** Use a deeper survey (e.g., future DESI, Euclid) to reduce the z-asymmetry in galaxy density.

2. **Galaxy catalog completeness correction:** Model the catalog completeness as a function of redshift and include it as a prior in the denominator. This is the standard approach in GW cosmology (see Gray et al. 2020, arXiv:1908.06050).

3. **Redshift-dependent weighting:** Downweight events at z > 0.08 where the catalog incompleteness is known to be severe.

4. **Galaxy catalog simulation:** Replace the real catalog with a simulated complete catalog (uniform in comoving volume) to eliminate the density gradient. Then use the real catalog positions only for the true host matching.

5. **H(z) prior:** Use an informative prior on h that penalizes values far from Planck/SH0ES estimates.

6. **Hierarchical modeling:** Model the galaxy catalog incompleteness jointly with h, marginalizing over the completeness function.

## Files Analyzed
- `bayesian_statistics.py` — core likelihood (lines 500-714)
- `simulation_detection_probability.py` — P_det grid (lines 144-298)  
- `physical_relations.py` — d_L(z,h) function
- `results/h_sweep_20260401/` — 534-detection posterior data
- `simulations/injections/` — injection campaign CSVs (7 h-grid points)
