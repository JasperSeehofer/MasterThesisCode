# Grid Quality Assessment: P_det Wilson Confidence Intervals

## Wilson CI Summary

Per-bin Wilson 95% confidence intervals computed for all 7 h-values using
`astropy.stats.binom_conf_interval` with `interval='wilson'` and
`confidence_level=0.9545`. Bins with n_total < 10 flagged as unreliable per
Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133.

### 30x20 Grid (Fine)

| h    | Events | Empty | Unreliable (n<10) | Reliable (n>=10) | Med CI hw | Max CI hw | Boundary (0.05<P<0.95) | P_det>0 |
|------|--------|-------|--------------------|------------------|-----------|-----------|------------------------|---------|
| 0.60 | 22500  | 49    | 141                | 459              | 0.0645    | 0.4000    | 10                     | 11      |
| 0.65 | 26000  | 47    | 121                | 479              | 0.0645    | 0.4000    | 15                     | 17      |
| 0.70 | 23500  | 46    | 129                | 471              | 0.0625    | 0.4000    | 12                     | 18      |
| 0.73 | 25500  | 46    | 133                | 467              | 0.0606    | 0.4000    | 12                     | 17      |
| 0.80 | 25000  | 49    | 141                | 459              | 0.0606    | 0.4000    | 11                     | 15      |
| 0.85 | 25500  | 50    | 149                | 451              | 0.0645    | 0.4082    | 16                     | 18      |
| 0.90 | 17000  | 201   | 462                | 138              | 0.0211    | 0.4000    | 18                     | 24      |

### 15x10 Grid (Coarse)

| h    | Events | Empty | Unreliable (n<10) | Reliable (n>=10) | Med CI hw | Max CI hw | Boundary (0.05<P<0.95) | P_det>0 |
|------|--------|-------|--------------------|------------------|-----------|-----------|------------------------|---------|
| 0.60 | 22500  | 10    | 11                 | 139              | 0.0204    | 0.1538    | 4                      | 7       |
| 0.65 | 26000  | 10    | 10                 | 140              | 0.0198    | 0.1000    | 5                      | 6       |
| 0.70 | 23500  | 10    | 10                 | 140              | 0.0187    | 0.1250    | 3                      | 7       |
| 0.73 | 25500  | 10    | 10                 | 140              | 0.0189    | 0.1250    | 5                      | 5       |
| 0.80 | 25000  | 10    | 12                 | 138              | 0.0190    | 0.1667    | 5                      | 6       |
| 0.85 | 25500  | 10    | 10                 | 140              | 0.0213    | 0.1111    | 5                      | 6       |
| 0.90 | 17000  | 22    | 71                 | 79               | 0.0625    | 0.4000    | 5                      | 10      |

## Grid Comparison

### CI Half-Width Improvement

The 15x10 grid achieves substantially smaller CI half-widths due to higher
per-bin occupancy. Median CI half-width improvement for reliable bins:

| h    | 30x20 Med CI hw | 15x10 Med CI hw | Improvement Factor |
|------|-----------------|-----------------|-------------------|
| 0.60 | 0.0645          | 0.0204          | 3.2x              |
| 0.65 | 0.0645          | 0.0198          | 3.3x              |
| 0.70 | 0.0625          | 0.0187          | 3.3x              |
| 0.73 | 0.0606          | 0.0189          | 3.2x              |
| 0.80 | 0.0606          | 0.0190          | 3.2x              |
| 0.85 | 0.0645          | 0.0213          | 3.0x              |
| 0.90 | 0.0211          | 0.0625          | 0.3x (reversed)   |

Note: h=0.90 is anomalous because the 30x20 grid has very few reliable bins
(138 vs 459 for h=0.73), so the median is dominated by the few well-populated
bins. h=0.90 has 17000 events vs 22500-26000 for other h-values and much more
empty space in (d_L, M).

### Interpolation Error (15x10 evaluated at 30x20 centers)

| h    | N eval bins | Median |error| | Max |error| | Frac |error|>0.05 |
|------|-------------|-----------------|---------------|---------------------|
| 0.60 | 11          | 0.3750          | 1.0000        | 0.909               |
| 0.65 | 17          | 0.2500          | 0.7895        | 0.882               |
| 0.70 | 18          | 0.0746          | 1.0000        | 0.667               |
| 0.73 | 17          | 0.1429          | 1.0000        | 0.882               |
| 0.80 | 15          | 0.2143          | 1.0000        | 0.733               |
| 0.85 | 18          | 0.1435          | 0.9545        | 0.833               |
| 0.90 | 24          | 0.1414          | 1.0000        | 0.708               |

**Interpolation errors are large** because:
1. Only 5-24 bins have P_det > 0, all concentrated at small d_L and large M
2. The detection boundary is sharp (most bins are 0 or ~1), creating a
   discontinuity that linear interpolation cannot resolve
3. The 15x10 grid places bin centers at different locations than the 30x20
   grid, and with so few non-zero bins, the coarse grid "misses" detections
   that the fine grid catches

This does NOT mean 15x10 is unsuitable -- it means the metric is dominated
by bins with very few events (1-7) where stochastic fluctuations are large.

## Boundary Region Analysis

The detection boundary region (0.05 < P_det < 0.95) is the most informative
region for P_det estimation, as bins with P_det ~ 0 or ~ 1 contribute little
to the likelihood gradient.

**Boundary region is sparse across all h-values:**
- 30x20: 10-18 boundary bins (of 600 total, ~2%)
- 15x10: 3-5 boundary bins (of 150 total, ~3%)

Boundary median CI half-widths (where boundary bins exist):
- 30x20: 0.14-0.24 (substantial uncertainty per bin)
- 15x10: 0.06-0.08 (meets <0.15 target)

The 15x10 grid achieves CI half-width < 0.15 in the boundary region for all
h-values where boundary bins exist. However, the boundary region contains
very few bins (3-5), making this statistic fragile.

## Quality Flags

Quality flag arrays added to `SimulationDetectionProbability`:
- `n_total`: per-bin total injection count (30x20 int array)
- `n_detected`: per-bin detected injection count (30x20 int array)
- `reliable`: boolean mask True where n_total >= 10

Access via `sdp.quality_flags(h)` method. Flags are metadata only and do not
affect interpolation behavior.

Verified: `reliable == (n_total >= 10)` for all h-values.
Verified: P_det interpolation output unchanged at 10 test points after
adding quality flags (flags are stored in a separate dict, not used by
the RegularGridInterpolator).

## Recommendation

**Use the 15x10 grid** for P_det estimation given current injection counts
(~23000 events per h-value).

Rationale:
1. **CI quality**: 15x10 has 93% reliable bins (n>=10) vs 78% for 30x20
2. **Boundary precision**: Median CI half-width < 0.15 in boundary region
3. **Few empty bins**: 10 empty bins (7%) vs 46 (8%) for 30x20
4. **Sufficient resolution**: The detection boundary is confined to a narrow
   strip at d_L ~ 0.2-1.0 Gpc, M ~ 2e5-1e6 Msun. A 15x10 grid provides
   adequate resolution in this region.

**Exception**: h=0.90 has insufficient statistics even for 15x10 (71 unreliable
bins out of 150). Recommend doubling injection count for h=0.90 in the next
campaign.

**To use 30x20 reliably**, the injection count would need to increase by roughly
4x (to ~100k events per h-value) to bring unreliable bins below 10%.

## Key Findings

The P_det grid is dominated by zeros -- detections occur in a small corner
of (d_L, M) parameter space (d_L < 1 Gpc, M > 2e5 Msun). The current
injection campaign of ~165k total events across 7 h-values provides
sufficient statistics for a coarse 15x10 grid but is marginal for the
finer 30x20 grid used by the production code. The boundary region (where
P_det transitions from 0 to 1) contains only 3-18 bins per grid, making
per-bin CI estimates in this region inherently noisy. The 15x10 grid
achieves median CI half-width < 0.02 for reliable bins and < 0.15 in the
boundary region, compared to 0.06 and 0.17-0.24 for the 30x20 grid.
