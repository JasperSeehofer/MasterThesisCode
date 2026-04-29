# Zero-Likelihood Diagnostic Report

## Summary

- **Total events:** 412
- **Events with zeros:** 0
- **Empty events (all NaN):** 0
- **All-zeros events:** 0

## Zero-Event Detail

No events with zero likelihoods.

## Zero Distribution by h-bin

| h-value | Number of zeros |
|---|---|
| 0.60 | 0 |
| 0.61 | 0 |
| 0.62 | 0 |
| 0.63 | 0 |
| 0.64 | 0 |
| 0.65 | 0 |
| 0.66 | 0 |
| 0.67 | 0 |
| 0.68 | 0 |
| 0.69 | 0 |
| 0.69 | 0 |
| 0.69 | 0 |
| 0.70 | 0 |
| 0.70 | 0 |
| 0.71 | 0 |
| 0.71 | 0 |
| 0.72 | 0 |
| 0.72 | 0 |
| 0.73 | 0 |
| 0.73 | 0 |
| 0.74 | 0 |
| 0.74 | 0 |
| 0.75 | 0 |
| 0.76 | 0 |
| 0.76 | 0 |
| 0.77 | 0 |
| 0.77 | 0 |
| 0.78 | 0 |
| 0.78 | 0 |
| 0.79 | 0 |
| 0.79 | 0 |
| 0.80 | 0 |
| 0.81 | 0 |
| 0.82 | 0 |
| 0.83 | 0 |
| 0.84 | 0 |
| 0.85 | 0 |
| 0.86 | 0 |

## Root Cause Analysis

Zero likelihoods arise when a detection event has no compatible host galaxy at the given Hubble constant value. **All-zeros** events have no viable host at any h-value, indicating the event lies outside the galaxy catalog coverage entirely. **Low-h-only** zeros occur at smaller h-values where the implied luminosity distance pushes the source beyond the catalog's redshift completeness boundary. **Partial-zeros** arise at coverage boundaries where the galaxy catalog transitions between complete and incomplete.

## Impact on Posterior

Under naive multiplication, a single zero at any h-bin drives the entire joint posterior to zero at that bin. With 0 zero-events, the naive posterior is dominated by the *absence* of catalog coverage rather than the actual likelihood information. Log-space combination with zero-handling strategies mitigates this.