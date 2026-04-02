# Requirements: EMRI Parameter Estimation v1.4

**Defined:** 2026-04-02
**Core Value:** Fix posterior combination numerical instability and deploy before pending cluster evaluation jobs run.

## Analysis

- [x] **ANAL-01**: Diagnostic report documenting zero-likelihood origins per h-bin (which events produce zeros, why no hosts found, catalog coverage gaps)
- [x] **ANAL-02**: Comparison table of all combination methods (naive, Option 1 exclude-zeros, Option 2 per-event-floor, Option 3 physics-floor) with MAP estimates for both with/without BH mass variants

## Numerical Fix

- [x] **NFIX-01**: Log-space posterior accumulation in post-processing combination script — replace `np.prod(likelihoods, axis=0)` with `np.sum(np.log(likelihoods), axis=0)` with shift-before-exp for numerical stability
- [x] **NFIX-02**: Physically motivated likelihood floor (Option 3) in `single_host_likelihood` in `bayesian_statistics.py` — when no host galaxy produces nonzero likelihood, assign a floor based on the faintest catalog galaxy at the error volume boundary. Requires `/physics-change` protocol.
- [x] **NFIX-03**: Replace `check_overflow` with proper underflow detection that catches product-to-zero (not just overflow-to-inf)

## Deployment

- [ ] **DEPL-01**: Updated code pushed to cluster `~/MasterThesisCode` before evaluate jobs start (22 simulate tasks + merge remaining)
- [x] **DEPL-02**: Validation run comparing new posteriors against existing baselines (naive MAP=0.72/0.86, Option 1 MAP=0.68/0.66)

## Post-Processing

- [x] **POST-01**: Standalone combination script that loads per-event posterior JSONs and produces the joint H0 posterior with log-space accumulation + configurable zero-handling (Option 1/2/3)

## Future Requirements (deferred)

- Catalog completeness model for systematic bias correction at high redshift
- Full log-space accumulation inside the evaluate pipeline itself (not just post-processing)

## Out of Scope

- Changing the h-value sweep grid (currently 15 points from 0.6 to 0.86)
- Re-running the simulation pipeline (only the evaluate jobs are in scope)
- Modifications to Pipeline A (`bayesian_inference.py`) — only Pipeline B (`bayesian_statistics.py`) is production

## Traceability

| REQ-ID | Phase | Plan | Status |
|--------|-------|------|--------|
| ANAL-01 | Phase 21 | — | Pending |
| ANAL-02 | Phase 21 | — | Pending |
| NFIX-01 | Phase 21 | — | Pending |
| NFIX-02 | Phase 22 | — | Pending |
| NFIX-03 | Phase 22 | — | Pending |
| DEPL-01 | Phase 23 | — | Pending |
| DEPL-02 | Phase 23 | — | Pending |
| POST-01 | Phase 21 | — | Pending |
