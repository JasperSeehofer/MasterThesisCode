---
slug: fisher-frame-mismatch-angles
status: resolved
trigger: "Ecliptic/equatorial frame mismatch in Fisher covariance and BallTree search radius"
created: 2026-04-27
updated: 2026-04-27
---

## Symptoms

**Expected behavior:** After Phase 43 ecliptic migration, both sky positions AND their Fisher covariance errors should be in the ecliptic frame for consistent BallTree search radius computation.

**Actual behavior:**
- `detection.py:101-112` reads ecliptic sky positions (`phiS`, `qS`) from CSV ✓
- But reads Fisher covariance errors (`delta_phiS_delta_phiS`, `delta_qS_delta_qS`, `delta_phiS_delta_qS`) which are still in equatorial frame ⚠️
- `handler.py:315-360` BallTree search radius computes `σ_multiplier × √λ_max(J Σ Jᵀ)` where `J = diag(|sin θ_ecl|, 1)` — uses ecliptic θ to rescale azimuthal width, but Σ covariance block is equatorial → mixed frame

**Error magnitude:** O(obliquity) ≈ 23.4° — scales search radius by `|sin θ_ecl|/|sin θ_eq|`, an O(1) multiplicative factor, not directional. MAP=0.730 is valid but search radius is slightly mis-scaled.

**When started:** After Phase 43 ecliptic migration (sky positions corrected to ecliptic, Fisher covariance not updated)

**Reproduction:** Read `detection.py` and `handler.py` — the frame inconsistency is structural, visible in code.

## Current Focus

hypothesis: "Fisher 2×2 sky covariance block is in equatorial frame while sky position and BallTree Jacobian use ecliptic frame; additional angle bugs may exist elsewhere in the codebase"
next_action: "Do comprehensive audit of all angle usage in detection.py, handler.py, parameter_estimation.py, LISA_configuration.py, datamodels/ to find all frame inconsistencies; prepare fix plan"
test: "n/a — diagnosis only"
expecting: "Confirm known bug + find any additional angle frame inconsistencies"
reasoning_checkpoint: ""

## Evidence

## Eliminated

## Resolution

root_cause: "Migration script (migrate_crb_to_ecliptic.py) rotated sky positions (qS, phiS) but NOT the Fisher covariance block. Pre-Phase-43 CSVs that went through migration have ecliptic positions with equatorial Fisher covariance. New simulation runs (post-Phase-43) are CORRECT — Fisher is computed with ecliptic inputs. 6 bugs found total."
fix: "See fix plan in Root Cause Report — primary fix is extending migrate_crb_to_ecliptic.py to also rotate Sigma_ecl = J_rot * Sigma_eq * J_rot^T for all angular Fisher cross-terms. Requires /physics-change protocol. Also add runtime frame validation in Detection.__init__."
verification: "Unit tests for covariance rotation; regression test that posterior mean doesn't shift >1σ after fix"
files_changed: "None yet — diagnosis only"
