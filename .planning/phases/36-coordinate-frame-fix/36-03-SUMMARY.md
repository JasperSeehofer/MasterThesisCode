---
phase: 36-coordinate-frame-fix
plan: "03"
subsystem: galaxy-catalogue
tags: [physics, coordinate-frame, balltree, fisher-matrix, regression-anchor]
dependency_graph:
  requires: ["36-02"]
  provides: ["COORD-04", "36-superset-regression.pkl"]
  affects: ["bayesian_statistics.py", "handler.py"]
tech_stack:
  added: []
  patterns:
    - "Eigenvalue radius: σ_mult × √λ_max(J Σ Jᵀ) with J = diag(|sin θ|, 1) for sky search on unit sphere"
    - "Backward-compatible default kwarg (cov_theta_phi=0.0) at signature tail to respect non-default-follows-default rule"
    - "Regression anchor pickle (D-24 schema) committed as binary artifact for Phase 40 VERIFY-02"
key_files:
  created:
    - scripts/generate_phase36_regression.py
    - .planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl
  modified:
    - master_thesis_code/galaxy_catalogue/handler.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
decisions:
  - "Placed cov_theta_phi at tail of get_possible_hosts_from_ball_tree signature (after sigma_multiplier) to satisfy Python non-default-follows-default rule; consistent with D-21"
  - "Old candidate set in regression script uses axis-aligned radius directly on corrected BallTree (option b from uncertainty_markers), isolating COORD-04 effect cleanly"
  - "Pre-fix and post-fix qS/phiS are identical for the ML point (waveform frame unchanged; only catalog frame changed in COORD-03)"
metrics:
  duration: "~4 minutes (10s script wall clock)"
  completed: "2026-04-22"
  tasks_completed: 4
  files_changed: 4
---

# Phase 36 Plan 03: COORD-04 Eigenvalue Sky Search Radius — Summary

**One-liner:** Eigenvalue sky search radius σ_mult·√λ_max(JΣJᵀ) with |sin θ| Jacobian replaces axis-aligned max(σ_φ,σ_θ) in `get_possible_hosts_from_ball_tree`; regression pickle confirms new ⊇ old on D-23 reference event.

## What Was Done

### Task 1: Extend signature + replace radius formula (handler.py)

Extended `GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree` with:
- New `cov_theta_phi: float = 0.0` kwarg at the signature tail (backward-compatible)
- Added full NumPy-style docstring
- Replaced `radius = max(phi_sigma, theta_sigma) * sigma_multiplier` with eigenvalue formula:

```python
# Eq. (eigenvalue of J Σ Jᵀ on 2×2 Fisher sky block); COORD-04 per
# .planning/phases/36-coordinate-frame-fix/36-CONTEXT.md D-22.
# J = diag(|sin θ|, 1) rescales the azimuthal std to great-circle distance
# on the unit sphere (ds² = dθ² + sin²θ dφ² — see detection.py:15-40).
sigma_matrix = np.array(
    [[phi_sigma**2, cov_theta_phi], [cov_theta_phi, theta_sigma**2]]
)
jacobian = np.diag([abs(np.sin(theta)), 1.0])
sigma_scaled = jacobian @ sigma_matrix @ jacobian.T
lambda_max = float(np.linalg.eigvalsh(sigma_scaled).max())
radius = float(sigma_multiplier * np.sqrt(max(lambda_max, 0.0)))
```

**Limiting cases verified:**
- θ = π/2, C=0, σ_φ=σ_θ=σ → radius = σ_mult·σ (isotropic equator) ✓
- θ → 0, σ_φ >> σ_θ, C=0 → radius = σ_mult·σ_θ (pole degeneracy suppressed) ✓
- θ = π/2, C = 0.5σ², σ_φ=σ_θ=σ → radius = σ_mult·√1.5·σ > old radius ✓

### Task 2: Update caller at bayesian_statistics.py

Added `cov_theta_phi=self.detection.theta_phi_covariance` to the `get_possible_hosts_from_ball_tree` call at line 778. `Detection.theta_phi_covariance` is already populated from `parameters["delta_phiS_delta_qS"]` (datamodels/detection.py:107). No datamodel changes needed.

### Task 3: Create regression CLI + produce pickle

Created `scripts/generate_phase36_regression.py` (295 LOC) following `merge_cramer_rao_bounds.py` template. Wall clock: 10s (within 2-min cap).

Pickle at `.planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl`:
- D-24 schema: 9 keys including `old_candidate_indices`, `new_candidate_indices`, `fisher_sky_2x2`, `git_commit`
- Reference event: row 29, qS=1.4468 rad, |qS-π/2|=0.1240 rad
- Old radius: 0.078153 rad → 18 candidates; New radius: 0.078213 rad → 18 candidates
- `old ⊆ new: True` (SC-4 satisfied)
- Fisher 2×2: [[7.29e-05, 1.07e-04], [1.07e-04, 2.71e-03]], det=1.87e-07 > 0 (PD)

### Task 4: Atomic commit

Commit `b2ef9c9` — `[PHYSICS] COORD-04: eigenvalue sky search radius on 2×2 Fisher covariance with |sin θ| Jacobian`. Pre-commit hooks (ruff, ruff-format, mypy) all passed. Test suite: 517 passed, 0 failures, 0 regressions.

## Deviations from Plan

None — plan executed exactly as written. The uncertainty_markers option (b) for the old radius — replicate axis-aligned radius inline in the script — was the cleanest approach and was used as documented.

## Verification Results

| Acceptance Test | Result |
|----------------|--------|
| test-isotropic-limit (θ=π/2, C=0, σ_φ=σ_θ=0.1) | PASSED (radius=0.2 to 1e-15) |
| test-pole-limit (θ=1e-6, σ_φ=1.0, σ_θ=0.01) | PASSED (radius=0.02 to 1e-15) |
| test-regression-superset (D-23 event) | PASSED (old ⊆ new, 18 ⊆ 18) |
| test-caller-signature (grep in bayesian_statistics.py) | PASSED (1 match) |
| test-integration-smoke (pytest -m "not gpu and not slow") | PASSED (517 passed) |
| test-coordinate-roundtrip (9 tests) | PASSED (9 passed, 0 xfail) |

## Self-Check

- handler.py modified: confirmed (eigenvalue radius, new docstring, cov_theta_phi kwarg)
- bayesian_statistics.py modified: confirmed (cov_theta_phi=self.detection.theta_phi_covariance at line 778)
- scripts/generate_phase36_regression.py created: confirmed (295 LOC)
- .planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl created: confirmed (533 bytes)
- Commit b2ef9c9 exists: confirmed via git log

## Self-Check: PASSED
