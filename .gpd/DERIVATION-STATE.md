# Derivation State (Cumulative)

This file is append-only. Each session appends its equations, conventions,
and results here before the CONTINUE-HERE file is deleted. This prevents
lossy compression across context resets.

---

## Session: 2026-04-24T12:41:42Z | Phase: 43-posterior-calibration-fix

### Equations Established

**Gray et al. (2020) dark siren posterior normalization (Eq. A.19):**
```
log p(h|data) = Σᵢ log[L_i(h)] - N × log[D(h)]
```
where:
- `L_i(h)` = per-event likelihood (catalog + completion terms)
- `D(h) = ∫₀^{z_max(h)} P_det(d_L(z,h)) × (4π c/H_0) × [c z / H_0 E(z)]² / E(z) dz`
  (completeness-corrected detectable comoving volume, units Mpc³/sr)
- The `-N log D(h)` term is the survey selection function normalization

**Critical finding:** `extract_baseline` in `evaluation_report.py:221` omits `D(h)`:
```python
log_posts = [r["log_posterior"] for r in posteriors]
map_idx = int(np.argmax(log_posts))  # no D(h) correction
```
Since `D(h)` grows monotonically with `h` (larger h → smaller d_L → more detectable events),
omitting it makes the raw `log_posterior` increase monotonically with h → MAP=h_max=0.86.

### Conventions Applied

- Sky frame (post-Phase-36): qS = ecliptic colatitude (polar angle from north ecliptic pole),
  phiS = ecliptic longitude; GLADE equatorial RA/Dec rotated via astropy BarycentricTrueEcliptic
- CRBs on disk (`simulations/prepared_cramer_rao_bounds.csv`): generated BEFORE Phase 36,
  so stored qS/phiS are in EQUATORIAL frame (not ecliptic)

### Intermediate Results

- H1 CONFIRMED (code trace): `extract_baseline` + `combine_posteriors` both lack D(h).
  `BayesianStatistics.evaluate()` DOES include D(h) via `precompute_completion_denominator` (line 350-358).
- H2 CONFIRMED (code trace): `Detection.__init__` reads `parameters["phiS"]`, `parameters["qS"]`
  from CRB CSV as BallTree search center. v2.2 BallTree is ecliptic; CRBs are equatorial → mismatch.
  `handler.py:get_possible_hosts_from_ball_tree(phi=self.detection.phi, theta=self.detection.theta)`
  searches at equatorial position in ecliptic catalog.
- VERIFY-02 MAP=0.735 was comparing v2.1 archive posteriors against themselves (not a real v2.2 run).
- KEY UNKNOWN: What does `--evaluate` (with D(h)) give on current CRBs (equatorial sky angles)?

### Approximations Used

- Gaussian GW measurement errors (Fisher matrix, valid SNR≥20)
- Completeness correction: f(z,h) × L_cat + (1-f(z,h)) × L_comp per Gray et al. (2020) Eq. 9

