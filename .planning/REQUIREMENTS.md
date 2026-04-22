# Requirements — v2.2 Pipeline Correctness

**Defined:** 2026-04-21
**Core Value:** Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.

## v2.2 Requirements

Fix all 10 findings from the 2026-04-21 pre-batch audit — two critical coordinate-frame bugs, statistical drift from Gray et al., P_det extrapolation asymmetry, h-hardcode in Fisher CRBs, uniform derivative_epsilon, and latent correctness hygiene — before investing new cluster compute on the next simulation batch and extended P_det injection run. Plan artifact: `~/.claude/plans/i-want-a-last-elegant-feather.md`.

### Coordinate Frame Correctness

- [x] **COORD-01**: Failing-test-first characterization exists: round-trip tests for equatorial↔ecliptic and polar↔latitude conventions, plus baseline counts of events in ±5° ecliptic-equator band from existing CRB CSV
- [x] **COORD-02**: BallTree embedding uses correct polar-to-Cartesian formula `(sin θ cos φ, sin θ sin φ, cos θ)` where θ is polar angle ∈ [0, π]; applied consistently in both `setup_galaxy_catalog_balltree` and `get_possible_hosts_from_ball_tree`
- [x] **COORD-02b**: 4D BallTree (`setup_4d_galaxy_catalog_balltree` / `find_closest_galaxy_to_coordinates`) uses spherical embedding on the sky sub-space — `_polar_to_cartesian(θ, φ)` on the (θ, φ) axes plus normalized z and normalized log M — instead of the flat `θ / π` + `φ / 2π` embedding that exhibits the same latitude-vs-polar bug as COORD-02
- [x] **COORD-03**: GLADE catalog angles are rotated from equatorial J2000 (RA, Dec) to ecliptic SSB (φ, θ_polar) via `astropy.coordinates.SkyCoord.transform_to(BarycentricTrueEcliptic())` during catalog ingestion; docstring documents the stored frame
- [x] **COORD-04**: Sky candidate-host search radius derived from 2×2 sky covariance eigendecomposition (including |sin θ| Jacobian on φ-component) rather than axis-aligned `max(σ_φ, σ_θ)`
- [ ] **COORD-05**: `_map_angles_to_spherical_coordinates` guarded against double-application via idempotency assertion on raw input range

### Statistical Correctness

- [ ] **STAT-01**: L_cat form reconciled with Gray et al. (2020) Eq. 24-25: either proven analytically equivalent to `(1/N) Σ_g N_g/D_g` under the code's implicit uniform 1/N galaxy prior (with docstring derivation + unit test), or replaced with the canonical form via `/physics-change` protocol
- [ ] **STAT-02**: Unit test with 3 synthetic galaxies reproduces both L_cat forms and asserts their agreement (or documents divergence with quantitative reason)
- [ ] **STAT-03**: P_det extrapolation is symmetric between numerator (`single_host_likelihood` integrand) and denominator (`precompute_completion_denominator`); either both use zero-fill or both use NN-fill
- [ ] **STAT-04**: Per-event diagnostic logs the fraction of quadrature weight landing outside the P_det injection grid, for both numerator and denominator integrals; warning fires if >5% for any event

### Parameter Estimation Correctness

- [ ] **PE-01**: `ParameterSpace.set_host_galaxy_parameters(host, h)` threads `h` explicitly through `dist()`; default h removed; regression test confirms 2× d_L ratio between h=0.5 and h=1.0
- [ ] **PE-02**: `derivative_epsilon` is per-parameter: relative (fractional-of-value) for scale parameters (M, mu, d_L, p0), absolute for angles (qS, phiS, qK, phiK, Phi_*) and eccentricity-like parameters (e0, x0, a); validated against Fisher determinant stability on one representative event
- [ ] **PE-03**: `LamCDMScenario.Omega_m` limits correctly ordered (`lower_limit < upper_limit`) with physically sensible range
- [ ] **PE-04**: SNR threshold has a single source of truth (`SNR_THRESHOLD` constant); `Model1CrossCheck`, pre-screen coefficient, evaluation filter, and injection quality gate all read from it
- [ ] **PE-05**: `SPEED_OF_LIGHT_KM_S = C / 1000` derived from `C`, not hardcoded; eliminates 0.07% inconsistency in `comoving_volume_element`

### HPC Hygiene (Safe Wins)

- [x] **HPC-01**: `parameter_estimation.py` is CPU-importable via `_get_xp(use_gpu)` shim; all `cp.*` calls gated; module passes `pytest -m "not gpu"` without `cupy` installed
- [x] **HPC-02**: `_crb_flush_interval` set to 25 (SIGTERM flush retained for tail safety); expected Lustre I/O reduction 5–20 s per SLURM array task
- [x] **HPC-03**: FFT cache is only cleared on explicit memory-pressure signal in `memory_management.py`, not per Fisher iteration; expected 20–100 ms savings per event
- [x] **HPC-04**: Dead `_crop_frequency_domain` removed from `parameter_estimation.py`
- [x] **HPC-05**: `flip_hx=True` in `waveform_generator.py` verified against current `fastlisaresponse` version; obsolete flag removed with `/physics-change` if so, documented if correct

### Visualization (Safe Wins)

- [x] **VIZ-01**: Production figures generated with `apply_style(use_latex=True)` when TeX is available; graceful fallback to mathtext when not; applied in `main.py:generate_figures`
- [x] **VIZ-02**: Static `plot_h0_convergence` displays bootstrap HDI band (16/84 percentile shading) via existing `convergence_analysis.compute_m_z_improvement_bank`

### Verification Gate

- [ ] **VERIFY-01**: Full regression suite passes on CPU (`uv run pytest -m "not gpu"`), including new coordinate round-trip tests
- [ ] **VERIFY-02**: Existing CRBs re-evaluated under fixed frame + fixed L_cat + fixed P_det + eigenvalue sky radius; posterior MAP at h=0.73 within 1% bias; abort new compute if MAP shifts >5% from v2.1 baseline
- [ ] **VERIFY-03**: 27-value h-sweep re-evaluated; convergence figure regenerated; M_z improvement interactive updated
- [ ] **VERIFY-04**: Anisotropy audit: H₀ MAP binned by `|qS − π/2|` quartiles shows no systematic trend (>1σ shift is a blocker)
- [ ] **VERIFY-05**: P_det quadrature-weight-outside-grid diagnostic (STAT-04) logged for every event in the re-evaluation; summary statistic reported

### Staged Cluster Campaign

- [ ] **CAMP-01**: Stage 1 — if VERIFY-05 reports >5% mean extrapolation weight per event, submit densified M×z×d_L injection grid (1.5× in each axis, extended d_L upper bound) to bwUniCluster gpu_h100; re-evaluate posteriors; compare to VERIFY-02 baseline
- [ ] **CAMP-02**: Stage 2 — if VERIFY-04 or Stage 1 reveals residual sky anisotropy, submit sky-dependent P_det injection campaign with (qS, phiS) as additional grid axes (6×12 sky grid); otherwise document that isotropic P_det marginalization is verified sufficient

## Future Requirements (deferred past v2.2)

### Remaining Physics Bugs

- **PHYS-01**: wCDM params `w_0`, `w_a` silently ignored in `dist()` (kwargs accepted but LCDM `hyp2f1` used unconditionally)
- **PHYS-02**: Pipeline A (`bayesian_inference.py`) hardcodes 10% σ(d_L) instead of per-source CRB
- **PHYS-03**: WMAP-era cosmology constants (`Omega_m=0.25`, `H=0.73`) — Planck 2018 best-fit differs (`Omega_m=0.3153`, `H=0.6736`)
- **PHYS-04**: Galaxy redshift uncertainty scaling `(1+z)^3` has no reference; standard forms scale as `(1+z)`

### Deferred Visualization (full overhaul)

- **VIZ-FUT-01**: Galaxy-density Mollweide with Fisher ellipses overlaid (paper-central figure)
- **VIZ-FUT-02**: P_det as pcolormesh heatmap with contours and scatter overlay of observed events
- **VIZ-FUT-03**: Ridgeline/violin posterior stack vs SNR threshold or event count

### Deferred HPC (not safe-win)

- **HPC-FUT-01**: CUDA-stream stencil waveform pipelining (est. 10–30% Fisher wall-time reduction); requires bit-identical validation

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full re-simulation campaign | User decision: re-evaluate existing CRBs under fixed frame, no re-sim needed |
| wCDM equation of state | Known bug PHYS-01, explicitly deferred |
| Pipeline A fixes | Pipeline B is production; A is cross-check only |
| WMAP cosmology update | PHYS-03, explicitly deferred |
| Full visualization overhaul | User decision: safe wins only in v2.2 |
| CUDA-stream refactor | User decision: safe HPC wins only in v2.2 |

## Traceability

| Requirement | Phase | Routing | Status |
|-------------|-------|---------|--------|
| COORD-01 | Phase 35 | GSD | Pending |
| COORD-02 | Phase 36 | GPD | Done |
| COORD-02b | Phase 36 | GPD | Done |
| COORD-03 | Phase 36 | GPD | Done |
| COORD-04 | Phase 36 | GPD | Done |
| COORD-05 | Phase 37 | GSD | Pending |
| STAT-01 | Phase 38 | GSD (if proof) / GPD (if fix) | Pending |
| STAT-02 | Phase 38 | GSD | Pending |
| STAT-03 | Phase 38 | GPD | Pending |
| STAT-04 | Phase 38 | GSD | Pending |
| PE-01 | Phase 37 | GPD | Pending |
| PE-02 | Phase 37 | GPD | Pending |
| PE-03 | Phase 37 | GSD | Pending |
| PE-04 | Phase 37 | GSD | Pending |
| PE-05 | Phase 37 | GSD | Pending |
| HPC-01 | Phase 39 | GSD | Done |
| HPC-02 | Phase 39 | GSD | Done |
| HPC-03 | Phase 39 | GSD | Done |
| HPC-04 | Phase 39 | GSD | Done |
| HPC-05 | Phase 39 | GSD (verify) / GPD (if removed) | Done (KEEP) |
| VIZ-01 | Phase 39 | GSD | Done |
| VIZ-02 | Phase 39 | GSD | Done |
| VERIFY-01 | Phase 40 | GSD | Pending |
| VERIFY-02 | Phase 40 | GPD (runs physics-changed code) | Pending |
| VERIFY-03 | Phase 40 | GSD | Pending |
| VERIFY-04 | Phase 40 | GSD | Pending |
| VERIFY-05 | Phase 40 | GSD | Pending |
| CAMP-01 | Phase 41 | GSD (conditional on VERIFY-05) | Pending |
| CAMP-02 | Phase 42 | GSD (conditional on VERIFY-04 / CAMP-01) | Pending |

**Coverage:**
- v2.2 requirements: 29 total
- Mapped to phases: 29 ✓ (100%)
- Unmapped: 0

**Physics-gate REQ-IDs** (trigger `/physics-change` protocol when executed): COORD-02, COORD-02b, COORD-03, COORD-04, PE-01, PE-02, STAT-03, STAT-01 (conditional on fix), HPC-05 (conditional on removal), VERIFY-02 (as runner of physics-changed code).

---
*Requirements defined: 2026-04-21*
*Last updated: 2026-04-23 — Phase 39 complete: HPC-01..HPC-05, VIZ-01, VIZ-02 checkboxes flipped; traceability status Pending → Done (HPC-05 followed KEEP path, no /physics-change triggered)*
