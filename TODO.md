# TODO's

## Physics Fixes (confirmed bugs — Physics Change Protocol required before each)

These all require presenting old formula / new formula / reference / dimensional analysis /
limiting case to the user for approval before any code is written (see CLAUDE.md).

- [ ] **[CRITICAL]** Fix comoving volume formula in `datamodels/galaxy.py:121`
      Exponent 2 → 3, prefactor 4π → 4π/3. Ref: Hogg (1999) arXiv:astro-ph/9905116 Eq. (28).
      `cv_grid = (4/3) * np.pi * (SPEED_OF_LIGHT / h0) ** 3 * cumulative_integral**3`

- [ ] **[HIGH]** Switch Fisher matrix to five-point stencil in `parameter_estimation.py:336`
      Replace `finite_difference_derivative()` call with `five_point_stencil_derivative()`.
      Ref: Vallisneri (2008) arXiv:gr-qc/0703086; Cutler & Flanagan (1994) PRD 49, 2658.

- [ ] **[MEDIUM]** Add galactic confusion noise to LISA PSD in `LISA_configuration.py`
      Implement `galactic_confusion_noise(frequencies, T_obs)` and add to
      `power_spectral_density_a_channel()`. Constants already defined in `constants.py:77–83`.
      Ref: Babak et al. (2023) arXiv:2303.15929 Eq. (17) and Table 1.

- [ ] **[MEDIUM]** Fix silent wCDM fallback in `physical_relations.py:72`
      Either (a) remove `w_0`, `w_a` args from `dist()` and document the ΛCDM assumption, or
      (b) fall back to numerical integration via `hubble_function()` when `w_0 ≠ -1` or `w_a ≠ 0`.
      Ref: Hogg (1999) arXiv:astro-ph/9905116 Eq. (14–16).

- [ ] **[MEDIUM]** Use per-source Fisher-matrix σ(d_L) in `bayesian_inference/bayesian_inference.py`
      Replace hardcoded `FRACTIONAL_LUMINOSITY_ERROR * d_L` with
      `delta_luminosity_distance_delta_luminosity_distance` from the `Detection` dataclass.

- [ ] **[LOW]** Update fiducial cosmological parameters in `constants.py:29–30` to Planck 2018:
      `OMEGA_M = 0.3153`, `OMEGA_DE = 0.6847`, `H = 0.6736`.
      Ref: Planck Collaboration (2020) arXiv:1807.06209 Table 2.

- [ ] **[LOW]** Document or fix galaxy redshift uncertainty scaling in `datamodels/galaxy.py:64`
      Current `(1+z)³` formula has no reference; add citation or switch to standard
      photometric `σ_z = 0.05(1+z)` or spectroscopic `σ_z = 0.001(1+z)`.

- [ ] **[IMPORTANT]** Designate and document the production Bayesian pipeline
      Document that Pipeline B (`BayesianStatistics` in `cosmological_model.py`) is the
      science-grade implementation and Pipeline A (`BayesianInference`) is a dev cross-check.
      Add a module-level docstring and a note in README.

## Physics / Science (pre-existing)

- [ ] coordinate transformation to orbital motion around sun
- [ ] check _s parameters. Barycenter same as orientation of binary wrt fixed frame
- [ ] check spin limits for parameter a
- [ ] What happens with the inclination for the Schwarzschild waveforms, because it is defined w.r.t. the angular momentum of the MBH.
- [ ] compute derivative w.r.t. sky localization in ssb again.
- [ ] use second detector from LISA
- [ ] function integration: has been reduced to positive integral because of negative frequency == complex conjugate. atm fs contains negative frequencies which is wrong

## Code Health (remaining)

- [ ] Extract `BayesianStatistics` from `cosmological_model.py` (~1611 lines) to
      `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- [ ] Fix unconditional `import cupy` at module level in `LISA_configuration.py`
      (blocks import on CPU machines without `try/except` guard)
- [ ] Raise test coverage gate in `pyproject.toml` (`fail_under`) above 25% as more
      tests are added; target ≥ 50% by thesis submission
- [ ] Tag git release `v0.1.0` once current branch is merged: `git tag v0.1.0`
- [ ] Add Codecov integration to CI for a coverage badge in README

## Done (Phase 10 — 2026-03-11, Plotting Refactor)

- [x] Create `master_thesis_code/plotting/` subpackage with factory functions
      (`_style.py`, `_helpers.py`, `emri_thesis.mplstyle`, `simulation_plots.py`,
      `bayesian_plots.py`, `evaluation_plots.py`, `model_plots.py`, `catalog_plots.py`,
      `physical_relations_plots.py`)
- [x] Create `callbacks.py` with `SimulationCallback` Protocol; wire into `data_simulation()`
- [x] Delete `ScientificPlotter`, `IS_PLOTTING_ACTIVATED`, `if_plotting_activated` decorator
- [x] Remove all matplotlib imports from computation modules (only in `plotting/` now)
- [x] Remove `__init__` plot side effects from `glade_completeness.py`, `detection_horizon.py`,
      `detection_distribution_simplified.py`, `emri_distribution.py`, `detection_fraction.py`
- [x] Extract ~1900 lines of plotting from `cosmological_model.py` (shrunk to ~1611 lines)
- [x] Extract plotting from `evaluation.py`, `handler.py`, `physical_relations.py`,
      `galaxy.py`, `emri_detection.py`, `bayesian_inference.py`, `memory_management.py`,
      `LISA_configuration.py`, `parameter_estimation.py`
- [x] Add `--generate_figures` CLI argument (stub handler)
- [x] Add `master_thesis_code_test/plotting/test_style.py` (9 tests)
- [x] Coverage increased from 28.83% to 36.19%

## Done (Phase 9 — 2026-03-10)

- [x] Physics & mathematics review: README "Scientific Background and Known Limitations" section
- [x] All eight confirmed issues documented in README, TODO, CHANGELOG, and CLAUDE.md
- [x] GitHub issues filed for all confirmed bugs

## Done (Phase 8 — 2026-03-10)

- [x] Add LICENSE file (MIT)
- [x] Add CONTRIBUTING.md and .editorconfig
- [x] Add pytest-cov + coverage gate (25%); CI uploads coverage.xml artifact
- [x] Add pip-audit to dev extras + CI security step
- [x] Add Dependabot (weekly pip + GitHub Actions updates)
- [x] Add `--seed` CLI arg; seed numpy in main(); write run_metadata.json per run
- [x] Fix `get_samples_from_comoving_volume` PNG side-effect (`save_plot=False`)
- [x] Rename `ParameterSpace.dist` → `luminosity_distance` (field, symbol, CSV cols)
- [x] Fix missed `"dist"` column references in `scripts/prepare_detections.py` and
      `scripts/estimate_hubble_constant.py`; patch existing simulation CSVs