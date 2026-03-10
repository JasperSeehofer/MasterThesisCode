# TODO's

## Physics / Science

- [ ] coordinate transformation to orbital motion around sun
- [ ] check _s parameters. Barycenter same as orientation of binary wrt fixed frame
- [ ] check spin limits for parameter a
- [ ] What happens with the inclination for the Schwarzschild waveforms, because it is defined w.r.t. the angular momentum of the MBH.
- [ ] compute derivative w.r.t. sky localization in ssb again.
- [ ] use second detector from LISA
- [ ] function integration: has been reduced to positive integral because of negative frequency == complex conjugate. atm fs contains negative frequencies which is wrong

## Code Health (remaining)

- [ ] Extract `BayesianStatistics` from `cosmological_model.py` (~3530 lines) to
      `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- [ ] Fix unconditional `import cupy` at module level in `LISA_configuration.py`
      (blocks import on CPU machines without `try/except` guard)
- [ ] Raise test coverage gate in `pyproject.toml` (`fail_under`) above 25% as more
      tests are added; target ≥ 50% by thesis submission
- [ ] Tag git release `v0.1.0` once current branch is merged: `git tag v0.1.0`
- [ ] Add Codecov integration to CI for a coverage badge in README

## Done (Phase 8 — 2026-03-10)

- [x] Add LICENSE file (MIT)
- [x] Add CONTRIBUTING.md and .editorconfig
- [x] Add pytest-cov + coverage gate (25%); CI uploads coverage.xml artifact
- [x] Add pip-audit to dev extras + CI security step
- [x] Add Dependabot (weekly pip + GitHub Actions updates)
- [x] Add `--seed` CLI arg; seed numpy in main(); write run_metadata.json per run
- [x] Fix `get_samples_from_comoving_volume` PNG side-effect (`save_plot=False`)
- [x] Rename `ParameterSpace.dist` → `luminosity_distance` (field, symbol, CSV cols)