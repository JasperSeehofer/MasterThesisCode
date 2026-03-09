# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [2026-03-10] — modern dev tooling: ruff, pre-commit, CI, mypy clean

### Added
- `ruff` and `pre-commit` added to `dev` dependency group in `pyproject.toml`.
- `[tool.ruff]` and `[tool.ruff.lint]` configuration in `pyproject.toml`: selects
  `E`, `F`, `I`, `UP`, `B`, `N` rules; line length 100; `target-version = "py313"`.
- `.pre-commit-config.yaml`: ruff (lint + format) and mypy run automatically on
  every `git commit`. mypy uses the project's local environment to avoid false positives.
- `.github/workflows/ci.yml`: CI pipeline runs ruff check, ruff format check, mypy,
  and pytest (CPU only, `not gpu`) on every push and pull request.
- `CLAUDE.md`: added "Dev Workflow" section documenting the linting commands and
  pre-commit usage.

### Changed
- `pyproject.toml`: `[tool.mypy]` `python_version` updated `"3.10"` → `"3.13"`;
  extended `ignore_missing_imports` overrides to cover `pandas`, `scipy`, `sklearn`,
  `mpl_toolkits`, `emcee`, and `tabulate`.
- All 39 source files and 10 test files brought to 0 mypy errors under strict settings
  (`disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`).
  Key changes across the codebase:
  - Removed all `from typing import List, Dict, Optional, Union`; replaced with
    native Python 3.10 syntax (`list[X]`, `dict[K,V]`, `X | None`, `X | Y`).
  - Removed `from __future__ import annotations` from `arguments.py` and
    `bayesian_inference_mwe.py` (per CLAUDE.md convention).
  - All public and private functions annotated with complete return types and
    parameter types.
  - `np.ndarray` → `npt.NDArray[np.float64]` / `npt.NDArray[np.floating[Any]]`
    throughout.
  - `np.trapz` → `np.trapezoid` (numpy 2.x rename).
  - Guarded `import cupy as cp` in `decorators.py`, `memory_management.py`,
    `LISA_configuration.py`, and `parameter_estimation.py` with
    `try/except ImportError` + `_CUPY_AVAILABLE` sentinel.
  - `decorators.py`: converted `TypeVar`-based generics to Python 3.12 type
    parameter syntax (`def f[F: Callable[..., Any]](func: F) -> F:`).
  - `lambda_cdm_analytic_distance` in `physical_relations.py` and
    `bayesian_inference_mwe.py`: removed `float()` cast (broke numpy 2.x when
    called with 1-D array from `fsolve`); `dist()` uses `np.asarray(...).flat[0]`
    for safe scalar extraction.
  - Fixed trailing-comma bug in `EMRIDetection.from_host_galaxy` that made
    `measured_luminosity_distance` and `measured_redshifted_mass` tuples instead
    of floats when `use_measurement_noise=False`.
  - Ruff-formatted all files to 100-character line length with isort ordering.

---

## [2026-03-09] — uv migration & Python 3.13 compatibility

### Added
- `pyproject.toml`: `[project]` section with dependency groups (`cpu`, `gpu`, `dev`),
  replacing the conda `environment.yml` as the authoritative dependency declaration.
- `uv.lock`: generated lock file replacing `conda-linux-64.lock`; commits exact resolved
  versions of all 150 transitive dependencies for bit-for-bit reproducibility.
- `.python-version`: pins to Python 3.13 (`fastlisaresponse` has no cp314 wheel yet).

### Changed
- Dependency manager switched from conda to [uv](https://docs.astral.sh/uv/).
  Motivation: faster installs, pure-pip workflow, `uv.lock` is simpler than
  `conda-lock`, and `fastemriwaveforms`/`fastlisaresponse` now ship cp313 wheels on PyPI.
- Updated `fastlisaresponse` 1.1.17 → 1.1.9 (latest stable with cp313 wheel; 1.1.17
  transitively pinned `numpy==1.26.0` via `lisaanalysistools`, which has no cp313 wheel).
- Updated `fastemriwaveforms` to 2.0.0rc1 (latest release; removes the numpy pin and
  ships a cp313 wheel on PyPI for the CPU variant).
- Scientific stack updated to current versions with cp313 wheels: numpy 2.4.3,
  scipy 1.17.1, matplotlib 3.10.8, pandas 3.0.1.
- `CLAUDE.md` Environment Setup section replaced with uv instructions.

### Fixed
- `BayesianInference` dataclass: `redshift_values` and `galaxy_distribution_at_redshifts`
  fields used bare `np.array([])` as defaults. Python 3.13 now explicitly rejects mutable
  defaults in dataclasses; replaced with `field(default_factory=lambda: np.array([]))`.

---

## [2025-05-08] — cosmological model & galaxy catalog refinements

### Changed
- `cosmological_model.py`: minor tuning of detection probability evaluation logic and
  integration limits in `BayesianStatistics`.
- `galaxy_catalogue/handler.py`: small adjustment to host-galaxy lookup parameters.

---

## [2025-05-04] — bugfix: detection probability (second round)

### Fixed
- `cosmological_model.py`: added plot of interpolated detection probability surface
  alongside the directly-evaluated one, to verify the interpolation is faithful.
  The root cause of the earlier divergence between the two was confirmed fixed.

---

## [2025-04-28] — bugfix: detection probability

### Fixed
- `cosmological_model.py`: phi boundary check was inverted (`phi >= 0` should be
  `phi < 0`); valid azimuth range is `[0, 2π)` so the out-of-range guard was
  accepting invalid values and rejecting valid ones.
- `cosmological_model.py`: `kde.evaluate(...)` returns a length-1 array, not a scalar;
  added `[0]` indexing so the detection probability is a float rather than an array,
  preventing silent broadcasting errors downstream.

### Added
- `cosmological_model.py`: `plot_detection_probability()` method for visual sanity-checking
  of the KDE-based detection probability over the (`d_L`, `M`, `φ`, `θ`) parameter space.

---

## [2025-04-30] — performance improvements & physical relations refactor

### Changed
- `physical_relations.py`: `dist()` now uses an analytic closed-form expression
  (`lambda_cdm_analytic_distance`) instead of the numerical `np.trapz` integral over
  redshift. Faster and avoids discretisation error.
- `physical_relations.py`: added `cached_dist()` with `@lru_cache(maxsize=1000)` so
  repeated calls at the same redshift/cosmology parameters hit the cache instead of
  recomputing the integral. Significant speedup for the inference loop.
- `cosmological_model.py`: extensive rework of the Bayesian inference evaluation loop;
  likelihood computation restructured around interpolated detection-probability functions
  rather than repeated KDE evaluation calls.
- `galaxy_catalogue/handler.py`: added `HostGalaxy.__eq__` and `__hash__` based on
  `catalog_index`, enabling deduplication of host candidates with a set; added
  `HostGalaxy.from_attributes()` classmethod for constructing instances without a
  full catalog row.

---

## [2025-04-25] — BallTree catalog lookups; inference via interpolated functions

### Changed
- `galaxy_catalogue/handler.py`: replaced linear-scan host-galaxy search with a
  scikit-learn `BallTree` on (φ, θ) sky coordinates. Lookup complexity drops from O(N)
  to O(log N) per query; dominant cost for large catalogs.
- `bayesian_inference/bayesian_inference_mwe.py`: inference now evaluates detection
  probability via interpolated functions over a precomputed grid instead of drawing
  Monte Carlo samples. Removes sample-size variance from the posterior and speeds up
  each likelihood call.
- `cosmological_model.py`: significant reduction in size (1 896 → ~300 lines) by
  removing dead evaluation scripts and consolidating the H₀ inference driver into
  `BayesianStatistics.evaluate()`.
- Tests in `test_bayesian_inference_mwe.py` updated to match the new function-based API.
