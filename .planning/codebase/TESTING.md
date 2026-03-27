# Testing Patterns

**Analysis Date:** 2026-03-25

## Test Framework

**Runner:**
- pytest (via `uv run pytest`)
- Config: `pyproject.toml` `[tool.pytest.ini_options]`

**Assertion Library:**
- Built-in `assert` statements (no third-party assertion library)

**Benchmarking:**
- pytest-benchmark (`pytest.mark.slow` + `benchmark` fixture)

**Run Commands:**
```bash
uv run pytest -m "not gpu and not slow"   # Fast CPU tests (default dev workflow)
uv run pytest -m "not gpu"                # All CPU tests including slow
uv run pytest                             # All tests (requires GPU)
uv run pytest -m slow --benchmark-only    # Benchmarks only
uv run pytest --save-test-plots           # Save integration test plots to test-artifacts/
```

## Test Markers

**Defined in `pyproject.toml`:**
- `gpu`: Requires CUDA GPU and cupy. Skip with `-m "not gpu"`.
- `slow`: Long-running test (>10s). Skip with `-m "not slow"`.

**Applied via `pytestmark` module-level:**
```python
pytestmark = pytest.mark.skipif(
    not _PE_AVAILABLE,
    reason="parameter_estimation unavailable",
)
```

**CI runs:** `pytest -m "not gpu and not slow"` in the `check` job; `pytest -m slow` in the `integration` job.

## Coverage

**Tool:** pytest-cov

**Configuration (`pyproject.toml`):**
```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short --cov=master_thesis_code --cov-report=term-missing --cov-report=xml"

[tool.coverage.run]
source = ["master_thesis_code"]

[tool.coverage.report]
fail_under = 25
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", "raise NotImplementedError"]
```

**Current coverage:** ~37% (gate at 25%)

**Coverage output:** `coverage.xml` uploaded as CI artifact.

## Test File Organization

**Location:** Separate `master_thesis_code_test/` directory, mirroring source layout.

**Structure:**
```
master_thesis_code_test/
├── __init__.py
├── conftest.py                          # Root conftest (xp fixture, plotting style, plot_output_dir)
├── test_constants.py                    # Tests for constants.py
├── test_benchmarks.py                   # Benchmark tests (slow marker)
├── physical_relations_test.py           # Tests for physical_relations.py
├── cosmological_model_test.py           # Tests for cosmological_model.py
├── decorators_test.py                   # Tests for decorators.py
├── LISA_configuration_test.py           # Tests for LISA_configuration.py
├── bayesian_inference/
│   └── test_bayesian_inference_mwe.py   # Tests for bayesian_inference pipeline A
├── datamodels/
│   ├── __init__.py
│   ├── parameter_space_test.py          # Tests for parameter_space.py
│   ├── test_detection.py                # Tests for detection.py
│   └── test_emri_detection.py           # Tests for emri_detection.py
├── parameter_estimation/
│   ├── __init__.py
│   └── parameter_estimation_test.py     # Tests for parameter_estimation.py
├── plotting/
│   ├── __init__.py
│   └── test_style.py                    # Tests for plotting foundation
├── integration/
│   ├── __init__.py
│   ├── conftest.py                      # Fixtures: fake_galaxy_catalog, cosmological_model
│   └── test_evaluation_pipeline.py      # Integration test for Pipeline B
└── fixtures/
    ├── __init__.py
    └── evaluation/
        ├── __init__.py
        └── generate_fixtures.py         # Script to generate fixture CSVs
```

**Naming convention:** Mixed — older files use `<module>_test.py`, newer files use `test_<module>.py`. Both are discovered by pytest.

## Fixtures (conftest.py)

**Root conftest (`master_thesis_code_test/conftest.py`):**

1. **`xp` fixture** — Parametrized over `numpy` (always) and `cupy` (when available). Use for tests that should run on both backends:
```python
@pytest.fixture(params=["numpy"] + (["cupy"] if _CUPY_AVAILABLE else []))
def xp(request: pytest.FixtureRequest) -> types.ModuleType:
    if request.param == "cupy":
        return cp
    return np
```

2. **`_plotting_style` fixture** — Session-scoped autouse. Calls `apply_style()` once to set Agg backend and load `emri_thesis.mplstyle`.

3. **`plot_output_dir` fixture** — Session-scoped. Returns a directory for plot output. With `--save-test-plots` writes to `test-artifacts/evaluation/`, otherwise uses a tmpdir.

4. **`pytest_addoption`** — Adds `--save-test-plots` CLI option.

**Integration conftest (`master_thesis_code_test/integration/conftest.py`):**

1. **`fake_galaxy_catalog` fixture** — Builds a `GalaxyCatalogueHandler` with 40 synthetic galaxies (5 detected events x 8 galaxies each: 5 mass-matching + 3 mass-mismatched). Bypasses `__init__` via `object.__new__()`.

2. **`cosmological_model` fixture** — Session-scoped `Model1CrossCheck` instance (slow ~3s MCMC burn-in).

## Test Patterns

**Physical correctness tests (analytical limits):**
```python
def test_dist_at_zero_redshift() -> None:
    """Fundamental analytical limit: dist(0) == 0.0."""
    result = dist(0)
    assert result == 0.0

def test_hubble_function_at_zero() -> None:
    """Normalization: hubble_function(0) == 1.0 for flat LCDM."""
    result = hubble_function(0)
    assert abs(result - 1.0) < 1e-10
```

**Bounds / property tests:**
```python
def test_randomize_parameters_all_within_bounds() -> None:
    ps = ParameterSpace()
    rng = np.random.default_rng(42)
    for _ in range(10):
        ps.randomize_parameters(rng)
        for symbol, value in ps._parameters_to_dict().items():
            param = getattr(ps, symbol)
            assert param.lower_limit <= value <= param.upper_limit
```

**Round-trip tests:**
```python
@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
def test_dist_round_trip(z: float) -> None:
    d = dist(z)
    z_recovered = dist_to_redshift(d)
    assert abs(z_recovered - z) < 1e-5
```

**Determinism / seed tests:**
```python
def test_randomize_parameters_deterministic_with_same_seed() -> None:
    ps1 = ParameterSpace()
    ps1.randomize_parameters(np.random.default_rng(42))
    result1 = ps1._parameters_to_dict()
    ps2 = ParameterSpace()
    ps2.randomize_parameters(np.random.default_rng(42))
    result2 = ps2._parameters_to_dict()
    for key in result1:
        assert result1[key] == result2[key]
```

## Mocking

**Framework:** `unittest.mock` (MagicMock, monkeypatch)

**Pattern — bypass heavy constructors with `__new__`:**
```python
def _make_minimal_pe(tmp_path: pathlib.Path) -> Any:
    pe = ParameterEstimation.__new__(ParameterEstimation)
    pe.parameter_space = ParameterSpace()
    pe.lisa_response_generator = MagicMock()
    pe.snr_check_generator = MagicMock()
    pe.lisa_configuration = MagicMock()
    return pe
```

**Pattern — monkeypatch module-level constants:**
```python
def test_save_cramer_rao_bound_creates_csv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)
```

**Pattern — patch methods on constructed objects:**
```python
with patch.object(pe, "finite_difference_derivative", return_value=mock_derivatives):
    F = pe.compute_fisher_information_matrix()
```

**What to mock:**
- GPU waveform generators (`lisa_response_generator`, `snr_check_generator`)
- File paths (via monkeypatch on module constants)
- Heavy computation methods when testing surrounding logic

**What NOT to mock:**
- Pure math functions (`dist`, `hubble_function`, `lambda_cdm_analytic_distance`)
- Dataclass construction and validation
- Physical constants

## GPU Test Guards

**Import guard pattern for modules that transitively import cupy:**
```python
try:
    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation
    _PE_AVAILABLE = True
except Exception:
    _PE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _PE_AVAILABLE, reason="parameter_estimation unavailable")
```

**GPU-specific tests use `@pytest.mark.gpu`:**
```python
@pytest.mark.gpu
def test_power_spectral_density_a_positive() -> None:
    import cupy as cp
    config = LisaTdiConfiguration()
    fs = cp.logspace(-4, 0, 100)
    psd = config.power_spectral_density(fs, channel="A")
    assert cp.all(psd > 0)
```

**CPU equivalents exist alongside GPU tests** in `LISA_configuration_test.py` — same logic tested with numpy arrays.

## Test Data / Fixtures

**Synthetic fixture CSVs:** Generated by `master_thesis_code_test/fixtures/evaluation/generate_fixtures.py`. Contains Cramer-Rao bounds for 5 detected events.

**Helper factories for test data:**
```python
def _make_series(dist=1.0, delta_dist=0.05, ...) -> pd.Series:
    """Build a pd.Series with column names expected by Detection.__init__."""
    return pd.Series({...})

def _make_emri_detection(seed: int = 0) -> EMRIDetection:
    rng = np.random.default_rng(seed)
    return EMRIDetection(measured_luminosity_distance=float(rng.uniform(0.5, 5.0)), ...)
```

**Integration test fixture builder** in `integration/conftest.py:build_galaxy_catalog_for_n_detections()` — constructs `GalaxyCatalogueHandler` with synthetic BallTree.

## Test Count and Status

**As of 2026-03-12:** 149 CPU tests pass, 18 deselected (14 GPU + 2 slow + 2 integration), 0 xfail.

**Test files by area:**

| File | Test Count (approx) | Area |
|------|---------------------|------|
| `physical_relations_test.py` | 17 | Physics functions |
| `bayesian_inference/test_bayesian_inference_mwe.py` | ~30 | Pipeline A, Galaxy/GalaxyCatalog |
| `datamodels/parameter_space_test.py` | 14 | Parameter space |
| `datamodels/test_detection.py` | 6 | Detection dataclass |
| `datamodels/test_emri_detection.py` | 4 | EMRIDetection dataclass |
| `cosmological_model_test.py` | 12 | Model1CrossCheck helpers |
| `LISA_configuration_test.py` | 11 (5 CPU + 6 GPU) | LISA PSD |
| `parameter_estimation/parameter_estimation_test.py` | 12 (5 CPU + 7 GPU) | Fisher matrix, CRB |
| `test_constants.py` | 5 | Physical constants |
| `decorators_test.py` | 3 | Timer decorator |
| `plotting/test_style.py` | 9 | Plotting utilities |
| `test_benchmarks.py` | 2 (slow) | Performance benchmarks |
| `integration/test_evaluation_pipeline.py` | ~5 | Pipeline B end-to-end |

## Test Coverage Gaps

**Not tested at all (no test file exists):**
- `master_thesis_code/main.py` — Main entry point, simulation loop, evaluation dispatch
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — Pipeline B production code (~986 lines, only integration tests)
- `master_thesis_code/bayesian_inference/detection_probability.py` — KDE detection probability (~344 lines)
- `master_thesis_code/galaxy_catalogue/handler.py` — GLADE catalog interface, BallTree lookups
- `master_thesis_code/galaxy_catalogue/parser.py` — Catalog file parsing
- `master_thesis_code/memory_management.py` — GPU memory tracking
- `master_thesis_code/arguments.py` — CLI argument parsing
- `master_thesis_code/parameter_estimation/evaluation.py` — Evaluation helpers
- `master_thesis_code/plotting/bayesian_plots.py`, `evaluation_plots.py`, `model_plots.py`, `catalog_plots.py`, `physical_relations_plots.py`, `simulation_plots.py` — All plot factory functions
- `master_thesis_code/M1_model_extracted_data/*.py` — All extracted data modules
- `master_thesis_code/galaxy_catalogue/glade_completeness.py` — Completeness calculations

**Partially tested (low coverage):**
- `master_thesis_code/cosmological_model.py` — Only `gaussian`, `polynomial`, `MBH_spin_distribution`, and `Detection` are tested. `Model1CrossCheck`, `LamCDMScenario`, `DarkEnergyScenario` are not unit-tested.
- `master_thesis_code/LISA_configuration.py` — PSD functions tested, but antenna patterns (`F_plus`, `F_cross`), frame transformations, and SSB conversion are untested.
- `master_thesis_code/parameter_estimation/parameter_estimation.py` — CSV I/O tested; core compute functions (`compute_signal_to_noise_ratio`, `compute_Cramer_Rao_bounds`, `finite_difference_derivative`, `five_point_stencil_derivative`) only tested indirectly via GPU integration tests.

**Priority gaps (highest impact):**
1. `bayesian_inference/bayesian_statistics.py` — Production pipeline, ~986 lines, only integration coverage
2. `bayesian_inference/detection_probability.py` — Critical probability computation, ~344 lines, no unit tests
3. `galaxy_catalogue/handler.py` — BallTree lookups, galaxy filtering, used by both pipelines
4. `main.py` — Entry point, simulation loop orchestration

---

*Testing analysis: 2026-03-25*
