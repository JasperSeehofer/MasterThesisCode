# Coding Conventions

**Analysis Date:** 2026-03-25

## Naming Patterns

**Files:**
- Source modules: `snake_case.py` (e.g., `physical_relations.py`, `parameter_space.py`)
- Exception: `LISA_configuration.py` uses UPPERCASE prefix (physics convention, ruff N999 ignored)
- Test files: two patterns coexist — `test_<module>.py` (newer) and `<module>_test.py` (older). Both are valid. New tests use the `test_` prefix.
- Private modules: `_style.py`, `_helpers.py` (underscore prefix in `plotting/`)

**Functions:**
- `snake_case` for all functions: `dist_to_redshift()`, `compute_fisher_information_matrix()`
- Physics symbols preserved in names: `S_OMS()`, `S_TM()`, `S_zz()` in `LISA_configuration.py`
- Ruff rules N802/N803/N806/N815/N816 are ignored to allow uppercase physics names

**Variables:**
- Physics variables use standard notation: `M`, `H`, `Omega_m`, `W_0`, `M_z`, `d_L`
- Prefixes `delta_` for uncertainties: `delta_dist`, `delta_phiS`
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `SNR_THRESHOLD`, `OMEGA_M`, `SPEED_OF_LIGHT_KM_S`)

**Types/Classes:**
- `PascalCase`: `ParameterSpace`, `BayesianInference`, `GalaxyCatalog`, `EMRIDetection`
- Exception classes: `PascalCase` + `Error` suffix: `ParameterEstimationError`, `WaveformGenerationError`

**Parameters on dataclasses:**
- Physics symbols used directly as field names: `M`, `mu`, `a`, `p0`, `e0`, `x0`, `qS`, `phiS`, `qK`, `phiK`

## Code Style

**Formatting:**
- Tool: `ruff format` (via `pyproject.toml` and pre-commit)
- Line length: 100 characters (`[tool.ruff] line-length = 100`)
- Target version: Python 3.13 (`target-version = "py313"`)

**Linting:**
- Tool: `ruff check`
- Rule sets enabled: `E` (pycodestyle), `F` (pyflakes), `I` (isort), `UP` (pyupgrade), `B` (bugbear), `N` (pep8-naming)
- Key ignores: `E501` (line length — formatter handles), `E402` (imports after mpl.rcParams), `N802`/`N803`/`N806`/`N815`/`N816` (physics uppercase), `N999` (LISA_configuration.py)
- Config: `pyproject.toml` `[tool.ruff]` and `[tool.ruff.lint]`

**EditorConfig:** `.editorconfig`
- 4-space indentation, LF line endings, UTF-8, trailing whitespace trimmed
- YAML: 2-space indent

## Type Annotations

**Policy:** All functions must have complete type annotations. Enforced by mypy (`disallow_untyped_defs = true`, `disallow_incomplete_defs = true`) in `pyproject.toml`.

**Style — Python 3.10+ native syntax:**
- Use `list[float]` not `List[float]`, `dict[str, int]` not `Dict[str, int]`
- Use `X | None` not `Optional[X]`
- Enforced by ruff `UP` rule set (pyupgrade)
- Do NOT add `from __future__ import annotations`

**NumPy arrays:**
```python
import numpy.typing as npt
def psd(frequencies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def integrate(a: npt.NDArray[np.floating[Any]]) -> float: ...
```
Never annotate bare `np.ndarray` without dtype parameter.

**CuPy arrays:** Annotate as `npt.NDArray[np.float64]` with a comment that cupy is also accepted at runtime. Never use `cp.ndarray` as a type annotation.

**Union return types for scalar/array functions:**
```python
def hubble_function(
    redshift: float | npt.NDArray[np.floating[Any]],
) -> float | npt.NDArray[np.floating[Any]]: ...
```

**Callable types:** Use `Callable` from `typing` (not `collections.abc.Callable` for type annotations), or `collections.abc.Callable` for runtime use. Decorators use `TypeVar` bound to `Callable[..., Any]`.

**mypy config:** `pyproject.toml` `[tool.mypy]`, `python_version = "3.13"`. Overrides ignore missing imports for: `astropy`, `cupy`, `few`, `fastlisaresponse`, `GPUtil`, `pandas`, `scipy`, `sklearn`, `emcee`, `tabulate`.

## Dataclass Patterns

**Standard usage:**
```python
@dataclass
class Parameter:
    symbol: str
    unit: str
    lower_limit: float
    upper_limit: float
    value: float = 0.0
    is_fixed: bool = False
```

**Mutable defaults — always use `field(default_factory=...)`:**
```python
@dataclass
class ParameterSpace:
    M: Parameter = field(
        default_factory=lambda: Parameter(
            symbol="M", unit="solar masses",
            lower_limit=1e4, upper_limit=1e7,
            randomize_by_distribution=log_uniform,
        )
    )
```
Python 3.13 raises `ValueError` on bare mutable defaults. This is a hard rule.

**`unsafe_hash=True`** used on `Galaxy` dataclass for hashability (needed for set/dict membership).

## Import Organization

**Order (enforced by ruff `I` / isort):**
1. Standard library (`functools`, `logging`, `os`, `pathlib`, `time`)
2. Third-party (`numpy`, `scipy`, `pandas`, `matplotlib`, `emcee`, `astropy`)
3. First-party (`master_thesis_code.*`)

**Configuration:** `[tool.ruff.lint.isort]` with `known-first-party = ["master_thesis_code"]`

**CuPy imports — always guarded:**
```python
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False
```

**Path aliases:** None used. All imports are absolute from `master_thesis_code`.

**Barrel files (`__init__.py`):**
- `master_thesis_code/__init__.py`: empty
- `master_thesis_code/plotting/__init__.py`: re-exports public API (`apply_style`, `get_figure`, `save_figure`, `make_colorbar`)
- Most subpackage `__init__.py` files are empty

**Backward-compatibility re-exports:** `cosmological_model.py` re-exports `BayesianStatistics`, `DetectionProbability`, and `Detection` for backward compatibility after extraction.

## Error Handling

**Custom exceptions:** Defined in `master_thesis_code/exceptions.py`:
- `ArgumentsError`, `ParameterEstimationError`, `TimeoutError`, `ParameterOutOfBoundsError`, `WaveformGenerationError`

**Pattern:** Raise specific exception classes. No bare `except:` clauses (ruff `BLE001` is not in the select list, but tests use `except Exception` with `# noqa: BLE001` when catching broad import failures).

## Logging

**Framework:** Python `logging` module.

**Pattern:**
```python
_LOGGER = logging.getLogger()  # module-level logger
_LOGGER.debug(f"Function {func.__name__!r} executed in {(end - start):.4f}s.")
```

**Log level:** Configurable via `--log_level` CLI argument. Default is `INFO`.

## Comments and Docstrings

**Module docstrings:** All modules have a one-line or multi-line docstring at the top:
```python
"""Physical constants, cosmological parameters, and simulation configuration."""
```

**Function docstrings:** Mixed styles:
- Newer/refactored code: NumPy-style with `Args:` / `Returns:` / `References:` / `Examples:` sections (see `physical_relations.py`)
- Some functions use Sphinx-style `Parameters`/`Returns` with dashes (see `_helpers.py:save_figure`)
- Older code: brief one-liners or no docstring

**Prescriptive rule:** Use Google/NumPy-style docstrings with `Args:`, `Returns:`, `References:` sections for new code. Include math with `.. math::` directives for physics functions.

**Physics reference comments:** Place directly above changed lines:
```python
# Eq. (X.Y) in Author et al. (YYYY), arXiv:XXXX.XXXXX
```

**Inline comments:** Used for units and physics meaning:
```python
C: float = float(ac.c.to(u.m / u.s).value)  # 299792458.0 m/s
```

## Git Conventions

**Physics changes:** Prefix commit subject with `[PHYSICS]`:
```
[PHYSICS] fix luminosity distance prefactor in dist()
```

**Regular changes:** Use conventional commit style observed in history:
```
refactor: replace bare np.random calls with Generator, fix USE_GPU hardcode
chore: remove unused files, obsolete conda/install scripts
```

**Pre-commit hooks (`.pre-commit-config.yaml`):**
1. `ruff --fix` (lint with auto-fix)
2. `ruff-format` (formatting)
3. `mypy` (type checking on `master_thesis_code/` and `master_thesis_code_test/`)

## GPU/CPU Portability Pattern

**Array namespace pattern (`xp`):** Resolve the array module once and use throughout:
```python
def _get_xp(use_gpu: bool) -> types.ModuleType:
    if use_gpu and _CUPY_AVAILABLE:
        return cp
    return np

def scalar_product(a, b, *, use_gpu: bool = False) -> float:
    xp = _get_xp(use_gpu)
    result = xp.trapz(...)
    return float(result.real)
```

**`USE_GPU` must never be hardcoded.** It flows from `--use_gpu` CLI argument through constructors.

## Protocol Pattern

**Structural subtyping** via `typing.Protocol` for decoupling:
```python
class SimulationCallback(Protocol):
    def on_simulation_start(self, total_steps: int) -> None: ...
    def on_snr_computed(self, step: int, snr: float, passed: bool) -> None: ...
```
Default no-op: `NullCallback` class in `callbacks.py`.

---

*Convention analysis: 2026-03-25*
