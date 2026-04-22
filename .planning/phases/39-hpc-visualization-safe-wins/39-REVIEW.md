---
status: issues
phase: 39
review_date: 2026-04-23
files_reviewed: 9
findings_count: 7
critical: 0
high: 0
medium: 1
low: 4
info: 2
---

# Phase 39: HPC & Visualization Safe Wins — Code Review

**Reviewed:** 2026-04-23
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues (1 medium, 4 low, 2 info — all advisory)

## Summary

Phase 39 implements seven non-physics hygiene tasks (HPC-01..HPC-05, VIZ-01,
VIZ-02). All ROADMAP success criteria pass and 540/540 CPU tests are GREEN.
Implementations cleanly follow the patterns laid out in 39-PATTERNS.md:
the `self._xp` / `self._fft` shim is applied consistently, the deleted
`_crop_frequency_domain` helper is gone (grep clean), the `free_gpu_memory`
API split is sound, the LaTeX gating is auto-detected, and the bootstrap
HDI band is wired through with backward-compatible defaults.

The findings below are all advisory — no critical bugs, no security issues,
no HPC-rule violations. The medium item is a latent robustness concern
about diagnostic GPU calls that could fire in a CPU-with-cupy-installed
configuration; the low items are style/consistency notes; the info items
are observations for future cleanup.

## Files Reviewed

- `master_thesis_code/parameter_estimation/parameter_estimation.py` (HPC-01, HPC-02, HPC-04)
- `master_thesis_code/memory_management.py` (HPC-03)
- `master_thesis_code/main.py` (HPC-03, VIZ-01, VIZ-02)
- `master_thesis_code/plotting/convergence_plots.py` (VIZ-02)
- `master_thesis_code/waveform_generator.py` (HPC-05)
- `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py`
- `master_thesis_code_test/plotting/test_convergence_plots.py`
- `master_thesis_code_test/test_generate_figures.py`
- `master_thesis_code_test/test_memory_management.py`

## Verification of HPC/GPU rules (CLAUDE.md)

- [x] No unguarded `cp.*` or `cufft.*` outside helpers — every direct
      `cp.*` use in `parameter_estimation.py` is gated by
      `_CUPY_AVAILABLE and cp is not None` (lines 236, 282, 401)
- [x] `_CUPY_AVAILABLE` guard preserved at module scope; `_get_xp` /
      `_get_fft` honor it correctly
- [x] `self._xp` / `self._fft` resolved once in `__init__` (lines 104-105),
      no per-call namespace lookups in the hot path
- [x] Vectorized array ops preserved — no per-element loops introduced
- [x] No new GPU→CPU transfers added in the hot path; `cp.asnumpy(...)` at
      line 402 is in `compute_Cramer_Rao_bounds`, called once per detection
      (existing behavior, not regressed)
- [x] `use_gpu` threaded through every constructor; no module-level GPU flag

## Verification of typing conventions (CLAUDE.md)

- [x] No new `from __future__ import annotations` introduced in changed files
      (the two pre-existing instances in `convergence_analysis.py` and
      `posterior_combination.py` are out of Phase 39 scope)
- [x] No `cp.ndarray` annotations added (zero hits in `master_thesis_code/`
      production source; lone `isinstance` check in `LISA_configuration.py:43`
      is pre-existing and runtime-only, not an annotation)
- [x] `list[...]` / `dict[...]` lowercase generics used throughout
- [x] `npt.NDArray[np.float64]` used for typed arrays in
      `convergence_plots.py` band block
- [x] Forward-reference string-literal annotation `"ImprovementBank | None"`
      used in `plot_h0_convergence` to avoid the circular import — correct
      pattern given the project's no-`__future__-annotations` rule
- [x] `_get_xp` and `_get_fft` return `types.ModuleType` — appropriate
      annotation; `# type: ignore[no-any-return]` pragmas justified

## Verification of mutable defaults

- [x] No new `@dataclass` definitions in changed files
- [x] No bare-mutable-default regressions — `_psd_cache`, `_crb_buffer` are
      assigned in `__init__` (not declared as class-level defaults)

## Verification of backward compatibility

- [x] `MemoryManagement.free_gpu_memory()` preserved as deprecated alias
      that emits `DeprecationWarning` and routes to
      `free_gpu_memory_if_pressured()` — D-09 satisfied
- [x] `plot_h0_convergence(bootstrap_bank=None)` is the new default;
      pre-VIZ-02 call sites work unchanged — D-24 satisfied
- [x] `apply_style()` (no kwarg) call sites at `main.py:36`,
      `bayesian_inference_mwe.py`, `fisher_plots.py` left untouched — D-21 satisfied

## Verification of test coverage

- [x] HPC-01: 6 tests in `TestArrayNamespaceShim` (helpers + attributes + CPU `_get_cached_psd`)
- [x] HPC-02: `test_sigterm_drain_with_flush_interval_25` covers 30-row
      auto-flush + manual drain
- [x] HPC-03: 4 new CPU tests (`free_memory_pool`, `clear_fft_cache`,
      `free_gpu_memory_if_pressured`, deprecation alias) — full no-op-on-CPU
      coverage
- [x] HPC-04: `test_dead_freq_crop_helper_is_removed` (negative assertion,
      runtime-built name to satisfy grep gate)
- [x] HPC-05: documentation-only change; verified via 39-05-VERIFICATION.md
      cross-check of `fastlisaresponse 1.1.17` source
- [x] VIZ-01: `TestApplyStyleLatexGating` covers both branches (mocked
      `shutil.which`)
- [x] VIZ-02: `test_plot_h0_convergence_with_bootstrap_bank_adds_fill_between`
      + backward-compat negative test

## Findings

### MEDIUM

#### MED-01: Diagnostic `cp.*` calls fire on CPU runs when cupy is installed

**File:** `master_thesis_code/parameter_estimation/parameter_estimation.py`
**Lines:** 236-238, 282-284, 401-402

The three preserved `cp.*` diagnostic sites are guarded by
`_CUPY_AVAILABLE and cp is not None` but **not** by `self._use_gpu`. On a
CPU-only run where cupy happens to be installed (e.g. dev machine with
`uv sync --extra gpu` for cross-build testing, or future CI lane that
installs cupy purely for static checks), these calls will:

- Lines 236-238 / 282-284: read GPU memory pool stats and log them as
  "GPU memory before/after derivatives", reporting nominal numbers from a
  pool that the CPU code path never wrote to.
- Line 402: `cp.asnumpy(fisher_information_matrix)` is called on what is
  now a numpy array (since `self._xp.zeros(...)` produced numpy in
  `compute_fisher_information_matrix` line 376). Modern cupy treats
  `cp.asnumpy(np.ndarray)` as an identity, so this is functionally
  correct — but it is a GPU-API call on a CPU path, which is exactly the
  pattern HPC-01 was designed to eliminate.

**Fix (recommended for a future quick task; not blocking Phase 39):**
Tighten the guards from `_CUPY_AVAILABLE and cp is not None` to
`self._use_gpu and _CUPY_AVAILABLE and cp is not None` at all three
sites. The CPU branch should use `np.asarray(fisher_information_matrix)`
(already the else-branch at line 404) and skip the diagnostic memory log
entirely. Example:

```python
# parameter_estimation.py:236-238 and :282-284
if self._use_gpu and _CUPY_AVAILABLE and cp is not None:
    pool = cp.get_default_memory_pool()
    _LOGGER.info(f"GPU memory before derivatives: {pool.total_bytes() / 1e9:.2f} GB")

# parameter_estimation.py:401-404
if self._use_gpu and _CUPY_AVAILABLE and cp is not None:
    fisher_np = cp.asnumpy(fisher_information_matrix)
else:
    fisher_np = np.asarray(fisher_information_matrix)
```

**Severity rationale:** Latent — only fires in a configuration
(CPU + cupy installed) that is unusual but legal. Not currently
triggered in the GREEN test suite (which runs `pytest -m "not gpu"` on a
machine without cupy). Worth noting because HPC-01's whole purpose was
to eliminate exactly this CPU/GPU coupling.

### LOW

#### LOW-01: `import warnings` inside method body in deprecated alias

**File:** `master_thesis_code/memory_management.py`
**Lines:** 75-85

The deprecated `free_gpu_memory()` alias does
`import warnings` inside the method body rather than at module top.
Module top would be more idiomatic (and consistent with the rest of the
codebase, e.g. `parameter_estimation.py:13`). The local-import pattern
is sometimes used to defer optional dependencies, but `warnings` is
stdlib and always available.

**Fix:** Add `import warnings` to the module's top-level imports (after
`import logging`, before the optional-import blocks) and remove the
in-method import.

**Severity rationale:** Stylistic only; no behavioral impact. Code
review surfaces it because the rest of the project consistently puts
stdlib imports at module top.

#### LOW-02: `_get_xp` / `_get_fft` use `# type: ignore[no-any-return]` rather than typed module returns

**File:** `master_thesis_code/parameter_estimation/parameter_estimation.py`
**Lines:** 52-63

Both helpers carry `# type: ignore[no-any-return]` on their cupy
branches. This is necessary because cupy has no mypy stubs (per
CLAUDE.md), so mypy infers `Any` from `cp` and complains about
returning `Any` where `types.ModuleType` is annotated. The pragma is
correct but a comment justifying it (mirroring the cupy/few/GPUtil
pragmas elsewhere in the project) would aid future readers.

**Fix (optional):**

```python
def _get_xp(use_gpu: bool) -> types.ModuleType:
    """Resolve the array namespace: cupy when use_gpu and cupy available, else numpy."""
    if use_gpu and _CUPY_AVAILABLE and cp is not None:
        # cupy has no mypy stubs (CLAUDE.md), so mypy infers Any here
        return cp  # type: ignore[no-any-return]
    return np
```

**Severity rationale:** Non-functional documentation improvement.

#### LOW-03: `compute_Cramer_Rao_bounds` `np.asarray` fall-through allocates an extra array on CPU

**File:** `master_thesis_code/parameter_estimation/parameter_estimation.py`
**Lines:** 401-404

When the CPU path is taken (the `else` branch at line 404),
`np.asarray(fisher_information_matrix)` runs on what is already a numpy
array (`self._xp.zeros(...)` at line 376 returned numpy). `np.asarray`
on a numpy array of matching dtype is a no-op (returns the same object),
so this is not a leak — but a brief comment would document the
intentional symmetry with the GPU branch.

**Fix (optional):** Add a one-line comment:

```python
else:
    # On CPU, fisher_information_matrix is already numpy (self._xp == np)
    fisher_np = np.asarray(fisher_information_matrix)
```

**Severity rationale:** No correctness or performance issue; readability only.

#### LOW-04: Local imports inside `_gen_h0_convergence` repeat what `generate_figures` already does

**File:** `master_thesis_code/main.py`
**Lines:** 1025-1029 (inside `_gen_h0_convergence`)

`compute_m_z_improvement_bank` is imported inside `_gen_h0_convergence`
(line 1027) and again inside `_gen_paper_m_z_improvement` further down
(line 1216). Each closure rebinds the import at first call. This is
consistent with the existing per-figure lazy-import pattern used
throughout `generate_figures` (rationale: keep figure failures isolated
when an optional dependency is missing for one figure).

**Fix:** None required — this matches the established pattern. Flagged
only as an observation: should the closures ever be refactored, the two
lazy-imports of `compute_m_z_improvement_bank` could share a single
top-of-`generate_figures` import. The cached-on-disk semantics
(noted in the inline comment at line 1034) make the duplicate import
harmless.

**Severity rationale:** Pattern-conformant; no action needed.

### INFO

#### INFO-01: `LISA_configuration.py` unconditional `cupy` import remains unaddressed

**File:** `master_thesis_code/LISA_configuration.py` (out of Phase 39 scope)

CLAUDE.md and the phase context (39-CONTEXT.md §Out of scope) explicitly
defer this fix to "when that file is next touched." HPC-01 successfully
makes `parameter_estimation.py` CPU-importable, but
`parameter_estimation.py` still imports `LISA_configuration` at module
top (line 43). Therefore the `python -c "from ...parameter_estimation
import ParameterEstimation"` smoke test at SC-1 only succeeds because
`sys.modules['cupy'] = None` is injected first by the test harness — the
underlying `LISA_configuration` import would still fail in a true
cupy-uninstalled environment.

The 39-VERIFICATION.md SC-1 evidence acknowledges this implicitly
(`sys.modules['cupy']=None` is in the verification command). This is a
known limitation, not a Phase 39 regression.

**Fix:** Out of scope. Apply the same `try/except ImportError` +
`_CUPY_AVAILABLE` guard to `LISA_configuration.py:cp` import when that
file is next touched (memory note `project_coordinate_bugs` and
CLAUDE.md Known Bug 1 already track this).

#### INFO-02: GitHub issue #2 marked open but actually resolved by Phase 10

**File:** GitHub repository state (out of Phase 39 scope)

39-VERIFICATION.md notes that GitHub issue #2 ("Fisher matrix forward
difference") is still open despite being fixed in Phase 10 (5-point
stencil now default). Phase 39 does not affect issue #2 directly, but
the GitHub-integration discipline in CLAUDE.md ("close issues with a
comment referencing the fix") was missed at Phase 10 close.

**Fix:** Out of scope for Phase 39. Recommend a one-liner cleanup task:

```bash
gh issue close 2 --comment "Resolved in Phase 10 (commit ?) — five-point stencil now default in compute_fisher_information_matrix; see CLAUDE.md Known Bug 4 [FIXED]."
```

---

_Reviewed: 2026-04-23_
_Reviewer: Claude (gsd-code-reviewer, Opus 4.7 1M)_
_Depth: standard_
