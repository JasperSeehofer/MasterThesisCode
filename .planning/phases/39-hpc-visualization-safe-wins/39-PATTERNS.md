# Phase 39: HPC & Visualization Safe Wins — Pattern Map

**Mapped:** 2026-04-22
**Files analyzed:** 5 production files (modified) + 1 test file (new)
**Analogs found:** 6 / 6

All targets are in-place edits of existing files; each already contains at least
one peer pattern that the executor must mirror. Nothing is net-new architecture.

---

## File Classification

| File to Modify | Role | Data Flow | Closest Analog (in-file or peer) | Match Quality |
|----------------|------|-----------|----------------------------------|---------------|
| `master_thesis_code/parameter_estimation/parameter_estimation.py` (HPC-01, HPC-02, HPC-04) | hot-path math (Fisher / inner product) | GPU array transform + batch I/O | `memory_management.py` `try/except ImportError` + `_CUPY_AVAILABLE` guard; self-analog `_get_cached_psd` for `self._xp`/`self._fft` attribute style | exact |
| `master_thesis_code/memory_management.py` (HPC-03) | memory manager (CLI glue + GPU utility) | command (no data flow) | self-analog: existing `free_gpu_memory` already splits pool vs cache internally (lines 43–48) | exact |
| `master_thesis_code/main.py` (HPC-03 call-site updates, VIZ-01, VIZ-02) | CLI glue / figure orchestration | request-response (manifest dispatch) | self-analog: existing `manifest.append((name, _gen_fn))` blocks at 1015–1030, 1195–1209; existing `apply_style()` call at line 803 | exact |
| `master_thesis_code/waveform_generator.py:58` (HPC-05) | physics config (constructor args for `ResponseWrapper`) | one-shot setup | self-analog: existing inline citation comment pattern used in `parameter_estimation.py:241` (`# Vallisneri (2008), arXiv:gr-qc/0703086`) | exact |
| `master_thesis_code/plotting/convergence_plots.py` (VIZ-02) | plotting factory | transform + render | `master_thesis_code/plotting/convergence_analysis.py:656-677` (`fill_between` + `plot` + `VARIANT_*` color + 1/sqrt(N) ref) | exact |
| `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py` (HPC-02 regression) | test | mock-based unit | self-analog: existing `test_crb_buffer_auto_flushes_at_interval` at line 251 and `_make_minimal_pe` factory at line 28 | exact |

---

## Pattern Assignments

### `master_thesis_code/parameter_estimation/parameter_estimation.py` (HPC-01 / HPC-02 / HPC-04)

#### HPC-01 — `self._xp` / `self._fft` shim

**Anchor pattern (keep as-is, extend)** — existing `try/except ImportError` at `parameter_estimation.py:19-27`:

```python
try:
    import cupy as cp
    import cupyx.scipy.fft as cufft

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cufft = None
    _CUPY_AVAILABLE = False
```

**Mirror pattern from `memory_management.py:32-37`** — `_CUPY_AVAILABLE and cp is not None` CPU fall-through:

```python
if _CUPY_AVAILABLE and cp is not None:
    self.memory_pool = cp.get_default_memory_pool()
    self._fft_cache = cp.fft.config.get_plan_cache()
else:
    self.memory_pool = None
    self._fft_cache = None
```

**Existing `self._<attr>` runtime-resolved attribute pattern (`__init__`, lines 73-100)** — this is where `_xp` / `_fft` attributes attach:

```python
def __init__(
    self,
    waveform_generation_type: WaveGeneratorType,
    parameter_space: ParameterSpace,
    *,
    use_gpu: bool = True,
    use_five_point_stencil: bool = True,
):
    self.parameter_space = parameter_space
    self._use_gpu = use_gpu
    self._use_five_point_stencil = use_five_point_stencil
    # ... lazy imports, lisa_configuration, caches ...
    self._psd_cache: dict[int, tuple[Any, Any, int, int]] = {}
    self._crb_buffer: list[dict] = []
    self._crb_flush_interval: int = 1
```

**Sites to rewrite (exact grep set)** — every `cp.*` / `cufft.*` hit inside the class body:

| Line | Current | Target |
|------|---------|--------|
| 110 | `fs_full = cufft.rfftfreq(n, self.dt)[1:]` | `fs_full = self._fft.rfftfreq(n, self.dt)[1:]` |
| 111 | `lower_idx = int(cp.argmax(fs_full >= MINIMAL_FREQUENCY))` | `lower_idx = int(self._xp.argmax(fs_full >= MINIMAL_FREQUENCY))` |
| 112 | `upper_idx = int(cp.argmax(fs_full >= MAXIMAL_FREQUENCY))` | `self._xp.argmax(...)` |
| 117 | `psd_stack = cp.stack([...])` | `self._xp.stack([...])` |
| 138 | `result = cp.stack(result)` | `self._xp.stack(result)` |
| 207, 253 | `pool = cp.get_default_memory_pool()` (diagnostic) | Guard with `if self._use_gpu and _CUPY_AVAILABLE and cp is not None:` (already guarded; keep `cp` direct — it only runs on GPU) |
| 273 | `return cp.array([...])` | `return self._xp.array([...])` |
| 317 | `a_ffts = cufft.rfft(...)` | `a_ffts = self._fft.rfft(...)` |
| 318 | `b_ffts_cc = cp.conjugate(cufft.rfft(...))` | `self._xp.conjugate(self._fft.rfft(...))` |
| 331 | `float(cp.trapz(...))` | `float(self._xp.trapz(...))` |
| 386 | `fisher_np = cp.asnumpy(fisher_information_matrix)` | Keep guarded: `if self._use_gpu and _CUPY_AVAILABLE and cp is not None: fisher_np = cp.asnumpy(...)` |
| 432 | `snr = cp.sqrt(...)` | `snr = self._xp.sqrt(...)` |
| 340-341 | `cp.argmax(...)` inside `_crop_frequency_domain` | **DELETE whole method** per HPC-04 |
| 359 | `xp = cp if (_CUPY_AVAILABLE and cp is not None) else np` (already in `compute_fisher_information_matrix`) | Replace with `xp = self._xp` (remove local resolution) |

**CPU fall-back warning pattern (D-04) — use `warnings.warn` once per instance:**

```python
# In __init__, after resolving self._xp / self._fft:
if use_gpu and not _CUPY_AVAILABLE:
    warnings.warn(
        "ParameterEstimation: use_gpu=True but cupy is not installed; "
        "falling back to numpy. Expect ~100x slowdown.",
        RuntimeWarning,
        stacklevel=2,
    )
```

`warnings` is already imported at line 12. No new module dependency.

#### HPC-02 — `_crb_flush_interval: int = 25`

**One-line change** at `parameter_estimation.py:99`:

```python
# Before
self._crb_flush_interval: int = 1
# After
self._crb_flush_interval: int = 25  # ROADMAP SC-2: batch CRB writes to reduce Lustre I/O
```

**SIGTERM drain check (existing, do not change)** — `main.py:346-354`:

```python
_pe_ref: list[ParameterEstimation] = []

def _sigterm_handler(signum: int, frame: object) -> None:
    if _pe_ref:
        _ROOT_LOGGER.warning("SIGTERM received — flushing buffered Cramér-Rao bounds...")
        _pe_ref[0].flush_pending_results()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)
```

#### HPC-04 — Delete `_crop_frequency_domain`

**Lines 334-347 to remove verbatim:**

```python
@staticmethod
def _crop_frequency_domain(fs: Any, integrant: Any) -> tuple[Any, Any]:
    if len(fs) != len(integrant):
        _LOGGER.warning("length of frequency domain and integrant are not equal.")
    # find lowest frequency
    lower_limit_index = cp.argmax(fs >= MINIMAL_FREQUENCY)
    upper_limit_index = cp.argmax(fs >= MAXIMAL_FREQUENCY)
    if upper_limit_index == 0:
        upper_limit_index = len(fs)
    return (
        fs[lower_limit_index:upper_limit_index],
        integrant[lower_limit_index:upper_limit_index],
    )
```

**Superseding function (keep, already in use)** — `_get_cached_psd` at lines 102-124 performs the identical crop via `lower_idx`/`upper_idx` slicing and caches the result.

**Grep gate:** `rg "_crop_frequency_domain" master_thesis_code/ master_thesis_code_test/` must return empty after removal.

---

### `master_thesis_code/memory_management.py` (HPC-03)

**Existing internal split (keep, surface publicly) — lines 32-48:**

```python
if _CUPY_AVAILABLE and cp is not None:
    self.memory_pool = cp.get_default_memory_pool()
    self._fft_cache = cp.fft.config.get_plan_cache()
else:
    self.memory_pool = None
    self._fft_cache = None

# ...

def free_gpu_memory(self) -> None:
    """Free GPU memory pool blocks and clear FFT plan cache. No-op on CPU."""
    if self.memory_pool is not None:
        self.memory_pool.free_all_blocks()
    if self._fft_cache is not None:
        self._fft_cache.clear()
```

**Target API (D-08, D-09, D-11):**

```python
def free_memory_pool(self) -> None:
    """Free CuPy memory-pool blocks. Safe to call every simulation step. No-op on CPU."""
    if self.memory_pool is not None:
        self.memory_pool.free_all_blocks()

def clear_fft_cache(self) -> None:
    """Clear FFT plan cache. Expensive — rebuilds on next rfft. No-op on CPU."""
    if self._fft_cache is not None:
        self._fft_cache.clear()

def free_gpu_memory_if_pressured(self, threshold: float = 0.8) -> None:
    """Free memory pool every call; clear FFT cache only when pool usage exceeds threshold.

    Args:
        threshold: Fraction of total GPU memory (default 0.8 → 64 GB on 80 GB H100).
    """
    self.free_memory_pool()
    if self.memory_pool is None or not _CUPY_AVAILABLE or cp is None:
        return
    try:
        _free, total = cp.cuda.runtime.memGetInfo()
    except Exception:  # noqa: BLE001 — tolerate absent runtime on CPU-only
        return
    used = int(self.memory_pool.total_bytes())
    if total > 0 and used / total >= threshold:
        _LOGGER.info(
            "FFT cache cleared — pool usage %.1f%% of %.1f GB",
            100.0 * used / total,
            total / 1e9,
        )
        self.clear_fft_cache()

def free_gpu_memory(self) -> None:
    """Deprecated alias. Routes to free_gpu_memory_if_pressured()."""
    import warnings
    warnings.warn(
        "free_gpu_memory() is deprecated; use free_gpu_memory_if_pressured() "
        "or free_memory_pool() directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    self.free_gpu_memory_if_pressured()
```

**Claude's discretion per D-68:** GPU-total source is `cp.cuda.runtime.memGetInfo()` (stdlib; avoids optional `GPUtil`). Telemetry is `_LOGGER.info` — helpful for Phase 40, per D-68 note.

**Grep gate (D-12):** `rg "_fft_cache.clear" master_thesis_code/` returns exactly one hit — inside `clear_fft_cache`.

---

### `master_thesis_code/main.py` (HPC-03 call-site updates, VIZ-01, VIZ-02)

#### HPC-03 call-site pattern

**Before (two sites):**

```python
# main.py:381 (data_simulation loop)
memory_management.free_gpu_memory()

# main.py:639 (injection campaign loop)
memory_management.free_gpu_memory()
```

**After:**

```python
# main.py:381
memory_management.free_gpu_memory_if_pressured()

# main.py:639
memory_management.free_gpu_memory_if_pressured()
```

No other call-site changes required — the three `MemoryManagement(use_gpu=...)` constructor calls at lines 307, 356, 588 are unchanged.

#### VIZ-01 — optional LaTeX gating

**Existing anchor at `main.py:800-803`:**

```python
from master_thesis_code.plotting._helpers import save_figure
from master_thesis_code.plotting._style import apply_style

apply_style()
```

**Target (per D-19):**

```python
import shutil  # at top of generate_figures; shutil is already a stdlib import at module scope
from master_thesis_code.plotting._helpers import save_figure
from master_thesis_code.plotting._style import apply_style

if shutil.which("latex"):
    apply_style(use_latex=True)
    _ROOT_LOGGER.info("LaTeX detected; rendering figures with text.usetex=True")
else:
    apply_style()
```

**Reference signature (do not modify)** — `plotting/_style.py:13-47` already supports `use_latex: bool = False` kwarg. No plotting code changes needed.

**Do NOT touch** the other `apply_style()` call sites (per D-21 & 39-CONTEXT.md §Known Constraints):

- `master_thesis_code/main.py:35` (module-load smoke)
- `master_thesis_code/plotting/fisher_plots.py:509`
- `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py:67`

#### VIZ-02 — wire `bootstrap_bank` into `_gen_h0_convergence`

**Existing factory call at `main.py:1015-1030`:**

```python
def _gen_h0_convergence() -> tuple[object, object] | None:
    if post_data is None:
        return None
    from master_thesis_code.plotting.convergence_plots import plot_h0_convergence

    h_vals, event_posts = post_data
    h_alt, ep_alt = post_data_with if post_data_with is not None else (None, None)
    return plot_h0_convergence(
        h_vals,
        event_posts,
        true_h=0.73,
        h_values_alt=h_alt,
        event_posteriors_alt=ep_alt,
    )
```

**Peer pattern at `main.py:1196-1209`** (already computes the bank for the M_z improvement figure):

```python
def _gen_paper_m_z_improvement() -> tuple[object, object] | None:
    from master_thesis_code.constants import H as TRUE_H
    from master_thesis_code.plotting.convergence_analysis import (
        compute_m_z_improvement_bank,
        plot_m_z_improvement_panels,
    )

    try:
        bank = compute_m_z_improvement_bank(Path(output_dir), h_true=float(TRUE_H))
    except (FileNotFoundError, ValueError, KeyError):
        return None
    if bank is None:
        return None
    return plot_m_z_improvement_panels(bank)
```

**Target at `main.py:1015-1030` — reuse the bank (D-25):**

```python
def _gen_h0_convergence() -> tuple[object, object] | None:
    if post_data is None:
        return None
    from master_thesis_code.constants import H as TRUE_H
    from master_thesis_code.plotting.convergence_analysis import (
        compute_m_z_improvement_bank,
    )
    from master_thesis_code.plotting.convergence_plots import plot_h0_convergence

    h_vals, event_posts = post_data
    h_alt, ep_alt = post_data_with if post_data_with is not None else (None, None)
    try:
        bootstrap_bank = compute_m_z_improvement_bank(Path(output_dir), h_true=float(TRUE_H))
    except (FileNotFoundError, ValueError, KeyError):
        bootstrap_bank = None
    return plot_h0_convergence(
        h_vals,
        event_posts,
        true_h=float(TRUE_H),
        h_values_alt=h_alt,
        event_posteriors_alt=ep_alt,
        bootstrap_bank=bootstrap_bank,
    )
```

Note: `compute_m_z_improvement_bank` is cached on disk (per its docstring), so calling it again from `_gen_h0_convergence` costs one JSON read, not a bootstrap recomputation.

---

### `master_thesis_code/waveform_generator.py:58` (HPC-05 primary path)

**Existing code (lines 56-69):**

```python
lisa_response_generator = ResponseWrapper(
    waveform_gen=_set_waveform_generator(waveform_generator_type, use_gpu=use_gpu),
    flip_hx=True,
    index_lambda=INDEX_LAMBDA,
    index_beta=INDEX_BETA,
    t0=T0,
    is_ecliptic_latitude=False,
    Tobs=T_observation,
    ...
)
```

**Peer comment style (inline physics citation)** — `parameter_estimation.py:241`:

```python
# 5-point stencil: (-f(+2h) + 8f(+h) - 8f(-h) + f(-2h)) / 12h
# Vallisneri (2008), arXiv:gr-qc/0703086
```

And `parameter_estimation.py:353`:

```python
# Vallisneri (2008), arXiv:gr-qc/0703086 — O(epsilon^4) stencil for accurate Fisher matrices
```

**Target — add a 2-line comment directly above `flip_hx=True` (D-16):**

```python
lisa_response_generator = ResponseWrapper(
    waveform_gen=_set_waveform_generator(waveform_generator_type, use_gpu=use_gpu),
    # flip_hx=True required when is_ecliptic_latitude=False; see fastlisaresponse
    # <version> source (ResponseWrapper.__init__) and Katz et al. (2022) arXiv:2204.06633
    flip_hx=True,
    index_lambda=INDEX_LAMBDA,
    ...
)
```

**Verification checklist (must complete before commit per D-15):**
1. `uv pip show fastlisaresponse` → record version string; substitute for `<version>`.
2. Read `ResponseWrapper.__init__` source from the installed package (`.venv/lib/python3.13/site-packages/fastlisaresponse/response.py` or similar).
3. Cross-check Katz et al. (2022) arXiv:2204.06633 — `is_ecliptic_latitude` kwarg semantics and the `flip_hx` sign flip.
4. If verification fails → escalate to HPC-05 fallback (removal path per D-17): triggers `/physics-change` skill, regression pickle, `[PHYSICS]` commit.

---

### `master_thesis_code/plotting/convergence_plots.py` (VIZ-02)

#### Signature extension (D-24)

**Current signature at line 50 adds one kwarg — preserves backward-compatibility:**

```python
def plot_h0_convergence(
    h_values: npt.NDArray[np.float64],
    event_posteriors: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    *,
    true_h: float | None = None,
    subset_sizes: list[int] | None = None,
    seed: int = 42,
    level: float = 0.68,
    h_values_alt: npt.NDArray[np.float64] | None = None,
    event_posteriors_alt: list[npt.NDArray[np.float64]] | None = None,
    label: str = r"Without $M_z$",
    label_alt: str = r"With $M_z$",
    color: str | None = None,
    color_alt: str | None = None,
    bootstrap_bank: "ImprovementBank | None" = None,  # NEW
    ax: None = None,
) -> tuple[Figure, npt.NDArray[np.object_]]:
```

Import for type annotation (put at top of file, avoid cycle with existing `convergence_analysis` import layout):

```python
from master_thesis_code.plotting.convergence_analysis import ImprovementBank
```

If a circular import appears, use `TYPE_CHECKING` guard — **but** CLAUDE.md forbids `from __future__ import annotations`, so keep the string literal annotation `"ImprovementBank | None"` with a direct import inside the function body for the isinstance check:

```python
if bootstrap_bank is not None:
    from master_thesis_code.plotting.convergence_analysis import ImprovementBank  # local import
    assert isinstance(bootstrap_bank, ImprovementBank)
```

#### Band-drawing pattern (D-22, D-23, D-27)

**Anchor pattern to mirror — `convergence_analysis.py:655-677`:**

```python
# ---- Top-left: HDI68 width vs N ----
ax_w.fill_between(sizes, w_no_lo, w_no_hi, color=VARIANT_NO_MASS, alpha=0.18, zorder=2)
ax_w.plot(
    sizes,
    w_no_med,
    "o-",
    color=VARIANT_NO_MASS,
    markersize=4,
    linewidth=1.2,
    label=r"Without $M_z$",
    zorder=3,
)
ax_w.fill_between(sizes, w_with_lo, w_with_hi, color=VARIANT_WITH_MASS, alpha=0.18, zorder=2)
ax_w.plot(
    sizes,
    w_with_med,
    "s--",
    color=VARIANT_WITH_MASS,
    markersize=4,
    linewidth=1.2,
    label=r"With $M_z$",
    zorder=3,
)
```

**Field path in `ImprovementBank` (source of the 16/84 band)** — `convergence_analysis.py:634-639`:

```python
w_no_med = np.asarray(bank.metrics_no_mass["hdi68_width"]["median"], dtype=np.float64)
w_no_lo = np.asarray(bank.metrics_no_mass["hdi68_width"]["p16"], dtype=np.float64)
w_no_hi = np.asarray(bank.metrics_no_mass["hdi68_width"]["p84"], dtype=np.float64)
w_with_med = np.asarray(bank.metrics_with_mass["hdi68_width"]["median"], dtype=np.float64)
w_with_lo = np.asarray(bank.metrics_with_mass["hdi68_width"]["p16"], dtype=np.float64)
w_with_hi = np.asarray(bank.metrics_with_mass["hdi68_width"]["p84"], dtype=np.float64)
```

**Target insertion — after the existing right-panel `ax_ci.plot(...)` calls at `convergence_plots.py:144` and 172, before the 1/sqrt(N) ref curve (line 174):**

```python
# --- Optional bootstrap HDI band on the right panel (VIZ-02) ---
if bootstrap_bank is not None:
    b_sizes = np.asarray(bootstrap_bank.sizes, dtype=np.float64)
    # Without M_z (primary variant)
    w_no_lo = np.asarray(
        bootstrap_bank.metrics_no_mass["hdi68_width"]["p16"], dtype=np.float64
    )
    w_no_hi = np.asarray(
        bootstrap_bank.metrics_no_mass["hdi68_width"]["p84"], dtype=np.float64
    )
    ax_ci.fill_between(b_sizes, w_no_lo, w_no_hi, color=color, alpha=0.2, zorder=2)
    if event_posteriors_alt is not None:
        w_with_lo = np.asarray(
            bootstrap_bank.metrics_with_mass["hdi68_width"]["p16"], dtype=np.float64
        )
        w_with_hi = np.asarray(
            bootstrap_bank.metrics_with_mass["hdi68_width"]["p84"], dtype=np.float64
        )
        ax_ci.fill_between(
            b_sizes, w_with_lo, w_with_hi, color=color_alt, alpha=0.2, zorder=2
        )
```

**Discretion choices made (aligned with D-101/103/105):**
- `alpha=0.2` — conventional 16/84 band; lower than the `0.18` in `convergence_analysis.py` because the right panel also hosts the `1/sqrt(N)` reference line, which should remain visually dominant (per 39-CONTEXT.md §Specifics).
- Band color tracks `color` / `color_alt` (the variant's line color) — no new legend entry; band piggybacks on the existing CI-width line label.
- `zorder=2` places band behind the median line (`"o-"` plot uses default `zorder=3` implicitly).

**Left panel (line 139, `ax_post.plot(...)`) — unchanged (D-27).**

---

### `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py` (HPC-02 regression)

**Self-analog test at lines 251-280 — `test_crb_buffer_auto_flushes_at_interval`:**

```python
def test_crb_buffer_auto_flushes_at_interval(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buffer must auto-flush once it reaches _crb_flush_interval rows."""
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    pe._crb_flush_interval = 2
    result_path = csv_path.replace("$index", "0")

    # 1st call: buffer has 1 row — no auto-flush yet
    pe.save_cramer_rao_bound({}, snr=10.0, simulation_index=0)
    assert not pathlib.Path(result_path).exists()
    # 2nd call: buffer reaches 2 → auto-flush
    pe.save_cramer_rao_bound({}, snr=11.0, simulation_index=0)
    assert pathlib.Path(result_path).exists()
    ...
```

**Factory at lines 28-44 — `_make_minimal_pe`** (reuse; test must call it, not construct `ParameterEstimation` directly, to avoid SIGILL-on-CPU).

**Target test (per D-07) — 30 rows with interval=25 → 1 auto-flush + 1 manual:**

```python
def test_sigterm_drain_with_flush_interval_25(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HPC-02 regression: interval=25 + 30 saves → auto-flush at 25 +
    manual flush drains remaining 5. Mirrors SIGTERM handler at main.py:351.
    """
    csv_path = str(tmp_path / "crb_simulation_$index.csv")
    monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", csv_path)

    pe = _make_minimal_pe(tmp_path)
    pe._crb_flush_interval = 25

    for i in range(30):
        pe.save_cramer_rao_bound({}, snr=10.0 + i, simulation_index=0)

    # After 25 saves: auto-flushed; 5 pending
    result_path = csv_path.replace("$index", "0")
    df_after_auto = pd.read_csv(result_path)
    assert len(df_after_auto) == 25
    assert len(pe._crb_buffer) == 5

    # Simulate SIGTERM handler path: flush drains remaining 5
    pe.flush_pending_results()
    df_final = pd.read_csv(result_path)
    assert len(df_final) == 30
    assert pe._crb_buffer == []
```

Mirrors the `_make_minimal_pe` mock factory and the `monkeypatch` + `CRAMER_RAO_BOUNDS_PATH` substitution pattern already established.

---

## Shared Patterns

### CPU-safe imports (cross-cutting, HPC-01 and HPC-03)

**Source:** `memory_management.py:14-20` (and mirrored in `parameter_estimation.py:19-27`)
**Apply to:** every module that uses cupy/cufft.

```python
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False
```

And the runtime guard inside instance code:

```python
if _CUPY_AVAILABLE and cp is not None:
    ...
else:
    ...
```

### Self-attribute namespace (HPC-01)

**Source:** `parameter_estimation.py:82-99` (`self._use_gpu`, `self._use_five_point_stencil`, `self._psd_cache`, `self._crb_buffer`, `self._crb_flush_interval`)
**Apply to:** HPC-01 adds `self._xp` and `self._fft` following the same pattern — runtime-resolved GPU/CPU switches attached once at `__init__` time.

### Inline physics citation (HPC-05)

**Source:** `parameter_estimation.py:241, 353` (Vallisneri 2008 stencil comments)
**Apply to:** HPC-05 `flip_hx=True` comment. Convention: 1–2 line `# ` comment directly above the cited line, with `arXiv:NNNN.NNNNN` format.

### Factory plotting contract (VIZ-02)

**Source:** every module in `master_thesis_code/plotting/` returns `(fig, ax_or_axes)`.
**Apply to:** `plot_h0_convergence` signature is already `(Figure, NDArray[object])`. The new `bootstrap_bank` kwarg must remain keyword-only and default to `None` to preserve the existing public contract.

### Test scaffolding (HPC-02)

**Source:** `parameter_estimation_test.py:28-44` (`_make_minimal_pe`), `:22-25` (`pytestmark = pytest.mark.skipif(not _PE_AVAILABLE, ...)`), `_load_crb_data` monkeypatch pattern
**Apply to:** new HPC-02 SIGTERM-drain test reuses the existing fixture factory and `monkeypatch.setattr(pe_module, "CRAMER_RAO_BOUNDS_PATH", ...)` idiom.

### Project style contract

**Source:** CLAUDE.md §Typing Conventions
- `list[...]` not `List[...]`
- No `from __future__ import annotations` (use string-literal annotations if needed)
- `npt.NDArray[np.float64]` for arrays
- `Callable` from `typing` (not lowercase `callable`)
- No bare mutable default in `@dataclass` (use `field(default_factory=...)`)

All modifications in this phase must pass `uv run mypy master_thesis_code/`.

---

## No Analog Found

*(None.)* Every target has a close in-file or peer-file pattern.

---

## Metadata

**Analog search scope:**
- `master_thesis_code/parameter_estimation/` (role: hot-path math)
- `master_thesis_code/` top-level (role: memory/CLI)
- `master_thesis_code/plotting/` (role: plotting factory)
- `master_thesis_code_test/` (role: tests)

**Files scanned:** 8 (of which 6 provided the anchor patterns cited above).

**Key invariants to preserve (from CLAUDE.md + 39-CONTEXT.md):**

1. GPU imports stay in `try/except ImportError` blocks.
2. `self._use_gpu` threads through every constructor (no module-level GPU flag).
3. Vectorized array ops only — no per-element Python loops in hot paths (HPC-01 must not regress the existing `xp.trapz` / batch `rfft` shape).
4. No GPU→CPU transfers per Fisher iteration (HPC-01 guards the diagnostic `cp.asnumpy(...)` at `compute_Cramer_Rao_bounds:386` behind `if self._use_gpu`).
5. Factory plotting functions return `(fig, axes)` (VIZ-02 keeps this).
6. HPC-05 software-only unless verification escalates — any escalation triggers `/physics-change` with `[PHYSICS]` commit prefix.
7. All Phase 36/37/38 tests must remain GREEN (D-30).
