# Phase 39: HPC & Visualization Safe Wins — Verification

**Date:** 2026-04-23
**Phase:** 39
**Requirements:** HPC-01, HPC-02, HPC-03, HPC-04, HPC-05, VIZ-01, VIZ-02

## Summary

All 7 ROADMAP Phase 39 success criteria PASS. 540 tests GREEN on the full CPU suite (524 Phase-38 baseline + 4 new from 39-04 + 8 new from 39-03 + 4 new from 39-02 = 540). Phase 36/37/38 regressions intact (9 + 4 + 3 = 16/16). Lint and mypy clean across the package (57 source files).

## Success Criteria

### SC-1 — HPC-01 CPU-importable parameter_estimation.py
**Status:** [x] PASS
**Evidence:**
- `python -c "import sys; sys.modules['cupy']=None; ...; from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation; print('OK')"` → `OK`
- 11 `self._xp` references and 5 `self._fft` references inside the class body (≥8 + ≥4 required)
- `_get_xp` and `_get_fft` module-local helpers added
- Commit: `b3cec75` (refactor) — see also `c0906db` (RED tests)

### SC-2 — HPC-02 _crb_flush_interval = 25 + SIGTERM drain
**Status:** [x] PASS
**Evidence:**
- `rg "_crb_flush_interval: int = 25" parameter_estimation.py` → 1 match (line 128)
- `test_sigterm_drain_with_flush_interval_25` → 1 passed
- Existing `test_crb_buffer_auto_flushes_at_interval` still passes (instance-level override to 2)
- Commit: `815ac4a` (perf) — see also `555304f` (RED test)

### SC-3 — HPC-03 FFT cache lifecycle decoupled
**Status:** [x] PASS
**Evidence:**
- `rg "_fft_cache\.clear" master_thesis_code/` → exactly **1** (inside `clear_fft_cache`)
- `rg "free_gpu_memory_if_pressured" main.py` → **2** (lines 381 and 639 both migrated)
- `MemoryManagement` now exposes `free_memory_pool`, `clear_fft_cache`, `free_gpu_memory_if_pressured(threshold=0.8)`; original `free_gpu_memory()` retained as `DeprecationWarning`-emitting alias (D-09)
- Pressure threshold computed via `cp.cuda.runtime.memGetInfo()` (no GPUtil dep)
- Commit: `49e3d60` (refactor) + `6feeeb3` (call-site migration) — see also `14504a7` (RED tests)

### SC-4 — HPC-04 _crop_frequency_domain removed
**Status:** [x] PASS
**Evidence:**
- `rg "_crop_frequency_domain" master_thesis_code/ master_thesis_code_test/` → **empty** (no matches)
- Method body removed; superseded by `_get_cached_psd`
- Commit: `b3cec75` (bundled with HPC-01 — single file, single refactor)

### SC-5 — HPC-05 flip_hx documented (KEEP path)
**Status:** [x] PASS (path: **KEEP**)
**Evidence:**
- `arXiv:2204.06633` cited in `master_thesis_code/waveform_generator.py:58` (2-line comment directly above `flip_hx=True`)
- Decision recorded in `.planning/phases/39-hpc-visualization-safe-wins/39-05-VERIFICATION.md` with `[x] KEEP` marker
- Verified against installed `fastlisaresponse 1.1.17` ResponseWrapper docstring (lines 670-671, 693-696) and `__call__` source (lines 819-821, 830-831). `few 2.0.0rc1` emits `h_+ - i*h_x`; wrapper conjugates to `h_+ + i*h_x`. Removing the flag would silently invert h_x sign in every TDI channel.
- No `[PHYSICS]` commit; no regression pickle needed (software-only)
- Commit: `35d9366` (citation comment) + `5b182d1` (VERIFICATION record)

### SC-6 — VIZ-01 LaTeX auto-detection
**Status:** [x] PASS
**Evidence:**
- `rg "shutil\.which\(\"latex\"\)" main.py` → 1 match (in `generate_figures`)
- `rg "apply_style\(use_latex=True\)" main.py` → 1 match (gated branch)
- 3 untouched `apply_style()` call sites verified intact (D-21)
- LaTeX-keyed smoke tests → 2 passed (TestApplyStyleLatexGating: with-LaTeX + without-LaTeX branches)
- Commit: `c1cbaac` (feat)

### SC-7 — VIZ-02 HDI band on convergence plot
**Status:** [x] PASS
**Evidence:**
- `rg "bootstrap_bank" convergence_plots.py` → **8** matches (≥6 required)
- `rg "bootstrap_bank=bootstrap_bank" main.py` → 1 match (`_gen_h0_convergence` wires `compute_m_z_improvement_bank` output)
- `bootstrap_bank: ImprovementBank | None = None` kwarg, backward-compatible default
- Right-panel-only `fill_between` at `alpha=0.2`, `zorder=2`, primary + alt variants
- Bootstrap-keyed tests → 2 passed
- Commit: `fd1953c` (convergence_plots refactor) + `c1cbaac` (main.py wiring)

## Regression Check (D-30 invariant)

- `test_coordinate_roundtrip.py` (Phase 36 ecliptic-frame fix): **9/9 GREEN**
- `test_parameter_space_h.py` (Phase 37 PE correctness): **4/4 GREEN**
- `test_l_cat_equivalence.py` (Phase 38 statistical correctness): **3/3 GREEN**
- Full CPU suite (`-m "not gpu and not slow"`): **540 passed, 6 skipped, 16 deselected** — 0 failures

## Lint / Type Gate

- `uv run ruff check master_thesis_code/` → All checks passed
- `uv run mypy master_thesis_code/` → Success: no issues found in 57 source files

## GitHub Integration

No open GitHub issues match HPC-01..HPC-05 or VIZ-01..VIZ-02 directly. Existing open issues (#4, #5, #6, #7, #8) cover separate physics/design items (wCDM, Pipeline A 10% distance error, WMAP cosmology, redshift uncertainty scaling, two-pipeline divergence) and are tracked separately for the Paper Submission milestone.

Note: Issue #2 ("Fisher matrix forward difference") is marked open but was actually fixed in Phase 10. Closing it is out of Phase 39 scope but recommended in a follow-up.

## Plan Summary Pointers

- `39-01-SUMMARY.md` — HPC-01 self._xp/self._fft shim + HPC-04 dead code removal
- `39-02-SUMMARY.md` — HPC-02 batch CRB writes + SIGTERM drain test
- `39-03-SUMMARY.md` — HPC-03 pressure-gated GPU memory API split
- `39-04-SUMMARY.md` — VIZ-01 LaTeX auto-detect + VIZ-02 bootstrap HDI band
- `39-05-SUMMARY.md` — HPC-05 flip_hx verification (KEEP)
- `39-05-VERIFICATION.md` — HPC-05 decision record with ResponseWrapper source excerpts
