# Phase 39: HPC & Visualization Safe Wins — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 39-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-22
**Phase:** 39-hpc-visualization-safe-wins
**Areas discussed:** HPC-01 shim style, HPC-03 cache policy, HPC-05 flip_hx, VIZ-02 HDI band

---

## HPC-01 shim style

| Option | Description | Selected |
|--------|-------------|----------|
| self._xp attribute | Follows CLAUDE.md canonical pattern. One _get_xp helper; all 13 sites become self._xp.* | ✓ |
| Per-call _CUPY_AVAILABLE guards | Minimal diff. Each cp.* site gets an if/else. Verbose but local. | |
| Module-level xp = cp or np | Simplest but ignores self._use_gpu — breaks CLI-driven flag rule. | |

**User's choice:** self._xp attribute (Recommended)
**Notes:** Canonical per CLAUDE.md HPC rules.

### Follow-up: FFT module gating

| Option | Description | Selected |
|--------|-------------|----------|
| self._fft = cufft or np.fft | Parallel to self._xp. Module-local _get_fft helper. | ✓ |
| Route FFT through self._xp.fft | Works but loses cupyx.scipy.fft (faster than cupy.fft). | |
| Keep cufft import guarded, branches inline | Minimal class change but duplicates dispatch logic. | |

**User's choice:** self._fft = cufft or np.fft (Recommended)

### Follow-up: Helper location

| Option | Description | Selected |
|--------|-------------|----------|
| Module-local inside parameter_estimation.py | CLAUDE.md example style. Scoped tight. | ✓ |
| New master_thesis_code/_array_namespace.py | Central utility; reusable for LISA_configuration.py later. | |

**User's choice:** Module-local (Recommended)
**Notes:** Extract later if a second module needs the pattern.

---

## HPC-03 cache policy

**Ground truth established during discussion:** audit claim of "per-Fisher-iteration FFT clear" is false — `free_gpu_memory()` is called per-simulation-step, not per-iteration. Parameter_estimation.py:207/253 only logs pool bytes, doesn't clear.

| Option | Description | Selected |
|--------|-------------|----------|
| Decouple + threshold | Split free_gpu_memory() into free_memory_pool() + clear_fft_cache(). Clear only when pool > threshold. | ✓ |
| Verify + document only | Current behavior already matches SC-3; just add test + close. No code change. | |
| Drop fft cache clear entirely | Only clear on dedicated call. Biggest perf win but memory risk. | |

**User's choice:** Decouple + threshold (Recommended)
**Notes:** Threshold default 80% of GPU total memory (Claude's discretion). Earns the REQ with a real optimization rather than a documentation-only close.

---

## HPC-05 flip_hx

| Option | Description | Selected |
|--------|-------------|----------|
| Verify + keep with citation | Read current fastlisaresponse source + arXiv:2204.06633 to confirm semantics; add citation comment. No /physics-change. | ✓ |
| Remove via /physics-change | Regression pickle + [PHYSICS] commit. Cleaner but bigger diff. | |
| Conditional: verify first, then decide | Plan two subtasks with inline decision point. | |

**User's choice:** Verify + keep with citation (Recommended)
**Notes:** Fallback to removal only if verification finds flag is obsolete/wrong. Recorded as conditional branch (D-17) inside the primary plan.

---

## VIZ-02 HDI band

| Option | Description | Selected |
|--------|-------------|----------|
| Right panel (CI width vs N) only | Matches ROADMAP SC-7 "sits inside the CI rails"; communicates convergence-rate uncertainty. | ✓ |
| Left panel (posterior curve) only | Bootstrap envelope of combined posterior. Visually striking but mostly subset-sampling noise. | |
| Both panels | More info density but left band can dominate visually. | |

**User's choice:** Right panel (CI width vs N) only (Recommended)
**Notes:** Bootstrap source is the existing `compute_m_z_improvement_bank` paired-bootstrap aggregator. Band is 16/84 percentile `fill_between` at each subset size.

---

## Claude's Discretion

- Exact signature of `_get_xp` / `_get_fft` helpers (module types vs proxy classes).
- One-time CPU-fallback WARNING mechanism (module-level flag vs `warnings.warn`).
- HPC-03 threshold value: 80% of GPU total memory (default); tunable via kwarg.
- Whether `free_gpu_memory_if_pressured()` logs a telemetry line on cache clear.
- Reading total GPU bytes: `cp.cuda.runtime.memGetInfo()[1]` vs `GPUtil`.
- VIZ-02 alpha (0.2-0.3) and legend entry for the shaded band.
- Wave ordering exact shape (guidance provided in D-32).
- CI lane choice for HPC-01 testability (`sys.modules["cupy"] = None` fixture vs dedicated tox lane).

## Deferred Ideas

- Extract `_get_xp` / `_get_fft` to shared module (only if second module needs it).
- `LISA_configuration.py` unconditional `cupy` import — fix when file is next touched.
- `--use_latex` CLI override — rejected as too niche.
- HPC-03 pressure-trigger telemetry line.
- Adaptive `_crb_flush_interval`.
- Band on left panel of `plot_h0_convergence` (rejected for visual clarity).
- HPC-05 escalation to removal (conditional on verification outcome).
- Full deprecation of `free_gpu_memory()` alias after HPC-03 lands.
