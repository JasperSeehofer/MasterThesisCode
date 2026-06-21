# Phase 39: HPC & Visualization Safe Wins — Context

**Gathered:** 2026-04-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Non-physics hygiene to prepare the pipeline for Phase 40 verification and the Stage 1/2 injection
campaigns. Seven requirements split across two areas:

- **HPC safe wins (HPC-01..HPC-05):** CPU-importable `parameter_estimation.py`, reduce Lustre I/O,
  decouple FFT plan-cache lifecycle from memory-pool freeing, remove dead code, verify `flip_hx`
  flag against the current `fastlisaresponse`.
- **Visualization safe wins (VIZ-01, VIZ-02):** Optional LaTeX rendering in production figures,
  bootstrap HDI band on the static convergence plot.

**In scope:**
- HPC-01: `parameter_estimation.py` CPU-importable via `self._xp` / `self._fft` shim
- HPC-02: `_crb_flush_interval = 25` (exact value per ROADMAP SC-2); SIGTERM handler retained
- HPC-03: `free_gpu_memory()` split — memory pool freed always, FFT cache cleared only on memory-pressure threshold
- HPC-04: delete `_crop_frequency_domain` (dead since `_get_cached_psd` replaced it)
- HPC-05: verify `flip_hx=True` against current `fastlisaresponse`; add citation comment (keep)
- VIZ-01: `apply_style(use_latex=True)` when `shutil.which("latex")` is truthy; mathtext fallback
- VIZ-02: 16/84 bootstrap HDI band on the right panel (CI width vs N) of `plot_h0_convergence`

**Out of scope:**
- Full posterior re-evaluation under v2.2 fixes — Phase 40 VERIFY-02
- Cluster job submission or new injection campaigns — Phases 41-42
- `LISA_configuration.py` unconditional `cupy` import — tracked in memory as known code-health bug; fix when that file is next touched
- Broader refactors or architectural changes — user decision "safe wins only" (2026-04-21 audit)
- Physics changes beyond conditional HPC-05 removal fallback
</domain>

<decisions>
## Implementation Decisions

### HPC-01: CPU-importable parameter_estimation.py

- **D-01:** Use the `self._xp` attribute pattern per CLAUDE.md HPC rules. Resolve once in `__init__` from a module-local `_get_xp(use_gpu)` helper; all 13 `cp.*` sites become `self._xp.*`.
- **D-02:** Parallel `self._fft` attribute for the FFT module. Module-local `_get_fft(use_gpu)` returns `cupyx.scipy.fft` when `use_gpu and _CUPY_AVAILABLE`, else `numpy.fft`. Both `cufft.rfft(x, axis=-1)` and `cufft.rfftfreq(n, dt)` have API-compatible `numpy.fft` peers.
- **D-03:** Helpers live module-local inside `parameter_estimation.py` (not a new shared module). Extraction to `master_thesis_code/_array_namespace.py` is deferred until a second module needs the pattern.
- **D-04:** Keep the `_CUPY_AVAILABLE` guard at module scope (already present) — `_get_xp` and `_get_fft` read it. CPU fall-back when `use_gpu=True` but `_CUPY_AVAILABLE=False` should emit a one-time WARNING (don't silently degrade).
- **D-05:** Acceptance criterion: `python -c "from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation"` succeeds in a venv where `uv pip uninstall cupy-cuda12x` has been run; `uv run pytest -m "not gpu"` runs tests without skipping on import failure.

### Claude's Discretion (HPC-01)
- Exact signature of `_get_xp` / `_get_fft` (module types vs proxy classes) — pick the shorter form.
- Whether to introduce a `float` type alias or leave `npt.NDArray[np.float64]` untouched — prefer the latter (CLAUDE.md Typing).
- One-time WARNING mechanism (module-level flag vs `warnings.warn`) — pick whichever is more idiomatic.

### HPC-02: CRB flush interval

- **D-06:** `self._crb_flush_interval: int = 25` exactly (ROADMAP SC-2). No configurability via CLI flag — the value is empirically tied to Lustre I/O characteristics, not per-run tuning.
- **D-07:** Verify SIGTERM handler (`main.py:351` calls `_pe_ref[0].flush_pending_results()`) still drains the buffer on termination. Add a unit/mocked test that writes 30 rows with interval=25 → 1 automatic flush + 1 manual flush on `flush_pending_results()`.

### HPC-03: FFT cache lifecycle

- **D-08:** Split `MemoryManagement.free_gpu_memory()` into two methods:
  - `free_memory_pool()` — always safe; frees memory-pool blocks only.
  - `clear_fft_cache()` — explicit cache invalidation; no-op on CPU.
- **D-09:** Introduce `free_gpu_memory_if_pressured(threshold: float = 0.8)` that calls both methods when `pool.total_bytes() / total_gpu_bytes() >= threshold`, and only `free_memory_pool()` otherwise. Existing `free_gpu_memory()` becomes a thin deprecated alias routing to the new pressure-gated call.
- **D-10:** Update `main.py:381` and `main.py:639` call sites to the new API. Per-simulation-step free of memory pool is retained; FFT cache survives across simulation steps until pressure threshold is crossed.
- **D-11:** Threshold default 0.8 (80% of total GPU memory). Rationale: H100 has 80 GB; cupy memory pool comfortably sits at ~40 GB in typical Fisher runs, so 80% ≈ 64 GB → cache only clears when allocation grows pathologically.
- **D-12:** Acceptance: grep `_fft_cache.clear` returns exactly one site (`clear_fft_cache()` method); no call site clears the cache per Fisher iteration.

### Claude's Discretion (HPC-03)
- Whether to read total GPU bytes via `cp.cuda.runtime.memGetInfo()[1]` or `GPUtil` (already imported).
- Whether the pressure trigger also logs a telemetry line — `_LOGGER.info("FFT cache cleared — pool usage {pct:.1%}")` is useful but not required.

### HPC-04: Dead code removal

- **D-13:** Delete `_crop_frequency_domain` from `parameter_estimation.py:334-347`. The function is already superseded by `_get_cached_psd` (line 102) which performs equivalent index selection + caching.
- **D-14:** Grep-verify no external caller: `rg "_crop_frequency_domain" master_thesis_code/ master_thesis_code_test/` must return empty after the removal.

### HPC-05: flip_hx verification (keep + citation)

- **D-15:** Primary plan: **verify and keep**. Read current `fastlisaresponse` source (`force_backend`-era — post 2024) + `arXiv:2204.06633` (Katz, Chua et al.) to confirm what `flip_hx=True` does today. Expected outcome: the flag maps ecliptic latitude to the correct LISA frame sign convention; `flip_hx=True` is still required because `is_ecliptic_latitude=False` is passed alongside.
- **D-16:** Add a reference comment directly above `flip_hx=True` in `waveform_generator.py:58`:
  ```python
  # flip_hx=True required when is_ecliptic_latitude=False; see fastlisaresponse
  # <version> source (ResponseWrapper.__init__) and Katz et al. (2022) arXiv:2204.06633
  ```
- **D-17:** **Fallback (conditional):** if verification finds `flip_hx=True` is obsolete or wrong given how Phase 36 now threads coordinates, switch to removal path: `/physics-change` with regression pickle (one seed-pinned event, pre-vs-post CRB identical to ~1e-12 after the flag is removed AND any downstream sign is compensated). `[PHYSICS]` commit prefix. Document decision in the plan's execution log.
- **D-18:** No physics change is expected under the primary path; HPC-05 remains software-only unless the verification escalates.

### VIZ-01: Optional LaTeX rendering

- **D-19:** In `main.py:generate_figures` (existing `apply_style()` call at `main.py:803`), detect TeX via `shutil.which("latex")` and call `apply_style(use_latex=True)` when truthy, else `apply_style()` (mathtext default per `_style.py:29`).
- **D-20:** Smoke test runs both branches: one mocked `which` returning `/usr/bin/latex`, one returning `None`. Verify no crash, `rcParams["text.usetex"]` true/false as expected.
- **D-21:** No new CLI flag. Auto-detection only. If a user needs to force mathtext on a machine with TeX installed, they can `PATH=$(echo $PATH | tr ':' '\n' | grep -v tex | paste -sd:)` — documented in `docs_src/` if needed. Decision: too niche to warrant a CLI surface.

### VIZ-02: Bootstrap HDI band on convergence plot

- **D-22:** Add 16/84 percentile shading on the **right panel only** (CI width vs N) of `plot_h0_convergence`. Matches ROADMAP SC-7's "sits inside the CI rails" phrasing — the CI curve is on the right panel.
- **D-23:** Source the bootstrap distribution from the existing `compute_m_z_improvement_bank` paired-bootstrap aggregator. The bank already computes per-subset-size bootstrap distributions (`DEFAULT_BOOTSTRAP = 200`, `DEFAULT_SEED = 20260410`); extract the 16/84 percentiles of CI-width per subset and `fill_between` between them on the right panel.
- **D-24:** Signature of `plot_h0_convergence` gains an optional `bootstrap_bank: ImprovementBank | None = None` kwarg. When provided, the band is drawn. When `None`, current behavior preserved (no band) — backward-compatible.
- **D-25:** `main.py:_gen_h0_convergence` (line 1015-1030) updates to pass the bank when `compute_m_z_improvement_bank(Path(output_dir), h_true=float(TRUE_H))` returns a non-None result.
- **D-26:** Visual smoke test: run the figure generator on `simulations/h_0_73`, assert the output PDF contains a `PolyCollection` artist (the `fill_between` result) on the second subplot.
- **D-27:** Left panel (posterior curve) gets no band — subset-sampling noise would confuse rather than clarify the statistical message.

### Claude's Discretion (VIZ-02)
- Alpha value of the shaded band (conventional 0.2-0.3 for 16/84).
- Whether to add a legend entry for the band or piggyback on the existing CI-width line label.
- Band color: match the `VARIANT_NO_MASS`/`VARIANT_WITH_MASS` color of the corresponding line (per variant).

### Cross-cutting Decisions

- **D-28:** HPC-05 is the only requirement that may escalate to a physics change. Plan explicitly branches: verification → decision point → (keep w/ citation comment) vs (`/physics-change` removal). Branch is resolved inline during execution, not pre-decided.
- **D-29:** All other requirements are software-only. Commit messages use standard prefixes (`refactor`, `perf`, `feat`, `test`, `docs`). No `[PHYSICS]` except conditional HPC-05 removal.
- **D-30:** All Phase 36/37/38 regressions must remain GREEN:
  - `master_thesis_code_test/test_coordinate_roundtrip.py` (9 tests)
  - `master_thesis_code_test/test_parameter_space_h.py` (Phase 37)
  - `test_l_cat_equivalence.py` (Phase 38)
  - Full suite baseline ≥524 tests (Phase 38 GREEN count).
- **D-31:** Test coverage for HPC-01 requires a way to exercise the CPU path. Options for CI lane tracking: (a) run existing `pytest -m "not gpu"` in a subprocess with `cupy` shadowed to raise `ImportError` at import time via `sys.modules["cupy"] = None` fixture, or (b) add a dedicated `tox`/`nox` lane with `uv sync --extra cpu --extra dev` followed by `pytest -m "not gpu"`. Claude's discretion — simpler option wins.
- **D-32:** Wave ordering (planner's discretion, guidance here): Wave 1 — HPC-01 (shim enables testability); Wave 2 — HPC-02, HPC-03, HPC-04, VIZ-01 (independent software fixes); Wave 3 — HPC-05 (verify; escalation branch); Wave 4 — VIZ-02 (depends on band data flow but independent of HPC fixes); Wave 5 — verification (SC-1..SC-7 harness).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` §Hardware/HPC (HPC-01..HPC-05) — exact REQ specs and expected I/O savings
- `.planning/REQUIREMENTS.md` §Visualization (VIZ-01, VIZ-02) — spec for LaTeX gating and HDI band
- `.planning/ROADMAP.md` §Phase 39 — success criteria SC-1..SC-7 (verbatim gate conditions)

### Prior phase context
- `.planning/phases/38-statistical-correctness/38-CONTEXT.md` — completed 2026-04-22; STAT-03 P_det zero-fill landed; 524 tests GREEN baseline
- `.planning/phases/37-parameter-estimation-correctness/37-CONTEXT.md` — completed 2026-04-22; per-parameter epsilon + h-threading landed
- `.planning/phases/36-coordinate-frame-fix/36-VERIFICATION.md` — coord frame regression pickle; must stay GREEN

### Code loci (production code to change)
- `master_thesis_code/parameter_estimation/parameter_estimation.py:19-27` — cupy/cufft import guards (starting point for HPC-01 shim)
- `master_thesis_code/parameter_estimation/parameter_estimation.py:99` — `_crb_flush_interval: int = 1` (HPC-02 target)
- `master_thesis_code/parameter_estimation/parameter_estimation.py:334-347` — `_crop_frequency_domain` (HPC-04 delete target)
- `master_thesis_code/memory_management.py:43-48` — `free_gpu_memory` (HPC-03 split target)
- `master_thesis_code/waveform_generator.py:58` — `flip_hx=True` (HPC-05 verification target)
- `master_thesis_code/main.py:803, 1015-1030, 1198-1209` — `apply_style` call + convergence figure wiring (VIZ-01, VIZ-02)
- `master_thesis_code/plotting/_style.py:13-47` — `apply_style(use_latex)` implementation (reference, no change)
- `master_thesis_code/plotting/convergence_plots.py:50-end` — `plot_h0_convergence` (VIZ-02 target)
- `master_thesis_code/plotting/convergence_analysis.py:341+` — `compute_m_z_improvement_bank` (VIZ-02 data source)

### External references
- `CLAUDE.md` §HPC / GPU Best Practices — mandatory `_get_xp` pattern, GPU imports must be guarded, vectorize operations, avoid GPU↔CPU transfers in hot paths
- `CLAUDE.md` §Typing Conventions — `list[...]` not `List[...]`; no `from __future__ import annotations`; `npt.NDArray[np.float64]` for arrays
- `CLAUDE.md` §Math/Physics Validation Workflow — physics-change protocol for conditional HPC-05 removal path
- Katz, Chua et al. (2022) `fastlisaresponse`, arXiv:2204.06633 — `ResponseWrapper` API; reference for HPC-05 `flip_hx` citation comment
- Vallisneri (2008), arXiv:gr-qc/0703086 — already-cited Fisher stencil reference; HPC regression must not perturb this behavior

### Existing tests (must remain GREEN)
- `master_thesis_code_test/test_coordinate_roundtrip.py` — 9 tests (Phase 36)
- `master_thesis_code_test/test_parameter_space_h.py` — Phase 37 regression
- `master_thesis_code_test/test_l_cat_equivalence.py` — Phase 38 STAT-02
- Full suite baseline ≥524 tests (Phase 38 GREEN)

### Skills/protocols (triggered during execution)
- `.claude/skills/physics-change/SKILL.md` — invoked if HPC-05 escalates to removal
- `.claude/skills/gpu-audit/SKILL.md` — run after HPC-01 refactor
- `.claude/skills/check/SKILL.md` — pre-commit gate (ruff + mypy + pytest)
- `.claude/skills/pre-commit-docs/SKILL.md` — CHANGELOG/TODO/CLAUDE.md sync check before commit
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `parameter_estimation.py:102` `_get_cached_psd` already caches PSD + frequency axis per waveform length — HPC-04 dead-code removal validates this is the superseding function.
- `memory_management.py` already splits memory-pool vs fft-cache concerns internally (`self.memory_pool`, `self._fft_cache`) — HPC-03 surfaces that split as public API.
- `plotting/_style.py:13-47` `apply_style(use_latex: bool)` — VIZ-01 just wires the flag; no new plotting logic needed.
- `plotting/convergence_analysis.py` `compute_m_z_improvement_bank` — VIZ-02 data source is already paired-bootstrap; band is a percentile lookup on existing arrays.
- `main.py:1015-1030` `_gen_h0_convergence` factory function — VIZ-02 wiring sits inside an existing `manifest.append()` block.

### Established Patterns
- `try/except ImportError` + `_CUPY_AVAILABLE` module flag (already used in `memory_management.py`, `parameter_estimation.py`, will be formalized in HPC-01).
- `self._<attr>` for runtime-resolved GPU/CPU switches (`self._use_gpu`, `self._use_five_point_stencil`).
- Factory functions in `plotting/` return `(fig, ax)` — preserved by VIZ-02.
- Pre-commit hook chain: ruff → mypy → pytest.
- Skills-driven workflow: `physics-change` for formula/flag changes; `check` before commit; `gpu-audit` after array/GPU code changes.

### Integration Points
- `main.py:307, 356, 588` instantiate `MemoryManagement(use_gpu=...)` — HPC-03 API changes propagate through these three call sites (plus the two `free_gpu_memory()` invocations).
- `main.py:803` `apply_style()` call sits inside `generate_figures` — single edit site for VIZ-01.
- `main.py:1022` `plot_h0_convergence(...)` call passes `h_values` and `event_posteriors` — VIZ-02 adds one kwarg.
- `main.py:1204` `compute_m_z_improvement_bank(Path(output_dir), h_true=float(TRUE_H))` already computed for the M_z panel — VIZ-02 reuses this result rather than recomputing.

### Known Constraints
- `fastlisaresponse` compiled extension crashes (SIGILL) on CPUs without AVX (`waveform_generator.py:45-48`). HPC-01 must not trigger that import at module load — the lazy-import pattern is already in place.
- `_crb_flush_interval = 1` (current) writes one row per detection; raising to 25 reduces writes per SLURM task but means up to 24 rows lost on uncaught crash. Mitigation: SIGTERM handler at `main.py:351` drains the buffer (HPC-02 test confirms).
- `apply_style()` is called multiple times across the module (`bayesian_inference_mwe.py:67`, `fisher_plots.py:509`, `main.py:35, 803`). VIZ-01 should only gate the **production figure** call at `main.py:803`; the rest stay mathtext.
</code_context>

<specifics>
## Specific Ideas

- "Safe wins only" (user decision from 2026-04-21 audit) — no architectural refactors, no multi-day explorations.
- `flip_hx=True` verification must actually read the `fastlisaresponse` source / arXiv paper — not just grep the project. The resulting citation comment lives permanently in the code.
- HPC-03 threshold value (80%) is a default. If GPU telemetry during Phase 40 verification shows cache pressure at lower utilization, tune down in a follow-up quick task.
- VIZ-02 band should be visually subtle (alpha 0.2-0.3). The two-panel figure already has the `1/sqrt(N)` reference line; the band must not visually compete with it.
- VIZ-01 auto-detection is all-or-nothing per run — if a user wants mixed mode (some figures LaTeX, some mathtext) they need a separate figure-generation call. Out of scope.
</specifics>

<deferred>
## Deferred Ideas

- **Extract `_get_xp` / `_get_fft` to shared module** (`master_thesis_code/_array_namespace.py`): only if a second module needs the pattern (e.g., `LISA_configuration.py` fix). Tracked as future refactor.
- **`LISA_configuration.py` unconditional `cupy` import**: known code-health bug, fix when that file is next touched for a physics/feature reason. Not in Phase 39 scope.
- **`--use_latex` CLI override**: rejected as too niche. Users who need to force mathtext despite TeX being installed can adjust `PATH` or edit `main.py`. Revisit if demand emerges.
- **HPC-03 pressure-trigger telemetry**: `_LOGGER.info("FFT cache cleared, pool usage {pct:.1%}")` helpful for Phase 40 diagnostics. Claude's discretion during implementation; not a gate.
- **Adaptive `_crb_flush_interval`**: tune based on job length / Lustre latency. Deferred — 25 is empirically well-behaved.
- **Band on left panel of `plot_h0_convergence`** (posterior curve): rejected — subset-sampling noise would confuse rather than inform. If bootstrap-over-posteriors is valuable, revisit as a separate figure in a future visualization phase.
- **HPC-05 escalation to removal**: only fires if `flip_hx` verification finds the flag obsolete/incorrect. If it does, `/physics-change` protocol is triggered inline during execution; no pre-plan required.
- **Deprecate `free_gpu_memory()` alias entirely**: once HPC-03 split lands, callers should migrate to `free_gpu_memory_if_pressured()`. Full deprecation is a quick task for a future phase.
</deferred>

---

*Phase: 39-hpc-visualization-safe-wins*
*Context gathered: 2026-04-22*
