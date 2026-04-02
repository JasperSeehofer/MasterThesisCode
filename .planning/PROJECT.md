# EMRI Parameter Estimation — HPC Integration

## What This Is

A gravitational wave parameter estimation pipeline for LISA Extreme Mass Ratio Inspirals (EMRIs). Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramér-Rao bounds, and (2) CPU-based Bayesian inference that evaluates the Hubble constant posterior. The v1.0 milestone delivered production-ready HPC/cluster support: both pipelines now run on bwUniCluster 3.0 at KIT as SLURM array jobs with a single-command submission, reproducible seeding, failure recovery, and complete documentation.

## Core Value

The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramér-Rao bounds for statistically meaningful Hubble constant posteriors.

## Shipped Milestones

- **v1.4 Posterior Numerical Stability** — shipped 2026-04-02 (Phases 21-23). Log-space accumulation, physics-motivated likelihood floor, underflow detection. Deployed to bwUniCluster at `5793f70`. Full details: `.planning/milestones/v1.4-ROADMAP.md`
- **v1.2 Production Campaign & Physics Corrections** — shipped 2026-04-01 (Phases 9-13, 11.1). Confusion noise + five-point stencil physics fixes, simulation-based P_det, 1000+ detection production campaign, H0 sweep baselines. Full details: `.planning/milestones/v1.2-ROADMAP.md`

## Shipped Milestones (continued)

- **v1.3 Visualization Overhaul** — shipped 2026-04-02 (Phases 14-19). Full publication-quality visualization stack: centralized style, Fisher data layer, enhanced existing plots, sky maps + corner plots + convergence diagnostics, 15-figure batch pipeline. Full details: `.planning/milestones/v1.3-ROADMAP.md`

## Current State (v1.3 Phase 19 complete 2026-04-02)

v1.3 Visualization Overhaul milestone complete. Phase 19 adds campaign dashboard composite (2x2 mosaic: H0 posterior, SNR, detection yield, sky map) and batch figure generation pipeline (`generate_figures()` with 15-entry manifest). PDF size optimization via `rasterized=True` on scatter calls and 2MB warning check. 342 tests pass at 63% coverage. All 6 phases (14-19) shipped.

## Current State (v1.3 Phase 18 complete 2026-04-02)

Phase 18 complete. Four new plot factory functions: Mollweide sky localization map, Fisher corner plots (via `corner` library), H0 convergence diagnostics (two-panel posterior narrowing + CI width), and detection efficiency curves with Wilson score CI. All follow (Figure, Axes) convention with smoke tests. 334 tests pass at 62% coverage.

## Current State (v1.4 Phase 23 complete 2026-04-02)

v1.4 milestone complete. Physics-floor strategy deployed to bwUniCluster (5793f70). Validated locally: physics-floor MAP=0.66 matches exclude MAP=0.66 (|Δ|=0.00 < 0.05 acceptance criterion — PASS). Cluster confirmed at correct commit; evaluate SLURM jobs were pending at deploy time and will run with the numerical stability fixes. Report at `results/v1.4-validation.md`.

## Current State (v1.3 Phase 21 complete 2026-04-02)

Posterior combination pipeline complete: log-space accumulation with 4 zero-handling strategies (naive, exclude, per-event-floor, physics-floor fallback), diagnostic reports identifying zero-event root causes, comparison tables with MAP estimates. CLI wiring via `--combine --strategy`. Validated against campaign data (naive MAP=0.86 reproduced). 299 tests pass at 58% coverage.

## Current State (v1.3 Phase 15 complete 2026-04-02)

Style infrastructure in place: centralized figure presets (single/double column), LaTeX toggle, semantic color palette, shared label constants, consolidated helpers. All 6 plot modules wired to shared imports. 246 tests pass. Foundation ready for Phases 16-19.

## Current State (v1.2 Phase 10 complete 2026-03-29)

Fisher matrix now uses O(epsilon^4) five-point stencil derivatives by default (PHYS-01 resolved). Condition number logging and CRB safety checks added. CRB timeout increased to 90s. 198 CPU tests pass at 40.8% coverage. Galactic confusion noise (Phase 9) and stencil derivatives (Phase 10) are both wired — ready for validation campaign on cluster.

## Current State (v1.1 shipped 2026-03-29)

Pipeline validated end-to-end on bwUniCluster 3.0. Smoke-test campaign (3 tasks, 10 steps, seed 100) produced 20 detections, 18 passed the 10% d_L error filter, and the evaluation pipeline generated an H0 posterior at h=0.73. All quantitative checks passed.

## Current State (v1.0 shipped 2026-03-27)

- **Cluster pipeline operational:** `submit_pipeline.sh --tasks N --steps M --seed S` submits the full simulate→merge→evaluate chain
- **186 CPU tests passing**, 43% coverage (gate at 25%)
- **~24,400 LOC** (Python + Shell), 9 cluster scripts in `cluster/`
- **Documentation complete:** `cluster/README.md` quickstart, `CLAUDE.md` cluster section, `README.md` HPC pointer
- **Known physics bugs remain** (Fisher stencil, confusion noise, wCDM silent, etc.) — tracked in CLAUDE.md

## Requirements

### Validated

- ✓ EMRI simulation pipeline (waveform generation, SNR, Fisher matrix, Cramér-Rao bounds) — existing
- ✓ Bayesian inference pipeline (H₀ posterior from detection catalog + galaxy catalog) — existing
- ✓ Seed-based reproducibility with `run_metadata.json` tracking — existing
- ✓ `--simulation_index` CLI flag for partitioning output CSVs — existing
- ✓ Merge and prepare scripts for post-simulation data processing — existing
- ✓ GPU memory management class — existing
- ✓ CuPy/CUDA GPU acceleration with guarded imports — existing
- ✓ Multiprocessing pool for Bayesian likelihood evaluation — existing
- ✓ Scientific plotting subpackage with callback-based simulation monitoring — existing
- ✓ Pre-commit hooks (ruff, mypy), CI pipeline, test suite — existing
- ✓ `--use_gpu` CLI flag threaded through all pipelines — v1.0 Phase 1
- ✓ `--num_workers` CLI flag for Bayesian inference pool size — v1.0 Phase 1
- ✓ CPU-safe `MemoryManagement` (guard GPUtil, no-op on CPU nodes) — v1.0 Phase 1
- ✓ Non-interactive merge script (`--delete-sources` flag) — v1.0 Phase 2
- ✓ Environment module loader (`modules.sh`) for bwUniCluster 3.0 — v1.0 Phase 3
- ✓ One-time cluster setup script (uv, workspace allocation) — v1.0 Phase 3
- ✓ SLURM metadata in `run_metadata.json` (job ID, array task ID, node, GPU info) — v1.0 Phase 4
- ✓ SLURM job scripts for simulation (GPU array), merge, and evaluation (CPU) — v1.0 Phase 4
- ✓ Workflow orchestrator chaining jobs via `--dependency=afterok` — v1.0 Phase 4
- ✓ Cluster documentation (quickstart, monitoring, troubleshooting) — v1.0 Phase 5
- ✓ CLAUDE.md and README.md updates for cluster deployment — v1.0 Phase 5
- ✓ Data reset: stale simulation outputs removed from tracking, .gitignore verified — v1.1 Phase 6
- ✓ SSH key-based cluster access with ControlMaster 2FA session reuse — v1.1 Phase 7
- ✓ Cluster environment preflight verified (modules, GPU partition, workspace, venv+imports) — v1.1 Phase 7
- ✓ Claude SSH integration via `ssh bwunicluster '<cmd>'` for direct cluster command execution — v1.1 Phase 7
- ✓ Test simulation run completed (3 tasks, 10 steps, seed 100) with timing data — v1.1 Phase 8
- ✓ Evaluation pipeline produces H₀ posterior from fresh Cramér-Rao bounds — v1.1 Phase 8
- ✓ Results validated (SNR physical, seeds correct, detection rates reasonable) — v1.1 Phase 8

### Active

- (No active milestone — all planned milestones shipped; run `/gsd:new-milestone` for next milestone)

### Recently Validated (v1.2 / v1.3 / v1.4)

- ✓ Fisher matrix upgrade to 5-point stencil derivative — v1.2 Phase 10
- ✓ Galactic confusion noise added to PSD — v1.2 Phase 9
- ✓ KDE detection probability replaced by simulation-based P_det — v1.2 Phase 11.1
- ✓ Production CRB catalog (1000+ detections, seed 200) — v1.2 Phase 12
- ✓ H₀ posterior sweep [0.6, 0.9], baselines documented — v1.2 Phase 13
- ✓ Log-space posterior accumulation (replace np.prod with log-sum-exp) — v1.4 Phase 21
- ✓ Physics-motivated likelihood floor in single_host_likelihood — v1.4 Phase 22
- ✓ Underflow detection replacing dead check_overflow — v1.4 Phase 22
- ✓ Updated code deployed to cluster at 5793f70, validation PASS — v1.4 Phase 23
- ✓ 23 smoke tests covering all plot factory functions + rcParams regression test — v1.3 Phase 14
- ✓ Centralized style system (_colors.py, _labels.py, REVTeX presets, LaTeX/mathtext toggle) — v1.3 Phase 15
- ✓ CRB data layer + Fisher plot factories (error ellipses, characteristic strain, uncertainty violins) — v1.3 Phase 16
- ✓ All existing plots upgraded: credible intervals, heatmap contours, injected-vs-recovered scatter — v1.3 Phase 17
- ✓ New plot types: Mollweide sky map, Fisher corner plots, H0 convergence diagnostics, efficiency curve — v1.3 Phase 18
- ✓ 15-figure batch pipeline (generate_figures() manifest) + 2×2 campaign dashboard — v1.3 Phase 19

### Out of Scope

- Multi-node MPI distribution — single GPU per simulation task is sufficient; array jobs provide parallelism
- Dask/Ray distributed computing — overkill for this use case
- GPU CI runners — cluster testing is manual; GitHub Actions stays CPU-only
- Checkpointing/resume — each array task is short enough; failed tasks can be re-submitted
- Self-hosted runners on cluster — security and maintenance overhead not justified for a thesis
- Apptainer container definition — module-based approach works well; container deferred as not needed for thesis

## Context

- **Cluster:** bwUniCluster 3.0 at KIT Karlsruhe (bwHPC federation), SLURM scheduler, GPU nodes with NVIDIA H100 GPUs (CUDA 12), `module load` environment system, workspace mechanism (`ws_allocate`)
- **Architecture fit:** `--simulation_index` maps to `SLURM_ARRAY_TASK_ID`. Each EMRI event is independent, making array jobs the ideal parallelization strategy.
- **All v1.0 code blockers resolved:** CPU-safe imports, batch-compatible scripts, environment setup, job infrastructure, documentation.
- **v1.1 cluster integration fixes:** `few` v2.0.0rc1 API updates (`force_backend="cuda12x"`, `ESAOrbits`), sbatch path resolution, SIGTERM flush handler, 30s waveform timeout.
- **Smoke-test results:** 20 detections from 30 EMRI events (3 tasks × 10 steps), 18 passed 10% d_L filter, H₀ posterior generated at h=0.73.
- **Resolved physics bugs:** Fisher stencil (fixed v1.2 Phase 10), confusion noise (fixed v1.2 Phase 9), posterior underflow/zeros (fixed v1.4).
- **Remaining physics bugs:** wCDM params silently ignored, hardcoded 10% σ(d_L), WMAP-era cosmology (Omega_m=0.25/H=0.73), galaxy redshift uncertainty scaling. All tracked in CLAUDE.md.

## Constraints

- **GPU:** CUDA 12 required for `cupy-cuda12x` and `fastemriwaveforms-cuda12x` — must use GPU partition on cluster
- **GSL:** Build-time requirement for `fastemriwaveforms` — must be available via module or container
- **uv:** Primary package manager; installed to `~/.local/bin` on login nodes
- **Workspace:** bwHPC workspaces expire (default 60 days, extendable) — results must be copied to persistent storage
- **Network:** Compute nodes may have restricted outbound access — all dependency installation happens on login nodes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| SLURM array jobs over single long job | Per-task time limits, automatic retry, better queue scheduling, natural fit with `--simulation_index` | ✓ Good — clean mapping, easy failure recovery |
| Module-based primary, Apptainer deferred | Simpler debugging and cluster storage integration for a thesis | ✓ Good — modules work well, container not needed |
| Seed = base + array task ID | Deterministic, reproducible, different per task | ✓ Good — exact reproducibility confirmed |
| `cluster/` directory in repo | Job scripts version-controlled alongside code they run | ✓ Good — single git pull deploys everything |
| `emri-merge`/`emri-prepare` entry points | Console scripts avoid fragile `python -m scripts.*` in sbatch | ✓ Good — clean CLI interface |
| afterok dependency chaining | Pipeline runs unattended after single submission | ✓ Good — three stages chain automatically |
| afterany for merge step | Merge runs even if some simulate tasks timeout | ✓ Good — collects partial results from timed-out tasks |
| `force_backend="cuda12x"` | few auto-detection fell back to CPU on GPU nodes | ✓ Good — explicit backend selection is reliable |
| 30s waveform timeout | Some EMRI parameter combinations hang indefinitely | ⚠️ Revisit — root cause TBD, mitigated not fixed |
| 10% d_L error threshold | Forward-diff Fisher matrix too imprecise for 5% | ⚠️ Revisit — will improve with 5-point stencil |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-02 after v1.3 milestone completion*
