# EMRI Dark Siren H₀ Inference

## What This Is

A dark siren inference pipeline for measuring the Hubble constant H₀ from LISA Extreme Mass Ratio Inspiral (EMRI) detections. Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramér-Rao bounds on bwUniCluster 3.0, and (2) CPU-based Bayesian inference that evaluates the H₀ posterior using the GLADE+ galaxy catalog with completeness correction (Gray et al. 2020). The thesis is complete; the project now focuses on finishing the research and wrapping results into a paper.

## Core Value

Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.

## Shipped Milestones

- **v1.4 Posterior Numerical Stability** — shipped 2026-04-02 (Phases 21-23). Log-space accumulation, physics-motivated likelihood floor, underflow detection. Deployed to bwUniCluster at `5793f70`. Full details: `.planning/milestones/v1.4-ROADMAP.md`
- **v1.3 Visualization Overhaul** — shipped 2026-04-02 (Phases 14-19). Full publication-quality visualization stack: centralized style, Fisher data layer, enhanced existing plots, sky maps + corner plots + convergence diagnostics, 15-figure batch pipeline. Full details: `.planning/milestones/v1.3-ROADMAP.md`
- **v1.2 Production Campaign & Physics Corrections** — shipped 2026-04-01 (Phases 9-13, 11.1). Confusion noise + five-point stencil physics fixes, simulation-based P_det, 1000+ detection production campaign, H0 sweep baselines. Full details: `.planning/milestones/v1.2-ROADMAP.md`
- **v1.1 Clean Simulation Campaign** — shipped 2026-03-29 (Phases 6-8). Data cleanup, cluster access, end-to-end validation. Full details: `.planning/milestones/v1.1-ROADMAP.md`
- **v1.0 EMRI HPC Integration** — shipped 2026-03-27 (Phases 1-5). CPU-safe imports, batch scripts, cluster env, SLURM jobs, documentation. Full details: `.planning/milestones/v1.0-ROADMAP.md`

## Current State (v1.4 complete, 2026-04-04)

All 5 milestones shipped (v1.0–v1.4). 23 phases, 41 plans completed across 9 days. Pipeline runs end-to-end on bwUniCluster 3.0: GPU-accelerated EMRI simulation → Cramér-Rao bounds → Bayesian H₀ posterior. Production CRB catalog (1000+ detections, seed 200) with corrected physics (five-point stencil, confusion noise, simulation-based P_det). Log-space posterior combination with physics-motivated likelihood floor deployed and validated. 15-figure publication-quality visualization pipeline. 342 tests at 63% coverage.

**Remaining physics bugs** (tracked in CLAUDE.md): wCDM params silently ignored, hardcoded 10% σ(d_L), WMAP-era cosmology, galaxy redshift uncertainty scaling.

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

## Current Milestone: v1.5 Galaxy Catalog Completeness Correction

**Goal:** Implement the Gray et al. (2020) completeness-corrected dark siren likelihood to eliminate the systematic H0 bias caused by GLADE+ incompleteness at z > 0.08.

**Target features:**
- GLADE+ completeness estimation f(z) from B-band luminosity comparison
- Completion term in the dark siren likelihood for uncataloged host galaxies
- Modified p_Di() combining catalog + completion terms weighted by f(z)
- Comoving volume element function in physical_relations.py
- Verification: limiting cases (f=1, f=0), bias reduction on 534-detection dataset

**Research specification:** `.gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md`

### Active

- [ ] GLADE+ completeness function f(z) estimated and validated
- [ ] Completion term added to dark siren likelihood (Gray et al. 2020 Eq. 9)
- [ ] p_Di() modified to combine catalog + completion terms
- [ ] Comoving volume element added to physical_relations.py
- [ ] Bias investigation tests re-run showing MAP shift toward h=0.73
- [ ] Cluster deployment and production evaluation

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
- Self-hosted runners on cluster — security and maintenance overhead not justified for this project
- Apptainer container definition — module-based approach works well; container not needed

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
*Last updated: 2026-04-04 — v1.5 milestone started*
