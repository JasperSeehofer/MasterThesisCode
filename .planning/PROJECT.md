# EMRI Dark Siren H₀ Inference

## What This Is

A dark siren inference pipeline for measuring the Hubble constant H₀ from LISA Extreme Mass Ratio Inspiral (EMRI) detections. Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramér-Rao bounds on bwUniCluster 3.0, and (2) CPU-based Bayesian inference that evaluates the H₀ posterior using the GLADE+ galaxy catalog with completeness correction (Gray et al. 2020). The thesis is complete; the project now focuses on finishing the research and wrapping results into a paper.

## Core Value

Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.

## Shipped Milestones

- **v2.1 H₀ Bias Resolution** — shipped 2026-04-09 (Phases 30–34, 10 plans). L_comp denominator fixed to full-volume D(h) per Gray Eq. A.19 → MAP 0.60→0.73, bias −17.8%→0.0%. Baseline infrastructure, catalog-only diagnostic, P_det grid validation, Fisher condition-number gate. Full details: `.planning/milestones/v2.1-biasres-ROADMAP.md`
- **v1.5 Galaxy Catalog Completeness Correction** — shipped 2026-04-04 (Phases 24-25, GPD-tracked). GLADE+ completeness function f(z,h) and Gray et al. (2020) Eq. 9 completeness-corrected likelihood. Full artifacts: `.gpd/phases/24-completeness-estimation/`, `.gpd/phases/25-likelihood-correction/`
- **v1.4 Posterior Numerical Stability** — shipped 2026-04-02 (Phases 21-23). Log-space accumulation, physics-motivated likelihood floor, underflow detection. Deployed to bwUniCluster at `5793f70`. Full details: `.planning/milestones/v1.4-ROADMAP.md`
- **v1.3 Visualization Overhaul** — shipped 2026-04-02 (Phases 14-19). Full publication-quality visualization stack: centralized style, Fisher data layer, enhanced existing plots, sky maps + corner plots + convergence diagnostics, 15-figure batch pipeline. Full details: `.planning/milestones/v1.3-ROADMAP.md`
- **v1.2 Production Campaign & Physics Corrections** — shipped 2026-04-01 (Phases 9-13, 11.1). Confusion noise + five-point stencil physics fixes, simulation-based P_det, 1000+ detection production campaign, H0 sweep baselines. Full details: `.planning/milestones/v1.2-ROADMAP.md`
- **v1.1 Clean Simulation Campaign** — shipped 2026-03-29 (Phases 6-8). Data cleanup, cluster access, end-to-end validation. Full details: `.planning/milestones/v1.1-ROADMAP.md`
- **v1.0 EMRI HPC Integration** — shipped 2026-03-27 (Phases 1-5). CPU-safe imports, batch scripts, cluster env, SLURM jobs, documentation. Full details: `.planning/milestones/v1.0-ROADMAP.md`

## Current State (v2.2 active, Phases 35–39 shipped, Phase 40 next)

All 7 prior milestones shipped (v1.0–v1.5, v2.1). v2.1 fixed the completion-term
bias to MAP=0.73 at bias=0.0% with N=59 events. The 2026-04-21 pre-batch audit
(across statistical, physical, and HPC axes) surfaced **10 new findings**, two
critical: (1) the GLADE catalog is matched to waveform ecliptic `qS`/`phiS` with
no equatorial→ecliptic rotation, and (2) the BallTree embedding mixes polar and
latitude conventions, collapsing all ecliptic-equator galaxies to a single axis
and artificially spreading near-polar galaxies.

v2.2 Pipeline Correctness addresses all 10 findings before the next cluster
batch. **Phases 35–39 complete (2026-04-21..2026-04-23):** coordinate frame
fixed (Phase 36), parameter estimation correctness restored (Phase 37 — h-threading
+ per-param epsilon), L_cat aligned with Gray Eq. 24-25 + symmetric P_det
zero-fill (Phase 38), HPC + visualization safe wins shipped (Phase 39 —
parameter_estimation.py CPU-importable, batched CRB writes, pressure-gated FFT
cache, dead code removed, flip_hx verified KEEP, LaTeX figures, HDI bands).
540 tests GREEN. **Phase 40 Verification Gate next** — re-evaluates existing
CRBs under all v2.2 fixes; abort gate triggers if MAP at h=0.73 shifts >5% from
v2.1 baseline. v2.0 Paper remains paused pending Phase 40 outcome. Plan artifact:
`~/.claude/plans/i-want-a-last-elegant-feather.md`.

**Remaining physics bugs** (tracked in CLAUDE.md, explicitly deferred past v2.2):
wCDM params silently ignored, Pipeline-A hardcoded 10% σ(d_L), WMAP-era
cosmology, galaxy redshift uncertainty (1+z)³ scaling.

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
- ✓ Baseline/comparison infrastructure (`BaselineSnapshot`, `--save_baseline`, `--compare_baseline`) — v2.1 Phase 30
- ✓ Catalog-only diagnostic (`--catalog_only` flag, per-event CSV of L_cat, L_comp, f_i) — v2.1 Phase 31
- ✓ Completion term: full-volume D(h) = ∫P_det·dVc/dz dz per Gray Eq. A.19, MAP 0.60→0.73, bias 0.0% — v2.1 Phase 32 (GPD)
- ✓ P_det grid resolution validated (60-bin configurable, full 38-point cluster sweep, zero delta vs 30-bin) — v2.1 Phase 33
- ✓ Fisher matrix quality gates (`allow_singular=True` removed, condition-number gate, `fisher_quality.csv`, `plot_fisher_diagnostics`) — v2.1 Phase 34

## Current Milestone: v2.2 Pipeline Correctness

**Goal:** Fix all 10 findings from the 2026-04-21 pre-batch audit — two critical coordinate-frame bugs (no equatorial→ecliptic rotation + singular BallTree embedding at ecliptic equator), plus L_cat drift from Gray Eq. 24-25, P_det extrapolation asymmetry, h-hardcode in Fisher CRBs, uniform derivative_epsilon, and latent correctness hygiene — before investing new cluster compute on the next simulation batch and extended P_det injection run.

**Target features:**
- Failing-test-first characterization of the coordinate bugs + per-event equator-fraction baseline
- Equatorial→ecliptic rotation via `astropy.coordinates.SkyCoord` + correct polar Cartesian embedding + eigenvalue-based sky search radius
- Thread `h` through `ParameterSpace.set_host_galaxy_parameters`
- Per-parameter `derivative_epsilon` (relative for M/mu/d_L, absolute for angles)
- L_cat form: analytical proof of Gray Eq. 24-25 equivalence OR canonical replacement (`/physics-change` gated)
- P_det extrapolation alignment between numerator and denominator + per-event quadrature-weight-outside-grid diagnostic
- Correctness hygiene: swapped `Omega_m` limits, unified SNR threshold, `SPEED_OF_LIGHT_KM_S = C / 1000`
- HPC safe wins: `_crb_flush_interval` 1→25, remove per-iteration FFT cache clear, `_get_xp` shim in `parameter_estimation.py`, dead code removal, `flip_hx=True` verification
- Visualization safe wins: `apply_style(use_latex=True)` in production figures, bootstrap HDI bands on static H₀ convergence plot
- Verification gate: re-evaluate existing CRBs under fixed frame; abort new compute if MAP at h=0.73 shifts >5%
- Staged cluster campaign: Stage 1 densifies M×z×d_L injection grid (conditional on diagnostics); Stage 2 adds sky-dependent P_det (conditional on Stage 1 anisotropy check)

**Context:** Current self-consistency (sim uses same buggy frame as eval) hid the bugs. Any real dark-siren science requires them fixed. User decisions: fix + re-evaluate (no re-simulation); prove L_cat equivalence (or fix); safe HPC/viz wins only (no CUDA-stream refactor, no full viz overhaul); staged injection campaign.

**Plan artifact:** `~/.claude/plans/i-want-a-last-elegant-feather.md`

**Audit artifacts:** `.claude/projects/…/memory/project_coordinate_bugs.md`, `project_audit_2026_04_21.md`

### Active

- [ ] Coordinate bug characterization (failing tests + equator-fraction baseline)
- [ ] Equatorial→ecliptic rotation + correct polar Cartesian embedding + sky ellipse search
- [ ] Thread `h` through `set_host_galaxy_parameters`; per-parameter `derivative_epsilon`
- [ ] L_cat equivalence proof (or canonical Gray-form fix)
- [ ] P_det extrapolation alignment; per-event diagnostic
- [ ] Correctness hygiene (Omega_m limits, SNR threshold, `C/1000`)
- [ ] HPC safe wins (`_crb_flush_interval`, FFT cache, `_get_xp` shim, `flip_hx` verify, dead code)
- [ ] Visualization safe wins (`use_latex=True`, HDI bands)
- [ ] Verification gate (re-evaluate existing CRBs; abort if MAP shifts >5%)
- [ ] Stage 1 injection: densified M×z×d_L grid (conditional on diagnostics)
- [ ] Stage 2 injection: sky-dependent P_det (conditional on Stage 1)

### Paused: v2.0 Paper (GPD-tracked)

**Status:** Phase 26 (Paper Draft) complete. Phase 27 (Production Run & Figures) paused — posterior bias must be resolved first.
**GPD details:** `.gpd/ROADMAP.md`, `.gpd/STATE.md`

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
*Last updated: 2026-04-23 — v2.2 Phases 35–39 complete (coord frame, PE correctness, statistical correctness, HPC/viz safe wins); Phase 40 Verification Gate next*
