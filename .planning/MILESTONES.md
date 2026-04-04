# Milestones

## v1.4 Posterior Numerical Stability (Shipped: 2026-04-02)

**Phases completed:** 3 phases (21–23), 5 plans, 299 tests passing

**Key accomplishments:**

- Log-space posterior accumulation (replace np.prod with log-sum-exp) with 4 zero-handling strategies: naive, exclude, per-event-floor, physics-floor (Phase 21)
- Post-processing combination script with CLI wiring (`--combine --strategy`), diagnostic reports identifying zero-event root causes, comparison tables with MAP estimates (Phase 21)
- Physics-motivated likelihood floor in `single_host_likelihood` using per-event min(nonzero); removed dead `check_overflow` code (Phase 22)
- Deployed to bwUniCluster at commit `5793f70` before pending evaluate jobs; validated physics-floor MAP=0.66 matches exclude MAP=0.66 (|Δ|=0.00 < 0.05 — PASS) (Phase 23)

---

## v1.3 Visualization Overhaul (Shipped: 2026-04-02)

**Phases completed:** 6 phases (14–19), 11 plans, 342 tests passing

**Key accomplishments:**

- 23 smoke tests covering all plot factory functions + 18-key rcParams regression test pinning emri_thesis.mplstyle against unintentional drift (Phase 14)
- Centralized style infrastructure: _colors.py semantic palette, _labels.py with 21 LaTeX constants, figure presets (3.375in/7.0in REVTeX widths), LaTeX/mathtext toggle (Phase 15)
- CRB data layer (_data.py) reconstructing 14×14 covariance matrices + Fisher plot factories: error ellipses, characteristic strain h_c(f) overlay on LISA PSD, parameter uncertainty violins/bars (Phase 16)
- Enhanced all existing plot modules: H0 posterior with 68%/95% credible intervals + Planck/SH0ES bands, SNR CDF overlay, detection yield with efficiency curve, P_det heatmaps with contours, injected-vs-recovered scatter with residuals (Phase 17)
- Four new plot factory functions: Mollweide sky localization map with localization ellipses, Fisher corner plots via `corner` library, H0 convergence two-panel (posterior narrowing + CI width vs N), detection efficiency with Wilson score CI (Phase 18)
- Campaign dashboard composite (2×2 mosaic: H0 posterior, SNR, detection yield, sky map) + manifest-driven generate_figures() producing 15 thesis PDFs with graceful degradation and 2 MB size checks (Phase 19)

---

## v1.2 Production Campaign & Physics Corrections (Shipped: 2026-04-01)

**Phases completed:** 6 phases (9–13, 11.1), 12 plans

**Key accomplishments:**

- Galactic confusion noise added to LISA PSD — galactic foreground component wired into `power_spectral_density()` (Phase 9)
- Fisher matrix derivatives upgraded from O(ε) forward-difference to O(ε⁴) five-point stencil with condition number logging (Phase 10)
- KDE detection probability replaced by simulation-based P_det with importance sampling (VRF 11.8–24.9×) and RegularGridInterpolator integration (Phase 11.1)
- Production CRB catalog: 1000+ detections from 100 tasks × 50 steps (seed 200) on bwUniCluster (Phase 12)
- H₀ posterior sweep over [0.6, 0.9] with baseline MAP values: 0.72 (with BH mass), 0.86 (without); zero-likelihood problem documented (Phase 13)

---

## v1.1 Clean Simulation Campaign (Shipped: 2026-03-29)

**Phases completed:** 3 phases, 4 plans, 5 tasks

**Key accomplishments:**

- Cleaned stale simulation artifacts from git tracking and verified `.gitignore` coverage
- Established SSH access to bwUniCluster 3.0 with ControlMaster 2FA session reuse and full environment preflight
- Fixed cluster integration issues: sbatch path resolution, `few` v2.0.0rc1 / `fastlisaresponse` v1.1.17 API updates, CUDA backend forcing
- Added simulation robustness: 30s waveform timeout, SIGTERM handler for buffer flush, ZeroDivisionError catches
- Completed smoke-test campaign: 3 tasks x 10 steps, 20 detections (18 passed filter), H0 posterior at h=0.73
- Validated end-to-end pipeline: all quantitative checks passed (SNR physical, seeds correct, files present)

---

## v1.0 EMRI HPC Integration (Shipped: 2026-03-27)

**Phases completed:** 5 phases, 9 plans, 18 tasks

**Key accomplishments:**

- CPU-safe MemoryManagement with guarded GPUtil import, use_gpu parameter, free_gpu_memory() method, and fixed circular import in main.py
- --use_gpu and --num_workers CLI flags added and threaded through data_simulation, snr_analysis, and evaluate call chains
- Argparse CLIs for merge and prepare scripts with --workdir/--delete-sources flags, zero interactive prompts, and emri-merge/emri-prepare console entry points
- SLURM env var capture in run_metadata.json with indexed filenames for array job traceability
- Three sbatch job scripts (simulate, merge, evaluate) for the simulate-merge-evaluate pipeline on bwUniCluster 3.0 with GPU array jobs and reproducible seeding
- Single-command SLURM pipeline submission with afterok dependency chaining and sacct-based failure recovery for array tasks
- Self-contained cluster/README.md with 5-command quickstart, ASCII pipeline diagram, worked example, troubleshooting, and script reference
- Cluster Deployment section in CLAUDE.md with CLI flags table and script inventory; Running on HPC section in README.md

---
