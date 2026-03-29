# Milestones

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
