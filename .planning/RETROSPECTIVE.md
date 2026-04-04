# Retrospective

## Milestone: v1.0 — EMRI HPC Integration

**Shipped:** 2026-03-27
**Phases:** 5 | **Plans:** 9

### What Was Built
- CPU-safe MemoryManagement with guarded GPUtil import and --use_gpu/--num_workers CLI flags
- Non-interactive merge/prepare scripts with emri-merge/emri-prepare console entry points
- modules.sh + setup.sh for repeatable bwUniCluster 3.0 environment setup
- Three-stage SLURM pipeline (simulate GPU array → merge → evaluate) with afterok dependency chaining
- Single-command pipeline submission (submit_pipeline.sh) with sacct-based failure recovery
- Self-contained cluster/README.md with quickstart, pipeline diagram, troubleshooting
- CLAUDE.md cluster section and README.md HPC pointer

### What Worked
- Phase decomposition was clean: each phase unblocked the next naturally (imports → scripts → env → jobs → docs)
- Shell scripts were straightforward to write and test conceptually without cluster access
- Discussion phase (05-CONTEXT.md) for documentation produced clear decisions that prevented scope creep
- Parallel plan execution within waves saved time (Phases 1, 4, 5)

### What Was Inefficient
- Phase 1 verification initially found gaps (Phase 1 checkbox never marked in ROADMAP), required manual fixups
- Rate limit hits during Phase 5 execution required manual recovery and worktree merge
- Some ROADMAP.md progress table entries fell out of sync with actual completion state
- Apptainer container was planned but deferred — could have been scoped out earlier

### Patterns Established
- `cluster/` directory as single location for all cluster scripts
- `modules.sh` as shared environment loader sourced by all sbatch scripts
- Per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID for reproducibility
- Console entry points (emri-merge, emri-prepare) instead of `python -m scripts.*`
- Blockquote callouts (> **Warning:**, > **Tip:**) for documentation emphasis

### Key Lessons
- Documentation phases benefit from a discussion/context session before planning — the D-01 through D-14 decisions made the plans nearly mechanical to execute
- Cluster scripts can be written and committed without cluster access; actual validation requires the cluster
- SLURM afterok dependency chaining is simple and reliable for linear pipelines

### Cost Observations
- Model mix: ~80% opus, ~20% sonnet (verifier only)
- Sessions: ~6 across all phases
- Notable: Documentation phase was the most token-efficient — clear context decisions meant near-zero rework

## Milestone: v1.1 — Clean Simulation Campaign

**Shipped:** 2026-03-29
**Phases:** 3 | **Plans:** 4

### What Was Built
- Cleaned stale simulation artifacts from git tracking, verified .gitignore coverage
- SSH ControlMaster to bwUniCluster 3.0 with 2FA session reuse, full environment preflight
- Fixed cluster integration issues: sbatch path resolution, few/fastlisaresponse API updates, CUDA backend forcing
- Simulation robustness: 30s waveform timeout, SIGTERM handler, ZeroDivisionError catches
- Smoke-test campaign: 3 tasks × 10 steps, 20 detections, 18 passed filter, H0 posterior at h=0.73
- End-to-end pipeline validation: all quantitative checks passed

### What Worked
- Iterative debugging cycle on the cluster was effective — push fix, submit, check logs, repeat
- SSH ControlMaster with 8h persistence made Claude's cluster command execution seamless
- SIGTERM handler + immediate flush prevented data loss from SLURM timeouts
- `force_backend="cuda12x"` resolved the GPU auto-detection failures cleanly

### What Was Inefficient
- Multiple rounds of waveform API fixes (6+ commits) — could have been caught by reading few/fastlisaresponse changelogs first
- ROADMAP.md and REQUIREMENTS.md never updated during phase execution — all still showed "Not started" / "Pending" at milestone completion
- Original plan parameters (5 tasks, 50 steps, seed 42) all changed during execution — planning was speculative without cluster timing data
- d_L error threshold raised from 5% to 10% as a workaround instead of fixing the Fisher stencil

### Patterns Established
- `force_backend="cuda12x"` for all LISA GPU tools (few, fastlisaresponse, lisatools)
- SIGTERM handler pattern for SLURM jobs that buffer output
- `afterany` (not `afterok`) for merge step when simulate tasks may timeout
- 30s alarm-based timeout for waveform computation hot path

### Key Lessons
1. Plan parameters for cluster jobs are guesses until you have actual timing data — plan for iteration
2. Waveform library API changes are the #1 cluster integration risk — check changelogs before assuming existing code works
3. ROADMAP/REQUIREMENTS state tracking needs to happen during execution, not retroactively at milestone close

### Cost Observations
- Model mix: ~90% opus, ~10% sonnet
- Sessions: ~4 across all phases
- Notable: Phase 8 was the most expensive — iterative cluster debugging required many round-trips

## Milestone: v1.2 — Production Campaign & Physics Corrections

**Shipped:** 2026-04-01
**Phases:** 6 | **Plans:** 12

### What Was Built
- Galactic confusion noise added to LISA PSD (PHYS-02 resolved)
- Fisher matrix derivatives upgraded to O(ε⁴) five-point stencil (PHYS-01 resolved)
- KDE detection probability replaced by simulation-based P_det with importance sampling (VRF 11.8–24.9×)
- Production CRB catalog: 1000+ detections from 100 tasks × 50 steps (seed 200)
- H₀ posterior sweep over [0.6, 0.9]; baseline MAP values documented (0.72/0.86)

### What Worked
- Physics Change Protocol (`/physics-change`) ensured dimensional analysis and limiting-case verification before every formula change
- Simulation-based P_det with importance sampling dramatically improved variance reduction (11.8–24.9×)
- Decimal phase numbering (11.1) cleanly handled the inserted P_det rework without disrupting the roadmap
- Production campaign on bwUniCluster delivered 1000+ detections reliably using the established SLURM infrastructure

### What Was Inefficient
- Phase 11.1 (simulation-based P_det) expanded from 1 planned phase to 5 plans — original scope underestimated the complexity of injection campaigns + importance sampling + grid interpolation + cluster scripts
- Zero-likelihood problem discovered only at the H₀ sweep stage (Phase 13), too late to fix within v1.2
- Some progress table dates were left as "-" instead of actual completion dates

### Patterns Established
- Physics Change Protocol as mandatory gate for formula modifications
- Importance sampling for detection probability estimation
- RegularGridInterpolator for efficient P_det lookups
- `submit_injection.sh` pattern for injection campaign cluster jobs

### Key Lessons
1. Physics corrections should be validated at small scale before production runs — Phase 11 validation caught issues early
2. Scope estimation for physics work is harder than software work — Phase 11.1 tripled in plan count
3. Zero-likelihood events are a fundamental challenge when combining per-event posteriors — should have been anticipated

### Cost Observations
- Model mix: ~85% opus, ~15% sonnet
- Sessions: ~5 across all phases
- Notable: Phase 11.1 was the most expensive — 5 plans for what was originally scoped as a single-phase fix

## Milestone: v1.4 — Posterior Numerical Stability

**Shipped:** 2026-04-02
**Phases:** 3 | **Plans:** 5

### What Was Built
- Log-space posterior accumulation replacing np.prod (avoids underflow to zero)
- Physics-motivated likelihood floor in single_host_likelihood (per-event min(nonzero))
- Post-processing combination script with 4 strategies and CLI wiring (--combine --strategy)
- Diagnostic reports identifying zero-event root causes
- Deployed to bwUniCluster at 5793f70; validated against baselines (PASS)

### What Worked
- Analysis done interactively in conversation first, then formalized in Phase 21 — saved rework
- Clean three-phase structure (analyze → fix → deploy) was natural and efficient
- Fast-forward merge for cluster deployment preserved linear history
- Validation criterion (|Δ MAP| < 0.05) gave a clear PASS/FAIL signal

### What Was Inefficient
- v1.4 analysis and coding were partially done before the milestone was formally defined — Phase 21 was partly documenting existing work
- "With BH mass" validation deferred because the campaign data directory didn't exist yet
- check_overflow removal required reading through dead code paths to confirm it was truly dead

### Patterns Established
- Log-sum-exp trick for numerical stability in posterior combination
- Per-event-min as physics-motivated floor (not arbitrary constant)
- StrEnum for Python 3.13 ruff-compliant strategy enumerations
- NaN to distinguish missing events from zero-likelihood events

### Key Lessons
1. Interactive analysis before formal planning is efficient for diagnosis — formalize only the solution
2. Numerical stability should be designed in from the start, not patched after production data reveals problems
3. Validation against baselines with a quantitative acceptance criterion makes deploy decisions objective

### Cost Observations
- Model mix: ~90% opus, ~10% sonnet
- Sessions: ~3 across all phases
- Notable: Most efficient milestone — tight scope, clear problem, well-understood solution space

## Cross-Milestone Trends

| Metric | v1.0 | v1.1 | v1.2 | v1.3 | v1.4 |
|--------|------|------|------|------|------|
| Phases | 5 | 3 | 6 | 6 | 3 |
| Plans | 9 | 4 | 12 | 11 | 5 |
| Days | 2 | 3 | 3 | 2 | 1 |
| Rework cycles | 1 (verification gap) | 6+ (waveform API) | 1 (P_det scope expansion) | 0 | 0 |
| Focus | Infrastructure | Cluster validation | Physics + production | Visualization | Numerical stability |
