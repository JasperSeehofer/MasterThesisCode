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

## Cross-Milestone Trends

| Metric | v1.0 |
|--------|------|
| Phases | 5 |
| Plans | 9 |
| Days | 2 |
| Rework cycles | 1 (Phase 1 verification gap) |
