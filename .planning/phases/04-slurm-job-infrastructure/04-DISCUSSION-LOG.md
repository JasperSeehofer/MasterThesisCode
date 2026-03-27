# Phase 4: SLURM Job Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-27
**Phase:** 04-slurm-job-infrastructure
**Areas discussed:** SLURM resource parameters, Pipeline invocation design, Output directory structure, Failure recovery

---

## SLURM Resource Parameters

| Option | Description | Selected |
|--------|-------------|----------|
| gpu_h100 partition | H100 nodes, up to 3 days, 12 nodes, 193GB/GPU | ✓ |
| gpu_a100_il partition | A100 nodes, up to 2 days, 10 nodes | |
| gpu_a100_short partition | A100, 30 min max — too tight | |

**User's choice:** `gpu_h100` for simulate, `cpu` for merge and evaluate
**Notes:** User aims for 1-2 hour tasks to minimize queue wait times. Simulation index approach was designed specifically for this — many short parallel tasks rather than one long run. Time limits are initial estimates; profiling planned. `sinfo` was blocked on the cluster; partition info obtained from `scontrol show partition` output and bwHPC wiki.

---

## Pipeline Invocation Design

| Option | Description | Selected |
|--------|-------------|----------|
| Positional args | `./submit_pipeline.sh 100 50 42` | |
| Flags (explicit) | `--tasks 100 --steps 50 --seed 42` | ✓ |
| Defaults with override | Sensible defaults, override individually | |

**User's choice:** Flags — explicit and self-documenting
**Notes:** No defaults — all three args required. User rejected time-limit override flags (`--time-simulate`, `--time-evaluate`) as unnecessary complexity. Edit sbatch files directly if limits need changing.

---

## Output Directory Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Option A: flat | Everything in $WORKSPACE root, metadata files would overwrite | |
| Option B: campaign directory | `$WORKSPACE/run_<date>_seed<N>/` with flat contents inside | ✓ |
| Option C: per-task subdirectories | `$WORKSPACE/tasks/0/`, `tasks/1/`, etc. | |

**User's choice:** Option B — campaign run directory
**Notes:** User initially asked for clarification on Option C ("what is task in this context"). After explanation, chose Option B. Metadata files indexed per task (`run_metadata_0.json`) to avoid overwrites within the flat directory. Multiple campaigns can coexist side by side in `$WORKSPACE`.

---

## Failure Recovery

| Option | Description | Selected |
|--------|-------------|----------|
| Manual sacct + sbatch | User runs sacct, manually resubmits with --array= | |
| Helper script | `resubmit_failed.sh` automates query + cleanup + resubmit | ✓ |

**User's choice:** Helper script with clean-before-resubmit
**Notes:** User specifically requested cleaning up output files from failed tasks before resubmission. Rationale: partial CSV writes from killed tasks could corrupt the merge step. The script deletes CSVs and metadata for failed indices, then resubmits only those. User considers this safety measure worthwhile, not overkill.

---

## Claude's Discretion

- sbatch script structure and SLURM output log naming
- sacct parsing details in resubmit_failed.sh
- How run directory name is passed between chained jobs
- CPU core allocation for evaluate job
- submit_pipeline.sh output format (summary table vs plain job IDs)

## Deferred Ideas

- **Profiling** — User wants profiling to tune time limits and understand bottlenecks. Future milestone.
- **Job monitoring helpers** — Tracked as MON-01 in REQUIREMENTS.md v2.
