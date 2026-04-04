---
phase: 4
slug: slurm-job-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-27
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (configured in pyproject.toml) |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest -m "not gpu and not slow" -x` |
| **Full suite command** | `uv run pytest -m "not gpu"` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -m "not gpu and not slow" -x`
- **After every plan wave:** Run `uv run pytest -m "not gpu"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | TRACE-01 | unit | `uv run pytest master_thesis_code_test/test_main_metadata.py -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | TRACE-01/D-09 | unit | `uv run pytest master_thesis_code_test/test_main_metadata.py -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | SLURM-01 | manual | `bash -n cluster/simulate.sbatch` | N/A | ⬜ pending |
| 04-02-02 | 02 | 1 | SLURM-02 | manual | `bash -n cluster/merge.sbatch` | N/A | ⬜ pending |
| 04-02-03 | 02 | 1 | SLURM-03 | manual | `bash -n cluster/evaluate.sbatch` | N/A | ⬜ pending |
| 04-03-01 | 03 | 2 | SLURM-04 | manual | `bash -n cluster/submit_pipeline.sh` | N/A | ⬜ pending |
| 04-03-02 | 03 | 2 | D-12/D-13 | manual | `bash -n cluster/resubmit_failed.sh` | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `master_thesis_code_test/test_main_metadata.py` — unit tests for `_write_run_metadata()` covering SLURM env vars (TRACE-01) and indexed filenames (D-09)

*Shell scripts (sbatch, submit_pipeline, resubmit_failed) are validated via `bash -n` syntax checks and manual cluster dry-runs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| simulate.sbatch submits GPU array job correctly | SLURM-01 | Requires SLURM scheduler on cluster | Submit with `--array=0-0` on `dev_gpu_h100` partition, verify output files |
| merge.sbatch runs emri-merge and emri-prepare | SLURM-02 | Requires merged CSV input from simulation | Run after a successful simulate job, verify merged CSV exists |
| evaluate.sbatch runs with correct --num_workers | SLURM-03 | Requires Cramér-Rao bounds input | Run after merge, verify evaluation output |
| submit_pipeline.sh chains 3 dependent jobs | SLURM-04 | Requires SLURM scheduler | Run with `--tasks 1 --steps 10 --seed 42` on dev partition, verify all 3 job IDs printed |
| Seed = base_seed + SLURM_ARRAY_TASK_ID | TRACE-02 | Seed arithmetic is in bash, verified by metadata output | Check `run_metadata_0.json` seed matches expected value |
| resubmit_failed.sh detects and resubmits failed tasks | D-12 | Requires failed SLURM jobs to query | Manually fail a task (e.g., cancel it), then run resubmit script |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
