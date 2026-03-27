# Phase 03 Plan 01 Summary: Cluster Environment Setup

## What was built

Two shell scripts for repeatable bwUniCluster 3.0 environment setup:

1. **`cluster/modules.sh`** — sourceable module loader
   - Loads 3 modules: `compiler/gnu/14.2`, `devel/cuda/12.8`, `devel/python/3.13.3-gnu-14.2`
   - GSL 2.6 is system-wide (no module needed)
   - Exports `$WORKSPACE`, `$PROJECT_ROOT`, `$VENV_PATH`
   - No `set -euo pipefail` (sourced into login shell)

2. **`cluster/setup.sh`** — idempotent first-time setup
   - Step 1: Sources `modules.sh`
   - Step 2: Installs `uv` if missing
   - Step 3: Allocates workspace `emri` (60 days) if missing
   - Step 4: Runs `uv sync --extra gpu`
   - Verification: imports `master_thesis_code`, `few`, `cupy`

## Verified on cluster (2026-03-27)

- All 3 modules load without errors
- `python3 --version` → 3.13.3
- `uv` 0.11.2 installed to `~/.local/bin`
- Workspace: `/pfs/work9/workspace/scratch/st_ac147838-emri`
- 59 packages installed including `cupy 13.6.0`, `fastemriwaveforms 2.0.0rc1`
- All 3 import smoke tests pass

## Deviations from plan

- **GSL module removed**: `numlib/gsl` does not exist on bwUniCluster 3.0; GSL 2.6 is installed system-wide
- **`set -euo pipefail` removed from `modules.sh`**: kills login shell when sourced and a command fails

## Files modified

- `cluster/modules.sh` (created, then patched)
- `cluster/setup.sh` (created)
- `.gitignore` (updated — unrelated repo cleanup)

## Duration

~20 min (including interactive cluster verification)
