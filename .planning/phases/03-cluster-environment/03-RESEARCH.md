# Phase 3: Cluster Environment - Research

**Researched:** 2026-03-26
**Domain:** HPC environment setup (bwUniCluster 3.0, module system, uv, workspace management)
**Confidence:** MEDIUM

## Summary

Phase 3 creates two shell scripts (`cluster/modules.sh` and `cluster/setup.sh`) and workspace integration for bwUniCluster 3.0. The cluster uses the Lmod module system for software, Lustre-based workspaces with 60-day expiry (extendable to 240 days), and provides NVIDIA H100/A100 GPU nodes. All key Python packages (`fastemriwaveforms-cuda12x`, `cupy-cuda12x`) ship pre-built manylinux wheels, so GSL is unlikely to be needed at install time -- but the module should still be loaded as a safety net for any source-build fallback.

The cluster's login nodes have internet access for `uv sync` and `curl` (uv installer). Compute nodes share the same HOME and workspace filesystems. The `uv` installer places the binary in `~/.local/bin/` without root access, which is the standard pattern for HPC user-space tools.

**Primary recommendation:** Write defensive, idempotent scripts with early failure and clear error messages. Hardcode best-guess module names from wiki documentation, with inline comments explaining how to verify and update them via `module spider`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Hardcode module load sequence: `compiler/gnu/14.2`, `devel/cuda/12.8`, `numlib/gsl`, `devel/python/3.13.3-gnu-14.2`. Fail early with clear error if any module not found.
- **D-02:** `modules.sh` should fail early with a clear error message if any module is not found.
- **D-03:** Check-and-skip pattern for uv: `command -v uv` first, only install if missing. Use official Astral installer.
- **D-04:** Check-and-allocate pattern: `ws_find emri` first; if no workspace, run `ws_allocate emri 60`. Idempotent.
- **D-05:** `modules.sh` resolves workspace path via `ws_find emri` and exports `$WORKSPACE`. All downstream scripts use `$WORKSPACE`.
- **D-06:** Use cluster Python module (`devel/python/3.13.3-gnu-14.2`), not uv-managed Python.

### Claude's Discretion
- Error handling and messaging in `setup.sh` (what to do if `uv sync` fails, etc.)
- Whether `modules.sh` also exports convenience variables like `$PROJECT_ROOT` or `$VENV_PATH`
- Verification steps at end of `setup.sh` (e.g., `python -c "import master_thesis_code"` smoke test)
- Script structure (functions vs sequential, logging style)

### Deferred Ideas (OUT OF SCOPE)
- Apptainer container (v2 requirement CONT-01, CONT-02)
- Workspace expiration warnings (Phase 5 documentation)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-01 | `cluster/modules.sh` defines all required environment modules (CUDA, Python, GSL, compiler) and is sourced by every job script | Module names confirmed from bwHPC wiki; Lmod system supports `module load` with fail-on-missing; script pattern documented below |
| ENV-02 | `cluster/setup.sh` automates first-time cluster setup: uv installation, workspace allocation, `uv sync --extra gpu`, and module verification | uv installer works in user-space (`~/.local/bin/`); workspace tools (`ws_allocate`, `ws_find`) documented; pre-built wheels eliminate GSL build requirement |
| ENV-03 | All simulation output uses bwHPC workspace paths (resolved via `ws_find`), not `$HOME` | Workspace paths on Lustre (`/pfs/work9`); `ws_find` returns full path; export as `$WORKSPACE` env var |
</phase_requirements>

## Standard Stack

This phase is shell scripts only -- no Python libraries to install. The "stack" is the cluster tooling.

### Core Tools
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| Lmod | (cluster-provided) | Environment module system | Standard on all bwHPC clusters |
| uv | latest (installer) | Python package/project manager | Project standard per CLAUDE.md |
| ws_allocate / ws_find | (cluster-provided) | bwHPC workspace management | Only workspace mechanism on bwUniCluster |
| bash | 4.x+ | Shell scripting | Login shell on cluster |

### Cluster Modules (D-01)
| Module | Purpose | Confidence |
|--------|---------|------------|
| `compiler/gnu/14.2` | GCC compiler toolchain | HIGH -- confirmed on bwHPC wiki |
| `devel/cuda/12.8` | CUDA toolkit for cupy/few | HIGH -- confirmed on bwHPC wiki |
| `numlib/gsl` | GSL (safety net for source builds) | MEDIUM -- category exists, default version loads |
| `devel/python/3.13.3-gnu-14.2` | Python 3.13 interpreter | MEDIUM -- version naming derived from wiki pattern, needs on-cluster verification |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Cluster Python module | uv-managed Python download | Extra download, not cluster-optimized, may conflict with module system |
| `ws_allocate` workspace | `$HOME` subdirectory | HOME quota is 500 GiB but clutters persistent storage; workspace is on Lustre (fast parallel I/O) |
| uv | pip + venv | Project uses uv exclusively; lockfile is `uv.lock` not `requirements.txt` |

## Architecture Patterns

### Recommended Directory Structure
```
cluster/
    modules.sh      # ENV-01: source this to load modules + export $WORKSPACE
    setup.sh        # ENV-02: first-time setup (uv install, workspace, venv)
```

### Pattern 1: Sourceable Module Script (`modules.sh`)
**What:** A script designed to be `source`d (not executed) that loads modules and exports environment variables. Used by `setup.sh` and all Phase 4 SLURM job scripts.
**When to use:** Every time cluster environment is needed.
**Example:**
```bash
#!/usr/bin/env bash
# cluster/modules.sh — Source this file; do not execute directly.
# Loads required environment modules and exports $WORKSPACE.
#
# Module names verified against bwUniCluster 3.0 wiki (2026-03).
# If a module fails to load, verify with: module spider <category>/<name>

set -euo pipefail

# --- Module loading with early failure ---
_load_or_die() {
    local mod="$1"
    if ! module load "$mod" 2>/dev/null; then
        echo "ERROR: Failed to load module '$mod'" >&2
        echo "       Run 'module spider ${mod%%/*}/' to see available versions." >&2
        return 1
    fi
    echo "  Loaded: $mod"
}

echo "Loading bwUniCluster modules..."
_load_or_die compiler/gnu/14.2
_load_or_die devel/cuda/12.8
_load_or_die numlib/gsl
_load_or_die devel/python/3.13.3-gnu-14.2

# --- Workspace resolution (D-05) ---
WORKSPACE="$(ws_find emri 2>/dev/null || true)"
if [[ -z "$WORKSPACE" ]]; then
    echo "WARNING: Workspace 'emri' not found. Run cluster/setup.sh first." >&2
else
    export WORKSPACE
    echo "  Workspace: $WORKSPACE"
fi

# --- Convenience variables (Claude's discretion) ---
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV_PATH="$PROJECT_ROOT/.venv"

# Add uv to PATH if installed in user-local
export PATH="$HOME/.local/bin:$PATH"
```

### Pattern 2: Idempotent Setup Script (`setup.sh`)
**What:** A script that can be re-run safely. Uses check-and-skip for uv, check-and-allocate for workspace.
**When to use:** First-time cluster setup and after any environment issue.
**Example:**
```bash
#!/usr/bin/env bash
# cluster/setup.sh — First-time cluster environment setup.
# Idempotent: safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Step 1: Load modules
echo "=== Step 1/4: Loading modules ==="
source "$SCRIPT_DIR/modules.sh"

# Step 2: Install uv (D-03: check-and-skip)
echo "=== Step 2/4: Checking uv ==="
if command -v uv &>/dev/null; then
    echo "  uv already installed: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "  Installed: $(uv --version)"
fi

# Step 3: Workspace (D-04: check-and-allocate)
echo "=== Step 3/4: Workspace ==="
WORKSPACE="$(ws_find emri 2>/dev/null || true)"
if [[ -z "$WORKSPACE" ]]; then
    echo "  Allocating workspace 'emri' (60 days)..."
    ws_allocate emri 60
    WORKSPACE="$(ws_find emri)"
fi
export WORKSPACE
echo "  Workspace: $WORKSPACE"

# Step 4: Create virtualenv and install dependencies
echo "=== Step 4/4: Python environment ==="
cd "$PROJECT_ROOT"
uv sync --extra gpu
echo "  Virtual environment: $PROJECT_ROOT/.venv"

# Smoke test
echo "=== Verification ==="
uv run python -c "import master_thesis_code; print('  master_thesis_code importable')"
uv run python -c "import few; print('  few (fastemriwaveforms) importable')"
uv run python -c "import cupy; print(f'  cupy {cupy.__version__} with CUDA')"
echo "Setup complete."
```

### Anti-Patterns to Avoid
- **Hardcoding workspace paths:** Never use `/pfs/work9/username-emri-...` directly. Always resolve via `ws_find emri`. The path includes a timestamp suffix that changes on re-allocation.
- **Running `uv sync` on compute nodes:** Compute nodes may have restricted network access. All package installation must happen on login nodes.
- **Using `set -e` in a sourced file without guard:** When `modules.sh` is sourced, `set -e` propagates to the caller's shell. Use `set -euo pipefail` but document that the script must be sourced with awareness.
- **Forgetting to `export PATH`:** The uv binary lives in `~/.local/bin/`, which may not be in the default PATH on all shells. Always prepend it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Module loading | Custom `LD_LIBRARY_PATH` manipulation | `module load` (Lmod) | Lmod handles dependencies, conflicts, PATH/LD_LIBRARY_PATH atomically |
| Workspace path resolution | Hardcoded `/pfs/work9/...` paths | `ws_find emri` | Path includes username + timestamp suffix; changes on re-allocation |
| Python version management | Download Python tarball | `devel/python/3.13.3-gnu-14.2` module | Cluster-compiled, tested, maintained by admins |
| Package installation | `pip install` from requirements.txt | `uv sync --extra gpu` | Project uses uv lockfile exclusively |

## Common Pitfalls

### Pitfall 1: Module Version Mismatch
**What goes wrong:** Module names change between cluster updates. `devel/python/3.13.3-gnu-14.2` might become `devel/python/3.13.5-gnu-14.2` after a cluster maintenance window.
**Why it happens:** bwUniCluster updates software periodically; wiki may lag behind.
**How to avoid:** `_load_or_die` function with `module spider` hint in error message. Add a comment in `modules.sh` with verification instructions.
**Warning signs:** `setup.sh` fails at module loading step.

### Pitfall 2: Workspace Expired Between Runs
**What goes wrong:** Workspace was allocated 60+ days ago, `ws_find emri` returns empty, all output paths break.
**Why it happens:** bwHPC workspaces have a hard expiry. Maximum lifetime is 240 days with 3 extensions.
**How to avoid:** `modules.sh` warns if workspace not found. Phase 5 docs should cover extension workflow.
**Warning signs:** `$WORKSPACE` is empty or points to non-existent path.

### Pitfall 3: `uv sync` Fails Due to Missing System Library
**What goes wrong:** A dependency needs a C library (GSL, LAPACK) not available in the module environment.
**Why it happens:** Pre-built wheels cover most cases, but a version mismatch or new dependency could trigger a source build.
**How to avoid:** Load `numlib/gsl` and `compiler/gnu/14.2` before `uv sync`. The compiler module provides standard build tools.
**Warning signs:** Build errors mentioning `gsl-config`, `gcc`, or missing headers.

### Pitfall 4: PATH Not Updated After uv Install
**What goes wrong:** `uv` is installed to `~/.local/bin/` but the current shell doesn't have it in PATH.
**Why it happens:** The installer modifies `~/.bashrc` but the current session doesn't re-source it.
**How to avoid:** Explicitly `export PATH="$HOME/.local/bin:$PATH"` in `setup.sh` after installing uv.
**Warning signs:** `command not found: uv` immediately after installation.

### Pitfall 5: `set -e` in Sourced Script Kills Caller Shell
**What goes wrong:** If `modules.sh` uses `set -e` and a command fails, it can terminate an interactive shell session.
**Why it happens:** `source` runs in the current shell context; `set -e` persists.
**How to avoid:** Use a subshell guard or save/restore `set` options. Alternatively, accept this behavior and document it. For SLURM job scripts (which are non-interactive), `set -e` is desirable. For interactive use, wrap in a function that traps errors.
**Warning signs:** Shell exits unexpectedly when sourcing `modules.sh`.

### Pitfall 6: `uv sync` Uses Wrong Python
**What goes wrong:** `uv` finds a system Python or a different module Python instead of the one loaded via `module load`.
**Why it happens:** `uv` has its own Python discovery logic. If `.python-version` says `3.13` and multiple 3.13 interpreters exist, it may pick the wrong one.
**How to avoid:** Ensure module Python is loaded BEFORE running `uv sync`. The `--python` flag can force a specific interpreter path if needed: `uv sync --extra gpu --python $(which python3)`.
**Warning signs:** Wrong Python version in `.venv/pyvenv.cfg`.

## Code Examples

### Checking Module Availability
```bash
# Verify a module exists before hardcoding it
module spider compiler/gnu/
# Shows all versions: compiler/gnu/11.4, compiler/gnu/14.2

module spider devel/cuda/
# Shows CUDA versions: devel/cuda/12.8

module spider numlib/gsl
# Shows GSL availability and any dependency requirements
```

### Workspace Commands
```bash
# Allocate (first time)
ws_allocate emri 60
# Output: /pfs/work9/ws/username-emri-0

# Find (subsequent times)
ws_find emri
# Output: /pfs/work9/ws/username-emri-0

# Extend before expiry
ws_extend emri 60

# Check remaining time
ws_list
```

### uv on HPC
```bash
# Install uv (no root needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Installs to ~/.local/bin/uv

# Create venv from module Python (D-06)
module load devel/python/3.13.3-gnu-14.2
uv sync --extra gpu
# Creates .venv/ using the module Python, installs all gpu dependencies from uv.lock
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pip + virtualenv on HPC | uv with lockfile | 2024-2025 | Deterministic, fast installs from lockfile; no `requirements.txt` drift |
| Conda on HPC | uv (project choice) | N/A | Avoids conda/module conflicts; uses system CUDA via modules |
| Manual module loading | Shared `modules.sh` script | This phase | Single source of truth for environment; all job scripts source it |

## Open Questions

1. **GSL module version string**
   - What we know: `numlib/gsl` category exists on bwUniCluster 3.0. The default (no version suffix) should work.
   - What's unclear: Exact version string if default loading fails (e.g., `numlib/gsl/2.7` or similar).
   - Recommendation: Use `numlib/gsl` (default). If it fails, `module spider numlib/gsl` will reveal available versions. Document this in `modules.sh` comment.

2. **Python module exact version**
   - What we know: `devel/python/3.13.3-gnu-14.2` is derived from bwHPC wiki pattern showing Python 3.13.x versions.
   - What's unclear: Whether 3.13.3 is current or has been superseded by 3.13.5 after a cluster update.
   - Recommendation: Hardcode `devel/python/3.13.3-gnu-14.2` per D-01. Error message from `_load_or_die` will guide the user to `module spider devel/python/` to find the correct version.

3. **Compute node network access**
   - What we know: Login nodes have internet access. Compute nodes share HOME and workspace filesystems.
   - What's unclear: Whether compute nodes can reach PyPI for downloads.
   - Recommendation: All `uv sync` / installation happens on login nodes only. Job scripts never install packages. This is safe regardless of network policy.

4. **`set -e` behavior when sourced interactively**
   - What we know: `set -e` in a sourced script propagates to the caller's shell.
   - What's unclear: Whether this causes usability issues for interactive sessions.
   - Recommendation: Use `set -e` (benefits for SLURM jobs outweigh interactive inconvenience). Document that interactive users should use `source cluster/modules.sh || true` or run in a subshell.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | bash + manual verification |
| Config file | none (shell scripts, not Python tests) |
| Quick run command | `bash -n cluster/modules.sh && bash -n cluster/setup.sh` (syntax check) |
| Full suite command | `source cluster/modules.sh` on bwUniCluster login node |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | `source cluster/modules.sh` loads modules without errors | manual (requires cluster) | `bash -n cluster/modules.sh` (syntax only) | Wave 0 |
| ENV-02 | `cluster/setup.sh` produces working `.venv` with GPU deps | manual (requires cluster) | `bash -n cluster/setup.sh` (syntax only) | Wave 0 |
| ENV-03 | Output uses workspace paths, not `$HOME` | manual (verify `$WORKSPACE` exported) | `source cluster/modules.sh && test -n "$WORKSPACE"` | Wave 0 |

### Sampling Rate
- **Per task commit:** `bash -n cluster/*.sh` (syntax validation, runs on dev machine)
- **Per wave merge:** N/A (full validation requires cluster)
- **Phase gate:** Manual execution on bwUniCluster login node

### Wave 0 Gaps
- [ ] `bash -n` syntax checking only validates shell syntax, not module availability
- [ ] Full ENV-01/ENV-02/ENV-03 validation requires interactive bwUniCluster access (documented blocker in STATE.md)

## Project Constraints (from CLAUDE.md)

- **uv is the only package manager** -- never use pip/conda directly
- **`uv sync --extra gpu`** is the documented cluster install command
- **GSL is listed as a build-time prerequisite** in CLAUDE.md -- load the module even though pre-built wheels likely eliminate the need
- **Pre-commit hooks** (ruff, mypy) run on commit -- shell scripts in `cluster/` are not subject to Python linting but should follow `.editorconfig` (4-space indent, LF, trim trailing whitespace)
- **`.python-version` pins Python 3.13** -- the cluster module must provide 3.13.x
- **Entry points `emri-merge` and `emri-prepare`** are registered in `pyproject.toml` and accept `--workdir` -- downstream SLURM scripts (Phase 4) will use these with `$WORKSPACE`
- **GSD workflow enforcement** -- changes should go through GSD workflow

## Sources

### Primary (HIGH confidence)
- [bwUniCluster 3.0 Software Modules wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0/Software_Modules) -- module naming conventions, `compiler/gnu/14.2`, `devel/cuda/12.8` confirmed
- [bwHPC Workspace wiki](https://wiki.bwhpc.de/e/Workspace) -- `ws_allocate`, `ws_find` syntax, 60-day lifetime, 3 extensions (240 days max)
- [bwUniCluster 3.0 Filesystem Details](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture/Filesystem_Details) -- workspace on `/pfs/work9` (Lustre), HOME quota 500 GiB
- [bwUniCluster 3.0 Batch Queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues) -- GPU partitions: `gpu_h100`, `gpu_a100_il`, `dev_gpu_h100`
- [uv Installation docs](https://docs.astral.sh/uv/getting-started/installation/) -- `curl -LsSf https://astral.sh/uv/install.sh | sh`, installs to `~/.local/bin/`
- `uv.lock` in project -- `cupy-cuda12x 13.6.0` and `fastemriwaveforms-cuda12x 2.0.0rc1` both have pre-built manylinux wheels

### Secondary (MEDIUM confidence)
- [bwUniCluster 3.0 Hardware and Architecture](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture) -- GPU node types (H100, A100), 4 GPUs per node
- `devel/python/3.13.3-gnu-14.2` module name -- derived from wiki pattern, needs on-cluster verification

### Tertiary (LOW confidence)
- Compute node network access policy -- not explicitly documented; assumed restricted based on HPC convention
- GSL module exact version -- `numlib/gsl` loads default; exact version unknown

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - module system and workspace tools are well-documented on bwHPC wiki
- Architecture: HIGH - script structure is straightforward shell scripting; patterns are well-established for HPC
- Pitfalls: MEDIUM - module version strings and network policy need on-cluster verification

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable HPC infrastructure; module versions may change at cluster maintenance windows)
