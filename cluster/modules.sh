#!/usr/bin/env bash
# cluster/modules.sh -- Source this file; do not execute directly.
# Loads required environment modules for bwUniCluster 3.0 and exports
# $WORKSPACE, $PROJECT_ROOT, $VENV_PATH.

# NOTE: Do NOT use "set -euo pipefail" here — this file is sourced into the
# user's login shell, and set -e would kill the session on any failure.

# ---------------------------------------------------------------------------
# Module loading with early failure
# ---------------------------------------------------------------------------

_load_or_die() {
    local mod="$1"
    if ! module load "$mod" 2>/dev/null; then
        echo "ERROR: Failed to load module '$mod'" >&2
        echo "       Run 'module spider ${mod%%/*}/' to see available versions." >&2
        return 1
    fi
    echo "  Loaded: $mod"
}

# Module names verified against bwUniCluster 3.0 wiki (2026-03).
echo "Loading bwUniCluster modules..."
_load_or_die compiler/gnu/14.2
_load_or_die devel/cuda/12.8
_load_or_die devel/python/3.13.3-gnu-14.2
# GSL 2.6 is installed system-wide — no module needed (verified 2026-03-27).
# If a module fails to load, verify with: module spider <category>/

# ---------------------------------------------------------------------------
# Workspace resolution (D-05)
# ---------------------------------------------------------------------------

WORKSPACE="$(ws_find emri 2>/dev/null || true)"
if [[ -z "$WORKSPACE" ]]; then
    echo "WARNING: Workspace 'emri' not found. Run cluster/setup.sh first." >&2
else
    export WORKSPACE
    echo "  Workspace: $WORKSPACE"
fi

# ---------------------------------------------------------------------------
# Convenience variables
# ---------------------------------------------------------------------------

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV_PATH="$PROJECT_ROOT/.venv"

# Ensure uv is findable (installed to ~/.local/bin by the Astral installer)
export PATH="$HOME/.local/bin:$PATH"
