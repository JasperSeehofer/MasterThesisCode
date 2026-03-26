#!/usr/bin/env bash
# cluster/setup.sh -- First-time cluster environment setup. Idempotent: safe to re-run.
# Loads modules, installs uv, allocates workspace, creates virtualenv with GPU deps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =========================================================================
# Step 1/4: Loading modules
# =========================================================================
echo "=== Step 1/4: Loading modules ==="
source "$SCRIPT_DIR/modules.sh"

# =========================================================================
# Step 2/4: Checking uv (D-03: check-and-skip)
# =========================================================================
echo ""
echo "=== Step 2/4: Checking uv ==="
if command -v uv &>/dev/null; then
    echo "  uv already installed: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "  Installed: $(uv --version)"
fi

# =========================================================================
# Step 3/4: Workspace (D-04: check-and-allocate)
# =========================================================================
echo ""
echo "=== Step 3/4: Workspace ==="
WORKSPACE="$(ws_find emri 2>/dev/null || true)"
if [[ -z "$WORKSPACE" ]]; then
    echo "  Allocating workspace 'emri' (60 days)..."
    ws_allocate emri 60
    WORKSPACE="$(ws_find emri)"
fi
export WORKSPACE
echo "  Workspace: $WORKSPACE"

# =========================================================================
# Step 4/4: Python environment (D-06: use cluster Python module)
# =========================================================================
echo ""
echo "=== Step 4/4: Python environment ==="
cd "$PROJECT_ROOT"
uv sync --extra gpu
echo "  Virtual environment: $PROJECT_ROOT/.venv"

# =========================================================================
# Verification
# =========================================================================
echo ""
echo "=== Verification ==="
uv run python -c "import master_thesis_code; print('  master_thesis_code importable')"
uv run python -c "import few; print('  few (fastemriwaveforms) importable')"
uv run python -c "import cupy; print(f'  cupy {cupy.__version__} with CUDA')"
echo ""
echo "Setup complete. Workspace: $WORKSPACE"
