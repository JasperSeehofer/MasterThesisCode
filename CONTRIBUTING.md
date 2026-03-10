# Contributing

## Development environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <repo>
cd MasterThesisCode
uv sync --extra cpu --extra dev
```

## Branching convention

- `main` — stable, tagged releases
- `claudes_sidequests` — active development branch
- Feature branches: `<topic>/<short-description>`

## Pre-commit hooks

Pre-commit hooks run ruff (lint + format) and mypy on every `git commit`. Install them once:

```bash
uv run pre-commit install
```

To run all hooks manually on all files:

```bash
uv run pre-commit run --all-files
```

## Running tests

```bash
# CPU-only (default for dev machines)
uv run pytest -m "not gpu and not slow"

# Including slow benchmarks
uv run pytest -m "not gpu"

# Single file
uv run pytest master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py
```

## Type checking

```bash
uv run mypy master_thesis_code/
```

## Physics change protocol

Any change touching a formula, physical constant, or waveform parameter is a **physics change**
and must follow this protocol before implementation:

1. State the **old formula** with file and line number
2. State the **new formula**
3. Provide a **reference** (DOI/arXiv + equation number) or step-by-step derivation
4. Show **dimensional analysis** (units in, units out)
5. Give at least one **limiting case** with a known analytical result

Prefix the commit subject with `[PHYSICS]`.

## Adding a dependency

```bash
uv add <package>              # core dependency
uv add --optional dev <pkg>   # dev-only
uv add --optional gpu <pkg>   # GPU cluster only
```

Commit both `pyproject.toml` and `uv.lock`.
