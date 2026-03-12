---
name: check
description: >
  Run the full quality gate: ruff lint, ruff format, mypy type checking, and
  pytest (CPU-only, no slow tests). Use before committing or after significant
  code changes. Reports results in priority order.
disable-model-invocation: true
argument-hint: [file_or_directory] (defaults to master_thesis_code/)
allowed-tools: Bash(uv run *)
---

## Quality Gate

Run all checks on the target (default: master_thesis_code/):

### Step 1: Ruff lint (auto-fix)
```bash
uv run ruff check --fix <target>
```

### Step 2: Ruff format check
```bash
uv run ruff format --check <target>
```
If formatting issues found, ask user before applying `ruff format`.

### Step 3: Mypy type check
```bash
uv run mypy <target>
```
Priority errors: `disallow_untyped_defs` violations > missing imports > other.

### Step 4: Tests (CPU-only, fast)
```bash
uv run pytest -m "not gpu and not slow" --tb=short -q
```
Report: passed/failed/skipped counts + coverage %.

### Summary format:
| Check       | Status | Issues |
|------------|--------|--------|
| Ruff lint   | PASS/FAIL | N issues |
| Ruff format | PASS/FAIL | N files |
| Mypy        | PASS/FAIL | N errors |
| Tests       | PASS/FAIL | N failed / M passed |

### Commit readiness:
- All PASS → "Ready to commit"
- Any FAIL → list blocking issues with file:line
