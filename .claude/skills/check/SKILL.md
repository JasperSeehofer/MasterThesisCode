---
name: check
description: >
  Run the full quality gate: ruff lint, ruff format, mypy type checking, and
  pytest (CPU-only, no slow tests). Use before committing or after significant
  code changes. Reports results in priority order.
disable-model-invocation: true
argument-hint: [file_or_directory] (defaults to master_thesis_code/)
allowed-tools: Bash(uv run *), Bash(git diff *), Bash(git status *), Bash(grep *)
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

### Step 5: Physics-gate ledger (evidence check, does not append)

If the working tree or index touches a physics-trigger file (`physical_relations.py`,
`constants.py`, `LISA_configuration.py`, `parameter_estimation/parameter_estimation.py`,
`datamodels/galaxy.py`, `bayesian_inference/**`, `cosmological_model.py`) with a numerical-value
change:

```bash
git diff --name-only HEAD
grep '^| 20' docs/gates/PHYSICS-GATE-LEDGER.md | tail -5
```

Expect a `presented`/`implemented` row dated today for that file. If there is none, the
`/physics-change` hard gate has no evidence it ran → report **FAIL** and route the user to
`/physics-change` before committing. Only `/physics-change` appends rows; this step reads.
A refactor/type/comment-only diff is not a physics change and needs no row.

### Summary format:
| Check       | Status | Issues |
|------------|--------|--------|
| Ruff lint   | PASS/FAIL | N issues |
| Ruff format | PASS/FAIL | N files |
| Mypy        | PASS/FAIL | N errors |
| Tests       | PASS/FAIL | N failed / M passed |
| Physics ledger | PASS/FAIL/N-A | missing row for file.py |

### Commit readiness:
- All PASS → "Ready to commit"
- Any FAIL → list blocking issues with file:line
