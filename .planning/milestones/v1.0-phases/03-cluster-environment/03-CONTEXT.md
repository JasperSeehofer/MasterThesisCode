# Phase 3: Cluster Environment - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Create and verify a repeatable environment setup for bwUniCluster 3.0. Delivers two shell scripts (`cluster/modules.sh` and `cluster/setup.sh`) and workspace path integration so that a fresh cluster account can go from zero to a working `.venv` with all GPU dependencies installed.

</domain>

<decisions>
## Implementation Decisions

### Module Names
- **D-01:** Hardcode best-guess module names from the bwHPC wiki. The load sequence is:
  ```bash
  module load compiler/gnu/14.2
  module load devel/cuda/12.8
  module load numlib/gsl
  module load devel/python/3.13.3-gnu-14.2
  ```
  These are derived from public bwHPC wiki documentation and need on-cluster verification. The GSL version string is not publicly documented — `numlib/gsl` loads the cluster default.
- **D-02:** `modules.sh` should fail early with a clear error message if any module is not found, so the user knows exactly what to fix.

### uv Installation
- **D-03:** Check-and-skip pattern: `command -v uv` first, only install if missing. When installing, use the official Astral installer (`curl -LsSf https://astral.sh/uv/install.sh | sh`), which installs to `~/.local/bin`.

### Workspace Integration
- **D-04:** Check-and-allocate pattern: `ws_find emri` first; if workspace doesn't exist, run `ws_allocate emri 60` (60-day duration). Idempotent — safe to re-run `setup.sh`.
- **D-05:** `modules.sh` resolves the workspace path via `ws_find emri` and exports `$WORKSPACE`. All downstream scripts (Phase 4 SLURM jobs) use `$WORKSPACE` as the single source of truth for output paths.

### Python Version
- **D-06:** Use the cluster Python module (`devel/python/3.13.3-gnu-14.2`), not uv-managed Python. `uv sync` creates the virtualenv from the module-provided Python. This avoids extra downloads and uses the cluster-optimized build.

### Claude's Discretion
- Error handling and messaging in `setup.sh` (what to do if `uv sync` fails, if modules can't be loaded, etc.)
- Whether `modules.sh` also exports convenience variables like `$PROJECT_ROOT` or `$VENV_PATH`
- Verification steps at the end of `setup.sh` (e.g., `python -c "import master_thesis_code"` smoke test)
- Script structure (functions vs sequential, logging style)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — ENV-01, ENV-02, ENV-03

### Prior Phase Context
- `.planning/phases/01-code-hardening/01-CONTEXT.md` — Import guard decisions, `--use_gpu`/`--num_workers` flag threading
- `.planning/phases/02-batch-compatibility/02-CONTEXT.md` — Entry points `emri-merge`/`emri-prepare`, `--workdir` pattern

### Codebase References
- `pyproject.toml` — Dependency groups (`gpu` extras: `cupy-cuda12x`, `fastemriwaveforms-cuda12x`), `[project.scripts]` entry points
- `master_thesis_code/constants.py` — File path constants that downstream scripts resolve relative to working directory
- `.python-version` — Pins Python 3.13

### External References
- [bwUniCluster 3.0 Software Modules wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0/Software_Modules) — Module naming conventions
- [bwHPC Workspace mechanism](https://wiki.bwhpc.de/e/Workspace) — `ws_allocate`, `ws_find`, expiration policies

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- No `cluster/` directory exists yet — this is greenfield
- `pyproject.toml` already defines `[project.optional-dependencies.gpu]` with `cupy-cuda12x` and `fastemriwaveforms-cuda12x`
- Entry points `emri-merge` and `emri-prepare` are registered and accept `--workdir` (Phase 2)

### Established Patterns
- `--use_gpu` flag defaults to `False` — scripts must explicitly enable GPU (Phase 1)
- `--num_workers` respects `os.sched_getaffinity(0) - 2` which honors SLURM cgroup limits (Phase 1)
- `uv sync --extra gpu` is the documented cluster install command (CLAUDE.md)

### Integration Points
- Phase 4 SLURM job scripts will `source cluster/modules.sh` to load environment and get `$WORKSPACE`
- `setup.sh` output: a working `.venv/` in the repo checkout with GPU dependencies installed
- `$WORKSPACE` feeds into `--workdir` arguments in Phase 4 batch scripts

</code_context>

<specifics>
## Specific Ideas

- Module names are best-guess from public wiki — first real cluster login should verify with `module spider` and update if needed
- STATE.md flagged "GSL module name unconfirmed" as a blocker — `numlib/gsl` is the best available guess

</specifics>

<deferred>
## Deferred Ideas

- **Apptainer container** — Container-based alternative to module setup is tracked as v2 requirement (CONT-01, CONT-02) in REQUIREMENTS.md. Not in Phase 3 scope.
- **Workspace expiration warnings** — Operational risk of workspace expiry belongs in Phase 5 documentation, not in setup scripts.

</deferred>

---

*Phase: 03-cluster-environment*
*Context gathered: 2026-03-26*
