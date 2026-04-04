# Phase 2: Batch Compatibility - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Post-simulation scripts (`merge_cramer_rao_bounds.py` and `prepare_detections.py`) run unattended in SLURM batch jobs without human interaction. This phase removes interactive prompts, adds proper CLI interfaces, and registers console entry points so scripts are discoverable and callable from batch job scripts.

</domain>

<decisions>
## Implementation Decisions

### Delete Behavior (merge script)
- **D-01:** `--delete-sources` flag = silent delete of per-index source CSVs after successful merge. No flag = keep files. No interactive `input()` prompt in either case — all 4 `input()` calls removed entirely.

### Script Invocation
- **D-02:** Register console entry points in `pyproject.toml` `[project.scripts]`. After `uv sync`, commands are available on `$PATH` without knowing script file locations.
- **D-03:** Naming convention: `emri-*` prefix. Specifically: `emri-merge` and `emri-prepare`. Tab-completable and namespaced.
- **D-04:** Each script gets argparse CLI and a proper `main()` function (required for entry points and batch invocability).

### Working Directory
- **D-05:** Scripts accept `--workdir` argument (optional, defaults to current directory). All paths from `constants.py` are resolved relative to the provided working directory. Consistent with how the main package takes a positional `working_dir` argument.

### Scope
- **D-06:** Only the 2 scripts required by BATCH-01 and BATCH-02: `merge_cramer_rao_bounds.py` and `prepare_detections.py`. `remove_detections_out_of_bounds.py` and `estimate_hubble_constant.py` are out of scope for this phase.

### Claude's Discretion
- Argparse argument naming and help text for the new CLI flags
- How `--workdir` interacts with the existing path constants (e.g. `os.path.join(workdir, CRAMER_RAO_BOUNDS_OUTPUT_PATH)` or a path-resolution helper)
- Error handling for missing files or empty merge sets
- Logging approach (print vs logging module)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scripts (primary targets)
- `scripts/merge_cramer_rao_bounds.py` — Has `main()` but 4 interactive `input()` calls blocking batch use. Merges per-index Cramér-Rao CSVs and undetected events CSVs.
- `scripts/prepare_detections.py` — All logic in `if __name__ == "__main__"` block, no `main()` function. Converts detection parameters to best-guess parameters.

### Path Constants
- `master_thesis_code/constants.py` — Defines `CRAMER_RAO_BOUNDS_PATH`, `CRAMER_RAO_BOUNDS_OUTPUT_PATH`, `UNDETECTED_EVENTS_PATH`, `UNDETECTED_EVENTS_OUTPUT_PATH`, `PREPARED_CRAMER_RAO_BOUNDS_PATH`

### Entry Point Registration
- `pyproject.toml` — `[project.scripts]` section for console entry points

### Requirements
- `.planning/REQUIREMENTS.md` — BATCH-01, BATCH-02

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `merge_cramer_rao_bounds.py` already has a `main()` function calling `merge_cramer_rao_bounds()` + `merge_undetected_events()` — structure is close, just needs argparse added
- `delete_files()` helper function already exists for cleanup logic
- Path constants in `constants.py` provide the relative paths to resolve against `--workdir`

### Established Patterns
- Main package uses positional `working_dir` argument (first arg to `python -m master_thesis_code`)
- CLI arguments defined via argparse in `arguments.py`
- `Detection` class imported from `cosmological_model` (backward-compat re-export) — used by `prepare_detections.py`

### Integration Points
- `pyproject.toml` `[project.scripts]` — new entry points: `emri-merge`, `emri-prepare`
- Phase 4 SLURM job scripts will call these entry points with `--workdir` pointing to the bwHPC workspace path
- `uv sync` must be re-run after adding entry points (updates `.venv/bin/`)

</code_context>

<specifics>
## Specific Ideas

- User wants cleanest UX for someone who didn't write the project — entry points provide discoverability via `emri-<tab>` completion
- Delete behavior should be fully non-interactive: flag = delete, no flag = keep, never prompt

</specifics>

<deferred>
## Deferred Ideas

- **Batch-harden remaining scripts** — `remove_detections_out_of_bounds.py` and `estimate_hubble_constant.py` also lack `main()` functions and could benefit from argparse + entry points. Deferred to a future phase or backlog item.

</deferred>

---

*Phase: 02-batch-compatibility*
*Context gathered: 2026-03-26*
