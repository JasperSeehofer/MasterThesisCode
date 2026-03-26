# Phase 2: Batch Compatibility - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 02-batch-compatibility
**Areas discussed:** Delete behavior, Script invocation, Working directory, Other scripts scope

---

## Delete Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Flag = silent delete | --delete-sources means delete without asking. No flag = keep files (no prompt either). Batch-safe by default, explicit opt-in for cleanup. | ✓ |
| Flag = delete, no flag = prompt | --delete-sources deletes silently. Without the flag, the current interactive prompt is preserved for manual runs. | |
| Always non-interactive | Remove all input() calls entirely. Add --delete-sources as the only way to delete. No interactive mode at all. | |

**User's choice:** Flag = silent delete
**Notes:** User initially asked for clarification about which files are being deleted. Explained: per-index source CSVs (e.g. `cramer_rao_bounds_0.csv`) are cleaned up after merging into the combined output file.

---

## Script Invocation

| Option | Description | Selected |
|--------|-------------|----------|
| argparse + main() per script | Each script gets argparse CLI and a main() function. Invoked as 'python scripts/merge_cramer_rao_bounds.py --delete-sources'. Simple, no package plumbing needed. | |
| Package with python -m | Add scripts/__init__.py, make it a package. Invoke as 'python -m scripts.merge_cramer_rao_bounds'. Requires package structure changes. | |
| pyproject.toml entry points | Register as console_scripts in pyproject.toml (e.g. 'emri-merge'). Invoke as shell commands after install. More setup but cleaner CLI. | ✓ |

**User's choice:** pyproject.toml entry points
**Notes:** User asked about cleanest UX for someone unfamiliar with the project. Confirmed that entry points provide the best discoverability (available on $PATH after install, tab-completable).

### Follow-up: Naming Convention

| Option | Description | Selected |
|--------|-------------|----------|
| emri-* prefix | emri-merge, emri-prepare. Short, namespaced, discoverable via 'emri-<tab>'. | ✓ |
| master-thesis-* prefix | master-thesis-merge, master-thesis-prepare. Matches package name but verbose. | |
| No prefix, descriptive names | merge-cramer-rao, prepare-detections. Short but risk of name collisions on cluster. | |

**User's choice:** emri-* prefix

---

## Working Directory

| Option | Description | Selected |
|--------|-------------|----------|
| Accept --workdir argument | Scripts take an optional --workdir (defaults to current directory). Paths from constants.py are resolved relative to it. Matches how the main package takes a positional working_dir. | ✓ |
| Use current directory only | Scripts always resolve paths relative to cwd. SLURM scripts must cd to the workspace before calling. Simpler code but requires discipline in job scripts. | |
| Environment variable | Read EMRI_WORKDIR env var (fallback to cwd). Set once in SLURM job preamble. Less explicit but convenient for chained scripts. | |

**User's choice:** Accept --workdir argument

---

## Other Scripts Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Include all 4 scripts | Since we're adding argparse + main() + entry points anyway, do all 4 scripts in one pass. Avoids a second touch later. | |
| Only the 2 required scripts | Stick to BATCH-01 (merge) and BATCH-02 (prepare). Touch the others when they're actually needed. Smaller scope, faster delivery. | ✓ |
| Required 2 + remove_detections | Include remove_detections_out_of_bounds.py since it's part of the pipeline. Leave estimate_hubble_constant.py for later. | |

**User's choice:** Only the 2 required scripts

---

## Claude's Discretion

- Argparse argument naming and help text
- Path resolution approach for --workdir
- Error handling and logging details

## Deferred Ideas

- Batch-harden `remove_detections_out_of_bounds.py` and `estimate_hubble_constant.py` (out of scope for Phase 2)
