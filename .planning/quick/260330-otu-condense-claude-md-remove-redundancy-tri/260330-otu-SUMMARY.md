---
phase: quick
plan: 260330-otu
subsystem: documentation
tags: [docs, claude-md, condensation]
dependency_graph:
  requires: []
  provides: [condensed-claude-md]
  affects: [all-claude-sessions]
tech_stack:
  added: []
  patterns: []
key_files:
  created: []
  modified: [CLAUDE.md]
decisions:
  - Removed 2 resolved bugs (cosmological_model size, comoving volume element) and kept 7 open bugs
  - Replaced ~80-line Technology Stack with 4-line summary pointing to pyproject.toml
  - Replaced ~90-line GSD Architecture with one-line reference to hand-written Architecture section
  - Condensed Conventions to 12-line summary of naming and docstring rules
  - Removed all redundant code examples from Typing, HPC/GPU, Testing sections
metrics:
  duration: 3m
  completed: "2026-03-30T15:58:14Z"
  tasks_completed: 1
  tasks_total: 1
  files_modified: 1
---

# Quick Task 260330-otu: Condense CLAUDE.md Summary

Reduced CLAUDE.md from 846 lines to 426 lines by removing resolved bugs, verbose code examples, and GSD-managed sections that duplicated hand-written content.

## Changes Made

### Task 1: Rewrite CLAUDE.md with condensed content
**Commit:** b47b46e

Specific condensations applied:
- **Known Bugs:** Removed 2 resolved items (cosmological_model.py size, comoving volume element). Kept 7 open bugs with concise descriptions.
- **Technology Stack (GSD):** Replaced ~80-line detailed listing with 4-line summary referencing pyproject.toml.
- **Architecture (GSD):** Replaced ~90-line duplicate with one-line reference to hand-written Architecture section above it.
- **Conventions (GSD):** Replaced ~48-line section with 12-line summary covering naming patterns, physics symbols, docstrings, and error types.
- **Typing Conventions:** Removed all code examples. Kept rules as bullet points (Python 3.10 syntax, npt.NDArray, CuPy, Callable, mypy).
- **HPC/GPU Best Practices:** Kept _get_xp pattern example. Replaced 5 subsections with bullet-point rules (no code blocks).
- **Testing Strategy:** Merged standalone "Running Tests" section. Removed all code examples. Kept running commands, GPU marker rule, xp fixture description, test priority list.
- **Cluster Deployment:** Removed Script Inventory table and Quick Reference code block. Kept intro + CLI flags table.
- **Dataclass Conventions:** Condensed from 23 lines to 5 lines (rule + correct example only).

### Preserved Sections (unchanged)
- Environment Setup, Dev Workflow, Running the Code
- Architecture (hand-written pipelines + module responsibilities)
- Skill-Driven Workflows (trigger table + physics-change files)
- Math/Physics Validation Workflow (full protocol)
- GSD Project, GSD Workflow Enforcement, Developer Profile

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

| Check | Result |
|-------|--------|
| Line count 350-450 | PASS: 426 lines |
| GSD start markers = 6 | PASS |
| GSD end markers = 6 | PASS |
| No RESOLVED/strikethrough text | PASS: 0 matches |
| _get_xp pattern present | PASS |
| Physics change protocol present | PASS |

## Self-Check: PASSED

- [x] CLAUDE.md exists and is 426 lines
- [x] Commit b47b46e exists in git log
- [x] All 6 GSD marker pairs intact
