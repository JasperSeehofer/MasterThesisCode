---
name: physics-change
description: >
  MANDATORY when modifying any physics formula, physical constant, waveform
  parameter, frequency limit, PSD coefficient, or galaxy/cosmological model.
  Enforces the Physics Change Protocol from CLAUDE.md. Triggers on changes to
  files: physical_relations.py, constants.py, LISA_configuration.py,
  parameter_estimation.py, galaxy.py, bayesian_inference.py, cosmological_model.py.
argument-hint: <description of the proposed change>
---

## Physics Change Protocol

You are about to modify physics/math code. This protocol is NON-NEGOTIABLE.

### Trigger files (any formula/constant change in these files requires this protocol):
- `master_thesis_code/physical_relations.py`
- `master_thesis_code/constants.py`
- `master_thesis_code/LISA_configuration.py`
- `master_thesis_code/parameter_estimation/parameter_estimation.py`
- `master_thesis_code/datamodels/galaxy.py`
- `master_thesis_code/bayesian_inference/bayesian_inference.py`
- `master_thesis_code/cosmological_model.py`

Refactoring, type annotations, import cleanup, and comment-only changes do NOT trigger
this protocol — only changes that alter a computed numerical value.

### Before writing ANY code, present all 5 items to the user:

1. **Old formula** — exact current expression, with `file_path:line_number`
2. **New formula** — proposed replacement expression
3. **Reference** — citation: arXiv ID or DOI + equation number, OR step-by-step derivation
4. **Dimensional analysis** — units of every input, units of output, consistency proof
5. **Limiting case** — at least one analytically known limit (e.g., z→0, f→0, M→0)

### STOP and wait for explicit user approval before implementing.

### After implementation, verify and report:
- [ ] Sign convention consistency with rest of codebase
- [ ] Dimensional consistency (no mixed units)
- [ ] Reference comment added directly above the changed line:
  ```python
  # Eq. (X.Y) in Author et al. (YYYY), arXiv:XXXX.XXXXX
  ```
- [ ] Regression test added BEFORE the change (asserting old value) so the diff is visible
- [ ] Git commit uses `[PHYSICS]` prefix

### Known physics bugs for reference:
!`grep -A2 "CRITICAL\|HIGH\|MEDIUM\|LOW" /home/jasper/Repositories/MasterThesisCode/CLAUDE.md | head -30`
