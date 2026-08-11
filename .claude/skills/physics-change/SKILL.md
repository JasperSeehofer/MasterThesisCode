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
- `darksiren_emri/physical_relations.py`
- `darksiren_emri/constants.py`
- `darksiren_emri/LISA_configuration.py`
- `darksiren_emri/parameter_estimation/parameter_estimation.py`
- `darksiren_emri/datamodels/galaxy.py`
- `darksiren_emri/bayesian_inference/bayesian_inference.py`
- `darksiren_emri/cosmological_model.py`

Refactoring, type annotations, import cleanup, and comment-only changes do NOT trigger
this protocol — only changes that alter a computed numerical value.

### Before writing ANY code, present all 5 items to the user:

1. **Old formula** — exact current expression, with `file_path:line_number`
2. **New formula** — proposed replacement expression
3. **Reference** — citation: arXiv ID or DOI + equation number, OR step-by-step derivation
4. **Dimensional analysis** — units of every input, units of output, consistency proof
5. **Limiting case** — at least one analytically known limit (e.g., z→0, f→0, M→0)

### STOP and wait for explicit user approval before implementing.

Once the user answers, append a ledger row (see below) — `APPROVED` or `REJECTED`. A `REJECTED`
verdict ends the protocol here; the row is still written.

### After implementation, verify and report:
- [ ] Sign convention consistency with rest of codebase
- [ ] Dimensional consistency (no mixed units)
- [ ] Reference comment added directly above the changed line:
  ```python
  # Eq. (X.Y) in Author et al. (YYYY), arXiv:XXXX.XXXXX
  ```
- [ ] Regression test added BEFORE the change (asserting old value) so the diff is visible
- [ ] Git commit uses `[PHYSICS]` prefix
- [ ] Ledger rows appended to `docs/gates/PHYSICS-GATE-LEDGER.md`

### Ledger — the gate must leave evidence

A `[PHYSICS]` commit with no ledger row is indistinguishable from a skipped gate. Append to
`docs/gates/PHYSICS-GATE-LEDGER.md` (read its header for the column contract) at each step
completion — never rewrite existing rows, never back-fill older commits:

| when | row |
|---|---|
| user answers the 5-item gate | `\| YYYY-MM-DD \| pre-commit \| presented \| APPROVED\|REJECTED \| file.py:line \| what changed \|` |
| code written after approval | `\| YYYY-MM-DD \| pre-commit \| implemented \| PASS \| file.py:line \| ref comment + regression test \|` |
| post-implementation checks reported | `\| YYYY-MM-DD \| <short-sha> \| verified \| PASS\|FAIL \| file.py:line \| sign + units + limit checked \|` |

Use `pre-commit` in the commit column until the commit exists, then update those rows to the
short SHA when it lands. Rows go at the bottom of the `## Ledger` table (newest last).

### Known physics bugs for reference:
!`grep -A2 "CRITICAL\|HIGH\|MEDIUM\|LOW" /home/jasper/Repositories/MasterThesisCode/CLAUDE.md | head -30`
