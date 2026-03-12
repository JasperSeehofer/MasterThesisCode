---
name: pre-commit-docs
description: >
  Check that CHANGELOG.md, TODO.md, CLAUDE.md, README.md, CONTRIBUTING.md, and
  memory files are consistent with the staged changes before committing. Reports
  a checklist of needed documentation updates — does not auto-edit.
argument-hint: (no arguments; operates on staged git changes)
allowed-tools: Bash(git *), Read, Grep, Glob
---

## Pre-Commit Documentation Consistency Check

Examine the staged changes (`git diff --cached`) and cross-reference against all
project documentation files. Report what needs updating before the commit.

### Step 1: Capture the staged diff
```bash
git diff --cached --stat
git diff --cached
```
Identify: files changed, functions/classes added/removed/renamed, constants changed,
bugs fixed, new CLI flags, new test files.

### Step 2: Check CHANGELOG.md

Read the `[Unreleased]` section of `CHANGELOG.md`.

- Are the staged changes reflected there? Every commit should have a changelog entry
  unless it is trivially small (typo fix, comment-only change).
- Use the correct subsection: `Added`, `Changed`, `Fixed`, `Removed`.
- Flag: "CHANGELOG.md: `[Unreleased]` section is empty — add entries for: ..."

### Step 3: Check TODO.md

- If a known bug was fixed (check for `[PHYSICS]` prefix in commit message, or changes
  to files listed in the TODO), is the corresponding `- [ ]` item checked off to `- [x]`?
- If new `# TODO` comments appear in staged code, should they be tracked in TODO.md?
- Flag: "TODO.md: item X should be marked done" or "TODO.md: new TODO in file:line not tracked"

### Step 4: Check CLAUDE.md

- **Architecture section**: If new files were created, modules renamed, or classes moved,
  is "Key Module Responsibilities" still accurate?
- **Known Bugs section**: If a listed bug was fixed, is it updated/removed?
- **Conventions sections**: If new patterns were introduced (new decorator, new CLI flag,
  new test marker), are they documented?
- **Skill-Driven Workflows**: If a new skill was added, is it in the trigger rules table?
- Flag: "CLAUDE.md: Known Bug N was fixed but still listed as open"

### Step 5: Check README.md

- If the "Known Limitations" section lists a bug that was just fixed, flag it.
- If new CLI arguments were added, check the usage examples.
- Flag: "README.md: limitation X was resolved — update or remove"

### Step 6: Check CONTRIBUTING.md

- If new workflow steps were added (new CLI flags, new test markers, new pre-commit hooks),
  check whether contributors need to know.
- This file rarely needs updates — only flag if something clearly changed.

### Step 7: Check Memory (MEMORY.md)

- If changes affect: architecture, bug status, test coverage numbers, key file locations,
  or branch information — flag that memory may be stale.
- Flag: "MEMORY.md: bug status / test coverage / architecture section may need update"

### Output format

```
## Pre-Commit Documentation Check

### CHANGELOG.md
- [ ] Add entry under [Unreleased] > Added: ...
- [x] No issues

### TODO.md
- [ ] Mark item "Fix comoving volume..." as done
- [x] No issues

### CLAUDE.md
- [ ] Update Known Bugs: bug 3 was fixed
- [x] No issues

### README.md
- [x] No issues

### CONTRIBUTING.md
- [x] No issues

### Memory (MEMORY.md)
- [ ] Update test coverage numbers
- [x] No issues

### Summary: N items need attention before committing.
```

If all checks pass, report: "All documentation is consistent. Ready to commit."
